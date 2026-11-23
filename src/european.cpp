#include "mcpricer/european.hpp"

#include <cmath>
#include <stdexcept>

#include "estimators.hpp"
#include "gbm.hpp"
#include "mcpricer/parallel.hpp"
#include "mcpricer/rng.hpp"
#include "mcpricer/statistics.hpp"

namespace mcp {

std::string_view to_string(VarianceReduction method) noexcept {
    switch (method) {
        case VarianceReduction::None: return "crude";
        case VarianceReduction::Antithetic: return "antithetic";
        case VarianceReduction::ControlVariate: return "control_variate";
        case VarianceReduction::AntitheticControlVariate: return "antithetic_control_variate";
    }
    return "unknown";
}

MCResult price_european_mc(const EuropeanOption& option, const Market& market, const MCConfig& config) {
    validate(option);
    validate(market);
    if (config.num_paths < 4) throw std::invalid_argument("num_paths must be at least 4");

    const detail::Stopwatch clock;
    const detail::TerminalGBM gbm(market, option.maturity);
    const OptionType type = option.type;
    const double strike = option.strike;
    const double df = gbm.discount;
    const std::uint64_t seed = config.seed;

    // Discounted payoff (the estimand) and discounted terminal price (the control).
    auto discounted = [&](double z, double& y, double& x) {
        const double s_t = gbm.terminal(z);
        y = df * payoff(type, s_t, strike);
        x = df * s_t;
    };

    // E[e^{-rT} S_T] = S_0 e^{-qT}
    const double control_mean = market.spot * std::exp(-market.dividend * option.maturity);
    const std::uint64_t n = config.num_paths;
    const std::uint64_t n_pairs = n / 2;

    MCResult result;
    switch (config.variance_reduction) {
        case VarianceReduction::None: {
            const auto stats = parallel_accumulate<RunningStats>(n, [&](std::uint64_t i, RunningStats& acc) {
                RandomStream rng(seed, i);
                double y, x;
                discounted(rng.normal(), y, x);
                acc.add(y);
            });
            result = detail::mean_estimator(stats, n);
            break;
        }
        case VarianceReduction::Antithetic: {
            const auto stats = parallel_accumulate<RunningStats>(n_pairs, [&](std::uint64_t i, RunningStats& acc) {
                RandomStream rng(seed, i);
                const double z = rng.normal();
                double y_plus, y_minus, x;
                discounted(z, y_plus, x);
                discounted(-z, y_minus, x);
                acc.add(0.5 * (y_plus + y_minus));
            });
            result = detail::mean_estimator(stats, 2 * n_pairs);
            break;
        }
        case VarianceReduction::ControlVariate: {
            const auto stats = parallel_accumulate<RunningCovariance>(n, [&](std::uint64_t i, RunningCovariance& acc) {
                RandomStream rng(seed, i);
                double y, x;
                discounted(rng.normal(), y, x);
                acc.add(y, x);
            });
            result = detail::control_variate_estimator(stats, control_mean, n);
            break;
        }
        case VarianceReduction::AntitheticControlVariate: {
            const auto stats =
                parallel_accumulate<RunningCovariance>(n_pairs, [&](std::uint64_t i, RunningCovariance& acc) {
                    RandomStream rng(seed, i);
                    const double z = rng.normal();
                    double y_plus, x_plus, y_minus, x_minus;
                    discounted(z, y_plus, x_plus);
                    discounted(-z, y_minus, x_minus);
                    acc.add(0.5 * (y_plus + y_minus), 0.5 * (x_plus + x_minus));
                });
            result = detail::control_variate_estimator(stats, control_mean, 2 * n_pairs);
            break;
        }
    }
    result.elapsed_seconds = clock.seconds();
    return result;
}

}  // namespace mcp
