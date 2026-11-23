#include "mcpricer/asian.hpp"

#include <cmath>
#include <stdexcept>

#include "estimators.hpp"
#include "mcpricer/normal.hpp"
#include "mcpricer/parallel.hpp"
#include "mcpricer/rng.hpp"
#include "mcpricer/statistics.hpp"

namespace mcp {

void validate(const AsianOption& option) {
    validate(EuropeanOption{option.type, option.strike, option.maturity});
    if (option.num_fixings == 0) throw std::invalid_argument("num_fixings must be positive");
}

double geometric_asian_price(const AsianOption& option, const Market& market) {
    validate(option);
    validate(market);
    const double n = static_cast<double>(option.num_fixings);
    const double t = option.maturity;
    const double sigma = market.vol;

    const double mean_log =
        std::log(market.spot) + (market.rate - market.dividend - 0.5 * sigma * sigma) * t * (n + 1.0) / (2.0 * n);
    const double var_log = sigma * sigma * t * (n + 1.0) * (2.0 * n + 1.0) / (6.0 * n * n);
    const double sd_log = std::sqrt(var_log);

    const double forward = std::exp(mean_log + 0.5 * var_log);  // E[G]
    const double d1 = (mean_log - std::log(option.strike) + var_log) / sd_log;
    const double d2 = d1 - sd_log;
    const double df = std::exp(-market.rate * t);
    if (option.type == OptionType::Call) {
        return df * (forward * norm_cdf(d1) - option.strike * norm_cdf(d2));
    }
    return df * (option.strike * norm_cdf(-d2) - forward * norm_cdf(-d1));
}

namespace {

struct AsianPathValues {
    double arithmetic;  // discounted payoff on the arithmetic average
    double geometric;   // discounted payoff on the geometric average
};

}  // namespace

MCResult price_asian_mc(const AsianOption& option, const Market& market, const MCConfig& config) {
    validate(option);
    validate(market);
    if (config.num_paths < 4) throw std::invalid_argument("num_paths must be at least 4");
    const bool use_cv = config.variance_reduction == VarianceReduction::ControlVariate ||
                        config.variance_reduction == VarianceReduction::AntitheticControlVariate;
    const bool antithetic = config.variance_reduction == VarianceReduction::Antithetic ||
                            config.variance_reduction == VarianceReduction::AntitheticControlVariate;
    if (use_cv && option.averaging == Averaging::Geometric) {
        throw std::invalid_argument("the geometric control variate requires an arithmetic Asian option");
    }

    const detail::Stopwatch clock;
    const std::uint32_t n_fix = option.num_fixings;
    const double inv_n = 1.0 / static_cast<double>(n_fix);
    const double dt = option.maturity * inv_n;
    const double drift = (market.rate - market.dividend - 0.5 * market.vol * market.vol) * dt;
    const double vol_sqrt_dt = market.vol * std::sqrt(dt);
    const double log_s0 = std::log(market.spot);
    const double df = std::exp(-market.rate * option.maturity);
    const OptionType type = option.type;
    const double strike = option.strike;
    const std::uint64_t seed = config.seed;

    // Simulates both the path driven by Z and, if requested, its mirror driven by -Z.
    auto simulate = [&](std::uint64_t i, AsianPathValues& plus, AsianPathValues& minus) {
        RandomStream rng(seed, i);
        double log_p = log_s0, log_m = log_s0;
        double sum_p = 0.0, sum_m = 0.0, sum_log_p = 0.0, sum_log_m = 0.0;
        for (std::uint32_t k = 0; k < n_fix; ++k) {
            const double dw = vol_sqrt_dt * rng.normal();
            log_p += drift + dw;
            sum_p += std::exp(log_p);
            sum_log_p += log_p;
            if (antithetic) {
                log_m += drift - dw;
                sum_m += std::exp(log_m);
                sum_log_m += log_m;
            }
        }
        plus = {df * payoff(type, sum_p * inv_n, strike), df * payoff(type, std::exp(sum_log_p * inv_n), strike)};
        if (antithetic) {
            minus = {df * payoff(type, sum_m * inv_n, strike), df * payoff(type, std::exp(sum_log_m * inv_n), strike)};
        }
    };
    auto target = [&](const AsianPathValues& v) {
        return option.averaging == Averaging::Arithmetic ? v.arithmetic : v.geometric;
    };

    const std::uint64_t n_samples = antithetic ? config.num_paths / 2 : config.num_paths;
    const std::uint64_t n_paths = antithetic ? 2 * n_samples : n_samples;

    MCResult result;
    if (use_cv) {
        const double control_mean = geometric_asian_price(option, market);
        const auto stats =
            parallel_accumulate<RunningCovariance>(n_samples, [&](std::uint64_t i, RunningCovariance& acc) {
                AsianPathValues p{}, m{};
                simulate(i, p, m);
                if (antithetic) {
                    acc.add(0.5 * (p.arithmetic + m.arithmetic), 0.5 * (p.geometric + m.geometric));
                } else {
                    acc.add(p.arithmetic, p.geometric);
                }
            });
        result = detail::control_variate_estimator(stats, control_mean, n_paths);
    } else {
        const auto stats = parallel_accumulate<RunningStats>(n_samples, [&](std::uint64_t i, RunningStats& acc) {
            AsianPathValues p{}, m{};
            simulate(i, p, m);
            acc.add(antithetic ? 0.5 * (target(p) + target(m)) : target(p));
        });
        result = detail::mean_estimator(stats, n_paths);
    }
    result.elapsed_seconds = clock.seconds();
    return result;
}

}  // namespace mcp
