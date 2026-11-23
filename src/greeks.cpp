#include "mcpricer/greeks.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "estimators.hpp"
#include "gbm.hpp"
#include "mcpricer/parallel.hpp"
#include "mcpricer/rng.hpp"
#include "mcpricer/statistics.hpp"

namespace mcp {

std::string_view to_string(Greek greek) noexcept {
    switch (greek) {
        case Greek::Delta: return "delta";
        case Greek::Gamma: return "gamma";
        case Greek::Vega: return "vega";
        case Greek::Rho: return "rho";
        case Greek::Theta: return "theta";
    }
    return "unknown";
}

std::string_view to_string(GreekMethod method) noexcept {
    switch (method) {
        case GreekMethod::FiniteDifferenceCRN: return "fd_crn";
        case GreekMethod::FiniteDifferenceIndependent: return "fd_independent";
        case GreekMethod::Pathwise: return "pathwise";
        case GreekMethod::LikelihoodRatio: return "likelihood_ratio";
    }
    return "unknown";
}

bool is_supported(Greek greek, GreekMethod method) noexcept {
    switch (method) {
        case GreekMethod::FiniteDifferenceCRN:
        case GreekMethod::FiniteDifferenceIndependent: return true;
        case GreekMethod::Pathwise: return greek != Greek::Gamma;
        case GreekMethod::LikelihoodRatio: return greek != Greek::Theta;
    }
    return false;
}

double default_bump(Greek greek, const Market& market) noexcept {
    switch (greek) {
        case Greek::Delta:
        case Greek::Gamma: return 0.01 * market.spot;
        case Greek::Vega: return 1e-3;
        case Greek::Rho: return 1e-4;
        case Greek::Theta: return 1.0 / 365.0;
    }
    return 0.0;
}

namespace {

/// One revaluation scenario of a finite-difference stencil.
struct Scenario {
    detail::TerminalGBM gbm;
    double weight;
};

std::vector<Scenario> fd_stencil(Greek greek, const EuropeanOption& option, const Market& market, double h) {
    auto bumped = [&](double d_spot, double d_vol, double d_rate, double d_maturity, double weight) {
        Market m = market;
        m.spot += d_spot;
        m.vol += d_vol;
        m.rate += d_rate;
        const double maturity = option.maturity + d_maturity;
        if (!(m.spot > 0.0) || !(m.vol > 0.0) || !(maturity > 0.0)) {
            throw std::invalid_argument("finite-difference bump " + std::to_string(h) + " is too large for " +
                                        std::string(to_string(greek)));
        }
        return Scenario{detail::TerminalGBM(m, maturity), weight};
    };

    const double c = 0.5 / h;  // central first difference
    switch (greek) {
        case Greek::Delta: return {bumped(h, 0, 0, 0, c), bumped(-h, 0, 0, 0, -c)};
        case Greek::Gamma: {
            const double w = 1.0 / (h * h);
            return {bumped(h, 0, 0, 0, w), bumped(0, 0, 0, 0, -2.0 * w), bumped(-h, 0, 0, 0, w)};
        }
        case Greek::Vega: return {bumped(0, h, 0, 0, c), bumped(0, -h, 0, 0, -c)};
        case Greek::Rho: return {bumped(0, 0, h, 0, c), bumped(0, 0, -h, 0, -c)};
        case Greek::Theta: return {bumped(0, 0, 0, h, -c), bumped(0, 0, 0, -h, c)};  // theta = -dV/dT
    }
    throw std::invalid_argument("unknown greek");
}

RunningStats finite_difference(Greek greek, bool common_random_numbers, const EuropeanOption& option,
                               const Market& market, const GreekConfig& config, double h) {
    const std::vector<Scenario> stencil = fd_stencil(greek, option, market, h);
    const OptionType type = option.type;
    const double strike = option.strike;
    const std::uint64_t seed = config.seed;

    return parallel_accumulate<RunningStats>(config.num_paths, [&](std::uint64_t i, RunningStats& acc) {
        double estimate = 0.0;
        if (common_random_numbers) {
            RandomStream rng(seed, i);
            const double z = rng.normal();
            for (const Scenario& s : stencil) {
                estimate += s.weight * s.gbm.discount * payoff(type, s.gbm.terminal(z), strike);
            }
        } else {
            std::uint32_t substream = 1;
            for (const Scenario& s : stencil) {
                RandomStream rng(seed, i, substream++);
                estimate += s.weight * s.gbm.discount * payoff(type, s.gbm.terminal(rng.normal()), strike);
            }
        }
        acc.add(estimate);
    });
}

RunningStats pathwise(Greek greek, const EuropeanOption& option, const Market& market, const GreekConfig& config) {
    const detail::TerminalGBM gbm(market, option.maturity);
    const bool is_call = option.type == OptionType::Call;
    const double k = option.strike;
    const double t = option.maturity;
    const double sqrt_t = std::sqrt(t);
    const double s0 = market.spot;
    const double sigma = market.vol;
    const double r = market.rate;
    const double mu = market.rate - market.dividend - 0.5 * sigma * sigma;
    const double df = gbm.discount;
    const std::uint64_t seed = config.seed;

    return parallel_accumulate<RunningStats>(config.num_paths, [&](std::uint64_t i, RunningStats& acc) {
        RandomStream rng(seed, i);
        const double z = rng.normal();
        const double s_t = gbm.terminal(z);
        // derivative of the payoff with respect to S_T (defined almost surely)
        const double dpayoff = is_call ? (s_t > k ? 1.0 : 0.0) : (s_t < k ? -1.0 : 0.0);
        double estimate = 0.0;
        switch (greek) {
            case Greek::Delta: estimate = df * dpayoff * s_t / s0; break;
            case Greek::Vega: estimate = df * dpayoff * s_t * (sqrt_t * z - sigma * t); break;
            case Greek::Rho: estimate = t * df * (dpayoff * s_t - payoff(option.type, s_t, k)); break;
            case Greek::Theta: {
                const double ds_dt = s_t * (mu + 0.5 * sigma * z / sqrt_t);
                estimate = -(-r * df * payoff(option.type, s_t, k) + df * dpayoff * ds_dt);
                break;
            }
            case Greek::Gamma: break;  // rejected by is_supported()
        }
        acc.add(estimate);
    });
}

RunningStats likelihood_ratio(Greek greek, const EuropeanOption& option, const Market& market,
                              const GreekConfig& config) {
    const detail::TerminalGBM gbm(market, option.maturity);
    const double t = option.maturity;
    const double sqrt_t = std::sqrt(t);
    const double s0 = market.spot;
    const double sigma = market.vol;
    const double df = gbm.discount;
    const std::uint64_t seed = config.seed;

    return parallel_accumulate<RunningStats>(config.num_paths, [&](std::uint64_t i, RunningStats& acc) {
        RandomStream rng(seed, i);
        const double z = rng.normal();
        const double value = df * payoff(option.type, gbm.terminal(z), option.strike);
        double score = 0.0;  // d/dtheta log density of S_T (second-order score for Gamma)
        switch (greek) {
            case Greek::Delta: score = z / (s0 * sigma * sqrt_t); break;
            case Greek::Gamma: score = (z * z - 1.0 - z * sigma * sqrt_t) / (s0 * s0 * sigma * sigma * t); break;
            case Greek::Vega: score = (z * z - 1.0) / sigma - z * sqrt_t; break;
            case Greek::Rho: score = z * sqrt_t / sigma - t; break;  // includes -T from the discount factor
            case Greek::Theta: break;                                // rejected by is_supported()
        }
        acc.add(value * score);
    });
}

}  // namespace

GreekResult mc_greek(Greek greek, GreekMethod method, const EuropeanOption& option, const Market& market,
                     const GreekConfig& config) {
    validate(option);
    validate(market);
    if (!is_supported(greek, method)) {
        throw std::invalid_argument(std::string(to_string(method)) + " estimator not available for " +
                                    std::string(to_string(greek)));
    }
    if (config.num_paths < 2) throw std::invalid_argument("num_paths must be at least 2");

    const detail::Stopwatch clock;
    GreekResult result;
    RunningStats stats;
    switch (method) {
        case GreekMethod::FiniteDifferenceCRN:
        case GreekMethod::FiniteDifferenceIndependent: {
            const double h = config.bump > 0.0 ? config.bump : default_bump(greek, market);
            stats = finite_difference(greek, method == GreekMethod::FiniteDifferenceCRN, option, market, config, h);
            result.bump = h;
            break;
        }
        case GreekMethod::Pathwise: stats = pathwise(greek, option, market, config); break;
        case GreekMethod::LikelihoodRatio: stats = likelihood_ratio(greek, option, market, config); break;
    }
    result.value = stats.mean();
    result.std_error = stats.std_error();
    result.num_paths = config.num_paths;
    result.elapsed_seconds = clock.seconds();
    return result;
}

}  // namespace mcp
