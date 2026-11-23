#include "mcpricer/qmc.hpp"

#include <stdexcept>

#include "estimators.hpp"
#include "gbm.hpp"
#include "mcpricer/normal.hpp"
#include "mcpricer/parallel.hpp"
#include "mcpricer/rng.hpp"
#include "mcpricer/statistics.hpp"

namespace mcp {

double rqmc_shifted_estimate(const EuropeanOption& option, const Market& market, std::uint64_t num_points,
                             double shift) {
    validate(option);
    validate(market);
    if (num_points == 0) throw std::invalid_argument("num_points must be positive");
    if (!(shift >= 0.0 && shift < 1.0)) throw std::invalid_argument("shift must lie in [0, 1)");

    const detail::TerminalGBM gbm(market, option.maturity);
    constexpr double kTwoPowMinus54 = 0.5 / 9007199254740992.0;

    const auto stats = parallel_accumulate<RunningStats>(num_points, [&](std::uint64_t i, RunningStats& acc) {
        double u = van_der_corput(i) + shift;
        if (u >= 1.0) u -= 1.0;
        if (u <= 0.0) u = kTwoPowMinus54;  // the quantile transform needs u in (0, 1)
        const double s_t = gbm.terminal(norm_inv_cdf(u));
        acc.add(gbm.discount * payoff(option.type, s_t, option.strike));
    });
    return stats.mean();
}

MCResult price_european_rqmc(const EuropeanOption& option, const Market& market, const RQMCConfig& config) {
    if (config.num_randomisations < 2) throw std::invalid_argument("at least two randomisations are needed");

    const detail::Stopwatch clock;
    RunningStats across;
    for (std::uint32_t r = 0; r < config.num_randomisations; ++r) {
        // Shifts come from a dedicated substream so they never coincide with path streams.
        RandomStream shift_rng(config.seed, r, /*substream=*/0xA5A5A5A5u);
        across.add(rqmc_shifted_estimate(option, market, config.num_points, shift_rng.uniform()));
    }

    MCResult result = detail::mean_estimator(across, config.num_points * config.num_randomisations);
    result.elapsed_seconds = clock.seconds();
    return result;
}

}  // namespace mcp
