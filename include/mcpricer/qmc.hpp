#pragma once

#include <cstdint>

#include "mcpricer/types.hpp"

namespace mcp {

/// Van der Corput radical inverse in base 2: the bits of i mirrored around
/// the binary point. It is the first coordinate of the Sobol' sequence.
[[nodiscard]] constexpr double van_der_corput(std::uint64_t i) noexcept {
    std::uint64_t r = 0;
    for (int b = 0; b < 64; ++b) {
        r = (r << 1) | (i & 1u);
        i >>= 1;
    }
    return static_cast<double>(r >> 11) * (1.0 / 9007199254740992.0);  // top 53 bits
}

/// Equal-weight quadrature of the discounted payoff on the first `num_points`
/// van der Corput points shifted by `shift` modulo 1 (one randomisation).
/// For a shift U ~ U(0, 1) this is an unbiased estimator of the price.
[[nodiscard]] double rqmc_shifted_estimate(const EuropeanOption& option, const Market& market, std::uint64_t num_points,
                                           double shift);

struct RQMCConfig {
    std::uint64_t num_points{1u << 16};    ///< points per randomisation
    std::uint32_t num_randomisations{32};  ///< independent random shifts
    std::uint64_t seed{1};
};

/// Randomised quasi-Monte Carlo price of a European option.
///
/// Each randomisation r applies a uniform random shift U_r modulo 1 to the
/// van der Corput points (Cranley-Patterson rotation), maps them through
/// the normal quantile and averages the discounted payoff. Every shifted
/// estimate is unbiased; the standard error is the dispersion of the
/// R estimates, so `num_samples` = R and `num_paths` = R * num_points.
[[nodiscard]] MCResult price_european_rqmc(const EuropeanOption& option, const Market& market,
                                           const RQMCConfig& config);

}  // namespace mcp
