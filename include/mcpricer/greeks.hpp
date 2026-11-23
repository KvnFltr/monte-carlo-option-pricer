#pragma once

#include <cstdint>
#include <string_view>

#include "mcpricer/types.hpp"

namespace mcp {

enum class Greek { Delta, Gamma, Vega, Rho, Theta };

enum class GreekMethod {
    FiniteDifferenceCRN,          ///< central bump-and-revalue, common random numbers
    FiniteDifferenceIndependent,  ///< central bump-and-revalue, independent draws per scenario
    Pathwise,                     ///< derivative of the payoff along each path (Broadie-Glasserman)
    LikelihoodRatio               ///< derivative of the density (score-function method)
};

[[nodiscard]] std::string_view to_string(Greek greek) noexcept;
[[nodiscard]] std::string_view to_string(GreekMethod method) noexcept;

/// Whether `method` yields an unbiased, finite-variance estimator of `greek` here.
/// Pathwise Gamma is unavailable (the payoff derivative is discontinuous) and
/// the likelihood-ratio Theta is not implemented.
[[nodiscard]] bool is_supported(Greek greek, GreekMethod method) noexcept;

/// Default bump h for finite differences: 1% of spot for Delta/Gamma,
/// 1e-3 for vol, 1e-4 for the rate and one calendar day for maturity.
[[nodiscard]] double default_bump(Greek greek, const Market& market) noexcept;

struct GreekConfig {
    std::uint64_t num_paths{1u << 20};
    std::uint64_t seed{1};
    double bump{0.0};  ///< absolute bump h; 0 selects default_bump()
};

struct GreekResult {
    double value{};
    double std_error{};
    double bump{};  ///< bump actually used (0 for pathwise / likelihood ratio)
    std::uint64_t num_paths{};
    double elapsed_seconds{};
};

/// Monte Carlo estimate of a Black-Scholes Greek (same conventions as BSGreeks).
///
/// Finite differences are central: (V(p + h) - V(p - h)) / 2h for first-order
/// Greeks and (V(S + h) - 2V(S) + V(S - h)) / h^2 for Gamma. With common random
/// numbers every scenario of path i reuses the same Gaussian draw; otherwise each
/// scenario gets its own independent stream. Theta is -dV/dT.
[[nodiscard]] GreekResult mc_greek(Greek greek, GreekMethod method, const EuropeanOption& option, const Market& market,
                                   const GreekConfig& config);

}  // namespace mcp
