#pragma once

#include <cstdint>
#include <string_view>

#include "mcpricer/types.hpp"

namespace mcp {

enum class VarianceReduction {
    None,                     ///< crude Monte Carlo
    Antithetic,               ///< pairs (Z, -Z)
    ControlVariate,           ///< control e^{-rT} S_T with regression coefficient
    AntitheticControlVariate  ///< both, the control acting on pair averages
};

[[nodiscard]] std::string_view to_string(VarianceReduction method) noexcept;

struct MCConfig {
    std::uint64_t num_paths{1u << 20};  ///< simulation budget: number of payoff evaluations
    std::uint64_t seed{1};
    VarianceReduction variance_reduction{VarianceReduction::None};
};

/// Monte Carlo price of a European option under geometric Brownian motion.
///
/// S_T is sampled exactly, S_T = S_0 exp((r - q - sigma^2/2) T + sigma sqrt(T) Z),
/// so the only error is statistical. Every method uses the same budget of
/// `num_paths` terminal prices: antithetic sampling draws num_paths / 2
/// independent pairs (Z, -Z). The control variate is the discounted terminal
/// price e^{-rT} S_T, whose expectation S_0 e^{-qT} is known.
///
/// Path i is driven by the random stream (seed, i), so a run with N paths
/// reproduces exactly the first N paths of any longer run with the same seed.
[[nodiscard]] MCResult price_european_mc(const EuropeanOption& option, const Market& market, const MCConfig& config);

}  // namespace mcp
