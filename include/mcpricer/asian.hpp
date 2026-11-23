#pragma once

#include <cstdint>

#include "mcpricer/european.hpp"
#include "mcpricer/types.hpp"

namespace mcp {

enum class Averaging { Arithmetic, Geometric };

/// Fixed-strike Asian option on the average of n equally spaced fixings
/// t_k = k T / n, k = 1, ..., n.
struct AsianOption {
    OptionType type{OptionType::Call};
    double strike{100.0};
    double maturity{1.0};
    std::uint32_t num_fixings{12};
    Averaging averaging{Averaging::Arithmetic};
};

void validate(const AsianOption& option);

/// Closed-form price of the discretely monitored geometric-average Asian option.
/// log G is Gaussian with mean log S_0 + (r - q - sigma^2/2) T (n+1)/(2n)
/// and variance sigma^2 T (n+1)(2n+1)/(6n^2), so the price is Black-like.
/// The `averaging` field of `option` is ignored.
[[nodiscard]] double geometric_asian_price(const AsianOption& option, const Market& market);

/// Monte Carlo price of an Asian option. Paths are simulated exactly on the
/// fixing grid. For an arithmetic option, the control variate is the
/// geometric Asian option with the same strike (Kemna & Vorst, 1990).
/// Control variates are rejected for a geometric option.
[[nodiscard]] MCResult price_asian_mc(const AsianOption& option, const Market& market, const MCConfig& config);

}  // namespace mcp
