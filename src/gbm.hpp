#pragma once

// Internal: exact sampling of the terminal price of a geometric Brownian motion.

#include <cmath>

#include "mcpricer/types.hpp"

namespace mcp::detail {

/// S_T = S_0 exp((r - q - sigma^2/2) T + sigma sqrt(T) Z), with discount factor e^{-rT}.
struct TerminalGBM {
    TerminalGBM(const Market& m, double maturity) noexcept
        : spot(m.spot),
          drift((m.rate - m.dividend - 0.5 * m.vol * m.vol) * maturity),
          vol_sqrt_t(m.vol * std::sqrt(maturity)),
          discount(std::exp(-m.rate * maturity)) {}

    [[nodiscard]] double terminal(double z) const noexcept { return spot * std::exp(drift + vol_sqrt_t * z); }

    double spot;
    double drift;
    double vol_sqrt_t;
    double discount;
};

}  // namespace mcp::detail
