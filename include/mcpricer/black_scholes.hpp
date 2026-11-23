#pragma once

#include "mcpricer/types.hpp"

namespace mcp {

/// Price and sensitivities of a European option in the Black-Scholes model.
///
/// Conventions: vega and rho per unit change (not per 1%), theta is the
/// derivative with respect to calendar time, dV/dt = -dV/dT, per year.
struct BSGreeks {
    double price{};
    double delta{};
    double gamma{};
    double vega{};
    double theta{};
    double rho{};
};

/// Closed-form Black-Scholes(-Merton) price.
[[nodiscard]] double bs_price(const EuropeanOption& option, const Market& market);

/// Closed-form price and Greeks.
[[nodiscard]] BSGreeks bs_greeks(const EuropeanOption& option, const Market& market);

}  // namespace mcp
