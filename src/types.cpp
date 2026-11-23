#include "mcpricer/types.hpp"

#include <cmath>
#include <stdexcept>

namespace mcp {

void validate(const Market& market) {
    if (!(market.spot > 0.0) || !std::isfinite(market.spot)) {
        throw std::invalid_argument("spot must be positive and finite");
    }
    if (!(market.vol > 0.0) || !std::isfinite(market.vol)) {
        throw std::invalid_argument("volatility must be positive and finite");
    }
    if (!std::isfinite(market.rate)) throw std::invalid_argument("rate must be finite");
    if (!std::isfinite(market.dividend)) throw std::invalid_argument("dividend yield must be finite");
}

void validate(const EuropeanOption& option) {
    if (!(option.strike > 0.0) || !std::isfinite(option.strike)) {
        throw std::invalid_argument("strike must be positive and finite");
    }
    if (!(option.maturity > 0.0) || !std::isfinite(option.maturity)) {
        throw std::invalid_argument("maturity must be positive and finite");
    }
}

}  // namespace mcp
