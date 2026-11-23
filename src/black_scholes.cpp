#include "mcpricer/black_scholes.hpp"

#include <cmath>

#include "mcpricer/normal.hpp"

namespace mcp {

namespace {

struct D12 {
    double d1;
    double d2;
};

D12 compute_d12(const EuropeanOption& o, const Market& m) {
    const double vol_sqrt_t = m.vol * std::sqrt(o.maturity);
    const double d1 =
        (std::log(m.spot / o.strike) + (m.rate - m.dividend + 0.5 * m.vol * m.vol) * o.maturity) / vol_sqrt_t;
    return {d1, d1 - vol_sqrt_t};
}

}  // namespace

double bs_price(const EuropeanOption& option, const Market& market) {
    validate(option);
    validate(market);
    const auto [d1, d2] = compute_d12(option, market);
    const double df = std::exp(-market.rate * option.maturity);
    const double fwd_df = market.spot * std::exp(-market.dividend * option.maturity);
    if (option.type == OptionType::Call) {
        return fwd_df * norm_cdf(d1) - option.strike * df * norm_cdf(d2);
    }
    return option.strike * df * norm_cdf(-d2) - fwd_df * norm_cdf(-d1);
}

BSGreeks bs_greeks(const EuropeanOption& option, const Market& market) {
    validate(option);
    validate(market);
    const double s = market.spot;
    const double k = option.strike;
    const double t = option.maturity;
    const double r = market.rate;
    const double q = market.dividend;
    const double sigma = market.vol;
    const double sqrt_t = std::sqrt(t);

    const auto [d1, d2] = compute_d12(option, market);
    const double df = std::exp(-r * t);
    const double div_df = std::exp(-q * t);
    const double pdf_d1 = norm_pdf(d1);

    BSGreeks g;
    g.gamma = div_df * pdf_d1 / (s * sigma * sqrt_t);
    g.vega = s * div_df * pdf_d1 * sqrt_t;
    const double theta_common = -s * div_df * pdf_d1 * sigma / (2.0 * sqrt_t);

    if (option.type == OptionType::Call) {
        const double nd1 = norm_cdf(d1);
        const double nd2 = norm_cdf(d2);
        g.price = s * div_df * nd1 - k * df * nd2;
        g.delta = div_df * nd1;
        g.theta = theta_common - r * k * df * nd2 + q * s * div_df * nd1;
        g.rho = k * t * df * nd2;
    } else {
        const double nmd1 = norm_cdf(-d1);
        const double nmd2 = norm_cdf(-d2);
        g.price = k * df * nmd2 - s * div_df * nmd1;
        g.delta = -div_df * nmd1;
        g.theta = theta_common + r * k * df * nmd2 - q * s * div_df * nmd1;
        g.rho = -k * t * df * nmd2;
    }
    return g;
}

}  // namespace mcp
