#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "mcpricer/black_scholes.hpp"

namespace {

using mcp::EuropeanOption;
using mcp::Market;
using mcp::OptionType;

TEST(BlackScholes, ReferencePrices) {
    // Reference values computed independently with SciPy.
    const Market m{100.0, 0.05, 0.0, 0.2};
    EXPECT_NEAR(mcp::bs_price({OptionType::Call, 100.0, 1.0}, m), 10.450583572185565, 1e-12);
    EXPECT_NEAR(mcp::bs_price({OptionType::Put, 100.0, 1.0}, m), 5.573526022256971, 1e-12);

    // Hull, Options, Futures and Other Derivatives: S=42, K=40, r=10%, sigma=20%, T=6 months.
    const Market hull{42.0, 0.10, 0.0, 0.2};
    EXPECT_NEAR(mcp::bs_price({OptionType::Call, 40.0, 0.5}, hull), 4.759422392871532, 1e-12);
    EXPECT_NEAR(mcp::bs_price({OptionType::Put, 40.0, 0.5}, hull), 0.8085993729000922, 1e-12);

    // With a dividend yield.
    const Market div{100.0, 0.03, 0.02, 0.25};
    EXPECT_NEAR(mcp::bs_price({OptionType::Call, 95.0, 0.75}, div), 11.363171865840123, 1e-12);
    EXPECT_NEAR(mcp::bs_price({OptionType::Put, 95.0, 0.75}, div), 5.738345438900808, 1e-12);
}

TEST(BlackScholes, PutCallParity) {
    for (double s : {60.0, 90.0, 100.0, 130.0}) {
        for (double q : {0.0, 0.03}) {
            for (double t : {0.1, 1.0, 5.0}) {
                const Market m{s, 0.04, q, 0.3};
                const double c = mcp::bs_price({OptionType::Call, 100.0, t}, m);
                const double p = mcp::bs_price({OptionType::Put, 100.0, t}, m);
                EXPECT_NEAR(c - p, s * std::exp(-q * t) - 100.0 * std::exp(-0.04 * t), 1e-11);
            }
        }
    }
}

// Each closed-form Greek must agree with a central difference of the closed-form price.
class GreeksVsFiniteDifferences : public ::testing::TestWithParam<std::tuple<OptionType, double, double>> {};

TEST_P(GreeksVsFiniteDifferences, Agree) {
    const auto [type, strike, dividend] = GetParam();
    const Market m{100.0, 0.05, dividend, 0.25};
    const EuropeanOption o{type, strike, 0.8};
    const mcp::BSGreeks g = mcp::bs_greeks(o, m);

    EXPECT_NEAR(g.price, mcp::bs_price(o, m), 1e-13);

    auto price_with = [&](double ds, double dvol, double dr, double dt) {
        Market mm = m;
        mm.spot += ds;
        mm.vol += dvol;
        mm.rate += dr;
        EuropeanOption oo = o;
        oo.maturity += dt;
        return mcp::bs_price(oo, mm);
    };
    const double hs = 1e-3, hg = 0.05, hv = 1e-5, hr = 1e-6, ht = 1e-6;
    EXPECT_NEAR(g.delta, (price_with(hs, 0, 0, 0) - price_with(-hs, 0, 0, 0)) / (2 * hs), 1e-8);
    EXPECT_NEAR(g.gamma, (price_with(hg, 0, 0, 0) - 2 * g.price + price_with(-hg, 0, 0, 0)) / (hg * hg), 1e-7);
    EXPECT_NEAR(g.vega, (price_with(0, hv, 0, 0) - price_with(0, -hv, 0, 0)) / (2 * hv), 1e-6);
    EXPECT_NEAR(g.rho, (price_with(0, 0, hr, 0) - price_with(0, 0, -hr, 0)) / (2 * hr), 1e-5);
    EXPECT_NEAR(g.theta, -(price_with(0, 0, 0, ht) - price_with(0, 0, 0, -ht)) / (2 * ht), 1e-5);
}

INSTANTIATE_TEST_SUITE_P(AllCases, GreeksVsFiniteDifferences,
                         ::testing::Combine(::testing::Values(OptionType::Call, OptionType::Put),
                                            ::testing::Values(80.0, 100.0, 125.0), ::testing::Values(0.0, 0.02)));

TEST(BlackScholes, CallPutGreekRelations) {
    const Market m{100.0, 0.05, 0.02, 0.2};
    const auto c = mcp::bs_greeks({OptionType::Call, 105.0, 1.5}, m);
    const auto p = mcp::bs_greeks({OptionType::Put, 105.0, 1.5}, m);
    EXPECT_NEAR(c.delta - p.delta, std::exp(-0.02 * 1.5), 1e-14);
    EXPECT_NEAR(c.gamma, p.gamma, 1e-15);
    EXPECT_NEAR(c.vega, p.vega, 1e-12);
    EXPECT_NEAR(c.rho - p.rho, 105.0 * 1.5 * std::exp(-0.05 * 1.5), 1e-11);
}

TEST(BlackScholes, AtTheMoneyGreeksReferenceValues) {
    const auto g = mcp::bs_greeks({OptionType::Call, 100.0, 1.0}, Market{100.0, 0.05, 0.0, 0.2});
    // d1 = 0.35, d2 = 0.15
    EXPECT_NEAR(g.delta, 0.6368306511756191, 1e-14);
    EXPECT_NEAR(g.gamma, 0.018762017345846895, 1e-15);
    EXPECT_NEAR(g.vega, 37.52403469169379, 1e-11);
}

TEST(BlackScholes, Limits) {
    const Market m{100.0, 0.05, 0.01, 0.2};
    // Deep in the money: the call is worth its discounted forward intrinsic value.
    EXPECT_NEAR(mcp::bs_price({OptionType::Call, 1.0, 1.0}, m), 100.0 * std::exp(-0.01) - std::exp(-0.05), 1e-12);
    // Deep out of the money: worthless.
    EXPECT_LT(mcp::bs_price({OptionType::Call, 1000.0, 1.0}, m), 1e-20);
    // Strictly increasing in volatility.
    double previous = 0.0;
    for (double vol = 0.05; vol < 1.0; vol += 0.05) {
        const double price = mcp::bs_price({OptionType::Call, 100.0, 1.0}, Market{100.0, 0.05, 0.01, vol});
        EXPECT_GT(price, previous);
        previous = price;
    }
}

TEST(BlackScholes, RejectsInvalidInputs) {
    const Market m{};
    EXPECT_THROW((void)mcp::bs_price({OptionType::Call, 100.0, 0.0}, m), std::invalid_argument);
    EXPECT_THROW((void)mcp::bs_price({OptionType::Call, -1.0, 1.0}, m), std::invalid_argument);
    EXPECT_THROW((void)mcp::bs_price({OptionType::Call, 100.0, 1.0}, Market{0.0, 0.05, 0.0, 0.2}),
                 std::invalid_argument);
    EXPECT_THROW((void)mcp::bs_price({OptionType::Call, 100.0, 1.0}, Market{100.0, 0.05, 0.0, 0.0}),
                 std::invalid_argument);
    EXPECT_THROW((void)mcp::bs_price({OptionType::Call, 100.0, 1.0}, Market{100.0, NAN, 0.0, 0.2}),
                 std::invalid_argument);
}

}  // namespace
