#include <gtest/gtest.h>

#include <stdexcept>

#include "mcpricer/asian.hpp"
#include "mcpricer/black_scholes.hpp"

namespace {

using mcp::AsianOption;
using mcp::Averaging;
using mcp::OptionType;
using VR = mcp::VarianceReduction;

TEST(GeometricAsian, ReferenceValue) {
    // Independent SciPy evaluation of the discrete geometric-average formula.
    const AsianOption o{OptionType::Call, 100.0, 1.0, 12, Averaging::Geometric};
    EXPECT_NEAR(mcp::geometric_asian_price(o, mcp::Market{}), 5.94020022163352, 1e-12);
}

TEST(GeometricAsian, SingleFixingIsEuropean) {
    const mcp::Market m{100.0, 0.04, 0.01, 0.3};
    for (const OptionType type : {OptionType::Call, OptionType::Put}) {
        const AsianOption a{type, 95.0, 2.0, 1, Averaging::Geometric};
        EXPECT_NEAR(mcp::geometric_asian_price(a, m), mcp::bs_price({type, 95.0, 2.0}, m), 1e-12);
    }
}

TEST(GeometricAsian, MonteCarloMatchesClosedForm) {
    const mcp::Market m{100.0, 0.05, 0.0, 0.3};
    for (const OptionType type : {OptionType::Call, OptionType::Put}) {
        const AsianOption o{type, 100.0, 1.0, 24, Averaging::Geometric};
        for (const VR method : {VR::None, VR::Antithetic}) {
            const auto r = mcp::price_asian_mc(o, m, {1u << 18, 31, method});
            EXPECT_NEAR(r.price, mcp::geometric_asian_price(o, m), 4.0 * r.std_error);
        }
    }
}

TEST(ArithmeticAsian, EstimatorsAgreeAndControlVariateIsPowerful) {
    const mcp::Market m{};
    const AsianOption o{OptionType::Call, 100.0, 1.0, 12, Averaging::Arithmetic};
    const auto crude = mcp::price_asian_mc(o, m, {1u << 18, 8, VR::None});
    const auto anti = mcp::price_asian_mc(o, m, {1u << 18, 8, VR::Antithetic});
    const auto cv = mcp::price_asian_mc(o, m, {1u << 18, 8, VR::ControlVariate});
    const auto both = mcp::price_asian_mc(o, m, {1u << 18, 8, VR::AntitheticControlVariate});

    EXPECT_NEAR(crude.price, cv.price, 4.0 * crude.std_error);
    EXPECT_NEAR(anti.price, cv.price, 4.0 * anti.std_error);
    EXPECT_NEAR(both.price, cv.price, 4.0 * (cv.std_error + both.std_error));
    // AM >= GM pathwise, so the arithmetic call is worth more than the geometric one.
    EXPECT_GT(cv.price, mcp::geometric_asian_price(o, m));
    // Kemna-Vorst control: correlation close to one, variance cut by orders of magnitude.
    EXPECT_GT(cv.cv_correlation, 0.99);
    EXPECT_GT(crude.variance_per_path / cv.variance_per_path, 100.0);
    EXPECT_NEAR(cv.cv_beta, 1.0, 0.1);
}

TEST(Asian, RejectsInvalidInputs) {
    const mcp::Market m{};
    EXPECT_THROW((void)mcp::geometric_asian_price({OptionType::Call, 100.0, 1.0, 0, Averaging::Geometric}, m),
                 std::invalid_argument);
    EXPECT_THROW((void)mcp::price_asian_mc({OptionType::Call, 100.0, 1.0, 12, Averaging::Geometric}, m,
                                           {1000, 1, VR::ControlVariate}),
                 std::invalid_argument);
}

}  // namespace
