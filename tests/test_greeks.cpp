#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "mcpricer/black_scholes.hpp"
#include "mcpricer/greeks.hpp"

namespace {

using mcp::Greek;
using mcp::GreekMethod;
using mcp::OptionType;

double analytic(Greek greek, const mcp::BSGreeks& g) {
    switch (greek) {
        case Greek::Delta: return g.delta;
        case Greek::Gamma: return g.gamma;
        case Greek::Vega: return g.vega;
        case Greek::Rho: return g.rho;
        case Greek::Theta: return g.theta;
    }
    return NAN;
}

class GreekEstimators : public ::testing::TestWithParam<std::tuple<Greek, GreekMethod, OptionType>> {};

TEST_P(GreekEstimators, AgreeWithClosedForm) {
    const auto [greek, method, type] = GetParam();
    const mcp::Market m{100.0, 0.05, 0.01, 0.25};
    const mcp::EuropeanOption o{type, 105.0, 0.75};
    if (!mcp::is_supported(greek, method)) {
        EXPECT_THROW((void)mcp::mc_greek(greek, method, o, m, {}), std::invalid_argument);
        return;
    }
    const mcp::GreekConfig config{1u << 18, 77, 0.0};
    const auto r = mcp::mc_greek(greek, method, o, m, config);
    const double exact = analytic(greek, mcp::bs_greeks(o, m));
    // Finite differences carry an O(h^2) bias; with the default bumps it is
    // far below the statistical error, but allow a small absolute margin.
    const double bias_margin = (method == GreekMethod::Pathwise || method == GreekMethod::LikelihoodRatio)
                                   ? 0.0
                                   : 1e-3 * std::abs(exact) + 1e-6;
    EXPECT_GT(r.std_error, 0.0);
    EXPECT_NEAR(r.value, exact, 4.0 * r.std_error + bias_margin)
        << mcp::to_string(greek) << " / " << mcp::to_string(method);
}

INSTANTIATE_TEST_SUITE_P(
    AllCombinations, GreekEstimators,
    ::testing::Combine(::testing::Values(Greek::Delta, Greek::Gamma, Greek::Vega, Greek::Rho, Greek::Theta),
                       ::testing::Values(GreekMethod::FiniteDifferenceCRN, GreekMethod::FiniteDifferenceIndependent,
                                         GreekMethod::Pathwise, GreekMethod::LikelihoodRatio),
                       ::testing::Values(OptionType::Call, OptionType::Put)));

TEST(Greeks, CommonRandomNumbersCollapseTheVariance) {
    const mcp::Market m{};
    const mcp::EuropeanOption o{};
    const mcp::GreekConfig config{1u << 16, 5, 0.5};
    for (const Greek greek : {Greek::Delta, Greek::Gamma}) {
        const auto crn = mcp::mc_greek(greek, GreekMethod::FiniteDifferenceCRN, o, m, config);
        const auto ind = mcp::mc_greek(greek, GreekMethod::FiniteDifferenceIndependent, o, m, config);
        EXPECT_LT(10.0 * crn.std_error, ind.std_error) << mcp::to_string(greek);
    }
}

TEST(Greeks, PathwiseDeltaBeatsLikelihoodRatio) {
    // For a Lipschitz payoff the pathwise estimator has the smaller variance.
    const mcp::Market m{};
    const mcp::EuropeanOption o{};
    const mcp::GreekConfig config{1u << 16, 9, 0.0};
    const auto pw = mcp::mc_greek(Greek::Delta, GreekMethod::Pathwise, o, m, config);
    const auto lr = mcp::mc_greek(Greek::Delta, GreekMethod::LikelihoodRatio, o, m, config);
    EXPECT_LT(pw.std_error, lr.std_error);
}

TEST(Greeks, BumpTooLargeIsRejected) {
    const mcp::Market m{};
    const mcp::EuropeanOption o{mcp::OptionType::Call, 100.0, 0.01};
    EXPECT_THROW((void)mcp::mc_greek(Greek::Theta, GreekMethod::FiniteDifferenceCRN, o, m, {1000, 1, 0.02}),
                 std::invalid_argument);
    EXPECT_THROW(
        (void)mcp::mc_greek(Greek::Delta, GreekMethod::FiniteDifferenceCRN, mcp::EuropeanOption{}, m, {1000, 1, 150.0}),
        std::invalid_argument);
}

}  // namespace
