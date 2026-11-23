#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "mcpricer/black_scholes.hpp"
#include "mcpricer/european.hpp"
#include "mcpricer/parallel.hpp"

namespace {

using mcp::EuropeanOption;
using mcp::Market;
using mcp::MCConfig;
using mcp::OptionType;
using VR = mcp::VarianceReduction;

constexpr VR kAllMethods[] = {VR::None, VR::Antithetic, VR::ControlVariate, VR::AntitheticControlVariate};

// The pricers are deterministic for a given seed, so these statistical checks
// cannot be flaky; the 4-sigma tolerance makes them robust to a change of seed.
class EuropeanMC : public ::testing::TestWithParam<std::tuple<VR, OptionType, double>> {};

TEST_P(EuropeanMC, PriceIsWithinFourStandardErrorsOfBlackScholes) {
    const auto [method, type, strike] = GetParam();
    const Market m{100.0, 0.05, 0.02, 0.25};
    const EuropeanOption o{type, strike, 1.0};
    const double exact = mcp::bs_price(o, m);
    const mcp::MCResult r = mcp::price_european_mc(o, m, MCConfig{1u << 18, 2024, method});

    EXPECT_GT(r.std_error, 0.0);
    EXPECT_NEAR(r.price, exact, 4.0 * r.std_error) << mcp::to_string(method);
    EXPECT_LT(r.std_error, 0.01 * exact + 0.01);
    EXPECT_NEAR(r.ci_upper - r.ci_lower, 2.0 * mcp::kZ975 * r.std_error, 1e-12);
    EXPECT_EQ(r.num_paths, 1u << 18);
}

INSTANTIATE_TEST_SUITE_P(AllMethods, EuropeanMC,
                         ::testing::Combine(::testing::ValuesIn(kAllMethods),
                                            ::testing::Values(OptionType::Call, OptionType::Put),
                                            ::testing::Values(80.0, 100.0, 120.0)));

TEST(EuropeanMC, StandardErrorDecreasesAsOneOverSqrtN) {
    const Market m{};
    const EuropeanOption o{};
    for (const VR method : kAllMethods) {
        const double se_small = mcp::price_european_mc(o, m, MCConfig{1u << 16, 7, method}).std_error;
        const double se_large = mcp::price_european_mc(o, m, MCConfig{1u << 20, 7, method}).std_error;
        // 16x more paths => 4x smaller standard error.
        EXPECT_NEAR(se_small / se_large, 4.0, 0.2) << mcp::to_string(method);
    }
}

TEST(EuropeanMC, VarianceReductionOrdering) {
    const Market m{};
    const EuropeanOption o{OptionType::Call, 100.0, 1.0};
    auto variance = [&](VR method) {
        return mcp::price_european_mc(o, m, MCConfig{1u << 18, 11, method}).variance_per_path;
    };
    const double crude = variance(VR::None);
    const double anti = variance(VR::Antithetic);
    const double cv = variance(VR::ControlVariate);
    const double both = variance(VR::AntitheticControlVariate);
    // A call payoff is monotone in Z, so antithetic sampling must reduce variance.
    EXPECT_LT(anti, crude);
    // corr(payoff, S_T) is ~0.92 at the money: the control removes most of the variance.
    EXPECT_LT(cv, 0.25 * crude);
    EXPECT_LT(both, cv);
}

TEST(EuropeanMC, ControlVariateCoefficientIsSensible) {
    const Market m{};
    const auto call =
        mcp::price_european_mc({OptionType::Call, 100.0, 1.0}, m, MCConfig{1u << 18, 3, VR::ControlVariate});
    const auto put =
        mcp::price_european_mc({OptionType::Put, 100.0, 1.0}, m, MCConfig{1u << 18, 3, VR::ControlVariate});
    EXPECT_GT(call.cv_beta, 0.0);
    EXPECT_LT(call.cv_beta, 1.0);
    EXPECT_LT(put.cv_beta, 0.0);
    EXPECT_GT(put.cv_beta, -1.0);
    EXPECT_GT(call.cv_correlation, 0.85);
    EXPECT_LT(put.cv_correlation, -0.5);
}

TEST(EuropeanMC, IsReproducibleAndNested) {
    const Market m{};
    const EuropeanOption o{};
    const auto a = mcp::price_european_mc(o, m, MCConfig{100'000, 42, VR::None});
    const auto b = mcp::price_european_mc(o, m, MCConfig{100'000, 42, VR::None});
    const auto c = mcp::price_european_mc(o, m, MCConfig{100'000, 43, VR::None});
    EXPECT_EQ(a.price, b.price);
    EXPECT_EQ(a.std_error, b.std_error);
    EXPECT_NE(a.price, c.price);
}

TEST(EuropeanMC, ResultIsBitwiseIdenticalForAnyThreadCount) {
    const Market m{};
    const EuropeanOption o{};
    const int saved = mcp::max_threads();
    for (const VR method : kAllMethods) {
        mcp::set_num_threads(1);
        const auto one = mcp::price_european_mc(o, m, MCConfig{300'001, 5, method});
        mcp::set_num_threads(std::max(3, saved));
        const auto many = mcp::price_european_mc(o, m, MCConfig{300'001, 5, method});
        EXPECT_EQ(one.price, many.price) << mcp::to_string(method);
        EXPECT_EQ(one.std_error, many.std_error) << mcp::to_string(method);
    }
    mcp::set_num_threads(saved);
}

TEST(EuropeanMC, EmpiricalCoverageOfConfidenceInterval) {
    // 400 independent replications: the 95% interval should contain the true
    // price about 380 times (binomial sd ~ 4.4).
    const Market m{};
    const EuropeanOption o{};
    const double exact = mcp::bs_price(o, m);
    int covered = 0;
    constexpr int replications = 400;
    for (int k = 0; k < replications; ++k) {
        const auto r = mcp::price_european_mc(o, m, MCConfig{20'000, 1000u + static_cast<unsigned>(k), VR::None});
        covered += (r.ci_lower <= exact && exact <= r.ci_upper);
    }
    EXPECT_NEAR(covered, 0.95 * replications, 4.0 * std::sqrt(replications * 0.95 * 0.05));
}

TEST(EuropeanMC, RejectsInvalidConfiguration) {
    EXPECT_THROW((void)mcp::price_european_mc(EuropeanOption{}, Market{}, MCConfig{2, 1, VR::None}),
                 std::invalid_argument);
    EXPECT_THROW((void)mcp::price_european_mc(EuropeanOption{OptionType::Call, 100.0, -1.0}, Market{}, MCConfig{}),
                 std::invalid_argument);
}

}  // namespace
