#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "mcpricer/black_scholes.hpp"
#include "mcpricer/european.hpp"
#include "mcpricer/qmc.hpp"

namespace {

TEST(VanDerCorput, FirstTerms) {
    const double expected[] = {0.0, 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875};
    for (std::uint64_t i = 0; i < 8; ++i) EXPECT_EQ(mcp::van_der_corput(i), expected[i]);
    static_assert(mcp::van_der_corput(3) == 0.75);
}

TEST(VanDerCorput, FirstPowerOfTwoPointsFormAGrid) {
    // {vdc(0), ..., vdc(2^m - 1)} = {0, 1/2^m, ..., (2^m - 1)/2^m}: one point per stratum.
    constexpr std::uint64_t n = 1024;
    std::vector<double> pts;
    for (std::uint64_t i = 0; i < n; ++i) pts.push_back(mcp::van_der_corput(i));
    std::sort(pts.begin(), pts.end());
    for (std::uint64_t i = 0; i < n; ++i) EXPECT_EQ(pts[i], static_cast<double>(i) / static_cast<double>(n));
}

TEST(RQMC, UnbiasedAndFarMoreAccurateThanMonteCarlo) {
    const mcp::Market m{};
    const mcp::EuropeanOption o{};
    const double exact = mcp::bs_price(o, m);
    const auto q = mcp::price_european_rqmc(o, m, {1u << 14, 32, 3});
    EXPECT_NEAR(q.price, exact, 4.0 * q.std_error);
    EXPECT_EQ(q.num_paths, (1u << 14) * 32u);

    const auto mc = mcp::price_european_mc(o, m, {q.num_paths, 3, mcp::VarianceReduction::None});
    EXPECT_LT(100.0 * q.variance_per_path, mc.variance_per_path);
}

}  // namespace
