#include <gtest/gtest.h>

#include <cmath>
#include <set>

#include "mcpricer/rng.hpp"
#include "mcpricer/statistics.hpp"

namespace {

using Counter = mcp::Philox4x32::Counter;
using Key = mcp::Philox4x32::Key;

// Known-answer tests from the Random123 distribution (kat_vectors, philox4x32 R=10).
TEST(Philox, KnownAnswerVectors) {
    EXPECT_EQ(mcp::Philox4x32::apply(Counter{0, 0, 0, 0}, Key{0, 0}),
              (Counter{0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u}));
    EXPECT_EQ(mcp::Philox4x32::apply(Counter{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu},
                                     Key{0xffffffffu, 0xffffffffu}),
              (Counter{0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu}));
    EXPECT_EQ(mcp::Philox4x32::apply(Counter{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u},
                                     Key{0xa4093822u, 0x299f31d0u}),
              (Counter{0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u}));
}

TEST(Philox, IsAConstantExpression) {
    constexpr Counter out = mcp::Philox4x32::apply(Counter{0, 0, 0, 0}, Key{0, 0});
    static_assert(out[0] == 0x6627e8d5u);
}

TEST(RandomStream, IsDeterministic) {
    mcp::RandomStream a(123, 7), b(123, 7);
    for (int i = 0; i < 1000; ++i) EXPECT_EQ(a.next_u64(), b.next_u64());
}

TEST(RandomStream, StreamsSeedsAndSubstreamsDiffer) {
    std::set<std::uint64_t> first_draws;
    for (std::uint64_t seed = 0; seed < 4; ++seed) {
        for (std::uint64_t stream = 0; stream < 64; ++stream) {
            for (std::uint32_t sub = 0; sub < 4; ++sub) {
                mcp::RandomStream rng(seed, stream, sub);
                first_draws.insert(rng.next_u64());
            }
        }
    }
    EXPECT_EQ(first_draws.size(), 4u * 64u * 4u);
}

TEST(RandomStream, UniformsLieInOpenUnitIntervalWithCorrectMoments) {
    mcp::RunningStats stats;
    mcp::RandomStream rng(2024, 0);
    constexpr int n = 1'000'000;
    for (int i = 0; i < n; ++i) {
        const double u = rng.uniform();
        ASSERT_GT(u, 0.0);
        ASSERT_LT(u, 1.0);
        stats.add(u);
    }
    // mean 1/2 and variance 1/12; tolerances are about 5 standard errors.
    EXPECT_NEAR(stats.mean(), 0.5, 5.0 * std::sqrt(1.0 / 12.0 / n));
    EXPECT_NEAR(stats.variance(), 1.0 / 12.0, 5.0 * std::sqrt(1.0 / 180.0 / n));
}

TEST(RandomStream, NormalsHaveStandardMoments) {
    mcp::RunningStats m1, m2, m3, m4;
    constexpr int n = 1'000'000;
    for (int i = 0; i < n; ++i) {
        // one draw per stream, exactly as in the pricers
        mcp::RandomStream rng(99, static_cast<std::uint64_t>(i));
        const double z = rng.normal();
        m1.add(z);
        m2.add(z * z);
        m3.add(z * z * z);
        m4.add(z * z * z * z);
    }
    EXPECT_NEAR(m1.mean(), 0.0, 5.0 * std::sqrt(1.0 / n));
    EXPECT_NEAR(m2.mean(), 1.0, 5.0 * std::sqrt(2.0 / n));
    EXPECT_NEAR(m3.mean(), 0.0, 5.0 * std::sqrt(15.0 / n));
    EXPECT_NEAR(m4.mean(), 3.0, 5.0 * std::sqrt(96.0 / n));
}

TEST(RandomStream, NormalTailFrequenciesMatchTheory) {
    // P(Z > 3) = 1.3499e-3; a chi-square style check on a few bins.
    constexpr int n = 2'000'000;
    int above3 = 0, below_minus2 = 0;
    mcp::RandomStream rng(5, 5);
    for (int i = 0; i < n; ++i) {
        const double z = rng.normal();
        above3 += z > 3.0;
        below_minus2 += z < -2.0;
    }
    const double p3 = 1.3498980316301e-3, p2 = 2.2750131948179e-2;
    EXPECT_NEAR(above3, n * p3, 5.0 * std::sqrt(n * p3 * (1 - p3)));
    EXPECT_NEAR(below_minus2, n * p2, 5.0 * std::sqrt(n * p2 * (1 - p2)));
}

}  // namespace
