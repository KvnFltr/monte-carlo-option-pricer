#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "mcpricer/parallel.hpp"
#include "mcpricer/statistics.hpp"

namespace {

std::vector<double> sample_data() {
    std::vector<double> v;
    for (int i = 0; i < 1000; ++i) v.push_back(1e6 + std::sin(0.37 * i) + 0.001 * i);  // large offset
    return v;
}

TEST(RunningStats, MatchesTwoPassFormulas) {
    const auto v = sample_data();
    double mean = 0.0;
    for (double x : v) mean += x;
    mean /= static_cast<double>(v.size());
    double ss = 0.0;
    for (double x : v) ss += (x - mean) * (x - mean);
    const double var = ss / static_cast<double>(v.size() - 1);

    mcp::RunningStats s;
    for (double x : v) s.add(x);
    EXPECT_EQ(s.count(), v.size());
    EXPECT_NEAR(s.mean(), mean, 1e-9);
    EXPECT_NEAR(s.variance() / var, 1.0, 1e-8);  // a naive sum of squares would lose ~12 digits here
    EXPECT_NEAR(s.std_error() / std::sqrt(var / 1000.0), 1.0, 1e-8);
}

TEST(RunningStats, MergeEqualsSequentialAccumulation) {
    const auto v = sample_data();
    mcp::RunningStats all, a, b, c;
    for (std::size_t i = 0; i < v.size(); ++i) {
        all.add(v[i]);
        (i < 123 ? a : i < 700 ? b : c).add(v[i]);
    }
    mcp::RunningStats merged;
    merged.merge(a);
    merged.merge(b);
    merged.merge(c);
    merged.merge(mcp::RunningStats{});  // merging an empty accumulator is a no-op
    EXPECT_EQ(merged.count(), all.count());
    EXPECT_NEAR(merged.mean(), all.mean(), 1e-8);  // ~1e-14 relative to the 1e6 offset
    EXPECT_NEAR(merged.variance() / all.variance(), 1.0, 1e-8);
}

TEST(RunningStats, EdgeCases) {
    mcp::RunningStats s;
    EXPECT_EQ(s.variance(), 0.0);
    EXPECT_EQ(s.std_error(), 0.0);
    s.add(3.0);
    EXPECT_EQ(s.mean(), 3.0);
    EXPECT_EQ(s.variance(), 0.0);
}

TEST(RunningCovariance, MomentsAndMerge) {
    mcp::RunningCovariance all, a, b;
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    constexpr int n = 500;
    for (int i = 0; i < n; ++i) {
        const double x = std::cos(0.1 * i) + 0.01 * i;
        const double y = 2.0 * x + std::sin(1.3 * i);
        all.add(y, x);
        (i % 3 == 0 ? a : b).add(y, x);
        sx += x;
        sy += y;
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
    }
    const double mx = sx / n, my = sy / n;
    const double vx = (sxx - n * mx * mx) / (n - 1);
    const double vy = (syy - n * my * my) / (n - 1);
    const double cxy = (sxy - n * mx * my) / (n - 1);

    EXPECT_NEAR(all.mean_x(), mx, 1e-12);
    EXPECT_NEAR(all.mean_y(), my, 1e-12);
    EXPECT_NEAR(all.var_x(), vx, 1e-10);
    EXPECT_NEAR(all.var_y(), vy, 1e-10);
    EXPECT_NEAR(all.cov(), cxy, 1e-10);
    EXPECT_NEAR(all.corr(), cxy / std::sqrt(vx * vy), 1e-12);

    a.merge(b);
    EXPECT_EQ(a.count(), all.count());
    EXPECT_NEAR(a.mean_y(), all.mean_y(), 1e-12);
    EXPECT_NEAR(a.cov(), all.cov(), 1e-10);
    EXPECT_NEAR(a.var_y(), all.var_y(), 1e-10);
}

TEST(ParallelAccumulate, CoversEveryIndexExactlyOnce) {
    for (std::uint64_t n : {1ull, 4095ull, 4096ull, 4097ull, 100'000ull}) {
        const auto s = mcp::parallel_accumulate<mcp::RunningStats>(
            n, [](std::uint64_t i, mcp::RunningStats& acc) { acc.add(static_cast<double>(i)); });
        EXPECT_EQ(s.count(), n);
        EXPECT_NEAR(s.mean(), (static_cast<double>(n) - 1.0) / 2.0, 1e-9 * static_cast<double>(n));
    }
}

TEST(ParallelAccumulate, ResultDoesNotDependOnThreadCount) {
    auto run = [] {
        return mcp::parallel_accumulate<mcp::RunningStats>(
            1'000'003, [](std::uint64_t i, mcp::RunningStats& acc) { acc.add(std::sin(static_cast<double>(i))); });
    };
    const int saved = mcp::max_threads();
    mcp::set_num_threads(1);
    const auto one = run();
    mcp::set_num_threads(std::max(2, saved));
    const auto many = run();
    mcp::set_num_threads(saved);
    // Bitwise equality, not just closeness.
    EXPECT_EQ(one.mean(), many.mean());
    EXPECT_EQ(one.variance(), many.variance());
}

}  // namespace
