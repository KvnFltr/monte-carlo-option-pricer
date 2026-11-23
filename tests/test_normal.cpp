#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "mcpricer/normal.hpp"

namespace {

TEST(Normal, PdfAndCdfReferenceValues) {
    EXPECT_NEAR(mcp::norm_pdf(0.0), 0.3989422804014327, 1e-16);
    EXPECT_DOUBLE_EQ(mcp::norm_cdf(0.0), 0.5);
    EXPECT_NEAR(mcp::norm_cdf(1.959963984540054), 0.975, 1e-15);
    EXPECT_NEAR(mcp::norm_cdf(-1.0), 0.15865525393145707, 1e-16);
    // Far tail: erfc keeps full relative precision.
    EXPECT_NEAR(mcp::norm_cdf(-10.0) / 7.61985302416047e-24, 1.0, 1e-12);
}

TEST(Normal, CdfSymmetry) {
    for (double x = -6.0; x <= 6.0; x += 0.25) {
        EXPECT_NEAR(mcp::norm_cdf(x) + mcp::norm_cdf(-x), 1.0, 1e-15) << "x = " << x;
    }
}

TEST(Normal, InverseCdfReferenceValues) {
    EXPECT_DOUBLE_EQ(mcp::norm_inv_cdf(0.5), 0.0);
    EXPECT_NEAR(mcp::norm_inv_cdf(0.975), 1.959963984540054, 1e-14);
    EXPECT_NEAR(mcp::norm_inv_cdf(0.025), -1.959963984540054, 1e-14);
    EXPECT_NEAR(mcp::norm_inv_cdf(0.001), -3.090232306167813, 1e-13);
    EXPECT_NEAR(mcp::norm_inv_cdf(1e-10), -6.361340902404056, 1e-12);
}

TEST(Normal, InverseCdfRoundTripIsAccurateToFewUlps) {
    // Relative accuracy of Phi(Phi^{-1}(p)) across ten orders of magnitude and both halves.
    for (double p = 1e-12; p < 0.5; p *= 1.7) {
        EXPECT_NEAR(mcp::norm_cdf(mcp::norm_inv_cdf(p)) / p, 1.0, 1e-13) << "p = " << p;
        if (p > 1e-4) {  // below that, 1 - p itself is not representable accurately enough
            const double x = mcp::norm_inv_cdf(1.0 - p);
            EXPECT_NEAR(mcp::norm_inv_cdf(p), -x, 1e-12 * std::abs(x)) << "p = " << p;
        }
    }
    // Phi^{-1}(Phi(x)) = x. In the upper tail Phi(x) rounds towards 1 and the
    // information is lost in the input, so the check stops at x = 5.
    for (double x = -8.0; x <= 5.0; x += 0.125) {
        EXPECT_NEAR(mcp::norm_inv_cdf(mcp::norm_cdf(x)), x, 1e-9 * (1.0 + std::abs(x))) << "x = " << x;
    }
}

TEST(Normal, InverseCdfIsMonotone) {
    double previous = -std::numeric_limits<double>::infinity();
    for (int k = 1; k < 100000; ++k) {
        const double x = mcp::norm_inv_cdf(k / 100000.0);
        EXPECT_GT(x, previous);
        previous = x;
    }
}

TEST(Normal, InverseCdfBoundaries) {
    EXPECT_EQ(mcp::norm_inv_cdf(0.0), -std::numeric_limits<double>::infinity());
    EXPECT_EQ(mcp::norm_inv_cdf(1.0), std::numeric_limits<double>::infinity());
    EXPECT_TRUE(std::isnan(mcp::norm_inv_cdf(-0.1)));
    EXPECT_TRUE(std::isnan(mcp::norm_inv_cdf(1.1)));
    EXPECT_TRUE(std::isnan(mcp::norm_inv_cdf(std::nan(""))));
}

}  // namespace
