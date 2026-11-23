#pragma once

// Internal helpers turning accumulated moments into an MCResult.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>

#include "mcpricer/statistics.hpp"
#include "mcpricer/types.hpp"

namespace mcp::detail {

inline void fill_interval(MCResult& r) noexcept {
    r.ci_lower = r.price - kZ975 * r.std_error;
    r.ci_upper = r.price + kZ975 * r.std_error;
    r.variance_per_path = r.std_error * r.std_error * static_cast<double>(r.num_paths);
}

/// Plain sample-mean estimator.
[[nodiscard]] inline MCResult mean_estimator(const RunningStats& s, std::uint64_t num_paths) noexcept {
    MCResult r;
    r.price = s.mean();
    r.std_error = s.std_error();
    r.num_paths = num_paths;
    r.num_samples = s.count();
    fill_interval(r);
    return r;
}

/// Control-variate estimator  Ybar - beta (Xbar - E[X])  with the
/// least-squares coefficient beta = Cov(Y, X) / Var(X) estimated on the same
/// sample. The residual variance uses n - 2 degrees of freedom (one for the
/// mean, one for beta). The bias introduced by estimating beta is O(1/n).
[[nodiscard]] inline MCResult control_variate_estimator(const RunningCovariance& s, double control_mean,
                                                        std::uint64_t num_paths) noexcept {
    const std::uint64_t n = s.count();
    const double var_x = s.var_x();
    const double beta = var_x > 0.0 ? s.cov() / var_x : 0.0;

    MCResult r;
    r.price = s.mean_y() - beta * (s.mean_x() - control_mean);
    if (n > 2) {
        const double nd = static_cast<double>(n);
        const double residual = std::max(0.0, s.var_y() - beta * s.cov()) * (nd - 1.0) / (nd - 2.0);
        r.std_error = std::sqrt(residual / nd);
    }
    r.cv_beta = beta;
    r.cv_correlation = s.corr();
    r.num_paths = num_paths;
    r.num_samples = n;
    fill_interval(r);
    return r;
}

class Stopwatch {
public:
    [[nodiscard]] double seconds() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    }

private:
    std::chrono::steady_clock::time_point start_{std::chrono::steady_clock::now()};
};

}  // namespace mcp::detail
