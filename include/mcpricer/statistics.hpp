#pragma once

#include <cmath>
#include <cstdint>

namespace mcp {

/// One-pass mean / variance accumulator (Welford, 1962).
/// `merge` implements the pairwise update of Chan, Golub & LeVeque (1983),
/// so partial results computed by different threads can be combined exactly.
class RunningStats {
public:
    void add(double x) noexcept {
        ++n_;
        const double delta = x - mean_;
        mean_ += delta / static_cast<double>(n_);
        m2_ += delta * (x - mean_);
    }

    void merge(const RunningStats& other) noexcept {
        if (other.n_ == 0) return;
        if (n_ == 0) {
            *this = other;
            return;
        }
        const double na = static_cast<double>(n_);
        const double nb = static_cast<double>(other.n_);
        const double n = na + nb;
        const double delta = other.mean_ - mean_;
        mean_ += delta * nb / n;
        m2_ += other.m2_ + delta * delta * na * nb / n;
        n_ += other.n_;
    }

    [[nodiscard]] std::uint64_t count() const noexcept { return n_; }
    [[nodiscard]] double mean() const noexcept { return mean_; }
    /// Unbiased sample variance.
    [[nodiscard]] double variance() const noexcept { return n_ > 1 ? m2_ / static_cast<double>(n_ - 1) : 0.0; }
    /// Standard error of the sample mean.
    [[nodiscard]] double std_error() const noexcept {
        return n_ > 0 ? std::sqrt(variance() / static_cast<double>(n_)) : 0.0;
    }

private:
    std::uint64_t n_{0};
    double mean_{0.0};
    double m2_{0.0};
};

/// One-pass accumulator of the first two moments of a pair (Y, X):
/// means, variances and covariance. Used for control variates.
class RunningCovariance {
public:
    void add(double y, double x) noexcept {
        ++n_;
        const double n = static_cast<double>(n_);
        const double dx = x - mean_x_;
        mean_x_ += dx / n;
        const double dy = y - mean_y_;
        mean_y_ += dy / n;
        m2_x_ += dx * (x - mean_x_);
        m2_y_ += dy * (y - mean_y_);
        c_xy_ += dx * (y - mean_y_);
    }

    void merge(const RunningCovariance& other) noexcept {
        if (other.n_ == 0) return;
        if (n_ == 0) {
            *this = other;
            return;
        }
        const double na = static_cast<double>(n_);
        const double nb = static_cast<double>(other.n_);
        const double n = na + nb;
        const double dx = other.mean_x_ - mean_x_;
        const double dy = other.mean_y_ - mean_y_;
        const double w = na * nb / n;
        mean_x_ += dx * nb / n;
        mean_y_ += dy * nb / n;
        m2_x_ += other.m2_x_ + dx * dx * w;
        m2_y_ += other.m2_y_ + dy * dy * w;
        c_xy_ += other.c_xy_ + dx * dy * w;
        n_ += other.n_;
    }

    [[nodiscard]] std::uint64_t count() const noexcept { return n_; }
    [[nodiscard]] double mean_y() const noexcept { return mean_y_; }
    [[nodiscard]] double mean_x() const noexcept { return mean_x_; }
    [[nodiscard]] double var_y() const noexcept { return n_ > 1 ? m2_y_ / dof() : 0.0; }
    [[nodiscard]] double var_x() const noexcept { return n_ > 1 ? m2_x_ / dof() : 0.0; }
    [[nodiscard]] double cov() const noexcept { return n_ > 1 ? c_xy_ / dof() : 0.0; }
    [[nodiscard]] double corr() const noexcept {
        const double denom = std::sqrt(m2_x_ * m2_y_);
        return denom > 0.0 ? c_xy_ / denom : 0.0;
    }

private:
    [[nodiscard]] double dof() const noexcept { return static_cast<double>(n_ - 1); }

    std::uint64_t n_{0};
    double mean_y_{0.0};
    double mean_x_{0.0};
    double m2_y_{0.0};
    double m2_x_{0.0};
    double c_xy_{0.0};
};

}  // namespace mcp
