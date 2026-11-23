#pragma once

#include <cstdint>
#include <limits>

namespace mcp {

enum class OptionType { Call, Put };

/// Black-Scholes market with a continuous dividend yield.
/// All rates are continuously compounded and annualised.
struct Market {
    double spot{100.0};    ///< S_0
    double rate{0.05};     ///< risk-free rate r
    double dividend{0.0};  ///< dividend yield q
    double vol{0.2};       ///< volatility sigma
};

/// Plain-vanilla European option.
struct EuropeanOption {
    OptionType type{OptionType::Call};
    double strike{100.0};  ///< K
    double maturity{1.0};  ///< T, in years
};

/// Throws std::invalid_argument if a parameter is outside its domain.
void validate(const Market& market);
void validate(const EuropeanOption& option);

/// Undiscounted vanilla payoff (S - K)^+ or (K - S)^+.
[[nodiscard]] inline double payoff(OptionType type, double spot, double strike) noexcept {
    const double intrinsic = type == OptionType::Call ? spot - strike : strike - spot;
    return intrinsic > 0.0 ? intrinsic : 0.0;
}

/// Output of every Monte Carlo pricing routine.
///
/// `variance_per_path` is N * std_error^2, i.e. the variance of the estimator
/// normalised by the number of simulated paths. It is the quantity to compare
/// across variance-reduction schemes at equal simulation budget.
struct MCResult {
    double price{};
    double std_error{};
    double ci_lower{};  ///< 95% asymptotic confidence interval
    double ci_upper{};
    double variance_per_path{};
    double cv_beta{std::numeric_limits<double>::quiet_NaN()};         ///< estimated control-variate coefficient
    double cv_correlation{std::numeric_limits<double>::quiet_NaN()};  ///< corr(payoff, control)
    std::uint64_t num_paths{};    ///< number of simulated paths (payoff evaluations)
    std::uint64_t num_samples{};  ///< number of i.i.d. samples behind the standard error
    double elapsed_seconds{};
};

/// 97.5% quantile of the standard normal distribution.
inline constexpr double kZ975 = 1.959963984540054;

}  // namespace mcp
