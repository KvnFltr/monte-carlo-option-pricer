// Runs every numerical experiment of the report and writes the raw results as CSV.
//
//   run_experiments [output_dir] [--quick]
//
// --quick shrinks every experiment by about two orders of magnitude (smoke test).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "mcpricer/mcpricer.hpp"

namespace fs = std::filesystem;
using VR = mcp::VarianceReduction;

namespace {

constexpr VR kMethods[] = {VR::None, VR::Antithetic, VR::ControlVariate, VR::AntitheticControlVariate};

// Base case used throughout the report.
const mcp::Market kMarket{100.0, 0.05, 0.0, 0.2};
const mcp::EuropeanOption kCall{mcp::OptionType::Call, 100.0, 1.0};

struct Scale {
    std::uint64_t max_convergence_paths;
    int replications;
    std::uint64_t table_paths;
    std::uint64_t greek_bump_paths;
    int greek_replications;
    std::uint64_t asian_paths;
    std::uint64_t asian_reference_paths;
    std::uint64_t performance_paths;
};

constexpr Scale kFull{1u << 20, 500, 1u << 22, 1u << 16, 200, 1u << 20, 1u << 25, 1u << 24};
constexpr Scale kQuick{1u << 14, 40, 1u << 16, 1u << 12, 20, 1u << 14, 1u << 18, 1u << 18};

/// Minimal CSV writer: header once, then rows of numbers and strings.
class Csv {
public:
    Csv(const fs::path& path, const char* header) : file_(path) {
        if (!file_) throw std::runtime_error("cannot open " + path.string());
        file_ << std::setprecision(12) << header << '\n';
    }

    template <class... Ts>
    void row(const Ts&... values) {
        const char* sep = "";
        ((file_ << sep << values, sep = ","), ...);
        file_ << '\n';
    }

private:
    std::ofstream file_;
};

std::vector<std::uint64_t> powers_of_two(int lo, int hi) {
    std::vector<std::uint64_t> v;
    for (int k = lo; k <= hi; ++k) v.push_back(std::uint64_t{1} << k);
    return v;
}

int log2_floor(std::uint64_t n) {
    int k = 0;
    while (n >>= 1) ++k;
    return k;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// ------------------------------------------------------------------ 1. convergence
void convergence(const fs::path& out, const Scale& s) {
    std::printf("[1/7] convergence in 1/sqrt(N) and coverage of the 95%% interval\n");
    const double exact = mcp::bs_price(kCall, kMarket);
    Csv csv(out / "convergence.csv",
            "method,n_paths,replications,rmse,mean_abs_error,mean_std_error,coverage_95,mean_time_s");
    for (const VR method : kMethods) {
        for (const std::uint64_t n : powers_of_two(7, log2_floor(s.max_convergence_paths))) {
            mcp::RunningStats sq_err, abs_err, se, time;
            int covered = 0;
            for (int r = 0; r < s.replications; ++r) {
                const auto res =
                    mcp::price_european_mc(kCall, kMarket, {n, 1'000'000u + static_cast<std::uint64_t>(r), method});
                const double err = res.price - exact;
                sq_err.add(err * err);
                abs_err.add(std::abs(err));
                se.add(res.std_error);
                time.add(res.elapsed_seconds);
                covered += (res.ci_lower <= exact && exact <= res.ci_upper);
            }
            csv.row(mcp::to_string(method), n, s.replications, std::sqrt(sq_err.mean()), abs_err.mean(), se.mean(),
                    static_cast<double>(covered) / s.replications, time.mean());
        }
    }
}

// ------------------------------------------------------------------ 2. running estimate
void running_estimate(const fs::path& out, const Scale& s) {
    std::printf("[2/7] running estimate of a single simulation\n");
    Csv csv(out / "running_estimate.csv", "method,n_paths,price,std_error,ci_lower,ci_upper");
    const int max_log = log2_floor(s.max_convergence_paths) + 2;
    for (const VR method : {VR::None, VR::AntitheticControlVariate}) {
        // Nested runs: with a counter-based RNG the run of size N is the prefix of every longer run.
        for (int k = 4 * 5; k <= 4 * max_log; ++k) {
            const auto n = static_cast<std::uint64_t>(std::llround(std::pow(2.0, k / 4.0)));
            const auto r = mcp::price_european_mc(kCall, kMarket, {n, 2024, method});
            csv.row(mcp::to_string(method), n, r.price, r.std_error, r.ci_lower, r.ci_upper);
        }
    }
}

// ------------------------------------------------------------------ 3. variance reduction
void variance_reduction(const fs::path& out, const Scale& s) {
    std::printf("[3/7] variance reduction across strikes\n");
    Csv csv(out / "variance_reduction.csv",
            "type,strike,method,price,std_error,bs_price,error,variance_per_path,vrf,time_s,efficiency_gain,"
            "cv_beta,cv_corr");
    for (const auto type : {mcp::OptionType::Call, mcp::OptionType::Put}) {
        for (double strike = 70.0; strike <= 130.0 + 1e-9; strike += 5.0) {
            const mcp::EuropeanOption o{type, strike, 1.0};
            const double exact = mcp::bs_price(o, kMarket);
            // Methods are interleaved and timed five times each so that CPU frequency
            // drift affects them alike; the median time is reported.
            constexpr std::size_t kNumMethods = std::size(kMethods);
            mcp::MCResult results[kNumMethods];
            std::vector<double> times[kNumMethods];
            for (int rep = 0; rep < 5; ++rep) {
                for (std::size_t j = 0; j < kNumMethods; ++j) {
                    results[j] = mcp::price_european_mc(o, kMarket, {s.table_paths, 7, kMethods[j]});
                    times[j].push_back(results[j].elapsed_seconds);
                }
            }
            const double crude_var = results[0].variance_per_path;
            const double crude_time = median(times[0]);
            for (std::size_t j = 0; j < kNumMethods; ++j) {
                const mcp::MCResult& r = results[j];
                const double t = median(times[j]);
                csv.row(type == mcp::OptionType::Call ? "call" : "put", strike, mcp::to_string(kMethods[j]), r.price,
                        r.std_error, exact, r.price - exact, r.variance_per_path, crude_var / r.variance_per_path, t,
                        (crude_var * crude_time) / (r.variance_per_path * t), r.cv_beta, r.cv_correlation);
            }
        }
    }
}

// ------------------------------------------------------------------ 4. Greeks
void greeks(const fs::path& out, const Scale& s) {
    std::printf("[4/7] Greeks: estimator comparison and bump-size study\n");
    const mcp::BSGreeks exact = mcp::bs_greeks(kCall, kMarket);
    auto analytic = [&](mcp::Greek g) {
        switch (g) {
            case mcp::Greek::Delta: return exact.delta;
            case mcp::Greek::Gamma: return exact.gamma;
            case mcp::Greek::Vega: return exact.vega;
            case mcp::Greek::Rho: return exact.rho;
            case mcp::Greek::Theta: return exact.theta;
        }
        return std::numeric_limits<double>::quiet_NaN();
    };
    constexpr mcp::Greek kGreeks[] = {mcp::Greek::Delta, mcp::Greek::Gamma, mcp::Greek::Vega, mcp::Greek::Rho,
                                      mcp::Greek::Theta};
    constexpr mcp::GreekMethod kGreekMethods[] = {mcp::GreekMethod::FiniteDifferenceCRN,
                                                  mcp::GreekMethod::FiniteDifferenceIndependent,
                                                  mcp::GreekMethod::Pathwise, mcp::GreekMethod::LikelihoodRatio};

    {
        Csv csv(out / "greeks_table.csv", "greek,method,value,std_error,analytic,bump,time_s");
        for (const auto g : kGreeks) {
            for (const auto m : kGreekMethods) {
                if (!mcp::is_supported(g, m)) continue;
                const auto r = mcp::mc_greek(g, m, kCall, kMarket, {s.table_paths, 11, 0.0});
                csv.row(mcp::to_string(g), mcp::to_string(m), r.value, r.std_error, analytic(g), r.bump,
                        r.elapsed_seconds);
            }
        }
    }

    // Bias / variance trade-off of the bump size h, with and without common random numbers.
    Csv csv(out / "greeks_bump.csv", "greek,method,h_rel,h,mean,bias,std,rmse,analytic,n_paths,replications");
    auto study = [&](mcp::Greek g, mcp::GreekMethod m, double h_rel) {
        const double h = h_rel * kMarket.spot;
        mcp::RunningStats est;
        for (int r = 0; r < s.greek_replications; ++r) {
            est.add(mcp::mc_greek(g, m, kCall, kMarket, {s.greek_bump_paths, 5000u + static_cast<std::uint64_t>(r), h})
                        .value);
        }
        const double bias = est.mean() - analytic(g);
        const double sd = std::sqrt(est.variance());
        csv.row(mcp::to_string(g), mcp::to_string(m), h_rel, h, est.mean(), bias, sd, std::sqrt(bias * bias + sd * sd),
                analytic(g), s.greek_bump_paths, s.greek_replications);
    };
    for (const auto g : {mcp::Greek::Delta, mcp::Greek::Gamma}) {
        for (const auto m : {mcp::GreekMethod::FiniteDifferenceCRN, mcp::GreekMethod::FiniteDifferenceIndependent}) {
            for (int k = 0; k <= 20; ++k) {
                study(g, m, std::pow(10.0, -4.0 + 0.175 * k));  // h/S0 from 1e-4 to ~0.32
            }
        }
    }
    // Unbiased benchmarks at the same budget (no bump).
    study(mcp::Greek::Delta, mcp::GreekMethod::Pathwise, NAN);
    study(mcp::Greek::Delta, mcp::GreekMethod::LikelihoodRatio, NAN);
    study(mcp::Greek::Gamma, mcp::GreekMethod::LikelihoodRatio, NAN);
}

// ------------------------------------------------------------------ 5. Asian option
void asian(const fs::path& out, const Scale& s) {
    std::printf("[5/7] arithmetic Asian option with the geometric control variate\n");
    Csv csv(out / "asian.csv",
            "strike,num_fixings,method,price,std_error,reference,reference_std_error,geometric_price,variance_per_path,"
            "vrf,time_s,efficiency_gain,cv_beta,cv_corr");
    for (const double strike : {90.0, 100.0, 110.0}) {
        const mcp::AsianOption o{mcp::OptionType::Call, strike, 1.0, 12, mcp::Averaging::Arithmetic};
        const auto ref = mcp::price_asian_mc(o, kMarket, {s.asian_reference_paths, 99, VR::AntitheticControlVariate});
        const double geo = mcp::geometric_asian_price(o, kMarket);
        constexpr std::size_t kNumMethods = std::size(kMethods);
        mcp::MCResult results[kNumMethods];
        std::vector<double> times[kNumMethods];
        for (int rep = 0; rep < 3; ++rep) {  // interleaved timings, as in the European study
            for (std::size_t j = 0; j < kNumMethods; ++j) {
                results[j] = mcp::price_asian_mc(o, kMarket, {s.asian_paths, 13, kMethods[j]});
                times[j].push_back(results[j].elapsed_seconds);
            }
        }
        const double crude_var = results[0].variance_per_path;
        const double crude_time = median(times[0]);
        for (std::size_t j = 0; j < kNumMethods; ++j) {
            const mcp::MCResult& r = results[j];
            const double t = median(times[j]);
            csv.row(strike, 12, mcp::to_string(kMethods[j]), r.price, r.std_error, ref.price, ref.std_error, geo,
                    r.variance_per_path, crude_var / r.variance_per_path, t,
                    (crude_var * crude_time) / (r.variance_per_path * t), r.cv_beta, r.cv_correlation);
        }
    }
}

// ------------------------------------------------------------------ 6. randomised QMC
void qmc(const fs::path& out, const Scale& s) {
    std::printf("[6/7] randomised quasi-Monte Carlo\n");
    const double exact = mcp::bs_price(kCall, kMarket);
    Csv csv(out / "qmc.csv", "method,n_paths,replications,rmse,mean_abs_error");
    for (const std::uint64_t n : powers_of_two(7, log2_floor(s.max_convergence_paths))) {
        mcp::RunningStats sq_err, abs_err;
        for (int r = 0; r < s.replications; ++r) {
            mcp::RandomStream shift_rng(31337, static_cast<std::uint64_t>(r), 0xA5A5A5A5u);
            const double err = mcp::rqmc_shifted_estimate(kCall, kMarket, n, shift_rng.uniform()) - exact;
            sq_err.add(err * err);
            abs_err.add(std::abs(err));
        }
        csv.row("rqmc_van_der_corput", n, s.replications, std::sqrt(sq_err.mean()), abs_err.mean());
    }
}

// ------------------------------------------------------------------ 7. performance
void performance(const fs::path& out, const Scale& s) {
    std::printf("[7/7] parallel scaling and reproducibility\n");
    Csv csv(out / "performance.csv", "threads,n_paths,time_s,paths_per_second,speedup,price,identical_to_serial");
    const int hw = mcp::max_threads();
    std::vector<int> counts{1, 2, 4, 8, 12, 16, 24, 32};
    counts.erase(std::remove_if(counts.begin(), counts.end(), [&](int c) { return c > hw; }), counts.end());
    if (counts.back() != hw) counts.push_back(hw);

    double serial_time = 0.0, serial_price = 0.0;
    for (const int t : counts) {
        mcp::set_num_threads(t);
        std::vector<double> times;
        double price = 0.0;
        for (int rep = 0; rep < 5; ++rep) {
            const auto r = mcp::price_european_mc(kCall, kMarket, {s.performance_paths, 1, VR::None});
            times.push_back(r.elapsed_seconds);
            price = r.price;
        }
        const double time = median(times);
        if (t == 1) {
            serial_time = time;
            serial_price = price;
        }
        csv.row(t, s.performance_paths, time, static_cast<double>(s.performance_paths) / time, serial_time / time,
                price, price == serial_price ? 1 : 0);
    }
    mcp::set_num_threads(hw);
}

void environment(const fs::path& out) {
    Csv csv(out / "environment.csv", "key,value");
#if defined(__clang__)
    csv.row("compiler", std::string("Clang ") + __clang_version__);
#elif defined(_MSC_VER)
    csv.row("compiler", "MSVC " + std::to_string(_MSC_VER / 100) + "." + std::to_string(_MSC_VER % 100));
#elif defined(__GNUC__)
    csv.row("compiler", std::string("GCC ") + __VERSION__);
#else
    csv.row("compiler", "unknown");
#endif
    csv.row("threads", std::to_string(mcp::max_threads()));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        fs::path out = "results";
        Scale scale = kFull;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--quick")
                scale = kQuick;
            else
                out = arg;
        }
        fs::create_directories(out);
        std::printf("writing results to %s (%d threads)\n", out.string().c_str(), mcp::max_threads());

        const auto start = std::chrono::steady_clock::now();
        environment(out);
        convergence(out, scale);
        running_estimate(out, scale);
        variance_reduction(out, scale);
        greeks(out, scale);
        asian(out, scale);
        qmc(out, scale);
        performance(out, scale);
        std::printf("done in %.1f s\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return EXIT_FAILURE;
    }
}
