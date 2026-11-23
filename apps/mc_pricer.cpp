// Command-line pricer: compares the Monte Carlo estimators with Black-Scholes.

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>

#include "mcpricer/mcpricer.hpp"

namespace {

struct Options {
    mcp::EuropeanOption option{};
    mcp::Market market{};
    std::uint64_t paths{1'000'000};
    std::uint64_t seed{42};
    int threads{0};
    bool greeks{false};
};

void print_usage() {
    std::puts(
        "Usage: mc_pricer [options]\n"
        "  --type call|put     option type              (default call)\n"
        "  --spot S            spot price               (default 100)\n"
        "  --strike K          strike                   (default 100)\n"
        "  --rate r            risk-free rate           (default 0.05)\n"
        "  --div q             dividend yield           (default 0)\n"
        "  --vol sigma         volatility               (default 0.2)\n"
        "  --maturity T        maturity in years        (default 1)\n"
        "  --paths N           number of paths          (default 1000000)\n"
        "  --seed s            random seed              (default 42)\n"
        "  --threads n         worker threads           (default: all cores)\n"
        "  --greeks            also estimate the Greeks\n"
        "  --help              show this message");
}

Options parse(int argc, char** argv) {
    Options o;
    // Options that take a value, and how to store it.
    const std::map<std::string_view, std::function<void(const std::string&)>> setters{
        {"--spot", [&](const std::string& v) { o.market.spot = std::stod(v); }},
        {"--strike", [&](const std::string& v) { o.option.strike = std::stod(v); }},
        {"--rate", [&](const std::string& v) { o.market.rate = std::stod(v); }},
        {"--div", [&](const std::string& v) { o.market.dividend = std::stod(v); }},
        {"--vol", [&](const std::string& v) { o.market.vol = std::stod(v); }},
        {"--maturity", [&](const std::string& v) { o.option.maturity = std::stod(v); }},
        {"--paths", [&](const std::string& v) { o.paths = std::stoull(v); }},
        {"--seed", [&](const std::string& v) { o.seed = std::stoull(v); }},
        {"--threads", [&](const std::string& v) { o.threads = std::stoi(v); }},
        {"--type",
         [&](const std::string& v) {
             if (v != "call" && v != "put") throw std::invalid_argument("--type must be call or put");
             o.option.type = v == "call" ? mcp::OptionType::Call : mcp::OptionType::Put;
         }},
    };

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage();
            std::exit(EXIT_SUCCESS);
        }
        if (arg == "--greeks") {
            o.greeks = true;
            continue;
        }
        const auto setter = setters.find(arg);
        if (setter == setters.end()) throw std::invalid_argument("unknown option " + std::string(arg));
        if (i + 1 >= argc) throw std::invalid_argument("missing value after " + std::string(arg));
        setter->second(argv[++i]);
    }
    mcp::validate(o.option);
    mcp::validate(o.market);
    return o;
}

void print_prices(const Options& o) {
    const double exact = mcp::bs_price(o.option, o.market);
    std::printf("Black-Scholes price        %.6f\n\n", exact);
    std::printf("Monte Carlo: %llu paths, seed %llu, %d thread(s)\n", static_cast<unsigned long long>(o.paths),
                static_cast<unsigned long long>(o.seed), mcp::max_threads());
    std::printf("%-28s %10s %10s %23s %8s %8s %10s\n", "method", "price", "std.err", "95% CI", "z-score", "VRF",
                "time [ms]");

    double crude_variance = 0.0;
    for (const auto method :
         {mcp::VarianceReduction::None, mcp::VarianceReduction::Antithetic, mcp::VarianceReduction::ControlVariate,
          mcp::VarianceReduction::AntitheticControlVariate}) {
        const mcp::MCResult r = mcp::price_european_mc(o.option, o.market, {o.paths, o.seed, method});
        if (method == mcp::VarianceReduction::None) crude_variance = r.variance_per_path;
        std::printf("%-28s %10.5f %10.5f   [%9.5f, %9.5f] %8.2f %8.1f %10.1f\n",
                    std::string(mcp::to_string(method)).c_str(), r.price, r.std_error, r.ci_lower, r.ci_upper,
                    (r.price - exact) / r.std_error, crude_variance / r.variance_per_path, 1e3 * r.elapsed_seconds);
    }
}

void print_greeks(const Options& o) {
    const mcp::BSGreeks exact = mcp::bs_greeks(o.option, o.market);
    const mcp::GreekConfig config{o.paths, o.seed, 0.0};
    std::printf("\nGreeks (value +/- standard error)\n");
    std::printf("%-6s %12s %24s %24s %24s\n", "greek", "analytic", "finite diff. (CRN)", "pathwise",
                "likelihood ratio");

    const struct {
        mcp::Greek greek;
        double analytic;
    } rows[] = {{mcp::Greek::Delta, exact.delta},
                {mcp::Greek::Gamma, exact.gamma},
                {mcp::Greek::Vega, exact.vega},
                {mcp::Greek::Rho, exact.rho},
                {mcp::Greek::Theta, exact.theta}};

    for (const auto& row : rows) {
        std::printf("%-6s %12.6f", std::string(mcp::to_string(row.greek)).c_str(), row.analytic);
        for (const auto method :
             {mcp::GreekMethod::FiniteDifferenceCRN, mcp::GreekMethod::Pathwise, mcp::GreekMethod::LikelihoodRatio}) {
            if (!mcp::is_supported(row.greek, method)) {
                std::printf(" %24s", "n/a");
                continue;
            }
            const mcp::GreekResult g = mcp::mc_greek(row.greek, method, o.option, o.market, config);
            char cell[64];
            std::snprintf(cell, sizeof cell, "%.5f +/- %.5f", g.value, g.std_error);
            std::printf(" %24s", cell);
        }
        std::printf("\n");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        if (o.threads > 0) mcp::set_num_threads(o.threads);

        std::printf("European %s  S0=%g  K=%g  r=%g  q=%g  sigma=%g  T=%g\n",
                    o.option.type == mcp::OptionType::Call ? "call" : "put", o.market.spot, o.option.strike,
                    o.market.rate, o.market.dividend, o.market.vol, o.option.maturity);
        print_prices(o);
        if (o.greeks) print_greeks(o);
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        print_usage();
        return EXIT_FAILURE;
    }
}
