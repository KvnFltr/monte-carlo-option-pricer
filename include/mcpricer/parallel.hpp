#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace mcp {

/// Number of threads used by the simulation kernels (1 without OpenMP).
[[nodiscard]] inline int max_threads() noexcept {
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/// Sets the number of threads used by subsequent simulations (no-op without OpenMP).
inline void set_num_threads(int n) noexcept {
#if defined(_OPENMP)
    omp_set_num_threads(std::max(1, n));
#else
    (void)n;
#endif
}

/// Samples per block of the parallel reduction. Fixed, so that the
/// partition of the sample, and hence the floating-point result, does not
/// depend on the number of threads.
inline constexpr std::uint64_t kBlockSize = 4096;

/// Runs `sample(i, acc)` for i = 0, ..., n - 1 and returns the merged accumulator.
///
/// The index range is cut into fixed blocks; each block is reduced into its own
/// accumulator (in parallel when OpenMP is enabled) and the partial results are
/// merged sequentially in block order. Combined with a counter-based RNG keyed
/// on the sample index, the output is bitwise identical for any thread count.
template <class Accumulator, class SampleFn>
[[nodiscard]] Accumulator parallel_accumulate(std::uint64_t n, SampleFn&& sample) {
    const std::uint64_t num_blocks = (n + kBlockSize - 1) / kBlockSize;
    std::vector<Accumulator> partial(num_blocks);
    const auto num_blocks_signed = static_cast<std::int64_t>(num_blocks);

    // Dynamic scheduling balances the load on hybrid CPUs (performance and
    // efficiency cores); determinism is unaffected because block b always
    // writes partial[b] and the merge order is fixed.
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
    for (std::int64_t b = 0; b < num_blocks_signed; ++b) {
        const std::uint64_t begin = static_cast<std::uint64_t>(b) * kBlockSize;
        const std::uint64_t end = std::min(n, begin + kBlockSize);
        Accumulator acc;
        for (std::uint64_t i = begin; i < end; ++i) sample(i, acc);
        partial[static_cast<std::size_t>(b)] = acc;
    }

    Accumulator total;
    for (const Accumulator& p : partial) total.merge(p);
    return total;
}

}  // namespace mcp
