#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "mcpricer/normal.hpp"

namespace mcp {

/// Philox4x32-10 counter-based pseudo-random bijection (Salmon et al., SC'11).
///
/// A counter-based generator maps (counter, key) to a random block with a
/// stateless keyed bijection. Any sub-stream can therefore be reached in O(1),
/// which is what makes the simulations reproducible independently of the
/// number of threads and gives common random numbers for free.
struct Philox4x32 {
    using Counter = std::array<std::uint32_t, 4>;
    using Key = std::array<std::uint32_t, 2>;

    [[nodiscard]] static constexpr Counter apply(Counter ctr, Key key) noexcept {
        ctr = round(ctr, key);
        for (int i = 1; i < 10; ++i) {
            key[0] += kW0;
            key[1] += kW1;
            ctr = round(ctr, key);
        }
        return ctr;
    }

private:
    static constexpr std::uint32_t kM0 = 0xD2511F53u;
    static constexpr std::uint32_t kM1 = 0xCD9E8D57u;
    static constexpr std::uint32_t kW0 = 0x9E3779B9u;  // golden ratio
    static constexpr std::uint32_t kW1 = 0xBB67AE85u;  // sqrt(3) - 1

    [[nodiscard]] static constexpr Counter round(const Counter& c, const Key& k) noexcept {
        const std::uint64_t p0 = std::uint64_t{kM0} * c[0];
        const std::uint64_t p1 = std::uint64_t{kM1} * c[2];
        const auto hi0 = static_cast<std::uint32_t>(p0 >> 32);
        const auto lo0 = static_cast<std::uint32_t>(p0);
        const auto hi1 = static_cast<std::uint32_t>(p1 >> 32);
        const auto lo1 = static_cast<std::uint32_t>(p1);
        return {hi1 ^ c[1] ^ k[0], lo1, hi0 ^ c[3] ^ k[1], lo0};
    }
};

/// Stream of uniform and Gaussian variates identified by (seed, stream, substream).
///
/// Counter layout: word 0 = block index, word 1 = substream, words 2-3 = stream.
/// The key is the 64-bit seed. In this library `stream` is the path index, so
/// path i always sees the same variates whatever the thread that simulates it.
class RandomStream {
public:
    RandomStream(std::uint64_t seed, std::uint64_t stream, std::uint32_t substream = 0) noexcept
        : key_{static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32)},
          ctr_{0u, substream, static_cast<std::uint32_t>(stream), static_cast<std::uint32_t>(stream >> 32)} {}

    /// 64 random bits.
    [[nodiscard]] std::uint64_t next_u64() noexcept {
        if (pos_ == 2) refill();
        return buffer_[pos_++];
    }

    /// Uniform variate on the open interval (0, 1) with 53 bits of resolution:
    /// (k + 1/2) 2^-53, k = 0, ..., 2^53 - 1. Never returns 0 or 1.
    [[nodiscard]] double uniform() noexcept {
        constexpr double kTwoPowMinus53 = 1.0 / 9007199254740992.0;
        return (static_cast<double>(next_u64() >> 11) + 0.5) * kTwoPowMinus53;
    }

    /// Standard normal variate by inversion: one uniform per Gaussian,
    /// which keeps the mapping path index -> variates simple and monotone.
    [[nodiscard]] double normal() noexcept { return norm_inv_cdf(uniform()); }

private:
    void refill() noexcept {
        const Philox4x32::Counter out = Philox4x32::apply(ctr_, key_);
        buffer_[0] = (std::uint64_t{out[1]} << 32) | out[0];
        buffer_[1] = (std::uint64_t{out[3]} << 32) | out[2];
        pos_ = 0;
        ++ctr_[0];  // 2^32 blocks = 2^33 doubles per stream
    }

    Philox4x32::Key key_;
    Philox4x32::Counter ctr_;
    std::array<std::uint64_t, 2> buffer_{};
    std::size_t pos_{2};
};

}  // namespace mcp
