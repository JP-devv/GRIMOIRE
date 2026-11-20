#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace b70 {
inline bool verify_attn_profile_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("GRIMOIRE_VERIFY_ATTN_PROFILE");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

inline void require_verify_attn_profiling_queue(const sycl::queue& q) {
    if (!q.has_property<sycl::property::queue::enable_profiling>())
        throw std::logic_error("GRIMOIRE_VERIFY_ATTN_PROFILE requires a profiling queue");
}

// Only call on completed events. These are device execution intervals, not
// queue time or fenced host time.
inline uint64_t verify_attn_device_ns(const sycl::event& event) {
    const auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    if (end <= start) throw std::runtime_error("invalid verifier attention device interval");
    return end - start;
}
} // namespace b70
