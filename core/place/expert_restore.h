/* Remember backing an expert cache surrendered to another model. Idle restoration
 * reclaims only previously used capacity; the ordinary mover chooses the experts.
 * Owned by the engine step loop, including its quiescent recall/park callbacks. */
#pragma once
#include <algorithm>
#include <cstdint>

namespace rad {
struct ExpertFlexRestore {
    void recalled(int64_t before, int64_t after) {
        if (after < before) previous_bytes_ = std::max(previous_bytes_, before);
    }
    // A changed ceiling cannot turn remembered capacity into an oversized request.
    int64_t limit(int64_t current, int64_t ceiling) {
        previous_bytes_ = std::min(previous_bytes_, std::max<int64_t>(ceiling, 0));
        if (current >= previous_bytes_) clear();
        return previous_bytes_;
    }
    bool ready(uint64_t now) const { return now >= retry_ns_; }
    // An idle cache may take free memory, but must not demand eviction of a peer.
    // Its backoff is separate so a new user request keeps the existing busy policy.
    int64_t refused(uint64_t now, bool idle_restore, int64_t remaining) {
        if (!idle_restore) return remaining;
        retry_ns_ = now + 1000000000ull;
        return 0;
    }
    void clear() { previous_bytes_ = 0; retry_ns_ = 0; }
private:
    int64_t previous_bytes_ = 0;
    uint64_t retry_ns_ = 0;
};
} // namespace rad
