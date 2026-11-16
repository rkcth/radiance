#pragma once

#include <chrono>
#include <string>

namespace rad {

struct ImageWaiter {
    std::string model;
    std::chrono::steady_clock::time_point enqueued;
    unsigned bypasses = 0;
};

/* Choose the first compatible request without passing an older request whose
 * count or age has reached its limit. Time is supplied by the dispatcher. */
template<class Iterator>
Iterator next_image_request(Iterator first, Iterator last, const std::string& active_model,
                            bool fifo, std::chrono::steady_clock::time_point now) {
    constexpr auto max_bypass_age = std::chrono::seconds(5);
    constexpr unsigned max_bypasses = 2;
    if (!fifo) {
        for (auto it = first; it != last; ++it) {
            if (it->model == active_model) return it;
            if (it->bypasses >= max_bypasses || now - it->enqueued >= max_bypass_age) break;
        }
    }
    return first;
}

}  /* namespace rad */
