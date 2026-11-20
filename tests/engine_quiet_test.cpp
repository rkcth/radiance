#include "rad_test.h"
#include "engine_priv.h"
#include <future>
#include <chrono>
namespace rad {
struct EngineQuietTestAccess {
    static int init(Engine& e) { auto r=std::make_unique<Engine::Rank>(); CtxDesc d;d.program=&r->program;RAD_TRY(r->ctx.init(d));e.ranks_.push_back(std::move(r));return RAD_OK; }
    static bool quiet_pending(Engine& e) { return e.quiet_pending_.load(); }
    static void service_quiet(Engine& e) { e.quiet_service(); }
    static void stop_quiet(Engine& e) { e.stopping_.store(true);e.quiet_cv_.notify_all(); }
};
}
using rad::Engine;
using Access=rad::EngineQuietTestAccess;
TEST(concurrent_quiet_callers_keep_their_own_callbacks) {
    Engine e(rad::Config{});
    REQUIRE_EQ(Access::init(e), RAD_OK);
    std::atomic<int> callbacks{0};
    std::future<int> first, second;
    struct StopBeforeJoining {
        Engine& engine;
        ~StopBeforeJoining() { Access::stop_quiet(engine); }
    } cleanup{e};
    const auto wait_for = [](auto ready) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!ready() && std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
        return ready();
    };
    first = std::async(std::launch::async, [&] {
        return e.run_quiet([&] { callbacks.fetch_add(1); return 11; });
    });
    REQUIRE(wait_for([&] { return Access::quiet_pending(e); }));
    second = std::async(std::launch::async, [&] {
        return e.run_quiet([&] { callbacks.fetch_add(1); return 22; });
    });

    Access::service_quiet(e);
    REQUIRE(wait_for([&] { return first.wait_for(std::chrono::seconds(0)) == std::future_status::ready; }));
    CHECK_EQ(first.get(), 11);
    REQUIRE(wait_for([&] { return Access::quiet_pending(e); }));
    Access::service_quiet(e);
    REQUIRE(wait_for([&] { return second.wait_for(std::chrono::seconds(0)) == std::future_status::ready; }));
    CHECK_EQ(second.get(), 22);
    CHECK_EQ(callbacks.load(), 2);
}
RAD_TEST_MAIN()
