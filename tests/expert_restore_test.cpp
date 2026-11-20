#include "rad_test.h"
#include "place/expert_restore.h"

TEST(a_fresh_or_unsuccessful_recall_never_grows_an_idle_cache) {
    rad::ExpertFlexRestore state;
    CHECK_EQ(state.limit(0, 1024), 0);
    state.recalled(512, 512);
    CHECK_EQ(state.limit(256, 1024), 0);
    state.recalled(512, 768);
    CHECK_EQ(state.limit(0, 1024), 0);
}

TEST(repeated_recalls_restore_the_original_capacity_without_overshooting) {
    rad::ExpertFlexRestore state;
    state.recalled(1000, 400);
    state.recalled(400, 200);
    CHECK_EQ(state.limit(200, 2000), 1000);
    // A fake broker first refuses, then grants fixed-size increments. The final
    // increment must be capped at the old watermark, not the larger ceiling.
    uint64_t now = 2000000000ull;
    CHECK_EQ(state.refused(now, true, 1800), 0);
    CHECK(!state.ready(now + 999999999ull));
    CHECK(state.ready(now + 1000000000ull));
    int64_t current = 200;
    while (const int64_t target = state.limit(current, 2000)) {
        const int64_t grant = std::min<int64_t>(256, target - current);
        REQUIRE(grant > 0);
        current += grant;
        CHECK(current <= 1000);
    }
    CHECK_EQ(current, 1000);
    CHECK_EQ(state.limit(current, 2000), 0);
}

TEST(idle_backoff_does_not_change_a_busy_engines_eviction_demand) {
    rad::ExpertFlexRestore state;
    state.recalled(1000, 0);
    CHECK_EQ(state.refused(100, true, 1000), 0);
    CHECK(!state.ready(101));
    CHECK_EQ(state.refused(101, false, 2000), 2000);
    CHECK_EQ(state.limit(0, 3000), 1000);
}

TEST(smaller_limits_and_parking_clear_stale_restore_intent) {
    rad::ExpertFlexRestore state;
    state.recalled(1000, 100);
    CHECK_EQ(state.limit(100, 600), 600);
    CHECK_EQ(state.limit(600, 600), 0);
    state.recalled(800, 200);
    state.clear();
    CHECK_EQ(state.limit(0, 2000), 0);
    state.recalled(800, 200);
    CHECK_EQ(state.limit(0, 0), 0);
}

RAD_TEST_MAIN()
