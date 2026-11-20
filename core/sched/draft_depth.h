/* draft_depth.h -- same-step adaptive depth of a coupled serial draft chain (--draft-depth adaptive).
 *
 * A fixed chain drafts N tokens every step and its verify pays for N + 1 rows whatever the head
 * thought of them. Here the chain of a step stops as soon as the head's own laws say the next draft
 * is unlikely to be accepted: round j (j >= 2) runs only while the chain's confidence stays at or
 * above the threshold,
 *
 *     round j runs  iff  j <= max  and  ( j <= min  or  conf(j - 1 - lag) >= stop ),
 *     conf(m) = prod_{i = 1..m} s_i,   conf(m <= 0) = 1,
 *
 * where s_i is read from round i's coupled state (abi/rad_spechub.h): the largest mass of the law
 * Q_i the round drew from (`top`), or the mass of the token it drew (`path`, the probability of
 * the drafted path). The first round that does not run ends the chain; its depth D is the number
 * of rounds that ran, and the next verify carries D drafts (D + 1 rows).
 *
 * `lag` is how many rounds the decision trails the chain: with lag 0 round j is decided by every
 * round before it, so the card waits for the host between two rounds; with lag 1 round j is decided
 * when round j - 2 has finished, while round j - 1 runs, so the host's decision (a readback of one
 * state row, then the next round's or the next verify's issue) hides behind a running round.
 *
 * Why this keeps the verify exact. The decision "x_j is drafted" reads Q_1 .. Q_{j-1-lag} and, for
 * `path`, x_1 .. x_{j-1-lag}: draft-side quantities of the chain's ANCESTORS of x_j only, never a
 * target output and never x_j itself. For a row-wise verify any such stopping time is exact (each
 * verified position is exact on its own). For the traversal walk (abi/rad_mtp_traversal.h) the step
 * of its induction at a node v needs v's number of children to be independent of the tokens drawn
 * AT v; the shape of a child's subtree may depend on the child's own token, because the induction
 * hypothesis is applied per child value. Here node v (the prefix x_1 .. x_{j-1}) has a child iff
 * the rule above holds, a function of v's own path. A rule that read x_j's own mass to admit x_j
 * would not be exact (tests/mtp_rrs_test.cpp shows the statistic that catches it, and one that
 * peeks at the target).
 *
 * Several sequences in one step: the load-aware rule (DraftDepthCfg::load) and the late choice of
 * the verify's rows (DraftDepthCfg::late), with the tables they read and the online calibration
 * that keeps those tables (DraftDepthOnline), further down.
 *
 * Plain code, no engine types beyond the wire row: the engine's driver, the scheduler's tests and
 * the exactness test share it.
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "rad_spechub.h"

namespace rad {

enum DraftDepthStat : int32_t {
    DRAFT_DEPTH_TOP  = 0,   /* s_i = max_x Q_i(x) */
    DRAFT_DEPTH_PATH = 1,   /* s_i = Q_i(x_i), the drawn token's mass */
};

struct DraftDepthCfg {
    bool   on      = false;
    int    max     = 0;      /* the deepest chain: rounds 1 .. max (the head's declared depth caps it) */
    int    min     = 1;      /* rounds that always run */
    int    lag     = 1;      /* rounds the decision trails the chain, 0 or more */
    int    stat    = DRAFT_DEPTH_TOP;
    double stop    = 0.85;   /* the chain goes on while conf >= stop */
    int    lawless = 3;      /* the depth of a row with no law (a greedy request, a malformed round) */

    /* ---- The load-aware rule (--draft-depth-load on), for a step that drafts for `load_from` or
     * more sequences (draft_depth_go_load, draft_depth_select below).
     *
     * `stop` was fitted for one sequence, where a draft's price is its verify row plus the head
     * round that makes it. With n drafting rows the prices part: the round is one pass for every
     * row, the verify row is each row's own, a row that takes the verify past a kernel's tile
     * (16, 32, 48 or 64 query rows) pays for the tile, and all of them are a different share of
     * the tokens the step makes. So the rule weighs expected accepted drafts against time:
     *
     *     p_i(j)  = cal[j](conf_i)                 the chance that row i's draft j is accepted
     *     cost(R) = row_ms R + sum of jump_ms[k] over the tiles R is past (R > jump_at[k])
     *     round j takes the m rows of largest p that maximise
     *         sum of their p  -  lam (cost(R + m) - cost(R))  -  lam round_ms (m > 0)
     *
     * with R the verify rows the step has so far and lam the tokens a millisecond the engine
     * makes at n sequences: a draft is made when it yields more than the same time does on
     * average. `cal` is the head's calibration, piecewise linear over the knots `cal_x`.
     *
     * Every number here starts from a table measured once (draft_depth_load_defaults) and, with
     * the online calibration (DraftDepthOnline below), follows the running server: the
     * acceptance by confidence from the verifies' own accepted counts, the rate and the prices
     * from the measured step times.
     *
     * A step with fewer drafting rows keeps the `stop` rule, so a single stream drafts exactly as
     * without this. Exactness is the stop rule's (above): whether x_j is drafted reads draft-side
     * quantities of x_j's ancestors, the same of the step's OTHER sequences (independent of this
     * sequence's draws) and numbers fixed before the step began -- the tables, which the online
     * calibration moves only between steps, from verifies and step times that are over. Those
     * are functions of draws at coordinates the step in hand never reads again (a draft stream is
     * keyed by the step's base position and its round, an acceptance by its verify position, and
     * each is consumed once), so given the past the rule is one with constants. Never x_j, never
     * a target output of the step. (Two requests served with one seed share those coordinates;
     * the same caveat as --draft-calib online.) */
    enum { kLoadSeqs = 8, kCalKnots = 8, kCalRounds = 8, kJumps = 4 };
    bool   load      = false;
    int    load_from = 5;        /* the fewest drafting rows of a load-aware step */
    int    load_min  = 2;        /* rounds a load-aware step always runs and drafts it never cuts */
    double load_lam[kLoadSeqs + 1]      = {};   /* by drafting rows n, [0] unused; n > kLoadSeqs takes the last */
    double load_row_ms[kLoadSeqs + 1]   = {};
    double load_round_ms[kLoadSeqs + 1] = {};
    int    load_jump_at[kJumps] = { 16, 32, 48, 64 };          /* a verify of more rows than this ... */
    double load_jump_ms[kLoadSeqs + 1][kJumps] = {};           /* ... costs this much more, once */
    double cal_x[kCalKnots] = {};            /* ascending conf knots */
    /* cal_p[t][j][k]: draft j's acceptance at conf(j - 2 + t) = cal_x[k]: the confidence the rule
     * reads one round behind (t 0), the last one before the draft (t 1), and the one that
     * includes the draft's own law (t 2, the late choice). */
    double cal_p[3][kCalRounds + 1][kCalKnots] = {};

    /* The late choice of the verify's rows (draft_depth_select). A round is one head pass over
     * the drafting rows, so once a round ran for a row the row holds its draft, and by the time
     * the next verify is issued the host has read more of the chain's laws than the rule had when
     * it decided the round: the statistics of every round but the last `late_lag`. `late` chooses
     * each row's verified prefix again from those, level by level: draft j of the rows still in
     * at j - 1, by p_i(j) = cal(conf_i(min(j, known))), the best prefix under the same cost model
     * (no round's price: the rounds have run).
     *
     * Exact for the same reason as the rule: draft j is verified by a function of the laws
     * Q_1 .. Q_j of its own chain -- Q_j is the law AT its parent, a function of the parent's path
     * and not of x_j -- and of the other sequences' laws up to their round j. Never of a round
     * past j: Q_{j+1} is conditioned on x_j, and a choice that read it would pick x_j by what
     * followed it. Only the `top` statistic may be read this way (the drawn token's own mass at
     * round j is x_j's). */
    bool   late     = false;
    int    late_lag = 1;      /* rounds the choice does not wait for: 1 = it is made while the last round runs */
};

/* One drafting row's chain while its step is drafted. */
struct DraftDepthRow {
    double conf   = 1.0;     /* conf(seen) */
    int    seen   = 0;       /* rounds whose state went into conf */
    int    depth  = 1;       /* rounds decided to run so far (round 1 always does) */
    bool   done   = false;   /* a round was refused: depth is final */
    bool   lawless = false;  /* no law: the fixed fallback depth */
    float  s[8]   = {};      /* each observed round's statistic (RAD_MTP_COUPLE_MAX_ROUNDS) */
    int    rule   = 1;       /* the round the load-aware rule sent the chain to (the late choice may cut it) */
};

/* Round `round`'s statistic from its state, or a negative number when the state carries no law
 * (status != 1: a greedy row or a shortlist the proposer could not turn into Q). */
inline double draft_depth_stat(const RadMtpCoupleState& s, int round, int stat) {
    if (s.status != 1 || s.round != round || s.count < 1 || s.count > 8) return -1.0;
    double top = 0.0, own = -1.0;
    for (int i = 0; i < s.count; ++i) {
        const double q = s.q[i];
        if (!(q > 0.0) || q > 1.0) return -1.0;
        if (q > top) top = q;
        if (s.ids[i] == s.x) own = q;
    }
    return stat == DRAFT_DEPTH_PATH ? own : top;
}

/* The round whose statistic decides round `j` (>= 2); <= 0 when nothing does (conf = 1). */
inline int draft_depth_need(const DraftDepthCfg& c, int j) { return j - 1 - (c.lag > 0 ? c.lag : 0); }

/* The start of a step's chain for one row. `lawless`: the row draws nothing (a greedy request). */
inline void draft_depth_begin(const DraftDepthCfg& c, DraftDepthRow* r, bool lawless) {
    *r = DraftDepthRow{};
    r->lawless = lawless;
    (void)c;
}

/* Feed round `round`'s statistic (rounds in order, each once). A negative one makes the row lawless. */
inline void draft_depth_observe(DraftDepthRow* r, int round, double s) {
    if (round != r->seen + 1) return;
    r->seen = round;
    if (s < 0.0) { r->lawless = true; return; }
    r->conf *= s;
    if (round >= 1 && round <= 8) r->s[round - 1] = (float)s;
}

/* Whether round `j` (>= 2) runs for this row, given every statistic up to round
 * draft_depth_need(c, j) was observed. Rounds are asked in order; once one is refused the chain has
 * ended. Records the answer in r->depth / r->done. */
inline bool draft_depth_go(const DraftDepthCfg& c, DraftDepthRow* r, int j) {
    if (r->done || j != r->depth + 1) return false;
    bool go;
    if (j > c.max) go = false;
    else if (r->lawless) go = j <= (c.lawless > 0 ? c.lawless : 1);
    else if (j <= c.min) go = true;
    else {
        /* conf(need): the row has seen exactly `need` rounds when asked in order with lag >= 0, or
         * more when the caller observed ahead; recompute is not possible, so the caller observes
         * no further than draft_depth_need before asking (the driver does). */
        go = r->conf >= c.stop;
    }
    if (go) r->depth = j; else r->done = true;
    return go;
}

/* ------------------------------------------------------------------ the load-aware rule */

/* conf(m) of a row whose first `m` rounds were observed. */
inline double draft_depth_conf(const DraftDepthRow& r, int m) {
    double c = 1.0;
    for (int i = 0; i < m && i < 8 && i < r.seen; ++i) c *= (double)r.s[i];
    return c;
}

/* The knots a calibration table has: cal_x ascends as far as it goes. */
inline int draft_depth_knots(const DraftDepthCfg& c) {
    int K = 1;
    while (K < DraftDepthCfg::kCalKnots && c.cal_x[K] > c.cal_x[K - 1]) ++K;
    return K;
}

/* `conf` between two knots: the lower one's index and the upper one's share, for a table's
 * interpolation and for the online counts alike. */
inline void draft_depth_knot_at(const DraftDepthCfg& c, double conf, int* k0, double* w1) {
    const int K = draft_depth_knots(c);
    const double* x = c.cal_x;
    if (!(conf > x[0])) { *k0 = 0; *w1 = 0.0; return; }
    for (int k = 1; k < K; ++k)
        if (conf <= x[k]) { *k0 = k - 1; *w1 = (conf - x[k - 1]) / (x[k] - x[k - 1]); return; }
    *k0 = K - 1; *w1 = 0.0;
}

/* cal[j](conf): the chance that draft j is accepted, from the confidence the decision reads:
 * table t is by conf(j - 2 + t) (DraftDepthCfg::cal_p). */
inline double draft_depth_cal(const DraftDepthCfg& c, int j, double conf, int t = 0) {
    if (j < 0) j = 0;
    if (j > DraftDepthCfg::kCalRounds) j = DraftDepthCfg::kCalRounds;
    if (t < 0) t = 0;
    if (t > 2) t = 2;
    int k0; double w1;
    draft_depth_knot_at(c, conf, &k0, &w1);
    const double* p = c.cal_p[t][j];
    return w1 > 0.0 ? p[k0] + (p[k0 + 1] - p[k0]) * w1 : p[k0];
}

inline int draft_depth_seqs(int n) {
    return n < 1 ? 1 : (n > DraftDepthCfg::kLoadSeqs ? (int)DraftDepthCfg::kLoadSeqs : n);
}

/* What a verify of R query rows costs under the load-aware rule's model, in milliseconds past a
 * constant. */
inline double draft_depth_load_cost(const DraftDepthCfg& c, int n, int R) {
    const int nn = draft_depth_seqs(n);
    if (R < 0) R = 0;
    double ms = c.load_row_ms[nn] * (double)R;
    for (int k = 0; k < DraftDepthCfg::kJumps; ++k)
        if (R > c.load_jump_at[k]) ms += c.load_jump_ms[nn][k];
    return ms;
}

/* Round `j` (>= 2) for the `n` drafting rows of one step under the load-aware rule; every
 * statistic up to round draft_depth_need(c, j) was observed. `other_rows` are the verify rows the
 * next step carries beside these rows' (sequences that do not draft). Rows whose chain ended stay
 * ended. Returns whether the round runs, with each row's answer in depth / done as draft_depth_go
 * leaves them (and in `rule`, which the late choice does not move). A row with no law, and the
 * rounds that always run, keep the stop rule's answers and make the round run. */
inline bool draft_depth_go_load(const DraftDepthCfg& c, DraftDepthRow* r, int n, int j, int other_rows = 0) {
    const int nn = draft_depth_seqs(n);
    const double lam = c.load_lam[nn];
    const int lmin = c.min > c.load_min ? c.min : c.load_min;
    enum { kMaxRows = 64 };
    int    cand[kMaxRows], n_cand = 0;
    double p[kMaxRows];
    bool   want[kMaxRows];
    int    R = other_rows < 0 ? 0 : other_rows, forced = 0;
    for (int i = 0; i < n && i < kMaxRows; ++i) {
        want[i] = false;
        R += 1 + (r[i].depth < 1 ? 1 : r[i].depth);
        if (r[i].done || j != r[i].depth + 1 || j > c.max) continue;
        if (r[i].lawless) want[i] = j <= (c.lawless > 0 ? c.lawless : 1);
        /* A round the rule has read nothing for (round 2, decided one round behind) runs as under
         * the stop rule: with no law to weigh, the prices alone would turn all drafting on or off
         * for the whole step, and a calibration that is a little off must not be able to do that. */
        else if (j <= lmin || draft_depth_need(c, j) <= 0) want[i] = true;
        else { p[i] = draft_depth_cal(c, j, r[i].conf, c.lag > 0 ? 0 : 1); cand[n_cand++] = i; }
        forced += want[i];
    }
    /* the candidates by falling p (stable: the earlier row first), then the best prefix */
    for (int a = 1; a < n_cand; ++a) {
        const int x = cand[a];
        int b = a;
        for (; b > 0 && p[cand[b - 1]] < p[x]; --b) cand[b] = cand[b - 1];
        cand[b] = x;
    }
    const double base = draft_depth_load_cost(c, n, R + forced);
    double sum = 0.0, best = 0.0;
    int    take = 0;
    for (int m = 1; m <= n_cand; ++m) {
        sum += p[cand[m - 1]];
        const double v = sum - lam * (draft_depth_load_cost(c, n, R + forced + m) - base) -
                         (forced ? 0.0 : lam * c.load_round_ms[nn]);
        if (v > best) { best = v; take = m; }
    }
    for (int m = 0; m < take; ++m) want[cand[m]] = true;
    const bool run = forced > 0 || take > 0;
    for (int i = 0; i < n && i < kMaxRows; ++i) {
        if (r[i].done || j != r[i].depth + 1) continue;
        if (run && want[i]) { r[i].depth = j; r[i].rule = j; } else r[i].done = true;
    }
    for (int i = kMaxRows; i < n; ++i) r[i].done = true;
    return run;
}

/* The late choice (DraftDepthCfg::late): the verified prefix of each of the `n` drafting rows of a
 * step whose `rounds` rounds were issued, from the statistics observed so far (each row's `seen`;
 * the driver observes every row up to rounds - late_lag). Level by level: draft j of the rows
 * still in at j - 1 by the best prefix of their p under the cost model, with no round's price.
 * A row with no law keeps the depth the rule gave it (at most `rounds`).
 * Leaves each row's answer in depth / done. */
inline void draft_depth_select(const DraftDepthCfg& c, DraftDepthRow* r, int n, int rounds, int other_rows = 0) {
    const int nn = draft_depth_seqs(n);
    const double lam = c.load_lam[nn];
    const int lmin = c.min > c.load_min ? c.min : c.load_min;
    enum { kMaxRows = 64 };
    if (n > kMaxRows) n = kMaxRows;
    int R = other_rows < 0 ? 0 : other_rows;
    for (int i = 0; i < n; ++i) {
        if (r[i].lawless) {
            if (r[i].depth > rounds) r[i].depth = rounds;
            if (r[i].depth < 1) r[i].depth = 1;
        } else {
            r[i].depth = 1;
        }
        r[i].done = r[i].lawless;
        R += 1 + r[i].depth;
    }
    for (int j = 2; j <= rounds && j <= c.max; ++j) {
        int    cand[kMaxRows], n_cand = 0;
        double p[kMaxRows];
        for (int i = 0; i < n; ++i) {
            if (r[i].done || r[i].depth != j - 1) continue;
            /* the confidence read: conf(min(j, seen)), never a round past j */
            const int m = r[i].seen < j ? r[i].seen : j;
            p[i] = j <= lmin ? 2.0 : draft_depth_cal(c, j, draft_depth_conf(r[i], m), m - (j - 2));
            cand[n_cand++] = i;
        }
        for (int a = 1; a < n_cand; ++a) {
            const int x = cand[a];
            int b = a;
            for (; b > 0 && p[cand[b - 1]] < p[x]; --b) cand[b] = cand[b - 1];
            cand[b] = x;
        }
        const double base = draft_depth_load_cost(c, n, R);
        double sum = 0.0, best = 0.0;
        int    take = 0;
        for (int m = 1; m <= n_cand; ++m) {
            sum += p[cand[m - 1]];
            const double v = sum - lam * (draft_depth_load_cost(c, n, R + m) - base);
            if (v > best || p[cand[m - 1]] > 1.5) { best = v; take = m; }   /* > 1.5: a round that always runs */
        }
        for (int m = 0; m < n_cand; ++m) {
            DraftDepthRow& x = r[cand[m]];
            if (m < take) x.depth = j; else x.done = true;
        }
        R += take;
        if (take == 0) break;
    }
    for (int i = 0; i < n; ++i) r[i].done = true;
}

/* The rule's tables as measured once: Qwen3.8-Flash-Next IQR-V4 on two R9700 (tensor parallel),
 * sampled at temperature 1.0, MTP head with coupling and traversal verify. They are where the
 * online calibration starts from and what `--draft-depth-load-calib static` serves. By drafting
 * rows 1 .. 8: the tokens a millisecond, a verify row's and a head round's milliseconds; the
 * verify's tile jumps past 16 / 32 / 48 / 64 rows; draft j's acceptance by confidence. */
inline void draft_depth_load_defaults(DraftDepthCfg* c) {
    static const double lam[8]   = { 0.2325, 0.3558, 0.4341, 0.5125, 0.5494, 0.5863, 0.6232, 0.6602 };
    static const double row[8]   = { 1.159, 0.824, 0.706, 0.587, 0.548, 0.510, 0.471, 0.432 };
    static const double round[8] = { 0.470, 0.540, 0.570, 0.600, 0.645, 0.690, 0.735, 0.780 };
    static const double jump[DraftDepthCfg::kJumps] = { 1.30, 1.40, 2.30, 1.00 };
    static const double x[8] = { 0, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99, 1 };
    static const double cal[3][4][8] = {
        { { 0.575, 0.575, 0.575, 0.575, 0.575, 0.575, 0.575, 0.575 },
          { 0.239, 0.239, 0.239, 0.279, 0.337, 0.420, 0.546, 0.809 },
          { 0.094, 0.119, 0.172, 0.288, 0.396, 0.526, 0.669, 0.855 },
          { 0.055, 0.088, 0.186, 0.322, 0.457, 0.579, 0.710, 0.847 } },
        { { 0.318, 0.318, 0.341, 0.403, 0.462, 0.575, 0.723, 0.890 },
          { 0.138, 0.185, 0.273, 0.380, 0.503, 0.660, 0.790, 0.931 },
          { 0.092, 0.137, 0.241, 0.411, 0.555, 0.711, 0.810, 0.909 },
          { 0.056, 0.112, 0.276, 0.414, 0.540, 0.651, 0.823, 0.916 } },
        { { 0.251, 0.305, 0.416, 0.532, 0.649, 0.809, 0.919, 0.981 },
          { 0.147, 0.220, 0.367, 0.519, 0.661, 0.824, 0.909, 0.971 },
          { 0.096, 0.166, 0.346, 0.508, 0.648, 0.783, 0.914, 0.970 },
          { 0.063, 0.142, 0.352, 0.478, 0.594, 0.769, 0.899, 0.968 } } };
    for (int n = 1; n <= DraftDepthCfg::kLoadSeqs; ++n) {
        c->load_lam[n] = lam[n - 1]; c->load_row_ms[n] = row[n - 1]; c->load_round_ms[n] = round[n - 1];
        for (int k = 0; k < DraftDepthCfg::kJumps; ++k) c->load_jump_ms[n][k] = jump[k];
    }
    c->load_lam[0] = lam[0]; c->load_row_ms[0] = row[0]; c->load_round_ms[0] = round[0];
    c->load_jump_at[0] = 16; c->load_jump_at[1] = 32; c->load_jump_at[2] = 48; c->load_jump_at[3] = 64;
    for (int k = 0; k < DraftDepthCfg::kCalKnots; ++k) c->cal_x[k] = x[k];
    for (int t = 0; t < 3; ++t)
        for (int j = 0; j <= DraftDepthCfg::kCalRounds; ++j)
            for (int k = 0; k < DraftDepthCfg::kCalKnots; ++k)
                c->cal_p[t][j][k] = j < 2 ? 1.0 : cal[t][(j > 5 ? 5 : j) - 2][k];
}

/* "a,b,c" into out[0 .. cap): the count read, or -1 when a word is no number in [lo, hi], or the
 * list is empty or longer than `cap` (the --draft-depth-load lists). */
inline int draft_depth_parse_list(const char* s, double* out, int cap, double lo, double hi) {
    if (!s || !*s) return -1;
    int n = 0;
    for (const char* q = s;;) {
        char* end = nullptr;
        const double v = std::strtod(q, &end);
        if (end == q || !(v >= lo && v <= hi) || n >= cap) return -1;
        out[n++] = v;
        if (*end == 0) break;
        if (*end != ',') return -1;
        q = end + 1;
    }
    return n;
}

/* ------------------------------------------------------------------ the online calibration
 *
 * The tables above were measured on one model, one pair of cards and one sampling temperature.
 * A server that runs something else (a lower temperature accepts more at the same confidence;
 * another card prices a row differently) measures them itself:
 *
 *   acceptance   every verified chain says, for each draft j it carried, whether the verify
 *                accepted it, beside the confidences the decisions read. They are counted at the
 *                knots of the table the decision used (a confidence between two knots goes to
 *                both, by its distance), on top of `prior_w` pseudo-observations of the static
 *                table, and the estimate is made monotone in the confidence. A knot the rule
 *                never drafts at keeps the static value.
 *   rate         tokens and milliseconds of the pure decode steps, a moving average by drafting
 *                rows (half-life 64 steps).
 *   prices       the same steps' milliseconds regressed on (1, verify rows, rounds, the tiles the
 *                rows are past), by drafting rows, with the static prices as a prior: a
 *                coefficient the steps do not tell apart (a round from the rows that come with
 *                it) stays near the measured table, one they do (a row, a tile) follows them.
 *
 * Table 0 (the rule's, read one round behind) counts a chain the rule sent on to round j and the
 * late choice then cut as "not accepted": what the round yielded is what the rule weighs.
 *
 * The driver feeds a chain when its verify is committed and a step when its passes have ended,
 * and takes the new tables between steps (draft_depth_online_refresh): see the exactness note at
 * DraftDepthCfg::load. Plain arithmetic; a few hundred flops a step. */
struct DraftDepthOnline {
    enum { kF = 3 + DraftDepthCfg::kJumps, kR = DraftDepthCfg::kCalRounds, kK = DraftDepthCfg::kCalKnots,
           kN = DraftDepthCfg::kLoadSeqs };
    double  prior_w  = 48.0;                 /* the static table's weight, in verified drafts a knot */
    double  cal_keep = 0.98923;              /* a refresh keeps this much of the counts: half in 64 */
    double  cal0[3][kR + 1][kK] = {};        /* the static tables */
    double  cn[3][kR + 1][kK]   = {}, ca[3][kR + 1][kK] = {};
    struct Lam { double tok = 0, ms = 0; int64_t n = 0; } lam[kN + 1];
    struct Reg { double S[kF][kF] = {}; double b[kF] = {}; double ms_avg = 0; int64_t n = 0; } reg[kN + 1];
    double  theta0[kN + 1][kF] = {};         /* the static prices: (base, row, round, jumps) */
    double  theta[kN + 1][kF]  = {};         /* the last fit, for the log */
    double  prec[kF]   = { 1e-4, 200.0, 600.0, 20.0, 20.0, 20.0, 20.0 };   /* the prior's weight */
    double  reg_keep   = 1.0 - 1.0 / 2048.0; /* a step keeps this much of the sums */
    int64_t chains = 0, steps = 0, outliers = 0, refreshes = 0;
};

/* Start from the tables `c` holds (the static ones, or the configured). */
inline void draft_depth_online_begin(const DraftDepthCfg& c, DraftDepthOnline* o) {
    *o = DraftDepthOnline{};
    for (int t = 0; t < 3; ++t)
        for (int j = 0; j <= DraftDepthOnline::kR; ++j)
            for (int k = 0; k < DraftDepthOnline::kK; ++k) o->cal0[t][j][k] = c.cal_p[t][j][k];
    for (int n = 0; n <= DraftDepthOnline::kN; ++n) {
        o->theta0[n][0] = 0.0;
        o->theta0[n][1] = c.load_row_ms[n];
        o->theta0[n][2] = c.load_round_ms[n];
        for (int k = 0; k < DraftDepthCfg::kJumps; ++k) o->theta0[n][3 + k] = c.load_jump_ms[n][k];
        for (int f = 0; f < DraftDepthOnline::kF; ++f) o->theta[n][f] = o->theta0[n][f];
    }
}

/* One verified chain: it carried `depth` drafts of which the verify accepted `accepted`; `s` are
 * the statistics of its rounds 1 .. n_s (the `top` ones). `rule_depth` is the round the load-aware
 * rule sent the chain to before the late choice (0: a step the stop rule decided). */
inline void draft_depth_online_chain(const DraftDepthCfg& c, DraftDepthOnline* o, int depth, int rule_depth,
                                     int accepted, const float* s, int n_s) {
    if (depth < 1 || n_s < 0) return;
    ++o->chains;
    const int rule_t = c.lag > 0 ? 0 : 1;
    const bool rule_cut = rule_depth > 0 && c.late && rule_t == 0;   /* table 0 is the rule's alone */
    const int top = depth > rule_depth ? depth : rule_depth;
    for (int j = 2; j <= top && j <= DraftDepthOnline::kR; ++j) {
        const bool verified = j <= depth;
        const double y = verified && accepted >= j ? 1.0 : 0.0;
        for (int t = 0; t < 3; ++t) {
            const int m = j - 2 + t;
            if (m > n_s || m > 8) continue;                 /* that round's statistic was not kept */
            if (t == 0 && rule_cut ? rule_depth < j : !verified) continue;
            double conf = 1.0;
            for (int i = 0; i < m; ++i) conf *= (double)s[i];
            int k0; double w1;
            draft_depth_knot_at(c, conf, &k0, &w1);
            o->cn[t][j][k0] += 1.0 - w1; o->ca[t][j][k0] += (1.0 - w1) * y;
            if (w1 > 0.0) { o->cn[t][j][k0 + 1] += w1; o->ca[t][j][k0 + 1] += w1 * y; }
        }
    }
}

/* One pure decode step whose passes have ended, `ms` after the step before it (a pure decode
 * step of the same `n` drafting rows): its verify's rows R, the rounds D its drafter ran, the
 * tokens it made. */
inline void draft_depth_online_step(const DraftDepthCfg& c, DraftDepthOnline* o, int n, int R, int D,
                                    double ms, int tokens) {
    if (!(ms > 0.0) || ms > 500.0 || n < 1 || R < 1 || tokens < 1) return;
    const int nn = draft_depth_seqs(n);
    DraftDepthOnline::Reg& G = o->reg[nn];
    /* a stall (a client arriving, a page fault) is no price of a row */
    if (G.n >= 32 && (ms > 2.5 * G.ms_avg || ms < 0.4 * G.ms_avg)) { ++o->outliers; return; }
    ++o->steps;
    const double wa = 1.0 / (double)(G.n + 1);
    G.ms_avg += (wa > 1.0 / 64.0 ? wa : 1.0 / 64.0) * (ms - G.ms_avg);
    ++G.n;
    DraftDepthOnline::Lam& L = o->lam[nn];
    const double wl = 1.0 / (double)(L.n + 1);
    const double w = wl > 0.010772 ? wl : 0.010772;          /* 1 - 2^(-1/64) */
    L.tok += w * ((double)tokens - L.tok);
    L.ms  += w * (ms - L.ms);
    ++L.n;
    double x[DraftDepthOnline::kF];
    x[0] = 1.0; x[1] = (double)R; x[2] = (double)D;
    for (int k = 0; k < DraftDepthCfg::kJumps; ++k) x[3 + k] = R > c.load_jump_at[k] ? 1.0 : 0.0;
    for (int a = 0; a < DraftDepthOnline::kF; ++a) {
        G.b[a] = G.b[a] * o->reg_keep + x[a] * ms;
        for (int b = 0; b < DraftDepthOnline::kF; ++b) G.S[a][b] = G.S[a][b] * o->reg_keep + x[a] * x[b];
    }
}

/* The tables from what was fed so far, into `c`. Between steps only. */
inline void draft_depth_online_refresh(DraftDepthCfg* c, DraftDepthOnline* o) {
    enum { kF = DraftDepthOnline::kF, kK = DraftDepthOnline::kK };
    ++o->refreshes;
    const int K = draft_depth_knots(*c);
    for (int t = 0; t < 3; ++t)
        for (int j = 2; j <= DraftDepthOnline::kR; ++j) {
            /* each knot's estimate with the static table as pseudo-observations, then pooled
             * until it does not fall as the confidence rises (pool adjacent violators) */
            double v[kK], w[kK];
            int    lo[kK], nb = 0;
            for (int k = 0; k < K; ++k) {
                const double wk = o->cn[t][j][k] + o->prior_w;
                const double vk = (o->ca[t][j][k] + o->prior_w * o->cal0[t][j][k]) / wk;
                v[nb] = vk; w[nb] = wk; lo[nb] = k; ++nb;
                while (nb > 1 && v[nb - 2] > v[nb - 1]) {
                    v[nb - 2] = (v[nb - 2] * w[nb - 2] + v[nb - 1] * w[nb - 1]) / (w[nb - 2] + w[nb - 1]);
                    w[nb - 2] += w[nb - 1];
                    --nb;
                }
            }
            for (int b = 0; b < nb; ++b) {
                const int hi = b + 1 < nb ? lo[b + 1] : K;
                const double p = v[b] < 0.0 ? 0.0 : (v[b] > 1.0 ? 1.0 : v[b]);
                for (int k = lo[b]; k < hi; ++k) c->cal_p[t][j][k] = p;
            }
            for (int k = 0; k < K; ++k) { o->cn[t][j][k] *= o->cal_keep; o->ca[t][j][k] *= o->cal_keep; }
        }
    for (int n = 1; n <= DraftDepthOnline::kN; ++n) {
        const DraftDepthOnline::Lam& L = o->lam[n];
        if (L.n >= 32 && L.ms > 0.0 && L.tok > 0.0) c->load_lam[n] = L.tok / L.ms;
        const DraftDepthOnline::Reg& G = o->reg[n];
        if (G.n < 64) continue;
        /* (S + P) theta = b + P theta0, by Cholesky: S + P is positive definite */
        double A[kF][kF], y[kF];
        for (int a = 0; a < kF; ++a) {
            y[a] = G.b[a] + o->prec[a] * o->theta0[n][a];
            for (int b = 0; b < kF; ++b) A[a][b] = G.S[a][b] + (a == b ? o->prec[a] : 0.0);
        }
        bool ok = true;
        for (int a = 0; a < kF && ok; ++a) {
            for (int b = 0; b <= a; ++b) {
                double sum = A[a][b];
                for (int q = 0; q < b; ++q) sum -= A[a][q] * A[b][q];
                if (a == b) { if (!(sum > 1e-12)) { ok = false; break; } A[a][a] = std::sqrt(sum); }
                else A[a][b] = sum / A[b][b];
            }
        }
        if (!ok) continue;
        for (int a = 0; a < kF; ++a) {
            double sum = y[a];
            for (int q = 0; q < a; ++q) sum -= A[a][q] * y[q];
            y[a] = sum / A[a][a];
        }
        for (int a = kF - 1; a >= 0; --a) {
            double sum = y[a];
            for (int q = a + 1; q < kF; ++q) sum -= A[q][a] * y[q];
            y[a] = sum / A[a][a];
        }
        bool finite = true;
        for (int a = 0; a < kF; ++a) finite = finite && y[a] == y[a] && y[a] > -1e6 && y[a] < 1e6;
        if (!finite) continue;
        auto clamp = [](double v, double lo_, double hi_) { return v < lo_ ? lo_ : (v > hi_ ? hi_ : v); };
        for (int a = 0; a < kF; ++a) o->theta[n][a] = y[a];
        c->load_row_ms[n]   = clamp(y[1], 0.05, 5.0);
        c->load_round_ms[n] = clamp(y[2], 0.10, 5.0);
        for (int k = 0; k < DraftDepthCfg::kJumps; ++k) c->load_jump_ms[n][k] = clamp(y[3 + k], 0.0, 20.0);
    }
}

/* The whole rule at once, for a chain whose statistics are all known (tests, offline): s[i] is
 * round i + 1's, n of them (n >= max - 1 - lag is enough). Returns the depth D in [1, max]. */
inline int draft_depth_of(const DraftDepthCfg& c, const double* s, int n) {
    DraftDepthRow r;
    draft_depth_begin(c, &r, false);
    for (int j = 2; j <= c.max; ++j) {
        const int need = draft_depth_need(c, j);
        while (r.seen < need && r.seen < n) draft_depth_observe(&r, r.seen + 1, s[r.seen]);
        if (!draft_depth_go(c, &r, j)) break;
    }
    return r.depth < 1 ? 1 : r.depth;
}

}  /* namespace rad */
