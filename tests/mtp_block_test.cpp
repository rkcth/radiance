/* tests/mtp_block_test.cpp -- Block Verification of a linear MTP draft chain (--draft-verify block,
 * abi/rad_mtp_block.h, arXiv 2403.10444), checked against its own law.
 *
 * The property: whatever the head's law Q and the chain's length g, the tokens a verify step commits
 * are a prefix of a sample of the TARGET's autoregressive law, and the step commits at least as many
 * tokens as the token-wise (maximal coupling) verify of the same drafts. Tests:
 *
 *   exact enumeration      every draft of a toy chain, the rule written out densely and independently
 *                          of the header, the law of the first three committed tokens summed in
 *                          closed form: equals the target's product law to 1e-12 (g = 1..4, three
 *                          head qualities); and the header's h_i equal the dense ones
 *   chi-square             the real shared code (the propose row, rad_mtp_couple_verify_row with the
 *                          block marking, the core's token-comparison walk) over 3e5 seeds: the joint
 *                          law of the first three tokens and the first token's marginal
 *   negative control       the same code with h_i nudged (a wrong acceptance rule) is NOT exact and
 *                          the same statistic sees it
 *   dominance              the mean committed tokens per step of block verification is at least the
 *                          token-wise verify's on the same drafts
 *   inert rows             a stale or non-consecutive state leaves the ordinary picks alone; rows
 *                          after the first of a block write nothing; a greedy row is untouched
 */
#include "rad_test.h"

#include "rad_mtp_block.h"
#include "../plugin/spechub/spechub.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <vector>

namespace {

constexpr int V = 12;   /* larger than the head's top 8: the target has mass off Q's support */

uint64_t H(uint64_t a) { return rad_mtp_rrs_mix(a); }

struct Law {
    int32_t pid[V];  float pval[V];  int pn = 0;
    double  p[V];
    int32_t qid[8];  float qval[8];
};

Law law_at(uint64_t prefix_hash, double head_noise) {
    Law L;
    float lt[V], lq[V];
    for (int i = 0; i < V; ++i) {
        lt[i] = (float)((H(prefix_hash * 131 + (uint64_t)i * 977 + 1) >> 40) % 3000) / 1000.0f - 1.5f;
        const float nz = (float)((H(prefix_hash * 257 + (uint64_t)i * 331 + 7) >> 40) % 3000) / 1000.0f - 1.5f;
        lq[i] = 0.75f * lt[i] + (float)head_noise * nz;
    }
    const uint64_t h = H(prefix_hash + 99);
    const int drop = h % 3 == 0 ? (int)((h >> 8) % V) : -1;
    double z = 0;
    for (int i = 0; i < V; ++i) { L.p[i] = 0; if (i != drop) { L.pid[L.pn] = i; L.pval[L.pn++] = lt[i]; } }
    for (int j = 0; j < L.pn; ++j) z += std::exp((double)L.pval[j]);
    for (int j = 0; j < L.pn; ++j) L.p[L.pid[j]] = std::exp((double)L.pval[j]) / z;
    int ord[V];
    for (int i = 0; i < V; ++i) ord[i] = i;
    std::stable_sort(ord, ord + V, [&](int a, int b) { return lq[a] > lq[b]; });
    for (int i = 0; i < 8; ++i) { L.qid[i] = ord[i]; L.qval[i] = lq[ord[i]]; }
    return L;
}

uint64_t child_hash(uint64_t ph, int tok) { return H(ph * 1000003ull + (uint64_t)tok + 17); }

int sample_law(const Law& L, uint64_t seed, uint64_t pos, uint64_t key) {
    double u = rad_spechub_uniform(seed, pos, key);
    int last = 0;
    for (int i = 0; i < V; ++i) {
        if (L.p[i] <= 0) continue;
        last = i;
        u -= L.p[i];
        if (u < 0) return i;
    }
    return last;
}

RadMtpCoupleDraw make_draw(uint64_t seed, uint64_t base) {
    RadMtpCoupleDraw d{};
    d.seed = seed; d.base = base; d.seq = 1; d.card = 0; d.mode = 1;
    return d;
}

/* ------------------------------------------------------------------ a drafted chain */

struct Chain {
    int      g = 0;
    int      tok[8] = {};            /* X_1 .. X_g */
    uint64_t hash[9] = {};           /* the prefix hash after X^i, i = 0 .. g */
    RadMtpCoupleState st[8] = {};    /* row i's coupled state (round i + 1) */
};

/* The chain drafted by the real propose row, round by round, over the head's law at the chain's own
 * prefix (x ~ Q at the draw's (seed, base, KEY_DRAFT + round)). */
void draft_chain(Chain* t, uint64_t seed, uint64_t base, int g, double noise, uint64_t h0 = 1) {
    uint64_t h = h0;
    t->g = g;
    t->hash[0] = h;
    for (int r = 1; r <= g; ++r) {
        const Law law = law_at(h, noise);
        int32_t idx[8];
        float   val[8];
        for (int i = 0; i < 8; ++i) { idx[i] = law.qid[i]; val[i] = law.qval[i]; }
        const RadMtpCoupleDraw draw = make_draw(seed, base);
        const int32_t card = 0;
        RadMtpCoupleProposeCall c{};
        c.idx = idx; c.val = val; c.card = &card;
        c.draw = reinterpret_cast<const uint32_t*>(&draw);
        RadMtpCoupleState rows[8] = {};
        c.state = reinterpret_cast<uint32_t*>(rows);
        c.M = 1; c.vocab = 1000;
        c.ld_idx = 8; c.ld_val = 8; c.ld_draw = RAD_MTP_COUPLE_DRAW_WORDS;
        c.st_round = RAD_MTP_COUPLE_STATE_WORDS; c.st_row = 0;
        c.round = r;
        rad_mtp_couple_propose_row(c, 0);
        t->st[r - 1] = rows[r - 1];
        t->tok[r - 1] = idx[0];
        h = child_hash(h, idx[0]);
        t->hash[r] = h;
    }
}

enum Mode { TOKEN, BLOCK, CONTROL };

/* The control's wrong rule: every inner h_i raised by a quarter (clipped at 1). */
void nudge(RadBlockPlan* pl) {
    for (int i = 1; i < pl->g; ++i) pl->h[i] = std::min(1.0, pl->h[i] * 1.25);
}

/* The sampler's planes of one step: g + 1 rows, the last the bonus row with no state. */
struct Planes {
    int32_t  ids[9 * V];
    float    val[9 * V];
    uint32_t params[9 * 24];
    RadMtpCoupleState st[9];
    int32_t  tok[9];
};

RadMtpCoupleVerifyCall call_of(Planes* pl, int g) {
    RadMtpCoupleVerifyCall c{};
    c.idx = pl->ids; c.val = pl->val; c.params = pl->params;
    c.state = reinterpret_cast<const uint32_t*>(pl->st);
    c.token = pl->tok;
    c.M = g + 1; c.n_cand = V; c.vocab = 1000;
    c.ld[0] = V; c.ld[1] = V; c.ld[2] = 24; c.ld[3] = RAD_MTP_COUPLE_STATE_WORDS; c.ld[4] = 1;
    return c;
}

/* One verify step over the chain, walked as the core does: the verify op's rows over the sampler's
 * planes, then the rule "a row's token that equals its draft accepts it". Writes the committed
 * tokens, extended to `want` by exact draws from the target. Returns the committed count before the
 * extension. */
int step(const Chain& t, uint64_t seed, uint64_t n_out, double noise, Mode mode, int want, int* out) {
    const int g = t.g;
    Planes pl;
    std::memset(&pl, 0, sizeof pl);
    for (int i = 0; i <= g; ++i) {
        const Law L = law_at(t.hash[i], noise);
        for (int j = 0; j < V; ++j) {
            pl.ids[i * V + j] = j < L.pn ? L.pid[j] : -1;
            pl.val[i * V + j] = j < L.pn ? L.pval[j] : 0.0f;
        }
        RadSampleParams* p = reinterpret_cast<RadSampleParams*>(pl.params + i * 24);
        p->seed = seed;
        p->pos = n_out + (uint64_t)i;
        pl.tok[i] = sample_law(L, seed, n_out + (uint64_t)i, 0);   /* the ordinary pick */
        if (i < g) {
            pl.st[i] = t.st[i];
            pl.st[i].pos = rad_mtp_couple_verify_pos(n_out, i + 1);
            pl.st[i].blk = mode == TOKEN ? 0 : g;
        }
    }
    const RadMtpCoupleVerifyCall c = call_of(&pl, g);
    if (mode == CONTROL) {
        RadBlockRow rows[8];
        RadBlockStat stat[8];
        for (int i = 0; i < g; ++i) {
            rows[i] = RadBlockRow{ pl.ids + i * V, pl.val + i * V, V, 0, 0.0f };
            rad_mtp_block_scan(&rows[i], 1000);
            CHECK(rad_mtp_block_stat(rows[i], pl.st[i], &stat[i]));
        }
        RadBlockPlan bp;
        rad_mtp_block_plan(pl.st, stat, g, &bp);
        nudge(&bp);
        const int tau = rad_mtp_block_tau(bp, seed, n_out);
        int32_t y = -1;
        if (tau < g)
            CHECK(rad_mtp_block_resid_draw(rows[tau], pl.st[tau], stat[tau], bp.p[tau],
                                           rad_spechub_uniform(seed, n_out + (uint64_t)tau,
                                                               RAD_SAMPLE_KEY_RESID), &y));
        rad_mtp_block_write(c, 0, pl.st, tau, g, y);
    } else {
        for (int i = 0; i < g; ++i) rad_mtp_couple_verify_row(c, i);
    }
    int len = 0;
    for (int i = 0; i <= g; ++i) {
        out[len++] = pl.tok[i];
        if (i == g || pl.tok[i] != t.tok[i]) break;
    }
    const int committed = len;
    uint64_t ph = t.hash[0];
    for (int i = 0; i < len; ++i) ph = child_hash(ph, out[i]);
    while (len < want) {
        const Law L = law_at(ph, noise);
        const int y = sample_law(L, seed ^ 0xabcdef1234ull, n_out + (uint64_t)len, 1);
        out[len++] = y;
        ph = child_hash(ph, y);
    }
    return committed;
}

double zscore(const std::vector<double>& counts, const std::vector<double>& expect, double n) {
    double chi = 0, pool_c = 0, pool_e = 0;
    int cells = 0;
    for (size_t i = 0; i < counts.size(); ++i) {
        const double e = expect[i] * n;
        if (e < 20) { pool_c += counts[i]; pool_e += e; continue; }
        chi += (counts[i] - e) * (counts[i] - e) / e;
        ++cells;
    }
    if (pool_e >= 5) { chi += (pool_c - pool_e) * (pool_c - pool_e) / pool_e; ++cells; }
    const double df = cells - 1;
    return (chi - df) / std::sqrt(2.0 * df);
}

struct Result { double z, first_z, mean_len, var_len; };

Result run(Mode mode, int g, int trials, double noise) {
    std::vector<double> cnt3((size_t)V * V * V, 0), cnt1((size_t)V, 0);
    double sum = 0, sq = 0;
    for (int s = 0; s < trials; ++s) {
        const uint64_t seed = 0x1234567ull + (uint64_t)s * 7919ull;
        Chain t;
        draft_chain(&t, seed, 0, g, noise);
        int out[16];
        const int len = step(t, seed, 0, noise, mode, 3, out);
        ++cnt1[(size_t)out[0]];
        ++cnt3[(size_t)((out[0] * V + out[1]) * V + out[2])];
        sum += len; sq += (double)len * len;
    }
    std::vector<double> e1((size_t)V, 0), e3((size_t)V * V * V, 0);
    const Law l0 = law_at(1, noise);
    for (int a = 0; a < V; ++a) {
        e1[(size_t)a] = l0.p[a];
        const Law l1 = law_at(child_hash(1, a), noise);
        for (int b = 0; b < V; ++b) {
            const Law l2 = law_at(child_hash(child_hash(1, a), b), noise);
            for (int c = 0; c < V; ++c) e3[(size_t)((a * V + b) * V + c)] = l0.p[a] * l1.p[b] * l2.p[c];
        }
    }
    const double mean = sum / trials;
    return { zscore(cnt3, e3, trials), zscore(cnt1, e1, trials), mean, sq / trials - mean * mean };
}

/* ------------------------------------------------------------------ the exact law */

/* The rule written densely, independent of the header: P_i, Q_{i+1} as length-V vectors. */
struct Dense { double P[V], Q[V]; };

void dense_h(const std::vector<Dense>& d, const int* x, int g, double* h, double* p) {
    p[0] = 1;
    for (int i = 1; i <= g; ++i)
        p[i] = std::min(1.0, p[i - 1] * d[(size_t)i - 1].P[x[i - 1]] / d[(size_t)i - 1].Q[x[i - 1]]);
    h[g] = p[g];
    for (int i = 0; i < g; ++i) {
        double S = 0;
        for (int v = 0; v < V; ++v) S += std::max(p[i] * d[(size_t)i].P[v] - d[(size_t)i].Q[v], 0.0);
        h[i] = S / (S + 1.0 - p[i]);
    }
}

/* The exact law of the first three committed tokens (the step, then the target's own draws), by
 * summing over every draft, tau and Y. Also compares the header's weights with the dense ones. */
double exact_error(int gmax, double noise, double* h_err, int adaptive = 0) {
    std::map<std::vector<int>, double> emit;   /* committed sequence -> probability */
    *h_err = 0;
    std::vector<int> x((size_t)gmax);
    std::vector<uint64_t> hash((size_t)gmax + 1);
    hash[0] = 1;
    std::function<void(int, double)> rec = [&](int i, double pr) {
        /* adaptive: the draft stage stops after X_i when a function of the drafted prefix says so
         * (what confidence pruning does); the block's length g is then draft-dependent. */
        const bool stop_here = adaptive && i >= 1 && i < gmax && (H(hash[(size_t)i] + 5) % 3) < (uint64_t)adaptive;
        const int g = i;
        if (i < gmax && !stop_here) {
            const Law L = law_at(hash[(size_t)i], noise);
            RadMtpCoupleState s{};
            CHECK(rad_mtp_couple_build_q(L.qid, L.qval, 1000, &s));
            for (int k = 0; k < s.count; ++k) {
                x[(size_t)i] = s.ids[k];
                hash[(size_t)i + 1] = child_hash(hash[(size_t)i], s.ids[k]);
                rec(i + 1, pr * s.q[k]);
            }
            return;
        }
        std::vector<Dense> d((size_t)g);
        std::vector<Law> laws;
        std::vector<RadMtpCoupleState> st((size_t)g);
        std::vector<RadBlockRow> rows((size_t)g);
        for (int j = 0; j < g; ++j) laws.push_back(law_at(hash[(size_t)j], noise));
        for (int j = 0; j < g; ++j) {
            CHECK(rad_mtp_couple_build_q(laws[(size_t)j].qid, laws[(size_t)j].qval, 1000, &st[(size_t)j]));
            st[(size_t)j].x = x[(size_t)j];
            for (int v = 0; v < V; ++v) { d[(size_t)j].P[v] = laws[(size_t)j].p[v]; d[(size_t)j].Q[v] = 0; }
            for (int k = 0; k < st[(size_t)j].count; ++k)
                d[(size_t)j].Q[st[(size_t)j].ids[k]] = st[(size_t)j].q[k];
            rows[(size_t)j] = RadBlockRow{ laws[(size_t)j].pid, laws[(size_t)j].pval,
                                           laws[(size_t)j].pn, 0, 0.0f };
            rad_mtp_block_scan(&rows[(size_t)j], 1000);
        }
        double h[9], p[9];
        dense_h(d, x.data(), g, h, p);
        RadBlockStat stat[8];
        for (int j = 0; j < g; ++j) CHECK(rad_mtp_block_stat(rows[(size_t)j], st[(size_t)j], &stat[j]));
        RadBlockPlan bp;
        rad_mtp_block_plan(st.data(), stat, g, &bp);
        for (int j = 0; j <= g; ++j) *h_err = std::max(*h_err, std::fabs(bp.h[j] - h[j]));
        double stop = 1;   /* probability that no larger i has accepted */
        for (int tau = g; tau >= 0; --tau) {
            const double pt = stop * h[tau];
            stop *= 1 - h[tau];
            if (pt <= 0) continue;
            std::vector<int> e(x.begin(), x.begin() + tau);
            if (tau == g) {
                const Law Lg = law_at(hash[(size_t)g], noise);
                for (int z = 0; z < V; ++z) {
                    if (Lg.p[z] <= 0) continue;
                    std::vector<int> e2 = e; e2.push_back(z);
                    emit[e2] += pr * pt * Lg.p[z];
                }
                continue;
            }
            double S = 0, r[V];
            for (int v = 0; v < V; ++v) {
                r[v] = std::max(p[tau] * d[(size_t)tau].P[v] - d[(size_t)tau].Q[v], 0.0);
                S += r[v];
            }
            if (!(S > 0)) { for (int v = 0; v < V; ++v) r[v] = d[(size_t)tau].P[v]; S = 1; }
            for (int v = 0; v < V; ++v) {
                if (r[v] <= 0) continue;
                std::vector<int> e2 = e; e2.push_back(v);
                emit[e2] += pr * pt * r[v] / S;
            }
        }
    };
    rec(0, 1.0);
    /* extend every committed sequence to three tokens with the target's conditionals */
    std::vector<double> got((size_t)V * V * V, 0);
    for (const auto& kv : emit) {
        const std::vector<int>& e = kv.first;
        std::function<void(std::vector<int>&, uint64_t, double)> ext =
            [&](std::vector<int>& w, uint64_t h, double pr) {
                if (w.size() == 3) { got[(size_t)((w[0] * V + w[1]) * V + w[2])] += pr; return; }
                const Law L = law_at(h, noise);
                for (int v = 0; v < V; ++v) {
                    if (L.p[v] <= 0) continue;
                    w.push_back(v);
                    ext(w, child_hash(h, v), pr * L.p[v]);
                    w.pop_back();
                }
            };
        std::vector<int> w(e.begin(), e.begin() + (long)std::min<size_t>(e.size(), 3));
        uint64_t h = 1;
        for (int t : w) h = child_hash(h, t);
        ext(w, h, kv.second);
    }
    const Law l0 = law_at(1, noise);
    double err = 0, tot = 0;
    for (int a = 0; a < V; ++a) {
        const Law l1 = law_at(child_hash(1, a), noise);
        for (int b = 0; b < V; ++b) {
            const Law l2 = law_at(child_hash(child_hash(1, a), b), noise);
            for (int c = 0; c < V; ++c) {
                const size_t i = (size_t)((a * V + b) * V + c);
                const double want = l0.p[a] * l1.p[b] * l2.p[c];
                err = std::max(err, std::fabs(got[i] - want));
                tot += got[i];
            }
        }
    }
    CHECK_NEAR(tot, 1.0, 1e-12);
    return err;
}

}  /* namespace */

TEST(block_verification_emits_exactly_the_target_law_by_enumeration) {
    for (double noise : { 0.3, 1.0, 3.0 })
        for (int g = 1; g <= 4; ++g) {
            double h_err = 0;
            const double err = exact_error(g, noise, &h_err);
            std::printf("  g=%d noise=%.1f  max |law - target| = %.2e   max |h - h_dense| = %.2e\n", g,
                        noise, err, h_err);
            CHECK(err < 1e-12);
            CHECK(h_err < 1e-12);
        }
}

/* A draft stage that stops by a function of the drafted prefix (confidence pruning, dynamic rows)
 * makes the block length draft-dependent. Printed, and checked to be exact. */
TEST(block_verification_with_a_draft_dependent_chain_length_is_exact) {
    for (double noise : { 0.3, 1.0, 3.0 })
        for (int adaptive : { 1, 2 }) {
            double h_err = 0;
            const double err = exact_error(4, noise, &h_err, adaptive);
            std::printf("  adaptive stop (%d/3 of prefixes) noise=%.1f: max |law - target| = %.2e\n",
                        adaptive, noise, err);
            CHECK(err < 1e-12);
        }
}

TEST(the_chain_verify_with_block_marking_commits_the_target_law) {
    const Result a = run(BLOCK, 3, 300000, 1.0);
    std::printf("  g=3 noise 1.0: z=%.2f first=%.2f mean tokens %.4f\n", a.z, a.first_z, a.mean_len);
    CHECK(a.z < 4.5);  CHECK(a.first_z < 4.5);
    const Result b = run(BLOCK, 3, 150000, 3.0);
    std::printf("  g=3 noise 3.0: z=%.2f first=%.2f mean tokens %.4f\n", b.z, b.first_z, b.mean_len);
    CHECK(b.z < 4.5);  CHECK(b.first_z < 4.5);
    const Result c = run(BLOCK, 5, 150000, 0.3);
    std::printf("  g=5 noise 0.3: z=%.2f first=%.2f mean tokens %.4f\n", c.z, c.first_z, c.mean_len);
    CHECK(c.z < 4.5);  CHECK(c.first_z < 4.5);
}

/* The same statistic sees a wrong acceptance rule. */
TEST(a_block_rule_with_nudged_weights_is_not_exact) {
    const Result r = run(CONTROL, 3, 300000, 1.0);
    std::printf("  nudged h (known wrong): z=%.1f first=%.1f\n", r.z, r.first_z);
    CHECK(r.z > 8.0);
}

/* Block verification commits at least as many tokens as the token-wise verify of the same drafts,
 * and both are exact (the token-wise one is the existing coupling). */
TEST(block_verification_commits_at_least_as_many_tokens_as_token_wise) {
    for (double noise : { 0.3, 1.0, 3.0 }) {
        const int trials = 200000;
        const Result tk = run(TOKEN, 3, trials, noise);
        const Result bl = run(BLOCK, 3, trials, noise);
        const double se = std::sqrt((tk.var_len + bl.var_len) / trials);
        std::printf("  noise %.1f: token-wise %.4f tokens/step (z=%.2f), block %.4f (z=%.2f), +%.4f\n",
                    noise, tk.mean_len, tk.z, bl.mean_len, bl.z, bl.mean_len - tk.mean_len);
        CHECK(tk.z < 4.5);
        CHECK(bl.z < 4.5);
        CHECK(bl.mean_len > tk.mean_len - 4 * se);
    }
}

/* A state that is not the one the sampler would have bound, or a block that is not consecutive,
 * leaves the ordinary picks alone; only the first row of a block writes. */
TEST(block_rows_that_do_not_bind_keep_their_ordinary_picks) {
    const int g = 3;
    Chain t;
    draft_chain(&t, 77, 0, g, 1.0);
    auto build = [&](Planes* pl, uint64_t seed) {
        std::memset(pl, 0, sizeof *pl);
        for (int i = 0; i <= g; ++i) {
            const Law L = law_at(t.hash[i], 1.0);
            for (int j = 0; j < V; ++j) {
                pl->ids[i * V + j] = j < L.pn ? L.pid[j] : -1;
                pl->val[i * V + j] = j < L.pn ? L.pval[j] : 0.0f;
            }
            RadSampleParams* p = reinterpret_cast<RadSampleParams*>(pl->params + i * 24);
            p->seed = seed; p->pos = 10 + (uint64_t)i;
            pl->tok[i] = 1000 + i;   /* a marker no verify could write */
            if (i < g) { pl->st[i] = t.st[i]; pl->st[i].pos = 10 + (uint64_t)i; pl->st[i].blk = g; }
        }
    };
    auto verify = [&](Planes* pl, int row0, int rows) {
        const RadMtpCoupleVerifyCall c = call_of(pl, g);
        for (int i = row0; i < row0 + rows; ++i) rad_mtp_couple_verify_row(c, i);
    };
    auto untouched = [&](const Planes& pl) {
        for (int i = 0; i <= g; ++i) if (pl.tok[i] != 1000 + i) return false;
        return true;
    };
    Planes pl;
    build(&pl, 77);
    verify(&pl, 0, g);
    CHECK(!untouched(pl));                       /* the block binds and writes */
    build(&pl, 77);
    verify(&pl, 1, g - 1);
    CHECK(untouched(pl));                        /* rows after the first write nothing */
    build(&pl, 77);
    pl.st[1].round = 3;
    verify(&pl, 0, g);
    CHECK(untouched(pl));                        /* a row of the wrong round ends the block's coupling */
    build(&pl, 77);
    pl.st[2].draw.seq = 9;
    verify(&pl, 0, g);
    CHECK(untouched(pl));                        /* another request's state */
    build(&pl, 77);
    pl.st[2].pos += 1;
    verify(&pl, 0, g);
    CHECK(untouched(pl));                        /* a stale position */
    build(&pl, 77);
    reinterpret_cast<RadSampleParams*>(pl.params + 2 * 24)->flags = RAD_SP_GREEDY;
    verify(&pl, 0, g);
    CHECK(untouched(pl));                        /* a greedy row */
    build(&pl, 78);
    verify(&pl, 0, g);
    CHECK(untouched(pl));                        /* a seed the state was not drawn under */
    build(&pl, 77);
    pl.st[1].nkids = 2;
    verify(&pl, 0, g);
    CHECK(untouched(pl));                        /* a tree node is not a chain row */
}


/* The residual draw over a plane WIDER than the workgroup's 128 lanes (several slots per lane, ids
 * on and off Q's support, holes): the lane-ordered walk of rad_mtp_block_resid_draw must still
 * sample max(p P - Q, 0) normalised. Chi-square over the plane's ids against the dense closed
 * form; the negative control draws from a residual with Q's mass halved (a wrong law) and the same
 * statistic must see it. Also the tree sum: stat over the wide plane equals a plain ascending sum
 * to rounding (the tree is a different order, not a different value). */
TEST(the_lane_ordered_residual_draw_over_a_wide_plane_is_exact) {
    const int N = 700;
    std::vector<int32_t> ids((size_t)N);
    std::vector<float> val((size_t)N);
    for (int j = 0; j < N; ++j) {
        ids[(size_t)j] = j % 11 == 5 ? -1 : 2 * j + 3;
        val[(size_t)j] = j % 11 == 5 ? -INFINITY : (float)((H((uint64_t)j * 7919 + 5) >> 40) % 4000) / 1000.0f - 2.0f;
    }
    int32_t qid[8]; float qv[8];
    for (int k = 0; k < 8; ++k) { qid[k] = 2 * (k * 83 + 1) + 3; qv[k] = 1.5f - 0.3f * (float)k; }
    RadMtpCoupleState st{};
    CHECK(rad_mtp_couple_build_q(qid, qv, 100000, &st));
    st.x = st.ids[1];
    RadBlockRow r{ ids.data(), val.data(), N, 0, 0.0f };
    rad_mtp_block_scan(&r, 100000);
    RadBlockStat t;
    CHECK(rad_mtp_block_stat(r, st, &t));
    double ref = 0, top = r.top;
    for (int j = 0; j < N; ++j) if (ids[(size_t)j] >= 0) ref += std::exp((double)val[(size_t)j] - top);
    CHECK(std::fabs(t.sum - ref) <= 1e-12 * ref);
    const double p = 0.8;
    std::map<int32_t, double> want;
    double tot = 0;
    for (int j = 0; j < N; ++j) {
        if (ids[(size_t)j] < 0) continue;
        const double P = std::exp((double)val[(size_t)j] - top) / ref;
        double Q = 0;
        for (int k = 0; k < st.count; ++k) if (st.ids[k] == ids[(size_t)j]) Q = st.q[k];
        const double m = std::max(p * P - Q, 0.0);
        want[ids[(size_t)j]] = m; tot += m;
    }
    const int D = 400000;
    std::map<int32_t, double> got, bad;
    for (int d = 0; d < D; ++d) {
        const double u = (double)(H((uint64_t)d * 2654435761ull + 11) >> 11) / 9007199254740992.0;
        int32_t y = -1;
        CHECK(rad_mtp_block_resid_draw(r, st, t, p, u, &y));
        got[y] += 1;
        /* control: the walk with the uniform skewed (a wrong law) */
        int32_t yb = -1;
        CHECK(rad_mtp_block_resid_draw(r, st, t, p, u * u, &yb));
        bad[yb] += 1;
    }
    auto chi = [&](std::map<int32_t, double>& c) {
        double x = 0, pc = 0, pe = 0; int cells = 0;
        for (auto& kv : want) {
            const double e = D * kv.second / tot, o = c.count(kv.first) ? c[kv.first] : 0;
            if (e < 5) { pc += o; pe += e; continue; }
            x += (o - e) * (o - e) / e; ++cells;
        }
        if (pe >= 5) { x += (pc - pe) * (pc - pe) / pe; ++cells; }
        return (x - (cells - 1)) / std::sqrt(2.0 * (cells - 1));   /* z of chi-square */
    };
    for (auto& kv : got) CHECK(want.count(kv.first) && want[kv.first] > 0);
    const double z = chi(got), zb = chi(bad);
    fprintf(stderr, "  wide plane residual draw: z=%.2f   negative control z=%.1f\n", z, zb);
    CHECK(std::fabs(z) < 4.5);
    CHECK(zb > 20);
}

RAD_TEST_MAIN()
