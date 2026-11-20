/* tests/mtp_rrs_test.cpp -- the verifications of a coupled draft (--draft-coupling on), checked
 * against their own law: recursive rejection sampling over a node's siblings (abi/rad_mtp_rrs.h) and
 * Traversal Verification (--draft-verify traversal, abi/rad_mtp_traversal.h).
 *
 * The property: whatever the head's law Q and the draft's shape, the tokens a verify step commits
 * are a prefix of a sample of the TARGET's autoregressive law. The tests run the real shared code
 * (the propose row, the verify row, the walk) over a toy model with a known target P(. | prefix) and
 * a different head Q(. | prefix), many seeds, and compare the empirical law of the emitted tokens
 * with the exact one (chi-square, deterministic seeds), or enumerate every draft tree of a small
 * model and compare the walk's law with the target's to 1e-10:
 *
 *   one node, 2..4 children   the first emitted token ~ P (single-node recursion, exact)
 *   chain, chain + siblings   R = 3 rounds, with and without three sibling drafts at the root,
 *                             walked by traversal (for a chain it is Block Verification's case)
 *   negative controls         the with-replacement verify, token-level ratios, no (1 - p) term and
 *                             a q that is never renormalised are NOT exact, and the same statistic
 *                             sees each of them
 *   degenerate                Q == P accepts the first child; disjoint supports reject all and emit
 *                             from P
 *   suffix step               a chain copied from the sequence's own history (--draft-suffix) emits
 *                             the target law beside the head's stale coupled states, alone and
 *                             mixed with adaptive-depth head steps; the same chain visible to the
 *                             coupled rows (what Request::draft_suffix prevents) does not
 *   adaptive depth            a chain cut by the rule of sched/draft_depth.h keeps the target law
 *                             under both verifies; two rules that read what they may not do not
 *   fixed point               the walk's integer exponential tracks exp to 2^-22, its law is the
 *                             softmax to FP32 accuracy, and by enumeration the walk emits exactly
 *                             the law it reads
 *
 * The step is walked the way the core walks it: the verify row writes the sampler's token for each
 * visited node and ADV_ACCEPT's rule (the first child whose token is that token) follows it; after
 * the step, the sequence continues from the exact target so the first three tokens of every trial
 * have a known joint law.
 */
#include "rad_test.h"

#include "rad_mtp_rrs.h"
#include "rad_mtp_traversal.h"
#include "../plugin/spechub/spechub.h"
#include "sched/draft_depth.h"

#include <algorithm>
#include <functional>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int V = 12;  /* the toy vocabulary: larger than the head's top 8, so the target has mass off Q's support */

uint64_t H(uint64_t a) { return rad_mtp_rrs_mix(a); }

/* The target's candidate plane at a prefix (ids and logits; some prefixes drop a token, as a
 * top-k / top-p would) and the head's shortlist (all V tokens, logits). The head is a noisy
 * version of the target, so acceptance is neither 0 nor 1. */
struct Law {
    int32_t pid[V];  float pval[V];  int pn = 0;      /* target candidates */
    double  p[V];                                      /* the exact target law over tokens */
    int32_t qid[8];  float qval[8];                    /* the head's shortlist */
};

double g_sharp = 1.0;   /* the laws' inverse temperature: a peaked target (T = 0.7 and a top-p) is the engine's regime */

Law law_at(uint64_t prefix_hash, double head_noise) {
    Law L;
    float lt[V], lq[V];
    for (int i = 0; i < V; ++i) {
        lt[i] = (float)g_sharp * ((float)((H(prefix_hash * 131 + (uint64_t)i * 977 + 1) >> 40) % 3000) / 1000.0f - 1.5f);
        const float nz = (float)((H(prefix_hash * 257 + (uint64_t)i * 331 + 7) >> 40) % 3000) / 1000.0f - 1.5f;
        lq[i] = 0.75f * lt[i] + (float)(head_noise * g_sharp) * nz;
    }
    const uint64_t h = H(prefix_hash + 99);
    const int drop = h % 3 == 0 ? (int)((h >> 8) % V) : -1;   /* a target token the filters removed */
    double z = 0;
    for (int i = 0; i < V; ++i) { L.p[i] = 0; if (i != drop) { L.pid[L.pn] = i; L.pval[L.pn++] = lt[i]; } }
    for (int j = 0; j < L.pn; ++j) z += std::exp((double)L.pval[j]);
    for (int j = 0; j < L.pn; ++j) L.p[L.pid[j]] = std::exp((double)L.pval[j]) / z;
    /* the head reports its 8 best (descending, ties to the lower id): the rest of the vocabulary is
     * off Q's support, where the target keeps its mass */
    int ord[V];
    for (int i = 0; i < V; ++i) ord[i] = i;
    std::stable_sort(ord, ord + V, [&](int a, int b) { return lq[a] > lq[b]; });
    for (int i = 0; i < 8; ++i) { L.qid[i] = ord[i]; L.qval[i] = lq[ord[i]]; }
    return L;
}

uint64_t child_hash(uint64_t ph, int tok) { return H(ph * 1000003ull + (uint64_t)tok + 17); }

/* A draw from the target law at the prefix: the ordinary pick. */
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

/* The verify row of the real plugin code over one candidate plane and one bound state. */
int verify(const RadMtpCoupleState& st, const Law& L, uint64_t seed, uint64_t pos, int fallback) {
    uint32_t params[24] = {};
    RadSampleParams* p = reinterpret_cast<RadSampleParams*>(params);
    p->seed = seed;
    p->pos = pos;
    int32_t tok = fallback;
    RadMtpCoupleVerifyCall c{};
    c.idx = L.pid; c.val = L.pval; c.params = params;
    c.state = reinterpret_cast<const uint32_t*>(&st);
    c.token = &tok;
    c.M = 1; c.n_cand = L.pn; c.vocab = 1000;
    c.ld[0] = L.pn; c.ld[1] = L.pn; c.ld[2] = 24; c.ld[3] = RAD_MTP_COUPLE_STATE_WORDS; c.ld[4] = 1;
    rad_mtp_couple_verify_row(c, 0);
    return tok;
}

/* ------------------------------------------------------------------ a toy drafted tree */

struct Tree {
    int      n = 0;                 /* nodes 1..n */
    int      tok[64] = {}, par[64] = {};
    uint64_t hash[64] = {};         /* the prefix hash of every node, node 0 the root */
    RadMtpCoupleState st[64] = {};  /* row k's coupled state (status 0: none) */
    bool     suffix = false;        /* a SuffixDecoding chain (Request::draft_suffix) */
};

/* The negative control of the suffix step: the coupled rows see its chain as their children. */
bool g_suffix_visible = false;

enum Kind { CHAIN, STAGE1 };
int g_scheme = 0;   /* sibling scheme of the root's drafts: 0 Gumbel-top-k, m + 1 RheoSampling */

/* The chain's drafting, through the real propose row: round r over the head's law at the chain
 * prefix. With L > 0 the root also drafts L siblings of its first child, sampled without
 * replacement from the same law (Gumbel-top-k, or RheoSampling's scheme), as a draft tree's root
 * does; its state then lists the children in draft order (abi/rad_mtp_rrs.h). */
void draft_chain(Tree* t, uint64_t seed, uint64_t base, int R, int L, double noise, uint64_t h0 = 1) {
    uint64_t h = h0;
    t->hash[0] = h;
    int n = 0;
    std::vector<int32_t> leaves;
    for (int r = 1; r <= R; ++r) {
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
        RadMtpCoupleState s = rows[r - 1];
        if (r == 1 && L > 0 && s.status == 1) {
            if (g_scheme > 0) rad_mtp_rrs_draw_rheo(&s, 0, 1 + L, g_scheme - 1);
            else              rad_mtp_rrs_draw(&s, 0, 1 + L);
            s.status = 1;
            for (int i = 0; i < 8; ++i) idx[i] = i < s.nkids ? s.ids[i] : -1;
        }
        t->st[r - 1] = s;
        const int pick = idx[0];
        if (r == 1)
            for (int i = 1; i < 8; ++i) {
                const int32_t id = idx[i];
                if (id < 0 || id >= 1000 || id == pick || (int)leaves.size() >= L) continue;
                if (std::find(leaves.begin(), leaves.end(), id) == leaves.end()) leaves.push_back(id);
            }
        ++n;
        t->tok[n] = pick; t->par[n] = n - 1;
        h = child_hash(h, pick);
        t->hash[n] = h;
    }
    for (int32_t id : leaves) {
        ++n;
        t->tok[n] = id; t->par[n] = 0;
        t->hash[n] = child_hash(t->hash[0], id);
    }
    t->n = n;
}

/* One verify step over the drafted tree, walked as the core does. Returns the emitted tokens
 * (the accepted nodes' tokens and the final one), extended by direct target draws to `want`. */
int walk(const Tree& t, uint64_t seed, uint64_t n_out, double noise, int want, int* out,
         int* accepted) {
    int len = 0, cur = 0, depth = 0;
    uint64_t h = t.hash[0];
    *accepted = 0;
    for (;;) {
        const Law law = law_at(h, noise);
        const uint64_t pos = n_out + (uint64_t)depth;
        /* the row's children, in node order, without repeats */
        int32_t kid[64];
        int nk = 0;
        for (int j = cur + 1; j <= t.n; ++j) {
            if (t.par[j] != cur) continue;
            if (t.suffix && !g_suffix_visible) continue;   /* Request::draft_suffix: no row binds a state */
            bool dup = false;
            for (int q = 0; q < nk; ++q) dup = dup || kid[q] == t.tok[j];
            if (!dup) kid[nk++] = t.tok[j];
        }
        int y = sample_law(law, seed, pos, 0);   /* the ordinary pick; an unbound row keeps it */
        const RadMtpCoupleState& c = t.st[cur];
        if (nk > 0 && c.status == 1) {
            const int kept = rad_mtp_rrs_prefix(c, kid, nk);
            if (kept > 0 && c.round == depth + 1 && c.x == kid[0]) {
                RadMtpCoupleState b = c;
                b.pos = rad_mtp_couple_verify_pos(n_out, depth + 1);
                if (c.nkids >= 2 || c.scheme) b.nkids = kept;
                y = verify(b, law, seed, pos, y);
            }
        }
        out[len++] = y;
        int nxt = -1;
        for (int j = cur + 1; j <= t.n && nxt < 0; ++j)
            if (t.par[j] == cur && t.tok[j] == y) nxt = j;
        if (nxt < 0) break;
        cur = nxt; ++depth; ++*accepted;
        h = child_hash(h, y);
    }
    /* the steps after this one: exact draws from the target given what was committed */
    uint64_t ph = t.hash[0];
    for (int i = 0; i < len; ++i) ph = child_hash(ph, out[i]);
    while (len < want) {
        const Law law = law_at(ph, noise);
        const int y = sample_law(law, seed ^ 0xabcdef1234ull, n_out + (uint64_t)len, 1);
        out[len++] = y;
        ph = child_hash(ph, y);
    }
    return len;
}

/* Chi-square of the empirical first-`m`-token law against the target's, cells with too little
 * expected mass pooled; returns (chi2 - df) / sqrt(2 df). */
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

struct Result { double z; double mean_acc; double first_z; };

/* Run `trials` steps of one drafted-tree kind and compare the first three emitted tokens' joint
 * law (and the first token's marginal) with the target's. */
Result run_kind(Kind kind, int trials, double noise,
                const std::function<void(Tree*, int)>* hook = nullptr) {
    std::vector<double> cnt3((size_t)V * V * V, 0), cnt1((size_t)V, 0);
    double acc = 0;
    for (int s = 0; s < trials; ++s) {
        const uint64_t seed = 0x1234567ull + (uint64_t)s * 7919ull;
        Tree t;
        if (kind == CHAIN)  draft_chain(&t, seed, 0, 3, 0, noise);
        else if (kind == STAGE1) draft_chain(&t, seed, 0, 3, 3, noise);
        if (hook) (*hook)(&t, s);
        int out[64], accepted = 0;
        walk(t, seed, 0, noise, 3, out, &accepted);
        ++cnt1[(size_t)out[0]];
        ++cnt3[(size_t)((out[0] * V + out[1]) * V + out[2])];
        acc += accepted;
    }
    std::vector<double> e1((size_t)V, 0), e3((size_t)V * V * V, 0);
    const uint64_t h0 = 1;
    const Law l0 = law_at(h0, noise);
    for (int a = 0; a < V; ++a) {
        e1[(size_t)a] = l0.p[a];
        const Law l1 = law_at(child_hash(h0, a), noise);
        for (int b = 0; b < V; ++b) {
            const Law l2 = law_at(child_hash(child_hash(h0, a), b), noise);
            for (int c = 0; c < V; ++c) e3[(size_t)((a * V + b) * V + c)] = l0.p[a] * l1.p[b] * l2.p[c];
        }
    }
    return { zscore(cnt3, e3, trials), acc / trials, zscore(cnt1, e1, trials) };
}


/* ------------------------------------------------------------------ traversal verification */

/* The walk's view of a drafted toy tree: the bound states exactly as `walk` binds them. */
struct TravTree {
    const Tree& t;
    double noise;
    uint64_t seed, n_out;
    Law laws[64];
    RadMtpCoupleState bound[64];
    bool has[64] = {};
    int depth[64] = {};
    TravTree(const Tree& tr, uint64_t sd, uint64_t no, double nz) : t(tr), noise(nz), seed(sd), n_out(no) {
        depth[0] = 0;
        for (int j = 0; j <= t.n; ++j) {
            if (j > 0) depth[j] = depth[t.par[j]] + 1;
            laws[j] = law_at(t.hash[j], noise);
            const RadMtpCoupleState& c = t.st[j];
            int32_t kid[64];
            int nk = 0;
            for (int q = j + 1; q <= t.n; ++q) {
                if (t.par[q] != j) continue;
                if (t.suffix && !g_suffix_visible) continue;   /* Request::draft_suffix */
                bool dup = false;
                for (int z = 0; z < nk; ++z) dup = dup || kid[z] == t.tok[q];
                if (!dup) kid[nk++] = t.tok[q];
            }
            if (nk > 0 && c.status == 1) {
                const int kept = rad_mtp_rrs_prefix(c, kid, nk);
                if (kept > 0 && c.round == depth[j] + 1 && c.x == kid[0]) {
                    bound[j] = c;
                    bound[j].pos = rad_mtp_couple_verify_pos(n_out, depth[j] + 1);
                    if (c.nkids >= 2 || c.scheme) bound[j].nkids = kept;
                    has[j] = true;
                }
            }
        }
    }
    const RadMtpCoupleState* state(int v) const { return has[v] ? &bound[v] : nullptr; }
    bool cand(int v, const int32_t** ids, const float** val, int64_t* n) const {
        *ids = laws[v].pid; *val = laws[v].pval; *n = laws[v].pn;
        return true;
    }
    int child(int v, int32_t token) const {
        for (int j = v + 1; j <= t.n; ++j)
            if (t.par[j] == v && t.tok[j] == token) return j;
        return -1;
    }
    int64_t vocab() const { return 1000; }
};

/* One step walked by traversal verification, then continued from the exact target. */
int walk_trav(const Tree& t, uint64_t seed, uint64_t n_out, double noise, int want, int* out,
              int* accepted, RadMtpTravOpt opt) {
    TravTree T(t, seed, n_out, noise);
    RadMtpTravOut o;
    int len = 0;
    *accepted = 0;
    if (rad_mtp_trav_walk(T, 0, seed, n_out, &o, opt)) {
        for (int i = 1; i < o.n; ++i) out[len++] = t.tok[o.node[i]];
        *accepted = o.n - 1;
        const int last = o.node[o.n - 1];
        out[len++] = o.has_y ? o.y : sample_law(T.laws[last], seed, n_out + (uint64_t)(o.n - 1), 0);
    } else {
        out[len++] = sample_law(T.laws[0], seed, n_out, 0);
    }
    uint64_t ph = t.hash[0];
    for (int i = 0; i < len; ++i) ph = child_hash(ph, out[i]);
    while (len < want) {
        const Law law = law_at(ph, noise);
        const int y = sample_law(law, seed ^ 0xabcdef1234ull, n_out + (uint64_t)len, 1);
        out[len++] = y;
        ph = child_hash(ph, y);
    }
    return len;
}

Result run_trav(Kind kind, int trials, double noise, RadMtpTravOpt opt) {
    std::vector<double> cnt3((size_t)V * V * V, 0), cnt1((size_t)V, 0);
    double acc = 0;
    for (int s = 0; s < trials; ++s) {
        const uint64_t seed = 0x1234567ull + (uint64_t)s * 7919ull;
        Tree t;
        if (kind == CHAIN)  draft_chain(&t, seed, 0, 3, 0, noise);
        else if (kind == STAGE1) draft_chain(&t, seed, 0, 3, 3, noise);
        int out[64], accepted = 0;
        walk_trav(t, seed, 0, noise, 3, out, &accepted, opt);
        ++cnt1[(size_t)out[0]];
        ++cnt3[(size_t)((out[0] * V + out[1]) * V + out[2])];
        acc += accepted;
    }
    std::vector<double> e1((size_t)V, 0), e3((size_t)V * V * V, 0);
    const uint64_t h0 = 1;
    const Law l0 = law_at(h0, noise);
    for (int a = 0; a < V; ++a) {
        e1[(size_t)a] = l0.p[a];
        const Law l1 = law_at(child_hash(h0, a), noise);
        for (int b = 0; b < V; ++b) {
            const Law l2 = law_at(child_hash(child_hash(h0, a), b), noise);
            for (int c = 0; c < V; ++c) e3[(size_t)((a * V + b) * V + c)] = l0.p[a] * l1.p[b] * l2.p[c];
        }
    }
    return { zscore(cnt3, e3, trials), acc / trials, zscore(cnt1, e1, trials) };
}

/* The first FOUR emitted tokens' joint: the fourth is the final token of a three-deep accepted path,
 * drawn from the walk's residual law, which the three-token statistic never reads. */
double run_trav4(Kind kind, int trials, double noise) {
    std::vector<double> cnt((size_t)V * V * V * V, 0), ex((size_t)V * V * V * V, 0);
    for (int s = 0; s < trials; ++s) {
        const uint64_t seed = 0x2468ace1ull + (uint64_t)s * 6151ull;
        Tree t;
        if (kind == CHAIN)  draft_chain(&t, seed, 0, 3, 0, noise);
        else if (kind == STAGE1) draft_chain(&t, seed, 0, 3, 3, noise);
        int out[64], accepted = 0;
        walk_trav(t, seed, 0, noise, 4, out, &accepted, RadMtpTravOpt{});
        ++cnt[(size_t)(((out[0] * V + out[1]) * V + out[2]) * V + out[3])];
    }
    const uint64_t h0 = 1;
    const Law l0 = law_at(h0, noise);
    for (int a = 0; a < V; ++a) {
        const uint64_t ha = child_hash(h0, a); const Law l1 = law_at(ha, noise);
        for (int b = 0; b < V; ++b) {
            const uint64_t hb = child_hash(ha, b); const Law l2 = law_at(hb, noise);
            for (int c = 0; c < V; ++c) {
                const Law l3 = law_at(child_hash(hb, c), noise);
                for (int d = 0; d < V; ++d)
                    ex[(size_t)(((a * V + b) * V + c) * V + d)] = l0.p[a] * l1.p[b] * l2.p[c] * l3.p[d];
            }
        }
    }
    return zscore(cnt, ex, trials);
}

}  /* namespace */

/* A whole generation: steps drafted at the committed prefix (base = the output length so far, the
 * engine's), each walked as the core walks it, until `want` tokens are out. The joint law of tokens
 * [1, 4) -- spanning the step boundaries, with token 0 and everything after marginalised -- and of
 * tokens [0, 3) must be the target's: this is what a draft or verify stream reused across steps
 * would break. */
Result run_steps(Kind kind, int trials, double noise, bool trav = false) {
    const int want = 6;
    std::vector<double> cnt_a((size_t)V * V * V, 0), cnt_b((size_t)V * V * V, 0);
    double steps = 0;
    for (int s = 0; s < trials; ++s) {
        const uint64_t seed = 0x7654321ull + (uint64_t)s * 104729ull;
        int out[64], len = 0;
        uint64_t h = 1;
        while (len < want) {
            Tree t;
            if (kind == CHAIN)       draft_chain(&t, seed, (uint64_t)len, 3, 0, noise, h);
            else if (kind == STAGE1) draft_chain(&t, seed, (uint64_t)len, 3, 3, noise, h);
            int step_out[64], accepted = 0;
            const int n = trav ? walk_trav(t, seed, (uint64_t)len, noise, 0, step_out, &accepted, RadMtpTravOpt{})
                               : walk(t, seed, (uint64_t)len, noise, 0, step_out, &accepted);
            for (int i = 0; i < n; ++i) { out[len++] = step_out[i]; h = child_hash(h, step_out[i]); }
            ++steps;
        }
        ++cnt_a[(size_t)((out[0] * V + out[1]) * V + out[2])];
        ++cnt_b[(size_t)((out[1] * V + out[2]) * V + out[3])];
    }
    /* expected: token 0 marginalised for (1,2,3) */
    std::vector<double> ea((size_t)V * V * V, 0), eb((size_t)V * V * V, 0);
    const uint64_t h0 = 1;
    const Law l0 = law_at(h0, noise);
    for (int a = 0; a < V; ++a) {
        const uint64_t ha = child_hash(h0, a);
        const Law l1 = law_at(ha, noise);
        for (int b = 0; b < V; ++b) {
            const uint64_t hb = child_hash(ha, b);
            const Law l2 = law_at(hb, noise);
            for (int c = 0; c < V; ++c) {
                const uint64_t hc = child_hash(hb, c);
                const Law l3 = law_at(hc, noise);
                ea[(size_t)((a * V + b) * V + c)] = l0.p[a] * l1.p[b] * l2.p[c];
                for (int d = 0; d < V; ++d)
                    eb[(size_t)((b * V + c) * V + d)] += l0.p[a] * l1.p[b] * l2.p[c] * l3.p[d];
            }
        }
    }
    return { zscore(cnt_a, ea, trials), steps / trials, zscore(cnt_b, eb, trials) };
}

TEST(a_generation_over_several_steps_keeps_the_target_law) {
    const int trials = 150000;
    const Kind kinds[] = { CHAIN, STAGE1 };
    const char* names[] = { "chain", "stage1" };
    for (int k = 0; k < 2; ++k) {
        const Result r = run_steps(kinds[k], trials, 1.0);
        std::printf("  %-8s steps=%.3f  z[0..3)=%.2f  z[1..4)=%.2f\n", names[k], r.mean_acc, r.z, r.first_z);
        CHECK(r.z < 4.5);
        CHECK(r.first_z < 4.5);
    }
}

/* The single-node recursion: m = 1..4 children drawn without replacement from Q, verified in draft
 * order, emit ~ P for every (P, Q) pair, including a P that lacks some of Q's tokens. */
TEST(one_node_with_several_children_emits_the_target_law) {
    const int trials = 400000;
    for (int m = 2; m <= 4; ++m)
        for (int pair = 0; pair < 6; ++pair) {
            const Law L = law_at(1000 + (uint64_t)pair, 1.0 + 0.4 * pair);
            std::vector<double> cnt((size_t)V, 0);
            for (int s = 0; s < trials; ++s) {
                RadMtpCoupleState st{};
                CHECK(rad_mtp_couple_build_q(L.qid, L.qval, 1000, &st));
                st.draw = make_draw(0x9e3779b97f4a7c15ull * (uint64_t)(s + 1), (uint64_t)s);
                st.status = 1; st.round = 1; st.pos = 5;
                rad_mtp_rrs_draw(&st, 0, m);
                CHECK(rad_mtp_couple_state_valid(st, 1000));
                CHECK_EQ(st.nkids, std::min(m, st.count));
                const int y = verify(st, L, st.draw.seed, 5, -1);
                CHECK(y >= 0 && y < V);
                ++cnt[(size_t)y];
            }
            std::vector<double> e(L.p, L.p + V);
            const double z = zscore(cnt, e, trials);
            if (z > 4.5) std::printf("  m=%d pair=%d z=%.2f\n", m, pair, z);
            CHECK(z < 4.5);
        }
}

/* The negative control: SpecInfer's with-replacement verify (q not removed or renormalised) on
 * drafts drawn without replacement is not exact, and the same statistic sees it. */
TEST(the_with_replacement_verify_is_not_exact_for_without_replacement_drafts) {
    const int trials = 400000, m = 3;
    const Law L = law_at(1002, 1.0);
    std::vector<double> cnt((size_t)V, 0);
    for (int s = 0; s < trials; ++s) {
        RadMtpCoupleState st{};
        rad_mtp_couple_build_q(L.qid, L.qval, 1000, &st);
        st.draw = make_draw(0x9e3779b97f4a7c15ull * (uint64_t)(s + 1), (uint64_t)s);
        rad_mtp_rrs_draw(&st, 0, m);
        double r[V], q[V];
        for (int i = 0; i < V; ++i) { r[i] = L.p[i]; q[i] = 0; }
        for (int k = 0; k < st.count; ++k) q[st.ids[k]] = st.q[k];
        int out = -1;
        for (int i = 0; i < m && out < 0; ++i) {
            const int x = st.ids[i];
            const double u = rad_mtp_rrs_uaccept(st.draw.seed, 5, i);
            if (r[x] >= q[x] || u * q[x] < r[x]) { out = x; break; }
            double z = 0;
            for (int k = 0; k < V; ++k) { r[k] = std::max(r[k] - q[k], 0.0); z += r[k]; }
            for (int k = 0; k < V; ++k) r[k] /= z;   /* q never renormalised, the child never removed */
        }
        if (out < 0) {
            double u = rad_spechub_uniform(st.draw.seed, 5, RAD_SAMPLE_KEY_RESID);
            out = 0;
            for (int k = 0; k < V; ++k) { if (r[k] <= 0) continue; out = k; u -= r[k]; if (u < 0) break; }
        }
        ++cnt[(size_t)out];
    }
    std::vector<double> e(L.p, L.p + V);
    const double z = zscore(cnt, e, trials);
    CHECK(z > 20.0);
}

TEST(a_chain_and_a_chain_with_root_siblings_emit_the_target_law) {
    const int trials = 300000;
    const Result chain = run_kind(CHAIN, trials, 1.0);
    const Result s1    = run_kind(STAGE1, trials, 1.0);
    std::printf("  chain z=%.2f acc=%.3f | root siblings z=%.2f acc=%.3f\n", chain.z, chain.mean_acc, s1.z, s1.mean_acc);
    CHECK(chain.z < 4.5);  CHECK(chain.first_z < 4.5);
    CHECK(s1.z < 4.5);     CHECK(s1.first_z < 4.5);
    /* siblings are drafted to be accepted: they accept at least as many drafts as the chain */
    CHECK(s1.mean_acc > chain.mean_acc);
    /* and a head far from the target changes the acceptance, never the law */
    const Result far = run_kind(STAGE1, 150000, 3.0);
    CHECK(far.z < 4.5);
}

/* Q == P: the first child is accepted whatever the seed; disjoint supports: nothing drafted is
 * accepted and the token is a draw from P. */
TEST(degenerate_laws) {
    const Law L = law_at(77, 1.0);
    /* P exactly the head's law: a plane equal to the shortlist */
    Law same = L;
    same.pn = 0;
    for (int i = 0; i < 8; ++i) { same.pid[same.pn] = L.qid[i]; same.pval[same.pn++] = L.qval[i]; }
    for (int s = 0; s < 2000; ++s) {
        RadMtpCoupleState st{};
        CHECK(rad_mtp_couple_build_q(L.qid, L.qval, 1000, &st));
        st.draw = make_draw(31ull * (uint64_t)(s + 1), 0);
        st.status = 1; st.round = 1; st.pos = 0;
        rad_mtp_rrs_draw(&st, 0, 3);
        CHECK_EQ(verify(st, same, st.draw.seed, 0, -7), st.ids[0]);
    }
    /* disjoint: the target lives on tokens the head never reports */
    Law dis{};
    dis.pn = 2; dis.pid[0] = 20; dis.pid[1] = 21; dis.pval[0] = 0.3f; dis.pval[1] = -0.4f;
    const double p20 = 1.0 / (1.0 + std::exp(-0.7));
    int n20 = 0;
    const int trials = 100000;
    for (int s = 0; s < trials; ++s) {
        RadMtpCoupleState st{};
        CHECK(rad_mtp_couple_build_q(L.qid, L.qval, 1000, &st));
        st.draw = make_draw(77ull * (uint64_t)(s + 1), 0);
        st.status = 1; st.round = 1; st.pos = 3;
        rad_mtp_rrs_draw(&st, 0, 4);
        const int y = verify(st, dis, st.draw.seed, 3, -7);
        CHECK(y == 20 || y == 21);
        n20 += y == 20;
    }
    CHECK(std::fabs((double)n20 / trials - p20) < 5 * std::sqrt(p20 * (1 - p20) / trials));
}

TEST(traversal_verification_emits_the_target_law_and_accepts_more) {
    const int trials = 200000;
    const RadMtpTravOpt ok{};
    const Result chain = run_trav(CHAIN, trials, 1.0, ok);
    const Result s1    = run_trav(STAGE1, trials, 1.0, ok);
    const Result rchain = run_kind(CHAIN, trials, 1.0);
    const Result rs1    = run_kind(STAGE1, trials, 1.0);
    std::printf("  traversal chain z=%.2f acc=%.3f (rowwise %.3f) | root siblings z=%.2f acc=%.3f (rowwise %.3f)\n",
                chain.z, chain.mean_acc, rchain.mean_acc, s1.z, s1.mean_acc, rs1.mean_acc);
    CHECK(chain.z < 4.5); CHECK(chain.first_z < 4.5);
    CHECK(s1.z < 4.5);    CHECK(s1.first_z < 4.5);
    CHECK(chain.mean_acc > rchain.mean_acc);
    CHECK(s1.mean_acc > rs1.mean_acc);
    const Result far = run_trav(STAGE1, 100000, 3.0, ok);
    CHECK(far.z < 4.5);
}

/* Each mistake the walk could make is caught by the same statistic: token-level ratios (1), no
 * (1 - p) term (2), the draft law never renormalised after a rejection (3). */
TEST(traversal_verification_mistakes_are_not_exact) {
    for (int ctl = 1; ctl <= 3; ++ctl) {
        RadMtpTravOpt o; o.ctl = ctl;
        const Result r = run_trav(STAGE1, 100000, 1.0, o);
        std::printf("  negative control %d: z=%.2f first_z=%.2f\n", ctl, r.z, r.first_z);
        CHECK(r.z > 20.0 || r.first_z > 20.0);
    }
}

/* The walk's final token (the fourth of a three-deep path) and a generation over several steps (no
 * uniform reused across steps). */
TEST(traversal_verification_final_token_and_steps_are_exact) {
    const Kind kinds[] = { CHAIN, STAGE1 };
    const char* names[] = { "chain", "root siblings" };
    for (int k = 0; k < 2; ++k) {
        const double z4 = run_trav4(kinds[k], 300000, 1.0);
        const Result r = run_steps(kinds[k], 100000, 1.0, true);
        std::printf("  trav %-14s z4=%.2f steps=%.3f z[0..3)=%.2f z[1..4)=%.2f\n", names[k], z4, r.mean_acc, r.z, r.first_z);
        CHECK(z4 < 4.5);
        CHECK(r.z < 4.5);
        CHECK(r.first_z < 4.5);
    }
}

/* Peaked laws: the target's inverse temperature 3 and 6 (a few tokens carry the mass, the acceptance
 * ratios sit at or above one on the trunk), where a cap or rounding at p = 1 would show. */
TEST(traversal_verification_is_exact_on_peaked_laws) {
    const double old = g_sharp;
    for (double sh : { 3.0, 6.0 }) {
        g_sharp = sh;
        for (Kind k : { CHAIN, STAGE1 }) {
            const Result r = run_trav(k, 150000, 1.0, RadMtpTravOpt{});
            const double z4 = run_trav4(k, 200000, 1.0);
            std::printf("  trav sharp=%.0f kind=%d z=%.2f first=%.2f z4=%.2f acc=%.3f\n", sh, (int)k, r.z, r.first_z, z4, r.mean_acc);
            CHECK(r.z < 4.5); CHECK(r.first_z < 4.5); CHECK(z4 < 4.5);
        }
    }
    g_sharp = old;
}

namespace {

constexpr int SV = 8;   /* toy vocab bound: tokens 0 .. V-1 with V <= 6 here */

struct SLaw {
    int     V = 0;
    int32_t pid[SV];  float pval[SV];  int pn = 0;   /* the target's candidate plane (its nucleus) */
    double  p[SV] = {};                              /* its exact law over tokens 0 .. V-1 */
    int32_t qid[8];   float qval[8];                 /* the head's shortlist, holes = -1 / -inf */
};

/* A target over `psup` of the V tokens (the others have no candidate, as after top-p) and a head over
 * `qsup` of them, independently chosen: the supports may differ, nest or be disjoint. */
SLaw small_law(uint64_t h, int V, int qsup, int psup, double qspread = 1.0) {
    SLaw L;
    L.V = V;
    int perm[SV], perm2[SV];
    for (int i = 0; i < V; ++i) { perm[i] = i; perm2[i] = i; }
    for (int i = V - 1; i > 0; --i) {
        std::swap(perm[i], perm[(int)(H(h * 31 + (uint64_t)i) % (uint64_t)(i + 1))]);
        std::swap(perm2[i], perm2[(int)(H(h * 37 + 5 + (uint64_t)i) % (uint64_t)(i + 1))]);
    }
    double z = 0;
    for (int j = 0; j < psup; ++j) {
        const float lg = (float)((H(h * 131 + (uint64_t)j * 977 + 1) >> 40) % 4000) / 1000.0f - 2.0f;
        L.pid[L.pn] = perm[j]; L.pval[L.pn++] = lg;
    }
    for (int j = 0; j < L.pn; ++j) z += std::exp((double)L.pval[j]);
    for (int j = 0; j < L.pn; ++j) L.p[L.pid[j]] = std::exp((double)L.pval[j]) / z;
    for (int i = 0; i < 8; ++i) { L.qid[i] = -1; L.qval[i] = -INFINITY; }
    for (int j = 0; j < qsup; ++j) {
        L.qid[j] = perm2[j];
        L.qval[j] = (float)(qspread * (((H(h * 257 + (uint64_t)j * 331 + 7) >> 40) % 4000) / 1000.0 - 2.0));
    }
    return L;
}

/* The step function f(u), u in [0, 1): its pieces (token, length) in order, found by bisection. */
std::vector<std::pair<int, double>> pieces(const std::function<int(double)>& f) {
    std::vector<std::pair<int, double>> out;
    double lo = 0;
    const double last = 1.0 - 0x1p-53;
    while (lo < 1) {
        const int t = f(lo);
        double b = 1.0;
        if (f(last) != t) {
            double a = lo, bb = last;
            for (int it = 0; it < 80; ++it) {
                const double mid = 0.5 * (a + bb);
                if (f(mid) == t) a = mid; else bb = mid;
            }
            b = bb;
        }
        out.emplace_back(t, b - lo);
        if (b >= 1.0) break;
        lo = b;
    }
    return out;
}

}  /* namespace */

/* ====================================================================================================
 * Traversal verification, checked EXACTLY and on its own arithmetic (abi/rad_mtp_traversal.h).
 *
 * The walk reads the target through integer masses (rad_mtp_trav_mass: an FP32 exponential in fixed
 * point), so the law it realises for a row is P~ = masses / their sum, a fixed function of the row.
 * The tests below check (1) that P~ is the softmax to FP32 accuracy; (2) by ENUMERATION that the walk
 * emits P~ exactly: every draft tree of a small model is built with its probability (Plackett-Luce
 * prefixes for Gumbel siblings, the tail draw's pieces for rheo), the walk's tests are scripted (test
 * t accepts with its own rate after t - 1 rejections, which the walk reports), and the law of the
 * first two emitted tokens comes out as P~(x1) P~(x2 | x1) to 1e-11; the with-replacement update, the
 * missing (1 - p) term and a rheo node walked as a Gumbel one miss by percents; (3) that the final
 * token's uniform maps onto the residual law (support slots, then the off-support candidates by
 * integer units) on a 700-slot plane; (4) the rheo walk by chi-square through the real drafters.
 * ================================================================================================== */
namespace {

/* The law the walk realises for a candidate plane: its integer masses over the tokens. */
void trav_int_law(const SLaw& L, double* p) {
    for (int x = 0; x < SV; ++x) p[x] = 0;
    if (L.pn <= 0) return;
    float top = -INFINITY;
    for (int j = 0; j < L.pn; ++j) top = std::max(top, L.pval[j]);
    uint64_t m[SV] = {}, sum = 0;
    for (int j = 0; j < L.pn; ++j) m[L.pid[j]] += rad_mtp_trav_mass(L.pval[j], top, RAD_MTP_TRAV_QBITS);
    for (int x = 0; x < SV; ++x) sum += m[x];
    for (int x = 0; x < SV; ++x) p[x] = (double)m[x] / (double)sum;
}

struct ETree {
    int n = 1;
    int par[40] = {};
    int32_t tok[40] = {};
    bool has[40] = {};
    RadMtpCoupleState st[40] = {};
    const SLaw* law[40] = {};
    const RadMtpCoupleState* state(int v) const { return has[v] ? &st[v] : nullptr; }
    bool cand(int v, const int32_t** ids, const float** val, int64_t* nn) const {
        *ids = law[v]->pid; *val = law[v]->pval; *nn = law[v]->pn;
        return true;
    }
    int child(int v, int32_t token) const {
        for (int j = v + 1; j < n; ++j)
            if (par[j] == v && tok[j] == token) return j;
        return -1;
    }
    int64_t vocab() const { return 1000; }
};

struct EDraft { double w = 0; RadMtpCoupleState st{}; int nk = 0; int32_t kid[4] = {}; };

/* Every way a node with the head's law of `L` drafts `want` children, with its probability: the
 * ordered prefixes of sampling without replacement (scheme 0), or the pieces of the rheo tail draw
 * (scheme m + 1). Empty when the head has no law (the node drafts nothing). */
std::vector<EDraft> edrafts(const SLaw& L, int scheme, int want, int round) {
    std::vector<EDraft> out;
    RadMtpCoupleState s0{};
    if (want <= 0 || !rad_mtp_couple_build_q(L.qid, L.qval, 1000, &s0)) return out;
    s0.draw = make_draw(77, 3); s0.status = 1; s0.round = round; s0.pos = 5;
    if (scheme > 0) {
        const int m = scheme - 1;
        auto drawn = [&](double ut) { RadMtpCoupleState s = s0; rad_mtp_rrs_draw_rheo_u(&s, ut, want, m); return s; };
        std::vector<std::pair<int, double>> pc;
        if (s0.count > m) pc = pieces([&](double ut) { return (int)drawn(ut).ids[m]; });
        else pc.emplace_back(0, 1.0);
        double lo = 0;
        for (const auto& x : pc) {
            EDraft d;
            d.w = x.second;
            d.st = drawn(lo + 0.5 * x.second);
            lo += x.second;
            d.nk = d.st.nkids;
            for (int k = 0; k < d.nk; ++k) d.kid[k] = d.st.ids[k];
            out.push_back(d);
        }
        return out;
    }
    const int a = std::min(want, s0.count);
    int idx[4] = {};
    std::function<void(int, double, double)> rec = [&](int depth, double w, double used) {
        if (depth == a) {
            EDraft d;
            d.w = w;
            d.st = s0;
            if (a >= 2) {
                bool taken[8] = {};
                int n = 0;
                for (int i = 0; i < a; ++i) { d.st.ids[n] = s0.ids[idx[i]]; d.st.q[n++] = s0.q[idx[i]]; taken[idx[i]] = true; }
                for (int i = 0; i < s0.count; ++i)
                    if (!taken[i]) { d.st.ids[n] = s0.ids[i]; d.st.q[n++] = s0.q[i]; }
                d.st.x = d.st.ids[0];
                d.st.nkids = a;
            } else {
                d.st.x = s0.ids[idx[0]];   /* the chain's wire: x anywhere in the shortlist's order */
                d.st.nkids = 0;
            }
            d.nk = a;
            for (int i = 0; i < a; ++i) d.kid[i] = s0.ids[idx[i]];
            out.push_back(d);
            return;
        }
        for (int i = 0; i < s0.count; ++i) {
            bool dup = false;
            for (int j = 0; j < depth; ++j) dup = dup || idx[j] == i;
            if (dup) continue;
            idx[depth] = i;
            rec(depth + 1, w * s0.q[i] / (1.0 - used), used + s0.q[i]);
        }
    };
    rec(0, 1.0, 0.0);
    return out;
}

/* The law of the token a walk emits at a path's last node, from the walked tree: the node's
 * residual (support slots, then its off-support candidates by their integer masses), or the plain
 * target when the node carries no law. */
void etree_final_law(const RadTravTree& kt, int c, const SLaw& L, double* law) {
    for (int x = 0; x < SV; ++x) law[x] = 0;
    double tot = 0;
    if (kt.ok[c]) {
        const double woff = kt.moff[c];
        tot = rad_mtp_trav_sum8(kt.m[c]) + woff;
        if (tot > 0) {
            for (int k = 0; k < 8; ++k)
                if (kt.ids[c][k] >= 0) law[kt.ids[c][k]] += kt.m[c][k] / tot;
            if (woff > 0 && kt.off[c] > 0)
                for (int j = 0; j < L.pn; ++j) {
                    bool on = false;
                    for (int k = 0; k < 8; ++k) on = on || kt.ids[c][k] == L.pid[j];
                    if (!on)
                        law[L.pid[j]] += woff / tot *
                            (double)rad_mtp_trav_mass(L.pval[j], kt.top[c], RAD_MTP_TRAV_QBITS) / (double)kt.off[c];
                }
        }
    }
    if (!(tot > 0)) trav_int_law(L, law);
}

/* Adds w * P(the first two emitted tokens = (x1, x2) | this draft tree) to `joint`. */
void etree_joint(const ETree& T, const SLaw* kid, int V, double w, double joint[SV][SV], int ctl,
                 double* leak) {
    double pt[64];
    int nt = 0;
    RadMtpTravOpt o;
    o.ctl = ctl; o.accept_at = 0; o.p_test = pt; o.n_tests = &nt; o.max_tests = 64;
    RadMtpTravOut out;
    rad_mtp_trav_walk(T, 0, 1, 1, &out, o);   /* every test rejects: the rates of tests 1 .. nt */
    double kidlaw[SV][SV];
    for (int x = 0; x < V; ++x) trav_int_law(kid[x], kidlaw[x]);
    double surv = 1;
    for (int t = 1; t <= nt; ++t) {
        const double pa = surv * pt[t - 1];
        surv *= 1.0 - pt[t - 1];
        if (!(pa > 0)) continue;
        RadTravTree kt;
        int ids[RAD_MTP_TRAV_MAX_ROWS];
        RadMtpTravOpt o2;
        o2.ctl = ctl; o2.accept_at = t; o2.u_final = 0.5;
        RadMtpTravOut po;
        CHECK(rad_mtp_trav_walk(T, 0, 1, 1, &po, o2, &kt, ids));
        const int f = po.node[po.n - 1];
        int c = 0;
        while (ids[c] != f) ++c;
        double fl[SV];
        etree_final_law(kt, c, *T.law[f], fl);
        if (po.n == 1) {
            for (int y = 0; y < V; ++y)
                for (int b = 0; b < V; ++b) joint[y][b] += w * pa * fl[y] * kidlaw[y][b];
        } else if (po.n == 2) {
            const int x1 = T.tok[po.node[1]];
            for (int b = 0; b < V; ++b) joint[x1][b] += w * pa * fl[b];
        } else {
            joint[T.tok[po.node[1]]][T.tok[po.node[2]]] += w * pa;
        }
    }
    *leak += w * surv;   /* the root accepts with rate 1: nothing may be left */
}

/* The exact joint law of the first two emitted tokens for a root that drafts `a` children, each
 * drafting `b`; returns the largest deviation from P~(x1) P~(x2 | x1). */
double trav_enum_case(const SLaw& root, const SLaw* kid, int V, int scheme, int a, int b, int ctl, long* trees) {
    double joint[SV][SV] = {}, leak = 0, wsum = 0;
    const std::vector<EDraft> roots = edrafts(root, scheme, a, 1);
    CHECK(!roots.empty());
    for (const EDraft& r0 : roots) {
        std::vector<EDraft> cd[4];
        for (int i = 0; i < r0.nk; ++i) cd[i] = edrafts(kid[r0.kid[i]], scheme, b, 2);
        int sel[4] = {};
        for (;;) {
            ETree T;
            T.has[0] = true; T.st[0] = r0.st; T.law[0] = &root; T.par[0] = -1;
            CHECK(rad_mtp_couple_state_valid(r0.st, 1000));
            double w = r0.w;
            for (int i = 0; i < r0.nk; ++i) {
                const int v = T.n++;
                T.par[v] = 0; T.tok[v] = r0.kid[i]; T.law[v] = &kid[r0.kid[i]];
                T.has[v] = !cd[i].empty();
                if (T.has[v]) { T.st[v] = cd[i][(size_t)sel[i]].st; w *= cd[i][(size_t)sel[i]].w; }
            }
            for (int i = 0; i < r0.nk; ++i) {
                if (cd[i].empty()) continue;
                const EDraft& d = cd[i][(size_t)sel[i]];
                for (int g = 0; g < d.nk; ++g) {
                    const int v = T.n++;
                    T.par[v] = 1 + i; T.tok[v] = d.kid[g]; T.law[v] = &kid[d.kid[g]];
                }
            }
            etree_joint(T, kid, V, w, joint, ctl, &leak);
            wsum += w;
            ++*trees;
            int i = 0;
            for (; i < r0.nk; ++i) {
                if (cd[i].empty()) continue;
                if (++sel[i] < (int)cd[i].size()) break;
                sel[i] = 0;
            }
            if (i >= r0.nk) break;
        }
    }
    CHECK(std::fabs(wsum - 1.0) < 1e-11);
    double p0[SV], worst = ctl ? 0.0 : std::fabs(leak);
    trav_int_law(root, p0);
    for (int x1 = 0; x1 < V; ++x1) {
        double p1[SV];
        trav_int_law(kid[x1], p1);
        for (int x2 = 0; x2 < V; ++x2) worst = std::max(worst, std::fabs(joint[x1][x2] - p0[x1] * p1[x2]));
    }
    return worst;
}

}  /* namespace */

/* (1) The fixed-point exponential against libm's, the integer conversions, and the law they give. */
TEST(the_walks_fixed_point_exponential_tracks_exp) {
    const double unit = (double)(1ull << RAD_MTP_TRAV_QBITS);
    double worst = 0;
    CHECK_EQ(rad_mtp_trav_mass(3.25f, 3.25f, RAD_MTP_TRAV_QBITS), 1ull << RAD_MTP_TRAV_QBITS);
    CHECK_EQ(rad_mtp_trav_mass(-INFINITY, 0.0f, RAD_MTP_TRAV_QBITS), 0ull);
    CHECK_EQ(rad_mtp_trav_mass(-30.0f, 0.0f, RAD_MTP_TRAV_QBITS), 0ull);
    const int N = 3000000;
    for (int i = 0; i <= N; ++i) {
        /* a dense sweep of the whole range, and the same arguments reached from other tops */
        const float x = -28.6f * (float)i / (float)N;
        const float top = i % 3 == 0 ? 0.0f : (float)((int)(H((uint64_t)i) % 6000) - 3000) / 100.0f;
        const float v = top + x;
        const float xr = v - top;   /* the argument the mass sees */
        const uint64_t m = rad_mtp_trav_mass(v, top, RAD_MTP_TRAV_QBITS);
        const double want = xr > -28.5f ? std::exp((double)xr) * unit : 0.0;
        const double err = std::fabs((double)m - want);
        CHECK(err <= want * 0x1p-22 + 1.0);
        if (want > 0x1p32) worst = std::max(worst, err / want);   /* where the floor's one unit is below 2^-32 */
        /* a coarser unit is the floor of the finer one */
        if (i % 16 == 0) CHECK_EQ(rad_mtp_trav_mass(v, top, 31), m >> 9);
    }
    std::printf("  fixed-point exp: worst relative error %.3g (2^-22 = %.3g) over %d arguments\n", worst, 0x1p-22, N);
    CHECK(worst < 0x1p-22);
    CHECK_EQ(rad_mtp_trav_qbits(20), 40);     CHECK_EQ(rad_mtp_trav_qbits(1024), 40);
    CHECK_EQ(rad_mtp_trav_qbits(4096), 40);   CHECK_EQ(rad_mtp_trav_qbits(4097), 39);
    CHECK_EQ(rad_mtp_trav_qbits(8192), 39);   CHECK_EQ(rad_mtp_trav_qbits(1ll << 31), 21);
    for (int i = 0; i < 200000; ++i) {
        const uint64_t x = H((uint64_t)i * 77 + 5) >> (11 + (i % 40));
        CHECK(rad_mtp_trav_u2d(x) == (double)x);
        CHECK_EQ(rad_mtp_trav_d2u(rad_mtp_trav_u2d(x)), x);
        if (x < (1ull << 50)) CHECK_EQ(rad_mtp_trav_d2u(rad_mtp_trav_u2d(x) + 0.75), x);
    }
    /* the law of a plane: the softmax, to FP32 accuracy */
    double wl = 0;
    for (int t = 0; t < 2000; ++t) {
        const Law L = law_at(31000 + (uint64_t)t, 1.0);
        float top = -INFINITY;
        CHECK(rad_mtp_trav_row_top(L.pid, L.pval, L.pn, 1000, &top));
        int32_t sid[8];
        for (int k = 0; k < 8; ++k) sid[k] = L.qid[k];
        RadTravLaw w;
        rad_mtp_trav_row_law(L.pid, L.pval, L.pn, top, sid, 8, &w);
        uint64_t on = 0;
        for (int k = 0; k < 8; ++k) {
            on += w.pq[k];
            wl = std::max(wl, std::fabs((double)w.pq[k] / (double)w.sum - L.p[sid[k]]));
        }
        CHECK_EQ(on + w.off, w.sum);
    }
    std::printf("  integer law against the FP64 softmax: worst |dp| = %.3g\n", wl);
    CHECK(wl < 5e-7);
}

/* (2) The walk emits the law it reads, exactly: Gumbel siblings and rheo (m = 1, 2), roots with one
 * to three children, children with none to two, supports that nest, differ or are disjoint. */
TEST(traversal_two_deep_joint_law_is_exact_by_enumeration) {
    double worst = 0;
    long trees = 0;
    int cases = 0;
    const struct { int scheme, a, b; } shapes[] = {
        { 0, 1, 0 }, { 0, 1, 1 }, { 0, 2, 0 }, { 0, 2, 1 }, { 0, 2, 2 }, { 0, 3, 1 }, { 0, 1, 2 },
        { 0, 3, 2 }, { 0, 2, 3 },
        { 2, 1, 1 }, { 2, 2, 1 }, { 2, 2, 2 }, { 3, 2, 1 }, { 3, 3, 1 }, { 3, 3, 3 }, { 1, 1, 1 }, { 4, 4, 2 },
    };
    for (int V = 4; V <= 6; ++V)
        for (int cfg = 0; cfg < 24; ++cfg) {
            const int qs0 = 1 + (cfg * 2 + 1) % V, ps0 = 1 + (cfg * 3 + 2) % V;
            const SLaw root = small_law((uint64_t)(61000 + V * 50 + cfg), V, qs0, ps0, cfg % 3 == 2 ? 2.5 : 1.0);
            SLaw kid[SV];
            for (int x = 0; x < V; ++x)
                kid[x] = small_law((uint64_t)(67000 + V * 70 + cfg * 13 + x), V, 1 + (cfg + x) % V,
                                   1 + (cfg + 2 * x + 1) % V);
            for (const auto& sh : shapes) {
                const double w = trav_enum_case(root, kid, V, sh.scheme, sh.a, sh.b, 0, &trees);
                worst = std::max(worst, w);
                if (!(w < 1e-11))
                    std::printf("  ENUM FAIL V=%d cfg=%d qs0=%d ps0=%d scheme=%d a=%d b=%d: %g\n", V, cfg, qs0, ps0,
                                sh.scheme, sh.a, sh.b, w);
                CHECK(w < 1e-11);
                ++cases;
            }
        }
    std::printf("  traversal enumeration: %d cases, %ld draft trees, worst |joint - P P| = %.2e\n", cases, trees, worst);
    /* the same statistic sees the mistakes */
    const SLaw root = small_law(61333, 5, 5, 5, 1.0);
    SLaw kid[SV];
    for (int x = 0; x < 5; ++x) kid[x] = small_law((uint64_t)(67333 + x), 5, 4, 5);
    long t2 = 0;
    const double good = trav_enum_case(root, kid, 5, 0, 2, 2, 0, &t2);
    const double c1 = trav_enum_case(root, kid, 5, 0, 2, 2, 1, &t2);
    const double c2 = trav_enum_case(root, kid, 5, 0, 2, 2, 2, &t2);
    const double c3 = trav_enum_case(root, kid, 5, 0, 2, 2, 3, &t2);
    const double rgood = trav_enum_case(root, kid, 5, 2, 2, 2, 0, &t2);
    const double c4 = trav_enum_case(root, kid, 5, 2, 2, 2, 4, &t2);
    std::printf("  controls: exact %.1e | token-level ratios %.3f | no (1 - p) %.3f | no renormalisation %.3f | "
                "rheo exact %.1e, rheo walked as gumbel %.3f\n", good, c1, c2, c3, rgood, c4);
    CHECK(good < 1e-11); CHECK(rgood < 1e-11);
    CHECK(c1 > 1e-3); CHECK(c2 > 1e-3); CHECK(c3 > 1e-3); CHECK(c4 > 1e-3);
}

/* (3) The final token's uniform against the residual law, on a 700-slot plane (holes, a state whose
 * support is partly off the plane): after both children are rejected the root accepts, and the pieces
 * of u -> token have the lengths the walked tree's residual gives each token. */
TEST(the_traversals_final_draw_is_the_residual_law_on_a_wide_plane) {
    constexpr int NC = 700, NV = 5000;
    int mixed = 0;   /* cases where the support slots and the off-support candidates both carry mass */
    for (int rep = 0; rep < 3; ++rep) {
        static int32_t ids[NC];
        static float val[NC];
        int live = 0;
        for (int c = 0; c < NC; ++c) {
            const uint64_t h = H(900000 + (uint64_t)rep * 7919 + (uint64_t)c);
            if (h % 11 == 0) { ids[c] = -1; val[c] = -INFINITY; continue; }
            ids[c] = 1 + c * 7;   /* distinct */
            /* a peaked head and a long tail that still carries a few percent */
            val[c] = (h % 13 == 1) ? -INFINITY
                   : c < 16 ? 6.0f - 0.9f * (float)c - (float)((h >> 20) % 1000) * 0.0005f
                            : 1.5f - 0.012f * (float)c - (float)((h >> 20) % 1000) * 0.002f;
            ++live;
        }
        int32_t qid[8];
        float qv[8];
        for (int k = 0; k < 8; ++k) {
            qid[k] = k < 5 ? 1 + k * 7 : NV - 10 - k;   /* five on the plane's head, three off it */
            qv[k] = 2.0f - (rep == 0 ? 0.35f : rep == 1 ? 0.6f : 0.15f) * (float)k;
        }
        struct WTree {
            RadMtpCoupleState st; int32_t c1, c2;
            const RadMtpCoupleState* state(int v) const { return v == 0 ? &st : nullptr; }
            bool cand(int, const int32_t** i, const float** v, int64_t* n) const { *i = ids; *v = val; *n = NC; return true; }
            int child(int v, int32_t t) const { return v == 0 ? (t == c1 ? 1 : t == c2 ? 2 : -1) : -1; }
            int64_t vocab() const { return NV; }
        } T;
        CHECK(rad_mtp_couple_build_q(qid, qv, NV, &T.st));
        T.st.draw = make_draw(4242 + (uint64_t)rep, 3); T.st.status = 1; T.st.round = 1; T.st.pos = 5;
        rad_mtp_rrs_draw(&T.st, 0, 2);
        T.c1 = T.st.ids[0]; T.c2 = T.st.ids[1];
        RadTravTree kt;
        int map[RAD_MTP_TRAV_MAX_ROWS];
        RadMtpTravOpt o;
        o.accept_at = 3; o.u_final = 0.25;   /* the two leaves reject, the root accepts */
        RadMtpTravOut out;
        CHECK(rad_mtp_trav_walk(T, 0, 1, 1, &out, o, &kt, map));
        CHECK_EQ(out.n, 1);
        CHECK(out.has_y);
        const double woff = kt.moff[0], tot = rad_mtp_trav_sum8(kt.m[0]) + woff;
        std::vector<double> want((size_t)NV, 0.0), got((size_t)NV, 0.0);
        for (int k = 0; k < 8; ++k) want[(size_t)kt.ids[0][k]] += kt.m[0][k] / tot;
        for (int c = 0; c < NC; ++c) {
            if (ids[c] < 0 || val[c] == -INFINITY) continue;
            bool on = false;
            for (int k = 0; k < 8; ++k) on = on || kt.ids[0][k] == ids[c];
            if (!on) want[(size_t)ids[c]] += woff / tot * (double)rad_mtp_trav_mass(val[c], kt.top[0], RAD_MTP_TRAV_QBITS) / (double)kt.off[0];
        }
        const auto pc = pieces([&](double u) {
            RadMtpTravOpt q;
            q.accept_at = 3; q.u_final = u;
            RadMtpTravOut po;
            rad_mtp_trav_walk(T, 0, 1, 1, &po, q);
            return (int)po.y;
        });
        for (const auto& x : pc) { CHECK(x.first >= 0 && x.first < NV); got[(size_t)x.first] += x.second; }
        double worst = 0, off_mass = 0;
        int support = 0;
        for (int y = 0; y < NV; ++y) {
            worst = std::max(worst, std::fabs(got[(size_t)y] - want[(size_t)y]));
            support += want[(size_t)y] > 0;
        }
        off_mass = woff / tot;
        std::printf("  wide final draw %d: %d live slots, %d tokens with mass, off-support share %.3f, %zu pieces, worst |dp| = %.2e\n",
                    rep, live, support, off_mass, pc.size(), worst);
        CHECK(worst < 1e-10);
        CHECK(pc.size() > 300);
        mixed += off_mass > 0.01 && off_mass < 0.99;
    }
    CHECK(mixed >= 2);
}

/* (4) RheoSampling siblings through the walk: the real propose row drafts the root's siblings, the
 * walk tries the tail child first and the deterministic ones as point masses. Exact; the same trees
 * walked as Gumbel ones (what a walk that ignores the scheme does) are not. */
TEST(traversal_verification_of_rheo_siblings_is_exact) {
    const int trials = 200000;
    double worst_bad = 0;
    for (int m = 1; m <= 2; ++m) {
        g_scheme = m + 1;
        const Result r = run_trav(STAGE1, trials, 1.0, RadMtpTravOpt{});
        const double z4 = run_trav4(STAGE1, trials, 1.0);
        const Result row = run_kind(STAGE1, trials, 1.0);
        RadMtpTravOpt bad;
        bad.ctl = 4;
        const Result b = run_trav(STAGE1, trials, 1.0, bad);
        g_scheme = 0;
        std::printf("  trav rheo m=%d z=%.2f first=%.2f z4=%.2f acc=%.3f (rowwise %.3f) | walked as gumbel: z=%.1f first=%.1f\n",
                    m, r.z, r.first_z, z4, r.mean_acc, row.mean_acc, b.z, b.first_z);
        CHECK(r.z < 4.5); CHECK(r.first_z < 4.5); CHECK(z4 < 4.5);
        worst_bad = std::max(worst_bad, std::max(b.z, b.first_z));
    }
    CHECK(worst_bad > 20.0);
}

/* ==================================================================== --draft-depth adaptive
 *
 * The chain of a step stops by the head's own laws (core/sched/draft_depth.h): round j is drafted
 * only while the product of the earlier rounds' statistic holds. The rule under test is the
 * engine's own function over the states the real propose row wrote; the verify is the real
 * row-wise coupling and the real traversal walk. A generation runs over several steps, so the
 * depth of every step depends on the tokens the steps before it committed and on its own drafts.
 * Exact for every rule the engine offers (top / path statistic, decided 0 or 1 round behind);
 * the two mistakes a rule can make are caught by the same statistic:
 *   control 1  admits a draft by ITS OWN mass under the head's law (a draft-side quantity, but of
 *              the token the decision admits: the chain keeps the drafts the head liked);
 *   control 2  peeks at the target: admits a draft only where the target's mass on it is at least
 *              the head's (the chain keeps the drafts the verify is going to like). */
using namespace rad;
namespace {

struct DepthRun { double z3, z1, mean_acc, mean_depth; double share[8]; };

/* A drafted chain of at most cfg.max rounds, cut by the rule (ctl 0) or by a control. */
int draft_depth_chain(Tree* t, uint64_t seed, uint64_t base, const DraftDepthCfg& cfg, double noise,
                      uint64_t h0, int ctl, double ctl_thr) {
    draft_chain(t, seed, base, cfg.max, 0, noise, h0);
    int D = cfg.max;
    if (ctl == 0) {
        double st[RAD_MTP_COUPLE_MAX_ROUNDS];
        for (int r = 1; r <= cfg.max; ++r) st[r - 1] = draft_depth_stat(t->st[r - 1], r, cfg.stat);
        D = draft_depth_of(cfg, st, cfg.max);
    } else {
        for (int r = 1; r <= cfg.max; ++r) {
            const RadMtpCoupleState& c = t->st[r - 1];
            double own = 0;
            for (int i = 0; i < c.count; ++i) if (c.ids[i] == c.x) own = c.q[i];
            bool keep;
            if (ctl == 1) keep = own >= ctl_thr;
            else {
                const Law law = law_at(t->hash[r - 1], noise);
                keep = law.p[c.x] >= own;
            }
            if (!keep) { D = r - 1; break; }
        }
    }
    /* the rounds past the stop never ran: no node, no state */
    for (int j = D; j < 64; ++j) t->st[j] = RadMtpCoupleState{};
    t->n = D;
    return D;
}

/* `trials` generations of three tokens, each step an adaptive chain verified by the traversal walk
 * (trav) or row by row; the joint law of the three against the target's. */
DepthRun run_depth(const DraftDepthCfg& cfg, bool trav, int trials, double noise, int ctl = 0,
                   double ctl_thr = 0) {
    std::vector<double> cnt3((size_t)V * V * V, 0), cnt1((size_t)V, 0);
    DepthRun R{};
    double steps = 0;
    for (int s = 0; s < trials; ++s) {
        const uint64_t seed = 0x51ed27ull + (uint64_t)s * 104729ull;
        int out[16], len = 0;
        uint64_t h = 1;
        while (len < 3) {
            Tree t;
            const int D = draft_depth_chain(&t, seed, (uint64_t)len, cfg, noise, h, ctl, ctl_thr);
            R.mean_depth += D; ++R.share[D < 8 ? D : 7]; ++steps;
            int step_out[64], accepted = 0, n;
            if (D == 0) {
                /* no draft at all (a control cut the first one): the ordinary pick */
                step_out[0] = sample_law(law_at(h, noise), seed, (uint64_t)len, 0);
                n = 1;
            } else if (trav) {
                n = walk_trav(t, seed, (uint64_t)len, noise, 0, step_out, &accepted, RadMtpTravOpt{});
            } else {
                n = walk(t, seed, (uint64_t)len, noise, 0, step_out, &accepted);
            }
            R.mean_acc += accepted;
            for (int i = 0; i < n && len < 16; ++i) { out[len++] = step_out[i]; h = child_hash(h, step_out[i]); }
        }
        ++cnt1[(size_t)out[0]];
        ++cnt3[(size_t)((out[0] * V + out[1]) * V + out[2])];
    }
    std::vector<double> e1((size_t)V, 0), e3((size_t)V * V * V, 0);
    const Law l0 = law_at(1, noise);
    for (int a = 0; a < V; ++a) {
        e1[(size_t)a] = l0.p[a];
        const uint64_t ha = child_hash(1, a);
        const Law l1 = law_at(ha, noise);
        for (int b = 0; b < V; ++b) {
            const Law l2 = law_at(child_hash(ha, b), noise);
            for (int c = 0; c < V; ++c) e3[(size_t)((a * V + b) * V + c)] = l0.p[a] * l1.p[b] * l2.p[c];
        }
    }
    R.z3 = zscore(cnt3, e3, trials);
    R.z1 = zscore(cnt1, e1, trials);
    R.mean_acc /= steps; R.mean_depth /= steps;
    for (double& x : R.share) x /= steps;
    return R;
}

DraftDepthCfg depth_cfg(int max, int lag, int stat, double stop, int min = 1) {
    DraftDepthCfg c;
    c.on = true; c.max = max; c.min = min; c.lag = lag; c.stat = stat; c.stop = stop;
    return c;
}

}  /* namespace */

/* The rule itself: rounds are decided in order, `lag` rounds behind, and the first refusal ends the
 * chain; min rounds always run, max bounds it, a row with no law takes the fixed fallback. */
TEST(draft_depth_rule_decides_each_round_from_the_rounds_it_may_read) {
    const double s[] = { 0.9, 0.8, 0.5, 0.9, 0.9 };   /* conf: .9 .72 .36 .324 .2916 */
    CHECK_EQ(draft_depth_of(depth_cfg(5, 0, 0, 0.7), s, 5), 3);    /* round 4 needs conf(3) = .36 */
    CHECK_EQ(draft_depth_of(depth_cfg(5, 1, 0, 0.7), s, 5), 4);    /* a round later: conf(2) admits round 4 */
    CHECK_EQ(draft_depth_of(depth_cfg(5, 0, 0, 0.95), s, 5), 1);
    CHECK_EQ(draft_depth_of(depth_cfg(5, 1, 0, 0.95), s, 5), 2);   /* round 2 reads nothing: it always runs */
    CHECK_EQ(draft_depth_of(depth_cfg(5, 0, 0, 0.0), s, 5), 5);
    CHECK_EQ(draft_depth_of(depth_cfg(3, 0, 0, 0.0), s, 5), 3);
    CHECK_EQ(draft_depth_of(depth_cfg(5, 0, 0, 0.95, 3), s, 5), 3);  /* min */
    CHECK_EQ(draft_depth_of(depth_cfg(1, 0, 0, 0.0), s, 5), 1);
    /* the engine's incremental form: a decision never reads past draft_depth_need */
    const DraftDepthCfg c = depth_cfg(5, 1, 0, 0.7);
    CHECK_EQ(draft_depth_need(c, 2), 0);
    CHECK_EQ(draft_depth_need(c, 5), 3);
    DraftDepthRow r;
    draft_depth_begin(c, &r, false);
    CHECK(draft_depth_go(c, &r, 2));
    draft_depth_observe(&r, 1, 0.9);
    CHECK(draft_depth_go(c, &r, 3));
    draft_depth_observe(&r, 2, 0.8);
    CHECK(draft_depth_go(c, &r, 4));
    draft_depth_observe(&r, 3, 0.5);
    CHECK(!draft_depth_go(c, &r, 5));
    CHECK(r.done); CHECK_EQ(r.depth, 4);
    CHECK(!draft_depth_go(c, &r, 5));           /* the chain has ended */
    /* no law: a greedy row, or a state the proposer could not make a law of */
    DraftDepthRow g;
    draft_depth_begin(c, &g, true);
    CHECK(draft_depth_go(c, &g, 2)); CHECK(draft_depth_go(c, &g, 3)); CHECK(!draft_depth_go(c, &g, 4));
    DraftDepthRow m;
    draft_depth_begin(c, &m, false);
    CHECK(draft_depth_go(c, &m, 2));
    draft_depth_observe(&m, 1, -1.0);
    CHECK(m.lawless);
    CHECK(draft_depth_go(c, &m, 3)); CHECK(!draft_depth_go(c, &m, 4));
    /* the statistic of a state: the largest mass, or the drawn token's; none without a law */
    RadMtpCoupleState st{};
    st.status = 1; st.round = 2; st.count = 3; st.x = 7;
    st.ids[0] = 5; st.ids[1] = 7; st.ids[2] = 9;
    st.q[0] = 0.6; st.q[1] = 0.3; st.q[2] = 0.1;
    CHECK(draft_depth_stat(st, 2, DRAFT_DEPTH_TOP) == 0.6);
    CHECK(draft_depth_stat(st, 2, DRAFT_DEPTH_PATH) == 0.3);
    CHECK(draft_depth_stat(st, 1, DRAFT_DEPTH_TOP) < 0);   /* another round's state */
    st.status = 0;
    CHECK(draft_depth_stat(st, 2, DRAFT_DEPTH_TOP) < 0);
}

TEST(adaptive_chain_depth_keeps_the_target_law) {
    const int trials = 300000;
    g_sharp = 2.0;
    struct C { int lag, stat; double stop; const char* name; };
    const C cs[] = { { 0, DRAFT_DEPTH_TOP, 0.36, "top lag0" },  { 1, DRAFT_DEPTH_TOP, 0.40, "top lag1" },
                     { 0, DRAFT_DEPTH_PATH, 0.12, "path lag0" }, { 1, DRAFT_DEPTH_PATH, 0.22, "path lag1" } };
    for (const C& c : cs) {
        const DraftDepthCfg cfg = depth_cfg(4, c.lag, c.stat, c.stop);
        const DepthRun tv = run_depth(cfg, true, trials, 0.6);
        const DepthRun rw = run_depth(cfg, false, trials, 0.6);
        std::printf("  depth %-9s stop %.2f: trav z3=%.2f z1=%.2f acc=%.3f | rowwise z3=%.2f z1=%.2f acc=%.3f | "
                    "depth %.2f shares %.2f %.2f %.2f %.2f\n", c.name, c.stop, tv.z3, tv.z1, tv.mean_acc,
                    rw.z3, rw.z1, rw.mean_acc, tv.mean_depth, tv.share[1], tv.share[2], tv.share[3], tv.share[4]);
        CHECK(tv.z3 < 4.5); CHECK(tv.z1 < 4.5);
        CHECK(rw.z3 < 4.5); CHECK(rw.z1 < 4.5);
        /* the rule really stops chains at different depths, and the walk gains from the longer ones */
        int used = 0;
        for (int d = 1; d <= 4; ++d) used += tv.share[d] > 0.10;
        CHECK(used >= 2);
        CHECK(tv.mean_acc >= rw.mean_acc);
    }
    /* a far head and a min of two rounds */
    const DepthRun far = run_depth(depth_cfg(4, 1, DRAFT_DEPTH_TOP, 0.5, 2), true, 200000, 2.0);
    CHECK(far.z3 < 4.5); CHECK(far.z1 < 4.5);
    g_sharp = 1.0;
}

/* ---- the load-aware rule (DraftDepthCfg::load): several sequences drafted in one step, each chain's
 * depth decided from its own and the OTHER rows' draft laws (draft_depth_go_load). Every sequence
 * must keep the target's law: its depth now depends on its neighbours' drafts, which are
 * independent of its own draws. `n_seq` generations of three tokens run side by side, a step for
 * all that are still short; with one left the stop rule decides, as in the engine. */
namespace {

struct LoadRun { double z3[4], z1[4], mean_acc, mean_depth, rounds, share[8]; long mismatch; };

/* The late choice with a deliberate mistake, for the control: draft j chosen by conf(min(j + peek,
 * seen)); peek 0 is draft_depth_select's own arithmetic (checked against it), peek 1 reads the law
 * that FOLLOWS x_j, which is conditioned on x_j. Rows and rounds priced as the rule prices them. */
void select_peek(const DraftDepthCfg& c, DraftDepthRow* r, int n, int rounds, int other_rows, int peek) {
    const double lam = c.load_lam[n < 1 ? 1 : (n > 8 ? 8 : n)];
    int R = other_rows;
    for (int i = 0; i < n; ++i) { r[i].depth = 1; r[i].done = false; R += 2; }
    for (int j = 2; j <= rounds && j <= c.max; ++j) {
        int cand[8], nc = 0;
        double p[8];
        for (int i = 0; i < n; ++i) {
            if (r[i].done || r[i].depth != j - 1) continue;
            const int m = std::min(j + peek, r[i].seen);
            p[i] = draft_depth_cal(c, j, draft_depth_conf(r[i], m), std::min(2, std::max(0, m - (j - 2))));
            cand[nc++] = i;
        }
        std::stable_sort(cand, cand + nc, [&](int a, int b) { return p[a] > p[b]; });
        const double base = draft_depth_load_cost(c, n, R);
        double sum = 0, best = 0;
        int take = 0;
        for (int m = 1; m <= nc; ++m) {
            sum += p[cand[m - 1]];
            const double v = sum - lam * (draft_depth_load_cost(c, n, R + m) - base);
            if (v > best) { best = v; take = m; }
        }
        for (int m = 0; m < nc; ++m) { if (m < take) r[cand[m]].depth = j; else r[cand[m]].done = true; }
        R += take;
        if (!take) break;
    }
    for (int i = 0; i < n; ++i) r[i].done = true;
}

/* `late`: 0 the rule alone; 1 the late choice behind it (cfg.late_lag rounds not waited for);
 * 2 the control that reads one round too far. */
LoadRun run_depth_load(const DraftDepthCfg& cfg0, int n_seq, bool trav, int trials, double noise,
                       int other_rows = 0, int late = 0, int online = 0) {
    /* `online`: 1 the tables follow the run (DraftDepthOnline, fed by every verified chain,
     * refreshed every 16 steps); 2 the control whose tables are made from the verify of the very
     * step they then decide. */
    DraftDepthCfg cfg = cfg0;
    DraftDepthOnline on;
    if (online) draft_depth_online_begin(cfg, &on);
    long fed = 0;
    std::vector<std::vector<double>> cnt3((size_t)n_seq, std::vector<double>((size_t)V * V * V, 0)),
                                     cnt1((size_t)n_seq, std::vector<double>((size_t)V, 0));
    LoadRun R{};
    double rows = 0, steps = 0;
    for (int s = 0; s < trials; ++s) {
        uint64_t seed[4], h[4];
        int out[4][16], len[4];
        for (int q = 0; q < n_seq; ++q) {
            seed[q] = 0x2f6b1ull + (uint64_t)s * 104729ull + (uint64_t)q * 0x9e3779b97f4a7c15ull;
            h[q] = 1; len[q] = 0;
        }
        for (;;) {
            int act[4], n = 0;
            for (int q = 0; q < n_seq; ++q) if (len[q] < 3) act[n++] = q;
            if (n == 0) break;
            std::vector<Tree> t((size_t)n);
            DraftDepthRow row[4];
            for (int i = 0; i < n; ++i) {
                draft_chain(&t[(size_t)i], seed[act[i]], (uint64_t)len[act[i]], cfg.max, 0, noise, h[act[i]]);
                draft_depth_begin(cfg, &row[i], false);
            }
            if (online == 2) {
                /* the control: what the target accepts of each whole chain, into the tables first */
                draft_depth_online_begin(cfg0, &on);
                on.prior_w = 1e-3;
                for (int i = 0; i < n; ++i) {
                    Tree tc = t[(size_t)i];
                    tc.n = cfg.max;
                    float sv[8];
                    int ns = 0, so_[64], acc = 0;
                    for (; ns < cfg.max && ns < 8; ++ns) sv[ns] = (float)draft_depth_stat(tc.st[ns], ns + 1, DRAFT_DEPTH_TOP);
                    if (trav) walk_trav(tc, seed[act[i]], (uint64_t)len[act[i]], noise, 0, so_, &acc, RadMtpTravOpt{});
                    else      walk(tc, seed[act[i]], (uint64_t)len[act[i]], noise, 0, so_, &acc);
                    draft_depth_online_chain(cfg, &on, cfg.max, 0, acc, sv, ns);
                }
                cfg = cfg0;
                draft_depth_online_refresh(&cfg, &on);
            }
            /* the engine's driver: rounds in order, every row observed up to the round a decision reads */
            int ran = 1;
            for (int j = 2; j <= cfg.max; ++j) {
                const int need = draft_depth_need(cfg, j);
                for (int i = 0; i < n; ++i)
                    while (row[i].seen < need)
                        draft_depth_observe(&row[i], row[i].seen + 1,
                                            draft_depth_stat(t[(size_t)i].st[row[i].seen], row[i].seen + 1, cfg.stat));
                bool any = false;
                if (cfg.load && n > 1) any = draft_depth_go_load(cfg, row, n, j, other_rows);
                else for (int i = 0; i < n; ++i) any = draft_depth_go(cfg, &row[i], j) || any;
                if (!any) break;
                ran = j;
            }
            if (late && n > 1 && ran >= 2) {
                /* the driver's wait: every row observed up to the rounds the choice may read */
                const int upto = late == 2 ? ran : std::max(1, ran - cfg.late_lag);
                for (int i = 0; i < n; ++i)
                    while (row[i].seen < upto)
                        draft_depth_observe(&row[i], row[i].seen + 1,
                                            draft_depth_stat(t[(size_t)i].st[row[i].seen], row[i].seen + 1, cfg.stat));
                if (late == 1) {
                    DraftDepthRow chk[4];
                    for (int i = 0; i < n; ++i) chk[i] = row[i];
                    draft_depth_select(cfg, row, n, ran, other_rows);
                    select_peek(cfg, chk, n, ran, other_rows, 0);
                    for (int i = 0; i < n; ++i) R.mismatch += chk[i].depth != row[i].depth;
                } else {
                    select_peek(cfg, row, n, ran, other_rows, 1);
                }
            }
            R.rounds += ran; ++steps;
            for (int i = 0; i < n; ++i) {
                const int q = act[i];
                const int D = row[i].depth < 1 ? 1 : row[i].depth;
                Tree& tr = t[(size_t)i];
                float sv[8];
                int   ns = 0;
                for (; ns < ran && ns < 8; ++ns) {
                    const double x = draft_depth_stat(tr.st[ns], ns + 1, DRAFT_DEPTH_TOP);
                    if (x < 0) break;
                    sv[ns] = (float)x;
                }
                for (int j = D; j < 64; ++j) tr.st[j] = RadMtpCoupleState{};
                tr.n = D;
                R.mean_depth += D; ++R.share[D < 8 ? D : 7]; ++rows;
                int step_out[64], accepted = 0;
                const int m = trav ? walk_trav(tr, seed[q], (uint64_t)len[q], noise, 0, step_out, &accepted, RadMtpTravOpt{})
                                   : walk(tr, seed[q], (uint64_t)len[q], noise, 0, step_out, &accepted);
                R.mean_acc += accepted;
                for (int k = 0; k < m && len[q] < 16; ++k) { out[q][len[q]++] = step_out[k]; h[q] = child_hash(h[q], step_out[k]); }
                /* the chain is settled: the engine feeds it when its verify is committed */
                if (online == 1)
                    draft_depth_online_chain(cfg, &on, D, cfg.load && n > 1 ? row[i].rule : 0, accepted, sv, ns);
            }
            if (online == 1 && (++fed & 15) == 0) draft_depth_online_refresh(&cfg, &on);
        }
        for (int q = 0; q < n_seq; ++q) {
            ++cnt1[(size_t)q][(size_t)out[q][0]];
            ++cnt3[(size_t)q][(size_t)((out[q][0] * V + out[q][1]) * V + out[q][2])];
        }
    }
    std::vector<double> e1((size_t)V, 0), e3((size_t)V * V * V, 0);
    const Law l0 = law_at(1, noise);
    for (int a = 0; a < V; ++a) {
        e1[(size_t)a] = l0.p[a];
        const uint64_t ha = child_hash(1, a);
        const Law l1 = law_at(ha, noise);
        for (int b = 0; b < V; ++b) {
            const Law l2 = law_at(child_hash(ha, b), noise);
            for (int c = 0; c < V; ++c) e3[(size_t)((a * V + b) * V + c)] = l0.p[a] * l1.p[b] * l2.p[c];
        }
    }
    for (int q = 0; q < n_seq; ++q) {
        R.z3[q] = zscore(cnt3[(size_t)q], e3, trials);
        R.z1[q] = zscore(cnt1[(size_t)q], e1, trials);
    }
    R.mean_acc /= rows; R.mean_depth /= rows; R.rounds /= steps;
    for (double& x : R.share) x /= rows;
    return R;
}

/* p = conf (cal_x 0..1), one price for a row and one for a round at every row count. */
DraftDepthCfg depth_cfg_load(int max, int lag, double stop, double trow, double tround) {
    DraftDepthCfg c = depth_cfg(max, lag, DRAFT_DEPTH_TOP, stop);
    c.load = true;
    c.load_min = 1;           /* the tests weigh round 2 too wherever the rule has read a law for it */
    /* one token a millisecond, so the prices in milliseconds are prices in drafts */
    for (int n = 0; n <= DraftDepthCfg::kLoadSeqs; ++n) {
        c.load_lam[n] = 1.0; c.load_row_ms[n] = trow; c.load_round_ms[n] = tround;
    }
    c.cal_x[0] = 0.0; c.cal_x[1] = 1.0;
    for (int f = 0; f < 3; ++f)
        for (int j = 0; j <= DraftDepthCfg::kCalRounds; ++j) { c.cal_p[f][j][0] = 0.0; c.cal_p[f][j][1] = 1.0; }
    return c;
}

}  /* namespace */

TEST(load_aware_depth_rule_prices_rows_and_rounds_apart) {
    DraftDepthCfg c = depth_cfg_load(5, 1, 0.85, 0.5, 0.6);
    CHECK(draft_depth_cal(c, 3, 0.25) == 0.25);
    CHECK(draft_depth_cal(c, 3, 2.0) == 1.0);
    /* three rows at conf .9, .8, .4 deciding round 3: two want it (gain .4 + .3 = .7 >= .6) */
    DraftDepthRow r[3];
    const double s1[3] = { 0.9, 0.8, 0.4 };
    for (int i = 0; i < 3; ++i) { draft_depth_begin(c, &r[i], false); }
    CHECK(draft_depth_go_load(c, r, 3, 2));                 /* round 2 reads nothing: conf 1, gain 1.5 */
    for (int i = 0; i < 3; ++i) draft_depth_observe(&r[i], 1, s1[i]);
    CHECK(draft_depth_go_load(c, r, 3, 3));
    CHECK_EQ(r[0].depth, 3); CHECK_EQ(r[1].depth, 3); CHECK_EQ(r[2].depth, 2);
    CHECK(r[2].done); CHECK(!r[0].done);
    /* round 4: only row 0 still wants (.9 * .9 = .81, gain .31 < .6): the round does not run for one row */
    const double s2[3] = { 0.9, 0.6, 0.9 };
    for (int i = 0; i < 3; ++i) draft_depth_observe(&r[i], 2, s2[i]);
    CHECK(!draft_depth_go_load(c, r, 3, 4));
    CHECK_EQ(r[0].depth, 3); CHECK(r[0].done); CHECK(r[1].done);
    /* a cheap round lets the one row go on; a greedy row (no law) keeps its fixed depth and forces its rounds */
    c.load_round_ms[3] = 0.2;
    DraftDepthRow g[3];
    for (int i = 0; i < 3; ++i) draft_depth_begin(c, &g[i], i == 2);
    CHECK(draft_depth_go_load(c, g, 3, 2));
    for (int i = 0; i < 3; ++i) draft_depth_observe(&g[i], 1, s1[i]);
    for (int i = 0; i < 2; ++i) g[i].conf = 0.1;            /* neither sampled row wants round 3 */
    CHECK(draft_depth_go_load(c, g, 3, 3));                 /* the lawless row's third round runs */
    CHECK_EQ(g[0].depth, 2); CHECK_EQ(g[1].depth, 2); CHECK_EQ(g[2].depth, 3);
    CHECK(!draft_depth_go_load(c, g, 3, 4));
    /* a tile's price: six rows at conf .8 beside a step of 30 rows, the tile past 32 rows at 1.5 drafts --
     * two fill the tile (gain 2 x .3), the other four would open the next one (4 x .3 < 1.5) and stay out */
    DraftDepthCfg t = depth_cfg_load(5, 1, 0.85, 0.5, 0.1);
    for (auto& jm : t.load_jump_ms) jm[1] = 1.5;          /* past 32 rows */
    DraftDepthRow q[6];
    for (int i = 0; i < 6; ++i) { draft_depth_begin(t, &q[i], false); q[i].depth = 2; draft_depth_observe(&q[i], 1, 0.8); }
    CHECK_EQ(draft_depth_load_cost(t, 6, 32) + 1.5 + 0.5, draft_depth_load_cost(t, 6, 33));
    CHECK(draft_depth_go_load(t, q, 6, 3, 30 - 6 * 3));      /* R = 30: the six rows' 18 and 12 others */
    int went = 0;
    for (int i = 0; i < 6; ++i) went += q[i].depth == 3;
    CHECK_EQ(went, 2);
    CHECK_EQ(q[0].depth, 3); CHECK_EQ(q[1].depth, 3); CHECK(q[5].done);
    /* without the tile's price all six go */
    for (auto& jm : t.load_jump_ms) jm[1] = 0.0;
    for (int i = 0; i < 6; ++i) { draft_depth_begin(t, &q[i], false); q[i].depth = 2; draft_depth_observe(&q[i], 1, 0.8); }
    CHECK(draft_depth_go_load(t, q, 6, 3, 12));
    went = 0;
    for (int i = 0; i < 6; ++i) went += q[i].depth == 3;
    CHECK_EQ(went, 6);
}

TEST(load_aware_chain_depth_keeps_the_target_law) {
    const int trials = 200000;
    g_sharp = 2.0;
    /* `tile`: a 16-row tile's price with seven other rows in the step, so the third round's rows
     * open a tile and the rule picks which of them by their p. */
    struct C { int n, lag; double trow, tround, tile; const char* name; };
    const C cs[] = { { 2, 1, 0.30, 0.25, 0.0, "2 rows lag1" }, { 3, 1, 0.35, 0.40, 0.0, "3 rows lag1" },
                     { 3, 0, 0.30, 0.30, 0.0, "3 rows lag0" }, { 3, 1, 0.20, 0.05, 0.45, "3 rows tile" } };
    for (const C& c : cs) {
        DraftDepthCfg cfg = depth_cfg_load(4, c.lag, 0.40, c.trow, c.tround);
        for (auto& jm : cfg.load_jump_ms) jm[0] = c.tile;     /* past 16 rows */
        for (int trav = 1; trav >= 0; --trav) {
            const LoadRun r = run_depth_load(cfg, c.n, trav != 0, trials, 0.6, c.tile > 0 ? 7 : 0);
            double z3 = 0, z1 = 0;
            for (int q = 0; q < c.n; ++q) { z3 = std::max(z3, r.z3[q]); z1 = std::max(z1, r.z1[q]); }
            std::printf("  load %-11s %s: worst z3=%.2f z1=%.2f acc=%.3f depth %.2f rounds %.2f shares %.2f %.2f %.2f %.2f\n",
                        c.name, trav ? "trav   " : "rowwise", z3, z1, r.mean_acc, r.mean_depth, r.rounds,
                        r.share[1], r.share[2], r.share[3], r.share[4]);
            CHECK(z3 < 4.5); CHECK(z1 < 4.5);
            /* the rule cuts chains at several depths, and stops the pass before the deepest chain's wish */
            int used = 0;
            for (int d = 1; d <= 4; ++d) used += r.share[d] > 0.05;
            CHECK(used >= 2);
        }
    }
    g_sharp = 1.0;
}

TEST(late_choice_of_the_verify_rows_reads_each_draft_by_the_laws_up_to_its_own) {
    DraftDepthCfg c = depth_cfg_load(5, 1, 0.85, 0.3, 0.0);
    /* a different table for each confidence read, so the test sees which one was taken */
    for (int j = 0; j <= DraftDepthCfg::kCalRounds; ++j) {
        c.cal_p[0][j][1] = 0.25;      /* conf(j - 2): never enough */
        c.cal_p[1][j][1] = 0.50;      /* conf(j - 1) */
        c.cal_p[2][j][1] = 1.00;      /* conf(j) */
    }
    DraftDepthRow r[2];
    const double s[2][4] = { { 0.9, 0.9, 0.9, 0.9 }, { 0.9, 0.5, 0.9, 0.9 } };
    for (int i = 0; i < 2; ++i) {
        draft_depth_begin(c, &r[i], false);
        for (int k = 0; k < 3; ++k) draft_depth_observe(&r[i], k + 1, s[i][k]);   /* rounds 1..3 read, 4 ran unread */
    }
    CHECK(draft_depth_conf(r[1], 2) > 0.449 && draft_depth_conf(r[1], 2) < 0.451);
    draft_depth_select(c, r, 2, 4);
    /* row 0: drafts 2 and 3 by conf(2) = .81 and conf(3) = .729 (full table), draft 4 by conf(3) through
     * the conf(j - 1) table: .36 >= .3 -- depth 4. Row 1: draft 2 by conf(2) = .45 >= .3, draft 3 by
     * conf(3) = .405 >= .3, draft 4 by .5 x .405 = .2 < .3 -- depth 3. */
    CHECK_EQ(r[0].depth, 4); CHECK_EQ(r[1].depth, 3);
    CHECK(r[0].done && r[1].done);
    /* nothing read past round 1: draft 2 by conf(1) (the conf(j - 1) table), draft 3 by conf(1) through the
     * conf(j - 2) table, which admits nothing */
    for (int i = 0; i < 2; ++i) { draft_depth_begin(c, &r[i], false); draft_depth_observe(&r[i], 1, s[i][0]); }
    draft_depth_select(c, r, 2, 4);
    CHECK_EQ(r[0].depth, 2); CHECK_EQ(r[1].depth, 2);
    /* a greedy row keeps the rule's depth, capped by the rounds that ran */
    DraftDepthRow g[2];
    draft_depth_begin(c, &g[0], true); g[0].depth = 3;
    draft_depth_begin(c, &g[1], false); draft_depth_observe(&g[1], 1, 0.9);
    draft_depth_select(c, g, 2, 2);
    CHECK_EQ(g[0].depth, 2); CHECK_EQ(g[1].depth, 2);
}

TEST(late_choice_keeps_the_target_law_and_reading_a_round_too_far_does_not) {
    const int trials = 200000;
    g_sharp = 2.0;
    struct C { int n, late_lag; double stop, trow, tile; int other; const char* name; };
    const C cs[] = { { 2, 1, 0.20, 0.10, 0.0, 0, "2 rows, last round unread" },
                     { 3, 0, 0.20, 0.14, 0.0, 0, "3 rows, every round read" },
                     { 3, 1, 0.20, 0.06, 0.25, 7, "3 rows, a tile" },
                     { 3, 2, 0.30, 0.10, 0.0, 0, "3 rows, no wait" } };
    for (const C& c : cs) {
        DraftDepthCfg cfg = depth_cfg_load(4, 1, c.stop, c.trow, 0.0);
        cfg.load = false;                 /* the rounds by the stop rule, the rows by the late choice */
        cfg.late = true; cfg.late_lag = c.late_lag;
        for (auto& jm : cfg.load_jump_ms) jm[0] = c.tile;     /* past 16 rows */
        for (int trav = 1; trav >= 0; --trav) {
            const LoadRun r = run_depth_load(cfg, c.n, trav != 0, trials, 0.6, c.other, 1);
            double z3 = 0, z1 = 0;
            for (int q = 0; q < c.n; ++q) { z3 = std::max(z3, r.z3[q]); z1 = std::max(z1, r.z1[q]); }
            std::printf("  late %-26s %s: worst z3=%.2f z1=%.2f acc=%.3f depth %.2f rounds %.2f shares %.2f %.2f %.2f %.2f\n",
                        c.name, trav ? "trav   " : "rowwise", z3, z1, r.mean_acc, r.mean_depth, r.rounds,
                        r.share[1], r.share[2], r.share[3], r.share[4]);
            CHECK(z3 < 4.5); CHECK(z1 < 4.5);
            CHECK_EQ(r.mismatch, 0L);
            int used = 0;
            for (int d = 1; d <= 4; ++d) used += r.share[d] > 0.05;
            CHECK(used >= 2);
        }
    }
    /* the control: the same choice reading conf(j + 1) for draft j */
    DraftDepthCfg bad = depth_cfg_load(4, 1, 0.20, 0.14, 0.0);
    bad.load = false; bad.late = true; bad.late_lag = 0;
    for (int trav = 1; trav >= 0; --trav) {
        const LoadRun r = run_depth_load(bad, 3, trav != 0, 100000, 0.6, 0, 2);
        double z3 = 0, z1 = 0;
        for (int q = 0; q < 3; ++q) { z3 = std::max(z3, r.z3[q]); z1 = std::max(z1, r.z1[q]); }
        std::printf("  late control (reads the round after the draft, %s): worst z3=%.1f z1=%.1f depth %.2f\n",
                    trav ? "traversal" : "rowwise", z3, z1, r.mean_depth);
        CHECK(z3 > 8.0 || z1 > 8.0);
    }
    g_sharp = 1.0;
}

TEST(load_aware_tables_start_from_the_measured_ones_and_lists_parse) {
    DraftDepthCfg c = depth_cfg(5, 1, DRAFT_DEPTH_TOP, 0.85);
    draft_depth_load_defaults(&c);
    CHECK_EQ(draft_depth_knots(c), 8);
    CHECK_NEAR(draft_depth_cal(c, 3, 1.0, 2), 0.971, 1e-12);
    CHECK_NEAR(draft_depth_cal(c, 3, 0.90, 2), 0.5 * (0.661 + 0.824), 1e-12);
    CHECK_NEAR(draft_depth_cal(c, 2, 0.123, 0), 0.575, 1e-12);       /* round 2 is read with nothing */
    CHECK_NEAR(draft_depth_cal(c, 7, 0.99, 1), 0.823, 1e-12);        /* past the measured rounds: the last */
    CHECK(c.load_lam[1] < c.load_lam[8] && c.load_row_ms[1] > c.load_row_ms[8]);
    /* a verify's price: rows, and each tile the rows are past, once */
    const double j0 = c.load_jump_ms[8][0], j1 = c.load_jump_ms[8][1], j2 = c.load_jump_ms[8][2];
    CHECK_NEAR(draft_depth_load_cost(c, 8, 16), 16 * c.load_row_ms[8], 1e-12);
    CHECK_NEAR(draft_depth_load_cost(c, 8, 17), 17 * c.load_row_ms[8] + j0, 1e-12);
    CHECK_NEAR(draft_depth_load_cost(c, 8, 33), 33 * c.load_row_ms[8] + j0 + j1, 1e-12);
    CHECK_NEAR(draft_depth_load_cost(c, 8, 49), 49 * c.load_row_ms[8] + j0 + j1 + j2, 1e-12);
    CHECK_NEAR(draft_depth_load_cost(c, 20, 8), draft_depth_load_cost(c, 8, 8), 1e-12);   /* past 8 sequences: the last */
    double v[8];
    CHECK_EQ(draft_depth_parse_list("0.5,0.25,2", v, 8, 0.0, 10.0), 3);
    CHECK(v[0] == 0.5 && v[1] == 0.25 && v[2] == 2.0);
    CHECK_EQ(draft_depth_parse_list("1", v, 8, 0.0, 10.0), 1);
    for (const char* bad : { "", "1,", ",1", "a", "1;2", "1,2,3,4,5", "-1", "11" })
        CHECK_EQ(draft_depth_parse_list(bad, v, 4, 0.0, 10.0), -1);
}

TEST(online_calibration_counts_each_verified_draft_at_the_confidence_its_decision_read) {
    DraftDepthCfg c = depth_cfg(5, 1, DRAFT_DEPTH_TOP, 0.85);
    draft_depth_load_defaults(&c);
    c.load = true; c.late = true;
    DraftDepthOnline o;
    draft_depth_online_begin(c, &o);
    auto sum = [&](const double (*a)[DraftDepthOnline::kK], int j) {
        double x = 0;
        for (int k = 0; k < DraftDepthOnline::kK; ++k) x += a[j][k];
        return x;
    };
    /* a chain of three drafts, two accepted; conf(1) = .9 lies half way between the knots .85 and .95 */
    const float s[3] = { 0.9f, 1.0f, 1.0f };
    draft_depth_online_chain(c, &o, 3, 3, 2, s, 3);
    for (int t = 0; t < 3; ++t) {
        CHECK_NEAR(sum(o.cn[t], 2), 1.0, 1e-9); CHECK_NEAR(sum(o.ca[t], 2), 1.0, 1e-9);   /* draft 2: accepted */
        CHECK_NEAR(sum(o.cn[t], 3), 1.0, 1e-9); CHECK_NEAR(sum(o.ca[t], 3), 0.0, 1e-9);   /* draft 3: not */
        CHECK_NEAR(sum(o.cn[t], 4), 0.0, 1e-9);                                           /* never made */
    }
    CHECK_NEAR(o.cn[0][2][7], 1.0, 1e-9);                    /* table 0 of draft 2 reads conf(0) = 1 */
    CHECK_NEAR(o.cn[0][3][4], 0.5, 1e-6); CHECK_NEAR(o.cn[0][3][5], 0.5, 1e-6);   /* draft 3 by conf(1) */
    CHECK_NEAR(o.cn[1][2][4], 0.5, 1e-6); CHECK_NEAR(o.cn[2][2][5], 0.5, 1e-6);
    /* the rule sent a chain to round 4 and the late choice cut it to 2: the rule's table (0) counts
     * drafts 3 and 4 as not accepted, the late choice's tables count nothing for them */
    draft_depth_online_begin(c, &o);
    draft_depth_online_chain(c, &o, 2, 4, 2, s, 3);
    CHECK_NEAR(sum(o.cn[0], 3), 1.0, 1e-9); CHECK_NEAR(sum(o.ca[0], 3), 0.0, 1e-9);
    CHECK_NEAR(sum(o.cn[0], 4), 1.0, 1e-9); CHECK_NEAR(sum(o.ca[0], 4), 0.0, 1e-9);
    CHECK_NEAR(sum(o.cn[1], 3), 0.0, 1e-9); CHECK_NEAR(sum(o.cn[2], 4), 0.0, 1e-9);
    CHECK_NEAR(sum(o.cn[2], 2), 1.0, 1e-9); CHECK_NEAR(sum(o.ca[2], 2), 1.0, 1e-9);
    /* a step the stop rule decided (rule 0): only what was verified */
    draft_depth_online_begin(c, &o);
    draft_depth_online_chain(c, &o, 2, 0, 0, s, 3);
    CHECK_NEAR(sum(o.cn[0], 2), 1.0, 1e-9); CHECK_NEAR(sum(o.cn[0], 3), 0.0, 1e-9);

    /* the estimate: the counts on top of the static table's pseudo-observations, monotone in conf */
    draft_depth_online_begin(c, &o);
    const float one[5] = { 1, 1, 1, 1, 1 };
    for (int i = 0; i < 4000; ++i) draft_depth_online_chain(c, &o, 3, 0, i % 2 ? 3 : 2, one, 3);
    const double before = c.cal_p[2][4][7];
    draft_depth_online_refresh(&c, &o);
    CHECK(c.cal_p[2][3][7] > 0.49 && c.cal_p[2][3][7] < 0.53);          /* half of draft 3 were accepted */
    for (int k = 1; k < 8; ++k) CHECK(c.cal_p[2][3][k] >= c.cal_p[2][3][k - 1]);
    CHECK(c.cal_p[2][3][6] <= c.cal_p[2][3][7]);                        /* the knots below were pooled down to it */
    CHECK(c.cal_p[2][2][7] > 0.99);                                     /* every draft 2 was */
    CHECK_NEAR(c.cal_p[2][4][7], before, 1e-12);                        /* a draft nobody made keeps the table */
    CHECK_EQ(o.chains, 4000L);
    /* counts fade: a refresh keeps cal_keep of them */
    const double n0 = o.cn[2][3][7];
    draft_depth_online_refresh(&c, &o);
    CHECK_NEAR(o.cn[2][3][7], n0 * o.cal_keep, 1e-6);
}

TEST(online_calibration_prices_rows_and_tiles_from_step_times) {
    DraftDepthCfg c = depth_cfg(5, 1, DRAFT_DEPTH_TOP, 0.85);
    draft_depth_load_defaults(&c);
    DraftDepthOnline o;
    draft_depth_online_begin(c, &o);
    const double row4 = c.load_row_ms[4];
    /* steps of 8 sequences: 14 + 0.5 R + 0.9 D + 2.0 past 32 rows + 0.7 past 48, a little noise */
    uint64_t x = 0x9e3779b97f4a7c15ull;
    auto rnd = [&]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return (double)(x >> 11) / 9007199254740992.0; };
    double tok = 0, ms_sum = 0;
    for (int i = 0; i < 8000; ++i) {
        const int R = 17 + (int)(rnd() * 35.0);                          /* 17 .. 51 */
        const int D = std::min(5, std::max(2, 2 + (R - 17) / 9 + (rnd() < 0.3 ? 1 : 0)));
        const double ms = 14.0 + 0.5 * R + 0.9 * D + (R > 32 ? 2.0 : 0.0) + (R > 48 ? 0.7 : 0.0) + (rnd() - 0.5) * 0.6;
        const int tk = 8 + R / 3;
        draft_depth_online_step(c, &o, 8, R, D, ms, tk);
        tok += tk; ms_sum += ms;
    }
    draft_depth_online_step(c, &o, 8, 30, 3, 300.0, 18);                  /* a stall: left out */
    CHECK_EQ(o.outliers, 1L);
    draft_depth_online_refresh(&c, &o);
    std::printf("  fitted at 8 sequences: row %.3f (0.5), round %.3f (0.9; prior %.3f), tiles %.2f %.2f %.2f %.2f "
                "(-, 2.0, 0.7, -), %.4f tok/ms\n", c.load_row_ms[8], c.load_round_ms[8], o.theta0[8][2],
                c.load_jump_ms[8][0], c.load_jump_ms[8][1], c.load_jump_ms[8][2], c.load_jump_ms[8][3], c.load_lam[8]);
    /* a price the steps tell apart lands on it, or at least most of the way from the table's */
    auto near = [](double est, double truth, double prior) {
        return std::fabs(est - truth) < 0.30 || std::fabs(est - truth) < 0.4 * std::fabs(prior - truth);
    };
    CHECK(std::fabs(c.load_row_ms[8] - 0.5) < 0.05);
    CHECK(near(c.load_jump_ms[8][1], 2.0, o.theta0[8][4]));
    CHECK(near(c.load_jump_ms[8][2], 0.7, o.theta0[8][5]));
    CHECK_NEAR(c.load_jump_ms[8][3], o.theta0[8][6], 1e-6);              /* no step was past 64 rows: the table's */
    CHECK(c.load_round_ms[8] > 0.5 && c.load_round_ms[8] < 1.3);          /* between the prior and the data */
    CHECK(std::fabs(c.load_lam[8] / (tok / ms_sum) - 1.0) < 0.08);        /* the last steps' rate */
    CHECK_NEAR(c.load_row_ms[4], row4, 1e-12);                            /* a row count nobody ran keeps the table */
    /* too few steps: the table stays */
    DraftDepthCfg d = depth_cfg(5, 1, DRAFT_DEPTH_TOP, 0.85);
    draft_depth_load_defaults(&d);
    DraftDepthOnline o2;
    draft_depth_online_begin(d, &o2);
    for (int i = 0; i < 20; ++i) draft_depth_online_step(d, &o2, 2, 6, 3, 50.0, 4);
    draft_depth_online_refresh(&d, &o2);
    CHECK_NEAR(d.load_row_ms[2], 0.824, 1e-12); CHECK_NEAR(d.load_lam[2], 0.3558, 1e-12);
}

TEST(online_calibration_keeps_the_target_law_and_tables_made_from_the_steps_own_verify_do_not) {
    const int trials = 120000;
    g_sharp = 2.0;
    /* The rule and the late choice with tables that follow the run: every chain a verify settled
     * is counted, the tables are remade every 16 steps, and each later step is decided by them.
     * They are functions of steps that are over, so every sequence keeps the target's law. */
    struct C { int n, late; double trow, tround, tile; int other; const char* name; };
    const C cs[] = { { 2, 0, 0.30, 0.25, 0.0, 0, "2 rows, the rule" }, { 3, 1, 0.14, 0.10, 0.0, 0, "3 rows, rule + late" },
                     { 3, 1, 0.08, 0.05, 0.30, 7, "3 rows, late, a tile" } };
    for (const C& c : cs) {
        DraftDepthCfg cfg = depth_cfg_load(4, 1, 0.40, c.trow, c.tround);
        for (auto& jm : cfg.load_jump_ms) jm[0] = c.tile;
        cfg.late = c.late != 0;
        /* start the tables well off (p = conf / 2), so the calibration has something to move */
        for (int t = 0; t < 3; ++t)
            for (int j = 0; j <= DraftDepthCfg::kCalRounds; ++j) cfg.cal_p[t][j][1] = 0.5;
        for (int trav = 1; trav >= 0; --trav) {
            const LoadRun r = run_depth_load(cfg, c.n, trav != 0, trials, 0.6, c.other, c.late, 1);
            double z3 = 0, z1 = 0;
            for (int q = 0; q < c.n; ++q) { z3 = std::max(z3, r.z3[q]); z1 = std::max(z1, r.z1[q]); }
            std::printf("  online %-21s %s: worst z3=%.2f z1=%.2f acc=%.3f depth %.2f rounds %.2f shares %.2f %.2f %.2f %.2f\n",
                        c.name, trav ? "trav   " : "rowwise", z3, z1, r.mean_acc, r.mean_depth, r.rounds,
                        r.share[1], r.share[2], r.share[3], r.share[4]);
            CHECK(z3 < 4.5); CHECK(z1 < 4.5);
            int used = 0;
            for (int d = 1; d <= 4; ++d) used += r.share[d] > 0.05;
            CHECK(used >= 2);
        }
    }
    /* the control: the tables made from what the target accepts of the step's own chains */
    DraftDepthCfg bad = depth_cfg_load(4, 1, 0.20, 0.45, 0.0);
    bad.load = false; bad.late = true; bad.late_lag = 0;
    for (int trav = 1; trav >= 0; --trav) {
        const LoadRun r = run_depth_load(bad, 2, trav != 0, 40000, 0.6, 0, 1, 2);
        double z3 = 0, z1 = 0;
        for (int q = 0; q < 2; ++q) { z3 = std::max(z3, r.z3[q]); z1 = std::max(z1, r.z1[q]); }
        std::printf("  online control (tables from the step's own verify, %s): worst z3=%.1f z1=%.1f depth %.2f\n",
                    trav ? "traversal" : "rowwise", z3, z1, r.mean_depth);
        CHECK(z3 > 8.0 || z1 > 8.0);
    }
    g_sharp = 1.0;
}

TEST(a_depth_rule_that_reads_the_admitted_draft_or_the_target_is_not_exact) {
    const int trials = 100000;
    g_sharp = 2.0;
    const DraftDepthCfg cfg = depth_cfg(4, 0, DRAFT_DEPTH_TOP, 0.0);
    for (int trav = 0; trav < 2; ++trav) {
        const DepthRun own = run_depth(cfg, trav != 0, trials, 0.6, 1, 0.30);
        const DepthRun peek = run_depth(cfg, trav != 0, trials, 0.6, 2, 0);
        std::printf("  depth controls (%s): own-mass z3=%.1f z1=%.1f depth %.2f | target peek z3=%.1f z1=%.1f depth %.2f\n",
                    trav ? "traversal" : "rowwise", own.z3, own.z1, own.mean_depth, peek.z3, peek.z1,
                    peek.mean_depth);
        CHECK(own.z3 > 20.0 || own.z1 > 20.0);
        CHECK(peek.z3 > 20.0 || peek.z1 > 20.0);
    }
    g_sharp = 1.0;
}

static int argmax_tok(const Law& L) {
    int b = 0;
    for (int i = 1; i < V; ++i) if (L.p[i] > L.p[b]) b = i;
    return b;
}

/* ------------------------------------------------------------------ a SuffixDecoding step
 *
 * --draft-suffix replaces the head's chain of a step by the sequence's own history (suffix.h): a
 * chain of up to 8 rows chosen from the committed text alone. The head had drafted (and DRAWN, under
 * --draft-coupling) a chain for the same positions, so the coupled states of that draft are still
 * in the sampler when the step is built. The step is exact because the scheduler marks it
 * Request::draft_suffix, which keeps the sampler from binding any state to a proposed row: each
 * row's emitted token is the target's plain draw and is followed iff it equals the proposal.
 * Visible to the coupled rows, a proposal that happens to equal the head's unused draw x would
 * select the row's coupling by the VALUE of that draw, which is biased. The statistic is the first
 * three tokens' joint law over 400000 steps; the negative control must fail it. */
static void suffix_chain(Tree* t, int s, int how, int len, double noise) {
    const int orig[4] = { 0, t->tok[1], t->tok[2], t->tok[3] };
    uint64_t h = t->hash[0];
    for (int i = 0; i < len; ++i) {
        int tk = argmax_tok(law_at(h, noise));           /* a history that repeats the likely text */
        if (how == 1 && i < 3) tk = orig[i + 1];         /* the head's own draw, by chance of the text */
        if (H((uint64_t)s * 977 + (uint64_t)i * 31 + 5) % 4 == 0) tk = (tk + 1) % V;   /* a damaged copy */
        t->tok[i + 1] = tk;
        t->par[i + 1] = i;
        h = child_hash(h, tk);
        t->hash[i + 1] = h;
    }
    t->n = len;
    t->suffix = true;                                    /* Request::draft_suffix: no row binds a state */
}

TEST(a_suffix_chain_beside_the_heads_stale_coupled_states_emits_the_target_law) {
    for (int how = 0; how < 2; ++how) {
        for (int len : { 3, 6, 8 }) {
            const std::function<void(Tree*, int)> h = [how, len](Tree* t, int s) { suffix_chain(t, s, how, len, 1.0); };
            g_suffix_visible = false;
            const Result ok = run_kind(CHAIN, 400000, 1.0, &h);
            std::printf("  suffix chain, rows %d, how %d, hidden: z=%.2f first=%.2f acc=%.2f\n", len, how,
                        ok.z, ok.first_z, ok.mean_acc);
            CHECK(ok.z < 4.5);
            CHECK(ok.first_z < 4.5);
        }
    }
    /* the negative control: the same chains with the rows visible to the coupled verify */
    double worst = 0;
    g_suffix_visible = true;
    for (int how = 0; how < 2; ++how) {
        const std::function<void(Tree*, int)> h = [how](Tree* t, int s) { suffix_chain(t, s, how, 6, 1.0); };
        const Result r = run_kind(CHAIN, 400000, 1.0, &h);
        std::printf("  suffix chain visible to the coupled rows (known wrong), how %d: z=%.1f first=%.1f\n",
                    how, r.z, r.first_z);
        worst = std::max(worst, std::max(r.z, r.first_z));
    }
    g_suffix_visible = false;
    CHECK(worst > 8.0);
}

/* A suffix step is the plain rule: the row's draw is the target's and the proposal is followed iff
 * it equals it. A chain whose first row is never the likeliest token still emits the target's law. */
TEST(a_suffix_chain_with_a_wrong_first_row_stays_exact) {
    const std::function<void(Tree*, int)> wrong = [](Tree* t, int s) {
        suffix_chain(t, s, 0, 6, 1.0);
        const Law L0 = law_at(t->hash[0], 1.0);
        t->tok[1] = (argmax_tok(L0) + 1 + (int)(H((uint64_t)s) % (V - 1))) % V;   /* never the likeliest, any other */
        t->hash[1] = child_hash(t->hash[0], t->tok[1]);
        for (int i = 1; i < t->n; ++i) t->hash[i + 1] = child_hash(t->hash[i], t->tok[i + 1]);
    };
    g_suffix_visible = false;
    const Result r = run_kind(CHAIN, 300000, 1.0, &wrong);
    std::printf("  suffix chain, first row wrong: z=%.2f first=%.2f\n", r.z, r.first_z);
    CHECK(r.z < 4.5);
    CHECK(r.first_z < 4.5);
}

/* ------------------------------------------------------------------ the draft law's recipe
 *
 * rad_mtp_couple_build_law: the draft temperature (dscale = 1 / T_draft) and the request's top-k /
 * top-p cut change WHICH Q the head drafts from, never the verify's exactness, because the state
 * stores the Q that was drawn from and the verify reads it. */

/* ==================================================================== --draft-suffix beside --draft-depth adaptive
 *
 * A step is either a suffix step (the committed text recurs: an uncoupled chain of 1..7 rows copied
 * from the history, Request::draft_suffix, no row binds a state) or an adaptive MTP chain (coupled, cut by
 * the head's own laws), chosen by a function of the committed text alone -- the history hash here,
 * the matcher over prompt ++ output in the engine. Both are verified by the traversal walk or row by
 * row, and a generation runs over several such steps, so each step's kind and depth depend on what
 * the steps before it committed. The negative control shows the suffix rows visible to the coupled
 * rows' child lists (the head's stale states of the same positions), which is biased. */
namespace {

DepthRun run_mixed(const DraftDepthCfg& cfg, bool trav, int trials, double noise, int share_pct, int how) {
    std::vector<double> cnt3((size_t)V * V * V, 0), cnt1((size_t)V, 0);
    DepthRun R{};
    double steps = 0, sfx = 0;
    for (int s = 0; s < trials; ++s) {
        const uint64_t seed = 0x7a31c5ull + (uint64_t)s * 130363ull;
        int out[16], len = 0;
        uint64_t h = 1;
        while (len < 3) {
            Tree t;
            /* the head drafts for this step either way (the history pass is paid behind a suffix step) */
            const int D = draft_depth_chain(&t, seed, (uint64_t)len, cfg, noise, h, 0, 0);
            const bool is_sfx = (int)(H(h * 31 + 7) % 100) < share_pct;
            if (is_sfx) {
                const int rows = 1 + (int)(H(h * 17 + 3) % 7);
                suffix_chain(&t, (int)(h % 100003), how, rows, 1.0);
                sfx += 1;
            }
            R.mean_depth += is_sfx ? t.n : D; ++steps;
            int step_out[64], accepted = 0, n;
            if (trav) n = walk_trav(t, seed, (uint64_t)len, noise, 0, step_out, &accepted, RadMtpTravOpt{});
            else      n = walk(t, seed, (uint64_t)len, noise, 0, step_out, &accepted);
            R.mean_acc += accepted;
            for (int i = 0; i < n && len < 16; ++i) { out[len++] = step_out[i]; h = child_hash(h, step_out[i]); }
        }
        ++cnt1[(size_t)out[0]];
        ++cnt3[(size_t)((out[0] * V + out[1]) * V + out[2])];
    }
    std::vector<double> e1((size_t)V, 0), e3((size_t)V * V * V, 0);
    const Law l0 = law_at(1, noise);
    for (int a = 0; a < V; ++a) {
        e1[(size_t)a] = l0.p[a];
        const uint64_t ha = child_hash(1, a);
        const Law l1 = law_at(ha, noise);
        for (int b = 0; b < V; ++b) {
            const Law l2 = law_at(child_hash(ha, b), noise);
            for (int c = 0; c < V; ++c) e3[(size_t)((a * V + b) * V + c)] = l0.p[a] * l1.p[b] * l2.p[c];
        }
    }
    R.z3 = zscore(cnt3, e3, trials);
    R.z1 = zscore(cnt1, e1, trials);
    R.mean_acc /= steps; R.mean_depth /= steps;
    R.share[0] = sfx / steps;
    return R;
}

}  /* namespace */

TEST(suffix_steps_beside_adaptive_depth_chains_emit_the_target_law) {
    g_sharp = 2.0;
    g_suffix_visible = false;
    for (int how = 0; how < 2; ++how)
        for (int pct : { 30, 70 }) {
            const DraftDepthCfg cfg = depth_cfg(4, 1, DRAFT_DEPTH_TOP, 0.40);
            const DepthRun tv = run_mixed(cfg, true, 250000, 0.6, pct, how);
            const DepthRun rw = run_mixed(cfg, false, 250000, 0.6, pct, how);
            std::printf("  suffix %d%% of steps (how %d): trav z3=%.2f z1=%.2f acc=%.3f | rowwise z3=%.2f z1=%.2f acc=%.3f | "
                        "suffix share %.2f rows/draft %.2f\n", pct, how, tv.z3, tv.z1, tv.mean_acc, rw.z3, rw.z1,
                        rw.mean_acc, tv.share[0], tv.mean_depth);
            CHECK(tv.z3 < 4.5); CHECK(tv.z1 < 4.5);
            CHECK(rw.z3 < 4.5); CHECK(rw.z1 < 4.5);
        }
    /* negative control: the suffix rows are children of the coupled rows (the head's own unused
     * draw may equal the copied token, selecting the coupling by the VALUE of that draw) */
    g_suffix_visible = true;
    double worst = 0;
    for (int trav = 0; trav < 2; ++trav) {
        /* every step a suffix step over a head that drafted its full chain (all its states are stale) */
        const DepthRun r = run_mixed(depth_cfg(4, 1, DRAFT_DEPTH_TOP, 0.0), trav != 0, 250000, 0.6, 100, 0);
        std::printf("  suffix rows visible to the coupled rows (known wrong, %s): z3=%.1f z1=%.1f\n",
                    trav ? "traversal" : "rowwise", r.z3, r.z1);
        worst = std::max(worst, std::max(r.z3, r.z1));
    }
    g_suffix_visible = false;
    g_sharp = 1.0;
    CHECK(worst > 8.0);
}

RAD_TEST_MAIN()
