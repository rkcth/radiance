/* suffix.h -- SuffixDecoding's proposal for a hybrid step selector (--draft-suffix, docs/GUIDE.md).
 *
 * SuffixDecoding (Oliaro et al., arXiv 2411.04975) drafts from a suffix tree of earlier text: for
 * the longest suffix of the current context that occurred before, it expands the tokens that
 * followed, each node scored by the product of its followers' frequencies (D, the expected number of
 * accepted tokens along its path). Here the tree is the per-request one (prompt ++ output of this
 * request; no tree over other requests), reduced to what a chain needs:
 *
 *   - the earlier occurrence comes from the existing NgramIndex (the --spec-ngram matcher): the
 *     latest earlier occurrence of the longest recurring suffix of `min_match..min_match+3` tokens;
 *   - the frequency rule comes from two count tables over `min_match`-token contexts: how often a
 *     context was followed by anything, and by each token. A proposal token at depth j has
 *     C_j = count(context_j + token_j) / count(context_j) with context_j the last `min_match`
 *     tokens of the sequence extended by the proposal so far, and the chain is cut where its path
 *     score D_j = prod C_i falls under `min_score`.
 *
 * It is a heuristic and only that: every proposed token is verified (the sampler's own draw must
 * equal it), so a collision in the tables, a stale count or a wrong match changes how many tokens a
 * step lands, never which tokens come out. The tables key on 64-bit hashes and store 32 bits of the
 * key; a clash merges two contexts' counts.
 *
 * The index is incremental: each call indexes the tokens committed since the last one, prompt
 * included at the first, so a sequence is indexed once over its lifetime. Memory is about 8 bytes a
 * token per table and per NgramIndex pattern length (4 lengths).
 */
#pragma once
#include "ngram.h"

#include <cstdint>
#include <vector>

namespace rad {

class SuffixIndex {
public:
    /* NgramIndex tries pattern lengths min_match .. min_match + kExtra, longest first: the proposal
     * follows the latest occurrence of the longest recurring suffix in that range. */
    enum { kExtra = 3, kMaxK = 12 };

    /* Forget the sequence; `owner` is the request id the next lookups are for. */
    void reset(uint64_t owner = 0) {
        ix_.reset(owner);
        followed_.clear();
        follow_.clear();
        filled_ = 0;
        k_ = 0;
    }
    uint64_t owner() const { return ix_.owner(); }

    struct Out {
        int   n = 0;          /* tokens proposed */
        int   match = 0;      /* how far the match behind the proposal runs back (>= min_match) */
        float score = 0.0f;   /* D of the last proposed node: the chance the whole chain is taken */
        float expect = 0.0f;  /* sum of D along the chain: expected accepted tokens */
    };

    /* The proposal for `seq`: at most `max_rows` tokens into `out` (the index is brought up to the
     * sequence first). n == 0 when the last `min_match` tokens did not occur before, or the first
     * token's own score is under `min_score`. 1 <= min_match <= kMaxK. */
    Out propose(const TokenSeq& seq, uint64_t owner, int min_match, int max_rows, float min_score,
                int32_t* out) {
        Out r;
        if (min_match < 1 || min_match > kMaxK || max_rows < 1 || seq.size() <= min_match) return r;
        if (ix_.owner() != owner || k_ != min_match) { reset(owner); k_ = min_match; }
        const int64_t L = seq.size();
        if (L < filled_) { reset(owner); k_ = min_match; }
        index_to(seq, L);
        int match = 0;
        const int n = ix_.propose(seq, min_match, min_match + kExtra, max_rows, out, &match, 64);
        if (n <= 0) return r;
        /* Score the chain: the context moves forward over the proposal's own tokens. */
        int32_t win[kMaxK];
        for (int i = 0; i < k_; ++i) win[i] = seq[L - k_ + i];
        float d = 1.0f, sum = 0.0f;
        int m = 0;
        for (; m < n; ++m) {
            const uint64_t h = hash_ctx(win, k_);
            const uint32_t a = followed_.get(h);
            const uint32_t b = follow_.get(mix(h, (uint64_t)(uint32_t)out[m]));
            if (a == 0 || b == 0) break;                   /* a path the history never took */
            const float c = (float)b / (float)a;
            if (d * c < min_score) break;
            d *= c;
            sum += d;
            for (int i = 0; i + 1 < k_; ++i) win[i] = win[i + 1];
            win[k_ - 1] = out[m];
        }
        r.n = m;
        r.match = match;
        r.score = m > 0 ? d : 0.0f;
        r.expect = sum;
        return r;
    }

    int64_t indexed() const { return filled_; }
    size_t  bytes() const { return ix_.bytes() + (followed_.t.capacity() + follow_.t.capacity()) * 8; }

private:
    /* Open addressing; an entry is key (the high 32 bits of a 64-bit hash, never 0) << 32 | count.
     * The probe starts from the stored key, so a rehash needs nothing else. */
    struct Table {
        std::vector<uint64_t> t;
        uint64_t mask = 0;
        int64_t  n = 0;
        void clear() { t.clear(); t.shrink_to_fit(); mask = 0; n = 0; }
        static uint32_t key_of(uint64_t h) { return (uint32_t)(h >> 32) | 1u; }
        static uint64_t start(uint32_t key) { return ((uint64_t)key * 0x9e3779b97f4a7c15ull) >> 29; }
        uint32_t get(uint64_t h) const {
            if (t.empty()) return 0;
            const uint32_t key = key_of(h);
            for (uint64_t i = start(key) & mask;; i = (i + 1) & mask) {
                const uint64_t e = t[i];
                if (e == 0) return 0;
                if ((uint32_t)(e >> 32) == key) return (uint32_t)e;
            }
        }
        void bump(uint64_t h) {
            if (t.empty() || (uint64_t)(n + 1) * 2 > mask + 1) grow();
            const uint32_t key = key_of(h);
            for (uint64_t i = start(key) & mask;; i = (i + 1) & mask) {
                uint64_t& e = t[i];
                if (e == 0) { e = (uint64_t)key << 32 | 1u; ++n; return; }
                if ((uint32_t)(e >> 32) == key) { if ((uint32_t)e != 0xffffffffu) ++e; return; }
            }
        }
        void grow() {
            std::vector<uint64_t> old;
            old.swap(t);
            const uint64_t cap = old.empty() ? 1024 : (mask + 1) * 2;
            t.assign(cap, 0);
            mask = cap - 1;
            for (uint64_t e : old) {
                if (!e) continue;
                for (uint64_t i = start((uint32_t)(e >> 32)) & mask;; i = (i + 1) & mask)
                    if (t[i] == 0) { t[i] = e; break; }
            }
        }
    };

    NgramIndex ix_;
    Table      followed_;    /* a k-token context -> how many times anything followed it */
    Table      follow_;      /* a k-token context and a token -> how many times that token did */
    int64_t    filled_ = 0;  /* positions < filled_ are counted */
    int        k_ = 0;

    static uint64_t mix(uint64_t h, uint64_t v) {
        h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h *= 0xff51afd7ed558ccdull;
        h ^= h >> 32;
        return h;
    }
    static uint64_t hash_ctx(const int32_t* t, int k) {
        uint64_t h = 0x84222325cbf29ce4ull;
        for (int i = 0; i < k; ++i) h = mix(h, (uint64_t)(uint32_t)t[i]);
        return h;
    }
    /* Count, for every position e in [filled_, upto) with k tokens before it, the context
     * seq[e-k..e-1] followed by seq[e]. */
    void index_to(const TokenSeq& seq, int64_t upto) {
        int32_t win[kMaxK];
        for (int64_t e = filled_ < k_ ? k_ : filled_; e < upto; ++e) {
            for (int i = 0; i < k_; ++i) win[i] = seq[e - k_ + i];
            const uint64_t h = hash_ctx(win, k_);
            followed_.bump(h);
            follow_.bump(mix(h, (uint64_t)(uint32_t)seq[e]));
        }
        filled_ = upto;
    }
};

}  /* namespace rad */
