/* plugin/spechub/spechub.hip -- the device launches for the coupling ops.
 *
 * The propose kernel is one thread a row over abi/rad_spechub.h's helpers. The verify is one
 * workgroup a row: the masses are computed in parallel, and the sums and walks run on one thread
 * in ascending candidate order, the order rad_mtp_couple_decide uses, so the token matches the
 * host reference bit for bit.
 *
 * Strict FP64: fast-math or FMA contraction would reorder those sums and change the answer, hence
 * the #error below and -ffp-contract=off in CMakeLists.txt.
 */
#ifdef __FAST_MATH__
#error "the MTP coupling requires strict FP64 arithmetic: build without -ffast-math"
#endif

#include <hip/hip_runtime.h>

#include "spechub.h"
#include "../../libr4d/r4d_emit.h"

namespace {

/* One thread a row: both ops are row-local and the M is the number of sampled positions. */
__global__ void propose(RadMtpCoupleProposeCall c) {
    const int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row < c.M) rad_mtp_couple_propose_row(c, row);
}

/* One workgroup a row, 128 threads. The weight and residual planes are in LDS: 16 KiB of FP64 at
 * n_cand 1024. */
constexpr int kThreads = 128;
constexpr int64_t kMaxCand = 1024;

/* The row's live candidates (id >= 0, value != -inf), compacted in ascending index order into `lj`, with
 * the validity flag (no id >= vocab, NaN or +inf among the ids >= 0) and the maximum value over the
 * ids >= 0: what the serial scans of the host reference read, in its order. The sampler's plane is
 * kSampleCandidates wide and mostly holes (top_k 20 leaves about 20 live), so the serial part afterwards
 * walks that many entries, not n_cand. Every thread of the workgroup calls it. */
__device__ void couple_live(const int32_t* ids, const float* val, int64_t n, int64_t vocab,
                            short* lj, int* n_live, int* valid, float* top) {
    __shared__ int wcnt[kThreads / 32], wbad[kThreads / 32], sbase;
    __shared__ float wmax[kThreads / 32];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    if (tid == 0) sbase = 0;
    int bad = 0;
    float mx = -INFINITY;
    for (int64_t c0 = 0; c0 < n; c0 += kThreads) {
        const int64_t j = c0 + tid;
        bool live = false;
        if (j < n) {
            const int32_t id = ids[j];
            const float v = val[j];
            if (id >= 0) {
                if (id >= vocab || v != v || v == INFINITY) bad = 1;
                else if (v > mx) mx = v;
                live = v != -INFINITY;
            }
        }
        const uint64_t ball = __ballot(live);
        const int below = __popcll(ball & ((1ull << lane) - 1ull));
        if (lane == 0) wcnt[wave] = __popcll(ball);
        __syncthreads();
        int off = sbase, tot = 0;
        for (int w = 0; w < kThreads / 32; ++w) {
            if (w < wave) off += wcnt[w];
            tot += wcnt[w];
        }
        if (live) lj[off + below] = (short)j;
        __syncthreads();
        if (tid == 0) sbase += tot;
    }
    for (int d = 16; d > 0; d >>= 1) {
        bad |= __shfl_xor(bad, d);
        mx = fmaxf(mx, __shfl_xor(mx, d));
    }
    if (lane == 0) { wbad[wave] = bad; wmax[wave] = mx; }
    __syncthreads();
    if (tid == 0) {
        int b = 0;
        float m = -INFINITY;
        for (int w = 0; w < kThreads / 32; ++w) { b |= wbad[w]; m = fmaxf(m, wmax[w]); }
        *valid = !b;
        *top = m;
        *n_live = sbase;
    }
    __syncthreads();
}

/* The live candidates of a row as the shared decisions read them (rad_mtp_rrs_decide_over,
 * rad_mtp_trav_*): the sampler's ids, the compaction (`lj`, ascending candidate order), each live
 * candidate's FP64 weight exp(val - top) and its slot in the node's support (count when outside),
 * all computed by the workgroup and stored in LDS. */
struct LiveCands {
    const int32_t* ids; const short* lj; const double* w; const unsigned char* k; int L;
    template <class F> __device__ void each(F&& f) const {
        for (int i = 0; i < L; ++i)
            if (!f(ids[lj[i]], w[i], (int)k[i])) return;
    }
};

/* --draft-verify block (abi/rad_mtp_block.h). The workgroup of a block's round-1 row decides the
 * whole block and writes the tokens of rows 0 .. tau; the block's other rows return. Every FP64 sum
 * over a candidate plane is the header's fixed lane tree (slot j to lane j % 128, ascending within
 * the lane, butterfly over each wave's 32 lanes, the four wave sums as ((0+1)+(2+3))): the
 * workgroup computes all exps and lane partials in parallel and thread 0 only runs the 8-wide
 * plan, so the result is the host reference's bit for bit (rad_mtp_block_decide). Called by every
 * thread of the workgroup. */
static_assert(kThreads == RAD_MTP_BLOCK_LANES && RAD_MTP_BLOCK_WAVE == 32, "the tree's shape");

__device__ void block_verify(const RadMtpCoupleVerifyCall& c, int64_t row) {
    constexpr int kW = kThreads / 32;
    __shared__ RadMtpCoupleState sst[RAD_MTP_BLOCK_MAX];
    __shared__ RadBlockStat      sstat[RAD_MTP_BLOCK_MAX];
    __shared__ RadBlockPlan      plan;
    __shared__ float             wmax[RAD_MTP_BLOCK_MAX][kW];
    __shared__ int               wbad[RAD_MTP_BLOCK_MAX][kW];
    __shared__ double            wacc[RAD_MTP_BLOCK_MAX][kW][10];
    __shared__ RadBlockLane      lanes[kThreads];
    __shared__ int   g_s, ok_s, tau_s;
    __shared__ int32_t y_s;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    if (tid == 0) {
        const auto& s0 = *reinterpret_cast<const RadMtpCoupleState*>(c.state + row * c.ld[3]);
        g_s = s0.blk;
        ok_s = s0.round == 1 && g_s >= 1 && g_s <= RAD_MTP_BLOCK_MAX && row + g_s <= c.M;
        if (ok_s) {
            const auto& p0 = *reinterpret_cast<const RadSampleParams*>(c.params + row * c.ld[2]);
            for (int i = 0; i < g_s && ok_s; ++i) {
                sst[i] = *reinterpret_cast<const RadMtpCoupleState*>(c.state + (row + i) * c.ld[3]);
                const auto& p = *reinterpret_cast<const RadSampleParams*>(c.params + (row + i) * c.ld[2]);
                ok_s = rad_mtp_block_row_ok(sst[0], sst[i], p0, p, i, g_s, c.vocab);
            }
        }
    }
    __syncthreads();
    if (!ok_s) return;
    const int g = g_s;
    const int64_t n = c.n_cand;
    /* every row's validity and maximum */
    for (int i = 0; i < g; ++i) {
        const int32_t* ids = c.idx + (row + i) * c.ld[0];
        const float*   val = c.val + (row + i) * c.ld[1];
        int bad = 0;
        float mx = -INFINITY;
        for (int64_t j = tid; j < n; j += kThreads) {
            const int32_t id = ids[j];
            const float v = val[j];
            if (id >= 0) {
                if (id >= c.vocab || v != v || v == INFINITY) bad = 1;
                else if (v > mx) mx = v;
            }
        }
        for (int d = 16; d > 0; d >>= 1) {
            bad |= __shfl_xor(bad, d);
            mx = fmaxf(mx, __shfl_xor(mx, d));
        }
        if (lane == 0) { wbad[i][wave] = bad; wmax[i][wave] = mx; }
    }
    __syncthreads();
    /* the lane partials and the butterfly of every row */
    for (int i = 0; i < g; ++i) {
        const int32_t* ids = c.idx + (row + i) * c.ld[0];
        const float*   val = c.val + (row + i) * c.ld[1];
        float top = wmax[i][0];
        int bad = wbad[i][0];
        for (int w = 1; w < kW; ++w) { top = fmaxf(top, wmax[i][w]); bad |= wbad[i][w]; }
        if (bad || top == -INFINITY) return;   /* uniform: the same LDS everywhere */
        RadBlockAcc a;
        for (int k = 0; k < 10; ++k) a.v[k] = 0;
        for (int64_t j = tid; j < n; j += kThreads)
            if (rad_mtp_block_live(ids[j], val[j])) rad_mtp_block_acc_add(&a, sst[i], ids[j], val[j], top);
        for (int d = 16; d > 0; d >>= 1)
            for (int k = 0; k < 10; ++k) a.v[k] = a.v[k] + __shfl_xor(a.v[k], d);
        if (lane == 0)
            for (int k = 0; k < 10; ++k) wacc[i][wave][k] = a.v[k];
    }
    __syncthreads();
    if (tid == 0) {
        for (int i = 0; i < g && ok_s; ++i) {
            RadBlockAcc t;
            for (int k = 0; k < 10; ++k)
                t.v[k] = (wacc[i][0][k] + wacc[i][1][k]) + (wacc[i][2][k] + wacc[i][3][k]);
            float top = wmax[i][0];
            for (int w = 1; w < kW; ++w) top = fmaxf(top, wmax[i][w]);
            if (!rad_mtp_block_stat_from(t, 1, top, sst[i], &sstat[i])) ok_s = 0;
        }
        if (ok_s) {
            const auto& p0 = *reinterpret_cast<const RadSampleParams*>(c.params + row * c.ld[2]);
            rad_mtp_block_plan(sst, sstat, g, &plan);
            tau_s = rad_mtp_block_tau(plan, p0.seed, p0.pos);
        }
        y_s = -1;
    }
    __syncthreads();
    if (!ok_s) return;
    const int tau = tau_s;
    const auto& p0 = *reinterpret_cast<const RadSampleParams*>(c.params + row * c.ld[2]);
    if (tau < g) {
        const int32_t* ids = c.idx + (row + tau) * c.ld[0];
        const float*   val = c.val + (row + tau) * c.ld[1];
        float top = wmax[tau][0];
        for (int w = 1; w < kW; ++w) top = fmaxf(top, wmax[tau][w]);
        RadBlockLane a{ 0, 0, -1, -1 };
        for (int64_t j = tid; j < n; j += kThreads)
            if (rad_mtp_block_live(ids[j], val[j]))
                rad_mtp_block_lane_add(&a, sst[tau], sstat[tau], plan.p[tau], ids[j], val[j], top);
        lanes[tid] = a;
        __syncthreads();
        if (tid == 0) {
            const RadBlockRow r{ ids, val, n, 1, top };
            if (!rad_mtp_block_resid_walk(r, sst[tau], sstat[tau], plan.p[tau],
                                          rad_spechub_uniform(p0.seed, p0.pos + (uint64_t)tau,
                                                              RAD_SAMPLE_KEY_RESID),
                                          lanes, &y_s))
                ok_s = 0;
        }
        __syncthreads();
        if (!ok_s) return;
    }
    if (tid == 0) rad_mtp_block_write(c, row, sst, tau, g, y_s);
}

/* Above kMaxCand the LDS planes would overflow, so one thread runs the serial helper. The helper
 * carries the block row's big private planes, which is why it is a kernel of its own: inside
 * couple_verify it gave every launch of that kernel a 17.8 KB/lane scratch request (queue
 * OUT_OF_RESOURCES on the first sampled tree). */
__global__ void couple_verify_wide(const int32_t* idx, const float* values, const uint32_t* params,
                                   const uint32_t* state, int32_t* token, int64_t M, int64_t n_cand,
                                   int64_t vocab, int64_t ld0, int64_t ld1, int64_t ld2, int64_t ld3,
                                   int64_t ld4) {
    RadMtpCoupleVerifyCall c{ idx, values, params, state, token, M, n_cand, vocab,
                              { ld0, ld1, ld2, ld3, ld4 } };
    if (blockIdx.x < M && threadIdx.x == 0) rad_mtp_couple_verify_row<true>(c, blockIdx.x);
}

__global__ void couple_verify(const int32_t* idx, const float* values, const uint32_t* params,
                              const uint32_t* state, int32_t* token, int64_t M, int64_t n_cand,
                              int64_t vocab, int64_t ld0, int64_t ld1, int64_t ld2, int64_t ld3,
                              int64_t ld4) {
    RadMtpCoupleVerifyCall c{ idx, values, params, state, token, M, n_cand, vocab,
                              { ld0, ld1, ld2, ld3, ld4 } };
    const int64_t row = blockIdx.x;
    if (row >= M) return;
    /* n_cand <= kMaxCand here: the launch sends wider planes to couple_verify_wide. */
    /* `weight` is P's unnormalised FP64 mass per live candidate and `mass` its residual (P-Q)+,
     * both indexed by the candidate's place in `lj`. */
    __shared__ double weight[kMaxCand], mass[kMaxCand], pq[8], sum;
    __shared__ short  lj[kMaxCand];
    __shared__ float  top;
    __shared__ int    eligible, nl, valid, first[8], ix, dominated;
    const auto& s = *reinterpret_cast<const RadMtpCoupleState*>(c.state + row * c.ld[3]);
    const auto& p = *reinterpret_cast<const RadSampleParams*>(c.params + row * c.ld[2]);
    const int32_t* ids = c.idx + row * c.ld[0];
    const float*   val = c.val + row * c.ld[1];
    /* The same eligibility gate as rad_mtp_couple_verify_row: an unbound row keeps the ordinary
     * token. */
    if (threadIdx.x == 0)
        eligible = s.status == 1 && !(p.flags & RAD_SP_GREEDY) && s.draw.seed == p.seed &&
                   s.pos == p.pos && rad_mtp_couple_state_valid(s, c.vocab);
    __syncthreads();
    if (!eligible) return;
    /* A chain row of a block-verified draft (--draft-verify block). */
    if (s.blk >= 1) {
#ifdef RAD_BLOCK_REPEAT
        /* microbenchmark build: the block is decided RAD_BLOCK_REPEAT times back to back (idempotent) */
        for (int rep = 0; rep < RAD_BLOCK_REPEAT; ++rep) { block_verify(c, row); __syncthreads(); }
#else
        block_verify(c, row);
#endif
        return;
    }
    /* A RheoSampling node (abi/rad_mtp_rrs.h): one thread runs the shared serial helper, which is
     * the host reference itself. Only a node with a RheoSampling scheme takes it. */
    if (s.scheme != 0) {
        if (threadIdx.x == 0) rad_mtp_couple_verify_row<false>(c, row);
        return;
    }
    couple_live(ids, val, c.n_cand, c.vocab, lj, &nl, &valid, &top);
    if (!valid || top == -INFINITY) return;
    const int L = nl;
    for (int i = threadIdx.x; i < L; i += blockDim.x) weight[i] = ::exp((double)val[lj[i]] - top);
    /* A tree node with several drafted children (abi/rad_mtp_rrs.h, recursive rejection sampling):
     * the workgroup has compacted the live candidates and computed every weight and every
     * candidate's slot in the node's support; one lane then runs the host reference's own decision
     * (rad_mtp_rrs_decide_over) over those stored values, in the same order, so the token is the
     * host's bit for bit. */
    if (s.nkids >= 2) {
        __shared__ unsigned char kslot[kMaxCand];
        __shared__ double        pq_x[8], sum_x;
        for (int i = threadIdx.x; i < L; i += blockDim.x) {
            const int32_t id = ids[lj[i]];
            int k = 0;
            while (k < s.count && id != s.ids[k]) ++k;
            kslot[i] = (unsigned char)k;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int32_t out = -1;
            if (rad_mtp_rrs_decide_over(s, LiveCands{ ids, lj, weight, kslot, L }, p.seed, p.pos, &out,
                                        pq_x, &sum_x)) {
                c.token[row * c.ld[4]] = out;
            }
        }
        return;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        sum = 0;
        for (int k = 0; k < s.count; ++k) { pq[k] = 0; first[k] = -1; }
        for (int i = 0; i < L; ++i) {
            const int32_t id = ids[lj[i]];
            const double w = weight[i];
            sum += w;
            for (int k = 0; k < s.count; ++k)
                if (id == s.ids[k]) {
                    if (first[k] < 0) first[k] = i;
                    pq[k] += w;
                    break;
                }
        }
        if (!(sum > 0) || !rad_spechub_finite(sum)) eligible = 0;
        if (eligible) {
            ix = -1;
            dominated = true;
            for (int k = 0; k < s.count; ++k) {
                pq[k] /= sum;
                dominated = dominated && pq[k] >= s.q[k];
                if (s.ids[k] == s.x) ix = k;
            }
            if (ix < 0) eligible = 0;
        }
    }
    __syncthreads();
    if (!eligible) return;
    /* Residual per live slot: a Q token contributes max(0, P-Q) once, at its first occurrence;
     * every other candidate contributes its own P. */
    for (int i = threadIdx.x; i < L; i += blockDim.x) {
        const int32_t id = ids[lj[i]];
        double m = 0;
        bool in_q = false;
        for (int k = 0; k < s.count; ++k)
            if (s.ids[k] == id) {
                in_q = true;
                if (first[k] == i) m = rad_spechub_positive(pq[k] - s.q[k]);
                break;
            }
        if (!in_q) m = weight[i] / sum;
        mass[i] = m;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    /* One thread, ascending candidate order: parallelising these sums would reorder them and
     * break bit-exactness with the host reference. */
    double resid = 0;
    for (int i = 0; i < L; ++i) resid += mass[i];
    int32_t out = -1;
    const double u_resid = rad_spechub_uniform(p.seed, p.pos, RAD_SAMPLE_KEY_RESID);
    if (!dominated && !(resid > 0 && rad_spechub_finite(resid))) {
        double cdf = u_resid * sum;
        for (int i = 0; i < L; ++i) {
            const double w = weight[i];
            if (w > 0) { out = ids[lj[i]]; cdf -= w; if (cdf < 0) break; }
        }
    } else if (pq[ix] >= s.q[ix] ||
               rad_spechub_uniform(p.seed, p.pos, RAD_SAMPLE_KEY_ACCEPT) * s.q[ix] < pq[ix]) {
        out = s.x;
    } else {
        double cdf = u_resid * resid;
        for (int i = 0; i < L; ++i) {
            const int32_t id = ids[lj[i]];
            if (id == s.x) continue;
            const double m = mass[i];
            if (m > 0) { out = id; cdf -= m; if (cdf < 0) break; }
        }
    }
    c.token[row * c.ld[4]] = out;
}

/* DFlash2's coupled walk: one workgroup a sequence. Per step the workgroup forms the
 * candidate-independent half `sa` of the edge, then thread (c, j) computes part j of candidate c's
 * edge (rad_dflash_edge_part, the host's serial fmaf chain over the same slice), and thread 0 adds
 * the parts in order, takes the step (rad_dflash_couple_step) and publishes the token as the next
 * predecessor. The arithmetic and its order are the host reference's, so tokens and states match
 * rad_ref_dflash_couple bit for bit. */
constexpr int kDfThreads = RAD_DFLASH_COUPLE_MAX_K * RAD_DFLASH_EDGE_PARTS;   /* 256 */

__global__ __launch_bounds__(kDfThreads) void dflash_couple(
    const int32_t* cand, const float* unary, const uint16_t* hp, const int32_t* anchor,
    const uint16_t* pred, const uint16_t* succ, const int32_t* card, const uint32_t* draw,
    int32_t* tokens, uint32_t* state, int64_t steps, int64_t K, int64_t R, int64_t vocab,
    int64_t astr, int64_t ld_cand, int64_t ld_unary, int64_t ld_hp, int64_t ld_draw,
    int64_t ld_tok, int64_t st_round, int64_t st_row, float q_temp) {
    __shared__ float   sa[RAD_DFLASH_COUPLE_MAX_R];
    __shared__ float   part[kDfThreads];
    __shared__ float   sc[RAD_DFLASH_COUPLE_MAX_K];
    __shared__ int32_t sid[RAD_DFLASH_COUPLE_MAX_K];
    __shared__ int32_t spid;
    /* The step's Q in the making: its live (id, score / q_temp) in selection order, their count
     * (-1: no Q), the top score, the FP64 weights, their sum and the normalised q. */
    __shared__ int32_t lid[8], slot_id[8];
    __shared__ float   lv[8], slot_v[8];
    __shared__ int     sn;
    __shared__ float   stop;
    __shared__ double  w[8], wq[8], ssum;
    __shared__ RadMtpCoupleState sst;
    const int64_t s   = blockIdx.x;
    const int     tid = threadIdx.x;
    const RadMtpCoupleDraw d = *reinterpret_cast<const RadMtpCoupleDraw*>(draw + s * ld_draw);
    int32_t* tok = tokens + s * ld_tok;
    uint32_t* st0 = state + s * st_row;
    if (d.mode != 1) {
        /* A greedy request: dflash_select's walk stands, and each round records it (status 0). */
        if (tid == 0)
            for (int64_t l = 0; l < steps; ++l)
                (void)rad_dflash_couple_step(sc, sid, (int)K, q_temp, d, false, (int)l + 1, vocab,
                                             tok[l],
                                             reinterpret_cast<RadMtpCoupleState*>(st0 + l * st_round));
        return;
    }
    const bool card_ok = d.card == card[s];
    if (tid == 0) spid = anchor[s * astr];
    __syncthreads();
    const int c = tid / RAD_DFLASH_EDGE_PARTS, j = tid % RAD_DFLASH_EDGE_PARTS;
    for (int64_t l = 0; l < steps; ++l) {
        const int64_t row = s * steps + l;
        /* What does not depend on the predecessor is loaded first, so its latency overlaps the
         * predecessor's codebook row: the candidate, its successor slice and its unary score. */
        const int32_t id = c < K ? cand[row * ld_cand + c] : -1;
        const bool    id_ok = id >= 0 && id < vocab;
        const bool    wide = R == 32 * RAD_DFLASH_EDGE_PARTS;
        uint4 v[4] = {};
        if (id_ok && wide) {
            /* DFlash2's rank 256: the slice's 32 values in four 16-byte loads issued together. A
             * load inside the fmaf chain would wait out one memory latency per element. */
            const uint4* src = reinterpret_cast<const uint4*>(succ + (int64_t)id * R + j * 32);
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = src[i];
        }
        const float un = tid < K ? unary[row * ld_unary + tid] : 0.0f;
        const int32_t pid = spid;
        const bool ok = pid >= 0 && pid < vocab;
        for (int64_t r = tid; r < R; r += kDfThreads)
            sa[r] = ok ? rad_dflash_bf2f(pred[(int64_t)pid * R + r]) *
                             (hp ? rad_dflash_bf2f(hp[row * ld_hp + r]) : 1.0f)
                       : 0.0f;
        __syncthreads();
        if (c < K) {
            float pv = 0.0f;
            if (id_ok) {
                if (wide) {
                    /* rad_dflash_edge_part's fmaf chain over the slice, in its order. */
                    const uint16_t* h = reinterpret_cast<const uint16_t*>(v);
                    const float* a = sa + j * 32;
#pragma unroll
                    for (int r = 0; r < 32; ++r) pv = fmaf(a[r], rad_dflash_bf2f(h[r]), pv);
                } else {
                    pv = rad_dflash_edge_part(sa, succ + (int64_t)id * R, R, j);
                }
            }
            part[tid] = pv;
            if (j == 0) sid[c] = id;
        }
        __syncthreads();
        /* Each candidate's score from its parts, in the host's order. */
        if (tid < K) {
            float acc = 0.0f;
            if (sid[tid] >= 0 && sid[tid] < vocab)
                for (int q = 0; q < RAD_DFLASH_EDGE_PARTS; ++q)
                    acc += part[tid * RAD_DFLASH_EDGE_PARTS + q];
            sc[tid] = un + acc;
        }
        if (tid < 8) slot_id[tid] = -2;
        __syncthreads();
        RadMtpCoupleState* so = reinterpret_cast<RadMtpCoupleState*>(st0 + l * st_round);
        /* rad_dflash_couple_step, spread over the workgroup: one lane walking the selection and
         * build_q's FP64 work serially waits out an LDS or FP64 latency per element. Each operation
         * and its order are the helper's, so the bits are too.
         *
         * The eight: candidate k's place is the number of eligible candidates ahead of it (a higher
         * score, or an equal one at a lower index), which is exactly the order the helper's repeated
         * first-maximum scan takes them in. */
        if (tid < K) {
            const int32_t id = sid[tid];
            const float   v  = sc[tid];
            if (id >= 0 && id < vocab && v == v) {
                int rank = 0;
                for (int k = 0; k < K; ++k) {
                    const int32_t ik = sid[k];
                    const float   vk = sc[k];
                    rank += (ik >= 0 && ik < vocab && vk == vk && (vk > v || (vk == v && k < tid)));
                }
                if (rank < 8) { slot_id[rank] = id; slot_v[rank] = v / q_temp; }
            }
        }
        __syncthreads();
        if (tid == 0) {
            sn = -1;
            if (card_ok) {
                int live = 0;
                bool bad = false;
                for (int n = 0; n < 8 && slot_id[n] != -2; ++n) {
                    const float v = slot_v[n];
                    /* build_q: +inf refuses the whole Q, -inf carries no mass, a repeat refuses. */
                    if (v == INFINITY) { bad = true; break; }
                    if (v == -INFINITY) continue;
                    for (int i = 0; i < live; ++i) bad = bad || lid[i] == slot_id[n];
                    if (bad) break;
                    lid[live] = slot_id[n];
                    lv[live++] = v;
                }
                if (!bad && live > 0) {
                    float top = lv[0];
                    for (int i = 1; i < live; ++i)
                        if (lv[i] > top) top = lv[i];
                    stop = top;
                    sn = live;
                }
            }
        }
        __syncthreads();
        if (tid < 8 && tid < sn) w[tid] = ::exp((double)lv[tid] - stop);
        __syncthreads();
        if (tid == 0 && sn > 0) {
            double sum = 0;
            for (int i = 0; i < sn; ++i) sum += w[i];
            ssum = sum;
        }
        __syncthreads();
        if (tid < 8 && tid < sn) wq[tid] = w[tid] / ssum;
        __syncthreads();
        /* The state is built in LDS: a private struct indexed by a runtime count lives in scratch,
         * and every access to it is a memory round trip. */
        if (tid == 0) {
            RadMtpCoupleState& s = sst;
            s.draw = d;
            s.pos = 0;
            s.round = (int)l + 1;
            s.status = -1;
            s.count = 0;
            s.nkids = 0;   /* the walk's position is one draw: the chain's wire */
            s.scheme = 0;
            for (int i = 0; i < 8; ++i) { s.ids[i] = -1; s.q[i] = 0; }
            for (int i = 0; i < sn; ++i)
                if (wq[i] > 0) { s.ids[s.count] = lid[i]; s.q[s.count++] = wq[i]; }
            if (s.count > 0) {
                const double u = rad_spechub_uniform(d.seed, d.base,
                                                     RAD_MTP_COUPLE_KEY_DRAFT + (uint64_t)(l + 1));
                s.x = s.ids[rad_mtp_couple_pick(s, u)];
                s.status = 1;
            } else {
                s.count = 0;
                for (int i = 0; i < 8; ++i) { s.ids[i] = -1; s.q[i] = 0; }
                int   best = 0;
                float bv = -INFINITY;
                for (int k = 0; k < K; ++k)
                    if (sc[k] > bv) { bv = sc[k]; best = k; }
                s.x = sid[best];
            }
            tok[l] = s.x;
            spid = s.x;
        }
        __syncthreads();
        /* One word a lane out to the state plane. */
        if (tid < (int)RAD_MTP_COUPLE_STATE_WORDS)
            reinterpret_cast<uint32_t*>(so)[tid] = reinterpret_cast<const uint32_t*>(&sst)[tid];
        /* The next step reads spid and overwrites sa, part and the Q scratch (the copy above reads
         * sst, which the next step writes only after its own barriers). */
    }
}

}  /* namespace */

int rad_mine_dflash_couple_launch(const RadArgs* a, RadStream stream) {
    RadDflashCoupleCall c{};
    const int rc = rad_dflash_couple_args(a, &c);
    if (rc != RAD_OK) return rc;
    return R4D_EMIT(stream, dflash_couple, dim3((unsigned)c.M), dim3(kDfThreads), 0,
                    c.cand, c.unary, c.hp, c.anchor, c.pred, c.succ, c.card, c.draw, c.tokens,
                    c.state, c.steps, c.K, c.R, c.vocab, c.anchor_stride, c.ld_cand, c.ld_unary,
                    c.ld_hp, c.ld_draw, c.ld_tok, c.st_round, c.st_row, c.q_temp);
}

int rad_mine_mtp_couple_propose_launch(const RadArgs* a, RadStream stream) {
    RadMtpCoupleProposeCall c{};
    const int rc = rad_mtp_couple_propose_args(a, &c);
    if (rc != RAD_OK) return rc;
    return R4D_EMIT(stream, propose, dim3((unsigned)((c.M + 63) / 64)), dim3(64), 0, c);
}

/* ------------------------------------------------------------------ traversal verification
 *
 * abi/rad_mtp_traversal.h, one workgroup a row; only a request's root row (topo parent -1) does
 * work, and it decides the whole request. Its four waves each take every fourth row of the request
 * and need nothing from each other until the tree is complete:
 *
 *   per row, one wave:  lane 0 reads the row's state (the gate, the canonical draft side) into LDS
 *       and hands the wave what it needs of it in registers; a lane a later row finds the node's
 *       child rows in the step's topology; the wave reads the row's candidate
 *       plane a pass of 32 slots at a time, one slot a lane (a top_k 20 plane is one live pass): the
 *       largest value and the validity, reduced over the wave in registers, then the integer masses
 *       (rad_mtp_trav_mass) of the passes that have a live slot, summed in all and on the node's
 *       support slots and reduced the same way -- integer sums, so the order is free and the totals
 *       are the host's serial ones bit for bit; lanes 0..7 turn them into the node's law, mass /
 *       total, 
 *   one barrier, the kernel's only one;
 *   wave 0, lane 0: the walk itself (rad_mtp_trav_walk_tree, the host's code over
 *       the tree in LDS: one FP64 division to enter a node, two and a few dozen multiply-adds when a
 *       child is rejected), the path's
 *       tokens and the final token's slot or off-support unit; then the wave finds the off-support
 *       candidate holding that unit (a prefix over the lanes of each pass, in registers).
 *
 * Lanes talk to each other in two ways only: through LDS ACROSS the barrier, and through registers
 * (v_readlane, DPP) inside a wave. A lane never reads from LDS what another lane wrote since the last
 * barrier: the hardware would order the two accesses of one wave, but the compiler, which sees a
 * thread at a time, may move the read before the write. No candidate plane is copied, no lane walks a
 * plane alone at any width, and the kernel has no scratch segment (nothing private is indexed at run
 * time). Tokens and paths are the host reference's bit for bit (rad_mtp_trav_verify_root). */
namespace {

template <int N>
__device__ __forceinline__ uint32_t tv_xor_lane(uint32_t v) {
    /* lane ^ N's value on DPP (row_xmask; 16 crosses the wave's two rows): every lane must be active */
    if constexpr (N < 16)
        return (uint32_t)__builtin_amdgcn_update_dpp(0, (int)v, 0x160 | N, 0xf, 0xf, true);
    else
        return (uint32_t)__builtin_amdgcn_permlanex16((int)v, (int)v, 0x76543210u, 0xFEDCBA98u, false, false);
}
template <int N>
__device__ __forceinline__ uint64_t tv_xor_add(uint64_t v) {
    return v + (((uint64_t)tv_xor_lane<N>((uint32_t)(v >> 32)) << 32) | tv_xor_lane<N>((uint32_t)v));
}
/* The wave's sum of a 64-bit integer, in every lane (integers: the order is free). */
__device__ __forceinline__ uint64_t tv_wave_sum(uint64_t v) {
    v = tv_xor_add<16>(v); v = tv_xor_add<8>(v); v = tv_xor_add<4>(v);
    v = tv_xor_add<2>(v);  v = tv_xor_add<1>(v);
    return v;
}
__device__ __forceinline__ uint32_t tv_wave_or(uint32_t v) {
    v |= tv_xor_lane<16>(v); v |= tv_xor_lane<8>(v); v |= tv_xor_lane<4>(v);
    v |= tv_xor_lane<2>(v);  v |= tv_xor_lane<1>(v);
    return v;
}
__device__ __forceinline__ float tv_fmax_lane(float a, uint32_t b) { return fmaxf(a, __uint_as_float(b)); }
__device__ __forceinline__ float tv_wave_max(float v) {
    v = tv_fmax_lane(v, tv_xor_lane<16>(__float_as_uint(v)));
    v = tv_fmax_lane(v, tv_xor_lane<8>(__float_as_uint(v)));
    v = tv_fmax_lane(v, tv_xor_lane<4>(__float_as_uint(v)));
    v = tv_fmax_lane(v, tv_xor_lane<2>(__float_as_uint(v)));
    v = tv_fmax_lane(v, tv_xor_lane<1>(__float_as_uint(v)));
    return v;
}
/* Lane 0's value in every lane of the wave (v_readlane): how one lane hands the wave a scalar
 * without LDS. Every lane must be active. */
__device__ __forceinline__ int32_t tv_lane0(int32_t v) { return __builtin_amdgcn_readlane(v, 0); }
__device__ __forceinline__ float tv_lane0f(float v) {
    return __uint_as_float((uint32_t)__builtin_amdgcn_readlane((int)__float_as_uint(v), 0));
}
/* A wave's inclusive prefix sum over its lanes (row_shr 1, 2, 4, 8 inside each row of sixteen, then
 * the first row's total into the second). Every lane must be active; sums below 2^32. */
__device__ __forceinline__ uint32_t tv_wave_prefix(uint32_t x, int lane) {
    x += (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x111, 0xF, 0xF, true);
    x += (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x112, 0xF, 0xF, true);
    x += (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x114, 0xF, 0xF, true);
    x += (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x118, 0xF, 0xF, true);
    const uint32_t lo = (uint32_t)__builtin_amdgcn_readlane((int)x, 15);
    return lane >= 16 ? x + lo : x;
}

constexpr int kTravRows  = RAD_MTP_TRAV_MAX_ROWS;
constexpr int kTravWaves = kThreads / 32;
constexpr int kTravPer   = 8;   /* passes a batch */

/* The op's operands as the kernel takes them: 32-bit geometry (the launch refuses anything wider),
 * which is half the scalar registers of 64-bit strides. */
struct TravArgs {
    const int32_t*  idx;
    const float*    val;
    const uint32_t* params;
    const uint32_t* state;
    int32_t*        token;
    const int32_t*  topo;
    uint32_t M, n_cand, vocab, ld0, ld1, ld2, ld3, ld4, ld_topo;
};

/* Eight slots of a row's plane, in registers. Lane l's slots are l, l + 32, l + 64, ... (pass p is
 * slots [32 p, 32 p + 32), one a lane), so the plane's live head -- 20 slots at top_k 20 -- is one
 * pass of the wave whatever its width; a batch is eight passes, sixteen loads issued together (a loop
 * that loaded a pass at a time would wait out a memory latency per pass). */
struct TvBatch {
    int32_t id[kTravPer];
    float   v[kTravPer];
};
/* Passes p0 .. p0 + 8 of lane `lane` of a row of n slots; holes past the row's end. `full`: the batch
 * lies inside the row for every lane, so the loads are unconditional. The ids first: a batch of holes
 * (every batch but the first at top_k 20) is known from them alone, and its values are not read. Returns
 * whether any lane of the wave has an id in the batch. */
__device__ __forceinline__ bool tv_load(const int32_t* ids, const float* val, int lane, uint32_t p0,
                                        uint32_t n, bool full, TvBatch* c) {
    const uint32_t j = (uint32_t)lane + 32u * p0;
    bool on = false;
    if (full) {
#pragma unroll
        for (int i = 0; i < kTravPer; ++i) { c->id[i] = ids[j + 32u * i]; on = on || c->id[i] >= 0; }
    } else {
#pragma unroll
        for (int i = 0; i < kTravPer; ++i) {
            c->id[i] = -1;
            if (j + 32u * i < n) c->id[i] = ids[j + 32u * i];
            on = on || c->id[i] >= 0;
        }
    }
#pragma unroll
    for (int i = 0; i < kTravPer; ++i) c->v[i] = -INFINITY;
    if (!__ballot(on)) return false;
    if (full) {
#pragma unroll
        for (int i = 0; i < kTravPer; ++i) c->v[i] = val[j + 32u * i];
    } else {
#pragma unroll
        for (int i = 0; i < kTravPer; ++i)
            if (j + 32u * i < n) c->v[i] = val[j + 32u * i];
    }
    return true;
}

/* One request, by its root row's workgroup. Called by every thread. */
__device__ __forceinline__ void trav_request(const TravArgs& c, uint32_t row) {
    __shared__ RadTravTree T;
    __shared__ int32_t  s_tok[kTravRows];
    __shared__ int      s_path[RAD_MTP_TRAV_MAX_DEPTH + 1];   /* lane 0's alone */
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const uint32_t avail = c.M - row < (uint32_t)kTravRows ? c.M - row : (uint32_t)kTravRows;
    const uint32_t nc = c.n_cand;
    const uint32_t npass = (nc + 31u) / 32u;
    const int qb = rad_mtp_trav_qbits(nc);

    /* The rows of the request: the root and the rows after it that have a parent (a lane a row). */
    uint64_t has = 1;
    for (uint32_t u0 = 0; u0 < avail; u0 += 32) {
        const uint32_t u = u0 + (uint32_t)lane;
        bool in = false;
        if (u >= 1 && u < avail) in = c.topo[(uint64_t)(row + u) * c.ld_topo] >= 0;
        has |= (uint64_t)(uint32_t)__ballot(in) << u0;
    }
    const int n = ~has ? (int)__builtin_ctzll(~has) : 64;

    /* This wave's rows: the node, the row's law. */
    for (int v = wave; v < n; v += kTravWaves) {
        const uint64_t rr = (uint64_t)row + (uint32_t)v;
        int elig = 0, nkv = 0;
        int32_t cid[8];
        for (int k = 0; k < 8; ++k) cid[k] = -1;
        if (lane == 0) {
            s_tok[v] = c.topo[rr * c.ld_topo + 1];
            /* the state by value: its words are loaded together and stay in registers (nothing below
             * indexes it at run time), instead of one load a check */
            const RadMtpCoupleState s = *reinterpret_cast<const RadMtpCoupleState*>(c.state + rr * c.ld3);
            const RadSampleParams p = *reinterpret_cast<const RadSampleParams*>(c.params + rr * c.ld2);
            double q[8];
            int nk = 0, nt = -1, tail = 0, swap = 0;
            elig = rad_mtp_trav_eligible(s, p, c.vocab) &&
                   rad_mtp_trav_canon(s, 0, cid, q, &nk, &nt, &tail, &swap);
            for (int k = 0; k < 8; ++k) {
                if (!elig) cid[k] = -1;
                T.ids[v][k] = cid[k];
                T.qr[v][k] = elig ? q[k] : 0.0;
            }
            nkv = elig ? nk : 0;
            T.nk[v] = nkv;
            T.nt[v] = nt;
            T.tail[v] = tail;
            T.ci[v] = 0;
        }
        /* lane 0's findings, to the wave in registers */
        elig = tv_lane0(elig);
        nkv = tv_lane0(nkv);
        int32_t sid[8];
        for (int k = 0; k < 8; ++k) sid[k] = tv_lane0(cid[k]);
        /* the node's child rows: a lane a later row of the request, the first row under this node
         * that carries the slot's token */
        int chrow[8];
        for (int k = 0; k < 8; ++k) chrow[k] = -1;
        if (nkv > 0) {
            for (int u0 = v + 1; u0 < n; u0 += 32) {
                const int u = u0 + lane;
                int32_t upar = -2, utok = -1;
                if (u < n) {
                    upar = c.topo[((uint64_t)row + (uint32_t)u) * c.ld_topo];
                    utok = c.topo[((uint64_t)row + (uint32_t)u) * c.ld_topo + 1];
                }
                const bool under = u < n && upar == (int32_t)(row + (uint32_t)v);
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const uint32_t mk = (uint32_t)__ballot(under && utok == sid[k]);
                    if (mk && chrow[k] < 0 && k < nkv) chrow[k] = u0 + (int)__builtin_ctz(mk);
                }
            }
        }
        if (lane == 0)
            for (int k = 0; k < 8; ++k) T.child[v][k] = chrow[k];
        const int32_t* ids = c.idx + rr * c.ld0;
        const float*   val = c.val + rr * c.ld1;
        float top = -INFINITY;
        uint32_t bad = 0;
        uint64_t sum = 0, pk[8];
        for (int k = 0; k < 8; ++k) pk[k] = 0;
        if (elig) {
            /* the largest value and the validity */
            for (uint32_t p0 = 0; p0 < npass; p0 += kTravPer) {
                TvBatch x;
                if (!tv_load(ids, val, lane, p0, nc, 32u * (p0 + kTravPer) <= nc, &x)) continue;
#pragma unroll
                for (int i = 0; i < kTravPer; ++i) {
                    const bool on = x.id[i] >= 0;
                    const bool inval = on && ((uint32_t)x.id[i] >= c.vocab || x.v[i] != x.v[i] || x.v[i] == INFINITY);
                    bad |= inval ? 1u : 0u;
                    top = on && !inval && x.v[i] > top ? x.v[i] : top;
                }
            }
            top = tv_wave_max(top);
            bad = tv_wave_or(bad);
            /* the masses: in all, and on the support slots */
            if (!bad && top != -INFINITY) {
                for (uint32_t p0 = 0; p0 < npass; p0 += kTravPer) {
                    TvBatch x;
                    /* a batch of holes (top_k 20: all but the first) adds nothing */
                    if (!tv_load(ids, val, lane, p0, nc, 32u * (p0 + kTravPer) <= nc, &x)) continue;
#pragma unroll
                    for (int i = 0; i < kTravPer; ++i) {
                        const bool live = x.id[i] >= 0 && x.v[i] != -INFINITY;
                        if (!__ballot(live)) continue;   /* nor does a pass with none (top_k 20: all but one) */
                        const uint64_t m = live ? rad_mtp_trav_mass(x.v[i], top, qb) : 0;
                        sum += m;
                        /* the support's ids sit in a few slots of the plane: a pass none of them is in
                         * costs the eight compares */
                        bool eq[8], anyeq = false;
#pragma unroll
                        for (int k = 0; k < 8; ++k) { eq[k] = x.id[i] == sid[k]; anyeq = anyeq || eq[k]; }
                        if (__ballot(anyeq)) {
#pragma unroll
                            for (int k = 0; k < 8; ++k) pk[k] += eq[k] ? m : 0;
                        }
                    }
                }
            }
            sum = tv_wave_sum(sum);
#pragma unroll
            for (int k = 0; k < 8; ++k) pk[k] = tv_wave_sum(pk[k]);
        }
        /* the node's law (rad_mtp_trav_set_law's values): lane k the support slot k, lane 0 the
         * scalars; every word of the tree is written once, by one lane */
        if (lane < 8) {
            const bool valid = elig && !bad && top != -INFINITY && sum > 0;
            uint64_t on = 0, mine = 0;
            for (int k = 0; k < 8; ++k) { on += pk[k]; mine = k == lane ? pk[k] : mine; }
            const double dsum = valid ? rad_mtp_trav_u2d(sum) : 1.0;
            const double rk = valid ? rad_mtp_trav_u2d(mine) / dsum : 0.0;
            T.m[v][lane] = rk;
            if (lane == 0) {
                const uint64_t off = sum - on;
                T.moff[v] = valid ? rad_mtp_trav_u2d(off) / dsum : 0.0;
                T.off[v] = valid ? off : 0;
                T.top[v] = top;
                T.ok[v] = valid;
            }
        }
    }
    __syncthreads();   /* the tree is complete */
    if (wave != 0) return;

    /* Lane 0: the walk (the host's code over the tree in LDS), the path's tokens. */
    int kind = 0, fnode = 0;
    uint32_t want_lo = 0, want_hi = 0;
    float ftop = 0.0f;
    int32_t fid[8];
    for (int k = 0; k < 8; ++k) fid[k] = -1;
    if (lane == 0) {
        const auto& p0 = *reinterpret_cast<const RadSampleParams*>(c.params + (uint64_t)row * c.ld2);
        const uint64_t seed = p0.seed, pos0 = p0.pos;
        int np = 0;
        if (rad_mtp_trav_walk_tree(T, seed, pos0, s_path, &np)) {
            const int f = s_path[np - 1];
            for (int i = 0; i + 1 < np; ++i)
                c.token[((uint64_t)row + (uint32_t)s_path[i]) * c.ld4] = s_tok[s_path[i + 1]];
            if (T.ok[f]) {
                const RadTravPick pk = rad_mtp_trav_final_pick(
                    T, f, rad_spechub_uniform(seed, pos0 + (uint64_t)(np - 1), RAD_SAMPLE_KEY_RESID));
                if (pk.kind == 1) c.token[((uint64_t)row + (uint32_t)f) * c.ld4] = T.ids[f][pk.slot];
                if (pk.kind == 2) {
                    kind = 2;
                    fnode = f;
                    want_lo = (uint32_t)pk.t;
                    want_hi = (uint32_t)(pk.t >> 32);
                    ftop = T.top[f];
                    for (int k = 0; k < 8; ++k) fid[k] = T.ids[f][k];
                }
            }
        }
    }
    if (tv_lane0(kind) != 2) return;

    /* The final token off the support: the candidate holding unit `want`, candidates in slot order
     * (pass by pass, the lanes of a pass in order). Lane 0's finding goes to the wave in registers; a
     * mass is split in two limbs so its prefix over the lanes is two 32-bit DPP scans. */
    {
        const int f = tv_lane0(fnode);
        const uint64_t want = ((uint64_t)(uint32_t)tv_lane0((int32_t)want_hi) << 32) | (uint32_t)tv_lane0((int32_t)want_lo);
        const float top = tv_lane0f(ftop);
        int32_t sid[8];
        for (int k = 0; k < 8; ++k) sid[k] = tv_lane0(fid[k]);
        const uint64_t rr = (uint64_t)row + (uint32_t)f;
        const int32_t* ids = c.idx + rr * c.ld0;
        const float*   val = c.val + rr * c.ld1;
        uint64_t before = 0;   /* the off-support mass of the passes already walked: the same in every lane */
        bool done = false;
        for (uint32_t p0 = 0; p0 < npass && !done; p0 += kTravPer) {
            TvBatch x;
            if (!tv_load(ids, val, lane, p0, nc, 32u * (p0 + kTravPer) <= nc, &x)) continue;
#pragma unroll
            for (int i = 0; i < kTravPer; ++i) {
                bool off = x.id[i] >= 0 && x.v[i] != -INFINITY;
#pragma unroll
                for (int k = 0; k < 8; ++k) off = off && x.id[i] != sid[k];
                if (done || !__ballot(off)) continue;
                const uint64_t m = off ? rad_mtp_trav_mass(x.v[i], top, qb) : 0;
                /* a mass is at most 2^41: limbs of 24 and 17 bits, whose prefixes over 32 lanes stay
                 * below 2^32 */
                const uint32_t plo = tv_wave_prefix((uint32_t)m & 0xffffffu, lane);
                const uint32_t phi = tv_wave_prefix((uint32_t)(m >> 24), lane);
                const uint64_t incl = ((uint64_t)phi << 24) + plo;
                const uint64_t all = ((uint64_t)(uint32_t)__builtin_amdgcn_readlane((int)phi, 31) << 24) +
                                     (uint32_t)__builtin_amdgcn_readlane((int)plo, 31);
                const uint64_t base = before + incl - m;
                const bool mine = m > 0 && want >= base && want - base < m;
                if (mine) c.token[rr * c.ld4] = x.id[i];
                done = __ballot(mine) != 0;
                before += all;
            }
        }
    }
}

}  /* namespace */

__global__ __launch_bounds__(kThreads) void trav_verify(
    const int32_t* idx, const float* values, const uint32_t* params, const uint32_t* state,
    int32_t* token, const int32_t* topo, uint32_t M, uint32_t n_cand, uint32_t vocab, uint32_t ld0,
    uint32_t ld1, uint32_t ld2, uint32_t ld3, uint32_t ld4, uint32_t ld_topo) {
    const uint32_t row = blockIdx.x;
    if (row >= M) return;
    if (topo[(uint64_t)row * ld_topo] >= 0) return;   /* not a request's root */
    const TravArgs c{ idx, values, params, state, token, topo, M, n_cand, vocab, ld0, ld1, ld2, ld3, ld4, ld_topo };
#ifdef RAD_TRAV_REPEAT
    /* microbenchmark build: the request is decided RAD_TRAV_REPEAT times back to back (idempotent) */
    for (int rep = 0; rep < RAD_TRAV_REPEAT; ++rep) { trav_request(c, row); __syncthreads(); }
#else
    trav_request(c, row);
#endif
}

int rad_mine_mtp_trav_verify_launch(const RadArgs* a, RadStream stream) {
    RadMtpTravCall c{};
    const int rc = rad_mtp_trav_verify_args(a, &c);
    if (rc != RAD_OK) return rc;
    /* the kernel's geometry is 32-bit */
    for (int i = 0; i < 5; ++i)
        if (c.v.ld[i] < 0 || c.v.ld[i] > INT32_MAX) return RAD_E_UNSUPPORTED;
    if (c.ld_topo < 0 || c.ld_topo > INT32_MAX) return RAD_E_UNSUPPORTED;
    return R4D_EMIT(stream, trav_verify, dim3((unsigned)c.v.M), dim3(kThreads), 0,
                    c.v.idx, c.v.val, c.v.params, c.v.state, c.v.token, c.topo, c.v.M, c.v.n_cand,
                    c.v.vocab, c.v.ld[0], c.v.ld[1], c.v.ld[2], c.v.ld[3], c.v.ld[4], c.ld_topo);
}

int rad_mine_mtp_couple_verify_launch(const RadArgs* a, RadStream stream) {
    RadMtpCoupleVerifyCall c{};
    const int rc = rad_mtp_couple_verify_args(a, &c);
    if (rc != RAD_OK) return rc;
    /* Separate pointer arguments let graph replay rebind each operand address independently: a
     * struct on the host would freeze the addresses it was built with. */
    if (c.n_cand > kMaxCand)
        return R4D_EMIT(stream, couple_verify_wide, dim3((unsigned)c.M), dim3(1), 0,
                        c.idx, c.val, c.params, c.state, c.token, c.M, c.n_cand, c.vocab,
                        c.ld[0], c.ld[1], c.ld[2], c.ld[3], c.ld[4]);
    return R4D_EMIT(stream, couple_verify, dim3((unsigned)c.M), dim3(kThreads), 0,
                    c.idx, c.val, c.params, c.state, c.token, c.M, c.n_cand, c.vocab,
                    c.ld[0], c.ld[1], c.ld[2], c.ld[3], c.ld[4]);
}
