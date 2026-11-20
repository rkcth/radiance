/* Derived from libr4d/r4d_logits_vocab.hip by the libhalo rename transform (2026-11-17),
 * then ported to gfx1151; see libhalo/README.md. */
// logits_gemm_vocab: the lm_head stored as width classes of the vocabulary's rows, each an IQR
// computed trellis (halo_iqr.h: tcq_code=mad, one f32 scale a row under a factor a 128-K block,
// fwht128) decoded where it is read.
//
// KL weighs a head row by its token's probability, so a vocabulary-weighted head gives the likely
// tokens' rows more bits than the rare ones'. The rows are sorted into classes of one code width
// each (whole 256-row trellis runs), every class a weight of its own whose rows each carry their
// vocabulary id, and the logits come out in vocabulary order: nothing after the head knows the rows
// were permuted. Stored at its id from the tile, a logit in a random row order is a write to a line
// of its own (+120 us of 760 at 8 tokens, +970 of 2350 at 64). So within a class the rows are in
// ascending id (the converter's order; a class's rows share a width, so their order is free), and
// above 32 tokens the tile stores at the ids directly, the large narrow class's rows then nearly
// contiguous; up to 32 it writes a staging plane in row order and a gather through `vrow` (each
// id's class and row) puts it in vocabulary order, cheaper there by the narrow classes' scattered
// rows (626 / 646 us against 668 / 706 at 8 / 16 tokens; 1618 against 1559 at 64). A rank's rows of
// each class hold only ids of that rank's share (an id outside it is dropped), so the head shards by
// rows as any lm_head does.
//
// The activation is the head's bf16 input rotated by the unnormalised Hadamard per 128 of K (the
// rotation the codes were quantised under) and held at sixteen bits: one f32 scale a (token, 128)
// over integers q = 128 hi + lo, |q| <= 127 * 128, hi and lo int8. The int8 grid the trunk's GEMMs
// use costs the head 2e-5 KL on its own (half the weighted head's quantisation cost); sixteen bits
// are below bf16's rounding. Each 128-K block's dot is then two int8 WMMA passes over the same
// decoded weight fragments, hi then lo, the first scaled by 128 in the accumulator: an exact
// integer per (row, token, block), folded once into f32.
//
// One launch covers every class: a block finds its class among the classes' row ranges (the
// widest first, so the narrow classes' cheaper blocks fill the tail) and runs that width's
// decoder. A class of one width alone would leave most of the card idle (the widest is a few
// thousand rows).

#include "halo.h"
#include "halo_args.h"
#include "halo_emit.h"
#include "halo_fp8_wmma.h"
#include "halo_fwht.h"
#include "halo_iqr.h"
#include "halo_iqr_dec.h"

#include <algorithm>

namespace {

using namespace halo_fp8;
using namespace halo_iqr;
using namespace halo_iqr_dev;

constexpr uint32_t kWave = 32, kBlock = 128, kFK = 256;
constexpr uint32_t kMaxClass = 4;
/* Tokens a pass of the tile: four 16-token tiles. More tokens are taken in passes, each reading
 * the head once (a prefill wants one row's logits; the all-position logits of a KL evaluation are
 * the case this serves). */
constexpr uint32_t kPassTok = 64;
/* |q| <= 127 * 128: hi = round(q / 128) stays in int8, lo = q - 128 hi in [-64, 64]. */
constexpr float kQMax = 16256.0f;

/* One width class, as the launch sees it. */
struct Cls {
    const uint8_t* W;       /* the IQR weight buffer (codes, then the computed code's A and B) */
    const uint8_t* S;       /* its scale buffer: the rows' f32 scales, then the blocks' factors */
    uint32_t N;             /* rows */
    uint32_t bits;          /* code width */
    const int32_t* ids;     /* the vocabulary id of each row, global, ascending */
    uint32_t blk0;          /* the first grid block of this class */
    uint32_t row0;          /* its first row in the classes' row order (the staging plane's) */
};
struct Head {
    Cls c[kMaxClass];       /* widest first */
    uint32_t n;             /* classes */
    uint32_t nkb;           /* K / 128 */
    int32_t off;            /* this rank's first vocabulary id */
    uint32_t n_vocab;       /* this rank's share */
};

/* ---- the activation: rotated, sixteen bits as two int8 planes ------------------------------
 *
 * A wave a (token, 128-K block): each lane's four values through halo_fwht_wave, the block's
 * absolute maximum, then q = rint(x * kQMax / amax). An all-zero block stores zeros under scale 0. */
__global__ __launch_bounds__(kWave) void vocab_quant_kernel(const uint16_t* __restrict__ X,
                                                            uint32_t x_ld, int8_t* __restrict__ XH,
                                                            int8_t* __restrict__ XL,
                                                            float* __restrict__ XS, uint32_t K) {
    const uint32_t m = blockIdx.x, kb = blockIdx.y, lane = threadIdx.x;
    const uint32_t nkb = K / kBlock;
    const uint2 raw = *(const uint2*)(X + (size_t)m * x_ld + kb * kBlock + 4u * lane);
    float v[4] = { __uint_as_float(raw.x << 16), __uint_as_float(raw.x & 0xFFFF0000u),
                   __uint_as_float(raw.y << 16), __uint_as_float(raw.y & 0xFFFF0000u) };
    halo_fwht_wave<128>(v, (int)lane);
    float a = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) a = fmaxf(a, fabsf(v[i]));
    a = halo_wave_max(a);
    const float inv = a > 0.0f ? kQMax / a : 0.0f;
    uint32_t hp = 0, lp = 0;
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        const int q = (int)fminf(fmaxf(rintf(v[i] * inv), -kQMax), kQMax);
        const int hi = (q + 64) >> 7, lo = q - (hi << 7);
        hp |= ((uint32_t)hi & 0xFFu) << (8u * i);
        lp |= ((uint32_t)lo & 0xFFu) << (8u * i);
    }
    const size_t o = (size_t)m * K + kb * kBlock + 4u * lane;
    *(uint32_t*)(XH + o) = hp;
    *(uint32_t*)(XL + o) = lp;
    if (lane == 0) XS[(size_t)m * nkb + kb] = a / kQMax;
}

/* ---- vocabulary order through the staging plane ---------------------------------------
 *
 * `vrow` is the head's inverse map, one word a vocabulary id: its class's code width in the top four
 * bits and its row in the class's whole tensor below, of which this rank holds rows
 * [rank * rows, (rank + 1) * rows). A row's tokens are one contiguous read, consecutive ids
 * consecutive writes. A staging row is `mp` floats (16-byte aligned). */
__global__ void vocab_gather_kernel(Head h, const float* __restrict__ st,
                                    const int32_t* __restrict__ vrow, float* __restrict__ out,
                                    uint32_t c_ld, uint32_t mp, uint32_t nb) {
    const uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= h.n_vocab) return;
    const uint32_t w = (uint32_t)vrow[v], b = w >> 28, rank = (uint32_t)h.off / h.n_vocab;
    uint32_t rc = 0, r0 = 0;
#pragma unroll
    for (uint32_t k = 0; k < kMaxClass; ++k)
        if (k < h.n && h.c[k].bits == b) { rc = h.c[k].N; r0 = h.c[k].row0; }
    const uint32_t j = (w & 0x0FFFFFFFu) - rank * rc;
    const bool ok = j < rc;
    const float4* const src = (const float4*)(st + (size_t)(ok ? r0 + j : 0u) * mp);
    for (uint32_t m4 = 0; m4 < nb; m4 += 4) {
        float4 q = src[m4 / 4];
        if (!ok) q = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float qv[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i)
            if (m4 + i < nb) out[(size_t)(m4 + i) * c_ld + v] = qv[i];
    }
}

/* ---- the decoder ---------------------------------------------------------------------------
 *
 * halo_iqr_dec.h's for the computed trellis under one scale a row, with the continuation read
 * compactly: a column's stream is 16 B bits, so each of a lane's four columns starts at bit 0 or
 * 16 of a dword, and the at most 15 bits a row's state takes from the unit above lie in that one
 * dword: four dword loads a unit, where Dec reads the unit above's whole segment again (fourteen
 * dwords at seven bits). */
template <uint32_t B>
struct HeadDec {
    static constexpr uint32_t L = 16, LANE_BITS = 64u * B, ND = LANE_BITS / 32u, CB = 16u * B;
    struct Raw { uint32_t d[ND]; uint32_t p[4]; };

    __device__ static Raw fetch(const uint8_t* W, uint32_t nkb, uint32_t rt, uint32_t kb,
                                uint32_t lane, bool live) {
        Raw r;
        constexpr size_t ub = 4u * LANE_BITS;
        load_dw<ND>(r.d, W + ((size_t)rt * nkb + kb) * ub + lane * (LANE_BITS / 8u), live);
        /* the unit above in the run of 256 rows (16 tiles), tail-biting at the run's first */
        const uint32_t rp = rt % 16u == 0u ? rt + 15u : rt - 1u;
        const uint32_t* up = (const uint32_t*)(W + ((size_t)rp * nkb + kb) * ub + lane * (LANE_BITS / 8u));
#pragma unroll
        for (uint32_t e = 0; e < 4; ++e) r.p[e] = live ? up[e * CB / 32u] : 0u;
        return r;
    }

    /* The lane's four values of each of the tile's 16 rows, packed: bytesum((st * A + Bc) &
     * 0x3F3F3F3F) - 126 a state. */
    __device__ static void rows(const Raw& r, uint32_t A, uint32_t Bc, uint32_t (&w)[16]) {
        /* a row's four values packed as soon as they are computed: the tile's NT accumulators
         * leave no room for all 64 at once */
#pragma unroll
        for (uint32_t q = 0; q < 16; ++q) {
            uint32_t t[4];
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e) {
                const uint32_t pos = e * CB + q * B, own = CB - q * B;
                const uint32_t st =
                    L <= own ? field<ND>(r.d, pos, L)
                             : field<ND>(r.d, pos, own) |
                                   ((r.p[e] >> ((e * CB) & 31u)) & ((1u << (L - own)) - 1u)) << own;
                const uint32_t h = (__umul24(st, A) + Bc) & 0x3F3F3F3Fu;
                t[e] = (uint32_t)__builtin_amdgcn_sudot4(false, (int)h, false, 0x01010101, -126, false);
            }
            w[15u - q] = pack4(pack2(t[0], t[1]), pack2(t[2], t[3]));
        }
    }
};

/* ---- the tile (gfx11) -------------------------------------------------------------------------
 *
 * libr4d's tile moved to gfx11's fragments (halo_wmma.h), as halo_gemm_iqr.hip's token tiles: MW
 * waves a block, a wave one 16-row tile of its class against NT 16-token tiles. Each 128-K unit's
 * codes are fetched a unit ahead, decoded to int8 (HeadDec), written row by row into the wave's
 * LDS tile and read back as eight A fragments (a lane's sixteen consecutive k of weight row
 * lane & 15, one 16-byte read). The activation's B fragments are a lane's token's sixteen
 * consecutive bytes of the hi and lo planes, read from global memory where the pass's planes sit
 * in L2 (libr4d stages them through LDS a 256-K chunk at a time). Accumulator slot t is weight row
 * 2t + (lane >> 4) of token lane & 15.
 *
 * Each unit's dot is libr4d's: the hi plane's int32 sum, times 128 in the accumulator, then the lo
 * plane's on top -- an exact integer per (row, token, block), converted and folded once in f32
 * with the block's factor and the token's scale, in ascending block order. */
template <class F, uint32_t NT, uint32_t MW, bool STAGE>
__device__ __forceinline__ void vocab_tile(const Head& h, const Cls& cl, uint32_t bx,
                                           const int8_t* __restrict__ XH,
                                           const int8_t* __restrict__ XL,
                                           const float* __restrict__ XS, float* __restrict__ out,
                                           uint32_t c_ld, uint32_t nb, uint8_t* s_tr) {
    using D = HeadDec<F::B>;
    static_assert(F::KIND == kTrellisMad && F::LRB == 0 && F::SC == kRowF32 && F::L == 16,
                  "a computed trellis, one scale a row");
    typedef int v4i_t __attribute__((ext_vector_type(4)));
    typedef int v8i_t __attribute__((ext_vector_type(8)));
    typedef float v8f_t __attribute__((ext_vector_type(8)));
    const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const uint32_t K = h.nkb * kBlock;

    Ctx c{};
    c.W = cl.W;
    c.S = cl.S;
    c.N = cl.N;
    c.nkb = h.nkb;
    c.run_tiles = 256u / 16u;
    uint8_t* const tr = s_tr + wave * kTrBytes;

    const uint32_t wrow0 = bx * MW * 16u + wave * 16u;
    const uint32_t arow = wrow0 / 16u;
    const bool alive = wrow0 < cl.N;
    if (!alive) return;   /* no block barrier below: a wave past its class has nothing to do */

    /* Token tile j's column is the pass's token 16 j + (lane & 15); a column past the pass reads
     * its last token, and its outputs are not stored. */
    uint32_t tok[NT];
    const int8_t* xh[NT];
    const int8_t* xl[NT];
#pragma unroll
    for (uint32_t j = 0; j < NT; ++j) {
        const uint32_t tj = j * 16u + (lane & 15u);
        tok[j] = tj < nb ? tj : nb - 1u;
        xh[j] = XH + (size_t)tok[j] * K;
        xl[j] = XL + (size_t)tok[j] * K;
    }
    v8f_t acc[NT];
#pragma unroll
    for (uint32_t j = 0; j < NT; ++j) acc[j] = v8f_t(0.0f);

    /* the computed code's A and B, after its 2^16 values (kept for the unrelayout) */
    constexpr Form fm{ F::KIND, F::B, F::LV, F::L, F::LRB, F::SC, 0 };
    const uint32_t* const ab = (const uint32_t*)(c.W + aux_off(fm, c.N, (int64_t)K));
    const uint32_t mA = ab[0], mB = ab[1];

    typename D::Raw cur = D::fetch(c.W, c.nkb, arow, 0, lane, true), nxt;
    for (uint32_t kb = 0; kb < c.nkb; ++kb) {
        if (kb + 1u < c.nkb) nxt = D::fetch(c.W, c.nkb, arow, kb + 1u, lane, true);
        uint32_t w[16];
        D::rows(cur, mA, mB, w);
        /* the unit through the wave's LDS tile: row q's four bytes of each lane, then each lane's
         * row (lane & 15) sixteen bytes a k step */
#pragma unroll
        for (uint32_t q = 0; q < 16; ++q) *(uint32_t*)(tr + q * kTrPitch + 4u * lane) = w[q];
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront", "local");
        v4i_t a[8];
#pragma unroll
        for (uint32_t ks = 0; ks < 8; ++ks)
            a[ks] = *(const v4i_t*)(tr + (lane & 15u) * kTrPitch + ks * 16u);
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront", "local");
        /* the block's factor (shared by every row) times each token's activation scale */
        const float f = fold_f32<F>(c, wrow0, kb);
#pragma unroll
        for (uint32_t j = 0; j < NT; ++j) {
            if (j * 16u >= nb) continue;   /* uniform */
            v8i_t raw = v8i_t(0);
            /* hi: 128 times its sum, exactly, in the accumulator; then lo on top */
#pragma unroll
            for (uint32_t ks = 0; ks < 8; ++ks) {
                const v4i_t b = *(const v4i_t*)(xh[j] + kb * kBlock + ks * 16u);
                raw = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a[ks], true, b, raw, false);
            }
#pragma unroll
            for (uint32_t t = 0; t < 8; ++t) raw[t] = raw[t] << 7;
#pragma unroll
            for (uint32_t ks = 0; ks < 8; ++ks) {
                const v4i_t b = *(const v4i_t*)(xl[j] + kb * kBlock + ks * 16u);
                raw = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a[ks], true, b, raw, false);
            }
            const float ws = f * XS[(size_t)tok[j] * h.nkb + kb];
#pragma unroll
            for (uint32_t t = 0; t < 8; ++t) acc[j][t] = fmaf((float)raw[t], ws, acc[j][t]);
        }
        cur = nxt;
    }

    /* A lane's eight accumulator rows are rows 2t + (lane >> 4) of the wave's tile, of one token.
     * STAGE: the logits go to the staging plane in the classes' row order, `c_ld` (the pass's
     * tokens rounded up to four) a row; else each goes to its id. */
    const uint32_t hh = lane >> 4;
    float rs[8];
    int32_t id[8];
#pragma unroll
    for (uint32_t t = 0; t < 8; ++t) {
        const uint32_t r = wrow0 + 2u * t + hh;
        rs[t] = ((const float*)cl.S)[r];
        id[t] = cl.ids[r];
    }
#pragma unroll
    for (uint32_t j = 0; j < NT; ++j) {
        const uint32_t tj = j * 16u + (lane & 15u);
        if (tj >= nb) continue;
        if constexpr (STAGE) {
#pragma unroll
            for (uint32_t t = 0; t < 8; ++t)
                out[(size_t)(cl.row0 + wrow0 + 2u * t + hh) * c_ld + tj] = acc[j][t] * rs[t];
        } else {
            float* const o = out + (size_t)tj * c_ld;
#pragma unroll
            for (uint32_t t = 0; t < 8; ++t) {
                const uint32_t v = (uint32_t)(id[t] - h.off);
                if (v < h.n_vocab) o[v] = acc[j][t] * rs[t];
            }
        }
    }
}

template <uint32_t B>
using VF = FormT<kTrellisMad, B, 0, 16, 0, kRowF32>;

/* The widths the classes may have: the computed trellis's built row forms. */
#define HALO_VOCAB_BITS(X) X(2) X(3) X(4) X(5) X(6) X(7)

template <uint32_t NT, uint32_t MW, bool STAGE>
__global__ __launch_bounds__(MW * kWave)
void vocab_tile_kernel(Head h, const int8_t* __restrict__ XH, const int8_t* __restrict__ XL,
                       const float* __restrict__ XS, float* __restrict__ out, uint32_t c_ld,
                       uint32_t nb) {
    __shared__ __attribute__((aligned(16))) uint8_t s_tr[MW * kTrBytes];
    /* the block's class: the last whose first block is at or below this one (uniform), copied
     * at constant indices so the argument block is never indexed at run time */
    Cls cl = h.c[0];
#pragma unroll
    for (uint32_t i = 1; i < kMaxClass; ++i)
        if (i < h.n && blockIdx.x >= h.c[i].blk0) cl = h.c[i];
    const uint32_t bx = blockIdx.x - cl.blk0;
    switch (cl.bits) {
#define HALO_VOCAB_CASE(b)                                                                        \
        case b: vocab_tile<VF<b>, NT, MW, STAGE>(h, cl, bx, XH, XL, XS, out, c_ld, nb, s_tr);     \
            break;
        HALO_VOCAB_BITS(HALO_VOCAB_CASE)
#undef HALO_VOCAB_CASE
        default: break;
    }
}

constexpr uint32_t kMW = 8;

struct Operands {
    const RadTensor* x = nullptr;
    const RadTensor* y = nullptr;
    const RadTensor *w[kMaxClass] = {}, *s[kMaxClass] = {}, *r[kMaxClass] = {}, *vrow = nullptr;
    long long M = 0, K = 0, rows[kMaxClass] = {};
    uint32_t n = 0;
};

/* Operand order: x, then (w, s, ids) a class, the inverse map, the logits (halo_rows.cpp's
 * oLogitsVocab). */
constexpr int kOpX = 0, kOpMap = 1 + 3 * (int)kMaxClass, kOpY = kOpMap + 1;

int operands(const RadArgs* a, Operands* g) {
    g->x = halo_opt(a, kOpX);
    g->y = halo_opt(a, kOpY);
    if (!g->x || !g->y) return HALO_NO(RAD_E_INVAL);
    if (g->x->dtype != RAD_BF16 || g->y->dtype != RAD_F32) return HALO_NO(RAD_E_DTYPE);
    if (g->x->rank < 2 || g->y->rank < 2) return HALO_NO(RAD_E_SHAPE);
    g->M = g->x->shape[0];
    g->K = halo_geti_or(a, "n_embd", 0);
    if (g->M < 1 || g->K < kFK || g->K % kFK || halo_numel(g->x) != g->M * g->K)
        return HALO_NO(RAD_E_SHAPE);
    if (g->y->shape[0] < g->M) return HALO_NO(RAD_E_SHAPE);
    static const char* kRows[kMaxClass] = { "rows0", "rows1", "rows2", "rows3" };
    for (uint32_t i = 0; i < kMaxClass; ++i) {
        const RadTensor* w = halo_opt(a, 1 + 3 * (int)i);
        const RadTensor* s = halo_opt(a, 2 + 3 * (int)i);
        const RadTensor* r = halo_opt(a, 3 + 3 * (int)i);
        const long long n = halo_geti_or(a, kRows[i], 0);
        if (!w && !s && !r && n == 0) break;
        if (!w || !s || !r || n < 256 || n % 256) return HALO_NO(RAD_E_SHAPE);
        if (!is_weight_dtype(w->dtype) || !is_scale_dtype(s->dtype) || r->dtype != RAD_I32)
            return HALO_NO(RAD_E_DTYPE);
        const uint32_t word = w->dtype & kWordMask;
        const Form f = form(word);
        if ((s->dtype & kWordMask) != word || f.kind != kTrellisMad || f.L != 16 || f.lrb != 0 ||
            f.scale != kRowF32 || f.lrun != 8 || f.bits < 2 || f.bits > 7)
            return HALO_NO(RAD_E_DTYPE);
        if (w->shape[0] != n || w->shape[1] != g->K || halo_numel(r) != n) return HALO_NO(RAD_E_SHAPE);
        if (!halo_aligned(w->data, 256) || !halo_aligned(s->data, 16) || !halo_aligned(r->data, 16))
            return HALO_NO(RAD_E_ALIGN);
        g->w[i] = w;
        g->s[i] = s;
        g->r[i] = r;
        g->rows[i] = n;
        g->n = i + 1;
    }
    if (g->n == 0) return HALO_NO(RAD_E_INVAL);
    g->vrow = halo_opt(a, kOpMap);
    if (!g->vrow || g->vrow->dtype != RAD_I32 || halo_numel(g->vrow) != halo_geti_or(a, "n_vocab", -1))
        return HALO_NO(RAD_E_SHAPE);
    if (halo_row_pitch(g->x) != g->K) return HALO_NO(RAD_E_STRIDE);
    return RAD_OK;
}

/* The staging plane's tokens: a pass of up to this many stages, a larger one stores directly
 * (staging 48 tokens measured 1175 us against the direct store's 1062 at 33). */
constexpr uint32_t kStageTok = 32;

/* Scratch: the two int8 planes and the scales for every token, then the staging plane. */
long long scratch_bytes(long long M, long long K, long long rows) {
    return 2 * align256(M * K) + align256(M * (K / kBlock) * 4) + align256(rows * kStageTok * 4);
}

}  // namespace

extern "C" int64_t halo_rad_logits_gemm_vocab_scratch(const RadArgs* a) {
    const long long M = halo_geti_or(a, "M", 0), K = halo_geti_or(a, "n_embd", 0);
    static const char* kRows[kMaxClass] = { "rows0", "rows1", "rows2", "rows3" };
    long long rows = 0;
    for (const char* k : kRows) rows += halo_geti_or(a, k, 0);
    if (M < 1 || K < 1) return 0;
    return (int64_t)scratch_bytes(M, K, rows);
}

extern "C" int halo_rad_logits_gemm_vocab(const RadArgs* a, RadStream s) {
    Operands g;
    const int rc = operands(a, &g);
    if (rc != RAD_OK) return rc;
    const long long M = g.M, K = g.K, nkb = K / kBlock;
    long long rows = 0;
    for (uint32_t i = 0; i < g.n; ++i) rows += g.rows[i];
    if (!a->scratch || (long long)a->scratch_bytes < scratch_bytes(M, K, rows))
        return HALO_NO(RAD_E_SHAPE);
    int8_t* const xh = (int8_t*)a->scratch;
    int8_t* const xl = xh + align256(M * K);
    float* const xs = (float*)(xl + align256(M * K));
    float* const stg = (float*)((uint8_t*)xs + align256(M * (K / kBlock) * 4));

    Head h{};
    h.n = g.n;
    h.nkb = (uint32_t)nkb;
    h.off = (int32_t)halo_geti_or(a, "vocab_offset", 0);
    h.n_vocab = (uint32_t)halo_geti_or(a, "n_vocab", 0);
    const long long c_ld = halo_row_pitch(g.y);
    if (h.n_vocab < 1 || c_ld < (long long)h.n_vocab) return HALO_NO(RAD_E_SHAPE);
    /* the widest class first: the narrow classes' cheaper blocks then fill the tail */
    uint32_t ord[kMaxClass];
    for (uint32_t i = 0; i < g.n; ++i) ord[i] = i;
    std::sort(ord, ord + g.n, [&](uint32_t x, uint32_t y) {
        return form(g.w[x]->dtype & kWordMask).bits > form(g.w[y]->dtype & kWordMask).bits;
    });
    uint32_t blocks = 0;
    for (uint32_t i = 0; i < g.n; ++i) {
        const uint32_t k = ord[i];
        Cls& c = h.c[i];
        c.W = (const uint8_t*)g.w[k]->data;
        c.S = (const uint8_t*)g.s[k]->data;
        c.ids = (const int32_t*)g.r[k]->data;
        c.N = (uint32_t)g.rows[k];
        c.bits = form(g.w[k]->dtype & kWordMask).bits;
        c.blk0 = blocks;
        c.row0 = blocks * kMW * 16u;
        blocks += c.N / (kMW * 16u);
    }

    if (HALO_EMIT(s, vocab_quant_kernel, dim3((uint32_t)M, (uint32_t)nkb), dim3(kWave), 0,
                 (const uint16_t*)g.x->data, (uint32_t)K, xh, xl, xs, (uint32_t)K) != RAD_OK)
        return HALO_NO(RAD_E_DEVICE);
    for (long long t0 = 0; t0 < M; t0 += kPassTok) {
        const uint32_t nb = (uint32_t)std::min<long long>(kPassTok, M - t0);
        const uint32_t nt = nb > 48 ? 4u : nb > 32 ? 3u : nb > 16 ? 2u : 1u;
        const dim3 grid(blocks, 1, 1);
        const int8_t* const ph = xh + t0 * K;
        const int8_t* const pl = xl + t0 * K;
        const float* const ps = xs + t0 * nkb;
        const bool stage = nb <= kStageTok;
        const uint32_t mp = (nb + 3u) / 4u * 4u;
        float* const po = stage ? stg : (float*)g.y->data + t0 * c_ld;
        const uint32_t ld = stage ? mp : (uint32_t)c_ld;
        int e;
#define HALO_VOCAB_GO(nt_, st_)                                                                   \
        e = HALO_EMIT(s, (vocab_tile_kernel<nt_, kMW, st_>), grid, dim3(kMW * kWave), 0, h, ph, pl, \
                     ps, po, ld, nb)
        if (nt == 4u) HALO_VOCAB_GO(4u, false);
        else if (nt == 3u) HALO_VOCAB_GO(3u, false);
        else if (nt == 2u) HALO_VOCAB_GO(2u, true);
        else HALO_VOCAB_GO(1u, true);
#undef HALO_VOCAB_GO
        if (e != RAD_OK) return HALO_NO(RAD_E_DEVICE);
        if (stage && HALO_EMIT(s, vocab_gather_kernel, dim3((h.n_vocab + 255u) / 256u), dim3(256), 0,
                              h, (const float*)stg, (const int32_t*)g.vrow->data,
                              (float*)g.y->data + t0 * c_ld, (uint32_t)c_ld, mp, nb) != RAD_OK)
            return HALO_NO(RAD_E_DEVICE);
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : HALO_NO(RAD_E_DEVICE);
}
