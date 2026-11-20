/* Derived from libr4d/r4d_gemm_iqr.hip by the libhalo rename transform (2026-11-17), then ported
 * to gfx1151; see libhalo/README.md. */
// halo_gemm_iqr.hip -- gemm_nt_q over IQR weights (halo_iqr.h), decoded from their codes where they
// are read; memory holds a B-bit weight in B bits. Two rows, as libr4d's:
//
//   gemm_iqr_nt_m64   M <= 64 (decode and speculative verify)
//   gemm_iqr_tiled    any M (prefill)
//
// Both run one launcher. At one token (two for 2- and 3-bit codes) a GEMV, libr4d's unchanged: a
// block a 16-row tile, its waves across K, each lane's decoded values one sudot4 against the
// activation -- no matrix instruction, so it ports as it is. Above, the 16-token tile, libr4d's
// iqr_m16_kernel moved to gfx11's fragments (halo_wmma.h): each 128-K unit is decoded once, written
// row by row into the wave's LDS tile and read back as eight A fragments (a lane's sixteen
// consecutive k of row lane & 15, one 16-byte read); the activation's B fragment is a lane's token's
// sixteen consecutive bytes, read from global memory where the M x K activation sits in L2; and
// accumulator slot t is weight row 2t + (lane >> 4) of token lane & 15. A block serves up to four
// token tiles (64 tokens) against its decoded unit, and grid.y takes further groups of 64 tokens,
// so the prefill row is the same kernel over more groups: each group decodes the weight again,
// which at a 2048-token chunk is 32 decodes a unit -- the cost libr4d's int8 prefill tile avoids
// by decoding into a scratch plane once. Not ported (yet): that tile, the 16 < M <= 64 narrow tile
// and the gated fold; libhalo's minimal IQR-V4 port serves the computed trellis under one f32 scale
// a row (halo_iqr.h's HALO_IQR_FORMS), the trunk form of IQR-V4.
//
// Every product is the one libr4d's tiles compute from the same int8 values (the unit's int32 sums
// biased so one fma folds them, flushed every kRbFlush units, the waves' partials summed through LDS
// in ascending order): the output is the product within f32 rounding.

#include "halo.h"
#include "halo_args.h"
#include "halo_chain.h"
#include "halo_common.h"
#include "halo_emit.h"
#include "halo_fp8_wmma.h"
#include "halo_iqr.h"
#include "halo_iqr_dec.h"
#include "halo_plugin.h"   /* halo_rad_gated_had_quant_i8g, the gated form's second launch */

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>

namespace {

using namespace halo_fp8;
using namespace halo_iqr;
using namespace halo_iqr_dev;

typedef int v4i_t __attribute__((ext_vector_type(4)));
typedef int v8i_t __attribute__((ext_vector_type(8)));
typedef float v8f_t __attribute__((ext_vector_type(8)));

constexpr uint32_t kWave = 32, kBlock = 128, kFK = 256;
/* The biased fold under one scale a row (libr4d's r4d_gemm_8a8_prefill.h): a unit's int32 sums
 * start at 0x4B400000, so a slot read as an f32 is 12582912 plus its sum and one fma folds it;
 * 12582912 times the folded factors leaves every kRbFlush units and at the store. */
constexpr int kRbBias = 0x4B400000;
constexpr float kRbBiasF = 12582912.0f;
constexpr uint32_t kRbFlush = 8;

__device__ __forceinline__ float pick(bool c, float x, float y) { return c ? x : y; }
/* Sixteen values a lane, each summed over the wave (libr4d's red16): lane l ends with the wave's
 * sum of value red16_row(l). */
__device__ __forceinline__ float red16(float (&a)[16], uint32_t lane) {
    const bool l1 = lane & 1u, l2 = lane & 2u, l4 = lane & 4u, l8 = lane & 8u;
    float b[8], c[4], d[2];
#pragma unroll
    for (uint32_t q = 0; q < 16; ++q) a[q] = halo_dpp_xor_add<0x161>(a[q]);
#pragma unroll
    for (uint32_t j = 0; j < 8; ++j) b[j] = halo_dpp_xor_add<0x162>(pick(l1, a[8 + j], a[j]));
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) c[j] = halo_dpp_xor_add<0x164>(pick(l2, b[4 + j], b[j]));
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) d[j] = halo_dpp_xor_add<0x168>(pick(l4, c[2 + j], c[j]));
    const float v = pick(l8, d[1], d[0]);
    return v + __shfl_xor(v, 16, 32);
}
__device__ __forceinline__ uint32_t red16_row(uint32_t lane) {
    return (lane & 1u) << 3 | (lane & 2u) << 1 | (lane & 4u) >> 1 | (lane & 8u) >> 3;
}

/* ---- the GEMV: one token, or two at 2 and 3 bits (libr4d's iqr_gemv_kernel, its computed
 * trellis under one f32 scale a row: each value's masked hash dotted with its column's activation
 * byte in every byte lane sums the value's bytes and multiplies in one instruction). */
template <class F, uint32_t MT, uint32_t KW, class CH>
__global__ __launch_bounds__(KW * kWave)
void iqr_gemv_kernel(Ctx c0, const int8_t* __restrict__ X, const float* __restrict__ XS,
                     uint16_t* __restrict__ Y, uint32_t N, uint32_t x_ld, uint32_t xs_ld) {
    if (CH::on) halo_chain_wait();
    using D = Dec<F, true>;
    static_assert(D::MAD && F::LRB == 0 && (F::SC == kRowF32 || (F::SC == kBf16 && MT == 1)),
                  "the computed trellis under a scale a row, or a bf16 scale a (row, 128) at one token");
    constexpr bool ROWF = F::SC == kRowF32;
    __shared__ float s_red[KW][16 * MT];
    const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1);
    const uint32_t wave = __builtin_amdgcn_readfirstlane(tid / kWave);
    const Ctx c = c0;
    const uint32_t rt = blockIdx.x;

    float acc[16][MT];
#pragma unroll
    for (uint32_t q = 0; q < 16; ++q)
#pragma unroll
        for (uint32_t m = 0; m < MT; ++m) acc[q][m] = 0.0f;
    float fsum[MT];
#pragma unroll
    for (uint32_t m = 0; m < MT; ++m) fsum[m] = 0.0f;
    uint32_t nflush = 0;

    typename D::Raw cur = D::fetch(c, rt, wave, lane, wave < c.nkb), nxt;
    for (uint32_t kb = wave; kb < c.nkb; kb += KW) {
        if (kb + KW < c.nkb) nxt = D::fetch(c, rt, kb + KW, lane, true);
        uint32_t xv[MT];
        float as[MT];
#pragma unroll
        for (uint32_t m = 0; m < MT; ++m) {
            xv[m] = *(const uint32_t*)(X + (size_t)m * x_ld + kb * kBlock + 4u * lane);
            as[m] = XS[(size_t)m * xs_ld + kb];
        }
        uint32_t h[16][4];
        D::hashes(cur, c, h);
        if constexpr (!ROWF) {
            /* A bf16 scale a (row, 128) (libr4d's UNS one-token form): each row's sum folded with
             * its own scale, the activation's bias sum taken off in the dot. */
            uint32_t xs[4];
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e) xs[e] = __builtin_amdgcn_perm(0u, xv[0], e * 0x01010101u);
            const int xsum = __builtin_amdgcn_sudot4(true, (int)xv[0], false, 0x7E7E7E7E, 0, false);
            const uint16_t* sp = (const uint16_t*)c.S + ((size_t)rt * c.nkb + kb) * 16u;
#pragma unroll
            for (uint32_t q = 0; q < 16; ++q) {
                const float wsq = bscale(sp[q]);
                int d = -xsum;
#pragma unroll
                for (uint32_t e = 0; e < 4; ++e)
                    d = __builtin_amdgcn_sudot4(false, (int)h[q][e], true, (int)xs[e], d, false);
                acc[q][0] = fmaf((float)d, wsq * as[0], acc[q][0]);
            }
            cur = nxt;
            continue;
        }
        const float fs = fold_f32<F>(c, 0u, kb);
#pragma unroll
        for (uint32_t m = 0; m < MT; ++m) {
            uint32_t xs[4];
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e) xs[e] = __builtin_amdgcn_perm(0u, xv[m], e * 0x01010101u);
            const float s = fs * as[m];
            fsum[m] += s;
            const int b0 = kRbBias - __builtin_amdgcn_sudot4(true, (int)xv[m], false, 0x7E7E7E7E, 0, false);
#pragma unroll
            for (uint32_t q = 0; q < 16; ++q) {
                int d = b0;
#pragma unroll
                for (uint32_t e = 0; e < 4; ++e)
                    d = __builtin_amdgcn_sudot4(false, (int)h[q][e], true, (int)xs[e], d, false);
                acc[q][m] = fmaf(__int_as_float(d), s, acc[q][m]);
            }
        }
        if (++nflush == kRbFlush) {
#pragma unroll
            for (uint32_t m = 0; m < MT; ++m) {
#pragma unroll
                for (uint32_t q = 0; q < 16; ++q) acc[q][m] = fmaf(-kRbBiasF, fsum[m], acc[q][m]);
                fsum[m] = 0.0f;
            }
            nflush = 0;
        }
        cur = nxt;
    }
    float rv[MT];
#pragma unroll
    for (uint32_t m = 0; m < MT; ++m) {
        float a16[16];
#pragma unroll
        for (uint32_t q = 0; q < 16; ++q) a16[q] = fmaf(-kRbBiasF, fsum[m], acc[q][m]);
        rv[m] = red16(a16, lane);
    }
    if (lane < 16u) {
#pragma unroll
        for (uint32_t m = 0; m < MT; ++m) s_red[wave][lane * MT + m] = rv[m];
    }
    __syncthreads();
    if (tid < 16u * MT) {
        const uint32_t q = red16_row(tid / MT), m = tid % MT;
        float v = 0.0f;
#pragma unroll
        for (uint32_t w = 0; w < KW; ++w) v += s_red[w][tid];
        v *= row_scale<F>(c, rt * 16u + q);
        Y[(size_t)m * N + rt * 16u + q] = f32_to_bf16(v);
    }
    if (CH::on) halo_chain_done();
}

/* ---- the token tiles: a block a 16-row weight tile and up to four 16-token tiles (grid.y the
 * groups of 64 tokens), its KW waves across K. */
template <class F, uint32_t KW, uint32_t NT, class CH>
__global__ __launch_bounds__(KW * kWave)
void iqr_tile_kernel(Ctx c0, const int8_t* __restrict__ X, const float* __restrict__ XS,
                     uint16_t* __restrict__ Y, uint32_t N, uint32_t M, uint32_t x_ld,
                     uint32_t xs_ld) {
    if (CH::on) halo_chain_wait();
    using D = Dec<F>;
    static_assert(D::MAD && F::LRB == 0 && (F::SC == kRowF32 || F::SC == kBf16),
                  "the computed trellis under a scale a row or a bf16 scale a (row, 128)");
    /* Under one f32 scale a row a unit's sums start biased and one fma folds them (BIAS); a bf16
     * scale a (row, 128) folds each converted sum with its row's scale. */
    constexpr bool BIAS = F::SC == kRowF32;
    __shared__ __attribute__((aligned(16))) uint8_t s_tr[KW * kTrBytes];
    __shared__ float s_red[KW][NT][8u * kWave];
    const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1);
    const uint32_t wave = __builtin_amdgcn_readfirstlane(tid / kWave);
    const Ctx c = c0;
    uint8_t* const tr = s_tr + wave * kTrBytes;
    const uint32_t rt = blockIdx.x, row0 = rt * 16u;
    const uint32_t m0 = blockIdx.y * 16u * NT;
    const uint32_t mg = M - m0 < 16u * NT ? M - m0 : 16u * NT;
    /* Token tile j is the group's tokens 16 j .. 16 j + 15; a column past the group reads its last
     * token, and its outputs are not stored. */
    uint32_t tok[NT];
    const int8_t* xrow[NT];
#pragma unroll
    for (uint32_t j = 0; j < NT; ++j) {
        const uint32_t tj = j * 16u + (lane & 15u);
        tok[j] = m0 + (tj < mg ? tj : mg - 1u);
        xrow[j] = X + (size_t)tok[j] * x_ld;
    }
    v8f_t out[NT];
    float corr[NT];
#pragma unroll
    for (uint32_t j = 0; j < NT; ++j) {
        out[j] = v8f_t(0.0f);
        corr[j] = 0.0f;
    }
    uint32_t nflush = 0;
    typename D::Raw cur = D::fetch(c, rt, wave, lane, wave < c.nkb), nxt;
    for (uint32_t kb = wave; kb < c.nkb; kb += KW) {
        if (kb + KW < c.nkb) nxt = D::fetch(c, rt, kb + KW, lane, true);
        uint32_t w[16];
        D::rows(cur, c, 1.0f, w);
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
        const float sbw = BIAS ? fold_f32<F>(c, row0, kb) : 0.0f;
        float wsr[8];   /* !BIAS: the lane's eight rows' (row, block) scales */
        if constexpr (!BIAS) {
#pragma unroll
            for (uint32_t t = 0; t < 8; ++t) wsr[t] = fold_f32<F>(c, row0 + 2u * t + (lane >> 4), kb);
        }
#pragma unroll
        for (uint32_t j = 0; j < NT; ++j) {
            if (j * 16u < mg) {
                v8i_t raw = v8i_t(BIAS ? kRbBias : 0);
#pragma unroll
                for (uint32_t ks = 0; ks < 8; ++ks) {
                    const v4i_t b = *(const v4i_t*)(xrow[j] + kb * kBlock + ks * 16u);
                    raw = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a[ks], true, b, raw, false);
                }
                const float as = XS[(size_t)tok[j] * xs_ld + kb];
                if constexpr (BIAS) {
                    const float sb = sbw * as;
                    corr[j] += sb;
#pragma unroll
                    for (uint32_t t = 0; t < 8; ++t)
                        out[j][t] = __builtin_fmaf(__int_as_float(raw[t]), sb, out[j][t]);
                } else {
#pragma unroll
                    for (uint32_t t = 0; t < 8; ++t)
                        out[j][t] = __builtin_fmaf((float)raw[t], wsr[t] * as, out[j][t]);
                }
            }
        }
        if (BIAS && ++nflush == kRbFlush) {
#pragma unroll
            for (uint32_t j = 0; j < NT; ++j) {
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t) out[j][t] = __builtin_fmaf(-kRbBiasF, corr[j], out[j][t]);
                corr[j] = 0.0f;
            }
            nflush = 0;
        }
        cur = nxt;
    }
#pragma unroll
    for (uint32_t j = 0; j < NT; ++j)
#pragma unroll
        for (uint32_t t = 0; t < 8; ++t)
            s_red[wave][j][t * kWave + lane] = __builtin_fmaf(-kRbBiasF, corr[j], out[j][t]);
    __syncthreads();
    if (wave == 0) {
        const uint32_t h = lane >> 4;
        float rs[8];
#pragma unroll
        for (uint32_t t = 0; t < 8; ++t) rs[t] = row_scale<F>(c, row0 + 2u * t + h);
#pragma unroll
        for (uint32_t j = 0; j < NT; ++j) {
            const uint32_t tj = j * 16u + (lane & 15u);
            if (tj >= mg) continue;
            uint16_t* yr = Y + (size_t)(m0 + tj) * N + row0;
#pragma unroll
            for (uint32_t t = 0; t < 8; ++t) {
                float a = 0.0f;
#pragma unroll
                for (uint32_t w = 0; w < KW; ++w) a += s_red[w][j][t * kWave + lane];
                yr[2u * t + h] = f32_to_bf16(a * rs[t]);
            }
        }
    }
    if (CH::on) halo_chain_done();
}

struct Operands {
    const RadTensor *a, *as, *b, *bs, *y;
    long long M, N, K, nkb, a_ld, sa_ld;
    uint32_t word;
    bool chain;   /* the chained forms (halo_chain.h): the engine resolved the row with chain = 1 */
};

int operands(const RadArgs* a, Operands* g) {
    g->a = halo_opt(a, 0);
    g->as = halo_opt(a, 1);
    g->b = halo_opt(a, 2);
    g->bs = halo_opt(a, 3);
    g->y = halo_opt(a, 4);
    g->chain = halo_geti_or(a, "chain", 0) != 0;   /* used at decode heights (launch_form) */
    if (!g->a || !g->as || !g->b || !g->bs || !g->y) return HALO_NO(RAD_E_INVAL);
    if (!is_weight_dtype(g->b->dtype) || !is_scale_dtype(g->bs->dtype)) return HALO_NO(RAD_E_DTYPE);
    g->word = g->b->dtype & kWordMask;
    if ((g->bs->dtype & kWordMask) != g->word) return HALO_NO(RAD_E_DTYPE);
    if (g->a->dtype != RAD_I8 || g->as->dtype != RAD_F32 || g->y->dtype != RAD_BF16)
        return HALO_NO(RAD_E_DTYPE);
    if (g->b->rank != 2 || g->y->rank < 2 || g->a->rank < 2 || g->as->rank != 2)
        return HALO_NO(RAD_E_SHAPE);
    g->M = g->y->shape[0];
    if (g->M < 1 || halo_numel(g->y) % g->M) return HALO_NO(RAD_E_SHAPE);
    g->N = halo_numel(g->y) / g->M;
    g->K = g->b->shape[1];
    if (g->b->shape[0] != g->N || g->N % 16 || g->K % kFK) return HALO_NO(RAD_E_SHAPE);
    g->nkb = g->K / kBlock;
    if (g->a->shape[0] != g->M || halo_numel(g->a) != g->M * g->K) return HALO_NO(RAD_E_SHAPE);
    if (g->as->shape[0] != g->M || g->as->shape[1] != g->nkb) return HALO_NO(RAD_E_SHAPE);
    g->a_ld = halo_row_pitch(g->a);
    g->sa_ld = halo_row_pitch(g->as);
    if (g->a_ld < 0 || g->sa_ld < 0 || halo_row_pitch(g->y) != g->N) return HALO_NO(RAD_E_STRIDE);
    if (!halo_aligned(g->a->data, 16) || !halo_aligned(g->b->data, 256)) return HALO_NO(RAD_E_ALIGN);
    if (g->a_ld % 16) return HALO_NO(RAD_E_ALIGN);   /* the B fragment's 16-byte reads */
    const Form f = form(g->word);
    if (is_trellis(f.kind) && g->N % ((long long)1 << f.lrun)) return HALO_NO(RAD_E_SHAPE);
    return RAD_OK;
}

Ctx ctx_of(const Operands& g) {
    Ctx c{};
    c.W = (const uint8_t*)g.b->data;
    c.S = (const uint8_t*)g.bs->data;
    c.N = (uint32_t)g.N;
    c.nkb = (uint32_t)g.nkb;
    const Form f = form(g.word);
    c.run_tiles = is_trellis(f.kind) ? (1u << f.lrun) / 16u : 1u;
    return c;
}

/* libr4d's measurement: the GEMV wins at one token on every width and at two on 2- and 3-bit
 * codes; from there the tile. */
template <class F>
constexpr uint32_t gemv_max() { return F::B <= 3u && F::SC == kRowF32 ? 2u : 1u; }

template <class F>
int launch_form(const Operands& g, RadStream stream) {
    const uint32_t N = (uint32_t)g.N, M = (uint32_t)g.M;
    const Ctx c = ctx_of(g);
    const int8_t* x = (const int8_t*)g.a->data;
    const float* xs = (const float*)g.as->data;
    uint16_t* y = (uint16_t*)g.y->data;
    const uint32_t a_ld = (uint32_t)g.a_ld, sa_ld = (uint32_t)g.sa_ld;
    int rc = RAD_E_SHAPE;
    /* Four waves a tile where the tiles alone nearly fill the card, eight on narrow weights. */
    const bool wide = N >= 2048u;
    Operands gc = g;
    gc.chain = g.chain && M <= 64u;   /* halo_chain.h's decode heights */
    if (M <= gemv_max<F>()) {
#define HALO_IQR_GV(mt_, kw_)                                                                     \
        rc = gc.chain                                                                              \
           ? HALO_EMIT(stream, (iqr_gemv_kernel<F, mt_, kw_, r4dchain>), dim3(N / 16u),           \
                       dim3(kw_ * kWave), 0, c, x, xs, y, N, a_ld, sa_ld)                         \
           : HALO_EMIT(stream, (iqr_gemv_kernel<F, mt_, kw_, HaloUnchained>), dim3(N / 16u),      \
                       dim3(kw_ * kWave), 0, c, x, xs, y, N, a_ld, sa_ld)
        if (M == 1u) { if (wide) HALO_IQR_GV(1u, 4u); else HALO_IQR_GV(1u, 8u); }
        else if constexpr (gemv_max<F>() >= 2u) {
            if (wide) HALO_IQR_GV(2u, 4u); else HALO_IQR_GV(2u, 8u);
        }
#undef HALO_IQR_GV
    } else {
        const uint32_t nt = M >= 64u ? 4u : (M + 15u) / 16u;
        const dim3 grid(N / 16u, (M + 16u * nt - 1u) / (16u * nt));
#define HALO_IQR_T(kw_, nt_)                                                                      \
        rc = gc.chain                                                                              \
           ? HALO_EMIT(stream, (iqr_tile_kernel<F, kw_, nt_, r4dchain>), grid, dim3(kw_ * kWave),  \
                       0, c, x, xs, y, N, M, a_ld, sa_ld)                                         \
           : HALO_EMIT(stream, (iqr_tile_kernel<F, kw_, nt_, HaloUnchained>), grid,               \
                       dim3(kw_ * kWave), 0, c, x, xs, y, N, M, a_ld, sa_ld)
#define HALO_IQR_TK(nt_) do { if (wide) HALO_IQR_T(4u, nt_); else HALO_IQR_T(8u, nt_); } while (0)
        switch (nt) {
            case 1u: HALO_IQR_TK(1u); break;
            case 2u: HALO_IQR_TK(2u); break;
            case 3u: HALO_IQR_TK(3u); break;
            default: HALO_IQR_TK(4u); break;
        }
#undef HALO_IQR_TK
#undef HALO_IQR_T
    }
    return rc == RAD_OK && hipGetLastError() == hipSuccess ? RAD_OK : HALO_NO(RAD_E_DEVICE);
}

int iqr_launch(const RadArgs* a, RadStream stream) {
    Operands g{};
    int rc = operands(a, &g);
    if (rc != RAD_OK) return rc;
    const uint32_t kw = kernel_word(g.word);
#define HALO_IQR_F(k, b, lv, l, lrb, sc)                                                          \
    if (kw == FormT<k, b, lv, l, lrb, sc>::word())                                                \
        return launch_form<FormT<k, b, lv, l, lrb, sc>>(g, stream);
    HALO_IQR_FORMS(HALO_IQR_F)
#undef HALO_IQR_F
    return HALO_NO(RAD_E_UNSUPPORTED);
}

/* gemm_nt_q_gated at i8a8h over an IQR [2n, K] gate_up plane: [0] a [1] a_scale [2] b [3] b_scale
 * [4] q [5] scale [6] out_bf16. The GEMM into the op's scratch ([M, 2n] bf16), then
 * gated_had_quant_i8g over it: the pair the op is defined as (libr4d folds the quantiser into the
 * GEMM's epilogue, saving the launch and the [M, 2n] round trip). */
int gated_launch(const RadArgs* a, RadStream stream) {
    const RadTensor* q = halo_opt(a, 4);
    const RadTensor* sc = halo_opt(a, 5);
    const RadTensor* yb = halo_opt(a, 6);
    const RadTensor* b = halo_opt(a, 2);
    const RadTensor* x = halo_opt(a, 0);
    if (!q || !sc || !b || !x || b->rank != 2 || x->rank < 2) return HALO_NO(RAD_E_INVAL);
    const long long M = x->shape[0], N = b->shape[0];
    if (M < 1 || N % 256) return HALO_NO(RAD_E_SHAPE);
    if (!a->scratch || a->scratch_bytes < align256(M * N * 2)) return HALO_NO(RAD_E_SCRATCH);
    RadTensor y{};
    y.data = a->scratch;
    y.dtype = RAD_BF16;
    y.rank = 2;
    y.shape[0] = M;
    y.shape[1] = N;
    y.stride[0] = N;
    y.stride[1] = 1;
    RadTensor t[5] = { a->t[0], a->t[1], a->t[2], a->t[3], y };
    RadArgs g = *a;
    g.t = t;
    g.n_t = 5;
    int rc = iqr_launch(&g, stream);
    if (rc != RAD_OK) return rc;
    RadTensor u[4] = { y, *q, *sc, RadTensor{} };
    if (yb) u[3] = *yb;
    RadArgs h = *a;
    h.t = u;
    h.n_t = yb ? 4 : 3;
    h.scratch = nullptr;
    h.scratch_bytes = 0;
    return halo_rad_gated_had_quant_i8g(&h, stream);
}

}  // namespace

extern "C" int halo_rad_gemm_iqr_gated(const RadArgs* a, RadStream s) { return gated_launch(a, s); }
extern "C" int64_t halo_rad_gemm_iqr_gated_scratch(const RadArgs* a) {
    const long long M = halo_geti_or(a, "M", 0), N = halo_geti_or(a, "N", 0);
    if (M < 1 || N < 2) return 0;
    return (int64_t)align256(M * N * 2);
}
extern "C" int halo_rad_gemm_iqr_m64(const RadArgs* a, RadStream s) { return iqr_launch(a, s); }
extern "C" int halo_rad_gemm_iqr_tiled(const RadArgs* a, RadStream s) { return iqr_launch(a, s); }
extern "C" int64_t halo_rad_gemm_iqr_scratch(const RadArgs*) { return 0; }
