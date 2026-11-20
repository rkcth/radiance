// halo_logits_gemm.hip - logits_gemm_fp8 / logits_gemm_i8: the lm_head projection over the
// stored head whose rows carry their own scales in the tail (halo_fp8_lm.h). C[M, n_vocab]
// f32 = X[M, n_embd] bf16 @ dequant(W)^T, fp32 accumulate.
//
// Two forms. The matvec (M = 1 on the fp8 head, M <= 3 on the i8 head): one wave a vocabulary
// row, the lanes walk the row's 16-byte chunks, decode the codes, fold the chunk's 128-group
// scale (read from the row's own tail) into the partial, and close with the xor butterfly. The
// WMMA form (every other M, below): the head decoded once into LDS per 64 rows of x. The fp8
// head's tail holds its 128-row block's scales repeated; the i8 head's tail holds the row's own
// -- both forms read the tail either way, so one kernel serves both rows.
#include "halo.h"
#include "halo_args.h"
#include "halo_common.h"
#include "halo_fp8_lm.h"
#include "halo_fp8_wmma.h"
#include "halo_iqr.h"
#include "halo_wmma.h"

#include <cstdint>

namespace {

/* An IQR head (halo_iqr.h's lm row: TB-bit codes indexing an int8 table of 2^TB entries, the row's
 * bf16 scales after its codes), libr4d's lm_table / lm_codes16: TB = 0 is a byte a code. The table
 * sits after the last row (lm_tab), and a kernel holds it in sixteen dwords (`tv`) for byte
 * permutes: a lookup of four codes is one v_perm a group of eight entries, the groups chosen
 * bytewise by the codes' upper bits. No matrix instruction: libr4d's code as it is. */
template <uint32_t TB>
__device__ __forceinline__ void lm_table(const uint8_t* W, uint32_t rows, uint32_t w_ld,
                                         uint32_t (&tv)[16]) {
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) tv[i] = 0u;
    if constexpr (TB != 0) {
        const uint32_t* t = (const uint32_t*)(W + ((size_t)rows * w_ld + 255u) / 256u * 256u);
#pragma unroll
        for (uint32_t i = 0; i < ((1u << TB) + 3u) / 4u; ++i) tv[i] = t[i];
    }
}
/* The table entries of four codes, one a byte of `x`. */
template <uint32_t TB>
__device__ __forceinline__ uint32_t lm_lookup4(uint32_t x, const uint32_t (&tv)[16]) {
    const uint32_t lo = x & 0x07070707u;
    uint32_t g[8];
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i)
        if (i < ((1u << TB) + 7u) / 8u) g[i] = __builtin_amdgcn_perm(tv[2 * i + 1], tv[2 * i], lo);
#pragma unroll
    for (uint32_t b = 3; b < TB; ++b) {
        /* bytes whose bit b is set take the upper half of each pair of groups */
        const uint32_t m = ((x >> b) & 0x01010101u) * 0xFFu;
        const uint32_t half = 1u << (b - 3);
#pragma unroll
        for (uint32_t i = 0; i < 8; i += 2 * half)
            if (i + half < ((1u << TB) + 7u) / 8u) g[i] = (g[i + half] & m) | (g[i] & ~m);
    }
    return g[0];
}
/* Sixteen consecutive codes from column c (a multiple of 16) of a row, as int8 bytes (TB = 0: the
 * row's own sixteen bytes). */
template <uint32_t TB>
__device__ __forceinline__ uint4 lm_codes16(const uint8_t* __restrict__ row, uint32_t c,
                                            const uint32_t (&tv)[16]) {
    if constexpr (TB == 0) {
        return *(const uint4*)(row + c);
    } else {
        /* 2*TB bytes from offset 2*TB*(c/16), read from the dword-aligned window over them */
        constexpr uint32_t ND = (2u * TB + 3u + 3u) / 4u;
        const uint32_t off = c / 16u * 2u * TB;
        const uint32_t* p = (const uint32_t*)(row + (off & ~3u));
        uint32_t w[ND];
#pragma unroll
        for (uint32_t i = 0; i < ND; ++i) w[i] = p[i];
        const uint32_t sh = (off & 3u) * 8u;
        uint32_t d[ND];
#pragma unroll
        for (uint32_t i = 0; i < ND; ++i)
            d[i] = __builtin_amdgcn_alignbit(i + 1u < ND ? w[i + 1u] : 0u, w[i], sh);
        uint32_t o[4];
#pragma unroll
        for (uint32_t q = 0; q < 4; ++q) {
            uint32_t x = 0;
#pragma unroll
            for (uint32_t j = 0; j < 4; ++j) {
                const uint32_t bit = (q * 4u + j) * TB, i = bit >> 5, s = bit & 31u;
                const uint32_t v = (s + TB <= 32u ? d[i] >> s
                                                  : __builtin_amdgcn_alignbit(d[i + 1u], d[i], s)) &
                                   ((1u << TB) - 1u);
                x |= v << (8u * j);
            }
            o[q] = lm_lookup4<TB>(x, tv);
        }
        return make_uint4(o[0], o[1], o[2], o[3]);
    }
}
/* Bytes of a row's codes before its scales. */
template <uint32_t TB>
__device__ __forceinline__ long lm_codes_bytes(long K) { return TB ? K / 8 * (long)TB : K; }

template <bool I8, int MT, uint32_t TB = 0>
__global__ void __launch_bounds__(128) logits_kernel(const uint16_t* __restrict__ x,
                                                     const uint8_t* __restrict__ w,
                                                     float* __restrict__ y,
                                                     int M, int N, long K, long stride,
                                                     long a_ld, long c_ld) {
    const int lane = (int)threadIdx.x & 31;
    const int n = ((int)blockIdx.x * 4 + ((int)threadIdx.x >> 5));
    if (n >= N) return;
    const int m0 = (int)blockIdx.y * MT;
    const int mt = min(MT, M - m0);
    uint32_t tv[16];
    lm_table<TB>(w, (uint32_t)N, (uint32_t)stride, tv);
    const uint8_t* wr = w + (long)n * stride;
    const uint16_t* sr = (const uint16_t*)(wr + lm_codes_bytes<TB>(K));
    const long nchunks = K >> 4;
    float acc[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[m] = 0.0f;
    for (long c = lane; c < nchunks; c += 32) {
        const long k0 = c << 4;
        const float wsc = bf16_to_f32(sr[k0 >> 7]);
        const uint4 wb = lm_codes16<TB>(wr, (uint32_t)k0, tv);
        const uint32_t qw[4] = { wb.x, wb.y, wb.z, wb.w };
        float wv[16];
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            if (I8) {
#pragma unroll
                for (int b = 0; b < 4; ++b)
                    wv[4 * d + b] = (float)(int8_t)((qw[d] >> (8 * b)) & 0xffu);
            } else {
                const v2f wlo = halo_fp8::e4m3_pk<false>(qw[d]);
                const v2f whi = halo_fp8::e4m3_pk<true>(qw[d]);
                wv[4 * d + 0] = wlo.x; wv[4 * d + 1] = wlo.y;
                wv[4 * d + 2] = whi.x; wv[4 * d + 3] = whi.y;
            }
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m >= mt) break;
            const uint16_t* xr = x + (long)(m0 + m) * a_ld + k0;
            float part = 0.0f;
#pragma unroll
            for (int e = 0; e < 16; ++e) part = fmaf(wv[e], bf16_to_f32(xr[e]), part);
            acc[m] = fmaf(wsc, part, acc[m]);
        }
    }
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        if (m >= mt) break;
        float v = acc[m];
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) v += __shfl_xor(v, mask);
        if (lane == 0) y[(long)(m0 + m) * c_ld + n] = v;
    }
}


/* ---- the WMMA form, for M past the matvec band ----------------------------------------------
 *
 * The matvec above streams the head once per 8 (or 32) rows of x and spends ~16 VALU a code a
 * row, so its time grows with M from about 8 rows on. This form reads and decodes the head ONCE
 * for up to 64 rows (96 at M = 65 .. 96) and multiplies on the bf16 WMMA:
 *
 *   C^T tile = head tile (B: lane l holds vocabulary row l & 15, sixteen consecutive k) x x^T
 *   (A: lane l holds x row l & 15), D lane l / element e = x row 2e + (l >> 4), vocabulary
 *   row l & 15 (halo_wmma.h's map).
 *
 * A block is four waves; each wave owns 16 vocabulary rows, all four share the x tile. Per
 * stage (KG 128-groups of k) every wave loads its head rows coalesced (eight lanes a 128-byte
 * row group), decodes each code ONCE into a wave-private bf16 tile in LDS, the block stages the
 * x tile beside it, and the loads of the next stage are issued before the WMMAs of this one.
 *
 * Each 128-group accumulates into a zeroed tmp and is folded into the accumulator with that
 * row's scale (one fma a group, as the scale is per vocabulary row and group). The WMMA sums
 * the sixteen products of a k step in its own order, so the result is NOT the matvec's bits:
 * against a double reference it is within ~1e-7 of sum|products| (the matvec ~2e-8). */
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x8 __attribute__((ext_vector_type(8)));

/* The block barrier for the LDS tiles alone. __syncthreads() fences global memory too, which
 * waits out the next group's loads (vmcnt(0)) and would serialise the prefetch; only the LDS
 * stores need to have landed. */
__device__ __forceinline__ void lg_lds_barrier() {
    __asm__ volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

/* fp8: the codes decode 2^-8 low (f16's exponent bias over E4M3's), the scale takes the 2^8. */
#define HALO_LG_FIX 256.0f

__device__ __forceinline__ float lg_h2f(uint32_t bits16) {
    union { uint16_t u; _Float16 h; } c;
    c.u = (uint16_t)bits16;
    return (float)c.h;
}

/* Sixteen head codes to sixteen bf16, EXACT and never subnormal (the WMMA flushes subnormal
 * bf16 inputs, measured: halo_wmma.h's deq16_e4m3 loses E4M3's subnormal codes that way).
 *
 * fp8: each code's sign and seven magnitude bits placed as an f16 (E4M3's exponent and mantissa
 * land in f16's, 2^-8 low, subnormals included as f16 subnormals), widened to f32 by
 * v_cvt_f32_f16 (which keeps f16 subnormals) and cut to bf16 (an E4M3 value has four
 * significant bits). i8: each code through f32. */
template <bool I8>
__device__ __forceinline__ u32x8 lg_dec16(const u32x4 q) {
    u32x8 r;
#pragma unroll
    for (int d = 0; d < 4; ++d) {
        const uint32_t v = q[d];
        float f0, f1, f2, f3;
        if constexpr (!I8) {
            /* bytes 0 and 2 to f16 halves (sign to bit 15, magnitude to bits 7..13); 1 and 3 */
            const uint32_t h02 = ((v << 8) & 0x80008000u) | ((v << 7) & 0x3f803f80u);
            const uint32_t h13 = (v & 0x80008000u) | ((v >> 1) & 0x3f803f80u);
            f0 = lg_h2f(h02); f2 = lg_h2f(h02 >> 16);
            f1 = lg_h2f(h13); f3 = lg_h2f(h13 >> 16);
        } else {
            f0 = (float)(int8_t)(v);
            f1 = (float)(int8_t)(v >> 8);
            f2 = (float)(int8_t)(v >> 16);
            f3 = (float)(int8_t)(v >> 24);
        }
        r[2 * d]     = __builtin_amdgcn_perm(__float_as_uint(f1), __float_as_uint(f0), 0x07060302u);
        r[2 * d + 1] = __builtin_amdgcn_perm(__float_as_uint(f3), __float_as_uint(f2), 0x07060302u);
    }
    return r;
}

/* Measured on gfx1151 at 248320 x 5120: four waves a block (eight and two lost), one 16-row
 * vocabulary tile a wave (two lost: the decoded tile and the accumulators double), two groups a
 * stage where the LDS allows (one group a stage is ~3% slower at M <= 48). */
#define HALO_LG_W 4
#define HALO_LG_NT 1
#define HALO_LG_KG 2
template <bool I8, int MT, int NT, uint32_t TB = 0>
__global__ void __launch_bounds__(32 * HALO_LG_W) logits_wmma_kernel(const uint16_t* __restrict__ x,
                                                          const uint8_t* __restrict__ w,
                                                          float* __restrict__ y,
                                                          int M, int N, long K, long stride,
                                                          long a_ld, long c_ld) {
    constexpr int W = HALO_LG_W;    /* waves a block */
    constexpr int KG = MT >= 4 ? 1 : HALO_LG_KG;  /* 128-groups a stage (LDS caps M > 48 at one) */
    constexpr int XP = 136;         /* tile row pitch in bf16: 256 B + 16 B pad */
    constexpr int XI = 8 * MT / W;  /* x staging loads a thread a group */
    static_assert(XI * W == 8 * MT, "x tile rows split evenly over the block");
    __shared__ __attribute__((aligned(16))) uint16_t sx[KG][16 * MT * XP];
    __shared__ __attribute__((aligned(16))) uint16_t sw[W][KG][16 * NT * XP];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31, wv = tid >> 5;
    const int col = lane & 15, h = lane >> 4;
    /* No early exit: every wave takes the block barriers. A wave past the vocabulary reads row
     * N-1 and stores nothing. */
    const long nb = ((long)blockIdx.x * W + wv) * (16 * NT);
    const int m0 = (int)blockIdx.y * (16 * MT);

    /* Coalesced staging loads, one base pointer each and 32-bit offsets. Head: load i covers
     * tile rows 4i .. 4i+3 of one group, lane l row 4i + (l>>3), bytes 16 (l & 7). x: load i
     * covers tile rows 2Wi .. 2Wi + 2W-1, thread t row 2Wi + (t>>4), bytes 16 (t & 15). */
    const long nlast = (long)N - 1;
    const long nbase = min(nb, nlast);
    const uint8_t* wb = w + nbase * stride;
    /* TB > 0 (an IQR head): a lane's sixteen codes are lm_codes16's from its row's start, so the
     * offset is the row's and the column is added at the fetch. */
    uint32_t tv[16];
    lm_table<TB>(w, (uint32_t)N, (uint32_t)stride, tv);
    uint32_t wo[4 * NT];
#pragma unroll
    for (int i = 0; i < 4 * NT; ++i)
        wo[i] = (uint32_t)((min(nb + 4 * i + (lane >> 3), nlast) - nbase) * stride) +
                (TB ? 0u : (uint32_t)(lane & 7) * 16);
    uint32_t so[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j)
        so[j] = (uint32_t)((min(nb + 16 * j + col, nlast) - nbase) * stride + lm_codes_bytes<TB>(K));
    const int mbase = min(m0, M - 1);
    const uint16_t* xb = x + (long)mbase * a_ld + (tid & 15) * 8;
    uint32_t xo[XI];
#pragma unroll
    for (int i = 0; i < XI; ++i)
        xo[i] = (uint32_t)((min(m0 + 2 * W * i + (tid >> 4), M - 1) - mbase) * a_ld);

    const int ng = (int)(K >> 7);
    u32x4 wq[KG][4 * NT], xq[KG][XI];
    uint32_t sq[KG][NT];
    auto fetch = [&](int g0) {
#pragma unroll
        for (int q = 0; q < KG; ++q) {
            const int gq = min(g0 + q, ng - 1);   /* past the end re-reads the last group */
            const uint32_t o = (uint32_t)gq * 128;
#pragma unroll
            for (int i = 0; i < 4 * NT; ++i) {
                if constexpr (TB == 0) {
                    wq[q][i] = *(const u32x4*)(wb + wo[i] + o);
                } else {
                    const uint4 c = lm_codes16<TB>(wb + wo[i], o + (uint32_t)(lane & 7) * 16, tv);
                    wq[q][i] = u32x4{ c.x, c.y, c.z, c.w };
                }
            }
#pragma unroll
            for (int i = 0; i < XI; ++i) xq[q][i] = *(const u32x4*)(xb + xo[i] + o);
#pragma unroll
            for (int j = 0; j < NT; ++j) sq[q][j] = *(const uint16_t*)(wb + so[j] + 2 * gq);
        }
    };
    fetch(0);

    v8f acc[MT][NT];
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int j = 0; j < NT; ++j) acc[t][j] = (v8f)0.f;
    for (int g = 0; g < ng; g += KG) {
        lg_lds_barrier();
        float sc[KG][NT];
#pragma unroll
        for (int q = 0; q < KG; ++q) {
#pragma unroll
            for (int i = 0; i < 4 * NT; ++i) {
                const u32x8 d = lg_dec16<I8>(wq[q][i]);
                uint16_t* p = &sw[wv][q][(4 * i + (lane >> 3)) * XP + (lane & 7) * 16];
                *(u32x4*)p = d.lo;
                *(u32x4*)(p + 8) = d.hi;
            }
#pragma unroll
            for (int i = 0; i < XI; ++i)
                *(u32x4*)&sx[q][(2 * W * i + (tid >> 4)) * XP + (tid & 15) * 8] = xq[q][i];
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const float sv = halo_fp8::bscale(sq[q][j]);
                sc[q][j] = I8 ? sv : sv * HALO_LG_FIX;
            }
        }
        fetch(g + KG);   /* the next stage's loads fly while this one computes */
        lg_lds_barrier();
#pragma unroll
        for (int q = 0; q < KG; ++q) {
            if (KG > 1 && g + q >= ng) break;   /* uniform: a K with an odd group count */
            v8f tmp[MT][NT];
#pragma unroll
            for (int t = 0; t < MT; ++t)
#pragma unroll
                for (int j = 0; j < NT; ++j) tmp[t][j] = (v8f)0.f;
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                v16bf b[NT];
#pragma unroll
                for (int j = 0; j < NT; ++j)
                    b[j] = halo::mk_ldg<v16bf>(&sw[wv][q][(16 * j + col) * XP + 16 * u]);
#pragma unroll
                for (int t = 0; t < MT; ++t) {
                    const v16bf a = halo::mk_ldg<v16bf>(&sx[q][(16 * t + col) * XP + 16 * u]);
#pragma unroll
                    for (int j = 0; j < NT; ++j) tmp[t][j] = halo::mma(a, b[j], tmp[t][j]);
                }
            }
#pragma unroll
            for (int t = 0; t < MT; ++t)
#pragma unroll
                for (int j = 0; j < NT; ++j)
#pragma unroll
                    for (int e = 0; e < 8; ++e)
                        acc[t][j][e] = fmaf(sc[q][j], tmp[t][j][e], acc[t][j][e]);
        }
    }
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        const long n = nb + 16 * j + col;
        if (n >= N) continue;
#pragma unroll
        for (int t = 0; t < MT; ++t)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int m = m0 + 16 * t + 2 * e + h;
                if (m < M) y[(long)m * c_ld + n] = acc[t][j][e];
            }
    }
}

/* The launch both entries share. The matvec keeps the rows it is already at the weight's
 * bandwidth for (one row on the fp8 head, three on the i8 head, measured on the part), and any
 * x or head the WMMA form's 16-byte loads cannot take. Above, one pass of the head covers up to
 * 64 rows, or 96 when M is 65 .. 96 (one six-tile pass beats two four-tile ones there; past 96
 * the six-tile wave is compute-bound and 64-row passes are cheaper). */
template <bool I8, uint32_t TB = 0>
void logits_launch(const uint16_t* xp, const uint8_t* wp, float* yp, int M, int N, long K,
                   long stride, long a_ld, long c_ld, hipStream_t st) {
    constexpr int MIN_M = I8 ? 4 : 2;
    if (M < MIN_M || (a_ld % 8) || (stride % 16) || ((uintptr_t)xp & 15) || ((uintptr_t)wp & 15)) {
        if (M <= 32) {
            const dim3 grid((unsigned)((N + 3) / 4), (unsigned)((M + 7) / 8));
            logits_kernel<I8, 8, TB><<<grid, 128, 0, st>>>(xp, wp, yp, M, N, K, stride, a_ld, c_ld);
        } else {
            const dim3 grid((unsigned)((N + 3) / 4), (unsigned)((M + 31) / 32));
            logits_kernel<I8, 32, TB><<<grid, 128, 0, st>>>(xp, wp, yp, M, N, K, stride, a_ld, c_ld);
        }
        return;
    }
    constexpr int NT = HALO_LG_NT;
    const unsigned gx = (unsigned)((N + 16 * HALO_LG_W * NT - 1) / (16 * HALO_LG_W * NT));
    const int mtiles = (M > 64 && M <= 96) ? 6 : (min(M, 64) + 15) / 16;
    const unsigned gy = (unsigned)((M + 16 * mtiles - 1) / (16 * mtiles));
    const dim3 grid(gx, gy);
    switch (mtiles) {
    case 1: logits_wmma_kernel<I8, 1, NT, TB><<<grid, 32 * HALO_LG_W, 0, st>>>(xp, wp, yp, M, N, K, stride, a_ld, c_ld); break;
    case 2: logits_wmma_kernel<I8, 2, NT, TB><<<grid, 32 * HALO_LG_W, 0, st>>>(xp, wp, yp, M, N, K, stride, a_ld, c_ld); break;
    case 3: logits_wmma_kernel<I8, 3, NT, TB><<<grid, 32 * HALO_LG_W, 0, st>>>(xp, wp, yp, M, N, K, stride, a_ld, c_ld); break;
    case 4: logits_wmma_kernel<I8, 4, NT, TB><<<grid, 32 * HALO_LG_W, 0, st>>>(xp, wp, yp, M, N, K, stride, a_ld, c_ld); break;
    default: logits_wmma_kernel<I8, 6, NT, TB><<<grid, 32 * HALO_LG_W, 0, st>>>(xp, wp, yp, M, N, K, stride, a_ld, c_ld); break;
    }
}

template <bool I8>
int logits_run(const RadArgs* a, RadStream stream, uint32_t wdtype) {
    const RadTensor* X = halo_opt(a, 0);
    const RadTensor* W = halo_opt(a, 1);
    const RadTensor* Y = halo_opt(a, 2);
    if (!X || !W || !Y) return HALO_NO(RAD_E_INVAL);
    if (X->dtype != RAD_BF16 || W->dtype != wdtype || Y->dtype != RAD_F32)
        return HALO_NO(RAD_E_DTYPE);
    const long long K = halo_geti_or(a, "n_embd", halo_dim(X, 1));
    const long long N = halo_geti_or(a, "n_vocab", halo_dim(W, 0));
    if (K <= 0 || (K % 128) || N <= 0) return HALO_NO(RAD_E_SHAPE);
    /* The rows are X's, as libr4d's rad_logits_gemm_8 reads them; `M` is the band. */
    long long M = 0, My = 0;
    const long long x_ld = halo_rows_pitch(X, K, &M);
    const long long y_ld = halo_rows_pitch(Y, N, &My);
    if (x_ld < 0 || y_ld < 0) return HALO_NO(RAD_E_STRIDE);
    if (M <= 0 || My < M) return HALO_NO(RAD_E_SHAPE);

    const long stride = halo_fp8::lm_row_bytes(K);
    const uint16_t* xp = (const uint16_t*)halo_p(X);
    const uint8_t* wp = (const uint8_t*)halo_p(W);
    float* yp = (float*)halo_p(Y);
    logits_launch<I8>(xp, wp, yp, (int)M, (int)N, K, stride, x_ld, y_ld, (hipStream_t)stream);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

}  /* namespace */

extern "C" int halo_rad_logits_gemm_fp8(const RadArgs* a, RadStream s) {
    return logits_run<false>(a, s, RAD_F8E4M3);
}
extern "C" int halo_rad_logits_gemm_i8(const RadArgs* a, RadStream s) {
    return logits_run<true>(a, s, RAD_I8);
}

/* An IQR head (halo_iqr.h's lm form): the int8 head's kernels with each sixteen codes read at their
 * width and looked up in the row form's table (lm_codes16), so the arithmetic is the int8 head's
 * over the same values. */
extern "C" int halo_rad_logits_gemm_iqr(const RadArgs* a, RadStream s) {
    const RadTensor* X = halo_opt(a, 0);
    const RadTensor* W = halo_opt(a, 1);
    const RadTensor* Y = halo_opt(a, 2);
    if (!X || !W || !Y) return HALO_NO(RAD_E_INVAL);
    if (X->dtype != RAD_BF16 || Y->dtype != RAD_F32 || !halo_iqr::is_weight_dtype(W->dtype))
        return HALO_NO(RAD_E_DTYPE);
    const uint32_t word = W->dtype & halo_iqr::kWordMask;
    if (!halo_iqr::lm_built(word)) return HALO_NO(RAD_E_DTYPE);
    const halo_iqr::Form f = halo_iqr::form(word);
    const long long N = halo_geti_or(a, "n_vocab", halo_dim(W, 0));
    const long long K = halo_geti_or(a, "n_embd", 0);
    if (N <= 0 || K <= 0 || K % 128) return HALO_NO(RAD_E_SHAPE);
    if (W->rank != 2 || W->shape[0] != N || W->shape[1] != K) return HALO_NO(RAD_E_SHAPE);
    const long stride = (long)halo_iqr::lm_row(f, K);
    long long M = 0, My = 0;
    const long long x_ld = halo_rows_pitch(X, K, &M);
    const long long y_ld = halo_rows_pitch(Y, N, &My);
    if (x_ld < 0 || y_ld < 0) return HALO_NO(RAD_E_STRIDE);
    if (M <= 0 || My < M) return HALO_NO(RAD_E_SHAPE);
    if (((uintptr_t)halo_p(X) & 15) || ((uintptr_t)halo_p(W) & 255)) return HALO_NO(RAD_E_ALIGN);
    const uint16_t* xp = (const uint16_t*)halo_p(X);
    const uint8_t* wp = (const uint8_t*)halo_p(W);
    float* yp = (float*)halo_p(Y);
    const hipStream_t st = (hipStream_t)s;
    switch (f.bits) {
    case 2: logits_launch<true, 2>(xp, wp, yp, (int)M, (int)N, K, stride, x_ld, y_ld, st); break;
    case 3: logits_launch<true, 3>(xp, wp, yp, (int)M, (int)N, K, stride, x_ld, y_ld, st); break;
    case 4: logits_launch<true, 4>(xp, wp, yp, (int)M, (int)N, K, stride, x_ld, y_ld, st); break;
    case 5: logits_launch<true, 5>(xp, wp, yp, (int)M, (int)N, K, stride, x_ld, y_ld, st); break;
    default: logits_launch<true, 6>(xp, wp, yp, (int)M, (int)N, K, stride, x_ld, y_ld, st); break;
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* The raw-pointer entries libr4d exports beside the rows (halo.h): same kernel, the
 * stored row's pitch spelled out. Negative codes, as libr4d's: -1 a null pointer, -2 a
 * non-positive extent, -4 K not a whole 128-block, -5 a weight pitch not a multiple of
 * 16, -6 the launch failed. */
template <bool I8>
static int logits_entry(long a, long wq, long c, int M, int K, int N, int a_ld, int c_ld,
                        int w_ld, long stream) {
    if (!a || !wq || !c) return -1;
    if (M <= 0 || K <= 0 || N <= 0) return -2;
    if (K % 128) return -4;
    if (!a_ld) a_ld = K;
    if (!c_ld) c_ld = N;
    if (!w_ld) w_ld = (int)halo_fp8::lm_row_bytes(K);
    if (w_ld % 16) return -5;
    const uint16_t* xp = (const uint16_t*)(uintptr_t)a;
    const uint8_t* wp = (const uint8_t*)(uintptr_t)wq;
    float* yp = (float*)(uintptr_t)c;
    logits_launch<I8>(xp, wp, yp, M, N, K, w_ld, a_ld, c_ld, (hipStream_t)stream);
    return hipGetLastError() == hipSuccess ? 0 : -6;
}

extern "C" int halo_logits_gemm_fp8(long a, long wq, long c, int M, int K, int N,
                                    int a_ld, int c_ld, int w_ld, long stream) {
    return logits_entry<false>(a, wq, c, M, K, N, a_ld, c_ld, w_ld, stream);
}
extern "C" int halo_logits_gemm_i8(long a, long wq, long c, int M, int K, int N,
                                   int a_ld, int c_ld, int w_ld, long stream) {
    return logits_entry<true>(a, wq, c, M, K, N, a_ld, c_ld, w_ld, stream);
}
