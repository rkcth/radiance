/* Derived from libr4d/r4d_attn_paged_h256_gqa2.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// Halo paged attention dispatch for Gemma 4 dense's sliding-window layers: head_dim 256, 2 queries
// per KV head, paged block 16, window 1024 (50 of 60 layers). Instantiations of the h256 prefill
// kernel template with WIN = 1, as the DFlash2 drafter leg does at head_dim 128.
//
// Both bands: decode (q_len * gqa <= 64, so q_len <= 32) and the query-tiled prefill kernel. The
// split-KV decode kernel is a body of its own (halo_attn_decode_h256_gqa6.hip) and this unit
// instantiates it at the end, window bit on.
#include "halo.h"
#include "halo_args.h"
#include "halo_attn_args.h"
#include "halo_common.h"
#include <cmath>
#include <algorithm>
using std::max;
#include "halo_attn_prefill_h256_gqa6.hip"

#define CHK4() do { hipError_t e_ = hipGetLastError(); if (e_ != hipSuccess) return -100 - (int)e_; } while (0)

#define B_HEAD_DIM   256
#define B_GQA          2
#define B_BLOCK_SIZE  16

// ---- prefill ---------------------------------------------------------------------------------
// The h256/gqa6 prefill option set but at K prefetch 4 (P_OPT there is 3954187); O_KWIDE is valid
// at head_dim 256. At gqa 2 a block covers NWARPS * 16 / 2 = 192 query positions. WIN is on;
// window <= 0 gives full attention.
#define P_NWARPS 24
#define P_TILE   48
#define P_OPT    3955211

template <int KVP, int XOPT = 0>
static int prefill_launch_h256_gqa2(const HaloArgs* a, hipStream_t s) {
    if (a->head_dim != B_HEAD_DIM || a->block_size != B_BLOCK_SIZE) return -1;
    if (a->q_heads / a->kv_heads != B_GQA) return -2;
    constexpr int BQ = (P_NWARPS * 16) / B_GQA;
    dim3 grid((a->q_len + BQ - 1) / BQ, a->kv_heads, a->num_seqs);
    hipLaunchKernelGGL((halo_attn_prefill_kernel<P_NWARPS, P_TILE, B_HEAD_DIM, B_GQA,
                                                 B_BLOCK_SIZE, KVP, P_OPT | XOPT, 1>),
                       grid, dim3(P_NWARPS * 32), 0, s, *a);
    CHK4();
    return 0;
}

extern "C" int halo_attn_prefill_h256_gqa2_fp8kv(const HaloArgs* a, hipStream_t s) {
    return prefill_launch_h256_gqa2<0>(a, s);
}
extern "C" int halo_attn_prefill_h256_gqa2_bf16kv(const HaloArgs* a, hipStream_t s) {
    return prefill_launch_h256_gqa2<1>(a, s);
}
extern "C" int halo_attn_prefill_h256_gqa2_i8kv(const HaloArgs* a, hipStream_t s) {
    return prefill_launch_h256_gqa2<0, O_KVI8>(a, s);
}

/* Prefill RadArgs wrappers, defined in this unit rather than the h256 dispatch. Absent descales
 * mean 1.0 (fill_paged leaves them null). */
extern "C" int halo_rad_attn_prefill_h256_gqa2_fp8(const RadArgs* a, RadStream s) {
    HaloArgs r;
    int rc = fill_paged(a, &r, B_HEAD_DIM);
    if (rc != RAD_OK) return rc;
    return halo_attn_prefill_h256_gqa2_fp8kv(&r, (hipStream_t)s);
}
extern "C" int halo_rad_attn_prefill_h256_gqa2_bf16(const RadArgs* a, RadStream s) {
    HaloArgs r;
    int rc = fill_paged(a, &r, B_HEAD_DIM);
    if (rc != RAD_OK) return rc;
    return halo_attn_prefill_h256_gqa2_bf16kv(&r, (hipStream_t)s);
}
/* The int8 cache: a token's row is 2 * 256 + 4 = 516 bytes. The query-tiled kernel here runs one
 * split, so the scratch hook names no partials. */
extern "C" int halo_rad_attn_prefill_h256_gqa2_i8(const RadArgs* a, RadStream s) {
    HaloArgs r;
    int rc = fill_paged_row(a, &r, B_HEAD_DIM, 2 * B_HEAD_DIM + 4);
    if (rc != RAD_OK) return rc;
    return halo_attn_prefill_h256_gqa2_i8kv(&r, (hipStream_t)s);
}

// ============================================================== the split-KV decode leg
//
// Same geometry, the decode band (q_len * gqa <= 64, so q_len <= 32), with the window bit on: the
// body halo_attn_decode_h256_gqa6.hip instantiated at h256/gqa2 with WIN = 1, which picks the
// windowed split law and compiles the window walk in. window <= 0 is full attention.
//
// attn_paged_qrow reads the same geometry with q at HaloArgs::q_ld (O_QLD), its own instances so the
// dense pair above keeps its registers.
#include "halo_attn_decode_h256_gqa6.hip"

#define B_TILE HALO_D_TILE
#define B_OPT  HALO_D_OPT

extern "C" long halo_attn_decode_h256_gqa2_scratch_bytes(const HaloArgs* a) {
    return decode_scratch_bytes<B_OPT, 1>(a);
}

extern "C" int halo_attn_decode_h256_gqa2_fp8kv(const HaloArgs* a, hipStream_t s) {
    return decode_launch<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT, 0, 1>(a, s);
}
extern "C" int halo_attn_decode_h256_gqa2_bf16kv(const HaloArgs* a, hipStream_t s) {
    return decode_launch<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT, 1, 1>(a, s);
}
extern "C" int halo_attn_decode_h256_gqa2_qld_fp8kv(const HaloArgs* a, hipStream_t s) {
    return decode_launch<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT | O_QLD, 0, 1>(a, s);
}
extern "C" int halo_attn_decode_h256_gqa2_qld_bf16kv(const HaloArgs* a, hipStream_t s) {
    return decode_launch<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT | O_QLD, 1, 1>(a, s);
}
// The int8 cache: K is widened from the paged cache a fragment at a time.
extern "C" int halo_attn_decode_h256_gqa2_i8kv(const HaloArgs* a, hipStream_t s) {
    return decode_launch<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT | O_KVI8, 0, 1>(a, s);
}
extern "C" int halo_attn_decode_h256_gqa2_qld_i8kv(const HaloArgs* a, hipStream_t s) {
    return decode_launch<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT | O_KVI8 | O_QLD, 0, 1>(a, s);
}
// The same under attn_paged_qrow_tree: a tree verify's mask (O_TREE), causal only.
extern "C" int halo_attn_decode_h256_gqa2_qld_tree_bf16kv(const HaloArgs* a, hipStream_t s) {
    return decode_launch<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT | O_QLD | O_TREE, 1, 1>(a, s);
}
extern "C" int halo_attn_decode_h256_gqa2_qld_tree_i8kv(const HaloArgs* a, hipStream_t s) {
    return decode_launch<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT | O_KVI8 | O_QLD | O_TREE, 0, 1>(a,
                                                                                              s);
}

/* ---- the plugin ABI entry points of the decode band */

extern "C" int64_t halo_rad_attn_decode_h256_gqa2_scratch(const RadArgs* a) {
    return decode_scratch<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT, 1>(a);
}
extern "C" int halo_rad_attn_decode_h256_gqa2_fp8(const RadArgs* a, RadStream s) {
    HaloArgs r;
    int rc = decode_common<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT, 1>(a, &r);
    if (rc != RAD_OK) return rc;
    return halo_attn_decode_h256_gqa2_fp8kv(&r, (hipStream_t)s);
}
extern "C" int halo_rad_attn_decode_h256_gqa2_bf16(const RadArgs* a, RadStream s) {
    HaloArgs r;
    int rc = decode_common<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT, 1>(a, &r);
    if (rc != RAD_OK) return rc;
    return halo_attn_decode_h256_gqa2_bf16kv(&r, (hipStream_t)s);
}

/* The same decode under attn_paged_qrow: q read at its own row pitch (HaloArgs::q_ld), the sliding
 * layers' q where a fused q|k|v projection left it. Same kernel, same arithmetic. */
extern "C" int64_t halo_rad_attn_qrow_decode_h256_gqa2_scratch(const RadArgs* a) {
    return decode_scratch<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT, 1>(a, 0, 1, true);
}
extern "C" int halo_rad_attn_qrow_decode_h256_gqa2_fp8(const RadArgs* a, RadStream s) {
    HaloArgs r;
    int rc = decode_common<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT, 1>(a, &r, 0, 1, true);
    if (rc != RAD_OK) return rc;
    return halo_attn_decode_h256_gqa2_qld_fp8kv(&r, (hipStream_t)s);
}
extern "C" int halo_rad_attn_qrow_decode_h256_gqa2_bf16(const RadArgs* a, RadStream s) {
    HaloArgs r;
    int rc = decode_common<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT, 1>(a, &r, 0, 1, true);
    if (rc != RAD_OK) return rc;
    return halo_attn_decode_h256_gqa2_qld_bf16kv(&r, (hipStream_t)s);
}

/* The int8 cache (a token's row 2 * 256 + 4 bytes), plain and under attn_paged_qrow. */
extern "C" int64_t halo_rad_attn_decode_h256_gqa2_i8_scratch(const RadArgs* a) {
    return decode_scratch<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT, 1>(a, 2 * B_HEAD_DIM + 4);
}
extern "C" int halo_rad_attn_decode_h256_gqa2_i8(const RadArgs* a, RadStream s) {
    HaloArgs r;
    int rc = decode_common<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT, 1>(a, &r, 2 * B_HEAD_DIM + 4);
    if (rc != RAD_OK) return rc;
    return halo_attn_decode_h256_gqa2_i8kv(&r, (hipStream_t)s);
}
extern "C" int64_t halo_rad_attn_qrow_decode_h256_gqa2_i8_scratch(const RadArgs* a) {
    return decode_scratch<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT, 1>(a, 2 * B_HEAD_DIM + 4, 1, true);
}
extern "C" int halo_rad_attn_qrow_decode_h256_gqa2_i8(const RadArgs* a, RadStream s) {
    HaloArgs r;
    int rc = decode_common<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT, 1>(a, &r, 2 * B_HEAD_DIM + 4, 1,
                                                                       true);
    if (rc != RAD_OK) return rc;
    return halo_attn_decode_h256_gqa2_qld_i8kv(&r, (hipStream_t)s);
}

/* ...under attn_paged_qrow_tree: the same, with the tree verify's mask from `spec_parent` (operand
 * 8, one int32 a query row; docs/OPS.md). Causal only, as the op is. */
static int tree_operand(const RadArgs* a, HaloArgs* r) {
    const RadTensor* tp = a->n_t > 8 ? halo_opt(a, 8) : nullptr;
    if (!tp || !tp->data) return HALO_NO(RAD_E_INVAL);
    if (tp->dtype != RAD_I32) return HALO_NO(RAD_E_DTYPE);
    if (halo_numel(tp) < halo_q_total(*r)) return HALO_NO(RAD_E_SHAPE);
    if (r->causal != 1 || r->causal_seq) return HALO_NO(RAD_E_UNSUPPORTED);
    r->spec_parent = (const int*)tp->data;
    return RAD_OK;
}
extern "C" int halo_rad_attn_qrow_tree_decode_h256_gqa2_i8(const RadArgs* a, RadStream s) {
    HaloArgs r;
    int rc = decode_common<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT, 1>(a, &r, 2 * B_HEAD_DIM + 4, 1,
                                                                       true);
    if (rc == RAD_OK) rc = tree_operand(a, &r);
    if (rc != RAD_OK) return rc;
    return halo_attn_decode_h256_gqa2_qld_tree_i8kv(&r, (hipStream_t)s);
}
extern "C" int halo_rad_attn_qrow_tree_decode_h256_gqa2_bf16(const RadArgs* a, RadStream s) {
    HaloArgs r;
    int rc = decode_common<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT, 1>(a, &r, 0, 1, true);
    if (rc == RAD_OK) rc = tree_operand(a, &r);
    if (rc != RAD_OK) return rc;
    return halo_attn_decode_h256_gqa2_qld_tree_bf16kv(&r, (hipStream_t)s);
}
