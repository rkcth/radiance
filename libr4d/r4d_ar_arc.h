#pragma once
// r4d_ar_arc.h: the lossless chunk codec of the coded all-reduce pushes, ~12.1 bits a bf16 value
// instead of 16. Shared by the four-rank coded two-shot (r4d_ar_twoshot_Nrank_exact.hip, where the
// format and its measurements are described) and the two-rank coded fused writes
// (r4d_ar_oneshot_2rank_exact.hip); each lays the coded words out in its own slot.
//
// A chunk is one int4 word (8 bf16 values); 8 consecutive chunks (64 values, 8 adjacent lanes of a
// wave) form a group that shares a base exponent, and each value keeps its sign, its 7 mantissa
// bits and a 4-bit exponent offset (0 = an exact zero, 1..15 = base .. base + 14): 8 values to 3
// dwords. A group holding a subnormal, an inf or NaN, or an exponent range past 14 codes to base
// 0xFF and its chunks travel raw. Decoding gives back the same 16 bits of every value.
#include <hip/hip_runtime.h>

__device__ __forceinline__ unsigned r4d_arc_shfl(unsigned v, int src) {
  return (unsigned)__builtin_amdgcn_ds_bpermute(src << 2, (int)v);
}
struct R4dArcEnc { unsigned w[3]; unsigned base; };
// Code this lane's chunk; every lane of the wave calls (a lane past the end with valid false).
__device__ __forceinline__ R4dArcEnc r4d_arc_enc(int4 v, bool valid) {
  const unsigned d[4] = {(unsigned)v.x, (unsigned)v.y, (unsigned)v.z, (unsigned)v.w};
  unsigned emin = 255u, emax = 0u, bad = 0u;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const unsigned u = (d[i >> 1] >> ((i & 1) * 16)) & 0xffffu;
    const unsigned e = (u >> 7) & 0xffu, m = u & 0x7fu;
    if (e) { emin = e < emin ? e : emin; emax = e > emax ? e : emax; }
    bad |= (unsigned)(e == 0u && m != 0u) | (unsigned)(e == 255u);
  }
  if (!valid) { emin = 255u; emax = 0u; bad = 0u; }
  const int lane = (int)(threadIdx.x & 31u);
#pragma unroll
  for (int s = 1; s < 8; s <<= 1) {
    const unsigned a = r4d_arc_shfl(emin, lane ^ s), b = r4d_arc_shfl(emax, lane ^ s);
    const unsigned c = r4d_arc_shfl(bad, lane ^ s);
    emin = a < emin ? a : emin; emax = b > emax ? b : emax; bad |= c;
  }
  R4dArcEnc r;
  const unsigned base = emin == 255u ? 1u : emin;
  r.base = (bad || emax > base + 14u) ? 0xFFu : base;
  unsigned c[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const unsigned u = (d[i >> 1] >> ((i & 1) * 16)) & 0xffffu;
    const unsigned e = (u >> 7) & 0xffu, m = u & 0x7fu, sg = u >> 15;
    c[i] = (sg << 11) | (m << 4) | (e ? (e - base + 1u) & 15u : 0u);
  }
  r.w[0] = c[0] | (c[1] << 12) | (c[2] << 24);
  r.w[1] = (c[2] >> 8) | (c[3] << 4) | (c[4] << 16) | (c[5] << 28);
  r.w[2] = (c[5] >> 4) | (c[6] << 8) | (c[7] << 20);
  return r;
}
__device__ __forceinline__ int4 r4d_arc_dec(const unsigned (&w)[3], unsigned base) {
  unsigned c[8];
  c[0] = w[0] & 0xfffu; c[1] = (w[0] >> 12) & 0xfffu; c[2] = (w[0] >> 24) | ((w[1] & 0xfu) << 8);
  c[3] = (w[1] >> 4) & 0xfffu; c[4] = (w[1] >> 16) & 0xfffu; c[5] = (w[1] >> 28) | ((w[2] & 0xffu) << 4);
  c[6] = (w[2] >> 8) & 0xfffu; c[7] = w[2] >> 20;
  unsigned u[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const unsigned off = c[i] & 15u;
    u[i] = ((c[i] >> 11) << 15) | ((off ? base + off - 1u : 0u) << 7) | ((c[i] >> 4) & 0x7fu);
  }
  int4 v;
  v.x = (int)(u[0] | (u[1] << 16)); v.y = (int)(u[2] | (u[3] << 16));
  v.z = (int)(u[4] | (u[5] << 16)); v.w = (int)(u[6] | (u[7] << 16));
  return v;
}
