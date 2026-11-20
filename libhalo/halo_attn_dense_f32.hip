/* Derived from libr4d/r4d_audio.hip (encoder_attn, r4d_rad_encoder_attn) by the libhalo rename
 * transform (2026-11-17); see libhalo/README.md. */
// attn_dense_h64_f32: f32 causal encoder attention without a global score plane. One workgroup a
// (query, head), at most 512 keys; only the reference audio encoder uses this f32 path, its
// sequence at most 250 rows. Plain f32 arithmetic with no wave-size or fragment dependence, so it
// ports unchanged; a unit of its own so the attention family's row does not wait on the audio
// unit's port (libr4d keeps it in r4d_audio.hip as r4d_rad_encoder_attn).
#include "halo.h"
#include "halo_args.h"
#include "halo_common.h"

#include <cmath>

namespace {

constexpr int kEncT = 256;

__device__ float enc_attn_sum(float v, float* buf) {
    const int t = threadIdx.x;
    buf[t] = v;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (t < s) buf[t] += buf[t + s];
        __syncthreads();
    }
    return buf[0];
}

/* One workgroup a (query, head), <=512 keys. Only the reference encoder uses this f32
 * attention path; its sequence is at most 250 rows, so no quadratic global score plane. */
__global__ __launch_bounds__(kEncT) void encoder_attn_f32(const float* q, const float* k,
                                                         const float* v, const int* cu,
                                                         float* out, int heads, int qp, int kp,
                                                         int vp, int op, int nseq, int div,
                                                         int window, float scale) {
    __shared__ float score[512], reduce[256];
    const int r = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
    (void)heads;
    int seq = 0;
    while (seq + 1 < nseq && cu[seq + 1] <= r * div) ++seq;
    const int first = max(cu[seq] / div, window > 0 ? r - window + 1 : 0);
    const int last = min(cu[seq + 1] / div, r + 1), count = last - first;
    if (count <= 0 || count > 512) return;
    float best = -INFINITY;
    for (int j = t; j < count; j += 256) {
        float dot = 0;
        for (int d = 0; d < 64; ++d)
            dot += q[(size_t)r * qp + h * 64 + d] * k[(size_t)(first + j) * kp + h * 64 + d];
        score[j] = dot * scale;
        best = fmaxf(best, score[j]);
    }
    reduce[t] = best;
    __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (t < s) reduce[t] = fmaxf(reduce[t], reduce[t + s]);
        __syncthreads();
    }
    best = reduce[0];
    __syncthreads();
    float sum = 0;
    for (int j = t; j < count; j += 256) { score[j] = expf(score[j] - best); sum += score[j]; }
    const float denom = enc_attn_sum(sum, reduce);
    __syncthreads();
    if (t < 64) {
        float result = 0;
        for (int j = 0; j < count; ++j)
            result += score[j] / denom * v[(size_t)(first + j) * vp + h * 64 + t];
        out[(size_t)r * op + h * 64 + t] = result;
    }
}

inline bool enc_f32(const RadTensor* t) { return t && t->dtype == RAD_F32; }
inline bool enc_i32(const RadTensor* t) { return t && t->dtype == RAD_I32; }

}  // namespace

/* attn_dense_h64_f32: q, k, v, cu_seqlens, out (r4d_rad_encoder_attn's operands and checks). */
extern "C" int halo_rad_attn_dense_h64_f32(const RadArgs* a, RadStream s) {
    const auto *q = halo_opt(a, 0), *k = halo_opt(a, 1), *v = halo_opt(a, 2),
               *cu = halo_opt(a, 3), *y = halo_opt(a, 4);
    if (!q || !k || !v || !cu || !y || !enc_f32(q) || !enc_f32(k) || !enc_f32(v) ||
        !enc_f32(y) || !enc_i32(cu)) return HALO_NO(RAD_E_DTYPE);
    if (halo_geti_or(a, "head_dim", 0) != 64 || halo_geti_or(a, "gqa", 1) != 1 ||
        halo_geti_or(a, "causal", 0) != 1) return HALO_NO(RAD_E_UNSUPPORTED);
    const long long n = halo_dim(q, (int)q->rank - 1), div = halo_geti_or(a, "cu_div", 1);
    if (n < 64 || n % 64 || n > 4096 || div < 1 || div > 0x7fffffffll) return HALO_NO(RAD_E_SHAPE);
    long long rq = 0, rk = 0, rv = 0, ry = 0;
    const auto qp = halo_rows_pitch(q, n, &rq), kp = halo_rows_pitch(k, n, &rk),
               vp = halo_rows_pitch(v, n, &rv), yp = halo_rows_pitch(y, n, &ry);
    if (qp < 0 || kp < 0 || vp < 0 || yp < 0 || !halo_contig(cu)) return HALO_NO(RAD_E_STRIDE);
    const auto ns = halo_numel(cu) - 1, window = halo_geti_or(a, "window", 0);
    if (ry < 1 || ry > 512 || rq < ry || rk < ry || rv < ry || ns < 1 || ns > 512 ||
        qp > 0x7fffffffll || kp > 0x7fffffffll || vp > 0x7fffffffll || yp > 0x7fffffffll ||
        window < 0 || window > 0x7fffffffll) return HALO_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(encoder_attn_f32, dim3((unsigned)ry, (unsigned)(n / 64)), dim3(kEncT), 0,
        (hipStream_t)s, (const float*)q->data, (const float*)k->data, (const float*)v->data,
        (const int*)cu->data, (float*)y->data, (int)(n / 64), (int)qp, (int)kp, (int)vp, (int)yp,
        (int)ns, (int)div, (int)window, (float)halo_getf(a, "scale", 0.125));
    return hipGetLastError() == hipSuccess ? RAD_OK : HALO_NO(RAD_E_DEVICE);
}
