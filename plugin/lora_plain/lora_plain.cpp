/* lora_plain -- kernel plugin: plain bf16 views of rotated int8 weight planes, so the core's
 * runtime LoRA machinery can attach adapters to W8A8-rotated trunk modules.
 *
 * The weights of the LoRA-targeted module kinds are stored as affine int8 codes with a bf16 (or
 * f16) scale per [1 x 128] and the "fwht128" transform: the stored plane holds W*H/128 per
 * 128-column block, H the unnormalised Sylvester Walsh-Hadamard matrix (libquant's transform,
 * docs "fwhtN": T = H_N / N block-diagonal over groups of 128 columns). The fused int8 GEMMs read
 * those planes against had_quant's rotated codes and never expose W itself.
 *
 * This plugin registers the plain fp GEMM op `gemm_nt` (dtype bf16) with a layout hook that
 * accepts that encoding and a relayout that decodes it to the plain basis at load:
 *
 *     plain_row = FWHT_128( dequant(row block) )        // (W*H/128)*H = W exactly, H^2 = 128 I
 *
 * rounded once to bf16. The launch delegates to libr4d's own exported bf16 GEMM entries, so the
 * arithmetic is byte-for-byte libr4d's; only the operand bytes differ.
 *
 * The core's site gate (core/lora/lora.cpp, op_is_site) accepts plain fp GEMMs over
 * transform-basis weights, so it injects lora_shrink / lora_expand around these GEMMs and runtime
 * PEFT adapters bind to them.
 *
 * Build: g++-14 -std=c++20 -O2 -fPIC -shared -I <sdk>/include/radiance lora_plain.cpp \
 *            -o <home>/kernels/lora_plain.so -ldl
 */
#include <dlfcn.h>
#include <math.h>
#include <mutex>
#include <stdint.h>
#include <string.h>
#include <cstdio>

#include "rad_abi.h"
#include "rad_abi_layout.h"
#include "rad_encoding.h"

RAD_PLUGIN_ABI_LAYOUT_EXPORT
extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }

extern "C" const RadPluginInfo* rad_plugin_info(void) {
    static const RadPluginInfo i = {
        RAD_PLUGIN_KERNEL, "lora_plain", "1.0",
        "plain bf16 relayouts of fwht128 w8 planes; launches delegate to libr4d bf16 GEMMs", "",
    };
    return &i;
}

/* ==================================================================== fp helpers */

static inline float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
    uint32_t u;
    if (exp == 0) {
        if (man == 0) { u = sign; }
        else {                              /* subnormal: renormalise */
            uint32_t m = man; int e = 0;
            while (!(m & 0x400u)) { m <<= 1; ++e; }
            m &= 0x3FFu;
            u = sign | ((uint32_t)(127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {
        u = sign | 0x7F800000u | (man << 13);
    } else {
        u = sign | ((uint32_t)(exp + (127 - 15)) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static inline uint16_t f32_to_bf16(float f) {   /* round-to-nearest-even, NaN-preserving */
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t)((u >> 16) | 0x0040u);  /* NaN quiet */
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return (uint16_t)(u >> 16);
}

static inline float scale_load(const void* base, uint32_t dtype, int64_t idx) {
    switch (dtype) {
        case RAD_BF16: return bf16_to_f32(((const uint16_t*)base)[idx]);
        case RAD_F16:  return f16_to_f32(((const uint16_t*)base)[idx]);
        case RAD_F32:  return ((const float*)base)[idx];
        default:       return 0.0f;
    }
}

/* libquant's fwht_inplace: unnormalised Sylvester WHT, butterfly a+b low and a-b high. */
static void fwht_inplace(float* v, int64_t n) {
    for (int64_t h = 1; h < n; h <<= 1)
        for (int64_t i = 0; i < n; i += h << 1)
            for (int64_t j = i; j < i + h; ++j) {
                const float a = v[j], b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
}

/* ==================================================================== encoding gate */

namespace {

struct Target {
    int i_codes, i_scale;         /* indices into the passed planes[] */
    uint32_t scale_dtype;
    int64_t N, K;
};

/* Recognise exactly the encoding this plugin converts: affine, fwht128, an int8 codes plane with
 * a block scale (bf16/f16/f32) per [1 x 128]. Returns false for everything else, which the hook
 * reports as RAD_E_UNSUPPORTED ("store as is") so all other weights pass through untouched. */
bool match_target(const RadEncoding* enc, const int* sel, const RadTensor* planes, int n_planes,
                  Target* out) {
    if (!enc || rad_enc_fwht(enc) != 128) return false;
    const RadEncPlane* codes = rad_enc_plane(enc, "codes");
    const RadEncPlane* scale = rad_enc_plane(enc, "scale");
    if (!codes || !scale) return false;
    if (codes->dtype != RAD_I8) return false;
    if (scale->dtype != RAD_BF16 && scale->dtype != RAD_F16 && scale->dtype != RAD_F32) return false;
    if (scale->block[0] != 1 || scale->block[1] != 128) return false;
    const int want_c = (int)(codes - enc->plane), want_s = (int)(scale - enc->plane);
    out->i_codes = out->i_scale = -1;
    for (int i = 0; i < n_planes; ++i) {
        if (sel[i] == want_c) out->i_codes = i;
        else if (sel[i] == want_s) out->i_scale = i;
    }
    if (out->i_codes < 0 || out->i_scale < 0) return false;
    const RadTensor* pc = &planes[out->i_codes];
    if (pc->rank != 2) return false;
    out->N = pc->shape[0];
    out->K = pc->shape[1];
    out->scale_dtype = scale->dtype;
    return out->N > 0 && out->K > 0 && (out->K % 128) == 0;
}

}  /* namespace */

/* ==================================================================== layout hooks */

namespace {

int layout(const RadParam*, int, int, const RadEncoding* enc, const int* sel,
           const RadTensor* planes, int n_planes, RadLayout* out) {
    Target t;
    if (!match_target(enc, sel, planes, n_planes, &t)) return RAD_E_UNSUPPORTED;  /* store as is */
    out->tag = "lora_plain.bf16.g128";
    out->dtype = RAD_BF16;
    out->rank = 2;
    out->shape[0] = t.N;
    out->shape[1] = t.K;
    out->align = 256;
    out->bytes = t.N * t.K * 2;
    return RAD_OK;
}

int relayout(const RadParam*, int, int, const RadEncoding* enc, const int* sel,
             const RadTensor* planes, int n_planes, void* dst, int64_t dst_bytes) {
    Target t;
    if (!match_target(enc, sel, planes, n_planes, &t)) return RAD_E_SHAPE;
    if (dst_bytes != t.N * t.K * 2) return RAD_E_SHAPE;
    const int8_t* codes = (const int8_t*)planes[t.i_codes].data;
    const void* scale = planes[t.i_scale].data;
    const int64_t srow = t.K / 128;
    uint16_t* out = (uint16_t*)dst;
    float v[128];
    for (int64_t n = 0; n < t.N; ++n) {
        for (int64_t b = 0; b < srow; ++b) {
            const float s = scale_load(scale, t.scale_dtype, n * srow + b);
            const int8_t* src = codes + n * t.K + b * 128;
            for (int i = 0; i < 128; ++i) v[i] = (float)src[i] * s;
            fwht_inplace(v, 128);
            uint16_t* d = out + n * t.K + b * 128;
            for (int i = 0; i < 128; ++i) d[i] = f32_to_bf16(v[i]);
        }
    }
    return RAD_OK;
}

}  /* namespace */

/* ==================================================================== launch delegation */

namespace {

typedef int (*LaunchFn)(const RadArgs*, RadStream);
typedef int (*TuneValidFn)(const RadParam*, int, const RadParam*, int);

void* g_r4d = nullptr;
std::once_flag g_once;

void init_r4d() {
    /* libr4d is the in-image kernel plugin; it is already loaded when we launch. Prefer the
     * resident copy (NOLOAD matches by soname), then the stock install path. */
    g_r4d = dlopen("libr4d.so", RTLD_NOLOAD | RTLD_NOW);
    if (!g_r4d) g_r4d = dlopen("/opt/radiance/share/radiance/kernels/libr4d.so", RTLD_NOW);
    if (!g_r4d)
        fprintf(stderr, "lora_plain: cannot resolve libr4d (%s); bf16 GEMM launches will fail\n",
                dlerror());
}

LaunchFn resolve(const char* sym) {
    std::call_once(g_once, init_r4d);
    if (!g_r4d) return nullptr;
    return (LaunchFn)dlsym(g_r4d, sym);
}

int launch(const RadArgs* a, RadStream s) {
    long long M = 0, N = 0, K = 0;
    rad_args_geti(a, "M", &M);
    rad_args_geti(a, "N", &N);
    rad_args_geti(a, "K", &K);
    const char* sym;
    if (M <= 16 && (K % 256) == 0) sym = "r4d_rad_gemm_bf16_m16";
    else if (M >= 65 && N >= 128 && (K % 64) == 0) sym = "r4d_rad_gemm_bf16_tiled";
    else sym = "r4d_rad_gemm_bf16_m64";
    LaunchFn fn = resolve(sym);
    if (!fn) return RAD_E_STATE;
    return fn(a, s);
}

int tune_valid(const RadParam* p, int n_p, const RadParam* choice, int n_c) {
    std::call_once(g_once, init_r4d);
    if (!g_r4d) return 1;
    TuneValidFn fn = (TuneValidFn)dlsym(g_r4d, "r4d_tune_valid_gemm");
    return fn ? fn(p, n_p, choice, n_c) : 1;
}

}  /* namespace */

/* ==================================================================== schema + rows */

namespace {

const RadParamSpec pGemmNt[] = {
    {"M", RAD_P_INT, RAD_REQUIRED},
    {"N", RAD_P_INT, RAD_REQUIRED},
    {"K", RAD_P_INT, RAD_REQUIRED},
    {"dtype", RAD_P_STR, RAD_REQUIRED},
};
const RadOperandSpec oGemmNt[] = {
    {"a", RAD_OPD_IN, 0},
    {"b", RAD_OPD_WEIGHT, 0},
    {"y", RAD_OPD_OUT, 0},
};

const RadOpSchema kSchemas[] = {
    {"gemm_nt", pGemmNt, 4, oGemmNt, 3, "C[M,N] = A[M,K] @ B[N,K]^T, unquantised."},
};

const RadConstraint cNt[] = {
    {"M", RAD_C_LE, 16, nullptr},
    {"K", RAD_C_DIV, 256, nullptr},
    {"dtype", RAD_C_IN, 0, "bf16"},
};
const RadConstraint cNt64[] = {
    {"M", RAD_C_LE, 2147483647, nullptr},
    {"K", RAD_C_DIV, 16, nullptr},
    {"dtype", RAD_C_IN, 0, "bf16"},
};
const RadConstraint cNtTiled[] = {
    {"M", RAD_C_GE, 65, nullptr},
    {"M", RAD_C_LE, 2147483647, nullptr},
    {"N", RAD_C_GE, 128, nullptr},
    {"K", RAD_C_DIV, 64, nullptr},
    {"dtype", RAD_C_IN, 0, "bf16"},
};

const int64_t kWv[] = {1, 2, 4};
const int64_t kSk[] = {0, 1, 2, 4, 5, 8, 10, 16, 20, 32};
const int64_t kMb[] = {0, 1, 2, 3, 4};
const int64_t kNpw[] = {1, 2, 4};
const int64_t kNtst[] = {0, 1};

const RadTunable tSkinny[] = {
    {"wv", kWv, 3, 1, "waves per block; wv*sk*32 must stay inside the block bound"},
    {"sk", kSk, 10, 0, "split-K segment count, reduced in LDS. 0 = the shape's own ladder"},
    {"mb", kMb, 5, 0, "16-row tiles one block covers; grid.y is ceil(ceil(M/16)/mb). 0 = all"},
    {"npw", kNpw, 3, 1, "N tiles per wave; widens the block to hide a dequantise"},
    {"nt", kNtst, 2, 1, "non-temporal weight loads"},
};
#define NELEM(a) ((int)(sizeof(a) / sizeof(a[0])))

/* Priorities mirror libr4d's band resolution: m16 wins the tiny-M band, tiled wins M >= 65,
 * m64 is the fallback that claims every other shape. */
const RadKernelInfo kRows[] = {
    {"lora_plain.gemm_bf16_m16", "gemm_nt", "gemm",
     "delegates to libr4d's bf16 skinny m16 GEMM over planes this plugin relaid out to plain bf16",
     "M <= 16; K divisible by SK and K/SK by 256; N, K otherwise runtime", "bf16 A / W / C",
     RAD_DOMAIN_DEVICE, 3,
     cNt, NELEM(cNt), tSkinny, NELEM(tSkinny), tune_valid,
     nullptr, nullptr, launch, nullptr, layout, relayout,
     nullptr, nullptr, nullptr, 0},
    {"lora_plain.gemm_bf16_m64", "gemm_nt", "gemm",
     "delegates to libr4d's bf16 skinny m64 GEMM over planes this plugin relaid out to plain bf16",
     "signed-int M, bounded row launches; K divisible by SK*16", "bf16 A / W / C",
     RAD_DOMAIN_DEVICE, 1,
     cNt64, NELEM(cNt64), tSkinny, NELEM(tSkinny), tune_valid,
     nullptr, nullptr, launch, nullptr, layout, relayout,
     nullptr, nullptr, nullptr, 0},
    {"lora_plain.gemm_bf16_tiled", "gemm_nt", "gemm",
     "delegates to libr4d's bf16 prefill tiled GEMM over planes this plugin relaid out to plain bf16",
     "64 < M; N >= 128; K % 64", "bf16 A / W / C",
     RAD_DOMAIN_DEVICE, 2,
     cNtTiled, NELEM(cNtTiled), tSkinny, NELEM(tSkinny), tune_valid,
     nullptr, nullptr, launch, nullptr, layout, relayout,
     nullptr, nullptr, nullptr, 0},
};

}  /* namespace */

extern "C" int rad_kernel_schema_count(void) { return NELEM(kSchemas); }
extern "C" const RadOpSchema* rad_kernel_schema_at(int i) {
    return (i >= 0 && i < NELEM(kSchemas)) ? &kSchemas[i] : nullptr;
}
extern "C" int rad_kernel_count(void) { return NELEM(kRows); }
extern "C" const RadKernelInfo* rad_kernel_at(int i) {
    return (i >= 0 && i < NELEM(kRows)) ? &kRows[i] : nullptr;
}
