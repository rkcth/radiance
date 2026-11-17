/* gen_lora_test.cpp -- runtime LoRA for generators: adapter files in every spelling and kind
 * (core/lora/adapter.h), their keys mapped through a generator's declared names, stacked at
 * strengths, packed per rank; and the request fields that name them (core/server/loras.h).
 *
 * Host only, no kernel plugin: the Program is made by the core Builder over an empty Registry, as
 * lora_test.cpp makes its own, with the generator's names declared through the builder (the calls
 * rad_lora_root / rad_lora_alias make). The reference is the test's own dense delta of every
 * checkpoint module, placed by model semantics -- which rows of which tensor a rank's declared
 * weight holds -- never by anything adapter.cpp or lora.cpp computes.
 */
#include "rad_test.h"

#include "build/rad_build.h"
#include "lora/adapter.h"
#include "lora/cpu_blas.h"
#include "lora/lora.h"
#include "plugin/registry.h"
#include "rad_plugin.h"
#include "server/loras.h"

#include <filesystem>
#include <arch/rad_arch.h>
#include <arch/rad_fp8.h>

#include "util/json.h"

#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <vector>
#include <sys/stat.h>

using namespace rad;
namespace AR = rad::arch;
using ojson = Json;

namespace {

Registry& empty_registry() {
    static Registry r;
    return r;
}

float val(uint32_t seed, uint64_t i) {
    uint64_t x = (uint64_t)seed * 0x9E3779B97F4A7C15ull + i * 0xBF58476D1CE4E5B9ull + 1;
    x ^= x >> 31; x *= 0x94D049BB133111EBull; x ^= x >> 29;
    return (float)((double)(x >> 11) / (double)(1ull << 53) - 0.5) * 0.5f;
}

/* ------------------------------------------------------------------ safetensors */
struct TW {
    std::string name, dtype;
    std::vector<int64_t> shape;
    std::vector<float> v;        /* written as dtype */
};
void write_st(const std::string& path, const std::vector<TW>& ts) {
    ojson h = ojson::object();
    std::vector<std::vector<uint8_t>> bytes;
    uint64_t off = 0;
    for (const TW& t : ts) {
        std::vector<uint8_t> b;
        if (t.dtype == "I64") { const int64_t v = 4; b.resize(8); std::memcpy(b.data(), &v, 8); }
        for (float f : t.v) {
            if (t.dtype == "F32") { uint8_t x[4]; std::memcpy(x, &f, 4); b.insert(b.end(), x, x + 4); }
            else {
                const uint16_t u = t.dtype == "F16" ? rad_f32_to_f16(f) : rad_f32_to_bf16(f);
                uint8_t x[2]; std::memcpy(x, &u, 2); b.insert(b.end(), x, x + 2);
            }
        }
        h[t.name] = { { "dtype", t.dtype }, { "shape", t.shape },
                      { "data_offsets", { off, off + b.size() } } };
        off += b.size();
        bytes.push_back(std::move(b));
    }
    h["__metadata__"] = { { "format", "pt" } };
    std::string s = h.dump();
    while (s.size() % 8) s += ' ';
    std::ofstream f(path, std::ios::binary);
    const uint64_t n = s.size();
    f.write((const char*)&n, 8);
    f.write(s.data(), (std::streamsize)s.size());
    for (const auto& b : bytes) f.write((const char*)b.data(), (std::streamsize)b.size());
}
/* A tensor of seeded values, and the values as the file holds them. */
TW tensor(const std::string& name, const std::string& dt, std::vector<int64_t> shape, uint32_t seed,
          float mul = 1.0f) {
    TW t{ name, dt, shape, {} };
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    for (int64_t i = 0; i < n; ++i) {
        float f = val(seed, (uint64_t)i) * mul;
        if (dt == "F16") f = rad_f16_to_f32(rad_f32_to_f16(f));
        else if (dt == "BF16") f = rad_bf16_to_f32(rad_f32_to_bf16(f));
        t.v.push_back(f);
    }
    return t;
}
TW scalar(const std::string& name, float f) { return TW{ name, "F32", {}, { f } }; }

/* ------------------------------------------------------------------ dense helpers */
struct Dense {
    int64_t N = 0, K = 0;
    std::vector<double> d;
    void add(const Dense& o, double w) {
        if (d.empty()) { N = o.N; K = o.K; d.assign(o.d.size(), 0.0); }
        for (size_t i = 0; i < d.size(); ++i) d[i] += w * o.d[i];
    }
};
Dense mul(const std::vector<float>& B, const std::vector<float>& A, int64_t N, int64_t r, int64_t K,
          double s) {
    Dense o{ N, K, std::vector<double>((size_t)(N * K), 0.0) };
    for (int64_t n = 0; n < N; ++n)
        for (int64_t j = 0; j < r; ++j)
            for (int64_t k = 0; k < K; ++k)
                o.d[(size_t)(n * K + k)] += s * (double)B[(size_t)(n * r + j)] * A[(size_t)(j * K + k)];
    return o;
}

/* ------------------------------------------------------------------ the generator */
/* A two-block DiT under "transformer.", as an image generator's plugin declares one: a fused
 * [q|k|v] (row parts, ROW), the output (COL), a SwiGLU re-laid from one checkpoint tensor whose
 * value half comes first (RAD_MAP_ROWS: the gate half moved first), the down projection (COL); a
 * text encoder's linear outside the root; and a DiT weight a fused GEMM reads. */
const int64_t kD = 128, kHD = 32, kNH = 8, kF = 512, kQ = kNH * kHD;

RadModelMeta gen_meta() {
    RadModelMeta m{};
    m.arch_id = "gentest"; m.name = "lora-test-gen"; m.quant = "fp8_e4m3";
    m.n_layers = 2; m.n_embd = kD; m.n_head = kNH; m.n_head_kv = kNH; m.head_dim = kHD;
    m.n_ff = kF; m.n_vocab = 512; m.n_ctx_train = 1024;
    m.rms_eps = 1e-5f; m.rope_theta = 1e4f; m.rope_scale = 1.0f;
    return m;
}

OpInfo gemm_op(const char* name, int64_t N, int64_t K, const char* dtype, int64_t hi,
               std::vector<rad_weight> w, std::vector<int32_t> opd) {
    OpInfo o;
    o.op = name;
    o.base.set_i("N", N);
    o.base.set_i("K", K);
    o.base.set_i("group", AR::RAD_FP8_BLOCK);
    o.base.set_s("dtype", dtype);
    o.ranged_key = "M";
    o.range_lo = 1;
    o.range_hi = hi;
    o.weights = std::move(w);
    o.weight_opd = std::move(opd);
    return o;
}
void push_op(Builder& b, OpInfo o) {
    o.index = (int32_t)b.program().ops.size();
    b.program().ops.push_back(std::move(o));
}

int linear(Builder& b, AR::Names& nm, const AR::Geom& g, const char* base, int64_t N, int64_t K,
           int shard, int layer, std::initializer_list<const char*> src,
           std::initializer_list<int64_t> parts = {}, bool mapped = false,
           const char* op = "gemm_nt_q") {
    AR::LinearFP8 lin{};
    AR::ActFP8 act{};
    lin.mapped = mapped;
    const int st = lin.declare((RadBuilder*)&b, nm, g, base, N, K, shard, AR::grp_layer(layer), src,
                               act, 0, 0, parts);
    if (st < 0 || !lin.w || !lin.ws) return st < 0 ? st : RAD_E_INVAL;
    push_op(b, gemm_op(op, N, K, "fp8a8", g.max_tok, { lin.w, lin.ws }, { 2, 3 }));
    return RAD_OK;
}

RadBuildCtx ctx_for(int rank, int world) {
    RadBuildCtx c{};
    c.rank = rank;
    c.world_size = world;
    c.max_tok = 256;
    c.max_seqs = 8;
    c.max_ctx = 1024;
    c.scope = "";
    return c;
}

/* Where a declared weight's rows on one rank come from: (checkpoint module, its full rows) per
 * row of y, and the full K columns of x. */
struct Truth {
    std::vector<std::pair<std::string, int64_t>> rows;
    std::vector<int64_t> cols;
};

int build_gen(int rank, int world, Program* out, std::map<std::string, Truth>* truth) {
    const RadModelMeta meta = gen_meta();
    const RadBuildCtx ctx = ctx_for(rank, world);
    AR::Geom g{};
    RAD_ARCH_TRY(AR::geom_from(g, &meta, &ctx, "bf16", RAD_BF16));
    Builder b(empty_registry(), meta, ctx);
    b.program().generator.layout = RAD_GEN_LAYOUT_JOINT_STEP;
    RAD_ARCH_TRY(AR::lora_names((RadBuilder*)&b, "transformer.", AR::kLoraDiffusionSpellings,
                                { { "blocks.#.attn.qkv", "transformer_blocks.#.attn.to_q", 0, kQ, 0 },
                                  { "blocks.#.attn.qkv", "transformer_blocks.#.attn.to_k", kQ, 2 * kQ, 0 },
                                  { "blocks.#.attn.qkv", "transformer_blocks.#.attn.to_v", 2 * kQ, 3 * kQ, 0 },
                                  { "blocks.#.attn.out_proj", "transformer_blocks.#.attn.to_out.0" },
                                  { "blocks.#.mlp.fc1", "transformer_blocks.#.ff.net.0.proj", 0, kF, kF },
                                  { "blocks.#.mlp.fc1", "transformer_blocks.#.ff.net.0.proj", kF, 2 * kF, 0 },
                                  { "blocks.#.mlp.fc2", "transformer_blocks.#.ff.net.2" } }));
    AR::Names nm("");
    const int64_t qr = g.q_dim(), fr = g.n_ff;
    for (int l = 0; l < 2; ++l) {
        const std::string P = "transformer.transformer_blocks." + std::to_string(l) + ".";
        auto ck = [&](const char* leaf) { return nm.ckpt("%s%s.weight", P.c_str(), leaf); };
        RAD_ARCH_TRY(linear(b, nm, g, nm.f("dit.%d.qkv", l), 3 * qr, kD, RAD_SHARD_ROW, l,
                            { ck("attn.to_q"), ck("attn.to_k"), ck("attn.to_v") }, { qr, qr, qr }));
        RAD_ARCH_TRY(linear(b, nm, g, nm.f("dit.%d.o", l), kD, qr, RAD_SHARD_COL, l,
                            { ck("attn.to_out.0") }));
        const char* fsrc = ck("ff.net.0.proj");
        RAD_ARCH_TRY(AR::map_rows((RadBuilder*)&b, nm.f("dit.%d.gate_up.weight", l),
                                  { { fsrc, (int)kF, kF }, { fsrc, 0, kF } }));
        RAD_ARCH_TRY(linear(b, nm, g, nm.f("dit.%d.gate_up", l), 2 * fr, kD, RAD_SHARD_ROW, l,
                            { fsrc }, { fr, fr }, true));
        RAD_ARCH_TRY(linear(b, nm, g, nm.f("dit.%d.down", l), kD, fr, RAD_SHARD_COL, l,
                            { ck("ff.net.2") }));

        Truth q, o, gu, dn;
        const char* part[3] = { "attn.to_q", "attn.to_k", "attn.to_v" };
        for (const char* p : part)
            for (int64_t i = rank * qr; i < (rank + 1) * qr; ++i) q.rows.push_back({ P + p, i });
        for (int64_t k = 0; k < kD; ++k) q.cols.push_back(k);
        for (int64_t i = 0; i < kD; ++i) o.rows.push_back({ P + "attn.to_out.0", i });
        for (int64_t k = rank * qr; k < (rank + 1) * qr; ++k) o.cols.push_back(k);
        for (int64_t i = rank * fr; i < (rank + 1) * fr; ++i) gu.rows.push_back({ P + "ff.net.0.proj", kF + i });
        for (int64_t i = rank * fr; i < (rank + 1) * fr; ++i) gu.rows.push_back({ P + "ff.net.0.proj", i });
        for (int64_t k = 0; k < kD; ++k) gu.cols.push_back(k);
        for (int64_t i = 0; i < kD; ++i) dn.rows.push_back({ P + "ff.net.2", i });
        for (int64_t k = rank * fr; k < (rank + 1) * fr; ++k) dn.cols.push_back(k);
        const std::string L = std::to_string(l);
        (*truth)["dit." + L + ".qkv.weight"] = q;
        (*truth)["dit." + L + ".o.weight"] = o;
        (*truth)["dit." + L + ".gate_up.weight"] = gu;
        (*truth)["dit." + L + ".down.weight"] = dn;
    }
    /* Outside the root: a text encoder's projection. */
    RAD_ARCH_TRY(linear(b, nm, g, "te.0.q", kD, kD, RAD_SHARD_NONE, 0,
                        { "text_encoder.layers.0.q_proj.weight" }));
    /* Under the root, read by a fused GEMM: it stays the base. */
    RAD_ARCH_TRY(linear(b, nm, g, "dit.fused", kD, kD, RAD_SHARD_NONE, 0,
                        { "transformer.proj_out.weight" }, {}, false, "gemm_nt_q_swiglu"));
    *out = std::move(b.program());
    return RAD_OK;
}

std::string new_dir(const std::string& name) {
    const std::string d = radtest::tmp_path(name);
    mkdir(d.c_str(), 0755);
    return d;
}

/* ------------------------------------------------------------------ the adapter files */
/* Two files, every spelling and kind, and the dense delta of every checkpoint module they reach
 * (strength 1). File 1: kohya LoRA on block 0's fused qkv; ComfyUI LoRA on block 0's fc1 (gate
 * half first); PEFT LoRA on block 1's down projection under the checkpoint's own name; LoHa on
 * block 1's out_proj; a full delta on block 0's fc2; a module of no model; a bias delta. File 2:
 * LoKr on block 1's qkv; a bare-keyed LoRA on block 0's qkv, stacking with file 1's. */
struct Files {
    std::string f1, f2;
    std::map<std::string, Dense> d1, d2;   /* by checkpoint module, full rows */
};

void place(std::map<std::string, Dense>* m, const std::string& mod, int64_t N_full, const Dense& d,
           int64_t from_lo, int64_t n, int64_t to_lo) {
    Dense& t = (*m)[mod];
    if (t.d.empty()) { t.N = N_full; t.K = d.K; t.d.assign((size_t)(N_full * d.K), 0.0); }
    for (int64_t r = 0; r < n; ++r)
        for (int64_t k = 0; k < d.K; ++k)
            t.d[(size_t)((to_lo + r) * d.K + k)] += d.d[(size_t)((from_lo + r) * d.K + k)];
}

Files make_files() {
    Files f;
    const std::string dir = new_dir("genlora");
    const std::string B0 = "transformer.transformer_blocks.0.", B1 = "transformer.transformer_blocks.1.";
    std::vector<TW> ts;
    {   /* kohya LoRA, rank 8, alpha 4: qkv of block 0 */
        TW dn = tensor("lora_unet_blocks_0_attn_qkv.lora_down.weight", "F16", { 8, kD }, 1);
        TW up = tensor("lora_unet_blocks_0_attn_qkv.lora_up.weight", "BF16", { 3 * kQ, 8 }, 2);
        const Dense d = mul(up.v, dn.v, 3 * kQ, 8, kD, 4.0 / 8.0);
        place(&f.d1, B0 + "attn.to_q", kQ, d, 0, kQ, 0);
        place(&f.d1, B0 + "attn.to_k", kQ, d, kQ, kQ, 0);
        place(&f.d1, B0 + "attn.to_v", kQ, d, 2 * kQ, kQ, 0);
        ts.push_back(dn); ts.push_back(up);
        TW al{ "lora_unet_blocks_0_attn_qkv.alpha", "I64", {}, {} };   /* as kohya writes some */
        ts.push_back(al);
    }
    {   /* ComfyUI LoRA, rank 4, no alpha: fc1 of block 0, [gate; up] */
        TW a = tensor("diffusion_model.blocks.0.mlp.fc1.lora_A.weight", "BF16", { 4, kD }, 3);
        TW b = tensor("diffusion_model.blocks.0.mlp.fc1.lora_B.weight", "F32", { 2 * kF, 4 }, 4);
        const Dense d = mul(b.v, a.v, 2 * kF, 4, kD, 1.0);
        place(&f.d1, B0 + "ff.net.0.proj", 2 * kF, d, 0, kF, kF);
        place(&f.d1, B0 + "ff.net.0.proj", 2 * kF, d, kF, kF, 0);
        ts.push_back(a); ts.push_back(b);
    }
    {   /* PEFT LoRA under the checkpoint's own name, rank 6, alpha 12 */
        TW a = tensor(B1 + "ff.net.2.lora_A.default.weight", "F32", { 6, kF }, 5);
        TW b = tensor(B1 + "ff.net.2.lora_B.default.weight", "F32", { kD, 6 }, 6);
        place(&f.d1, B1 + "ff.net.2", kD, mul(b.v, a.v, kD, 6, kF, 12.0 / 6.0), 0, kD, 0);
        ts.push_back(a); ts.push_back(b); ts.push_back(scalar(B1 + "ff.net.2.alpha", 12.0f));
    }
    {   /* LoHa, ranks 4 and 3, alpha 2: out_proj of block 1 */
        const std::string k = "diffusion_model.blocks.1.attn.out_proj.";
        TW w1a = tensor(k + "hada_w1_a", "F32", { kD, 4 }, 7), w1b = tensor(k + "hada_w1_b", "F32", { 4, kQ }, 8);
        TW w2a = tensor(k + "hada_w2_a", "F32", { kD, 3 }, 9), w2b = tensor(k + "hada_w2_b", "F32", { 3, kQ }, 10);
        const Dense p1 = mul(w1a.v, w1b.v, kD, 4, kQ, 1.0), p2 = mul(w2a.v, w2b.v, kD, 3, kQ, 1.0);
        Dense d{ kD, kQ, std::vector<double>(p1.d.size()) };
        for (size_t i = 0; i < d.d.size(); ++i) d.d[i] = 0.5 * p1.d[i] * p2.d[i];
        place(&f.d1, B1 + "attn.to_out.0", kD, d, 0, kD, 0);
        ts.insert(ts.end(), { w1a, w1b, w2a, w2b, scalar(k + "alpha", 2.0f) });
    }
    {   /* a full delta: fc2 of block 0 */
        TW df = tensor("diffusion_model.blocks.0.mlp.fc2.diff", "F32", { kD, kF }, 11, 0.1f);
        Dense d{ kD, kF, std::vector<double>(df.v.begin(), df.v.end()) };
        place(&f.d1, B0 + "ff.net.2", kD, d, 0, kD, 0);
        ts.push_back(df);
    }
    ts.push_back(tensor("diffusion_model.nowhere.lora_A.weight", "F32", { 2, 8 }, 12));
    ts.push_back(tensor("diffusion_model.nowhere.lora_B.weight", "F32", { 8, 2 }, 13));
    ts.push_back(tensor("diffusion_model.blocks.0.mlp.fc2.diff_b", "F32", { kD }, 14));
    f.f1 = dir + "/one.safetensors";
    write_st(f.f1, ts);

    ts.clear();
    {   /* LoKr: w1 [3, 2] full, w2 = w2_a [kQ, 4] @ w2_b [4, kD / 2], alpha 8 (scale 8 / 4) */
        const std::string k = "lora_unet_blocks_1_attn_qkv.";
        TW w1 = tensor(k + "lokr_w1", "F32", { 3, 2 }, 15);
        TW w2a = tensor(k + "lokr_w2_a", "F32", { kQ, 4 }, 16), w2b = tensor(k + "lokr_w2_b", "F32", { 4, kD / 2 }, 17);
        const Dense w2 = mul(w2a.v, w2b.v, kQ, 4, kD / 2, 1.0);
        Dense d{ 3 * kQ, kD, std::vector<double>((size_t)(3 * kQ * kD)) };
        for (int64_t i1 = 0; i1 < 3; ++i1)
            for (int64_t j1 = 0; j1 < 2; ++j1)
                for (int64_t i2 = 0; i2 < kQ; ++i2)
                    for (int64_t j2 = 0; j2 < kD / 2; ++j2)
                        d.d[(size_t)((i1 * kQ + i2) * kD + j1 * (kD / 2) + j2)] =
                            2.0 * w1.v[(size_t)(i1 * 2 + j1)] * w2.d[(size_t)(i2 * (kD / 2) + j2)];
        place(&f.d2, B1 + "attn.to_q", kQ, d, 0, kQ, 0);
        place(&f.d2, B1 + "attn.to_k", kQ, d, kQ, kQ, 0);
        place(&f.d2, B1 + "attn.to_v", kQ, d, 2 * kQ, kQ, 0);
        ts.insert(ts.end(), { w1, w2a, w2b, scalar(k + "alpha", 8.0f) });
    }
    {   /* bare keys, rank 8: qkv of block 0, stacked on file 1's */
        TW a = tensor("blocks.0.attn.qkv.lora_A.weight", "BF16", { 8, kD }, 18);
        TW b = tensor("blocks.0.attn.qkv.lora_B.weight", "BF16", { 3 * kQ, 8 }, 19);
        const Dense d = mul(b.v, a.v, 3 * kQ, 8, kD, 1.0);
        place(&f.d2, B0 + "attn.to_q", kQ, d, 0, kQ, 0);
        place(&f.d2, B0 + "attn.to_k", kQ, d, kQ, kQ, 0);
        place(&f.d2, B0 + "attn.to_v", kQ, d, 2 * kQ, kQ, 0);
        ts.push_back(a); ts.push_back(b);
    }
    f.f2 = dir + "/two.safetensors";
    write_st(f.f2, ts);
    return f;
}

/* A packed slot's product, [N_r, K_r]. */
std::vector<double> slot_delta(const lora::Site& s, const lora::PackedSite& p) {
    std::vector<double> d((size_t)(s.N_r * s.K_r), 0.0);
    for (int64_t y = 0; y < s.N_r; ++y) {
        for (int64_t j = 0; j < s.R; ++j) {
            const double b = rad_bf16_to_f32(p.b[(size_t)(y * s.R + j)]);
            if (b == 0.0) continue;
            for (int64_t k = 0; k < s.K_r; ++k)
                d[(size_t)(y * s.K_r + k)] += b * rad_bf16_to_f32(p.a[(size_t)(j * s.K_r + k)]);
        }
        for (int64_t k = 0; k < s.K_r; ++k) d[(size_t)(y * s.K_r + k)] *= p.scale[(size_t)y];
    }
    return d;
}

}  /* namespace */

/* ================================================================== the files */
TEST(adapter_files_group_by_module_and_refuse_what_they_cannot_reproduce) {
    const Files f = make_files();
    lora::AdapterFile a;
    std::string err;
    REQUIRE_EQ(a.open(f.f1, &err), RAD_OK);
    CHECK_EQ(a.modules().size(), (size_t)6);
    CHECK(a.skipped().empty());
    CHECK_EQ(a.biases().size(), (size_t)1);
    CHECK_EQ(std::string(a.kind_name()), std::string("LoRA"));
    const std::string dir = new_dir("genlora_bad");
    write_st(dir + "/dora.safetensors", { tensor("m.lora_A.weight", "F32", { 2, 4 }, 1),
                                          tensor("m.lora_B.weight", "F32", { 4, 2 }, 2),
                                          tensor("m.dora_scale", "F32", { 4, 1 }, 3) });
    CHECK_EQ(a.open(dir + "/dora.safetensors", &err), (int)RAD_E_UNSUPPORTED);
    CHECK(err.find("DoRA") != std::string::npos);
    write_st(dir + "/half.safetensors", { tensor("m.lora_A.weight", "F32", { 2, 4 }, 1) });
    CHECK_EQ(a.open(dir + "/half.safetensors", &err), (int)RAD_E_FORMAT);
}

TEST(comfyui_bias_deltas_compose_at_the_requested_strengths) {
    const std::string dir = new_dir("genlora_bias");
    const std::string path = dir + "/bias.safetensors";
    const std::string module = "transformer.transformer_blocks.0.attn.to_out.0";
    write_st(path, {tensor(module + ".diff_b", "F32", {kD}, 7)});
    auto file = std::make_shared<lora::AdapterFile>();
    std::string err;
    REQUIRE_EQ(file->open(path, &err), RAD_OK);
    REQUIRE_EQ(file->biases().size(), (size_t)1);
    Program prog;
    std::map<std::string, Truth> truth;
    REQUIRE_EQ(build_gen(0, 1, &prog, &truth), RAD_OK);
    std::vector<lora::Site> sites;
    REQUIRE_EQ(lora::build_sites(prog, {"*"}, 16, &sites, &err), RAD_OK);
    lora::KeyMap keys(prog, sites);
    std::map<std::string, std::vector<float>> biases;
    REQUIRE_EQ(lora::compose_biases(keys, {{"one", file, 0.6}, {"two", file, 1.0}},
                                    &biases, &err), RAD_OK);
    REQUIRE(biases.count(module));
    REQUIRE_EQ(biases.at(module).size(), (size_t)kD);
    for (int64_t i = 0; i < kD; ++i)
        CHECK(std::abs(biases.at(module)[(size_t)i] - 1.6f * val(7, (size_t)i)) < 1e-6f);
}

TEST(snapshotted_adapter_composes_its_original_bias_after_replacement) {
    const std::string dir = new_dir("genlora_bias");
    const std::string path = dir + "/bias.safetensors";
    const std::string module = "transformer.transformer_blocks.0.attn.to_out.0";
    write_st(path, {tensor(module + ".diff_b", "F32", {kD}, 7)});
    auto file = std::make_shared<lora::AdapterFile>();
    std::string err;
    CHECK_EQ(file->open_snapshot(path, 8, &err), RAD_E_FULL);
    CHECK_EQ(file->size_bytes(), 0u);
    REQUIRE_EQ(file->open_snapshot(path, 4096, &err), RAD_OK);
    CHECK(file->size_bytes() > 0);
    write_st(path, {tensor(module + ".diff_b", "F32", {kD}, 19)});
    REQUIRE_EQ(file->biases().size(), (size_t)1);
    Program prog;
    std::map<std::string, Truth> truth;
    REQUIRE_EQ(build_gen(0, 1, &prog, &truth), RAD_OK);
    std::vector<lora::Site> sites;
    REQUIRE_EQ(lora::build_sites(prog, {"*"}, 16, &sites, &err), RAD_OK);
    lora::KeyMap keys(prog, sites);
    std::map<std::string, std::vector<float>> biases;
    REQUIRE_EQ(lora::compose_biases(keys, {{"one", file, 0.6}, {"two", file, 1.0}},
                                    &biases, &err), RAD_OK);
    REQUIRE(biases.count(module));
    REQUIRE_EQ(biases.at(module).size(), (size_t)kD);
    for (int64_t i = 0; i < kD; ++i)
        CHECK(std::abs(biases.at(module)[(size_t)i] - 1.6f * val(7, (size_t)i)) < 1e-6f);

    write_st(path, {tensor(module + ".lora_A.weight", "F32", {2, kD}, 1)});
    CHECK_EQ(file->open_snapshot(path, 4096, &err), RAD_E_FORMAT);
    CHECK_EQ(file->size_bytes(), 0u);
    CHECK(file->modules().empty());
    CHECK(file->biases().empty());
    write_st(path, {tensor(module + ".diff_b", "F32", {kD}, 19)});
    REQUIRE_EQ(file->open_snapshot(path, 4096, &err), RAD_OK);
    CHECK_EQ(file->biases().size(), 1u);
}

TEST(adapter_snapshot_pool_charges_live_revisions_once) {
    const std::string dir = new_dir("genlora_snapshot_pool");
    const std::string path = dir + "/active.safetensors";
    const std::string replacement = dir + "/new.safetensors";
    write_st(path, {tensor("layer.diff_b", "F32", {kD}, 7)});
    write_st(replacement, {tensor("layer.diff_b", "F32", {kD}, 19)});
    lora::AdapterSnapshots pool;
    std::shared_ptr<const lora::AdapterSnapshots::Source> old, same, next;
    std::string err;
    const std::string invalid = dir + "/incomplete.safetensors";
    write_st(invalid, {tensor("layer.lora_A.weight", "F32", {2, kD}, 1)});
    CHECK_EQ(pool.acquire(invalid, 4096, 0, &old, &err), RAD_E_FORMAT);
    CHECK(!old);
    CHECK_EQ(pool.used_bytes(), 0u);
    REQUIRE_EQ(pool.acquire(path, std::filesystem::file_size(path), 0, &old, &err), RAD_OK);
    const auto budget = old->file.size_bytes();
    REQUIRE_EQ(pool.acquire(path, budget, 0, &same, &err), RAD_OK);
    CHECK(same == old);
    CHECK_EQ(pool.used_bytes(), budget);
    REQUIRE_EQ(std::rename(replacement.c_str(), path.c_str()), 0);
    int64_t pressure_bytes = -1;
    CHECK_EQ(pool.acquire(path, budget, 0, &next, &err, &pressure_bytes), RAD_E_FULL);
    CHECK_EQ(pressure_bytes, 0);
    CHECK(!next);
#if defined(__linux__)
    CHECK_EQ(pool.acquire(path, budget * 2, std::numeric_limits<int64_t>::max(),
                          &next, &err, &pressure_bytes), RAD_E_FULL);
    CHECK_EQ(pressure_bytes, static_cast<int64_t>(std::filesystem::file_size(path)));
    CHECK(!next);
    CHECK_EQ(pool.used_bytes(), budget);
#endif
    REQUIRE_EQ(pool.acquire(path, budget * 2, 0, &next, &err, &pressure_bytes), RAD_OK);
    CHECK_EQ(pressure_bytes, 0);
    CHECK(next->revision != old->revision);
    CHECK_EQ(pool.used_bytes(), 2 * budget);
    CHECK(std::memcmp(next->file.biases().at("layer")->data,
                      old->file.biases().at("layer")->data, kD * sizeof(float)) != 0);
    next.reset();
    same.reset();
    CHECK_EQ(pool.acquire(path, budget, 0, &next, &err), RAD_E_FULL);
    old.reset();
    CHECK_EQ(pool.used_bytes(), 0u);
    REQUIRE_EQ(pool.acquire(path, budget, 0, &next, &err), RAD_OK);
    CHECK_EQ(next->file.size_bytes(), budget);
}

TEST(krea_projector_lora_resolves_and_composes_without_a_gemm_site) {
    const std::string dir = new_dir("genlora_projector");
    const std::string path = dir + "/projector.safetensors";
    const std::string source = "diffusion_model.txtfusion.projector";
    const std::string module = "transformer.text_fusion.projector";
    write_st(path, {tensor(source + ".lora_down.weight", "F32", {1, 12}, 7),
                    tensor(source + ".lora_up.weight", "F32", {1, 1}, 8)});
    auto file = std::make_shared<lora::AdapterFile>();
    std::string err;
    REQUIRE_EQ(file->open(path, &err), RAD_OK);

    Program prog;
    prog.lora_roots.push_back({"transformer.", {"diffusion_model."}});
    prog.lora_aliases.push_back({"transformer.", "txtfusion.projector", "text_fusion.projector"});
    auto& projector = prog.lora_replacements[1];
    projector.module = module;
    projector.rows = 12;
    projector.row_mixer = true;
    lora::KeyMap keys(prog, {});
    std::map<std::string, lora::Composite> deltas;
    lora::ComposeReport report;
    REQUIRE_EQ(lora::compose(keys, {{"turbo", file, 0.6}}, 128, &deltas, &report, &err), RAD_OK);
    REQUIRE_EQ(report.modules, 1);
    REQUIRE_EQ(report.exact, 1);
    REQUIRE(deltas.count(module));
    const auto& delta = deltas.at(module);
    REQUIRE_EQ(delta.rows, 1);
    REQUIRE_EQ(delta.K, 12);
    REQUIRE_EQ(delta.r, 1);
    for (size_t i = 0; i < 12; ++i)
        CHECK(std::abs(delta.B[0] * delta.A[i] - 0.6f * val(8, 0) * val(7, i)) < 1e-6f);
}

/* A delta given as an operator truncates to its best rank-r form: exact at or above its rank, and
 * otherwise keeping what the sketch says it keeps. */
TEST(openblas_adapter_math_matches_dense_references) {
    const auto* blas = lora::cpu::blas();
    if (std::getenv("RAD_TEST_REQUIRE_LORA_BLAS")) REQUIRE(blas != nullptr);
    if (!blas) return;
    constexpr int rows = 12, cols = 11, rank = 10, keep = 3;
    std::vector<float> B(rows * rank), A(rank * cols);
    const float diagonal[] = {5, -4, 4, 2, 1};
    for (int i = 0; i < 5; ++i) {
        B[i * rank + i] = 1;
        A[i * cols + i] = diagonal[i];
    }
    // These two signed factors cancel, and the remaining ranks are zero.
    B[7 * rank + 5] = 1;
    B[7 * rank + 6] = 1;
    A[5 * cols + 7] = 3;
    A[6 * cols + 7] = -3;
    double energy = 0, retained = 0;
    REQUIRE(lora::cpu::energy(*blas, B, A, rows, rank, cols, &energy));
    CHECK_NEAR(energy, 62.0, 1e-10);
    std::vector<float> b, a;
    REQUIRE(lora::cpu::compress(*blas, B, A, rows, cols, rank, keep, &b, &a, &retained));
    CHECK_NEAR(retained, 57.0 / 62.0, 1e-6);
    const auto delta = mul(b, a, rows, keep, cols, 1.0);
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < cols; ++col)
            CHECK_NEAR(delta.d[(size_t)row * cols + col],
                       row == col && row < keep ? diagonal[row] : 0.0f, 1e-5);

    constexpr int a_rows = 2, a_cols = 3, b_rows = 3, b_cols = 2, sketch = 4, r2 = 2;
    const float w1[] = {1, -2, 3, 4, 1, -1};
    const float w2a[] = {1, 2, 0, -1, 4, 1};
    const float w2b[] = {1, 2, -1, 3};
    float w2[b_rows * b_cols] = {};
    for (int i = 0; i < b_rows; ++i)
        for (int j = 0; j < b_cols; ++j)
            for (int r = 0; r < r2; ++r)
                w2[i * b_cols + j] += w2a[i * r2 + r] * w2b[r * b_cols + j];
    float input[(a_rows*b_rows + a_cols*b_cols) * sketch] = {};
    for (size_t i = 0; i < std::size(input); ++i) input[i] = (int)(i % 7) - 3;
    for (bool transpose : {false, true}) {
        const int n = transpose ? a_cols * b_cols : a_rows * b_rows;
        const int k = transpose ? a_rows * b_rows : a_cols * b_cols;
        for (bool factored : {false, true}) {
            std::vector<float> output((size_t)n * sketch);
            REQUIRE(lora::cpu::apply_lokr(*blas, transpose, input, output.data(), sketch,
                                         a_rows, a_cols, b_rows, b_cols, r2,
                                         w1, factored ? nullptr : w2, w2a, w2b));
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < sketch; ++j) {
                    float expected = 0;
                    for (int h = 0; h < k; ++h) {
                        const int row = transpose ? h : i;
                        const int col = transpose ? i : h;
                        expected += w1[(row / b_rows) * a_cols + col / b_cols] *
                                    w2[(row % b_rows) * b_cols + col % b_cols] * input[h * sketch + j];
                    }
                    CHECK_NEAR(output[i * sketch + j], expected, 1e-5);
                }
        }
    }
}

TEST(truncation_is_exact_within_the_rank_and_honest_past_it) {
    const int64_t N = 96, K = 80;
    std::vector<float> D((size_t)(N * K));
    /* rank 12 */
    std::vector<float> U((size_t)(N * 12)), V((size_t)(12 * K));
    for (size_t i = 0; i < U.size(); ++i) U[i] = val(21, i);
    for (size_t i = 0; i < V.size(); ++i) V[i] = val(22, i);
    const Dense ref = mul(U, V, N, 12, K, 1.0);
    for (size_t i = 0; i < D.size(); ++i) D[i] = (float)ref.d[i];
    lora::Delta d;
    d.N = N; d.K = K; d.rank_bound = std::min(N, K);
    d.apply = [&](const float* X, int64_t l, float* Y) {
        for (int64_t n = 0; n < N; ++n)
            for (int64_t j = 0; j < l; ++j) {
                double a = 0;
                for (int64_t k = 0; k < K; ++k) a += (double)D[(size_t)(n * K + k)] * X[k * l + j];
                Y[n * l + j] = (float)a;
            }
    };
    d.apply_t = [&](const float* Y, int64_t l, float* X) {
        for (int64_t k = 0; k < K; ++k)
            for (int64_t j = 0; j < l; ++j) {
                double a = 0;
                for (int64_t n = 0; n < N; ++n) a += (double)D[(size_t)(n * K + k)] * Y[n * l + j];
                X[k * l + j] = (float)a;
            }
    };
    std::vector<float> B, A;
    double kept = 0;
    lora::truncate(d, 16, &B, &A, &kept);
    const Dense got = mul(B, A, N, 16, K, 1.0);
    double e = 0, t = 0;
    for (size_t i = 0; i < got.d.size(); ++i) { e += std::pow(got.d[i] - ref.d[i], 2); t += ref.d[i] * ref.d[i]; }
    CHECK(e / t < 1e-8);
    CHECK_NEAR(kept, 1.0, 1e-6);
    /* rank 6 of a rank-12 matrix: what is lost is what the report says is lost */
    lora::truncate(d, 6, &B, &A, &kept);
    const Dense g6 = mul(B, A, N, 6, K, 1.0);
    e = 0;
    for (size_t i = 0; i < g6.d.size(); ++i) e += std::pow(g6.d[i] - ref.d[i], 2);
    CHECK(kept < 0.99 && kept > 0.2);
    CHECK_NEAR(1.0 - e / t, kept, 0.02);
}

/* ================================================================== the names */
TEST(keys_resolve_through_spellings_aliases_and_kohya) {
    Program P;
    std::map<std::string, Truth> truth;
    REQUIRE_EQ(build_gen(0, 1, &P, &truth), RAD_OK);
    std::vector<lora::Site> sites;
    std::string err;
    REQUIRE_EQ(lora::build_sites(P, { "*" }, 16, &sites, &err), RAD_OK);
    /* 2 blocks x 4 weights; the text encoder's and the fused GEMM's are not sites */
    CHECK_EQ(sites.size(), (size_t)8);
    for (const lora::Site& s : sites) CHECK(s.weight.rfind("dit.", 0) == 0 && s.weight != "dit.fused.weight");
    const lora::KeyMap km(P, sites);
    std::string why;
    auto one = [&](const std::string& k) { return km.resolve(k, &why); };
    std::vector<lora::Target> t = one("lora_unet_blocks_1_attn_qkv");
    REQUIRE_EQ(t.size(), (size_t)3);
    CHECK_EQ(t[1].module, std::string("transformer.transformer_blocks.1.attn.to_k"));
    CHECK_EQ(t[1].from_lo, kQ);
    CHECK_EQ(t[1].from_hi, 2 * kQ);
    t = one("diffusion_model.blocks.0.mlp.fc2");
    REQUIRE_EQ(t.size(), (size_t)1);
    CHECK_EQ(t[0].module, std::string("transformer.transformer_blocks.0.ff.net.2"));
    t = one("transformer.transformer_blocks.0.attn.to_out.0");
    REQUIRE_EQ(t.size(), (size_t)1);
    t = one("lora_unet_transformer_blocks_0_attn_to_out_0");
    REQUIRE_EQ(t.size(), (size_t)1);
    CHECK_EQ(t[0].module, std::string("transformer.transformer_blocks.0.attn.to_out.0"));
    t = one("blocks.1.mlp.fc1");
    CHECK_EQ(t.size(), (size_t)2);
    CHECK(one("diffusion_model.proj_out").empty());
    CHECK(why.find("no adapter path") != std::string::npos);
    CHECK(one("text_encoder.layers.0.q_proj").empty());
    CHECK(one("diffusion_model.blocks.7.attn.qkv").empty());
}

/* ================================================================== stacked, packed */
/* Packed compact, as the engine packs a generator's slot: every pool's product is the dense delta,
 * and its ranks are taken from 0. */
TEST(stacked_adapters_of_every_kind_pack_to_their_dense_delta_on_every_rank) {
    const Files f = make_files();
    auto a1 = std::make_shared<lora::AdapterFile>(), a2 = std::make_shared<lora::AdapterFile>();
    std::string err;
    REQUIRE_EQ(a1->open(f.f1, &err), RAD_OK);
    REQUIRE_EQ(a2->open(f.f2, &err), RAD_OK);
    const double w1 = 0.8, w2 = -0.5;
    std::map<std::string, Dense> want;
    for (const auto& [k, d] : f.d1) want[k].add(d, w1);
    for (const auto& [k, d] : f.d2) want[k].add(d, w2);
    const int R = 24;
    for (int world : { 1, 2 }) {
        for (int rank = 0; rank < world; ++rank) {
            Program P;
            std::map<std::string, Truth> truth;
            REQUIRE_EQ(build_gen(rank, world, &P, &truth), RAD_OK);
            std::vector<lora::Site> sites;
            std::vector<std::string> pools;
            REQUIRE_EQ(lora::build_sites(P, { "*" }, R, &sites, &err, &pools), RAD_OK);
            const lora::KeyMap km(P, sites);
            std::map<std::string, lora::Composite> comp;
            lora::ComposeReport rep;
            const int cs = lora::compose(km, { { "one", a1, w1 }, { "two", a2, w2 } }, R, &comp, &rep, &err);
            if (cs != RAD_OK) fprintf(stderr, "    world %d rank %d: %s\n", world, rank, err.c_str());
            REQUIRE_EQ(cs, RAD_OK);
            CHECK_EQ(rep.unplaced.size(), (size_t)1);
            CHECK(rep.skipped.empty());
            /* block 0's fc2 (a full delta, rank 128) is the one module past rank 24 */
            CHECK_EQ(rep.modules - rep.exact, (int64_t)1);
            CHECK_EQ(rep.worst_module, std::string("transformer.transformer_blocks.0.ff.net.2"));
            const std::map<std::string, lora::AdModule> mods = lora::as_modules(comp, km);
            lora::AdapterInfo info;
            std::vector<lora::PackedSite> packed;
            const int ps = lora::pack_modules(sites, R, mods, &info, &packed, &err, /*compact=*/true);
            if (ps != RAD_OK) fprintf(stderr, "    world %d rank %d: %s\n", world, rank, err.c_str());
            REQUIRE_EQ(ps, RAD_OK);
            REQUIRE_EQ(packed.size(), pools.size());
            for (size_t p = 0; p < pools.size(); ++p) {
                const lora::Site* sp = nullptr;
                for (const lora::Site& x : sites) if (x.pool == (int32_t)p && !sp) sp = &x;
                REQUIRE(sp != nullptr);
                const lora::Site& s = *sp;
                const Truth& t = truth.at(s.weight);
                /* compact: the ranks taken from 0, within the pool's R */
                const lora::PackedSite& pk = packed[(size_t)s.pool];
                CHECK(pk.empty || (pk.r_used > 0 && pk.r_used <= s.R));
                for (int64_t y = 0; y < s.N_r && !pk.empty; ++y)
                    for (int64_t j = pk.r_used; j < s.R; ++j)
                        CHECK_EQ(pk.b[(size_t)(y * s.R + j)], (uint16_t)0);
                REQUIRE_EQ((int64_t)t.rows.size(), s.N_r);
                REQUIRE_EQ((int64_t)t.cols.size(), s.K_r);
                const std::vector<double> got = slot_delta(s, packed[(size_t)s.pool]);
                double e = 0, tot = 0, mx = 0;
                bool truncated = false;
                for (int64_t y = 0; y < s.N_r; ++y) {
                    const auto& [mod, row] = t.rows[(size_t)y];
                    truncated = truncated || mod == "transformer.transformer_blocks.0.ff.net.2";
                    auto it = want.find(mod);
                    for (int64_t k = 0; k < s.K_r; ++k) {
                        const double ref = it == want.end() ? 0.0
                                           : it->second.d[(size_t)(row * it->second.K + t.cols[(size_t)k])];
                        const double g = got[(size_t)(y * s.K_r + k)];
                        e += (g - ref) * (g - ref);
                        tot += ref * ref;
                        mx = std::max(mx, std::fabs(g - ref) / (std::fabs(ref) + 1e-3));
                    }
                }
                const double rel = tot > 0 ? std::sqrt(e / tot) : std::sqrt(e);
                if (rel > (truncated ? 0.9 : 0.01))
                    fprintf(stderr, "    world %d rank %d %s: relative error %.4g\n", world, rank,
                            s.weight.c_str(), rel);
                /* bf16 factors hold a delta to ~0.5%; the truncated one to what its rank keeps */
                CHECK(rel <= (truncated ? 0.9 : 0.01));
                if (truncated && world == 1)
                    CHECK_NEAR(1.0 - rel * rel, rep.worst_kept, 0.03);
            }
        }
    }
}

TEST(rotated_generators_pack_lora_in_the_activation_basis) {
    const std::string path = new_dir("genlora-fwht128") + "/one.safetensors";
    const std::string block = "transformer.transformer_blocks.0.";
    const TW q = tensor(block + "attn.to_q.lora_A.weight", "F32", {1, kD}, 71);
    const TW down = tensor(block + "ff.net.2.lora_A.weight", "F32", {1, kF}, 72);
    write_st(path, {q, down,
        tensor(block + "attn.to_q.lora_B.weight", "BF16", {kQ, 1}, 73),
        tensor(block + "ff.net.2.lora_B.weight", "BF16", {kD, 1}, 74)});
    auto adapter = std::make_shared<lora::AdapterFile>();
    std::string err;
    REQUIRE_EQ(adapter->open(path, &err), RAD_OK);
    for (int rank : {0, 1}) {
        Program program;
        std::map<std::string, Truth> truth;
        REQUIRE_EQ(build_gen(rank, 2, &program, &truth), RAD_OK);
        for (OpInfo& op : program.ops) {
            if (op.op != "gemm_nt_q") continue;
            WeightInfo& w = program.weights[op.weights[0]];
            if (w.name != "dit.0.qkv.weight" && w.name != "dit.0.down.weight") continue;
            const bool u4 = w.name == "dit.0.down.weight";
            w.enc_known = true;
            rad_enc_clear(&w.enc);
            rad_enc_copy_str(w.enc.scheme, "affine");
            rad_enc_copy_str(w.enc.transform, "fwht128");
            rad_enc_add_plane(&w.enc, "codes", u4 ? RAD_U4 : RAD_I8, 1, 1);
            if (u4) {
                rad_enc_add_plane(&w.enc, "scale", RAD_F16, 1, 128);
                rad_enc_add_plane(&w.enc, "zero", RAD_U8, 1, 128);
            }
            op.base.set_s("dtype", u4 ? "w4a8i8_asym" : "i8a8h");
            op.base.set_i("group", 256);
            CHECK_EQ(lora::op_is_site(program, op, nullptr, nullptr), !u4);
            op.base.set_i("group", 128);
            CHECK(lora::op_is_site(program, op, nullptr, nullptr));
            if (u4) {
                w.enc.plane[1].block[1] = 64;
                CHECK(!lora::op_is_site(program, op, nullptr, nullptr));
                w.enc.plane[1].block[1] = 128;
            }
            op.base.set_i("pb1", 16);
            CHECK(!lora::op_is_site(program, op, nullptr, nullptr));
            op.base.set_i("pb1", 0);
            std::strcpy(w.enc.transform, "fwht64");
            CHECK(!lora::op_is_site(program, op, nullptr, nullptr));
            std::strcpy(w.enc.transform, "fwht128");
        }
        std::vector<lora::Site> sites;
        REQUIRE_EQ(lora::build_sites(program, {"*"}, 16, &sites, &err), RAD_OK);
        const lora::KeyMap keys(program, sites);
        std::map<std::string, lora::Composite> modules;
        lora::ComposeReport report;
        REQUIRE_EQ(lora::compose(keys, {{"one", adapter, 1.0}}, 16, &modules, &report, &err),
                   RAD_OK);
        CHECK_EQ(report.exact, 2);
        CHECK(report.unplaced.empty());
        lora::AdapterInfo info;
        std::vector<lora::PackedSite> packed;
        REQUIRE_EQ(lora::pack_modules(sites, 16, lora::as_modules(modules, keys), &info,
                                      &packed, &err, /*compact=*/true), RAD_OK);
        int rotated_sites = 0;
        for (const lora::Site& site : sites) {
            if (!site.fwht128_input) continue;
            ++rotated_sites;
            const auto& slot = packed[(size_t)site.pool];
            CHECK(!slot.empty);
            CHECK_EQ(slot.r_used, 1);
            const auto& original = site.weight == "dit.0.qkv.weight" ? q.v : down.v;
            const auto& columns = truth.at(site.weight).cols;
            for (int64_t i = 0; i < site.K_r; ++i) {
                const int64_t start = (i / 128) * 128;
                double value = 0;
                for (uint32_t j = 0; j < 128; ++j) {
                    const double coefficient = rad_bf16_to_f32(slot.a[(size_t)(start + j)]);
                    const auto parity = std::popcount(uint32_t(i % 128) & j) % 2;
                    value += parity ? -coefficient : coefficient;
                }
                CHECK_NEAR(value, original[(size_t)columns[(size_t)i]], 0.002);
            }
        }
        CHECK_EQ(rotated_sites, 2);
    }
}

/* ================================================================== the request fields */
TEST(requests_name_adapters_in_a_field_or_in_prompt_tags) {
    std::vector<std::pair<std::string, double>> out;
    std::string field, why;
    std::string prompt = "a cat, <lora:pixel art:0.7>, sitting <lora:detail>";
    Json body = { { "loras", { "style", "ink:0.5", { { "name", "x:y" }, { "weight", 2 } } } } };
    REQUIRE_EQ(server::parse_loras(body, &prompt, &out, &field, &why), RAD_OK);
    REQUIRE_EQ(out.size(), (size_t)5);
    CHECK_EQ(out[0].first, std::string("style"));
    CHECK_EQ(out[0].second, 1.0);
    CHECK_EQ(out[1].first, std::string("ink"));
    CHECK_EQ(out[1].second, 0.5);
    CHECK_EQ(out[2].first, std::string("x:y"));
    CHECK_EQ(out[2].second, 2.0);
    CHECK_EQ(out[3].first, std::string("pixel art"));
    CHECK_NEAR(out[3].second, 0.7, 1e-12);
    CHECK_EQ(out[4].first, std::string("detail"));
    CHECK_EQ(prompt, std::string("a cat, sitting"));
    body = { { "lora", { { "a", 0.25 }, { "b", 1 } } } };
    prompt = "p";
    REQUIRE_EQ(server::parse_loras(body, &prompt, &out, &field, &why), RAD_OK);
    CHECK_EQ(out.size(), (size_t)2);
    body = { { "loras", { 3 } } };
    CHECK_EQ(server::parse_loras(body, &prompt, &out, &field, &why), (int)RAD_E_INVAL);
    CHECK_EQ(field, std::string("loras"));
}

RAD_TEST_MAIN()
