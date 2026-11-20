/* Ling-3.0's declaration and step (arch/bailing_hybrid), without a GPU or checkpoint: the IQ8R
 * container data/recipes/ling-3.0-tiny-iq8r.recipe makes, at one and two ranks, and the bf16 and
 * four-bit containers, tiny and flash (with its MTP layer). */
#define RAD_ARCH_NO_EXPORTS 1
#include "arch_fakes.h"
#include "rad_plugin.h"
#include "../arch/bailing_hybrid/bailing_hybrid.cpp"

using namespace rad_fake;

namespace {

/* Ling-3.0-tiny's configuration as rad-convert records it. */
MetaKV tiny_meta() {
    RadModelMeta m{};
    m.arch_id = "bailinghybrid";
    m.name = "test-ling-tiny";
    m.quant = "";
    m.n_layers = 24;
    m.n_embd = 1536;
    m.n_head = 16;
    m.n_head_kv = 16;
    m.head_dim = 128;
    m.n_ff = 4608;
    m.n_vocab = 157184;
    m.n_ctx_train = 131072;
    m.rms_eps = 1e-6f;
    m.rope_theta = 6e6f;
    m.n_expert = 128;
    m.n_expert_used = 8;
    MetaKV k(m);
    k.set("num_attention_heads", "16").set("head_dim", "128").set("qk_nope_head_dim", "128")
     .set("qk_rope_head_dim", "64").set("v_head_dim", "128").set("kv_lora_rank", "512")
     .set("num_experts", "128").set("num_experts_per_tok", "8").set("n_group", "8")
     .set("topk_group", "4").set("moe_intermediate_size", "512")
     .set("moe_shared_expert_intermediate_size", "512").set("num_shared_experts", "1")
     .set("intermediate_size", "4608").set("first_k_dense_replace", "1")
     .set("layer_group_size", "4").set("short_conv_kernel_size", "4").set("q_lora_rank", "256")
     .set("kda_lower_bound", "-5").set("routed_scaling_factor", "2.5").set("no_kda_lora", "1")
     .set("kda_safe_gate", "1").set("rope_interleave", "1")
     .set("gated_attention_proj_granularity_type", "head_wise").set("score_function", "sigmoid")
     .set("num_nextn_predict_layers", "0").set("norm_topk_prob", "1");
    return k;
}

/* Ling-3.0-flash's: 42 layers of 2560, MLA every sixth with no query LoRA, two dense layers, 512
 * experts of 768, the SwiGLU clamped in the last layers, one MTP layer. */
MetaKV flash_meta() {
    MetaKV k = tiny_meta();
    k.m.name = "test-ling-flash";
    k.m.n_layers = 42;
    k.m.n_embd = 2560;
    k.m.n_head = 32;
    k.m.n_head_kv = 32;
    k.m.n_ff = 6144;
    k.m.n_expert = 512;
    std::string routed, shared;
    for (int l = 0; l < 42; ++l) {
        routed += std::string(l ? "," : "") + (l >= 35 ? "4" : "0");
        shared += std::string(l ? "," : "") + (l >= 40 ? "7" : l >= 34 ? "5" : "0");
    }
    k.set("num_attention_heads", "32").set("num_experts", "512")
     .set("moe_intermediate_size", "768").set("moe_shared_expert_intermediate_size", "768")
     .set("intermediate_size", "6144").set("first_k_dense_replace", "2")
     .set("layer_group_size", "6").set("q_lora_rank", "0").set("num_nextn_predict_layers", "1")
     .set("expert_swiglu_limit_list", routed.c_str())
     .set("share_expert_swiglu_limit_list", shared.c_str());
    return k;
}

/* IQ8R as libquant states it: int8 codes rotated by fwht128, a u8 scale per [1 x 256] of K under
 * a bf16 scale a row. */
RadEncoding iq8r() {
    RadEncoding e = rad_enc_affine(RAD_I8, RAD_U8, 1, 256);
    rad_enc_add_plane(&e, "scale.1", RAD_BF16, 1, 0);
    rad_enc_copy_str(e.transform, "fwht128");
    return e;
}

/* Int8 rotated by fwht128 at a bf16 scale a (row, 128 of K): the experts' form. */
RadEncoding i8h() {
    RadEncoding e = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    rad_enc_copy_str(e.transform, "fwht128");
    return e;
}

const RadEncoding kI4 = rad_enc_affine(RAD_I4, RAD_BF16, 1, 128);

/* data/recipes/ling-3.0-tiny-iq8r.recipe's encodings by declared name, every other weight plain
 * bf16. `attn_gate.weight` is the KDA output gate; the MLA's head gate is `attn_head_gate`. */
void iq8r_encs(RadBuilder* b) {
    b->encs.clear();
    for (const char* t : { "ffn_gate_up_exps", "ffn_down_exps", "ffn_down_shexp" })
        b->encs.push_back({ t, i8h() });
    for (const char* t : { "attn_qkv.weight", "attn_forget.weight", "attn_gate.weight",
                           "attn_output.weight", "ffn_gate_up.weight", "ffn_down.weight",
                           "ffn_gate_up_shexp" })
        b->encs.push_back({ t, iq8r() });
    b->encs.push_back({ "", rad_enc_plain(RAD_BF16) });
}

/* data/recipes/ling-3.0-flash-iq8r-w4nl64.recipe's: the IQ8R trunk of iq8r_encs with the routed
 * experts as rotated four-bit codes into the w4nl codebook, an E4M3 scale a (row, 64 of K) under
 * one f32. */
void flash_w4nl64_encs(RadBuilder* b) {
    iq8r_encs(b);
    RadEncoding w4{};
    rad_enc_clear(&w4);
    rad_enc_copy_str(w4.scheme, "affine");
    rad_enc_add_plane(&w4, "codes", RAD_U4, 1, 1);
    rad_enc_add_plane(&w4, "scale", RAD_F8E4M3, 1, 64);
    rad_enc_add_plane(&w4, "scale.1", RAD_F32, 0, 0);
    rad_enc_add_table(&w4, "table", RAD_F32, 1, 16);
    rad_enc_copy_str(w4.transform, "fwht128");
    for (auto& e : b->encs)
        if (e.first == "ffn_gate_up_exps" || e.first == "ffn_down_exps") e.second = w4;
}

/* data/recipes/ling3-w4a8-rtn.recipe's: four-bit routed experts, everything else bf16. */
void w4_encs(RadBuilder* b) {
    b->encs.clear();
    b->encs.push_back({ "ffn_gate_up_exps", kI4 });
    b->encs.push_back({ "ffn_down_exps", kI4 });
    b->encs.push_back({ "", rad_enc_plain(RAD_BF16) });
}

/* The bf16 checkpoint. */
void bf16_encs(RadBuilder* b) {
    b->encs.clear();
    b->encs.push_back({ "", rad_enc_plain(RAD_BF16) });
}

bool is_mla(int l) { return (l + 1) % 4 == 0; }

const std::string& wname(const RadBuilder& b, rad_weight w) { return b.weights[w - 1].name; }

/* The buffer a name declared, 0 if none. */
rad_buf buf(const RadBuilder& b, const char* n) {
    for (size_t i = 0; i < b.bufs.size(); ++i)
        if (b.bufs[i].name == n) return (rad_buf)(i + 1);
    return 0;
}

/* graph_hash with the name maps in declared-name order (stable, so maps that extend one name keep
 * theirs): where among the other declarations a map is made (the trunk probe maps layer 0's
 * q|k|v ahead of the buffers) is not part of what the core builds from it. */
uint64_t graph_hash_maps_by_name(const RadBuilder& b0) {
    RadBuilder b = b0;
    std::stable_sort(b.maps.begin(), b.maps.end(),
                     [](const RecMap& x, const RecMap& y) { return x.declared < y.declared; });
    return graph_hash(b);
}

/* A decode step, a prefill of two prompts and a mixed step, each pass clean. */
void steps_lint_clean(RadBuilder& b, int rank, int world) {
    const auto dc = kit(b, { 1, 1, 1 }, {});
    const auto pf = kit(b, {}, { 64, 100 }, 0);
    const auto mx = kit(b, { 1, 1 }, { 120 }, 300);
    for (const Kit* k : { dc.get(), pf.get(), mx.get() }) {
        auto v = run_passes(b, bailing_hybrid::step, { k }, rank, world);
        CHECK_PASSES(b, v);
    }
}

}  /* namespace */

/* The IQ8R container at one rank and both ranks of two: every trunk linear the recipe quantises is
 * one gemm_nt_q at i8a8h, its N and K this rank's, reading the codes of the producer that feeds it
 * at that producer's group; the linears the recipe keeps bf16 are gemm_nt over the norm's bf16
 * output; the grouped GEMMs are the int8 rows. No E4M3 code and no unrotated int8 quantiser
 * exist, and no linear quantises its own input. */
TEST(iq8r_container_declares_the_rotated_int8_trunk) {
    for (int world : { 1, 2 })
        for (int rank = 0; rank < world; ++rank) {
            MetaKV mk = tiny_meta();
            RadBuilder b;
            iq8r_encs(&b);
            RadBuildCtx ctx = build_ctx(rank, world);
            REQUIRE_EQ(bailing_hybrid::declare(&b, mk.meta(), &ctx), RAD_OK);
            const bailing_hybrid::Model& m = bailing_hybrid::g_model[rank];
            CHECK(m.trunk_rot);
            CHECK(m.norm_fed);
            CHECK_EQ(m.trunk_g8, 256);
            const int64_t H = 16 / world, W = 128 * H, FF = 4608 / world, SH = 512 / world,
                          EX = 512 / world;
            const rad_buf normed = buf(b, "normed"), nq = buf(b, "normed.q8"),
                          ns = buf(b, "normed.s8"), kq = buf(b, "kda_out.q8"),
                          vq = buf(b, "mla_value.q8"), dq = buf(b, "dense_ff.q8"),
                          sq = buf(b, "shared_ff.q8"), eq = buf(b, "expert_ff.q8"),
                          qa = buf(b, "mla_q_a");
            REQUIRE(normed && nq && ns && kq && vq && dq && sq && eq && qa);
            CHECK_EQ(m.act.q8, (uint32_t)nq);

            /* ---- each quantised linear: N, K, group, the codes it reads. */
            struct Want { int64_t N, K, group; rad_buf in; int n; };
            std::map<std::string, Want> want;
            for (int l = 0; l < 24; ++l) {
                char n[96];
                if (is_mla(l)) {
                    std::snprintf(n, sizeof n, "blk.%d.attn_output.weight", l);
                    want[n] = { 1536, H * 128, 256, vq, 0 };
                } else {
                    std::snprintf(n, sizeof n, "blk.%d.attn_qkv.weight", l);
                    want[n] = { 3 * W, 1536, 256, nq, 0 };
                    std::snprintf(n, sizeof n, "blk.%d.attn_forget.weight", l);
                    want[n] = { W, 1536, 256, nq, 0 };
                    std::snprintf(n, sizeof n, "blk.%d.attn_gate.weight", l);
                    want[n] = { W, 1536, 256, nq, 0 };
                    std::snprintf(n, sizeof n, "blk.%d.attn_output.weight", l);
                    want[n] = { 1536, W, 256, kq, 0 };
                }
                if (l == 0) {
                    want["blk.0.ffn_gate_up.weight"] = { 2 * FF, 1536, 256, nq, 0 };
                    want["blk.0.ffn_down.weight"] = { 1536, FF, 256, dq, 0 };
                } else {
                    std::snprintf(n, sizeof n, "blk.%d.ffn_gate_up_shexp.weight", l);
                    want[n] = { 2 * SH, 1536, 256, nq, 0 };
                    std::snprintf(n, sizeof n, "blk.%d.ffn_down_shexp.weight", l);
                    want[n] = { 1536, SH, 128, sq, 0 };
                }
            }
            int q = 0;
            for (const auto& o : b.ops) {
                if (o.op != "gemm_nt_q") continue;
                ++q;
                CHECK_EQ(pstr(&o, "dtype"), std::string("i8a8h"));
                REQUIRE(!o.w.empty());
                auto it = want.find(wname(b, o.w[0]));
                REQUIRE(it != want.end());
                Want& x = it->second;
                ++x.n;
                CHECK_EQ(pint(&o, "N"), x.N);
                CHECK_EQ(pint(&o, "K"), x.K);
                CHECK_EQ(pint(&o, "group"), x.group);
                CHECK(in(o.reads, x.in));
            }
            CHECK_EQ(q, 18 * 4 + 6 + 2 + 23 * 2);
            for (const auto& [name, x] : want) CHECK_EQ(x.n, 1);

            /* ---- the linears the recipe keeps bf16, over the norm's bf16 output (the query parts
             * over the LoRA's normed latent). */
            int keep = 0;
            for (const auto& o : b.ops) {
                if (o.op != "gemm_nt" || o.w.empty()) continue;
                const std::string& n = wname(b, o.w[0]);
                const bool lora = n.find("attn_q_rope") != std::string::npos ||
                                  n.find("attn_q_nope") != std::string::npos;
                if (n.find("attn_beta") == std::string::npos &&
                    n.find("attn_q_a.") == std::string::npos &&
                    n.find("attn_kv_a.") == std::string::npos &&
                    n.find("attn_head_gate") == std::string::npos && !lora)
                    continue;
                ++keep;
                CHECK_EQ(pstr(&o, "dtype"), std::string("bf16"));
                CHECK(in(o.reads, lora ? qa : normed));
                CHECK(!in(o.reads, nq));
            }
            CHECK_EQ(keep, 18 + 6 * 5);

            /* ---- the producers and their groups. */
            int norms = 0, outq = 0, dense = 0, shared = 0, routed = 0;
            for (const auto& o : b.ops) {
                if (o.op == "rmsnorm_had_quant_i8g") {
                    ++norms;
                    CHECK_EQ(pint(&o, "group"), 256);
                    CHECK_EQ(pint(&o, "n"), 1536);
                    CHECK(in(o.writes, nq) && in(o.writes, ns) && in(o.writes, normed));
                } else if (o.op == "had_quant_act_i8g") {
                    ++outq;
                    CHECK_EQ(pint(&o, "group"), 256);
                    const bool kda = in(o.writes, kq);
                    CHECK(kda || in(o.writes, vq));
                    CHECK_EQ(pint(&o, "n"), kda ? W : H * 128);
                } else if (o.op == "gated_had_quant_i8g") {
                    if (in(o.writes, dq)) {
                        ++dense;
                        CHECK_EQ(pint(&o, "group"), 256);
                        CHECK_EQ(pint(&o, "n"), FF);
                    } else if (in(o.writes, sq)) {
                        ++shared;
                        CHECK_EQ(pint(&o, "group"), 128);
                        CHECK_EQ(pint(&o, "n"), SH);
                    } else {
                        CHECK(in(o.writes, eq));
                        ++routed;
                        CHECK_EQ(pint(&o, "group"), 128);
                        CHECK_EQ(pint(&o, "n"), EX);
                    }
                }
            }
            CHECK_EQ(norms, 48);
            CHECK_EQ(outq, 18 + 6);
            CHECK_EQ(dense, 1);
            CHECK_EQ(shared, 23);
            CHECK_EQ(routed, 23);
            for (const char* op : { "rmsnorm_quant_fp8", "ar_rmsnorm_quant_fp8",
                                    "ar_rmsnorm_had_quant_i8g", "quant_act_fp8", "quant_act_i8g",
                                    "gated_quant_fp8", "had_quant_act_fp8", "gated_had_quant_fp8",
                                    "swiglu_step", "silu_mul" })
                CHECK_EQ(count_op(b, op), 0);
            for (const char* n : { "normed.q", "kda_out.q", "mla_value.q", "mla_q_a.q",
                                   "dense_ff.q", "shared_ff.q", "expert_ff.q", "normed_rot.q",
                                   "normed_rot.q8" })
                CHECK(find_b(b, n) == nullptr);

            /* ---- the routed experts: rotated int8 rows, gate_up over the norm's codes at 256,
             * down over the gated product's at 128, each rank a slice of every expert. */
            int gu = 0, dn = 0;
            for (const auto& o : b.ops) {
                if (o.op != "moe_gemm_q") continue;
                CHECK_EQ(pstr(&o, "dtype"), std::string("i8a8h"));
                const bool token = pstr(&o, "a_order") == "token";
                CHECK_EQ(pint(&o, "a_group"), token ? 256 : 128);
                CHECK(in(o.reads, token ? nq : eq));
                CHECK_EQ(pint(&o, "N"), token ? 2 * EX : 1536);
                CHECK_EQ(pint(&o, "K"), token ? 1536 : EX);
                CHECK_EQ(pint(&o, "n_expert"), 128);
                (token ? gu : dn) += 1;
            }
            CHECK_EQ(gu, 23);
            CHECK_EQ(dn, 23);
            const RecWeight* e0 = find_w(b, "blk.1.ffn_gate_up_exps.127.weight");
            REQUIRE(e0 != nullptr);
            CHECK_EQ(e0->d.dtype, (uint32_t)RAD_I8);
            CHECK_EQ((int)e0->d.shard, (int)RAD_SHARD_ROW);
            steps_lint_clean(b, rank, world);
        }
}

/* The IQ8R container shards as the bf16 one does: every weight both declare outside the routed
 * experts is split the same way, and the collectives are the same standalone all-reduces (the plugin's norms take no
 * collective, so no `ar_` norm is declared at two ranks). */
TEST(iq8r_container_follows_the_plugins_sharding) {
    for (int world : { 1, 2 }) {
        MetaKV mk = tiny_meta();
        RadBuilder q, f;
        iq8r_encs(&q);
        bf16_encs(&f);
        RadBuildCtx cq = build_ctx(0, world), cf = build_ctx(0, world);
        cf.slot_base = 4;
        REQUIRE_EQ(bailing_hybrid::declare(&q, mk.meta(), &cq), RAD_OK);
        REQUIRE_EQ(bailing_hybrid::declare(&f, mk.meta(), &cf), RAD_OK);
        int same = 0;
        for (const RecWeight& w : q.weights) {
            const RecWeight* o = find_w(f, w.name);
            if (!o || w.name.find("_exps.") != std::string::npos) continue;
            ++same;
            CHECK_EQ(w.d.shard, o->d.shard);
            CHECK_EQ(w.d.n_row_parts, o->d.n_row_parts);
        }
        CHECK(same > 300);
        CHECK_EQ(count_op(q, "all_reduce"), count_op(f, "all_reduce"));
        CHECK_EQ(count_op(q, "all_reduce"), world == 2 ? 1 + 24 + 24 : 0);
        /* Rotated int8 experts are a slice of every expert on each rank, where bf16 ones are
         * whole on one. */
        CHECK(find_w(q, "blk.1.ffn_gate_up_exps.0.weight") != nullptr);
        CHECK(find_w(q, "blk.1.ffn_gate_up_exps.127.weight") != nullptr);
    }
}

/* Where no loaded library serves the rotating norm (a CPU-only engine), the norms write E4M3
 * codes no linear reads, every int8 trunk linear quantises the norm's bf16 output itself at the
 * trunk's group, and the routed gate_up reads a rotated int8 copy written once a layer at 128. */
TEST(iq8r_container_without_the_rotating_norm_quantises_each_input) {
    for (int world : { 1, 2 }) {
        MetaKV mk = tiny_meta();
        RadBuilder b;
        b.undeclared.insert("rmsnorm_had_quant_i8g");
        iq8r_encs(&b);
        RadBuildCtx ctx = build_ctx(0, world);
        REQUIRE_EQ(bailing_hybrid::declare(&b, mk.meta(), &ctx), RAD_OK);
        CHECK(bailing_hybrid::g_model[0].trunk_rot);
        CHECK(!bailing_hybrid::g_model[0].norm_fed);
        CHECK_EQ(count_op(b, "rmsnorm_quant_fp8"), 48);
        const rad_buf normed = buf(b, "normed"), hr = buf(b, "normed_rot.q8");
        REQUIRE(hr != 0);
        int own = 0, copy = 0;
        for (const auto& o : b.ops) {
            if (o.op == "had_quant_act_i8g" && in(o.reads, normed)) {
                const bool c = in(o.writes, hr);
                (c ? copy : own) += 1;
                CHECK_EQ(pint(&o, "group"), c ? 128 : 256);
            }
            if (o.op == "moe_gemm_q" && pstr(&o, "a_order") == "token") {
                CHECK_EQ(pint(&o, "a_group"), 128);
                CHECK(in(o.reads, hr));
            }
        }
        /* q|k|v, forget and gate on 18 layers, the dense gate_up, 23 shared gate_ups. */
        CHECK_EQ(own, 18 * 3 + 1 + 23);
        CHECK_EQ(copy, 23);
        steps_lint_clean(b, 0, world);
    }
}

/* A shared expert whose down projection is stored IQ8R too takes its input codes at 256. */
TEST(iq8r_shared_down_sets_its_inputs_group) {
    MetaKV mk = tiny_meta();
    RadBuilder b;
    iq8r_encs(&b);
    b.encs.insert(b.encs.begin(), { "ffn_down_shexp", iq8r() });
    RadBuildCtx ctx = build_ctx();
    REQUIRE_EQ(bailing_hybrid::declare(&b, mk.meta(), &ctx), RAD_OK);
    const rad_buf sq = buf(b, "shared_ff.q8");
    int n = 0;
    for (const auto& o : b.ops)
        if (o.op == "gated_had_quant_i8g" && in(o.writes, sq)) {
            CHECK_EQ(pint(&o, "group"), 256);
            ++n;
        }
    CHECK_EQ(n, 23);
    CHECK_EQ(find_b(b, "shared_ff.s8")->d.shape[1], 2);
}

/* The bf16 and four-bit containers at one rank and both ranks of two, tiny and flash (with its
 * MTP layer under speculation), pinned by their graphs' fingerprints: no int8 op, E4M3 codes from
 * the norms, and the four-bit experts reading the norm's E4M3 pair at w4a8. */
TEST(bf16_and_four_bit_containers_keep_their_graphs) {
    struct Case { bool flash, w4; int world, rank; uint64_t hash; };
    const Case cases[] = {
        { false, false, 1, 0, 0x86ab7d6c2a6c7d9aull },
        { false, false, 2, 0, 0x08a9d5eac89cbfdcull },
        { false, false, 2, 1, 0x6bbc016892e93dacull },
        { false, true,  1, 0, 0xcdcf0243f2a0c792ull },
        { false, true,  2, 0, 0xda4c82105c0feb47ull },
        { false, true,  2, 1, 0xd94daa2c5c90352bull },
        { true,  false, 1, 0, 0x21f7c41f985de35eull },
        { true,  false, 2, 0, 0x0420376396574defull },
        { true,  false, 2, 1, 0xb7df9b2e13bc4ae6ull },
        { true,  true,  1, 0, 0xf9dec754da195628ull },
        { true,  true,  2, 0, 0x7da4ef872c9304e9ull },
        { true,  true,  2, 1, 0xcc97ada7f98fca85ull },
    };
    for (const Case& c : cases) {
        MetaKV mk = c.flash ? flash_meta() : tiny_meta();
        RadBuilder b;
        if (c.w4) w4_encs(&b); else bf16_encs(&b);
        RadBuildCtx ctx = build_ctx(c.rank, c.world);
        if (c.flash) ctx.max_spec = 3;
        REQUIRE_EQ(bailing_hybrid::declare(&b, mk.meta(), &ctx), RAD_OK);
        const bailing_hybrid::Model& m = bailing_hybrid::g_model[c.rank];
        CHECK(!m.trunk_rot);
        CHECK_EQ(m.have_mtp, c.flash);
        CHECK_EQ(graph_hash_maps_by_name(b), c.hash);
        CHECK(count_op(b, "rmsnorm_quant_fp8") > 0);
        for (const char* op : { "rmsnorm_had_quant_i8g", "had_quant_act_i8g",
                                "gated_had_quant_i8g", "gemm_nt_q" })
            CHECK_EQ(count_op(b, op), 0);
        if (c.w4) {
            const rad_buf nq = buf(b, "normed.q");
            int moe = 0;
            for (const auto& o : b.ops)
                if (o.op == "moe_gemm_q") {
                    CHECK_EQ(pstr(&o, "dtype"), std::string("w4a8"));
                    if (pstr(&o, "a_order") == "token") CHECK(in(o.reads, nq));
                    ++moe;
                }
            CHECK_EQ(moe, 2 * (c.flash ? 41 : 23));
        }
        if (!c.flash) steps_lint_clean(b, c.rank, c.world);
    }
}

/* What the plugin refuses by name: a rotated trunk over unrotated experts, rotated int8 experts
 * under an unrotated trunk, and the tiny IQ8R trunk at four ranks, whose 1152-column dense slice
 * cuts a 256-column scale group. */
TEST(refuses_mismatched_forms) {
    {
        MetaKV mk = tiny_meta();
        RadBuilder b;
        iq8r_encs(&b);
        for (auto& e : b.encs)
            if (e.first == "ffn_gate_up_exps" || e.first == "ffn_down_exps") e.second = kI4;
        RadBuildCtx ctx = build_ctx();
        CHECK_EQ(stderr_of_rc([&] { return bailing_hybrid::declare(&b, mk.meta(), &ctx); }),
                 RAD_E_UNSUPPORTED);
    }
    {
        MetaKV mk = tiny_meta();
        RadBuilder b;
        w4_encs(&b);
        for (auto& e : b.encs)
            if (e.first == "ffn_gate_up_exps" || e.first == "ffn_down_exps") e.second = i8h();
        RadBuildCtx ctx = build_ctx();
        CHECK_EQ(stderr_of_rc([&] { return bailing_hybrid::declare(&b, mk.meta(), &ctx); }),
                 RAD_E_UNSUPPORTED);
    }
    {
        MetaKV mk = tiny_meta();
        RadBuilder b;
        iq8r_encs(&b);
        RadBuildCtx ctx = build_ctx(0, 4);
        CHECK_EQ(stderr_of_rc([&] { return bailing_hybrid::declare(&b, mk.meta(), &ctx); }),
                 RAD_E_UNSUPPORTED);
    }
}

/* Flash's IQ8R trunk with the clamp lists cut declares, MTP layer included: the drafter's MLA and
 * routed feed-forward read the trunk's codes like any trunk layer. */
TEST(flash_iq8r_without_clamps_declares_its_drafter_on_the_int8_trunk) {
    for (int world : { 1, 2 }) {
        MetaKV mk = flash_meta();
        mk.drop("expert_swiglu_limit_list").drop("share_expert_swiglu_limit_list");
        RadBuilder b;
        iq8r_encs(&b);
        RadBuildCtx ctx = build_ctx(0, world);
        ctx.max_spec = 3;
        REQUIRE_EQ(bailing_hybrid::declare(&b, mk.meta(), &ctx), RAD_OK);
        const bailing_hybrid::Model& m = bailing_hybrid::g_model[0];
        CHECK(m.trunk_rot);
        CHECK(m.have_mtp);
        const RecWeight* d = find_w(b, "blk.42.attn_output.weight");
        REQUIRE(d != nullptr);
        CHECK_EQ(d->d.dtype, (uint32_t)RAD_I8);
        /* 40 routed layers and the drafter's; two norms a layer. */
        CHECK_EQ(count_op(b, "moe_gemm_q"), 2 * 41);
        CHECK_EQ(count_op(b, "rmsnorm_had_quant_i8g"), 2 * 43);
    }
}

/* Flash's IQ8R trunk with its clamp lists declares at one, two and four ranks. Each clamped gated
 * product (routed in layers 35-41, shared in 34-41) is two launches: the clamp into the bf16 plane
 * (swiglu_step) and the rotating int8 quantiser reading it (had_quant_act_i8g, at the 128-column
 * group of the int8 down projection that reads the codes). Against the same container without
 * the clamp lists, those 15 gated_had_quant_i8g become 15 such pairs and no other op changes; no
 * unrotated int8 and no E4M3 quantiser exists, and the steps run lint-clean. */
TEST(flash_iq8r_clamps_its_gated_products_then_quantises_them_rotated) {
    for (int world : { 1, 2, 4 }) {
        for (int rank = 0; rank < world; ++rank) {
            MetaKV plain = flash_meta();
            plain.drop("expert_swiglu_limit_list").drop("share_expert_swiglu_limit_list");
            RadBuilder b0;
            iq8r_encs(&b0);
            RadBuildCtx c0 = build_ctx(rank, world);
            REQUIRE_EQ(bailing_hybrid::declare(&b0, plain.meta(), &c0), RAD_OK);
            MetaKV mk = flash_meta();
            RadBuilder b;
            iq8r_encs(&b);
            RadBuildCtx ctx = build_ctx(rank, world);
            REQUIRE_EQ(bailing_hybrid::declare(&b, mk.meta(), &ctx), RAD_OK);
            CHECK(bailing_hybrid::g_model[0].trunk_rot);
            CHECK_EQ(count_op(b, "swiglu_step"), 15);
            CHECK_EQ(count_op(b, "had_quant_act_i8g") - count_op(b0, "had_quant_act_i8g"), 15);
            CHECK_EQ(count_op(b0, "gated_had_quant_i8g") - count_op(b, "gated_had_quant_i8g"), 15);
            CHECK_EQ(b.ops.size(), b0.ops.size() + 15);
            CHECK_EQ(count_op(b, "quant_act_i8g"), 0);
            CHECK_EQ(count_op(b, "quant_act_fp8"), 0);
            CHECK_EQ(count_op(b, "swiglu_clamped"), 0);
            for (size_t i = 0; i + 1 < b.ops.size(); ++i) {
                if (b.ops[i].op != "swiglu_step") continue;
                const RecOp& q = b.ops[i + 1];
                CHECK_EQ(q.op, std::string("had_quant_act_i8g"));
                REQUIRE_EQ(b.ops[i].writes.size(), (size_t)1);
                CHECK(std::find(q.reads.begin(), q.reads.end(), b.ops[i].writes[0]) !=
                      q.reads.end());
                const RecParam* g = param(&q, "group");
                REQUIRE(g != nullptr);
                CHECK_EQ(g->ival, 128LL);
            }
            steps_lint_clean(b, rank, world);
        }
    }
}

/* The flash recipe's own container: the routed experts are the w4nl codebook at 64 of K, read
 * against the trunk's rotated int8 codes (moe_gemm_q or moe_gemm_q_mod4 at w4nl64a8i8h). The
 * clamped layers take the same clamp-then-quantise pairs as with int8 experts, at every rank of
 * one, two and four. */
TEST(flash_iq8r_with_w4nl64_experts_declares_with_its_clamps) {
    for (int world : { 1, 2, 4 }) {
        for (int rank = 0; rank < world; ++rank) {
            MetaKV mk = flash_meta();
            RadBuilder b;
            flash_w4nl64_encs(&b);
            REQUIRE(rad_enc_valid(&b.encs[0].second));
            RadBuildCtx ctx = build_ctx(rank, world);
            REQUIRE_EQ(bailing_hybrid::declare(&b, mk.meta(), &ctx), RAD_OK);
            CHECK(bailing_hybrid::g_model[0].trunk_rot);
            int routed = 0;
            for (const RecOp& o : b.ops) {
                /* moe_gemm_q, and its four-class form where a rank holds a quarter of K. */
                if (o.op.rfind("moe_gemm_q", 0) != 0) continue;
                CHECK_EQ(pstr(&o, "dtype"), std::string("w4nl64a8i8h"));
                ++routed;
            }
            CHECK(routed > 0);
            CHECK_EQ(count_op(b, "swiglu_step"), 15);
            CHECK_EQ(count_op(b, "quant_act_i8g"), 0);
            CHECK_EQ(count_op(b, "quant_act_fp8"), 0);
            for (size_t i = 0; i + 1 < b.ops.size(); ++i) {
                if (b.ops[i].op != "swiglu_step") continue;
                CHECK_EQ(b.ops[i + 1].op, std::string("had_quant_act_i8g"));
            }
            steps_lint_clean(b, rank, world);
        }
    }
}

/* Per-instance state lives in the declaring instance's slot, so a second engine in the process
 * leaves the first one's model alone. */
TEST(declares_into_its_own_slot) {
    MetaKV mk = tiny_meta();
    RadBuilder b0, b1;
    iq8r_encs(&b0);
    w4_encs(&b1);
    RadBuildCtx c0 = build_ctx();
    RadBuildCtx c1 = build_ctx();
    c1.slot_base = 4;
    REQUIRE_EQ(bailing_hybrid::declare(&b0, mk.meta(), &c0), RAD_OK);
    REQUIRE_EQ(bailing_hybrid::declare(&b1, mk.meta(), &c1), RAD_OK);
    CHECK(bailing_hybrid::g_model[0].trunk_rot);
    CHECK(!bailing_hybrid::g_model[4].trunk_rot);
    CHECK_EQ(bailing_hybrid::g_model[4].s.n_layer, 24);
}

RAD_TEST_MAIN()
