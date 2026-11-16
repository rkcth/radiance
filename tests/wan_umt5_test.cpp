#include "arch_fakes.h"
#include "../arch/wan/wan_conditioning.h"
#include "../arch/wan/wan_denoiser.h"
#include "../arch/wan/wan_dit_io.h"
#include "../arch/wan/wan_time.h"
#include "../arch/wan/wan_vae_time.h"
#include "../arch/wan/wan_vae_encoder.h"
#include "../arch/wan/wan_vae_decoder.h"
#define RAD_ARCH_NO_EXPORTS
#include "../arch/wan/wan.cpp"

static RadVideoGenDecl declared_wan{};
extern "C" int rad_declare_video_generator(RadBuilder*, const RadVideoGenDecl* decl) {
    declared_wan = *decl;
    return RAD_OK;
}


TEST(wan_umt5_declares_each_checkpoint_tensor_and_runs_packed_conditioning) {
    RadBuilder builder{};
    rad::arch::Names names("wan");
    wan::Umt5Encoder encoder;
    REQUIRE_EQ(encoder.declare(&builder, names), RAD_OK);
    const auto* token_weight = find_w(builder, "wan.umt5.token.weight");
    REQUIRE(token_weight);
    CHECK_EQ(token_weight->d.shape[0], 256384);
    CHECK_EQ(token_weight->d.shape[1], 4096);
    const auto* output = find_b(builder, "wan.umt5.encoded");
    REQUIRE(output);
    CHECK_EQ(output->d.shape[0], 1024);
    CHECK_EQ(output->d.shape[1], 4096);

    std::set<std::string> sources;
    size_t source_count = 0;
    for (const auto& map : builder.maps) {
        source_count += map.src.size();
        sources.insert(map.src.begin(), map.src.end());
    }
    CHECK_EQ(source_count, 242u);
    CHECK_EQ(sources.size(), source_count);
    const auto* token_map = find_map(builder, "wan.umt5.token.weight");
    REQUIRE(token_map);
    CHECK(token_map->src == std::vector<std::string>{"shared.weight"});
    const auto* qkv = find_map(builder, "wan.umt5.block.0.qkv.weight");
    const auto* ffn = find_map(builder, "wan.umt5.block.23.gate_up.weight");
    REQUIRE(qkv && ffn);
    CHECK(qkv->src == (std::vector<std::string>{
        "encoder.block.0.layer.0.SelfAttention.q.weight",
        "encoder.block.0.layer.0.SelfAttention.k.weight",
        "encoder.block.0.layer.0.SelfAttention.v.weight"}));
    CHECK(ffn->src == (std::vector<std::string>{
        "encoder.block.23.layer.1.DenseReluDense.wi_0.weight",
        "encoder.block.23.layer.1.DenseReluDense.wi_1.weight"}));

    const int32_t tokens[] = {20, 21, 1, 57, 1};
    const int32_t boundaries[] = {0, 3, 5};
    RadBatch batch{};
    batch.n_tok = 5;
    batch.n_seq = 2;
    batch.token_ids = tokens;
    batch.cu_seqlens = boundaries;
    RadCtx run{};
    run.b = &builder;
    run.batch = &batch;
    encoder.step(&run, batch);
    REQUIRE(run.step_fail.empty());

    int layer = 0;
    for (const auto& issue : run.issues) {
        const auto& op = builder.ops.at(issue.op - 1);
        CHECK_EQ(issue.n, batch.n_tok);
        if (op.op != "attn_umt5") continue;
        REQUIRE_EQ(issue.opd.size(), 6u);
        CHECK_EQ(pint(&op, "heads"), 64);
        for (int i = 0; i < 3; ++i) {
            CHECK_EQ(issue.opd[i].handle, issue.opd[0].handle);
            CHECK_EQ(issue.opd[i].offset, i * 4096);
            CHECK_EQ(issue.opd[i].cols, 4096);
        }
        CHECK(issue.opd[3].raw == boundaries);
        CHECK_EQ(issue.opd[3].rows, 3);
        const auto& bias = builder.weights.at(issue.opd[4].handle - 1);
        CHECK_EQ(bias.d.group.layer, layer++);
        CHECK_EQ(bias.d.shape[0], 32);
        CHECK_EQ(bias.d.shape[1], 64);
    }
    CHECK_EQ(layer, 24);
    REQUIRE(!run.issues.empty());
    CHECK_EQ(run.issues.back().opd.back().handle, encoder.encoded);
    CHECK_LINT(builder, &run);
}

TEST(wan_umt5_rejects_geometry_outside_one_or_two_tokenized_prompts) {
    RadCtx run{};
    RadBatch batch{};
    batch.n_tok = 513;
    batch.n_seq = 1;
    const int32_t token = 1;
    const int32_t boundaries[] = {0, 513};
    batch.token_ids = &token;
    batch.cu_seqlens = boundaries;
    wan::Umt5Encoder{}.step(&run, batch);
    CHECK(!run.step_fail.empty());
    CHECK(run.issues.empty());
}

TEST(wan_text_conditioning_projects_padded_prompts_into_the_right_video_lane) {
    RadBuilder builder{};
    rad::arch::Names names("wan");
    wan::TextConditioning text;
    REQUIRE_EQ(text.declare(&builder, names, 2), RAD_OK);
    const auto* w = find_w(builder, "wan.text.proj1.weight");
    REQUIRE(w);
    CHECK_EQ(w->d.shape[0], 3072);
    CHECK_EQ(w->d.shape[1], 4096);
    CHECK_EQ(w->d.dtype, (uint32_t)RAD_BF16);
    const auto* persistent = find_b(builder, "wan.text.projected");
    REQUIRE(persistent);
    CHECK_EQ(persistent->d.kind, RAD_BUF_PERSIST);
    CHECK_EQ(persistent->d.shape[0], 2048);
    CHECK_EQ(persistent->d.shape[1], 3072);

    const int32_t tokens[] = {20, 21, 1, 57, 1}, boundaries[] = {0, 3, 5};
    RadVideoGenBatch video{};
    video.lane = 1;
    video.n_neg = 2;
    RadBatch batch{};
    batch.n_tok = 5;
    batch.n_seq = 2;
    batch.token_ids = tokens;
    batch.cu_seqlens = boundaries;
    batch.vgen_b = &video;
    RadCtx run{};
    run.b = &builder;
    run.batch = &batch;
    text.step(&run, batch);
    REQUIRE(run.step_fail.empty());
    REQUIRE(run.issues.size() >= 5);
    const size_t end = run.issues.size();
    const auto& fill = run.issues[end - 5];
    const auto& positive = run.issues[end - 4];
    const auto& negative = run.issues[end - 3];
    const auto& proj1 = run.issues[end - 2];
    const auto& proj2 = run.issues[end - 1];
    CHECK(builder.ops.at(fill.op - 1).op == "fill");
    CHECK_EQ(fill.n, 1024);
    CHECK_EQ(positive.n, 3);
    CHECK_EQ(positive.opd.at(0).offset, 0);
    CHECK_EQ(positive.opd.at(1).offset, 0);
    CHECK_EQ(negative.n, 2);
    CHECK_EQ(negative.opd.at(0).offset, 3 * 4096);
    CHECK_EQ(negative.opd.at(1).offset, 512 * 4096);
    CHECK_EQ(pstr(&builder.ops.at(proj1.op - 1), "act"), std::string("gelu_tanh"));
    CHECK_EQ(proj1.n, 1024);
    CHECK_EQ(proj2.n, 1024);
    CHECK_EQ(proj2.opd.back().handle, text.buffer());
    CHECK_EQ(proj2.opd.back().offset, text.context(1, 0).offset);
    CHECK_EQ(text.context(1, 1).offset, 1536 * 3072);
    CHECK_LINT(builder, &run);

    video.n_neg = 0;
    batch.n_tok = 3;
    batch.n_seq = 1;
    RadCtx single{};
    single.b = &builder;
    single.batch = &batch;
    text.step(&single, batch);
    REQUIRE(single.step_fail.empty());
    CHECK_EQ(single.issues.size() + 1, run.issues.size());
    CHECK_EQ(single.issues.back().opd.back().rows, 512);
    CHECK_LINT(builder, &single);

    video.lane = 2;
    RadCtx invalid{};
    invalid.b = &builder;
    text.step(&invalid, batch);
    CHECK(!invalid.step_fail.empty());
    CHECK(invalid.issues.empty());
}

TEST(wan_time_conditioning_uses_zero_only_for_the_reference_frame) {
    RadBuilder builder{};
    rad::arch::Names names("wan");
    wan::TimeConditioning time;
    REQUIRE_EQ(time.declare(&builder, names), RAD_OK);
    REQUIRE_EQ(builder.weights.size(), 6u);
    for (const auto& w : builder.weights) CHECK_EQ(w.d.dtype, (uint32_t)RAD_F32);
    CHECK_EQ(pflt(find_op(builder, "timestep_embed_f64"), "max_period"), 10000.0);
    CHECK_EQ(pint(find_op(builder, "gemm_nt_bias", 2), "N"), 18432);
    const float schedule[] = {999.0f, 0.999f};
    RadVideoGenBatch video{};
    video.sched = schedule;
    RadBatch batch{};
    for (int image : {0, 1}) {
        video.has_image = image;
        RadCtx run{};
        run.b = &builder;
        run.batch = &batch;
        time.step(&run, video);
        REQUIRE(run.step_fail.empty());
        REQUIRE_EQ(run.issues.size(), (size_t)(7 + image));
        if (image) CHECK(builder.ops.at(run.issues[0].op - 1).op == "fill");
        const auto& copy = run.issues.at(image);
        CHECK(copy.opd.at(0).raw == schedule);
        CHECK_EQ(copy.opd.at(1).offset, image);
        CHECK_EQ(run.issues.at(image + 1).n, 1 + image);
        CHECK_EQ(run.issues.back().opd.back().handle, time.modulation);
        CHECK_EQ(run.issues.back().opd.back().rows, 1 + image);
        CHECK_LINT(builder, &run);
    }
}

TEST(wan_denoiser_io_uses_latent_geometry_and_preserves_weight_shape) {
    using namespace rad::arch;
    RadBuilder builder{};
    Names names("wan");
    const rad_buf latent = decl_b(&builder, "latent", RAD_F32, {64, 48}, RAD_BUF_PERSIST);
    const rad_buf head = decl_b(&builder, "head", RAD_F32, {16, 192}, RAD_BUF_PERSIST);
    const rad_buf velocity = decl_b(&builder, "velocity", RAD_F32, {64, 48}, RAD_BUF_PERSIST);
    wan::DitPatchIO io;
    REQUIRE_EQ(io.declare_input(&builder, names, latent, 16), RAD_OK);
    REQUIRE_EQ(io.declare_output(&builder, head, velocity, RAD_F32), RAD_OK);
    const auto* weight = find_w(builder, "wan.dit.patch_embedding.weight");
    REQUIRE(weight);
    CHECK_EQ(weight->d.rank, 5u);
    CHECK_EQ(weight->d.shape[1], 48);
    CHECK_EQ(weight->d.shape[2], 1);
    CHECK_EQ(weight->d.shape[3], 2);
    CHECK_EQ(weight->d.shape[4], 2);
    CHECK_EQ(pint(find_op(builder, "gemm_nt_bias"), "K"), 192);

    int32_t dims[RAD_VGEOM_N]{};
    dims[RAD_VGEOM_LAT_FRAMES] = 2;
    dims[RAD_VGEOM_LAT_H] = 4;
    dims[RAD_VGEOM_LAT_W] = 6;
    RadVideoGenBatch video{};
    video.geom.lat_frames = 2;
    video.geom.lat_h = 4;
    video.geom.lat_w = 6;
    video.geom_d = dims;
    RadCtx run{};
    run.b = &builder;
    const int64_t rows = io.step_input(&run, video, RAD_B(latent));
    REQUIRE_EQ(rows, 12);
    io.step_output(&run, video, rows, RAD_B(head), RAD_B(velocity));
    REQUIRE(run.step_fail.empty());
    REQUIRE_EQ(run.issues.size(), 4u);
    CHECK_EQ(run.issues[0].opd.at(0).rows, 48);
    CHECK_EQ(run.issues[1].opd.back().handle, io.hidden);
    CHECK_EQ(run.issues[2].opd.back().handle, io.rotary);
    CHECK_EQ(run.issues[3].opd.at(0).rows, 12);
    CHECK_EQ(run.issues[3].opd.back().rows, 48);
    for (const auto& issue : run.issues)
        for (const auto& operand : issue.opd)
            if (operand.kind == RAD_OPK_RAW) CHECK(operand.raw == dims + RAD_VGEOM_LAT_FRAMES);
    CHECK_LINT(builder, &run);
}

TEST(wan_denoiser_records_all_blocks_with_fp32_residual_and_velocity) {
    using namespace rad::arch;
    RadBuilder builder{};
    Names names("wan");
    const rad_buf latent = decl_b(&builder, "latent", RAD_F32, {64, 48}, RAD_BUF_PERSIST);
    const rad_buf text = decl_b(&builder, "text", RAD_BF16, {512, 3072}, RAD_BUF_PERSIST);
    const rad_buf velocity = decl_b(&builder, "velocity", RAD_F32, {64, 48}, RAD_BUF_PERSIST);
    wan::Denoiser denoiser;
    REQUIRE_EQ(denoiser.declare(&builder, names, latent, text, velocity, 16), RAD_OK);
    CHECK_EQ(count_op(builder, "attn_dense"), 30);
    CHECK_EQ(count_op(builder, "attn_cross"), 30);
    CHECK_EQ(count_op(builder, "wan_gated_add"), 90);
    CHECK_EQ(count_op(builder, "wan_mod_norm"), 61);
    CHECK_EQ(pint(find_op(builder, "wan_mod_norm", 0), "round_norm"), 1);
    CHECK_EQ(pint(find_op(builder, "wan_mod_norm", 2), "round_norm"), 0);
    CHECK_EQ(pint(find_op(builder, "wan_mod_norm", 60), "group"), -1);
    const auto* head = find_w(builder, "wan.dit.head.weight");
    const auto* state = find_b(builder, "wan.dit.state");
    REQUIRE(head && state);
    CHECK_EQ(head->d.dtype, (uint32_t)RAD_F32);
    CHECK_EQ(head->d.shape[0], 192);
    CHECK_EQ(state->d.dtype, (uint32_t)RAD_F32);
    const auto* self_qkv = find_map(builder, "wan.dit.block.0.self.qkv.weight");
    REQUIRE(self_qkv);
    REQUIRE_EQ(self_qkv->src.size(), 3u);
    CHECK_EQ(self_qkv->src[1], std::string("transformer.blocks.0.attn1.to_k.weight"));
    const auto* last_norm = find_map(builder, "wan.dit.block.29.cross.norm_k.weight");
    REQUIRE(last_norm);
    CHECK_EQ(last_norm->src.at(0), std::string("transformer.blocks.29.attn2.norm_k.weight"));

    int32_t geometry[RAD_VGEOM_N]{};
    geometry[RAD_VGEOM_LAT_FRAMES] = 2;
    geometry[RAD_VGEOM_LAT_H] = 4;
    geometry[RAD_VGEOM_LAT_W] = 6;
    geometry[RAD_VGEOM_FIRST_CU + 1] = 6;
    int32_t video_cu[] = {0, 12};
    const float scheduler[] = {999, 1};
    RadVideoGenBatch video{};
    video.geom.lat_frames = 2;
    video.geom.lat_h = 4;
    video.geom.lat_w = 6;
    video.geom_d = geometry;
    video.lat_cu = video_cu;
    video.sched = scheduler;
    for (int image : {0, 1}) {
        video.has_image = image;
        RadCtx run{};
        run.b = &builder;
        CHECK_OK(denoiser.step(&run, video, RAD_B(latent), RAD_B(text), RAD_B(velocity)));
        REQUIRE(run.step_fail.empty());
        for (const auto& issue : run.issues) {
            const auto& op = builder.ops.at(issue.op - 1);
            if (op.op == "wan_mod_norm") {
                CHECK_EQ(issue.opd.at(1).rows, 1 + image);
                if (image)
                    CHECK(issue.opd.at(3).raw == geometry + RAD_VGEOM_FIRST_CU + 1);
                else
                    CHECK_EQ(issue.opd.at(3).kind, RAD_OPK_NONE);
            } else if (op.op == "attn_cross") {
                CHECK_EQ(issue.opd.at(0).rows, 12);
                CHECK_EQ(issue.opd.at(1).rows, 512);
            }
        }
        REQUIRE(!run.issues.empty());
        CHECK_EQ(run.issues.back().opd.back().handle, velocity);
        CHECK_EQ(run.issues.back().opd.back().rows, 48);
        CHECK_LINT(builder, &run);
    }
}

TEST(wan_vae_time_cache_bypasses_first_chunk_and_isolates_lanes) {
    using namespace rad::arch;
    RadBuilder b{};
    Names names("wan");
    const rad_buf in = decl_b(&b, "in", RAD_BF16, {16, 8}, RAD_BUF_PERSIST);
    const rad_buf out = decl_b(&b, "out", RAD_BF16, {32, 8}, RAD_BUF_PERSIST);
    const int32_t dims[] = {1, 2, 2};
    const wan::VaeVolume volume{1, 2, 2, praw(dims, RAD_I32, 3)};
    const wan::VaeChunkCapacity limit{8, 2, 2};
    const wan::VaeChunkCapacity wide{9000, 2, 2};
    CHECK(!wide.fits({1, 1, 9000, volume.geometry}, 0));
    CHECK(!limit.fits({1, 2, 2, RAD_NONE}, 0));
    wan::VaeTemporalUp up;
    REQUIRE_EQ(up.declare(&b, names, "vae.up", "vae.decoder.up.time_conv",
                         limit, 8, in, out, 0, grp_layer(0)), RAD_OK);
    const auto* weight = find_w(b, "wan.vae.up.weight");
    REQUIRE(weight);
    CHECK_EQ(weight->d.rank, 5u);
    CHECK_EQ(weight->d.shape[0], 16);
    CHECK_EQ(weight->d.shape[1], 8);
    CHECK_EQ(weight->d.shape[2], 3);
    CHECK_EQ(weight->d.shape[3], 1);
    for (int chunk : {0, 1, 2}) {
        RadCtx run{};
        run.b = &b;
        up.step(&run, volume, 1, chunk, RAD_B(in), RAD_B(out));
        REQUIRE(run.step_fail.empty());
        if (chunk == 0) {
            REQUIRE_EQ(run.issues.size(), 1u);
            CHECK(b.ops.at(run.issues[0].op - 1).op == "cast");
            CHECK_EQ(run.issues[0].opd.back().rows, 4);
        } else {
            REQUIRE_EQ(run.issues.size(), 3u);
            const auto& issue = run.issues.front();
            const auto& op = b.ops.at(issue.op - 1);
            CHECK(op.op == "wan_vae_history");
            CHECK_EQ(pint(&op, "reset"), chunk == 1);
            CHECK_EQ(issue.opd.at(1).offset, 2 * 8 * 8);
            CHECK_EQ(issue.opd.at(1).rows, 8);
            CHECK_EQ(run.issues.back().opd.back().rows, 8);
        }
        CHECK_LINT(b, &run);
    }
}

TEST(wan_vae_causal_convolution_excludes_history_outputs) {
    using namespace rad::arch;
    RadBuilder b{};
    Names names("wan");
    const auto in = decl_b(&b, "in", RAD_BF16, {16, 8}, RAD_BUF_PERSIST);
    const auto out = decl_b(&b, "out", RAD_BF16, {16, 8}, RAD_BUF_PERSIST);
    wan::VaeCausalConv conv;
    REQUIRE_EQ(conv.declare(&b, names, "vae.conv", "vae.decoder.conv_in", {8, 2, 2},
                           8, 8, in, out, 0, grp_layer(0)), RAD_OK);
    int32_t dims[] = {2, 2, 2};
    RadCtx run{};
    run.b = &b;
    conv.step(&run, {2, 2, 2, praw(dims, RAD_I32, 3)}, 0, false, RAD_B(in), RAD_B(out));
    REQUIRE_EQ(run.issues.size(), 4u);
    CHECK_EQ(run.issues[2].n, 16);
    CHECK_EQ(run.issues[3].opd.at(0).offset, 2 * 4 * 8);
    CHECK_EQ(run.issues[3].opd.back().rows, 8);
    CHECK_LINT(b, &run);
}

TEST(wan_vae_encoder_uses_last_temporal_taps_and_spatial_checkpoint_weights) {
    using namespace rad::arch;
    RadBuilder b{};
    Names names("wan");
    const auto input = decl_b(&b, "patches", RAD_BF16, {12 * 64, 12}, RAD_BUF_PERSIST);
    const auto cu = decl_b(&b, "offsets", RAD_I32, {2}, RAD_BUF_PERSIST);
    wan::VaeEncoder enc;
    REQUIRE_EQ(enc.declare(&b, names, 12, input, cu, 2), RAD_OK);
    const auto* conv = find_map(b, "wan.vae.encoder.conv_in.weight");
    const auto* resample = find_map(b, "wan.vae.encoder.down_blocks.2.downsampler.resample.1.weight");
    const auto* attention = find_map(b, "wan.vae.encoder.mid_block.attentions.0.to_qkv.weight");
    REQUIRE(conv && resample && attention);
    CHECK_EQ(conv->mode, RAD_MAP_TAP);
    CHECK_EQ(resample->mode, RAD_MAP_COPY);
    CHECK_EQ(attention->mode, RAD_MAP_COPY);
    CHECK(attention->src == (std::vector<std::string>{"vae.encoder.mid_block.attentions.0.to_qkv.weight"}));
    const auto* w = find_w(b, "wan.vae.encoder.conv_in.weight");
    REQUIRE(w);
    CHECK_EQ(w->d.shape[1], 12);
    CHECK_EQ(w->d.rank, 4u);
    RadCtx run{};
    run.b = &b;
    REQUIRE_EQ(enc.step(&run, 2, 3), RAD_OK);
    int shortcuts = 0, attention_calls = 0;
    for (const auto& issue : run.issues) {
        const auto* op = b.op_of(issue.op);
        if (op->op == "l2norm_chan") CHECK_EQ(pint(op, "round_length"), 1);
        if (op->op == "s2d_avg_add") {
            CHECK_EQ(pint(op, "ft"), shortcuts == 0 ? 1 : 2);
            ++shortcuts;
        }
        if (op->op == "attn_dense") {
            ++attention_calls;
            CHECK_EQ(pint(op, "head_dim"), 640);
            CHECK_EQ(issue.n, 6);
            CHECK_EQ(issue.opd[3].handle, cu);
        }
    }
    CHECK_EQ(shortcuts, 3);
    CHECK_EQ(attention_calls, 1);
    CHECK_EQ(enc.mean(6).handle, enc.output());
    CHECK_EQ(enc.mean(6).cols, 48);
    CHECK_LINT(b, &run);
}

TEST(wan_vae_decoder_emits_one_frame_then_four_and_declares_causal_history) {
    using namespace rad::arch;
    RadBuilder b{};
    Names names("wan");
    const auto latent = decl_b(&b, "latent", RAD_BF16, {24, 48}, RAD_BUF_PERSIST);
    const auto geom = decl_b(&b, "geometry", RAD_I32, {3}, RAD_BUF_PERSIST);
    const auto rgb = decl_b(&b, "frames", RAD_U8, {4 * 24 * 256, 3}, RAD_BUF_PERSIST);
    wan::VaeDecoder decoder;
    REQUIRE_EQ(decoder.declare(&b, names, 24, 2, latent, geom, grp_layer(3)), RAD_OK);
    REQUIRE_EQ(decoder.declare_rgb(&b, names, rgb), RAD_OK);
    auto* mid = find_map(b, "wan.vae.decoder.mid_block.attentions.0.to_qkv.weight");
    REQUIRE(mid);
    CHECK_EQ(mid->mode, RAD_MAP_COPY);
    auto* after = find_w(b, "wan.vae.decoder.up_blocks.3.resnets.0.conv1.weight");
    REQUIRE(after);
    CHECK_EQ(after->d.shape[0], 256);
    CHECK_EQ(after->d.shape[1], 512);
    CHECK_EQ(after->d.shape[2], 3);
    auto* time = find_w(b, "wan.vae.decoder.up_blocks.1.upsampler.time_conv.weight");
    REQUIRE(time);
    CHECK_EQ(time->d.shape[0], 2048);
    CHECK_EQ(time->d.shape[1], 1024);
    CHECK(!find_w(b, "wan.vae.decoder.up_blocks.2.upsampler.time_conv.weight"));

    for (int chunk = 0; chunk < 3; ++chunk) {
        RadCtx run{};
        run.b = &b;
        REQUIRE_EQ(decoder.step(&run, 4, 6, 1, chunk), RAD_OK);
        CHECK(run.step_fail.empty());
        REQUIRE(!run.issues.empty());
        int spatial_history = 0, time_history = 0, shortcuts = 0;
        for (const auto& issue : run.issues) {
            const auto* op = b.op_of(issue.op);
            if (op->op == "wan_vae_history") {
                if (pint(op, "temporal")) {
                    ++time_history;
                    CHECK_EQ(pint(op, "reset"), chunk == 1);
                } else {
                    ++spatial_history;
                    CHECK_EQ(pint(op, "reset"), chunk == 0);
                }
            }
            if (op->op == "wan_vae_expand" && !pint(op, "temporal")) {
                CHECK_EQ(pint(op, "drop"), chunk == 0 && shortcuts < 2 ? 1 : 0);
                ++shortcuts;
            }
        }
        const int64_t frames = chunk == 0 ? 1 : 4;
        const auto& end = run.issues.back();
        CHECK(b.op_of(end.op)->op == "tile_stitch");
        const auto& cast = run.issues.at(run.issues.size() - 3);
        CHECK(b.op_of(cast.op)->op == "cast");
        CHECK_EQ(cast.opd.front().handle, decoder.output());
        CHECK_EQ(cast.opd.back().handle, end.opd.front().handle);
        CHECK_EQ(end.opd.front().rows, frames * 24 * 64);
        CHECK_EQ(end.opd.back().rows, frames * 4 * 16);
        CHECK_EQ(end.opd.back().cols, 6 * 16 * 3);
        CHECK(end.opd.back().raw == rad_buf_ptr(&run, rgb));
        CHECK_EQ(spatial_history, 30);
        CHECK_EQ(time_history, chunk == 0 ? 0 : 2);
        CHECK_EQ(shortcuts, 3);
        CHECK_LINT(b, &run);
    }
}

TEST(wan_video_passes_use_both_guidance_branches_and_restore_the_first_frame) {
    using namespace rad::arch;
    RadBuilder b{};
    MetaKV fields(RadModelMeta{});
    std::string mean, stddev;
    for (int i = 0; i < 48; ++i) {
        mean += i ? " 0" : "0";
        stddev += i ? " 1" : "1";
    }
    fields.set("transformer.num_layers", "30")
          .set("transformer.num_attention_heads", "24")
          .set("transformer.attention_head_dim", "128")
          .set("transformer.in_channels", "48")
          .set("transformer.out_channels", "48")
          .set("vae.z_dim", "48")
          .set("vae.base_dim", "160")
          .set("vae.decoder_base_dim", "256")
          .set("vae.patch_size", "2")
          .set("vae.latents_mean", mean.c_str())
          .set("vae.latents_std", stddev.c_str());
    const auto* meta = fields.meta();
    RadArchProbe probe{};
    REQUIRE_EQ(wan::probe(meta, &probe), RAD_OK);
    auto build = build_ctx(0, 1);
    build.max_tok = probe.gen_rows;
    REQUIRE_EQ(wan::declare(&b, meta, &build), RAD_OK);
    CHECK_EQ(declared_wan.frames_tail, -3);
    CHECK_EQ(declared_wan.sched_kind, RAD_VIDEO_SCHEDULE_UNIPC);
    CHECK_EQ(declared_wan.guidance, RAD_VIDEO_GUIDANCE_CFG);
    const auto* velocities = find_b(b, "dit.velocities");
    REQUIRE(velocities);
    CHECK_EQ(velocities->d.kind, RAD_BUF_PERSIST);

    float schedule[16]{}, guidance[8]{5.0f};
    uint32_t seed[4]{};
    int32_t geom[RAD_VGEOM_N]{}, span[4]{0, 3, 0, 0}, lane = 0, first = 1;
    RadVideoGenBatch v{};
    v.geom.lat_frames = 3;
    v.geom.lat_h = v.geom.lat_w = 2;
    v.geom_d = geom;
    v.lat_cu = span;
    v.lane_d = &lane;
    v.first_d = &first;
    v.seed = seed;
    v.sched = schedule;
    v.guidance_h = v.guidance = guidance;
    v.has_image = 1;
    RadBatch batch{};
    batch.gen = RAD_GEN_PASS_VSTEP;
    batch.vgen_b = &v;
    RadCtx run{};
    run.b = &b;
    run.batch = &batch;
    wan::step(&run, &batch);
    REQUIRE(run.step_fail.empty());

    int unpatches = 0;
    for (const auto& issue : run.issues)
        if (b.op_of(issue.op)->op == "wan_unpatch") ++unpatches;
    CHECK_EQ(unpatches, 2);
    REQUIRE(run.issues.size() > 2u);
    const auto& sampler = run.issues.at(run.issues.size() - 2);
    CHECK(b.op_of(sampler.op)->op == "flow_unipc_step");
    CHECK_EQ(sampler.n, 12);
    CHECK_EQ(sampler.opd[0].handle, sampler.opd[4].handle);
    CHECK_EQ(sampler.opd[4].offset, wan::kMaxVoxels * wan::kChannels);
    const auto& pin = run.issues.back();
    CHECK(b.op_of(pin.op)->op == "cast");
    CHECK_EQ(pin.n, 4);
    CHECK_EQ(pin.opd.back().handle, sampler.opd[3].handle);
    CHECK_LINT(b, &run);
}

RAD_TEST_MAIN()
