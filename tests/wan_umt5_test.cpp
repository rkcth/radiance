#include "arch_fakes.h"
#include "../arch/wan/wan_conditioning.h"
#include "../arch/wan/wan_dit_io.h"
#include "../arch/wan/wan_time.h"

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
    const auto* qkv = find_map(builder, "wan.umt5.block.0.qkv.weight");
    const auto* ffn = find_map(builder, "wan.umt5.block.23.gate_up.weight");
    REQUIRE(qkv && ffn);
    CHECK(qkv->src == (std::vector<std::string>{
        "text_encoder.encoder.block.0.layer.0.SelfAttention.q.weight",
        "text_encoder.encoder.block.0.layer.0.SelfAttention.k.weight",
        "text_encoder.encoder.block.0.layer.0.SelfAttention.v.weight"}));
    CHECK(ffn->src == (std::vector<std::string>{
        "text_encoder.encoder.block.23.layer.1.DenseReluDense.wi_0.weight",
        "text_encoder.encoder.block.23.layer.1.DenseReluDense.wi_1.weight"}));

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

RAD_TEST_MAIN()
