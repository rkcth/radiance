#include "arch_fakes.h"
#include "../arch/wan/wan_umt5.h"

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

RAD_TEST_MAIN()
