/* tests/images_test.cpp -- OpenAI's images API over an image generator (core/server/images.cpp).
 *
 * The generator is faked (server_fakes.h: FakeImageGen finishes every request on submit), so
 * these pin what the server owns: which requests are refused and under which field, the request
 * it hands the generator (sizes rounded to the model's multiple, steps from `quality`, a seed a
 * JavaScript client can read back), the usage split between text and pictures, the background an
 * `auto` resolves to, failures mapped to 400 or 500 in both the body and a stream, multipart
 * uploads, and the stream's events. Cases that encode or decode a picture need the media decoder
 * and skip without it.
 */
#include "server_fakes.h"

#include "image_queue.h"
#include "lora/adapter.h"
#include "mm/media.h"

#include <future>

namespace {

struct Images {
    FakeImageGen img;
    FakeTokenizer image_tok;
    std::unique_ptr<Fixture> f;
    explicit Images(bool image_only = true) {
        f.reset(new Fixture(Fixture::Tweak{ [this, image_only](ServerOptions& o, Deps& d) {
            d.images = { &img, &image_tok };
            o.image_model = image_only;
        } }));
    }
    HttpResponse post(const char* path, const json& body) { return f->post(path, body.dump()); }
    json gen(const json& body, int status = 200) {
        HttpResponse r = post("/v1/images/generations", body);
        CHECK_EQ(r.status, status);
        return json::parse(r.body, nullptr, false);
    }
    const gen::Request& last() const { return *img.got.back(); }
};

/* A PNG of w x h with every pixel `rgba`. */
std::string png(int32_t w, int32_t h, uint8_t alpha_left, uint8_t alpha_right) {
    std::vector<uint8_t> px((size_t)w * h * 4, 200);
    for (int32_t y = 0; y < h; ++y)
        for (int32_t x = 0; x < w; ++x) px[((size_t)y * w + x) * 4 + 3] = x < w / 2 ? alpha_left : alpha_right;
    std::vector<uint8_t> file;
    std::string why;
    if (mm::encode_image(px.data(), w, h, 4, "png", 0, &file, &why) < 0) return std::string();
    return std::string(file.begin(), file.end());
}

std::string data_url(const std::string& bytes) {
    return "data:image/png;base64," + base64_encode(bytes.data(), bytes.size());
}

}  /* namespace */

/* ================================================================== refusals */

/* A request this deployment cannot serve is a 400 naming the field it is about, before anything
 * is queued: unknown fields, a missing or non-string prompt, n outside 1..lanes, url output (no
 * picture is stored), a transparent jpeg, previews without a stream, a step count on a model with
 * its own schedule, half a size (naming the half that is missing), a malformed or oversized size,
 * guidance without a negative prompt, and pictures sent to the endpoint that takes none. */
TEST(an_images_request_that_cannot_be_served_names_its_field) {
    Images s;
    s.img.inf.fixed_schedule = true;
    s.img.inf.fixed_steps = 8;
    struct Case { json body; const char* param; };
    const Case cases[] = {
        { { { "prompt", "x" }, { "wobble", 1 } }, "wobble" },
        { json::object(), "prompt" },
        { { { "prompt", 5 } }, "prompt" },
        { { { "prompt", "x" }, { "n", 0 } }, "n" },
        { { { "prompt", "x" }, { "n", 5 } }, "n" },
        { { { "prompt", "x" }, { "response_format", "url" } }, "response_format" },
        { { { "prompt", "x" }, { "output_format", "jpeg" }, { "background", "transparent" } }, "background" },
        { { { "prompt", "x" }, { "partial_images", 2 } }, "partial_images" },
        { { { "prompt", "x" }, { "num_inference_steps", 20 } }, "num_inference_steps" },
        { { { "prompt", "x" }, { "width", 256 } }, "height" },
        { { { "prompt", "x" }, { "height", 256 } }, "width" },
        { { { "prompt", "x" }, { "size", "12x" } }, "size" },
        { { { "prompt", "x" }, { "size", "2048x2048" } }, "size" },
        { { { "prompt", "x" }, { "true_cfg_scale", 4.0 } }, "true_cfg_scale" },
        { { { "prompt", "x" }, { "image", "data:image/png;base64,AAAA" } }, "image" },
    };
    for (const Case& c : cases) {
        const json e = s.gen(c.body, 400);
        CHECK_EQ(str_at(e["error"], "param"), std::string(c.param));
        if (str_at(e["error"], "param") != c.param) fprintf(stderr, "    %s\n", c.body.dump().c_str());
    }
    CHECK(s.gen({ { "prompt", "x" }, { "response_format", "url" } }, 400)["error"]["supported"] ==
          json::array({ "b64_json" }));
    CHECK(s.img.got.empty());
    /* An edit needs a picture; a model that reads none takes no edit at all. */
    CHECK_EQ(str_at(json::parse(s.post("/v1/images/edits", { { "prompt", "x" } }).body)["error"], "param"),
             std::string("image"));
    s.img.inf.max_refs = 0;
    CHECK_EQ(s.post("/v1/images/edits", { { "prompt", "x" } }).status, 501);
}

/* An image model generates no text, so the text endpoints point at the images endpoints, and its
 * model entry says what it does: image generation and editing, no grammar, no video. */
TEST(an_image_model_names_the_images_endpoints_and_its_capabilities) {
    Images s;
    HttpResponse r = s.f->post("/v1/chat/completions", R"({"messages":[{"role":"user","content":"hi"}]})");
    CHECK_EQ(r.status, 501);
    CHECK(has(r.body, "/v1/images/generations"));
    CHECK_EQ(s.f->post("/v1/completions", R"({"prompt":"x"})").status, 501);
    const json caps = json::parse(s.f->get("/v1/models").body)["data"][0]["capabilities"];
    CHECK(caps.value("image_generation", false));
    CHECK(caps.value("image_edit", false));
    CHECK(!caps.value("grammar", true));
    CHECK(!caps.value("video", true));
}

/* ================================================================== the request handed over */

TEST(image_and_text_endpoints_use_their_own_tokenizers) {
    if (!mm::decoder_available()) return;
    Images s(/*image_only=*/false);
    s.gen({ { "prompt", "sky" } });
    CHECK(s.image_tok.last_parse_special);
    CHECK(!s.f->tok.last_parse_special);

    s.image_tok.last_parse_special = false;
    CHECK_EQ(s.post("/v1/completions", { { "prompt", "hello" }, { "max_tokens", 1 } }).status, 200);
    CHECK(s.f->tok.last_parse_special);
    CHECK(!s.image_tok.last_parse_special);
}

/* What reaches the generator: sides rounded down to the model's multiple, steps from `quality`
 * (half the default for low, three quarters for medium) unless named, the seed given or a fresh
 * one under 2^53 (returned for reuse, and JavaScript reads JSON numbers as doubles), and each
 * image's seed counting up from it. */
TEST(the_generator_gets_rounded_sizes_quality_steps_and_a_readable_seed) {
    if (!mm::decoder_available()) return;
    Images s;
    const json a = s.gen({ { "prompt", "x" }, { "size", "300x200" }, { "quality", "low" },
                           { "seed", 5 }, { "n", 2 } });
    CHECK_EQ(s.last().width, 288);
    CHECK_EQ(s.last().height, 192);
    CHECK_EQ(s.last().steps, 10);
    CHECK_EQ(s.last().seed, (uint64_t)5);
    CHECK_EQ(a.value("size", ""), std::string("288x192"));
    REQUIRE(a["data"].size() == 2);
    CHECK_EQ(a["data"][1]["seed"].get<uint64_t>(), (uint64_t)6);
    std::vector<uint8_t> file;
    REQUIRE(base64_decode(a["data"][0]["b64_json"].get<std::string>(), &file));
    CHECK(file.size() > 8 && file[1] == 'P' && file[2] == 'N' && file[3] == 'G');

    s.gen({ { "prompt", "x" }, { "quality", "medium" } });
    CHECK_EQ(s.last().steps, 15);
    CHECK_EQ(s.last().width, 256);
    CHECK_EQ(s.last().height, 128);
    s.gen({ { "prompt", "x" }, { "quality", "low" }, { "num_inference_steps", 7 } });
    CHECK_EQ(s.last().steps, 7);
    for (const json& seed : { json(nullptr), json(-1) }) {
        json body = { { "prompt", "x" } };
        if (!seed.is_null()) body["seed"] = seed;
        const json r = s.gen(body);
        CHECK(s.last().seed < ((uint64_t)1 << 53));
        CHECK(r["data"][0]["seed"].get<uint64_t>() < ((uint64_t)1 << 53));
    }
}

TEST(auto_edit_size_respects_a_tight_pixel_limit_after_rounding) {
    if (!mm::decoder_available()) return;
    Images s;
    s.img.inf.default_width = s.img.inf.default_height = 1024;
    s.img.inf.max_tokens = 4096;
    const std::string pic = png(128, 96, 255, 255);
    REQUIRE(!pic.empty());
    const HttpResponse res = s.post("/v1/images/edits",
        { { "prompt", "x" }, { "image", data_url(pic) } });
    REQUIRE_EQ(res.status, 200);
    CHECK_EQ(s.last().width, 1152);
    CHECK_EQ(s.last().height, 896);
}

TEST(krea_outfill_scales_the_long_edge_and_preserves_the_source_in_the_canvas) {
    if (!mm::decoder_available()) return;
    Images s;
    s.img.inf.refs_fit_output = true;
    s.img.inf.size_multiple = 16;
    const json request = { {"prompt", "A portrait in a studio"}, {"size", "1024x1024"},
                           {"image", data_url(png(48, 64, 255, 255))},
                           {"outfill", { {"horizontal", "right"} }} };
    const HttpResponse response = s.post("/v1/images/edits", request);
    REQUIRE_EQ(response.status, 200);
    CHECK_EQ(s.last().outfill.x, 256);
    CHECK_EQ(s.last().outfill.y, 0);
    CHECK_EQ(s.last().outfill.w, 768);
    CHECK_EQ(s.last().outfill.h, 1024);
    REQUIRE_EQ(s.last().refs.size(), (size_t)1);
    CHECK_EQ(s.last().refs[0].w, 336);
    CHECK_EQ(s.last().refs[0].h, 448);
    CHECK(s.img.frames.empty());
    const json answer = json::parse(response.body);
    std::vector<uint8_t> png_bytes;
    REQUIRE(base64_decode(answer["data"][0]["b64_json"].get<std::string>(), &png_bytes));
    mm::FrameRGBA result;
    std::string why;
    REQUIRE_EQ(mm::decode_image_rgba(png_bytes.data(), png_bytes.size(), 1024 * 1024, &result, &why),
               RAD_OK);
    const auto red = [&](int x, int y) { return (int)result.rgba[((size_t)y * 1024 + x) * 4]; };
    CHECK_EQ(red(10, 100), 0);
    CHECK_EQ(red(256, 100), 0);
    CHECK_EQ(red(272, 100), 100);
    CHECK_EQ(red(288, 100), 200);
    json other = request;
    other["size"] = "768x768";
    CHECK_EQ(s.post("/v1/images/edits", other).status, 400);
    other = request;
    other["ref_boost"] = 4.0;
    CHECK_EQ(s.post("/v1/images/edits", other).status, 400);
}

/* Usage counts the pictures' token runs as image tokens, once in each branch that carries them:
 * twice with guidance (the negative prompt is a second conditioning). An edit's pictures are
 * resized to the model's reference area at their own aspect before the encoder sees them. */
TEST(usage_counts_each_pictures_tokens_in_every_branch) {
    if (!mm::decoder_available()) return;
    Images s;
    const std::string pic = png(40, 20, 255, 255);
    REQUIRE(!pic.empty());
    const json plain = json::parse(s.post("/v1/images/edits",
        { { "prompt", "x" }, { "image", data_url(pic) } }).body);
    const json& u = plain["usage"];
    CHECK_EQ(u["input_tokens_details"].value("image_tokens", (int64_t)-1), (int64_t)3);
    CHECK_EQ(u["input_tokens_details"].value("text_tokens", (int64_t)-1),
             u.value("input_tokens", (int64_t)0) - 3);
    REQUIRE(!s.img.frames.empty());
    CHECK(s.img.frames.back() == std::make_pair(96, 32));
    CHECK_EQ(plain.value("size", ""), std::string("256x128"));
    const json guided = json::parse(s.post("/v1/images/edits",
        { { "prompt", "x" }, { "image", data_url(pic) }, { "negative_prompt", "" },
          { "true_cfg_scale", 4.0 } }).body);
    CHECK_EQ(guided["usage"]["input_tokens_details"].value("image_tokens", (int64_t)-1), (int64_t)6);
    CHECK(s.last().has_neg);
}

/* A model that reads its prompt through a second tokenizer gets the user's prompt as that tokenizer
 * encodes it, in each branch, and a prompt longer than it reads is refused under `prompt`. A
 * prompt shorter than the model's minimum after the fixed prefix is padded with its padding token
 * (an empty prompt is one, and is not turned into " " first). */
TEST(a_second_tokenizer_and_a_minimum_prompt_reach_the_generator) {
    Images s;
    s.img.inf.aux_tokens = true;
    s.img.inf.aux_max_tokens = 4;
    s.img.inf.min_tokens = 2;
    s.img.inf.pad_token = 9;
    s.gen({ { "prompt", "ab" }, { "negative_prompt", "c" }, { "true_cfg_scale", 4.0 } });
    REQUIRE(!s.img.got.empty());
    CHECK(s.last().pos.aux == std::vector<int32_t>({ 100 + 'a', 100 + 'b', 1 }));
    CHECK(s.last().neg.aux == std::vector<int32_t>({ 100 + 'c', 1 }));
    /* "c" is one token after the prefix, padded to two. */
    const gen::Branch& n = s.last().neg;
    REQUIRE((int64_t)n.tokens.size() - n.n_drop == 2);
    CHECK_EQ(n.tokens[(size_t)n.n_drop], (int32_t)'c');
    CHECK_EQ(n.tokens.back(), (int32_t)9);
    s.gen({ { "prompt", "" } });
    const gen::Branch& p = s.last().pos;
    REQUIRE((int64_t)p.tokens.size() - p.n_drop == 2);
    CHECK_EQ(p.tokens[(size_t)p.n_drop], (int32_t)9);
    CHECK(p.aux == std::vector<int32_t>({ 1 }));
    /* An empty negative prompt stays empty for such a model: all padding. */
    s.gen({ { "prompt", "x" }, { "negative_prompt", "" }, { "true_cfg_scale", 4.0 } });
    const gen::Branch& e = s.last().neg;
    REQUIRE((int64_t)e.tokens.size() - e.n_drop == 2);
    CHECK_EQ(e.tokens[(size_t)e.n_drop], (int32_t)9);
    CHECK(e.aux == std::vector<int32_t>({ 1 }));
    const json longp = s.gen({ { "prompt", "abcd" } }, 400);
    CHECK_EQ(str_at(longp["error"], "param"), std::string("prompt"));
}

/* A discrete-time model takes `sampler` and `scheduler` by ComfyUI's names, defaulting to its
 * own, and refuses one it does not serve; a flow-matching model refuses both. A prompt read in
 * windows reaches the generator cut into runs between their markers, each padded to the window,
 * after the template's fixed prefix, and a prompt that fills its last window exactly takes no
 * empty one after it. */
TEST(a_discrete_model_takes_its_sampler_and_reads_the_prompt_in_windows) {
    Images s;
    const json flow = s.gen({ { "prompt", "x" }, { "sampler", "euler" } }, 400);
    CHECK_EQ(str_at(flow["error"], "param"), std::string("sampler"));
    s.img.inf.samplers = (1u << RAD_GEN_SAMPLER_EULER) | (1u << RAD_GEN_SAMPLER_DPMPP_2M);
    s.img.inf.default_sampler = RAD_GEN_SAMPLER_EULER;
    s.img.inf.default_scheduler = RAD_GEN_SCHED_NORMAL;
    s.img.inf.chunk_tokens = 4;
    s.img.inf.chunk_bos = 2;
    s.img.inf.chunk_eos = 3;
    s.img.inf.chunk_pad = 9;
    s.gen({ { "prompt", "abcdefghij" } });
    REQUIRE(!s.img.got.empty());
    CHECK_EQ(s.last().sampler, (int32_t)RAD_GEN_SAMPLER_EULER);
    CHECK_EQ(s.last().scheduler, (int32_t)RAD_GEN_SCHED_NORMAL);
    const std::vector<int32_t> want = { '<', 's', '>', 2, 'a', 'b', 'c', 'd', 3,
                                        2, 'e', 'f', 'g', 'h', 3, 2, 'i', 'j', 3, 9, 9 };
    CHECK(s.last().pos.tokens == want);
    CHECK_EQ(s.last().pos.n_drop, (int32_t)3);
    s.gen({ { "prompt", "x" }, { "sampler", "dpmpp_2m" }, { "scheduler", "karras" } });
    CHECK_EQ(s.last().sampler, (int32_t)RAD_GEN_SAMPLER_DPMPP_2M);
    CHECK_EQ(s.last().scheduler, (int32_t)RAD_GEN_SCHED_KARRAS);
    /* A prompt that fills a window takes no second one. */
    s.gen({ { "prompt", "abcd" } });
    const std::vector<int32_t> one = { '<', 's', '>', 2, 'a', 'b', 'c', 'd', 3 };
    CHECK(s.last().pos.tokens == one);
    const json bad = s.gen({ { "prompt", "x" }, { "sampler", "euler_ancestral" } }, 400);
    CHECK_EQ(str_at(bad["error"], "param"), std::string("sampler"));
    CHECK(bad["error"]["supported"] == json::array({ "euler", "dpmpp_2m" }));
    const json badk = s.gen({ { "prompt", "x" }, { "scheduler", "ddim_uniform" } }, 400);
    CHECK_EQ(str_at(badk["error"], "param"), std::string("scheduler"));
}

/* `rebalance` is a boolean, JSON or form ("true", "1", "false", "0"). `true` reaches the generator
 * as RAD_GEN_COND_REBALANCE only on a model that declares it, and is a 400 naming the field on
 * one that does not; `false` is accepted by every model. */
TEST(rebalance_reaches_a_model_that_declares_it_and_is_refused_by_one_that_does_not) {
    Images s;
    const json no = s.gen({ { "prompt", "x" }, { "rebalance", true } }, 400);
    CHECK_EQ(str_at(no["error"], "param"), std::string("rebalance"));
    CHECK(s.img.got.empty());
    s.gen({ { "prompt", "x" }, { "rebalance", false } });
    REQUIRE(!s.img.got.empty());
    CHECK_EQ(s.last().cond_opts, (int32_t)0);
    s.img.inf.cond_opts = RAD_GEN_COND_REBALANCE;
    for (const json& bad : { json("yes"), json(1), json::array() }) {
        const json e = s.gen({ { "prompt", "x" }, { "rebalance", bad } }, 400);
        CHECK_EQ(str_at(e["error"], "param"), std::string("rebalance"));
    }
    s.gen({ { "prompt", "x" }, { "rebalance", true } });
    CHECK_EQ(s.last().cond_opts, (int32_t)RAD_GEN_COND_REBALANCE);
    s.gen({ { "prompt", "x" } });
    CHECK_EQ(s.last().cond_opts, (int32_t)0);
    s.gen({ { "prompt", "x" }, { "rebalance", nullptr } });
    CHECK_EQ(s.last().cond_opts, (int32_t)0);
    /* The form's text values. */
    HttpRequest q;
    q.method = "POST";
    q.path = "/v1/images/generations";
    q.form = { FormPart{ "prompt", "", "", "a cat" }, FormPart{ "rebalance", "", "", "1" } };
    CHECK_EQ(s.f->srv->router().dispatch(q).status, 200);
    CHECK_EQ(s.last().cond_opts, (int32_t)RAD_GEN_COND_REBALANCE);
    q.form[1].data = "false";
    CHECK_EQ(s.f->srv->router().dispatch(q).status, 200);
    CHECK_EQ(s.last().cond_opts, (int32_t)0);
    q.form[1].data = "on";
    CHECK_EQ(s.f->srv->router().dispatch(q).status, 400);
}

/* `rebalance` also names a preset: "legacy" is `true`, "balanced", "detail" and "subtle" set
 * RAD_GEN_COND_REBALANCE with the preset's row in RAD_GEN_COND_REBALANCE_PRESET, in JSON and in a
 * form. A model whose rebalance declares no presets takes `true` and "legacy" and refuses the
 * others with a 400 naming the field; an unknown name is a 400 everywhere. */
TEST(rebalance_presets_reach_a_model_that_declares_them) {
    Images s;
    const int32_t R = RAD_GEN_COND_REBALANCE, sh = RAD_GEN_COND_REBALANCE_PRESET_SHIFT;
    s.img.inf.cond_opts = RAD_GEN_COND_REBALANCE;
    s.gen({ { "prompt", "x" }, { "rebalance", "legacy" } });
    CHECK_EQ(s.last().cond_opts, R);
    for (const char* p : { "balanced", "detail", "subtle" }) {
        const json e = s.gen({ { "prompt", "x" }, { "rebalance", p } }, 400);
        CHECK_EQ(str_at(e["error"], "param"), std::string("rebalance"));
    }
    s.img.inf.cond_opts = RAD_GEN_COND_REBALANCE | RAD_GEN_COND_REBALANCE_PRESET;
    const struct { const char* name; int32_t opts; } kWant[] = {
        { "legacy", R }, { "balanced", R | 1 << sh }, { "detail", R | 2 << sh }, { "subtle", R | 3 << sh },
    };
    for (const auto& w : kWant) {
        s.gen({ { "prompt", "x" }, { "rebalance", w.name } });
        CHECK_EQ(s.last().cond_opts, w.opts);
    }
    s.gen({ { "prompt", "x" }, { "rebalance", true } });
    CHECK_EQ(s.last().cond_opts, R);
    for (const json& bad : { json("Balanced"), json("strong"), json("") }) {
        const json e = s.gen({ { "prompt", "x" }, { "rebalance", bad } }, 400);
        CHECK_EQ(str_at(e["error"], "param"), std::string("rebalance"));
    }
    HttpRequest q;
    q.method = "POST";
    q.path = "/v1/images/generations";
    q.form = { FormPart{ "prompt", "", "", "a cat" }, FormPart{ "rebalance", "", "", "detail" } };
    CHECK_EQ(s.f->srv->router().dispatch(q).status, 200);
    CHECK_EQ(s.last().cond_opts, R | 2 << sh);
    q.form[1].data = "nope";
    CHECK_EQ(s.f->srv->router().dispatch(q).status, 400);
}

/* `background: auto` keeps an edited picture's transparency and is opaque otherwise; a mask must
 * be the first picture's size. */
/* A model whose card states guidance as cond + s (cond - uncond) and guides by default (Krea 2):
 * a request naming nothing is guided at the card's scale with the model's own empty negative
 * prompt, handed on as s + 1; `guidance_scale` is the same field as `true_cfg_scale`, 0 turns it
 * off, and naming both is refused. An RGB model refuses a transparent background. */
TEST(guidance_follows_the_model_cards_terms_and_its_default_negative) {
    Images s;
    s.img.inf.cfg_original = true;
    s.img.inf.default_cfg = 4.5;
    s.img.inf.has_default_negative = true;
    s.img.inf.default_negative = "";
    s.img.inf.pixel_channels = 3;
    s.gen({ { "prompt", "a fox" } });
    REQUIRE(!s.img.got.empty());
    CHECK(s.last().has_neg);
    CHECK_EQ(s.last().cfg, 5.5f);
    s.gen({ { "prompt", "a fox" }, { "guidance_scale", 2.0 } });
    CHECK(s.last().has_neg);
    CHECK_EQ(s.last().cfg, 3.0f);
    s.gen({ { "prompt", "a fox" }, { "guidance_scale", 0.0 } });
    CHECK(!s.last().has_neg);
    const json both = s.gen({ { "prompt", "x" }, { "guidance_scale", 2.0 }, { "true_cfg_scale", 2.0 } },
                            400);
    CHECK_EQ(str_at(both["error"], "param"), std::string("guidance_scale"));
    const json alpha = s.gen({ { "prompt", "x" }, { "background", "transparent" } }, 400);
    CHECK_EQ(str_at(alpha["error"], "param"), std::string("background"));
}

TEST(an_auto_background_follows_the_pictures_alpha) {
    if (!mm::decoder_available()) return;
    Images s;
    const std::string opaque = png(40, 20, 255, 255), half = png(40, 20, 0, 255);
    REQUIRE(!opaque.empty() && !half.empty());
    auto edit = [&](const std::string& pic) {
        return json::parse(s.post("/v1/images/edits", { { "prompt", "x" }, { "image", data_url(pic) } }).body);
    };
    CHECK_EQ(edit(opaque).value("background", ""), std::string("opaque"));
    CHECK_EQ(edit(half).value("background", ""), std::string("transparent"));
    CHECK_EQ(s.gen({ { "prompt", "x" } }).value("background", ""), std::string("opaque"));
    const json m = json::parse(s.post("/v1/images/edits",
        { { "prompt", "x" }, { "image", data_url(opaque) }, { "mask", data_url(png(10, 10, 0, 0)) } }).body);
    CHECK_EQ(str_at(m["error"], "param"), std::string("mask"));
}

/* ================================================================== failures */

/* The generator's refusal at submit is the client's 400 when it says the request can never run
 * (RAD_E_INVAL) and a 503 otherwise; a request that failed while running is a 500 carrying the
 * generator's reason, or a 400 for its admission refusal. A stream ends in an error event of the
 * same type. A prompt the model's template does not open with its fixed turn is the server's
 * failure, a 500 typed as one. */
TEST(a_generators_failure_is_a_400_or_a_500_streamed_or_not) {
    if (!mm::decoder_available()) return;
    Images s;
    s.img.submit_status = RAD_E_INVAL;
    s.img.submit_why = "too long for the cache";
    CHECK(has(s.gen({ { "prompt", "x" } }, 400)["error"].value("message", ""), "too long"));
    s.img.submit_status = RAD_E_IO;
    s.gen({ { "prompt", "x" } }, 503);
    s.img.submit_status = RAD_OK;
    CHECK_EQ(s.f->srv->in_flight(), (int64_t)0);

    for (int st : { RAD_E_INVAL, RAD_E_IO }) {
        s.img.finish_status = st;
        s.img.finish_why = "the decoder ran out of memory";
        const json e = s.gen({ { "prompt", "x" } }, st == RAD_E_INVAL ? 400 : 500)["error"];
        CHECK_EQ(e.value("message", ""), s.img.finish_why);
        CHECK_EQ(e.value("type", ""), std::string(st == RAD_E_INVAL ? "invalid_request_error"
                                                                    : "server_error"));
        HttpResponse r = s.post("/v1/images/generations", { { "prompt", "x" }, { "stream", true } });
        CHECK_EQ(r.status, 200);
        const std::vector<json> ev = sse_events(drain(r));
        REQUIRE(!ev.empty());
        CHECK_EQ(ev.back().value("type", ""), std::string("error"));
        CHECK_EQ(ev.back()["error"].value("type", ""), e.value("type", ""));
    }
    s.img.finish_status = RAD_OK;
    CHECK_EQ(s.f->srv->in_flight(), (int64_t)0);

    s.img.inf.drop_prefix = "<not the template's turn>";
    const json e = s.gen({ { "prompt", "x" } }, 500)["error"];
    CHECK_EQ(e.value("type", ""), std::string("internal_server_error"));
}

/* ================================================================== the wire forms */

/* An upload as multipart form-data: fields arrive as text (n as "2") and pictures as `image[]`
 * parts. */
TEST(a_multipart_edit_reads_text_fields_and_picture_parts) {
    if (!mm::decoder_available()) return;
    Images s;
    const std::string pic = png(40, 20, 255, 255);
    HttpRequest q;
    q.method = "POST";
    q.path = "/v1/images/edits";
    q.form = { FormPart{ "prompt", "", "", "a cat" }, FormPart{ "n", "", "", "2" },
               FormPart{ "image[]", "a.png", "image/png", pic },
               FormPart{ "image[]", "b.png", "image/png", pic } };
    HttpResponse r = s.f->srv->router().dispatch(q);
    CHECK_EQ(r.status, 200);
    CHECK_EQ(json::parse(r.body)["data"].size(), (size_t)2);
    CHECK_EQ(s.last().n, 2);
    CHECK_EQ(s.last().refs.size(), (size_t)2);
    q.form[1].data = "two";
    CHECK_EQ(s.f->srv->router().dispatch(q).status, 400);
}

/* A stream sends each preview as a partial_image event numbered from 0, then the image as
 * `completed` with its usage, per image. */
TEST(a_streamed_generation_sends_previews_then_the_image) {
    if (!mm::decoder_available()) return;
    Images s;
    HttpResponse r = s.post("/v1/images/generations",
                            { { "prompt", "x" }, { "stream", true }, { "partial_images", 2 } });
    CHECK_EQ(r.status, 200);
    const std::vector<json> ev = sse_events(drain(r));
    REQUIRE(ev.size() == 3);
    CHECK_EQ(ev[0].value("type", ""), std::string("image_generation.partial_image"));
    CHECK_EQ(ev[0].value("partial_image_index", -1), 0);
    CHECK_EQ(ev[1].value("partial_image_index", -1), 1);
    CHECK_EQ(ev[2].value("type", ""), std::string("image_generation.completed"));
    CHECK(ev[2].contains("usage"));
    CHECK(!ev[2].contains("partial_image_index"));
    CHECK_EQ(s.f->srv->in_flight(), (int64_t)0);
}

TEST(shared_streamed_images_report_dispatch_timing) {
    if (!mm::decoder_available()) return;
    FakeImageGen image;
    FakeTokenizer tokenizer;
    Fixture f(Fixture::Tweak{[&](ServerOptions& o, Deps&) {
        o.queue_image = [&](const std::string&, size_t, size_t, size_t,
                            const std::function<bool()>&, Deps::ImageEndpoint* endpoint,
                            std::shared_ptr<void>*, ImageDispatchTiming* timing, std::string*) {
            timing->selection = ImageSelection::Warm;
            *endpoint = {&image, &tokenizer};
            return RAD_OK;
        };
    }});
    auto response = f.post("/v1/images/generations", R"({"prompt":"x","stream":true})");
    REQUIRE_EQ(response.status, 200);
    const auto header = std::find_if(response.headers.begin(), response.headers.end(),
        [](const auto& entry) { return entry.first == "Server-Timing"; });
    REQUIRE(header != response.headers.end());
    CHECK(header->second.starts_with("image_queue;dur=0.000, image_selection;dur="));
    CHECK(header->second.ends_with(", image_activation;desc=\"warm\""));
    CHECK_EQ(sse_events(drain(response)).size(), 1u);
}

TEST(shared_image_releases_adapter_source_before_generation) {
    if (!mm::decoder_available()) return;
    struct Image : FakeImageGen {
        std::weak_ptr<const lora::AdapterFile> source;
        int submit(std::unique_ptr<gen::Request> request, std::string* why) override {
            CHECK(source.expired());
            CHECK(request->adapter_request_lease);
            CHECK(request->lora_hold);
            CHECK_EQ(request->lora_slot, 0);
            return FakeImageGen::submit(std::move(request), why);
        }
    } image;
    FakeTokenizer tokenizer;
    Fixture f(Fixture::Tweak{[&](ServerOptions& o, Deps&) {
        o.pin_image_loras = [&](const auto&, auto* adapters, auto* lease, auto*, bool reclaim_cache) {
            CHECK(reclaim_cache);
            auto source = std::make_shared<lora::AdapterFile>();
            image.source = source;
            adapters->push_back({{"style", std::move(source), 0.6}, "revision"});
            *lease = std::make_shared<int>(0);
            return RAD_OK;
        };
        o.queue_image = [&](const std::string&, size_t, size_t, size_t,
                            const std::function<bool()>&, Deps::ImageEndpoint* endpoint,
                            std::shared_ptr<void>*, ImageDispatchTiming*, std::string*) {
            *endpoint = {&image, &tokenizer};
            endpoint->pinned_lora_acquire = [&](const auto& adapters, auto* slot, auto* hold,
                                               auto*, auto*) {
                CHECK_EQ(adapters.size(), 1u);
                CHECK(!image.source.expired());
                *slot = 0;
                *hold = std::make_shared<int>(0);
                return RAD_OK;
            };
            return RAD_OK;
        };
    }});
    const auto response = f.post("/v1/images/generations",
        R"({"prompt":"x","loras":[{"name":"style","weight":0.6}]})");
    CHECK_EQ(response.status, 200);
    CHECK_EQ(image.got.size(), 1u);
}

TEST(shared_image_waiters_reserve_admission_and_release_it_on_shutdown) {
    std::promise<void> waiting;
    auto queued = waiting.get_future();
    Fixture f(Fixture::Tweak{[&](ServerOptions& o, Deps&) {
        o.queue_depth = 1;
        o.queue_image = [&](const std::string&, size_t, size_t, size_t,
                            const std::function<bool()>& cancelled, Deps::ImageEndpoint*,
                            std::shared_ptr<void>*, ImageDispatchTiming*, std::string*) {
            waiting.set_value();
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!cancelled() && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            return cancelled() ? RAD_E_CANCELLED : RAD_E_STATE;
        };
    }});
    auto image = std::async(std::launch::async, [&] {
        return f.post("/v1/images/generations", R"({"prompt":"a red square","model":"second"})");
    });
    REQUIRE(queued.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    const auto health = f.get("/health");
    CHECK_EQ(health.status, 200);
    CHECK_EQ(json::parse(health.body)["in_flight"].get<int>(), 1);
    CHECK_EQ(f.post("/completion", R"({"prompt":"hello","n_predict":1})").status, 429);
    const auto full = f.post("/v1/images/generations", R"({"prompt":"a blue square"})");
    CHECK_EQ(full.status, 429);
    CHECK(std::any_of(full.headers.begin(), full.headers.end(),
                      [](const auto& h) { return h.first == "Retry-After"; }));
    f.srv->stop();
    CHECK_EQ(image.get().status, 503);
    CHECK_EQ(json::parse(f.get("/health").body)["in_flight"].get<int>(), 0);
}

TEST(shared_image_ordering_respects_the_age_threshold) {
    using namespace std::chrono;
    const auto now = steady_clock::time_point{} + seconds(10);
    std::vector<ImageWaiter> requests{
        {"other", now - seconds(5) + milliseconds(1)},
        {"active", now},
    };
    auto select = [&](bool fifo, steady_clock::time_point when) {
        return next_image_request(requests.begin(), requests.end(), "active", fifo, when)
               - requests.begin();
    };
    CHECK_EQ(select(false, now), 1);
    CHECK_EQ(select(false, now + milliseconds(1)), 0);
    CHECK_EQ(select(true, now), 0);
    requests[0].bypasses = 2;
    CHECK_EQ(select(false, now), 0);
}

TEST(image_lifecycle_reports_backend_failures_as_service_unavailable) {
    int status = RAD_E_IO;
    auto fail = [&](std::string* why) {
        *why = "image lifecycle operation failed";
        return status;
    };
    Fixture f(Fixture::Tweak{[&](ServerOptions& o, Deps&) {
        o.queue_image = [&](const std::string&, size_t, size_t, size_t,
                            const std::function<bool()>&, Deps::ImageEndpoint*,
                            std::shared_ptr<void>*, ImageDispatchTiming*, std::string* why) {
            return fail(why);
        };
        o.register_image = o.load_image = [&](const std::string&, const std::string&, std::string* why) {
            return fail(why);
        };
        o.unload_image = [&](const std::string&, const std::function<bool()>&, std::string* why) {
            return fail(why);
        };
    }});
    struct Request { const char* path; const char* body; };
    const Request requests[] = {
        {"/v1/images/generations", R"({"prompt":"a red square","model":"next"})"},
        {"/v1/images/register_model", R"({"model":"next","path":"/models/next.rad"})"},
        {"/v1/images/load_model", R"({"model":"next","path":"/models/next.rad"})"},
        {"/v1/images/unload_model", R"({"model":"next"})"},
    };
    for (int failure : {RAD_E_IO, RAD_E_DEVICE, RAD_E_NOMEM, RAD_E_INVAL}) {
        status = failure;
        const bool unavailable = failure != RAD_E_INVAL;
        for (const auto& request : requests) {
            const auto reply = f.post(request.path, request.body);
            CHECK_EQ(reply.status, unavailable ? 503 : 400);
            const auto error = json::parse(reply.body)["error"];
            CHECK_EQ(error.value("type", ""), unavailable ? "service_unavailable" : "invalid_request_error");
            CHECK_EQ(error.value("param", ""), "model");
            CHECK(error.value("message", "").find("image lifecycle operation failed") != std::string::npos);
        }
        CHECK_EQ(f.srv->in_flight(), 0);
    }
}

TEST(image_registration_requires_local_operator_on_keyless_listeners) {
    int calls = 0;
    Fixture f(Fixture::Tweak{[&](ServerOptions& o, Deps&) {
        o.register_image = [&](const std::string&, const std::string&, std::string*) {
            ++calls;
            return RAD_OK;
        };
    }});
    HttpRequest req;
    req.method = "POST";
    req.path = "/v1/images/register_model";
    req.body = R"({"model":"secondary","path":"/models/secondary.rad"})";
    req.raw_peer = "192.0.2.1:12345";
    CHECK_EQ(f.srv->router().dispatch(req).status, 403);
    CHECK_EQ(calls, 0);
    req.raw_peer.clear();
    CHECK_EQ(f.srv->router().dispatch(req).status, 200);
    CHECK_EQ(calls, 1);
}

TEST(image_status_requires_local_operator_on_keyless_listeners) {
    Fixture f(Fixture::Tweak{[](ServerOptions& o, Deps&) {
        o.image_status = [] { return R"({"queued":0})"; };
    }});
    HttpRequest req;
    req.path = "/v1/images/status";
    req.raw_peer = "192.0.2.1:12345";
    CHECK_EQ(f.srv->router().dispatch(req).status, 403);
    req.raw_peer.clear();
    const auto response = f.srv->router().dispatch(req);
    CHECK_EQ(response.status, 200);
    CHECK_EQ(json::parse(response.body)["queued"].get<int>(), 0);
}

TEST(image_unload_requires_local_operator_on_keyless_listeners) {
    int calls = 0;
    Fixture f(Fixture::Tweak{[&](ServerOptions& o, Deps&) {
        o.unload_image = [&](const std::string& model, const std::function<bool()>& cancelled,
                             std::string*) {
            CHECK_EQ(model, "secondary");
            CHECK(!cancelled());
            ++calls;
            return RAD_OK;
        };
    }});
    HttpRequest req;
    req.method = "POST";
    req.path = "/v1/images/unload_model";
    req.body = R"({"model":"secondary"})";
    req.raw_peer = "192.0.2.1:12345";
    CHECK_EQ(f.srv->router().dispatch(req).status, 403);
    CHECK_EQ(calls, 0);
    req.raw_peer.clear();
    CHECK_EQ(f.srv->router().dispatch(req).status, 200);
    CHECK_EQ(calls, 1);
}

RAD_TEST_MAIN()
