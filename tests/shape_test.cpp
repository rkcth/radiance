/* shape_test.cpp -- the shape generator's host geometry (core/gen/shape.h) and its pass driver
 * (core/gen/shape_driver.h). The plugin's side of each pass is played by the test: a query pass is
 * answered with a sphere's field at the points the driver staged.
 */
#include "sched_fakes.h"

using namespace sched_fake;

#include "gen/shape_driver.h"
#include "rad_plugin.h"

#include <random>
#include "util/simd.h"
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace rad;

namespace {

/* A small declaration: a 28-pixel encoder picture in 14-pixel patches, a latent of 8 rows of 4,
 * a step of both branches with 5 conditioning rows each. */
RadShapeGenDecl decl_small() {
    RadShapeGenDecl d{};
    d.canvas = 32;
    d.border_ratio = 0.15;
    d.enc_size = 28;
    d.enc_patch = 14;
    for (int c = 0; c < 3; ++c) { d.pix_mean[c] = 0.5; d.pix_std[c] = 0.25; }
    d.latent_rows = 8;
    d.latent_channels = 4;
    d.step_rows = 2 * (5 + 8);
    d.default_steps = 4;
    d.max_steps = 8;
    d.default_guidance = 5.0;
    d.bounds = 1.0;
    d.level = 0.0;
    d.default_resolution = 64;
    d.min_resolution = 8;
    d.max_resolution = 128;
    d.max_lanes = 2;
    d.max_query = 4096;
    d.field_out = 1;
    return d;
}

/* An RGBA picture w x h, opaque inside the box [x0, x1) x [y0, y1) and transparent elsewhere. */
mm::FrameRGBA picture(int32_t w, int32_t h, int32_t x0, int32_t y0, int32_t x1, int32_t y1) {
    mm::FrameRGBA f;
    f.w = w;
    f.h = h;
    f.rgba.assign((size_t)w * h * 4, 0);
    for (int32_t y = y0; y < y1; ++y)
        for (int32_t x = x0; x < x1; ++x) {
            uint8_t* p = &f.rgba[((size_t)y * w + x) * 4];
            p[0] = 200; p[1] = 50; p[2] = 10; p[3] = 255;
        }
    return f;
}

gen::ShapeGenInfo info_small() { return gen::shape_info(decl_small(), "test"); }

/* A sphere of radius `r` at the origin, as an occupancy logit: positive inside. */
float sphere(float x, float y, float z, float r) { return r - std::sqrt(x * x + y * y + z * z); }

/* The field of a sphere on level r's grid. */
std::vector<float> sphere_grid(int32_t r, double bounds, float radius) {
    const std::vector<float> ax = gen::shape_axis(bounds, r);
    const int64_t n = (int64_t)r + 1;
    std::vector<float> f((size_t)(n * n * n));
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j)
            for (int64_t k = 0; k < n; ++k)
                f[(size_t)((i * n + j) * n + k)] = sphere(ax[(size_t)i], ax[(size_t)j],
                                                          ax[(size_t)k], radius);
    return f;
}

struct ShapeFixture {
    Program prog;
    gen::ShapeDriver drv;
    void build() {
        prog.shape_gen = decl_small();
        prog.sgen_name = "test";
        prog.ctx.max_tok = 4096;
        prog.ctx.max_seqs = 1;
    }
    int start() { return drv.init(prog, { gen::ShapeDriver::RankIO{} }); }
    std::unique_ptr<gen::ShapeRequest> req(int32_t steps = 0, int32_t res = 16) {
        auto r = std::make_unique<gen::ShapeRequest>();
        std::string why;
        if (gen::shape_picture(info_small(), picture(40, 30, 5, 3, 30, 27), &r->patches, &why) < 0)
            return nullptr;
        r->steps = steps;
        r->resolution = res;
        r->seed = 7;
        r->ticket = std::make_shared<gen::ShapeTicket>();
        return r;
    }
    /* The plugin's answer to a query pass: the sphere's field at each staged point, steepened
     * `gain` times as an occupancy logit is (the model's cross zero within a cell or two). */
    void answer(const RadBatch* b, float radius, float gain = 20.0f) {
        const RadShapeGenBatch* S = b->sgen_b;
        for (const gen::IPassDriver::Readback& rb : drv.readbacks()) {
            float* x = static_cast<float*>(rb.host);
            for (int64_t i = 0; i < S->n_query; ++i)
                x[i] = gain * sphere(S->query[3 * i], S->query[3 * i + 1], S->query[3 * i + 2],
                                     radius);
        }
    }
};

char kind_of(int gen) {
    switch (gen) {
    case RAD_GEN_PASS_SCOND:   return 'C';
    case RAD_GEN_PASS_SSTEP:   return 'S';
    case RAD_GEN_PASS_SLATENT: return 'L';
    case RAD_GEN_PASS_SQUERY:  return 'Q';
    default:                   return '?';
    }
}

/* Every edge of a closed, consistently wound mesh is in exactly two faces, once each way. */
bool watertight(const gen::ShapeMesh& m) {
    std::map<std::pair<uint32_t, uint32_t>, int> e;
    for (size_t t = 0; t + 2 < m.f.size(); t += 3)
        for (int q = 0; q < 3; ++q) ++e[{ m.f[t + q], m.f[t + (q + 1) % 3] }];
    for (const auto& [k, n] : e) {
        if (n != 1) return false;
        auto it = e.find({ k.second, k.first });
        if (it == e.end() || it->second != 1) return false;
    }
    return !e.empty();
}

/* The signed volume of a closed mesh: positive when its faces wind counter-clockwise seen from
 * outside. */
double volume(const gen::ShapeMesh& m) {
    double v = 0.0;
    for (size_t t = 0; t + 2 < m.f.size(); t += 3) {
        const float* a = &m.v[(size_t)m.f[t] * 3];
        const float* b = &m.v[(size_t)m.f[t + 1] * 3];
        const float* c = &m.v[(size_t)m.f[t + 2] * 3];
        v += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) +
              a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0;
    }
    return v;
}

}  /* namespace */

/* The sampler's grid points are numpy's linspace(0, 1) in f32, the last exactly 1. */
TEST(the_sigmas_are_linspace_from_noise_to_the_shape) {
    const std::vector<float> s = gen::shape_sigmas(50);
    REQUIRE_EQ(s.size(), (size_t)50);
    CHECK_EQ(s.front(), 0.0f);
    CHECK_EQ(s.back(), 1.0f);
    CHECK_EQ(s[1], (float)(1.0 / 49.0));
    CHECK_EQ(s[30], (float)(30 * (1.0 / 49.0)));
    for (size_t i = 1; i < s.size(); ++i) CHECK(s[i] > s[i - 1]);
    CHECK(gen::shape_sigmas(1).empty());
    CHECK_EQ(gen::shape_sigmas(2).size(), (size_t)2);
}

/* The grid's levels: halved while at least 63, coarsest first; one level below 63. */
TEST(the_grid_levels_halve_down_to_63) {
    CHECK(gen::shape_levels(256) == (std::vector<int32_t>{ 64, 128, 256 }));
    CHECK(gen::shape_levels(384) == (std::vector<int32_t>{ 96, 192, 384 }));
    CHECK(gen::shape_levels(512) == (std::vector<int32_t>{ 64, 128, 256, 512 }));
    CHECK(gen::shape_levels(63) == (std::vector<int32_t>{ 63 }));
    CHECK(gen::shape_levels(32) == (std::vector<int32_t>{ 32 }));
    CHECK(gen::shape_levels(130) == (std::vector<int32_t>{ 65, 130 }));
    const std::vector<float> a = gen::shape_axis(1.01, 4);
    REQUIRE_EQ(a.size(), (size_t)5);
    CHECK_EQ(a[0], -1.01f);
    CHECK_EQ(a[2], (float)(2 * (2.02 / 4) - 1.01));
    CHECK_EQ(a[4], 1.01f);
}

/* The finer level samples a shell around the coarse surface and nothing far from it. */
TEST(refinement_samples_only_near_the_surface) {
    const int32_t rp = 32, r = 64;
    const float radius = 0.5f;
    const std::vector<float> coarse = sphere_grid(rp, 1.0, radius);
    for (bool last : { false, true }) {
        const std::vector<uint32_t> pts = gen::shape_refine(coarse, rp, r, 0.0, last);
        REQUIRE(!pts.empty());
        const std::vector<float> ax = gen::shape_axis(1.0, r);
        const int64_t n = r + 1;
        double far = 0.0;
        for (uint32_t p : pts) {
            const float d = std::fabs(sphere(ax[p / (n * n)], ax[p / n % n], ax[p % n], radius));
            far = std::max(far, (double)d);
        }
        /* Within 0.95 of zero counts as near (the field here is a distance), plus the dilation. */
        CHECK(far < 0.95 + 4.0 * 2.0 / rp);
        CHECK((int64_t)pts.size() < n * n * n);
        /* Every fine cell the surface crosses has all eight corners sampled. */
        std::vector<char> in((size_t)(n * n * n), 0);
        for (uint32_t p : pts) in[p] = 1;
        bool covered = true;
        for (int64_t i = 0; i + 1 < n && covered; ++i)
            for (int64_t j = 0; j + 1 < n && covered; ++j)
                for (int64_t k = 0; k + 1 < n && covered; ++k) {
                    int pos = 0, neg = 0;
                    for (int c = 0; c < 8; ++c) {
                        const float v = sphere(ax[i + (c & 1)], ax[j + (c >> 1 & 1)],
                                               ax[k + (c >> 2 & 1)], radius);
                        (v > 0 ? pos : neg)++;
                    }
                    if (!pos || !neg) continue;
                    for (int c = 0; c < 8; ++c)
                        covered = covered &&
                                  in[(size_t)(((i + (c & 1)) * n + j + (c >> 1 & 1)) * n + k +
                                              (c >> 2 & 1))];
                }
        CHECK(covered);
    }
}

/* Marching cubes over a sphere: a closed surface wound outwards, its vertices on the sphere
 * (scaled by r / (r + 1), the reference's placement), its volume the ball's. */
TEST(marching_cubes_makes_a_closed_outward_sphere) {
    const int32_t r = 48;
    const float radius = 0.6f;
    gen::ShapeMesh m;
    gen::shape_mesh(sphere_grid(r, 1.0, radius), r, 1.0, 0.0, &m);
    REQUIRE(!m.f.empty());
    CHECK(watertight(m));
    const double scale = (double)r / (r + 1);
    double worst = 0.0;
    for (size_t i = 0; i + 2 < m.v.size(); i += 3) {
        const double x = m.v[i] + 1.0 / (r + 1), y = m.v[i + 1] + 1.0 / (r + 1),
                     z = m.v[i + 2] + 1.0 / (r + 1);
        worst = std::max(worst, std::fabs(std::sqrt(x * x + y * y + z * z) - radius * scale));
    }
    CHECK(worst < 2.0 / r);
    const double ball = 4.0 / 3.0 * M_PI * std::pow(radius * scale, 3);
    CHECK(volume(m) > 0.0);
    CHECK(std::fabs(volume(m) - ball) / ball < 0.02);
}

/* Every inside/outside pattern of a cube gives a closed surface: a field of random signs. */
TEST(marching_cubes_is_closed_for_every_cube_case) {
    const int32_t r = 12;
    const int64_t n = r + 1;
    std::vector<float> f((size_t)(n * n * n));
    uint32_t s = 12345;
    for (float& v : f) {
        s = s * 1664525u + 1013904223u;
        v = (float)((s >> 8) & 0xffff) / 65535.0f - 0.5f;
    }
    /* A border outside the shape, so the surface closes inside the grid. */
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j)
            for (int64_t k = 0; k < n; ++k)
                if (!i || !j || !k || i == r || j == r || k == r)
                    f[(size_t)((i * n + j) * n + k)] = -1.0f;
    gen::ShapeMesh m;
    gen::shape_mesh(f, r, 1.0, 0.0, &m);
    REQUIRE(!m.f.empty());
    CHECK(watertight(m));
    CHECK(volume(m) > 0.0);
}

/* A cube with a corner not sampled makes nothing, and an unsampled grid no surface. */
TEST(marching_cubes_skips_unsampled_cubes) {
    const int32_t r = 16;
    std::vector<float> f = sphere_grid(r, 1.0, 0.5f);
    gen::ShapeMesh whole;
    gen::shape_mesh(f, r, 1.0, 0.0, &whole);
    std::vector<float> holes = f;
    const int64_t n = r + 1;
    holes[(size_t)((8 * n + 8) * n + 4)] = NAN;   /* a point on the sphere's surface band */
    gen::ShapeMesh m;
    gen::shape_mesh(holes, r, 1.0, 0.0, &m);
    CHECK(m.f.size() < whole.f.size());
    std::vector<float> none(f.size(), NAN);
    gen::shape_mesh(none, r, 1.0, 0.0, &m);
    CHECK(m.f.empty());
}

/* The files: a GLB with its header and chunks sized, and the three others. */
TEST(the_mesh_exports_as_glb_obj_ply_and_stl) {
    gen::ShapeMesh m;
    gen::shape_mesh(sphere_grid(16, 1.0, 0.5f), 16, 1.0, 0.0, &m);
    REQUIRE(!m.f.empty());
    std::string bytes, mime, why;
    CHECK_OK(gen::shape_export(m, "glb", &bytes, &mime, &why));
    CHECK_EQ(mime, std::string("model/gltf-binary"));
    REQUIRE(bytes.size() > 28);
    uint32_t hdr[5];
    std::memcpy(hdr, bytes.data(), sizeof hdr);
    CHECK_EQ(hdr[0], 0x46546C67u);
    CHECK_EQ(hdr[1], 2u);
    CHECK_EQ(hdr[2], (uint32_t)bytes.size());
    CHECK_EQ(hdr[3] % 4, 0u);
    CHECK_EQ(hdr[4], 0x4E4F534Au);
    uint32_t bin[2];
    std::memcpy(bin, bytes.data() + 20 + hdr[3], sizeof bin);
    CHECK_EQ(bin[1], 0x004E4942u);
    CHECK_EQ((size_t)bin[0], m.v.size() * 4 * 2 + m.f.size() * 4);
    CHECK_OK(gen::shape_export(m, "obj", &bytes, &mime, &why));
    CHECK(bytes.rfind("v ", 0) == 0);
    CHECK_OK(gen::shape_export(m, "ply", &bytes, &mime, &why));
    CHECK(bytes.rfind("ply\n", 0) == 0);
    CHECK_OK(gen::shape_export(m, "stl", &bytes, &mime, &why));
    CHECK_EQ(bytes.size(), (size_t)84 + m.f.size() / 3 * 50);
    CHECK(gen::shape_export(m, "fbx", &bytes, &mime, &why) < 0);
}

/* The picture: the alpha's box centred on white, normalised, cut into patches; an empty alpha
 * refused. */
TEST(the_picture_is_recentred_on_white_and_cut_into_patches) {
    const gen::ShapeGenInfo g = info_small();
    std::vector<float> p;
    std::string why;
    CHECK_OK(gen::shape_picture(g, picture(40, 30, 5, 3, 30, 27), &p, &why));
    REQUIRE_EQ(p.size(), (size_t)(2 * 2 * 3 * 14 * 14));
    /* The corners are white, (1 - 0.5) / 0.25 = 2 in every channel; the centre the colour. */
    const float white = (float)((1.0 - 0.5) / 0.25);
    CHECK_NEAR(p[0], white, 1e-3);
    CHECK_NEAR(p[(size_t)(2 * 14 * 14)], white, 1e-3);   /* blue, pixel (0, 0) */
    const int64_t pd = 3 * 14 * 14;
    /* Patch (1, 1) holds pixels (14.., 14..); its first pixel, red channel, is the box's colour. */
    CHECK_NEAR(p[(size_t)(3 * pd)], (float)((200.0 / 255.0 - 0.5) / 0.25), 0.1);
    CHECK(gen::shape_picture(g, picture(20, 20, 0, 0, 0, 0), &p, &why) < 0);
    CHECK(gen::shape_picture(g, picture(20, 20, 5, 5, 6, 15), &p, &why) < 0);
}

/* A request runs its picture, its steps, the latent, then the grid level by level; the field the
 * ticket hands back meshes into the sphere the queries were answered with. */
TEST(a_request_runs_cond_steps_latent_and_the_grid_levels) {
    ShapeFixture f;
    f.build();
    CHECK_OK(f.start());
    auto r = f.req(4, 128);
    REQUIRE(r != nullptr);
    auto t = r->ticket;
    std::string why;
    CHECK_OK(f.drv.submit(std::move(r), &why));
    CHECK_EQ(t->steps, 3);
    CHECK_EQ(t->resolution, 128);
    std::string kinds;
    const std::vector<float> sig = gen::shape_sigmas(4);
    int steps = 0;
    for (int i = 0; i < 1000; ++i) {
        const RadBatch* b = f.drv.next(i);
        if (!b) break;
        const RadShapeGenBatch* S = b->sgen_b;
        REQUIRE(S != nullptr);
        kinds += kind_of(b->gen);
        CHECK_EQ(S->lanes[0], S->lane);
        CHECK_EQ(S->lat_cu[2], 16);
        if (b->gen == RAD_GEN_PASS_SCOND) CHECK_EQ(b->n_tok, (int64_t)4);
        if (b->gen == RAD_GEN_PASS_SSTEP) {
            CHECK_EQ(b->n_tok, (int64_t)26);
            CHECK_EQ(S->sigma[0], sig[(size_t)steps]);
            CHECK_EQ(S->sigma_next[0], sig[(size_t)steps + 1]);
            CHECK_EQ(S->cfg[0], 5.0f);
            CHECK_EQ(S->pair[0], 1);
            CHECK_EQ(S->pair[1], -2);
            CHECK_EQ(S->flags[0], steps == 0 ? (int32_t)RAD_GEN_F_FIRST : 0);
            CHECK_EQ(S->flags[1], 0);
            CHECK_EQ(S->seed[0], 7u);
            ++steps;
        }
        if (b->gen == RAD_GEN_PASS_SQUERY) {
            CHECK(S->n_query > 0 && S->n_query <= 4096);
            CHECK_EQ(S->query_cu[1], (int32_t)S->n_query);
            f.answer(b, 0.5f);
        }
        CHECK_OK(f.drv.done());
    }
    /* 65^3 coarse points in passes of 4096, then the finer level's shell. */
    REQUIRE(kinds.size() > 5);
    CHECK_EQ(kinds.substr(0, 5), std::string("CSSSL"));
    CHECK(kinds.find_first_not_of('Q', 5) == std::string::npos);
    CHECK((int64_t)(kinds.size() - 5) >= (65 * 65 * 65 + 4095) / 4096);
    REQUIRE(t->finished());
    CHECK_OK(t->status());
    int32_t res = 0;
    std::vector<float> field = t->take_field(&res);
    CHECK_EQ(res, 128);
    gen::ShapeMesh m;
    gen::shape_mesh(field, res, 1.0, 0.0, &m);
    REQUIRE(!m.f.empty());
    CHECK(watertight(m));
    CHECK(volume(m) > 0.0);
    /* Hierarchical sampling queried far fewer points than the dense grid has. */
    CHECK(t->points > 65 * 65 * 65);
    CHECK(t->points < (int64_t)129 * 129 * 129 / 4);
}

/* Dense sampling queries every point of the finest grid, in one level. */
TEST(dense_sampling_queries_every_point_once) {
    ShapeFixture f;
    f.build();
    CHECK_OK(f.start());
    auto r = f.req(2, 20);
    REQUIRE(r != nullptr);
    r->dense = true;
    auto t = r->ticket;
    std::string why;
    CHECK_OK(f.drv.submit(std::move(r), &why));
    for (int i = 0; i < 100; ++i) {
        const RadBatch* b = f.drv.next(i);
        if (!b) break;
        if (b->gen == RAD_GEN_PASS_SQUERY) f.answer(b, 0.5f);
        CHECK_OK(f.drv.done());
    }
    REQUIRE(t->finished());
    CHECK_EQ(t->points, (int64_t)21 * 21 * 21);
}

/* What a request may not name is refused at submit, with the range. */
TEST(a_request_out_of_range_is_refused_at_submit) {
    ShapeFixture f;
    f.build();
    CHECK_OK(f.start());
    std::string why;
    auto r = f.req(9);
    CHECK(f.drv.submit(std::move(r), &why) < 0);
    CHECK(why.find("num_inference_steps") != std::string::npos);
    r = f.req(4, 200);
    CHECK(f.drv.submit(std::move(r), &why) < 0);
    CHECK(why.find("octree_resolution") != std::string::npos);
    r = f.req(4);
    r->guidance = -1.0;
    CHECK(f.drv.submit(std::move(r), &why) < 0);
    r = f.req(4);
    r->patches.pop_back();
    CHECK(f.drv.submit(std::move(r), &why) < 0);
}

/* A field that crosses the level nowhere finishes after the coarsest level. */
TEST(a_field_without_a_surface_ends_after_its_first_level) {
    ShapeFixture f;
    f.build();
    CHECK_OK(f.start());
    auto r = f.req(2, 128);
    auto t = r->ticket;
    std::string why;
    CHECK_OK(f.drv.submit(std::move(r), &why));
    int queries = 0;
    for (int i = 0; i < 1000; ++i) {
        const RadBatch* b = f.drv.next(i);
        if (!b) break;
        if (b->gen == RAD_GEN_PASS_SQUERY) {
            f.answer(b, -5.0f);   /* a sphere of negative radius: outside everywhere */
            ++queries;
        }
        CHECK_OK(f.drv.done());
    }
    REQUIRE(t->finished());
    CHECK_EQ(queries, (65 * 65 * 65 + 4095) / 4096);
    int32_t res = 0;
    gen::ShapeMesh m;
    gen::shape_mesh(t->take_field(&res), res, 1.0, 0.0, &m);
    CHECK(m.f.empty());
}

/* ================================================================== bitmap kernels vs per point */
/* The per-point forms shape_refine and shape_mesh had before they classified rows with
 * simd::level_bits_f32, with the case table they were built on, kept as the reference the
 * bitmap forms must match exactly. */
namespace {

/* A 3 x 3 x 3 dilation of an n^3 mask (zero outside), as separable maxima along each axis. */
void ref_dilate(std::vector<uint8_t>& m, int64_t n) {
    std::vector<uint8_t> t(m.size());
    /* One axis: each voxel ORs its two neighbours along it, where they exist. Along the
     * contiguous axis a line of n voxels at a time; along the others whole rows of `stride`
     * voxels, each with the rows either side of it. */
    auto pass = [&](const std::vector<uint8_t>& src, std::vector<uint8_t>& dst, int64_t stride) {
        const uint8_t* s = src.data();
        uint8_t* d = dst.data();
        const int64_t total = (int64_t)src.size(), plane = n * stride;
        for (int64_t base = 0; base < total; base += plane) {
            if (stride == 1) {
                simd::or_adjacent_u8(d + base, s + base, (size_t)n);
                continue;
            }
            for (int64_t c = 0; c < n; ++c) {
                const uint8_t* a = s + base + c * stride;
                simd::or3_u8(d + base + c * stride, a, c > 0 ? a - stride : nullptr,
                             c + 1 < n ? a + stride : nullptr, (size_t)stride);
            }
        }
    };
    pass(m, t, 1);
    pass(t, m, n);
    pass(m, t, n * n);
    m.swap(t);
}


std::vector<uint32_t> ref_refine(const std::vector<float>& prev, int32_t rp, int32_t r,
                                   double level, bool last) {
    const int64_t np = (int64_t)rp + 1, n = (int64_t)r + 1;
    std::vector<uint8_t> near((size_t)(np * np * np), 0);
    auto sgn = [](float v) { return (v > 0.0f) - (v < 0.0f); };
    const float lv = (float)level;
    for (int64_t i = 0; i < np; ++i)
        for (int64_t j = 0; j < np; ++j)
            for (int64_t k = 0; k < np; ++k) {
                const int64_t p = (i * np + j) * np + k;
                const float raw = prev[(size_t)p];
                if (std::isnan(raw)) continue;
                const float v = raw - lv;
                const int s = sgn(v);
                bool cross = std::fabs(v) < 0.95f;
                const int64_t nb[6][3] = { { i - 1, j, k }, { i + 1, j, k }, { i, j - 1, k },
                                           { i, j + 1, k }, { i, j, k - 1 }, { i, j, k + 1 } };
                for (int q = 0; q < 6 && !cross; ++q) {
                    const int64_t a = nb[q][0], b = nb[q][1], c = nb[q][2];
                    /* Outside the grid the reference replicates the border, and a neighbour not
                     * sampled stands in as the point itself: neither crosses. */
                    if (a < 0 || a >= np || b < 0 || b >= np || c < 0 || c >= np) continue;
                    const float u = prev[(size_t)((a * np + b) * np + c)];
                    if (std::isnan(u)) continue;
                    cross = sgn(u - lv) != s;
                }
                near[(size_t)p] = cross;
            }
    if (!last) ref_dilate(near, np);
    std::vector<uint8_t> mark((size_t)(n * n * n), 0);
    auto up = [&](int64_t c) { return (c * r + rp / 2) / rp; };
    for (int64_t i = 0; i < np; ++i)
        for (int64_t j = 0; j < np; ++j)
            for (int64_t k = 0; k < np; ++k)
                if (near[(size_t)((i * np + j) * np + k)])
                    mark[(size_t)((up(i) * n + up(j)) * n + up(k))] = 1;
    ref_dilate(mark, n);
    if (last) ref_dilate(mark, n);
    std::vector<uint32_t> pts;
    for (int64_t p = 0; p < (int64_t)mark.size(); ++p)
        if (mark[(size_t)p]) pts.push_back((uint32_t)p);
    return pts;
}


/* Corner c of a cube is at (c & 1, c >> 1 & 1, c >> 2 & 1) along the grid's (i, j, k). Edge e is
 * along axis e / 4 from corner kEdge[e][0] to kEdge[e][1]. */
struct McRef {
    int32_t edge[12][2];
    /* Each case's triangles as edge triples, -1 ended. */
    int8_t  tri[256][31];
};

McRef build_ref_tables() {
    McRef T{};
    int ne = 0;
    for (int a = 0; a < 3; ++a)
        for (int c = 0; c < 8; ++c)
            if (!(c >> a & 1)) { T.edge[ne][0] = c; T.edge[ne][1] = c | (1 << a); ++ne; }
    auto edge_of = [&](int c0, int c1) {
        for (int e = 0; e < 12; ++e)
            if ((T.edge[e][0] == c0 && T.edge[e][1] == c1) ||
                (T.edge[e][0] == c1 && T.edge[e][1] == c0))
                return e;
        return -1;
    };
    auto pos = [](int c, double* p) { p[0] = c & 1; p[1] = c >> 1 & 1; p[2] = c >> 2 & 1; };
    for (int m = 0; m < 256; ++m) {
        for (int i = 0; i < 31; ++i) T.tri[m][i] = -1;
        if (m == 0 || m == 255) continue;
        auto in = [&](int c) { return (m >> c & 1) != 0; };
        /* Each face's segments between its crossing edges. A face with all four edges crossing
         * (its inside corners diagonal) separates the inside corners, the rule both cubes that
         * share the face apply, so the surface has no cracks. */
        std::vector<std::array<int, 2>> seg;
        for (int a = 0; a < 3; ++a)
            for (int s = 0; s < 2; ++s) {
                const int u = (a + 1) % 3, v = (a + 2) % 3;
                const int base = s << a;
                const int cyc[4] = { base, base | (1 << u), base | (1 << u) | (1 << v),
                                     base | (1 << v) };
                int fe[4];
                bool cr[4];
                int ncr = 0;
                for (int q = 0; q < 4; ++q) {
                    fe[q] = edge_of(cyc[q], cyc[(q + 1) % 4]);
                    cr[q] = in(cyc[q]) != in(cyc[(q + 1) % 4]);
                    ncr += cr[q];
                }
                if (ncr == 2) {
                    int a0 = -1, a1 = -1;
                    for (int q = 0; q < 4; ++q) {
                        if (!cr[q]) continue;
                        if (a0 < 0) a0 = fe[q]; else a1 = fe[q];
                    }
                    seg.push_back({ a0, a1 });
                } else if (ncr == 4) {
                    for (int q = 0; q < 4; ++q)
                        if (in(cyc[q])) seg.push_back({ fe[(q + 3) % 4], fe[q] });
                }
            }
        /* The segments close into loops; each crossing edge is in exactly two. */
        std::vector<bool> used(seg.size(), false);
        int nt = 0;
        for (size_t s0 = 0; s0 < seg.size(); ++s0) {
            if (used[s0]) continue;
            std::vector<int> loop = { seg[s0][0], seg[s0][1] };
            used[s0] = true;
            for (bool grew = true; grew;) {
                grew = false;
                for (size_t q = 0; q < seg.size(); ++q) {
                    if (used[q]) continue;
                    const int tail = loop.back();
                    int nx = -1;
                    if (seg[q][0] == tail) nx = seg[q][1];
                    else if (seg[q][1] == tail) nx = seg[q][0];
                    if (nx < 0) continue;
                    used[q] = true;
                    grew = true;
                    if (nx != loop.front()) loop.push_back(nx);
                }
            }
            /* Orientation: the loop's normal (Newell's, over the edges' midpoints) points from the
             * inside corners to the outside ones. */
            double nrm[3] = { 0, 0, 0 }, out[3] = { 0, 0, 0 };
            std::vector<std::array<double, 3>> mid;
            for (int e : loop) {
                double p0[3], p1[3];
                pos(T.edge[e][0], p0);
                pos(T.edge[e][1], p1);
                mid.push_back({ (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, (p0[2] + p1[2]) / 2 });
                const double sg = in(T.edge[e][0]) ? 1.0 : -1.0;
                for (int d = 0; d < 3; ++d) out[d] += sg * (p1[d] - p0[d]);
            }
            for (size_t q = 0; q < mid.size(); ++q) {
                const auto& c = mid[q];
                const auto& d = mid[(q + 1) % mid.size()];
                nrm[0] += (c[1] - d[1]) * (c[2] + d[2]);
                nrm[1] += (c[2] - d[2]) * (c[0] + d[0]);
                nrm[2] += (c[0] - d[0]) * (c[1] + d[1]);
            }
            if (nrm[0] * out[0] + nrm[1] * out[1] + nrm[2] * out[2] < 0)
                std::reverse(loop.begin(), loop.end());
            for (size_t q = 1; q + 1 < loop.size() && nt < 10; ++q) {
                T.tri[m][3 * nt] = (int8_t)loop[0];
                T.tri[m][3 * nt + 1] = (int8_t)loop[q];
                T.tri[m][3 * nt + 2] = (int8_t)loop[q + 1];
                ++nt;
            }
        }
    }
    return T;
}

const McRef& mc_ref() {
    static const McRef t = build_ref_tables();
    return t;
}

void ref_mesh(const std::vector<float>& field, int32_t r, double bounds, double level,
                gen::ShapeMesh* out) {
    out->v.clear();
    out->f.clear();
    const McRef& T = mc_ref();
    const int64_t n = (int64_t)r + 1, nn = n * n;
    const float lv = (float)level;
    const double sc = 2.0 * bounds / (double)n;
    /* An edge's vertex by its first grid point: the x edges between planes i and i + 1, and the
     * y and z edges of planes i and i + 1. */
    std::vector<int32_t> ex((size_t)nn), ey[2], ez[2];
    for (int p = 0; p < 2; ++p) { ey[p].assign((size_t)nn, -1); ez[p].assign((size_t)nn, -1); }
    auto value = [&](int64_t i, int64_t j, int64_t k) {
        return field[(size_t)((i * n + j) * n + k)];
    };
    for (int64_t i = 0; i + 1 < n; ++i) {
        std::fill(ex.begin(), ex.end(), -1);
        if (i > 0) {
            std::swap(ey[0], ey[1]);
            std::swap(ez[0], ez[1]);
            std::fill(ey[1].begin(), ey[1].end(), -1);
            std::fill(ez[1].begin(), ez[1].end(), -1);
        }
        for (int64_t j = 0; j + 1 < n; ++j)
            for (int64_t k = 0; k + 1 < n; ++k) {
                float v[8];
                bool skip = false;
                int m = 0;
                for (int c = 0; c < 8 && !skip; ++c) {
                    v[c] = value(i + (c & 1), j + (c >> 1 & 1), k + (c >> 2 & 1));
                    if (std::isnan(v[c])) skip = true;
                    else if (v[c] > lv) m |= 1 << c;
                }
                if (skip || m == 0 || m == 255) continue;
                for (int t = 0; t < 30 && T.tri[m][t] >= 0; ++t) {
                    const int e = T.tri[m][t];
                    const int c0 = T.edge[e][0], c1 = T.edge[e][1], a = e / 4;
                    const int64_t bi = c0 & 1, bj = c0 >> 1 & 1, bk = c0 >> 2 & 1;
                    int32_t* slot = a == 0 ? &ex[(size_t)((j + bj) * n + k + bk)]
                                  : a == 1 ? &ey[bi][(size_t)(j * n + k + bk)]
                                           : &ez[bi][(size_t)((j + bj) * n + k)];
                    if (*slot < 0) {
                        const float t0 = v[c0], t1 = v[c1];
                        const float f = t1 != t0 ? (lv - t0) / (t1 - t0) : 0.5f;
                        float g[3] = { (float)(i + bi), (float)(j + bj), (float)(k + bk) };
                        g[a] += f;
                        *slot = (int32_t)(out->v.size() / 3);
                        for (int d = 0; d < 3; ++d) out->v.push_back((float)(g[d] * sc - bounds));
                    }
                    out->f.push_back((uint32_t)*slot);
                }
            }
    }
}


/* A field with NaNs, values exactly at the level, and a surface crossing everywhere. */
std::vector<float> rough_field(int64_t n, std::mt19937& rng) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> f((size_t)(n * n * n));
    for (auto& x : f) {
        const float y = u(rng);
        x = y > 0.97f ? std::nanf("") : y > 0.9f ? 0.0f : y * 1.5f;
    }
    return f;
}

}  /* namespace */

TEST(shape_mesh_and_refine_match_the_per_point_forms_at_every_simd_level) {
    std::mt19937 rng(20261107);
    const int top = simd::detect_level();
    for (int lvl = 0; lvl <= top; ++lvl) {
        simd::force_level(lvl);
        for (int32_t r : { 1, 2, 5, 31, 62, 63, 64, 65, 66, 90 }) {
            const int64_t n = (int64_t)r + 1;
            const std::vector<float> f = rough_field(n, rng);
            gen::ShapeMesh got, want;
            gen::shape_mesh(f, r, 1.0, 0.0, &got);
            ref_mesh(f, r, 1.0, 0.0, &want);
            CHECK(got.f == want.f);
            CHECK(got.v == want.v);
            for (bool last : { false, true }) {
                const int32_t rf = r * 2;
                CHECK(gen::shape_refine(f, r, rf, 0.0, last) == ref_refine(f, r, rf, 0.0, last));
                CHECK(gen::shape_refine(f, r, rf, 0.3, last) == ref_refine(f, r, rf, 0.3, last));
            }
        }
    }
    simd::force_level(top);
}

RAD_TEST_MAIN()
