#include "service.h"

#include "build/startup.h"
#include "device/device.h"
#include "engine_hostmem.h"
#include "engine_priv.h"
#include "format/radfile.h"
#include "gen/driver.h"
#include "mm/media.h"
#include "util/stallwatch.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <limits>
#include <thread>

#include "util/json.h"

namespace rad {
namespace {

Config image_config(const Config& text) {
    Config image;
    image.model = text.image_model;
    image.lora = text.lora;
    image.radiance_home = text.radiance_home;
    image.kernel_hierarchy = text.kernel_hierarchy;
    image.accept_reference_kernels = text.accept_reference_kernels;
    image.proceed_incompatible = text.proceed_incompatible;
    image.tp = text.tp;
    if (text.image_embedding_placement != TablePlacement::Auto) {
        image.embedding_placement = text.image_embedding_placement;
        image.embedding_placement_given = true;
    }
    image.device_ordinals = text.device_ordinals;
    image.p2p = text.p2p;
    image.n_spec = 0;
    /* One guided conditioning can occupy two slots. */
    image.max_seqs = 2;
    image.given["--max-num-seqs"] = "shared image runtime";
    image.image_gen_max_pixels = text.image_gen_max_pixels ? text.image_gen_max_pixels : 1024 * 1024;
    image.image_gen_max_refs = text.image_gen_max_refs ? text.image_gen_max_refs : 1;
    image.gpu_memory_mib = text.gpu_memory_mib;
    image.gpu_headroom_mib = text.gpu_headroom_mib;
    image.host_headroom_mib = text.host_headroom_mib;
    return image;
}

constexpr int64_t kMiB = 1024 * 1024;
constexpr int64_t kCopyChunk = 256 * kMiB;

/* The file identifies the source bytes. Compare each rank's storage and layout before
 * using those bytes in a different engine. No addresses survive in this description. */
struct CachedWeight {
    std::string name, source, layout;
    std::vector<int64_t> placement;
    bool operator==(const CachedWeight&) const = default;
};

bool describe_weight_pool(const Program& program, const Pool& pool,
                          std::vector<CachedWeight>* weights) {
    const auto base = reinterpret_cast<uintptr_t>(pool.base());
    for (size_t i = 1; i < program.weights.size(); ++i) {
        const auto& w = program.weights[i];
        /* Mapped weights are rebound and host weights are loaded separately. */
        if (w.tier == Tier::Mapped || w.tier == Tier::HostPinned ||
            (!w.ptr && w.decl.optional)) continue;
        const auto addr = reinterpret_cast<uintptr_t>(w.ptr);
        if (w.tier != Tier::VRAM || addr < base || addr - base > uint64_t(pool.used()) ||
            w.stored_bytes < 0 || w.stored_bytes > pool.used() - int64_t(addr - base)) return false;
        CachedWeight key{w.name, w.source, w.layout_tag,
                         {int64_t(i), int64_t(addr - base), w.stored_bytes, w.stored_dtype,
                          w.stored_rank, w.decl.dtype, w.decl.rank, w.decl.shard,
                          w.decl.n_row_parts, w.shard_lo, w.shard_hi, w.n_part_spans, w.n_sel,
                          w.identity, w.widen, w.narrow, w.lay_op, w.lay_band, w.lay_dom,
                          w.lay_operand}};
        key.placement.insert(key.placement.end(), w.stored_shape, w.stored_shape + w.stored_rank);
        key.placement.insert(key.placement.end(), w.decl.shape, w.decl.shape + w.decl.rank);
        key.placement.insert(key.placement.end(), w.decl.row_parts,
                             w.decl.row_parts + w.decl.n_row_parts);
        for (int p = 0; p < w.n_part_spans; ++p)
            key.placement.insert(key.placement.end(), {w.part_lo[p], w.part_hi[p], w.part_total[p]});
        for (int p = 0; p < w.n_sel; ++p)
            key.placement.insert(key.placement.end(), {w.sel[p], w.sel_rows[p], w.sel_cols[p]});
        weights->push_back(std::move(key));
    }
    return true;
}

}  /* namespace */

struct SharedService::ImageMemory {
    ImageMemory(int device, int backing_kind) : device(device), backing_kind(backing_kind) {}

    const int device;
    const int backing_kind;
    char* weights = nullptr;
    void* backing = nullptr;
    int64_t weight_bytes = 0;
    int64_t offered_bytes = 0;
    bool weights_lent = false;
    std::vector<CachedWeight> layout;

    ~ImageMemory() { release_backing(); }

    void release_backing() {
        if (backing && rad_dev_set(device) >= 0) rad_dev_free(backing, backing_kind);
        backing = nullptr;
    }

    int copy(RadStream stream, bool restore) {
        auto* host = static_cast<char*>(backing);
        for (int64_t off = 0; off < weight_bytes;) {
            const int64_t bytes = std::min(weight_bytes - off, kCopyChunk);
            void* dst = restore ? weights + off : host + off;
            const void* src = restore ? host + off : weights + off;
            RAD_TRY(rad_memcpy_async(dst, src, bytes, stream));
            RAD_TRY(rad_stream_sync(stream));
            off += bytes;
        }
        return RAD_OK;
    }
};

struct SharedService::UnloadedExpertMemory {
    UnloadedExpertMemory(int device, int64_t bytes) : device(device), bytes(bytes) {}
    ~UnloadedExpertMemory() {
        if (base && rad_dev_set(device) >= 0) rad_dev_free(base, RAD_MEM_DEVICE);
    }
    const int device;
    const int64_t bytes;
    char* base = nullptr;
};

struct SharedService::CachedImage {
    RadFile::Identity identity;
    std::string path;
    int64_t bytes;
    TablePlacement embedding_placement;
    ImageSnapshot ranks;
};

SharedService::SharedService(Config cfg) : image_template_(image_config(cfg)), text_(std::move(cfg)) {
    text_.shared_ = this;
    text_.service_role_ = Engine::ServiceRole::PrimaryText;
    text_.cfg_.lora = LoraConfig{};
    text_.cfg_.image_gen_max_pixels = text_.cfg_.image_gen_max_refs = 0;
}

SharedService::~SharedService() = default;

SharedService::RegistryOwner::~RegistryOwner() {
    rad_tools_registry().close();
}

int SharedService::run(std::unique_ptr<StallWatch> watch) {
    const Config& cfg = text_.cfg_;
    if (cfg.kld() || cfg.calib() || cfg.debug_graph || cfg.deterministic) {
        RAD_ERR("--image-model requires HTTP serving with deterministic mode off");
        return RAD_E_UNSUPPORTED;
    }
    if (cfg.n_spec < 0 && cfg.spec_ngram == 0) {
        text_.cfg_.n_spec = 0;
        text_.decide("--num-speculative-tokens", "0",
                     "shared mode enables speculation only when explicitly requested");
    }
    text_.startup_begin_ns_ = rad_mono_ns();
    RAD_TRY(text_.declare_model());
    if (cfg.image_text_rank >= 0) {
        image_template_.tp = 1;
        image_template_.device_ordinals = {text_.ranks_[(size_t)cfg.image_text_rank]->device};
    }
    if (text_.tree_) {
        RAD_ERR("--image-model requires a linear draft chain because tree verification issues ahead");
        return RAD_E_UNSUPPORTED;
    }
    if (text_.program().is_pass_driven() || text_.program().logits_buf == RAD_NULL_HANDLE) {
        RAD_ERR("--model must generate text when --image-model is used");
        return RAD_E_UNSUPPORTED;
    }
    if (!mm::decoder_available()) {
        RAD_ERR("--image-model requires an image codec (RAD_WITH_FFMPEG)");
        return RAD_E_UNSUPPORTED;
    }
    /* The text planner measures the already-resident image allocations and
     * reserves the remaining image KV budget and lazy allocations separately. */
    RAD_TRY(plan_image_memory());
    if (const int status = text_.load_model(); status < 0) {
        RAD_ERR("shared service: the text model must fit with the image loan fully recalled; "
                "see the rank memory plan above");
        return status;
    }
    RAD_TRY(lend_image_memory());
    return text_.serve(std::move(watch));
}

int SharedService::read_free_vram(std::vector<int64_t>* free) const {
    free->clear();
    for (const auto& rank : text_.ranks_) {
        RadDeviceProps props{};
        RAD_TRY(rad_dev_props(rank->device, &props));
        free->push_back(props.vram_free);
    }
    return RAD_OK;
}

int SharedService::plan_image_memory() {
    const auto& additional = text_.cfg_.image_model_reserve;

    /* Measure each candidate alone, with the same bounds and adapter pools the
     * live service will use. Leave the configured first model loaded for serving. */
    std::vector<int64_t> baseline, free;
    RAD_TRY(read_free_vram(&baseline));
    std::vector<int64_t> minimum = baseline;
    std::vector<int64_t> largest((size_t)image_template_.tp, 0);
    auto measure = [&](const Config& cfg, const GeneratorKind* expected_kind = nullptr) -> int {
        RAD_INFO("shared image memory: measuring '%s' for startup budget", cfg.model.c_str());
        RAD_TRY(prepare_image(cfg, {}, expected_kind));
        RAD_TRY(read_free_vram(&free));
        for (size_t i = 0; i < largest.size(); ++i) {
            const size_t rank = image_text_rank(i);
            minimum[rank] = std::min(minimum[rank], free[rank] - image_growth_bytes((int)i));
            largest[i] = std::max(largest[i], image_->shared_image_footprint_[i]);
        }
        return RAD_OK;
    };
    std::set<std::string> measured;
    for (const auto& path : additional) {
        if (path == image_template_.model || !measured.insert(path).second) continue;
        Config candidate = image_template_;
        candidate.model = path;
        RAD_TRY(measure(candidate));
        const std::string id = std::filesystem::path(path).stem().string();
        const auto [it, inserted] = image_models_.emplace(
            id, ImageRegistration{path, image_->file_->identity(), next_image_revision_++,
                                  false, generator_kind(image_->program())});
        if (id.empty() || (!inserted && it->second.path != path)) {
            RAD_ERR("shared image model '%s': reserved checkpoints need distinct filename stems",
                    path.c_str());
            return RAD_E_INVAL;
        }
        image_.reset();
    }
    constexpr auto initial_kind = GeneratorKind::Image;
    RAD_TRY(measure(image_template_, &initial_kind));
    image_extra_reserve_.assign(free.size(), 0);
    const int64_t reload_headroom = text_.cfg_.image_reload_headroom_mib * kMiB;
    if (reload_headroom)
        RAD_INFO("shared image memory: reserving %s per generator rank for reloads",
                 humanb(reload_headroom).c_str());
    for (size_t i = 0; i < largest.size(); ++i) {
        const size_t rank = image_text_rank(i);
        const int64_t active = free[rank] - image_growth_bytes((int)i);
        const int64_t planned_extra = largest[i] - image_->shared_image_footprint_[i];
        const int64_t measured_extra = std::max(active - minimum[rank], planned_extra);
        if (reload_headroom > std::max<int64_t>(active - measured_extra, 0)) {
            RAD_ERR("shared image memory: text rank %zu cannot reserve %s of reload headroom "
                    "after budgeting the generators", rank, humanb(reload_headroom).c_str());
            return RAD_E_NOMEM;
        }
        image_extra_reserve_[rank] = measured_extra + reload_headroom;
        RAD_INFO("shared image memory: text rank %zu reserves %s beyond the initial image profile",
                 rank, humanb(image_extra_reserve_[rank]).c_str());
    }
    image_budget_bytes_ = std::move(largest);
    return RAD_OK;
}

int SharedService::check_image_footprint(const std::vector<int64_t>& footprint) const {
    if (image_budget_bytes_.empty()) return RAD_OK;  /* measuring startup candidates */
    if (image_budget_bytes_.size() != footprint.size()) return RAD_E_STATE;
    for (size_t i = 0; i < footprint.size(); ++i) {
        if (footprint[i] <= image_budget_bytes_[i]) continue;
        RAD_ERR("shared image memory: rank %zu needs %s, above the %s startup reservation; "
                "include this checkpoint in --image-model-reserve at launch",
                i, humanb(footprint[i]).c_str(), humanb(image_budget_bytes_[i]).c_str());
        /* Evicting host snapshots cannot change the startup VRAM reservation. */
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

size_t SharedService::image_text_rank(size_t image_rank) const {
    const int selected = text_.cfg_.image_text_rank;
    return selected < 0 ? image_rank : (size_t)selected;
}

int64_t SharedService::image_reserve_bytes(int rank) const {
    const int selected = text_.cfg_.image_text_rank;
    if (selected >= 0 && rank != selected) return 0;
    return image_growth_bytes(selected < 0 ? rank : 0) + image_extra_reserve_[(size_t)rank];
}

int SharedService::prepare_image(Config cfg, std::shared_ptr<RadFile> opened,
                                 const GeneratorKind* expected_kind, bool* cached_placement) {
    if (cached_placement) *cached_placement = false;
    auto image = std::make_unique<Engine>(std::move(cfg));
    image->shared_ = this;
    image->service_role_ = Engine::ServiceRole::Generator;
    image->reuse_registry_ = true;
    image->startup_begin_ns_ = rad_mono_ns();
    image->occupied_arch_plugins_.insert(text_.plugins_.arch_plugin);
    if (text_.supplemental_text_)
        image->occupied_arch_plugins_.insert(text_.supplemental_text_->plugins_.arch_plugin);
    image->file_ = std::move(opened);
    RAD_TRY(image->declare_model());
    const uint64_t declared = rad_mono_ns();
    if (image->ranks_.size() != (size_t)image_template_.tp) return RAD_E_STATE;
    for (size_t i = 0; i < image->ranks_.size(); ++i) {
        const size_t rank = image_text_rank(i);
        if (rank >= text_.ranks_.size() || image->ranks_[i]->device != text_.ranks_[rank]->device) {
            RAD_ERR("shared image rank %zu does not bind to its planned text rank %zu", i, rank);
            return RAD_E_STATE;
        }
    }
    if (!image->file_) {
        RAD_ERR("shared image checkpoint '%s' must be a .rad container", image->cfg_.model.c_str());
        return RAD_E_FORMAT;
    }
    const Program& program = image->program();
    if ((!program.is_generator() && !program.is_audio_generator() && !program.is_video_generator()) ||
        (expected_kind && generator_kind(program) != *expected_kind)) {
        if (expected_kind && *expected_kind == GeneratorKind::Image)
            RAD_ERR("shared image checkpoint must declare an image generator");
        else
            RAD_ERR("shared checkpoint must declare the requested image, audio or video generator");
        return RAD_E_UNSUPPORTED;
    }
    if (program.is_video_generator())
        image->cfg_.tower_placement = text_.cfg_.video_tower_placement;
    /* Changing embedding tiers moves weights into or out of the cached device
     * pool. Reuse a matching snapshot's layout before asking auto placement
     * again. If its host allocations cannot fit, activation evicts the optional
     * snapshots and retries with the original configuration. */
    if (!image->cfg_.embedding_placement_given) {
        const auto cached = std::find_if(image_cache_.begin(), image_cache_.end(),
            [&](const auto& entry) { return entry->identity == image->file_->identity(); });
        if (cached != image_cache_.end()) {
            image->cfg_.embedding_placement = (*cached)->embedding_placement;
            image->cfg_.embedding_placement_given = true;
            if (cached_placement) *cached_placement = true;
            RAD_INFO("shared image cache: retaining %s embedding placement for '%s'",
                     table_placement_name((*cached)->embedding_placement), image->file_->path());
        }
    }
    const uint64_t load_begin = rad_mono_ns();
    RAD_TRY(image->load_model());
    const uint64_t loaded = rad_mono_ns();
    if (!image->file_->matches_open_file(image->file_->fd())) {
        RAD_ERR("shared image checkpoint '%s' changed during loading", image->cfg_.model.c_str());
        return RAD_E_FORMAT;
    }
    RAD_TRY(image->prepare_runtime());
    const uint64_t ready = rad_mono_ns();
    RAD_INFO("shared image preparation for '%s': declare %.1f ms, load model %.1f ms, "
             "prepare runtime %.1f ms", image->cfg_.model.c_str(),
             double(declared - image->startup_begin_ns_) / 1e6,
             double(loaded - load_begin) / 1e6, double(ready - loaded) / 1e6);
    image_ = std::move(image);
    return RAD_OK;
}

int SharedService::image_adapter_request_count(const std::string& name) const {
    const std::lock_guard lock(image_adapter_requests_->mutex);
    const auto it = image_adapter_requests_->counts.find(name);
    return it == image_adapter_requests_->counts.end() ? 0 : it->second;
}

int SharedService::describe_generator(GeneratorEndpoint* result) {
    return image_->with_server_dependencies([&](server::Deps& d, server::ServerOptions& o) {
        result->model_id = std::move(o.model_id);
        result->audio = std::move(d.audio);
        result->video = std::move(d.videos);
        result->image = std::move(d.images);
        if (!*result) return RAD_E_STATE;
        if (!result->image) return RAD_OK;
        auto* endpoint = &result->image;
        endpoint->model_id = result->model_id;
        endpoint->lora_acquire = std::move(o.gen_lora_acquire);
        if (endpoint->lora_acquire) {
            auto* engine = image_.get();
            endpoint->validate_loras = [engine](const auto& adapters, auto* why) {
                return engine->gen_lora_validate(adapters, why);
            };
            endpoint->pinned_lora_acquire = [this, engine](const auto& adapters, auto* slot, auto* hold,
                                                    auto* note, auto* why) {
                const int status = engine->gen_lora_acquire_pinned(adapters, slot, hold, note, why);
                if (status >= 0) {
                    const std::lock_guard lock(image_lora_mutex_);
                    for (const auto& adapter : adapters) {
                        const auto it = image_lora_paths_.find(adapter.use.name);
                        if (it == image_lora_paths_.end() || !it->second.empty()) continue;
                        /* The request was admitted before unload. Its slot can drain
                         * with its hold, but must not remain cached after retirement. */
                        int running = 0;
                        std::string ignored;
                        engine->lora_unload(adapter.use.name, &ignored, &running, true);
                    }
                }
                return status;
            };
            endpoint->lora_load = [this, load = std::move(o.lora_load)](
                const std::string& name, const std::string& path, std::string* why) -> int {
                const std::lock_guard lock(image_lora_mutex_);
                if (path.find('\0') != std::string::npos) {
                    *why = "expected a local adapter path";
                    return RAD_E_INVAL;
                }
                std::error_code ec;
                const auto file = std::filesystem::canonical(path, ec);
                if (ec || !std::filesystem::is_regular_file(file, ec)) {
                    *why = "expected a local adapter file";
                    return RAD_E_NOTFOUND;
                }
                const auto previous = image_lora_paths_.find(name);
                if (previous != image_lora_paths_.end() && !previous->second.empty()) {
                    *why = "an adapter named '" + name + "' is already loaded";
                    return RAD_E_INVAL;
                }
                if (previous != image_lora_paths_.end() && image_adapter_request_count(name)) {
                    *why = "the previous requests using '" + name + "' are still draining";
                    return RAD_E_STATE;
                }
                const int status = load(name, file.string(), why);
                if (status >= 0) image_lora_paths_[name] = file.string();
                return status;
            };
            endpoint->lora_unload = [this, engine](
                const std::string& name, std::string* why, int* in_flight) -> int {
                const std::lock_guard lock(image_lora_mutex_);
                const auto it = image_lora_paths_.find(name);
                if (it != image_lora_paths_.end() && it->second.empty()) {
                    *in_flight = image_adapter_request_count(name);
                    if (*in_flight) return RAD_OK;
                    *why = "no adapter named '" + name + "' is loaded";
                    return RAD_E_NOTFOUND;
                }
                const bool known = it != image_lora_paths_.end() ||
                                   image_adapter_request_count(name) > 0;
                const int status = engine->lora_unload(name, why, in_flight, known);
                if (status >= 0) {
                    image_lora_paths_[name].clear();
                    *in_flight = image_adapter_request_count(name);
                }
                return status;
            };
        }
        return RAD_OK;
    });
}

void SharedService::publish_generator(GeneratorEndpoint endpoint, bool register_revision) {
    const std::lock_guard lock(image_admission_);
    const std::lock_guard queue(image_queue_mutex_);
    const auto it = image_models_.find(endpoint.model_id);
    const auto& identity = image_->file_->identity();
    if (register_revision && (it == image_models_.end() || it->second.identity != identity)) {
        image_models_.insert_or_assign(endpoint.model_id,
            ImageRegistration{image_->cfg_.model, identity, next_image_revision_++,
                              false, endpoint.kind()});
    } else if (it != image_models_.end() && it->second.identity == identity) {
        it->second.activation_failed = false;
        if (register_revision) it->second.path = image_->cfg_.model;
    }
    published_image_model_ = endpoint.model_id;
    activating_image_model_.clear();
    image_endpoint_ = std::move(endpoint);
}

SharedService::GeneratorEndpoint SharedService::acquire_generator() {
    const std::lock_guard lock(image_admission_);
    return image_endpoint_draining_ ? GeneratorEndpoint{} : image_endpoint_;
}

server::Deps::ImageEndpoint SharedService::acquire_image() {
    return acquire_generator().image;
}

SharedService::GeneratorKind SharedService::generator_kind(const Program& program) {
    if (program.is_audio_generator()) return GeneratorKind::Audio;
    if (program.is_video_generator()) return GeneratorKind::Video;
    return GeneratorKind::Image;
}

SharedService::GeneratorKind SharedService::generator_kind(const std::string& model) {
    const std::lock_guard lock(image_queue_mutex_);
    const auto it = image_models_.find(model);
    return it == image_models_.end() ? GeneratorKind::Image : it->second.kind;
}

int SharedService::verify_generator(const GeneratorEndpoint& endpoint) {
    server::Deps deps;
    deps.images = endpoint.image;
    deps.audio = endpoint.audio;
    deps.videos = endpoint.video;
    server::ServerOptions opt;
    opt.gen_lora_acquire = endpoint.image.lora_acquire;
    server::Server srv(deps, opt);
    Json probe;
    server::HttpRequest request;
    request.method = "POST";
    if (endpoint.audio) {
        const auto& info = endpoint.audio.gen->info();
        probe = {{"input", "Hello."}, {"seed", 7},
                 {"max_new_tokens", std::min<int64_t>(info.max_frames, 8)}};
        if (!info.tmpl.empty()) probe["instructions"] = "A quiet piano melody.";
        if (info.default_steps > 0) probe["num_inference_steps"] = 1;
        request.path = "/v1/audio/speech";
        RAD_INFO("shared audio startup check: generating with '%s'", endpoint.model_id.c_str());
    } else if (endpoint.video) {
        const auto& info = endpoint.video.gen->info();
        const double seconds = double(info.min_chunks * info.frames_per_chunk + info.frames_tail) *
                               double(info.fps_den) / double(info.fps_num);
        const int64_t side = ((info.min_side + info.size_multiple - 1) / info.size_multiple) *
                             info.size_multiple;
        const bool square_fits = side <= info.max_area / side;
        const int64_t width = square_fits ? side : info.default_width;
        const int64_t height = square_fits ? side : info.default_height;
        probe = {{"prompt", "A red square"}, {"seed", 7}, {"seconds", seconds},
                 {"size", std::to_string(width) + "x" + std::to_string(height)}};
        if (!info.fixed_steps) probe["num_inference_steps"] = info.step_evals ? 1 : 2;
        request.path = "/v1/videos";
        RAD_INFO("shared video startup check: generating with '%s'", endpoint.model_id.c_str());
    } else {
        const auto& info = endpoint.image.gen->info();
        probe = {{"model", endpoint.model_id}, {"prompt", "A red square"},
                 {"seed", 7}, {"size", "auto"}, {"output_format", "png"}};
        if (!info.fixed_schedule) probe["num_inference_steps"] = 1;
        request.path = "/v1/images/generations";
        RAD_INFO("shared image startup check: generating with '%s' at %lldx%lld",
                 endpoint.model_id.c_str(), (long long)info.default_width,
                 (long long)info.default_height);
    }
    request.body = probe.dump();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(3);
    request.closed = [deadline] { return std::chrono::steady_clock::now() >= deadline; };
    const auto result = srv.router().dispatch(std::move(request));
    bool completed = result.status == 200;
    std::string failure = result.body;
    if (completed && endpoint.video) {
        failure = "video job did not complete before the startup deadline";
        const auto created = Json::parse(result.body, nullptr, false);
        completed = false;
        if (created.is_object() && created.contains("id") && created["id"].is_string()) {
            server::HttpRequest poll;
            poll.path = "/v1/videos/" + created["id"].get<std::string>();
            while (std::chrono::steady_clock::now() < deadline) {
                const auto progress = srv.router().dispatch(poll);
                if (progress.status != 200) {
                    failure = progress.body;
                    break;
                }
                const auto body = Json::parse(progress.body, nullptr, false);
                if (!body.is_object()) break;
                const auto status = body.value("status", std::string{});
                if (status == "completed") { completed = true; break; }
                if (status == "failed") {
                    failure = progress.body;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
            }
        }
    }
    const char* kind = endpoint.audio ? "audio" : endpoint.video ? "video" : "image";
    if (!completed || !wait_until_parked(deadline)) {
        RAD_ERR("shared %s startup check failed (HTTP %d): %s", kind,
                result.status, completed ? "generator memory did not park" : failure.c_str());
        return RAD_E_STATE;
    }
    RAD_INFO("shared %s startup check passed for '%s'", kind, endpoint.model_id.c_str());
    return RAD_OK;
}

int SharedService::detach_image_memory(bool retain) {
    RAD_TRY(drain(text_));
    for (const auto& rank : text_.ranks_) {
        RAD_TRY(rad_dev_set(rank->device));
        RAD_TRY(rank->mover.reclaim(Mover::kLendImage, 0, &rank->heat));
    }
    for (const auto& rank : text_.ranks_) RAD_TRY(rank->mover.detach_image_loan());
    if (retain) cache_image_weights();
    memory_.clear();
    restored_memory_.clear();
    if (!unloaded_expert_memory_.empty()) {
        unloaded_expert_memory_.clear();
        return rad_dev_set(text_.ranks_.front()->device);
    }
    return RAD_OK;
}

int SharedService::offer_unloaded_expert_memory() {
    /* Caller holds dispatch after the generator and its previous loan retire. */
    const auto& cfg = text_.cfg_;
    const int64_t limit = cfg.unloaded_image_expert_pool_mib * kMiB;
    if (!limit) return RAD_OK;
    if (image_ || !unloaded_expert_memory_.empty()) return RAD_E_STATE;
    const int64_t default_headroom = Config{}.gpu_headroom_mib * kMiB;
    const int64_t configured_headroom = std::min(cfg.gpu_headroom_mib, INT64_MAX / kMiB) * kMiB;
    std::vector<std::unique_ptr<UnloadedExpertMemory>> pending(image_budget_bytes_.size());
    const auto decline = [&] {
        pending.clear();
        return rad_dev_set(text_.ranks_.front()->device);
    };
    std::vector<int64_t> requested(pending.size(), 0), available(pending.size(), 0);
    for (size_t i = 0; i < pending.size(); ++i) {
        const auto& rank = *text_.ranks_[image_text_rank(i)];
        /* Integrated devices share the host RAM reserve used by startup planning. */
        if (dev_integrated(rank.device)) continue;
        requested[i] = std::min({limit, rank.mover.lendable_host_bytes(), image_budget_bytes_[i]});
    }
    /* Free VRAM can become visible on one card before another after image teardown.
     * Sample every eligible rank before allocating, then give the driver a short
     * shared window to publish the release. Use the latest safe capacity if a
     * card remains constrained. */
    using Clock = std::chrono::steady_clock;
    using namespace std::chrono_literals;
    const auto began = Clock::now();
    const auto deadline = began + 250ms;
    int samples = 0;
    for (;;) {
        ++samples;
        bool short_on_memory = false;
        for (size_t i = 0; i < pending.size(); ++i) {
            if (requested[i] <= RAD_ALIGN_UNIT) continue;
            const auto& rank = *text_.ranks_[image_text_rank(i)];
            RadDeviceProps props{};
            if (rad_dev_set(rank.device) < 0 || rad_dev_props(rank.device, &props) < 0 ||
                props.vram_free < 0 || props.vram_free > props.vram_bytes) {
                RAD_WARN("unloaded image expert pool: could not inspect text rank %d", rank.index);
                return decline();
            }
            /* An explicit cap replaces configured headroom. Compute that margin from this
             * card's capacity rather than charging the common startup margin a second time. */
            const int64_t cap_headroom = cfg.gpu_memory_mib > 0
                ? props.vram_bytes - std::min(cfg.gpu_memory_mib, props.vram_bytes / kMiB) * kMiB
                : configured_headroom;
            int64_t free = props.vram_free;
            free -= std::min(free, std::max(cap_headroom, default_headroom));
            free -= std::min(free, std::max<int64_t>(0, text_.late_lazy_bytes_));
            const auto& kv = rank.pools.kv();
            free -= std::min(free, std::max<int64_t>(0, kv.bytes() - kv.committed()));
            available[i] = std::min(requested[i], free);
            short_on_memory |= available[i] < requested[i];
        }
        const auto now = Clock::now();
        if (!short_on_memory || now >= deadline) break;
        std::this_thread::sleep_until(std::min(now + 25ms, deadline));
    }
    const double settle_ms = std::chrono::duration<double, std::milli>(Clock::now() - began).count();
    for (size_t i = 0; i < pending.size(); ++i) {
        const auto& rank = *text_.ranks_[image_text_rank(i)];
        const int64_t bytes = available[i];
        if (samples > 1 && requested[i] > RAD_ALIGN_UNIT)
            RAD_INFO("unloaded image expert pool: text rank %d settled in %.1f ms (%d samples);"
                     " requested %s, available %s", rank.index, settle_ms, samples,
                     humanb(requested[i]).c_str(), humanb(bytes).c_str());
        if (bytes <= RAD_ALIGN_UNIT) continue;
        if (rad_dev_set(rank.device) < 0) return decline();
        auto allocation = std::make_unique<UnloadedExpertMemory>(rank.device, bytes);
        allocation->base = static_cast<char*>(rad_dev_alloc(bytes, RAD_MEM_DEVICE));
        if (!allocation->base) {
            RAD_WARN("unloaded image expert pool: rank %d could not allocate %s",
                     rank.index, humanb(bytes).c_str());
            return decline();
        }
        pending[i] = std::move(allocation);
    }

    /* Keep ownership if a later rollback cannot detach a slot safely. */
    unloaded_expert_memory_ = std::move(pending);
    for (size_t i = 0; i < unloaded_expert_memory_.size(); ++i) {
        auto& mem = unloaded_expert_memory_[i];
        if (!mem) continue;
        auto& rank = *text_.ranks_[image_text_rank(i)];
        int st = rad_dev_set(rank.device);
        if (st >= 0)
            st = rank.mover.lend_from(Mover::kLendImage, {{mem->base + mem->bytes, mem->bytes, 1, 1}});
        if (st < 0) {
            RAD_WARN("unloaded image expert pool: rank %d could not attach slots", rank.index);
            return detach_image_memory();
        }
        if (!rank.mover.loan_ceiling(Mover::kLendImage)) mem.reset();
    }
    refill_at_ = 0;
    for (size_t i = 0; i < unloaded_expert_memory_.size(); ++i) {
        const auto& mem = unloaded_expert_memory_[i];
        if (!mem) continue;
        auto& rank = *text_.ranks_[image_text_rank(i)];
        RAD_TRY(rank.mover.set_loan(Mover::kLendImage, rank.mover.loan_ceiling(Mover::kLendImage)));
        RAD_INFO("unloaded image expert pool: text rank %d allocated %s",
                 rank.index, humanb(mem->bytes).c_str());
    }
    if (std::none_of(unloaded_expert_memory_.begin(), unloaded_expert_memory_.end(),
                     [](const auto& mem) { return mem != nullptr; }))
        unloaded_expert_memory_.clear();
    return rad_dev_set(text_.ranks_.front()->device);
}

int64_t SharedService::remaining_image_lora_budget() const {
    if (!image_template_.lora.enabled()) return 0;
    const auto budget = static_cast<uint64_t>(text_.cfg_.image_lora_snapshot_mib) << 20;
    return static_cast<int64_t>(budget - std::min(budget, image_lora_snapshots_.used_bytes()));
}

void SharedService::trim_image_cache(int64_t active_bytes, int64_t incoming_bytes) {
    const int64_t budget = text_.cfg_.image_model_cache_mib * kMiB;
    const int64_t available = budget - std::min(active_bytes, budget);
    while (!image_cache_.empty()) {
        std::string why;
        if (image_cache_bytes_ > available) {
            why = "inactive snapshot budget " + humanb(available);
        /* The inactive cache may yield to allocations for serving. Include the
         * unspent adapter budget when a new weight or staging allocation needs
         * to fit, so optional snapshots do not consume space intended for requests. */
        } else if (!incoming_bytes || lockable_room(incoming_bytes, remaining_image_lora_budget(),
                                                   text_.host_headroom_bytes(), false, &why)) {
            break;
        } else {
            why = "host reservation " + humanb(incoming_bytes) + ": " + why;
        }
        evict_oldest_image_snapshot(why);
    }
}

void SharedService::evict_oldest_image_snapshot(const std::string& why) {
    const auto& oldest = *image_cache_.front();
    RAD_INFO("shared image cache: evict '%s' (%s): %s", oldest.path.c_str(),
             humanb(oldest.bytes).c_str(), why.c_str());
    image_cache_bytes_ -= oldest.bytes;
    image_cache_.pop_front();
}

bool SharedService::make_room_for_image_adapter(int64_t bytes) {
    const std::unique_lock update(image_update_, std::try_to_lock);
    if (!update.owns_lock() || failed()) return false;
    std::string why;
    const int64_t margin = text_.host_headroom_bytes();
    if (lockable_room(bytes, 0, margin, false, &why)) return true;
    if (image_cache_.empty()) return false;

    /* Unmapping pinned snapshots can affect the device even when they are inactive. */
    const auto dispatch = lock_image_dispatch();
    if (failed()) return false;
    if (lockable_room(bytes, 0, margin, false, &why)) return true;
    if (drain(text_) < 0 || (image_ && drain(*image_) < 0)) return false;
    while (!lockable_room(bytes, 0, margin, false, &why)) {
        if (image_cache_.empty()) return false;
        evict_oldest_image_snapshot("queued adapter host memory: " + why);
    }
    return true;
}

void SharedService::cache_image_weights() {
    const int64_t budget = text_.cfg_.image_model_cache_mib * kMiB;
    if (!budget || !image_ || !image_->file_ || memory_.empty()) return;
    int64_t bytes = 0;
    for (size_t i = 0; i < memory_.size(); ++i) {
        auto& mem = *memory_[i];
        const auto& rank = *image_->ranks_[i];
        if (!mem.backing || mem.weight_bytes != rank.pools.weights().used() ||
            mem.weight_bytes > budget - bytes) return;
        mem.layout.clear();
        if (!describe_weight_pool(rank.program, rank.pools.weights(), &mem.layout)) return;
        bytes += mem.weight_bytes;
    }

    const auto& identity = image_->file_->identity();
    for (auto it = image_cache_.begin(); it != image_cache_.end(); ++it) {
        if ((*it)->identity != identity) continue;
        image_cache_bytes_ -= (*it)->bytes;
        image_cache_.erase(it);
        break;
    }
    /* The active checkpoint already owned these pins. Defer eviction until the next
     * checkpoint has claimed its snapshot, so A -> B -> A can promote A even if it
     * is the least recently used. */
    auto entry = std::make_unique<CachedImage>();
    entry->identity = identity;
    entry->path = image_->file_->path();
    entry->bytes = bytes;
    entry->embedding_placement = image_->cfg_.embedding_resolved;
    for (auto& mem : memory_) {
        /* Parked device weights may already have been overwritten by text experts. */
        mem->weights = nullptr;
        mem->offered_bytes = 0;
    }
    entry->ranks = std::move(memory_);
    image_cache_.push_back(std::move(entry));
    image_cache_bytes_ += bytes;
    RAD_INFO("shared image cache: retained '%s' (%s total)", image_->file_->path(),
             humanb(image_cache_bytes_).c_str());
}

int SharedService::restore_image_weights(Engine& image, bool* restored) {
    *restored = false;
    if (!image.file_ || image_cache_.empty()) return RAD_OK;
    for (auto it = image_cache_.begin(); it != image_cache_.end();) {
        const bool same_file = (*it)->identity == image.file_->identity();
        if (!same_file && (*it)->path != image.file_->path()) {
            ++it;
            continue;
        }
        auto entry = std::move(*it);
        image_cache_.erase(it);
        image_cache_bytes_ -= entry->bytes;
        if (!same_file || entry->ranks.size() != image.ranks_.size()) {
            RAD_INFO("shared image cache: stale checkpoint or rank count for '%s'",
                     image.file_->path());
            break;
        }
        for (size_t i = 0; i < image.ranks_.size(); ++i) {
            const auto& rank = *image.ranks_[i];
            const auto& mem = *entry->ranks[i];
            std::vector<CachedWeight> layout;
            if (mem.device != rank.device || mem.weight_bytes != rank.pools.weights().used() ||
                !describe_weight_pool(rank.program, rank.pools.weights(), &layout) ||
                layout != mem.layout) {
                RAD_INFO("shared image cache: rank %zu storage changed for '%s'",
                         i, image.file_->path());
                return RAD_OK;
            }
        }

        const uint64_t begin = rad_mono_ns();
        for (size_t i = 0; i < image.ranks_.size(); ++i) {
            auto& rank = *image.ranks_[i];
            auto& mem = *entry->ranks[i];
            mem.weights = static_cast<char*>(rank.pools.weights().base());
            RAD_TRY(rad_dev_set(rank.device));
            RAD_TRY(mem.copy(rank.mover.transfer_stream(), true));
        }
        restored_memory_ = std::move(entry->ranks);
        *restored = true;
        RAD_INFO("shared image cache: restored '%s' (%s) in %.1f ms",
                 image.file_->path(), humanb(entry->bytes).c_str(),
                 double(rad_mono_ns() - begin) / 1e6);
        break;
    }
    return RAD_OK;
}

std::vector<std::string> SharedService::image_model_ids() {
    const std::lock_guard lock(image_queue_mutex_);
    std::vector<std::string> ids;
    for (const auto& entry : image_models_) ids.push_back(entry.first);
    return ids;
}

std::string SharedService::image_status() {
    using json = Json;
    /* The cache is owned by the model update. A status poll never waits for GPU bring-up. */
    const std::unique_lock update(image_update_, std::try_to_lock);
    const std::lock_guard queue(image_queue_mutex_);
    const auto& active = published_image_model_;
    std::map<std::string, size_t> queued;
    for (const auto& waiter : image_waiters_) ++queued[waiter.model];
    json models = json::array();
    for (const auto& [name, registration] : image_models_) {
        const bool cached = update.owns_lock() &&
            std::any_of(image_cache_.begin(), image_cache_.end(), [&](const auto& entry) {
                return entry->identity == registration.identity;
            });
        const char* state = name == draining_image_model_ ? "draining" :
                            name == active ? "active" :
                            name == activating_image_model_ ? "activating" :
                            registration.activation_failed ? "failed" :
                            cached ? "host_cached" : "registered";
        models.push_back({{"id", name}, {"revision", std::to_string(registration.revision)},
                          {"state", state}, {"queued", queued[name]}});
    }
    /* These are independent samples. Source snapshots can retire before the
     * adapter-name leases held through generation. */
    size_t adapter_holds = 0;
    {
        const std::lock_guard adapters(image_adapter_requests_->mutex);
        for (const auto& entry : image_adapter_requests_->counts)
            adapter_holds += static_cast<size_t>(entry.second);
    }
    return json{{"active_model", active.empty() ? json(nullptr) : json(active)},
                {"activating_model", activating_image_model_.empty()
                                     ? json(nullptr) : json(activating_image_model_)},
                {"queued", image_waiters_.size()}, {"cache_status_available", update.owns_lock()},
                {"inactive_cache_bytes", update.owns_lock() ? json(image_cache_bytes_) : json(nullptr)},
                {"adapter_request_holds", adapter_holds},
                {"adapter_snapshot_bytes", image_lora_snapshots_.used_bytes()},
                {"models", std::move(models)}}.dump();
}

int SharedService::register_image_models(const std::string& text_id, const std::string& initial_id) {
    if (image_models_.contains(initial_id) || image_models_.contains(text_id)) {
        RAD_ERR("reserved image model IDs must differ from the initial image and text model IDs");
        return RAD_E_INVAL;
    }
    image_models_.emplace(initial_id,
                          ImageRegistration{image_->cfg_.model, image_->file_->identity(), next_image_revision_++});
    selected_image_model_ = default_image_model_ = initial_id;
    return RAD_OK;
}

SharedService::ImageTurn::~ImageTurn() {
    if (!service) return;
    const std::lock_guard guard(service->image_queue_mutex_);
    service->image_turn_in_use_ = false;
    if (operator_load) service->image_operator_load_ = false;
    service->image_queue_turn_.notify_all();
}

int SharedService::queue_generator(const std::string& model, GeneratorKind kind, size_t bytes, size_t max_waiters,
                               size_t max_bytes, const std::function<bool()>& cancelled,
                               GeneratorEndpoint* endpoint, std::shared_ptr<void>* hold,
                               server::ImageDispatchTiming* timing, std::string* why) {
    using namespace std::chrono;
    constexpr auto poll = milliseconds(25);
    auto turn = std::make_shared<ImageTurn>();

    std::unique_lock lock(image_queue_mutex_);
    const std::string name = model.empty() ? default_image_model_ : model;
    if (!draining_image_model_.empty()) {
        *why = "an operator is draining the image queue";
        return RAD_E_FULL;
    }
    const auto registered = image_models_.find(name);
    if (registered == image_models_.end() || registered->second.kind != kind) {
        *why = std::string("no registered ") + (kind == GeneratorKind::Video ? "video" : kind == GeneratorKind::Audio ? "audio" : "image") + " model named " + name;
        return RAD_E_NOTFOUND;
    }
    const ImageRegistration registration = registered->second;
    if (image_waiters_.size() >= max_waiters || bytes > max_bytes ||
        image_waiting_bytes_ > max_bytes - bytes) {
        *why = "the image model queue is full";
        return RAD_E_FULL;
    }
    const auto waiter = image_waiters_.insert(image_waiters_.end(), {name, steady_clock::now()});
    image_waiting_bytes_ += bytes;
    if (image_turn_in_use_ || image_waiters_.size() > 1)
        RAD_INFO("shared image queue: waiting for '%s'", name.c_str());
    auto remove = [&] {
        image_waiting_bytes_ -= bytes;
        image_waiters_.erase(waiter);
        image_queue_turn_.notify_all();
    };
    for (;;) {
        if (cancelled() || failed()) {
            remove();
            return failed() ? RAD_E_STATE : RAD_E_CANCELLED;
        }
        if (!image_turn_in_use_) {
            const auto next = next_image_request(image_waiters_.begin(), image_waiters_.end(),
                                                 selected_image_model_, text_.cfg_.image_model_fifo,
                                                 steady_clock::now());
            if (next == waiter) {
                timing->queued_ns = duration_cast<nanoseconds>(
                    steady_clock::now() - waiter->enqueued).count();
                for (auto it = image_waiters_.begin(); it != waiter; ++it) ++it->bypasses;
                turn->service = this;
                *hold = std::move(turn);
                image_turn_in_use_ = true;
                remove();
                break;
            }
        }
        image_queue_turn_.wait_for(lock, poll);
    }
    lock.unlock();

    *endpoint = acquire_generator();
    if (*endpoint && endpoint->model_id == name && image_->file_->identity() == registration.identity)
        return cancelled() ? RAD_E_CANCELLED : RAD_OK;
    *endpoint = {};
    for (;;) {
        if (cancelled()) return RAD_E_CANCELLED;
        bool busy = false;
        const int st = load_image(registration.path, name, why, &registration.identity, &busy,
                                  &timing->selection, &kind);
        if (busy) {
            /* The previous driver job can release its hold just before it parks. */
            std::this_thread::sleep_for(poll);
            continue;
        }
        if (st < 0) {
            const auto current = acquire_generator();
            const std::lock_guard guard(image_queue_mutex_);
            if (current) selected_image_model_ = current.model_id;
            if (auto it = image_models_.find(name); it != image_models_.end() &&
                it->second.revision == registration.revision)
                it->second.activation_failed = true;
            return st;
        }
        break;
    }
    *endpoint = acquire_generator();
    {
        const std::lock_guard guard(image_queue_mutex_);
        selected_image_model_ = endpoint->model_id;
        if (kind == GeneratorKind::Image) default_image_model_ = endpoint->model_id;
    }
    return cancelled() ? RAD_E_CANCELLED : *endpoint ? RAD_OK : RAD_E_STATE;
}

int SharedService::load_image_admin(const std::string& path, const std::string& name,
                                    std::string* why) {
    ImageTurn turn;
    std::unique_lock lock(image_queue_mutex_);
    if (image_turn_in_use_ || !image_waiters_.empty() || !draining_image_model_.empty()) {
        *why = "image requests already own the model queue";
        return RAD_E_FULL;
    }
    image_turn_in_use_ = true;
    turn.service = this;
    turn.operator_load = image_operator_load_ = true;
    GeneratorKind existing_kind = GeneratorKind::Image;
    const auto existing = image_models_.find(name);
    const GeneratorKind* expected_kind = nullptr;
    if (existing != image_models_.end()) {
        existing_kind = existing->second.kind;
        expected_kind = &existing_kind;
    }
    lock.unlock();
    const int st = load_image(path, name, why, nullptr, nullptr, nullptr, expected_kind);
    lock.lock();
    if (st >= 0) {
        selected_image_model_ = name;
        if (image_models_.at(name).kind == GeneratorKind::Image) default_image_model_ = name;
    }
    return st;
}

int SharedService::unload_image_admin(const std::string& name,
                                      const std::function<bool()>& cancelled, std::string* why) {
    using namespace std::chrono_literals;
    struct DrainStatus {
        SharedService& service;
        bool armed = false;
        ~DrainStatus() {
            if (!armed) return;
            const std::lock_guard admission(service.image_admission_);
            service.image_endpoint_draining_ = false;
            const std::lock_guard lock(service.image_queue_mutex_);
            service.draining_image_model_.clear();
            service.image_queue_turn_.notify_all();
        }
    } drain_status{*this};

    RadFile::Identity identity{};
    {
        std::unique_lock lock(image_queue_mutex_);
        const auto found = image_models_.find(name);
        if (found == image_models_.end()) {
            *why = "no registered image model named " + name;
            return RAD_E_NOTFOUND;
        }
        if (!draining_image_model_.empty() || image_operator_load_) {
            *why = "an operator image update is already in progress";
            return RAD_E_FULL;
        }
        identity = found->second.identity;
        draining_image_model_ = name;
        drain_status.armed = true;
        while (image_turn_in_use_ || !image_waiters_.empty()) {
            if (cancelled() || failed()) return RAD_E_STATE;
            image_queue_turn_.wait_for(lock, 25ms);
        }
    }

    const std::lock_guard update(image_update_);
    bool retired = false;
    for (;;) {
        if (cancelled() || failed()) return RAD_E_STATE;
        {
            const std::lock_guard admission(image_admission_);
            const auto dispatch = lock_image_dispatch();
            if (image_endpoint_.model_id != name) break;
            image_endpoint_draining_ = true;
            /* The turn retires when the driver erases its job. Let the step loop park
             * before detaching and let outstanding catalog/adapter leases finish. */
            if (!active_ && !image_->gen_->busy() && image_endpoint_.use_count() == 1) {
                const int st = detach_image_memory();
                if (st < 0) {
                    image_failure_.store(st, std::memory_order_release);
                    return st;
                }
                const std::lock_guard queue(image_queue_mutex_);
                image_endpoint_ = {};
                published_image_model_.clear();
                retired = true;
                break;
            }
        }
        std::this_thread::sleep_for(25ms);
    }

    /* The turn and update locks exclude activation. Stop without holding dispatch,
     * which the generator loop needs to exit. */
    if (retired) image_->stop_runtime();
    const auto dispatch = lock_image_dispatch();
    if (const int st = drain(text_); st < 0) {
        image_failure_.store(st, std::memory_order_release);
        return st;
    }
    if (retired) {
        image_.reset();
        const int status = offer_unloaded_expert_memory();
        if (status < 0) {
            image_failure_.store(status, std::memory_order_release);
            return status;
        }
    }
    for (auto it = image_cache_.begin(); it != image_cache_.end();) {
        if ((*it)->identity == identity) {
            image_cache_bytes_ -= (*it)->bytes;
            it = image_cache_.erase(it);
        } else {
            ++it;
        }
    }
    return RAD_OK;
}

int SharedService::open_image_registration(const std::string& path, ImageRegistration* registration,
                                           std::string* why, std::shared_ptr<RadFile>* opened) {
    if (path.find('\0') != std::string::npos || std::filesystem::path(path).extension() != ".rad") {
        *why = "expected a local .rad checkpoint path";
        return RAD_E_INVAL;
    }
    std::error_code ec;
    const auto file = std::filesystem::canonical(path, ec);
    if (ec || !std::filesystem::is_regular_file(file, ec)) {
        *why = "expected an existing local .rad checkpoint";
        return RAD_E_NOTFOUND;
    }
    auto checkpoint = std::make_shared<RadFile>();
    if (const int st = checkpoint->open(file.c_str()); st < 0) {
        *why = "could not read the requested .rad checkpoint";
        return st;
    }
    *registration = {file.string(), checkpoint->identity()};
    if (opened) *opened = std::move(checkpoint);
    return RAD_OK;
}

int SharedService::register_image_admin(const std::string& path, const std::string& name,
                                        std::string* why) {
    ImageRegistration registration;
    RAD_TRY(open_image_registration(path, &registration, why));
    const std::lock_guard lock(image_queue_mutex_);
    if (image_operator_load_ || !draining_image_model_.empty()) {
        *why = "an operator image load is already in progress";
        return RAD_E_FULL;
    }
    const auto existing = image_models_.find(name);
    if (existing != image_models_.end()) {
        if (existing->second.identity == registration.identity) return RAD_OK;
        *why = "this model name is already registered; use a new name or the operator load endpoint";
        return RAD_E_DUPLICATE;
    }
    registration.revision = next_image_revision_++;
    image_models_.emplace(name, std::move(registration));
    return RAD_OK;
}

int SharedService::load_image(const std::string& path, const std::string& name, std::string* why,
                              const RadFile::Identity* expected, bool* busy,
                              server::ImageSelection* selection, const GeneratorKind* expected_kind) {
    if (busy) *busy = false;
    ImageRegistration registration;
    std::shared_ptr<RadFile> opened;
    RAD_TRY(open_image_registration(path, &registration, why, &opened));
    const auto& requested_identity = registration.identity;
    if (expected && requested_identity != *expected) {
        *why = "the registered checkpoint changed; an operator must reload this model";
        return RAD_E_FORMAT;
    }
    const auto& file = registration.path;

    std::unique_lock update(image_update_, std::try_to_lock);
    if (!update.owns_lock()) {
        if (busy) *busy = true;
        *why = "another image model load is in progress";
        return RAD_E_FULL;
    }
    /* Scope the activating status to this transaction, including probe and rollback. */
    std::string previous_id;
    bool had_unloaded_expert_memory = false;
    uint64_t retire_begin = 0;
    struct ActivationStatus {
        SharedService& service;
        ~ActivationStatus() {
            const std::lock_guard lock(service.image_queue_mutex_);
            service.activating_image_model_.clear();
        }
    } activation_status{*this};

    {
        const std::lock_guard admission(image_admission_);
        if (image_endpoint_.use_count() > 1) {
            if (busy) *busy = true;
            *why = "an image or adapter request is still using the active model";
            return RAD_E_FULL;
        }
        const auto dispatch = lock_image_dispatch();
        if (failed()) {
            *why = "the shared service is shutting down";
            return RAD_E_STATE;
        }
        if (active_ || (image_ && image_->gen_->busy())) {
            if (busy) *busy = true;
            *why = "the active image model is still processing a request";
            return RAD_E_FULL;
        }
        if (image_ && image_->file_ && image_->file_->identity() == requested_identity &&
            image_->cfg_.model == file && image_endpoint_.model_id == name)
            return RAD_OK;
        retire_begin = rad_mono_ns();
        had_unloaded_expert_memory = !unloaded_expert_memory_.empty();
        if (const int st = detach_image_memory(true); st < 0) {
            image_failure_.store(st, std::memory_order_release);
            return st;
        }
        const std::lock_guard queue(image_queue_mutex_);
        previous_id = std::move(published_image_model_);
        published_image_model_.clear();
        image_endpoint_ = {};
        activating_image_model_ = name;
    }

    Config previous;
    std::shared_ptr<RadFile> previous_file;
    if (image_) {
        previous_file = image_->file_;
        previous = image_template_;
        previous.model = image_->cfg_.model;
        previous.served_model_name = image_->cfg_.served_model_name;
        image_->stop_runtime();
        const auto dispatch = lock_image_dispatch();
        image_.reset();
    }
    Config next = image_template_;
    next.model = file;
    next.served_model_name = name;
    bool cached_placement = false;
    auto activate = [&](Config cfg, const std::shared_ptr<RadFile>& checkpoint,
                        bool register_revision, const GeneratorKind* required_kind) -> int {
        const uint64_t prepare_begin = rad_mono_ns();
        GeneratorEndpoint endpoint;
        uint64_t dispatch_acquired;
        {
            const auto dispatch = lock_image_dispatch();
            dispatch_acquired = rad_mono_ns();
            RAD_TRY(prepare_image(std::move(cfg), checkpoint, required_kind, &cached_placement));
            RAD_TRY(describe_generator(&endpoint));
        }
        const bool restored = !restored_memory_.empty();
        const uint64_t snapshot_begin = rad_mono_ns();
        RAD_TRY(lend_image_memory());
        const uint64_t runtime_begin = rad_mono_ns();
        RAD_TRY(image_->start_runtime());
        const uint64_t probe_begin = rad_mono_ns();
        const int st = verify_generator(endpoint);
        if (st < 0) {
            image_->stop_runtime();
            return st;
        }
        const uint64_t ready = rad_mono_ns();
        RAD_INFO("shared image activation phases for '%s': prepare %.1f ms, snapshot and lend %.1f ms, "
                 "start runtime %.1f ms, readiness probe %.1f ms (dispatch wait %.1f ms)",
                 endpoint.model_id.c_str(), double(snapshot_begin - prepare_begin) / 1e6,
                 double(runtime_begin - snapshot_begin) / 1e6,
                 double(probe_begin - runtime_begin) / 1e6, double(ready - probe_begin) / 1e6,
                 double(dispatch_acquired - prepare_begin) / 1e6);
        publish_generator(std::move(endpoint), register_revision);
        if (selection)
            *selection = restored ? server::ImageSelection::Warm : server::ImageSelection::Cold;
        return RAD_OK;
    };
    auto discard = [&](bool evict_cache) -> int {
        const auto dispatch = lock_image_dispatch();
        if (failed() || active_) return RAD_E_STATE;
        RAD_TRY(detach_image_memory());
        image_.reset();
        if (evict_cache) trim_image_cache(std::numeric_limits<int64_t>::max(), 0);
        return RAD_OK;
    };

    const uint64_t begin = rad_mono_ns();
    if (previous_file)
        RAD_INFO("shared image model: retired '%s' in %.1f ms", previous_id.c_str(),
                 double(begin - retire_begin) / 1e6);
    auto activate_with_retry = [&](const Config& cfg, const std::shared_ptr<RadFile>& checkpoint,
                                   bool register_revision, const GeneratorKind* required_kind) -> int {
        const int st = activate(cfg, checkpoint, register_revision, required_kind);
        if (st != RAD_E_NOMEM || (!cached_placement && image_cache_.empty())) return st;
        if (discard(true) < 0) {
            image_failure_.store(RAD_E_STATE, std::memory_order_release);
            return RAD_E_STATE;
        }
        return activate(cfg, checkpoint, register_revision, required_kind);
    };
    const int st = activate_with_retry(next, opened, expected == nullptr, expected_kind);
    if (failed()) {
        *why = "image model load failed while releasing GPU memory";
        return RAD_E_STATE;
    }
    if (st >= 0) {
        RAD_INFO("shared image model: '%s' ready in %.1f ms", name.c_str(),
                 double(rad_mono_ns() - begin) / 1e6);
        return st;
    }
    if (discard(false) < 0) {
        image_failure_.store(RAD_E_STATE, std::memory_order_release);
        *why = "image model load failed during GPU execution";
        return RAD_E_STATE;
    }
    if (!previous.model.empty()) {
        {
            const std::lock_guard lock(image_queue_mutex_);
            activating_image_model_ = previous_id;
        }
        RAD_WARN("shared image model '%s' failed to load (%s); restoring the previous checkpoint",
                 name.c_str(), rad_strerror(st));
        if (activate_with_retry(previous, previous_file, false, nullptr) >= 0) {
            *why = std::string("image model load failed (") + rad_strerror(st) +
                   "); the previous model is ready";
            return st;
        }
        if (discard(false) < 0) image_failure_.store(RAD_E_STATE, std::memory_order_release);
    }
    if (had_unloaded_expert_memory && !failed()) {
        const auto dispatch = lock_image_dispatch();
        if (const int status = offer_unloaded_expert_memory(); status < 0)
            image_failure_.store(status, std::memory_order_release);
    }
    *why = "image model load failed and no image model is ready";
    return RAD_E_STATE;
}

int64_t SharedService::image_growth_bytes(int rank) const {
    const auto& image_rank = *image_->ranks_[(size_t)rank];
    const auto& kv = image_rank.pools.kv();
    const bool has_kv = image_rank.program.kv_groups.size() > 1;
    return (has_kv ? kv.bytes() - kv.committed() : 0) + image_->late_lazy_bytes_;
}

int SharedService::lend_image_memory() {
    /* ROCm host mapping changes may stall queues that are still using the device.
     * Keep text drained through snapshot allocation, copy, and cleanup. Declare
     * the lock before pending so rollback frees happen while dispatch is held. */
    const auto dispatch = lock_image_dispatch();
    if (failed()) return RAD_E_STATE;
    RAD_TRY(drain(text_));
    RAD_TRY(drain(*image_));
    ImageSnapshot pending;
    pending.reserve(image_->ranks_.size());
    int64_t backing_bytes = 0, incoming_bytes = 0;
    const bool lend_weights = text_.cfg_.image_weight_loan;
    if (!lend_weights) RAD_INFO("shared image memory: generator weights stay in VRAM");
    const bool pageable = text_.cfg_.image_snapshot_pageable;
    const char* const memory_mode = pageable ? "pageable" : "pinned";
    const int backing_kind = pageable ? RAD_MEM_HOST : RAD_MEM_HOST_PINNED;
    const int64_t budget = text_.cfg_.image_model_cache_mib * kMiB;
    int64_t full_snapshot_bytes = 0;
    bool cache_all_ranks = budget > 0 && image_->file_;
    for (const auto& rank : image_->ranks_) {
        const int64_t bytes = rank->pools.weights().used();
        if (bytes <= 0 || bytes > budget - full_snapshot_bytes) {
            cache_all_ranks = false;
            break;
        }
        full_snapshot_bytes += bytes;
    }
    for (size_t i = 0; i < image_->ranks_.size(); ++i) {
        auto& img = *image_->ranks_[i];
        auto& txt = *text_.ranks_[image_text_rank(i)];
        if (img.device != txt.device) return RAD_E_STATE;
        RAD_TRY(rad_dev_set(img.device));
        auto mem = std::make_unique<ImageMemory>(img.device, backing_kind);
        mem->weights = static_cast<char*>(img.pools.weights().base());
        const int64_t weights = img.pools.weights().bytes();
        const int64_t arena = img.ctx.arena_allocation_bytes();
        const int64_t transient = arena - img.program.arena_fixed_bytes;
        std::vector<LoanStrip> strips;
        if (lend_weights && weights > 0)
            strips.push_back({mem->weights + weights, weights, 1, 1});
        if (transient > 0)
            strips.push_back({static_cast<char*>(img.ctx.arena_base()) + arena, transient, 1, 1});
        RAD_TRY(txt.mover.lend_from(Mover::kLendImage, strips));
        mem->offered_bytes = txt.mover.loan_ceiling(Mover::kLendImage);
        mem->weights_lent = lend_weights && mem->offered_bytes > 0;
        if (mem->weights_lent) mem->weight_bytes = img.pools.weights().used();
        if (mem->weight_bytes > std::numeric_limits<int64_t>::max() - backing_bytes)
            return RAD_E_NOMEM;
        backing_bytes += mem->weight_bytes;
        if (!restored_memory_.empty() && mem->weight_bytes) {
            /* Validated cache ranks and the service's fixed snapshot mode keep the
             * device and allocator the same on both sides of each backing swap. */
            std::swap(mem->backing, restored_memory_[i]->backing);
        } else {
            incoming_bytes += mem->weight_bytes;
        }
        pending.push_back(std::move(mem));
    }

    /* A warm promotion already owns every rank's copy. Keep optional ranks only when
     * the full checkpoint fits, so dense or partially lending text can also warm-switch. */
    if (cache_all_ranks) {
        for (size_t i = 0; i < pending.size(); ++i) {
            const auto& rank = *image_->ranks_[i];
            std::vector<CachedWeight> layout;
            if (!describe_weight_pool(rank.program, rank.pools.weights(), &layout)) {
                cache_all_ranks = false;
                break;
            }
        }
    }
    if (cache_all_ranks) trim_image_cache(full_snapshot_bytes, incoming_bytes);
    else trim_image_cache(backing_bytes, incoming_bytes);

    /* The slots have zero loan depth until the snapshots are ready. */
    std::string why;
    if (incoming_bytes && !lockable_room(incoming_bytes, 0, text_.host_headroom_bytes(), false, &why)) {
        RAD_ERR("shared image weights need %s of %s RAM: %s",
                humanb(incoming_bytes).c_str(), memory_mode, why.c_str());
        return RAD_E_NOMEM;
    }
    uint64_t snapshot_alloc_ns = 0, snapshot_copy_ns = 0;
    auto capture_snapshot = [&](ImageMemory& mem, size_t rank) -> int {
        RAD_INFO("shared image snapshot: allocating %s of %s host memory for rank %zu (device %d)",
                 humanb(mem.weight_bytes).c_str(), memory_mode, rank, mem.device);
        const uint64_t alloc_begin = rad_mono_ns();
        mem.backing = rad_dev_alloc(mem.weight_bytes, mem.backing_kind);
        const uint64_t alloc_elapsed = rad_mono_ns() - alloc_begin;
        snapshot_alloc_ns += alloc_elapsed;
        if (!mem.backing) return RAD_E_NOMEM;
        RAD_INFO("shared image snapshot: rank %zu allocated in %.1f ms; copying device weights",
                 rank, double(alloc_elapsed) / 1e6);
        const uint64_t copy_begin = rad_mono_ns();
        const int status = mem.copy(image_->ranks_[rank]->ctx.stream(), false);
        snapshot_copy_ns += rad_mono_ns() - copy_begin;
        return status;
    };
    for (size_t i = 0; i < pending.size(); ++i) {
        auto& mem = *pending[i];
        if (!mem.weight_bytes || mem.backing) continue;
        RAD_TRY(rad_dev_set(mem.device));
        int status = capture_snapshot(mem, i);
        if (!mem.backing && !image_cache_.empty()) {
            trim_image_cache(std::numeric_limits<int64_t>::max(), 0);
            RAD_TRY(rad_dev_set(mem.device));
            status = capture_snapshot(mem, i);
        }
        if (!mem.backing) {
            const char* detail = rad_dev_last_error();
            RAD_ERR("shared image weights: rank %zu could not allocate %s of %s host memory for expert recall%s%s", i,
                    humanb(mem.weight_bytes).c_str(), memory_mode, *detail ? ": " : "", detail);
            return RAD_E_NOMEM;
        }
        RAD_TRY(status);
    }
    if (cache_all_ranks && full_snapshot_bytes > backing_bytes) {
        int64_t extra_bytes = 0;
        for (size_t i = 0; i < pending.size(); ++i) {
            if (!pending[i]->weight_bytes && restored_memory_.empty())
                extra_bytes += image_->ranks_[i]->pools.weights().used();
        }
        if (!extra_bytes || lockable_room(extra_bytes, remaining_image_lora_budget(),
                                         text_.host_headroom_bytes(), false, &why)) {
            for (size_t i = 0; i < pending.size(); ++i) {
                auto& mem = *pending[i];
                if (mem.weight_bytes) continue;
                const int64_t bytes = image_->ranks_[i]->pools.weights().used();
                RAD_TRY(rad_dev_set(mem.device));
                if (!restored_memory_.empty()) std::swap(mem.backing, restored_memory_[i]->backing);
                else {
                    mem.weight_bytes = bytes;
                    const int status = capture_snapshot(mem, i);
                    if (!mem.backing) {
                        mem.weight_bytes = 0;
                        break;
                    }
                    RAD_TRY(status);
                }
                mem.weight_bytes = bytes;
            }
        }
        const bool complete = std::all_of(pending.begin(), pending.end(),
                                          [](const auto& mem) { return mem->backing != nullptr; });
        if (complete) backing_bytes = full_snapshot_bytes;
        else {
            RAD_INFO("shared image cache: optional host snapshot unavailable for '%s'",
                     image_->file_->path());
            for (auto& mem : pending) {
                if (mem->weights_lent || !mem->backing) continue;
                mem->release_backing();
                mem->weight_bytes = 0;
            }
        }
    }
    restored_memory_.clear();
    if (snapshot_alloc_ns || snapshot_copy_ns)
        RAD_INFO("shared image snapshot capture: %s %.1f ms, device-to-host copies %.1f ms",
                 pageable ? "pageable allocation" : "host pinning",
                 double(snapshot_alloc_ns) / 1e6, double(snapshot_copy_ns) / 1e6);
    memory_ = std::move(pending);
    RAD_TRY(offer_image_memory());
    int64_t offered = 0;
    for (const auto& mem : memory_) offered += mem->offered_bytes;
    RAD_INFO("shared image memory: %s of immutable weights cached in host RAM; "
             "expert loan depth %s across ranks", humanb(backing_bytes).c_str(),
             humanb(offered).c_str());
    return RAD_OK;
}

int SharedService::drain(Engine& engine) {
    for (const auto& rank : engine.ranks_) {
        RAD_TRY(rad_dev_set(rank->device));
        RAD_TRY(rank->ctx.drain_streams());
        if (RadStream transfer = rank->mover.transfer_stream()) RAD_TRY(rad_stream_sync(transfer));
    }
    return rad_dev_set(engine.ranks_[0]->device);
}

int SharedService::offer_image_memory() {
    for (size_t i = 0; i < memory_.size(); ++i) {
        auto& rank = *text_.ranks_[image_text_rank(i)];
        RAD_TRY(rad_dev_set(rank.device));
        RAD_TRY(rank.mover.set_loan(Mover::kLendImage, memory_[i]->offered_bytes));
    }
    refill_at_ = 0;
    return rad_dev_set(text_.ranks_[0]->device);
}

int SharedService::recall_image_memory() {
    const uint64_t begin = rad_mono_ns();
    RAD_TRY(drain(text_));
    int64_t held = 0;
    for (auto& rank : text_.ranks_) {
        RAD_TRY(rad_dev_set(rank->device));
        held += rank->mover.loan_held(Mover::kLendImage);
        const int status = rank->mover.reclaim(Mover::kLendImage, 0, &rank->heat);
        if (status < 0 || rank->mover.loan_held(Mover::kLendImage) != 0) {
            RAD_ERR("shared image memory: recall failed on rank %d", rank->index);
            return status < 0 ? status : RAD_E_STATE;
        }
    }
    for (size_t i = 0; i < memory_.size(); ++i) {
        if (!memory_[i]->weights_lent) continue;
        auto& rank = *image_->ranks_[i];
        RAD_TRY(rad_dev_set(rank.device));
        RAD_TRY(memory_[i]->copy(rank.ctx.stream(), true));
    }
    RAD_INFO("shared image memory: recalled and restored in %.1f ms (held depth %s across ranks)",
             (double)(rad_mono_ns() - begin) / 1e6, humanb(held).c_str());
    return rad_dev_set(image_->ranks_[0]->device);
}

std::unique_lock<std::mutex> SharedService::lock_image_dispatch() {
    /* A busy text loop can otherwise reacquire the mutex for many consecutive
     * steps while an image lifecycle operation waits. The text loop yields at
     * its next step boundary. Once an operation owns dispatch_, the mutex itself
     * keeps text out until the operation releases it. */
    image_dispatch_waiters_.fetch_add(1, std::memory_order_acq_rel);
    std::unique_lock lock(dispatch_);
    image_dispatch_waiters_.fetch_sub(1, std::memory_order_acq_rel);
    return lock;
}

int SharedService::enter_text() {
    text_issued_ = true;
    if (active_ || image_waiting_.load(std::memory_order_acquire)) return RAD_OK;

    const uint64_t now = rad_mono_ns();
    if (now < refill_at_) return RAD_OK;
    refill_at_ = now + 100000000ull;
    for (auto& rank : text_.ranks_) {
        if (rank->mover.loan(Mover::kLendImage) == 0) continue;
        RAD_TRY(rad_dev_set(rank->device));
        RAD_TRY(rank->mover.idle_tick(&rank->heat));
    }
    return rad_dev_set(text_.ranks_[0]->device);
}

int SharedService::enter_image(bool* runnable) {
    *runnable = false;
    if (!image_->gen_->busy() || !text_issued_) return RAD_OK;
    if (text_.loop_status_.load(std::memory_order_acquire) < 0) return RAD_E_STATE;
    if (!active_) {
        RAD_TRY(recall_image_memory());
        active_ = true;
        text_steps_during_image_ = 0;
        drafts_verified_during_image_ = 0;
        drafts_accepted_during_image_ = 0;
    } else {
        RAD_TRY(drain(text_));
    }
    *runnable = true;
    return RAD_OK;
}

int SharedService::leave_image() {
    text_issued_ = false;
    const bool pending = image_->gen_->busy();
    image_waiting_.store(pending, std::memory_order_release);
    turn_.notify_all();
    if (pending) return RAD_OK;
    RAD_TRY(drain(*image_));
    active_ = false;
    RAD_TRY(offer_image_memory());
    RAD_INFO("shared image memory: parked; expert refill enabled (%lld text steps during images)",
             (long long)text_steps_during_image_);
    if (text_.cfg_.n_spec > 0)
        RAD_INFO("shared image memory: text draft verification during images: %lld checked, %lld accepted",
                 (long long)drafts_verified_during_image_, (long long)drafts_accepted_during_image_);
    return RAD_OK;
}

bool SharedService::wait_until_parked(std::chrono::steady_clock::time_point deadline) {
    std::unique_lock lock(dispatch_);
    return turn_.wait_until(lock, deadline, [this] {
        return failed() || (!active_ && !image_->gen_->busy());
    }) && !failed();
}

bool SharedService::failed() const {
    return image_failure_.load(std::memory_order_acquire) < 0 ||
           text_.loop_status_.load(std::memory_order_acquire) < 0;
}

void SharedService::shutdown_images(int status) {
    const std::lock_guard lock(dispatch_);
    if (image_) image_->gen_->shutdown("the shared service stopped", status);
}

}  /* namespace rad */
