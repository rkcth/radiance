/* engine_lora.cpp -- runtime LoRA on the engine side: which modules adapters may target (fixed
 * before declare), and loading an adapter into a slot of every rank's pools.
 *
 * A load packs the adapter for each rank (lora::load_adapter, host only), refuses a module no rank
 * placed, copies each rank's bytes into the slot on a stream of its own, waits, and only then
 * publishes the name. A request can name the adapter only after the publish, so no step reads a
 * slot while it is written.
 *
 * A generator keeps its adapter files in a library instead (--lora, --lora-dir, the load
 * endpoint), and a slot holds one combination of them at given strengths, named by the
 * combination: the first request that asks for it composes it (lora/adapter.h) into a free slot,
 * or the least recently used one no request holds, and later ones find it there. */
#include "engine_priv.h"
#include "service.h"
#include "lora/lora.h"
#include "rad_plugin.h"

#include "util/json.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <dirent.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <thread>

namespace rad {

namespace {

/* adapter_config.json's target_modules as leaf names. PEFT also accepts a regex string and
 * "all-linear"; neither names leaves this engine can check before declare, so both are refused
 * with the flag that does. */
int read_targets(const std::string& dir, std::set<std::string>* out, std::string* err) {
    const std::string path = dir + "/adapter_config.json";
    std::ifstream f(path);
    if (!f) { *err = "cannot read " + path; return RAD_E_NOTFOUND; }
    Json j;
    try { f >> j; } catch (const std::exception& e) {
        *err = path + ": " + e.what();
        return RAD_E_INVAL;
    }
    const auto it = j.find("target_modules");
    if (it == j.end() || it->is_null()) { *err = path + " names no target_modules"; return RAD_E_INVAL; }
    if (it->is_string()) {
        *err = path + ": target_modules is '" + it->get<std::string>() +
               "', not a list of module names; give them with --lora-target-modules";
        return RAD_E_UNSUPPORTED;
    }
    if (!it->is_array()) { *err = path + ": target_modules is not a list"; return RAD_E_INVAL; }
    for (const auto& m : *it) {
        if (!m.is_string()) { *err = path + ": a target_modules entry is not a string"; return RAD_E_INVAL; }
        std::string s = m.get<std::string>();
        const size_t dot = s.rfind('.');
        if (dot != std::string::npos) s = s.substr(dot + 1);
        if (!s.empty()) out->insert(s);
    }
    return RAD_OK;
}

/* A generator's adapter name: a file stem under --lora-dir, possibly in a subdirectory, never
 * leaving it. */
bool valid_name(const std::string& n) {
    if (n.empty() || n.size() > 200 || n[0] == '/' || n[0] == '.') return false;
    if (n.find("..") != std::string::npos || n.find("//") != std::string::npos) return false;
    for (char c : n)
        if (!(std::isalnum((unsigned char)c) || c == '_' || c == '-' || c == '.' || c == '/' ||
              c == ' ' || c == '(' || c == ')' || c == '+' || c == ','))
            return false;
    return true;
}

bool is_file(const std::string& p) {
    struct stat sb{};
    return stat(p.c_str(), &sb) == 0 && S_ISREG(sb.st_mode);
}

/* Discovered adapters must stay inside the configured local library. */
std::string discovered_path(const std::string& dir, const std::string& name) {
    if (dir.empty() || !valid_name(name)) return {};
    std::error_code ec;
    const auto root = std::filesystem::canonical(dir, ec);
    if (ec) return {};
    const auto file = std::filesystem::canonical(root / (name + ".safetensors"), ec);
    if (ec || !std::filesystem::is_regular_file(file, ec)) return {};
    const auto relative = file.lexically_relative(root);
    if (relative.empty() || *relative.begin() == "..") return {};
    return file.string();
}

std::set<std::string> discovered_adapters(const std::string& directory) {
    std::set<std::string> n;
    if (!directory.empty()) {
        /* Two levels: the directory and its subdirectories, as a UI's lora folder is laid out. */
        std::vector<std::string> dirs{ "" };
        for (size_t i = 0; i < dirs.size(); ++i) {
            const std::string rel = dirs[i];
            DIR* d = opendir((directory + "/" + rel).c_str());
            if (!d) continue;
            while (dirent* e = readdir(d)) {
                const std::string fn = e->d_name;
                if (fn.empty() || fn[0] == '.') continue;
                const std::string p = rel + fn;
                struct stat sb{};
                if (stat((directory + "/" + p).c_str(), &sb) != 0) continue;
                if (S_ISDIR(sb.st_mode)) { if (rel.empty()) dirs.push_back(p + "/"); continue; }
                const size_t k = fn.size();
                if (k > 12 && fn.compare(k - 12, 12, ".safetensors") == 0) {
                    const std::string stem = p.substr(0, p.size() - 12);
                    if (!discovered_path(directory, stem).empty()) n.insert(stem);
                }
            }
            closedir(d);
        }
    }
    return n;
}

/* '|' is excluded from adapter names and never appears in a printed strength. */
std::string combo_key(const std::vector<Engine::LoraUse>& uses,
                      const std::vector<std::string>& versions = {}) {
    std::string k;
    for (size_t i = 0; i < uses.size(); ++i) {
        const auto& u = uses[i];
        k += fmt("%s%s*%.17g", k.empty() ? "" : "|", u.name.c_str(), u.weight);
        if (!versions.empty()) k += "@" + versions[i];
    }
    return k;
}

std::string adapter_identity(const std::string& path) {
    struct stat st {};
    if (path.empty() || stat(path.c_str(), &st) != 0) return {};
    return fmt("%s:%llu:%llu:%lld:%lld:%ld:%lld:%ld", path.c_str(),
               (unsigned long long)st.st_dev, (unsigned long long)st.st_ino,
               (long long)st.st_size, (long long)st.st_mtim.tv_sec, st.st_mtim.tv_nsec,
               (long long)st.st_ctim.tv_sec, st.st_ctim.tv_nsec);
}

}  /* namespace */

std::vector<std::string> SharedService::image_lora_names() {
    const std::lock_guard lock(image_lora_mutex_);
    auto names = discovered_adapters(image_template_.lora.dir);
    std::set<std::string> startup_names;
    const auto apply = [&](const std::string& name, const std::string& path) {
        if (is_file(path)) names.insert(name);
        else names.erase(name);
    };
    for (const auto& [name, path] : image_template_.lora.adapters)
        if (startup_names.insert(name).second) apply(name, path);
    for (const auto& [name, path] : image_lora_paths_) apply(name, path);
    return {names.begin(), names.end()};
}

int SharedService::pin_image_loras(const std::vector<std::pair<std::string, double>>& in,
                                   std::vector<lora::PinnedAdapter>* out, std::shared_ptr<void>* lease,
                                   std::string* why, bool reclaim_cache) {
    if (!image_template_.lora.enabled()) {
        *why = "the shared image service was started without adapters";
        return RAD_E_UNSUPPORTED;
    }
    std::map<std::string, double> strengths;
    for (const auto& [name, weight] : in) {
        if (!valid_name(name) || !std::isfinite(weight) || std::abs(weight) > 16.0) {
            *why = "invalid adapter name or strength for '" + name + "'";
            return RAD_E_INVAL;
        }
        strengths[name] += weight;
    }
    std::erase_if(strengths, [](const auto& entry) { return entry.second == 0.0; });
    if (strengths.size() > 16) {
        *why = "at most 16 adapters may be used in one request";
        return RAD_E_INVAL;
    }

    std::vector<lora::PinnedAdapter> resolved;
    int64_t pressure_bytes = 0;
    auto pin_sources = [&]() -> int {
        const std::lock_guard lock(image_lora_mutex_);
        pressure_bytes = 0;
        const uint64_t budget = static_cast<uint64_t>(text_.cfg_.image_lora_snapshot_mib) << 20;
        std::map<std::string, std::string> paths;
        for (const auto& [name, weight] : strengths) {
            if (std::abs(weight) > 16.0) {
                *why = "combined adapter strength exceeds [-16, 16] for '" + name + "'";
                return RAD_E_INVAL;
            }
            std::string path;
            if (const auto it = image_lora_paths_.find(name); it != image_lora_paths_.end()) {
                path = it->second;
            } else {
                for (const auto& [adapter, configured] : image_template_.lora.adapters)
                    if (name == adapter) { path = configured; break; }
                if (path.empty()) path = discovered_path(image_template_.lora.dir, name);
            }
            if (path.empty()) {
                *why = "no available adapter named '" + name + "'";
                return RAD_E_NOTFOUND;
            }
            paths.emplace(name, std::move(path));
        }
        for (const auto& [name, weight] : strengths) {
            std::shared_ptr<const lora::AdapterSnapshots::Source> source;
            RAD_TRY(image_lora_snapshots_.acquire(paths.at(name), budget, text_.host_headroom_bytes(),
                                                 &source, why, &pressure_bytes));
            resolved.push_back({{name, {source, &source->file}, weight}, source->revision});
        }

        if (!resolved.empty()) {
            struct RequestLease {
                std::shared_ptr<ImageAdapterRequests> requests;
                std::vector<std::string> names;
                size_t charged = 0;
                ~RequestLease() {
                    const std::lock_guard lock(requests->mutex);
                    for (size_t i = 0; i < charged; ++i) {
                        const auto it = requests->counts.find(names[i]);
                        if (--it->second == 0) requests->counts.erase(it);
                    }
                }
            };
            auto hold = std::make_shared<RequestLease>();
            hold->requests = image_adapter_requests_;
            for (const auto& adapter : resolved) hold->names.push_back(adapter.use.name);
            const std::lock_guard count_lock(hold->requests->mutex);
            for (const auto& name : hold->names) {
                ++hold->requests->counts[name];
                ++hold->charged;
            }
            *lease = std::move(hold);
        }
        *out = std::move(resolved);
        return RAD_OK;
    };
    const int status = pin_sources();
    /* Keep already acquired sources charged while checking if the failed source fits.
     * Re-resolve everything after eviction to respect concurrent adapter updates. */
    if (!reclaim_cache || !pressure_bytes || !make_room_for_image_adapter(pressure_bytes)) return status;
    resolved.clear();
    why->clear();
    return pin_sources();
}

int Engine::lora_targets() {
    if (!cfg_.lora.enabled()) return RAD_OK;
    lora_targets_ = std::set<std::string>(cfg_.lora.targets.begin(), cfg_.lora.targets.end());
    if (lora_targets_.empty()) {
        /* A PEFT directory names its targets; an adapter file (a generator's) and a directory of
         * them do not, and a generator's default is every plain GEMM under its declared roots. A
         * text model refuses "*" at declare. */
        for (const auto& [name, dir] : cfg_.lora.adapters) {
            if (is_file(dir)) { lora_targets_.insert("*"); continue; }
            std::string err;
            const int s = read_targets(dir, &lora_targets_, &err);
            if (s < 0) {
                RAD_ERR("lora: adapter '%s': %s", name.c_str(), err.c_str());
                return s;
            }
        }
        if (cfg_.lora.adapters.empty()) lora_targets_.insert("*");
    }
    lora_table_.reset(cfg_.lora.slots(), &lora_epoch_);
    return RAD_OK;
}

int Engine::lora_start() {
    if (!cfg_.lora.enabled()) return RAD_OK;
    lora_gen_ = ranks_[0]->program.is_pass_driven();
    if (!lora_gen_ && cfg_.lora.slots() < (int)cfg_.lora.adapters.size()) {
        RAD_ERR("lora: --max-loras %d holds fewer adapters than the %zu that --lora names",
                cfg_.lora.slots(), cfg_.lora.adapters.size());
        return RAD_E_INVAL;
    }
    size_t n_sites = 0;
    for (const auto& r : ranks_) n_sites += r->program.lora_sites.size();
    if (n_sites == 0) {
        std::string t;
        for (const std::string& x : lora_targets_) t += (t.empty() ? "" : ",") + x;
        RAD_ERR("lora: no GEMM of this model multiplies a weight of the targeted modules (%s)",
                t.c_str());
        return RAD_E_UNSUPPORTED;
    }
    /* Every target is a module some rank carries adapters on. A leaf the model lacks, or holds
     * only outside the trunk (embedding, lm_head, a tower, the drafter), would target nothing. */
    if (!lora_targets_.count("*")) {
        std::set<std::string> carried;
        for (const auto& r : ranks_)
            for (const lora::Site& site : r->lora_sites)
                for (const lora::SiteSource& src : site.sources)
                    if (src.targeted) carried.insert(src.module.substr(src.module.rfind('.') + 1));
        for (const std::string& x : lora_targets_)
            if (!carried.count(x)) {
                RAD_ERR("lora: the target module '%s' is no trunk linear of this model, so no "
                        "adapter could land on it; leave it out of --lora-target-modules (or of "
                        "the adapters' target_modules)", x.c_str());
                return RAD_E_UNSUPPORTED;
            }
    }
    /* The adapter ops run in their GEMM's domain and read the pools on the card, so a targeted
     * GEMM placed on the host could serve no adapter request. On the host backend every op and
     * the pools are in host memory. */
    if (!device_is_host())
        for (const auto& r : ranks_)
            for (const LoraIssueSite& site : r->program.lora_sites)
                if (r->ctx.op_domain((rad_op)site.gemm) != RAD_DOMAIN_DEVICE) {
                    RAD_ERR("lora: rank %d: the GEMM over '%s' is placed on the host, and its "
                            "adapter ops read pools on the card; keep the targeted weights in "
                            "VRAM or leave the module out of --lora-target-modules", r->index,
                            r->program.lora_pools[(size_t)site.pool].weight.c_str());
                    return RAD_E_UNSUPPORTED;
                }
    {
        const Program& P = ranks_[0]->program;
        int64_t slot_bytes = 0;
        for (const LoraPool& p : P.lora_pools) slot_bytes += 2 * p.R * p.K + 2 * p.N * p.R + 4 * p.N;
        std::string t;
        for (const std::string& x : lora_targets_) t += (t.empty() ? "" : ",") + x;
        RAD_INFO("lora: %zu GEMMs over %zu weights carry adapters on rank 0 (%s); %d slots of rank "
                 "%d, %s a slot, %s on the card", P.lora_sites.size(), P.lora_pools.size(), t.c_str(),
                 cfg_.lora.slots(), cfg_.lora.rank(lora_gen_), humanb(slot_bytes).c_str(),
                 humanb(slot_bytes * cfg_.lora.slots()).c_str());
    }
    if (lora_gen_) return gen_lora_start();
    for (const auto& [name, dir] : cfg_.lora.adapters) {
        std::string err;
        const int s = lora_load(name, dir, &err);
        if (s < 0) {
            RAD_ERR("lora: adapter '%s' (%s): %s", name.c_str(), dir.c_str(), err.c_str());
            return s;
        }
    }
    return RAD_OK;
}

int Engine::lora_load(const std::string& name, const std::string& dir, std::string* err) {
    std::lock_guard<std::mutex> lk(lora_mu_);
    if (!cfg_.lora.enabled()) { *err = "the engine was started without adapters"; return RAD_E_STATE; }
    if (name.empty()) { *err = "an adapter needs a name"; return RAD_E_INVAL; }
    if (lora_gen_) {
        /* A generator's adapter goes into the library; a slot takes it when a request asks. */
        if (!valid_name(name)) { *err = "'" + name + "' is not an adapter name"; return RAD_E_INVAL; }
        {
            std::lock_guard<std::mutex> lk2(lora_lib_mu_);
            if (lora_lib_.count(name)) { *err = "an adapter named '" + name + "' is already loaded"; return RAD_E_INVAL; }
        }
        auto f = std::make_shared<lora::AdapterFile>();
        RAD_TRY(f->open(dir, err));
        int placed = 0, rank = 0;
        std::string first;
        for (const lora::FileModule& m : f->modules()) {
            std::string why;
            if (!lora_keys_->resolve(m.key, &why).empty()) { ++placed; continue; }
            if (first.empty()) first = m.key + ": " + why;
        }
        for (const auto& [key, tensor] : f->biases()) {
            (void)tensor;
            std::string why;
            if (!lora_keys_->resolve(key, &why).empty()) ++placed;
        }
        for (const lora::FileModule& m : f->modules())
            if (m.kind == lora::FileModule::LORA && m.t[0]) rank = std::max(rank, (int)m.t[0]->shape[0]);
        if (placed == 0) {
            *err = fmt("none of its %zu modules is a module of this model (%s)",
                       f->modules().size() + f->biases().size(),
                       first.c_str());
            return RAD_E_NOTFOUND;
        }
        RAD_INFO("lora: '%s' -- %s, %d of %zu modules on this model%s%s%s", name.c_str(),
                 f->kind_name(), placed, f->modules().size() + f->biases().size(),
                 rank ? fmt(", rank %d", rank).c_str() : "", first.empty() ? "" : "; not placed: ",
                 first.c_str());
        std::lock_guard<std::mutex> lk2(lora_lib_mu_);
        lora_lib_[name] = { f, {}, next_adapter_version_++ };
        gen_lora_unloaded_.erase(name);
        return RAD_OK;
    }
    /* A slot that is still cooling (lora/slots.h) is waited for here rather than answered with a
     * retry: the step loop goes round idle or busy, so the wait is milliseconds, and a hot swap
     * (unload, then load at once) would otherwise fail under traffic. Only other loads wait on
     * lora_mu_ meanwhile; requests and unloads use the table alone. */
    const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    int slot = -1;
    for (;;) {
        bool cooling = false;
        const int s = lora_table_.reserve(name, &slot, &cooling, err);
        if (s >= 0) break;
        if (s != RAD_E_FULL || !cooling || stopping_.load(std::memory_order_acquire) ||
            std::chrono::steady_clock::now() >= give_up)
            return s;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    /* ---- pack, per rank (host only) */
    std::vector<lora::AdapterInfo> infos(ranks_.size());
    std::vector<std::vector<lora::PackedSite>> packed(ranks_.size());
    for (size_t ri = 0; ri < ranks_.size(); ++ri) {
        const int s = lora::load_adapter(dir, ranks_[ri]->lora_sites, cfg_.lora.rank(false), &infos[ri],
                                         &packed[ri], err);
        if (s < 0) {
            if (ranks_.size() > 1) *err = fmt("rank %zu: ", ri) + *err;
            return s;
        }
    }
    {
        const int s = lora::check_placed(infos, err, &ranks_[0]->program, &lora_targets_);
        if (s < 0) return s;
    }

    RAD_TRY(lora_copy(slot, packed, err));

    lora_table_.publish(slot, name, dir);
    const lora::AdapterInfo& a = infos[0];
    RAD_INFO("lora: '%s' in slot %d -- %zu modules, rank %lld, alpha %g%s, base %s", name.c_str(),
             slot, a.modules.size(), (long long)a.max_module_rank, a.alpha, a.rslora ? " (rsLoRA)" : "",
             a.base_model.empty() ? "(unnamed)" : a.base_model.c_str());
    return RAD_OK;
}

int Engine::lora_copy(int slot, const std::vector<std::vector<lora::PackedSite>>& packed,
                      std::string* err) {
    /* Shared image uploads already hold the dispatch lock. Standalone uploads need
     * a separate stream because the compute stream may be recording a pass. */
    for (size_t ri = 0; ri < ranks_.size(); ++ri) {
        Rank& r = *ranks_[ri];
        const Program& P = r.program;
        if (packed[ri].size() != P.lora_pools.size()) {
            *err = fmt("rank %zu: the adapter was packed for %zu pools, the program has %zu", ri,
                       packed[ri].size(), P.lora_pools.size());
            return RAD_E_STATE;
        }
        if (rad_dev_set(r.device) < 0) { *err = fmt("rank %zu: cannot bind its device", ri); return RAD_E_DEVICE; }
        RadStream st = shared_ ? r.ctx.stream() : nullptr;
        if (!st && (shared_ || rad_stream_create(&st, 0) < 0)) { *err = fmt("rank %zu: no copy stream", ri); return RAD_E_DEVICE; }
        int rc = RAD_OK;
        for (size_t p = 0; p < P.lora_pools.size() && rc >= 0; ++p) {
            const LoraPool& lp = P.lora_pools[p];
            const lora::PackedSite& ps = packed[ri][p];
            const int64_t ab = 2 * lp.R * lp.K, bb = 2 * lp.N * lp.R, sb = 4 * lp.N;
            char* da = (char*)P.buffers[lp.a].ptr;
            char* db = (char*)P.buffers[lp.b].ptr;
            char* ds = (char*)P.buffers[lp.scale].ptr;
            if (!da || !db || !ds) {
                rc = RAD_E_STATE;
                *err = "the adapter pool of '" + lp.weight + "' has no buffer address";
                break;
            }
            da += (int64_t)slot * ab; db += (int64_t)slot * bb; ds += (int64_t)slot * sb;
            /* A pool this adapter does not touch is zeroed, not left as it was: the bytes are what
             * the slot's last adapter or the arena held, and a scale of 0 times a NaN is a NaN. */
            if (ps.empty) {
                if (rad_memset_async(da, 0, ab, st) < 0 || rad_memset_async(db, 0, bb, st) < 0 ||
                    rad_memset_async(ds, 0, sb, st) < 0)
                    rc = RAD_E_DEVICE;
                continue;
            }
            if ((int64_t)ps.a.size() * 2 != ab || (int64_t)ps.b.size() * 2 != bb ||
                (int64_t)ps.scale.size() * 4 != sb) {
                *err = fmt("rank %zu: pool '%s' was packed as %zu/%zu/%zu elements, its slot holds "
                           "%lld/%lld/%lld", ri, lp.weight.c_str(), ps.a.size(), ps.b.size(),
                           ps.scale.size(), (long long)(ab / 2), (long long)(bb / 2), (long long)(sb / 4));
                rc = RAD_E_STATE;
                break;
            }
            if (rad_memcpy_async(da, ps.a.data(), ab, st) < 0 ||
                rad_memcpy_async(db, ps.b.data(), bb, st) < 0 ||
                rad_memcpy_async(ds, ps.scale.data(), sb, st) < 0)
                rc = RAD_E_DEVICE;
        }
        const int sy = rad_stream_sync(st);
        if (!shared_) rad_stream_destroy(st);
        if (rc >= 0 && sy < 0) rc = RAD_E_DEVICE;
        if (rc < 0) {
            if (err->empty()) *err = fmt("rank %zu: the copy into slot %d failed", ri, slot);
            return rc;
        }
    }
    return RAD_OK;
}

int Engine::lora_acquire(const std::string& name, int32_t* slot, uint64_t* uid,
                         std::shared_ptr<void>* hold) {
    return lora_table_.acquire(name, slot, uid, hold);
}

int Engine::lora_unload(const std::string& name, std::string* err, int* in_flight,
                        bool known_to_service) {
    if (!lora_gen_) return lora_table_.unload(name, in_flight, err);
    std::lock_guard<std::mutex> update(lora_mu_);
    /* A generator's adapter leaves the library, and every combination holding it its slot (at
     * once when idle, after its requests otherwise). */
    {
        std::lock_guard<std::mutex> lk(lora_lib_mu_);
        if (!lora_lib_.erase(name) && !known_to_service &&
            (!shared_ || gen_lora_unloaded_.count(name) ||
             discovered_path(cfg_.lora.dir, name).empty())) {
            *err = "no adapter named '" + name + "' is loaded";
            return RAD_E_NOTFOUND;
        }
        if (shared_) gen_lora_unloaded_.insert(name);
    }
    *in_flight = 0;
    for (const std::string& k : lora_table_.names()) {
        bool uses = false;
        size_t at = 0;
        while (at <= k.size()) {
            const size_t plus = std::min(k.find('|', at), k.size());
            const std::string part = k.substr(at, plus - at);
            if (part.substr(0, part.rfind('*')) == name) uses = true;
            at = plus + 1;
        }
        if (!uses) continue;
        int n = 0;
        std::string e;
        lora_table_.unload(k, &n, &e);
        *in_flight += n;
    }
    return RAD_OK;
}

std::vector<std::string> Engine::lora_names() const {
    return lora_gen_ ? gen_lora_names() : lora_table_.names();
}

/* ================================================================== generator adapters */
int Engine::gen_lora_start() {
    /* Every slot's run of the slot-row table holds its own index (Program::lora_rows). */
    for (auto& rp : ranks_) {
        Rank& r = *rp;
        const Program& P = r.program;
        if (!P.lora_rows || P.lora_rows_n <= 0) continue;
        RAD_TRY(rad_dev_set(r.device));
        for (const auto& [weight, bias] : P.lora_replacements) {
            auto& base = r.lora_base_weights[weight];
            base.resize((size_t)bias.rows);
            const void* src = r.ctx.weight_ptr(weight);
            if (!src) return RAD_E_STATE;
            RAD_TRY(rad_memcpy_async(base.data(), src, bias.rows * 2, r.ctx.stream()));
            RAD_TRY(rad_stream_sync(r.ctx.stream()));
        }
        const int S = cfg_.lora.slots();
        std::vector<int32_t> v((size_t)(S * P.lora_rows_n));
        for (int s = 0; s < S; ++s)
            std::fill(v.begin() + (int64_t)s * P.lora_rows_n, v.begin() + (int64_t)(s + 1) * P.lora_rows_n, s);
        void* d = P.buffers[P.lora_rows].ptr;
        if (!d || rad_dev_set(r.device) < 0) {
            RAD_ERR("lora: rank %d: the slot-row table has no address", r.index);
            return RAD_E_STATE;
        }
        const RadStream st = r.ctx.stream();
        if (!st) return RAD_E_DEVICE;
        const int rc = rad_memcpy_async(d, v.data(), (int64_t)v.size() * 4, st);
        const int sy = rad_stream_sync(st);
        if (rc < 0 || sy < 0) return RAD_E_DEVICE;
    }
    lora_keys_ = std::make_unique<lora::KeyMap>(ranks_[0]->program, ranks_[0]->lora_sites);
    for (const auto& [name, path] : cfg_.lora.adapters) {
        std::string err;
        const int s = lora_load(name, path, &err);
        if (s < 0) {
            RAD_ERR("lora: adapter '%s' (%s): %s", name.c_str(), path.c_str(), err.c_str());
            return s;
        }
    }
    if (!cfg_.lora.dir.empty())
        RAD_INFO("lora: requests may name the %zu adapter files under %s", gen_lora_names().size(),
                 cfg_.lora.dir.c_str());
    return RAD_OK;
}

int Engine::gen_lora_prepare_replacements(
    const std::map<std::string, lora::Composite>& modules,
    const std::map<std::string, std::vector<float>>& deltas,
    LoraReplacements* replacements, std::string* err) const {
    replacements->resize(ranks_.size());
    for (size_t i = 0; i < ranks_.size(); ++i) {
        const auto& r = *ranks_[i];
        std::set<std::string> matched;
        for (const auto& [weight, bias] : r.program.lora_replacements) {
            auto data = r.lora_base_weights.at(weight);
            const auto delta = deltas.find(bias.module);
            if (bias.row_mixer) {
                if (const auto it = modules.find(bias.module); it != modules.end()) {
                    const auto& factors = it->second;
                    if (factors.rows != 1 || factors.K != bias.rows) {
                        *err = "adapter shape disagrees with row mixer " + bias.module;
                        return RAD_E_SHAPE;
                    }
                    for (int64_t i = 0; i < bias.rows; ++i) {
                        float adjustment = 0;
                        for (int64_t j = 0; j < factors.r; ++j)
                            adjustment += factors.B[(size_t)j] * factors.A[(size_t)(j * factors.K + i)];
                        data[(size_t)i] = rad_f32_to_bf16(rad_bf16_to_f32(data[(size_t)i]) + adjustment);
                    }
                }
            } else if (delta != deltas.end()) {
                const int64_t rows = bias.source_rows.empty() ? bias.rows :
                    1 + *std::max_element(bias.source_rows.begin(), bias.source_rows.end());
                if ((int64_t)delta->second.size() != rows) {
                    *err = "bias shape disagrees with checkpoint module " + bias.module;
                    return RAD_E_SHAPE;
                }
                matched.insert(delta->first);
                for (int64_t i = 0; i < bias.rows; ++i) {
                    const auto row = bias.source_rows.empty() ? i : bias.source_rows[(size_t)i];
                    if (row >= (int64_t)delta->second.size()) {
                        *err = "bias row mapping exceeds " + bias.module;
                        return RAD_E_SHAPE;
                    }
                    data[(size_t)i] = rad_f32_to_bf16(rad_bf16_to_f32(data[(size_t)i]) +
                                                     delta->second[(size_t)row]);
                }
            }
            (*replacements)[i].push_back({bias.slots, std::move(data)});
        }
        if (matched.size() != deltas.size()) {
            for (const auto& [module, values] : deltas) {
                (void)values;
                if (!matched.count(module)) {
                    *err = "no supported bias operand for " + module;
                    return RAD_E_UNSUPPORTED;
                }
            }
        }
    }
    return RAD_OK;
}

int Engine::gen_lora_copy_replacements(int slot, const LoraReplacements& replacements) {
    for (size_t rank = 0; rank < ranks_.size(); ++rank) {
        auto& r = *ranks_[rank];
        if (replacements[rank].empty()) continue;
        RAD_TRY(rad_dev_set(r.device));
        for (const auto& [buffer, data] : replacements[rank]) {
            auto* dst = static_cast<uint16_t*>(r.ctx.buf_ptr(buffer));
            if (!dst) return RAD_E_STATE;
            RadStream stream = r.ctx.stream();
            if (!shared_) RAD_TRY(rad_stream_create(&stream, 0));
            const int copy = rad_memcpy_async(dst + slot * data.size(), data.data(),
                                              data.size() * sizeof(uint16_t), stream);
            const int sync = rad_stream_sync(stream);
            if (!shared_) rad_stream_destroy(stream);
            RAD_TRY(copy);
            RAD_TRY(sync);
        }
    }
    return RAD_OK;
}

int Engine::gen_lora_open(const std::string& name, std::shared_ptr<const lora::AdapterFile>* out,
                          uint64_t* version, std::string* err) {
    const std::string path = discovered_path(cfg_.lora.dir, name);
    const std::string identity = shared_ ? adapter_identity(path) : std::string{};
    {
        const std::lock_guard lock(lora_lib_mu_);
        auto it = lora_lib_.find(name);
        if (it != lora_lib_.end() &&
            (it->second.identity.empty() || it->second.identity == identity)) {
            *out = it->second.file;
            *version = it->second.version;
            return RAD_OK;
        }
        if (gen_lora_unloaded_.count(name)) {
            *err = "adapter '" + name + "' was unloaded; load it before using it again";
            return RAD_E_NOTFOUND;
        }
        if (it != lora_lib_.end()) lora_lib_.erase(it);
    }
    if (path.empty() || (shared_ && identity.empty())) {
        *err = "no adapter named '" + name + "'" +
               (cfg_.lora.dir.empty() ? std::string(" is loaded")
                                      : " is loaded or under --lora-dir (GET /v1/loras lists them)");
        return RAD_E_NOTFOUND;
    }
    auto f = std::make_shared<lora::AdapterFile>();
    RAD_TRY(f->open(path, err));
    if (shared_ && adapter_identity(path) != identity) {
        *err = "the adapter changed while it was being opened; retry";
        return RAD_E_STATE;
    }
    RAD_INFO("lora: '%s' opened from --lora-dir -- %s, %zu modules", name.c_str(), f->kind_name(),
             f->modules().size());
    const std::lock_guard lock(lora_lib_mu_);
    *version = next_adapter_version_++;
    lora_lib_[name] = { f, identity, *version };
    *out = std::move(f);
    return RAD_OK;
}

std::vector<std::string> Engine::gen_lora_names() const {
    auto n = discovered_adapters(cfg_.lora.dir);
    const std::lock_guard lock(lora_lib_mu_);
    for (const auto& [name, entry] : lora_lib_)
        if (entry.identity.empty()) n.insert(name);
    for (const auto& name : gen_lora_unloaded_) n.erase(name);
    return std::vector<std::string>(n.begin(), n.end());
}

int Engine::gen_lora_acquire(const std::vector<LoraUse>& uses_in, int32_t* slot,
                             std::shared_ptr<void>* hold, std::string* note, std::string* err) {
    return gen_lora_acquire_impl(uses_in, nullptr, slot, hold, note, err);
}

int Engine::gen_lora_acquire_pinned(const std::vector<lora::PinnedAdapter>& adapters,
                                    int32_t* slot, std::shared_ptr<void>* hold,
                                    std::string* note, std::string* err) {
    std::vector<LoraUse> uses;
    for (const auto& adapter : adapters) uses.push_back({adapter.use.name, adapter.use.weight});
    return gen_lora_acquire_impl(uses, &adapters, slot, hold, note, err);
}

int Engine::gen_lora_prepare(const std::vector<lora::Strength>& ss,
                             std::vector<std::vector<lora::PackedSite>>* packed,
                             LoraReplacements* replacements, lora::ComposeReport* report,
                             size_t* bias_count, std::string* err) const {
    const int rank = cfg_.lora.rank(true);
    std::map<std::string, lora::Composite> comp;
    auto& rep = *report;
    RAD_TRY(lora::compose(*lora_keys_, ss, rank, &comp, &rep, err));
    std::map<std::string, std::vector<float>> biases;
    RAD_TRY(lora::compose_biases(*lora_keys_, ss, &biases, err));
    if (rep.modules == 0 && biases.empty()) {
        *err = "none of the adapters' modules is a module of this model" +
               (rep.unplaced.empty() ? std::string() : " (" + rep.unplaced[0] + ")");
        return RAD_E_NOTFOUND;
    }
    if (shared_ && (!rep.unplaced.empty() || !rep.skipped.empty() || rep.exact != rep.modules)) {
        const std::string detail = !rep.unplaced.empty() ? rep.unplaced.front()
                                 : !rep.skipped.empty() ? rep.skipped.front()
                                 : fmt("%s exceeds the configured adapter rank %d",
                                       rep.worst_module.c_str(), rank);
        *err = "shared image mode requires an exact adapter match: " + detail;
        return RAD_E_INVAL;
    }
    const std::map<std::string, lora::AdModule> mods = lora::as_modules(comp, *lora_keys_);
    packed->resize(ranks_.size());
    for (size_t ri = 0; ri < ranks_.size(); ++ri) {
        lora::AdapterInfo info;
        const int s = lora::pack_modules(ranks_[ri]->lora_sites, rank, mods, &info, &(*packed)[ri], err,
                                         /*compact=*/true);
        if (s < 0) {
            if (ranks_.size() > 1) *err = fmt("rank %zu: ", ri) + *err;
            return s;
        }
        /* Pools past the last one a site names hold nothing (lora_declare's dropped GEMMs). */
        (*packed)[ri].resize(ranks_[ri]->program.lora_pools.size());
    }
    *bias_count = biases.size();
    return gen_lora_prepare_replacements(comp, biases, replacements, err);
}

int Engine::gen_lora_validate(const std::vector<lora::PinnedAdapter>& adapters, std::string* err) {
    const std::unique_lock lock(lora_mu_, std::try_to_lock);
    if (!lock.owns_lock()) {
        *err = "adapter composition is already in progress";
        return RAD_E_FULL;
    }
    std::vector<lora::Strength> uses;
    for (const auto& adapter : adapters) uses.push_back(adapter.use);
    std::vector<std::vector<lora::PackedSite>> packed;
    LoraReplacements replacements;
    lora::ComposeReport report;
    size_t bias_count = 0;
    return gen_lora_prepare(uses, &packed, &replacements, &report, &bias_count, err);
}

int Engine::gen_lora_acquire_impl(const std::vector<LoraUse>& uses_in,
                                  const std::vector<lora::PinnedAdapter>* pinned,
                                  int32_t* slot, std::shared_ptr<void>* hold,
                                  std::string* note, std::string* err) {
    *slot = -1;
    hold->reset();
    if (note) note->clear();
    if (!lora_gen_) { *err = "this model takes no adapters (start it with --lora or --lora-dir)"; return RAD_E_STATE; }
    /* One entry per adapter, strengths of a repeated name summed; strength 0 is no adapter. */
    std::map<std::string, double> by;
    for (const LoraUse& u : uses_in) {
        if (!std::isfinite(u.weight) || std::fabs(u.weight) > 16.0) {
            *err = fmt("adapter '%s': strength %g is outside [-16, 16]", u.name.c_str(), u.weight);
            return RAD_E_INVAL;
        }
        by[u.name] += u.weight;
    }
    std::vector<LoraUse> uses;
    for (const auto& [n, w] : by) if (w != 0.0) uses.push_back({ n, w });
    if (uses.empty()) return RAD_OK;
    if (uses.size() > 16) { *err = fmt("%zu adapters in one request; at most 16", uses.size()); return RAD_E_INVAL; }
    std::string key;
    uint64_t uid = 0;
    if (!shared_) {
        key = combo_key(uses);
        if (lora_table_.acquire(key, slot, &uid, hold) == RAD_OK) return RAD_OK;
    }

    std::lock_guard<std::mutex> lk(lora_mu_);
    if (!shared_ && lora_table_.acquire(key, slot, &uid, hold) == RAD_OK) return RAD_OK;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<lora::Strength> ss;
    std::vector<std::string> versions;
    if (pinned) {
        for (const auto& adapter : *pinned) {
            ss.push_back(adapter.use);
            versions.push_back(adapter.revision);
        }
    } else {
        for (const LoraUse& u : uses) {
            lora::Strength s;
            s.name = u.name;
            s.weight = u.weight;
            uint64_t version = 0;
            RAD_TRY(gen_lora_open(u.name, &s.file, &version, err));
            versions.push_back(std::to_string(version));
            ss.push_back(std::move(s));
        }
    }
    if (shared_) {
        key = combo_key(uses, versions);
        if (lora_table_.acquire(key, slot, &uid, hold) == RAD_OK) return RAD_OK;
    }
    std::vector<std::vector<lora::PackedSite>> packed;
    LoraReplacements replacements;
    lora::ComposeReport rep;
    size_t bias_count = 0;
    RAD_TRY(gen_lora_prepare(ss, &packed, &replacements, &rep, &bias_count, err));
    const int rank = cfg_.lora.rank(true);
    /* Finish validation and packing before evicting an idle combination. A rejected
     * adapter should leave the current slot available to subsequent requests. */
    const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    int sl = -1;
    for (;;) {
        bool cooling = false;
        const int s = lora_table_.reserve(key, &sl, &cooling, err);
        if (s >= 0) break;
        if (s == RAD_E_FULL && !cooling) {
            std::string ev;
            if (lora_table_.evict_idle(&ev) == RAD_OK) continue;
            *err = fmt("all %d adapter slots are held by running requests with other adapters "
                       "(--max-loras); retry when one finishes", cfg_.lora.slots());
            return RAD_E_FULL;
        }
        if (s != RAD_E_FULL || stopping_.load(std::memory_order_acquire) ||
            std::chrono::steady_clock::now() >= give_up)
            return s;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    /* The adapter pools are outside the expert loan. Use the image lifecycle
     * turn so continuously runnable text yields before the adapter upload. */
    std::unique_lock<std::mutex> dispatch;
    if (shared_) {
        dispatch = shared_->lock_image_dispatch();
        if (shared_->failed()) {
            *err = "the shared service is shutting down";
            return RAD_E_STATE;
        }
        if (const int st = shared_->drain(shared_->text_); st < 0) {
            *err = fmt("could not drain text work before uploading the image adapter: %s",
                       rad_strerror(st));
            return st;
        }
    }
    if (const int st = gen_lora_copy_replacements(sl, replacements); st < 0) {
        *err = fmt("could not upload image adapter replacement weights to slot %d: %s",
                   sl, rad_strerror(st));
        return st;
    }
    RAD_TRY(lora_copy(sl, packed, err));
    if (bias_count)
        RAD_INFO("lora: slot %d includes %zu bias deltas", sl, bias_count);
    /* The ranks each pool takes in this slot, before a pass can name the slot (the publish). */
    for (size_t ri = 0; ri < ranks_.size(); ++ri) {
        Program& P = ranks_[ri]->program;
        const size_t S = (size_t)cfg_.lora.slots();
        for (size_t p = 0; p < P.lora_pools.size() && p < packed[ri].size(); ++p)
            if (p * S + (size_t)sl < P.lora_rused.size())
                P.lora_rused[p * S + (size_t)sl] = packed[ri][p].empty ? 0 : (int32_t)packed[ri][p].r_used;
    }
    lora_table_.publish(sl, key, "");
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::string n;
    if (!rep.unplaced.empty())
        n += fmt("%zu adapter module%s not on this model (%s)", rep.unplaced.size(),
                 rep.unplaced.size() == 1 ? "" : "s", rep.unplaced[0].c_str());
    if (rep.exact < rep.modules)
        n += fmt("%s%lld module%s past rank %d were truncated to it, keeping at worst %.1f%% of the "
                 "delta (%s)", n.empty() ? "" : "; ", (long long)(rep.modules - rep.exact),
                 rep.modules - rep.exact == 1 ? "" : "s", rank, 100.0 * rep.worst_kept,
                 rep.worst_module.c_str());
    RAD_INFO("lora: slot %d <- %s: %lld modules (%lld exact) in %.0f ms%s%s", sl, key.c_str(),
             (long long)rep.modules, (long long)rep.exact, ms, n.empty() ? "" : "; ", n.c_str());
    if (note) *note = n;
    if (lora_table_.acquire(key, slot, &uid, hold) != RAD_OK) {
        *err = "the combination's slot was taken back before the request could hold it; retry";
        return RAD_E_STATE;
    }
    return RAD_OK;
}

}  /* namespace rad */
