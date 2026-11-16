#pragma once

#include "engine.h"
#include "format/radfile.h"
#include "image_queue.h"
#include "server/server.h"

#include <list>

namespace rad {

/* The HTTP listener belongs to the text engine. Both dispatch loops take turns
 * through this service before either can stage GPU work. */
class SharedService {
public:
    explicit SharedService(Config cfg);
    ~SharedService();

    int run(std::unique_ptr<StallWatch> watch);

private:
    friend class Engine;
    struct ImageMemory;
    struct CachedImage;
    struct UnloadedExpertMemory;
    using ImageSnapshot = std::vector<std::unique_ptr<ImageMemory>>;

    using GeneratorKind = server::GeneratorKind;
    static GeneratorKind generator_kind(const Program& program);

    struct GeneratorEndpoint {
        server::Deps::ImageEndpoint image;
        server::Deps::AudioEndpoint audio;
        server::Deps::VideoEndpoint video;
        std::string model_id;
        GeneratorKind kind() const {
            if (audio) return GeneratorKind::Audio;
            if (video) return GeneratorKind::Video;
            return GeneratorKind::Image;
        }
        long use_count() const {
            if (audio) return audio.owner.use_count();
            if (video) return video.owner.use_count();
            return image.owner.use_count();
        }
        explicit operator bool() const { return image || audio || video; }
    };

    GeneratorEndpoint acquire_generator();
    server::Deps::ImageEndpoint acquire_image();
    GeneratorKind generator_kind(const std::string& model);
    std::vector<std::string> image_model_ids();
    std::string image_status();
    int unload_image_admin(const std::string& name, const std::function<bool()>& cancelled,
                           std::string* why);
    int register_image_models(const std::string& text_id, const std::string& initial_id);
    int queue_generator(const std::string& model, GeneratorKind kind, size_t bytes, size_t max_waiters, size_t max_bytes,
                    const std::function<bool()>& cancelled, GeneratorEndpoint* endpoint,
                    std::shared_ptr<void>* hold, server::ImageDispatchTiming* timing, std::string* why);
    std::vector<std::string> image_lora_names();
    int pin_image_loras(const std::vector<std::pair<std::string, double>>& in,
                        std::vector<lora::PinnedAdapter>* out, std::shared_ptr<void>* lease,
                        std::string* why, bool reclaim_cache);
    int load_image_admin(const std::string& path, const std::string& name, std::string* why);
    struct ImageRegistration {
        std::string path;
        RadFile::Identity identity;
        uint64_t revision = 0;
        bool activation_failed = false;
        GeneratorKind kind = GeneratorKind::Image;
    };
    static int open_image_registration(const std::string& path, ImageRegistration* registration,
                                       std::string* why, std::shared_ptr<RadFile>* opened = nullptr);
    int register_image_admin(const std::string& path, const std::string& name, std::string* why);
    int load_image(const std::string& path, const std::string& name, std::string* why,
                   const RadFile::Identity* expected = nullptr, bool* busy = nullptr,
                   server::ImageSelection* selection = nullptr,
                   const GeneratorKind* expected_kind = nullptr);
    int prepare_image(Config cfg, std::shared_ptr<RadFile> opened = {},
                      const GeneratorKind* expected_kind = nullptr, bool* cached_placement = nullptr);
    int plan_image_memory();
    int check_image_footprint(const std::vector<int64_t>& footprint) const;
    int read_free_vram(std::vector<int64_t>* free) const;
    int describe_generator(GeneratorEndpoint* endpoint);
    int verify_generator(const GeneratorEndpoint& endpoint);
    int detach_image_memory(bool retain = false);
    int offer_unloaded_expert_memory();
    void cache_image_weights();
    int64_t remaining_image_lora_budget() const;
    void trim_image_cache(int64_t active_bytes, int64_t incoming_bytes);
    void evict_oldest_image_snapshot(const std::string& why);
    bool make_room_for_image_adapter(int64_t bytes);
    int restore_image_weights(Engine& image, bool* restored);
    void publish_generator(GeneratorEndpoint endpoint, bool register_revision = false);

    std::unique_lock<std::mutex> lock_image_dispatch();
    int enter_text();
    int enter_image(bool* runnable);
    int leave_image();
    bool wait_until_parked(std::chrono::steady_clock::time_point deadline);
    bool failed() const;
    void shutdown_images(int status);

    int lend_image_memory();
    int recall_image_memory();
    int offer_image_memory();
    int drain(Engine& engine);
    size_t image_text_rank(size_t image_rank) const;
    int64_t image_growth_bytes(int rank) const;
    int64_t image_reserve_bytes(int rank) const;

    struct ImageTurn {
        SharedService* service = nullptr;
        bool operator_load = false;
        ~ImageTurn();
    };
    std::mutex image_queue_mutex_;
    std::condition_variable image_queue_turn_;
    std::list<ImageWaiter> image_waiters_;
    std::map<std::string, ImageRegistration> image_models_;
    uint64_t next_image_revision_ = 1;
    std::string selected_image_model_;
    std::string default_image_model_;
    std::string activating_image_model_;
    std::string published_image_model_;
    std::string draining_image_model_;
    size_t image_waiting_bytes_ = 0;
    bool image_turn_in_use_ = false;
    bool image_operator_load_ = false;
    std::mutex image_lora_mutex_;
    lora::AdapterSnapshots image_lora_snapshots_;
    struct ImageAdapterRequests {
        std::mutex mutex;
        std::map<std::string, int> counts;
    };
    std::shared_ptr<ImageAdapterRequests> image_adapter_requests_ =
        std::make_shared<ImageAdapterRequests>();
    int image_adapter_request_count(const std::string& name) const;
    /* Empty paths are unload tombstones overriding discovered or startup sources. */
    std::map<std::string, std::string> image_lora_paths_;

    /* Reverse destruction keeps image allocations and the registry alive until
     * the text mover has retired its last use of borrowed addresses. */
    struct RegistryOwner { ~RegistryOwner(); } registry_;
    Config image_template_;  /* device binding is finalized after text declaration */
    std::unique_ptr<Engine> image_;
    std::vector<std::unique_ptr<UnloadedExpertMemory>> unloaded_expert_memory_;
    Engine text_;
    ImageSnapshot memory_;
    ImageSnapshot restored_memory_;
    std::list<std::unique_ptr<CachedImage>> image_cache_;  /* oldest first */
    int64_t image_cache_bytes_ = 0;
    std::vector<int64_t> image_extra_reserve_;
    std::vector<int64_t> image_budget_bytes_;

    std::mutex image_update_;
    std::mutex image_admission_;
    GeneratorEndpoint image_endpoint_;
    bool image_endpoint_draining_ = false;
    std::atomic<int> image_failure_{RAD_OK};
    std::mutex dispatch_;
    std::atomic<unsigned> image_dispatch_waiters_{0};
    std::condition_variable turn_;
    bool active_ = false;
    bool text_issued_ = false;
    std::atomic<bool> image_waiting_{false};
    uint64_t refill_at_ = 0;
    int64_t text_steps_during_image_ = 0;
    int64_t drafts_verified_during_image_ = 0;
    int64_t drafts_accepted_during_image_ = 0;
};

}  /* namespace rad */
