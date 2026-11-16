/* engine_gen.cpp -- a generator's loop: the driver's passes (core/gen/pass.h; driver.h for
 * images, audio_driver.h for audio, video_driver.h for video, shape_driver.h for shapes) run on the
 * ranks like token steps, one at a time, each drained before the next.
 *
 * Draining is cheap: a pass is tens to thousands of milliseconds of device work, and the next
 * pass depends only on the driver's host bookkeeping, so the wait costs the microseconds of
 * building it. Issuing ahead would need a second staging region and a commit protocol.
 */
#include "engine_priv.h"
#include "service.h"
#include "gen/driver.h"
#include "gen/pass.h"

#include <chrono>
#include <cstdlib>
#include <string_view>
#include <thread>

namespace rad {

/* A generator pass's kind, for the buffer probe's lines. */
static const char* pass_name(int gen) {
    switch (gen) {
    case RAD_GEN_PASS_COND:    return "cond";
    case RAD_GEN_PASS_DENOISE: return "denoise";
    case RAD_GEN_PASS_DECODE:  return "decode";
    case RAD_GEN_PASS_PROMPT:  return "prompt";
    case RAD_GEN_PASS_FRAME:   return "frame";
    case RAD_GEN_PASS_WINDOW:  return "window";
    case RAD_GEN_PASS_VOCODE:  return "vocode";
    case RAD_GEN_PASS_VCOND:   return "vcond";
    case RAD_GEN_PASS_VSTEP:   return "vstep";
    case RAD_GEN_PASS_ADECODE: return "adecode";
    case RAD_GEN_PASS_VDECODE: return "vdecode";
    case RAD_GEN_PASS_VUPSCALE: return "vupscale";
    case RAD_GEN_PASS_VENCODE: return "vencode";
    case RAD_GEN_PASS_AENCODE: return "aencode";
    case RAD_GEN_PASS_SCOND:   return "scond";
    case RAD_GEN_PASS_SSTEP:   return "sstep";
    case RAD_GEN_PASS_SLATENT: return "slatent";
    case RAD_GEN_PASS_SQUERY:  return "squery";
    default:                   return "";
    }
}

int Engine::gen_loop() {
    if (rad_dev_set(ranks_[0]->device) < 0)
        RAD_ERR("generator loop: could not bind rank 0's device");
    int step = 0, idle = 0;
    const char* timing = std::getenv("RADIANCE_IMAGE_PASS_TIMING");
    const bool trace_image_passes = shared_ && img_gen_ && timing &&
                                    std::string_view(timing) == "1";
    /* An error ends the driver's requests, then the loop. */
    auto stop = [&](int st) {
        if (shared_) shared_->image_failure_.store(st, std::memory_order_release);
        gen_->shutdown("the engine stopped on an error and is not serving", st);
        return fail_loop(st);
    };
    while (!stopping_.load(std::memory_order_acquire)) {
        /* A round of the loop, idle or not: what lets a freed adapter slot cool (lora/slots.h). */
        lora_epoch_.fetch_add(1, std::memory_order_acq_rel);
        quiet_service();
        std::unique_lock<std::mutex> dispatch;
        if (shared_ && gen_) {
            const bool pending = gen_->busy();
            shared_->image_waiting_.store(pending, std::memory_order_release);
            if (pending) {
                dispatch = std::unique_lock<std::mutex>(shared_->dispatch_);
                bool runnable = false;
                if (const int st = shared_->enter_image(&runnable); st < 0) return stop(st);
                if (!runnable) dispatch.unlock();
            }
            if (!dispatch.owns_lock()) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                continue;
            }
        }
        const uint64_t pass_begin = trace_image_passes ? rad_mono_ns() : 0;
        const RadBatch* b = gen_->next(step);
        if (!b) {
            if (dispatch.owns_lock()) {
                if (const int st = shared_->leave_image(); st < 0) return stop(st);
                dispatch.unlock();
            }
            step_batch_.store(nullptr, std::memory_order_release);
            if (hold_req_.load(std::memory_order_acquire)) {
                hold_point();
                continue;
            }
            if (idle < 200) ++idle;
            std::this_thread::sleep_for(std::chrono::microseconds(idle * 50));
            continue;
        }
        idle = 0;
        /* A hybrid model's recurrent state: the slots a newly admitted request was granted. */
        clear_fresh_states();
        if (const int ps = prepare_release(b); ps < 0) return stop(ps);
        for (auto& rk : ranks_) rk->ctx.set_gen_lora_slot(gen_->lora_slot());
        issue_what_.store(kIssueModel, std::memory_order_release);
        step_batch_.store(b, std::memory_order_release);
        barrier_->arrive_and_wait();      /* release the ranks */
        barrier_->arrive_and_wait();      /* wait for them */
        int st = step_status_.load(std::memory_order_acquire);
        if (st < 0) {
            RAD_ERR("generator pass %d failed: %s -- the engine cannot continue", step,
                    rad_strerror(st));
            return stop(st);
        }
        /* What the pass leaves for the host (an image, a window's samples, the frames' status),
         * copied behind it on rank 0's stream; every rank holds the same values. */
        for (const gen::IPassDriver::Readback& r : gen_->readbacks()) {
            if (st < 0 || r.bytes <= 0) continue;
            const void* src = ranks_[0]->ctx.buf_ptr(r.buf);
            if (!src) {
                RAD_ERR("the generator's output buffer '%s' was declared but never planned",
                        ranks_[0]->program.buffers[r.buf].name.c_str());
                return stop(RAD_E_STATE);
            }
            st = rad_memcpy_async(r.host, src, r.bytes, ranks_[0]->ctx.stream());
        }
        for (auto& rk : ranks_)
            if (st >= 0) st = rad_stream_sync(rk->ctx.stream());
        if (st < 0) {
            RAD_ERR("generator pass %d: the device failed: %s", step, rad_strerror(st));
            return stop(st);
        }
        if (trace_image_passes) {
            /* n_img is populated by the joint image/text denoiser layout. */
            const int64_t joint_image_rows = b->gen == RAD_GEN_PASS_DENOISE && b->gen_b
                                         ? b->gen_b->n_img : 0;
            RAD_INFO("shared image pass: %s step=%d took %.3f ms (rows=%lld joint_image_rows=%lld entries=%lld)",
                     pass_name(b->gen), step, double(rad_mono_ns() - pass_begin) / 1e6,
                     (long long)b->n_tok, (long long)joint_image_rows, (long long)b->n_seq);
        }
        dump_buffers(ranks_[0]->ctx, step, pass_name(b->gen));
        gen_->done();
        if (dispatch.owns_lock()) {
            if (const int st = shared_->leave_image(); st < 0) return stop(st);
        }
        ++step;
    }
    return RAD_OK;
}

}  /* namespace rad */
