/* Derived from libr4d/r4d_chain.h by the libhalo rename transform (2026-10-26);
 * see libhalo/README.md. Op schemas, row names and stored weight layouts are
 * shared with libr4d by design. Everything else in this file is libhalo's own
 * and diverges as it is measured on gfx1151. */
// halo_chain.h -- a chained launch's wait and completion (RadKernelChainFn, abi/rad_abi.h).
//
// Between two dependent kernels on one queue the packet processor waits for the first to finish
// before it launches the second (the AQL barrier bit): on 4x R9700, ~1.3 ms of a 14 ms DFlash2 step
// on the 31B. A chained launch drops the bit. The engine writes a record into the
// kernel's argument slot, and the kernel waits for its queue's completion counter to reach the
// blocks issued before it, then adds its own blocks as they finish. A kernel's chained form
// (`CHAIN` true) brackets its body with halo_chain_wait / halo_chain_done; the plain form never reads
// the record, so a launch from outside the engine (halo_selftest, rad-kbench) is untouched.
//
// Every block must reach halo_chain_done exactly once, so a kernel whose blocks return early keeps
// its body in a __device__ function called from the chained wrapper.
//
// The wait's acquire has to cover every read behind it, and an agent-scope fence does not: it
// invalidates the vector caches, while a load whose address every lane shares is a scalar load,
// served from the scalar data cache (a GEMM's row scales, the routed kernels' counts and offsets).
// What that cache holds is what a launch read from the same arena address a layer earlier. The
// packet processor's acquire empties it when it launches the kernel, which is before the kernels
// in front have written; halo_chain_wait therefore empties it behind the fence itself
// (halo_chain_acquire), as libr4d's wait does, where the stale read was measured (gfx1201).
#pragma once
#include <hip/hip_runtime.h>
#include "rad_abi.h"

/* A kernel template's chain parameter, for a kernel that brackets its own body: `r4dchain`
 * instantiates the chained form and puts the name the backend knows a chained kernel by into the
 * instantiation's symbol (rad_abi.h: the engine gives a record only to a dispatch whose kernel name
 * holds "r4dchain", so the tag keeps libr4d's spelling), HaloUnchained the plain form. */
struct r4dchain      { static constexpr bool on = true; };
struct HaloUnchained { static constexpr bool on = false; };

/* Host side: whether the launch being made should take its chained form. A row's entry point sets
 * it from its `chain` parameter for the extent of its (synchronous) launches, so an emit helper
 * deep inside the entry reads it without the flag threaded through every signature. */
inline bool& halo_chain_flag() {
    static thread_local bool on = false;
    return on;
}
/* The chained forms are for decode heights, as libr4d's: in a prefill step a chained kernel's blocks
 * wait resident on the CUs while the long kernel before them runs, which costs more than the launch
 * gap saved (on a Radeon 8060S, a Flash-Next prefill 3% slower with them). True where the row
 * resolved with `chain` and the launch's first operand has at most `rows_max` rows. */
inline bool halo_chain_small(const RadArgs* a, long long rows_max = 64) {
    long long on = 0;
    if (!a || !rad_args_geti(a, "chain", &on) || !on || a->n_t < 1 || !a->t) return false;
    return a->t[0].rank >= 1 && a->t[0].shape[0] <= rows_max;
}
struct HaloChainScope {
    bool prev;
    explicit HaloChainScope(bool on) : prev(halo_chain_flag()) { halo_chain_flag() = on; }
    ~HaloChainScope() { halo_chain_flag() = prev; }
    HaloChainScope(const HaloChainScope&) = delete;
    HaloChainScope& operator=(const HaloChainScope&) = delete;
};

#if !defined(__HIP__)
/* The host SIMT emulator (tests/halo_emu) compiles some of these sources as C++: no engine record
 * ever reaches a launch there, so a chained form is its plain form. */
__device__ __forceinline__ void halo_chain_wait() {}
__device__ __forceinline__ void halo_chain_done() {}
#else

struct HaloChainRec {
    unsigned long long  magic, target;
    unsigned long long* ctr;
    unsigned long long  check;
};

// The record, or null where it is not the engine's. Read from the argument segment (scalar loads)
// at both ends of the kernel, so nothing is held across the body: a register-heavy body (the
// split-KV decode attention) would otherwise pay for the pointer.
__device__ __forceinline__ const HaloChainRec* halo_chain_rec() {
    const HaloChainRec* r = (const HaloChainRec*)((const char*)__builtin_amdgcn_kernarg_segment_ptr() +
                                                RAD_CHAIN_KARG_OFF);
    const unsigned long long magic = r->magic, target = r->target, check = r->check;
    const unsigned long long* ctr = r->ctr;
    if (magic != RAD_CHAIN_MAGIC ||
        check != (magic ^ target ^ (unsigned long long)(uintptr_t)ctr ^ RAD_CHAIN_CHECK))
        return nullptr;
    return r;
}

// Before the first read of an operand: one lane waits for the counter to reach the target.
// Every resident block polls the same word, so a wide kernel launched behind a long one (a GEMM
// behind a junction's all-reduce) would put hundreds of loads into one L2 channel every few dozen
// clocks for as long as the wait lasts, slowing whatever else that channel serves. The first
// HALO_CHAIN_FAST polls are ~64 clocks apart, the rest HALO_CHAIN_SLOW x 64: a long wait ends at most
// that much late and puts 1/HALO_CHAIN_SLOW of the load on the channel.
#ifndef HALO_CHAIN_FAST
#define HALO_CHAIN_FAST 8
#endif
#ifndef HALO_CHAIN_SLOW
#define HALO_CHAIN_SLOW 16
#endif
// The acquire behind a chained wait: the vector caches by the fence, the scalar data cache by
// hand, which the fence leaves (see the head of this file). Every wave, for the compute unit it
// runs on.
__device__ __forceinline__ void halo_chain_acquire() {
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    asm volatile("s_dcache_inv" ::: "memory");
}

__device__ __forceinline__ void halo_chain_wait() {
    const HaloChainRec* r = halo_chain_rec();
    if (!r) return;
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        const unsigned long long target = r->target;
        unsigned long long* ctr = r->ctr;
        int n = 0;
        while (__hip_atomic_load(ctr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < target) {
            if (n < HALO_CHAIN_FAST) {
                __builtin_amdgcn_s_sleep(1);
                ++n;
            } else {
                __builtin_amdgcn_s_sleep(HALO_CHAIN_SLOW);
            }
        }
    }
    __syncthreads();
    halo_chain_acquire();
}

// After the block's last write: release, then one add a block.
__device__ __forceinline__ void halo_chain_done() {
    const HaloChainRec* r = halo_chain_rec();
    if (!r) return;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
        __hip_atomic_fetch_add(r->ctr, 1ull, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}
#endif  /* __HIP__ */
