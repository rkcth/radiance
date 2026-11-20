#pragma once
// r4d_ar_seq.h: the first step of every all-reduce and all-gather kernel (its handshake sequence
// number and scratch slot) and the wrap-safe wait after the push. Shared by the 2-rank and wide
// families: kernels sharing an instance's buffers must take the step identically or the ranks
// disagree.
//
// Counter allocation: 2 * nbw words, zeroed once (nbw = block capacity the flag array was sized
// for), then the landed counter (r4d_ar_link_landed):
//
//   [0, nbw)       Handshake counters, one per block. Block b advances word b once per launch it
//                  runs in, publishes it to the peer and waits for the peer's. Per block so a
//                  block idle through narrow launches resumes one step past its peer's last flag;
//                  the wrap-safe compare needs that gap far under 2^31.
//
//   [nbw, 2*nbw)   Parity counters. Every launch advances all nbw of them once, whatever its block
//                  count (block b takes words b, b + nb, b + 2*nb, ...), so each equals the launch
//                  count and a block reads the slot parity off its own word.
//
//   [2*nbw]        Landed counter: blocks of the running launch that have seen their peer's push
//                  land; back at zero when the launch ends.
//
// The slot cannot come from the handshake counter. Consecutive launches on one instance must use
// opposite scratch slots, since launch k+1 may push while the peer still reduces launch k. Block
// counts and per-block byte ranges vary by launch, so handshake counters drift apart per block,
// and a block of launch k+1 could push into the slot a different block of launch k is still
// reading. The parity counters move in lockstep, so the slot alternates per launch for every block.
//
// Cost: one more returning atomic, issued from a second wave so its round trip overlaps the
// handshake's, plus nbw / nb fire-and-forget atomics over the other threads. Those only need to
// land before the next launch, which the kernel boundary guarantees. All state is device-resident,
// so a captured graph replays.
#include <hip/hip_runtime.h>

// Sets `s_seq` to this block's handshake value and `s_slot` to the launch's scratch slot (0 or 1),
// then barriers so every thread can read both. `nbw` is the capacity the counters were allocated
// for, and gridDim.x must not exceed it. `steps` handshake values are taken at once (s_seq is the
// first): a kernel that handshakes more than twice per launch reserves the flag values it
// publishes, so every kernel on the instance stays in the same monotone flag space. All ranks must
// take the same number for the same block.
//
// The barrier fences LDS only, since it publishes just the two LDS words. A full __syncthreads
// would wait for the outstanding parity atomics and invalidate the vector caches, putting that
// traffic on the path to the first push.
__device__ __forceinline__ void r4d_ar_seq_take(unsigned int* __restrict__ ctrs, unsigned int nbw,
                                                unsigned int& s_seq, unsigned int& s_slot,
                                                unsigned int steps = 1u) {
  const unsigned int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  unsigned int* const par = ctrs + nbw;
  const unsigned int pt = nt > 32u ? 32u : 0u;   // first lane of the second wave, if there is one
  if (tid == 0) s_seq = atomicAdd(&ctrs[b], steps) + 1u;
  if (tid == pt) s_slot = (atomicAdd(&par[b], 1u) + 1u) & 1u;
  for (unsigned int j = b + (tid + 1u) * nb; j < nbw; j += nt * nb) atomicAdd(&par[j], 1u);
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// The engine's hold on the link, released by the launch (rad_abi.h, rad_kernel_collective
// returning 2). The word is odd exactly while the engine holds the link for this launch and
// nothing else writes it meanwhile, so each block's first thread reads it once at the start
// (r4d_ar_link_word), where the host-memory read overlaps the push instead of delaying the reduce.
//
// Every block calls r4d_ar_link_landed once its handshake has seen the peer's push land. The last
// block knows all of the peer's bytes have arrived, so the link is free for the rest of the
// kernel; it decrements the word if odd, atomically so it commutes with any barrier counting the
// same word down. `ctr` counts blocks and returns to zero at the end of the launch. The DMA engine
// polls the word past every cache, so it is accessed at system scope.
__device__ __forceinline__ long long r4d_ar_link_word(const long long* gate, unsigned int tid) {
  return gate && tid == 0 ? __hip_atomic_load(gate, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)
                          : 0;
}

__device__ __forceinline__ void r4d_ar_link_landed(long long* gate, long long word,
                                                   unsigned int* ctr, unsigned int nb,
                                                   unsigned int tid) {
  if (!gate || tid != 0) return;
  const unsigned int prev = __hip_atomic_fetch_add(ctr, 1u, __ATOMIC_RELAXED,
                                                   __HIP_MEMORY_SCOPE_AGENT);
  if (prev + 1u != nb) return;
  __hip_atomic_store(ctr, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  if (word & 1) __hip_atomic_fetch_add(gate, -1LL, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

// True while `flag` has not reached `want`. The u32 counters wrap, so compare the signed
// difference: a peer at or past `want` (possibly a launch ahead) reads as arrived across the wrap,
// where a plain `flag < want` would stop waiting once `want` wraps to a small number.
__device__ __forceinline__ bool r4d_ar_seq_behind(unsigned int flag, unsigned int want) {
  return (int)(flag - want) < 0;
}

// ---- The push check. A peer's push over PCIe P2P can land after its release flag (measured on 2x
// R9700: the reader sometimes found the slot's previous contents right after the handshake, and
// the new bytes ~1 us later), so the flag alone does not say the bytes are there. The pushed bytes
// wait in the writer's L2 for the release's writeback, and that writeback and the flag store reach
// the peer by different paths. The writer sums mix(dword, index) over what it pushes and stores the
// block's sum, salted with its handshake value, into the peer's check word for (slot, block):
// fine-grained words after the flags (r4d_ar_entry.hip sizes the allocation). The reader sums the
// same over the bytes it is about to read and reads again until the two agree. A stale element and
// a stale check word (an older handshake value) both disagree, so only the landed push passes; the
// order the two land in does not matter. The sum is order-free, so writer and reader may split the
// bytes differently.
#define R4D_ARCHK_SALT 0x6C8E9CF5u
// wall_clock64 ticks (100 MHz): 2 s. A push that never lands is a lockstep fault, so the reader traps.
#define R4D_ARCHK_DEADLINE 200000000ull
#define R4D_ARCHK_U16 0x80000000u    // index space of 16-bit fields (wire scales), apart from dwords

// MurmurHash3's finalizer over (value, index): every bit of the value reaches every bit of the term,
// so a sum of terms changes when any value does. A linear mix is not enough: (v ^ c_e) summed over
// a run of indices can lose v entirely, and a message of one repeated value (a test pattern, a
// zero-filled row) would then pass whatever the slot held.
__device__ __forceinline__ unsigned int r4d_archk_mix(unsigned int v, unsigned int e) {
  unsigned int x = v ^ (e * 0x9E3779B9u);
  x ^= x >> 16; x *= 0x85EBCA6Bu;
  x ^= x >> 13; x *= 0xC2B2AE35u;
  x ^= x >> 16;
  return x;
}
// A 16B word at dword index `e`.
__device__ __forceinline__ unsigned int r4d_archk_u4(int4 v, unsigned int e) {
  return r4d_archk_mix((unsigned int)v.x, e) + r4d_archk_mix((unsigned int)v.y, e + 1u) +
         r4d_archk_mix((unsigned int)v.z, e + 2u) + r4d_archk_mix((unsigned int)v.w, e + 3u);
}
__device__ __forceinline__ unsigned int r4d_archk_u4(uint4 v, unsigned int e) {
  return r4d_archk_mix(v.x, e) + r4d_archk_mix(v.y, e + 1u) + r4d_archk_mix(v.z, e + 2u) +
         r4d_archk_mix(v.w, e + 3u);
}
// The check word of (slot, block) in a flags allocation: two parity slots of nbw words after the
// nbw flags.
__device__ __forceinline__ unsigned int* r4d_archk_word(unsigned int* flags, unsigned int nbw,
                                                        unsigned int slot, unsigned int b) {
  return flags + nbw + slot * nbw + b;
}
__device__ __forceinline__ unsigned int r4d_archk_wave_sum(unsigned int h) {
  for (int o = 16; o > 0; o >>= 1) h += __shfl_xor(h, o, 32);
  return h;
}

// Writer: every thread of the block calls this after its pushes, before the handshake's barrier.
// Thread 0 stores the block's sum into the peer's check word.
__device__ __forceinline__ void r4d_archk_put(unsigned int h, unsigned int* peer_flags,
                                              unsigned int nbw, unsigned int slot,
                                              unsigned int b, unsigned int seq,
                                              bool release = false) {
  __shared__ unsigned int s_ckp[32];
  const unsigned int tid = threadIdx.x, nt = blockDim.x;
  h = r4d_archk_wave_sum(h);
  if ((tid & 31u) == 0u) s_ckp[tid >> 5] = h;
  __syncthreads();
  if (tid == 0) {
    unsigned int t = 0u;
    for (unsigned int w = 0; w < (nt + 31u) / 32u; ++w) t += s_ckp[w];
    /* The pushed bytes sit in this card's L2 until a system-scope release writes them back. The
     * flag's release, right after this, does that, so the word itself can be relaxed (the reader
     * re-reads until the bytes agree with it). Where the flag went out before the push (the test
     * mode, drain 4) nothing else would write them back before the kernel ends: a release. */
    unsigned int* const cw = r4d_archk_word(peer_flags, nbw, slot, b);
    if (release)
      __hip_atomic_store(cw, t + seq * R4D_ARCHK_SALT, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    else
      __hip_atomic_store(cw, t + seq * R4D_ARCHK_SALT, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

// Reader: every thread of the block calls this after the handshake; `rehash(h)` adds the thread's
// share of the bytes the block is about to read, with the same loads the kernel then makes.
template <typename F>
__device__ __forceinline__ void r4d_archk_wait(unsigned int* my_flags, unsigned int nbw,
                                               unsigned int slot, unsigned int b,
                                               unsigned int seq, const F& rehash) {
  __shared__ unsigned int s_ckw[32];
  __shared__ unsigned int s_ckok;
  const unsigned int tid = threadIdx.x, nt = blockDim.x;
  const unsigned long long t0 = tid == 0 ? wall_clock64() : 0ull;
  for (;;) {
    /* The check word first, so its round trip overlaps the re-hash's loads. Read before the bytes
     * it is compared with, it can only be older than them, and an older word just fails this try. */
    const unsigned int want = tid == 0 ? __hip_atomic_load(r4d_archk_word(my_flags, nbw, slot, b),
                                                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)
                                       : 0u;
    unsigned int h = 0u;
    rehash(h);
    h = r4d_archk_wave_sum(h);
    if ((tid & 31u) == 0u) s_ckw[tid >> 5] = h;
    __syncthreads();
    if (tid == 0) {
      unsigned int t = 0u;
      for (unsigned int w = 0; w < (nt + 31u) / 32u; ++w) t += s_ckw[w];
      const bool ok = t + seq * R4D_ARCHK_SALT == want;
      if (!ok && wall_clock64() - t0 > R4D_ARCHK_DEADLINE) __builtin_trap();
      s_ckok = ok ? 1u : 0u;
    }
    __syncthreads();
    if (s_ckok) return;
    __syncthreads();   // s_ckw and s_ckok are rewritten by the next try
    /* Drop the vector caches, so the next try reads the bytes from where the push lands. */
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    __builtin_amdgcn_s_sleep(2);
  }
}

// ---- The handshake of a kernel whose threads each own at most one word of the block's message
// (a decode or verify step: r4d_ar_hc_write_reg_kernel, r4d_ar_oneshot_2rank_exact.hip). Such a
// kernel is a chain of dependent round trips, and this is the plain kernels' protocol -- the same
// check word, the same flag value, a pass only behind the peer's flag and on an agreeing sum --
// with the chain shortened:
//
//  * Barriers order LDS only (r4d_lds_barrier). A __syncthreads also waits for every outstanding
//    load and store and invalidates the vector caches: 0.26 us against 0.03 on this part, five
//    times a launch in the plain kernels.
//  * The flag poll is a relaxed system-scope load. As an acquire load every poll is followed by a
//    system-scope cache invalidate (0.35 us, and whatever the card had cached for the kernels
//    behind goes with it), for as long as the peer is late. The slot is fine-grained memory, read
//    past the caches, and the push check re-reads behind a system-scope acquire whenever the
//    bytes do not agree, exactly as r4d_archk_wait does.
//  * `push` stores the thread's word and `rehash` loads the peer's into the caller's registers, so
//    the reduce reads nothing again.
//
// drain 4 is the plain kernels' test mode (r4d_ar_late_push_pin): the flag ahead of the push and
// the push ~20 us behind it.
#define R4D_ARW_SPIN_MAX 4000000000ULL

// A barrier that orders LDS only.
__device__ __forceinline__ void r4d_lds_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// The wake word of a 2-rank instance: the word behind the flags and the two slots' check words
// (r4d_ar_entry.hip sizes the allocation). Nothing reads it.
//
// The path between two cards idles when nothing crosses it. On 2x R9700 behind an EPYC's two root
// complexes, with each card's own link busy all the while: a store into the peer's memory more
// than ~20 us after the last one becomes visible there ~4 us later than one behind recent traffic,
// and the path is back at full speed ~2-4 us after that first store. A step's all-reduces are
// ~100 us apart, and a handshake's last store is its flag, so the whole of that wait lands on the
// flag: both ranks sit in their polls for it (5.9 us a mixer all-reduce of Qwen3.8-Flash-Next
// where neither rank was late, against 2.4 behind a store 2-20 us earlier). A block's first lane
// stores into the wake word before it waits for its turn, which is as early as the kernel knows it
// is about to push: the path wakes while the block waits, takes its counters and pushes.
__device__ __forceinline__ void r4d_ar_link_wake(unsigned int* peer_flags, unsigned int nbw) {
  if (threadIdx.x == 0)
    __hip_atomic_store(&peer_flags[3u * nbw], blockIdx.x, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

template <typename P, typename F>
__device__ __forceinline__ void r4d_ar_word_handshake(unsigned int* peer_flags,
                                                      unsigned int* my_flags, unsigned int nbw,
                                                      unsigned int slot, unsigned int b,
                                                      unsigned int seq, int drain, const P& push,
                                                      const F& rehash) {
  __shared__ unsigned int s_wput[32];
  __shared__ unsigned int s_wchk[32];
  __shared__ unsigned int s_wok;
  const unsigned int tid = threadIdx.x, nt = blockDim.x, nw = (nt + 31u) / 32u;
  const bool late = drain == 4;
  if (late) {
    if (tid == 0)
      __hip_atomic_store(&peer_flags[b], seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    const unsigned long long t0 = wall_clock64();
    while (wall_clock64() - t0 < 2000ull) __builtin_amdgcn_s_sleep(8);   // 100 MHz: 20 us
  }
  unsigned int hck = 0u;
  push(hck);
  hck = r4d_archk_wave_sum(hck);
  if ((tid & 31u) == 0u) s_wput[tid >> 5] = hck;
  /* Every wave's pushes are out before the one barrier in front of the flag (the drain modes of
   * the plain kernels; 2 is thread 0's fence below). */
  if (drain == 1) __threadfence_system();
  else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
  r4d_lds_barrier();
  if (tid == 0) {
    unsigned int t = 0u;
    for (unsigned int w = 0; w < nw; ++w) t += s_wput[w];
    unsigned int* const cw = r4d_archk_word(peer_flags, nbw, slot, b);
    if (late) {
      /* Nothing else writes the pushes back before the kernel ends: a release (r4d_archk_put). */
      __hip_atomic_store(cw, t + seq * R4D_ARCHK_SALT, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    } else {
      /* The flag's release writes the pushes back, so the check word itself can be relaxed. */
      __hip_atomic_store(cw, t + seq * R4D_ARCHK_SALT, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      if (drain == 2) __threadfence_system();
      __hip_atomic_store(&peer_flags[b], seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    unsigned long long z = 0;
    /* A handshake that cannot complete is a lockstep violation; trap rather than sum stale bytes. */
    while (r4d_ar_seq_behind(__hip_atomic_load(&my_flags[b], __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_SYSTEM), seq))
      if (++z > R4D_ARW_SPIN_MAX) __builtin_trap();
  }
  r4d_lds_barrier();
  const unsigned long long t0 = tid == 0 ? wall_clock64() : 0ull;
  for (;;) {
    /* The check word first, so its round trip overlaps the re-hash's loads (r4d_archk_wait). */
    const unsigned int want = tid == 0 ? __hip_atomic_load(r4d_archk_word(my_flags, nbw, slot, b),
                                                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)
                                       : 0u;
    unsigned int h = 0u;
    rehash(h);
    h = r4d_archk_wave_sum(h);
    if ((tid & 31u) == 0u) s_wchk[tid >> 5] = h;
    r4d_lds_barrier();
    if (tid == 0) {
      unsigned int t = 0u;
      for (unsigned int w = 0; w < nw; ++w) t += s_wchk[w];
      const bool ok = t + seq * R4D_ARCHK_SALT == want;
      if (!ok && wall_clock64() - t0 > R4D_ARCHK_DEADLINE) __builtin_trap();
      s_wok = ok ? 1u : 0u;
    }
    r4d_lds_barrier();
    if (s_wok) return;
    r4d_lds_barrier();   // s_wchk and s_wok are rewritten by the next try
    /* Drop the vector caches, so the next try reads the bytes from where the push lands. */
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    __builtin_amdgcn_s_sleep(2);
  }
}

// ---- The host mailbox's one read. A host twin's slots are pinned host memory (r4d_ar_entry.hip),
// which the card reads past every cache over its own link, so reading the slot for the check and
// again for the reduce crosses the link twice: at --p2p off a prefill all-reduce's reads doubled
// (#323). Such an instance also has a mirror of its slots in this rank's VRAM, `md` bytes from
// them (`stage` - my_scratch; 0 over P2P, where the slots are VRAM already). The re-hash reads each
// word through r4d_archk_keep, which also stores it into the mirror, and once the check passes the
// kernel moves its slot pointers onto the mirror (r4d_archk_moved) for the reduce. Only this block
// writes and reads its words of the mirror, and the check's barriers order the two; a failed try's
// copy is overwritten by the next.
template <typename V>
__device__ __forceinline__ V r4d_archk_keep(const V* p, long md) {
  const V v = *p;
  if (md) *reinterpret_cast<V*>(reinterpret_cast<uintptr_t>(p) + md) = v;
  return v;
}
template <typename V>
__device__ __forceinline__ const V* r4d_archk_moved(const V* p, long md) {
  return reinterpret_cast<const V*>(reinterpret_cast<uintptr_t>(p) + md);
}
// The offset of an instance's mirror from its slots, from the launcher's `stage` (0: none).
__device__ __forceinline__ long r4d_archk_md(long stage, const void* my_scratch) {
  return stage ? stage - (long)reinterpret_cast<uintptr_t>(my_scratch) : 0;
}
