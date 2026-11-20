// r4d_ar_oneshot_2rank_exact.hip: r4d_ar_oneshot_2rank_exact -- multi-block one-shot P2P-BAR
// all-reduce, exactly two ranks, for dual R9700 (gfx1201/RDNA4, PCIe). One-shot: each rank pushes
// its whole input, so scratch is sized by the full message and there is no ring or reduce-scatter
// stage. Two ranks is a hard constraint: the handshake is one peer flag per block. Pure HIP behind
// the C ABI in r4d.h; the caller passes raw pointers, element count, dtype and stream.
//
// Design:
//  * Push: each rank writes its input into the peer's IPC scratch, then reduces local input +
//    peer data (now in its own scratch) into out. Only scratch and flags are IPC-shared; input and
//    output stay local, so graph capture needs no buffer registration.
//  * Multi-block with per-chunk flags and no grid barrier: block b owns a contiguous chunk of 16B
//    words, pushes it, sets peer_flags[b], spins on my_flags[b] and reduces its chunk. This
//    spreads the push (PCIe) and the reduce (memory) across CUs.
//  * Graph-safe: seq is a device-resident per-block counter the kernel increments. Both ranks run
//    the same launch sequence (SPMD), so seq_ctr[b] stays in lockstep across ranks.
//  * Reuse: scratch is double-buffered by a per-launch parity (r4d_ar_seq.h explains why not
//    seq&1); same-stream serialisation plus the handshake drain a slot before reuse.
//  * Fine-grained memory (hipDeviceMallocFinegrained) is uncached and posted straight to the
//    fabric, so system-scope release/acquire skip the L2 writeback. The drain/acq fences are
//    selectable (default drain=1/acq=0 on fine-grained).
#include "r4d.h"
#include "r4d_chain.h"
#include "r4d_ar_seq.h"
#include "r4d_ar_hc.h"      // the gather word and the hc write word, shared with the rotated wire
#include "r4d_ar_arc.h"     // the lossless chunk codec of the coded pushes
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#define R4D_AR_SPIN_MAX 4000000000ULL
#define R4D_AR_MAX_BLOCKS 512
#define HIPOK(e) do { hipError_t _e = (e); if (_e != hipSuccess) \
    throw std::runtime_error(std::string("hip: ") + hipGetErrorString(_e)); } while (0)

__device__ __forceinline__ void store_sys_rel(unsigned int* p, unsigned int v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned int load_sys_acq(const unsigned int* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ float rd_to_f(const float& x) { return x; }
__device__ __forceinline__ float rd_to_f(const __half& x) { return __half2float(x); }
__device__ __forceinline__ float rd_to_f(const __hip_bfloat16& x) { return (float)x; }
__device__ __forceinline__ void rd_from_f(float v, float& o) { o = v; }
__device__ __forceinline__ void rd_from_f(float v, __half& o) { o = __float2half(v); }
__device__ __forceinline__ void rd_from_f(float v, __hip_bfloat16& o) { o = (__hip_bfloat16)v; }

// Vectorised 128-bit add: an int4 packs LANES elements of T, summed in fp32.
template <typename T, int LANES>
__device__ __forceinline__ int4 vadd(const int4& a, const int4& b) {
  int4 r;
  const T* pa = reinterpret_cast<const T*>(&a);
  const T* pb = reinterpret_cast<const T*>(&b);
  T* pr = reinterpret_cast<T*>(&r);
#pragma unroll
  for (int j = 0; j < LANES; ++j) rd_from_f(rd_to_f(pa[j]) + rd_to_f(pb[j]), pr[j]);
  return r;
}

// Multi-block one-shot all-reduce. Each block owns words [start,end) of the 16B-word space and
// drives its own flag handshake; no cross-block sync.
// CH `r4dchain`: the chained form (r4d_chain.h); no block returns early.
template <typename T, int LANES, class CH>
__global__ void r4d_ar_oneshot_2rank_exact_kernel(
    const int4* __restrict__ in4, int4* peer_base, const int4* my_base, int4* __restrict__ out4,
    unsigned int* peer_flags, unsigned int* my_flags, unsigned int* seq_ctrs,
    int n16, int slot_stride16, int drain, int acq, unsigned int nbw, long stage) {
  if (CH::on) r4d_chain_wait();
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  r4d_ar_seq_take(seq_ctrs, nbw, s_seq, s_slot);   // device-resident, replay-safe
  const unsigned int s = s_seq;
  const int slot = (int)s_slot;
  int4* ps = peer_base + (size_t)slot * slot_stride16;
  const int4* ms = my_base + (size_t)slot * slot_stride16;
  const long md = r4d_archk_md(stage, my_base);   // the host mailbox's VRAM mirror (r4d_ar_seq.h)

  const int w = (n16 + nb - 1) / nb;      // 16B words per block
  const int start = b * w;
  int end = start + w; if (end > n16) end = n16;
  unsigned int hck = 0u;   // the push check (r4d_ar_seq.h)

  // drain 4, tests only (r4d_ar_late_push_pin, r4d_selftest): the flag goes out before the push
  // and the push ~20 us after it, so the peer sees a flag whose bytes have not landed -- what a
  // push over PCIe P2P now and then does on its own -- on every launch.
  if (drain == 4) {
    if (tid == 0) store_sys_rel(&peer_flags[b], s);
    const unsigned long long t0 = wall_clock64();
    while (wall_clock64() - t0 < 2000ull) __builtin_amdgcn_s_sleep(8);   // 100 MHz: 20 us
  }
  // Push the chunk to the peer slot, 4x unrolled for memory-level parallelism.
  {
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
      int4 v0 = in4[k], v1 = in4[k + nt], v2 = in4[k + 2 * nt], v3 = in4[k + 3 * nt];
      ps[k] = v0; ps[k + nt] = v1; ps[k + 2 * nt] = v2; ps[k + 3 * nt] = v3;
      hck += r4d_archk_u4(v0, 4u * (unsigned)k) + r4d_archk_u4(v1, 4u * (unsigned)(k + nt)) +
             r4d_archk_u4(v2, 4u * (unsigned)(k + 2 * nt)) + r4d_archk_u4(v3, 4u * (unsigned)(k + 3 * nt));
    }
    for (; k < end; k += nt) { const int4 v = in4[k]; ps[k] = v; hck += r4d_archk_u4(v, 4u * (unsigned)k); }
  }
  // Drain modes, ordering the pushes before the release flag so the peer sees data first:
  //  0 = none (relies on PCIe posted-write ordering and the release flag)
  //  1 = all-thread __threadfence_system: safe, but an L2 writeback-invalidate per wave, wasted
  //      on fine-grained memory and costly at high nt*nb
  //  2 = single-thread __threadfence_system: cheap, but drains only wave 0's stores
  //  3 = all-thread s_wait_storecnt 0: every wave drains its posted writes with no L2
  //      writeback-invalidate; PCIe keeps posted writes to one peer ordered, so the release flag
  //      lands after the data. Correct only for fine-grained memory (coarse leaves data in L2).
  r4d_archk_put(hck, peer_flags, nbw, (unsigned)slot, (unsigned)b, s, drain == 4);
  if (drain == 1) __threadfence_system();
  else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
  __syncthreads();
  if (tid == 0) {
    if (drain == 2) __threadfence_system();
    if (drain != 4) store_sys_rel(&peer_flags[b], s);
    unsigned long long z = 0;
    // A handshake that cannot complete is a lockstep violation; trap rather than sum stale bytes.
    while (r4d_ar_seq_behind(load_sys_acq(&my_flags[b]), s)) {
      if (++z > R4D_AR_SPIN_MAX) __builtin_trap();
    }
  }
  __syncthreads();
  if (acq) __threadfence_system();
  r4d_archk_wait(my_flags, nbw, (unsigned)slot, (unsigned)b, s, [&](unsigned int& h) {
    for (int k = start + tid; k < end; k += nt)
      h += r4d_archk_u4(r4d_archk_keep(&ms[k], md), 4u * (unsigned)k);
  });
  ms = r4d_archk_moved(ms, md);
  // Reduce the chunk, 4x unrolled: out = local input + peer input (staged in local scratch).
  {
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
      out4[k]          = vadd<T, LANES>(in4[k],          ms[k]);
      out4[k + nt]     = vadd<T, LANES>(in4[k + nt],     ms[k + nt]);
      out4[k + 2 * nt] = vadd<T, LANES>(in4[k + 2 * nt], ms[k + 2 * nt]);
      out4[k + 3 * nt] = vadd<T, LANES>(in4[k + 3 * nt], ms[k + 3 * nt]);
    }
    for (; k < end; k += nt) out4[k] = vadd<T, LANES>(in4[k], ms[k]);
  }
  if (CH::on) r4d_chain_done();
}

// The same all-reduce with the gated residual write as its epilogue. `y` is the message: each word
// is reduced as above and stored back, then the hc streams take
// h[r][c*n + i] += inj[r][c] * y[r][i] in hc_write's arithmetic (r4d_hc_bf16.hip) over the stored
// bf16 value. The output matches the two-kernel pair byte for byte without the second launch (a
// dependent launch gap is about 3 us on this part, against a 1-2 us write).
//
// n is a multiple of eight, so a word never straddles a row and the block that reduces it owns
// every stream element behind it.
// CH `r4dchain`: the chained form (r4d_chain.h); no block returns early.
template <class CH>
__global__ void r4d_ar_hc_write_kernel(
    int4* __restrict__ y4, int4* peer_base, const int4* my_base, unsigned int* peer_flags,
    unsigned int* my_flags, unsigned int* seq_ctrs, const unsigned short* __restrict__ inj,
    unsigned short* __restrict__ h, int n16, int n, int hc, long irow, long hrow,
    int slot_stride16, int drain, int acq, unsigned int nbw, long long* gate,
    unsigned int* gate_ctr, long stage) {
  if (CH::on) r4d_chain_wait();
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  const long long gate_word = r4d_ar_link_word(gate, (unsigned int)tid);
  r4d_ar_seq_take(seq_ctrs, nbw, s_seq, s_slot);   // device-resident, replay-safe
  const unsigned int s = s_seq;
  const int slot = (int)s_slot;
  int4* ps = peer_base + (size_t)slot * slot_stride16;
  const int4* ms = my_base + (size_t)slot * slot_stride16;
  const long md = r4d_archk_md(stage, my_base);   // the host mailbox's VRAM mirror (r4d_ar_seq.h)

  const int w = (n16 + nb - 1) / nb;      // 16B words per block
  const int start = b * w;
  int end = start + w; if (end > n16) end = n16;
  unsigned int hck = 0u;   // the push check (r4d_ar_seq.h)

  {
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
      int4 v0 = y4[k], v1 = y4[k + nt], v2 = y4[k + 2 * nt], v3 = y4[k + 3 * nt];
      ps[k] = v0; ps[k + nt] = v1; ps[k + 2 * nt] = v2; ps[k + 3 * nt] = v3;
      hck += r4d_archk_u4(v0, 4u * (unsigned)k) + r4d_archk_u4(v1, 4u * (unsigned)(k + nt)) +
             r4d_archk_u4(v2, 4u * (unsigned)(k + 2 * nt)) + r4d_archk_u4(v3, 4u * (unsigned)(k + 3 * nt));
    }
    for (; k < end; k += nt) { const int4 v = y4[k]; ps[k] = v; hck += r4d_archk_u4(v, 4u * (unsigned)k); }
  }
  r4d_archk_put(hck, peer_flags, nbw, (unsigned)slot, (unsigned)b, s);
  if (drain == 1) __threadfence_system();
  else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
  __syncthreads();
  if (tid == 0) {
    if (drain == 2) __threadfence_system();
    store_sys_rel(&peer_flags[b], s);
    unsigned long long z = 0;
    while (r4d_ar_seq_behind(load_sys_acq(&my_flags[b]), s)) {
      if (++z > R4D_AR_SPIN_MAX) __builtin_trap();
    }
  }
  __syncthreads();
  if (acq) __threadfence_system();
  r4d_archk_wait(my_flags, nbw, (unsigned)slot, (unsigned)b, s, [&](unsigned int& h) {
    for (int k = start + tid; k < end; k += nt)
      h += r4d_archk_u4(r4d_archk_keep(&ms[k], md), 4u * (unsigned)k);
  });
  ms = r4d_archk_moved(ms, md);
  r4d_ar_link_landed(gate, gate_word, gate_ctr, (unsigned int)nb, (unsigned int)tid);

  const int wpr = n >> 3;                 // 16B words a row
  for (int k = start + tid; k < end; k += nt) {
    const int4 r = vadd<__hip_bfloat16, 8>(y4[k], ms[k]);
    y4[k] = r;
    const int row = k / wpr, i0 = (k - row * wpr) * 8;
    r4d_ar_hc_write_word(inj, h, row, i0, n, hc, irow, hrow,
                         reinterpret_cast<const unsigned short*>(&r));
  }
  if (CH::on) r4d_chain_done();
}

// The same write with the MoE gather as its prologue. The routed experts' rows go back to token
// order, y[t][i] = sum over t's slots, ascending, of w[t][j] * y_expert[slot][i] rounded to bf16,
// with the shared arm folded in as moe_gather does; the word is pushed to the peer and kept as this
// rank's message. The gather uses moe_gather's round-to-nearest-even narrows (r4d_common.h), the hc
// write keeps hc_write's, so the bytes match the three separate kernels with two fewer launches.
//
// A block's words sit in at most R4D_ARG_ROWS rows (the launcher sizes the grid for it), and their
// slots are found once, in LDS, in one pass over the sorted list.
//
// That table does not suit a prefill chunk: eight rows per block means hundreds of blocks at 2048
// rows, each scanning the whole sorted list and paying its own flag round trip, where a few dozen
// already saturate the fabric. With INV the slots come from an inverse table instead
// (inv[token * top_k + slot] = sorted row holding it, or -1), built by r4d_ar_gather_inv_kernel
// beforehand, and the block count stays the all-reduce's. The gather's reads then overlap the push
// of words already summed. The arithmetic is identical to the table form.

/* inv[f] = i for every sorted row i holding slot f, -1 for a slot no row holds. One block, so the
 * clear and the scatter are separated by a barrier. */
__global__ void r4d_ar_gather_inv_kernel(int* __restrict__ inv, const int* __restrict__ sorted,
                                         int T) {
  for (int x = threadIdx.x; x < T; x += blockDim.x) inv[x] = -1;
  __syncthreads();
  for (int i = threadIdx.x; i < T; i += blockDim.x) {
    const int f = sorted[i];
    if (f >= 0 && f < T) inv[f] = i;
  }
}

// CH `r4dchain`: the chained form (r4d_chain.h); no block returns early.
template <bool INV, class CH>
__global__ void r4d_ar_gather_hc_write_kernel(
    int4* __restrict__ y4, int4* peer_base, const int4* my_base, unsigned int* peer_flags,
    unsigned int* my_flags, unsigned int* seq_ctrs, const unsigned short* __restrict__ inj,
    unsigned short* __restrict__ h, int n16, int n, int hc, long irow, long hrow,
    int slot_stride16, int drain, int acq, unsigned int nbw, R4dArGather G,
    const int* __restrict__ inv, long long* gate, unsigned int* gate_ctr, long stage) {
  if (CH::on) r4d_chain_wait();
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  __shared__ int   s_i[R4D_ARG_ROWS * R4D_ARG_SLOTS];
  __shared__ float s_w[R4D_ARG_ROWS * R4D_ARG_SLOTS];
  __shared__ float s_g[R4D_ARG_ROWS];
  const long long gate_word = r4d_ar_link_word(gate, (unsigned int)tid);
  r4d_ar_seq_take(seq_ctrs, nbw, s_seq, s_slot);   // device-resident, replay-safe
  const unsigned int s = s_seq;
  const int slot = (int)s_slot;
  int4* ps = peer_base + (size_t)slot * slot_stride16;
  const int4* ms = my_base + (size_t)slot * slot_stride16;
  const long md = r4d_archk_md(stage, my_base);   // the host mailbox's VRAM mirror (r4d_ar_seq.h)

  const int w = (n16 + nb - 1) / nb;      // 16B words per block
  const int start = b * w;
  int end = start + w; if (end > n16) end = n16;
  unsigned int hck = 0u;   // the push check (r4d_ar_seq.h)
  const int wpr = n >> 3;                 // 16B words a row
  const int row_lo = start / wpr;
  const int nrow = end > start ? (end - 1) / wpr - row_lo + 1 : 0;
  const int K = G.top_k;

  /* Slot table. A slot no sorted row claims reads -1, so clear and scan are separated by a
   * barrier, as in moe_gather. */
  if (!INV) {
    for (int x = tid; x < nrow * K; x += nt) s_i[x] = -1;
    __syncthreads();
    {
      const int lo = row_lo * K, hi = (row_lo + nrow) * K;
      for (int i = tid; i < G.T; i += nt) {
        const int f = G.sorted[i];
        if (f >= lo && f < hi) s_i[f - lo] = i;
      }
    }
    for (int x = tid; x < nrow * K; x += nt)
      s_w[x] = bf2f(G.ew[(long)(row_lo + x / K) * G.ew_ld + x % K]);
    if (G.sh)
      for (int r = tid; r < nrow; r += nt) {
        const float v0 = bf2f(G.sg[(long)(row_lo + r) * G.sg_ld]);
        s_g[r] = G.sig ? (1.0f / (1.0f + __expf(-v0))) : v0;
      }
    __syncthreads();
  }

  /* Gather one word per thread per pass; push it and keep it. */
  for (int k = start + tid; k < end; k += nt) {
    const int row = k / wpr, i0 = (k - row * wpr) * 8;
    const int4 out = r4d_ar_gather_word<INV>(G, inv, s_i, s_w, s_g, row, row - row_lo, i0);
    y4[k] = out;
    ps[k] = out;
    hck += r4d_archk_u4(out, 4u * (unsigned)k);
  }
  r4d_archk_put(hck, peer_flags, nbw, (unsigned)slot, (unsigned)b, s);
  if (drain == 1) __threadfence_system();
  else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
  __syncthreads();
  if (tid == 0) {
    if (drain == 2) __threadfence_system();
    store_sys_rel(&peer_flags[b], s);
    unsigned long long z = 0;
    while (r4d_ar_seq_behind(load_sys_acq(&my_flags[b]), s)) {
      if (++z > R4D_AR_SPIN_MAX) __builtin_trap();
    }
  }
  __syncthreads();
  if (acq) __threadfence_system();
  r4d_archk_wait(my_flags, nbw, (unsigned)slot, (unsigned)b, s, [&](unsigned int& h) {
    for (int k = start + tid; k < end; k += nt)
      h += r4d_archk_u4(r4d_archk_keep(&ms[k], md), 4u * (unsigned)k);
  });
  ms = r4d_archk_moved(ms, md);
  r4d_ar_link_landed(gate, gate_word, gate_ctr, (unsigned int)nb, (unsigned int)tid);

  for (int k = start + tid; k < end; k += nt) {
    const int4 r = vadd<__hip_bfloat16, 8>(y4[k], ms[k]);
    y4[k] = r;
    const int row = k / wpr, i0 = (k - row * wpr) * 8;
    r4d_ar_hc_write_word(inj, h, row, i0, n, hc, irow, hrow,
                         reinterpret_cast<const unsigned short*>(&r));
  }
  if (CH::on) r4d_chain_done();
}

// ---- The register form of the two fused writes: every thread owns at most one word of its block
// (the launchers' rule: a decode or verify step's message, 320 words a block at four rows of 2560).
// The kernel is then a chain of dependent round trips, not a stream of bytes, and this form
// shortens the chain. The arithmetic, the wire, the flag values and the counters are the plain
// kernels', so the bytes are too, and the two forms follow one another on one instance.
//
//  * The path to the peer is woken before the block waits for its turn (r4d_ar_link_wake).
//  * The word of `y`, the row's gains and the word's stream words are loaded before the first
//    atomic returns and stay in registers: the push does not wait for the slot parity and then a
//    load, and nothing is read again behind the handshake.
//  * The handshake is r4d_ar_word_handshake (r4d_ar_seq.h): LDS-only barriers, a relaxed poll.
//
// On 2x R9700, Qwen3.8-Flash-Next at tp 2 (96 of these a verify pass, chain 3 with coupling):
// -2.7% a step against the plain kernels, the wake about half of it. P2P instances only: a host
// mailbox keeps the plain kernels and their mirror. `acq` has no meaning here (nothing is read
// behind the check).
template <class CH>
__global__ void r4d_ar_hc_write_reg_kernel(
    int4* __restrict__ y4, int4* peer_base, const int4* my_base, unsigned int* peer_flags,
    unsigned int* my_flags, unsigned int* seq_ctrs, const unsigned short* __restrict__ inj,
    unsigned short* __restrict__ h, int n16, int n, int hc, long irow, long hrow,
    int slot_stride16, int drain, unsigned int nbw, long long* gate, unsigned int* gate_ctr) {
  r4d_ar_link_wake(peer_flags, nbw);
  if (CH::on) r4d_chain_wait();
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x;
  __shared__ unsigned int s_seq, s_slot;
  const long long gate_word = r4d_ar_link_word(gate, (unsigned int)tid);
  const int w = (n16 + nb - 1) / nb;      // 16B words per block, at most the block's threads
  const int start = b * w;
  int end = start + w; if (end > n16) end = n16;
  const int k = start + tid;
  const bool live = k < end;
  const int kk = live ? k : 0;            // a thread without a word reads word 0 and drops it
  const int wpr = n >> 3;                 // 16B words a row
  const int row = kk / wpr, i0 = (kk - row * wpr) * 8;

  /* Everything that does not depend on the peer, issued before the first atomic returns. */
  const int4 v = y4[kk];
  R4dArHcPre pre;
  r4d_ar_hc_write_load(inj, h, row, i0, n, hc, irow, hrow, pre);

  r4d_ar_seq_take(seq_ctrs, nbw, s_seq, s_slot);   // device-resident, replay-safe
  const unsigned int s = s_seq;
  const int slot = (int)s_slot;
  int4* ps = peer_base + (size_t)slot * slot_stride16;
  const int4* ms = my_base + (size_t)slot * slot_stride16;

  int4 m = make_int4(0, 0, 0, 0);
  r4d_ar_word_handshake(peer_flags, my_flags, nbw, (unsigned)slot, (unsigned)b, s, drain,
      [&](unsigned int& hh) {
        if (live) { ps[k] = v; hh = r4d_archk_u4(v, 4u * (unsigned)k); }
      },
      [&](unsigned int& hh) {
        if (live) { m = ms[k]; hh += r4d_archk_u4(m, 4u * (unsigned)k); }
      });
  r4d_ar_link_landed(gate, gate_word, gate_ctr, (unsigned int)nb, (unsigned int)tid);

  if (live) {
    const int4 r = vadd<__hip_bfloat16, 8>(v, m);
    y4[k] = r;
    r4d_ar_hc_write_apply(h, row, i0, n, hc, hrow, reinterpret_cast<const unsigned short*>(&r),
                          pre);
  }
  if (CH::on) r4d_chain_done();
}

// The gather's write in the register form: the LDS slot table as r4d_ar_gather_hc_write_kernel
// builds it, then one word a thread, gathered, pushed and kept. Decode table only (no inverse).
template <class CH>
__global__ void r4d_ar_gather_hc_write_reg_kernel(
    int4* __restrict__ y4, int4* peer_base, const int4* my_base, unsigned int* peer_flags,
    unsigned int* my_flags, unsigned int* seq_ctrs, const unsigned short* __restrict__ inj,
    unsigned short* __restrict__ h, int n16, int n, int hc, long irow, long hrow,
    int slot_stride16, int drain, unsigned int nbw, R4dArGather G, long long* gate,
    unsigned int* gate_ctr) {
  r4d_ar_link_wake(peer_flags, nbw);
  if (CH::on) r4d_chain_wait();
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  __shared__ int   s_i[R4D_ARG_ROWS * R4D_ARG_SLOTS];
  __shared__ float s_w[R4D_ARG_ROWS * R4D_ARG_SLOTS];
  __shared__ float s_g[R4D_ARG_ROWS];
  const long long gate_word = r4d_ar_link_word(gate, (unsigned int)tid);
  const int w = (n16 + nb - 1) / nb;      // 16B words per block, at most the block's threads
  const int start = b * w;
  int end = start + w; if (end > n16) end = n16;
  const int wpr = n >> 3;                 // 16B words a row
  const int row_lo = start / wpr;
  const int nrow = end > start ? (end - 1) / wpr - row_lo + 1 : 0;
  const int K = G.top_k;
  const int k = start + tid;
  const bool live = k < end;
  const int kk = live ? k : (end > start ? start : 0);   // a thread without a word drops its loads
  const int row = kk / wpr, i0 = (kk - row * wpr) * 8;

  R4dArHcPre pre;
  r4d_ar_hc_write_load(inj, h, row, i0, n, hc, irow, hrow, pre);

  r4d_ar_seq_take(seq_ctrs, nbw, s_seq, s_slot);   // device-resident, replay-safe
  const unsigned int s = s_seq;
  const int slot = (int)s_slot;
  int4* ps = peer_base + (size_t)slot * slot_stride16;
  const int4* ms = my_base + (size_t)slot * slot_stride16;

  /* Slot table, as r4d_ar_gather_hc_write_kernel's. */
  for (int x = tid; x < nrow * K; x += nt) s_i[x] = -1;
  __syncthreads();
  {
    const int lo = row_lo * K, hi = (row_lo + nrow) * K;
    for (int i = tid; i < G.T; i += nt) {
      const int f = G.sorted[i];
      if (f >= lo && f < hi) s_i[f - lo] = i;
    }
  }
  for (int x = tid; x < nrow * K; x += nt)
    s_w[x] = bf2f(G.ew[(long)(row_lo + x / K) * G.ew_ld + x % K]);
  if (G.sh)
    for (int r = tid; r < nrow; r += nt) {
      const float v0 = bf2f(G.sg[(long)(row_lo + r) * G.sg_ld]);
      s_g[r] = G.sig ? (1.0f / (1.0f + __expf(-v0))) : v0;
    }
  __syncthreads();

  int4 v = make_int4(0, 0, 0, 0);
  if (live) v = r4d_ar_gather_word<false>(G, nullptr, s_i, s_w, s_g, row, row - row_lo, i0);
  int4 m = make_int4(0, 0, 0, 0);
  r4d_ar_word_handshake(peer_flags, my_flags, nbw, (unsigned)slot, (unsigned)b, s, drain,
      [&](unsigned int& hh) {
        if (live) { ps[k] = v; hh = r4d_archk_u4(v, 4u * (unsigned)k); }
      },
      [&](unsigned int& hh) {
        if (live) { m = ms[k]; hh += r4d_archk_u4(m, 4u * (unsigned)k); }
      });
  r4d_ar_link_landed(gate, gate_word, gate_ctr, (unsigned int)nb, (unsigned int)tid);

  if (live) {
    const int4 r = vadd<__hip_bfloat16, 8>(v, m);
    y4[k] = r;
    r4d_ar_hc_write_apply(h, row, i0, n, hc, hrow, reinterpret_cast<const unsigned short*>(&r),
                          pre);
  }
  if (CH::on) r4d_chain_done();
}

// ---- The coded form of the two fused writes: the push carries ~12.1 bits a bf16 value instead of
// 16, losslessly (the four-rank coded two-shot's chunk codec, r4d_ar_arc.h), for the messages whose
// time is the link's: a prefill chunk (2048 rows of 2560 are 10.5 MB each way a reduction) and a
// many-stream decode or verify step. The peer decodes the same bf16 words it would have been sent
// raw and folds them in the same arithmetic, so y and h are the plain kernels' byte for byte.
//
// A slot holds one coded message of n16 chunks (16-byte words), n16 a multiple of 32:
//
//   packed  u32[3 * n16]   at 0            three dwords a chunk
//   bases   u32[n16 / 32]  at 12 * n16     a base-exponent byte for each group of 8 chunks
//   spill   u32[n16]       behind, 16-byte aligned: the fourth dword of a raw group's chunks
//
// A wave codes 32 consecutive chunks and redistributes the packed dwords among its lanes, so that
// each of its three stores is 128 contiguous bytes and the P2P writes stay coalesced, as in the
// four-rank form. A group that does not code (a subnormal, an inf or NaN, an exponent range past
// 14: base byte 0xFF) travels raw in the same place: the first three dwords of each of its chunks
// in the chunk's packed cell and the fourth in the spill, so a slot is 16.125 bytes a chunk
// whatever the data (r4d_ar_entry.hip sizes a two-rank bf16 slot for it) and a message of raw
// groups only costs a spill store a chunk.
//
// The push check (r4d_ar_seq.h) runs over the dwords as they cross the link: the writer sums
// mix(dword, position) over what it stores, the reader over what it is about to decode, each
// region in its own index space. A stale base byte makes the reader sum a spill dword the writer
// did not store, or miss one it did, and the sums disagree as for any other stale dword.
//
// P2P instances only, like the register form: a host mailbox keeps the plain kernels and their
// mirror.
struct R4dArc2 { unsigned* packed; unsigned* bases; unsigned* spill; };
__host__ __device__ constexpr size_t r4d_arc2_spill_off(long n16) {
  return ((size_t)12 * n16 + (size_t)n16 / 8 + 15) & ~(size_t)15;
}
// Bytes of a coded message of n16 chunks, n16 a multiple of 32.
__host__ __device__ constexpr size_t r4d_arc2_bytes(long n16) {
  return r4d_arc2_spill_off(n16) + (size_t)4 * n16;
}
__device__ __forceinline__ R4dArc2 r4d_arc2_at(const void* p, int n16) {
  char* b = (char*)p;
  return R4dArc2{(unsigned*)b, (unsigned*)(b + (size_t)12 * n16),
                 (unsigned*)(b + r4d_arc2_spill_off(n16))};
}
#define R4D_ARC2_BASES 0x40000000u   // the push check's index spaces of the bases and the spill,
#define R4D_ARC2_SPILL 0x80000000u   // apart from the packed dwords' (under 2^30)

// Code this lane's chunk `v` of the wave's chunks [k0, k0 + 32) into the peer's slot; every lane of
// the wave calls. Returns the lane's share of the push check.
__device__ __forceinline__ unsigned r4d_arc2_put(const R4dArc2& d, int k0, int4 v) {
  const int lane = (int)(threadIdx.x & 31u);
  R4dArcEnc e = r4d_arc_enc(v, true);
  const bool raw = e.base == 0xFFu;      /* uniform over the lane's group of 8 */
  if (raw) { e.w[0] = (unsigned)v.x; e.w[1] = (unsigned)v.y; e.w[2] = (unsigned)v.z; }
  unsigned h = 0u;
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int p = 32 * j + lane, sl = p / 3, wi = p - sl * 3;
    const unsigned a0 = r4d_arc_shfl(e.w[0], sl), a1 = r4d_arc_shfl(e.w[1], sl);
    const unsigned a2 = r4d_arc_shfl(e.w[2], sl);
    const unsigned val = wi == 0 ? a0 : wi == 1 ? a1 : a2;
    const unsigned at = 3u * (unsigned)k0 + (unsigned)p;
    d.packed[at] = val;
    h += r4d_archk_mix(val, at);
  }
  const unsigned b0 = r4d_arc_shfl(e.base, 0), b1 = r4d_arc_shfl(e.base, 8);
  const unsigned b2 = r4d_arc_shfl(e.base, 16), b3 = r4d_arc_shfl(e.base, 24);
  if (lane == 0) {
    const unsigned bw = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    d.bases[k0 >> 5] = bw;
    h += r4d_archk_mix(bw, R4D_ARC2_BASES + (unsigned)(k0 >> 5));
  }
  if (raw) {
    d.spill[k0 + lane] = (unsigned)v.w;
    h += r4d_archk_mix((unsigned)v.w, R4D_ARC2_SPILL + (unsigned)(k0 + lane));
  }
  return h;
}
// The reader's share of the push check over the wave's chunks [k0, k0 + 32): the dwords
// r4d_arc2_get is about to load, in the writer's index spaces.
__device__ __forceinline__ unsigned r4d_arc2_sum(const R4dArc2& s, int k0) {
  const int lane = (int)(threadIdx.x & 31u);
  const unsigned bw = s.bases[k0 >> 5];
  unsigned h = lane == 0 ? r4d_archk_mix(bw, R4D_ARC2_BASES + (unsigned)(k0 >> 5)) : 0u;
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    const unsigned at = 3u * (unsigned)k0 + (unsigned)(32 * j + lane);
    h += r4d_archk_mix(s.packed[at], at);
  }
  if (((bw >> ((lane >> 3) * 8)) & 0xffu) == 0xFFu)
    h += r4d_archk_mix(s.spill[k0 + lane], R4D_ARC2_SPILL + (unsigned)(k0 + lane));
  return h;
}
// This lane's chunk of the wave's chunks [k0, k0 + 32) back from a slot; every lane calls.
__device__ __forceinline__ int4 r4d_arc2_get(const R4dArc2& s, int k0) {
  const int lane = (int)(threadIdx.x & 31u);
  unsigned ld[3];
#pragma unroll
  for (int j = 0; j < 3; ++j) ld[j] = s.packed[3u * (unsigned)k0 + (unsigned)(32 * j + lane)];
  const unsigned base = (s.bases[k0 >> 5] >> ((lane >> 3) * 8)) & 0xffu;
  unsigned w[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const int q = 3 * lane + i, sl = q & 31, j = q >> 5;
    const unsigned a0 = r4d_arc_shfl(ld[0], sl), a1 = r4d_arc_shfl(ld[1], sl);
    const unsigned a2 = r4d_arc_shfl(ld[2], sl);
    w[i] = j == 0 ? a0 : j == 1 ? a1 : a2;
  }
  if (base == 0xFFu)
    return make_int4((int)w[0], (int)w[1], (int)w[2], (int)s.spill[k0 + lane]);
  return r4d_arc_dec(w, base);
}

// r4d_ar_hc_write_kernel and r4d_ar_gather_hc_write_kernel with the coded push. MODE 0: `y` is the
// message (the plain write); 1: the gather in front, its slot table in LDS; 2: the gather in front
// by the inverse table. Blocks own whole waves' chunks (the launcher: n16 and the block size are
// multiples of 32, and so is the thread count), so every lane of a wave is in each pass together.
// CH `r4dchain`: the chained form (r4d_chain.h); no block returns early.
template <int MODE, class CH>
__global__ void r4d_ar_hc_write_coded_kernel(
    int4* __restrict__ y4, int4* peer_base, const int4* my_base, unsigned int* peer_flags,
    unsigned int* my_flags, unsigned int* seq_ctrs, const unsigned short* __restrict__ inj,
    unsigned short* __restrict__ h, int n16, int n, int hc, long irow, long hrow,
    int slot_stride16, int drain, int acq, unsigned int nbw, R4dArGather G,
    const int* __restrict__ inv, long long* gate, unsigned int* gate_ctr) {
  if (CH::on) r4d_chain_wait();
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  __shared__ int   s_i[R4D_ARG_ROWS * R4D_ARG_SLOTS];
  __shared__ float s_w[R4D_ARG_ROWS * R4D_ARG_SLOTS];
  __shared__ float s_g[R4D_ARG_ROWS];
  const long long gate_word = r4d_ar_link_word(gate, (unsigned int)tid);
  r4d_ar_seq_take(seq_ctrs, nbw, s_seq, s_slot);   // device-resident, replay-safe
  const unsigned int s = s_seq;
  const int slot = (int)s_slot;
  const R4dArc2 ps = r4d_arc2_at(peer_base + (size_t)slot * slot_stride16, n16);
  const R4dArc2 ms = r4d_arc2_at(my_base + (size_t)slot * slot_stride16, n16);

  const int w = ((n16 + nb - 1) / nb + 31) & ~31;   // 16B words per block, whole waves
  int start = b * w; if (start > n16) start = n16;
  int end = start + w; if (end > n16) end = n16;
  const int lane = tid & 31;
  unsigned int hck = 0u;   // the push check (r4d_ar_seq.h)
  const int wpr = n >> 3;                 // 16B words a row
  const int row_lo = start / wpr;
  const int nrow = end > start ? (end - 1) / wpr - row_lo + 1 : 0;

  if (MODE == 1) {
    /* Slot table, as r4d_ar_gather_hc_write_kernel's. */
    const int K = G.top_k;
    for (int x = tid; x < nrow * K; x += nt) s_i[x] = -1;
    __syncthreads();
    {
      const int lo = row_lo * K, hi = (row_lo + nrow) * K;
      for (int i = tid; i < G.T; i += nt) {
        const int f = G.sorted[i];
        if (f >= lo && f < hi) s_i[f - lo] = i;
      }
    }
    for (int x = tid; x < nrow * K; x += nt)
      s_w[x] = bf2f(G.ew[(long)(row_lo + x / K) * G.ew_ld + x % K]);
    if (G.sh)
      for (int r = tid; r < nrow; r += nt) {
        const float v0 = bf2f(G.sg[(long)(row_lo + r) * G.sg_ld]);
        s_g[r] = G.sig ? (1.0f / (1.0f + __expf(-v0))) : v0;
      }
    __syncthreads();
  }

  /* drain 4, tests only (r4d_ar_late_push_pin): the flag ahead of the push and the push ~20 us
   * behind it, as in r4d_ar_oneshot_2rank_exact_kernel, so the push check re-reads coded bytes. */
  if (drain == 4) {
    if (tid == 0) store_sys_rel(&peer_flags[b], s);
    const unsigned long long t0 = wall_clock64();
    while (wall_clock64() - t0 < 2000ull) __builtin_amdgcn_s_sleep(8);   // 100 MHz: 20 us
  }
  /* Code and push; the gather forms produce the word first and keep it as this rank's message. */
  for (int k = start + tid; k < end; k += nt) {
    int4 v;
    if (MODE == 0) {
      v = y4[k];
    } else {
      const int row = k / wpr, i0 = (k - row * wpr) * 8;
      v = r4d_ar_gather_word<MODE == 2>(G, inv, s_i, s_w, s_g, row, row - row_lo, i0);
      y4[k] = v;
    }
    hck += r4d_arc2_put(ps, k - lane, v);
  }
  r4d_archk_put(hck, peer_flags, nbw, (unsigned)slot, (unsigned)b, s, drain == 4);
  if (drain == 1) __threadfence_system();
  else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
  __syncthreads();
  if (tid == 0) {
    if (drain == 2) __threadfence_system();
    if (drain != 4) store_sys_rel(&peer_flags[b], s);
    unsigned long long z = 0;
    while (r4d_ar_seq_behind(load_sys_acq(&my_flags[b]), s)) {
      if (++z > R4D_AR_SPIN_MAX) __builtin_trap();
    }
  }
  __syncthreads();
  if (acq) __threadfence_system();
  r4d_archk_wait(my_flags, nbw, (unsigned)slot, (unsigned)b, s, [&](unsigned int& hh) {
    for (int k = start + tid; k < end; k += nt) hh += r4d_arc2_sum(ms, k - lane);
  });
  r4d_ar_link_landed(gate, gate_word, gate_ctr, (unsigned int)nb, (unsigned int)tid);

  for (int k = start + tid; k < end; k += nt) {
    const int4 r = vadd<__hip_bfloat16, 8>(y4[k], r4d_arc2_get(ms, k - lane));
    y4[k] = r;
    const int row = k / wpr, i0 = (k - row * wpr) * 8;
    r4d_ar_hc_write_word(inj, h, row, i0, n, hc, irow, hrow,
                         reinterpret_cast<const unsigned short*>(&r));
  }
  if (CH::on) r4d_chain_done();
}

// ---- The staged coded form: r4d_ar_hc_write_coded_kernel with its block's chunks cut into `S`
// parts, each with its own handshake, so the link and the card's memory work at the same time.
//
// The coded kernel runs in two phases: every block codes and pushes its part of the message (the
// link's time: 31.5 MB of an 8192-row chunk at ~25 GB/s, 1.3 ms), then decodes the peer's, adds it
// and writes the four streams (the card memory's time: 0.42 GB read and written, ~0.8 ms). All
// blocks do the same at once, so a reduction takes the sum of the two. Here a block pushes part j,
// then folds part j - 1 while part j crosses the link, then publishes part j: the reduction takes
// about the longer of the two.
//
// Each part is a handshake unit of its own: virtual block v = b * S + j owns the part's flag, its
// handshake counter, its check word (r4d_ar_seq.h) and the part's chunks, as block v of a grid of
// nb * S blocks would, so the units need nb * S <= nbw. The slot parity is the block's, taken as
// every kernel of the family takes it. The arithmetic, the coded words and the fold order of every
// value are the coded kernel's: y and h are its bytes, and so the plain kernels'.
#define R4D_ARC_STAGES_MAX 16
template <int MODE, class CH>
__global__ void r4d_ar_hc_write_coded_staged_kernel(
    int4* __restrict__ y4, int4* peer_base, const int4* my_base, unsigned int* peer_flags,
    unsigned int* my_flags, unsigned int* seq_ctrs, const unsigned short* __restrict__ inj,
    unsigned short* __restrict__ h, int n16, int n, int hc, long irow, long hrow,
    int slot_stride16, int drain, int acq, unsigned int nbw, R4dArGather G,
    const int* __restrict__ inv, long long* gate, unsigned int* gate_ctr, int S) {
  if (CH::on) r4d_chain_wait();
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seqv[R4D_ARC_STAGES_MAX], s_slot;
  __shared__ int   s_i[R4D_ARG_ROWS * R4D_ARG_SLOTS];
  __shared__ float s_w[R4D_ARG_ROWS * R4D_ARG_SLOTS];
  __shared__ float s_g[R4D_ARG_ROWS];
  const long long gate_word = r4d_ar_link_word(gate, (unsigned int)tid);
  {
    /* r4d_ar_seq_take over the block's S units: one handshake counter each, the parity once. */
    unsigned int* const par = seq_ctrs + nbw;
    const unsigned int pt = nt > 32 ? 32u : 0u;
    if (tid < S) s_seqv[tid] = atomicAdd(&seq_ctrs[b * S + tid], 1u) + 1u;
    if ((unsigned)tid == pt) s_slot = (atomicAdd(&par[b], 1u) + 1u) & 1u;
    for (unsigned int j = b + (tid + 1u) * nb; j < nbw; j += nt * nb) atomicAdd(&par[j], 1u);
    r4d_lds_barrier();
  }
  const int slot = (int)s_slot;
  const R4dArc2 ps = r4d_arc2_at(peer_base + (size_t)slot * slot_stride16, n16);
  const R4dArc2 ms = r4d_arc2_at(my_base + (size_t)slot * slot_stride16, n16);

  const int w = ((n16 + nb - 1) / nb + 31) & ~31;   // 16B words per block, whole waves
  int start = b * w; if (start > n16) start = n16;
  int end = start + w; if (end > n16) end = n16;
  const int ws = ((w + S - 1) / S + 31) & ~31;      // a part's words, whole waves
  const int lane = tid & 31;
  const int wpr = n >> 3;                 // 16B words a row
  const int row_lo = start / wpr;
  const int nrow = end > start ? (end - 1) / wpr - row_lo + 1 : 0;

  if (MODE == 1) {
    const int K = G.top_k;
    for (int x = tid; x < nrow * K; x += nt) s_i[x] = -1;
    __syncthreads();
    {
      const int lo = row_lo * K, hi = (row_lo + nrow) * K;
      for (int i = tid; i < G.T; i += nt) {
        const int f = G.sorted[i];
        if (f >= lo && f < hi) s_i[f - lo] = i;
      }
    }
    for (int x = tid; x < nrow * K; x += nt)
      s_w[x] = bf2f(G.ew[(long)(row_lo + x / K) * G.ew_ld + x % K]);
    if (G.sh)
      for (int r = tid; r < nrow; r += nt) {
        const float v0 = bf2f(G.sg[(long)(row_lo + r) * G.sg_ld]);
        s_g[r] = G.sig ? (1.0f / (1.0f + __expf(-v0))) : v0;
      }
    __syncthreads();
  }
  auto lo_of = [&](int j) { const int x = start + j * ws; return x < end ? x : end; };

  /* Code and push part j, then its check word and, once the stores are out, its flag. The fold of
   * the part before runs between the stores and the flag (below), beside the transfer. */
  auto push = [&](int j) -> unsigned int {
    unsigned int hck = 0u;
    const int e = lo_of(j + 1);
    for (int k = lo_of(j) + tid; k < e; k += nt) {
      int4 v;
      if (MODE == 0) {
        v = y4[k];
      } else {
        const int row = k / wpr, i0 = (k - row * wpr) * 8;
        v = r4d_ar_gather_word<MODE == 2>(G, inv, s_i, s_w, s_g, row, row - row_lo, i0);
        y4[k] = v;
      }
      hck += r4d_arc2_put(ps, k - lane, v);
    }
    return hck;
  };
  auto publish = [&](int j, unsigned int hck) {
    const unsigned int v = (unsigned)(b * S + j);
    r4d_archk_put(hck, peer_flags, nbw, (unsigned)slot, v, s_seqv[j]);
    if (drain == 1) __threadfence_system();
    else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
    __syncthreads();
    if (tid == 0) {
      if (drain == 2) __threadfence_system();
      store_sys_rel(&peer_flags[v], s_seqv[j]);
    }
  };
  auto fold = [&](int j) {
    const unsigned int v = (unsigned)(b * S + j);
    const unsigned int sq = s_seqv[j];
    const int s0 = lo_of(j), e = lo_of(j + 1);
    /* A relaxed poll, as the register form's (r4d_ar_seq.h): an acquire load is followed by a
     * system-scope cache invalidate on every poll, which also drops what kernels running beside
     * this one have cached. The slot is fine-grained memory read
     * past the caches, and the push check re-reads behind an acquire whenever the bytes do not
     * agree. */
    if (tid == 0) {
      unsigned long long z = 0;
      while (r4d_ar_seq_behind(__hip_atomic_load(&my_flags[v], __ATOMIC_RELAXED,
                                                 __HIP_MEMORY_SCOPE_SYSTEM), sq)) {
        if (++z > R4D_AR_SPIN_MAX) __builtin_trap();
        __builtin_amdgcn_s_sleep(1);
      }
    }
    __syncthreads();
    if (acq) __threadfence_system();
    r4d_archk_wait(my_flags, nbw, (unsigned)slot, v, sq, [&](unsigned int& hh) {
      for (int k = s0 + tid; k < e; k += nt) hh += r4d_arc2_sum(ms, k - lane);
    });
    if (j == S - 1)
      r4d_ar_link_landed(gate, gate_word, gate_ctr, (unsigned int)nb, (unsigned int)tid);
    for (int k = s0 + tid; k < e; k += nt) {
      const int4 r = vadd<__hip_bfloat16, 8>(y4[k], r4d_arc2_get(ms, k - lane));
      y4[k] = r;
      const int row = k / wpr, i0 = (k - row * wpr) * 8;
      r4d_ar_hc_write_word(inj, h, row, i0, n, hc, irow, hrow,
                           reinterpret_cast<const unsigned short*>(&r));
    }
  };

  publish(0, push(0));
  for (int j = 1; j < S; ++j) {
    const unsigned int hck = push(j);
    fold(j - 1);
    publish(j, hck);
  }
  fold(S - 1);
  if (CH::on) r4d_chain_done();
}

/* Parts a block of the staged coded form takes, 0 for the single-phase coded kernel. A coded launch
 * takes the rule (r4d_ar_coded_stages_pin overrides it in tests and benchmarks): the write 8 parts
 * from 4096 KiB, the gather form 6 parts from 6144 KiB; 0 or 1 parts is the single-phase kernel.
 * Both ranks decide from the same launch, message and grid. On 2x R9700, rows
 * of 2560, 48 blocks of 1024, coded -> staged, us: write 1024 rows 203 -> 190, 2048 433 -> 378,
 * 4096 1023 -> 821, 8192 2015 -> 1639; gather 1536 rows 336 -> 326 (S4), 2048 482 -> 445,
 * 4096 1077 -> ~1005, 8192 2235 -> 2015. A gather below ~1.5K rows is slower staged. */
static int g_ar_stages_pin = -1;
extern "C" void r4d_ar_coded_stages_pin(int stages) { g_ar_stages_pin = stages; }
static int ar_coded_stages(int n16, int nb, unsigned nbw, bool gather = false) {
  constexpr int  kStages = 8, kStagesG = 6;
  constexpr long kMin16 = 4096L * 1024L / 16L, kMin16G = 6144L * 1024L / 16L;
  static_assert(kStages <= R4D_ARC_STAGES_MAX && kStagesG <= R4D_ARC_STAGES_MAX,
                "the staged form's parts fit its flag layout");
  int s;
  if (g_ar_stages_pin >= 0) s = g_ar_stages_pin > R4D_ARC_STAGES_MAX ? R4D_ARC_STAGES_MAX
                                                                           : g_ar_stages_pin;
  else if (gather) s = n16 >= kMin16G ? kStagesG : 0;
  else s = n16 >= kMin16 ? kStages : 0;
  int S = s;
  while (S > 1 && (long)nb * S > (long)nbw) --S;
  return S >= 2 ? S : 0;
}

/* Whether a message of n16 chunks fits the coded form in a slot of `slot_stride16` words on a P2P
 * instance (`stage` 0): whole waves' chunks, and the coded image within the slot. */
int r4d_ar_coded2_ok(long n16, long slot_stride16, long stage) {
  return n16 > 0 && n16 % 32 == 0 && !stage &&
         r4d_arc2_bytes(n16) <= (size_t)slot_stride16 * 16;
}
/* Bytes a slot needs to hold the coded form of any message of up to n16 chunks. */
long r4d_ar_coded2_bytes(long n16) { return (long)r4d_arc2_bytes((n16 + 31) / 32 * 32); }

/* r4d_ar_form_pin: the form of the fused writes at a message of one word a thread. -1 is the
 * launchers' rule (the register form on a P2P instance), 0 the plain kernels, 1 the rule again.
 * A test hook (r4d_selftest compares the two forms byte for byte); nothing else sets it. */
static int g_ar_form = -1;
extern "C" void r4d_ar_form_pin(int form) { g_ar_form = form; }

/* r4d_ar_late_push_pin: 1 makes every 2-rank exact all-reduce, and the register form of the fused
 * writes, send its flags before its pushes (drain 4). A test hook (r4d_selftest); nothing else
 * sets it. */
static int g_ar_late_push = 0;
extern "C" void r4d_ar_late_push_pin(int on) { g_ar_late_push = on; }

/* Whether a fused write takes the register form: one word a thread, the streams within the
 * registers' bound, and the slots in the peer's VRAM (`stage` 0: a host mailbox, read over the
 * link, keeps the plain kernels and their mirror). Both ranks decide from the same M, n and grid. */
static bool ar_reg_form(int n16, int nb, int nt, long hc, long stage) {
  return g_ar_form != 0 && (n16 + nb - 1) / nb <= nt && hc <= R4D_ARHC_MAX && !stage;
}

// The IPC handle goes into a caller-owned buffer of R4D_AR_HANDLE_BYTES so this stays a C ABI.
static_assert(sizeof(hipIpcMemHandle_t) <= R4D_AR_HANDLE_BYTES, "ipc handle larger than the ABI buffer");

long r4d_ar_ipc_alloc(long size, int finegrained, char* out_handle, int* out_len) {
  void* p = nullptr;
  if (finegrained) HIPOK(hipExtMallocWithFlags(&p, size, hipDeviceMallocFinegrained));
  else HIPOK(hipMalloc(&p, size));
  HIPOK(hipMemset(p, 0, size));
  // The zeroes must land before the pointer reaches a peer, or its first push or flag store could
  // race the queued memset.
  HIPOK(hipStreamSynchronize(nullptr));
  // The handle is for an out-of-process peer. libr4d's ranks are threads of one process and share
  // the pointer itself (r4d_ar_entry.hip), so a refused export is not fatal: beside another engine
  // on the same cards hipIpcGetMemHandle answers invalid argument while the allocation is fine.
  hipIpcMemHandle_t h;
  if (hipIpcGetMemHandle(&h, p) == hipSuccess) {
    std::memcpy(out_handle, &h, sizeof(h));
    *out_len = (int)sizeof(h);
  } else {
    (void)hipGetLastError();
    std::memset(out_handle, 0, R4D_AR_HANDLE_BYTES);
    *out_len = 0;
  }
  return (long)p;
}
// r4d_p2p_copy2d as a kernel with a bounded workgroup count.
//
// hipMemcpy2DAsync is not a DMA engine on this part: it runs as
// `__amd_rocclr_copyBufferRectAligned`, one 16-thread workgroup per 16 bytes of a row (62080
// workgroups for 64 x 248 KB). Alone it reaches link rate, but beside the GEMM it should hide
// under, it fills every scheduling slot and slows that GEMM by about 60%.
//
// A peer copy needs only a small grid: 32 workgroups of 64 threads is within 0.1% of the best
// configuration at every size. The grid is an argument and defaults to at most 64 workgroups.
//
// 16-byte accesses: every pitch, the width and both base addresses must be 16-byte aligned; the
// caller checks and falls back to r4d_p2p_copy2d otherwise.
typedef unsigned r4d_u4v __attribute__((ext_vector_type(4)));

__global__ __launch_bounds__(256) void r4d_p2p_copy2d_k(const r4d_u4v* __restrict__ src,
                                                        r4d_u4v* __restrict__ dst, long spitch16,
                                                        long dpitch16, long wpr, int height) {
  const long cstride = (long)gridDim.x * blockDim.x;
  for (int r = blockIdx.y; r < height; r += gridDim.y) {
    const r4d_u4v* s = src + (long)r * spitch16;
    r4d_u4v* d = dst + (long)r * dpitch16;
    for (long c = (long)blockIdx.x * blockDim.x + threadIdx.x; c < wpr; c += cstride) d[c] = s[c];
  }
}

int r4d_p2p_copy2d_kernel(long dst, long src, long dpitch, long spitch, long width, long height,
                          int bx, int by, int threads, long stream) {
  if ((dst | src | dpitch | spitch | width) & 15L) return -1;
  if (height <= 0 || width <= 0) return 0;
  if (threads <= 0 || threads > 256 || (threads & 31)) return -2;
  const long wpr = width / 16;
  if (bx <= 0) bx = (int)((wpr + threads - 1) / threads);
  if (bx > 64) bx = 64;
  if (by <= 0) by = (int)height;
  if (by > (int)height) by = (int)height;
  hipLaunchKernelGGL(r4d_p2p_copy2d_k, dim3(bx, by), dim3(threads), 0, (hipStream_t)stream,
                     (const r4d_u4v*)src, (r4d_u4v*)dst, spitch / 16, dpitch / 16, wpr,
                     (int)height);
  hipError_t e = hipGetLastError();
  return e == hipSuccess ? 0 : -100 - (int)e;
}

// A system-scope release on `stream`, for when this device has written into peer memory that
// another agent will read. A peer write from a copy or kernel is posted and retires long before it
// lands; only a system-scope release orders it against a later handshake on a different path. One
// thread suffices, since the fence is a memory-system operation.
__global__ void r4d_p2p_fence_kernel() { __threadfence_system(); }

int r4d_p2p_fence(long stream) {
  hipLaunchKernelGGL(r4d_p2p_fence_kernel, dim3(1), dim3(1), 0, (hipStream_t)stream);
  hipError_t e = hipGetLastError();
  return e == hipSuccess ? 0 : -100 - (int)e;
}

// A strided device-to-device copy: a two-rank all-gather once the peer's buffer is IPC-mapped.
// The runtime's copy path is the fast one for peer traffic on this part (13.52 GB/s per direction
// with both directions busy, against 9.30 for a b128-load kernel and 10.1 for RCCL).
// `dpitch`, `spitch` and `width` are bytes; `height` is rows.
int r4d_p2p_copy2d(long dst, long src, long dpitch, long spitch, long width, long height,
                   long stream) {
  hipError_t e = hipMemcpy2DAsync((void*)dst, (size_t)dpitch, (const void*)src, (size_t)spitch,
                                  (size_t)width, (size_t)height, hipMemcpyDeviceToDevice,
                                  (hipStream_t)stream);
  return e == hipSuccess ? 0 : -100 - (int)e;
}

void r4d_ar_hc_write(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                     long seq_ctrs, long slot_stride16, long y, long inj, long h, long M, long n,
                     long hc, long irow, long hrow, long stream_i, long nblocks, long nthreads,
                     long drain, long acq, long gate, long gate_ctr, long stage, long chain,
                     long coded) {
  hipStream_t st = (hipStream_t)stream_i;
  if (n <= 0 || M <= 0 || n % 8) throw std::runtime_error("ar_hc_write: n must be a positive multiple of 8");
  if (hc < 1 || hrow % 8) throw std::runtime_error("ar_hc_write: the streams must be whole 16B words");
  if (M * n * 2 / 16 > slot_stride16) throw std::runtime_error("ar_hc_write: message larger than a slot");
  const int n16 = (int)(M * n * 2 / 16);
  int nb = (int)nblocks;
  if (nb < 1) nb = 1;
  if (nb > n16) nb = n16;
  if (nb > R4D_AR_MAX_BLOCKS) nb = R4D_AR_MAX_BLOCKS;
  const unsigned nbw = (unsigned)r4d_ar_max_blocks();
  if ((unsigned)nb > nbw) nb = (int)nbw;
  int nt = (int)nthreads;
  if (nt < 64) nt = 64;
  if (nt > 1024) nt = 1024;
  nt &= ~31;   // whole waves: the push check's wave sum reads all 32 lanes
  if (coded && r4d_ar_coded2_ok(n16, slot_stride16, stage)) {
    if (nb > n16 / 32) nb = n16 / 32;   /* a block owns whole waves' chunks */
    if (g_ar_late_push) drain = 4;      /* tests only: every push lands after its flag */
    R4dArGather G;
    std::memset(&G, 0, sizeof G);
    if (const int S = drain == 4 ? 0 : ar_coded_stages(n16, nb, nbw)) {
#define R4D_ARHWS(CH_)                                                                          \
  r4d_ar_hc_write_coded_staged_kernel<0, CH_><<<nb, nt, 0, st>>>(                               \
      (int4*)y, (int4*)peer_scratch, (const int4*)my_scratch, (unsigned*)peer_flags,           \
      (unsigned*)my_flags, (unsigned*)seq_ctrs, (const unsigned short*)inj, (unsigned short*)h, \
      n16, (int)n, (int)hc, irow, hrow, (int)slot_stride16, (int)drain, (int)acq, nbw, G,      \
      nullptr, (long long*)gate, (unsigned int*)gate_ctr, S)
      if (chain) R4D_ARHWS(r4dchain); else R4D_ARHWS(R4dUnchained);
#undef R4D_ARHWS
      return;
    }
#define R4D_ARHWC(CH_)                                                                          \
  r4d_ar_hc_write_coded_kernel<0, CH_><<<nb, nt, 0, st>>>(                                      \
      (int4*)y, (int4*)peer_scratch, (const int4*)my_scratch, (unsigned*)peer_flags,           \
      (unsigned*)my_flags, (unsigned*)seq_ctrs, (const unsigned short*)inj, (unsigned short*)h, \
      n16, (int)n, (int)hc, irow, hrow, (int)slot_stride16, (int)drain, (int)acq, nbw, G,      \
      nullptr, (long long*)gate, (unsigned int*)gate_ctr)
    if (chain) R4D_ARHWC(r4dchain); else R4D_ARHWC(R4dUnchained);
#undef R4D_ARHWC
    return;
  }
  if (ar_reg_form(n16, nb, nt, hc, stage)) {
    /* Tests only: every push lands after its flag (drain 4). */
    const int dr = g_ar_late_push ? 4 : (int)drain;
#define R4D_ARHWR(CH_)                                                                          \
  r4d_ar_hc_write_reg_kernel<CH_><<<nb, nt, 0, st>>>(                                           \
      (int4*)y, (int4*)peer_scratch, (const int4*)my_scratch, (unsigned*)peer_flags,           \
      (unsigned*)my_flags, (unsigned*)seq_ctrs, (const unsigned short*)inj, (unsigned short*)h, \
      n16, (int)n, (int)hc, irow, hrow, (int)slot_stride16, dr, nbw, (long long*)gate,         \
      (unsigned int*)gate_ctr)
    if (chain) R4D_ARHWR(r4dchain); else R4D_ARHWR(R4dUnchained);
#undef R4D_ARHWR
    return;
  }
#define R4D_ARHW(CH_)                                                                           \
  r4d_ar_hc_write_kernel<CH_><<<nb, nt, 0, st>>>(                                               \
      (int4*)y, (int4*)peer_scratch, (const int4*)my_scratch, (unsigned*)peer_flags,           \
      (unsigned*)my_flags, (unsigned*)seq_ctrs, (const unsigned short*)inj, (unsigned short*)h, \
      n16, (int)n, (int)hc, irow, hrow, (int)slot_stride16, (int)drain, (int)acq, nbw,         \
      (long long*)gate, (unsigned int*)gate_ctr, stage)
  if (chain) R4D_ARHW(r4dchain); else R4D_ARHW(R4dUnchained);
#undef R4D_ARHW
}

int r4d_ar_gather_hc_write(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                           long seq_ctrs, long slot_stride16, long y, long inj, long h, long M,
                           long n, long hc, long irow, long hrow, long ye, long ew, long sorted,
                           long sh, long sg, long ye_ld, long ew_ld, long sh_ld, long sg_ld,
                           long top_k, long sig, long stream_i, long nblocks, long nthreads,
                           long drain, long acq, long inv, long gate, long gate_ctr, long stage,
                           long chain, long coded) {
  hipStream_t st = (hipStream_t)stream_i;
  if (n <= 0 || M <= 0 || n % 8 || hc < 1 || hrow % 8) return -1;
  if (top_k < 1 || top_k > R4D_ARG_SLOTS) return -2;
  if (ye_ld % 8 || (sh && sh_ld < n)) return -3;
  if (M * n * 2 / 16 > slot_stride16) return -5;      /* larger than a slot */
  const int n16 = (int)(M * n * 2 / 16);
  const int wpr = (int)(n / 8);
  const unsigned nbw = (unsigned)r4d_ar_max_blocks();
  int nb = (int)nblocks;
  if (nb < 1) nb = 1;
  /* Enough blocks that none spans more than R4D_ARG_ROWS rows: a block of w words touches at most
   * (w - 1) / wpr + 2 rows. Both ranks compute the same count from the same M and n, as the
   * per-block handshake requires. */
  int wmax = (R4D_ARG_ROWS - 2) * wpr + 1;
  /* The coded form (r4d_ar_hc_write_coded_kernel) gives a block whole waves' chunks, so its LDS
   * table bounds a block at the waves within R4D_ARG_ROWS - 2 rows. Rows too short for that to
   * be one wave leave the inverse table (a bound of one word asks for it below), or the plain
   * kernels. */
  const int wmax_c = ((R4D_ARG_ROWS - 2) * wpr) & ~31;
  bool cod = coded && r4d_ar_coded2_ok(n16, slot_stride16, stage);
  if (cod && wmax_c >= 32) wmax = wmax_c;
  else if (cod && inv != 0 && n16 > nb) wmax = 1;
  else cod = false;
  const int nb_min = (n16 + wmax - 1) / wmax;
  /* Use the inverse table when the LDS table would need more blocks than requested (a prefill
   * chunk, never a decode step); both ranks decide from the same M and n. The block count stays
   * the requested one: at a 2048-row chunk, 48, 96 and 192 blocks are no faster than 24. */
  const bool by_inv = inv != 0 && nb_min > nb;
  if (!by_inv && nb < nb_min) nb = nb_min;
  if (nb > n16) nb = n16;
  if (cod && nb > n16 / 32) nb = n16 / 32;
  if (nb > R4D_AR_MAX_BLOCKS || (unsigned)nb > nbw) return -4;
  int nt = (int)nthreads;
  if (nt < 64) nt = 64;
  if (nt > 1024) nt = 1024;
  nt &= ~31;   // whole waves: the push check's wave sum reads all 32 lanes
  /* Zeroed including padding: the struct is passed by value, so its tail padding is part of the
   * kernel-argument bytes a recorded pass's audit compares. */
  R4dArGather G;
  std::memset(&G, 0, sizeof G);
  G.ye = (const unsigned short*)ye;
  G.ew = (const unsigned short*)ew;
  G.sorted = (const int*)sorted;
  G.sh = (const unsigned short*)sh;
  G.sg = (const unsigned short*)sg;
  G.ye_ld = ye_ld; G.ew_ld = ew_ld; G.sh_ld = sh_ld; G.sg_ld = sg_ld;
  G.top_k = (int)top_k; G.T = (int)(M * top_k); G.sig = (int)sig;
  /* The inverse table's two launches stay plain; the decode form's one has a chained form. */
#define R4D_ARGHW(INV_, CH_, inv_)                                                              \
    r4d_ar_gather_hc_write_kernel<INV_, CH_><<<nb, nt, 0, st>>>(                                \
        (int4*)y, (int4*)peer_scratch, (const int4*)my_scratch, (unsigned*)peer_flags,         \
        (unsigned*)my_flags, (unsigned*)seq_ctrs, (const unsigned short*)inj, (unsigned short*)h, \
        n16, (int)n, (int)hc, irow, hrow, (int)slot_stride16, (int)drain, (int)acq, nbw, G,    \
        inv_, (long long*)gate, (unsigned int*)gate_ctr, stage)
  const int stg = cod && !g_ar_late_push ? ar_coded_stages(n16, nb, nbw, true) : 0;
  if (stg) {
#define R4D_ARGHWS(MODE_, CH_, inv_)                                                            \
    r4d_ar_hc_write_coded_staged_kernel<MODE_, CH_><<<nb, nt, 0, st>>>(                         \
        (int4*)y, (int4*)peer_scratch, (const int4*)my_scratch, (unsigned*)peer_flags,         \
        (unsigned*)my_flags, (unsigned*)seq_ctrs, (const unsigned short*)inj, (unsigned short*)h, \
        n16, (int)n, (int)hc, irow, hrow, (int)slot_stride16, (int)drain, (int)acq, nbw, G,    \
        inv_, (long long*)gate, (unsigned int*)gate_ctr, stg)
    if (by_inv) {
      r4d_ar_gather_inv_kernel<<<1, 1024, 0, st>>>((int*)inv, (const int*)sorted, G.T);
      R4D_ARGHWS(2, R4dUnchained, (const int*)inv);
    } else if (chain) {
      R4D_ARGHWS(1, r4dchain, nullptr);
    } else {
      R4D_ARGHWS(1, R4dUnchained, nullptr);
    }
#undef R4D_ARGHWS
  } else if (cod) {
    if (g_ar_late_push) drain = 4;      /* tests only: every push lands after its flag */
#define R4D_ARGHWC(MODE_, CH_, inv_)                                                            \
    r4d_ar_hc_write_coded_kernel<MODE_, CH_><<<nb, nt, 0, st>>>(                                \
        (int4*)y, (int4*)peer_scratch, (const int4*)my_scratch, (unsigned*)peer_flags,         \
        (unsigned*)my_flags, (unsigned*)seq_ctrs, (const unsigned short*)inj, (unsigned short*)h, \
        n16, (int)n, (int)hc, irow, hrow, (int)slot_stride16, (int)drain, (int)acq, nbw, G,    \
        inv_, (long long*)gate, (unsigned int*)gate_ctr)
    if (by_inv) {
      r4d_ar_gather_inv_kernel<<<1, 1024, 0, st>>>((int*)inv, (const int*)sorted, G.T);
      R4D_ARGHWC(2, R4dUnchained, (const int*)inv);
    } else if (chain) {
      R4D_ARGHWC(1, r4dchain, nullptr);
    } else {
      R4D_ARGHWC(1, R4dUnchained, nullptr);
    }
#undef R4D_ARGHWC
  } else if (!by_inv && ar_reg_form(n16, nb, nt, hc, stage)) {
    /* Tests only: every push lands after its flag (drain 4). */
    const int dr = g_ar_late_push ? 4 : (int)drain;
#define R4D_ARGHWR(CH_)                                                                         \
    r4d_ar_gather_hc_write_reg_kernel<CH_><<<nb, nt, 0, st>>>(                                  \
        (int4*)y, (int4*)peer_scratch, (const int4*)my_scratch, (unsigned*)peer_flags,         \
        (unsigned*)my_flags, (unsigned*)seq_ctrs, (const unsigned short*)inj, (unsigned short*)h, \
        n16, (int)n, (int)hc, irow, hrow, (int)slot_stride16, dr, nbw, G, (long long*)gate,    \
        (unsigned int*)gate_ctr)
    if (chain) R4D_ARGHWR(r4dchain); else R4D_ARGHWR(R4dUnchained);
#undef R4D_ARGHWR
  } else if (by_inv) {
    r4d_ar_gather_inv_kernel<<<1, 1024, 0, st>>>((int*)inv, (const int*)sorted, G.T);
    R4D_ARGHW(true, R4dUnchained, (const int*)inv);
  } else if (chain) {
    R4D_ARGHW(false, r4dchain, nullptr);
  } else {
    R4D_ARGHW(false, R4dUnchained, nullptr);
  }
#undef R4D_ARGHW
  return (int)hipGetLastError();
}

/* The inverse slot table alone, for the wide fused gather (r4d_ar_oneshot_Nrank_exact.hip); same
 * kernel and launch as above. */
int r4d_ar_gather_inv(long inv, long sorted, long T, long stream_i) {
  if (!inv || !sorted || T < 1) return -1;
  r4d_ar_gather_inv_kernel<<<1, 1024, 0, (hipStream_t)stream_i>>>((int*)inv, (const int*)sorted,
                                                                  (int)T);
  return (int)hipGetLastError();
}

void r4d_ar_ipc_free(long p) { if (p) (void)hipFree((void*)p); }

// Host mailbox. Coherent so a peer's stores are not held in a CPU cache line, mapped so cards can
// address it, portable so every card can (not only the current one). Cards reach it over the same
// fine-grained path as a peer's VRAM scratch, so the kernels' fences apply unchanged. The card
// address must equal the CPU address, because one pointer is published to every rank.
long r4d_ar_host_alloc(long size) {
  void* p = nullptr;
  HIPOK(hipHostMalloc(&p, size, hipHostMallocCoherent | hipHostMallocMapped |
                                    hipHostMallocPortable));
  std::memset(p, 0, (size_t)size);
  void* d = nullptr;
  const hipError_t e = hipHostGetDevicePointer(&d, p, 0);
  if (e != hipSuccess || d != p) {
    (void)hipHostFree(p);
    throw std::runtime_error(e != hipSuccess
        ? std::string("host mailbox: no card address: ") + hipGetErrorString(e)
        : std::string("host mailbox: the card address differs from the CPU address"));
  }
  return (long)p;
}
void r4d_ar_host_free(long p) { if (p) (void)hipHostFree((void*)p); }

// Enable direct peer access. Ranks are threads of one process sharing an address space, so a peer
// pointer is valid once this has run and no IPC handle exchange is needed (see r4d_ar_entry.hip).
void r4d_ar_ipc_enable_peer(long peer) {
  hipError_t e = hipDeviceEnablePeerAccess((int)peer, 0);
  if (e == hipErrorPeerAccessAlreadyEnabled) { (void)hipGetLastError(); return; }
  if (e != hipSuccess)
    throw std::runtime_error(std::string("enable_peer: ") + hipGetErrorString(e));
}

void r4d_ar_oneshot_2rank_exact(long peer_scratch, long my_scratch, long peer_flags,
                            long my_flags, long seq_ctrs, long slot_stride16,
                            long inp, long out, long n_elem, long dtype,
                            long stream_i, long nblocks, long nthreads,
                            long drain, long acq, long stage, long chain) {
  hipStream_t st = (hipStream_t)stream_i;
  const int esize = (dtype == 2) ? 4 : 2;
  const long nbytes = (long)n_elem * esize;
  if (nbytes % 16 != 0) throw std::runtime_error("all_reduce: n_elem*esize not 16B-aligned");
  if (nbytes / 16 > slot_stride16) throw std::runtime_error("all_reduce: message larger than a slot");
  const int n16 = (int)(nbytes / 16);
  int nb = (int)nblocks;
  if (nb < 1) nb = 1;
  if (nb > n16) nb = n16;
  if (nb > R4D_AR_MAX_BLOCKS) nb = R4D_AR_MAX_BLOCKS;
  // Block capacity of the counter allocation. Every 2-rank kernel on an instance must agree on it,
  // because the parity counters start that many words in (r4d_ar_seq.h).
  const unsigned nbw = (unsigned)r4d_ar_max_blocks();
  if ((unsigned)nb > nbw) nb = (int)nbw;
  int nt = (int)nthreads;
  if (nt < 64) nt = 64;
  if (nt > 1024) nt = 1024;
  nt &= ~31;   // whole waves: the push check's wave sum reads all 32 lanes
  // Tests only: every push lands after its flag (drain 4 above). Not in the tuning table, so no
  // tuner can pick it.
  if (g_ar_late_push) drain = 4;
#define LK(T, LN, CH_) r4d_ar_oneshot_2rank_exact_kernel<T, LN, CH_><<<nb, nt, 0, st>>>(   \
    (const int4*)inp, (int4*)peer_scratch, (const int4*)my_scratch, (int4*)out,        \
    (unsigned*)peer_flags, (unsigned*)my_flags, (unsigned*)seq_ctrs, n16,              \
    (int)slot_stride16, (int)drain, (int)acq, nbw, stage)
#define L(T, LN) do { if (chain) LK(T, LN, r4dchain); else LK(T, LN, R4dUnchained); } while (0)
  if (dtype == 0) L(__hip_bfloat16, 8);
  else if (dtype == 1) L(__half, 8);
  else if (dtype == 2) L(float, 4);
  else throw std::runtime_error("all_reduce: bad dtype");
#undef L
#undef LK
}

