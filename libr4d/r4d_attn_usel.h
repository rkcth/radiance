// r4d_attn_usel.h -- the prefill form of QSA's sparse attention: a tile of query rows over the
// union of their page lists.
//
// A sparse launch is one launch row per query (q_len 1), each row with its own page list (`sel`,
// 512 pages and the partial one) and key count (docs/QSA.md). The split-KV decode kernel
// (r4d_attn_decode_body) serves it one wave a row, and each row gathers its own 2051 keys from the
// paged cache: at a 2048-row prefill chunk that is 4.3 GB of K and V a layer and rank, and the
// launch runs at about a ninth of the card's bf16 WMMA rate (about 3 ms a layer on a R9700, 12 %
// of a Qwen3.8-Flash-Next prefill chunk). The rows of a chunk are consecutive queries over one
// history, so their lists overlap: eight of them name 1000 to 1750 distinct pages, not 4104.
//
// This form reads each page once per tile of rows. Three kernels where the decode form has one:
//
//   r4d_usel_build_kernel   a workgroup a tile of G rows: the union of the rows' page lists,
//                           ordered by the first and the last row of the tile that lists a page
//                           and then by its place in that first row's list, and beside every page
//                           of the union the rows that list it, a bit a row. Order and bits depend
//                           on which entries of the lists name the same page and on nothing else,
//                           never on a page id's value, so the result does not move with the
//                           cache's allocation.
//   r4d_attn_usel_kernel    the decode kernel's arithmetic a row (a wave a launch row, lazy
//                           reference-max softmax, f16 operands, folded query scale), but K and V
//                           staged once in LDS for all the tile's rows, sixteen keys of the union
//                           at a time: a key's score is masked to -inf where the row does not
//                           list the key's page (or sees less of its partial page), and a wave
//                           whose row lists none of the sixteen keys' pages skips their matmuls.
//                           It stores the decode kernel's partial (O / l as f16, m, l) as the one
//                           segment of its row. Launched twice, as two instances: one for the
//                           tiles that hold slots no row sees (below), one for the others.
//   r4d_attn_decode_usel_fb_kernel
//                           the decode kernel itself, for the tiles whose rows share too little
//                           for a tile to pay (rows of unrelated sequences): the builder flags
//                           them, this kernel serves exactly those and the tiled one the rest.
//
// The merge is the decode launch's, unchanged (splits 1), so the output row, the gate and the
// quantiser are computed from the partial as the decode launch computes them.
//
// What a row attends is exactly what the decode form attends: the keys of its own list, each once.
// A score is the decode kernel's bit for bit, and p = 2^(s - m_ref), the rescale rule, the partial
// and the merge are its arithmetic. What differs is the order in which the softmax meets a row's
// keys: the decode form takes them sixteen at a time in the order of the row's list, this form in
// the order of the tile's union. That moves two roundings: m_ref, the max of the first sixteen
// keys a row meets (and of each later sixteen that outgrow it by 14 octaves), against which every
// p is rounded to f16; and the f32 sums. So the two forms agree to the rounding of an f16 p
// (2^-11) and of an f32 accumulation over 2051 terms, not bit for bit; r4d_selftest states what it
// measures, and that each is as close to an f64 evaluation of the row's own key set as the other.
// A row's result depends on the other rows of its tile through that order only.
//
// What a workgroup loads is more than its rows attend: every page of the union, for all its rows,
// where the decode kernel loads no key past its row's count (it clamps the key index). A key a
// row does not see takes a score of -inf and a p of 0, which is nothing only against a finite V,
// and not every slot of a cache is one: behind the last token of the block a sequence ends in
// nothing has ever been written, the memory holds what it held before, which read as bf16 is as
// often inf or NaN as a number, and 0 * inf in one row's V product is a NaN in every row of the
// workgroup that walks that tile of keys. So the form stages no slot the decode form would not
// have read for a row of the same tile: beside each page of the union the builder leaves the
// number of its slots that no row of the tile sees (the page is some rows' partial page and
// nobody's whole one), and those slots are staged as zeros, K and V. Every other slot is one the
// decode form reads for one of the tile's own rows.
//
// The launcher (r4d_attn_paged_gqa8.hip) takes this form only where a test pins it: it agrees
// with the decode form to rounding, not bit for bit, and measured no faster in a served model
// (docs/QSA.md).
#pragma once

#include "r4d.h"
#include "r4d_common.h"
#include "r4d_dt16.h"

// What the builder leaves for the two attention kernels.
struct R4DUsel {
    const int*      utab;    // [tiles][upitch] page ids of a tile's union, in its order; above
                             //        kUselPageBits, the page's slots no row of the tile sees
    const uint16_t* usig;    // [tiles][upitch] the tile's rows that list the page, a bit a row
    const int*      ucnt;    // [tiles] pages in the union; kUselHidden: some slot of them is unseen
    const int*      fb;      // [tiles] 1: the tile takes the decode kernel
    const uint32_t* prow;    // [rows] the row's partial page: its place in the union (0xffff: none)
                             //        in the low half, the keys the row sees of it in the high
    int upitch;
};

constexpr int kUselMaxW    = 520;    // widest page list a row (the table's pitch) the form takes
constexpr int kUselMaxG    = 16;     // most rows a tile: a bit a row in 16
constexpr int kUselThreads = 256;    // the builder's workgroup
constexpr int kUselSlots   = 8192;   // its hash table, in LDS
constexpr int kUselCap     = 6144;   // most pages a union it builds: past it the tile is handed back
/* A table entry is a page id below and, above, how many of the page's last slots no row of the
 * tile sees (0 to 3). A tile that names a page id past the mask is handed back. */
constexpr int      kUselPageBits = 30;
constexpr uint32_t kUselPageMask = (1u << kUselPageBits) - 1u;
/* In a tile's page count: the union holds a page with unseen slots. A chunk that starts on a page
 * has them in its prompt's last tile at most, and the attention kernel that looks at the entries'
 * counts is another instance (HID) than the one that does not: the kernel has no register to
 * spare, and the masks cost the plain walk 5 % to 9 % when they are compiled into it. */
constexpr int      kUselHidden   = 1 << 30;
/* Pages a tile's table is padded with behind its union (the last page again, listed by no row):
 * the attention kernel reads the table a few tiles ahead and a tile's sixteen keys whole, with no
 * test for the union's end. */
constexpr int kUselPad     = 32;
constexpr int kUselAhead   = 3;      // tiles a stager holds in flight
/* A tile is tiled when its rows' lists sum to at least 3/2 of their union (a chunk's sum to 2.4 to
 * 4 times it, unrelated rows' to 1); a limit of kUselNoLimit or more tiles every tile (tests). */
constexpr int kUselShareNum = 3, kUselShareDen = 2;
constexpr int kUselNoLimit  = 1 << 20;

/* The union of a tile's page lists, and the rows that list each of its pages.
 *
 * Entry (r, j) is slot j of the tile's row r, live while j < nsel(r) = ceil(seqused / 4) and the
 * page id is not negative. A row's entries belong to TPR = 256 / G consecutive threads, CH of them
 * a thread in slot order.
 *
 *   1  every live entry's page goes into the tile's hash table (LDS, compare-and-swap, linear
 *      probing) and ORs its row's bit into the page's sixteen: the rows that list the page,
 *      whatever order the threads ran in. A claim counts the distinct pages; at kUselCap the tile
 *      stops and is handed back (so is one whose union passes `lim`, or whose rows share too
 *      little), which is decided by the number of distinct pages alone.
 *   2  an entry is its page's first appearance when its row is the lowest bit there, and the
 *      highest is the last row that lists the page. The first entries are ranked by (first row,
 *      last row, slot): a count a thread by last row, a prefix over the threads of the row, the
 *      row's totals as the buckets (first row, last row) and a prefix over the G * G buckets. The
 *      rank is the page's place in the union: page id and row bits are stored there, and the rank
 *      replaces the bits in the table. With the page id goes the number of the page's slots that
 *      no row of the tile sees: none where a row lists the page whole, else four less the most
 *      keys a row sees of it as its partial page (its last, when it sees fewer than four).
 *   3  the table is padded behind the union (kUselPad pages: the last page again, listed by no
 *      row), and a row's partial page (its last, when it sees fewer than four keys of it) is
 *      looked up and its place and key count left in `prow`.
 *
 * The order is for the attention kernel's skip: a page read by rows a to b of the tile sits
 * beside the others read from row a on and until row b, so the pages one row reads lie in runs
 * and the sixteen-key m-tiles that hold none of them are many (on a chunk's real lists a row
 * meets 36 % to 61 % of its tile's m-tiles this way where its own list alone is 30 % to 52 % of
 * them, and in order of first appearance 49 % to 70 %; tiles of eight rows). */
template <int G>
__global__ __launch_bounds__(kUselThreads)
void r4d_usel_build_kernel(const int* __restrict__ bt, const int* __restrict__ su, int rows,
                           int W, int* __restrict__ utab, uint16_t* __restrict__ usig,
                           int* __restrict__ ucnt, int* __restrict__ fb,
                           uint32_t* __restrict__ prow, int upitch, int lim) {
    constexpr int NT = kUselThreads, NW = NT / 32, S = kUselSlots;
    constexpr int TPR = NT / G;                                /* threads a row */
    constexpr int CH  = (kUselMaxW + TPR - 1) / TPR;           /* entries a thread */
    static_assert(G <= kUselMaxG && G * G <= NT && CH <= 64 && (S & (S - 1)) == 0 &&
                  kUselCap + NT < S,
                  "a bucket a thread, a thread's entries bits of one register, room to probe");
    __shared__ int      s_key[S];                              /* page id, -1 empty */
    __shared__ uint32_t s_sig[S / 2];                          /* sixteen bits a slot: rows, then rank */
    __shared__ uint8_t  s_cnt[NT * G];                         /* first entries of a thread by last row */
    __shared__ uint16_t s_pre[NT * G];                         /* ... of the row's threads before it */
    __shared__ uint16_t s_tot[G * G];                          /* ... of a row: the buckets */
    __shared__ uint16_t s_base[G * G];                         /* first entries in the buckets before */
    __shared__ int      s_nsel[G], s_last[G];
    __shared__ int      s_part[G];                             /* a row's partial page, -1 none */
    __shared__ uint32_t s_wt[NW];
    __shared__ uint32_t s_n, s_over, s_hid;
    __shared__ int      s_end;                                 /* the union's last entry */

    const int t = threadIdx.x, lane = t & 31, wave = t >> 5;
    const int g = blockIdx.x, r0 = g * G;
    const int r = t / TPR, sub = t - r * TPR;                  /* this thread's row and place in it */
    const bool mine = r < G;
    const int j0 = sub * CH;
    int*      ut = utab + (size_t)g * upitch;
    uint16_t* us = usig + (size_t)g * upitch;
    uint16_t* const sig16 = (uint16_t*)s_sig;                  /* slot h's sixteen bits */

    /* ---- 0: the table empty, the counts clear, the rows' page counts. */
    for (int i = t; i < S; i += NT) s_key[i] = -1;
    for (int i = t; i < S / 2; i += NT) s_sig[i] = 0u;
    for (int b = 0; b < G; ++b) s_cnt[t * G + b] = 0;
    if (t < G * G) s_tot[t] = 0;
    if (t == 0) { s_n = 0u; s_over = 0u; s_hid = 0u; s_end = 0; }
    if (t < G) {
        const int used = (r0 + t < rows) ? su[r0 + t] : 0;
        const int n = used > 0 ? (used + 3) / 4 : 0;
        const int keys = n <= W ? used - 4 * (n - 1) : 4;
        s_nsel[t] = n < W ? n : W;
        s_last[t] = keys;
        s_part[t] = (n > 0 && keys < 4) ? bt[(size_t)(r0 + t) * W + n - 1] : -1;
    }
    __syncthreads();

    /* The slot of a page that is in the table. */
    auto slot_of = [&](int page) {
        unsigned h = ((unsigned)page * 0x9E3779B1u) >> 12 & (unsigned)(S - 1);
        while (s_key[h] != page) h = (h + 1u) & (unsigned)(S - 1);
        return h;
    };

    /* ---- 1: insert. */
    const int nrow = mine ? s_nsel[r] : 0;
    const int* brow = bt + (size_t)(r0 + (mine ? r : 0)) * W;
    for (int i = 0; i < CH; ++i) {
        const int j = j0 + i;
        if (j >= nrow || s_over) break;
        const int page = brow[j];
        if (page < 0) continue;
        if ((uint32_t)page > kUselPageMask) { s_over = 1u; break; }
        unsigned h = ((unsigned)page * 0x9E3779B1u) >> 12 & (unsigned)(S - 1);
        for (;;) {
            const int old = atomicCAS(s_key + h, -1, page);
            if (old == -1) {
                if (atomicAdd(&s_n, 1u) >= (unsigned)kUselCap) s_over = 1u;
                break;
            }
            if (old == page) break;
            h = (h + 1u) & (unsigned)(S - 1);
        }
        atomicOr(s_sig + (h >> 1), (1u << r) << (16u * (h & 1u)));
    }
    __syncthreads();
    if (s_over) {
        if (t == 0) { ucnt[g] = 0; fb[g] = 1; }
        return;
    }

    /* ---- 2: the first appearances, counted by last row. */
    unsigned long long first = 0ull;
    for (int i = 0; i < CH; ++i) {
        const int j = j0 + i;
        if (j >= nrow) break;
        const int page = brow[j];
        if (page < 0) continue;
        const unsigned sig = sig16[slot_of(page)];
        if ((sig & (0u - sig)) != (1u << r)) continue;
        first |= 1ull << i;
        s_cnt[t * G + (31 - __builtin_clz(sig))] += 1;
    }
    __syncthreads();
    /* The first entries of the row's threads before this one, by last row; the row's last thread
     * leaves the row's totals. */
    if (mine) {
        for (int b = 0; b < G; ++b) {
            unsigned p = 0;
            for (int q = 0; q < sub; ++q) p += s_cnt[(r * TPR + q) * G + b];
            s_pre[t * G + b] = (uint16_t)p;
            if (sub == TPR - 1) s_tot[r * G + b] = (uint16_t)(p + s_cnt[t * G + b]);
        }
    }
    __syncthreads();
    /* A prefix over the buckets, a bucket a thread. */
    const uint32_t tot  = t < G * G ? s_tot[t] : 0u;
    const uint32_t incl = r4d_wave_prefix_u32(tot, lane);
    if (lane == 31) s_wt[wave] = incl;
    __syncthreads();
    uint32_t base = incl - tot, total = 0;
    for (int w = 0; w < NW; ++w) {
        const uint32_t c = s_wt[w];
        if (w < wave) base += c;
        total += c;
    }
    if (t < G * G) s_base[t] = (uint16_t)base;
    __syncthreads();
    /* A first entry's slot is read and written here by its own thread alone. */
    for (int i = 0; i < CH; ++i) {
        if (!((first >> i) & 1ull)) continue;
        const int page = brow[j0 + i];
        const unsigned h = slot_of(page);
        const unsigned sig = sig16[h];
        const int b = 31 - __builtin_clz(sig);
        const unsigned rank = (unsigned)s_base[r * G + b] + s_pre[t * G + b];
        s_pre[t * G + b] += 1;
        /* The most keys a row of the tile sees of the page. */
        int seen = 0;
        for (unsigned l = sig; l; l &= l - 1u) {
            const int o = __builtin_ctz(l);
            const int k = s_part[o] == page ? s_last[o] : 4;
            seen = k > seen ? k : seen;
        }
        const int ent = (int)((uint32_t)page | ((uint32_t)(4 - seen) << kUselPageBits));
        ut[rank] = ent;
        us[rank] = (uint16_t)sig;
        sig16[h] = (uint16_t)rank;
        if (rank + 1u == total) s_end = ent;
        if (seen < 4) s_hid = 1u;
    }
    __syncthreads();
    if (t == 0) {
        /* Handed back: a union past `lim`, or (unless `lim` says every tile is tiled) one the rows
         * share too little of. Sharing is what the tile is for: where the lists name `total` pages
         * between them and sum to less than kUselShare times that, the rows are unrelated
         * sequences' and a workgroup would walk their keys one row at a time. */
        int sum = 0;
        for (int i = 0; i < G; ++i) sum += s_nsel[i];
        const bool thin = lim < kUselNoLimit && (int)total * kUselShareNum > sum * kUselShareDen;
        ucnt[g] = (int)total | (s_hid ? kUselHidden : 0);
        fb[g] = ((int)total > lim || thin) ? 1 : 0;
    }

    /* ---- 3: the table's padding behind the union (the last entry again, its unseen slots
     * with it; from LDS, since the table itself was stored by another wave a moment ago), and
     * the rows' partial pages. */
    if (t < kUselPad && total > 0) {
        ut[total + t] = s_end;
        us[total + t] = 0;
    }
    if (t < G && r0 + t < rows) {
        const int n = s_nsel[t], keys = s_last[t];
        uint32_t pr = 0xffffu | (4u << 16);
        if (n > 0 && keys < 4) {
            const int page = bt[(size_t)(r0 + t) * W + n - 1];
            if (page >= 0) pr = (uint32_t)sig16[slot_of(page)] | ((uint32_t)keys << 16);
        }
        prow[r0 + t] = pr;
    }
}

/* One of a uint4's four words by a per-lane index. */
__device__ __forceinline__ uint32_t r4d_usel_pick(uint4 v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

/* The attention over a tile's union. Grid (tiles, kv_heads); bf16 query and bf16 cache, 4-token
 * pages, sixteen keys (four pages) a KV tile.
 *
 * A workgroup is NWARPS waves. The first G = NWARPS - NS each hold one launch row of the tile, its
 * heads in the wave's first GQA rows and copies of the first ones behind them, as the decode
 * kernel's one wave does: a query's heads share a wave, skip a tile together and rescale together
 * on the decode kernel's rule.
 *
 * K and V of a tile of keys go through LDS once for all the workgroup's rows. There are two LDS
 * tiles: tile ti is read from LDS tile ti & 1 while tile ti + 1 is staged into the other, one
 * barrier a tile.
 *
 * NS > 0: the last NS waves are stagers. They hold no query: they fetch the union's K and V,
 * kUselAhead tiles ahead in registers, and store one tile a step, so a wave with queries reads
 * nothing but LDS. NS == 0: every wave stages its share of the next tile behind its own matmuls.
 *
 * Nothing in the walk is a scalar load: a tile's page ids and row bits are read by every lane as
 * vector loads two tiles ahead and used from registers. A scalar load is waited for where it is
 * issued, and a wave that skips a tile has no matmuls to hide that wait under.
 *
 * HID: the instance for the tiles whose union holds unseen slots (kUselHidden), which stages
 * zeros for them; the other instance serves the tiles without and reads no entry's count. Both
 * are launched over every tile, and a tile is one's or the other's. */
template <int NWARPS, int GQA, int NS, int HID>
__global__ __launch_bounds__(NWARPS * 32)
void r4d_attn_usel_kernel(const R4DArgs a, const R4DUsel u) {
    constexpr int HEAD_DIM = 256, BS = 4, PF = 2, TILE = 16;
    typedef DT16<1> D;
    typedef D::frag frag16;
    constexpr int G      = NWARPS - NS;                         // waves with a query: rows a tile
    constexpr int NKS    = HEAD_DIM / 16;
    constexpr int KSTR   = HEAD_DIM + 8;
    constexpr int VSTR   = TILE + 8;
    constexpr int NB     = TILE / BS;
    constexpr int SW     = NS ? NS : NWARPS;                    // waves that stage
    constexpr int KITEMS = TILE * (HEAD_DIM / 8);               // 16-byte K loads a tile
    constexpr int NTR    = (TILE / 8) * (HEAD_DIM / 32);        // transposed V loads a tile, by wave
    constexpr int SKP    = KITEMS / (SW * 32);                  // a staging thread's K loads a tile
    constexpr int SVP    = NTR / SW;                            // a staging wave's V loads a tile
    constexpr float PSHIFT = D::SHIFT, PGROW = D::GROW;
    static_assert(GQA <= 16 && G >= 1 && G <= kUselMaxG && NB == 4 && NKS % PF == 0 &&
                  KITEMS % (SW * 32) == 0 && NTR % SW == 0,
                  "a query a wave, a bit a row of the tile, whole loads a staging thread");
    constexpr int KSZ = TILE * KSTR, VSZ = HEAD_DIM * VSTR;
    __shared__ uint16_t sK[2 * KSZ];
    __shared__ uint16_t sV16[2 * VSZ];

    const int tid = threadIdx.x, lane = tid & 31;
    const int warp = __builtin_amdgcn_readfirstlane(tid >> 5);
    __builtin_assume(tid < NWARPS * 32);
    const int c = lane & 15, h = lane >> 4;
    const int g = blockIdx.x, kvh = blockIdx.y;
    if (u.fb[g]) return;
    const int uc  = u.ucnt[g];
    const int np  = uc & (kUselHidden - 1);
    if (((uc & kUselHidden) != 0) != (HID != 0)) return;    /* the other instance's tile */
    const int ctx = np * BS;
    if (ctx <= 0) return;                    /* every row of the tile is padding: the merge zeroes */
    const int ntiles = (ctx + TILE - 1) / TILE;
    const int* __restrict__ bt = u.utab + (size_t)g * u.upitch;
    const uint16_t* __restrict__ sg = u.usig + (size_t)g * u.upitch;

    /* ---- staging: thread sid of the SW staging waves takes K items sid, sid + SW * 32, ... (a
     * key's 32 sixteen-byte chunks are one wave's) and wave sw the V items sw, sw + SW, ... (eight
     * keys by 32 dims, transposed across each eight lanes). `ids` are the tile's four table
     * entries. */
    const int sw = NS ? warp - G : warp, sid = sw * 32 + lane;
    const int vg = lane >> 3, vj = lane & 7;
    struct Regs { uint4 k[SKP]; uint4 v[SVP]; };
    auto ids_of = [&](int ti) { return *(const uint4*)(bt + (size_t)ti * NB); };
    /* Whether a row of the tile sees key `key` of the staged tile, by its page's entry: the slots
     * no row sees are staged as zeros (see the head of this file). */
    auto seen = [](uint32_t ent, int key) {
        return (uint32_t)(key % BS) + (ent >> kUselPageBits) < (uint32_t)BS;
    };
    /* ... as a mask over a word of two sixteen-bit values, slots `slot` and `slot + 1`. */
    auto seen2 = [&](uint32_t ent, int slot) {
        return (seen(ent, slot) ? 0x0000ffffu : 0u) | (seen(ent, slot + 1) ? 0xffff0000u : 0u);
    };
    auto fetch = [&](uint4 ids, Regs& o) {
        #pragma unroll
        for (int p = 0; p < SKP; ++p) {
            const int i = sid + p * (SW * 32);
            const int key = i / (HEAD_DIM / 8), ch = i - key * (HEAD_DIM / 8);
            const uint32_t ent = r4d_usel_pick(ids, key / BS);
            const size_t off = (size_t)(HID ? ent & kUselPageMask : ent) * a.kv_block_stride
                             + (size_t)kvh * a.kv_head_stride + (size_t)(key % BS) * (2 * HEAD_DIM);
            uint4 kw = *(const uint4*)((const uint16_t*)a.kv + off + ch * 8);
            if constexpr (HID) kw = seen(ent, key) ? kw : make_uint4(0u, 0u, 0u, 0u);
            o.k[p] = kw;
        }
        #pragma unroll
        for (int r = 0; r < SVP; ++r) {
            constexpr int DG = HEAD_DIM / 32;
            const int idx = sw + r * SW;
            const int kg = idx / DG, dg = idx - kg * DG;
            const int key = kg * 8 + vj;
            const int d0 = dg * 32 + vg * 8;
            const uint32_t ent = r4d_usel_pick(ids, key / BS);
            const size_t off = (size_t)(HID ? ent & kUselPageMask : ent) * a.kv_block_stride
                             + (size_t)kvh * a.kv_head_stride
                             + (size_t)(key % BS) * (2 * HEAD_DIM) + HEAD_DIM + d0;
            /* The transposed load leaves a lane one dim of the group's eight keys, two pages of
             * four slots: words x, y the first page's, z, w the second's. */
            uint4 vw = __builtin_bit_cast(uint4, load_tr_b128((const uint16_t*)a.kv + off));
            if constexpr (HID) {
                const uint32_t e0 = r4d_usel_pick(ids, kg * (8 / BS));
                const uint32_t e1 = r4d_usel_pick(ids, kg * (8 / BS) + 1);
                vw = make_uint4(vw.x & seen2(e0, 0), vw.y & seen2(e0, 2),
                                vw.z & seen2(e1, 0), vw.w & seen2(e1, 2));
            }
            o.v[r] = vw;
        }
    };
    auto store = [&](int ti, const Regs& o) {
        uint16_t* kt = sK + (ti & 1) * KSZ;
        uint16_t* vt = sV16 + (ti & 1) * VSZ;
        #pragma unroll
        for (int p = 0; p < SKP; ++p) {
            const int i = sid + p * (SW * 32);
            const int key = i / (HEAD_DIM / 8), ch = i - key * (HEAD_DIM / 8);
            const uint4 r = o.k[p];
            uint16_t* kd = &kt[key * KSTR + ch * 8];
            *(uint2*)(kd)     = make_uint2(D::from_bf16w(r.x), D::from_bf16w(r.y));
            *(uint2*)(kd + 4) = make_uint2(D::from_bf16w(r.z), D::from_bf16w(r.w));
        }
        #pragma unroll
        for (int r = 0; r < SVP; ++r) {
            constexpr int DG = HEAD_DIM / 32;
            const int idx = sw + r * SW;
            const int kg = idx / DG, dg = idx - kg * DG;
            const int d0 = dg * 32 + vg * 8;
            const uint4 v = o.v[r];
            uint16_t* dst = &vt[(d0 + vj) * VSTR + kg * 8];
            *(uint2*)(dst)     = make_uint2(D::from_bf16w(v.x), D::from_bf16w(v.y));
            *(uint2*)(dst + 4) = make_uint2(D::from_bf16w(v.z), D::from_bf16w(v.w));
        }
    };

    if constexpr (NS > 0) {
        if (warp >= G) {
            /* A stager: ahead[i] is tile ti + 1 + i in flight, `nxt` the ids of the tile fetched
             * next. */
            Regs ahead[kUselAhead];
            uint4 nxt;
            {
                Regs t0;
                fetch(ids_of(0), t0);
                #pragma unroll
                for (int i = 0; i < kUselAhead; ++i) fetch(ids_of(1 + i), ahead[i]);
                nxt = ids_of(1 + kUselAhead);
                store(0, t0);
            }
            lds_barrier();
            for (int ti = 0; ti < ntiles; ++ti) {
                if (ti + 1 < ntiles) store(ti + 1, ahead[0]);
                #pragma unroll
                for (int i = 0; i + 1 < kUselAhead; ++i) ahead[i] = ahead[i + 1];
                /* Past the union the table is padded for kUselPad pages; no further. */
                if (ti + 1 + kUselAhead < ntiles + 2) fetch(nxt, ahead[kUselAhead - 1]);
                if (ti + 2 + kUselAhead < ntiles + 2) nxt = ids_of(ti + 2 + kUselAhead);
                lds_barrier();
            }
            return;
        }
    }

    /* ---- a wave with a query */
    const int qpos = g * G + warp;
    /* `rowq`: the wave's query exists (its copies behind the heads read the query's keys too, as
     * the decode kernel's do); `live`: the row is one of its heads. */
    const bool rowq = qpos < a.num_seqs;
    const bool live = rowq && c < GQA;
    const int qrow = rowq ? qpos : a.num_seqs - 1;
    const int qhead = kvh * GQA + (c < GQA ? c : c - GQA);
    /* The row's partial page: its place in the union and the keys the row sees of it. */
    const uint32_t pr  = u.prow[qrow];
    const int      pu  = (int)(pr & 0xffffu);
    const uint32_t pcm = (1u << (pr >> 16)) - 1u;

    const float kdesc = a.k_descale ? a.k_descale[qrow * a.kv_heads + kvh] : 1.0f;
    const float mul   = a.scale * kdesc * 1.44269504089f;

    frag16 qf[NKS];
    {
        const uint16_t* qp = (const uint16_t*)a.q + ((size_t)qrow * a.q_heads + qhead) * HEAD_DIM;
        #pragma unroll
        for (int t = 0; t < NKS; ++t) {
            const uint2 l0 = *(const uint2*)(qp + 16 * t + 8 * h);
            const uint2 h0 = *(const uint2*)(qp + 16 * t + 8 * h + 4);
            qf[t] = D::mk(make_uint2(r4d_qcvt<D, 1, 1>(l0.x, mul), r4d_qcvt<D, 1, 1>(l0.y, mul)),
                          make_uint2(r4d_qcvt<D, 1, 1>(h0.x, mul), r4d_qcvt<D, 1, 1>(h0.y, mul)));
        }
    }

    float m_ref = -3.0e38f;
    float off   = PSHIFT + 3.0e38f;
    float l_i   = 0.0f;
    v8f acc[NKS];
    #pragma unroll
    for (int t = 0; t < NKS; ++t) acc[t] = (v8f){0,0,0,0,0,0,0,0};

    /* The row bits of a tile's four pages, sixteen a page: this lane's two pages are one word. */
    auto sig_of = [&](int ti) {
        const uint2 w = *(const uint2*)(sg + (size_t)ti * NB);
        return h ? w.y : w.x;
    };
    /* A tile's matmuls from its staged K and V. `sig`: the row bits of the lane's two pages;
     * `p0`: the place of the tile's first page in the union. */
    auto tile = [&](const uint16_t* kt, const uint16_t* vt, uint32_t sig, int p0) {
        /* The lane's eight keys are two pages: the row reads a page it lists, all of it or, of its
         * partial page, the keys it sees. */
        const int pa = p0 + 2 * h;
        uint32_t m8 = ((sig >> warp) & 1u) * 0x0fu | ((sig >> (16 + warp)) & 1u) * 0xf0u;
        if (pa == pu)     m8 &= pcm | 0xf0u;
        if (pa + 1 == pu) m8 &= (pcm << 4) | 0x0fu;
        if (!rowq) m8 = 0u;
        /* No head of this wave's query reads a key of the tile: its p are all zero. */
        if (!wave_any(m8 != 0u)) return;

        v8f s = (v8f){0,0,0,0,0,0,0,0};
        const uint16_t* krow = &kt[c * KSTR] + 8 * h;
        #pragma unroll
        for (int gk = 0; gk < NKS; gk += PF) {
            frag16 kb[PF];
            #pragma unroll
            for (int j = 0; j < PF; ++j) {
                const uint4 w = *(const uint4*)(krow + 16 * (gk + j));
                kb[j] = D::mk(make_uint2(w.x, w.y), make_uint2(w.z, w.w));
            }
            sched_barrier();
            #pragma unroll
            for (int j = 0; j < PF; ++j) s = D::wmma(kb[j], qf[gk + j], s);
        }
        float sc[8];
        #pragma unroll
        for (int e = 0; e < 8; ++e) sc[e] = s[e];
        if (wave_any(m8 != 0xffu)) {
            #pragma unroll
            for (int e = 0; e < 8; ++e) sc[e] = ((m8 >> e) & 1u) ? sc[e] : -INFINITY;
        }

        float smax = sc[0];
        #pragma unroll
        for (int e = 1; e < 8; ++e) smax = fmaxf(smax, sc[e]);
        /* Every head of the query moves to its max when one of them outgrows its reference, the
         * decode kernel's rule (the two half-waves hold one row). */
        const float rmax = fmaxf(smax, swap16(smax));
        if (wave_any(rmax > m_ref + PGROW)) {
            const float mnew = fmaxf(fmaxf(m_ref, rmax), -3.0e38f);
            const float alpha = ex2(m_ref - mnew);
            #pragma unroll
            for (int dt = 0; dt < NKS; ++dt)
                #pragma unroll
                for (int e = 0; e < 8; ++e) acc[dt][e] *= alpha;
            l_i *= alpha;
            m_ref = mnew;
            off   = PSHIFT - mnew;
        }

        uint32_t pw[4];
        #pragma unroll
        for (int e = 0; e < 8; e += 2) pw[e >> 1] = D::pk(ex2(sc[e] + off), ex2(sc[e + 1] + off));
        {
            const v2hf one = {(__fp16)1.0f, (__fp16)1.0f};
            #pragma unroll
            for (int e = 0; e < 4; ++e)
                l_i = __builtin_amdgcn_fdot2(__builtin_bit_cast(v2hf, pw[e]), one, l_i, false);
        }
        const frag16 p16 = D::mk(make_uint2(pw[0], pw[1]), make_uint2(pw[2], pw[3]));
        const uint16_t* vbase = &vt[c * VSTR] + 8 * h;
        #pragma unroll
        for (int gk = 0; gk < NKS; gk += PF) {
            frag16 vb[PF];
            #pragma unroll
            for (int j = 0; j < PF; ++j) {
                const uint4 w = *(const uint4*)(vbase + (gk + j) * 16 * VSTR);
                vb[j] = D::mk(make_uint2(w.x, w.y), make_uint2(w.z, w.w));
            }
            sched_barrier();
            #pragma unroll
            for (int j = 0; j < PF; ++j) acc[gk + j] = D::wmma(vb[j], p16, acc[gk + j]);
        }
    };

    /* The walk: tile ti's row bits were loaded two tiles before. Without stagers this wave also
     * fetches tile ti + 1 before its matmuls and stores it behind them; that tile's ids were
     * loaded a tile before, so the fetch waits on nothing. */
    uint32_t sg0 = sig_of(0), sg1 = sig_of(1);
    Regs nx;
    uint4 ids1;
    if constexpr (NS == 0) {
        fetch(ids_of(0), nx);
        ids1 = ids_of(1);
        store(0, nx);
    }
    lds_barrier();
    for (int ti = 0; ti < ntiles; ++ti) {
        const int cur = ti & 1;
        const uint32_t sg2 = sig_of(ti + 2);
        uint4 ids2;
        if constexpr (NS == 0) {
            fetch(ids1, nx);
            ids2 = ids_of(ti + 2);
        }
        tile(sK + cur * KSZ, sV16 + cur * VSZ, sg0, ti * NB);
        if constexpr (NS == 0) {
            if (ti + 1 < ntiles) store(ti + 1, nx);
            ids1 = ids2;
        }
        lds_barrier();
        sg0 = sg1;
        sg1 = sg2;
    }

    if (!live) return;
    /* The decode kernel's partial (r4d_attn_decode_body, PF16) as the row's one segment: O / l in
     * f16, then m and l. */
    const float lrow = l_i + swap16_fi(l_i);
    const size_t idx  = (size_t)qpos * a.q_heads + qhead;
    const size_t nrow = (size_t)a.num_seqs * a.q_heads;
    uint16_t* so = (uint16_t*)a.scratch;
    float* sm = (float*)(so + nrow * HEAD_DIM);
    float* sl = sm + nrow;
    uint16_t* op = so + idx * HEAD_DIM;
    const float inv = (lrow > 0.0f) ? (1.0f / lrow) : 0.0f;
    #pragma unroll
    for (int dt = 0; dt < NKS; ++dt) {
        uint32_t tw[4];
        #pragma unroll
        for (int e = 0; e < 8; e += 2) tw[e >> 1] = D::pk(acc[dt][e] * inv, acc[dt][e + 1] * inv);
        *(uint4*)(op + dt * 16 + 8 * h) = make_uint4(tw[0], tw[1], tw[2], tw[3]);
    }
    if (h == 0) { sm[idx] = m_ref; sl[idx] = lrow; }
}

/* The decode kernel for the tiles the builder flagged: launch row `blockIdx.z` runs exactly as the
 * decode launch would run it, the others return. */
template <int NWARPS, int TILE, int HEAD_DIM, int GQA, int BS, int KVP, int OPT>
__global__ __launch_bounds__(NWARPS * 32)
void r4d_attn_decode_usel_fb_kernel(const R4DArgs a, int splits, const int* fb, int group) {
    if (!fb[(int)blockIdx.z / group]) return;
    r4d_attn_decode_body<NWARPS, TILE, HEAD_DIM, GQA, BS, KVP, OPT, 0>(a, splits);
}

/* The form's scratch behind the launch's partials: offsets from `base` (the partials' bytes,
 * rounded up), sized for `rows` launch rows of page lists `W` wide in tiles of `G`. */
struct R4DUselLay {
    size_t ucnt, fb, prow, utab, usig, end;
    int tiles, upitch;
};
static inline R4DUselLay r4d_usel_layout(size_t base, int rows, int W, int G) {
    R4DUselLay l{};
    l.tiles  = (rows + G - 1) / G;
    l.upitch = G * W + kUselPad;                   /* the union and its padding */
    size_t o = (base + 63) & ~(size_t)63;
    l.ucnt  = o; o += (size_t)l.tiles * 4;
    l.fb    = o; o += (size_t)l.tiles * 4;
    l.prow  = o; o += (size_t)rows * 4;
    o = (o + 63) & ~(size_t)63;
    l.utab  = o; o += (size_t)l.tiles * l.upitch * 4;
    o = (o + 63) & ~(size_t)63;
    l.usig  = o; o += (size_t)l.tiles * l.upitch * 2;
    l.end   = (o + 63) & ~(size_t)63;
    return l;
}
