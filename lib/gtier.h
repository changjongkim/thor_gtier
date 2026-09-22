// gTier -- a zero-copy NVMe-to-GPU data path for coherent SoCs, plus the
// baselines it is measured against, behind one interface.
//
// On a coherent SoC there is no separate device memory, so pinned memory from
// cudaHostAllocMapped is simultaneously (a) a valid O_DIRECT DMA target for the
// NVMe controller and (b) directly addressable by the GPU.  The drive writes
// into memory the GPU already reads: no bounce buffer, no cudaMemcpy, no page
// cache.  That is unavailable on a discrete GPU without GPUDirect Storage, and
// GDS is absent here (nvfs 0.0, compat mode).
//
// Every backend below answers the same request -- "give me these byte ranges,
// GPU-addressable" -- so throughput, amplification and request counts compare
// directly on identical workloads.

#ifndef GTIER_H
#define GTIER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GTIER_BACKEND_GTIER = 0,  // pinned window + io_uring + O_DIRECT  (ours)
    GTIER_BACKEND_MMAP_GPU,   // mmap; the GPU faults                 (the OS path)
    GTIER_BACKEND_MMAP_CPU,   // mmap; CPU threads fault, GPU then reads
    GTIER_BACKEND_PREAD_COPY, // pread to host, cudaMemcpy to device  (FlexGen/ZeRO pattern)
    GTIER_BACKEND_CUFILE,     // cuFile / GPUDirect Storage           (NVIDIA)
    GTIER_BACKEND_UVM,        // cudaMallocManaged + prefetch         (DeepUM pattern)
    GTIER_BACKEND_COUNT
} gtier_backend;

const char *gtier_backend_name(gtier_backend b);

typedef struct {
    gtier_backend backend;
    size_t slot_bytes;    // staging granularity; 0 -> 1 MiB
    int    slots;         // window depth in slots; 0 -> 8
    int    queue_depth;   // io_uring depth; 0 -> slots
    size_t merge_gap;     // coalesce threshold; SIZE_MAX -> calibrate empirically

    // Residency cache.  Caching needs fixed-size blocks, which means fetching
    // bytes the caller did not ask for -- exactly what rule 1 forbids for
    // one-shot access.  With reuse that amplification amortises, so the two
    // regimes have a crossover this flag lets us measure.
    // Residency policy.
    //   GTIER_CACHE_NONE   every range fetched exactly; no reuse, no amplification
    //   GTIER_CACHE_BLOCK  everything goes through fixed blocks; reuse repays the
    //                      amplification, and a miss pays it for nothing
    //   GTIER_CACHE_HYBRID hits served from resident blocks, misses fetched
    //                      exactly and admitted only once seen twice.  Measured
    //                      to be the right rule: caching swings 240x on whether
    //                      the working set fits, and the two failure modes are
    //                      opposite, so neither pure policy is safe.
    int    cache_policy;  // gtier_cache_policy
    int    cache_blocks;  // resident blocks; 0 -> use all slots
    int    admit_after;   // HYBRID: accesses to a block before admitting it; 0 -> 2

    // The window has to hold the resident set and, at the same time, the misses
    // one fetch can produce -- a miss under HYBRID lands in a scratch slot that
    // no resident block may be evicted for.  Declare the largest fetch you will
    // issue and the library reserves scratch accordingly.
    int    max_fetch_ranges;  // 0 -> a quarter of the window

    // Ablation: after fetching, copy each slot into a separate device buffer,
    // which is exactly the extra step cuFile and pread+copy take.  Running
    // gtier with and without it isolates what that copy costs in situ, instead
    // of inferring it from a standalone cudaMemcpy microbenchmark.
    int    ablate_copy;
} gtier_config;

typedef enum {
    GTIER_CACHE_NONE = 0,
    GTIER_CACHE_BLOCK,
    GTIER_CACHE_HYBRID,
    // The two pure policies fail in opposite directions and the boundary
    // between them is sharp -- 240x -- so what matters is not which block to
    // admit but which regime you are in.  ADAPTIVE watches the hit rate and
    // switches, with hysteresis so it does not oscillate at the edge.
    GTIER_CACHE_ADAPTIVE,
    // Weight streaming sweeps the whole model once per token, which is LRU's
    // pathological case: with a window smaller than the model every block is
    // evicted before it comes round again, so partial residency buys nothing.
    // PIN fills the window once and then stops evicting, turning a window of
    // W/N of the model into a W/N hit rate instead of zero.
    GTIER_CACHE_PIN,
} gtier_cache_policy;

typedef struct gtier gtier;

typedef struct {
    uint64_t off;
    size_t   len;
} gtier_range;

gtier *gtier_open(const char *path, const gtier_config *cfg);
void   gtier_close(gtier *g);

// Fetches n ranges; dev_out[i] receives a device pointer to range i.  Pointers
// remain valid until the next fetch (or, with a cache, until evicted).
int  gtier_fetch(gtier *g, const gtier_range *r, int n, void **dev_out);

size_t gtier_calibrate_merge_gap(gtier *g);

typedef struct {
    uint64_t ranges_requested;
    uint64_t reads_issued;
    uint64_t bytes_useful;
    uint64_t bytes_read;      // what the device actually delivered
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t admitted;        // blocks promoted into the cache
    uint64_t exact_fetches;   // misses served without amplification
    uint64_t switches;        // ADAPTIVE: regime changes
    double   seconds;
} gtier_stats;
void gtier_get_stats(const gtier *g, gtier_stats *out);
void gtier_reset_stats(gtier *g);

#ifdef __cplusplus
}
#endif
#endif  // GTIER_H
