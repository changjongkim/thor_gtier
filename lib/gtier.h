// gTier -- a zero-copy NVMe-to-GPU data path for coherent SoCs.
//
// On a coherent SoC there is no separate device memory, so host pinned memory
// obtained with cudaHostAllocMapped is simultaneously (a) a valid O_DIRECT DMA
// target for the NVMe controller and (b) directly addressable by the GPU.  The
// drive writes into memory the GPU already reads: no bounce buffer, no
// cudaMemcpy, no page cache.  That path is unavailable on a discrete GPU
// without GPUDirect Storage, and GDS is absent on this platform (nvfs 0.0,
// compat mode).
//
// The library exists because the alternative the hardware offers -- letting the
// GPU fault on file-backed mmap -- is serviced a page at a time and reaches
// only 3-4% of the device.  See results/GRANULARITY.md.
//
// Planning follows two measured rules (results/GRANULARITY_FLOOR.md):
//   1. Never amplify an isolated range.  The device's bandwidth grows
//      sublinearly in request size, so reading extra bytes always costs more
//      than the bandwidth it buys.
//   2. Do coalesce neighbours.  Merging two ranges across a small gap trades
//      those gap bytes for one fewer request, which pays below a threshold
//      the library calibrates on the device it is running on.

#ifndef GTIER_H
#define GTIER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t slot_bytes;    // staging granularity; 0 -> 1 MiB (measured optimum)
    int    slots;         // window depth in slots; 0 -> 8
    int    queue_depth;   // io_uring depth; 0 -> slots
    size_t merge_gap;     // coalesce threshold in bytes; SIZE_MAX -> calibrate
} gtier_config;

typedef struct gtier gtier;

typedef struct {
    uint64_t off;         // byte offset in the file
    size_t   len;         // byte length
} gtier_range;

// Opens path with O_DIRECT and allocates the pinned, GPU-mapped window.
gtier *gtier_open(const char *path, const gtier_config *cfg);
void   gtier_close(gtier *g);

// Fetches n ranges and returns, in dev_out[i], a device pointer to range i.
// Pointers stay valid until the next gtier_fetch or gtier_release.
// Returns 0 on success, negative errno otherwise.
int  gtier_fetch(gtier *g, const gtier_range *r, int n, void **dev_out);
void gtier_release(gtier *g);

// Measures this device's bandwidth-size curve and returns the gap below which
// merging two requests beats issuing both.  Called automatically when
// merge_gap is SIZE_MAX.
size_t gtier_calibrate_merge_gap(gtier *g);

// Statistics for the last fetch.
typedef struct {
    uint64_t ranges_requested;
    uint64_t reads_issued;     // after coalescing
    uint64_t bytes_useful;     // what the caller asked for
    uint64_t bytes_read;       // what the device delivered
    double   seconds;
} gtier_stats;
void gtier_get_stats(const gtier *g, gtier_stats *out);

#ifdef __cplusplus
}
#endif
#endif  // GTIER_H
