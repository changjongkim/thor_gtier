// Exercises the library on the sparse-fetch case and reports what the caller
// actually gets: useful bandwidth, read amplification, and how many device
// requests the planner issued.  The GPU consumes the data through the same
// pinned window the drive wrote into, so nothing is copied.
#include "gtier.h"

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(c) do { cudaError_t s_=(c); if (s_!=cudaSuccess) { \
  std::fprintf(stderr,"CUDA %d: %s\n",__LINE__,cudaGetErrorString(s_)); exit(1);} } while(0)

// Touch every 64 B line of each fetched range, proving the GPU reads the
// window directly.
__global__ void consume(const uint8_t *const *ptrs, const size_t *lens, int n,
                        unsigned long long *sink) {
    int r = blockIdx.x;
    if (r >= n) return;
    const uint8_t *p = ptrs[r];
    unsigned long long acc = 0;
    for (size_t i = threadIdx.x * 64; i < lens[r]; i += (size_t)blockDim.x * 64)
        acc += p[i];
    if (acc) atomicAdd(sink, acc);
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/home/thor/kcj/mmap_gpu/real32.bin";
    size_t item = argc > 2 ? strtoull(argv[2], 0, 10) : 65536;
    int    n    = argc > 3 ? atoi(argv[3]) : 8;
    size_t gap  = argc > 4 ? strtoull(argv[4], 0, 10) : SIZE_MAX;
    int    iters= argc > 5 ? atoi(argv[5]) : 512;
    int    clustered = argc > 6 ? atoi(argv[6]) : 0;  // 1 = neighbours, tests merging

    gtier_config cfg{};
    cfg.slot_bytes = 1u << 20;
    cfg.slots = n > 16 ? n : 16;
    cfg.queue_depth = cfg.slots;
    cfg.merge_gap = gap;
    gtier *g = gtier_open(path, &cfg);
    if (!g) { std::fprintf(stderr, "gtier_open failed\n"); return 1; }

    unsigned long long *sink; CK(cudaMalloc(&sink, sizeof(*sink)));
    CK(cudaMemset(sink, 0, sizeof(*sink)));
    const uint8_t **d_ptrs; size_t *d_lens;
    CK(cudaMallocManaged(&d_ptrs, n * sizeof(*d_ptrs)));
    CK(cudaMallocManaged(&d_lens, n * sizeof(*d_lens)));

    std::vector<gtier_range> rs(n);
    std::vector<void *> outs(n);
    unsigned long long seed = 0x9E3779B97F4A7C15ull;
    auto rnd = [&] { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed; };

    // one untimed pass so the ring and window are warm
    for (int i = 0; i < n; ++i) rs[i] = {(rnd() % (16ull << 30)) & ~4095ull, item};
    gtier_fetch(g, rs.data(), n, outs.data());

    uint64_t useful = 0, read = 0, reqs = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
        if (clustered) {
            uint64_t base = (rnd() % (16ull << 30)) & ~4095ull;
            for (int i = 0; i < n; ++i) rs[i] = {base + (uint64_t)i * (item * 2), item};
        } else {
            for (int i = 0; i < n; ++i) rs[i] = {(rnd() % (16ull << 30)) & ~4095ull, item};
        }
        if (gtier_fetch(g, rs.data(), n, outs.data()) != 0) {
            std::fprintf(stderr, "fetch failed\n"); return 1;
        }
        for (int i = 0; i < n; ++i) { d_ptrs[i] = (const uint8_t *)outs[i]; d_lens[i] = item; }
        consume<<<n, 256>>>(d_ptrs, d_lens, n, sink);
        CK(cudaDeviceSynchronize());
        gtier_stats s; gtier_get_stats(g, &s);
        useful += s.bytes_useful; read += s.bytes_read; reqs += s.reads_issued;
    }
    double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    char gapstr[32];
    if (gap == SIZE_MAX) std::snprintf(gapstr, sizeof(gapstr), "auto");
    else std::snprintf(gapstr, sizeof(gapstr), "%zuK", gap >> 10);
    std::printf("item=%7zuB n=%2d %-9s gap=%-9s | reqs/fetch=%4.1f amp=%5.2fx "
                "useful=%6.3f GiB/s\n",
                item, n, clustered ? "clustered" : "scattered", gapstr,
                (double)reqs / iters, (double)read / (double)useful,
                (double)useful / (1ull << 30) / t);
    gtier_close(g);
    return 0;
}
