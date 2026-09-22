// Drives every backend over identical workloads so the numbers compare.
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

__global__ void consume(const uint8_t *const *ptrs, const size_t *lens, int n,
                        unsigned long long *sink) {
    int r = blockIdx.x;
    if (r >= n) return;
    const uint8_t *p = ptrs[r];
    unsigned long long acc = 0;
    for (size_t i = (size_t)threadIdx.x * 64; i < lens[r]; i += (size_t)blockDim.x * 64)
        acc += p[i];
    if (acc) atomicAdd(sink, acc);
}

struct Opt {
    const char *path = "/home/thor/kcj/mmap_gpu/real32.bin";
    size_t item = 65536;
    int n = 64, iters = 128, cache = 0, slots = 0;
    size_t slot = 1u << 20;
    uint64_t span = 16ull << 30;  // region the ranges are drawn from
    int reuse = 0;                // 0 = fresh offsets; k>0 = cycle over k blocks
    int only = -1;                // run a single backend (so the driver can
                                  // drop caches between them)
};

static double run(gtier_backend b, const Opt &o, gtier_stats *agg) {
    gtier_config cfg{};
    cfg.backend = b;
    cfg.slot_bytes = o.slot;
    cfg.slots = o.slots ? o.slots : std::max(o.n, 8);
    cfg.queue_depth = cfg.slots;
    cfg.merge_gap = 0;           // measured: never merge
    cfg.cache_blocks = o.cache;
    gtier *g = gtier_open(o.path, &cfg);
    if (!g) return -1;

    unsigned long long *sink; CK(cudaMalloc(&sink, sizeof(*sink)));
    CK(cudaMemset(sink, 0, sizeof(*sink)));
    const uint8_t **dp; size_t *dl;
    CK(cudaMallocManaged(&dp, o.n * sizeof(*dp)));
    CK(cudaMallocManaged(&dl, o.n * sizeof(*dl)));

    std::vector<gtier_range> rs(o.n);
    std::vector<void *> outs(o.n);
    uint64_t seed = 0x9E3779B97F4A7C15ull;
    auto rnd = [&] { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed; };
    auto gen = [&](int it) {
        for (int i = 0; i < o.n; ++i) {
            uint64_t off;
            if (o.reuse > 0) {
                uint64_t blkidx = (uint64_t)((it * o.n + i) % o.reuse);
                off = blkidx * o.slot + (rnd() % (o.slot - o.item));
            } else {
                off = rnd() % (o.span - o.item);
            }
            rs[i] = {off & ~(uint64_t)4095, o.item};
        }
    };

    gen(0); gtier_fetch(g, rs.data(), o.n, outs.data());   // warm
    gtier_reset_stats(g);

    gtier_stats tot{};
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < o.iters; ++it) {
        gen(it);
        if (gtier_fetch(g, rs.data(), o.n, outs.data()) != 0) { gtier_close(g); return -1; }
        for (int i = 0; i < o.n; ++i) { dp[i] = (const uint8_t *)outs[i]; dl[i] = o.item; }
        consume<<<o.n, 256>>>(dp, dl, o.n, sink);
        if (cudaDeviceSynchronize() != cudaSuccess) { gtier_close(g); return -1; }
        gtier_stats s; gtier_get_stats(g, &s);
        tot.reads_issued += s.reads_issued; tot.bytes_useful += s.bytes_useful;
        tot.bytes_read += s.bytes_read; tot.cache_hits += s.cache_hits;
        tot.cache_misses += s.cache_misses;
    }
    double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    tot.seconds = t;
    *agg = tot;
    cudaFree(sink); cudaFree(dp); cudaFree(dl);
    gtier_close(g);
    return (double)tot.bytes_useful / (1ull << 30) / t;
}

int main(int argc, char **argv) {
    Opt o;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i]; auto nx = [&] { return argv[++i]; };
        if (s == "--file") o.path = nx();
        else if (s == "--item") o.item = strtoull(nx(), 0, 10);
        else if (s == "--n") o.n = atoi(nx());
        else if (s == "--iters") o.iters = atoi(nx());
        else if (s == "--slot") o.slot = strtoull(nx(), 0, 10);
        else if (s == "--cache") o.cache = atoi(nx());
        else if (s == "--reuse") o.reuse = atoi(nx());
        else if (s == "--span") o.span = strtoull(nx(), 0, 10) << 30;
        else if (s == "--only") o.only = atoi(nx());
        else if (s == "--quiet") { /* header suppressed */ }
    }
    if (o.only < 0) {
        std::printf("file=%s item=%zuB n=%d slot=%zuKiB cache=%d reuse=%d iters=%d\n",
                    o.path, o.item, o.n, o.slot >> 10, o.cache, o.reuse, o.iters);
        std::printf("%-12s %10s %8s %8s %10s\n", "backend", "useful", "amp", "reads", "hitrate");
    }
    for (int b = 0; b < GTIER_BACKEND_COUNT; ++b) {
        if (o.only >= 0 && b != o.only) continue;
        gtier_stats s{};
        double bw = run((gtier_backend)b, o, &s);
        if (bw < 0) { std::printf("%-12s %10s\n", gtier_backend_name((gtier_backend)b), "n/a"); continue; }
        double amp = s.bytes_useful ? (double)s.bytes_read / s.bytes_useful : 0;
        double hr = (s.cache_hits + s.cache_misses)
                        ? 100.0 * s.cache_hits / (s.cache_hits + s.cache_misses) : 0;
        std::printf("%-12s %7.3f GiB/s %7.2fx %8.1f %9.1f%%\n",
                    gtier_backend_name((gtier_backend)b), bw, amp,
                    (double)s.reads_issued / o.iters, hr);
    }
    return 0;
}
