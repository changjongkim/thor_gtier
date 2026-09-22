// Drives every backend over identical workloads so the numbers compare.
#include "gtier.h"

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <future>
#include <vector>

#define CK(c) do { cudaError_t s_=(c); if (s_!=cudaSuccess) { \
  std::fprintf(stderr,"CUDA %d: %s\n",__LINE__,cudaGetErrorString(s_)); exit(1);} } while(0)

// `work` sets arithmetic intensity: extra FLOPs per byte touched.  Sweeping it
// finds where GPU time reaches I/O time, which is the only place overlapping
// the two can pay.
__global__ void consume(const uint8_t *const *ptrs, const size_t *lens, int n,
                        unsigned long long *sink, int work) {
    int r = blockIdx.x;
    if (r >= n) return;
    const uint8_t *p = ptrs[r];
    unsigned long long acc = 0;
    for (size_t i = (size_t)threadIdx.x * 64; i < lens[r]; i += (size_t)blockDim.x * 64) {
        float v = (float)p[i];
        for (int k = 0; k < work; ++k) v = fmaf(v, 1.000001f, 0.5f);
        acc += (unsigned long long)v;
    }
    if (acc) atomicAdd(sink, acc);
}

struct Opt {
    const char *path = "/home/thor/kcj/mmap_gpu/real32.bin";
    size_t item = 65536;
    int n = 64, iters = 128, slots = 0, policy = 0, admit = 2;
    size_t slot = 1u << 20;
    uint64_t span = 16ull << 30;  // region the ranges are drawn from
    int reuse = 0;                // 0 = fresh offsets; k>0 = cycle over k blocks
    int only = -1;                // run a single backend (so the driver can
                                  // drop caches between them)
    int pipeline = 0;             // 1 = overlap the next fetch with this GPU batch
    int work = 0;                 // FLOPs per touched byte, to vary GPU time
};

// Each pipeline stage gets its own handle, so its window and ring are disjoint
// and a fetch can be in flight while the GPU consumes the previous batch.  That
// is what a real client does, and applying it uniformly keeps the comparison
// fair -- including for mmap-gpu, which cannot benefit because its I/O happens
// inside the kernel, not before it.
struct Stage {
    gtier *g = nullptr;
    const uint8_t **dp = nullptr;
    size_t *dl = nullptr;
    std::vector<gtier_range> rs;
    std::vector<void *> outs;
    cudaStream_t stream{};
};

static double run(gtier_backend b, const Opt &o, gtier_stats *agg) {
    const int nstage = o.pipeline ? 2 : 1;
    gtier_config cfg{};
    cfg.backend = b;
    cfg.slot_bytes = o.slot;
    cfg.slots = o.slots ? o.slots : std::max(o.n, 8);
    if (b == GTIER_BACKEND_GTIER && o.policy >= GTIER_CACHE_HYBRID)
        cfg.slots = std::max(cfg.slots, o.n * 2);
    cfg.queue_depth = cfg.slots;
    cfg.merge_gap = 0;
    cfg.cache_policy = o.policy;
    cfg.admit_after = o.admit;
    cfg.max_fetch_ranges = o.n;

    std::vector<Stage> st(nstage);
    for (int i = 0; i < nstage; ++i) {
        st[i].g = gtier_open(o.path, &cfg);
        if (!st[i].g) { for (auto &s2 : st) if (s2.g) gtier_close(s2.g); return -1; }
        CK(cudaMallocManaged(&st[i].dp, o.n * sizeof(const uint8_t *)));
        CK(cudaMallocManaged(&st[i].dl, o.n * sizeof(size_t)));
        CK(cudaStreamCreate(&st[i].stream));
        st[i].rs.resize(o.n);
        st[i].outs.resize(o.n);
    }

    unsigned long long *sink; CK(cudaMalloc(&sink, sizeof(*sink)));
    CK(cudaMemset(sink, 0, sizeof(*sink)));

    uint64_t seed = 0x9E3779B97F4A7C15ull;
    auto rnd = [&] { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed; };
    auto gen = [&](std::vector<gtier_range> &rs, int it) {
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

    gen(st[0].rs, 0);
    gtier_fetch(st[0].g, st[0].rs.data(), o.n, st[0].outs.data());   // warm
    for (auto &s2 : st) gtier_reset_stats(s2.g);

    gtier_stats tot{};
    auto collect = [&](Stage &s2) {
        gtier_stats x; gtier_get_stats(s2.g, &x);
        tot.reads_issued += x.reads_issued; tot.bytes_useful += x.bytes_useful;
        tot.bytes_read += x.bytes_read; tot.cache_hits += x.cache_hits;
        tot.cache_misses += x.cache_misses; tot.admitted += x.admitted;
        tot.exact_fetches += x.exact_fetches;
    };

    auto t0 = std::chrono::steady_clock::now();
    if (nstage == 1) {
        for (int it = 0; it < o.iters; ++it) {
            Stage &s2 = st[0];
            gen(s2.rs, it);
            if (gtier_fetch(s2.g, s2.rs.data(), o.n, s2.outs.data()) != 0) goto fail;
            for (int i = 0; i < o.n; ++i) { s2.dp[i] = (const uint8_t *)s2.outs[i]; s2.dl[i] = o.item; }
            consume<<<o.n, 256, 0, s2.stream>>>(s2.dp, s2.dl, o.n, sink, o.work);
            if (cudaStreamSynchronize(s2.stream) != cudaSuccess) goto fail;
            collect(s2);
        }
    } else {
        // Stage 0's fetch runs while stage 1's kernel is on the GPU, and back.
        std::future<int> inflight;
        int cur = 0;
        gen(st[cur].rs, 0);
        int rc = gtier_fetch(st[cur].g, st[cur].rs.data(), o.n, st[cur].outs.data());
        if (rc != 0) goto fail;
        for (int it = 0; it < o.iters; ++it) {
            Stage &s2 = st[cur];
            int nxt = 1 - cur;
            if (it + 1 < o.iters) {
                gen(st[nxt].rs, it + 1);
                Stage *ns = &st[nxt];
                inflight = std::async(std::launch::async, [ns, &o] {
                    return gtier_fetch(ns->g, ns->rs.data(), o.n, ns->outs.data());
                });
            }
            for (int i = 0; i < o.n; ++i) { s2.dp[i] = (const uint8_t *)s2.outs[i]; s2.dl[i] = o.item; }
            consume<<<o.n, 256, 0, s2.stream>>>(s2.dp, s2.dl, o.n, sink, o.work);
            if (cudaStreamSynchronize(s2.stream) != cudaSuccess) goto fail;
            collect(s2);
            if (it + 1 < o.iters && inflight.get() != 0) goto fail;
            cur = nxt;
        }
    }
    {
        double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        tot.seconds = t;
        *agg = tot;
        cudaFree(sink);
        for (auto &s2 : st) { cudaFree(s2.dp); cudaFree(s2.dl);
                              cudaStreamDestroy(s2.stream); gtier_close(s2.g); }
        return (double)tot.bytes_useful / (1ull << 30) / t;
    }
fail:
    cudaFree(sink);
    for (auto &s2 : st) { if (s2.dp) cudaFree(s2.dp); if (s2.dl) cudaFree(s2.dl);
                          if (s2.g) gtier_close(s2.g); }
    return -1;
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
        else if (s == "--policy") o.policy = atoi(nx());
        else if (s == "--admit") o.admit = atoi(nx());
        else if (s == "--pipeline") o.pipeline = atoi(nx());
        else if (s == "--work") o.work = atoi(nx());
        else if (s == "--reuse") o.reuse = atoi(nx());
        else if (s == "--span") o.span = strtoull(nx(), 0, 10) << 30;
        else if (s == "--only") o.only = atoi(nx());
        else if (s == "--quiet") { /* header suppressed */ }
    }
    if (o.only < 0) {
        std::printf("file=%s item=%zuB n=%d slot=%zuKiB policy=%d reuse=%d iters=%d\n",
                    o.path, o.item, o.n, o.slot >> 10, o.policy, o.reuse, o.iters);
        std::printf("%-12s %10s %8s %8s %10s %8s %8s\n", "backend", "useful", "amp", "reads", "hitrate", "admit", "exact");
    }
    for (int b = 0; b < GTIER_BACKEND_COUNT; ++b) {
        if (o.only >= 0 && b != o.only) continue;
        gtier_stats s{};
        double bw = run((gtier_backend)b, o, &s);
        if (bw < 0) { std::printf("%-12s %10s\n", gtier_backend_name((gtier_backend)b), "n/a"); continue; }
        double amp = s.bytes_useful ? (double)s.bytes_read / s.bytes_useful : 0;
        double hr = (s.cache_hits + s.cache_misses)
                        ? 100.0 * s.cache_hits / (s.cache_hits + s.cache_misses) : 0;
        std::printf("%-12s %7.3f GiB/s %7.2fx %8.1f %9.1f%% %8.1f %8.1f\n",
                    gtier_backend_name((gtier_backend)b), bw, amp,
                    (double)s.reads_issued / o.iters, hr,
                    (double)s.admitted / o.iters, (double)s.exact_fetches / o.iters);
    }
    return 0;
}
