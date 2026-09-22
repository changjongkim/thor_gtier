// Runs the real access pattern of weight-streaming inference over a real model.
//
// A GGUF model is a set of tensors at known offsets.  Decoding one token with
// the weights held on storage means reading, per layer and in order, that
// layer's tensors -- for a dense model all of them, for a mixture of experts
// only the experts the router picked.  That sequence of byte ranges is the
// trace; feeding it to each backend measures what inference would actually get.
#include "gguf.h"
#include "gtier.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(c) do { cudaError_t s_=(c); if (s_!=cudaSuccess) { \
  std::fprintf(stderr,"CUDA %d: %s\n",__LINE__,cudaGetErrorString(s_)); exit(1);} } while(0)

__global__ void consume(const uint8_t *const *p, const size_t *l, int n,
                        unsigned long long *sink) {
    int r = blockIdx.x; if (r >= n) return;
    unsigned long long acc = 0;
    for (size_t i = (size_t)threadIdx.x * 64; i < l[r]; i += (size_t)blockDim.x * 64)
        acc += p[r][i];
    if (acc) atomicAdd(sink, acc);
}

struct Shard { std::string path; gguf_model m; gtier *g = nullptr; };

// One read in the trace, tagged with the shard that holds it.
struct Req { int shard; gtier_range r; };

int main(int argc, char **argv) {
    std::vector<std::string> paths;
    int backend = 0, policy = 0, layers_per_batch = 1, tokens = 8, slots = 64;
    size_t slot = 1u << 20, split = 0;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i]; auto nx = [&] { return argv[++i]; };
        if (s == "--shard") paths.push_back(nx());
        else if (s == "--backend") backend = atoi(nx());
        else if (s == "--policy") policy = atoi(nx());
        else if (s == "--tokens") tokens = atoi(nx());
        else if (s == "--layers") layers_per_batch = atoi(nx());
        else if (s == "--slot") slot = strtoull(nx(), 0, 10);
        else if (s == "--slots") slots = atoi(nx());
        else if (s == "--split") split = strtoull(nx(), 0, 10);  // cap per read
    }
    if (paths.empty()) { std::fprintf(stderr, "--shard required\n"); return 1; }

    std::vector<Shard> sh(paths.size());
    uint64_t model_bytes = 0;
    int n_layers = 0;
    for (size_t i = 0; i < paths.size(); ++i) {
        sh[i].path = paths[i];
        if (gguf_load(paths[i].c_str(), &sh[i].m)) {
            std::fprintf(stderr, "parse failed: %s\n", paths[i].c_str()); return 1;
        }
        for (int t = 0; t < sh[i].m.n; ++t) model_bytes += sh[i].m.t[t].size;
        n_layers = std::max(n_layers, sh[i].m.n_layers);
    }

    // Trace: layer order, and within a layer the tensors in file order.  A read
    // larger than a slot is split, which is not amplification -- the bytes are
    // all wanted.
    std::vector<std::vector<Req>> by_layer(n_layers + 1);
    for (size_t i = 0; i < sh.size(); ++i)
        for (int t = 0; t < sh[i].m.n; ++t) {
            const gguf_tensor &tt = sh[i].m.t[t];
            if (!tt.size) continue;
            int L = tt.layer < 0 ? n_layers : tt.layer;
            // leave room for O_DIRECT alignment growth at both ends, so an
            // aligned range always still fits one slot
            size_t cap = split ? split : (slot - 2 * 4096);
            // Cut on the block grid as well as at the cap, so every range lies
            // inside one block and one slot -- what both the cached and the
            // exact paths require to return a single contiguous pointer.
            uint64_t pos = tt.offset, end = tt.offset + tt.size;
            while (pos < end) {
                uint64_t block_end = (pos / slot + 1) * slot;
                uint64_t stop = std::min({end, pos + (uint64_t)cap, block_end});
                by_layer[L].push_back({(int)i, {pos, (size_t)(stop - pos)}});
                pos = stop;
            }
        }

    // --slots is the window across the whole model, so split it over the
    // shards; each shard gets its own handle and a share of the slots.  Giving
    // every shard the full window would ask for shards x window bytes.
    int per_shard = (int)(slots / sh.size());
    if (per_shard < 8) per_shard = 8;
    gtier_config cfg{};
    cfg.backend = (gtier_backend)backend;
    cfg.slot_bytes = slot; cfg.slots = per_shard; cfg.queue_depth = per_shard;
    cfg.merge_gap = 0; cfg.cache_policy = policy; cfg.admit_after = 2;
    cfg.max_fetch_ranges = (policy == 4) ? std::max(8, per_shard / 16)
                                         : per_shard / 2;
    for (auto &s : sh) {
        s.g = gtier_open(s.path.c_str(), &cfg);
        if (!s.g) { std::fprintf(stderr, "gtier_open failed\n"); return 1; }
    }

    unsigned long long *sink; CK(cudaMalloc(&sink, sizeof(*sink)));
    CK(cudaMemset(sink, 0, sizeof(*sink)));
    const uint8_t **dp; size_t *dl;
    CK(cudaMallocManaged(&dp, per_shard * sizeof(*dp)));
    CK(cudaMallocManaged(&dl, per_shard * sizeof(*dl)));

    uint64_t useful = 0;
    gtier_stats agg{};
    auto t0 = std::chrono::steady_clock::now();
    for (int tok = 0; tok < tokens; ++tok) {
        for (int L = 0; L <= n_layers; L += layers_per_batch) {
            std::vector<Req> batch;
            for (int k = 0; k < layers_per_batch && L + k <= n_layers; ++k)
                batch.insert(batch.end(), by_layer[L + k].begin(), by_layer[L + k].end());
            for (size_t pos = 0; pos < batch.size(); ) {
                // group a run of same-shard requests, bounded by the window
                int s = batch[pos].shard;
                std::vector<gtier_range> rs;
                size_t end = pos;
                while (end < batch.size() && batch[end].shard == s &&
                       rs.size() < (size_t)cfg.max_fetch_ranges) {
                    rs.push_back(batch[end].r); ++end;
                }
                std::vector<void *> outs(rs.size());
                int rc = gtier_fetch(sh[s].g, rs.data(), (int)rs.size(), outs.data());
                if (rc != 0) {
                    size_t mx = 0; uint64_t mo = 0;
                    for (auto &x : rs) if (x.len > mx) { mx = x.len; mo = x.off; }
                    std::fprintf(stderr,
                        "fetch failed: %s  ranges=%zu maxlen=%zu off=%llu "
                        "(off%%4096=%llu, aligned span=%llu, slot=%zu)\n",
                        strerror(-rc), rs.size(), mx, (unsigned long long)mo,
                        (unsigned long long)(mo % 4096),
                        (unsigned long long)(((mo + mx + 4095) & ~4095ull) - (mo & ~4095ull)),
                        slot);
                    return 1;
                }
                for (size_t i = 0; i < rs.size(); ++i) {
                    dp[i] = (const uint8_t *)outs[i]; dl[i] = rs[i].len; useful += rs[i].len;
                }
                consume<<<(int)rs.size(), 256>>>(dp, dl, (int)rs.size(), sink);
                CK(cudaDeviceSynchronize());
                gtier_stats x; gtier_get_stats(sh[s].g, &x);
                agg.cache_hits += x.cache_hits; agg.cache_misses += x.cache_misses;
                agg.bytes_read += x.bytes_read; agg.admitted += x.admitted;
                agg.switches += x.switches;
                pos = end;
            }
        }
    }
    double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double win = (double)per_shard * sh.size() * slot / 1073741824.0;
    std::printf("%-11s policy=%d slot=%4zuKiB window=%6.2f GiB (%5.1f%% of model) | "
                "%6.3f GiB/s  %7.4f tok/s  hit=%.1f%% amp=%.2fx sw=%llu\n",
                gtier_backend_name((gtier_backend)backend), policy, slot >> 10,
                win, 100.0 * win / (model_bytes / 1073741824.0),
                (double)useful / (1ull << 30) / t, tokens / t,
                (agg.cache_hits + agg.cache_misses)
                    ? 100.0 * agg.cache_hits / (agg.cache_hits + agg.cache_misses) : 0.0,
                useful ? (double)agg.bytes_read / useful : 0.0,
                (unsigned long long)agg.switches);
    for (auto &s : sh) { gtier_close(s.g); gguf_free(&s.m); }
    return 0;
}
