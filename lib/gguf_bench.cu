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
#include <cmath>
#include <random>
#include <string>
#include <utility>
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
    // Decode steps serve `batch` sequences at once.  Each draws its own
    // experts, but a layer reads their union once -- which is the whole reason
    // batching helps a routed model: the bytes per token fall as the draws
    // start overlapping.
    int batch = 1;
    size_t slot = 1u << 20, split = 0;
    int async_io = 0;
    // MoE routing.  Expert weights are stored stacked -- one tensor per layer
    // holding all experts -- so a token that activates k of E experts touches
    // k/E of each stacked tensor, not all of it.  Reading the whole tensor, as
    // a dense sweep does, overstates the work by E/k.
    int experts = 0, active = 0;
    double skew = 0.0;   // Zipf exponent; 0 = uniform routing
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i]; auto nx = [&] { return argv[++i]; };
        if (s == "--shard") paths.push_back(nx());
        else if (s == "--backend") backend = atoi(nx());
        else if (s == "--policy") policy = atoi(nx());
        else if (s == "--tokens") tokens = atoi(nx());
        else if (s == "--batch") batch = atoi(nx());
        else if (s == "--layers") layers_per_batch = atoi(nx());
        else if (s == "--slot") slot = strtoull(nx(), 0, 10);
        else if (s == "--slots") slots = atoi(nx());
        else if (s == "--split") split = strtoull(nx(), 0, 10);  // cap per read
        else if (s == "--async") async_io = atoi(nx());
        else if (s == "--experts") experts = atoi(nx());
        else if (s == "--active") active = atoi(nx());
        else if (s == "--skew") skew = atof(nx());
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
    // Routed expert selection.  Real routing is skewed -- a few experts carry
    // much of the traffic -- so a Zipf draw is closer than a uniform one, and
    // the skew is what makes a resident cache worth having.
    std::mt19937_64 rng(20260923);
    std::vector<double> zipf;
    if (experts > 0) {
        zipf.resize(experts);
        double sum = 0;
        for (int i = 0; i < experts; ++i) {
            zipf[i] = skew > 0 ? 1.0 / std::pow(i + 1, skew) : 1.0;
            sum += zipf[i];
        }
        for (auto &v : zipf) v /= sum;
        for (int i = 1; i < experts; ++i) zipf[i] += zipf[i - 1];
    }
    auto draw_experts = [&](std::vector<int> &out) {
        out.clear();
        std::uniform_real_distribution<double> u(0.0, 1.0);
        while ((int)out.size() < active) {
            double x = u(rng);
            int e = (int)(std::lower_bound(zipf.begin(), zipf.end(), x) - zipf.begin());
            if (e >= experts) e = experts - 1;
            if (std::find(out.begin(), out.end(), e) == out.end()) out.push_back(e);
        }
    };

    // Routing is per layer, not per tensor: a layer's gate/up/down for one
    // expert are read together or not at all.  And it is redrawn for every
    // token, because that is what a router does -- holding one draw for the
    // whole run would let the page cache serve every token after the first,
    // which measures RAM rather than storage.
    if (batch < 1) batch = 1;
    int steps = (tokens + batch - 1) / batch;
    std::vector<std::vector<std::vector<int>>> picked(steps);
    uint64_t union_sum = 0;
    for (int tk = 0; tk < steps; ++tk) {
        picked[tk].resize(n_layers + 1);
        if (experts > 0 && active > 0)
            for (int L = 0; L <= n_layers; ++L) {
                std::vector<int> u, one;
                for (int s2 = 0; s2 < batch; ++s2) {
                    draw_experts(one);
                    for (int e : one)
                        if (std::find(u.begin(), u.end(), e) == u.end())
                            u.push_back(e);
                }
                std::sort(u.begin(), u.end());
                union_sum += u.size();
                picked[tk][L] = u;
            }
    }

    std::vector<std::vector<std::vector<Req>>> by_tok_layer(steps);
    for (int tk = 0; tk < steps; ++tk) {
    std::vector<std::vector<Req>> &by_layer = by_tok_layer[tk];
    by_layer.resize(n_layers + 1);
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
            // A stacked expert tensor contributes only the slices the router
            // picked; everything else contributes whole.
            std::vector<std::pair<uint64_t, uint64_t>> spans;
            const std::vector<int> &pick = picked[tk][L];
            if (experts > 0 && active > 0 && tt.expert >= 0) {
                if (sh[i].m.stacked_experts) {
                    // one tensor holds every expert; read the routed slices
                    uint64_t per = tt.size / experts;
                    for (int e : pick)
                        spans.emplace_back(tt.offset + (uint64_t)e * per, per);
                } else if (std::find(pick.begin(), pick.end(), tt.expert)
                           != pick.end()) {
                    // one tensor per expert; read it whole, or not at all
                    spans.emplace_back(tt.offset, tt.size);
                }
            } else {
                spans.emplace_back(tt.offset, tt.size);
            }
            for (auto &sp : spans) {
                uint64_t pos = sp.first, end = sp.first + sp.second;
                while (pos < end) {
                    uint64_t block_end = (pos / slot + 1) * slot;
                    uint64_t stop = std::min({end, pos + (uint64_t)cap, block_end});
                    by_layer[L].push_back({(int)i, {pos, (size_t)(stop - pos)}});
                    pos = stop;
                }
            }
        }
    }

    // --slots is the window across the whole model, so split it over the
    // shards; each shard gets its own handle and a share of the slots.  Giving
    // every shard the full window would ask for shards x window bytes.
    int per_shard = (int)(slots / sh.size());
    if (per_shard < 32) per_shard = 32;
    // Tickets own disjoint slots, so an async handle needs the window split
    // GTIER_MAX_INFLIGHT ways on top of the per-shard split.
    // A ticket owns slots/GTIER_MAX_INFLIGHT of the window, so the async path
    // needs enough slots to divide; it must not silently enlarge the window,
    // or it would be compared against the other backends at more memory.
    if (async_io) per_shard = std::max(per_shard, 8 * GTIER_MAX_INFLIGHT);
    gtier_config cfg{};
    cfg.backend = (gtier_backend)backend;
    cfg.slot_bytes = slot; cfg.slots = per_shard; cfg.queue_depth = per_shard;
    cfg.merge_gap = 0; cfg.cache_policy = policy; cfg.admit_after = 2;
    // A ticket owns slots/GTIER_MAX_INFLIGHT of the window, so an async batch
    // can never exceed that.  PIN also needs scratch it reuses across waves.
    cfg.max_fetch_ranges = async_io ? std::max(4, per_shard / GTIER_MAX_INFLIGHT)
                          : (policy == 4) ? std::max(8, per_shard / 16)
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
    for (int tok = 0; tok < steps; ++tok) {
        for (int L = 0; L <= n_layers; L += layers_per_batch) {
            std::vector<Req> batch;
            for (int k = 0; k < layers_per_batch && L + k <= n_layers; ++k)
                batch.insert(batch.end(), by_tok_layer[tok][L + k].begin(),
                             by_tok_layer[tok][L + k].end());
            // Split the batch into per-shard runs first, so the async path can
            // have the next run in flight while this one is consumed.
            struct Run { int shard; std::vector<gtier_range> rs; };
            std::vector<Run> runs;
            for (size_t pos = 0; pos < batch.size(); ) {
                int sh_id = batch[pos].shard;
                Run rn; rn.shard = sh_id;
                size_t end = pos;
                while (end < batch.size() && batch[end].shard == sh_id &&
                       rn.rs.size() < (size_t)cfg.max_fetch_ranges) {
                    rn.rs.push_back(batch[end].r); ++end;
                }
                runs.push_back(std::move(rn));
                pos = end;
            }

            if (async_io) {
                // One run in flight while the previous one is consumed.
                std::vector<void *> outs(cfg.max_fetch_ranges);
                gtier_ticket tk[2];
                bool live[2] = {false, false};
                for (size_t i = 0; i <= runs.size(); ++i) {
                    int cur = (int)(i & 1), nxt = cur ^ 1;
                    if (i < runs.size()) {
                        int rc = gtier_submit(sh[runs[i].shard].g, runs[i].rs.data(),
                                              (int)runs[i].rs.size(), &tk[cur]);
                        if (rc != 0) {
                            std::fprintf(stderr, "submit failed: %s (ranges=%zu)\n",
                                         strerror(-rc), runs[i].rs.size());
                            return 1;
                        }
                        live[cur] = true;
                    }
                    if (i == 0) continue;            // nothing to collect yet
                    size_t done = i - 1;
                    int slot = (int)(done & 1);
                    if (!live[slot]) continue;
                    if (gtier_wait(sh[runs[done].shard].g, &tk[slot], outs.data()) != 0) {
                        std::fprintf(stderr, "async wait failed\n"); return 1;
                    }
                    live[slot] = false;
                    for (size_t k = 0; k < runs[done].rs.size(); ++k) {
                        dp[k] = (const uint8_t *)outs[k];
                        dl[k] = runs[done].rs[k].len;
                        useful += runs[done].rs[k].len;
                    }
                    consume<<<(int)runs[done].rs.size(), 256>>>(
                        dp, dl, (int)runs[done].rs.size(), sink);
                    CK(cudaDeviceSynchronize());
                    gtier_stats x; gtier_get_stats(sh[runs[done].shard].g, &x);
                    agg.bytes_read += x.bytes_read;
                }
                continue;
            }

            for (size_t ri = 0; ri < runs.size(); ++ri) {
                int s = runs[ri].shard;
                std::vector<gtier_range> &rs = runs[ri].rs;
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
            }
        }
    }
    double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double win = (double)per_shard * sh.size() * slot / 1073741824.0;
    // uniq is the mean number of distinct experts a layer had to read per
    // step: it is 'active' at batch 1 and climbs toward 'experts' as the draws
    // of a larger batch overlap, so bytes/token falls by batch/uniq.
    double uniq = (experts > 0 && steps) ? (double)union_sum / steps / (n_layers + 1) : 0;
    std::printf("%-11s pol=%d slot=%4zuKiB win=%6.2f GiB b=%-3d uniq=%5.1f | "
                "%6.3f GiB/s %7.4f tok/s %6.1f MiB/tok hit=%.1f%% amp=%.2fx\n",
                gtier_backend_name((gtier_backend)backend), policy, slot >> 10,
                win, batch, uniq,
                (double)useful / (1ull << 30) / t, (steps * batch) / t,
                (double)useful / (1ull << 20) / (steps * batch),
                (agg.cache_hits + agg.cache_misses)
                    ? 100.0 * agg.cache_hits / (agg.cache_hits + agg.cache_misses) : 0.0,
                useful ? (double)agg.bytes_read / useful : 0.0);
    for (auto &s : sh) { gtier_close(s.g); gguf_free(&s.m); }
    return 0;
}
