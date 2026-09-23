// Serve a MoE model from storage under a fixed memory budget, replaying the
// routing the model actually produced, and report the two numbers a serving
// system is judged by: how long the prefill waits on I/O, and how long each
// decoded token waits on I/O.
//
// Everything here exists because of one measurement: with a good residency
// set, decode stops being I/O bound and prefill does not.  A policy is
// therefore judged on what it does to prefill, and on whether prefill wrecks
// what decode needs -- which is exactly what an LRU does, since prefill reads
// the union once and evicts the popular experts on the way through.
#include "gtier.h"
#include "gguf.h"
#include "serve.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define CK(x) do{cudaError_t s_=(x); if(s_!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA %d: %s\n",__LINE__,cudaGetErrorString(s_)); exit(1);} }while(0)

// ------------------------------------------------------------------ routing
struct Request {
    std::string name;
    int n_prefill = 0, n_decode = 0;
    std::vector<int16_t> prefill, decode;   // [token][layer][topk]
};
struct Trace {
    int n_layers = 0, n_experts = 0, topk = 0;
    std::vector<Request> req;
};

static bool load_trace(const char *path, Trace &t) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    char magic[4]; uint32_t L, E, K, N;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "GRT1", 4)) { fclose(f); return false; }
    if (fread(&L,4,1,f)!=1||fread(&E,4,1,f)!=1||fread(&K,4,1,f)!=1||fread(&N,4,1,f)!=1) {
        fclose(f); return false;
    }
    t.n_layers=(int)L; t.n_experts=(int)E; t.topk=(int)K;
    for (uint32_t i = 0; i < N; ++i) {
        Request r; uint32_t nl, np, nd;
        if (fread(&nl,4,1,f)!=1) { fclose(f); return false; }
        r.name.resize(nl);
        if (fread(&r.name[0],1,nl,f)!=nl) { fclose(f); return false; }
        if (fread(&np,4,1,f)!=1||fread(&nd,4,1,f)!=1) { fclose(f); return false; }
        r.n_prefill=(int)np; r.n_decode=(int)nd;
        r.prefill.resize((size_t)np*L*K); r.decode.resize((size_t)nd*L*K);
        if (fread(r.prefill.data(),2,r.prefill.size(),f)!=r.prefill.size()) { fclose(f); return false; }
        if (fread(r.decode.data(),2,r.decode.size(),f)!=r.decode.size()) { fclose(f); return false; }
        t.req.push_back(std::move(r));
    }
    fclose(f);
    return true;
}

// ------------------------------------------------------------------- model
// One routed unit = one (layer, expert): the three projections are read
// together or not at all, so they are held together too.
struct Unit {
    std::vector<std::pair<int, gtier_range>> parts;   // (shard, range)
    uint64_t bytes = 0;
};
struct Shard { std::string path; gguf_model m; gtier *g = nullptr; };

// ------------------------------------------------------------- residency
// The arena is real memory: a policy that claims to hold 40 GiB pays for it,
// which is the whole point of comparing policies under one budget.
struct Arena {
    uint8_t *base = nullptr;
    uint64_t cap = 0, used = 0;
    std::unordered_map<int, uint64_t> at;    // unit id -> offset
    bool put(int id, uint64_t bytes) {
        if (used + bytes > cap) return false;
        at[id] = used; used += bytes;
        return true;
    }
    bool has(int id) const { return at.count(id) != 0; }
};

static const char *policy_name(int p) {
    switch (p) {
        case SERVE_NONE: return "none";
        case SERVE_LRU: return "lru";
        case SERVE_LRU_PHASE: return "lru+phase";
        case SERVE_PERLAYER: return "per-layer";
        case SERVE_PREFIX: return "prefix";
    }
    return "?";
}

__global__ void touch(const uint8_t *const *p, const size_t *l, int n,
                      unsigned long long *sink) {
    int r = blockIdx.x; if (r >= n) return;
    unsigned long long acc = 0;
    for (size_t i=(size_t)threadIdx.x*64; i<l[r]; i+=(size_t)blockDim.x*64) acc += p[r][i];
    if (acc) atomicAdd(sink, acc);
}

int main(int argc, char **argv) {
    std::vector<std::string> paths;
    const char *trace_path = "results/SCOPE/routing.bin";
    double budget_gib = 24.0;        // total memory for residency + window
    double window_gib = 0.5;         // staging window (saturates here, sec 4.14)
    int policy = SERVE_PERLAYER, repeats = 1, prefix_tokens = 30, max_decode = 0;
    size_t slot = 4u << 20;
    bool verbose = false;
    int backend = GTIER_BACKEND_GTIER;
    double path_overhead_gib = 0.0;

    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto nx = [&]{ return argv[++i]; };
        if (s=="--shard") paths.push_back(nx());
        else if (s=="--trace") trace_path = nx();
        else if (s=="--budget") budget_gib = atof(nx());
        else if (s=="--window") window_gib = atof(nx());
        else if (s=="--policy") policy = atoi(nx());
        else if (s=="--repeats") repeats = atoi(nx());
        else if (s=="--prefix-tokens") prefix_tokens = atoi(nx());
        else if (s=="--slot") slot = strtoull(nx(),0,10);
        else if (s=="--max-decode") max_decode = atoi(nx());
        else if (s=="--backend") backend = atoi(nx());
        // The data path's own footprint is charged against the same budget:
        // whatever it holds beyond the declared window is memory residency
        // does not get, which is the whole argument of sec 1.6.
        else if (s=="--path-overhead") path_overhead_gib = atof(nx());
        else if (s=="--verbose") verbose = true;
    }
    if (paths.empty()) { std::fprintf(stderr,"--shard required\n"); return 1; }

    Trace tr;
    if (!load_trace(trace_path, tr)) { std::fprintf(stderr,"trace load failed\n"); return 1; }

    std::vector<Shard> sh(paths.size());
    uint64_t model_bytes = 0;
    for (size_t i=0;i<paths.size();++i) {
        sh[i].path = paths[i];
        if (gguf_load(paths[i].c_str(), &sh[i].m)) {
            std::fprintf(stderr,"parse failed: %s\n", paths[i].c_str()); return 1;
        }
        for (int t=0;t<sh[i].m.n;++t) model_bytes += sh[i].m.t[t].size;
    }

    // (layer, expert) -> unit.  Everything that is not a routed expert is
    // read on every token regardless, so an engine keeps it resident; it is
    // charged to the budget up front rather than streamed.
    int L = tr.n_layers, E = tr.n_experts, K = tr.topk;
    std::vector<Unit> unit((size_t)L*E);
    uint64_t always_bytes = 0;
    for (size_t i=0;i<sh.size();++i)
        for (int t=0;t<sh[i].m.n;++t) {
            const gguf_tensor &tt = sh[i].m.t[t];
            if (!tt.size) continue;
            if (tt.layer>=0 && tt.layer<L && tt.expert>=0 && tt.expert<E) {
                Unit &u = unit[(size_t)tt.layer*E + tt.expert];
                // split on the slot grid so every piece fits one slot
                uint64_t pos = tt.offset, end = tt.offset + tt.size;
                size_t cap = slot - 2*4096;
                while (pos < end) {
                    uint64_t blk = (pos/slot + 1)*slot;
                    uint64_t stop = std::min({end, pos+(uint64_t)cap, blk});
                    u.parts.push_back({(int)i, {pos, (size_t)(stop-pos)}});
                    pos = stop;
                }
                u.bytes += tt.size;
            } else {
                always_bytes += tt.size;
            }
        }
    uint64_t unit_bytes = 0; int n_units = 0;
    for (auto &u : unit) if (u.bytes) { unit_bytes += u.bytes; ++n_units; }
    if (!n_units) { std::fprintf(stderr,"no expert units found\n"); return 1; }
    uint64_t per_unit = unit_bytes / n_units;

    // --- budget split -----------------------------------------------------
    uint64_t B = (uint64_t)(budget_gib * (1ull<<30));
    uint64_t W = (uint64_t)(window_gib * (1ull<<30));
    if (always_bytes + W >= B) {
        std::fprintf(stderr,"budget too small: always-resident %.2f GiB + window %.2f GiB\n",
                     always_bytes/1073741824.0, W/1073741824.0);
        return 1;
    }
    uint64_t OV = (uint64_t)(path_overhead_gib * (1ull<<30));
    if (always_bytes + W + OV >= B) {
        std::fprintf(stderr,"budget too small once the data path's %.2f GiB is charged\n",
                     OV/1073741824.0);
        return 1;
    }
    uint64_t R = (policy == SERVE_NONE) ? 0 : B - always_bytes - W - OV;

    // --- per-layer popularity from the trace's decode phase ----------------
    // Ordering must be per layer: a layer's top half carries ~90% of its own
    // traffic while the same fraction aggregated across layers carries ~58%,
    // so a global order cannot see the structure (sec 2.3a).
    std::vector<uint64_t> cnt((size_t)L*E, 0);
    for (auto &r : tr.req)
        for (int t=0;t<r.n_decode;++t)
            for (int l=0;l<L;++l)
                for (int k=0;k<K;++k) {
                    int e = r.decode[((size_t)t*L + l)*K + k];
                    if (e>=0) cnt[(size_t)l*E+e]++;
                }
    std::vector<int> order;                      // unit ids, best first
    {
        std::vector<std::pair<double,int>> sc;
        for (int l=0;l<L;++l) {
            uint64_t tot=0; for (int e=0;e<E;++e) tot += cnt[(size_t)l*E+e];
            if (!tot) tot = 1;
            for (int e=0;e<E;++e) {
                int id=(int)((size_t)l*E+e);
                if (unit[id].bytes) sc.push_back({(double)cnt[id]/tot, id});
            }
        }
        std::sort(sc.begin(), sc.end(),
                  [](auto&a,auto&b){ return a.first > b.first; });
        for (auto &x : sc) order.push_back(x.second);
    }

    // --- the shared prefix's union (sec 3.5) -------------------------------
    // Routing is a deterministic function of hidden state, so requests that
    // share a prefix select the same experts over it; that union is the same
    // every request and is worth pinning once.
    std::unordered_set<int> prefix_union;
    for (auto &r : tr.req) {
        if (r.name.rfind("shared", 0) != 0) continue;
        for (int t=0;t<std::min(prefix_tokens, r.n_prefill);++t)
            for (int l=0;l<L;++l)
                for (int k=0;k<K;++k) {
                    int e = r.prefill[((size_t)t*L + l)*K + k];
                    if (e>=0) prefix_union.insert((int)((size_t)l*E+e));
                }
    }

    // --- gtier handles ----------------------------------------------------
    int per_shard = std::max(32, (int)(W / slot / sh.size()));
    gtier_config cfg{};
    cfg.backend = (gtier_backend)backend;
    cfg.slot_bytes = slot; cfg.slots = per_shard; cfg.queue_depth = per_shard;
    cfg.cache_policy = GTIER_CACHE_NONE;
    cfg.max_fetch_ranges = std::max(4, per_shard/GTIER_MAX_INFLIGHT);
    for (auto &s : sh) {
        s.g = gtier_open(s.path.c_str(), &cfg);
        if (!s.g) { std::fprintf(stderr,"gtier_open failed\n"); return 1; }
    }

    // --- residency --------------------------------------------------------
    Arena arena;
    arena.cap = R;
    if (R) {
        if (cudaHostAlloc((void**)&arena.base, R, cudaHostAllocMapped) != cudaSuccess) {
            std::fprintf(stderr,"residency alloc of %.2f GiB failed\n", R/1073741824.0);
            return 1;
        }
        std::memset(arena.base, 0, 1<<20);      // first-touch a little
    }
    std::list<int> lru; std::unordered_map<int,std::list<int>::iterator> lru_at;

    auto resident = [&](int id)->bool { return arena.has(id); };
    auto admit_static = [&](const std::vector<int> &ids) {
        for (int id : ids) {
            if (arena.has(id) || !unit[id].bytes) continue;
            if (!arena.put(id, unit[id].bytes)) break;
        }
    };
    if (policy==SERVE_PREFIX) {
        std::vector<int> pre(prefix_union.begin(), prefix_union.end());
        std::sort(pre.begin(), pre.end());
        admit_static(pre);                       // pinned first, never evicted
    }
    if (policy==SERVE_PERLAYER||policy==SERVE_PREFIX)
        admit_static(order);

    unsigned long long *sink; CK(cudaMalloc(&sink,sizeof(*sink)));
    CK(cudaMemset(sink,0,sizeof(*sink)));
    const uint8_t **dp; size_t *dl;
    CK(cudaMallocManaged(&dp, cfg.max_fetch_ranges*sizeof(*dp)));
    CK(cudaMallocManaged(&dl, cfg.max_fetch_ranges*sizeof(*dl)));

    // Fetch a set of units that are not resident, through the window.
    auto fetch_units = [&](const std::vector<int> &ids, uint64_t &bytes)->double {
        std::vector<std::vector<gtier_range>> byshard(sh.size());
        for (int id : ids)
            for (auto &pr : unit[id].parts) byshard[pr.first].push_back(pr.second);
        auto t0 = std::chrono::steady_clock::now();
        for (size_t s=0; s<sh.size(); ++s) {
            auto &v = byshard[s];
            for (size_t o=0; o<v.size(); o += cfg.max_fetch_ranges) {
                int n = (int)std::min((size_t)cfg.max_fetch_ranges, v.size()-o);
                std::vector<void*> outs(n);
                if (gtier_fetch(sh[s].g, v.data()+o, n, outs.data())) {
                    std::fprintf(stderr,"fetch failed (n=%d)\n", n); exit(1);
                }
                for (int k=0;k<n;++k){ dp[k]=(const uint8_t*)outs[k]; dl[k]=v[o+k].len; bytes+=v[o+k].len; }
                touch<<<n,256>>>(dp,dl,n,sink);
                CK(cudaDeviceSynchronize());
            }
        }
        return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    };

    // LRU admission: hold what was just used, evicting the least recent.
    auto lru_admit = [&](int id) {
        if (arena.has(id)) {
            lru.erase(lru_at[id]); lru.push_front(id); lru_at[id]=lru.begin();
            return;
        }
        while (arena.used + unit[id].bytes > arena.cap && !lru.empty()) {
            int v = lru.back(); lru.pop_back(); lru_at.erase(v);
            arena.used -= unit[v].bytes; arena.at.erase(v);
        }
        if (arena.used + unit[id].bytes <= arena.cap) {
            arena.at[id] = arena.used; arena.used += unit[id].bytes;
            lru.push_front(id); lru_at[id]=lru.begin();
        }
    };

    std::printf("model %.1f GiB  units %d x %.2f MiB  always-resident %.2f GiB\n",
                model_bytes/1073741824.0, n_units, per_unit/1048576.0,
                always_bytes/1073741824.0);
    std::printf("budget %.2f GiB  window %.2f  path-overhead %.2f  residency %.2f GiB "
                "(%.1f%% of experts)  policy=%s  backend=%s\n",
                B/1073741824.0, W/1073741824.0, OV/1073741824.0, R/1073741824.0,
                R ? 100.0*R/unit_bytes : 0.0, policy_name(policy),
                gtier_backend_name((gtier_backend)backend));
    if (policy==SERVE_PREFIX)
        std::printf("prefix union: %zu units (%.2f GiB) pinned\n",
                    prefix_union.size(), prefix_union.size()*per_unit/1073741824.0);

    double ttft_sum=0, tpot_sum=0; uint64_t pre_bytes=0, dec_bytes=0;
    int n_pre=0, n_dec=0;
    for (int rep=0; rep<repeats; ++rep)
    for (auto &r : tr.req) {
        // ---- prefill: the union over the prompt ---------------------------
        std::unordered_set<int> u;
        for (int t=0;t<r.n_prefill;++t)
            for (int l=0;l<L;++l)
                for (int k=0;k<K;++k) {
                    int e = r.prefill[((size_t)t*L+l)*K+k];
                    if (e>=0) u.insert((int)((size_t)l*E+e));
                }
        std::vector<int> miss;
        for (int id : u) if (!resident(id) && unit[id].bytes) miss.push_back(id);
        std::sort(miss.begin(), miss.end());
        uint64_t b=0; double dt = fetch_units(miss, b);
        pre_bytes += b; ttft_sum += dt; ++n_pre;
        // An LRU has no way to know prefill will not reuse these, so it admits
        // them -- and in doing so evicts what decode needs.  SERVE_PHASE is
        // exactly the decision not to.
        if (policy==SERVE_LRU) for (int id : miss) lru_admit(id);

        // ---- decode -------------------------------------------------------
        int n_dec_run = max_decode ? std::min(max_decode, r.n_decode) : r.n_decode;
        for (int t=0;t<n_dec_run;++t) {
            std::unordered_set<int> need;
            for (int l=0;l<L;++l)
                for (int k=0;k<K;++k) {
                    int e = r.decode[((size_t)t*L+l)*K+k];
                    if (e>=0) need.insert((int)((size_t)l*E+e));
                }
            std::vector<int> dm;
            for (int id : need) if (!resident(id) && unit[id].bytes) dm.push_back(id);
            std::sort(dm.begin(), dm.end());
            uint64_t db=0; double ddt = fetch_units(dm, db);
            dec_bytes += db; tpot_sum += ddt; ++n_dec;
            if (policy==SERVE_LRU || policy==SERVE_LRU_PHASE)
                for (int id : need) if (unit[id].bytes) lru_admit(id);
        }
        if (verbose)
            std::printf("  %-14s prefill miss %4zu/%zu\n", r.name.c_str(), miss.size(), u.size());
    }

    std::printf("%-10s | TTFT(io) %7.3f s  prefill %7.2f GiB | "
                "TPOT(io) %7.2f ms  decode %7.3f GiB/tok | total %7.2f GiB\n",
                policy_name(policy), ttft_sum/n_pre, pre_bytes/1073741824.0/n_pre,
                tpot_sum/n_dec*1e3, dec_bytes/1073741824.0/n_dec,
                (pre_bytes+dec_bytes)/1073741824.0);

    for (auto &s : sh) { gtier_close(s.g); gguf_free(&s.m); }
    if (arena.base) cudaFreeHost(arena.base);
    return 0;
}
