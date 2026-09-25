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
#include "moe_ffn.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <list>
#include <map>
#include <set>
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
// A routed unit is one (layer, expert), and the arithmetic needs to know
// which of its three matrices is which, so they are kept apart rather than
// flattened into a list of ranges.
struct Proj { int shard = -1; gtier_range r{0,0}; };
struct Unit {
    Proj gate, up, down;
    std::vector<std::pair<int, gtier_range>> parts;   // (shard, range), all three
    uint64_t bytes = 0;
    bool complete() const { return gate.shard>=0 && up.shard>=0 && down.shard>=0; }
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
        case SERVE_ONLINE: return "online";
        case SERVE_ONLINE_PREFIX: return "online+prefix";
        case SERVE_MULTIPREFIX: return "multi-prefix";
        case SERVE_UNIFIED: return "unified";
        case SERVE_UNIFIED_ONLINE: return "unified-online";
        case SERVE_MOEINF: return "moe-inf*";
        case SERVE_MIXTRAL: return "mixtral*";
        case SERVE_FULL: return "ledger";
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
    // Where the initial counts come from.  They must not come from the trace
    // being served: counts taken from the requests under test tell the policy
    // their future.  Given no profile trace, the run is labelled in-sample.
    std::vector<const char*> profile_paths;
    double budget_gib = 24.0;        // total memory for residency + window
    double window_gib = 0.5;         // staging window (saturates here, sec 4.14)
    int policy = SERVE_PERLAYER, repeats = 1, prefix_tokens = 30, max_decode = 0;
    // How much of residency prefixes may take before they start evicting each
    // other.  Without a cap several system prompts would crowd out everything
    // popularity-ordered residency needs.
    double prefix_budget_gib = 0.0;      // 0 = no cap
    // How much better a candidate has to be before it displaces a resident.
    double displace_margin = 0.0;
    // How many times a unit must be seen before it may become resident.  At
    // one, every count is one on the first pass, ties are everywhere, and the
    // arena fills with whatever arrived first -- which is prefill, since it
    // comes first in every request.  A decode-hot unit reaches two within a
    // single request; a unit the prompt merely touches once reaches it only
    // across requests, which is the distinction the counts are for.
    int admit_after = 1;
    double half_life = 0;    // serving steps; 0 = no decay (pure frequency)
    // Prediction mode (default).  The value of a resident unit is the
    // probability that the coming decode routes to it, estimated from two
    // sources: this request's own prefill routing (how often its prompt chose
    // the unit) and the decode routing seen so far.  mix is the weight of the
    // first.  "--mix off" restores the count utility above.
    bool use_pred = true; double mix = 0.5; bool selective = true;
    int batch = 1; bool overlap = false;
    // Terms of the estimate, each scaled to [0,1]:
    //   w_rec * 2^-(age/H)        recency, age in serving steps
    //   mix * pf / max pf         this request's prompt routing
    //   (1-mix) * hist / max      decode routing seen so far
    //   w_req * creq / max        this request's own decode routing
    double rec_half = 8, w_rec = 1.0, w_req = 0.0;
    // A prefill hit and a decode hit are both one prevented read, but they are
    // not equally predictive: a prompt touches a unit once and moves on, while
    // a decode-hot unit is touched again on the next token and the one after.
    // The weight is how much more a decode hit says about the future.
    double decode_weight = 4.0;
    // With --decode-weight auto the weight is what it should mean: how much
    // more a byte read at decode costs than a byte read at prefill.  Prefill
    // reads a whole union at depth and runs near the device's bandwidth;
    // decode reads a few units per layer and pays latency.  Both are measured
    // as the run goes, so the weight is the ratio of observed seconds per byte.
    bool decode_weight_auto = false;
    double pre_io_s = 0, pre_io_b = 0, dec_io_s = 0, dec_io_b = 0;
    // How much the profile's counts are trusted relative to what is observed.
    // One means a profiled hit and an observed hit weigh the same; zero means
    // there is no profile and everything is learned.
    double profile_weight = 0.0;
    size_t slot = 4u << 20;
    bool verbose = false;
    int backend = GTIER_BACKEND_GTIER;
    double path_overhead_gib = 0.0;
    // Decode is compute-bound once residency is good, so the I/O a token
    // actually waits on is what is left after the arithmetic hides it.  This
    // is also the budget a background admission has to fit inside if it is
    // not to stall the stream.
    double compute_ms_token = 0.89;      // 22B active params at 49.5 TFLOP/s
    double compute_ms_prompt_token = 0.89;
    bool interleave = false;             // mix prefill and decode (sec 3.6 limit)
    // How many layers ahead the router's choice is known.  This is not a free
    // parameter: a plain MoE learns layer L's experts only on reaching layer
    // L, so its fetches serialise with its arithmetic, while Pre-gated MoE
    // moves the gate one layer earlier precisely to buy one layer of overlap.
    // The driver defaulted to fetching a whole token's layers in one batch,
    // which hands every policy full lookahead for nothing; naming it makes
    // that a measured choice instead of a hidden gift.
    int lookahead = 0;                   // 0 = all layers at once (full)
    // With this on the driver stops reporting transfer seconds and reports
    // token times instead: the routed experts' feed-forward is actually run,
    // out of whatever memory the weights happen to be in.
    bool do_compute = false;
    // The continuous-submission path (sec 3.3).  On by default: the serving
    // measurements are of the whole stack, and leaving it off measured a data
    // path this work does not propose.
    bool use_async = true;
    // Toggles for the audit: a component that cannot be turned off cannot be
    // shown to be doing anything, and three of them turned out to be inert
    // before anyone checked.
    bool use_live_set = true;
    bool use_prefix_pin = false;
    int dim_hidden = 2048, dim_inter = 768;

    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto nx = [&]{ return argv[++i]; };
        if (s=="--shard") paths.push_back(nx());
        else if (s=="--trace") trace_path = nx();
        else if (s=="--profile-trace") profile_paths.push_back(nx());
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
        else if (s=="--compute-ms") compute_ms_token = atof(nx());
        else if (s=="--prompt-compute-ms") compute_ms_prompt_token = atof(nx());
        else if (s=="--interleave") interleave = true;
        else if (s=="--lookahead") lookahead = atoi(nx());
        else if (s=="--compute") do_compute = true;
        else if (s=="--no-async") use_async = false;
        else if (s=="--no-live-set") use_live_set = false;
        else if (s=="--no-prefix-pin") use_prefix_pin = false;
        else if (s=="--prefix-pin") use_prefix_pin = true;
        else if (s=="--hidden") dim_hidden = atoi(nx());
        else if (s=="--inter") dim_inter = atoi(nx());
        else if (s=="--prefix-budget") prefix_budget_gib = atof(nx());
        else if (s=="--margin") displace_margin = atof(nx());
        else if (s=="--admit-after") admit_after = atoi(nx());
        else if (s=="--half-life") half_life = atof(nx());
        else if (s=="--mix") { const char *v = nx();
            if (!std::strcmp(v,"off")) use_pred = false; else mix = atof(v); }
        else if (s=="--selective") selective = atoi(nx()) != 0;
        else if (s=="--batch") batch = std::max(1, atoi(nx()));
        else if (s=="--overlap") overlap = true;
        else if (s=="--rec-half") rec_half = atof(nx());
        else if (s=="--w-rec") w_rec = atof(nx());
        else if (s=="--w-req") w_req = atof(nx());
        else if (s=="--decode-weight") {
            const char *v = nx();
            if (!std::strcmp(v,"auto")) decode_weight_auto = true; else decode_weight = atof(v);
        }
        else if (s=="--profile-weight") profile_weight = atof(nx());
        else if (s=="--verbose") verbose = true;
    }
    if (paths.empty()) { std::fprintf(stderr,"--shard required\n"); return 1; }
    // Continuous submission is part of the gTier path; the other backends
    // answer each fetch synchronously, which is what they are compared as.
    if (backend != GTIER_BACKEND_GTIER) use_async = false;

    Trace tr;
    if (!load_trace(trace_path, tr)) { std::fprintf(stderr,"trace load failed\n"); return 1; }
    std::vector<Request> prof_req;
    for (auto pp : profile_paths) {
        Trace pt;
        if (!load_trace(pp, pt)) { std::fprintf(stderr,"profile trace load failed: %s\n", pp); return 1; }
        if (pt.n_layers!=tr.n_layers || pt.n_experts!=tr.n_experts || pt.topk!=tr.topk) {
            std::fprintf(stderr,"profile trace %s has a different model shape\n", pp); return 1;
        }
        for (auto &r : pt.req) prof_req.push_back(std::move(r));
    }
    const std::vector<Request> &preq = profile_paths.empty() ? tr.req : prof_req;
    std::printf("profile: %s (%zu requests)\n",
                profile_paths.empty() ? "IN-SAMPLE (served trace)" : "held-out trace", preq.size());

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
            // GGUF stores a layer's experts stacked in one tensor per
            // projection (ffn_{gate,up,down}_exps); expert e is the e-th of
            // E equal contiguous slices.  safetensors has one tensor each.
            bool stacked = std::strstr(tt.name,"_exps") != nullptr;
            int e_lo = tt.expert, e_hi = tt.expert + 1;
            if (stacked) { e_lo = 0; e_hi = E; }
            if (stacked && tt.size % E) {
                std::fprintf(stderr,"%s: size %llu not divisible by %d experts\n",
                             tt.name,(unsigned long long)tt.size,E); return 1;
            }
            if (tt.layer>=0 && tt.layer<L && tt.expert>=0 && tt.expert<E)
            for (int ex = e_lo; ex < e_hi; ++ex) {
                uint64_t sz  = stacked ? tt.size / E : tt.size;
                uint64_t off = tt.offset + (stacked ? (uint64_t)ex * sz : 0);
                Unit &u = unit[(size_t)tt.layer*E + ex];
                // One projection must land in one slot for the GEMM to see it
                // as a single matrix; at 3 MiB against a 4 MiB slot it does.
                Proj *pj = (std::strstr(tt.name,"gate_proj") || std::strstr(tt.name,"ffn_gate_exps")) ? &u.gate
                         : (std::strstr(tt.name,"up_proj")   || std::strstr(tt.name,"ffn_up_exps"))   ? &u.up
                         : (std::strstr(tt.name,"down_proj") || std::strstr(tt.name,"ffn_down_exps")) ? &u.down : nullptr;
                if (pj) { pj->shard = (int)i; pj->r = {off, (size_t)sz}; }
                // split on the slot grid so every piece fits one slot
                uint64_t pos = off, end = off + sz;
                size_t cap = slot - 2*4096;
                while (pos < end) {
                    uint64_t blk = (pos/slot + 1)*slot;
                    uint64_t stop = std::min({end, pos+(uint64_t)cap, blk});
                    u.parts.push_back({(int)i, {pos, (size_t)(stop-pos)}});
                    pos = stop;
                }
                u.bytes += sz;
            }
            else {
                always_bytes += tt.size;
            }
        }
    // The FFN kernels read bf16; quantised GGUF blocks need a dequantising
    // kernel, so on those models compute stays the per-token model.
    bool any_stacked = false;
    for (auto &x : sh) any_stacked |= x.m.stacked_experts != 0;
    if (do_compute && any_stacked) {
        std::fprintf(stderr,"--compute needs bf16 per-expert tensors; "
                            "use --compute-ms for GGUF models\n");
        return 1;
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
    for (auto &r : preq)
        for (int t=0;t<r.n_decode;++t)
            for (int l=0;l<L;++l)
                for (int k=0;k<K;++k) {
                    int e = r.decode[((size_t)t*L + l)*K + k];
                    if (e>=0) cnt[(size_t)l*E+e]++;
                }
    // What a resident unit is worth is the reads it prevents, and it prevents
    // them in both phases.  Prefill reads a unit once per request whose prompt
    // union contains it; decode reads it once per token that routes to it.
    // Counting both on one scale is what lets prefix value and popularity
    // value be compared instead of competing.
    std::vector<uint64_t> pre_hits((size_t)L*E, 0), dec_hits((size_t)L*E, 0);
    for (auto &r : preq) {
        std::unordered_set<int> u;
        for (int t=0;t<r.n_prefill;++t)
            for (int l=0;l<L;++l)
                for (int k=0;k<K;++k) {
                    int e = r.prefill[((size_t)t*L+l)*K+k];
                    if (e>=0) u.insert((int)((size_t)l*E+e));
                }
        for (int id : u) pre_hits[id]++;
        int nd = max_decode ? std::min(max_decode, r.n_decode) : r.n_decode;
        for (int t=0;t<nd;++t)
            for (int l=0;l<L;++l)
                for (int k=0;k<K;++k) {
                    int e = r.decode[((size_t)t*L+l)*K+k];
                    if (e>=0) dec_hits[(size_t)l*E+e]++;
                }
    }
    // The counts the value is computed from.  A profile seeds them; the run
    // keeps adding to them.  Declared here because residency is filled from
    // them before the first request.
    std::vector<uint64_t> obs_pre((size_t)L*E,0), obs_dec((size_t)L*E,0);
    // The utility LEDGER ranks by.  A plain count is frequency only, and a
    // cache ranked by frequency alone keeps what was popular in the profile
    // and evicts what was just admitted before it can be reused -- decode
    // reuses recent units (consecutive tokens share 3.30 of 8), so that loses
    // to LRU once the counts stop describing the future.  Each hit therefore
    // decays with a half-life measured in serving steps (one per decode token,
    // one per prefill), as in LRFU: a short half-life ranks by recency, an
    // infinite one by frequency.  The two phases add into the same value.
    std::vector<double> uval((size_t)L*E, 0.0), ut((size_t)L*E, 0.0);
    double clk = 0.0;
    auto udecayed = [&](int id) -> double {
        if (half_life <= 0) return uval[id];
        return uval[id] * std::exp2(-(clk - ut[id]) / half_life);
    };
    auto uhit = [&](int id, double w) { uval[id] = udecayed(id) + w; ut[id] = clk; };
    std::vector<double> cur_pf((size_t)L*E, 0.0), hist((size_t)L*E, 0.0);
    std::vector<double> last_use((size_t)L*E, -1e9), creq((size_t)L*E, 0.0);
    double cur_pf_sum = 0, cur_pf_max = 0, hist_sum = 0, hist_max = 0, creq_max = 0;
    int pf_known_layer = -1;      // prefill routing is known layer by layer
    auto pscore = [&](int id) -> double {
        if (!use_pred) return udecayed(id);
        double a = (id / E <= pf_known_layer && cur_pf_max > 0) ? cur_pf[id] / cur_pf_max : 0.0;
        double h = hist_max > 0 ? hist[id] / hist_max : 0.0;
        double c = creq_max > 0 ? creq[id] / creq_max : 0.0;
        double rc = std::exp2(-(clk - last_use[id]) / rec_half);
        return w_rec * rc + mix * a + (1.0 - mix) * h + w_req * c;
    };
    std::vector<int> unified_order;
    {
        std::vector<std::pair<double,int>> sc;
        for (size_t id=0; id<unit.size(); ++id)
            if (unit[id].bytes)
                sc.push_back({(double)(pre_hits[id] + dec_hits[id]), (int)id});
        std::sort(sc.begin(), sc.end(),
                  [](auto&a,auto&b){ return a.first > b.first; });
        for (auto &x : sc) unified_order.push_back(x.second);
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
    // A request's prefix family is the part of its name before the last '_':
    // shared_1/2/3 carry one system prompt, and any other family carries its
    // own.  Real deployments run several -- one per assistant, per tenant --
    // so the union of one prefix is not the whole story; what matters is what
    // happens when they do not all fit.
    auto family_of = [](const std::string &n)->std::string {
        size_t p = n.rfind('_');
        return (p==std::string::npos) ? n : n.substr(0,p);
    };
    std::map<std::string, std::unordered_set<int>> family_union;
    for (auto &r : tr.req) {
        std::string fam = family_of(r.name);
        auto &u = family_union[fam];
        for (int t=0;t<std::min(prefix_tokens, r.n_prefill);++t)
            for (int l=0;l<L;++l)
                for (int k=0;k<K;++k) {
                    int e = r.prefill[((size_t)t*L + l)*K + k];
                    if (e>=0) u.insert((int)((size_t)l*E+e));
                }
    }
    // A name is not evidence.  Two requests belong to the same prefix family
    // only if they route the same way over the prefix window -- which they do
    // when they really share the text, because routing is a deterministic
    // function of hidden state.  Grouping by name alone would pin the union of
    // requests that merely sort together.
    std::map<std::string,std::vector<int>> family_members;   // fam -> req idx
    for (size_t i=0;i<tr.req.size();++i) family_members[family_of(tr.req[i].name)].push_back((int)i);
    std::map<std::string,double> family_agree;
    std::map<std::string,int> family_count;
    for (auto &kv : family_members) {
        auto &v = kv.second;
        family_count[kv.first] = (int)v.size();
        if (v.size() < 2) { family_agree[kv.first] = 0.0; continue; }
        long same = 0, tot = 0;
        for (int t=0;t<prefix_tokens;++t)
            for (int l=0;l<L;++l) {
                std::unordered_set<int> a;
                bool ok = true;
                for (size_t m=0;m<v.size();++m) {
                    Request &rr = tr.req[v[m]];
                    if (t >= rr.n_prefill) { ok = false; break; }
                    std::unordered_set<int> s;
                    for (int k=0;k<K;++k) {
                        int e = rr.prefill[((size_t)t*L+l)*K+k];
                        if (e>=0) s.insert(e);
                    }
                    if (!m) a = s;
                    else { std::unordered_set<int> in; for (int e:s) if (a.count(e)) in.insert(e); a = in; }
                }
                if (!ok) continue;
                same += (long)a.size(); tot += K;
            }
        family_agree[kv.first] = tot ? (double)same/tot : 0.0;
    }
    // A family qualifies when its members agree on nearly every expert over
    // the window; below that the shared text is not shared after all.
    const double AGREE_MIN = 0.90;
    std::unordered_set<int> prefix_union;
    std::vector<std::string> pin_families;
    for (auto &kv : family_union)
        if (family_count[kv.first] > 1 && family_agree[kv.first] >= AGREE_MIN) {
            prefix_union.insert(kv.second.begin(), kv.second.end());
            pin_families.push_back(kv.first);
        }

    // --- gtier handles ----------------------------------------------------
    int per_shard = std::max(32, (int)(W / slot / sh.size()));
    if (use_async) per_shard = std::max(per_shard, 8 * GTIER_MAX_INFLIGHT);
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

    // Prefix table.  Each family holds its union; a family that no longer fits
    // evicts the least recently served one, but only the units that family
    // holds alone -- units another live prefix also needs stay.  Without this
    // a second system prompt would simply fail to be pinned, or would crowd
    // out everything popularity-ordered residency needs.
    struct PinnedFam { std::unordered_set<int> units; uint64_t seq = 0; };
    std::map<std::string, PinnedFam> pinned_fam;
    uint64_t pin_bytes = 0, pin_seq = 0, pin_evictions = 0;
    uint64_t PIN_CAP = prefix_budget_gib > 0
                     ? (uint64_t)(prefix_budget_gib * (1ull<<30)) : UINT64_MAX;

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
    // The online variants learn the prefix union the same way a server would:
    // from the first request that carries it.  Recorded here so the pin can be
    // applied when that request is seen, not before.
    std::unordered_set<int> pinned;
    if (policy==SERVE_PERLAYER||policy==SERVE_PREFIX)
        admit_static(order);
    if (policy==SERVE_UNIFIED)
        admit_static(unified_order);
    // The scheme is one policy with one value.  A profile is not a different
    // policy; it is where the counts start.  Given one, the counts begin at
    // its numbers and residency is filled by them before the first request;
    // without one they begin at zero and the first requests fill it.  Either
    // way the same counts keep accumulating from then on, so a profile that
    // turns out to be wrong is corrected rather than obeyed.
    if (policy==SERVE_FULL && profile_weight > 0) {
        for (size_t id=0; id<unit.size(); ++id) {
            obs_pre[id] = (uint64_t)(pre_hits[id] * profile_weight);
            obs_dec[id] = (uint64_t)(dec_hits[id] * profile_weight);
            uval[id] = ((double)pre_hits[id] + decode_weight * (double)dec_hits[id]) * profile_weight;
            hist[id] = (double)dec_hits[id] * profile_weight; hist_sum += hist[id];
            hist_max = std::max(hist_max, hist[id]);
        }
        admit_static(unified_order);
    }

    // Until now the arena recorded offsets and held nothing, which was enough
    // to say which reads a policy avoids.  Running the arithmetic needs the
    // bytes to be there, so the admitted set is read in once, and a unit's
    // three projections are laid out back to back so the GEMMs can find them.
    cublasHandle_t blas = nullptr;
    moe_dims dims{dim_hidden, dim_inter};
    void *d_x=nullptr, *d_g=nullptr, *d_u=nullptr, *d_h=nullptr, *d_y=nullptr;
    const void **d_ptrs = nullptr;        // gate,up,down,x,g,u,h,y arrays
    std::vector<const void*> h_ptrs;
    size_t ptr_cap = 0;
    if (do_compute) {
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr,"cublasCreate failed\n"); return 1;
        }
        // A layer's experts run in one batched call, so every buffer holds K
        // slices rather than one.
        size_t hb = (size_t)dim_hidden*2, ib = (size_t)dim_inter*2;
        CK(cudaMalloc(&d_x, hb)); CK(cudaMalloc(&d_y, hb*K));
        CK(cudaMalloc(&d_g, ib*K)); CK(cudaMalloc(&d_u, ib*K)); CK(cudaMalloc(&d_h, ib*K));
        CK(cudaMemset(d_x, 0x3c, hb));      // a plausible bf16 pattern near 1.0
        CK(cudaMemset(d_y, 0, hb*K));
        ptr_cap = (size_t)8*K*L;                 // a whole token's layers
        CK(cudaMalloc(&d_ptrs, sizeof(void*) * ptr_cap));
        auto tf = std::chrono::steady_clock::now();
        uint64_t filled = 0;
        for (auto &kv : arena.at) {
            Unit &un = unit[kv.first];
            if (!un.complete()) continue;
            const Proj *ps[3] = {&un.gate, &un.up, &un.down};
            uint64_t off = kv.second;
            for (int q=0;q<3;++q) {
                void *out = nullptr; gtier_range rr = ps[q]->r;
                if (gtier_fetch(sh[ps[q]->shard].g, &rr, 1, &out)) {
                    std::fprintf(stderr,"arena fill failed\n"); return 1;
                }
                CK(cudaMemcpy(arena.base+off, out, rr.len, cudaMemcpyDefault));
                off += rr.len; filled += rr.len;
            }
        }
        std::printf("arena filled: %.2f GiB in %.1f s\n", filled/1073741824.0,
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now()-tf).count());
    }

    unsigned long long *sink; CK(cudaMalloc(&sink,sizeof(*sink)));
    CK(cudaMemset(sink,0,sizeof(*sink)));
    const uint8_t **dp; size_t *dl;
    CK(cudaMallocManaged(&dp, cfg.max_fetch_ranges*sizeof(*dp)));
    CK(cudaMallocManaged(&dl, cfg.max_fetch_ranges*sizeof(*dl)));

    // Where a unit's three matrices are right now: the arena if it is
    // resident, or the slot it just landed in.  The arithmetic reads from
    // whichever without copying, which is what the staging plane promised.
    std::unordered_map<int, std::array<const void*,3>> where;

    // Fetch a set of units that are not resident, through the window.
    auto fetch_units = [&](const std::vector<int> &ids, uint64_t &bytes)->double {
        // Ranges are grouped by shard for the queue's sake, so a note of which
        // unit and which projection each one belongs to travels alongside.
        struct Tag { int unit; int proj; };
        std::vector<std::vector<gtier_range>> byshard(sh.size());
        std::vector<std::vector<Tag>> tags(sh.size());
        for (int id : ids) {
            const Unit &un = unit[id];
            const Proj *ps[3] = {&un.gate, &un.up, &un.down};
            if (do_compute && un.complete()) {
                for (int q=0;q<3;++q) {
                    byshard[ps[q]->shard].push_back(ps[q]->r);
                    tags[ps[q]->shard].push_back({id,q});
                }
            } else {
                for (auto &pr : un.parts) {
                    byshard[pr.first].push_back(pr.second);
                    tags[pr.first].push_back({id,-1});
                }
            }
        }
        auto t0 = std::chrono::steady_clock::now();
        for (size_t s=0; s<sh.size(); ++s) {
            auto &v = byshard[s];
            // Continuous submission: the next batch goes to the ring before
            // this one is collected, so the queue is never drained between
            // them.  gtier_fetch submits and waits for everything, which
            // empties the ring at every boundary and costs up to 31.7% at
            // shallow depth (sec 4.6).  The tickets own disjoint slots, so
            // two can be outstanding at once.
            size_t nb = (v.size() + cfg.max_fetch_ranges - 1) / cfg.max_fetch_ranges;
            gtier_ticket tk[2]; bool live_tk[2] = {false,false};
            std::vector<void*> outs(cfg.max_fetch_ranges);
            for (size_t b = 0; b <= nb; ++b) {
                if (b < nb) {
                    size_t o = b * cfg.max_fetch_ranges;
                    int n = (int)std::min((size_t)cfg.max_fetch_ranges, v.size()-o);
                    int cur = (int)(b & 1);
                    if (use_async) {
                        if (gtier_submit(sh[s].g, v.data()+o, n, &tk[cur])) {
                            std::fprintf(stderr,"submit failed (n=%d)\n", n); exit(1);
                        }
                        live_tk[cur] = true;
                    }
                }
                if (b == 0 && use_async) continue;        // nothing to collect yet
                size_t done = use_async ? b - 1 : b;
                if (done >= nb) break;
                size_t o = done * cfg.max_fetch_ranges;
                int n = (int)std::min((size_t)cfg.max_fetch_ranges, v.size()-o);
                int slot_i = (int)(done & 1);
                if (use_async) {
                    if (!live_tk[slot_i]) continue;
                    if (gtier_wait(sh[s].g, &tk[slot_i], outs.data())) {
                        std::fprintf(stderr,"wait failed\n"); exit(1);
                    }
                    live_tk[slot_i] = false;
                } else if (gtier_fetch(sh[s].g, v.data()+o, n, outs.data())) {
                    std::fprintf(stderr,"fetch failed (n=%d)\n", n); exit(1);
                }
                for (int k=0;k<n;++k){
                    dp[k]=(const uint8_t*)outs[k]; dl[k]=v[o+k].len; bytes+=v[o+k].len;
                    const Tag &tg = tags[s][o+k];
                    if (tg.proj >= 0) where[tg.unit][tg.proj] = outs[k];
                }
                if (!do_compute) {           // the arithmetic is the touch
                    touch<<<n,256>>>(dp,dl,n,sink);
                    CK(cudaDeviceSynchronize());
                }
            }
        }
        if (do_compute) CK(cudaDeviceSynchronize());
        return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    };

    // Run the routed experts' feed-forward for a group of units, a layer at a
    // time.  The pointer arrays for every layer in the group are staged and
    // uploaded once: a copy per layer meant 48 synchronous transfers per
    // token, which cost more than the arithmetic they were describing.
    auto run_ffn = [&](const std::vector<int> &ids)->double {
        if (!do_compute || ids.empty()) return 0.0;
        auto t0 = std::chrono::steady_clock::now();
        std::map<int, std::vector<int>> by_layer;
        for (int id : ids) if (unit[id].complete()) by_layer[id / E].push_back(id);
        if (by_layer.empty()) return 0.0;

        struct Job { int n; size_t base; };     // base = index into h_ptrs
        std::vector<Job> jobs;
        h_ptrs.clear();
        for (auto &kv : by_layer) {
            auto &v = kv.second;
            int n = (int)std::min<size_t>(v.size(), (size_t)K);
            std::array<std::vector<const void*>,8> col;
            for (auto &cvec : col) cvec.reserve(n);
            for (int j2=0;j2<n;++j2) {
                const Unit &un = unit[v[j2]];
                const void *w[3];
                auto it2 = arena.at.find(v[j2]);
                if (it2 != arena.at.end()) {
                    uint64_t off = it2->second;
                    w[0] = arena.base + off;
                    w[1] = arena.base + off + un.gate.r.len;
                    w[2] = arena.base + off + un.gate.r.len + un.up.r.len;
                } else {
                    auto wit = where.find(v[j2]);
                    if (wit == where.end() || !wit->second[0]) break;
                    for (int q=0;q<3;++q) w[q] = wit->second[q];
                }
                size_t slot = col[0].size();
                col[0].push_back(w[0]); col[1].push_back(w[1]); col[2].push_back(w[2]);
                col[3].push_back(d_x);
                col[4].push_back((const uint8_t*)d_g + slot*dim_inter*2);
                col[5].push_back((const uint8_t*)d_u + slot*dim_inter*2);
                col[6].push_back((const uint8_t*)d_h + slot*dim_inter*2);
                col[7].push_back((const uint8_t*)d_y + slot*dim_hidden*2);
            }
            int have = (int)col[0].size();
            if (!have) continue;
            jobs.push_back({have, h_ptrs.size()});
            for (auto &cvec : col) h_ptrs.insert(h_ptrs.end(), cvec.begin(), cvec.end());
        }
        if (jobs.empty()) return 0.0;
        if (h_ptrs.size() > ptr_cap) {
            if (d_ptrs) cudaFree(d_ptrs);
            ptr_cap = h_ptrs.size() * 2;
            CK(cudaMalloc(&d_ptrs, sizeof(void*) * ptr_cap));
        }
        CK(cudaMemcpy(d_ptrs, h_ptrs.data(), sizeof(void*)*h_ptrs.size(),
                      cudaMemcpyHostToDevice));
        for (auto &jb : jobs) {
            int rc = moe_layer_ffn(blas, dims, jb.n, nullptr, nullptr, nullptr,
                                   d_x, d_g, d_u, d_h, d_y, d_ptrs + jb.base, 0);
            if (rc) { std::fprintf(stderr,"layer ffn failed rc=%d\n", rc); exit(1); }
        }
        CK(cudaDeviceSynchronize());
        return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    };

    // Per-layer LFU learned online.  The static orderings above are an oracle
    // -- they rank by counts taken from the whole trace, which a running
    // system does not have.  This one starts empty, counts what it sees, and
    // swaps a resident unit out only for one that has been seen more often in
    // the same layer, so the ordering it converges to is per layer without
    // anyone being told the distribution.
    std::vector<uint64_t> seen((size_t)L*E, 0);
    std::vector<std::vector<int>> res_of_layer(L);
    auto online_admit = [&](int id) {
        seen[id]++;
        if (arena.has(id) || !unit[id].bytes) return;
        int l = id / E;
        if (arena.put(id, unit[id].bytes)) { res_of_layer[l].push_back(id); return; }
        // full: swap out this layer's least-seen resident, if it is worse
        auto &v = res_of_layer[l];
        if (v.empty()) return;
        int worst = v[0];
        for (int q : v) if (seen[q] < seen[worst]) worst = q;
        if (seen[worst] >= seen[id]) return;
        // sizes are uniform per unit, so the slot can be reused in place
        uint64_t off = arena.at[worst];
        arena.at.erase(worst); arena.at[id] = off;
        v.erase(std::find(v.begin(), v.end(), worst)); v.push_back(id);
    };

    // Admission into free space only: fill what nobody holds, evict nothing.
    // This is what a phase that will not reuse its own reads is entitled to.
    auto lru_admit_free_only = [&](int id) {
        if (arena.has(id) || !unit[id].bytes) return;
        if (arena.used + unit[id].bytes > arena.cap) return;
        arena.at[id] = arena.used; arena.used += unit[id].bytes;
        lru.push_back(id);                       // coldest: decode evicts it first
        lru_at[id] = std::prev(lru.end());
    };

    // The same value learned online: a running system does not have the
    // counts above, but it sees the hits as they happen and can keep its own.
    std::vector<std::vector<int>> res_by_layer_u(L);
    auto unified_admit = [&](int id, bool from_prefill) {
        if (from_prefill) obs_pre[id]++; else obs_dec[id]++;
        if (!unit[id].bytes) return;
        double v = (double)(obs_pre[id] + obs_dec[id]);
        if (arena.has(id)) return;
        if (arena.put(id, unit[id].bytes)) { res_by_layer_u[id/E].push_back(id); return; }
        // full: displace the weakest resident anywhere, not just in this layer,
        // because the value is already comparable across layers.
        int worst = -1; double wv = 1e18;
        for (auto &kv : arena.at) {
            double q = (double)(obs_pre[kv.first] + obs_dec[kv.first]);
            if (q < wv) { wv = q; worst = kv.first; }
        }
        if (worst < 0 || wv >= v) return;
        uint64_t off = arena.at[worst];
        arena.at.erase(worst); arena.at[id] = off;
    };

    // Sequence-level activation matrix, the shape MoE-Infinity's prefetcher
    // assumes: experts a sequence has already used are the ones it will use
    // again, so they are retained ahead of anything else.  Implemented as an
    // LRU whose ordering is overridden by membership in the current
    // sequence's activation set.
    std::unordered_set<int> seq_active;
    auto moeinf_admit = [&](int id) {
        seq_active.insert(id);
        if (arena.has(id) || !unit[id].bytes) return;
        while (arena.used + unit[id].bytes > arena.cap) {
            int victim = -1;
            for (auto it2 = lru.rbegin(); it2 != lru.rend(); ++it2)
                if (!seq_active.count(*it2)) { victim = *it2; break; }
            if (victim < 0 && !lru.empty()) victim = lru.back();   // all active
            if (victim < 0) return;
            lru.erase(lru_at[victim]); lru_at.erase(victim);
            arena.used -= unit[victim].bytes; arena.at.erase(victim);
        }
        arena.at[id] = arena.used; arena.used += unit[id].bytes;
        lru.push_front(id); lru_at[id] = lru.begin();
    };

    // The whole scheme's admission.  One value per unit, learned rather than
    // given, and both phases admit by it.
    //
    // An earlier version kept the rule that prefill may take free space and
    // may not displace anything, which is right when there is no value to
    // compare against -- a prefill reads the union once and does not return,
    // so an LRU that admits from it evicts what decode needs.  With a value
    // function that rule becomes harmful: it counts a prefill hit and then
    // forbids it from acting, so units the prompt keeps returning to can never
    // become resident.  Measured, it left prefill I/O at 1.343 s against the
    // oracle's 0.574 while per-token I/O was already at the oracle's level.
    // The value already knows a prefill unit is worth one hit and a hot decode
    // unit many; the rule was a substitute for not knowing that.
    std::unordered_set<int> pinned_units;      // prefix families, never displaced
    // Units this request has already touched.  The utility is a count of past
    // use, which says how often a unit is wanted but not when it is wanted
    // next -- and eviction is a question about when.  Sequence locality gives
    // the missing evidence: consecutive decode tokens share 3.30 of 8 experts
    // against 0.50 for independent draws (sec 2.3c), so a unit this request
    // has just used is very likely wanted again before the counts catch up.
    // Excluding those from displacement is what stops a token's own working
    // set being evicted and re-read a token later.
    // What is in use *now*.  Scoped to the current token, not the request:
    // a request's prefill union is 79% of the experts, so holding that for the
    // whole request leaves nothing evictable -- measured, 150,312 admissions
    // blocked with every non-pinned resident marked live.  A token's own 384
    // units are what must not be evicted out from under it.
    std::unordered_set<int> live_set;
    // Counters, because three guesses at why the hit rate was low were all
    // wrong and the policy needs to say what it is actually doing.
    uint64_t adm_free=0, adm_swap=0, adm_blocked_pin=0, adm_blocked_live=0,
             adm_blocked_none=0, adm_already=0, adm_reject=0;
    auto full_admit = [&](int id, bool from_prefill) {
        if (from_prefill) obs_pre[id]++; else obs_dec[id]++;
        uhit(id, from_prefill ? 1.0 : decode_weight);
        last_use[id] = clk;
        if (!from_prefill) {
            hist[id] += 1.0; hist_sum += 1.0; hist_max = std::max(hist_max, hist[id]);
            creq[id] += 1.0; creq_max = std::max(creq_max, creq[id]);
        }
        if (!unit[id].bytes) return;
        if (arena.has(id)) { adm_already++; return; }
        if (obs_pre[id] + obs_dec[id] < (uint64_t)admit_after) return;
        if (arena.put(id, unit[id].bytes)) { adm_free++; return; }   // free space

        // Admission is driven by demand and eviction by the utility.  Being
        // accessed now is evidence the counts cannot carry: a unit routed to
        // rarely still has to be read when it is routed to, and refusing it
        // because its count is low pays that read on every occurrence.  What
        // the utility decides is not whether to take something in but which
        // resident to give up for it -- and never one this request is still
        // using, since sequence locality says it is wanted again shortly
        // (3.30 of 8 experts shared between consecutive tokens, sec 2.3c).
        int worst = -1; double wv = 1e18;
        uint64_t n_pin = 0, n_live = 0;
        for (auto &kv : arena.at) {
            if (pinned_units.count(kv.first)) { n_pin++; continue; }
            if (use_live_set && live_set.count(kv.first)) { n_live++; continue; }
            double q = pscore(kv.first);
            if (q < wv) { wv = q; worst = kv.first; }
        }
        if (worst < 0) {
            if (n_pin >= arena.at.size())                 adm_blocked_pin++;
            else if (n_live >= arena.at.size() - n_pin)   adm_blocked_live++;
            else                                          adm_blocked_none++;
            return;
        }
        // The bytes are already in the window, so taking the unit in costs a
        // copy and nothing else; the only question is whether it is worth
        // more than what it would displace.
        if (use_pred && selective && pscore(id) <= wv) { adm_reject++; return; }
        adm_swap++;
        uint64_t off = arena.at[worst];
        arena.at.erase(worst); arena.at[id] = off;
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
    if (policy==SERVE_PREFIX || policy==SERVE_MULTIPREFIX || policy==SERVE_ONLINE_PREFIX) {
        std::printf("prefix families:");
        for (auto &kv : family_agree)
            if (family_count[kv.first] > 1)
                std::printf(" %s(n=%d,agree=%.2f%s)", kv.first.c_str(),
                            family_count[kv.first], kv.second,
                            kv.second>=AGREE_MIN ? ",pin" : "");
        std::printf("\n");
        std::printf("prefix union (whole-trace reference; online policies learn their own): %zu units (%.2f GiB) over %zu families\n",
                    prefix_union.size(), prefix_union.size()*per_unit/1073741824.0,
                    pin_families.size());
    }

    double ttft_sum=0, tpot_sum=0; uint64_t pre_bytes=0, dec_bytes=0;
    int n_pre=0, n_dec=0;
    uint64_t prompt_tok_sum = 0;
    // What a token actually waits for.  Decode is compute-bound once residency
    // is good, so I/O that fits inside the arithmetic costs nothing; only the
    // excess is a stall.  A background admission has the same budget: it is
    // free exactly while it fits in that shadow.
    double stall_sum = 0, shadow_bytes_sum = 0, bg_bytes_sum = 0;
    double compute_sum = 0;          // real arithmetic, when --compute is on
    const double BW = 5.682 * 1073741824.0;   // measured device ceiling, B/s
    // --- schedule ---------------------------------------------------------
    // A step is either a request's prefill (-1) or one of its decoded tokens.
    // Sequential order is one request at a time, which is what a single user
    // on an edge device produces.  --interleave round-robins instead, so
    // prefills and decodes are in flight together the way continuous batching
    // puts them -- and that is precisely where a global phase switch cannot
    // be made, so it is measured rather than assumed away.
    struct Step { int req; int tok; };       // tok = -1 means prefill
    std::vector<Step> sched;
    {
        std::vector<std::vector<Step>> per(tr.req.size());
        for (size_t i=0;i<tr.req.size();++i) {
            per[i].push_back({(int)i,-1});
            int nd = max_decode ? std::min(max_decode, tr.req[i].n_decode)
                                : tr.req[i].n_decode;
            for (int t=0;t<nd;++t) per[i].push_back({(int)i,t});
        }
        for (int rep=0; rep<repeats; ++rep) {
            if (!interleave) {
                for (auto &v : per) for (auto &s : v) sched.push_back(s);
            } else {
                size_t mx=0; for (auto &v:per) mx=std::max(mx,v.size());
                for (size_t k=0;k<mx;++k)
                    for (auto &v : per) if (k<v.size()) sched.push_back(v[k]);
            }
        }
    }

    // Pin a family's union, evicting the least recently served family's
    // exclusive units if the cap is in the way.
    // A family is learned as a server would learn it: the first member's
    // routing over the prefix window is kept; a later member that agrees with
    // it on at least AGREE_MIN of the experts qualifies the family, and only
    // then is the first member's union pinned.  Nothing about requests not yet
    // served is used.
    std::map<std::string, std::vector<int16_t>> fam_first;   // [t][l][k]
    std::map<std::string, std::unordered_set<int>> fam_learned;
    std::set<std::string> fam_ok;
    auto learn_family = [&](const std::string &fam, const Request &r) -> bool {
        int T = std::min(prefix_tokens, r.n_prefill);
        auto f = fam_first.find(fam);
        if (f == fam_first.end()) {
            std::vector<int16_t> v(r.prefill.begin(), r.prefill.begin() + (size_t)T*L*K);
            auto &u = fam_learned[fam];
            for (int t=0;t<T;++t) for (int l=0;l<L;++l) for (int k=0;k<K;++k) {
                int e = v[((size_t)t*L+l)*K+k];
                if (e>=0) u.insert((int)((size_t)l*E+e));
            }
            fam_first[fam] = std::move(v);
            return false;
        }
        if (fam_ok.count(fam)) return false;
        int T0 = (int)(f->second.size() / ((size_t)L*K)), Tm = std::min(T, T0);
        long same = 0, tot = 0;
        for (int t=0;t<Tm;++t) for (int l=0;l<L;++l) {
            std::unordered_set<int> a;
            for (int k=0;k<K;++k) { int e = f->second[((size_t)t*L+l)*K+k]; if (e>=0) a.insert(e); }
            for (int k=0;k<K;++k) { int e = r.prefill[((size_t)t*L+l)*K+k]; if (e>=0 && a.count(e)) ++same; }
            tot += K;
        }
        if (tot && (double)same/tot >= AGREE_MIN) { fam_ok.insert(fam); return true; }
        return false;
    };
    auto pin_family = [&](const std::string &fam) {
        if (!fam_ok.count(fam)) return;
        auto it = fam_learned.find(fam);
        auto &pf = pinned_fam[fam];
        pf.seq = ++pin_seq;
        if (!pf.units.empty()) return;                 // already pinned
        uint64_t want = 0;
        for (int id : it->second) if (unit[id].bytes && !arena.has(id)) want += unit[id].bytes;
        while (pin_bytes + want > PIN_CAP && pinned_fam.size() > 1) {
            auto victim = pinned_fam.end();
            for (auto i2 = pinned_fam.begin(); i2 != pinned_fam.end(); ++i2)
                if (i2->first != fam && (victim==pinned_fam.end() || i2->second.seq < victim->second.seq))
                    victim = i2;
            if (victim == pinned_fam.end()) break;
            for (int id : victim->second.units) {
                bool shared_with_live = false;
                for (auto &kv : pinned_fam)
                    if (&kv.second != &victim->second && kv.second.units.count(id)) { shared_with_live=true; break; }
                if (shared_with_live) continue;
                if (arena.at.count(id)) { arena.used -= unit[id].bytes; arena.at.erase(id); }
                pin_bytes -= unit[id].bytes;
            }
            pinned_fam.erase(victim);
            ++pin_evictions;
        }
        for (int id : it->second) {
            if (!unit[id].bytes || arena.has(id)) continue;
            if (pin_bytes + unit[id].bytes > PIN_CAP) break;
            if (!arena.put(id, unit[id].bytes)) break;
            pf.units.insert(id); pin_bytes += unit[id].bytes;
            pinned_units.insert(id);
        }
    };

    // --- serve ------------------------------------------------------------
    // Static batching: requests are taken B at a time in trace order.  A
    // batch's prefill reads the union of its prompts' units once, and each
    // decode step reads the union of what its still-active requests route to,
    // so a unit wanted by several requests in the batch is read once for all
    // of them.  B = 1 is one request at a time.
    //
    // Time.  I/O is measured; the arithmetic is the measured kernels under
    // --compute and the calibrated cost otherwise.  A decode step is bound by
    // the bytes it touches, so its cost scales with the dense weights plus the
    // routed units it touches: c_d * (dense + |need| * unit) / (dense + K*L*unit).
    // Prompt arithmetic is compute-bound and adds up over the batch's tokens.
    // Without --overlap the phase costs I/O + arithmetic; with it the layers
    // are pipelined (layer l+1 is read while layer l computes), which costs
    // max(io, c) + min(io, c) / L -- the slower of the two plus one layer's
    // worth of the faster to fill the pipe.
    auto phase_time = [&](double io, double c) -> double {
        if (!overlap) return io + c;
        return std::max(io, c) + std::min(io, c) / L;
    };
    const double unit_avg = (double)per_unit;
    const double dense_b = (double)always_bytes;
    std::vector<int> prev_need;              // last step's units, for speculation
    double req_time_sum = 0, ttft_time_sum = 0, tok_time_sum = 0, wall_sum = 0;
    uint64_t n_req_done = 0, n_tok_done = 0;
    std::vector<int> order_req;
    for (int rep=0; rep<repeats; ++rep)
        for (size_t i=0;i<tr.req.size();++i) order_req.push_back((int)i);
    for (size_t b0 = 0; b0 < order_req.size(); b0 += batch) {
        std::vector<int> br(order_req.begin()+b0,
                            order_req.begin()+std::min(order_req.size(), b0+(size_t)batch));
        // ---- prefill ------------------------------------------------------
        clk += 1.0;
        bool learn = (policy==SERVE_MULTIPREFIX || policy==SERVE_ONLINE_PREFIX
                      || policy==SERVE_FULL) && use_prefix_pin;
        if (learn) for (int ri : br) pin_family(family_of(tr.req[ri].name));
        std::unordered_set<int> u;
        double prompt_c = 0;
        for (int ri : br) {
            Request &r = tr.req[ri];
            for (int t=0;t<r.n_prefill;++t)
                for (int l=0;l<L;++l)
                    for (int k=0;k<K;++k) {
                        int e = r.prefill[((size_t)t*L+l)*K+k];
                        if (e>=0) u.insert((int)((size_t)l*E+e));
                    }
            prompt_c += r.n_prefill * compute_ms_prompt_token * 1e-3;
            prompt_tok_sum += r.n_prefill;
        }
        std::vector<int> miss;
        for (int id : u) if (!resident(id) && unit[id].bytes) miss.push_back(id);
        std::sort(miss.begin(), miss.end());
        uint64_t b=0; double dt = fetch_units(miss, b);
        pre_bytes += b; ttft_sum += dt; n_pre += (int)br.size();
        if (b) { pre_io_s += dt; pre_io_b += b; }
        double t_prefill = phase_time(dt, prompt_c);
        stall_sum += std::max(0.0, dt - prompt_c);
        if (learn) for (int ri : br)
            if (learn_family(family_of(tr.req[ri].name), tr.req[ri]))
                pin_family(family_of(tr.req[ri].name));
        if (policy==SERVE_LRU) for (int id : miss) lru_admit(id);
        if (policy==SERVE_LRU_PHASE) for (int id : miss) lru_admit_free_only(id);
        if (policy==SERVE_UNIFIED_ONLINE)
            for (int id : u) if (unit[id].bytes) unified_admit(id, true);
        if (policy==SERVE_FULL) {
            live_set.clear();
            std::fill(cur_pf.begin(), cur_pf.end(), 0.0); cur_pf_sum = 0; cur_pf_max = 0;
            std::fill(creq.begin(), creq.end(), 0.0); creq_max = 0;
            for (int ri : br) {
                Request &r = tr.req[ri];
                for (int t=0;t<r.n_prefill;++t)
                    for (int l=0;l<L;++l)
                        for (int k=0;k<K;++k) {
                            int e = r.prefill[((size_t)t*L+l)*K+k];
                            if (e>=0) { cur_pf[(size_t)l*E+e] += 1.0; cur_pf_sum += 1.0; }
                        }
            }
            for (double v : cur_pf) cur_pf_max = std::max(cur_pf_max, v);
            // In layer order: when layer l's experts pass through the
            // window, the prompts' routing is known up to layer l only.
            std::vector<int> us(u.begin(), u.end());
            std::sort(us.begin(), us.end());
            for (int id : us) {
                pf_known_layer = id / E;
                if (unit[id].bytes) full_admit(id, true);
            }
            pf_known_layer = L - 1;
        }
        if (policy==SERVE_MOEINF || policy==SERVE_MIXTRAL) seq_active.clear();
        if (policy==SERVE_MOEINF) for (int id : miss) lru_admit_free_only(id);
        if (policy==SERVE_MIXTRAL) for (int id : miss) lru_admit(id);

        // ---- decode -------------------------------------------------------
        std::vector<int> nd(br.size());
        int T = 0;
        for (size_t j=0;j<br.size();++j) {
            Request &r = tr.req[br[j]];
            nd[j] = max_decode ? std::min(max_decode, r.n_decode) : r.n_decode;
            T = std::max(T, nd[j]);
        }
        std::vector<double> t_req(br.size(), t_prefill);
        double t_batch = t_prefill;
        for (int tok=0; tok<T; ++tok) {
            clk += 1.0;
            std::unordered_set<int> need;
            int active = 0;
            for (size_t j=0;j<br.size();++j) {
                if (tok >= nd[j]) continue;
                ++active;
                Request &r = tr.req[br[j]];
                for (int l=0;l<L;++l)
                    for (int k=0;k<K;++k) {
                        int e = r.decode[((size_t)tok*L+l)*K+k];
                        if (e>=0) need.insert((int)((size_t)l*E+e));
                    }
            }
            uint64_t db=0; double ddt = 0, dct = 0;
            int grp = lookahead > 0 ? lookahead : L;
            for (int l0=0; l0<L; l0+=grp) {
                std::vector<int> dm, all;
                for (int id : need) {
                    int l = id / E;
                    if (l < l0 || l >= std::min(L,l0+grp) || !unit[id].bytes) continue;
                    all.push_back(id);
                    if (!resident(id)) dm.push_back(id);
                }
                std::sort(dm.begin(), dm.end()); std::sort(all.begin(), all.end());
                ddt += fetch_units(dm, db);
                dct += run_ffn(all);
            }
            double c_step = do_compute ? dct
                : compute_ms_token * 1e-3 * (dense_b + need.size() * unit_avg)
                                          / (dense_b + (double)K * L * unit_avg);
            double t_step = phase_time(ddt, c_step);
            compute_sum += c_step;
            dec_bytes += db; tpot_sum += ddt; n_dec += active;
            if (db) { dec_io_s += ddt; dec_io_b += db; }
            if (decode_weight_auto && pre_io_b > 0 && dec_io_b > 0)
                decode_weight = std::min(64.0, std::max(1.0,
                    (dec_io_s / dec_io_b) / (pre_io_s / pre_io_b)));
            stall_sum += std::max(0.0, ddt - c_step);
            t_batch += t_step;
            for (size_t j=0;j<br.size();++j)
                if (tok < nd[j]) { t_req[j] += t_step; tok_time_sum += t_step; ++n_tok_done; }
            if (policy==SERVE_LRU || policy==SERVE_LRU_PHASE)
                for (int id : need) if (unit[id].bytes) lru_admit(id);
            if (policy==SERVE_UNIFIED_ONLINE)
                for (int id : need) if (unit[id].bytes) unified_admit(id, false);
            if (policy==SERVE_FULL) {
                live_set.clear();
                for (int id : need) if (unit[id].bytes) live_set.insert(id);
                for (int id : need) if (unit[id].bytes) full_admit(id, false);
            }
            if (policy==SERVE_MOEINF)
                for (int id : need) if (unit[id].bytes) moeinf_admit(id);
            if (policy==SERVE_MIXTRAL) {
                for (int id : need) if (unit[id].bytes) lru_admit(id);
                for (int id : prev_need) if (unit[id].bytes && !arena.has(id))
                    lru_admit(id);
                prev_need.assign(need.begin(), need.end());
            }
            if (policy==SERVE_ONLINE || policy==SERVE_ONLINE_PREFIX
                || policy==SERVE_MULTIPREFIX)
                for (int id : need) if (unit[id].bytes) {
                    if (!arena.has(id)) bg_bytes_sum += unit[id].bytes;
                    online_admit(id);
                }
        }
        for (size_t j=0;j<br.size();++j) {
            req_time_sum += t_req[j]; ttft_time_sum += t_prefill; ++n_req_done;
        }
        wall_sum += t_batch;
    }

    std::printf("%-13s | TTFT(io) %7.3f s  prefill %7.2f GiB | "
                "look=%-3d | TPOT(io) %7.2f ms  decode %7.3f GiB/tok | stall %7.3f s | "
                "bg %6.2f GiB vs shadow %6.2f GiB | pins %zu evict %llu | total %7.2f GiB\n",
                policy_name(policy), ttft_sum/n_pre, pre_bytes/1073741824.0/n_pre,
                lookahead, tpot_sum/n_dec*1e3, dec_bytes/1073741824.0/n_dec, stall_sum,
                bg_bytes_sum/1073741824.0, shadow_bytes_sum/1073741824.0,
                pinned_fam.size(), (unsigned long long)pin_evictions,
                (pre_bytes+dec_bytes)/1073741824.0);
    if (policy == SERVE_FULL)
        std::printf("%-13s | admissions: free %llu swap %llu already %llu | "
                    "blocked: all-pinned %llu all-live %llu other %llu | "
                    "resident %zu of %d units | rejected %llu\n", policy_name(policy),
                    (unsigned long long)adm_free, (unsigned long long)adm_swap,
                    (unsigned long long)adm_already,
                    (unsigned long long)adm_blocked_pin,
                    (unsigned long long)adm_blocked_live,
                    (unsigned long long)adm_blocked_none,
                    arena.at.size(), n_units, (unsigned long long)adm_reject);
    if (do_compute) {
        // A token's cost is its I/O plus its arithmetic, and the reciprocal is
        // the rate the engine literature reports.
        //
        // The arithmetic is reported twice, because the kernels here are a
        // straightforward implementation rather than a tuned one.  A decode
        // token reads every routed weight once and reuses none, so the work is
        // bound by memory and its floor is (bytes touched) / 254 GB/s, the
        // measured GPU read bandwidth (gtier/mapped_read_bw.cu, and the same
        // whether the bytes sit in cudaMalloc'd or mapped memory).  The
        // measured kernels run about three and a half times that.  Since the
        // same kernels run under every policy, the comparison between policies
        // is unaffected; what the gap moves is the absolute rate, so both ends
        // are given and the truth is between them.
        double io_tok = tpot_sum / n_dec;
        double cp_tok = compute_sum / n_dec;
        double touched = (double)K * L * per_unit;          // bytes per token
        double floor_tok = touched / (254.0e9);
        auto rate = [&](double c){ double t = io_tok + c; return t>0 ? 1.0/t : 0.0; };
        std::printf("%-13s | io %7.2f ms/tok | compute %7.2f ms (floor %6.2f) "
                    "-> **%7.3f tok/s** (ceiling %7.3f) | prefill I/O %7.3f s\n",
                    policy_name(policy), io_tok*1e3, cp_tok*1e3, floor_tok*1e3,
                    rate(cp_tok), rate(floor_tok), ttft_sum/n_pre);
        // tok/s alone hides the prefill, and prefill is where the I/O is: a
        // policy can win the rate and still take longer to answer.  What a
        // user waits for is the whole request, so it is reported at several
        // generation lengths.
        double pre = ttft_sum / n_pre, per = io_tok + cp_tok;
        std::printf("%-13s | request seconds  n=4 %7.3f  n=32 %7.3f  "
                    "n=64 %7.3f  n=256 %7.3f\n", policy_name(policy),
                    pre+4*per, pre+32*per, pre+64*per, pre+256*per);
    }

    // One line per run for the tables.  I/O is measured; the arithmetic is
    // the measured kernels under --compute and the calibrated per-token cost
    // otherwise (--compute-ms, --prompt-compute-ms).  It is the same for every
    // policy on a model, so the policies differ only in what they read.  The
    // phases are charged serially, which is the conservative reading.
    if (policy == SERVE_FULL)
        std::printf("value: %s  mix %.2f  selective %d | count mode: W %.2f (%s) half-life %g\n",
                    use_pred ? "decode-probability estimate" : "counts", mix, (int)selective,
                    decode_weight, decode_weight_auto ? "measured cost ratio" : "fixed", half_life);
    if (n_req_done && n_tok_done) {
        double ttft = ttft_time_sum / n_req_done;
        double tpot = tok_time_sum / n_tok_done;
        double req  = req_time_sum / n_req_done;
        std::printf("RESULT policy=%s backend=%s batch=%d overlap=%d budget=%.2f window=%.2f "
                    "requests=%llu prompt_tok=%.1f decode_tok=%.1f ttft_s=%.4f tpot_ms=%.3f "
                    "request_s=%.4f throughput_tok_s=%.4f prefill_io_s=%.4f decode_io_ms=%.3f "
                    "prefill_gib=%.3f decode_gib_tok=%.4f compute=%s\n",
                    policy_name(policy), gtier_backend_name((gtier_backend)backend), batch,
                    (int)overlap, budget_gib, window_gib, (unsigned long long)n_req_done,
                    (double)prompt_tok_sum / n_req_done, (double)n_tok_done / n_req_done,
                    ttft, tpot * 1e3, req, (double)n_tok_done / wall_sum,
                    ttft_sum / n_pre, tpot_sum / n_dec * 1e3,
                    pre_bytes / 1073741824.0 / n_pre, dec_bytes / 1073741824.0 / n_dec,
                    do_compute ? "measured" : "calibrated");
    }

    for (auto &s : sh) { gtier_close(s.g); gguf_free(&s.m); }
    if (arena.base) cudaFreeHost(arena.base);
    return 0;
}
