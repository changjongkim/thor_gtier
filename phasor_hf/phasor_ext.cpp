// PHASOR engine for Hugging Face transformers: expert residency and staging
// behind the MoE blocks of a real model.
//
// The model runs unchanged except that each MoE block asks this engine for
// the weights of the experts its router chose.  An expert is served from the
// residency arena when it is there, and otherwise read by gTier into a staging
// slot; either way the caller gets CUDA tensors that alias the bytes where
// they landed (cudaHostAlloc(Mapped) memory is GPU-addressable on this SoC),
// so no copy is made on the way to the matmul.  Whether a staged expert is
// then admitted into the arena, and what it displaces, is PHASOR's residency
// policy: the value of a unit is its estimated probability of decode use --
// recency, decode history and (auxiliary) this request's prompt routing.
//
// Reads are issued only for experts a router has already chosen, so the
// pipeline is causal: within a layer, the next chunk of experts is read while
// the current chunk computes (submit/collect), never the next layer's.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "gtier.h"

namespace {

struct Proj { int file = -1; uint64_t off = 0; uint64_t len = 0; int64_t rows = 0, cols = 0; };
struct Unit { Proj p[3]; uint64_t bytes = 0; };   // gate, up, down

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) \
    throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e)); } while (0)

class Engine {
public:
    Engine(std::vector<std::string> files, int L, int E, double window_gib, int64_t slot_mib,
           double arena_gib, std::string policy, double mix, double rec_half, double w_rec)
        : L_(L), E_(E), units_((size_t)L * E), policy_(policy), mix_(mix),
          rec_half_(rec_half), w_rec_(w_rec) {
        size_t slot = (size_t)slot_mib << 20;
        // One window and one ring for every shard (gtier_add_file).
        int slots = std::max(8, (int)((window_gib * (1ull << 30)) / slot));
        slots = std::max(slots, 2 * GTIER_MAX_INFLIGHT);
        gtier_config cfg{};
        cfg.backend = GTIER_BACKEND_GTIER;
        cfg.slot_bytes = slot; cfg.slots = slots; cfg.queue_depth = slots;
        cfg.cache_policy = GTIER_CACHE_NONE;
        cfg.max_fetch_ranges = std::max(4, slots / GTIER_MAX_INFLIGHT);
        max_ranges_ = cfg.max_fetch_ranges;
        g_ = gtier_open(files[0].c_str(), &cfg);
        if (!g_) throw std::runtime_error("gtier_open failed: " + files[0]);
        for (size_t i = 1; i < files.size(); ++i)
            if (gtier_add_file(g_, files[i].c_str()) != (int)i)
                throw std::runtime_error("gtier_add_file failed: " + files[i]);
        arena_bytes_ = (uint64_t)(arena_gib * (1ull << 30));
        last_.assign(units_.size(), -1e18); hist_.assign(units_.size(), 0);
        pf_.assign(units_.size(), 0);
    }
    ~Engine() {
        if (g_) gtier_close(g_);
        if (arena_) cudaFreeHost(arena_);
    }

    void add_unit(int layer, int expert, std::vector<std::vector<int64_t>> projs) {
        Unit &u = units_[(size_t)layer * E_ + expert];
        for (int q = 0; q < 3; ++q) {
            auto &v = projs[q];   // file, off, len, rows, cols
            u.p[q] = Proj{(int)v[0], (uint64_t)v[1], (uint64_t)v[2], v[3], v[4]};
            u.bytes += (uint64_t)v[2];
        }
    }

    // After every unit is registered: the arena holds whole units back to back.
    void finalize() {
        for (auto &u : units_) unit_bytes_ = std::max(unit_bytes_, u.bytes);
        n_slots_ = unit_bytes_ ? (int)(arena_bytes_ / unit_bytes_) : 0;
        if (n_slots_ > 0) {
            CK(cudaHostAlloc((void **)&arena_, (size_t)n_slots_ * unit_bytes_, cudaHostAllocMapped));
            CK(cudaHostGetDevicePointer((void **)&arena_dev_, arena_, 0));
        }
        free_.clear();
        for (int s = n_slots_ - 1; s >= 0; --s) free_.push_back(s);
    }
    int arena_units() const { return n_slots_; }

    // One serving step: a prefill forward or one decode token.
    void step() { clk_ += 1.0; }
    // A new request: its prompt routing replaces the previous one's.
    void begin_request() { std::fill(pf_.begin(), pf_.end(), 0.0); pf_max_ = 0; }
    // Prompt routing of one layer, known once that layer's router has run.
    void note_prefill(int layer, torch::Tensor counts) {
        auto c = counts.to(torch::kCPU).to(torch::kFloat64).contiguous();
        const double *p = c.data_ptr<double>();
        for (int e = 0; e < E_; ++e) {
            pf_[(size_t)layer * E_ + e] = p[e];
            pf_max_ = std::max(pf_max_, p[e]);
        }
    }

    // Submit the reads of the experts in `ids` of `layer` that are not
    // resident; returns a handle.  At most GTIER_MAX_INFLIGHT tickets per file
    // are open at once.
    int64_t submit(int layer, std::vector<int64_t> ids) {
        Pending pd; pd.layer = layer; pd.ids = ids;
        std::vector<gtier_range> v;
        for (auto e : ids) {
            int id = layer * E_ + (int)e;
            // A resident unit a pending handle will read may not be evicted
            // before that handle is collected.
            if (res_.count(id)) { ++pin_[id]; pd.pinned.push_back(id); continue; }
            for (int q = 0; q < 3; ++q) {
                auto &p = units_[id].p[q];
                v.push_back({GTIER_FILE_OFF(p.file, p.off), (size_t)p.len});
                pd.tags[0].push_back({id, q});
            }
        }
        for (size_t o = 0; o < v.size(); o += max_ranges_) {
            int n = (int)std::min(v.size() - o, (size_t)max_ranges_);
            gtier_ticket t;
            int rc = gtier_submit(g_, v.data() + o, n, &t);
            if (rc) throw std::runtime_error("gtier_submit " + std::to_string(rc));
            pd.tickets.push_back({0, t, (int)o, n});
        }
        int64_t h = next_handle_++;
        pending_[h] = std::move(pd);
        return h;
    }

    // Collect a submitted set: per expert, [gate, up, down] CUDA tensors that
    // alias the arena or a staging slot.  Staged tensors stay valid until
    // their ticket's slots are reused, i.e. until two more submits later.
    std::vector<std::vector<torch::Tensor>> collect(int64_t h, bool from_prefill) {
        auto it = pending_.find(h);
        if (it == pending_.end()) throw std::runtime_error("unknown handle");
        Pending pd = std::move(it->second); pending_.erase(it);
        std::unordered_map<int, std::array<void *, 3>> staged;
        for (auto &tk : pd.tickets) {
            std::vector<void *> out(tk.n);
            if (gtier_wait(g_, &tk.t, out.data())) throw std::runtime_error("gtier_wait");
            auto &tags = pd.tags[tk.file];
            for (int k = 0; k < tk.n; ++k) {
                auto &tg = tags[tk.off + k];
                staged[tg.first][tg.second] = out[k];
                bytes_read_ += units_[tg.first].p[tg.second].len;
            }
        }
        std::vector<std::vector<torch::Tensor>> r;
        auto opt = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, 0);
        for (auto e : pd.ids) {
            int id = pd.layer * E_ + (int)e;
            last_[id] = clk_;
            if (!from_prefill) { hist_[id] += 1; hist_max_ = std::max(hist_max_, hist_[id]); }
            std::vector<torch::Tensor> w;
            auto rs = res_.find(id);
            uint8_t *base = nullptr;
            if (rs != res_.end()) { base = arena_dev_ + (size_t)rs->second * unit_bytes_; ++hits_; }
            for (int q = 0; q < 3; ++q) {
                auto &p = units_[id].p[q];
                void *ptr;
                if (base) { ptr = base; base += p.len; }
                else ptr = staged[id][q];
                // cudaHostAlloc(Mapped) memory is GPU-addressable on this SoC;
                // naming the device skips torch's host-pointer check.
                w.push_back(at::for_blob(ptr, {p.rows, p.cols}).options(opt)
                                .target_device(c10::Device(c10::kCUDA, 0)).make_tensor());
            }
            if (rs == res_.end()) { ++misses_; admit(id, staged[id]); }
            r.push_back(std::move(w));
        }
        for (int id : pd.pinned) if (--pin_[id] == 0) pin_.erase(id);
        return r;
    }

    std::vector<double> stats() const {
        return {(double)hits_, (double)misses_, (double)bytes_read_, (double)res_.size(),
                (double)n_slots_, (double)unit_bytes_};
    }

private:
    struct Tk { int file; gtier_ticket t; int off; int n; };
    struct Pending {
        int layer = 0; std::vector<int64_t> ids; std::vector<Tk> tickets; std::vector<int> pinned;
        std::unordered_map<int, std::vector<std::pair<int, int>>> tags;
    };

    double value(int id) const {
        if (policy_ == "lru") return last_[id];
        double rc = std::exp2(-(clk_ - last_[id]) / rec_half_);
        double h = hist_max_ > 0 ? hist_[id] / hist_max_ : 0.0;
        double a = pf_max_ > 0 ? pf_[id] / pf_max_ : 0.0;
        return w_rec_ * rc + mix_ * a + (1.0 - mix_) * h;
    }

    // A staged unit is already in memory; admitting it costs a copy.  It
    // enters a free slot, or replaces the least valuable resident when it is
    // worth more (PHASOR), or the least recent (LRU).
    void admit(int id, std::array<void *, 3> &src) {
        if (n_slots_ == 0) return;
        int slot = -1;
        if (!free_.empty()) { slot = free_.back(); free_.pop_back(); }
        else {
            int worst = -1; double wv = 1e300;
            for (auto &kv : res_) {
                if (last_[kv.first] == clk_) continue;       // in use by this step
                if (pin_.count(kv.first)) continue;          // a pending handle reads it
                double v = value(kv.first);
                if (v < wv) { wv = v; worst = kv.first; }
            }
            if (worst < 0) return;
            if (policy_ != "lru" && value(id) <= wv) return;
            slot = res_[worst]; res_.erase(worst);
        }
        uint8_t *dst = arena_dev_ + (size_t)slot * unit_bytes_;
        for (int q = 0; q < 3; ++q) {
            CK(cudaMemcpyAsync(dst, src[q], units_[id].p[q].len, cudaMemcpyDefault));
            dst += units_[id].p[q].len;
        }
        res_[id] = slot;
    }

    int L_, E_;
    std::vector<Unit> units_;
    gtier *g_ = nullptr;
    int max_ranges_ = 0;
    uint64_t arena_bytes_ = 0, unit_bytes_ = 0;
    int n_slots_ = 0;
    uint8_t *arena_ = nullptr, *arena_dev_ = nullptr;
    std::vector<int> free_;
    std::unordered_map<int, int> res_, pin_;
    std::string policy_;
    double mix_, rec_half_, w_rec_;
    double clk_ = 0;
    std::vector<double> last_, hist_, pf_;
    double hist_max_ = 0, pf_max_ = 0;
    std::unordered_map<int64_t, Pending> pending_;
    int64_t next_handle_ = 1;
    uint64_t hits_ = 0, misses_ = 0, bytes_read_ = 0;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<Engine>(m, "Engine")
        .def(pybind11::init<std::vector<std::string>, int, int, double, int64_t, double,
                            std::string, double, double, double>())
        .def("add_unit", &Engine::add_unit)
        .def("finalize", &Engine::finalize)
        .def("arena_units", &Engine::arena_units)
        .def("step", &Engine::step)
        .def("begin_request", &Engine::begin_request)
        .def("note_prefill", &Engine::note_prefill)
        .def("submit", &Engine::submit)
        .def("collect", &Engine::collect)
        .def("stats", &Engine::stats);
}
