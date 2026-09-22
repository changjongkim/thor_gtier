#include "gtier.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <liburing.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
constexpr size_t kAlign = 4096;  // O_DIRECT alignment
inline uint64_t align_down(uint64_t x) { return x & ~(uint64_t)(kAlign - 1); }
inline uint64_t align_up(uint64_t x) { return (x + kAlign - 1) & ~(uint64_t)(kAlign - 1); }
inline double now_s() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

struct gtier {
    int fd = -1;
    off_t file_bytes = 0;
    gtier_config cfg{};
    io_uring ring{};
    bool ring_ready = false;
    std::vector<uint8_t *> host;  // pinned, GPU-mapped slots
    std::vector<uint8_t *> dev;
    gtier_stats stats{};
};

gtier *gtier_open(const char *path, const gtier_config *user_cfg) {
    gtier *g = new gtier();
    g->cfg = user_cfg ? *user_cfg : gtier_config{};
    if (!g->cfg.slot_bytes) g->cfg.slot_bytes = 1u << 20;   // measured optimum
    if (!g->cfg.slots) g->cfg.slots = 8;
    if (!g->cfg.queue_depth) g->cfg.queue_depth = g->cfg.slots;

    g->fd = open(path, O_RDONLY | O_DIRECT);
    if (g->fd < 0) { delete g; return nullptr; }
    g->file_bytes = lseek(g->fd, 0, SEEK_END);

    // The window is pinned and device-mapped: one allocation serves as both the
    // DMA destination and the GPU's view of the data.
    g->host.resize(g->cfg.slots);
    g->dev.resize(g->cfg.slots);
    for (int i = 0; i < g->cfg.slots; ++i) {
        if (cudaHostAlloc((void **)&g->host[i], g->cfg.slot_bytes,
                          cudaHostAllocMapped) != cudaSuccess) {
            gtier_close(g); return nullptr;
        }
        if (cudaHostGetDevicePointer((void **)&g->dev[i], g->host[i], 0) != cudaSuccess) {
            gtier_close(g); return nullptr;
        }
        if ((uintptr_t)g->host[i] % kAlign) { gtier_close(g); return nullptr; }
    }

    if (io_uring_queue_init(g->cfg.queue_depth, &g->ring, 0) < 0) {
        gtier_close(g); return nullptr;
    }
    g->ring_ready = true;

    if (g->cfg.merge_gap == SIZE_MAX) g->cfg.merge_gap = gtier_calibrate_merge_gap(g);
    return g;
}

void gtier_close(gtier *g) {
    if (!g) return;
    if (g->ring_ready) io_uring_queue_exit(&g->ring);
    for (auto *p : g->host) if (p) cudaFreeHost(p);
    if (g->fd >= 0) close(g->fd);
    delete g;
}

namespace {
// One planned device read, plus which caller ranges it satisfies.
struct Plan {
    uint64_t off;   // aligned
    size_t len;     // aligned
    std::vector<int> members;
};

// Rule 1 forbids inflating an isolated range, so each range is only aligned out
// to the O_DIRECT boundary.  Rule 2 merges neighbours whose gap is under the
// calibrated threshold, which trades gap bytes for one fewer request.
std::vector<Plan> plan_reads(const gtier_range *r, int n, size_t merge_gap,
                             size_t slot_bytes, off_t file_bytes) {
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return r[a].off < r[b].off; });

    std::vector<Plan> plans;
    for (int k = 0; k < n; ++k) {
        int i = idx[k];
        uint64_t lo = align_down(r[i].off);
        uint64_t hi = std::min<uint64_t>(align_up(r[i].off + r[i].len), file_bytes);
        if (!plans.empty()) {
            Plan &p = plans.back();
            uint64_t pend = p.off + p.len;
            // merge only if the gap is small AND the result still fits a slot
            if (lo >= pend && lo - pend <= merge_gap &&
                hi - p.off <= slot_bytes) {
                p.len = hi - p.off;
                p.members.push_back(i);
                continue;
            }
            if (lo < pend && hi - p.off <= slot_bytes) {  // overlapping
                p.len = std::max<uint64_t>(p.len, hi - p.off);
                p.members.push_back(i);
                continue;
            }
        }
        plans.push_back(Plan{lo, (size_t)(hi - lo), {i}});
    }
    return plans;
}
}  // namespace

int gtier_fetch(gtier *g, const gtier_range *r, int n, void **dev_out) {
    if (!g || n <= 0) return -EINVAL;
    auto plans = plan_reads(r, n, g->cfg.merge_gap, g->cfg.slot_bytes, g->file_bytes);
    if ((int)plans.size() > g->cfg.slots) return -ENOSPC;  // caller must batch

    double t0 = now_s();
    uint64_t bytes_read = 0;
    for (size_t p = 0; p < plans.size(); ++p) {
        io_uring_sqe *sqe = io_uring_get_sqe(&g->ring);
        if (!sqe) return -EBUSY;
        io_uring_prep_read(sqe, g->fd, g->host[p], plans[p].len, plans[p].off);
        io_uring_sqe_set_data64(sqe, p);
    }
    io_uring_submit(&g->ring);
    for (size_t done = 0; done < plans.size(); ++done) {
        io_uring_cqe *cqe;
        if (io_uring_wait_cqe(&g->ring, &cqe) < 0) return -EIO;
        int res = cqe->res;
        io_uring_cqe_seen(&g->ring, cqe);
        if (res < 0) return res;
        bytes_read += res;
    }
    double t = now_s() - t0;

    uint64_t useful = 0;
    for (size_t p = 0; p < plans.size(); ++p)
        for (int m : plans[p].members) {
            dev_out[m] = g->dev[p] + (r[m].off - plans[p].off);
            useful += r[m].len;
        }

    g->stats = gtier_stats{(uint64_t)n, (uint64_t)plans.size(), useful, bytes_read, t};
    return 0;
}

void gtier_release(gtier *g) { (void)g; }

void gtier_get_stats(const gtier *g, gtier_stats *out) { if (g && out) *out = g->stats; }

size_t gtier_calibrate_merge_gap(gtier *g) {
    // A parametric model -- fixed request cost plus a per-byte cost -- does not
    // describe this device: its bandwidth-versus-size curve is neither linear
    // nor monotonic, so extrapolating a merge threshold from two point
    // measurements gives the wrong answer.  Calibrate by running the planner
    // itself over candidate thresholds on clustered ranges and keeping the one
    // that delivers the most useful bytes per second.
    const size_t item = 64u << 10;
    const int n = std::max(8, std::min(g->cfg.slots, 32));
    const size_t candidates[] = {0, 16u << 10, 64u << 10, 256u << 10, 1u << 20};

    std::vector<gtier_range> rs(n);
    std::vector<void *> outs(n);
    size_t best_gap = 0;
    double best_bw = -1;
    uint64_t seed = 0x243F6A8885A308D3ull;

    const size_t saved = g->cfg.merge_gap;
    for (size_t cand : candidates) {
        if (cand > g->cfg.slot_bytes) continue;
        g->cfg.merge_gap = cand;
        double t0 = now_s();
        uint64_t useful = 0;
        for (int it = 0; it < 24; ++it) {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            uint64_t base = (seed % (uint64_t)(g->file_bytes / 2)) & ~(uint64_t)(kAlign - 1);
            for (int i = 0; i < n; ++i) rs[i] = {base + (uint64_t)i * item * 2, item};
            if (gtier_fetch(g, rs.data(), n, outs.data()) != 0) break;
            useful += (uint64_t)n * item;
        }
        double bw = (double)useful / (now_s() - t0);
        if (bw > best_bw) { best_bw = bw; best_gap = cand; }
    }
    g->cfg.merge_gap = saved;
    return best_gap;
}
