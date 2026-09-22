#include "gtier.h"

#include <cuda_runtime.h>
#include <cufile.h>
#include <errno.h>
#include <fcntl.h>
#include <liburing.h>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <list>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {
constexpr size_t kAlign = 4096;
inline uint64_t adown(uint64_t x) { return x & ~(uint64_t)(kAlign - 1); }
inline uint64_t aup(uint64_t x) { return (x + kAlign - 1) & ~(uint64_t)(kAlign - 1); }
inline double now_s() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

const char *gtier_backend_name(gtier_backend b) {
    switch (b) {
        case GTIER_BACKEND_GTIER:      return "gtier";
        case GTIER_BACKEND_MMAP_GPU:   return "mmap-gpu";
        case GTIER_BACKEND_MMAP_CPU:   return "mmap-cpu";
        case GTIER_BACKEND_PREAD_COPY: return "pread+copy";
        case GTIER_BACKEND_CUFILE:     return "cufile";
        case GTIER_BACKEND_UVM:        return "uvm";
        default:                       return "?";
    }
}

struct gtier {
    gtier_config cfg{};
    int fd = -1;
    off_t file_bytes = 0;
    gtier_stats st{};

    // gtier / cufile / pread backends
    io_uring ring{};
    bool ring_ready = false;
    std::vector<uint8_t *> host, dev;      // pinned, GPU-mapped window
    std::vector<uint8_t *> devbuf;         // cudaMalloc targets (pread/cufile)
    CUfileHandle_t cufh{};
    bool cufile_ready = false;

    // mmap backends
    uint8_t *map = nullptr;
    uint8_t *map_dev = nullptr;

    // uvm backend
    uint8_t *uvm = nullptr;

    // block cache: aligned block index -> slot
    std::unordered_map<uint64_t, int> cache;
    std::list<uint64_t> lru;               // front = most recent
    std::vector<uint64_t> slot_block;      // slot -> block index (or UINT64_MAX)
    std::vector<std::list<uint64_t>::iterator> slot_lru;
};

// ---------------------------------------------------------------- open/close

static bool init_window(gtier *g) {
    g->host.resize(g->cfg.slots);
    g->dev.resize(g->cfg.slots);
    for (int i = 0; i < g->cfg.slots; ++i) {
        if (cudaHostAlloc((void **)&g->host[i], g->cfg.slot_bytes,
                          cudaHostAllocMapped) != cudaSuccess) return false;
        if (cudaHostGetDevicePointer((void **)&g->dev[i], g->host[i], 0) != cudaSuccess)
            return false;
        if ((uintptr_t)g->host[i] % kAlign) return false;
    }
    return true;
}

gtier *gtier_open(const char *path, const gtier_config *user) {
    gtier *g = new gtier();
    g->cfg = user ? *user : gtier_config{};
    if (!g->cfg.slot_bytes) g->cfg.slot_bytes = 1u << 20;
    if (!g->cfg.slots) g->cfg.slots = 8;
    if (!g->cfg.queue_depth) g->cfg.queue_depth = g->cfg.slots;

    const bool direct = g->cfg.backend == GTIER_BACKEND_GTIER ||
                        g->cfg.backend == GTIER_BACKEND_CUFILE ||
                        g->cfg.backend == GTIER_BACKEND_PREAD_COPY;
    g->fd = open(path, O_RDONLY | (direct ? O_DIRECT : 0));
    if (g->fd < 0) { delete g; return nullptr; }
    g->file_bytes = lseek(g->fd, 0, SEEK_END);

    switch (g->cfg.backend) {
        case GTIER_BACKEND_GTIER:
            if (!init_window(g)) { gtier_close(g); return nullptr; }
            if (io_uring_queue_init(g->cfg.queue_depth, &g->ring, 0) < 0) {
                gtier_close(g); return nullptr;
            }
            g->ring_ready = true;
            if (g->cfg.merge_gap == SIZE_MAX)
                g->cfg.merge_gap = gtier_calibrate_merge_gap(g);
            break;

        case GTIER_BACKEND_PREAD_COPY:
            // The discrete-GPU pattern: read into host memory, copy to a device
            // buffer.  On this SoC the copy is redundant, which is the point.
            if (!init_window(g)) { gtier_close(g); return nullptr; }
            g->devbuf.resize(g->cfg.slots);
            for (int i = 0; i < g->cfg.slots; ++i)
                if (cudaMalloc((void **)&g->devbuf[i], g->cfg.slot_bytes) != cudaSuccess) {
                    gtier_close(g); return nullptr;
                }
            break;

        case GTIER_BACKEND_CUFILE: {
            cuFileDriverOpen();
            CUfileDescr_t d{}; d.handle.fd = g->fd; d.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
            if (cuFileHandleRegister(&g->cufh, &d).err != CU_FILE_SUCCESS) {
                gtier_close(g); return nullptr;
            }
            g->cufile_ready = true;
            g->devbuf.resize(g->cfg.slots);
            for (int i = 0; i < g->cfg.slots; ++i) {
                if (cudaMalloc((void **)&g->devbuf[i], g->cfg.slot_bytes) != cudaSuccess) {
                    gtier_close(g); return nullptr;
                }
                cuFileBufRegister(g->devbuf[i], g->cfg.slot_bytes, 0);
            }
            break;
        }

        case GTIER_BACKEND_MMAP_GPU:
        case GTIER_BACKEND_MMAP_CPU:
            g->map = (uint8_t *)mmap(nullptr, g->file_bytes, PROT_READ, MAP_SHARED, g->fd, 0);
            if (g->map == MAP_FAILED) { g->map = nullptr; gtier_close(g); return nullptr; }
            g->map_dev = g->map;   // coherent SoC: the GPU dereferences it directly
            break;

        case GTIER_BACKEND_UVM:
            // cudaMallocManaged staging, the shape DeepUM-style systems assume.
            if (cudaMallocManaged((void **)&g->uvm,
                                  (size_t)g->cfg.slots * g->cfg.slot_bytes) != cudaSuccess) {
                gtier_close(g); return nullptr;
            }
            break;
        default: gtier_close(g); return nullptr;
    }

    if (g->cfg.cache_blocks > 0) {
        g->slot_block.assign(g->cfg.slots, UINT64_MAX);
        g->slot_lru.resize(g->cfg.slots);
    }
    return g;
}

void gtier_close(gtier *g) {
    if (!g) return;
    if (g->ring_ready) io_uring_queue_exit(&g->ring);
    for (auto *p : g->host) if (p) cudaFreeHost(p);
    for (auto *p : g->devbuf) if (p) cudaFree(p);
    if (g->cufile_ready) { cuFileHandleDeregister(g->cufh); cuFileDriverClose(); }
    if (g->map) munmap(g->map, g->file_bytes);
    if (g->uvm) cudaFree(g->uvm);
    if (g->fd >= 0) close(g->fd);
    delete g;
}

void gtier_get_stats(const gtier *g, gtier_stats *o) { if (g && o) *o = g->st; }
void gtier_reset_stats(gtier *g) { if (g) g->st = gtier_stats{}; }

// ------------------------------------------------------------------ planning

namespace {
struct Plan { uint64_t off; size_t len; std::vector<int> members; };

std::vector<Plan> plan_exact(const gtier_range *r, int n, size_t merge_gap,
                             size_t slot_bytes, off_t file_bytes) {
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return r[a].off < r[b].off; });
    std::vector<Plan> ps;
    for (int k = 0; k < n; ++k) {
        int i = idx[k];
        uint64_t lo = adown(r[i].off);
        uint64_t hi = std::min<uint64_t>(aup(r[i].off + r[i].len), (uint64_t)file_bytes);
        if (!ps.empty()) {
            Plan &p = ps.back();
            uint64_t pend = p.off + p.len;
            if (hi > p.off && hi - p.off <= slot_bytes &&
                (lo < pend || lo - pend <= merge_gap)) {
                p.len = std::max<uint64_t>(p.len, hi - p.off);
                p.members.push_back(i);
                continue;
            }
        }
        ps.push_back(Plan{lo, (size_t)(hi - lo), {i}});
    }
    return ps;
}
}  // namespace

// ------------------------------------------------------------------- backends

static int fetch_gtier(gtier *g, const gtier_range *r, int n, void **out) {
    const size_t blk = g->cfg.slot_bytes;
    double t0 = now_s();
    uint64_t bytes_read = 0, useful = 0, issued = 0, hits = 0, misses = 0;

    if (g->cfg.cache_blocks > 0) {
        // Fixed-size blocks so a block can be reused; amplification is the
        // price, and reuse is what repays it.
        std::vector<uint64_t> want;
        want.reserve(n);
        for (int i = 0; i < n; ++i) want.push_back(r[i].off / blk);
        std::sort(want.begin(), want.end());
        want.erase(std::unique(want.begin(), want.end()), want.end());

        std::vector<std::pair<uint64_t, int>> to_read;
        for (uint64_t b : want) {
            auto it = g->cache.find(b);
            if (it != g->cache.end()) {
                ++hits;
                g->lru.erase(g->slot_lru[it->second]);
                g->lru.push_front(b);
                g->slot_lru[it->second] = g->lru.begin();
                continue;
            }
            ++misses;
            int slot;
            if ((int)g->cache.size() < g->cfg.slots) {
                slot = (int)g->cache.size();
            } else {
                uint64_t victim = g->lru.back();
                slot = g->cache[victim];
                g->cache.erase(victim);
                g->lru.pop_back();
            }
            g->cache[b] = slot;
            g->slot_block[slot] = b;
            g->lru.push_front(b);
            g->slot_lru[slot] = g->lru.begin();
            to_read.emplace_back(b, slot);
        }
        for (auto &pr : to_read) {
            io_uring_sqe *s = io_uring_get_sqe(&g->ring);
            if (!s) return -EBUSY;
            size_t len = std::min<size_t>(blk, (size_t)(g->file_bytes - pr.first * blk));
            len = aup(len);
            io_uring_prep_read(s, g->fd, g->host[pr.second], len, (off_t)(pr.first * blk));
            io_uring_sqe_set_data64(s, pr.second);
        }
        if (!to_read.empty()) io_uring_submit(&g->ring);
        for (size_t d = 0; d < to_read.size(); ++d) {
            io_uring_cqe *c;
            if (io_uring_wait_cqe(&g->ring, &c) < 0) return -EIO;
            int res = c->res; io_uring_cqe_seen(&g->ring, c);
            if (res < 0) return res;
            bytes_read += res;
        }
        issued = to_read.size();
        for (int i = 0; i < n; ++i) {
            uint64_t b = r[i].off / blk;
            out[i] = g->dev[g->cache[b]] + (r[i].off - b * blk);
            useful += r[i].len;
        }
    } else {
        auto ps = plan_exact(r, n, g->cfg.merge_gap, blk, g->file_bytes);
        if ((int)ps.size() > g->cfg.slots) return -ENOSPC;
        for (size_t p = 0; p < ps.size(); ++p) {
            io_uring_sqe *s = io_uring_get_sqe(&g->ring);
            if (!s) return -EBUSY;
            io_uring_prep_read(s, g->fd, g->host[p], ps[p].len, (off_t)ps[p].off);
            io_uring_sqe_set_data64(s, p);
        }
        io_uring_submit(&g->ring);
        for (size_t d = 0; d < ps.size(); ++d) {
            io_uring_cqe *c;
            if (io_uring_wait_cqe(&g->ring, &c) < 0) return -EIO;
            int res = c->res; io_uring_cqe_seen(&g->ring, c);
            if (res < 0) return res;
            bytes_read += res;
        }
        issued = ps.size();
        for (size_t p = 0; p < ps.size(); ++p)
            for (int m : ps[p].members) {
                out[m] = g->dev[p] + (r[m].off - ps[p].off);
                useful += r[m].len;
            }
    }
    g->st = gtier_stats{(uint64_t)n, issued, useful, bytes_read, hits, misses, now_s() - t0};
    return 0;
}

// The OS path: hand the GPU a pointer into the mapping and let it fault.
static int fetch_mmap_gpu(gtier *g, const gtier_range *r, int n, void **out) {
    double t0 = now_s();
    uint64_t useful = 0;
    for (int i = 0; i < n; ++i) { out[i] = g->map_dev + r[i].off; useful += r[i].len; }
    g->st = gtier_stats{(uint64_t)n, 0, useful, 0, 0, 0, now_s() - t0};
    return 0;
}

// Same mapping, but CPU threads fault the pages in first so the kernel's
// readahead applies; the GPU then reads resident memory.
static int fetch_mmap_cpu(gtier *g, const gtier_range *r, int n, void **out) {
    double t0 = now_s();
    uint64_t useful = 0;
    const int nt = std::min(8, std::max(1, n));
    std::vector<std::thread> ts;
    volatile uint64_t sink = 0;
    for (int t = 0; t < nt; ++t)
        ts.emplace_back([&, t] {
            uint64_t acc = 0;
            for (int i = t; i < n; i += nt)
                for (size_t k = 0; k < r[i].len; k += kAlign) acc += g->map[r[i].off + k];
            sink += acc;
        });
    for (auto &t : ts) t.join();
    for (int i = 0; i < n; ++i) { out[i] = g->map_dev + r[i].off; useful += r[i].len; }
    g->st = gtier_stats{(uint64_t)n, (uint64_t)n, useful, useful, 0, 0, now_s() - t0};
    return 0;
}

// pread into pinned host memory, then cudaMemcpy to a device buffer -- what an
// offloading system written for a discrete GPU does.
static int fetch_pread_copy(gtier *g, const gtier_range *r, int n, void **out) {
    auto ps = plan_exact(r, n, 0, g->cfg.slot_bytes, g->file_bytes);
    if ((int)ps.size() > g->cfg.slots) return -ENOSPC;
    double t0 = now_s();
    std::vector<ssize_t> got(ps.size());
    const int nt = std::min<int>(8, ps.size());
    std::vector<std::thread> ts;
    for (int t = 0; t < nt; ++t)
        ts.emplace_back([&, t] {
            for (size_t p = t; p < ps.size(); p += nt)
                got[p] = pread(g->fd, g->host[p], ps[p].len, (off_t)ps[p].off);
        });
    for (auto &t : ts) t.join();
    uint64_t bytes = 0, useful = 0;
    for (size_t p = 0; p < ps.size(); ++p) {
        if (got[p] < 0) return -EIO;
        bytes += got[p];
        if (cudaMemcpyAsync(g->devbuf[p], g->host[p], got[p], cudaMemcpyHostToDevice) != cudaSuccess)
            return -EIO;
    }
    cudaDeviceSynchronize();
    for (size_t p = 0; p < ps.size(); ++p)
        for (int m : ps[p].members) {
            out[m] = g->devbuf[p] + (r[m].off - ps[p].off);
            useful += r[m].len;
        }
    g->st = gtier_stats{(uint64_t)n, ps.size(), useful, bytes, 0, 0, now_s() - t0};
    return 0;
}

static int fetch_cufile(gtier *g, const gtier_range *r, int n, void **out) {
    auto ps = plan_exact(r, n, 0, g->cfg.slot_bytes, g->file_bytes);
    if ((int)ps.size() > g->cfg.slots) return -ENOSPC;
    double t0 = now_s();
    std::vector<ssize_t> got(ps.size());
    const int nt = std::min<int>(8, ps.size());
    std::vector<std::thread> ts;
    for (int t = 0; t < nt; ++t)
        ts.emplace_back([&, t] {
            for (size_t p = t; p < ps.size(); p += nt)
                got[p] = cuFileRead(g->cufh, g->devbuf[p], ps[p].len, (off_t)ps[p].off, 0);
        });
    for (auto &t : ts) t.join();
    uint64_t bytes = 0, useful = 0;
    for (size_t p = 0; p < ps.size(); ++p) {
        if (got[p] < 0) return -EIO;
        bytes += got[p];
    }
    for (size_t p = 0; p < ps.size(); ++p)
        for (int m : ps[p].members) {
            out[m] = g->devbuf[p] + (r[m].off - ps[p].off);
            useful += r[m].len;
        }
    g->st = gtier_stats{(uint64_t)n, ps.size(), useful, bytes, 0, 0, now_s() - t0};
    return 0;
}

static int fetch_uvm(gtier *g, const gtier_range *r, int n, void **out) {
    auto ps = plan_exact(r, n, 0, g->cfg.slot_bytes, g->file_bytes);
    if ((int)ps.size() > g->cfg.slots) return -ENOSPC;
    double t0 = now_s();
    uint64_t bytes = 0, useful = 0;
    int dev = 0; cudaGetDevice(&dev);
    for (size_t p = 0; p < ps.size(); ++p) {
        uint8_t *dst = g->uvm + p * g->cfg.slot_bytes;
        ssize_t got = pread(g->fd, dst, ps[p].len, (off_t)ps[p].off);
        if (got < 0) return -errno;
        bytes += got;
#if CUDART_VERSION >= 13000
        cudaMemLocation loc{}; loc.type = cudaMemLocationTypeDevice; loc.id = dev;
        cudaMemPrefetchAsync(dst, got, loc, 0, 0);
#else
        cudaMemPrefetchAsync(dst, got, dev, 0);
#endif
    }
    cudaDeviceSynchronize();
    for (size_t p = 0; p < ps.size(); ++p)
        for (int m : ps[p].members) {
            out[m] = g->uvm + p * g->cfg.slot_bytes + (r[m].off - ps[p].off);
            useful += r[m].len;
        }
    g->st = gtier_stats{(uint64_t)n, ps.size(), useful, bytes, 0, 0, now_s() - t0};
    return 0;
}

int gtier_fetch(gtier *g, const gtier_range *r, int n, void **out) {
    if (!g || n <= 0) return -EINVAL;
    switch (g->cfg.backend) {
        case GTIER_BACKEND_GTIER:      return fetch_gtier(g, r, n, out);
        case GTIER_BACKEND_MMAP_GPU:   return fetch_mmap_gpu(g, r, n, out);
        case GTIER_BACKEND_MMAP_CPU:   return fetch_mmap_cpu(g, r, n, out);
        case GTIER_BACKEND_PREAD_COPY: return fetch_pread_copy(g, r, n, out);
        case GTIER_BACKEND_CUFILE:     return fetch_cufile(g, r, n, out);
        case GTIER_BACKEND_UVM:        return fetch_uvm(g, r, n, out);
        default:                       return -EINVAL;
    }
}

size_t gtier_calibrate_merge_gap(gtier *g) {
    // A parametric model does not describe this device -- its bandwidth-size
    // curve is neither linear nor monotonic -- so run the planner itself over
    // candidate thresholds and keep whichever delivers the most useful bytes.
    const size_t item = 64u << 10;
    const int n = std::max(8, std::min(g->cfg.slots, 32));
    const size_t cands[] = {0, 16u << 10, 64u << 10, 256u << 10, 1u << 20};
    std::vector<gtier_range> rs(n);
    std::vector<void *> outs(n);
    size_t best = 0; double best_bw = -1;
    uint64_t seed = 0x243F6A8885A308D3ull;
    const size_t saved = g->cfg.merge_gap;
    const int saved_cache = g->cfg.cache_blocks;
    g->cfg.cache_blocks = 0;
    for (size_t c : cands) {
        if (c > g->cfg.slot_bytes) continue;
        g->cfg.merge_gap = c;
        double t0 = now_s(); uint64_t useful = 0;
        for (int it = 0; it < 24; ++it) {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            uint64_t base = adown(seed % (uint64_t)(g->file_bytes / 2));
            for (int i = 0; i < n; ++i) rs[i] = {base + (uint64_t)i * item * 2, item};
            if (gtier_fetch(g, rs.data(), n, outs.data()) != 0) break;
            useful += (uint64_t)n * item;
        }
        double bw = (double)useful / (now_s() - t0);
        if (bw > best_bw) { best_bw = bw; best = c; }
    }
    g->cfg.merge_gap = saved;
    g->cfg.cache_blocks = saved_cache;
    return best;
}
