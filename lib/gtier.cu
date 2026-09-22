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
#include <unordered_set>
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
    // Resident blocks.  cache maps a block index to the slot holding it; the
    // LRU list orders them.  Slots above cache_slots are scratch, used for the
    // exact fetches that serve misses without amplifying them.
    std::unordered_map<uint64_t, int> cache;
    std::list<uint64_t> lru;               // front = most recent
    std::vector<uint64_t> slot_block;
    std::vector<std::list<uint64_t>::iterator> slot_lru;
    std::unordered_map<uint64_t, int> seen; // HYBRID: accesses before admission
    int cache_slots = 0;                   // slots reserved for resident blocks

    // ADAPTIVE regime state.  caching = true means blocks are being admitted.
    //
    // The detector must not count a first touch against the hit rate.  A cold
    // start is all compulsory misses and looks exactly like thrashing, so a
    // naive hit-rate trigger leaves the caching regime during the first pass
    // over the data and never returns.  `ever` records which blocks have been
    // seen at least once; only misses on those are capacity misses and evidence
    // that the working set does not fit.
    bool caching = true;
    std::unordered_set<uint64_t> ever;
    uint64_t win_hits = 0, win_reqs = 0, switches = 0;
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

    if (!g->cfg.admit_after) g->cfg.admit_after = 2;
    if (g->cfg.cache_policy != GTIER_CACHE_NONE) {
        // Hybrid keeps a quarter of the window as scratch so a miss always has
        // somewhere to land without evicting a resident block.
        int want = g->cfg.cache_blocks ? g->cfg.cache_blocks : g->cfg.slots;
        // Only HYBRID needs scratch: it serves misses with exact fetches that
        // must not evict a resident block.  ADAPTIVE admits misses to blocks
        // while caching and uses the slots directly once it stops, so reserving
        // scratch for it only shrinks the resident set.
        if (g->cfg.cache_policy == GTIER_CACHE_HYBRID ||
            g->cfg.cache_policy == GTIER_CACHE_PIN) {
            // Scratch holds the exact fetches that serve non-admitted misses.
            // HYBRID can miss on every range of a fetch, so it needs a full
            // fetch's worth.  PIN only falls back once the cache is full and
            // then reuses scratch across waves, so a small reserve suffices --
            // and taking more would shrink the resident set, which is the whole
            // point of pinning.
            int scratch_need = g->cfg.max_fetch_ranges ? g->cfg.max_fetch_ranges
                                                       : std::max(1, g->cfg.slots / 4);
            if (g->cfg.cache_policy == GTIER_CACHE_PIN)
                scratch_need = std::min(scratch_need, std::max(8, g->cfg.slots / 16));
            scratch_need = std::min(scratch_need, g->cfg.slots - 1);
            want = std::min(want, g->cfg.slots - scratch_need);
            if (want < 1) want = 1;
        }
        g->cache_slots = std::min(want, g->cfg.slots);
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

// O_DIRECT needs the length aligned as well as the offset.  A file whose size
// is not a multiple of the block size therefore cannot have its tail read by
// clamping to EOF -- that yields an unaligned length and EINVAL.  Reading an
// aligned length past EOF is legal and simply returns short, so the plan keeps
// the aligned span and lets the short read happen.
std::vector<Plan> plan_exact(const gtier_range *r, int n, size_t merge_gap,
                             size_t slot_bytes, off_t file_bytes) {
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return r[a].off < r[b].off; });
    std::vector<Plan> ps;
    for (int k = 0; k < n; ++k) {
        int i = idx[k];
        uint64_t lo = adown(r[i].off);
        uint64_t hi = aup(r[i].off + r[i].len);
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
        // A range must land in one slot: the caller gets a single pointer, so
        // its bytes have to be contiguous.  O_DIRECT alignment can push a
        // slot-sized range over the slot, so callers must keep ranges below
        // slot_bytes minus two alignment units.  Oversized ranges are rejected
        // rather than silently split across slots.
        if (hi - lo > slot_bytes) { ps.clear(); return ps; }
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
    uint64_t admitted = 0, exact = 0;

    if (g->cfg.cache_policy == GTIER_CACHE_NONE) {
        auto ps = plan_exact(r, n, g->cfg.merge_gap, blk, g->file_bytes);
        if (ps.empty()) return -E2BIG;                 // a range exceeds a slot
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
            if (res < 0) return res;   // a short read at EOF is fine; res >= 0
            bytes_read += res;
        }
        issued = ps.size();
        for (size_t p = 0; p < ps.size(); ++p)
            for (int m : ps[p].members) {
                out[m] = g->dev[p] + (r[m].off - ps[p].off);
                useful += r[m].len;
            }
        g->st = gtier_stats{(uint64_t)n, issued, useful, bytes_read, 0, 0, 0, 0, 0, now_s() - t0};
        return 0;
    }

    // Cached policies.  A block is the unit of residency; whether a miss is
    // also fetched as a block is what separates BLOCK from HYBRID.
    //
    // ADAPTIVE decides per fetch.  While the resident set is serving most
    // requests, admitting blocks is right and worth its amplification; once the
    // hit rate falls the working set no longer fits and every admission pays
    // amplification for a block that will be evicted before reuse, which is 8x
    // worse than fetching exactly.  Hysteresis (leave below 40%, return above
    // 70%) keeps it from oscillating at the boundary.
    bool hybrid = g->cfg.cache_policy == GTIER_CACHE_HYBRID;
    if (g->cfg.cache_policy == GTIER_CACHE_ADAPTIVE) {
        if (g->win_reqs >= 512) {
            double hr = (double)g->win_hits / g->win_reqs;
            if (g->caching && hr < 0.40) { g->caching = false; ++g->switches; }
            else if (!g->caching && hr > 0.70) { g->caching = true; ++g->switches; }
            g->win_hits = g->win_reqs = 0;
        }
        if (!g->caching) {
            // exact fetches only, but keep probing residency so the detector
            // can notice the working set shrinking again
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
            for (size_t p = 0; p < ps.size(); ++p)
                for (int m : ps[p].members) {
                    out[m] = g->dev[p] + (r[m].off - ps[p].off);
                    useful += r[m].len;
                }
            for (int i = 0; i < n; ++i) {
                uint64_t b = r[i].off / blk;
                if (g->ever.count(b)) {
                    ++g->win_reqs;
                    if (g->cache.count(b)) ++g->win_hits;
                } else {
                    g->ever.insert(b);
                }
            }
            g->st = gtier_stats{(uint64_t)n, ps.size(), useful, bytes_read,
                                0, (uint64_t)n, 0, ps.size(), g->switches, now_s() - t0};
            return 0;
        }
        hybrid = false;   // in the caching regime, admit blocks
    }

    auto touch = [&](uint64_t b, int slot) {
        g->lru.erase(g->slot_lru[slot]);
        g->lru.push_front(b);
        g->slot_lru[slot] = g->lru.begin();
    };
    const bool pinning = g->cfg.cache_policy == GTIER_CACHE_PIN;
    auto victim_slot = [&]() {
        if ((int)g->cache.size() < g->cache_slots) return (int)g->cache.size();
        if (pinning) return -1;            // full and pinned: admit nothing more
        uint64_t v = g->lru.back();
        int slot = g->cache[v];
        g->cache.erase(v);
        g->lru.pop_back();
        return slot;
    };

    struct Pending { uint64_t off; size_t len; int slot; bool block; };
    std::vector<Pending> pend;
    std::vector<int> range_slot(n, -1);
    std::vector<uint64_t> range_base(n, 0);
    std::unordered_map<uint64_t, int> planned;   // block -> slot, this fetch
    int scratch = g->cache_slots;

    for (int i = 0; i < n; ++i) {
        uint64_t b = r[i].off / blk;
        // A cached range is served from one block, so it must lie inside one.
        // Straddling would hand back a pointer whose bytes run off the end of
        // the slot.  The caller aligns its ranges to the block grid.
        if ((r[i].off + r[i].len - 1) / blk != b) return -E2BIG;
        auto c = g->cache.find(b);
        if (c != g->cache.end()) {
            ++hits;
            touch(b, c->second);
            range_slot[i] = c->second;
            range_base[i] = b * blk;
            continue;
        }
        auto pl = planned.find(b);
        if (pl != planned.end()) {           // already being fetched this round
            range_slot[i] = pl->second;
            range_base[i] = b * blk;
            continue;
        }
        ++misses;

        // Admit on the configured access count.  A block seen once may never be
        // seen again; paying block amplification for it is the failure mode
        // that makes pure BLOCK caching 8x worse than exact fetching when the
        // working set does not fit.
        bool admit = !hybrid;
        if (hybrid && ++g->seen[b] >= g->cfg.admit_after) { admit = true; g->seen.erase(b); }

        int slot = admit ? victim_slot() : -1;
        if (slot < 0) admit = false;        // pinned cache full: fetch exactly
        if (admit) {
            g->cache[b] = slot;
            g->slot_block[slot] = b;
            g->lru.push_front(b);
            g->slot_lru[slot] = g->lru.begin();
            planned[b] = slot;
            size_t len = aup(std::min<size_t>(blk, (size_t)(g->file_bytes - b * blk)));
            pend.push_back({b * blk, len, slot, true});
            range_slot[i] = slot;
            range_base[i] = b * blk;
            ++admitted;
        } else {
            if (scratch >= g->cfg.slots) return -ENOSPC;   // caller must batch smaller
            slot = scratch++;
            uint64_t lo = adown(r[i].off);
            uint64_t hi = aup(r[i].off + r[i].len);
            pend.push_back({lo, (size_t)(hi - lo), slot, false});
            range_slot[i] = slot;
            range_base[i] = lo;
            ++exact;
        }
    }

    for (auto &p : pend) {
        io_uring_sqe *s = io_uring_get_sqe(&g->ring);
        if (!s) return -EBUSY;
        io_uring_prep_read(s, g->fd, g->host[p.slot], p.len, (off_t)p.off);
        io_uring_sqe_set_data64(s, (uint64_t)p.slot);
    }
    if (!pend.empty()) io_uring_submit(&g->ring);
    for (size_t d = 0; d < pend.size(); ++d) {
        io_uring_cqe *c;
        if (io_uring_wait_cqe(&g->ring, &c) < 0) return -EIO;
        int res = c->res; io_uring_cqe_seen(&g->ring, c);
        if (res < 0) return res;
        bytes_read += res;
    }
    issued = pend.size();
    for (int i = 0; i < n; ++i) {
        out[i] = g->dev[range_slot[i]] + (r[i].off - range_base[i]);
        useful += r[i].len;
    }
    // Score the window on capacity behaviour only: a first touch is
    // compulsory and says nothing about whether the set fits.
    {
        uint64_t scored = 0, scored_hits = 0;
        for (int i = 0; i < n; ++i) {
            uint64_t b = r[i].off / blk;
            if (g->ever.count(b)) {
                ++scored;
                if (g->cache.count(b)) ++scored_hits;
            } else {
                g->ever.insert(b);
            }
        }
        g->win_hits += scored_hits; g->win_reqs += scored;
    }
    g->st = gtier_stats{(uint64_t)n, issued, useful, bytes_read,
                        hits, misses, admitted, exact, g->switches, now_s() - t0};
    return 0;
}

// The OS path: hand the GPU a pointer into the mapping and let it fault.
static int fetch_mmap_gpu(gtier *g, const gtier_range *r, int n, void **out) {
    double t0 = now_s();
    uint64_t useful = 0;
    for (int i = 0; i < n; ++i) { out[i] = g->map_dev + r[i].off; useful += r[i].len; }
    g->st = gtier_stats{(uint64_t)n, 0, useful, 0, 0, 0, 0, 0, 0, now_s() - t0};
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
    g->st = gtier_stats{(uint64_t)n, (uint64_t)n, useful, useful, 0, 0, 0, 0, 0, now_s() - t0};
    return 0;
}

// pread into pinned host memory, then cudaMemcpy to a device buffer -- what an
// offloading system written for a discrete GPU does.
static int fetch_pread_copy(gtier *g, const gtier_range *r, int n, void **out) {
    auto ps = plan_exact(r, n, 0, g->cfg.slot_bytes, g->file_bytes);
    if (ps.empty()) return -E2BIG;
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
    g->st = gtier_stats{(uint64_t)n, ps.size(), useful, bytes, 0, 0, 0, 0, 0, now_s() - t0};
    return 0;
}

static int fetch_cufile(gtier *g, const gtier_range *r, int n, void **out) {
    auto ps = plan_exact(r, n, 0, g->cfg.slot_bytes, g->file_bytes);
    if (ps.empty()) return -E2BIG;
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
    g->st = gtier_stats{(uint64_t)n, ps.size(), useful, bytes, 0, 0, 0, 0, 0, now_s() - t0};
    return 0;
}

static int fetch_uvm(gtier *g, const gtier_range *r, int n, void **out) {
    auto ps = plan_exact(r, n, 0, g->cfg.slot_bytes, g->file_bytes);
    if (ps.empty()) return -E2BIG;
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
    g->st = gtier_stats{(uint64_t)n, ps.size(), useful, bytes, 0, 0, 0, 0, 0, now_s() - t0};
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
    const int saved_pol = g->cfg.cache_policy;
    g->cfg.cache_policy = GTIER_CACHE_NONE;
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
    g->cfg.cache_policy = saved_pol;
    return best;
}
