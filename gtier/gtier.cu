// gTier v0 -- an explicitly managed GPU residency window for out-of-core data
// on a coherent SoC, measured against the OS demand-paging path.
//
// Two questions, one binary:
//
//   (1) go/kill.  The OS path (GPU faulting on file-backed mmap) delivers
//       0.19-1.35 GiB/s on this device and fails nondeterministically above
//       ~48 GiB.  Does a bounded, pinned window fed by io_uring beat it, and
//       does it stay up past the sizes where mmap dies?
//
//   (2) the exchange rate.  Every offloading system -- FlexGen, LLM in a Flash,
//       PowerInfer, ZeRO-Infinity -- spends CPU cycles to save I/O, on the
//       premise that the CPU is free while the GPU computes.  On this SoC the
//       GPU alone already reaches 91% of memory bandwidth, so CPU work is
//       charged against the GPU.  --cpu-work bytes/slot measures what that
//       costs, which is the number that inverts those designs.
//
// The GPU does identical work in both modes: one 8-byte read per 4 KiB of file,
// so "effective bandwidth" means the same thing on both paths.

#include <cuda_runtime.h>
#include <fcntl.h>
#include <liburing.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

#define CK(c)                                                              \
  do {                                                                     \
    cudaError_t s_ = (c);                                                  \
    if (s_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__,          \
                   cudaGetErrorString(s_));                                \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

constexpr size_t kPage = 4096;

// One 8-byte read per 4 KiB, matching the OS-path probe so the two modes are
// directly comparable.
__global__ void consume(const double *p, size_t pages, double *sink) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  double acc = 0;
  for (size_t k = i; k < pages; k += stride) acc += p[k * (kPage / 8)];
  if (acc != 0.0) atomicAdd(sink, acc);
}

struct Args {
  std::string path;
  double gib = 32.0;
  size_t slot_kib = 65536;
  int slots = 8;
  int depth = 8;
  int cpu_load = 0;      // background CPU streaming threads
  size_t load_mib = 256; // private buffer per load thread
  std::string mode = "gtier";  // gtier | mmap
};

// -------- OS demand-paging path, for the baseline column -------------------
static int run_mmap(const Args &a, size_t bytes, double *sink) {
  int fd = open(a.path.c_str(), O_RDWR | O_CREAT, 0644);
  if (fd < 0 || ftruncate(fd, bytes)) { perror("open/ftruncate"); return 1; }
  void *p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (p == MAP_FAILED) { perror("mmap"); return 1; }
  size_t pages = bytes / kPage;
  auto t0 = Clock::now();
  consume<<<1024, 256>>>((const double *)p, pages, sink);
  cudaError_t e = cudaDeviceSynchronize();
  double t = secs(t0, Clock::now());
  if (e != cudaSuccess) {
    std::printf("mode=mmap  FAILED after %.2fs: %s\n", t, cudaGetErrorString(e));
    munmap(p, bytes); close(fd); return 2;
  }
  std::printf("mode=mmap  OK  %.2fs  %.3f GiB/s\n", t, a.gib / t);
  munmap(p, bytes); close(fd);
  return 0;
}

// -------- same mmap, same access pattern, but the CPU touches it -----------
// The kernel's readahead is what makes sequential mmap tolerable for a CPU.
// Running the identical walk from CPU threads isolates whether the problem is
// mmap and 4 KiB pages in general, or specifically the GPU fault path.
static int run_mmap_cpu(const Args &a, size_t bytes, double *) {
  int fd = open(a.path.c_str(), O_RDWR);
  if (fd < 0) { perror("open"); return 1; }
  void *p = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
  if (p == MAP_FAILED) { perror("mmap"); return 1; }
  size_t pages = bytes / kPage;
  int nt = a.cpu_load > 0 ? a.cpu_load : 8;
  std::atomic<uint64_t> sink{0};
  auto t0 = Clock::now();
  std::vector<std::thread> ts;
  for (int t = 0; t < nt; ++t)
    ts.emplace_back([&, t] {
      const double *d = (const double *)p;
      uint64_t acc = 0;
      for (size_t k = t; k < pages; k += nt) acc += (uint64_t)d[k * (kPage / 8)];
      sink += acc;
    });
  for (auto &t : ts) t.join();
  double tsec = secs(t0, Clock::now());
  std::printf("mode=mmap-cpu threads=%d  OK  %.2fs  %.3f GiB/s\n",
              nt, tsec, a.gib / tsec);
  munmap(p, bytes); close(fd);
  return 0;
}

// -------- gTier: pinned window + io_uring, GPU never faults ----------------
static int run_gtier(const Args &a, size_t bytes, double *sink) {
  const size_t slot_bytes = a.slot_kib << 10;
  const size_t n_chunks = (bytes + slot_bytes - 1) / slot_bytes;

  int fd = open(a.path.c_str(), O_RDONLY | O_DIRECT);
  if (fd < 0) { perror("open O_DIRECT"); return 1; }

  // The window is pinned and device-mapped, so the GPU reads it at memory
  // speed and the kernel can never reclaim it out from under a translation.
  std::vector<uint8_t *> host(a.slots);
  std::vector<uint8_t *> dev(a.slots);
  for (int i = 0; i < a.slots; ++i) {
    CK(cudaHostAlloc((void **)&host[i], slot_bytes, cudaHostAllocMapped));
    CK(cudaHostGetDevicePointer((void **)&dev[i], host[i], 0));
    if ((uintptr_t)host[i] % kPage) {
      std::fprintf(stderr, "slot not page aligned; O_DIRECT will fail\n");
      return 1;
    }
  }

  io_uring ring;
  if (io_uring_queue_init(a.depth, &ring, 0) < 0) { perror("io_uring"); return 1; }

  std::vector<cudaEvent_t> ev(a.slots);
  for (auto &e : ev) CK(cudaEventCreate(&e));
  std::vector<bool> used(a.slots, false);

  auto submit = [&](int slot, size_t chunk) {
    io_uring_sqe *sqe = io_uring_get_sqe(&ring);
    size_t off = chunk * slot_bytes;
    size_t len = std::min(slot_bytes, bytes - off);
    len = (len + kPage - 1) / kPage * kPage;  // O_DIRECT length alignment
    io_uring_prep_read(sqe, fd, host[slot], len, off);
    io_uring_sqe_set_data64(sqe, slot);
    io_uring_submit(&ring);
  };

  // Controlled background CPU load.  Every offloading system spends CPU cycles
  // to save I/O on the premise that the CPU is free while the GPU computes.
  // Here the two share one memory controller, so the CPU's traffic is charged
  // against the GPU.  Each worker streams over its own private buffer, so this
  // measures pure bandwidth contention -- no shared lines, no pipeline coupling.
  std::atomic<bool> load_stop{false};
  std::atomic<uint64_t> load_bytes{0};
  std::vector<std::thread> load_threads;
  for (int t = 0; t < a.cpu_load; ++t) {
    load_threads.emplace_back([&, t] {
      const size_t n = (a.load_mib << 20) / sizeof(double);
      std::vector<double> buf(n, 1.0);
      uint64_t moved = 0;
      volatile double sink_local = 0;
      while (!load_stop.load(std::memory_order_relaxed)) {
        double acc = 0;
        for (size_t i = 0; i < n; ++i) acc += buf[i];
        sink_local = acc;
        moved += n * sizeof(double);
      }
      (void)sink_local;
      load_bytes += moved;
    });
  }

  const int prime = std::min<size_t>(a.slots, n_chunks);
  auto t0 = Clock::now();
  for (int i = 0; i < prime; ++i) { submit(i, i); used[i] = true; }

  size_t next_chunk = prime, done = 0;
  while (done < n_chunks) {
    io_uring_cqe *cqe;
    if (io_uring_wait_cqe(&ring, &cqe) < 0) { perror("wait_cqe"); return 1; }
    int slot = (int)io_uring_cqe_get_data64(cqe);
    int res = cqe->res;
    io_uring_cqe_seen(&ring, cqe);
    if (res < 0) { std::fprintf(stderr, "read: %s\n", strerror(-res)); return 1; }

    consume<<<1024, 256>>>((const double *)dev[slot], (size_t)res / kPage, sink);
    CK(cudaEventRecord(ev[slot]));
    ++done;

    if (next_chunk < n_chunks) {
      CK(cudaEventSynchronize(ev[slot]));  // slot is free again
      submit(slot, next_chunk++);
    }
  }
  CK(cudaDeviceSynchronize());
  double t = secs(t0, Clock::now());
  load_stop = true;
  for (auto &th : load_threads) th.join();
  double cpu_gibs = (load_bytes / (double)(1ull << 30)) / t;
  std::printf("mode=gtier slot=%zuKiB x%d depth=%d cpu_load=%d  OK  "
              "%.2fs  gpu=%.3f GiB/s  cpu=%.3f GiB/s  total=%.3f GiB/s\n",
              a.slot_kib, a.slots, a.depth, a.cpu_load,
              t, a.gib / t, cpu_gibs, a.gib / t + cpu_gibs);

  io_uring_queue_exit(&ring);
  for (int i = 0; i < a.slots; ++i) { cudaFreeHost(host[i]); cudaEventDestroy(ev[i]); }
  close(fd);
  return 0;
}

int main(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto nxt = [&] { return argv[++i]; };
    if (s == "--file") a.path = nxt();
    else if (s == "--gib") a.gib = atof(nxt());
    else if (s == "--slot-mib") a.slot_kib = strtoull(nxt(), nullptr, 10) << 10;
    else if (s == "--slot-kib") a.slot_kib = strtoull(nxt(), nullptr, 10);
    else if (s == "--slots") a.slots = atoi(nxt());
    else if (s == "--depth") a.depth = atoi(nxt());
    else if (s == "--cpu-load") a.cpu_load = atoi(nxt());
    else if (s == "--load-mib") a.load_mib = strtoull(nxt(), nullptr, 10);
    else if (s == "--mode") a.mode = nxt();
    else { std::fprintf(stderr, "unknown arg %s\n", s.c_str()); return 1; }
  }
  if (a.path.empty()) { std::fprintf(stderr, "--file required\n"); return 1; }
  size_t bytes = (size_t)(a.gib * (1ull << 30));
  bytes = bytes / kPage * kPage;

  double *sink = nullptr;
  CK(cudaMalloc(&sink, sizeof(double)));
  CK(cudaMemset(sink, 0, sizeof(double)));

  std::printf("file=%s  %.1f GiB  ", a.path.c_str(), a.gib);
  std::fflush(stdout);
  if (a.mode == "mmap") return run_mmap(a, bytes, sink);
  if (a.mode == "mmap-cpu") return run_mmap_cpu(a, bytes, sink);
  return run_gtier(a, bytes, sink);
}
