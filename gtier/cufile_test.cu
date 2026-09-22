// Does NVIDIA's own GPUDirect Storage path work on Thor, or fall back to POSIX
// bounce buffers?  This decides whether the vendor already solves NVMe->GPU on
// coherent SoCs, or whether the gap is real.
#include <cufile.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstring>
using Clock = std::chrono::steady_clock;

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "/home/thor/kcj/mmap_gpu/real32.bin";
  size_t chunk = argc > 2 ? strtoull(argv[2], 0, 10) << 20 : 16ull << 20;
  size_t total = argc > 3 ? strtoull(argv[3], 0, 10) << 30 : (4ull << 30);

  CUfileError_t st = cuFileDriverOpen();
  if (st.err != CU_FILE_SUCCESS) {
    std::printf("cuFileDriverOpen failed: %d  (no nvidia-fs driver?)\n", st.err);
  } else {
    CUfileDrvProps_t p{};
    if (cuFileDriverGetProperties(&p).err == CU_FILE_SUCCESS) {
      std::printf("cuFile driver: nvfs major=%u minor=%u\n",
                  p.nvfs.major_version, p.nvfs.minor_version);
      std::printf("  dstatusflags=0x%x  dcontrolflags=0x%x\n",
                  p.nvfs.dstatusflags, p.nvfs.dcontrolflags);
      std::printf("  poll_thresh=%zu  max_direct_io=%zu  device_cache=%u KiB\n",
                  p.nvfs.poll_thresh_size, p.nvfs.max_direct_io_size,
                  p.max_device_cache_size);
      std::printf("  GDS supported (dstatusflags bit0=%d)\n", p.nvfs.dstatusflags & 1);
    }
  }

  int fd = open(path, O_RDONLY | O_DIRECT);
  if (fd < 0) { perror("open"); return 1; }
  CUfileDescr_t d{}; CUfileHandle_t h;
  d.handle.fd = fd; d.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  if (cuFileHandleRegister(&h, &d).err != CU_FILE_SUCCESS) {
    std::printf("cuFileHandleRegister failed\n"); return 1;
  }
  void *dbuf = nullptr;
  cudaMalloc(&dbuf, chunk);
  if (cuFileBufRegister(dbuf, chunk, 0).err != CU_FILE_SUCCESS)
    std::printf("cuFileBufRegister failed (compat mode likely)\n");

  auto t0 = Clock::now();
  size_t done = 0;
  while (done < total) {
    ssize_t r = cuFileRead(h, dbuf, chunk, (off_t)done, 0);
    if (r <= 0) { std::printf("cuFileRead -> %zd\n", r); break; }
    done += r;
  }
  double t = std::chrono::duration<double>(Clock::now() - t0).count();
  std::printf("cuFileRead: %.2f GiB in %.2fs -> %.3f GiB/s (chunk %zu MiB)\n",
              done / 1073741824.0, t, done / 1073741824.0 / t, chunk >> 20);
  cuFileBufDeregister(dbuf); cuFileHandleDeregister(h); cuFileDriverClose();
  close(fd); return 0;
}
