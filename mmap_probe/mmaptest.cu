// Can a GPU kernel on Thor directly dereference mmap'd, file-backed memory,
// with the OS servicing the faults?  If yes, out-of-core GPU computing needs no
// manual chunking pipeline -- the page cache IS the tier.
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
using Clock = std::chrono::steady_clock;
static double sec(Clock::time_point a, Clock::time_point b){
  return std::chrono::duration<double>(b-a).count(); }
#define CK(c) do{cudaError_t s=(c); if(s!=cudaSuccess){ \
  printf("  CUDA error line %d: %s\n",__LINE__,cudaGetErrorString(s)); return -1;} }while(0)

__global__ void touch(const double *p, size_t n, size_t stride, double *out){
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
  size_t st = (size_t)gridDim.x*blockDim.x;
  double acc = 0;
  for(size_t k=i; k<n; k+=st) acc += p[k*stride];
  atomicAdd(out, acc);
}

int main(int argc, char** argv){
  const char* path = argc>1 ? argv[1] : "/home/thor/kcj/mmap_gpu/big.bin";
  double gib = argc>2 ? atof(argv[2]) : 8.0;
  size_t bytes = (size_t)(gib*1073741824.0);
  int fd = open(path, O_RDWR|O_CREAT, 0644);
  if(fd<0){ perror("open"); return 1; }
  if(ftruncate(fd, bytes)){ perror("ftruncate"); return 1; }
  void* p = mmap(nullptr, bytes, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  if(p==MAP_FAILED){ perror("mmap"); return 1; }
  printf("mmap %.1f GiB at %p (file-backed, MAP_SHARED)\n", gib, p);

  // seed a little so pages are allocated on disk
  memset(p, 1, 1u<<20);

  double *dout=nullptr; CK(cudaMalloc(&dout,sizeof(double)));
  CK(cudaMemset(dout,0,sizeof(double)));
  size_t n = bytes/sizeof(double);
  size_t stride = 1;
  // stride so we touch one double per 4KiB page -> pure page-fault behaviour
  size_t pages = bytes/4096;
  printf("kernel will touch %zu doubles (one per 4KiB page)\n", pages);

  auto t0=Clock::now();
  touch<<<1024,256>>>((const double*)p, pages, 512 /*doubles per 4KiB page*/, dout);
  cudaError_t err = cudaDeviceSynchronize();
  double t = sec(t0,Clock::now());
  if(err!=cudaSuccess){
    printf("  RESULT: GPU CANNOT fault on file-backed mmap -> %s\n", cudaGetErrorString(err));
    munmap(p,bytes); close(fd); return 2;
  }
  double h=0; cudaMemcpy(&h,dout,sizeof(double),cudaMemcpyDeviceToHost);
  printf("  RESULT: GPU READ FILE-BACKED mmap OK  checksum=%.1f\n", h);
  printf("  %.2f s for %.1f GiB touched -> %.2f GiB/s effective\n", t, gib, gib/t);
  munmap(p,bytes); close(fd);
  return 0;
}
