// Can CPU+GPU together move more bytes than the GPU alone on Thor?
// Decides whether coherent CPU-GPU co-execution can help a bandwidth-bound
// workload such as state-vector simulation (arithmetic intensity ~0.38 F/B).
#include <cuda_runtime.h>
#include <pthread.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b){
  return std::chrono::duration<double,std::milli>(b-a).count(); }
#define CK(c) do{cudaError_t s=(c); if(s!=cudaSuccess){printf("%d %s\n",__LINE__,cudaGetErrorString(s));exit(1);} }while(0)

__global__ void stream_triad(double2 *a, const double2 *b, size_t n, double s){
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
  size_t st = (size_t)gridDim.x*blockDim.x;
  for(; i<n; i+=st){ a[i].x += s*b[i].x; a[i].y += s*b[i].y; }
}

static void cpu_triad(double *a, const double *b, size_t n, double s, int tid, int nthreads){
  cpu_set_t set; CPU_ZERO(&set); CPU_SET(tid, &set);
  pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  size_t chunk=(n+nthreads-1)/nthreads, lo=chunk*tid, hi=std::min(n,lo+chunk);
  for(size_t i=lo;i<hi;++i) a[i]+=s*b[i];
}

int main(int argc,char**argv){
  size_t bytes = 4ull<<30;                   // 4 GiB per array
  int cpu_threads = 12;
  if(argc>1) cpu_threads=atoi(argv[1]);
  size_t n2 = bytes/sizeof(double2), nd = bytes/sizeof(double);
  double2 *ga_h,*ga_d,*gb_h,*gb_d; double *ca,*cb;
  CK(cudaHostAlloc(&ga_h,bytes,cudaHostAllocMapped)); CK(cudaHostGetDevicePointer(&ga_d,ga_h,0));
  CK(cudaHostAlloc(&gb_h,bytes,cudaHostAllocMapped)); CK(cudaHostGetDevicePointer(&gb_d,gb_h,0));
  CK(cudaHostAlloc(&ca,bytes,cudaHostAllocMapped));
  CK(cudaHostAlloc(&cb,bytes,cudaHostAllocMapped));
  memset(ga_h,1,bytes); memset(gb_h,1,bytes); memset(ca,1,bytes); memset(cb,1,bytes);
  const double gpu_bytes = 3.0*bytes, cpu_bytes = 3.0*bytes;  // read a, read b, write a

  auto gpu_run=[&](){ stream_triad<<<4096,256>>>(ga_d,gb_d,n2,1.5); CK(cudaDeviceSynchronize()); };
  auto cpu_run=[&](){ std::vector<std::thread> t;
    for(int i=0;i<cpu_threads;++i) t.emplace_back(cpu_triad,ca,cb,nd,1.5,i,cpu_threads);
    for(auto&x:t) x.join(); };

  gpu_run(); cpu_run();                       // warm

  printf("array=%.1f GiB  cpu_threads=%d\n", bytes/1073741824.0, cpu_threads);
  double best_g=0,best_c=0,best_b=0;
  for(int r=0;r<5;++r){
    auto t0=Clock::now(); gpu_run(); double tg=ms(t0,Clock::now());
    best_g=std::max(best_g, gpu_bytes/1e6/tg);
    t0=Clock::now(); cpu_run(); double tc=ms(t0,Clock::now());
    best_c=std::max(best_c, cpu_bytes/1e6/tc);
    // concurrent
    t0=Clock::now();
    std::thread cth(cpu_run);
    gpu_run(); cth.join();
    double tb=ms(t0,Clock::now());
    best_b=std::max(best_b,(gpu_bytes+cpu_bytes)/1e6/tb);
  }
  printf("GPU alone      : %7.1f GB/s\n", best_g);
  printf("CPU alone (%2d) : %7.1f GB/s\n", cpu_threads, best_c);
  printf("CPU+GPU concur : %7.1f GB/s\n", best_b);
  printf("-> combined / GPU-alone = %.3fx   (>1.10 means co-execution can add bandwidth)\n", best_b/best_g);
  return 0;
}
