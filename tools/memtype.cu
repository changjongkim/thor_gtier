// GPU read bandwidth of the same kernel over buffers from cudaMalloc,
// cudaHostAlloc(Mapped) (PHASOR's arena/window), cudaHostAlloc default, and
// cudaMallocManaged.  384 buffers of 3 MiB = one Qwen3 decode step's experts.
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
__global__ void rd(const uint4 *const *p, size_t n16, unsigned long long *sink) {
  const uint4 *b = p[blockIdx.y]; unsigned long long acc = 0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v = b[i]; acc += v.x ^ v.y ^ v.z ^ v.w; }
  if (acc == 42) atomicAdd(sink, acc);
}
int main() {
  const int NB = 384; const size_t SZ = 3u << 20;
  const char *names[4] = {"cudaMalloc", "cudaHostAlloc(Mapped)", "cudaHostAlloc(default)", "cudaMallocManaged"};
  unsigned long long *sink; cudaMalloc(&sink, 8);
  for (int kind = 0; kind < 4; ++kind) {
    std::vector<void *> h(NB); std::vector<const uint4 *> d(NB);
    for (int i = 0; i < NB; ++i) {
      if (kind == 0) cudaMalloc(&h[i], SZ);
      else if (kind == 1) { cudaHostAlloc(&h[i], SZ, cudaHostAllocMapped); }
      else if (kind == 2) cudaHostAlloc(&h[i], SZ, cudaHostAllocDefault);
      else cudaMallocManaged(&h[i], SZ);
      void *dp = h[i]; if (kind == 1) cudaHostGetDevicePointer(&dp, h[i], 0);
      cudaMemset(dp, 1, SZ); d[i] = (const uint4 *)dp;
    }
    const uint4 **dd; cudaMalloc(&dd, NB * sizeof(void *)); cudaMemcpy(dd, d.data(), NB * sizeof(void *), cudaMemcpyHostToDevice);
    dim3 g(16, NB); cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    rd<<<g, 256>>>(dd, SZ / 16, sink); cudaDeviceSynchronize();
    cudaEventRecord(a); for (int r = 0; r < 20; ++r) rd<<<g, 256>>>(dd, SZ / 16, sink); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    printf("%-24s %7.1f GiB/s  (%.2f ms per 1.125 GiB step)\n", names[kind], 20.0 * NB * SZ / (1 << 30) / (ms / 1e3), ms / 20);
    for (int i = 0; i < NB; ++i) { if (kind == 0 || kind == 3) cudaFree(h[i]); else cudaFreeHost(h[i]); }
    cudaFree(dd);
  }
  printf("err: %s\n", cudaGetErrorString(cudaGetLastError()));
}
