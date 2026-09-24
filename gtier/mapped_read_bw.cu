// Does the GPU read mapped host memory as fast as it reads its own?
//
// The staging plane hands the arithmetic a pointer into cudaHostAlloc'd
// memory rather than copying into cudaMalloc'd memory, which saves the copy.
// If the GPU then reads that memory more slowly, the saving is paid back on
// every pass over the weights -- and a MoE decode passes over 3.4 GiB per
// token.  That is worth knowing before claiming the copy was free.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void stream_read(const float4 *p, size_t n4, float4 *sink) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float4 acc = make_float4(0,0,0,0);
    for (; i < n4; i += stride) {
        float4 v = p[i];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    if (acc.x == 1234.5f) *sink = acc;       // never true; keeps the loads
}

static double bw(const float4 *p, size_t bytes, float4 *sink, int iters) {
    size_t n4 = bytes / sizeof(float4);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    stream_read<<<1024, 256>>>(p, n4, sink); cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) stream_read<<<1024, 256>>>(p, n4, sink);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)bytes * iters / (ms / 1e3) / 1e9;
}

int main(int argc, char **argv) {
    size_t mb = argc > 1 ? atoll(argv[1]) : 512;
    int iters = argc > 2 ? atoi(argv[2]) : 20;
    size_t bytes = mb << 20;
    float4 *sink; cudaMalloc(&sink, sizeof(float4));

    void *dev = nullptr, *host = nullptr, *mgd = nullptr;
    if (cudaMalloc(&dev, bytes) != cudaSuccess) { printf("cudaMalloc failed\n"); return 1; }
    cudaMemset(dev, 1, bytes);
    if (cudaHostAlloc(&host, bytes, cudaHostAllocMapped) != cudaSuccess) {
        printf("cudaHostAlloc failed\n"); return 1;
    }
    memset(host, 1, bytes);
    void *host_dev = nullptr; cudaHostGetDevicePointer(&host_dev, host, 0);
    if (cudaMallocManaged(&mgd, bytes) == cudaSuccess) { memset(mgd, 1, bytes); }

    printf("GPU read bandwidth over %zu MiB, %d passes\n", mb, iters);
    printf("  cudaMalloc (device)          %7.1f GB/s\n", bw((float4*)dev, bytes, sink, iters));
    printf("  cudaHostAlloc(Mapped)        %7.1f GB/s\n", bw((float4*)host_dev, bytes, sink, iters));
    if (mgd) printf("  cudaMallocManaged            %7.1f GB/s\n", bw((float4*)mgd, bytes, sink, iters));
    return 0;
}
