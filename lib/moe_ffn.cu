// The arithmetic a layer's routed experts actually do, so the driver reports
// token times rather than transfer times.
//
// The weights are not copied here.  gTier hands back a pointer the GPU can
// dereference -- that is the staging plane's claim -- so the kernels read the
// bytes that landed from the drive in place, and a resident expert is read
// from the arena in place.  Measured, the GPU reads cudaHostAlloc'd mapped
// memory at 254.0 GB/s against 254.1 for its own, so reading in place costs
// nothing (gtier/mapped_read_bw.cu).
//
// Hand-written rather than cuBLAS.  A decode token is one vector against
// 3.4 GiB of weights: every byte is read once and no byte is reused, so the
// work is bound by memory and the floor is 3.4 GiB / 254 GB/s = 14.4 ms.
// cublasGemmBatchedEx with n=1 took 59.3 ms for it, four times the floor,
// which would have drowned the I/O differences the driver exists to compare.
#include "moe_ffn.h"
#include <cuda_bf16.h>
#include <cstdio>

#define WARP 32
// Eight bf16 per lane, so a warp asks for 512 bytes at a time instead of 64.
// The rows are 2048 and 768 elements and the allocations are slot-aligned, so
// the eightfold split is exact and the loads stay aligned.
#define VEC 8
struct bf8 { __nv_bfloat16 v[VEC]; };

// out[e][i] = silu(gate[e][i].x) * (up[e][i].x), one warp per output row.
__global__ void gate_up_silu(const void *const *gate, const void *const *up,
                             const __nv_bfloat16 *x, __nv_bfloat16 *h,
                             int hidden, int inter) {
    int e = blockIdx.y;
    int row = blockIdx.x * (blockDim.x / WARP) + (threadIdx.x / WARP);
    if (row >= inter) return;
    int lane = threadIdx.x % WARP;
    const bf8 *gw = (const bf8 *)((const __nv_bfloat16 *)gate[e] + (size_t)row * hidden);
    const bf8 *uw = (const bf8 *)((const __nv_bfloat16 *)up[e]   + (size_t)row * hidden);
    const bf8 *xv8 = (const bf8 *)x;
    float ga = 0.f, ua = 0.f;
    int nv = hidden / VEC;
    for (int j = lane; j < nv; j += WARP) {
        bf8 gv = gw[j], uv = uw[j], xv = xv8[j];
#pragma unroll
        for (int q = 0; q < VEC; ++q) {
            float xq = __bfloat162float(xv.v[q]);
            ga = fmaf(__bfloat162float(gv.v[q]), xq, ga);
            ua = fmaf(__bfloat162float(uv.v[q]), xq, ua);
        }
    }
    for (int o = WARP/2; o; o >>= 1) {
        ga += __shfl_down_sync(0xffffffff, ga, o);
        ua += __shfl_down_sync(0xffffffff, ua, o);
    }
    if (!lane) h[(size_t)e*inter + row] = __float2bfloat16(ga / (1.f + __expf(-ga)) * ua);
}

// y[i] = sum over experts of down[e][i] . h[e], one warp per output row.
__global__ void down_accum(const void *const *down, const __nv_bfloat16 *h,
                           __nv_bfloat16 *y, int hidden, int inter, int n_exp) {
    int row = blockIdx.x * (blockDim.x / WARP) + (threadIdx.x / WARP);
    if (row >= hidden) return;
    int lane = threadIdx.x % WARP;
    float acc = 0.f;
    for (int e = 0; e < n_exp; ++e) {
        const bf8 *dw = (const bf8 *)((const __nv_bfloat16 *)down[e] + (size_t)row * inter);
        const bf8 *hv = (const bf8 *)(h + (size_t)e * inter);
        float a = 0.f;
        int nv = inter / VEC;
        for (int j = lane; j < nv; j += WARP) {
            bf8 dv = dw[j], hh = hv[j];
#pragma unroll
            for (int q = 0; q < VEC; ++q)
                a = fmaf(__bfloat162float(dv.v[q]), __bfloat162float(hh.v[q]), a);
        }
        for (int o = WARP/2; o; o >>= 1) a += __shfl_down_sync(0xffffffff, a, o);
        if (!lane) acc += a;
    }
    if (!lane) y[row] = __float2bfloat16(acc);
}

int moe_layer_ffn(cublasHandle_t, const moe_dims &d, int n_exp,
                  const void *const *, const void *const *, const void *const *,
                  const void *x, void *, void *, void *h_buf, void *y_accum,
                  const void **dev_ptrs, cudaStream_t stream) {
    if (n_exp <= 0) return 0;
    const void **p_gate = dev_ptrs;
    const void **p_up   = p_gate + n_exp;
    const void **p_down = p_up   + n_exp;

    const int TPB = 256, WPB = TPB / WARP;
    dim3 g1((d.inter + WPB - 1) / WPB, n_exp);
    gate_up_silu<<<g1, TPB, 0, stream>>>(p_gate, p_up, (const __nv_bfloat16 *)x,
                                         (__nv_bfloat16 *)h_buf, d.hidden, d.inter);
    int g2 = (d.hidden + WPB - 1) / WPB;
    down_accum<<<g2, TPB, 0, stream>>>(p_down, (const __nv_bfloat16 *)h_buf,
                                       (__nv_bfloat16 *)y_accum, d.hidden, d.inter, n_exp);
    return 0;
}
