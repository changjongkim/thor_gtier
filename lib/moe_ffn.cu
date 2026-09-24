// The arithmetic a routed expert actually does, so the driver's timings are
// token times rather than transfer times.
//
// The weights are not copied here.  gTier hands back a pointer that the GPU
// can dereference -- that is the whole claim of the staging plane -- so cuBLAS
// is pointed straight at the bytes that landed from the drive, and a resident
// expert is read from the arena in place.  If the claim were false this would
// not run.
//
// Qwen3-MoE shapes, from the checkpoint: gate and up are [I, H] and down is
// [H, I], with H = hidden and I = moe_intermediate.  Row-major there is
// column-major transposed here, which is why the ops below are what they are.
#include "moe_ffn.h"
#include <cuda_bf16.h>
#include <cstdio>

// h = silu(g) * u, elementwise, in bf16.
__global__ void silu_mul(const __nv_bfloat16 *g, const __nv_bfloat16 *u,
                         __nv_bfloat16 *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = __bfloat162float(g[i]);
    float uv = __bfloat162float(u[i]);
    out[i] = __float2bfloat16(gv / (1.0f + __expf(-gv)) * uv);
}

// y += down * silu(gate * x) * (up * x)
int moe_expert_ffn(cublasHandle_t h, const moe_dims &d,
                   const void *gate_w, const void *up_w, const void *down_w,
                   const void *x, void *g_buf, void *u_buf, void *h_buf,
                   void *y_accum, cudaStream_t stream) {
    const float one = 1.0f, zero = 0.0f;
    cublasSetStream(h, stream);
    // gate and up: [I,H] row-major times x[H] -> [I].  Column-major sees an
    // [H,I] matrix, so no transpose and m=H is the fast dimension.
    if (cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, d.inter, 1, d.hidden,
                     &one, gate_w, CUDA_R_16BF, d.hidden, x, CUDA_R_16BF, d.hidden,
                     &zero, g_buf, CUDA_R_16BF, d.inter,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
        return -1;
    if (cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, d.inter, 1, d.hidden,
                     &one, up_w, CUDA_R_16BF, d.hidden, x, CUDA_R_16BF, d.hidden,
                     &zero, u_buf, CUDA_R_16BF, d.inter,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
        return -1;
    int threads = 256, blocks = (d.inter + threads - 1) / threads;
    silu_mul<<<blocks, threads, 0, stream>>>((const __nv_bfloat16 *)g_buf,
                                             (const __nv_bfloat16 *)u_buf,
                                             (__nv_bfloat16 *)h_buf, d.inter);
    // down: [H,I] row-major times h[I] -> [H], accumulated across experts.
    if (cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, d.hidden, 1, d.inter,
                     &one, down_w, CUDA_R_16BF, d.inter, h_buf, CUDA_R_16BF, d.inter,
                     &one, y_accum, CUDA_R_16BF, d.hidden,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
        return -1;
    return 0;
}
