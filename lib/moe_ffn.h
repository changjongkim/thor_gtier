#ifndef MOE_FFN_H
#define MOE_FFN_H
#include <cublas_v2.h>
#include <cuda_runtime.h>

// One layer's shapes; Qwen3-30B-A3B is hidden 2048, inter 768.
struct moe_dims { int hidden; int inter; };

// Runs one routed expert's SwiGLU feed-forward straight out of whatever
// memory the weights already live in, accumulating into y_accum.
int moe_expert_ffn(cublasHandle_t h, const moe_dims &d,
                   const void *gate_w, const void *up_w, const void *down_w,
                   const void *x, void *g_buf, void *u_buf, void *h_buf,
                   void *y_accum, cudaStream_t stream);
#endif
