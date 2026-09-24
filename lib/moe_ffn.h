#ifndef MOE_FFN_H
#define MOE_FFN_H
#include <cublas_v2.h>
#include <cuda_runtime.h>

// One layer's shapes; Qwen3-30B-A3B is hidden 2048, intermediate 768.
struct moe_dims { int hidden; int inter; };

// Runs a layer's routed experts' SwiGLU feed-forward, reading the weights
// wherever they already are.  dev_ptrs is device memory holding the pointer
// arrays laid out as gate[n] up[n] down[n] ... ; only the first three are
// read.  The cublasHandle_t is unused and kept so callers need not change.
int moe_layer_ffn(cublasHandle_t h, const moe_dims &d, int n_exp,
                  const void *const *gate_w, const void *const *up_w,
                  const void *const *down_w, const void *x,
                  void *g_buf, void *u_buf, void *h_buf, void *y_accum,
                  const void **dev_ptrs, cudaStream_t stream);
#endif
