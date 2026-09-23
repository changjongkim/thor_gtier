// Minimal GGUF reader: enough to recover where each tensor lives in the file.
// The access pattern of inference is a sequence of byte ranges over those
// tensors, so this turns a real model into a real trace.
#ifndef GGUF_H
#define GGUF_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char     name[128];
    uint64_t offset;      // absolute byte offset in the file
    uint64_t size;        // bytes
    int      layer;       // blk.N -> N, else -1
    int      expert;      // per-expert tensor index, else -1
    int      is_ffn;      // part of the feed-forward (expert) path
} gguf_tensor;

typedef struct {
    gguf_tensor *t;
    int n;
    uint64_t data_offset;
    uint64_t file_bytes;
    int n_layers;
    // GGUF stacks a layer's experts into one tensor, so a routed read is a
    // slice of it; safetensors stores one tensor per expert, so a routed read
    // is a whole tensor.  The trace has to know which.
    int stacked_experts;
} gguf_model;

int  gguf_load(const char *path, gguf_model *m);
int  safetensors_load(const char *path, gguf_model *m);
void gguf_free(gguf_model *m);

#ifdef __cplusplus
}
#endif
#endif
