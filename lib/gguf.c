#include "gguf.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { GGUF_MAGIC = 0x46554747 };

// GGUF metadata value types
enum { T_U8=0,T_I8,T_U16,T_I16,T_U32,T_I32,T_F32,T_BOOL,T_STR,T_ARR,T_U64,T_I64,T_F64 };

static int rd(int fd, void *p, size_t n) { return read(fd, p, n) == (ssize_t)n ? 0 : -1; }
static int rd_u32(int fd, uint32_t *v) { return rd(fd, v, 4); }
static int rd_u64(int fd, uint64_t *v) { return rd(fd, v, 8); }
static int rd_str(int fd, char *buf, size_t cap) {
    uint64_t n; if (rd_u64(fd, &n)) return -1;
    char tmp[1024]; size_t take = n < sizeof(tmp) ? n : sizeof(tmp);
    if (rd(fd, tmp, take)) return -1;
    if (n > take && lseek(fd, n - take, SEEK_CUR) < 0) return -1;
    size_t c = take < cap - 1 ? take : cap - 1;
    memcpy(buf, tmp, c); buf[c] = 0;
    return 0;
}

static int skip_value(int fd, uint32_t type);
static int skip_array(int fd) {
    uint32_t et; uint64_t n;
    if (rd_u32(fd, &et) || rd_u64(fd, &n)) return -1;
    for (uint64_t i = 0; i < n; ++i) if (skip_value(fd, et)) return -1;
    return 0;
}
static int skip_value(int fd, uint32_t type) {
    static const int sz[] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
    if (type == T_STR) { char b[8]; return rd_str(fd, b, sizeof b); }
    if (type == T_ARR) return skip_array(fd);
    if (type > T_F64) return -1;
    return lseek(fd, sz[type], SEEK_CUR) < 0 ? -1 : 0;
}

// ggml type -> (block bytes, elements per block)
static void type_block(uint32_t t, uint64_t *bytes, uint64_t *els) {
    switch (t) {
        case 0:  *bytes=4;   *els=1;   break; // F32
        case 1:  *bytes=2;   *els=1;   break; // F16
        case 2:  *bytes=18;  *els=32;  break; // Q4_0
        case 3:  *bytes=20;  *els=32;  break; // Q4_1
        case 6:  *bytes=22;  *els=32;  break; // Q5_0
        case 7:  *bytes=24;  *els=32;  break; // Q5_1
        case 8:  *bytes=34;  *els=32;  break; // Q8_0
        case 9:  *bytes=36;  *els=32;  break; // Q8_1
        case 10: *bytes=84;  *els=256; break; // Q2_K
        case 11: *bytes=110; *els=256; break; // Q3_K
        case 12: *bytes=144; *els=256; break; // Q4_K
        case 13: *bytes=176; *els=256; break; // Q5_K
        case 14: *bytes=210; *els=256; break; // Q6_K
        case 15: *bytes=292; *els=256; break; // Q8_K
        case 30: *bytes=2;   *els=1;   break; // BF16
        default: *bytes=2;   *els=1;   break;
    }
}

int gguf_load(const char *path, gguf_model *m) {
    memset(m, 0, sizeof *m);
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    m->file_bytes = lseek(fd, 0, SEEK_END); lseek(fd, 0, SEEK_SET);

    uint32_t magic, version; uint64_t n_tensors, n_kv;
    if (rd_u32(fd,&magic) || magic != GGUF_MAGIC) { close(fd); return -1; }
    if (rd_u32(fd,&version) || rd_u64(fd,&n_tensors) || rd_u64(fd,&n_kv)) { close(fd); return -1; }

    uint32_t alignment = 32;
    for (uint64_t i = 0; i < n_kv; ++i) {
        char key[256], sval[256];
        uint32_t type;
        if (rd_str(fd, key, sizeof key) || rd_u32(fd, &type)) { close(fd); return -1; }
        if (!strcmp(key, "general.alignment") && type == T_U32) {
            if (rd_u32(fd, &alignment)) { close(fd); return -1; }
        } else if (type == T_STR) {
            if (rd_str(fd, sval, sizeof sval)) { close(fd); return -1; }
        } else if (skip_value(fd, type)) { close(fd); return -1; }
    }

    m->t = (gguf_tensor *)calloc(n_tensors, sizeof(gguf_tensor));
    m->n = (int)n_tensors;
    for (uint64_t i = 0; i < n_tensors; ++i) {
        gguf_tensor *t = &m->t[i];
        uint32_t ndim, type; uint64_t dims[4] = {1,1,1,1};
        if (rd_str(fd, t->name, sizeof t->name) || rd_u32(fd, &ndim)) { close(fd); return -1; }
        for (uint32_t d = 0; d < ndim && d < 4; ++d)
            if (rd_u64(fd, &dims[d])) { close(fd); return -1; }
        if (rd_u32(fd, &type) || rd_u64(fd, &t->offset)) { close(fd); return -1; }
        uint64_t bb, be; type_block(type, &bb, &be);
        uint64_t els = dims[0]*dims[1]*dims[2]*dims[3];
        t->size = els / be * bb;
        t->layer = -1; t->expert = -1;
        if (!strncmp(t->name, "blk.", 4)) t->layer = atoi(t->name + 4);
        t->is_ffn = strstr(t->name, "ffn_") != NULL;
        const char *e = strstr(t->name, "_exps");
        if (e) t->expert = 0;   // stacked expert tensor; sliced at trace time
        if (t->layer + 1 > m->n_layers) m->n_layers = t->layer + 1;
    }
    uint64_t pos = lseek(fd, 0, SEEK_CUR);
    m->data_offset = (pos + alignment - 1) / alignment * alignment;
    for (int i = 0; i < m->n; ++i) m->t[i].offset += m->data_offset;
    close(fd);
    return 0;
}

void gguf_free(gguf_model *m) { free(m->t); m->t = NULL; m->n = 0; }
