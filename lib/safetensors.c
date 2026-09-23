// Minimal safetensors reader.  It fills the same table the GGUF reader does,
// so one trace driver runs over either format and the comparison against an
// engine that only speaks safetensors is over the very same bytes on disk.
//
// Layout: 8-byte little-endian header length N, then N bytes of JSON, then the
// tensor blob.  Each entry carries data_offsets relative to the start of that
// blob, so the absolute offset is 8 + N + begin.
//
// The JSON is walked rather than fully parsed: at depth 1 the keys are tensor
// names and each value is an object holding data_offsets.  That is all the
// trace needs, and it avoids pulling in a JSON library for six fields.
#define _GNU_SOURCE
#include "gguf.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

static const char *skip_ws(const char *p, const char *e) {
    while (p < e && (*p==' '||*p=='\t'||*p=='\n'||*p=='\r')) ++p;
    return p;
}

// Reads a JSON string into out (truncating), returns the char after the close
// quote, or NULL on malformed input.
static const char *rd_jstr(const char *p, const char *e, char *out, size_t cap) {
    if (p >= e || *p != '"') return NULL;
    ++p;
    size_t n = 0;
    while (p < e && *p != '"') {
        if (*p == '\\') { if (++p >= e) return NULL; }
        if (out && n + 1 < cap) out[n++] = *p;
        ++p;
    }
    if (p >= e) return NULL;
    if (out) out[n] = 0;
    return p + 1;
}

// Skips one JSON value, returning the char after it.
static const char *skip_val(const char *p, const char *e) {
    p = skip_ws(p, e);
    if (p >= e) return NULL;
    if (*p == '"') return rd_jstr(p, e, NULL, 0);
    if (*p == '{' || *p == '[') {
        char open = *p, close = open == '{' ? '}' : ']';
        int d = 0;
        while (p < e) {
            if (*p == '"') { p = rd_jstr(p, e, NULL, 0); if (!p) return NULL; continue; }
            if (*p == open) ++d;
            else if (*p == close && --d == 0) return p + 1;
            ++p;
        }
        return NULL;
    }
    while (p < e && *p!=','&&*p!='}'&&*p!=']'&&*p!=' '&&*p!='\n'&&*p!='\t'&&*p!='\r') ++p;
    return p;
}

// Pulls data_offsets out of one tensor's value object without descending into
// anything else it may carry.
static int val_offsets(const char *p, const char *e, uint64_t *beg, uint64_t *end) {
    p = skip_ws(p, e);
    if (p >= e || *p != '{') return -1;
    const char *stop = skip_val(p, e);
    if (!stop) return -1;
    ++p;
    int found = 0;
    for (;;) {
        p = skip_ws(p, e);
        if (p >= e || *p == '}') break;
        char key[64];
        p = rd_jstr(p, e, key, sizeof key);
        if (!p) return -1;
        p = skip_ws(p, e);
        if (p >= e || *p != ':') return -1;
        ++p;
        if (!strcmp(key, "data_offsets")) {
            const char *q = skip_ws(p, e);
            if (q >= e || *q != '[') return -1;
            if (sscanf(q + 1, "%llu , %llu", (unsigned long long *)beg,
                       (unsigned long long *)end) != 2) return -1;
            found = 1;
        }
        p = skip_val(p, e);
        if (!p) return -1;
        p = skip_ws(p, e);
        if (p < e && *p == ',') ++p;
    }
    return found ? 0 : -1;
}

// model.layers.<L>.mlp.experts.<E>.<proj>.weight is the shape that matters:
// the layer and, where present, which routed expert this tensor belongs to.
static void classify(gguf_tensor *t) {
    t->layer = -1; t->expert = -1; t->is_ffn = 0;
    const char *L = strstr(t->name, "layers.");
    if (L) t->layer = atoi(L + 7);
    const char *E = strstr(t->name, "experts.");
    if (E) t->expert = atoi(E + 8);
    t->is_ffn = strstr(t->name, ".mlp.") != NULL;
}

int safetensors_load(const char *path, gguf_model *m) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat st;
    if (fstat(fd, &st)) { close(fd); return -1; }
    m->file_bytes = (uint64_t)st.st_size;

    uint64_t hdr = 0;
    if (read(fd, &hdr, 8) != 8 || hdr == 0 || hdr > (uint64_t)st.st_size) {
        close(fd); return -1;
    }
    char *js = (char *)malloc(hdr + 1);
    if (!js) { close(fd); return -1; }
    if ((uint64_t)pread(fd, js, hdr, 8) != hdr) { free(js); close(fd); return -1; }
    js[hdr] = 0;
    close(fd);
    m->data_offset = 8 + hdr;

    const char *p = js, *e = js + hdr;
    p = skip_ws(p, e);
    if (p >= e || *p != '{') { free(js); return -1; }
    ++p;

    int cap = 64;
    m->t = (gguf_tensor *)calloc(cap, sizeof(gguf_tensor));
    m->n = 0; m->n_layers = 0;
    for (;;) {
        p = skip_ws(p, e);
        if (p >= e || *p == '}') break;
        char name[128];
        p = rd_jstr(p, e, name, sizeof name);
        if (!p) { free(js); gguf_free(m); return -1; }
        p = skip_ws(p, e);
        if (p >= e || *p != ':') { free(js); gguf_free(m); return -1; }
        ++p;
        uint64_t beg = 0, end = 0;
        // __metadata__ is a string map, not a tensor; it simply has no offsets
        if (strcmp(name, "__metadata__") && !val_offsets(p, e, &beg, &end)) {
            if (m->n == cap) {
                cap *= 2;
                m->t = (gguf_tensor *)realloc(m->t, cap * sizeof(gguf_tensor));
            }
            gguf_tensor *t = &m->t[m->n++];
            memset(t, 0, sizeof *t);
            snprintf(t->name, sizeof t->name, "%s", name);
            t->offset = m->data_offset + beg;
            t->size = end - beg;
            classify(t);
            if (t->layer + 1 > m->n_layers) m->n_layers = t->layer + 1;
        }
        p = skip_val(p, e);
        if (!p) { free(js); gguf_free(m); return -1; }
        p = skip_ws(p, e);
        if (p < e && *p == ',') ++p;
    }
    free(js);
    return m->n ? 0 : -1;
}
