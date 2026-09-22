#include "gguf.h"
#include <stdio.h>
int main(int argc, char **argv) {
    gguf_model m;
    if (gguf_load(argv[1], &m)) { fprintf(stderr, "parse failed\n"); return 1; }
    printf("tensors=%d layers=%d data_off=%llu file=%.2f GiB\n",
           m.n, m.n_layers, (unsigned long long)m.data_offset, m.file_bytes/1073741824.0);
    unsigned long long tot=0, ffn=0; int nexp=0;
    for (int i = 0; i < m.n; ++i) {
        tot += m.t[i].size;
        if (m.t[i].is_ffn) ffn += m.t[i].size;
        if (m.t[i].expert >= 0) ++nexp;
        if (i < 8 || (argc>2 && m.t[i].layer==0))
            printf("  %-40s layer=%3d off=%12llu size=%10.2f MiB %s\n", m.t[i].name,
                   m.t[i].layer, (unsigned long long)m.t[i].offset,
                   m.t[i].size/1048576.0, m.t[i].expert>=0?"[exps]":"");
    }
    printf("total tensor bytes = %.2f GiB (ffn %.2f GiB), stacked-expert tensors=%d\n",
           tot/1073741824.0, ffn/1073741824.0, nexp);
    gguf_free(&m); return 0;
}
