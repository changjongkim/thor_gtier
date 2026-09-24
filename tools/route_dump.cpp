// Record the experts a GGUF MoE model routes to, from inside llama.cpp.
//
// Used where the HF checkpoint cannot be loaded: Mixtral-8x7B's 87 GiB of
// bf16 is OOM-killed during load on this 122 GiB machine, while its Q4_K_M
// GGUF fits with room to spare.  It is also the model a deployment would
// route with, which makes it the more relevant capture, not the lesser one.
//
// llama.cpp names each layer's selected experts "ffn_moe_topk-<layer>" in
// the graph (src/llama-graph.cpp), an int32 tensor of [n_expert_used,
// n_tokens].  The eval callback asks for exactly those and copies them out.
//
//   route_dump <model.gguf> <workload.json-as-lines> <out.bin>
//
// The workload is given as a pre-flattened text file -- one request per
// record, "name\tmax_new\tprompt-with-\\n-escaped" -- so this does not need
// a JSON parser.  Output records are
//   u32 tag | u8 phase(0 prefill,1 decode) | u16 layer | u32 pos | i32 experts[k]
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct Rec { uint32_t tag; uint8_t phase; uint16_t layer; uint32_t pos; std::vector<int32_t> ex; };
struct State {
    uint32_t tag = 0; uint8_t phase = 0; uint32_t base = 0; int k = 0;
    std::vector<Rec> recs;
    std::vector<uint8_t> buf;
};

static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto *st = (State *) ud;
    const char *nm = ggml_get_name(t);
    if (strncmp(nm, "ffn_moe_topk-", 13) != 0) return !ask ? true : false;
    if (ask) return true;
    int layer = atoi(nm + 13);
    int ne0 = (int) t->ne[0], ne1 = (int) t->ne[1];
    st->k = ne0;
    size_t n = ggml_nbytes(t);
    st->buf.resize(n);
    ggml_backend_tensor_get(t, st->buf.data(), 0, n);
    const int32_t *d = (const int32_t *) st->buf.data();
    for (int tok = 0; tok < ne1; ++tok) {
        Rec r{st->tag, st->phase, (uint16_t) layer, st->base + (uint32_t) tok, {}};
        r.ex.assign(d + (size_t) tok * ne0, d + (size_t) tok * ne0 + ne0);
        st->recs.push_back(std::move(r));
    }
    return true;
}

static std::string unescape(const std::string &s) {
    std::string o; o.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            char c = s[++i];
            o += (c == 'n') ? '\n' : (c == 't') ? '\t' : c;
        } else o += s[i];
    }
    return o;
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s model.gguf workload.tsv out.bin n_gpu_layers [max_prompt]\n", argv[0]); return 2; }
    // Never defaulted.  All layers on the GPU is the configuration that
    // restarted the host five times on the 132 GiB model; the caller has to
    // say how many, and the pipeline runs that number past memguard first.
    int n_gpu_layers = atoi(argv[4]);
    int max_prompt = argc > 5 ? atoi(argv[5]) : 8192;

    llama_backend_init();
    State st;
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab *vocab = llama_model_get_vocab(model);

    auto cp = llama_context_default_params();
    cp.n_ctx = max_prompt + 256;
    // One ubatch per prompt, so a token's position in the batch is its
    // position in the prompt and the records need no reassembly.
    cp.n_batch = cp.n_ubatch = max_prompt;
    cp.cb_eval = cb; cp.cb_eval_user_data = &st;
    llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }

    std::ifstream in(argv[2]);
    std::string line; std::vector<std::string> names;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string name, mn, prompt;
        std::getline(ls, name, '\t'); std::getline(ls, mn, '\t'); std::getline(ls, prompt);
        if (name.empty()) continue;
        prompt = unescape(prompt);
        int max_new = atoi(mn.c_str());

        std::vector<llama_token> toks(prompt.size() + 16);
        int nt = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(),
                                toks.data(), (int) toks.size(), true, false);
        if (nt < 0) { toks.resize(-nt); nt = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), toks.data(), (int) toks.size(), true, false); }
        if (nt > max_prompt) nt = max_prompt;
        toks.resize(nt);

        llama_memory_clear(llama_get_memory(ctx), true);
        st.tag = (uint32_t) names.size(); names.push_back(name);

        st.phase = 0; st.base = 0;
        llama_batch b = llama_batch_get_one(toks.data(), nt);
        if (llama_decode(ctx, b)) { fprintf(stderr, "prefill failed: %s\n", name.c_str()); continue; }

        st.phase = 1;
        for (int s = 0; s < max_new; ++s) {
            const float *lg = llama_get_logits_ith(ctx, -1);
            int nv = llama_vocab_n_tokens(vocab);
            llama_token best = 0; float bv = lg[0];
            for (int v = 1; v < nv; ++v) if (lg[v] > bv) { bv = lg[v]; best = v; }
            st.base = (uint32_t) s;
            llama_batch nb = llama_batch_get_one(&best, 1);
            if (llama_decode(ctx, nb)) break;
        }
        fprintf(stderr, "  %s: %d prompt tok, +%d\n", name.c_str(), nt, max_new);
    }

    FILE *f = fopen(argv[3], "wb");
    uint32_t nn = (uint32_t) names.size(), k = (uint32_t) st.k, nr = (uint32_t) st.recs.size();
    fwrite("RDMP", 1, 4, f); fwrite(&k, 4, 1, f); fwrite(&nn, 4, 1, f);
    for (auto &n : names) { uint32_t l = (uint32_t) n.size(); fwrite(&l, 4, 1, f); fwrite(n.data(), 1, l, f); }
    fwrite(&nr, 4, 1, f);
    for (auto &r : st.recs) {
        fwrite(&r.tag, 4, 1, f); fwrite(&r.phase, 1, 1, f);
        fwrite(&r.layer, 2, 1, f); fwrite(&r.pos, 4, 1, f);
        fwrite(r.ex.data(), 4, r.ex.size(), f);
    }
    fclose(f);
    fprintf(stderr, "wrote %s: %u records, k=%u, %u prompts\n", argv[3], nr, k, nn);
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
