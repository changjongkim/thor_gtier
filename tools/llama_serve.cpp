// Serve a workload's prompts through llama.cpp and time each request, so the
// engine a deployment would actually run is measured on the same requests as
// the serving driver: same prompts, same truncation, same number of new
// tokens, greedy decoding.
//
// Configuration is llama.cpp's standard one for MoE models that do not fit:
// every non-expert tensor on the GPU, the routed experts (ffn_*_exps) left in
// the mmap'd file on the CPU side (what --cpu-moe does).  Run it under the
// same memory cap as everything else (scripts/in_cgroup.sh) and the page
// cache holding those experts is bounded by the budget.
//
//   llama_serve <model.gguf> <workload.tsv> <n_threads> [max_prompt]
//
// Prints one line per request and a RESULT line in the driver's format.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static double now_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
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
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.gguf workload.tsv n_threads [max_prompt]\n", argv[0]);
        return 2;
    }
    int n_threads = atoi(argv[3]);
    int max_prompt = argc > 4 ? atoi(argv[4]) : 8192;

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;          // every layer's dense part on the GPU ...
    mp.use_mmap = true;
    // ... and the routed experts stay in the mapped file (--cpu-moe).
    static llama_model_tensor_buft_override ov[2];
    ov[0].pattern = "\\.ffn_(up|down|gate)_exps";
    ov[0].buft = ggml_backend_cpu_buffer_type();
    ov[1].pattern = nullptr; ov[1].buft = nullptr;
    mp.tensor_buft_overrides = ov;
    llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab *vocab = llama_model_get_vocab(model);

    auto cp = llama_context_default_params();
    cp.n_ctx = max_prompt + 256;
    cp.n_batch = cp.n_ubatch = 2048;
    cp.n_threads = cp.n_threads_batch = n_threads;
    llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }

    std::ifstream in(argv[2]);
    std::string line;
    double s_ttft = 0, s_tpot = 0, s_req = 0, s_ptok = 0, s_dtok = 0;
    int n = 0;
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
        if (nt < 0) {
            toks.resize(-nt);
            nt = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(),
                                toks.data(), (int) toks.size(), true, false);
        }
        if (nt > max_prompt) nt = max_prompt;
        toks.resize(nt);
        llama_memory_clear(llama_get_memory(ctx), true);

        double t0 = now_s();
        // Prompt in n_batch chunks, as the server does.
        bool ok = true;
        for (int p = 0; p < nt; p += cp.n_batch) {
            int m = std::min((int) cp.n_batch, nt - p);
            llama_batch b = llama_batch_get_one(toks.data() + p, m);
            if (llama_decode(ctx, b)) { ok = false; break; }
        }
        if (!ok) { fprintf(stderr, "prefill failed: %s\n", name.c_str()); continue; }
        int nv = llama_vocab_n_tokens(vocab);
        auto argmax = [&]() {
            const float *lg = llama_get_logits_ith(ctx, -1);
            llama_token best = 0; float bv = lg[0];
            for (int v = 1; v < nv; ++v) if (lg[v] > bv) { bv = lg[v]; best = v; }
            return best;
        };
        llama_token tok = argmax();
        double t_first = now_s();                  // first token is known here
        int produced = 1;
        for (int s = 1; s < max_new; ++s) {
            llama_batch nb = llama_batch_get_one(&tok, 1);
            if (llama_decode(ctx, nb)) break;
            tok = argmax();
            ++produced;
        }
        double t_end = now_s();
        double ttft = t_first - t0;
        double tpot = produced > 1 ? (t_end - t_first) / (produced - 1) : 0.0;
        printf("req %s prompt_tok=%d new_tok=%d ttft_s=%.4f tpot_ms=%.3f request_s=%.4f\n",
               name.c_str(), nt, produced, ttft, tpot * 1e3, t_end - t0);
        fflush(stdout);
        s_ttft += ttft; s_tpot += tpot * produced; s_req += t_end - t0;
        s_ptok += nt; s_dtok += produced; ++n;
    }
    if (n)
        printf("RESULT policy=llama.cpp backend=mmap batch=1 overlap=1 requests=%d "
               "prompt_tok=%.1f decode_tok=%.1f ttft_s=%.4f tpot_ms=%.3f request_s=%.4f "
               "throughput_tok_s=%.4f compute=measured\n",
               n, s_ptok / n, s_dtok / n, s_ttft / n, s_tpot / s_dtok * 1e3, s_req / n,
               s_dtok / s_req);
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
