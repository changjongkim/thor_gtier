#!/usr/bin/env python3
"""results/PREP/FINEMOE.md from stage 8's logs (format of MOE_INFINITY.md)."""
import glob, json, os, re
P = "/home/thor/kcj/thor_gtier/results/PREP"; O = "/home/thor/kcj/thor_gtier/results/MATRIX5/qwen30b"
F = "/home/thor/kcj/thor_gtier/results/FINEMOE"
def rd(p): return open(p).read() if os.path.exists(p) else ""
smoke = rd(f"{P}/finemoe/smoke.log")
L, E, H, I = 48, 128, 2048, 768
pinned = L * E * 3 * H * I * 2 / 2**30
out = ["# FineMoE (EuroSys'26) on the unified-memory Thor", "",
       "- Code: its release (github.com/IntelliSys-Lab/FineMoE-EuroSys26, commit 80717e9) with the Qwen3-MoE port",
       "  (`third_party/finemoe_qwen3_sm110.patch`: attention graphs without the Qwen3.5 output gate or linear",
       "  attention, no shared expert, model-type checks); run through `scripts/finemoe_serve.py`.",
       "- Design: the whole checkpoint is loaded to CPU (`from_pretrained(device_map=\"cpu\")`), then every expert",
       f"  is copied into one pinned host buffer (`model_offload.py:72`, `torch.empty(..., pin_memory=True)`):",
       f"  {pinned:.1f} GiB for Qwen3-30B-A3B, before the GPU expert cache (`cache_size` slots). Its README asks",
       "  for 192 GB of host memory. On this SoC host and GPU memory are one 122.8 GiB pool.", ""]
if "RESULT" not in smoke:
    g = re.findall(r"HOSTGUARD[^\n]*", smoke)
    out += ["## Result: cannot serve on this device", "",
            f"- Load test (2 MMLU prompts, 4 GiB GPU cache, host guard at 12 GiB available): "
            f"{g[-1] if g else 'failed: ' + (re.findall(r'[A-Za-z]*Error[^\n]*', smoke) or ['see log'])[-1]}",
            "- Loading needs the CPU copy of the checkpoint and the pinned expert buffer at once, above what the",
            "  pool holds beside the OS; FineMoE is reported as cannot run within any budget here, with this",
            "  evidence, rather than modified.", "", "Log: `results/PREP/finemoe/smoke.log`."]
else:
    tok = rd(f"{P}/finemoe/tok.log")
    def j(p):
        try: return json.load(open(p))
        except Exception: return None
    th, ou = j(f"{F}/fidelity_theirs.json"), j(f"{F}/fidelity_ours.json")
    out += ["## Serves", "", f"- Smoke: `{(re.findall(r'^RESULT[^\n]*', smoke, re.M) or [''])[0]}`"]
    if th and ou:
        out += [f"- Fidelity (8 MMLU prompts, same cache and maps): FineMoE's measure() {th['request_s']:.3f} s,"
                f" our runner {ou['request_s']:.3f} s ({(ou['request_s']/th['request_s']-1)*100:+.1f}%)."]
    out += ["", "## E1 cells", ""]
    for w in ("mmlu", "sharegpt", "longbench"):
        for f in ("0.25", "0.45", "0.65", "1.08"):
            t = rd(f"{O}/{w}/finemoe_{f}.txt")
            r = re.findall(r"^(RESULT|NORUN)[^\n]*", t, re.M)
            line = re.findall(r"^(?:RESULT|NORUN)[^\n]*", t, re.M)
            out.append(f"- {w} {f}: {line[-1][:150] if line else '-'}")
    out += ["", "Logs: `results/PREP/finemoe/`, `results/FINEMOE/`, `results/MATRIX5/qwen30b/*/finemoe_*`."]
print("\n".join(out))
