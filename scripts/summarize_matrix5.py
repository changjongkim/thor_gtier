#!/usr/bin/env python3
"""Tables for the architecture-level matrix (results/MATRIX5)."""
import json, os, re
R = "/home/thor/kcj/thor_gtier/results/MATRIX5"
MODELS = [("qwen30b", "Qwen3-30B-A3B bf16", 57.0), ("mixtral8x7b", "Mixtral-8x7B bf16", 87.0)]
WL = [("mmlu", "MMLU"), ("sharegpt", "ShareGPT"), ("longbench", "LongBench")]
SYS = [("phasor", "PHASOR"), ("zipmoe", "ZipMoE"), ("moeinf", "MoE-Infinity"), ("flashmoe", "FlashMoE*"),
       ("duoserve", "DuoServe*"), ("fiddler", "Fiddler"), ("mixoff", "Mixtral-offloading (2-bit)")]
ABL = [("pfall", "prefill admitted by value (no free-slot rule)"), ("lru", "LRU (value and admission)"),
       ("lrupfree", "LRU value, PHASOR admission"), ("count", "count utility"),
       ("pread", "pread+copy data path (same memory)"), ("copy", "extra device copy (arena kept)"),
       ("nopipe", "no intra-layer pipeline"), ("noprompt", "no prompt-routing term"),
       ("admitall", "admit every staged decode unit")]
FR = ["0.25", "0.45", "0.65", "1.08"]
SMALL = ["0.20", "0.15", "0.10", "0.05"]


def load(prefix):
    """(result dict | 'norun' | None, note)"""
    t = prefix + ".txt"; j = prefix + ".json"
    txt = open(t).read() if os.path.exists(t) else ""
    if "NORUN" in txt and "RESULT" not in txt:
        return "norun", re.search(r"reason=(\S+)", txt).group(1) if "reason=" in txt else ""
    if not os.path.exists(j): return None, ""
    try: d = json.load(open(j))
    except Exception: return None, ""
    m = re.search(r"cgroup_io_read_gib=([\d.]+)", txt)
    if m: d["io_gib"] = float(m.group(1))
    return d, ""


def e1(m, w, k, f):
    """The E1 cell, falling back to a lowered-knob retry when the calibrated run hit the cap."""
    d, why = load(f"{R}/{m}/{w}/{k}_{f}")
    if d == "norun":
        for kk in ("0.85", "0.7", "0.55"):
            r, _ = load(f"{R}/{m}/{w}/{k}_{f}_k{kk}")
            if r and r != "norun": return r, f"knob x{kk}"
    return d, why


def target(m, f, gb):
    p = f"{R}/{m}/memcal/phasor_{gb * float(f):.2f}.json"
    return json.load(open(p))["peak_gib"] if os.path.exists(p) else None


def cell(d, note, ph, tg):
    if d == "norun": return "cannot run" + (f" ({note})" if note else "")
    if not d: return "-"
    s = f"{d['request_s']:.2f} / {d['ttft_s']:.2f} / {d['tpot_ms']:.0f} / {d.get('peak_gib', float('nan')):.1f}"
    if tg and d.get("peak_gib", 0) > 1.05 * tg: s += " (over)"
    if ph and ph != "norun" and d is not ph: s += f" [{d['request_s'] / ph['request_s']:.2f}x]"
    if note: s += f" ({note})"
    return s


print("# Architecture-level matrix\n")
print("Cell: request s / TTFT s / TPOT ms / peak GiB (MemAvailable drop) [request time relative to PHASOR].")
print("Same prompts for every system. Each baseline runs at the knob memcal found to match PHASOR's measured peak at")
print("that budget (two MMLU prompts); every run is capped at 1.05 x that peak + 0.5 GiB. `(over)`: the run's peak")
print("exceeded 1.05 x PHASOR's peak (memory it grew into on longer workloads). `(knob xK)`: the calibrated run hit")
print("the cap and the cell is its retry at K x the calibrated knob. `*` = reimplemented (no released code).\n")
for m, mname, gb in MODELS:
    if not os.path.isdir(f"{R}/{m}"): continue
    print(f"## {mname}\n")
    for w, wname in WL:
        print(f"### {wname}\n")
        print("| system | " + " | ".join(f"{float(f):.0%} ({gb * float(f):.1f} GiB)" for f in FR) + " |")
        print("|---" * (len(FR) + 1) + "|")
        phs = {f: e1(m, w, "phasor", f)[0] for f in FR}
        for k, name in SYS:
            row = []
            for f in FR:
                d, note = e1(m, w, k, f)
                row.append(cell(d, note, phs[f] if k != "phasor" else None, target(m, f, gb)))
            if all(c == "-" for c in row): continue
            print(f"| {name} | " + " | ".join(row) + " |")
        if w == "mmlu":
            nom = []
            for k, name in SYS[1:]:
                row = [cell(*load(f"{R}/{m}/{w}/{k}_{f}_nominal"), None, target(m, f, gb)) for f in FR]
                if any(c != "-" for c in row): nom.append(f"| {name} (own setting) | " + " | ".join(row) + " |")
            if nom:
                print("\nReference: each baseline at its own setting for the nominal budget (not equal memory):\n")
                print("| system | " + " | ".join(f"{float(f):.0%}" for f in FR) + " |")
                print("|---" * (len(FR) + 1) + "|"); print("\n".join(nom))
        base, _ = load(f"{R}/{m}/{w}/phasor_0.45")
        rows = [(n, load(f"{R}/{m}/{w}/abl_{k}")[0]) for k, n in ABL]
        if base and base != "norun" and any(d for _, d in rows):
            print(f"\nE7 ablation at 45% (request s relative to PHASOR {base['request_s']:.2f} s):\n")
            for n, d in rows:
                if d and d != "norun":
                    print(f"- {n}: {d['request_s']:.2f} s ({d['request_s'] / base['request_s'] - 1:+.0%}), "
                          f"TTFT {d['ttft_s']:.2f} s, TPOT {d['tpot_ms']:.0f} ms, peak {d.get('peak_gib', 0):.1f} GiB")
        print()
    # E2
    print("### E2: small budgets (MMLU, each system at its own setting)\n")
    print("| system | " + " | ".join(f"{float(f):.0%} ({gb * float(f):.2f} GiB)" for f in SMALL) + " |")
    print("|---" * (len(SMALL) + 1) + "|")
    for k, name in SYS:
        row = [cell(*load(f"{R}/{m}/mmlu/{k}_{f}"), None, None) for f in SMALL]
        if any(c != "-" for c in row): print(f"| {name} | " + " | ".join(row) + " |")
    print()
    X = f"{R}/{m}/extras"
    # E5
    if os.path.isdir(f"{X}/e5"):
        print("### E5: PHASOR latency breakdown at 45%\n")
        print("| workload | phase | steps | I/O wait | expert matmuls | other MoE | unit hit rate |")
        print("|---|---|---:|---:|---:|---:|---:|")
        for w, wname in WL:
            d, _ = load(f"{X}/e5/{w}_phasor")
            if not d or d == "norun" or "profile_s" not in d: continue
            for ph in ("prefill", "decode"):
                x = d["profile_s"].get(ph)
                if not x: continue
                st = x["layers"] / (32 if m == "mixtral8x7b" else 48)
                h, mi = x.get("hits", 0), x.get("misses", 0)
                unit = "s/request" if ph == "prefill" else "ms/step"
                sc = 1.0 if ph == "prefill" else 1e3
                print(f"| {wname} | {ph} ({unit}) | {st:.0f} | {x['wait'] / st * sc:.2f} | {x['expert'] / st * sc:.2f} | "
                      f"{(x['moe'] - x['wait'] - x['expert']) / st * sc:.2f} | {h / max(h + mi, 1):.3f} |")
        print()
    # E9
    if os.path.isdir(f"{X}/e9"):
        print("### E9: staging window at 45% (PHASOR, request s / TTFT s / TPOT ms / peak GiB)\n")
        for w, wname in WL:
            fs = sorted(f for f in os.listdir(f"{X}/e9") if f.startswith(w + "_") and f.endswith(".json"))
            if not fs: continue
            items = [f"window {f.split('win')[1][:-5]} GiB: {cell(load(f'{X}/e9/{f[:-5]}')[0], '', None, None)}" for f in fs]
            base, _ = load(f"{R}/{m}/{w}/phasor_0.45")
            if base and base != "norun": items.insert(0, f"E1 window: {cell(base, '', None, None)}")
            print(f"- {wname}: " + "; ".join(items))
        print()
    # E3
    if os.path.isdir(f"{X}/e3"):
        print("### E3: batching at 45% (tokens/s; group request s)\n")
        print("| workload | system | batch 1 | batch 4 | batch 8 |")
        print("|---|---|---:|---:|---:|")
        for w, wname in WL:
            for k, name in SYS:
                b1, _ = e1(m, w, k, "0.45")
                cells = []
                if b1 and b1 != "norun":
                    cells.append(f"{sum(r['new_tok'] for r in b1['rows']) / sum(r['request_s'] for r in b1['rows']):.2f}; {b1['request_s']:.2f}")
                else: cells.append("-")
                for B in (4, 8):
                    d, why = load(f"{X}/e3/{w}_{k}_b{B}")
                    cells.append("cannot run" if d == "norun" else
                                 (f"{d['tok_per_s']:.2f}; {d['request_s']:.2f}" if d and "tok_per_s" in d else "-"))
                if any(c != "-" for c in cells[1:]): print(f"| {wname} | {name} | " + " | ".join(cells) + " |")
        print()
