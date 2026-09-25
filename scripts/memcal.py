#!/usr/bin/env python3
"""Equal memory for every system.  Given a target peak (PHASOR's measured peak
MemAvailable drop at a budget), find the --budget-gib to pass to another system
so that its measured peak lands within 5% of the target: secant search over
2-prompt MMLU runs (peak is close to linear in each system's memory knob).

usage: memcal.py <target-peak-gib> <initial-budget> <out.json> -- <runner cmd with {B} and {OUT}>
"""
import json, subprocess, sys
target, b0, out = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
cmd = sys.argv[sys.argv.index("--") + 1:]

def peak(b):
    o = out + f".b{b:.2f}.json"
    c = [x.replace("{B}", f"{b:.2f}").replace("{OUT}", o) for x in cmd]
    r = subprocess.run(c, capture_output=True, text=True)
    try:
        return json.load(open(o))["peak_gib"]
    except Exception:
        sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:]); return None

pts = []
b = b0
for it in range(5):
    p = peak(b)
    if p is None:
        break
    pts.append((b, p)); print(f"budget {b:.2f} -> peak {p:.2f} (target {target:.2f})", flush=True)
    if abs(p - target) <= 0.05 * target:
        break
    if len(pts) == 1:
        b = max(0.5, b + (target - p))                    # peak moves about 1:1 with the knob
    else:
        (b1, p1), (b2, p2) = pts[-2], pts[-1]
        slope = (p2 - p1) / (b2 - b1) if b2 != b1 else 1.0
        b = max(0.5, b2 + (target - p2) / (slope if abs(slope) > 0.05 else 1.0))
# Only settings that stay within the target count: a system that cannot get
# under PHASOR's memory at any setting does not run at this budget.
ok = [x for x in pts if x[1] <= target * 1.05]
best = min(ok, key=lambda x: abs(x[1] - target)) if ok else (None, None)
json.dump({"target_peak_gib": target, "budget_gib": best[0], "peak_gib": best[1], "tries": pts}, open(out, "w"))
print(f"CAL target={target:.2f} budget={best[0]} peak={best[1]}")
