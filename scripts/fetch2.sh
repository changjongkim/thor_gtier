#!/bin/bash
# Phase 2: Qwen3-235B-A22B (MoE, 22B active) as a size knob across Thor's
# 122 GiB DRAM boundary.  Same architecture and access pattern; only bytes change.
set -u
while ! grep -q "ALL DONE" /home/thor/kcj/models/fetch.log 2>/dev/null; do sleep 60; done
cd /home/thor/kcj/models
REPO=unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF
dl() { # subdir(quant), outdir
  local q="$1" out="$2"; mkdir -p "$out"
  files=$(curl -s "https://huggingface.co/api/models/$REPO" | python3 -c "
import sys,json
d=json.load(sys.stdin)
print('\n'.join(sorted(s['rfilename'] for s in d.get('siblings',[])
      if s['rfilename'].startswith('$q/') and s['rfilename'].endswith('.gguf'))))")
  for f in $files; do
    b=$(basename $f)
    [ -s "$out/$b" ] && { echo "  have $b"; continue; }
    echo "  get $b"
    curl -sL --retry 5 -C - -o "$out/$b" "https://huggingface.co/$REPO/resolve/main/$f" || echo "  FAIL $f"
  done
  echo "== $out : $(du -sh $out 2>/dev/null | cut -f1)  free: $(df -h /home/thor/kcj|tail -1|awk '{print $4}')"
}
echo "### 235B Q3_K_M  (104.7 GiB, 0.86x DRAM - FITS)" ; dl Q3_K_M  moe235b_q3km
echo "### 235B Q4_K_M  (132.4 GiB, 1.09x DRAM - EXCEEDS)"; dl Q4_K_M moe235b_q4km
echo "PHASE2 DONE"; df -h /home/thor/kcj | tail -1
