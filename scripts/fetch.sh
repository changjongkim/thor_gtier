#!/bin/bash
# Download the Qwen GGUF ladder that spans Thor's measured mmap regimes.
set -u
cd /home/thor/kcj/models
dl() { # repo, pattern, outdir
  local repo="$1" pat="$2" out="$3"
  mkdir -p "$out"
  files=$(curl -s "https://huggingface.co/api/models/$repo" | python3 -c "
import sys,json
d=json.load(sys.stdin)
print('\n'.join(sorted(s['rfilename'] for s in d.get('siblings',[])
      if s['rfilename'].endswith('.gguf') and '$pat' in s['rfilename'].lower())))")
  for f in $files; do
    if [ -s "$out/$(basename $f)" ]; then echo "  have $(basename $f)"; continue; fi
    echo "  get $(basename $f)"
    curl -sL --retry 5 -C - -o "$out/$(basename $f)" \
      "https://huggingface.co/$repo/resolve/main/$f" || echo "  FAIL $f"
  done
  echo "== $out : $(du -sh $out 2>/dev/null | cut -f1)"
}
echo "### 7B q8_0"  ; dl Qwen/Qwen2.5-7B-Instruct-GGUF  q8_0 qwen7b_q8
echo "### 14B q8_0" ; dl Qwen/Qwen2.5-14B-Instruct-GGUF q8_0 qwen14b_q8
echo "### 32B q8_0" ; dl Qwen/Qwen2.5-32B-Instruct-GGUF q8_0 qwen32b_q8
echo "### 72B q8_0" ; dl Qwen/Qwen2.5-72B-Instruct-GGUF q8_0 qwen72b_q8
echo "ALL DONE"; df -h /home/thor/kcj | tail -1
