#!/bin/bash
# Refuse to launch a run that would ask for more memory than the machine has.
#
# Five host restarts came from one configuration: llama.cpp with -ngl 99 on a
# 132.4 GiB model, on a device whose GPU and CPU share 122.8 GiB and which has
# no swap.  It did not fail the process -- the kernel went down with it, 18 to
# 39 minutes in, taking whatever else was running.  A memory cgroup does not
# help because it does not charge cudaMalloc (results/HF_MOE/FOOTPRINT.md), so
# the check has to happen before the process starts.
#
#   memguard.sh --need <GiB> -- <command...>
#
# Also usable as a library: `source memguard.sh` then `mg_check <GiB>`.
set -u

mg_total_gib(){ awk '/^MemTotal:/{printf "%.1f", $2/1048576}' /proc/meminfo; }
mg_avail_gib(){ awk '/^MemAvailable:/{printf "%.1f", $2/1048576}' /proc/meminfo; }

# The ceiling is deliberately below total: the kernel, the page tables for a
# hundred-gigabyte mapping, and the CUDA context all need room, and the
# failure mode for getting this wrong is not a slow run.
MG_CEILING_FRAC=${MG_CEILING_FRAC:-0.85}

mg_check(){
  local need=$1
  local total ceiling
  total=$(mg_total_gib)
  ceiling=$(awk -v t="$total" -v f="$MG_CEILING_FRAC" 'BEGIN{printf "%.1f", t*f}')
  if awk -v n="$need" -v c="$ceiling" 'BEGIN{exit !(n>c)}'; then
    echo "memguard: REFUSED -- needs ${need} GiB, ceiling ${ceiling} GiB of ${total} total" >&2
    echo "memguard: this is the configuration that took the host down five times" >&2
    return 1
  fi
  echo "memguard: ok -- needs ${need} GiB, ceiling ${ceiling} GiB, available $(mg_avail_gib)" >&2
  return 0
}

# Resident demand of a GGUF model at a given number of offloaded layers.
#   mg_gguf_need <model-dir> <ngl> <n_layers> [extra_gib]
mg_gguf_need(){
  local dir=$1 ngl=$2 nl=$3 extra=${4:-8}
  local bytes; bytes=$(du -sb "$dir"/*.gguf 2>/dev/null | awk '{s+=$1} END{print s+0}')
  awk -v b="$bytes" -v g="$ngl" -v n="$nl" -v e="$extra" 'BEGIN{
    gib = b/1073741824
    frac = (g>=n || g<0) ? 1.0 : g/n
    # Whatever is not offloaded still has to be read, but only the offloaded
    # part has to be resident all at once.
    printf "%.1f", gib*frac + e
  }'
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  need=""; while [ $# -gt 0 ]; do
    case "$1" in --need) need=$2; shift 2;; --) shift; break;; *) shift;; esac
  done
  [ -z "$need" ] && { echo "usage: memguard.sh --need <GiB> -- <cmd...>" >&2; exit 2; }
  mg_check "$need" || exit 3
  exec "$@"
fi
