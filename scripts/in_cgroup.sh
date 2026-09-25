#!/bin/bash
# Run a command under a memory cap that counts what the kernel attributes to
# it, page cache included: an mmap-based system otherwise keeps the whole
# model in the page cache outside any budget, which is not a budgeted run.
# usage: in_cgroup.sh <name> <max-bytes|max> <command...>
# The cgroup is fresh per run, so memory.peak is that run's peak.
#
# Host guard.  cudaMalloc is not charged to the cgroup, so a system whose GPU
# cache plus pinned host copy exceeds the unified pool can still exhaust the
# host (seen: MoE-Infinity, 58.7 GiB pinned + its GPU cache, left the kernel
# compacting for 95 min at 10 GiB available).  If MemAvailable falls below
# GUARD_GIB (default 12) the whole cgroup is killed and the run reports it.
CG=/sys/fs/cgroup/ledger_bench/$1; shift
MAX=$1; shift
GUARD_KIB=$(awk -v g="${GUARD_GIB:-12}" 'BEGIN{printf "%d", g*1048576}')
sudo -n rmdir "$CG" 2>/dev/null
sudo -n mkdir -p "$CG" || exit 97
echo "$MAX" | sudo -n tee "$CG/memory.max" >/dev/null
echo 0 | sudo -n tee "$CG/memory.swap.max" >/dev/null 2>&1
killcg(){ echo 1 | sudo -n tee "$CG/cgroup.kill" >/dev/null 2>&1; }
# the subshell moves itself into the cgroup, then becomes the command; the
# guard loop below stays outside it so cgroup.kill does not take it down
( sudo -n sh -c "echo $BASHPID > $CG/cgroup.procs" || exit 98
  exec "$@" ) &
pid=$!
trap 'killcg; wait $pid; exit 143' TERM INT
guard=0
while kill -0 $pid 2>/dev/null; do
  a=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
  if [ "$a" -lt "$GUARD_KIB" ]; then
    echo "HOSTGUARD kill: MemAvailable $((a/1024)) MiB < ${GUARD_GIB:-12} GiB (cgroup charges do not include cudaMalloc)" >&2
    killcg; guard=1; break
  fi
  sleep 0.5
done
wait $pid; rc=$?
[ $guard = 1 ] && exit 137
exit $rc
