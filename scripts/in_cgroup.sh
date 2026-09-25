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
guard=0; pinned_since=0; tick=0
maxb=$(cat "$CG/memory.max")
while kill -0 $pid 2>/dev/null; do
  # Cap thrash: memory the kernel cannot reclaim (anon + shmem, i.e. pinned
  # host buffers; there is no swap) held at >= 97% of the cap for 2 min means
  # the run only survives by reclaiming page cache it immediately needs again
  # (seen: a pinned host LRU at 19 GiB of a 21 GiB cap, no request in 20 min).
  # It cannot run within this memory; stop it rather than wait for the timeout.
  tick=$((tick+1))
  if [ "$maxb" != max ] && [ $((tick % 10)) = 0 ]; then
    u=$(awk '/^anon /{a=$2} /^shmem /{s=$2} END{print a+s}' "$CG/memory.stat")
    if [ "$u" -ge $((maxb / 100 * 97)) ]; then
      [ $pinned_since = 0 ] && pinned_since=$SECONDS
      if [ $((SECONDS - pinned_since)) -ge 120 ]; then
        echo "CAPTHRASH kill: unreclaimable $((u>>20)) MiB >= 97% of the $((maxb>>20)) MiB cap for 120 s" >&2
        killcg; guard=1; break
      fi
    else pinned_since=0; fi
  fi
  a=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
  if [ "$a" -lt "$GUARD_KIB" ]; then
    echo "HOSTGUARD kill: MemAvailable $((a/1024)) MiB < ${GUARD_GIB:-12} GiB (cgroup charges do not include cudaMalloc)" >&2
    killcg; guard=1; break
  fi
  sleep 0.5
done
wait $pid; rc=$?
# E4/E6 accounting for the whole run (every process and thread of the system):
# bytes read from block devices, and CPU time
awk '{for(i=2;i<=NF;i++) if($i ~ /^rbytes=/){split($i,a,"="); s+=a[2]}} END{printf "cgroup_io_read_gib=%.3f\n", s/1073741824}' "$CG/io.stat" 2>/dev/null
awk '/^usage_usec/{u=$2} /^user_usec/{us=$2} /^system_usec/{sy=$2} END{printf "cgroup_cpu_s=%.1f cgroup_user_s=%.1f cgroup_sys_s=%.1f\n", u/1e6, us/1e6, sy/1e6}' "$CG/cpu.stat" 2>/dev/null
[ $guard = 1 ] && exit 137
exit $rc
