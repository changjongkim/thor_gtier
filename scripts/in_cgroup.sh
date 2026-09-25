#!/bin/bash
# Run a command under a memory cap that counts what the kernel attributes to
# it, page cache included: an mmap-based system otherwise keeps the whole
# model in the page cache outside any budget, which is not a budgeted run.
# usage: in_cgroup.sh <name> <max-bytes|max> <command...>
# The cgroup is fresh per run, so memory.peak is that run's peak.
CG=/sys/fs/cgroup/ledger_bench/$1; shift
MAX=$1; shift
sudo -n rmdir "$CG" 2>/dev/null
sudo -n mkdir -p "$CG" || exit 97
echo "$MAX" | sudo -n tee "$CG/memory.max" >/dev/null
echo 0 | sudo -n tee "$CG/memory.swap.max" >/dev/null 2>&1
sudo -n sh -c "echo $$ > $CG/cgroup.procs" || exit 98
exec "$@"
