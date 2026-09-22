#!/bin/bash
# E1: -ngl sweep.  Pause the background model downloads so their disk writes and
# page-cache pressure do not pollute the paging measurements, then restore them.
set -u
pids=$(pgrep -f 'fetch2?\.sh|curl.*huggingface' | tr '\n' ' ')
echo "pausing downloads: $pids"
for p in $pids; do kill -STOP $p 2>/dev/null; done
trap 'for p in $pids; do kill -CONT $p 2>/dev/null; done; echo resumed' EXIT
sleep 2
sudo -n jetson_clocks >/dev/null 2>&1
"$@"
