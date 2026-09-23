#!/bin/bash
# One queue at a time.  Two queues sharing the NVMe measure each other rather
# than the device, and every number in this repository is a device number.
LOCK=/tmp/gtier_queue.lock
exec 9>"$LOCK"
flock 9
"$@"
