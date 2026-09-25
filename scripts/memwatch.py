"""Peak memory of a serving run on a unified-memory device.

cgroup memory.peak misses cudaMalloc (a GPU-side cache escapes it), RSS misses
it too, cudaMemGetInfo double counts mapped memory; the drop in MemAvailable is
the one measure that sees every allocation (sec 4.4).  A thread samples it every
0.2 s from before the model is built until the run ends."""
import threading, time

def mem_avail_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return 0.0

class MemWatch:
    def __init__(self, period=0.2):
        self.base = mem_avail_gib(); self.low = self.base
        self.stop = False; self.period = period
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start()
    def _run(self):
        while not self.stop:
            self.low = min(self.low, mem_avail_gib()); time.sleep(self.period)
    def peak_gib(self):
        self.low = min(self.low, mem_avail_gib()); return self.base - self.low
    def close(self):
        self.stop = True; self.t.join(timeout=1)
