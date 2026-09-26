#!/usr/bin/env python3
"""Keep the page cache of the given model files empty while a run is alive.

Equal memory counts every byte a system keeps in memory.  A system that reads
its weights through the page cache (buffered pread) otherwise gets a second,
uncounted cache: MemAvailable treats page cache as free, and the cgroup lets it
fill the cap.  Dropping the files' clean pages every 0.2 s makes buffered reads
behave like direct I/O for every system alike, without touching their code
(PHASOR reads with O_DIRECT and is unaffected).

usage: scrub_cache.py <pid> <glob> [<glob> ...]   (exits when <pid> exits)
"""
import glob, os, sys, time
pid = int(sys.argv[1])
files = sorted({f for g in sys.argv[2:] for f in glob.glob(g) if os.path.isfile(f)})
fds = [os.open(f, os.O_RDONLY) for f in files]
print(f"scrub: {len(fds)} files", file=sys.stderr, flush=True)
while True:
    try: os.kill(pid, 0)
    except OSError: break
    for fd in fds: os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    time.sleep(0.2)
