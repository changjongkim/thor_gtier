// CPU cost of a timed region.  getrusage(RUSAGE_SELF) covers every thread of
// the process (live or joined, io_uring workers included).  Work the kernel
// does on the process's behalf in other contexts -- GPU fault service threads,
// completion interrupts -- is not charged to it, so the whole-machine busy
// time from /proc/stat is recorded next to it.
#pragma once
#include <sys/resource.h>
#include <chrono>
#include <cstdio>
#include <unistd.h>

struct CpuSnap {
    double user = 0, sys = 0, busy = 0, wall = 0;
    static CpuSnap now() {
        CpuSnap s;
        rusage r{}; getrusage(RUSAGE_SELF, &r);
        s.user = r.ru_utime.tv_sec + r.ru_utime.tv_usec * 1e-6;
        s.sys = r.ru_stime.tv_sec + r.ru_stime.tv_usec * 1e-6;
        if (FILE *f = std::fopen("/proc/stat", "r")) {
            unsigned long long v[8] = {};
            if (std::fscanf(f, "cpu %llu %llu %llu %llu %llu %llu %llu %llu",
                            &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]) == 8)
                // user nice system [idle iowait] irq softirq steal
                s.busy = (double)(v[0] + v[1] + v[2] + v[5] + v[6] + v[7]) / sysconf(_SC_CLK_TCK);
            std::fclose(f);
        }
        s.wall = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        return s;
    }
};

// One machine-readable line per timed region.
inline void cpu_report(const char *backend, size_t item, double gib,
                       const CpuSnap &a, const CpuSnap &b) {
    double w = b.wall - a.wall, u = b.user - a.user, s = b.sys - a.sys, m = b.busy - a.busy;
    std::printf("CPU backend=%s item=%zu gib=%.4f wall_s=%.4f user_s=%.4f sys_s=%.4f "
                "machine_busy_s=%.3f gibps=%.4f user_s_per_gib=%.4f sys_s_per_gib=%.4f "
                "avg_cores=%.3f machine_cores=%.3f\n",
                backend, item, gib, w, u, s, m, gib / w, u / gib, s / gib, (u + s) / w, m / w);
}
