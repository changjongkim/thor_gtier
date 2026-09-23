// Is there duplicate content to exploit?  Quantised weights encode many float
// values into few codes, so identical blocks are not impossible.  If a
// meaningful fraction of the model is duplicated, a content-addressed cache
// would hold more of it in the same window -- which is the only lever left once
// the transfer path is at the device ceiling.
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static uint64_t fnv(const uint8_t *p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char **argv) {
    const char *path = argv[1];
    size_t blk = argc > 2 ? strtoull(argv[2], 0, 10) : 4096;
    size_t cap_gib = argc > 3 ? strtoull(argv[3], 0, 10) : 8;

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    size_t limit = cap_gib << 30, n = limit / blk;
    uint8_t *buf = malloc(blk);
    uint64_t *h = malloc(n * sizeof(uint64_t));
    size_t got = 0;
    for (size_t i = 0; i < n; ++i) {
        if (read(fd, buf, blk) != (ssize_t)blk) break;
        h[got++] = fnv(buf, blk);
    }
    close(fd);

    int cmp(const void *a, const void *b) {
        uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
        return x < y ? -1 : x > y;
    }
    qsort(h, got, sizeof(uint64_t), cmp);
    size_t uniq = got ? 1 : 0, maxrun = 1, run = 1;
    for (size_t i = 1; i < got; ++i) {
        if (h[i] != h[i - 1]) { ++uniq; if (run > maxrun) maxrun = run; run = 1; }
        else ++run;
    }
    if (run > maxrun) maxrun = run;
    printf("block=%6zuB  scanned=%.1f GiB  blocks=%zu  unique=%zu  "
           "dup=%.2f%%  largest dup group=%zu\n",
           blk, (double)got * blk / 1073741824.0, got, uniq,
           100.0 * (got - uniq) / (got ? got : 1), maxrun);
    return 0;
}
