# Slower SSD by NVMe power state (paper 4.8)

The kernel has no block-I/O throttling (CONFIG_BLK_DEV_THROTTLING unset), so cgroup `io.max` cannot
emulate a slower device. The drive (WD PC SN5000S, DRAM-less) has three operational power states;
capping it lowers its read bandwidth. Measured 2026-09-27 13:16 between two matrix runs,
`lib/gtier_bench` (gtier async, O_DIRECT, 3 GiB span, 4 GiB read):

| power state | max power | 4 MiB reads | 1 MiB reads |
|---|---:|---:|---:|
| PS0 (default) | 5.6 W | 3.52 / 3.31 GiB/s | 3.90 / 4.01 GiB/s |
| PS1 | 3.0 W | 1.33 GiB/s | 1.31 GiB/s |
| PS2 | 2.5 W | 0.73 GiB/s | 0.68 GiB/s |

(PS0 measured before and after.) Set with `nvme set-feature /dev/nvme0 --feature-id=2 --value=N`
(`-v` is verbose in this nvme-cli: an earlier probe with `-v N` left PS0 in place and is void).
Not saved across reboots. Stage 6 runs every system at 45% on MMLU at PS1 and PS2 and restores PS0 on exit.
