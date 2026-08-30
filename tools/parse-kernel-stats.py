#!/usr/bin/env python3
"""Aggregate 'Shrike kernel' / 'Shrike gap' diagnostic blocks.

Usage: parse-kernel-stats.py <file> [lastN]
Each generation emits role lines, a total line, gap lines, and a busy/span
line (the block terminator). Averages the last N blocks (default 12).
"""
import re
import statistics
import sys

path = sys.argv[1]
last_n = int(sys.argv[2]) if len(sys.argv) > 2 else 12

blocks = []
cur = {"roles": {}, "gaps": {}, "occ": None}
for line in open(path, errors="ignore"):
    m = re.search(r"Shrike kernel role=(\S+) gpu_ms=([\d.]+) per_token_ms=([\d.]+) count=(\d+)", line)
    if m:
        cur["roles"][m.group(1)] = (float(m.group(3)), int(m.group(4)))
        continue
    m = re.search(r"Shrike gap (\S+) total_ms=([\d.]+) per_token_ms=([\d.]+) count=(\d+)", line)
    if m:
        cur["gaps"][m.group(1)] = float(m.group(3))
        continue
    m = re.search(r"Shrike kernel busy_ms=([\d.]+) span_ms=([\d.]+) occupancy=([\d.]+)% "
                  r"busy_share_of_decode=([\d.]+)% busy_per_token_ms=([\d.]+)", line)
    if m:
        cur["occ"] = tuple(float(m.group(i)) for i in range(1, 6))
        blocks.append(cur)
        cur = {"roles": {}, "gaps": {}, "occ": None}

blocks = blocks[-last_n:]
print(f"blocks aggregated: {len(blocks)}")
if not blocks:
    sys.exit(0)

def mean_sd(vals):
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

roles = sorted({r for b in blocks for r in b["roles"]})
stats = []
for r in roles:
    vals = [b["roles"][r][0] for b in blocks if r in b["roles"]]
    counts = [b["roles"][r][1] for b in blocks if r in b["roles"]]
    m, sd = mean_sd(vals)
    stats.append((m, sd, statistics.mean(counts), len(vals), r))
stats.sort(reverse=True)
total = sum(m for m, *_ in stats)
print(f"\nper-role GPU per-token ms (mean of {len(blocks)} runs; role sums overlap by design)")
for m, sd, cnt, n, r in stats:
    print(f"  {r:32s} {m:8.3f}  sd {sd:6.3f}   {100*m/total:5.1f}%  count/run {cnt:7.1f}  (n={n})")
print(f"  {'SUM OF ROLES':32s} {total:8.3f}")

occs = [b["occ"] for b in blocks if b["occ"]]
if occs:
    busy_pt = [o[4] for o in occs]
    occ_pct = [o[2] for o in occs]
    share = [o[3] for o in occs]
    print(f"\nqueue occupancy: busy/span {statistics.mean(occ_pct):.1f}%   "
          f"busy_share_of_decode {statistics.mean(share):.1f}%   "
          f"busy_per_token {statistics.mean(busy_pt):.3f} ms")

gaps = sorted({g for b in blocks for g in b["gaps"]})
gstats = []
for g in gaps:
    vals = [b["gaps"].get(g, 0.0) for b in blocks]
    gstats.append((statistics.mean(vals), g))
gstats.sort(reverse=True)
print("\ntop gaps, per-token ms (GPU idle attributed to transition)")
for m, g in gstats[:8]:
    print(f"  {g:56s} {m:7.3f}")
