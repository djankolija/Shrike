#!/usr/bin/env python3
"""Mean/sd of every numeric field on the last N 'Shrike runner' lines."""
import re
import statistics
import sys

path = sys.argv[1]
last_n = int(sys.argv[2]) if len(sys.argv) > 2 else 12

rows = []
for line in open(path, errors="ignore"):
    if "Shrike runner" not in line:
        continue
    rows.append(dict(re.findall(r"(\w+)=([\d.]+)", line)))
rows = rows[-last_n:]
print(f"lines aggregated: {len(rows)}")

for key in rows[0]:
    vals = [float(r[key]) for r in rows if key in r]
    m = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"  {key:28s} {m:10.4f}  sd {sd:8.4f}")
