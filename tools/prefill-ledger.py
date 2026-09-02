#!/usr/bin/env python3
"""Per-request prefill ledger from a ShrikeServer log with SHRIKE_KERNEL_STATS.

Splits the log into per-request blocks at the 'Shrike kernel busy_ms' line
(the block terminator), pairs each block with the preceding gen_diag line for
the prompt token count when present, and reports every prefill_* role
normalized per PROMPT token (the server's per_token_ms divides by generated
tokens). Pass the prompt token count explicitly when the log has no gen_diag
line (the production launch); prefill-measure.sh prints it from the response.

Usage: prefill-ledger.py <server.log> [lastN] [prompt_tokens]
"""
import re
import sys

path = sys.argv[1]
last_n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
prompt_override = int(sys.argv[3]) if len(sys.argv) > 3 else None

blocks = []
cur = {"roles": {}, "gaps": [], "prefill_tokens": None, "generated": None,
       "runner": {}, "busy": None}
for line in open(path, errors="ignore"):
    m = re.search(r"Shrike gen_diag prefill=(\d+) generated=(\d+)", line)
    if m:
        cur["prefill_tokens"] = int(m.group(1))
        cur["generated"] = int(m.group(2))
        continue
    if "Shrike runner " in line:
        cur["runner"] = dict(re.findall(r"(\w+)=([\d.]+)", line))
        continue
    m = re.search(r"Shrike kernel role=(\S+) gpu_ms=([\d.]+) per_token_ms=[\d.]+ count=(\d+)", line)
    if m:
        cur["roles"][m.group(1)] = (float(m.group(2)), int(m.group(3)))
        continue
    m = re.search(r"Shrike gap (\S+) total_ms=([\d.]+) per_token_ms=[\d.]+ count=(\d+)", line)
    if m:
        split = re.search(r"host_ms=([\d.]+) driver_ms=([\d.]+) queue_ms=([\d.]+)", line)
        cur["gaps"].append((m.group(1), float(m.group(2)), int(m.group(3)),
                            tuple(float(g) for g in split.groups()) if split else None))
        continue
    m = re.search(r"Shrike kernel busy_ms=([\d.]+) span_ms=([\d.]+) occupancy=([\d.]+)%", line)
    if m:
        cur["busy"] = tuple(float(m.group(i)) for i in range(1, 4))
        blocks.append(cur)
        cur = {"roles": {}, "gaps": [], "prefill_tokens": None, "generated": None,
               "runner": {}, "busy": None}

if not blocks:
    print(f"no kernel-stats blocks found in {path} (is SHRIKE_KERNEL_STATS=1 set?)",
          file=sys.stderr)
    sys.exit(1)

for b in blocks[-last_n:]:
    n = b["prefill_tokens"] or prompt_override or 0
    print(f"=== prompt_tokens={n} generated={b['generated']} "
          f"busy_ms={b['busy'][0]:.0f} span_ms={b['busy'][1]:.0f} occupancy={b['busy'][2]:.1f}%")
    prefill_total = 0.0
    for role, (ms, count) in sorted(b["roles"].items(), key=lambda kv: -kv[1][0]):
        is_prefill = role.startswith("prefill")
        if is_prefill:
            prefill_total += ms
        per_prompt = f"{ms / n:8.3f} ms/prompt-tok" if (is_prefill and n) else " " * 21
        print(f"  {role:24s} gpu_ms={ms:10.1f} count={count:6d} {per_prompt}")
    if n:
        role_sum = (f"  {'PREFILL ROLE SUM':24s} gpu_ms={prefill_total:10.1f} "
                    f"{'':13s}{prefill_total / n:8.3f} ms/prompt-tok")
        if prefill_total:
            role_sum += f"  ({n / (prefill_total / 1000):.0f} tok/s GPU-only)"
        print(role_sum)
        print(f"  {'GPU BUSY / GAPS':24s} {b['busy'][0] / n:8.3f} / "
              f"{(b['busy'][1] - b['busy'][0]) / n:.3f} ms/prompt-tok")
    for key in ("expert_hits_prefill", "expert_misses_prefill",
                "expert_hit_rate_prefill", "io_ms", "wait_ms"):
        if key in b["runner"]:
            print(f"  runner.{key}={b['runner'][key]}")
    for name, ms, count, split in b["gaps"][:4]:
        line = f"  gap {name:28s} total_ms={ms:9.1f}"
        if split:
            line += (f" host_ms={split[0]:8.1f} driver_ms={split[1]:8.1f}"
                     f" queue_ms={split[2]:8.1f}")
        print(f"{line} count={count}")
