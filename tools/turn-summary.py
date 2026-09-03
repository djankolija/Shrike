#!/usr/bin/env python3
"""Summarize a turn-rig.sh server log, one row per request.

The server's own timing line (prefill_s / decode_s / completed-in / prompt /
cached / completion), busy vs span, the prefill role sum, and the host terms
of the three per-layer-chunk gaps.

Usage: turn-summary.py <server log>
"""
import re
import sys

log = open(sys.argv[1], errors="ignore").read().splitlines()
blocks, cur = [], None
for ln in log:
    if " request " in ln and " generating" in ln:
        cur = [ln]
    elif cur is not None:
        cur.append(ln)
        if " completed in " in ln:
            blocks.append(cur); cur = []


def grab(pattern, text, cast=float, default=None):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else default


for b in blocks:
    text = "\n".join(b)
    done = grab(r"completed in ([\d.]+)s", text)
    prompt = grab(r"prompt=(\d+)", text, int); cached = grab(r"cached=(\d+)", text, int)
    comp = grab(r"completion=(\d+)", text, int)
    pre = grab(r"prefill_s=([\d.]+)", text); dec = grab(r"decode_s=([\d.]+)", text)
    busy = grab(r"busy_ms=([\d.]+) span_ms", text); span = grab(r"span_ms=([\d.]+)", text)
    roles = {}
    for m in re.finditer(r"role=(prefill_\w+|attn_layer_\w+|moe_\w+|head_logits) gpu_ms=([\d.]+)", text):
        roles[m.group(1)] = roles.get(m.group(1), 0.0) + float(m.group(2))
    prefill_gpu = sum(v for k, v in roles.items() if k.startswith("prefill_"))
    gaps = {}
    for m in re.finditer(r"gap (\S+)->(\S+) total_ms=\s*([\d.]+) per_token_ms=[\d.]+ count=(\d+) host_ms=\s*([\d.]+)", text):
        gaps[f"{m.group(1)}->{m.group(2)}"] = (float(m.group(3)), float(m.group(5)), int(m.group(4)))
    new = (prompt or 0) - (cached or 0)
    print(f"prompt={prompt} cached={cached} new={new} completion={comp} | wall(server)={done} s prefill_s={pre} decode_s={dec} "
          f"outside={None if done is None or pre is None or dec is None else round(done - pre - dec, 2)} | busy={busy} span={span} "
          f"| prefill GPU={prefill_gpu:.0f} ms ({(prefill_gpu / new) if new else 0:.2f} ms/new tok)")
    for k in ("prefill_shared_expert->prefill_routed_tile", "prefill_routed_tile->prefill_routed_tile", "prefill_moe_reduce->prefill_gdn_router"):
        if k in gaps:
            t, h, c = gaps[k]
            print(f"    {k}: total {t:.0f} host {h:.0f} over {c}")
