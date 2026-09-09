#!/usr/bin/env python3
"""decode-rows.py <server log> [tokens-*.json ...]

One row per request from a decode-rig.sh server log: the server's timing line
(prefill_s, decode_s, decode_tok_s, completion), the runner's decode pool and
I/O counters (expert_hit_rate_decode, expert_misses_decode, hit_fixup_layers,
io_ms, io_fixup_wake_ms, io_fetch_ms, io_hidden_pct), and the two decode gaps
that hold the miss window (the miss window from moe_phase1_hit, or from
moe_spec_routed when the speculative command computes the hits, or from
layer_linear / layer_kv once the layer is one command, to
moe_phase1_miss_fixup_phase2), per token. A tokens-*.json from
decode-stream-client.py adds the streamed answer's wall per token (the mean of
consecutive arrivals after the first chunk) as a separate line.
"""
import json
import re
import statistics
import sys

GAPS = {
    "window": r"gap (?:moe_phase1_hit|moe_spec_routed|layer_linear|layer_kv)->moe_phase1_miss_fixup_phase2 total_ms=\s*[\d.]+ per_token_ms=([\d.]+) count=(\d+)",
    "adopted": r"gap (?:moe_phase1_hit|moe_spec_routed|layer_linear|layer_kv)->moe_phase1_miss_fixup_phase2_adopted total_ms=\s*[\d.]+ per_token_ms=([\d.]+) count=(\d+)",
}
RUNNER = ["expert_hit_rate_decode", "expert_misses_decode", "hit_fixup_layers", "io_ms",
          "io_fixup_wake_ms", "io_fetch_ms", "io_hidden_pct", "cache_plan_ms",
          "prefetch_begin_ms", "prefetch_issued", "prefetch_adopted", "prefetch_reclaimed",
          "prefetch_deferred", "prefetch_overlapped", "prefetch_late", "prefetch_refused", "prefetch_failed",
          "prefetch_joined", "prefetch_landed_hits", "prefetch_before_classify",
          "prefetch_during_tail", "prefetch_during_lt50us", "prefetch_during_50_150us",
          "prefetch_during_gt150us", "prefetch_after_classify", "prefetch_race_unknown",
          "prefetch_hook_failed",
          "router_readback_ms", "path_pin_ms", "path_submit_ms", "path_argbuf_ms",
          "path_hit_encode_ms", "path_fixup_build_ms", "path_hit_commit_to_kernel_ms",
          "path_hit_kernel_to_gpu_ms", "path_fixup_commit_to_kernel_ms", "path_router_wake_ms",
          "path_router_wake_fallbacks"]


def grab(pattern, text, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None


def gap(pattern, text):
    m = re.search(pattern, text)
    return (float(m.group(1)), int(m.group(2))) if m else None


def fmt_gap(value, tokens):
    if value is None:
        return "n/a"
    per_token, count = value
    per_layer = per_token * tokens / count if tokens and count else 0.0
    return f"{per_token:.3f} ({count} layers, {per_layer:.2f} ms each)"


def fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


log_path, token_paths = sys.argv[1], sys.argv[2:]
lines = open(log_path, errors="ignore").read().splitlines()
prefill = next((m.group(0) for ln in lines
                if (m := re.search(r"prefill_router_bits=(\S+(?: \S+=\S+)*)", ln))), None)
print(f"== {log_path.rsplit('/', 1)[-1]}  [{prefill}]")

blocks, cur = [], None
for ln in lines:
    if " request " in ln and " generating" in ln:
        if cur:
            blocks.append(cur)
        cur = [ln]
    elif cur is not None:
        cur.append(ln)
if cur:
    blocks.append(cur)

for block in blocks:
    text = "\n".join(block)
    prompt = grab(r"prompt=(\d+)", text, int)
    cached = grab(r"cached=(\d+)", text, int)
    completion = grab(r"completion=(\d+)", text, int)
    wall = grab(r"completed in ([\d.]+)s", text)
    prefill = grab(r"prefill_s=([\d.]+)", text)
    decode = grab(r"decode_s=([\d.]+)", text)
    tok_s = grab(r"decode_tok_s=([\d.]+)", text)
    runner = {key: grab(rf"{key}=([\d.]+)", text) for key in RUNNER}
    gaps = {key: gap(pattern, text) for key, pattern in GAPS.items()}
    print(f"  prompt={prompt} cached={cached} completion={completion} wall={fmt(wall)}s "
          f"prefill_s={fmt(prefill)} decode_s={fmt(decode)} decode_tok_s={fmt(tok_s, 2)} | "
          f"hit_rate={fmt(runner['expert_hit_rate_decode'], 4)} "
          f"misses={fmt(runner['expert_misses_decode'], 0)} "
          f"fixup_layers={fmt(runner['hit_fixup_layers'], 0)} "
          f"io_ms={fmt(runner['io_ms'])} wake_ms={fmt(runner['io_fixup_wake_ms'])} "
          f"fetch_ms={fmt(runner['io_fetch_ms'])} hidden_pct={fmt(runner['io_hidden_pct'], 1)} "
          f"plan_ms={fmt(runner['cache_plan_ms'])} | "
          f"window_ms/tok={fmt_gap(gaps['window'], completion)} "
          f"adopted_ms/tok={fmt_gap(gaps['adopted'], completion)}")
    if runner["path_pin_ms"] is not None:
        print(f"    path: router_wake={fmt(runner['path_router_wake_ms'])} "
              f"wake_fallbacks={fmt(runner['path_router_wake_fallbacks'], 0)} "
              f"readback={fmt(runner['router_readback_ms'])} plan={fmt(runner['cache_plan_ms'])} "
              f"pin={fmt(runner['path_pin_ms'])} submit={fmt(runner['path_submit_ms'])} "
              f"argbuf={fmt(runner['path_argbuf_ms'])} hit_encode={fmt(runner['path_hit_encode_ms'])} "
              f"fixup_build={fmt(runner['path_fixup_build_ms'])} | "
              f"hit commit>kernel={fmt(runner['path_hit_commit_to_kernel_ms'])} "
              f"kernel>gpu={fmt(runner['path_hit_kernel_to_gpu_ms'])} "
              f"fixup commit>kernel={fmt(runner['path_fixup_commit_to_kernel_ms'])} (ms per token)")
    if runner["prefetch_issued"] is not None:
        print(f"    prefetch: begin_ms={fmt(runner['prefetch_begin_ms'])} "
              f"issued={fmt(runner['prefetch_issued'], 0)} adopted={fmt(runner['prefetch_adopted'], 0)} "
              f"reclaimed={fmt(runner['prefetch_reclaimed'], 0)} "
              f"late={fmt(runner['prefetch_late'], 0)} joined={fmt(runner['prefetch_joined'], 0)} "
              f"refused={fmt(runner['prefetch_refused'], 0)} failed={fmt(runner['prefetch_failed'], 0)} "
              f"deferred={fmt(runner['prefetch_deferred'], 0)} "
              f"overlapped={fmt(runner['prefetch_overlapped'], 0)} "
              f"landed_hits={fmt(runner['prefetch_landed_hits'], 0)} "
              f"before_classify={fmt(runner['prefetch_before_classify'], 0)} "
              f"during_tail={fmt(runner['prefetch_during_tail'], 0)} "
              f"(lt50us={fmt(runner['prefetch_during_lt50us'], 0)} "
              f"50_150us={fmt(runner['prefetch_during_50_150us'], 0)} "
              f"gt150us={fmt(runner['prefetch_during_gt150us'], 0)}) "
              f"after_classify={fmt(runner['prefetch_after_classify'], 0)} "
              f"race_unknown={fmt(runner['prefetch_race_unknown'], 0)} "
              f"hook_failed={fmt(runner['prefetch_hook_failed'], 0)}")

for path in token_paths:
    arrivals = [t[1] for t in json.load(open(path))["tokens"]]
    if len(arrivals) < 3:
        print(f"  {path.rsplit('/', 1)[-1]}: {len(arrivals)} tokens, no wall")
        continue
    walls = [b - a for a, b in zip(arrivals[1:], arrivals[2:])]
    print(f"  {path.rsplit('/', 1)[-1]}: {len(arrivals)} tokens, first chunk {arrivals[0]:.0f} ms, "
          f"wall/token mean {statistics.mean(walls):.2f} ms (median {statistics.median(walls):.2f}, "
          f"{1000 / statistics.mean(walls):.2f} tok/s)")
