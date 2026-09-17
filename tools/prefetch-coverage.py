#!/usr/bin/env python3
"""Price a next-layer expert prefetch offline (v14 decode pass II, Task 1 Step 1).

Two independent modes, each reading a different capture and needing no
runtime code:

  join <trace.jsonl>
    A SHRIKE_PREFETCH_TRACE capture: one JSON line per (position, layer)
    decode plan, written by RealForwardRunner.recordPrefetchTrace. Fields:
    position, layer, probe_distance (d, constant across the capture),
    experts (the routed top-k at this layer), misses (the demand plan's own
    miss identities at this layer), resident (the expert ids resident in
    this layer's pool, captured before planning), next_layer_prediction
    (the router probe's top-k for layer L + d, empty when L + d is past the
    last layer).

    For every (request, position, L) whose next_layer_prediction is
    non-empty, join it to the row at (request, position, L + d): the target
    T = L + d, absent set A = that row's own misses, and P_M = the first M
    entries of next_layer_prediction (exactly the runtime's own
    predictedNextLayer.prefix(M) where encodeDecodeRoutedMoE calls
    predictivePrefetch.begin). What the
    scheme would fetch is F_M = P_M minus T's own resident set. Reports,
    per capture and per layer, at each M in --top-m: per-miss recall (sum
    |A & P_M| / sum |A|, over rows with a non-empty A), nonresident
    precision (sum |F_M & A| / sum |F_M|, over every joined row), and
    full-layer coverage p (the fraction of rows with a non-empty A where A
    is a subset of P_M) -- p, not recall, is the prize's multiplier, since a
    partly covered layer still pays the serial read. Also reports wasted
    reads per token (sum |F_M - A| / tokens found) and the modelled prize p
    x --prize-ms, marked MODELLED.

    A request boundary is inferred, since the capture carries no request
    marker of its own: any row whose position is less than the previous
    row's position opens a new request (positions repeat across one
    token's 40-odd layer lines, then increase by one per token, so a true
    decrease is the only signal available).
    The rule misses a boundary when a follow-up request's prompt is longer
    than the previous request's last decode position (its first position
    then rises instead of falling); the archived captures hold one
    streamed answer per lifetime, where this cannot occur.

  history <trace>
    A SHRIKE_ROUTE_TRACE capture, replayed with tools/expert-pool-replay.py
    (imported, never edited) at production's pool configuration, mirroring
    the archived recorder loop-files/step0-layer-misses.py's own hooking:
    group_into_chunks is wrapped to learn each plan's real layer index and
    make_pool is wrapped so every LayerPool's plan() is observed. Unlike
    that recorder, the hook here runs BEFORE the inner plan (so the absent
    set reflects the pool's state prior to this plan's own evictions) and
    records the full per-decode row: request id, position, layer, the
    demanded experts, and the absent set (the demanded experts not present
    in pool.slot_expert, i.e. not resident anywhere).

    Evaluates three router-free predictors of the absent set at
    (request, position, layer), over rows with a non-empty absent set:
    prev-token (the previous position's demanded set at the same layer),
    last-n-union (the union of the last n positions' demanded sets at the
    same layer, for each n in --history-n, using whatever predecessors
    exist when fewer than n are available), and same-position-prev-layer
    (the demanded set at (position, layer - 1)). Reports per-miss recall
    and full-layer coverage p for each. Also reports the absent-set size
    histogram, the mean missing layers per token, and the replay's own
    total decode misses per request (to check against a box's own counts).

Both modes print human-readable markdown tables to stdout and, with
--json <path>, write the same numbers as JSON. Every number is labelled;
the modelled prize (and anything derived from it) is marked MODELLED,
everything else is measured directly from the capture.

Usage:
  prefetch-coverage.py join <trace.jsonl> [--top-m 4,8] [--prize-ms 11.5]
                       [--wall-ms MS] [--json PATH]
  prefetch-coverage.py history <trace> [--history-n 2,4,8] [--slots 128]
                       [--json PATH]
  prefetch-coverage.py --self-test
"""
import argparse
import importlib.util
import json
import os
import sys
import tempfile
from collections import defaultdict

DEFAULT_TOP_M = "4,8"
DEFAULT_HISTORY_N = "2,4,8"
DEFAULT_PRIZE_MS = 11.5
DEFAULT_SLOTS = 128
HISTOGRAM_MAX = 8


def parse_int_list(raw):
    return [int(x) for x in raw.split(",") if x.strip()]


def fmt_ratio(x):
    return "n/a" if x is None else f"{x:.4f}"


def fmt_ms(x):
    return "n/a" if x is None else f"{x:.3f}"


def fmt_num(x):
    return "n/a" if x is None else f"{x:.4f}"


def render_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


RANKING_MISMATCHES = {"rows": 0, "mismatched": 0}


def load_jsonl(path):
    """The capture's plan rows. A v20 ranking row ({position, layer,
    probe_ranking}, the probe's order past its top-8, written after the
    position's plan rows) is folded into the latest plan row at its (position,
    layer) as `probe_ranking`; the count whose first eight differ from that
    row's own top-8 is kept in RANKING_MISMATCHES (a stale slot)."""
    rows = []
    latest = {}
    with open(path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if "probe_ranking" in row and "experts" not in row:
                plan = latest.get((row["position"], row["layer"]))
                if plan is None:
                    continue
                ranking = row["probe_ranking"]
                top = plan.get("next_layer_prediction") or []
                RANKING_MISMATCHES["rows"] += 1
                if ranking[:len(top)] != top:
                    RANKING_MISMATCHES["mismatched"] += 1
                    continue
                plan["probe_ranking"] = ranking
                continue
            rows.append(row)
            latest[(row["position"], row["layer"])] = row
    if RANKING_MISMATCHES["rows"]:
        print(f"probe rankings: {RANKING_MISMATCHES['rows']} rows, "
              f"{RANKING_MISMATCHES['mismatched']} whose top-8 differ from the plan row's "
              f"(dropped)")
    return rows


def assign_requests(rows):
    request_ids = []
    rid = 0
    prev_position = None
    for row in rows:
        position = row["position"]
        if prev_position is None or position < prev_position:
            rid += 1
        request_ids.append(rid)
        prev_position = position
    return request_ids


def index_rows(rows, request_ids):
    by_rp = defaultdict(dict)
    for row, rid in zip(rows, request_ids):
        by_rp[(rid, row["position"])][row["layer"]] = row
    return by_rp


def build_joins(rows, request_ids, by_rp):
    joins = []
    missing_target = 0
    for row, rid in zip(rows, request_ids):
        prediction = row.get("probe_ranking") or row.get("next_layer_prediction") or []
        if not prediction:
            continue
        target_layer = row["layer"] + row["probe_distance"]
        target_row = by_rp.get((rid, row["position"]), {}).get(target_layer)
        if target_row is None:
            missing_target += 1
            continue
        joins.append({
            "rid": rid,
            "position": row["position"],
            "layer": row["layer"],
            "target": target_layer,
            "absent": set(target_row.get("misses", [])),
            "resident_target": set(target_row.get("resident", [])),
            "prediction": prediction,
        })
    return joins, missing_target


def compute_join_stats(joins, top_m_list, num_tokens):
    overall = {}
    per_layer = {}
    for m in top_m_list:
        recall_num = recall_den = 0
        cover_num = cover_den = 0
        prec_num = prec_den = 0
        wasted_num = 0
        layer_acc = defaultdict(lambda: {"recall_num": 0, "recall_den": 0,
                                         "cover_num": 0, "cover_den": 0,
                                         "prec_num": 0, "prec_den": 0})
        for j in joins:
            absent = j["absent"]
            predicted = set(j["prediction"][:m])
            fetched = predicted - j["resident_target"]
            la = layer_acc[j["layer"]]
            prec_num += len(fetched & absent)
            prec_den += len(fetched)
            wasted_num += len(fetched - absent)
            la["prec_num"] += len(fetched & absent)
            la["prec_den"] += len(fetched)
            if absent:
                recall_num += len(absent & predicted)
                recall_den += len(absent)
                cover_den += 1
                la["recall_num"] += len(absent & predicted)
                la["recall_den"] += len(absent)
                la["cover_den"] += 1
                if absent <= predicted:
                    cover_num += 1
                    la["cover_num"] += 1
        overall[m] = {
            "recall": recall_num / recall_den if recall_den else None,
            "precision": prec_num / prec_den if prec_den else None,
            "coverage": cover_num / cover_den if cover_den else None,
            "wasted_per_token": wasted_num / num_tokens if num_tokens else None,
        }
        per_layer[m] = {}
        for layer, la in layer_acc.items():
            per_layer[m][layer] = {
                "recall": la["recall_num"] / la["recall_den"] if la["recall_den"] else None,
                "precision": la["prec_num"] / la["prec_den"] if la["prec_den"] else None,
                "coverage": la["cover_num"] / la["cover_den"] if la["cover_den"] else None,
            }
    return overall, per_layer


def compute_absent_stats(rows):
    sizes = [len(row.get("misses", [])) for row in rows]
    missing = [s for s in sizes if s > 0]
    mean_absent = sum(missing) / len(missing) if missing else 0.0
    histogram = {k: 0 for k in range(1, HISTOGRAM_MAX + 1)}
    for s in missing:
        if 1 <= s <= HISTOGRAM_MAX:
            histogram[s] += 1
    return mean_absent, histogram


def missing_layers_per_token(rows, request_ids):
    by_token = defaultdict(int)
    tokens = set()
    for row, rid in zip(rows, request_ids):
        key = (rid, row["position"])
        tokens.add(key)
        if len(row.get("misses", [])) > 0:
            by_token[key] += 1
    if not tokens:
        return 0.0
    return sum(by_token.get(k, 0) for k in tokens) / len(tokens)


def render_histogram_table(histogram):
    rows = [[size, histogram[size]] for size in range(1, HISTOGRAM_MAX + 1)]
    return render_table(["absent-set size", "count (measured)"], rows)


def run_join(args):
    top_m_list = parse_int_list(args.top_m)
    rows = load_jsonl(args.trace)
    if not rows:
        print(f"join: {args.trace} has no rows")
        return
    request_ids = assign_requests(rows)
    by_rp = index_rows(rows, request_ids)
    joins, missing_target = build_joins(rows, request_ids, by_rp)
    tokens = {(rid, row["position"]) for row, rid in zip(rows, request_ids)}
    probe_distances = sorted({row["probe_distance"] for row in rows})
    mean_absent, histogram = compute_absent_stats(rows)
    mlpt = missing_layers_per_token(rows, request_ids)
    overall, per_layer = compute_join_stats(joins, top_m_list, len(tokens))

    print(f"# join: {args.trace}\n")
    overview_rows = [
        ["requests found (measured)", len(set(request_ids))],
        ["tokens found (measured)", len(tokens)],
        ["probe_distance (measured)", ",".join(str(d) for d in probe_distances)],
        ["joined rows (measured)", len(joins)],
        ["rows with a prediction but no target line (measured)", missing_target],
        ["mean absent-set size over missing layers (measured)", fmt_num(mean_absent)],
        ["missing layers per token, mean (measured)", fmt_num(mlpt)],
    ]
    print(render_table(["metric", "value"], overview_rows))
    print()
    print("## Absent-set size histogram (measured)\n")
    print(render_histogram_table(histogram))
    print()

    print("## Overall, per top-M (measured)\n")
    overall_rows = []
    for m in top_m_list:
        s = overall[m]
        overall_rows.append([m, fmt_ratio(s["recall"]), fmt_ratio(s["precision"]),
                             fmt_ratio(s["coverage"]), fmt_num(s["wasted_per_token"])])
    print(render_table(["M", "per-miss recall", "nonresident precision",
                        "full-layer coverage p", "wasted reads/token"], overall_rows))
    print()

    for m in top_m_list:
        print(f"## Per layer, M={m} (measured)\n")
        layer_rows = []
        for layer in sorted(per_layer[m]):
            s = per_layer[m][layer]
            layer_rows.append([layer, fmt_ratio(s["recall"]), fmt_ratio(s["precision"]),
                               fmt_ratio(s["coverage"])])
        print(render_table(["layer L (predicts L + probe_distance)", "per-miss recall",
                            "nonresident precision", "full-layer coverage p"], layer_rows))
        print()

    print("## Modelled prize, p x --prize-ms (MODELLED)\n")
    prize = {}
    prize_rows = []
    for m in top_m_list:
        p = overall[m]["coverage"]
        prize_ms = p * args.prize_ms if p is not None else None
        decode_ms = None
        tok_s = None
        if prize_ms is not None and args.wall_ms is not None:
            decode_ms = args.wall_ms - prize_ms
            if decode_ms > 0:
                tok_s = 1000.0 / decode_ms
        prize[m] = {"p": p, "prize_ms": prize_ms, "decode_ms": decode_ms, "tok_s": tok_s}
        prize_rows.append([m, fmt_ratio(p), fmt_ms(prize_ms), fmt_ms(decode_ms),
                           fmt_num(tok_s)])
    print(render_table(["M", "p (measured)", "prize_ms (MODELLED)",
                        "decode_ms (MODELLED)", "tok/s (MODELLED)"], prize_rows))

    if args.json:
        out = {
            "mode": "join",
            "trace": args.trace,
            "requests": len(set(request_ids)),
            "tokens": len(tokens),
            "probe_distances": probe_distances,
            "joined_rows": len(joins),
            "rows_missing_target": missing_target,
            "mean_absent_size": mean_absent,
            "missing_layers_per_token": mlpt,
            "absent_histogram": histogram,
            "overall": overall,
            "per_layer": per_layer,
            "prize": prize,
        }
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)


def load_replay_module():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "expert-pool-replay.py")
    spec = importlib.util.spec_from_file_location("expert_pool_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replay_with_hooks(module, lines, slots):
    records = []
    layer_counter = [0]
    current = {"layer": -1, "item": None}

    original_group = module.group_into_chunks

    def counting_group(layer_lines):
        layer = layer_counter[0]
        layer_counter[0] += 1
        for item in original_group(layer_lines):
            current["layer"] = layer
            current["item"] = item
            yield item

    original_make_pool = module.make_pool

    def recording_make_pool(*args, **kwargs):
        pool = original_make_pool(*args, **kwargs)
        inner = pool.plan

        def plan(*pargs, **pkwargs):
            item = current["item"]
            if item is not None and item[0] == "decode":
                _kind, position, experts, label = item
                if label[0] == "request":
                    resident = {e for e in pool.slot_expert if e >= 0}
                    absent = [e for e in experts if e not in resident]
                    records.append({
                        "rid": label[1],
                        "position": position,
                        "layer": current["layer"],
                        "experts": list(experts),
                        "absent": absent,
                    })
            return inner(*pargs, **pkwargs)

        pool.plan = plan
        return pool

    module.group_into_chunks = counting_group
    module.make_pool = recording_make_pool
    try:
        stats, _total_compulsory, _settle_stats, _settle_meta, _profile = module.replay(
            lines, slots, module.parse_policy("aging-lfu"), None,
            module.DEFAULT_AVOID_LOOKBACK, "one", "resident-first", False, None, None,
            protect="chunk", sweep_tail=module.DEFAULT_SWEEP_TAIL, sweep_head_factor=6)
    finally:
        module.group_into_chunks = original_group
        module.make_pool = original_make_pool
    return records, stats


def compute_history_predictors(records, history_n_list):
    demand_by_key = {(r["rid"], r["layer"], r["position"]): set(r["experts"]) for r in records}
    names = ["prev-token"] + [f"last-{n}-union" for n in history_n_list] \
        + ["same-position-prev-layer"]
    acc = {name: {"recall_num": 0, "recall_den": 0, "cover_num": 0, "cover_den": 0}
           for name in names}

    def apply(name, absent, predicted):
        a = acc[name]
        a["recall_num"] += len(absent & predicted)
        a["recall_den"] += len(absent)
        a["cover_den"] += 1
        if absent <= predicted:
            a["cover_num"] += 1

    for r in records:
        absent = set(r["absent"])
        if not absent:
            continue
        rid, layer, position = r["rid"], r["layer"], r["position"]

        prev_key = (rid, layer, position - 1)
        if prev_key in demand_by_key:
            apply("prev-token", absent, demand_by_key[prev_key])

        for n in history_n_list:
            union = set()
            found_any = False
            for k in range(1, n + 1):
                key = (rid, layer, position - k)
                if key in demand_by_key:
                    union |= demand_by_key[key]
                    found_any = True
            if found_any:
                apply(f"last-{n}-union", absent, union)

        same_layer_key = (rid, layer - 1, position)
        if layer > 0 and same_layer_key in demand_by_key:
            apply("same-position-prev-layer", absent, demand_by_key[same_layer_key])

    result = {}
    for name, a in acc.items():
        result[name] = {
            "recall": a["recall_num"] / a["recall_den"] if a["recall_den"] else None,
            "coverage": a["cover_num"] / a["cover_den"] if a["cover_den"] else None,
        }
    return result, names


def compute_history_absent_stats(records):
    sizes = [len(r["absent"]) for r in records]
    missing = [s for s in sizes if s > 0]
    mean_absent = sum(missing) / len(missing) if missing else 0.0
    histogram = {k: 0 for k in range(1, HISTOGRAM_MAX + 1)}
    for s in missing:
        if 1 <= s <= HISTOGRAM_MAX:
            histogram[s] += 1
    return mean_absent, histogram


def history_missing_layers_per_token(records):
    by_token = defaultdict(int)
    tokens = set()
    for r in records:
        key = (r["rid"], r["position"])
        tokens.add(key)
        if len(r["absent"]) > 0:
            by_token[key] += 1
    if not tokens:
        return 0.0
    return sum(by_token.get(k, 0) for k in tokens) / len(tokens)


def run_history(args):
    module = load_replay_module()
    lines = module.load_trace(args.trace)
    history_n_list = parse_int_list(args.history_n)
    records, stats = replay_with_hooks(module, lines, args.slots)

    requests = sorted({r["rid"] for r in records})
    tokens = {(r["rid"], r["position"]) for r in records}
    mean_absent, histogram = compute_history_absent_stats(records)
    mlpt = history_missing_layers_per_token(records)
    predictors, predictor_order = compute_history_predictors(records, history_n_list)
    misses_per_request = {rid: stats[rid]["decode"][1] for rid in requests if rid in stats}

    print(f"# history: {args.trace}\n")
    overview_rows = [
        ["slots (measured, production config)", args.slots],
        ["requests found (measured)", len(requests)],
        ["tokens found (measured)", len(tokens)],
        ["mean absent-set size over missing layers (measured)", fmt_num(mean_absent)],
        ["missing layers per token, mean (measured)", fmt_num(mlpt)],
    ]
    print(render_table(["metric", "value"], overview_rows))
    print()
    print("## Absent-set size histogram (measured)\n")
    print(render_histogram_table(histogram))
    print()

    print("## Replay decode misses per request (measured)\n")
    miss_rows = [[rid, misses_per_request.get(rid, "n/a")] for rid in requests]
    print(render_table(["request", "decode misses"], miss_rows))
    print()

    print("## Router-free predictors, over rows with a non-empty absent set (measured)\n")
    predictor_rows = []
    for name in predictor_order:
        s = predictors[name]
        predictor_rows.append([name, fmt_ratio(s["recall"]), fmt_ratio(s["coverage"])])
    print(render_table(["predictor", "per-miss recall", "full-layer coverage p"],
                       predictor_rows))

    if args.json:
        out = {
            "mode": "history",
            "trace": args.trace,
            "slots": args.slots,
            "requests": len(requests),
            "tokens": len(tokens),
            "mean_absent_size": mean_absent,
            "missing_layers_per_token": mlpt,
            "absent_histogram": histogram,
            "decode_misses_per_request": misses_per_request,
            "predictors": predictors,
        }
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)


def _check(failures, label, actual, expected):
    if actual != expected:
        failures.append(f"{label}: got {actual!r}, expected {expected!r}")


def _self_test_join(failures):
    rows = [
        {"position": 10, "layer": 0, "probe_distance": 1, "experts": [1, 2],
         "misses": [1], "resident": [], "next_layer_prediction": [1, 2, 3, 4]},
        {"position": 10, "layer": 1, "probe_distance": 1, "experts": [2, 5, 3, 9],
         "misses": [2, 5], "resident": [3], "next_layer_prediction": [5, 6, 1, 2]},
        {"position": 10, "layer": 2, "probe_distance": 1, "experts": [6],
         "misses": [6], "resident": [1, 2], "next_layer_prediction": []},
        {"position": 11, "layer": 0, "probe_distance": 1, "experts": [1, 2],
         "misses": [], "resident": [1, 2], "next_layer_prediction": [2, 3, 4, 5]},
        {"position": 11, "layer": 1, "probe_distance": 1, "experts": [2, 3],
         "misses": [], "resident": [2, 3, 7, 8], "next_layer_prediction": [7, 8, 9, 1]},
        {"position": 11, "layer": 2, "probe_distance": 1, "experts": [9],
         "misses": [9], "resident": [7], "next_layer_prediction": []},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "prefetch.jsonl")
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        loaded = load_jsonl(path)

    request_ids = assign_requests(loaded)
    _check(failures, "join: single request", len(set(request_ids)), 1)
    by_rp = index_rows(loaded, request_ids)
    joins, missing_target = build_joins(loaded, request_ids, by_rp)
    _check(failures, "join: 4 joined rows (2 layer-2 rows have no prediction)",
          len(joins), 4)
    _check(failures, "join: no missing targets", missing_target, 0)

    tokens = {(rid, row["position"]) for row, rid in zip(loaded, request_ids)}
    _check(failures, "join: 2 tokens found", len(tokens), 2)

    mean_absent, histogram = compute_absent_stats(loaded)
    _check(failures, "join: mean absent size over missing layers", mean_absent, 1.25)
    _check(failures, "join: absent histogram",
          histogram, {1: 3, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0})
    mlpt = missing_layers_per_token(loaded, request_ids)
    _check(failures, "join: missing layers per token", mlpt, 2.0)

    overall, per_layer = compute_join_stats(joins, [2, 4], len(tokens))
    _check(failures, "join: overall recall@2", overall[2]["recall"], 0.5)
    _check(failures, "join: overall coverage@2", overall[2]["coverage"], 1 / 3)
    _check(failures, "join: overall precision@2", overall[2]["precision"], 0.4)
    _check(failures, "join: overall wasted/token@2", overall[2]["wasted_per_token"], 1.5)
    _check(failures, "join: overall recall@4", overall[4]["recall"], 0.75)
    _check(failures, "join: overall coverage@4", overall[4]["coverage"], 2 / 3)
    _check(failures, "join: overall precision@4", overall[4]["precision"], 0.3)
    _check(failures, "join: overall wasted/token@4", overall[4]["wasted_per_token"], 3.5)

    _check(failures, "join: layer0 recall@2", per_layer[2][0]["recall"], 0.5)
    _check(failures, "join: layer0 coverage@2", per_layer[2][0]["coverage"], 0.0)
    _check(failures, "join: layer0 precision@2", per_layer[2][0]["precision"], 0.5)
    _check(failures, "join: layer0 recall@4", per_layer[4][0]["recall"], 0.5)
    _check(failures, "join: layer0 coverage@4", per_layer[4][0]["coverage"], 0.0)
    _check(failures, "join: layer0 precision@4", per_layer[4][0]["precision"], 0.2)

    _check(failures, "join: layer1 recall@2", per_layer[2][1]["recall"], 0.5)
    _check(failures, "join: layer1 coverage@2", per_layer[2][1]["coverage"], 0.5)
    _check(failures, "join: layer1 precision@2", per_layer[2][1]["precision"], 1 / 3)
    _check(failures, "join: layer1 recall@4", per_layer[4][1]["recall"], 1.0)
    _check(failures, "join: layer1 coverage@4", per_layer[4][1]["coverage"], 1.0)
    _check(failures, "join: layer1 precision@4", per_layer[4][1]["precision"], 0.4)


def _self_test_history(failures):
    module = load_replay_module()
    trace_text = "\n".join([
        "r 0 1",
        "0 0 1 2", "0 1 1 2",
        "p 0 99 0 1 | 1:5",
        "1 0 1 3", "1 1 1 3",
        "2 0 1 3", "2 1 1 3",
        "3 0 2 4", "3 1 2 4",
        "4 0 1 5", "4 1 1 5",
    ])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "route.trace")
        with open(path, "w") as f:
            f.write(trace_text + "\n")
        lines = module.load_trace(path)

    records, stats = replay_with_hooks(module, lines, slots=2)
    _check(failures, "history: 10 decode rows recorded", len(records), 10)
    _check(failures, "history: total decode misses (replay's own count)",
          stats[1]["decode"][1], 14)

    mean_absent, histogram = compute_history_absent_stats(records)
    _check(failures, "history: mean absent size over missing layers", mean_absent, 1.75)
    _check(failures, "history: absent histogram",
          histogram, {1: 2, 2: 6, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0})
    mlpt = history_missing_layers_per_token(records)
    _check(failures, "history: missing layers per token", mlpt, 1.6)

    predictors, _order = compute_history_predictors(records, [2])
    _check(failures, "history: prev-token recall", predictors["prev-token"]["recall"], 0.0)
    _check(failures, "history: prev-token coverage", predictors["prev-token"]["coverage"], 0.0)
    _check(failures, "history: last-2-union recall",
          predictors["last-2-union"]["recall"], 0.2)
    _check(failures, "history: last-2-union coverage",
          predictors["last-2-union"]["coverage"], 0.0)
    _check(failures, "history: same-position-prev-layer recall",
          predictors["same-position-prev-layer"]["recall"], 1.0)
    _check(failures, "history: same-position-prev-layer coverage",
          predictors["same-position-prev-layer"]["coverage"], 1.0)


def _self_test_rankings(failures):
    """A ranking row folds into its plan row when its first entries are the
    row's own top-k; one whose prefix differs is counted and dropped."""
    rows = [
        {"position": 5, "layer": 0, "probe_distance": 1, "experts": [1, 2], "misses": [],
         "resident": [1, 2], "next_layer_prediction": [3, 4]},
        {"position": 5, "layer": 1, "probe_distance": 1, "experts": [3, 4], "misses": [],
         "resident": [3, 4], "next_layer_prediction": [5, 6]},
        {"position": 5, "layer": 0, "probe_ranking": [3, 4, 7, 8]},
        {"position": 5, "layer": 1, "probe_ranking": [6, 5, 9, 1]},
    ]
    RANKING_MISMATCHES["rows"] = 0
    RANKING_MISMATCHES["mismatched"] = 0
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "capture.jsonl")
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        loaded = load_jsonl(path)
    _check(failures, "rankings: plan rows kept", len(loaded), 2)
    _check(failures, "rankings: matching prefix folded", loaded[0].get("probe_ranking"), [3, 4, 7, 8])
    _check(failures, "rankings: mismatched prefix dropped", loaded[1].get("probe_ranking"), None)
    _check(failures, "rankings: counts", dict(RANKING_MISMATCHES), {"rows": 2, "mismatched": 1})
    RANKING_MISMATCHES["rows"] = 0
    RANKING_MISMATCHES["mismatched"] = 0


def self_test():
    failures = []
    _self_test_join(failures)
    _self_test_history(failures)
    _self_test_rankings(failures)
    if failures:
        print(f"SELF-TEST FAILED ({len(failures)} of many checks):")
        for f in failures:
            print(f"  {f}")
        return 1
    print("self-test: all checks passed")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Price a next-layer expert prefetch offline against a "
                    "SHRIKE_PREFETCH_TRACE or SHRIKE_ROUTE_TRACE capture.")
    parser.add_argument("--self-test", action="store_true",
                        help="run the built-in synthetic-capture checks and exit")
    sub = parser.add_subparsers(dest="mode")

    join_p = sub.add_parser(
        "join", help="join a SHRIKE_PREFETCH_TRACE capture's own next-layer "
                     "prediction against its own recorded misses")
    join_p.add_argument("trace", help="path to a SHRIKE_PREFETCH_TRACE JSONL capture")
    join_p.add_argument("--top-m", default=DEFAULT_TOP_M,
                        help=f"comma-separated prefix lengths of next_layer_prediction "
                             f"to evaluate (default {DEFAULT_TOP_M})")
    join_p.add_argument("--prize-ms", type=float, default=DEFAULT_PRIZE_MS,
                        help="the single-layer perfect-prediction saving per full-layer "
                             f"coverage hit, ms per token (default {DEFAULT_PRIZE_MS}, "
                             "the design doc's card-shape figure); the resulting prize "
                             "is MODELLED, not measured")
    join_p.add_argument("--wall-ms", type=float, default=None,
                        help="measured wall ms per token, to report a modelled "
                             "decode ms/tok and tok/s alongside the prize (default: "
                             "omit those columns)")
    join_p.add_argument("--json", default=None,
                        help="write the same numbers as JSON to this path")

    hist_p = sub.add_parser(
        "history", help="replay a SHRIKE_ROUTE_TRACE capture and evaluate "
                        "router-free predictors of each layer's absent set")
    hist_p.add_argument("trace", help="path to a SHRIKE_ROUTE_TRACE capture")
    hist_p.add_argument("--history-n", default=DEFAULT_HISTORY_N,
                        help="comma-separated window sizes for the last-n-union "
                             f"predictor (default {DEFAULT_HISTORY_N})")
    hist_p.add_argument("--slots", type=int, default=DEFAULT_SLOTS,
                        help=f"slots per layer (default {DEFAULT_SLOTS}, production)")
    hist_p.add_argument("--json", default=None,
                        help="write the same numbers as JSON to this path")

    args = parser.parse_args()

    if args.self_test:
        sys.exit(self_test())

    if not args.mode:
        parser.error("a mode (join or history) is required unless --self-test is given")

    if args.mode == "join":
        run_join(args)
    else:
        run_history(args)


if __name__ == "__main__":
    main()
