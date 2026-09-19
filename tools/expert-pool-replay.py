#!/usr/bin/env python3
"""Replay a SHRIKE_ROUTE_TRACE capture against an expert-pool eviction policy.

A trace is one line per plan the pool actually received: a decode layer is
bare `position layer e0 e1 ...`, a prefill tile is marked `p position layer
tile e0 e1 ... [| n0 n1 ...]` or `[| n0:l0 n1:l1 ...]` (nK is eK's row count
in this tile, lK its highest row index in the chunk; either suffix form, or
none, parses, oldest captures having neither and older ones having counts
without last rows), and a request's start is marked `r cachedTokens
promptTokens` (see `RealForwardRunner.formatRouteTraceLine`). This replays
the trace against a chosen policy over a per-layer pool of a fixed slot
count and reports hits / misses per phase per request, split into
compulsory (the first time this layer ever requests that expert, cache
size aside) and capacity (a re-miss of an expert this layer has requested
before) misses; compulsory is therefore the same for every policy on the
same trace, which is the cross-check that the split is wired correctly.

Policies (`--policy`): lru, lfu, aging-lfu (optionally `aging-lfu:<period>`,
default 1024), belady (the clairvoyant optimal, using the whole trace's
future), slru:<protected_share> (Segmented LRU, Karedla/Love/Wherry 1994:
two LRU segments, probationary and protected; a probationary hit promotes,
protected demotes its LRU entry to probationary once it exceeds its share
of the slots; default share 0.5), arc (Adaptive Replacement Cache, Megiddo
& Modha, FAST 2003: resident LRU lists T1/T2 for recency/frequency plus
ghost lists B1/B2 of recently evicted keys, adapting a target size p from
ghost hits), lru-2 (LRU-K with K=2, O'Neil/O'Neil/Weikum 1993: evicts the
resident whose second-most-recent reference is oldest; fewer than 2
references is evicted first).

Fidelity, each checked against PreadExpertStreamer.swift before being coded
here:
  - hits are reserved before victim selection (:686), so a hit can never
    also be chosen as a miss's victim within the same plan.
  - a loading or pinned slot is never an eviction candidate (:1156-1157);
    this replay has no in-flight state (it is a synchronous, single-pass
    replay of a trace already ordered as the pool received it), so the
    only ineligible slots are the plan's own hit-reserved set and, for a
    prefill tile, the modelled `avoidingSlots` below.
  - victims are ranked by (count, oldest slotLastUse, lower slot index) for
    the lfu family and by (slotLastUse, lower slot index) for lru
    (:1179-1192); an empty (never-loaded) slot always sorts first because
    its `expertUseCount` is treated as -1, matching `shouldEvictSlot`'s
    direct `lhsExpert < rhsExpert` branch when either side is empty.
  - `expertUseCount` is incremented once per expert per plan and never
    reset (:698-699, :514); `--prefill-weight rows` increments it by the
    expert's row count on a prefill plan instead of 1 (decode plans are
    unchanged either way) -- errors out if the trace has no row counts.
  - the aging-lfu halving (`count >>= 1` for every expert) fires when the
    layer's completed-plan count is a positive multiple of the period,
    checked before the halved plan's own hit/miss accounting (:664-670).
  - one plan per tile in prefill, one plan per layer per token in decode.
  - a prefill tile's `avoidingSlots` is the held slots of the open and
    pending batches at fetch depth 2. With production's constants (the
    replay's tile depth is production's maxPendingDepth 2, its tile batch
    production's 1 tile per batch, its fetch depth production's
    fetchLookahead 1), tracing the
    lookahead's own avoidingSlots union shows the held set reaches
    `maxPendingDepth + 1` = 3 preceding tiles at steady state. `--avoid-
    lookback` exposes this (default 3). A chunk boundary (a tile index
    that is not its predecessor's successor, within one layer's own
    sequence) resets the lookback window, since a new `executePrefillChunk`
    call starts with empty `pendingBatches` / `openBatch` state. If the
    modelled avoidingSlots would leave too few eligible slots to place a
    plan, it is dropped for that one plan.
  - a request's span is delimited by `r` lines when the trace has any: `r`
    line N opens request N, and request N runs until the next `r` line (or
    end of trace). Within that span, `p` lines before the first bare line
    are the request's own prefill; a bare line marks its decode has begun;
    any `p` line after that (before the next `r`) is the prompt cache's
    settle re-prefill running between requests, which mutates the pool but
    is reported as its own phase, not folded into either request's count.
    When the trace has no `r` line at all (a capture predating this
    marker), request boundaries fall back to the position-based heuristic
    below, and no settle separation is attempted (a settle chunk in such a
    trace is folded into the following request, as measured).
  - `--sweep-order rows-asc|rows-desc` re-tiles a prefill chunk offline
    from its (expert, rows) set (its original tiles concatenated, in file
    order), sorted by row count, into new tiles of 8; `rows-asc` puts the
    prompt's hottest (highest-row) experts in the chunk's last tile.
    `last-asc|last-desc` instead sort by each expert's own last row in the
    chunk (ties by rows ascending then expert id, kept ascending regardless
    of direction): the recency-ordered sweep, so the pool leaves prefill
    holding whichever experts the prompt's own final tokens routed to,
    approximating the ideal post-sweep state without needing decode's own
    future. `--sweep-carry on` alternates the sort direction chunk to
    chunk within a layer (matching the resident sweep's per-chunk carry,
    whose flip never resets across a request boundary); `off` (default)
    uses the same direction for every chunk. This re-tiling cannot
    reproduce the real scheduler's own tile composition (which follows the
    routed groups' natural order and packs by slot-budget fit, not a fixed
    width of 8) or its `avoidingSlots` (recomputed here from the new,
    synthetic tile boundaries, not the real batch/commit schedule).
  - `resident` is production's only sweep order since v13 T5 step 2 (the
    replay keeps `carry` for comparison). `--sweep-order resident-first`
    re-tiles each chunk from the pool's own state at the moment the chunk
    begins (per layer): the chunk's experts ranked `last-asc` (ties by
    rows ascending then expert id),
    split into a resident group (present in the pool, rank order kept)
    and an absent group (rank order kept). When either group is empty
    (the cold-pool case, or a chunk the pool already holds in full), this
    defers to `resident-first-grouped` at its own default tail
    (`DEFAULT_SWEEP_TAIL`, not this call's own `--sweep-tail`, which
    governs only `resident-first-grouped` itself): with the resident
    group empty, that composition's head/tail split runs over the whole
    ranked list, degenerating to one packed group when the absent count
    is within the default tail (matching the Swift `recencyBalanced`'s
    own order in that case, so the first-turn prize on a small chunk is
    unchanged) and splitting head/tail as usual once it is not (a large
    cold chunk, where the split still holds -- verified against the
    replayed acceptance traces, not asserted as an identity). Otherwise:
    with `T = ceil((R + A) / 8)` tiles, the smallest head length `h` tiles
    (0 <= h < T) is found such that `R - 8h <= slots - sweep_head_factor
    * ceil(A / (T - h))` -- the free slots left after the head cover
    `--sweep-head-factor` tiles' worth of misses for however many mixed
    tiles remain, so protection cannot starve; the first `8h` residents
    (in rank order) form a pure-resident head. The remaining `M = T - h`
    tiles each receive a contiguous, uniform slice of the absent group in
    rank order (tile `j` gets ranks `floor(jA/M)..floor((j+1)A/M)-1`, so
    the last tile holds the most recent absent experts), then the
    leftover residents fill the tiles' free slots heaviest-first the same
    way `_pack_by_rows` does, into the tile with the lowest total weight
    that still has room, ties to the lower tile index. The head residents
    and the tiles' experts (absent then residents, in placement order)
    concatenate and re-tile flat in runs of 8, so the chunk's tile count
    matches `index`'s exactly. This head rule assumes the production tile
    width of 8 (`RETILE_SIZE` against the Swift's `schedulerConfig.tileExperts`,
    which the fitting step can shrink below 8 on a tight slot budget). This
    is what the resident sweep's per-chunk carry
    approximates by alternating direction chunk to chunk, made exact
    through the pool's residency, with every layer's misses spread across
    its tiles instead of collected in the chunk's last ones. `--sweep-
    order resident-first-grouped` is round 1's landed order this
    composition replaces: resident, head and tail (the absent group's
    most recent `--sweep-tail` experts, default 96) each packed by row
    weight into their own bins, concatenated resident-head-tail and
    re-tiled flat -- kept for round 1's own rows. `--sweep-order
    resident-first-plain` is the step-zero order `resident-first-grouped`
    replaced: resident group then absent group, each in `last-asc` order
    with no row-weight packing, tiled flat in runs of 8 across the
    concatenation. `--sweep-carry` has no effect on any of the three
    orders (residency read off the pool's live `slot_expert` when the
    chunk's first tile is planned).
  - `--protect chunk` (default, matching production's chunk protection,
    always on since v13 T4): while a chunk's tiles are replayed, a slot holding an
    expert one of the chunk's not-yet-replayed tiles still needs is
    ineligible as a victim, in addition to `avoiding`. Modelled the same
    way `RealForwardRunner.PrefillChunkExpertProtection` maintains it: the
    chunk's `remaining` set starts as the union of every tile's own
    experts, each tile's own experts are removed from it immediately before
    that tile is planned (never protecting a tile against itself), and the
    protected slot set is read off the pool's live `slot_expert` at plan
    time. The graded fallback matches the streamer's `selectVictimSlots`:
    a plan that cannot place every miss while both `protect` and `avoiding`
    apply retries with `protect` dropped (keeping `avoiding`) before this
    tool's own pre-existing avoiding-only fallback. `--protect off` replays
    a capture taken before the knob existed, or prices the counterfactual.
  - a capture's own `p`-line count for a request must equal that request's
    routed tile count, or the capture is missing a plan the pool actually
    made (fix round 1: a starved primary plan in `resolveTileFetchBegin` /
    the depth-1 loop used to fall back to a second, untraced and
    unprotected plan inside `beginFetchForTile`; the trace line now emits
    once per tile after the fetch is resolved, whichever path planned it,
    and that second plan now receives the same `protectedExperts` the
    primary attempt would have). This tool has no way to detect a missing
    line from the trace alone; verify it against the request's own routed
    tile count when auditing a new capture.

Usage:
  expert-pool-replay.py <trace> --policy lru|lfu|aging-lfu[:period]|belady|
                         slru[:share]|arc|lru-2
                         [--slots N] [--layer L] [--avoid-lookback N]
                         [--prefill-weight one|rows]
                         [--sweep-order index|rows-asc|rows-desc|last-asc|last-desc|
                                        resident-first|resident-first-grouped|
                                        resident-first-plain]
                         [--sweep-tail K]
                         [--sweep-head-factor F]
                         [--sweep-carry off|on]
                         [--protect off|chunk]
                         [--phase-policy prefill=<policy>,decode=<policy>]
                         [--profile WINDOW]
  expert-pool-replay.py <trace> --policy ... --expect <file>
  expert-pool-replay.py --self-test
"""
import argparse
import bisect
import json
import sys
from collections import OrderedDict, defaultdict, deque

DEFAULT_SLOTS = 128
DEFAULT_AGING_PERIOD = 1024
# production's maxPendingDepth (2) + 1, the steady-state
# held-tile count at 1 tile per batch and fetch depth 2 (see module docstring).
DEFAULT_AVOID_LOOKBACK = 3
DEFAULT_SLRU_PROTECTED_SHARE = 0.5
# The production tile width; the Swift's fitting() narrows tileExperts below 8 on
# a small cache, which no re-tiling order here models.
RETILE_SIZE = 8
DEFAULT_SWEEP_TAIL = 96
DEFAULT_SWEEP_HEAD_FACTOR = 6


def parse_line(raw):
    """One trace line -> (kind, position, layer, tile, experts, row_counts,
    last_rows).

    kind is "r" (a request's start), "p" (one prefill tile) or "decode" (one
    decode layer). An "r" line has no layer or tile of its own; `position`
    holds cachedTokens and `layer` holds promptTokens for that line only.
    `row_counts` is the `| n0 n1 ...` suffix on a "p" line, or None when the
    line has no suffix (an older capture, or any non-"p" line). Each token
    in the suffix is either a bare count or `count:lastRow`; `last_rows` is
    the parsed lastRow list, or None when no token carries one (an older
    capture with counts but no last-use)."""
    parts = raw.split()
    if not parts:
        return None
    if parts[0] == "r":
        cached_tokens, prompt_tokens = int(parts[1]), int(parts[2])
        return "r", cached_tokens, prompt_tokens, None, [], None, None
    if parts[0] == "p":
        position, layer, tile = int(parts[1]), int(parts[2]), int(parts[3])
        rest = parts[4:]
        if "|" in rest:
            sep = rest.index("|")
            experts = [int(x) for x in rest[:sep]]
            row_counts = []
            last_rows = []
            for token in rest[sep + 1:]:
                fields = token.split(":")
                row_counts.append(int(fields[0]))
                last_rows.append(int(fields[1]) if len(fields) > 1 else None)
            if all(last_row is None for last_row in last_rows):
                last_rows = None
        else:
            experts = [int(x) for x in rest]
            row_counts = None
            last_rows = None
        return "p", position, layer, tile, experts, row_counts, last_rows
    if parts[0] == "q":
        # v20 S0.5: a prefill position's own top-k at one layer
        # (`q position layer e0 ...`), beside the tile's `p` line. Not a plan.
        position, layer = int(parts[1]), int(parts[2])
        experts = [int(x) for x in parts[3:]]
        return "q", position, layer, None, experts, None, None
    if parts[0] == "t":
        # v20 S0.5: the input token id of a decode position (`t position id`).
        return "t", int(parts[1]), None, None, [int(parts[2])], None, None
    position, layer = int(parts[0]), int(parts[1])
    experts = [int(x) for x in parts[2:]]
    return "decode", position, layer, None, experts, None, None


AUX_KINDS = ("q", "t")


def load_trace(path, keep_aux=False):
    """The trace's plan lines. The v20 auxiliary kinds (`q`, `t`) are not
    plans and are dropped unless `keep_aux`, so the request segmentation and
    the pool see the same lines the pool received."""
    lines = []
    with open(path) as f:
        for raw in f:
            parsed = parse_line(raw)
            if parsed is None:
                continue
            if parsed[0] in AUX_KINDS and not keep_aux:
                continue
            lines.append(parsed)
    return lines


def has_request_markers(lines):
    return any(kind == "r" for kind, *_ in lines)


def has_row_counts(lines):
    return any(row_counts for kind, _p, _l, _t, _e, row_counts, _lr in lines if kind == "p")


def has_last_rows(lines):
    return any(last_rows for kind, _p, _l, _t, _e, _rc, last_rows in lines if kind == "p")


def segment_requests(lines):
    """Assigns a 1-based request id to each parsed line: a `p` line that
    follows a non-prefill line opens a new request, and the first line
    always opens request 1. Used only when the trace has no `r` line.

    A decode-only capture predating this trace's prefill line has no `p`
    line to mark a boundary with, so a bare line whose position is not its
    predecessor's immediate successor (nor a repeat of it, for the other 39
    layers of the same token) also opens a new request. This heuristic does
    not separate a settle chunk from the request it precedes."""
    request_ids = []
    request = 0
    prev_is_prefill = None
    last_decode_position = None
    for kind, position, *_ in lines:
        is_prefill = kind == "p"
        new_request = False
        if prev_is_prefill is None:
            new_request = True
        elif is_prefill and not prev_is_prefill:
            new_request = True
        elif not is_prefill and not prev_is_prefill and last_decode_position is not None:
            if position not in (last_decode_position, last_decode_position + 1):
                new_request = True
        if new_request:
            request += 1
        if not is_prefill:
            last_decode_position = position
        request_ids.append(request)
        prev_is_prefill = is_prefill
    return request_ids


def label_lines_by_marker(lines):
    """Used only when the trace has at least one `r` line. Returns
    (labels, settle_meta): labels[i] is ("marker",) for an `r` line itself
    (not replayed), ("request", request_id, "prefill"|"decode") for a line
    belonging to a request's own work, or ("settle", chunk_id) for a `p`
    line that runs after that request's decode has begun and before the
    next `r` line. settle_meta[chunk_id] = {"first_position", "tile_count"}.

    A chunk boundary (used to tell one settle chunk from the next, and from
    the request's own prefill) is a `p` line whose (layer, tile) does not
    continue the previous `p` line's: the layer dropped back down (a new
    `executePrefillChunk` call starting again at layer 0), or the same
    layer's tile index did not advance by exactly one."""
    labels = []
    settle_meta = {}
    request_id = 0
    decode_seen = False
    prev_chunk_key = None
    settle_chunk_id = -1
    for kind, position, layer, tile, _experts, _row_counts, _last_rows in lines:
        if kind == "r":
            request_id += 1
            decode_seen = False
            prev_chunk_key = None
            labels.append(("marker",))
            continue
        if kind == "p":
            is_new_chunk = (prev_chunk_key is None
                            or layer < prev_chunk_key[0]
                            or (layer == prev_chunk_key[0] and tile != prev_chunk_key[1] + 1))
            prev_chunk_key = (layer, tile)
            if decode_seen:
                if is_new_chunk:
                    settle_chunk_id += 1
                    settle_meta[settle_chunk_id] = {"first_position": position, "tile_count": 0}
                settle_meta[settle_chunk_id]["tile_count"] += 1
                labels.append(("settle", settle_chunk_id))
            else:
                labels.append(("request", request_id, "prefill"))
        else:
            decode_seen = True
            labels.append(("request", request_id, "decode"))
    return labels, settle_meta


def group_into_chunks(layer_lines):
    """layer_lines: (kind, position, tile, experts, row_counts, last_rows,
    label). Returns items: ("decode", position, experts, label), or
    ("chunk", tiles, label) where `tiles` is a list of tiles, each a list
    of (expert, rows, last_row) triples in the tile's recorded order (rows
    defaults to 1 and last_row to None when the trace carries neither). A
    chunk boundary is the same tile-discontinuity rule as
    `label_lines_by_marker`, applied within this one layer's own
    sequence."""
    items = []
    current_label = None
    current_tiles = None
    prev_tile = None
    for kind, position, tile, experts, row_counts, last_rows, label in layer_lines:
        if kind == "p":
            is_new_chunk = prev_tile is None or tile != prev_tile + 1
            if is_new_chunk:
                if current_tiles is not None:
                    items.append(("chunk", current_tiles, current_label))
                current_tiles = []
                current_label = label
            rows = row_counts if row_counts is not None else [1] * len(experts)
            lasts = last_rows if last_rows is not None else [None] * len(experts)
            current_tiles.append(list(zip(experts, rows, lasts)))
            prev_tile = tile
        else:
            if current_tiles is not None:
                items.append(("chunk", current_tiles, current_label))
                current_tiles = None
            prev_tile = None
            items.append(("decode", position, experts, label))
    if current_tiles is not None:
        items.append(("chunk", current_tiles, current_label))
    return items


class Policy:
    def __init__(self, name, param=None):
        self.name = name
        self.param = param

    def label(self):
        if self.name == "aging-lfu":
            return f"aging-lfu:{self.param}"
        if self.name == "slru":
            return f"slru:{self.param}"
        return self.name


def parse_policy(raw):
    if ":" in raw:
        name, _, param_raw = raw.partition(":")
    else:
        name, param_raw = raw, None
    if name == "aging-lfu":
        return Policy(name, int(param_raw) if param_raw else DEFAULT_AGING_PERIOD)
    if name == "slru":
        return Policy(name, float(param_raw) if param_raw else DEFAULT_SLRU_PROTECTED_SHARE)
    return Policy(name, param_raw)


PHASE_POLICY_ALLOWED = {"lru", "lfu", "aging-lfu", "belady"}


def parse_phase_policy(raw):
    parts = {}
    for item in raw.split(","):
        key, _, value = item.partition("=")
        parts[key] = value
    if set(parts) != {"prefill", "decode"}:
        raise ValueError("--phase-policy needs exactly 'prefill=...,decode=...'")
    result = {}
    for phase, raw_policy in parts.items():
        policy = parse_policy(raw_policy)
        if policy.name not in PHASE_POLICY_ALLOWED:
            raise ValueError(
                f"--phase-policy only supports {sorted(PHASE_POLICY_ALLOWED)}, "
                f"got '{policy.name}' for {phase}")
        result[phase] = policy
    return result


def load_prefetch_fills(path, top_m):
    """(target layer, position) -> the router probe's top-M prediction for that
    layer, from a SHRIKE_PREFETCH_TRACE capture (one JSON line per decode plan;
    the prediction at (position, L) names layer L + probe_distance). This is
    what v15 Task 2's speculative landing would fill into the target's pool
    before the target's own plan at that position."""
    fills = {}
    with open(path) as handle:
        for raw in handle:
            row = json.loads(raw)
            prediction = row.get("next_layer_prediction") or []
            if not prediction:
                continue
            fills[(row["layer"] + row["probe_distance"], row["position"])] = prediction[:top_m]
    return fills


def load_two_distance_queue(path_d1, path_d2, top_m, chained=False):
    """The two-distance queue (v15's scheduled step zero): the window that opens
    at layer L (its demand batch's completion or its all-hit plan) serves the
    router probe's prediction for L + 1 (the d1 capture's line at (position, L))
    when that prediction names an expert absent from L + 1's resident set (the
    capture's own), else the prediction for L + 2 (the d2 capture's line at the
    same (position, L)). Returns (target layer, position) -> candidates: the d1
    prediction first, then the d2 prediction if the window two layers back was
    free for it; `chained` lets every window serve both in sequence (the second
    read still has two layers of lead)."""
    d1 = {}
    for raw in open(path_d1):
        row = json.loads(raw)
        d1[(row["position"], row["layer"])] = row
    d2 = {}
    for raw in open(path_d2):
        row = json.loads(raw)
        d2[(row["position"], row["layer"])] = row
    fills = {}
    for (position, layer), row in d1.items():
        prediction = (row.get("next_layer_prediction") or [])[:top_m]
        target = layer + row["probe_distance"]
        if prediction:
            fills.setdefault((target, position), []).extend(prediction)
        # The window at L is free for L + 2 when L + 1's prediction is all resident.
        next_row = d1.get((position, target))
        window_free = chained or not prediction or (
            next_row is not None and not (set(prediction) - set(next_row.get("resident", []))))
        row2 = d2.get((position, layer))
        prediction2 = (row2.get("next_layer_prediction") or [])[:top_m] if row2 else []
        if window_free and prediction2:
            fills.setdefault((layer + row2["probe_distance"], position), []).extend(prediction2)
    return fills


def parse_layer_set(spec, num_layers=40):
    """`0,30-39` -> the set {0, 30, ..., 39}; `all` -> every layer."""
    if spec is None or spec.strip() == "all":
        return frozenset(range(num_layers))
    layers = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = part.split("-", 1)
            layers.update(range(int(low), int(high) + 1))
        else:
            layers.add(int(part))
    return frozenset(layers)


def first_request_decode(lines):
    """(ordered decode positions, {(layer, position): experts}) of the trace's
    first request: the lines from its `r` marker to the next one, or the
    first segment of the position heuristic when the trace has no marker."""
    if has_request_markers(lines):
        started = False
        chosen = []
        for line in lines:
            if line[0] == "r":
                if started:
                    break
                started = True
                continue
            if started:
                chosen.append(line)
    else:
        request_ids = segment_requests(lines)
        chosen = [line for line, rid in zip(lines, request_ids) if rid == 1]
    order = []
    seen = set()
    routes = {}
    for kind, position, layer, _tile, experts, _rc, _lr in chosen:
        if kind != "decode":
            continue
        if position not in seen:
            seen.add(position)
            order.append(position)
        routes[(layer, position)] = list(experts)
    return order, routes


def table_prediction(history, source, width):
    """The table's entry as a list, most recent first, capped at `width`."""
    if not history or source == "none":
        return []
    if source == "last":
        return list(history[-1])[:width]
    if source in ("last2", "last3"):
        depth = 2 if source == "last2" else 3
        merged = []
        seen = set()
        for route in reversed(history[-depth:]):
            for expert in route:
                if expert not in seen:
                    seen.add(expert)
                    merged.append(expert)
        return merged[:width]
    if source == "freq":
        counts = defaultdict(int)
        first_seen = {}
        for index, route in enumerate(reversed(history)):
            for rank, expert in enumerate(route):
                counts[expert] += 1
                first_seen.setdefault(expert, (index, rank))
        ranked = sorted(counts, key=lambda e: (-counts[e], first_seen[e]))
        return ranked[:width]
    raise ValueError(f"unknown table source {source}")


def build_table_fills(lines, pieces, layers, width=8, source="last",
                      union_previous=False, draft_n=0, draft_layers=frozenset(),
                      prompt_pieces=None, seed_routes=None, info=None):
    """v20 S0.1: the token-id table as prefetch fills, computed from the
    trace's own first request. `pieces` is that request's streamed token
    text per decode position (the input token of the pass at that position,
    the s02 alignment: the streamed pieces and the decode positions are
    equal in count and matched by index); the table is keyed by the piece,
    a stand-in for the id until a capture carries `t` lines.

    Per served layer L, the prediction at position i is the table's entry for
    pieces[i] built from the earlier positions only (`source`: the last
    occurrence's route, the union of the last two or three, or the most
    frequent), plus, with `union_previous`, the route at position i-1 at the
    same layer (this token's own route, known a pass ahead like the id is);
    capped at `width`. For a layer in `draft_layers` the key is instead the
    prompt-lookup draft for pieces[i]: the piece that followed the most
    recent earlier occurrence of the `draft_n`-gram ending at pieces[i-1],
    over `prompt_pieces` plus the answer so far; no draft, no fill.
    `seed_routes` ({(layer, prompt index): experts}, from a capture's `q`
    lines) pre-fills the table from the prompt when `prompt_pieces` is given.
    Returns (layer, position) -> prediction; `info`, when a dict, receives
    the counts (positions, issued and covered per layer, the draft's
    proposals and hits)."""
    order, routes = first_request_decode(lines)
    if len(order) != len(pieces):
        raise ValueError(
            f"the trace's first request has {len(order)} decode positions and the token "
            f"stream {len(pieces)} pieces; they must match one to one")
    history = {layer: defaultdict(list) for layer in layers}
    if seed_routes and prompt_pieces:
        for (layer, index), experts in sorted(seed_routes.items(), key=lambda kv: kv[0][1]):
            if layer in history and index < len(prompt_pieces):
                history[layer][prompt_pieces[index]].append(list(experts))
    context = list(prompt_pieces or [])
    # n-gram -> the end (exclusive) of its most recent occurrence; the
    # n-gram ending at the prompt's last piece is registered at position 0's
    # lookup, not before it, so no lookup can find itself.
    ngram_last_end = {}
    if draft_n > 0:
        for end in range(draft_n, len(context)):
            ngram_last_end[tuple(context[end - draft_n:end])] = end
    fills = {}
    issued = defaultdict(int)
    covered = defaultdict(int)
    predicted_total = defaultdict(int)
    proposals = 0
    draft_hits = 0
    prompt_len = len(context)
    for i, position in enumerate(order):
        piece = pieces[i]
        draft = None
        if draft_n > 0 and prompt_len + i >= draft_n:
            end = prompt_len + i
            key = tuple(context[end - draft_n:end])
            follower_end = ngram_last_end.get(key)
            if follower_end is not None:
                draft = context[follower_end]
                proposals += 1
                if draft == piece:
                    draft_hits += 1
            ngram_last_end[key] = end
        for layer in layers:
            key = draft if layer in draft_layers else piece
            prediction = []
            if key is not None:
                prediction = table_prediction(history[layer].get(key), source, width)
                if prediction:
                    covered[layer] += 1
            if union_previous and i > 0:
                previous = routes.get((layer, order[i - 1]), [])
                for expert in previous:
                    if expert not in prediction:
                        prediction.append(expert)
                prediction = prediction[:width]
            if prediction:
                fills[(layer, position)] = prediction
                issued[layer] += 1
                predicted_total[layer] += len(prediction)
        for layer in layers:
            route = routes.get((layer, position))
            if route is not None:
                history[layer][piece].append(route)
        context.append(piece)
    if info is not None:
        info["positions"] = len(order)
        info["issued"] = dict(issued)
        info["covered"] = dict(covered)
        info["predicted"] = dict(predicted_total)
        info["draft"] = {"n": draft_n, "proposals": proposals, "hits": draft_hits}
    return fills


def load_pieces(path):
    """The streamed pieces of a rig token file (`tokens-*.json`), one per
    decode position, or the `pieces` of a `--tokenize` file."""
    with open(path) as handle:
        body = json.load(handle)
    if "pieces" in body:
        return list(body["pieces"])
    return [row[2] for row in body["tokens"]]


def load_seed_routes(path):
    """{(layer, prompt index): experts} from a capture's `q` lines."""
    seeds = {}
    for line in load_trace(path, keep_aux=True):
        if line[0] == "q":
            _kind, position, layer, _tile, experts, _rc, _lr = line
            seeds[(layer, position)] = list(experts)
    return seeds


LAYER_GROUPS = (("0-3", range(0, 4)), ("4-9", range(4, 10)), ("10-19", range(10, 20)),
                ("20-29", range(20, 30)), ("30-39", range(30, 40)), ("all", range(0, 40)))


def print_table_report(info, fill_stats, layers, request_id=1):
    positions = max(info.get("positions", 0), 1)
    per_layer = fill_stats.get("per_layer", {})
    misses = fill_stats.get("misses_per_layer", {})
    per_position = fill_stats.get("per_position", {})
    served = sorted(layers)
    print(f"  table fills: positions={positions} layers={len(served)} "
          f"(served {served[0]}..{served[-1]})" if served else "  table fills: no layers")
    draft = info.get("draft") or {}
    if draft.get("n"):
        rate = draft["hits"] / draft["proposals"] if draft["proposals"] else 0.0
        print(f"  draft prompt-lookup:{draft['n']}: proposals={draft['proposals']} "
              f"({draft['proposals'] / positions:.2f} per position) hits={draft['hits']} "
              f"(rate {rate:.3f})")
    print("  group    issued/pos  fills/pos  useful/pos  wasted/pos  misses/pos  reads/pos")
    for name, group in LAYER_GROUPS:
        members = [layer for layer in group if layer in layers] if name != "all" else list(group)
        issued = sum(info.get("issued", {}).get(layer, 0) for layer in members)
        fills = sum(per_layer.get(layer, (0, 0, 0))[0] for layer in members)
        useful = sum(per_layer.get(layer, (0, 0, 0))[1] for layer in members)
        wasted = sum(per_layer.get(layer, (0, 0, 0))[2] for layer in members)
        missed = sum(misses.get((request_id, layer), 0) for layer in members)
        print(f"  {name:<8} {issued / positions:>10.2f} {fills / positions:>10.2f} "
              f"{useful / positions:>11.2f} {wasted / positions:>11.2f} "
              f"{missed / positions:>11.2f} {(fills + missed) / positions:>10.2f}")
    if per_position:
        values = list(per_position.values())
        print(f"  cells at the pass start: mean={sum(values) / positions:.2f} "
              f"max={max(values)} (fills placed per position across the served layers)")


def build_future_occurrences(expert_lists):
    """expert -> sorted list of plan indices (within this layer's own
    sequence) at which it is requested; used only by belady."""
    occurrences = defaultdict(list)
    for index, experts in enumerate(expert_lists):
        for expert in experts:
            occurrences[expert].append(index)
    return occurrences


class LayerPool:
    """One layer's slot cache, replayed from a cold start. Serves lru, lfu,
    aging-lfu and belady; `policy_override` lets one call use a different
    policy than `self.policy` (for `--phase-policy`)."""

    def __init__(self, slots, policy, future_occurrences=None):
        self.slots = slots
        self.policy = policy
        self.future_occurrences = future_occurrences or {}
        self.slot_expert = [-1] * slots
        self.slot_last_use = [0] * slots
        self.use_clock = 0
        self.expert_use_count = defaultdict(int)
        self.seen_experts = set()
        self.plans_done = 0
        self.compulsory = 0
        self.capacity = 0
        self.last_assigned = set()
        self.filled_unused = set()
        self.fills = 0
        self.useful_fills = 0
        self.wasted_fills = 0
        # `pool`: a fill lands in a victim slot and stays (v16's landing in
        # the pool's own slot). `ring`: a fill lives beside the pool until its
        # layer's plan, counts a hit there and is then gone (the ring's cells
        # addressable by the classifier, no victim, nothing retained).
        # `ring-retain`: the same, but a hit expert is then placed in a victim
        # slot, which was production before v16's merge (the ring with
        # adoption) and is the merge's own profile (the swap at the plan).
        self.fill_mode = "pool"
        self.ring = set()

    def fill(self, candidates, budget):
        """Places up to `budget` predicted experts that are not resident into
        victim slots chosen by the policy, resident at once with no plan's use
        accounting (the exact route's later hit supplies it). The previous
        plan's slots are ineligible, as the streamer's pinned slots are; a
        filled expert evicted before any plan hits it counts as wasted. In the
        ring modes the fill is held beside the pool instead, no victim."""
        placed = 0
        if self.fill_mode != "pool":
            for expert in candidates:
                if placed >= budget:
                    break
                if expert in self.slot_expert or expert in self.ring:
                    continue
                self.ring.add(expert)
                self.fills += 1
                placed += 1
            return placed
        for expert in candidates:
            if placed >= budget:
                break
            if expert in self.slot_expert:
                continue
            eligible = [slot for slot in range(self.slots) if slot not in self.last_assigned]
            if not eligible:
                break
            eligible.sort(key=lambda slot: self._victim_key(slot, self.policy))
            victim = eligible[0]
            if victim in self.filled_unused:
                self.wasted_fills += 1
                self.filled_unused.discard(victim)
            self.slot_expert[victim] = expert
            self.slot_last_use[victim] = self.use_clock
            self.filled_unused.add(victim)
            self.fills += 1
            placed += 1
        return placed

    def _next_use_after(self, expert, index):
        occurrences = self.future_occurrences.get(expert)
        if not occurrences:
            return None
        pos = bisect.bisect_right(occurrences, index)
        return occurrences[pos] if pos < len(occurrences) else None

    def _victim_key(self, slot, policy):
        if policy.name == "lru":
            return (self.slot_last_use[slot], slot)
        if policy.name == "belady":
            expert = self.slot_expert[slot]
            if expert < 0:
                return (0, 0, slot)
            next_use = self._next_use_after(expert, self.plans_done)
            if next_use is None:
                return (1, 0, slot)
            return (2, -next_use, slot)
        expert = self.slot_expert[slot]
        count = self.expert_use_count[expert] if expert >= 0 else -1
        return (count, self.slot_last_use[slot], slot)

    def plan(self, experts, avoiding=frozenset(), protect=frozenset(),
             policy_override=None, weights=None):
        """Places `experts`, returns (hits, misses, assigned_slots). `protect`
        (the replay's chunk protection) is dropped first when too few
        slots are eligible, matching the streamer's own graded fallback for
        `protectedExperts`; `avoiding` (the modelled avoidingSlots) is
        dropped next if that still is not enough, this tool's own
        pre-existing (and separately documented) fallback for a lookback
        that leaves no eligible slot."""
        active = policy_override or self.policy
        if active.name == "aging-lfu" and self.plans_done > 0 \
                and self.plans_done % active.param == 0:
            for expert in list(self.expert_use_count):
                self.expert_use_count[expert] >>= 1

        newly_seen = [expert not in self.seen_experts for expert in experts]
        self.seen_experts.update(experts)

        assigned = [-1] * len(experts)
        reserved = set()
        hits = 0
        for i, expert in enumerate(experts):
            for slot in range(self.slots):
                if slot in reserved:
                    continue
                if self.slot_expert[slot] == expert:
                    assigned[i] = slot
                    reserved.add(slot)
                    hits += 1
                    if slot in self.filled_unused:
                        self.useful_fills += 1
                        self.filled_unused.discard(slot)
                    break

        # A ring fill the exact route wants is a hit served from the ring's
        # cell; `ring-retain` then gives it a victim slot like a miss, without
        # counting it one. Whatever the plan did not want is wasted, and the
        # ring is empty again after the plan either way.
        ring_hit_indices = [i for i in range(len(experts))
                            if assigned[i] == -1 and experts[i] in self.ring]
        hits += len(ring_hit_indices)
        self.useful_fills += len(ring_hit_indices)
        self.wasted_fills += len(self.ring) - len(ring_hit_indices)
        self.ring = set()
        retain_indices = ring_hit_indices if self.fill_mode == "ring-retain" else []
        for i in ring_hit_indices:
            assigned[i] = -2

        miss_indices = [i for i in range(len(experts)) if assigned[i] == -1]
        needed = len(miss_indices) + len(retain_indices)
        eligible = [s for s in range(self.slots)
                    if s not in reserved and s not in avoiding and s not in protect]
        if needed > len(eligible):
            eligible = [s for s in range(self.slots) if s not in reserved and s not in avoiding]
        if needed > len(eligible):
            eligible = [s for s in range(self.slots) if s not in reserved]

        eligible.sort(key=lambda slot: self._victim_key(slot, active))
        victims = eligible[:len(miss_indices)]
        retain_victims = eligible[len(miss_indices):needed]

        clock = self.use_clock + 1
        self.use_clock = clock
        increments = weights if weights is not None else [1] * len(experts)
        for expert, increment in zip(experts, increments):
            self.expert_use_count[expert] += increment
        for slot in assigned:
            if slot >= 0:
                self.slot_last_use[slot] = clock

        for miss_i, slot in zip(miss_indices, victims):
            if newly_seen[miss_i]:
                self.compulsory += 1
            else:
                self.capacity += 1
            if slot in self.filled_unused:
                self.wasted_fills += 1
                self.filled_unused.discard(slot)
            self.slot_expert[slot] = experts[miss_i]
            self.slot_last_use[slot] = clock
            assigned[miss_i] = slot
        for retain_i, slot in zip(retain_indices, retain_victims):
            if slot in self.filled_unused:
                self.wasted_fills += 1
                self.filled_unused.discard(slot)
            self.slot_expert[slot] = experts[retain_i]
            self.slot_last_use[slot] = clock
            assigned[retain_i] = slot

        self.plans_done += 1
        self.last_assigned = set(slot for slot in assigned if slot >= 0)
        return hits, len(miss_indices), assigned


class LRU2LayerPool(LayerPool):
    """LRU-K with K=2 (O'Neil, O'Neil & Weikum, SIGMOD 1993): the victim is
    the resident whose second-most-recent reference is oldest; a resident
    with fewer than 2 references is evicted first."""

    def __init__(self, slots):
        super().__init__(slots, Policy("lru-2"))
        self.expert_history = defaultdict(list)

    def _victim_key(self, slot, policy):
        expert = self.slot_expert[slot]
        if expert < 0:
            return (-1, self.slot_last_use[slot], slot)
        history = self.expert_history.get(expert, [])
        key = history[0] if len(history) >= 2 else -1
        return (key, self.slot_last_use[slot], slot)

    def plan(self, experts, avoiding=frozenset(), protect=frozenset(),
             policy_override=None, weights=None):
        result = super().plan(experts, avoiding, protect, policy_override, weights)
        clock = self.use_clock
        for expert in experts:
            history = self.expert_history[expert]
            history.append(clock)
            if len(history) > 2:
                history.pop(0)
        return result


class SLRULayerPool:
    """Segmented LRU (Karedla, Love & Wherry 1994): two LRU segments,
    probationary and protected (protected_share of the slots); a
    probationary hit promotes to protected MRU, a protected hit refreshes
    it to protected MRU, and protected demotes its LRU entry to
    probationary MRU once it exceeds its share."""

    def __init__(self, slots, protected_share):
        self.slots = slots
        self.protected_capacity = max(1, round(protected_share * slots))
        self.slot_expert = [-1] * slots
        self.expert_slot = {}
        self.probation = OrderedDict((slot, None) for slot in range(slots))
        self.protected = OrderedDict()
        self.seen_experts = set()
        self.compulsory = 0
        self.capacity = 0

    def plan(self, experts, avoiding=frozenset(), protect=frozenset(),
             policy_override=None, weights=None):
        newly_seen = [expert not in self.seen_experts for expert in experts]
        self.seen_experts.update(experts)
        assigned = [-1] * len(experts)
        hits = 0
        for i, expert in enumerate(experts):
            slot = self.expert_slot.get(expert)
            if slot is None:
                continue
            assigned[i] = slot
            hits += 1
            if slot in self.probation:
                del self.probation[slot]
                self._promote(slot)
            else:
                self.protected.move_to_end(slot)

        miss_indices = [i for i in range(len(experts)) if assigned[i] == -1]
        for miss_i in miss_indices:
            slot = self._select_victim(avoiding, protect)
            if newly_seen[miss_i]:
                self.compulsory += 1
            else:
                self.capacity += 1
            if self.slot_expert[slot] != -1:
                del self.expert_slot[self.slot_expert[slot]]
            expert = experts[miss_i]
            self.slot_expert[slot] = expert
            self.expert_slot[expert] = slot
            self.probation[slot] = None
            assigned[miss_i] = slot
        return hits, len(miss_indices), assigned

    def _promote(self, slot):
        self.protected[slot] = None
        if len(self.protected) > self.protected_capacity:
            demoted, _ = self.protected.popitem(last=False)
            self.probation[demoted] = None

    def _select_victim(self, avoiding, protect=frozenset()):
        combined = avoiding | protect
        for exclude in (combined, avoiding):
            for slot in list(self.probation.keys()):
                if slot not in exclude:
                    del self.probation[slot]
                    return slot
        if self.probation:
            slot = next(iter(self.probation))
            del self.probation[slot]
            return slot
        for exclude in (combined, avoiding):
            for slot in list(self.protected.keys()):
                if slot not in exclude:
                    del self.protected[slot]
                    return slot
        slot = next(iter(self.protected))
        del self.protected[slot]
        return slot


class ArcLayerPool:
    """Adaptive Replacement Cache (Megiddo & Modha, FAST 2003): resident
    LRU lists T1 (recency) and T2 (frequency) plus ghost LRU lists B1 and
    B2 of recently evicted keys, adapting the recency target size p from a
    ghost hit's list. Every slot starts as a distinct sentinel entry in T1
    so an initial fill is an ordinary T1 eviction, oldest slot first."""

    def __init__(self, slots):
        self.slots = slots
        self.c = slots
        self.p = 0
        self.expert_slot = {}
        self.slot_expert = [-1] * slots
        self.t1 = OrderedDict()
        self.t2 = OrderedDict()
        self.b1 = OrderedDict()
        self.b2 = OrderedDict()
        for slot in range(slots):
            sentinel = -1 - slot
            self.t1[sentinel] = None
            self.expert_slot[sentinel] = slot
        self.seen_experts = set()
        self.compulsory = 0
        self.capacity = 0

    def plan(self, experts, avoiding=frozenset(), protect=frozenset(),
             policy_override=None, weights=None):
        newly_seen = [expert not in self.seen_experts for expert in experts]
        self.seen_experts.update(experts)
        assigned = [-1] * len(experts)
        hits = 0
        misses = 0
        for i, expert in enumerate(experts):
            slot, hit = self._request(expert, avoiding, protect)
            assigned[i] = slot
            if hit:
                hits += 1
            else:
                misses += 1
                if newly_seen[i]:
                    self.compulsory += 1
                else:
                    self.capacity += 1
        return hits, misses, assigned

    def _lru_pop(self, ordered, avoiding, protect=frozenset()):
        combined = avoiding | protect
        for exclude in (combined, avoiding):
            for key in list(ordered.keys()):
                slot = self.expert_slot.get(key)
                if slot is None or slot not in exclude:
                    del ordered[key]
                    return key
        key = next(iter(ordered))
        del ordered[key]
        return key

    def _replace(self, favor_t2, avoiding, protect=frozenset()):
        use_t1 = bool(self.t1) and (len(self.t1) > self.p
                                    or (favor_t2 and len(self.t1) == self.p))
        source = self.t1 if use_t1 else self.t2
        if not source:
            source = self.t2 if source is self.t1 else self.t1
        evicted = self._lru_pop(source, avoiding, protect)
        slot = self.expert_slot.pop(evicted)
        self.slot_expert[slot] = -1
        if evicted >= 0:
            ghost = self.b1 if source is self.t1 else self.b2
            ghost[evicted] = None
        return slot

    def _request(self, expert, avoiding, protect=frozenset()):
        if expert in self.t1:
            del self.t1[expert]
            slot = self.expert_slot[expert]
            self.t2[expert] = None
            return slot, True
        if expert in self.t2:
            self.t2.move_to_end(expert)
            return self.expert_slot[expert], True
        if expert in self.b1:
            del self.b1[expert]
            delta = max(1, len(self.b2) // len(self.b1)) if self.b1 else max(1, len(self.b2))
            self.p = min(self.c, self.p + delta)
            slot = self._replace(False, avoiding, protect)
        elif expert in self.b2:
            del self.b2[expert]
            delta = max(1, len(self.b1) // len(self.b2)) if self.b2 else max(1, len(self.b1))
            self.p = max(0, self.p - delta)
            slot = self._replace(True, avoiding, protect)
        else:
            total_t1_b1 = len(self.t1) + len(self.b1)
            total_all = total_t1_b1 + len(self.t2) + len(self.b2)
            if total_t1_b1 == self.c:
                if len(self.t1) < self.c:
                    if self.b1:
                        del self.b1[next(iter(self.b1))]
                    slot = self._replace(False, avoiding, protect)
                else:
                    evicted = self._lru_pop(self.t1, avoiding, protect)
                    slot = self.expert_slot.pop(evicted)
                    self.slot_expert[slot] = -1
            elif total_t1_b1 < self.c and total_all >= self.c:
                if total_all >= 2 * self.c and self.b2:
                    del self.b2[next(iter(self.b2))]
                slot = self._replace(False, avoiding, protect)
            else:
                slot = self._replace(False, avoiding, protect)
            self.expert_slot[expert] = slot
            self.slot_expert[slot] = expert
            self.t1[expert] = None
            return slot, False
        self.expert_slot[expert] = slot
        self.slot_expert[slot] = expert
        self.t2[expert] = None
        return slot, False


def make_pool(slots, policy, future_occurrences=None):
    if policy.name == "slru":
        share = policy.param if policy.param is not None else DEFAULT_SLRU_PROTECTED_SHARE
        return SLRULayerPool(slots, share)
    if policy.name == "arc":
        return ArcLayerPool(slots)
    if policy.name == "lru-2":
        return LRU2LayerPool(slots)
    return LayerPool(slots, policy, future_occurrences)


def _accumulate(stats, settle_stats, label, hits, misses, compulsory, capacity):
    if label[0] == "request":
        _kind, request_id, phase = label
        row = stats[request_id][phase]
    else:
        _kind, chunk_id = label
        row = settle_stats[chunk_id]
    row[0] += hits
    row[1] += misses
    row[2] += compulsory
    row[3] += capacity


def _retile(tiles, sweep_order, reverse):
    """`tiles`: a list of tiles, each a list of (expert, rows, last_row)
    triples. `rows-asc`/`rows-desc` sort by row count, ties left in file
    order (a stable sort). `last-asc`/`last-desc` sort by each expert's
    last row in the chunk, ties broken by rows ascending then expert id,
    regardless of direction."""
    flat = [triple for tile in tiles for triple in tile]
    if sweep_order in ("rows-asc", "rows-desc"):
        flat = sorted(flat, key=lambda triple: triple[1], reverse=reverse)
    else:
        sign = -1 if reverse else 1
        flat = sorted(flat, key=lambda triple: (sign * triple[2], triple[1], triple[0]))
    return [flat[i:i + RETILE_SIZE] for i in range(0, len(flat), RETILE_SIZE)]


def _pack_by_rows(group):
    """Bins `group` (a list of (expert, rows, last_row) triples) into
    `ceil(len(group) / RETILE_SIZE)` tiles the way the Swift `packByRows`
    does: heaviest expert first, ties by expert id ascending, each placed
    into the open tile with the lowest total weight, ties to the lower
    tile index."""
    if not group:
        return []
    tile_count = (len(group) + RETILE_SIZE - 1) // RETILE_SIZE
    tiles = [[] for _ in range(tile_count)]
    weights = [0] * tile_count
    for triple in sorted(group, key=lambda triple: (-triple[1], triple[0])):
        open_tiles = [i for i in range(tile_count) if len(tiles[i]) < RETILE_SIZE]
        i = min(open_tiles, key=lambda j: (weights[j], j))
        tiles[i].append(triple)
        weights[i] += triple[1]
    return tiles


def _retile_resident_first_plain(tiles, resident):
    """`tiles` as in `_retile`; `resident` the set of experts the pool
    holds when the chunk begins. The chunk's resident experts come first,
    then the absent ones, each group in `last-asc` order (ties by rows
    ascending then expert id), re-tiled by RETILE_SIZE across the
    concatenation; this is the `resident-first-plain` order."""
    flat = [triple for tile in tiles for triple in tile]
    key = lambda triple: (triple[0] not in resident, triple[2], triple[1], triple[0])
    flat = sorted(flat, key=key)
    return [flat[i:i + RETILE_SIZE] for i in range(0, len(flat), RETILE_SIZE)]


def _retile_resident_first_grouped(tiles, resident, tail):
    """`tiles` as in `_retile`; `resident` the set of experts the pool
    holds when the chunk begins; `tail` the count of the absent group's
    most recent experts to carry as their own group. Splits the chunk's
    experts into resident and absent (each ranked `last-asc` only to pick
    the tail), takes the absent group's last `min(tail, len(absent))` as
    the tail and the rest as the head, packs resident/head/tail each by
    row weight with `_pack_by_rows`, concatenates the three groups'
    bins resident, head, tail, and re-tiles the concatenation flat in
    runs of RETILE_SIZE; this is round 1's landed `resident-first-grouped`
    order (v13 T5 step 1, fix-up 1: `resident-first` is now the
    interleaved composition below)."""
    flat = [triple for tile in tiles for triple in tile]
    key = lambda triple: (triple[2], triple[1], triple[0])
    resident_group = sorted((t for t in flat if t[0] in resident), key=key)
    absent = sorted((t for t in flat if t[0] not in resident), key=key)
    k = min(tail, len(absent))
    head = absent[:len(absent) - k] if k else absent
    tail_group = absent[len(absent) - k:] if k else []
    packed = _pack_by_rows(resident_group) + _pack_by_rows(head) + _pack_by_rows(tail_group)
    flat_order = [triple for tile in packed for triple in tile]
    return [flat_order[i:i + RETILE_SIZE] for i in range(0, len(flat_order), RETILE_SIZE)]


def _retile_resident_first(tiles, resident, slots, head_factor):
    """`tiles` as in `_retile`; `resident` the set of experts the pool
    holds when the chunk begins; `slots` the pool's slot count;
    `head_factor` (`--sweep-head-factor`) the tile multiplier the head
    rule protects against starving. Ranks the chunk `last-asc` (ties by
    rows ascending then expert id), splits into a resident and an absent
    group (each keeping rank order). If either group is empty, or the head
    search exhausts every tile without leaving a mixed tile behind (so the
    absent group would otherwise vanish from the order), defers to
    `_retile_resident_first_grouped` at its default tail (`DEFAULT_SWEEP_TAIL`,
    unaffected by this call's own `--sweep-tail`, which governs only
    `resident-first-grouped` itself) -- the cold-pool case, matching the
    Swift `recencyBalanced`'s own order whenever the absent group is no
    larger than that default tail (so the grouped composition's head is
    itself empty and it reduces to one packed group), which is the usual
    first-chunk case. Otherwise, with `T = ceil((R + A) / RETILE_SIZE)` tiles,
    finds the smallest head length `h` tiles (0 <= h < T) such that
    `R - RETILE_SIZE*h <= slots - head_factor * ceil(A / (T - h))`; the
    first `RETILE_SIZE*h` residents (rank order, not row-packed) form the
    head. The remaining `M = T - h` tiles each take a contiguous,
    uniform slice of the absent group in rank order (tile `j`: ranks
    `floor(jA/M)..floor((j+1)A/M)-1`, so the last tile holds the most
    recent), then the leftover residents fill the tiles' free slots
    heaviest-first (rows descending, ties by expert id ascending), each
    into the tile with the lowest total weight that still has room, ties
    to the lower tile index. The head residents and the tiles'
    concatenated experts (absent then residents, in placement order)
    re-tile flat in runs of RETILE_SIZE; this is the `resident-first`
    order."""
    flat = [triple for tile in tiles for triple in tile]
    key = lambda triple: (triple[2], triple[1], triple[0])
    resident_group = sorted((t for t in flat if t[0] in resident), key=key)
    absent_group = sorted((t for t in flat if t[0] not in resident), key=key)
    r = len(resident_group)
    a = len(absent_group)
    if r == 0 or a == 0:
        return _retile_resident_first_grouped(tiles, resident, DEFAULT_SWEEP_TAIL)

    tile_count = (r + a + RETILE_SIZE - 1) // RETILE_SIZE
    head = 0
    while head < tile_count:
        remaining_tiles = tile_count - head
        m = (a + remaining_tiles - 1) // remaining_tiles
        if r - RETILE_SIZE * head <= slots - head_factor * m:
            break
        head += 1
    head_residents = resident_group[:RETILE_SIZE * head]
    rest_residents = resident_group[RETILE_SIZE * head:]
    mixed_tiles = tile_count - head
    if mixed_tiles == 0:
        return _retile_resident_first_grouped(tiles, resident, DEFAULT_SWEEP_TAIL)

    bins = [[] for _ in range(mixed_tiles)]
    weights = [0] * mixed_tiles
    for j in range(mixed_tiles):
        lo = (j * a) // mixed_tiles
        hi = ((j + 1) * a) // mixed_tiles
        bins[j] = absent_group[lo:hi]
        weights[j] = sum(t[1] for t in bins[j])

    for triple in sorted(rest_residents, key=lambda t: (-t[1], t[0])):
        open_bins = [i for i in range(mixed_tiles) if len(bins[i]) < RETILE_SIZE]
        if not open_bins:
            break
        i = min(open_bins, key=lambda j: (weights[j], j))
        bins[i].append(triple)
        weights[i] += triple[1]

    order = head_residents + [t for b in bins for t in b]
    return [order[i:i + RETILE_SIZE] for i in range(0, len(order), RETILE_SIZE)]


def replay(lines, slots, policy, layer_filter=None, avoid_lookback=DEFAULT_AVOID_LOOKBACK,
           prefill_weight="one", sweep_order="index", sweep_carry=False,
           phase_policy=None, profile_window=None, protect="chunk",
           sweep_tail=DEFAULT_SWEEP_TAIL, sweep_head_factor=DEFAULT_SWEEP_HEAD_FACTOR,
           fills=None, fill_budget=1, fill_stats=None, fill_mode="pool",
           predicted_future=None, predicted_protect=None):
    """Returns (stats, total_compulsory, settle_stats, settle_meta, profile).
    `slots` is one count for every layer or a {layer: count} map (v20 S0.4,
    the split); `predicted_future` ((layer, position) -> experts) replaces
    the trace's own future on the belady path for decode plans, which
    leaks the future tokens' identities and is a bound, not a policy;
    `predicted_protect` (the same shape) is the online form at a horizon
    of one: a decode plan's victims exclude the slots holding the experts
    predicted for the layer's next decode position.
    `fills` ((layer, position) -> predicted experts, see load_prefetch_fills)
    places up to `fill_budget` of them into the layer's pool (`fill_mode`
    pool) or beside it (ring, ring-retain; see LayerPool) before the decode
    plan at that position; `fill_stats`, when a dict, receives the totals
    (fills, useful, wasted, unused_at_end).

    stats[request_id] = {"prefill": [hits, misses, compulsory, capacity],
                          "decode": [hits, misses, compulsory, capacity]}
    settle_stats[chunk_id] = [hits, misses, compulsory, capacity]
    settle_meta[chunk_id] = {"first_position": int, "tile_count": int}
    profile[request_id][window_index] = decode misses in that window, or
    None when `profile_window` is not given.
    """
    if prefill_weight == "rows" and not has_row_counts(lines):
        raise ValueError(
            "--prefill-weight rows requires a trace with row counts (p lines with a '|' suffix)")
    if sweep_order in ("rows-asc", "rows-desc") and not has_row_counts(lines):
        raise ValueError(
            f"--sweep-order {sweep_order} requires a trace with row counts "
            "(p lines with a '|' suffix)")
    if (sweep_order in ("last-asc", "last-desc", "resident-first",
                        "resident-first-grouped", "resident-first-plain")
            and not has_last_rows(lines)):
        raise ValueError(
            f"--sweep-order {sweep_order} requires a trace with last-row counts "
            "(p lines with a 'count:lastRow' suffix)")

    if has_request_markers(lines):
        labels, settle_meta = label_lines_by_marker(lines)
    else:
        request_ids = segment_requests(lines)
        labels = [("request", rid, "prefill" if kind == "p" else "decode")
                  for (kind, *_), rid in zip(lines, request_ids)]
        settle_meta = {}

    first_decode_position = {}
    if profile_window:
        for (kind, position, *_rest), label in zip(lines, labels):
            if kind == "decode" and label[0] == "request":
                first_decode_position.setdefault(label[1], position)

    by_layer = defaultdict(list)
    for (kind, position, layer, tile, experts, row_counts, last_rows), label in zip(lines, labels):
        if label[0] == "marker":
            continue
        if layer_filter is not None and layer != layer_filter:
            continue
        by_layer[layer].append((kind, position, tile, experts, row_counts, last_rows, label))

    stats = defaultdict(lambda: {"prefill": [0, 0, 0, 0], "decode": [0, 0, 0, 0]})
    settle_stats = defaultdict(lambda: [0, 0, 0, 0])
    profile = defaultdict(lambda: defaultdict(int)) if profile_window else None
    total_compulsory = 0

    for layer, layer_lines in by_layer.items():
        if predicted_future is not None:
            # v20 S0.4: the clairvoyant path sees a predicted future (the
            # table's entries per decode position) in place of the trace's.
            future = build_future_occurrences(
                [predicted_future.get((layer, position), []) if kind == "decode" else experts
                 for kind, position, _t, experts, _rc, _lr, _l in layer_lines])
        else:
            future = build_future_occurrences(
                [experts for _k, _p, _t, experts, _rc, _lr, _l in layer_lines])
        layer_slots = slots[layer] if isinstance(slots, dict) else slots
        pool = make_pool(layer_slots, policy, future)
        if fills is not None and not hasattr(pool, "fill"):
            raise ValueError(f"speculative fills are modelled for the lru / lfu / aging-lfu / belady "
                             f"pool only, not {policy.label()}")
        if fills is not None:
            pool.fill_mode = fill_mode
        lookback = deque(maxlen=avoid_lookback) if avoid_lookback > 0 else deque()
        next_reverse = sweep_order in ("rows-desc", "last-desc")
        decode_positions = [p for kind, p, _t, _e, _rc, _lr, _l in layer_lines if kind == "decode"]
        decode_index = 0

        for item in group_into_chunks(layer_lines):
            if item[0] == "decode":
                _kind, position, experts, label = item
                active = phase_policy["decode"] if phase_policy else None
                protect_set = frozenset()
                if predicted_protect is not None:
                    decode_index += 1
                    if decode_index < len(decode_positions):
                        wanted = predicted_protect.get((layer, decode_positions[decode_index]), ())
                        protect_set = frozenset(
                            slot for slot, expert in enumerate(pool.slot_expert)
                            if expert >= 0 and expert in wanted)
                if fills is not None:
                    candidates = fills.get((layer, position))
                    if candidates:
                        # A plain list takes `fill_budget`; a list of
                        # (candidates, budget) pairs gives each source its own
                        # (v20: the probe at its in-flight budget beside the table).
                        if isinstance(candidates[0], tuple):
                            placed = sum(pool.fill(group, budget) for group, budget in candidates)
                        else:
                            placed = pool.fill(candidates, fill_budget)
                        if fill_stats is not None and placed:
                            per_position = fill_stats.setdefault("per_position", defaultdict(int))
                            per_position[position] += placed
                before = (pool.compulsory, pool.capacity)
                hits, misses, _assigned = pool.plan(experts, protect=protect_set,
                                                    policy_override=active)
                _accumulate(stats, settle_stats, label, hits, misses,
                           pool.compulsory - before[0], pool.capacity - before[1])
                if fill_stats is not None and label[0] == "request":
                    misses_per_layer = fill_stats.setdefault("misses_per_layer", defaultdict(int))
                    misses_per_layer[(label[1], layer)] += misses
                    if misses:
                        miss_layers = fill_stats.setdefault("miss_layers", defaultdict(int))
                        miss_layers[(label[1], layer)] += 1
                if profile_window and label[0] == "request":
                    base = first_decode_position.get(label[1], position)
                    window_index = (position - base) // profile_window
                    profile[label[1]][window_index] += misses
                    profile[label[1]][("capacity", window_index)] += pool.capacity - before[1]
                lookback.clear()
                continue

            _kind, original_tiles, label = item
            if sweep_order == "index":
                new_tiles = original_tiles
            elif sweep_order == "resident-first":
                resident = frozenset(expert for expert in pool.slot_expert if expert >= 0)
                new_tiles = _retile_resident_first(original_tiles, resident, layer_slots, sweep_head_factor)
            elif sweep_order == "resident-first-grouped":
                resident = frozenset(expert for expert in pool.slot_expert if expert >= 0)
                new_tiles = _retile_resident_first_grouped(original_tiles, resident, sweep_tail)
            elif sweep_order == "resident-first-plain":
                resident = frozenset(expert for expert in pool.slot_expert if expert >= 0)
                new_tiles = _retile_resident_first_plain(original_tiles, resident)
            else:
                reverse = (next_reverse if sweep_carry
                          else sweep_order in ("rows-desc", "last-desc"))
                new_tiles = _retile(original_tiles, sweep_order, reverse)
                if sweep_carry:
                    next_reverse = not next_reverse

            lookback.clear()
            active = phase_policy["prefill"] if phase_policy else None
            pending = [set(expert for expert, _rows, _last in tile) for tile in new_tiles]
            remaining = set().union(*pending) if pending else set()
            for tile_index, new_tile in enumerate(new_tiles):
                tile_experts = [expert for expert, _rows, _last in new_tile]
                tile_rows = [rows for _expert, rows, _last in new_tile]
                remaining -= pending[tile_index]
                avoiding = frozenset(slot for _tile, held in lookback for slot in held)
                protect_set = (frozenset(slot for slot, expert in enumerate(pool.slot_expert)
                                         if expert in remaining)
                              if protect == "chunk" else frozenset())
                weights = tile_rows if prefill_weight == "rows" else None
                before = (pool.compulsory, pool.capacity)
                hits, misses, assigned = pool.plan(
                    tile_experts, avoiding=avoiding, protect=protect_set,
                    policy_override=active, weights=weights)
                _accumulate(stats, settle_stats, label, hits, misses,
                           pool.compulsory - before[0], pool.capacity - before[1])
                lookback.append((None, [slot for slot in assigned if slot >= 0]))
        total_compulsory += pool.compulsory
        if fill_stats is not None and hasattr(pool, "fills"):
            fill_stats["fills"] = fill_stats.get("fills", 0) + pool.fills
            fill_stats["useful"] = fill_stats.get("useful", 0) + pool.useful_fills
            fill_stats["wasted"] = fill_stats.get("wasted", 0) + pool.wasted_fills
            fill_stats["unused_at_end"] = fill_stats.get("unused_at_end", 0) + len(pool.filled_unused)
            fill_stats.setdefault("per_layer", {})[layer] = (
                pool.fills, pool.useful_fills, pool.wasted_fills)

    return stats, total_compulsory, settle_stats, settle_meta, profile


def print_report(stats, total_compulsory, policy, slots, layer_filter, avoid_lookback,
                 settle_stats=None, settle_meta=None, protect="chunk"):
    layer_note = f" layer={layer_filter}" if layer_filter is not None else ""
    print(f"policy={policy.label()} slots={slots} avoid_lookback={avoid_lookback} "
          f"protect={protect}{layer_note}")
    print(f"  compulsory (cold, first touch, all replayed layers) = {total_compulsory}")
    for request_id in sorted(stats):
        row = stats[request_id]
        pre_hits, pre_miss, pre_comp, pre_cap = row["prefill"]
        dec_hits, dec_miss, dec_comp, dec_cap = row["decode"]
        print(f"  request {request_id}: "
              f"prefill hits={pre_hits} misses={pre_miss} (compulsory={pre_comp} capacity={pre_cap}) | "
              f"decode hits={dec_hits} misses={dec_miss} (compulsory={dec_comp} capacity={dec_cap})")
    if settle_meta:
        for chunk_id in sorted(settle_meta):
            meta = settle_meta[chunk_id]
            hits, misses, comp, cap = settle_stats.get(chunk_id, [0, 0, 0, 0])
            print(f"  settle chunk {chunk_id}: first_position={meta['first_position']} "
                  f"tiles={meta['tile_count']} hits={hits} misses={misses} "
                  f"(compulsory={comp} capacity={cap})")


def print_expect_deltas(stats, expect_path):
    rows = []
    with open(expect_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                rows.append((int(parts[0]), int(parts[1])))
            else:
                rows.append((None, int(parts[0])))
    print(f"  --expect {expect_path}: replayed vs measured "
          "(prefill_misses decode_misses; a one-integer line stays decode-only)")
    for i, request_id in enumerate(sorted(stats)):
        replayed_prefill = stats[request_id]["prefill"][1]
        replayed_decode = stats[request_id]["decode"][1]
        if i >= len(rows):
            print(f"    request {request_id}: replayed prefill={replayed_prefill} "
                  f"decode={replayed_decode} measured=<none given>")
            continue
        expected_prefill, expected_decode = rows[i]
        if expected_prefill is not None:
            print(f"    request {request_id}: prefill replayed={replayed_prefill} "
                  f"measured={expected_prefill} delta={replayed_prefill - expected_prefill}")
        print(f"    request {request_id}: decode replayed={replayed_decode} "
              f"measured={expected_decode} delta={replayed_decode - expected_decode}")


def print_profile(profile, window):
    """Misses per window, then the capacity misses per window (the rest are
    compulsory, the request's first touch of that layer's expert)."""
    print(f"  --profile {window}: decode misses per window, summed over replayed layers")
    for request_id in sorted(profile):
        windows = profile[request_id]
        indices = [key for key in windows if isinstance(key, int)]
        last = max(indices) if indices else -1
        values = [windows.get(i, 0) for i in range(last + 1)]
        capacity = [windows.get(("capacity", i), 0) for i in range(last + 1)]
        print(f"    request {request_id}: " + "/".join(str(v) for v in values))
        print(f"    request {request_id} capacity: " + "/".join(str(v) for v in capacity))


# --- self-test -------------------------------------------------------------

def _lines_from_text(text):
    return [parse_line(raw) for raw in text.strip("\n").split("\n") if raw.strip()]


def _run(text, slots, policy_raw, **kwargs):
    lines = _lines_from_text(text)
    return replay(lines, slots, parse_policy(policy_raw), **kwargs)


def self_test():
    failures = []

    def check(label, actual, expected):
        if actual != expected:
            failures.append(f"{label}: got {actual}, expected {expected}")

    # Dataset 1: single-layer, single-expert-per-plan decode sequence
    # A A A B B C A (slots=2), hand-computed in the task report.
    letter_ids = {"A": 1, "B": 2, "C": 3}
    decode_trace = "\n".join(
        f"{i} 0 {letter_ids[expert]}" for i, expert in
        enumerate(["A", "A", "A", "B", "B", "C", "A"]))

    stats, _, _, _, _ = _run(decode_trace, slots=2, policy_raw="lru")
    check("lru hits", stats[1]["decode"][0], 3)
    check("lru misses", stats[1]["decode"][1], 4)
    check("lru compulsory", stats[1]["decode"][2], 3)
    check("lru capacity", stats[1]["decode"][3], 1)
    _, _, _, _, profile = _run(decode_trace, slots=2, policy_raw="lru", profile_window=1)
    check("profile misses per window", [profile[1].get(i, 0) for i in range(7)], [1, 0, 0, 1, 0, 1, 1])
    check("profile capacity per window", [profile[1].get(("capacity", i), 0) for i in range(7)],
          [0, 0, 0, 0, 0, 0, 1])

    # Speculative fills (v15 Task 2's pricing): layer 1 requests A, B, C at
    # positions 0, 1, 2 with two slots; a right fill of B before position 1 is a
    # hit there, a wrong fill of D takes the empty slot, survives B's miss (A
    # is older) and is evicted by C's.
    fill_trace = "\n".join(f"{i} 1 {letter_ids[expert]}" for i, expert in enumerate(["A", "B", "C"]))
    fill_stats = {}
    stats, _, _, _, _ = _run(fill_trace, slots=2, policy_raw="lru",
                             fills={(1, 1): [letter_ids["B"]]}, fill_stats=fill_stats)
    check("fill right hits", stats[1]["decode"][0], 1)
    check("fill right misses", stats[1]["decode"][1], 2)
    check("fill right counters", (fill_stats["fills"], fill_stats["useful"], fill_stats["wasted"]), (1, 1, 0))
    fill_stats = {}
    stats, _, _, _, _ = _run(fill_trace, slots=2, policy_raw="lru",
                             fills={(1, 1): [4]}, fill_stats=fill_stats)
    check("fill wrong misses", stats[1]["decode"][1], 3)
    check("fill wrong counters", (fill_stats["fills"], fill_stats["useful"], fill_stats["wasted"]), (1, 0, 1))
    # The fill modes on A, B, C, B: a right fill of B before position 1 is a
    # hit in every mode; in `ring` nothing is retained, so B's return at
    # position 3 misses (A, the older resident, is evicted for it), while
    # `pool` and `ring-retain` keep B and hit; a wrong ring fill is wasted at
    # its plan and costs the pool nothing.
    return_trace = "\n".join(f"{i} 1 {letter_ids[expert]}" for i, expert in enumerate(["A", "B", "C", "B"]))
    for mode, expected_hits, expected_misses in (("pool", 2, 2), ("ring", 1, 3), ("ring-retain", 2, 2)):
        fill_stats = {}
        stats, _, _, _, _ = _run(return_trace, slots=2, policy_raw="lru",
                                 fills={(1, 1): [letter_ids["B"]]}, fill_stats=fill_stats,
                                 fill_mode=mode)
        check(f"fill mode {mode} hits", stats[1]["decode"][0], expected_hits)
        check(f"fill mode {mode} misses", stats[1]["decode"][1], expected_misses)
        check(f"fill mode {mode} counters",
              (fill_stats["fills"], fill_stats["useful"], fill_stats["wasted"]), (1, 1, 0))
    fill_stats = {}
    stats, _, _, _, _ = _run(return_trace, slots=2, policy_raw="lru",
                             fills={(1, 1): [4]}, fill_stats=fill_stats, fill_mode="ring")
    check("ring wrong fill hits", stats[1]["decode"][0], 1)
    check("ring wrong fill misses", stats[1]["decode"][1], 3)
    check("ring wrong fill counters",
          (fill_stats["fills"], fill_stats["useful"], fill_stats["wasted"]), (1, 0, 1))
    # Per-source budgets: the first source's two candidates at budget 1 place
    # one (the wrong 4), the second source's B at its own budget lands the hit.
    fill_stats = {}
    stats, _, _, _, _ = _run(return_trace, slots=2, policy_raw="lru",
                             fills={(1, 1): [([4, 5], 1), ([letter_ids["B"]], 8)]},
                             fill_stats=fill_stats, fill_mode="ring")
    check("grouped fills hits", stats[1]["decode"][0], 1)
    check("grouped fills counters",
          (fill_stats["fills"], fill_stats["useful"], fill_stats["wasted"]), (2, 1, 1))
    check("grouped fills per position", dict(fill_stats["per_position"]), {1: 2})
    # v20 S0.1: the token-id table as fills. Layer 1 routes A, B, A, C, B, A
    # at positions 0..5 under pieces x, y, x, z, y, x: at position 2 the table's
    # entry for x is position 0's route, at 4 y's is position 1's, at 5 x's is
    # the last occurrence's (position 2), so `last` issues three fills, every
    # one right; `last2` at position 5 unions positions 2 and 0 (both A);
    # `freq` the same. A trace with an `r` line and a second request keeps the
    # second request out of the table. A prefill `p` line before decode is not
    # a decode position. The union with the previous position adds position
    # i-1's route: at position 1 that is A, wrong for B, so one wasted fill.
    table_lines = _lines_from_text("\n".join(
        ["r 0 4", "p 0 1 0 1 2"]
        + [f"{i} 1 {letter_ids[e]}" for i, e in enumerate(["A", "B", "A", "C", "B", "A"])]
        + ["r 0 4"] + [f"{i} 1 {letter_ids['C']}" for i in range(2)]))
    table_pieces = ["x", "y", "x", "z", "y", "x"]
    info = {}
    table_fills = build_table_fills(table_lines, table_pieces, frozenset([1]),
                                    width=8, source="last", info=info)
    check("table last fills", table_fills,
          {(1, 2): [letter_ids["A"]], (1, 4): [letter_ids["B"]], (1, 5): [letter_ids["A"]]})
    check("table info", (info["positions"], info["issued"], info["covered"]),
          (6, {1: 3}, {1: 3}))
    fill_stats = {}
    stats, _, _, _, _ = _run("\n".join(
        [f"{i} 1 {letter_ids[e]}" for i, e in enumerate(["A", "B", "A", "C", "B", "A"])]),
        slots=1, policy_raw="lru", fills=table_fills, fill_stats=fill_stats,
        fill_budget=8, fill_mode="ring")
    check("table ring hits", stats[1]["decode"][0], 3)
    check("table ring counters",
          (fill_stats["fills"], fill_stats["useful"], fill_stats["wasted"]), (3, 3, 0))
    check("table per-layer counters", fill_stats["per_layer"], {1: (3, 3, 0)})
    check("table per-position", dict(fill_stats["per_position"]), {2: 1, 4: 1, 5: 1})
    check("table misses per layer", dict(fill_stats["misses_per_layer"]), {(1, 1): 3})
    union_fills = build_table_fills(table_lines, table_pieces, frozenset([1]),
                                    width=8, source="last", union_previous=True)
    check("table union previous at 1", union_fills[(1, 1)], [letter_ids["A"]])
    check("table union previous at 3", union_fills[(1, 3)], [letter_ids["A"]])
    check("table last2 at 5", build_table_fills(table_lines, table_pieces, frozenset([1]),
                                                source="last2")[(1, 5)], [letter_ids["A"]])
    check("table freq at 5", build_table_fills(table_lines, table_pieces, frozenset([1]),
                                               source="freq")[(1, 5)], [letter_ids["A"]])
    check("table freq prediction order",
          table_prediction([[1, 2], [3, 2], [3, 4]], "freq", 8), [3, 2, 4, 1])
    check("table last3 merged", table_prediction([[1, 2], [3, 2], [5, 4]], "last3", 3), [5, 4, 3])
    # The draft over the prompt p q y x y then the answer x y x z y x, 2-grams:
    # at position 1 the 2-gram (y, x) ending at the answer's first piece last
    # occurred in the prompt followed by y, a hit with no table entry for y yet;
    # at 2 the 2-gram (x, y) ending at the prompt's tail was followed by x, a
    # hit, and x's entry is position 0's route, a right fill; at 3 (y, x) is
    # followed by y in the answer, a miss against z, and y's entry fills B for
    # the real C, a wasted fill; positions 4 and 5 have no earlier 2-gram.
    draft_info = {}
    draft_fills = build_table_fills(
        table_lines, table_pieces, frozenset([1]), width=8, source="last",
        draft_n=2, draft_layers=frozenset([1]), prompt_pieces=["p", "q", "y", "x", "y"],
        info=draft_info)
    check("draft proposals", (draft_info["draft"]["proposals"], draft_info["draft"]["hits"]),
          (3, 2))
    check("draft fills", draft_fills, {(1, 2): [letter_ids["A"]], (1, 3): [letter_ids["B"]]})
    seeded = build_table_fills(
        table_lines, table_pieces, frozenset([1]), width=8, source="last",
        prompt_pieces=["p", "q", "x", "y", "z"], seed_routes={(1, 2): [letter_ids["C"]]})
    check("table seeded from the prompt", seeded[(1, 0)], [letter_ids["C"]])
    check("layer set", sorted(parse_layer_set("0,30-31,5")), [0, 5, 30, 31])
    check("layer set all", len(parse_layer_set("all")), 40)
    try:
        build_table_fills(table_lines, table_pieces[:3], frozenset([1]))
        check("table length mismatch raises", False, True)
    except ValueError:
        pass
    # The two-distance queue keys each capture's target by its own probe_distance:
    # the second capture here is at distance 3, so its fill lands three layers on.
    import os
    import tempfile
    d1_rows = [{"position": 0, "layer": 3, "probe_distance": 1, "next_layer_prediction": [7], "resident": []},
               {"position": 0, "layer": 4, "probe_distance": 1, "next_layer_prediction": [9], "resident": [7]}]
    d2_rows = [{"position": 0, "layer": 3, "probe_distance": 3, "next_layer_prediction": [11], "resident": []}]
    with tempfile.TemporaryDirectory() as tmp:
        path_d1 = os.path.join(tmp, "d1.jsonl")
        path_d2 = os.path.join(tmp, "d2.jsonl")
        with open(path_d1, "w") as handle:
            handle.write("\n".join(json.dumps(row) for row in d1_rows) + "\n")
        with open(path_d2, "w") as handle:
            handle.write("\n".join(json.dumps(row) for row in d2_rows) + "\n")
        queue = load_two_distance_queue(path_d1, path_d2, top_m=8)
    check("queue targets by probe distance", dict(queue), {(4, 0): [7], (6, 0): [11], (5, 0): [9]})

    stats, _, _, _, _ = _run(decode_trace, slots=2, policy_raw="lfu")
    check("lfu hits", stats[1]["decode"][0], 4)
    check("lfu misses", stats[1]["decode"][1], 3)
    check("lfu compulsory", stats[1]["decode"][2], 3)
    check("lfu capacity", stats[1]["decode"][3], 0)

    stats, _, _, _, _ = _run(decode_trace, slots=2, policy_raw="belady")
    check("belady hits", stats[1]["decode"][0], 4)
    check("belady misses", stats[1]["decode"][1], 3)
    check("belady compulsory", stats[1]["decode"][2], 3)
    check("belady capacity", stats[1]["decode"][3], 0)

    aging_trace = decode_trace
    stats, _, _, _, _ = _run(aging_trace, slots=2, policy_raw="aging-lfu:5")
    check("aging-lfu hits", stats[1]["decode"][0], 3)
    check("aging-lfu misses", stats[1]["decode"][1], 4)
    check("aging-lfu compulsory", stats[1]["decode"][2], 3)
    check("aging-lfu capacity", stats[1]["decode"][3], 1)

    # Dataset 3 (no `r` line): two requests via the position heuristic,
    # exercising segmentation, phase split and avoidingSlots.
    mixed_trace = "\n".join([
        "0 0 10", "1 0 10", "2 0 10", "3 0 20",
        "p 10 0 0 30", "p 10 0 1 40",
        "11 0 30",
    ])
    stats, _, _, _, _ = _run(mixed_trace, slots=2, policy_raw="lfu", avoid_lookback=1)
    check("req1 decode hits", stats[1]["decode"][0], 2)
    check("req1 decode misses", stats[1]["decode"][1], 2)
    check("req1 prefill lines", stats[1]["prefill"], [0, 0, 0, 0])
    check("req2 prefill hits", stats[2]["prefill"][0], 0)
    check("req2 prefill misses", stats[2]["prefill"][1], 2)
    check("req2 prefill compulsory (30 and 40 both new)", stats[2]["prefill"][2], 2)
    check("req2 decode hits (avoidingSlots protected expert 30)",
          stats[2]["decode"][0], 1)
    check("req2 decode misses", stats[2]["decode"][1], 0)

    # Dataset 4: `r`-delimited requests with a settle chunk between them.
    settle_trace = "\n".join([
        "r 0 4",
        "p 0 0 0 10",
        "1 0 10",
        "p 5 0 0 20",
        "r 6 5",
        "p 6 0 0 10",
        "7 0 10",
    ])
    stats, _, settle_stats, settle_meta, _ = _run(settle_trace, slots=2, policy_raw="lfu")
    check("req1 prefill", stats[1]["prefill"], [0, 1, 1, 0])
    check("req1 decode", stats[1]["decode"], [1, 0, 0, 0])
    check("settle chunk count", len(settle_meta), 1)
    check("settle first_position", settle_meta[0]["first_position"], 5)
    check("settle tile_count", settle_meta[0]["tile_count"], 1)
    check("settle stats", settle_stats[0], [0, 1, 1, 0])
    check("req2 prefill (10 survived the settle)", stats[2]["prefill"], [1, 0, 0, 0])
    check("req2 decode", stats[2]["decode"], [1, 0, 0, 0])

    # Dataset 5: the lookback knob. Single layer, slots=3, tiles A A B C D A.
    lookback_trace = "\n".join([
        "p 0 0 0 1", "p 0 0 1 1", "p 0 0 2 2", "p 0 0 3 3", "p 0 0 4 4", "p 0 0 5 1",
    ])
    stats1, _, _, _, _ = _run(lookback_trace, slots=3, policy_raw="lfu", avoid_lookback=1)
    check("lookback=1 hits", stats1[1]["prefill"][0], 2)
    check("lookback=1 misses", stats1[1]["prefill"][1], 4)
    check("lookback=1 compulsory", stats1[1]["prefill"][2], 4)
    check("lookback=1 capacity", stats1[1]["prefill"][3], 0)

    stats2, _, _, _, _ = _run(lookback_trace, slots=3, policy_raw="lfu", avoid_lookback=2)
    check("lookback=2 hits", stats2[1]["prefill"][0], 1)
    check("lookback=2 misses", stats2[1]["prefill"][1], 5)
    check("lookback=2 compulsory", stats2[1]["prefill"][2], 4)
    check("lookback=2 capacity", stats2[1]["prefill"][3], 1)

    # Dataset 6: --prefill-weight rows. Tile0 loads A (rows=10) and B
    # (rows=1), both compulsory, same clock -> tied on lastUse under weight
    # "one" (both count 1), broken by slot index (slot0=A evicted next).
    # Under weight "rows" A's count is 10 and B's is 1, so B is the cheaper
    # LFU victim instead. Tile1 then needs one eviction (expert C, new).
    weight_trace = "\n".join([
        "p 0 0 0 1 2 | 10 1",
        "p 0 0 1 3",
    ])
    stats_one, _, _, _, _ = _run(weight_trace, slots=2, policy_raw="lfu", prefill_weight="one")
    check("weight=one: 3 misses so far (A, B compulsory, C capacity)",
          stats_one[1]["prefill"][1], 3)
    stats_rows, _, _, _, _ = _run(weight_trace, slots=2, policy_raw="lfu", prefill_weight="rows")
    check("weight=rows: also 3 misses so far", stats_rows[1]["prefill"][1], 3)

    probe_a_evicted = "\n".join([weight_trace, "p 5 0 0 1"])
    stats_one2, _, _, _, _ = _run(probe_a_evicted, slots=2, policy_raw="lfu", prefill_weight="one")
    check("weight=one: A re-miss (A was the tie-break victim)",
          stats_one2[1]["prefill"][1], 4)
    probe_b_evicted = "\n".join([weight_trace, "p 5 0 0 2"])
    stats_rows2, _, _, _, _ = _run(probe_b_evicted, slots=2, policy_raw="lfu",
                                   prefill_weight="rows")
    check("weight=rows: B re-miss (B was the cheap victim, not A)",
          stats_rows2[1]["prefill"][1], 4)

    try:
        _run("p 0 0 0 1 2", slots=2, policy_raw="lfu", prefill_weight="rows")
        failures.append("--prefill-weight rows should error on a trace with no row counts")
    except ValueError:
        pass

    # Dataset 7: --sweep-order. One chunk, 9 experts 1..9 with rows equal to
    # their id (so rows-asc keeps them in the same order as index, and
    # rows-desc reverses it), slots=8: rows-asc's re-tiling puts 1..8 in the
    # first (new) tile and 9 alone in the second, which then evicts the
    # tie-broken slot0 (expert 1); rows-desc puts 9..2 first and 1 alone
    # second, evicting slot0 (expert 9) instead.
    sweep_trace = "p 0 0 0 " + " ".join(str(e) for e in range(1, 10)) \
        + " | " + " ".join(str(e) for e in range(1, 10))
    lines_asc = _lines_from_text(sweep_trace)
    stats_asc, _, _, _, _ = replay(lines_asc, 8, parse_policy("lfu"), sweep_order="rows-asc")
    check("rows-asc: 9 present", stats_asc[1]["prefill"][0], 0)
    lines_probe9 = _lines_from_text(sweep_trace + "\np 5 0 0 9")
    stats_asc9, _, _, _, _ = replay(lines_probe9, 8, parse_policy("lfu"), sweep_order="rows-asc")
    check("rows-asc: 9 is a hit (freshly loaded)", stats_asc9[1]["prefill"][0], 1)
    lines_probe1 = _lines_from_text(sweep_trace + "\np 5 0 0 1")
    stats_asc1, _, _, _, _ = replay(lines_probe1, 8, parse_policy("lfu"), sweep_order="rows-asc")
    check("rows-asc: 1 is a miss (evicted)", stats_asc1[1]["prefill"][0], 0)

    stats_desc9, _, _, _, _ = replay(lines_probe9, 8, parse_policy("lfu"), sweep_order="rows-desc")
    check("rows-desc: 9 is a miss (evicted)", stats_desc9[1]["prefill"][0], 0)
    stats_desc1, _, _, _, _ = replay(lines_probe1, 8, parse_policy("lfu"), sweep_order="rows-desc")
    check("rows-desc: 1 is a hit (freshly loaded)", stats_desc1[1]["prefill"][0], 1)

    # Dataset 8: --sweep-carry. A second 9-expert chunk (ids 11..19, tile
    # indices restarting at 0) right after the first, no decode between
    # them. Chunk 1 (ascending) leaves the pool with expert 9 in the slot
    # tied oldest and experts 2-8 in the other 7 (verified against dataset
    # 7's own trace, so the two tie for eviction priority throughout chunk
    # 2's own first tile). carry=on flips chunk 2 to descending, which
    # lands 12 (not 19) in that same tied slot, so 12 is what chunk 2's own
    # second tile evicts; carry=off keeps chunk 2 ascending, landing 18
    # there instead. Confirmed against the tool's own output before being
    # written here (both directions checked, not just the expected one).
    chunk2 = "p 0 0 0 " + " ".join(str(e) for e in range(11, 20)) \
        + " | " + " ".join(str(e) for e in range(1, 10))
    carried_trace = sweep_trace + "\n" + chunk2

    def probe_hit(probe_expert, carry):
        lines = _lines_from_text(carried_trace + f"\np 5 0 0 {probe_expert}")
        stats, _, _, _, _ = replay(lines, 8, parse_policy("lfu"),
                                   sweep_order="rows-asc", sweep_carry=carry)
        return stats[1]["prefill"][0]

    check("sweep-carry on: chunk 2 flips to desc, 12 evicted", probe_hit(12, True), 0)
    check("sweep-carry on: chunk 2 flips to desc, 18 survives", probe_hit(18, True), 1)
    check("sweep-carry off: chunk 2 stays asc, 12 survives", probe_hit(12, False), 1)
    check("sweep-carry off: chunk 2 stays asc, 18 evicted", probe_hit(18, False), 0)

    try:
        _run("p 0 0 0 1 2 3", slots=8, policy_raw="lfu", sweep_order="rows-asc")
        failures.append("--sweep-order rows-asc should error on a trace with no row counts")
    except ValueError:
        pass

    # Dataset 8b: --sweep-order last-asc/last-desc. Same 9-expert shape as
    # dataset 7, slots=8, but rows and last-row are deliberately
    # anti-correlated (expert k has rows=10-k, last=k) so a last-row sort
    # gives a different tiling than a rows sort: last-asc orders 1..9 (tile0
    # = 1..8, tile1 = [9] alone), last-desc orders 9..1 (tile0 = 9..2, tile1
    # = [1] alone). Both directions tie on count and clock in tile0 (a cold
    # fill), so tile1's single eviction always takes slot0, whichever
    # expert landed there first in that direction's own order.
    last_row_trace = "p 0 0 0 " + " ".join(str(e) for e in range(1, 10)) \
        + " | " + " ".join(f"{10 - e}:{e}" for e in range(1, 10))
    lines_last9 = _lines_from_text(last_row_trace + "\np 5 0 0 9 | 1:9")
    stats_last_asc9, _, _, _, _ = replay(lines_last9, 8, parse_policy("lfu"),
                                         sweep_order="last-asc")
    check("last-asc: 9 (highest last-row) is a hit (freshly loaded)",
          stats_last_asc9[1]["prefill"][0], 1)
    lines_last1 = _lines_from_text(last_row_trace + "\np 5 0 0 1 | 1:1")
    stats_last_asc1, _, _, _, _ = replay(lines_last1, 8, parse_policy("lfu"),
                                         sweep_order="last-asc")
    check("last-asc: 1 (lowest last-row) is a miss (evicted)",
          stats_last_asc1[1]["prefill"][0], 0)
    stats_last_desc9, _, _, _, _ = replay(lines_last9, 8, parse_policy("lfu"),
                                          sweep_order="last-desc")
    check("last-desc: 9 is a miss (evicted)", stats_last_desc9[1]["prefill"][0], 0)
    stats_last_desc1, _, _, _, _ = replay(lines_last1, 8, parse_policy("lfu"),
                                          sweep_order="last-desc")
    check("last-desc: 1 is a hit (freshly loaded)", stats_last_desc1[1]["prefill"][0], 1)

    # _retile's own tie-break, isolated from pool dynamics: three experts
    # sharing one last-row (10) but distinct row counts; ties break by rows
    # ascending then expert id, the same order regardless of direction.
    tied_tiles = [[(3, 8, 10), (1, 5, 10), (2, 2, 10)]]
    expected_tie_order = [(2, 2, 10), (1, 5, 10), (3, 8, 10)]
    check("_retile last-asc tie-break (rows ascending)",
          _retile(tied_tiles, "last-asc", reverse=False), [expected_tie_order])
    check("_retile last-desc tie-break (still rows ascending)",
          _retile(tied_tiles, "last-desc", reverse=True), [expected_tie_order])

    try:
        _run("p 0 0 0 1 2 3 | 1 1 1", slots=8, policy_raw="lfu", sweep_order="last-asc")
        failures.append("--sweep-order last-asc should error on a trace with no last-row counts")
    except ValueError:
        pass

    # Dataset 8c: --sweep-order resident-first-grouped{,-plain}. slots=8,
    # lfu. Chunk 1 fills the pool with 1..8. Chunk 2 needs the residents 1
    # and 2 (the chunk's most recent rows, 20 and 21) plus the absent 9..15
    # (rows 1..7): nine experts, two tiles. last-asc puts 9..15 and 1 in
    # tile0 and 2 alone in tile1, so tile0's seven misses need seven of the
    # six unprotected slots, protection drops, 2 is evicted and re-missed in
    # tile1 (chunk 2: 1 hit, 8 misses). Both resident orders sweep 1 and 2
    # first (tile0 = 1 2 9..14, tile1 = 15): both hits harvested before any
    # eviction (chunk 2: 2 hits, 7 misses). resident-first-grouped's default
    # --sweep-tail (96) covers every absent expert here (only 7 of them), so
    # its head is empty and its tail is the whole absent group packed by row
    # weight; every absent row is 1, so the packing's tie-break (expert id
    # ascending) reproduces last-asc's own 9..15 order and the composition
    # matches -plain exactly on this trace.
    resident_trace = "\n".join([
        "p 0 0 0 1 2 3 4 5 6 7 8 | " + " ".join("1:0" for _ in range(8)),
        "p 5 0 0 1 2 9 10 11 12 13 14 | 1:20 1:21 1:1 1:2 1:3 1:4 1:5 1:6",
        "p 5 0 1 15 | 1:7",
    ])
    stats_last_asc, _, _, _, _ = _run(resident_trace, slots=8, policy_raw="lfu",
                                      sweep_order="last-asc")
    check("last-asc: one hit across both chunks (2 evicted before its tile)",
          stats_last_asc[1]["prefill"][:2], [1, 16])
    stats_rf_plain, _, _, _, _ = _run(resident_trace, slots=8, policy_raw="lfu",
                                      sweep_order="resident-first-plain")
    check("resident-first-plain: both residents hit before any eviction",
          stats_rf_plain[1]["prefill"][:2], [2, 15])
    stats_rf_grouped, _, _, _, _ = _run(resident_trace, slots=8, policy_raw="lfu",
                                        sweep_order="resident-first-grouped")
    check("resident-first-grouped: same totals as -plain on this trace (see comment above)",
          stats_rf_grouped[1]["prefill"][:2], [2, 15])
    stats_rf_plain_carry, _, _, _, _ = _run(resident_trace, slots=8, policy_raw="lfu",
                                            sweep_order="resident-first-plain", sweep_carry=True)
    check("resident-first-plain: --sweep-carry has no effect",
          stats_rf_plain_carry[1]["prefill"][:2], [2, 15])
    stats_rf_grouped_carry, _, _, _, _ = _run(resident_trace, slots=8, policy_raw="lfu",
                                              sweep_order="resident-first-grouped", sweep_carry=True)
    check("resident-first-grouped: --sweep-carry has no effect",
          stats_rf_grouped_carry[1]["prefill"][:2], [2, 15])

    # _retile_resident_first_plain's own order, isolated from pool
    # dynamics: the resident expert leads regardless of its last row;
    # inside each group last-asc with the rows-then-id tie-break.
    mixed_tiles = [[(5, 3, 9), (1, 1, 4), (7, 2, 9), (3, 1, 9)]]
    check("_retile_resident_first_plain: resident first, then last-asc with ties by rows then id",
          _retile_resident_first_plain(mixed_tiles, frozenset({7})),
          [[(7, 2, 9), (1, 1, 4), (3, 1, 9), (5, 3, 9)]])

    # _retile_resident_first_grouped (round 1's composition), isolated from
    # pool dynamics. resident = {1, 2, 3}: 1 and 2 tie on rows (5, tie broken
    # by id) ahead of 3 (rows 2), regardless of all three residents' last
    # rows sitting above every absent expert's; the group split does not
    # read recency. absent = {10..15}, last rows 10..15 ascending, ranked
    # last-asc only to pick the tail. With --sweep-tail 3 the tail is
    # {13, 14, 15} and the head {10, 11, 12}; packByRows orders the head by
    # rows descending (10:3, 12:2, 11:1) and the tail by rows descending
    # with a tie (13:4, 15:4 tied, id breaks it before 14:1). The 9-expert
    # concatenation (3 resident + 3 head + 3 tail) re-tiled flat in runs of
    # 8 gives a first tile spanning all three groups and a second tile of
    # the tail's last expert, ceil(9 / 8) = 2.
    composition_tiles = [[
        (1, 5, 100), (2, 5, 90), (3, 2, 40),
        (10, 3, 10), (11, 1, 11), (12, 2, 12),
        (13, 4, 13), (14, 1, 14), (15, 4, 15),
    ]]
    composition_resident = frozenset({1, 2, 3})
    check("_retile_resident_first_grouped: resident (with a rows tie), head, tail packed and "
          "flat-retiled, tail=3",
          _retile_resident_first_grouped(composition_tiles, composition_resident, 3),
          [[(1, 5, 100), (2, 5, 90), (3, 2, 40), (10, 3, 10), (12, 2, 12), (11, 1, 11),
            (13, 4, 13), (15, 4, 15)],
           [(14, 1, 14)]])
    check("_retile_resident_first_grouped: --sweep-tail changes the tail's membership (tail=1)",
          _retile_resident_first_grouped(composition_tiles, composition_resident, 1),
          [[(1, 5, 100), (2, 5, 90), (3, 2, 40), (13, 4, 13), (10, 3, 10), (12, 2, 12),
            (11, 1, 11), (14, 1, 14)],
           [(15, 4, 15)]])

    # The same composition through replay, tail=3 as above: chunk 1 fills
    # all 8 slots with 1, 2, 3, 20..24; chunk 2 is composition_tiles' 9
    # experts (1, 2, 3 resident, 10..15 absent). tile0 = 1 2 3 10 12 11 13
    # 15 (8 experts): 3 hits reserved, 5 misses against the 5 free slots
    # (20..24; nothing protected since only 14 remains for tile1 and it is
    # not yet resident). tile1 = 14: 1 miss (avoiding drops once it would
    # leave no eligible slot, the same fallback dataset 8c exercises).
    # Chunk 2: 3 hits, 6 misses; total across both chunks 3 hits, 14
    # misses (chunk 1 is 8 compulsory misses against the empty pool).
    composition_trace = "\n".join([
        "p 0 0 0 1 2 3 20 21 22 23 24 | " + " ".join("1:0" for _ in range(8)),
        "p 5 0 0 1 2 3 10 11 12 13 14 15 | "
        "5:100 5:90 2:40 3:10 1:11 2:12 4:13 1:14 4:15",
    ])
    stats_composition, _, _, _, _ = _run(composition_trace, slots=8, policy_raw="lfu",
                                         sweep_order="resident-first-grouped", sweep_tail=3)
    check("resident-first-grouped through replay: the composition's tiling drives real "
          "hits/misses",
          stats_composition[1]["prefill"][:2], [3, 14])

    # Dataset 8d: --sweep-order resident-first (the interleaved composition,
    # fix-up 1). All isolated calls use RETILE_SIZE=8 tiles.
    #
    # (i) R = 0 or A = 0: defers to _retile_resident_first_grouped at its
    # default tail (96), regardless of slots/head_factor (both given
    # nonsense values below to prove they are not consulted). With only 5
    # experts, well under the tail, the grouped composition's own head is
    # empty and its tail is the whole group: a single _pack_by_rows call.
    # Heaviest first, ties by expert id: 3 and 5 tie on rows (4), 3 wins;
    # then 1 (3), 4 (2), 2 (1).
    fallback_tiles = [[
        (1, 3, 10), (2, 1, 11), (3, 4, 12), (4, 2, 13), (5, 4, 14),
    ]]
    check("resident-first: empty resident defers to resident-first-grouped's own order",
          _retile_resident_first(fallback_tiles, frozenset(), slots=0, head_factor=0),
          [[(3, 4, 12), (5, 4, 14), (1, 3, 10), (4, 2, 13), (2, 1, 11)]])
    check("resident-first: all-resident chunk defers to the same grouped order",
          _retile_resident_first(fallback_tiles, frozenset({1, 2, 3, 4, 5}), slots=0, head_factor=0),
          [[(3, 4, 12), (5, 4, 14), (1, 3, 10), (4, 2, 13), (2, 1, 11)]])
    # A chunk larger than the default tail: the grouped composition's own
    # head/tail split is no longer a no-op, so this only holds as the
    # documented delegation, not as an identity with a single packed
    # group. 100 absent experts (ids 1..100, rows 1, last-row = id, so
    # already rank-ordered) exceed the default tail (96): the grouped
    # path's head keeps the 4 oldest (1..4) and its tail packs the other
    # 96 -- verified equal to calling the grouped composition directly.
    large_absent_tiles = [[(e, 1, e) for e in range(1, 101)]]
    check("resident-first: a chunk past the default tail still matches "
          "resident-first-grouped exactly, not a single packed group",
          _retile_resident_first(large_absent_tiles, frozenset(), slots=0, head_factor=0),
          _retile_resident_first_grouped(large_absent_tiles, frozenset(), DEFAULT_SWEEP_TAIL))

    # (i-b) the head search can also exhaust every tile without ever
    # breaking, leaving mixed_tiles at zero: R = 1, A = 100, T = ceil(101/8)
    # = 13, and at slots=1, head_factor=6 (default) the inequality fails
    # for every head 0..12, so the loop used to reach head=13 and return
    # only the head residents, silently dropping the whole absent group.
    # Run through replay (chunk 1 primes the pool's one slot with 999 so
    # chunk 2 sees it resident) and check that every one of the chunk's
    # 101 experts still reaches a pool.plan lookup, not just the resident.
    guard_absent = list(range(1, 101))
    guard_trace = "\n".join([
        "p 0 0 0 999 | 1:0",
        "p 5 0 0 " + " ".join(["999"] + [str(e) for e in guard_absent]) + " | "
        + " ".join(["1:0"] + [f"1:{e}" for e in guard_absent]),
    ])
    stats_guard, _, _, _, _ = _run(guard_trace, slots=1, policy_raw="lfu",
                                    sweep_order="resident-first")
    check("resident-first: the empty-mixed-tile guard keeps every absent expert of "
          "the chunk in the order (slots=1, R=1, A=100 exhausts the head search)",
          sum(stats_guard[1]["prefill"][:2]), 102)

    # (ii) h = 0: R = 4 residents (90..93, rows 9/7/5/3 so their heaviest-
    # first fill order is 90, 91, 92, 93 with no rows tie among them),
    # A = 20 absent (ids 1..20, last-row = id so already rank-ordered,
    # every row = 1). T = ceil(24 / 8) = 3. At h = 0: m = ceil(20 / 3) = 7,
    # 4 - 0 <= 50 - 1*7 = 43, so h = 0 (slots=50, head_factor=1). The
    # absent group spreads 6/7/7 across the 3 tiles (ranks 0..5, 6..12,
    # 13..19; the last tile holds 14..20, the most recent). Free slots per
    # tile: 2, 1, 1 (4 total, matching R). Filling heaviest-first: 90
    # (rows 9) goes to the lightest tile (bin0, weight 6) -> bin0 = 7,
    # weight 15; 91 (rows 7) ties bin1/bin2 at weight 7, lower index wins
    # (bin1) -> bin1 full, weight 14; 92 (rows 5) takes bin2 (only one open,
    # weight 7) -> bin2 full, weight 12; 93 (rows 3) takes the remaining
    # open slot, bin0 -> bin0 full, weight 18. Every tile ends at exactly 8
    # (the flat tile count is 3, matching ceil(24 / 8) with no re-tiling
    # needed since each bin is already RETILE_SIZE wide).
    resident_ids = [90, 91, 92, 93]
    resident_rows = {90: 9, 91: 7, 92: 5, 93: 3}
    h0_tiles = [[(e, resident_rows[e], 200 + e) for e in resident_ids]
                + [(e, 1, e) for e in range(1, 21)]]
    h0_resident = frozenset(resident_ids)
    h0_order = _retile_resident_first(h0_tiles, h0_resident, slots=50, head_factor=1)
    check("resident-first: h=0, uniform 6/7/7 absent spread, recency kept, ties to the "
          "lower tile index",
          h0_order,
          [[(e, 1, e) for e in range(1, 7)] + [(90, 9, 290), (93, 3, 293)],
           [(e, 1, e) for e in range(7, 14)] + [(91, 7, 291)],
           [(e, 1, e) for e in range(14, 21)] + [(92, 5, 292)]])
    check("resident-first: h=0 case's flat tile count matches ceil((R+A)/8)", len(h0_order), 3)

    # (iii) h > 0: R = 9 residents (1..9, last-row = 9+id so rank order is
    # 1..9), A = 3 absent (100..102, last-row = id so already rank-ordered).
    # T = ceil(12 / 8) = 2. At h = 0: m = ceil(3 / 2) = 2, 9 <= 10 - 1*2 = 8
    # is false; at h = 1: m = ceil(3 / 1) = 3, 9 - 8 = 1 <= 10 - 1*3 = 7 is
    # true (slots=10, head_factor=1), so h = 1: residents nearly fill the
    # pool (9 of 10 slots) and the head rule protects the one mixed tile.
    # Head = the 8 lowest-ranked residents (1..8, rank order, not
    # row-packed); the ninth resident (9, the highest-ranked / most recent)
    # is the only leftover, placed into the sole mixed tile alongside the
    # 3 absent experts.
    hpos_tiles = [[(e, 1, 9 + e) for e in range(1, 10)] + [(e, 1, e - 100) for e in range(100, 103)]]
    hpos_resident = frozenset(range(1, 10))
    hpos_order = _retile_resident_first(hpos_tiles, hpos_resident, slots=10, head_factor=1)
    check("resident-first: h>0 because residents nearly fill the pool",
          hpos_order,
          [[(e, 1, 9 + e) for e in range(1, 9)],
           [(100, 1, 0), (101, 1, 1), (102, 1, 2), (9, 1, 18)]])

    try:
        _run("p 0 0 0 1 2 3 | 1 1 1", slots=8, policy_raw="lfu", sweep_order="resident-first")
        failures.append("--sweep-order resident-first should error on a trace with no last-row counts")
    except ValueError:
        pass

    try:
        _run("p 0 0 0 1 2 3 | 1 1 1", slots=8, policy_raw="lfu",
             sweep_order="resident-first-grouped")
        failures.append("--sweep-order resident-first-grouped should error on a trace with no "
                        "last-row counts")
    except ValueError:
        pass

    try:
        _run("p 0 0 0 1 2 3 | 1 1 1", slots=8, policy_raw="lfu",
             sweep_order="resident-first-plain")
        failures.append("--sweep-order resident-first-plain should error on a trace with no "
                        "last-row counts")
    except ValueError:
        pass

    # Dataset 9: slru. slots=4, protected_share=0.5 (protected_capacity=2).
    # A is requested 3 times (2 hits -> promoted to protected, refreshed);
    # B once (stays in probation). C, D, E, F fill/evict; a re-request of A
    # should still hit (protected), while an early probation entry is the
    # one that gets recycled.
    slru_trace = "\n".join([
        "p 0 0 0 1", "p 0 0 1 1", "p 0 0 2 1",
        "p 0 0 3 2",
        "p 0 0 4 3", "p 0 0 5 4", "p 0 0 6 5", "p 0 0 7 6",
        "p 0 0 8 1",
    ])
    stats_slru, _, _, _, _ = _run(slru_trace, slots=4, policy_raw="slru:0.5")
    check("slru: A (promoted to protected) survives four probation churns",
          stats_slru[1]["prefill"][0], 3)

    # Dataset 10: arc. A pure LRU-ish access pattern (no repeats) should
    # behave like a plain miss-everything cache at first, then a repeated
    # expert should hit via T1/T2 promotion.
    arc_trace = "\n".join([
        "p 0 0 0 1", "p 0 0 1 2", "p 0 0 2 1",
    ])
    stats_arc, _, _, _, _ = _run(arc_trace, slots=4, policy_raw="arc")
    check("arc: repeated expert 1 hits", stats_arc[1]["prefill"][0], 1)
    check("arc: two compulsory misses", stats_arc[1]["prefill"][2], 2)

    # Dataset 11: lru-2. Expert A touched twice early then goes cold while
    # B and C alternate (2+ references each); a later capacity pressure
    # should prefer evicting the resident with fewest / oldest references.
    lru2_trace = "\n".join([
        "p 0 0 0 1", "p 0 0 1 1",
        "p 0 0 2 2", "p 0 0 3 2",
        "p 0 0 4 3",
    ])
    stats_lru2, _, _, _, _ = _run(lru2_trace, slots=2, policy_raw="lru-2")
    check("lru-2 baseline misses (A, B compulsory, C capacity)", stats_lru2[1]["prefill"][1], 3)
    lines_lru2_probe = _lines_from_text(lru2_trace + "\np 5 0 0 1")
    stats_lru2b, _, _, _, _ = replay(lines_lru2_probe, 2, parse_policy("lru-2"))
    check("lru-2: A (2nd-to-last reference oldest) evicted before B, so A re-misses",
          stats_lru2b[1]["prefill"][1], 4)

    # Dataset 12: --phase-policy. Prefill builds A to a high count over 3
    # touches and B once; a decode-phase miss for C then evicts by
    # decode's OWN policy (lru: oldest lastUse, i.e. A) rather than the
    # base/prefill policy (lfu: lowest count, i.e. B).
    phase_trace = "\n".join([
        "p 0 0 0 1", "p 0 0 1 1", "p 0 0 2 1", "p 0 0 3 2",
        "4 0 3",
    ])
    stats_phase, _, _, _, _ = _run(phase_trace, slots=2, policy_raw="lfu",
                                   phase_policy={"prefill": parse_policy("lfu"),
                                                 "decode": parse_policy("lru")})
    check("phase-policy: decode's lru evicted A, not lfu's choice of B",
          stats_phase[1]["decode"][1], 1)
    lines_phase_probe = _lines_from_text(phase_trace + "\n5 0 2")
    stats_phase_b, _, _, _, _ = replay(lines_phase_probe, 2, parse_policy("lfu"),
                                       phase_policy={"prefill": parse_policy("lfu"),
                                                     "decode": parse_policy("lru")})
    check("phase-policy: B (lru would not have picked it) still resident",
          stats_phase_b[1]["decode"][0], 1)

    # Dataset 13: --profile. Two requests (via `r`), each a 4-token decode
    # at window=2: request 1 misses everything (cold, slots=1) split 2/2
    # across the two windows; request 2 (positions restart at its own
    # first decode line) is identical.
    profile_trace = "\n".join([
        "r 0 1", "p 0 0 0 1",
        "1 0 2", "2 0 3", "3 0 4", "4 0 5",
        "r 5 1", "p 5 0 0 1",
        "6 0 2", "7 0 3", "8 0 4", "9 0 5",
    ])
    _, _, _, _, profile = _run(profile_trace, slots=1, policy_raw="lru", profile_window=2)
    check("profile req1 windows", {key: value for key, value in profile[1].items() if isinstance(key, int)}, {0: 2, 1: 2})
    check("profile req2 windows", {key: value for key, value in profile[2].items() if isinstance(key, int)}, {0: 2, 1: 2})

    # Dataset 14: --protect. Request 1's decode (slots=2, lfu) leaves slot A
    # holding expert 1 at count 2 (a hit bumped it) and slot B holding
    # expert 2 at count 1 -- the plain-LFU victim. Request 2's prefill chunk
    # has two tiles: tile0 needs new expert 3 (a miss), tile1 needs expert 2
    # (still resident, not yet planned). Under `chunk`, tile0's victim
    # selection excludes slot B (expert 2 is in `remaining`), so it takes
    # the worse-by-count slot A instead, and tile1 then hits; under `off`
    # tile0 evicts slot B as plain lfu would, and tile1 re-misses expert 2.
    protect_trace = "\n".join([
        "r 0 3", "0 0 1", "1 0 2", "2 0 1",
        "r 2 10", "p 3 0 0 3", "p 3 0 1 2",
    ])
    stats_protect_on, _, _, _, _ = _run(protect_trace, slots=2, policy_raw="lfu",
                                        protect="chunk")
    check("protect=chunk: expert 3 miss, expert 2 protected and hit",
          stats_protect_on[2]["prefill"], [1, 1, 1, 0])
    stats_protect_off, _, _, _, _ = _run(protect_trace, slots=2, policy_raw="lfu",
                                         protect="off")
    check("protect=off: expert 2 evicted by plain lfu, then re-misses",
          stats_protect_off[2]["prefill"], [0, 2, 1, 1])

    # Dataset 15: --protect's starvation fallback. Request 1's decode fills
    # all four slots (experts 1-4, count 1 each). Request 2's prefill
    # chunk's first tile needs new expert 9 (a miss); its other four tiles
    # need experts 1-4, all four still resident and all in `remaining` when
    # tile0 plans, so every slot is nominally protected -- zero eligible --
    # and the plan must fall back to plain lfu rather than fail. A first
    # tile's `avoiding` (the lookback) is always empty (just cleared for
    # the new chunk), so this is protection's own fallback in isolation:
    # dropping `protect` for that one plan leaves exactly the eligible set
    # `--protect off` would have used from the start, so the two runs give
    # the identical (if unglamorous) result below -- the graceful
    # degradation working as designed, not a test that found no effect.
    # The cascade through the chunk's remaining four tiles (each request
    # re-missing the expert the previous tile's own eviction just took)
    # is this replay tool's already-documented avoidingSlots-lookback
    # behavior, unrelated to protection.
    fallback_trace = "\n".join([
        "r 0 5", "0 0 1", "1 0 2", "2 0 3", "3 0 4",
        "r 4 10", "p 5 0 0 9", "p 5 0 1 1", "p 5 0 2 2", "p 5 0 3 3", "p 5 0 4 4",
    ])
    stats_fallback_on, _, _, _, _ = _run(fallback_trace, slots=4, policy_raw="lfu",
                                         protect="chunk")
    check("protect=chunk starvation fallback: plan still succeeds",
          stats_fallback_on[2]["prefill"], [0, 5, 1, 4])
    stats_fallback_off, _, _, _, _ = _run(fallback_trace, slots=4, policy_raw="lfu",
                                          protect="off")
    check("protect=chunk under full starvation matches protect=off exactly",
          stats_fallback_on[2]["prefill"], stats_fallback_off[2]["prefill"])

    if failures:
        print(f"SELF-TEST FAILED ({len(failures)} of many checks):")
        for f in failures:
            print(f"  {f}")
        return 1
    print("self-test: all checks passed")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Replay a SHRIKE_ROUTE_TRACE capture against an expert-pool eviction policy.")
    parser.add_argument("trace", nargs="?", help="path to a SHRIKE_ROUTE_TRACE file")
    parser.add_argument("--slots", type=int, default=DEFAULT_SLOTS,
                         help=f"slots per layer (default {DEFAULT_SLOTS})")
    parser.add_argument("--policy", default="aging-lfu",
                         help="lru | lfu | aging-lfu[:period] | belady | slru[:share] | "
                              "arc | lru-2")
    parser.add_argument("--layer", type=int, default=None,
                         help="restrict the replay to one layer")
    parser.add_argument("--avoid-lookback", type=int, default=DEFAULT_AVOID_LOOKBACK,
                         help="preceding tiles whose slots are ineligible as victims "
                              f"(default {DEFAULT_AVOID_LOOKBACK}, the production bound)")
    parser.add_argument("--prefill-weight", choices=["one", "rows"], default="one",
                         help="a prefill plan's expertUseCount increment: 1 (production) "
                              "or the expert's row count (needs a trace with row counts)")
    parser.add_argument("--sweep-order",
                         choices=["index", "rows-asc", "rows-desc", "last-asc", "last-desc",
                                  "resident-first", "resident-first-grouped",
                                  "resident-first-plain"],
                         default="index",
                         help="index replays each chunk's recorded tiles; rows-asc/rows-desc "
                              "re-tile a chunk by row count (needs row counts); last-asc/"
                              "last-desc re-tile by each expert's last row in the chunk, ties "
                              "by rows ascending then expert id (needs last-row counts); "
                              "resident-first splits the chunk into resident and absent "
                              "pool-resident groups (rank order kept in each), gives the "
                              "residents a pure head only while --sweep-head-factor tiles' "
                              "worth of misses would starve protection, spreads the absent "
                              "group uniformly by recency across the remaining tiles, and "
                              "fills their free slots with the leftover residents "
                              "heaviest-first, tiled flat; resident-first-grouped is round "
                              "1's landed order (resident/head/tail, the absent group's "
                              "--sweep-tail most recent experts as the tail, each packed by "
                              "row weight and tiled flat); resident-first-plain is the same "
                              "split with no row-weight packing, resident group then absent "
                              "group each in last-asc order (all three need last-row counts; "
                              "--sweep-carry has no effect on any)")
    parser.add_argument("--sweep-tail", type=int, default=DEFAULT_SWEEP_TAIL,
                         help="the absent group's most recent experts packed as their own "
                              f"group under --sweep-order resident-first-grouped (default "
                              f"{DEFAULT_SWEEP_TAIL}, the replay's own sweep tail); "
                              "must be a positive integer")
    parser.add_argument("--sweep-head-factor", type=int, default=DEFAULT_SWEEP_HEAD_FACTOR,
                         help="the tile multiplier --sweep-order resident-first's head rule "
                              f"protects against starving (default {DEFAULT_SWEEP_HEAD_FACTOR}); "
                              "must be a positive integer")
    parser.add_argument("--sweep-carry", choices=["off", "on"], default="off",
                         help="alternate the sweep-order direction chunk to chunk (off: fixed)")
    parser.add_argument("--protect", choices=["off", "chunk"], default="chunk",
                         help="chunk (default, matching production): a prefill tile's victim "
                              "selection also excludes a slot holding an expert the chunk's "
                              "not-yet-replayed tiles still need, with the streamer's own "
                              "graded fallback; off replays captures from before chunk "
                              "protection (v13 T4)")
    parser.add_argument("--phase-policy", default=None,
                         help="prefill=<policy>,decode=<policy>, restricted to "
                              f"{sorted(PHASE_POLICY_ALLOWED)}")
    parser.add_argument("--profile", type=int, default=None, dest="profile_window",
                         help="print decode misses per this many positions, per request")
    parser.add_argument("--expect", default=None,
                         help="file of measured miss counts, one request per line "
                              "('prefill_misses decode_misses', or one integer for a "
                              "decode-only comparison), to print beside the replayed counts")
    parser.add_argument("--speculative-fills", default=None, metavar="PREFETCH_TRACE",
                        help="a SHRIKE_PREFETCH_TRACE capture from the same lifetime: the "
                             "router probe's prediction for layer L + d at each position is "
                             "filled into L + d's pool before its plan there (v15 Task 2's "
                             "speculative landing, modelled)")
    parser.add_argument("--speculative-fills-2", default=None, metavar="PREFETCH_TRACE_D2",
                        help="with --speculative-fills: the same lifetime's distance-2 capture; "
                             "a layer's window serves its next layer's prediction, or the layer "
                             "after that when the next needs nothing (the two-distance queue)")
    parser.add_argument("--queue-mode", choices=["free", "chained"], default="free",
                        help="with --speculative-fills-2: 'free' spends a window on the layer "
                             "after next only when the next needs nothing, 'chained' spends every "
                             "window on both in sequence (default free)")
    parser.add_argument("--fill-top-m", type=int, default=8,
                        help="prefix of the prediction considered per layer (default 8)")
    parser.add_argument("--fill-budget", type=int, default=1,
                        help="fills placed per layer per position (default 1, the ring's "
                             "one read in flight)")
    parser.add_argument("--fill-mode", choices=["pool", "ring", "ring-retain"], default="pool",
                        help="where a fill lives: pool (default) lands in a victim slot and "
                             "stays (the landing in the pool's own slot); ring is held beside "
                             "the pool until its layer's plan, a hit there, then gone (the ring "
                             "addressable by the classifier, nothing retained); ring-retain "
                             "then places a hit expert in a victim slot (the ring with "
                             "adoption before v16's merge, and the merge's swap)")
    parser.add_argument("--table-fills", default=None, metavar="TOKENS_JSON",
                        help="v20: the token-id table as fills, keyed by the streamed piece per "
                             "decode position of the trace's first request (a rig tokens-*.json); "
                             "fills go through --fill-mode ring with --table-cells per layer")
    parser.add_argument("--table-layers", default="all",
                        help="with --table-fills: the served layers, e.g. 0,30-39 (default all)")
    parser.add_argument("--table-width", type=int, default=8,
                        help="with --table-fills: the prediction's cap per layer (default 8)")
    parser.add_argument("--table-source", choices=["last", "last2", "last3", "freq", "none"],
                        default="last",
                        help="with --table-fills: the entry is the last occurrence's route, the "
                             "union of the last two or three, or the most frequent experts; none "
                             "keeps the table empty so --union-previous is priced alone")
    parser.add_argument("--table-cells", type=int, default=None,
                        help="with --table-fills: the cell budget per layer per position "
                             "(default: --table-width)")
    parser.add_argument("--union-previous", action="store_true",
                        help="with --table-fills: add the previous position's route at the same "
                             "layer to the prediction")
    parser.add_argument("--draft", default=None, metavar="prompt-lookup:N",
                        help="with --table-fills: key --draft-layers by the prompt-lookup draft "
                             "(the piece after the last earlier occurrence of the N-gram ending "
                             "at the previous piece) instead of the real piece; no draft, no fill")
    parser.add_argument("--draft-layers", default="0",
                        help="with --draft: the layers keyed by the draft (default 0)")
    parser.add_argument("--prompt-pieces", default=None, metavar="TOKENIZE_JSON",
                        help="with --draft or --table-seed: the prompt's pieces from "
                             "`shrike generate --tokenize`, so the draft matches into the prompt")
    parser.add_argument("--table-seed", choices=["none", "prefill"], default="none",
                        help="with --table-fills and --prompt-pieces: seed the table from the "
                             "capture's q lines (the prefill's per-token routes; v20 S0.5)")
    parser.add_argument("--table-future", action="store_true",
                        help="with --table-fills and --policy belady: the clairvoyant path sees "
                             "the table's predictions as the future in place of the trace's, and "
                             "no fill is placed (v20 S0.4, knowledge in the policy)")
    parser.add_argument("--table-protect", action="store_true",
                        help="with --table-fills: the online form at a horizon of one, a decode "
                             "plan's victims exclude the experts the table predicts for the "
                             "layer's next position; no fill is placed (v20 S0.4)")
    parser.add_argument("--slots-json", default=None, metavar="FILE",
                        help="a JSON map of layer -> slots overriding --slots per layer "
                             "(v20 S0.4, the split)")
    parser.add_argument("--self-test", action="store_true",
                         help="run the built-in synthetic-trace checks and exit")
    args = parser.parse_args()

    if args.self_test:
        sys.exit(self_test())

    if not args.trace:
        parser.error("a trace path is required unless --self-test is given")

    if args.sweep_tail <= 0:
        parser.error("--sweep-tail must be a positive integer")

    if args.sweep_head_factor <= 0:
        parser.error("--sweep-head-factor must be a positive integer")

    lines = load_trace(args.trace)
    policy = parse_policy(args.policy)
    if args.speculative_fills and args.speculative_fills_2:
        fills = load_two_distance_queue(args.speculative_fills, args.speculative_fills_2,
                                        args.fill_top_m, chained=args.queue_mode == "chained")
    elif args.speculative_fills:
        fills = load_prefetch_fills(args.speculative_fills, args.fill_top_m)
    else:
        fills = None
    table_info = None
    table_layers = None
    fill_budget = args.fill_budget
    fill_mode = args.fill_mode
    if args.table_fills:
        if fills is not None:
            parser.error("--table-fills cannot be combined with --speculative-fills")
        table_layers = parse_layer_set(args.table_layers)
        draft_n = 0
        draft_layers = frozenset()
        if args.draft:
            name, _sep, count = args.draft.partition(":")
            if name != "prompt-lookup" or not count.isdigit() or int(count) < 1:
                parser.error("--draft takes prompt-lookup:N with N a positive integer")
            draft_n = int(count)
            draft_layers = parse_layer_set(args.draft_layers)
        prompt_pieces = load_pieces(args.prompt_pieces) if args.prompt_pieces else None
        if args.draft and prompt_pieces is None:
            parser.error("--draft needs --prompt-pieces so the draft can match into the prompt")
        seed_routes = None
        if args.table_seed == "prefill":
            if prompt_pieces is None:
                parser.error("--table-seed prefill needs --prompt-pieces")
            seed_routes = load_seed_routes(args.trace)
            if not seed_routes:
                parser.error("--table-seed prefill: the trace carries no q lines")
        table_info = {}
        try:
            fills = build_table_fills(
                lines, load_pieces(args.table_fills), table_layers,
                width=args.table_width, source=args.table_source,
                union_previous=args.union_previous, draft_n=draft_n,
                draft_layers=draft_layers, prompt_pieces=prompt_pieces,
                seed_routes=seed_routes, info=table_info)
        except ValueError as e:
            parser.error(str(e))
        fill_budget = args.table_cells if args.table_cells is not None else args.table_width
        # A landing the plan wants is swapped into the pool (v16's merge), so
        # the table's fills follow production's ring-retain profile unless told otherwise.
        fill_mode = args.fill_mode if args.fill_mode != "pool" else "ring-retain"
    predicted_future = None
    if args.table_future:
        if table_info is None or policy.name != "belady":
            parser.error("--table-future needs --table-fills and --policy belady")
        predicted_future = fills
        fills = None
        table_info = None
    predicted_protect = None
    if args.table_protect:
        if table_info is None:
            parser.error("--table-protect needs --table-fills")
        predicted_protect = fills
        fills = None
        table_info = None
    slots = args.slots
    if args.slots_json:
        with open(args.slots_json) as handle:
            slots = {int(layer): int(count) for layer, count in json.load(handle).items()}
    fill_stats = {} if fills is not None else None
    phase_policy = None
    if args.phase_policy:
        try:
            phase_policy = parse_phase_policy(args.phase_policy)
        except ValueError as e:
            parser.error(str(e))

    try:
        stats, total_compulsory, settle_stats, settle_meta, profile = replay(
            lines, slots, policy, args.layer, args.avoid_lookback,
            args.prefill_weight, args.sweep_order, args.sweep_carry == "on",
            phase_policy, args.profile_window, protect=args.protect,
            sweep_tail=args.sweep_tail, sweep_head_factor=args.sweep_head_factor,
            fills=fills, fill_budget=fill_budget, fill_stats=fill_stats,
            fill_mode=fill_mode, predicted_future=predicted_future,
            predicted_protect=predicted_protect)
    except ValueError as e:
        parser.error(str(e))

    print_report(stats, total_compulsory, policy, args.slots, args.layer,
                 args.avoid_lookback, settle_stats, settle_meta, protect=args.protect)
    if fill_stats is not None and table_info is None:
        print(f"  speculative fills (top-m={args.fill_top_m} budget={fill_budget} "
              f"mode={fill_mode}): "
              f"placed={fill_stats.get('fills', 0)} useful={fill_stats.get('useful', 0)} "
              f"wasted={fill_stats.get('wasted', 0)} "
              f"unused_at_end={fill_stats.get('unused_at_end', 0)}")
    if table_info is not None:
        print(f"  table (source={args.table_source} width={args.table_width} cells={fill_budget} "
              f"union_previous={'on' if args.union_previous else 'off'} "
              f"seed={args.table_seed}): placed={fill_stats.get('fills', 0)} "
              f"useful={fill_stats.get('useful', 0)} wasted={fill_stats.get('wasted', 0)}")
        print_table_report(table_info, fill_stats, table_layers)
    if args.profile_window:
        print_profile(profile, args.profile_window)
    if args.expect:
        print_expect_deltas(stats, args.expect)


if __name__ == "__main__":
    main()
