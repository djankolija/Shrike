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
    pending batches at fetch depth 2. With the shipped defaults
    (`SHRIKE_PREFILL_TILE_DEPTH` unset -> maxPendingDepth 2,
    `SHRIKE_PREFILL_TILE_BATCH` unset -> 1 tile per batch,
    `SHRIKE_PREFILL_FETCH_DEPTH` unset -> fetchLookahead 1), tracing the
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
    chunk within a layer (matching `PrefillSweepMode.carry`'s per-chunk
    flip, which never resets across a request boundary); `off` (default)
    uses the same direction for every chunk. This re-tiling cannot
    reproduce the real scheduler's own tile composition (which follows the
    routed groups' natural order and packs by slot-budget fit, not a fixed
    width of 8) or its `avoidingSlots` (recomputed here from the new,
    synthetic tile boundaries, not the real batch/commit schedule).

Usage:
  expert-pool-replay.py <trace> --policy lru|lfu|aging-lfu[:period]|belady|
                         slru[:share]|arc|lru-2
                         [--slots N] [--layer L] [--avoid-lookback N]
                         [--prefill-weight one|rows]
                         [--sweep-order index|rows-asc|rows-desc|last-asc|last-desc]
                         [--sweep-carry off|on]
                         [--phase-policy prefill=<policy>,decode=<policy>]
                         [--profile WINDOW]
  expert-pool-replay.py <trace> --policy ... --expect <file>
  expert-pool-replay.py --self-test
"""
import argparse
import bisect
import sys
from collections import OrderedDict, defaultdict, deque

DEFAULT_SLOTS = 128
DEFAULT_AGING_PERIOD = 1024
# maxPendingDepth (2, unset SHRIKE_PREFILL_TILE_DEPTH) + 1, the steady-state
# held-tile count at 1 tile per batch and fetch depth 2 (see module docstring).
DEFAULT_AVOID_LOOKBACK = 3
DEFAULT_SLRU_PROTECTED_SHARE = 0.5
RETILE_SIZE = 8


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
    position, layer = int(parts[0]), int(parts[1])
    experts = [int(x) for x in parts[2:]]
    return "decode", position, layer, None, experts, None, None


def load_trace(path):
    lines = []
    with open(path) as f:
        for raw in f:
            parsed = parse_line(raw)
            if parsed is not None:
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

    def plan(self, experts, avoiding=frozenset(), policy_override=None, weights=None):
        """Places `experts`, returns (hits, misses, assigned_slots)."""
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
                    break

        miss_indices = [i for i in range(len(experts)) if assigned[i] == -1]
        eligible = [s for s in range(self.slots)
                    if s not in reserved and s not in avoiding]
        if len(miss_indices) > len(eligible):
            eligible = [s for s in range(self.slots) if s not in reserved]

        eligible.sort(key=lambda slot: self._victim_key(slot, active))
        victims = eligible[:len(miss_indices)]

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
            self.slot_expert[slot] = experts[miss_i]
            self.slot_last_use[slot] = clock
            assigned[miss_i] = slot

        self.plans_done += 1
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

    def plan(self, experts, avoiding=frozenset(), policy_override=None, weights=None):
        result = super().plan(experts, avoiding, policy_override, weights)
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

    def plan(self, experts, avoiding=frozenset(), policy_override=None, weights=None):
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
            slot = self._select_victim(avoiding)
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

    def _select_victim(self, avoiding):
        for slot in list(self.probation.keys()):
            if slot not in avoiding:
                del self.probation[slot]
                return slot
        if self.probation:
            slot = next(iter(self.probation))
            del self.probation[slot]
            return slot
        for slot in list(self.protected.keys()):
            if slot not in avoiding:
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

    def plan(self, experts, avoiding=frozenset(), policy_override=None, weights=None):
        newly_seen = [expert not in self.seen_experts for expert in experts]
        self.seen_experts.update(experts)
        assigned = [-1] * len(experts)
        hits = 0
        misses = 0
        for i, expert in enumerate(experts):
            slot, hit = self._request(expert, avoiding)
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

    def _lru_pop(self, ordered, avoiding):
        for key in list(ordered.keys()):
            slot = self.expert_slot.get(key)
            if slot is None or slot not in avoiding:
                del ordered[key]
                return key
        key = next(iter(ordered))
        del ordered[key]
        return key

    def _replace(self, favor_t2, avoiding):
        use_t1 = bool(self.t1) and (len(self.t1) > self.p
                                    or (favor_t2 and len(self.t1) == self.p))
        source = self.t1 if use_t1 else self.t2
        if not source:
            source = self.t2 if source is self.t1 else self.t1
        evicted = self._lru_pop(source, avoiding)
        slot = self.expert_slot.pop(evicted)
        self.slot_expert[slot] = -1
        if evicted >= 0:
            ghost = self.b1 if source is self.t1 else self.b2
            ghost[evicted] = None
        return slot

    def _request(self, expert, avoiding):
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
            slot = self._replace(False, avoiding)
        elif expert in self.b2:
            del self.b2[expert]
            delta = max(1, len(self.b1) // len(self.b2)) if self.b2 else max(1, len(self.b1))
            self.p = max(0, self.p - delta)
            slot = self._replace(True, avoiding)
        else:
            total_t1_b1 = len(self.t1) + len(self.b1)
            total_all = total_t1_b1 + len(self.t2) + len(self.b2)
            if total_t1_b1 == self.c:
                if len(self.t1) < self.c:
                    if self.b1:
                        del self.b1[next(iter(self.b1))]
                    slot = self._replace(False, avoiding)
                else:
                    evicted = self._lru_pop(self.t1, avoiding)
                    slot = self.expert_slot.pop(evicted)
                    self.slot_expert[slot] = -1
            elif total_t1_b1 < self.c and total_all >= self.c:
                if total_all >= 2 * self.c and self.b2:
                    del self.b2[next(iter(self.b2))]
                slot = self._replace(False, avoiding)
            else:
                slot = self._replace(False, avoiding)
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


def replay(lines, slots, policy, layer_filter=None, avoid_lookback=DEFAULT_AVOID_LOOKBACK,
           prefill_weight="one", sweep_order="index", sweep_carry=False,
           phase_policy=None, profile_window=None):
    """Returns (stats, total_compulsory, settle_stats, settle_meta, profile).

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
    if sweep_order in ("last-asc", "last-desc") and not has_last_rows(lines):
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
        future = build_future_occurrences(
            [experts for _k, _p, _t, experts, _rc, _lr, _l in layer_lines])
        pool = make_pool(slots, policy, future)
        lookback = deque(maxlen=avoid_lookback) if avoid_lookback > 0 else deque()
        next_reverse = sweep_order in ("rows-desc", "last-desc")

        for item in group_into_chunks(layer_lines):
            if item[0] == "decode":
                _kind, position, experts, label = item
                active = phase_policy["decode"] if phase_policy else None
                before = (pool.compulsory, pool.capacity)
                hits, misses, _assigned = pool.plan(experts, policy_override=active)
                _accumulate(stats, settle_stats, label, hits, misses,
                           pool.compulsory - before[0], pool.capacity - before[1])
                if profile_window and label[0] == "request":
                    base = first_decode_position.get(label[1], position)
                    window_index = (position - base) // profile_window
                    profile[label[1]][window_index] += misses
                lookback.clear()
                continue

            _kind, original_tiles, label = item
            if sweep_order == "index":
                new_tiles = original_tiles
            else:
                reverse = (next_reverse if sweep_carry
                          else sweep_order in ("rows-desc", "last-desc"))
                new_tiles = _retile(original_tiles, sweep_order, reverse)
                if sweep_carry:
                    next_reverse = not next_reverse

            lookback.clear()
            active = phase_policy["prefill"] if phase_policy else None
            for new_tile in new_tiles:
                tile_experts = [expert for expert, _rows, _last in new_tile]
                tile_rows = [rows for _expert, rows, _last in new_tile]
                avoiding = frozenset(slot for _tile, held in lookback for slot in held)
                weights = tile_rows if prefill_weight == "rows" else None
                before = (pool.compulsory, pool.capacity)
                hits, misses, assigned = pool.plan(
                    tile_experts, avoiding=avoiding, policy_override=active, weights=weights)
                _accumulate(stats, settle_stats, label, hits, misses,
                           pool.compulsory - before[0], pool.capacity - before[1])
                lookback.append((None, [slot for slot in assigned if slot >= 0]))
        total_compulsory += pool.compulsory

    return stats, total_compulsory, settle_stats, settle_meta, profile


def print_report(stats, total_compulsory, policy, slots, layer_filter, avoid_lookback,
                 settle_stats=None, settle_meta=None):
    layer_note = f" layer={layer_filter}" if layer_filter is not None else ""
    print(f"policy={policy.label()} slots={slots} avoid_lookback={avoid_lookback}{layer_note}")
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
    print(f"  --profile {window}: decode misses per window, summed over replayed layers")
    for request_id in sorted(profile):
        windows = profile[request_id]
        last = max(windows) if windows else -1
        values = [windows.get(i, 0) for i in range(last + 1)]
        print(f"    request {request_id}: " + "/".join(str(v) for v in values))


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
    check("profile req1 windows", dict(profile[1]), {0: 2, 1: 2})
    check("profile req2 windows", dict(profile[2]), {0: 2, 1: 2})

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
                         choices=["index", "rows-asc", "rows-desc", "last-asc", "last-desc"],
                         default="index",
                         help="index replays each chunk's recorded tiles; rows-asc/rows-desc "
                              "re-tile a chunk by row count (needs row counts); last-asc/"
                              "last-desc re-tile by each expert's last row in the chunk, ties "
                              "by rows ascending then expert id (needs last-row counts)")
    parser.add_argument("--sweep-carry", choices=["off", "on"], default="off",
                         help="alternate the sweep-order direction chunk to chunk (off: fixed)")
    parser.add_argument("--phase-policy", default=None,
                         help="prefill=<policy>,decode=<policy>, restricted to "
                              f"{sorted(PHASE_POLICY_ALLOWED)}")
    parser.add_argument("--profile", type=int, default=None, dest="profile_window",
                         help="print decode misses per this many positions, per request")
    parser.add_argument("--expect", default=None,
                         help="file of measured miss counts, one request per line "
                              "('prefill_misses decode_misses', or one integer for a "
                              "decode-only comparison), to print beside the replayed counts")
    parser.add_argument("--self-test", action="store_true",
                         help="run the built-in synthetic-trace checks and exit")
    args = parser.parse_args()

    if args.self_test:
        sys.exit(self_test())

    if not args.trace:
        parser.error("a trace path is required unless --self-test is given")

    lines = load_trace(args.trace)
    policy = parse_policy(args.policy)
    phase_policy = None
    if args.phase_policy:
        try:
            phase_policy = parse_phase_policy(args.phase_policy)
        except ValueError as e:
            parser.error(str(e))

    try:
        stats, total_compulsory, settle_stats, settle_meta, profile = replay(
            lines, args.slots, policy, args.layer, args.avoid_lookback,
            args.prefill_weight, args.sweep_order, args.sweep_carry == "on",
            phase_policy, args.profile_window)
    except ValueError as e:
        parser.error(str(e))

    print_report(stats, total_compulsory, policy, args.slots, args.layer,
                 args.avoid_lookback, settle_stats, settle_meta)
    if args.profile_window:
        print_profile(profile, args.profile_window)
    if args.expect:
        print_expect_deltas(stats, args.expect)


if __name__ == "__main__":
    main()
