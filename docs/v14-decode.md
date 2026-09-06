# v14, decode pass II: the answer's wall on the shapes a card actually answers

## The problem

v13 ([v13-the-turn.md](v13-the-turn.md)) closed the turn's prefill: on the mini a
warm first turn of 300 / 1k / 2k tokens went 5.46 / 8.05 / 11.21 s to 3.10 / 5.90 /
10.29, and a follow-up turn 2.13 to 1.39 s. What is left of a card is the answer.
Step zero measured three of them on the deployed build, and the split is not close:

| request (mini, production launch) | prompt / completion | wall | prefill_s | decode_s | decode tok/s | decode's share of the wall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| the 2k card's tX | 2,125 / 219 | 29.308 s | 11.102 | 16.190 | 13.53 | 55 % |
| the 300 prompt | 289 / 314 | 29.523 s | 5.492 | 21.952 | 14.30 | 74 % |
| the 1k prompt | 1,069 / 405 | 37.885 s | 7.945 | 27.924 | 14.50 | 74 % |

Source: `step0-out/step0-{card,d512-300,d512-1k}/server-mini-step0-*.log`, the
`Shrike generation` and request-completion lines. An LLMBench card answers 600 to
800 tokens ([v13-the-turn.md](v13-the-turn.md) "The problem"), so at these rates one
card answer is **44.6 to 59.4 s**, and a tool-heavy card pays it once per round.

The chapter's unit is therefore **the wall of one decoded token**, decomposed, not a
rate quoted in the abstract. Step zero decomposes it three ways that agree, and the
answer is blunt: **of a 69 to 74 ms token, 45 ms is GPU work and 26 ms is the GPU
standing still**, most of it in one named window where the routed stage waits for an
expert to arrive from the SSD. A token that never missed would run at 22.2 tok/s on
the card's shape (its own GPU busy time), against the 13.5 measured: **the same 700
tokens would take 31.5 s instead of 52.0 s.**

## Step zero: the ledger (mini, v13's close build, production launch)

Measurement only, no code, no branch, one server lifetime per shape. Build: the
binary of `b13fbfe`, the pre-amend form of v13's close commit `c04e43b` (the
two-loop collapse); the fix wave that followed it (`e0bba79`) changed no decode
path and v13 closed golden-identical, merged to `main` as `5316d2a`, this
branch's base. Box: the Mac
mini (16 GB), the bare production launch (`--ram-budget 8G`, 128 expert slots per
layer, `SHRIKE_PREFILL_SWEEP=resident`, `SHRIKE_EXPERT_CACHE_PROTECT=chunk`,
`aging-lfu`, fetch depth 2) plus `SHRIKE_RUNNER_STATS=1 SHRIKE_KERNEL_STATS=1` and
`SHRIKE_ROUTE_TRACE`, with the cold request STREAMED so every token's arrival time
is recorded beside its route trace. Every number is MEASURED unless marked modelled.
Raw data at `~/.claude/handoffs/archive/shrike-v14-step0/`: `step0-out/<shape>/`
(the token arrivals, the route trace, the server log, `layer-misses.json`),
`ledger/` (the probe output, the fits, the replayed profiles), `loop-files/` (every
script). The rows below and their sources are also in that archive's
`step0-ledger.md`, and the session ledger's "Decode step zero" entries carry them
(`.superpowers/sdd/v13-implementation-plan/progress.md`).

**A note on the `file:line` citations in this document.** Every one was re-verified
against this branch's base, commit `5316d2a` (the merged v13 close), when the
document was copied onto the branch on 2026-09-06: the citations into
`RealForwardRunner.swift` and `tools/expert-pool-replay.py` had drifted by a
uniform offset since the draft (v13's close fix round) and were re-anchored; every
other file's held. The exception is `docs/v10-implementation-plan.md`, cited
against the working-tree copy carrying the peer's uncommitted P3 follow-on (a
50-line insertion after its line 155); those resolve exactly once that edit lands.
The symbol names are the stable part.

### 1. Where a decoded token's wall goes

Per new token, from the server's own kernel-role and gap counters (every `*_ms`
field in a `Shrike runner` line and every `per_token_ms` in a `Shrike kernel` /
`Shrike gap` line is divided by the request's new-token count,
`ServerInference.swift:1960-1962` and `:2042-2073`), beside the wall measured from
the streamed arrivals:

| term (ms per token) | the 2k card | the 300 prompt | the 1k prompt | source |
| --- | ---: | ---: | ---: | --- |
| wall (streamed arrivals, mean) | 74.27 | 70.15 | 69.08 | `ledger/step0-fit-aligned.txt` |
| GPU busy, decode roles summed | 44.99 | 41.52 | 42.20 | the `Shrike kernel role=` lines |
| ... attention linear (GDN layers) | 15.57 | 15.77 | 15.29 | " |
| ... attention KV | 9.84 | 5.66 | 7.44 | " |
| ... routed MoE, speculative pass | 8.98 | 9.06 | 9.22 | " |
| ... LM head (`head_logits`) | 4.89 | 5.05 | 4.82 | " |
| ... routed phase-1 hit + miss fixup | 5.54 | 5.81 | 5.24 | " |
| ... sample + embed | 0.18 | 0.18 | 0.17 | " |
| **GPU idle: `moe_phase1_hit` to `moe_phase1_miss_fixup_phase2`** | **19.68** | **19.24** | **18.03** | the `Shrike gap` lines |
| GPU idle: `moe_spec_routed` to `moe_phase1_hit` | 4.13 | 4.19 | 3.84 | " |
| GPU idle: `moe_spec_routed` to `attn_layer_linear` | 1.40 | 1.18 | 1.25 | " |
| GPU idle: `attn_layer_linear` to `moe_spec_routed` | 1.12 | 1.11 | 1.12 | " |
| GPU idle: other listed decode transitions | 0 | 0.45 | 0.43 | " |
| unaccounted (gaps below the log's top-8 cut) | 2.95 | 2.46 | 2.21 | by difference |

**Readings.** (1) The ledger closes to 3 to 4 %, and the only unlisted terms are
transitions the log truncates at eight entries, so nothing large is hiding.
(2) **The GPU is idle 35 % of a decoded token, and three quarters of that idle is one
transition**: the phase-1 hit kernel has finished the layer's resident experts and
the miss-fixup kernel cannot start until the missing experts have landed. Its
`host_ms` is **0.0** on all three shapes, so the host is never late to submit: the
command buffer is waiting on the shared event the reader signals. (3) The decode
GPU busy total is the same quantity as the regression's miss-free body below
(44.99 against 44.89, 41.52 against 41.28, 42.20 against 42.62), which is two
instruments agreeing on what a token would cost with a perfect pool.

### 2. The wall per token is body plus 0.85 ms per expert miss

Every decode plan's misses per (position, layer) from `tools/expert-pool-replay.py`
at production's configuration (`loop-files/step0-layer-misses.py` imports the tool
rather than editing it; the replay reproduces the box's decode miss totals exactly
on all three shapes: 6,689 / 9,468 / 11,398), joined to the streamed arrivals by
`loop-files/step0-fit.py`. A decode position k in the trace is the pass that emits
token k+1, whose wall is arrival[k+1] minus arrival[k].

| answer (prompt rows, joined tokens) | decode s, tok/s | misses per token | fit: body + cost x misses | r², rms | first miss in a layer / each further one | layers with a miss per token | modelled exposed ms per token |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: |
| the 2k card (2,125 rows, 217) | 16.19 s, 13.53 | 30.7 | 48.8 + 0.831 | 0.88, 4.9 ms | 1.19 / 0.61 ms | 18.3 of 40 | 29.4 of 74.3 (40 %) |
| the 300 prompt (289 rows, 312) | 21.95 s, 14.30 | 30.3 | 44.0 + 0.865 | 0.89, 4.7 ms | 1.10 / 0.72 ms | 18.8 | 28.9 of 70.2 (41 %) |
| the 1k prompt (1,069 rows, 403) | 27.92 s, 14.50 | 28.2 | 45.5 + 0.837 | 0.92, 4.0 ms | 1.10 / 0.67 ms | 17.4 | 26.5 of 69.1 (38 %) |

Source: `ledger/step0-fit-aligned.txt`. A term for layers with three or more misses
adds nothing. The structural fit (two features) is the one the chapter uses: the
**first miss in a layer** is a read the layer waits for serially, and each **further
miss in the same layer** is overlapped by the reader's four threads and costs 0.61
to 0.72 ms rather than nothing.

Three instruments now price the same object and agree: the regression's first-miss
coefficient (1.19 / 1.10 / 1.10 ms), the box's own `io_ms` divided by
`hit_fixup_layers` (1.19 / 1.16 / 1.16 ms), and the kernel-gap counter divided by
the same layer count (**1.08 / 1.03 / 1.04 ms**). The box's `hit_fixup_layers` per
token (18.18 / 18.70 / 17.39) is the replay's "layers with a miss per token" (18.3 /
18.8 / 17.4) to within 1 %, which is what lets the two ledgers be added.

Of the gap's 1.03 to 1.08 ms per missing layer, **0.165 to 0.176 ms is spent after
the bytes have landed**: `io_fixup_wake_ms` (3.20 / 3.14 / 2.88 ms per token) is
`gpuStartTime` of the routed command buffer minus the storage operation's
completion (`RealForwardRunner.swift:6193-6197`). That is 16 % of the miss gap and
4.2 to 4.5 % of the whole token, and it is dead by construction.

The host-side path around the fetch is **not** where the time is: per token,
`router_readback_ms` 0.008 to 0.009, `cache_plan_ms` 0.155 to 0.212,
`io_queue_ms` 0.53 to 0.57. Their sum is under 0.8 ms per token, 1 % of the wall.

### 3. By answer position: the rate swings 11 to 18 tok/s inside one answer

16-token windows of the 1k answer, regenerated from the archived data with the
aligned `loop-files/step0-fit.py` (the window table shipped in
`ledger/step0-fit.txt` is the pre-alignment run, whose per-token fit is superseded;
window means shift by at most one token):

| tokens | wall/tok ms | misses/tok | layers with a miss | tok/s |
| --- | ---: | ---: | ---: | ---: |
| 1 to 16 | 84.35 | 41.8 | 23.9 | 11.9 |
| 65 to 96 | 63.7 to 64.9 | 23.6 to 23.8 | 16.1 to 16.9 | 15.6 |
| **129 to 176** | **53.7 to 55.8** | **10.3 to 13.1** | **9.1 to 10.3** | **18.0 to 18.6** |
| 257 to 272 | 90.40 | 49.8 | 25.2 | 11.1 |
| 385 to 403 | 81.2 to 83.6 | 36.5 to 41.0 | 21.2 to 21.3 | 12.0 |

The card's answer does the same (87.5 ms at 42.1 misses in its first window, a
trough of 61.9 ms at 13.3 misses at tokens 81 to 96, then 72 to 81 ms at 30 to 38).
Belady replayed on the same traces at 128 slots also rises in the later stretches
(the card by 16-token window: 129 / 88 / 90 / 73 / 81 / 77 / 272 / 290 / 219 / 215 /
218 / 242 / 211 / 153 against production's 674 / 507 / 491 / 412 / 328 / 213 / 535 /
616 / 486 / 480 / 504 / 536 / 537 / 370; `ledger/step0-profile.txt`), so a stretch of
text whose 16 tokens need more than 128 distinct experts in a layer misses under any
policy. **The rate is the miss count, at every scale the ledger looks at.**

### 4. The drive under decode's spacing

`loop-files/ssd-inflight-probe.c` on the mini (whole-expert 1,769,472 B `F_NOCACHE`
preads at random (layer, expert) over the 40 packed-expert files, K threads each
pausing `gap` between its own reads, seed 1, 400 reads per thread;
`ledger/step0-inflight-run1.txt`):

| arm | per-read p50 ms | aggregate |
| --- | ---: | ---: |
| K = 1, back-to-back | 0.774 | 2.31 GB/s |
| K = 1, gap 0.25 / 0.5 / 1 / 2 / 5 ms | 0.810 / 0.836 / 0.949 / 0.972 / 0.991 | 1.50 / 1.12 / 0.72 / 0.45 / 0.21 GB/s |
| K = 2 / 3 / 4 / 8, gap 1 ms | 1.076 / 1.184 / 1.107 / 2.655 | 1.37 / 1.99 / 2.70 / 3.49 GB/s |
| K = 2 / 3 / 4 / 8, back-to-back | 1.075 / 1.558 / 2.047 / 4.100 | 3.30 / 3.47 / 3.50 / 3.51 GB/s |

**Readings.** Decode's roughly 1 ms spacing pays **+0.175 ms per read** over a busy
drive (0.774 to 0.949); the drive saturates near **3.5 GB/s from three reads in
flight**; two reads overlapped cost 1.08 ms each, so a pair completes in about
1.1 ms instead of 1.9 ms serial. The probe's own header states the framing it was
built for: decode today is K = 1 with a 1 ms gap, and a lookahead that fetches the
next layer's misses under the current layer's compute is K = 2 at the same gap.

**Decode is not bandwidth-bound.** 30.7 / 30.3 / 28.2 misses per token at 1,769,472
B each is 54.3 / 53.6 / 49.9 MB per token, which over the measured wall is
**0.73 / 0.76 / 0.72 GB/s against the drive's 3.5**: a headroom factor of 4.6 to
4.8. The wall is read latency and serialization, not bytes.

### 5. What the pool has left

Replayed at production's configuration and against the clairvoyant policy at 128
slots per layer (`ledger/step0-profile.txt`, aging-lfu / resident-first /
protect chunk):

| trace | production decode misses | Belady | reduction | compulsory (production) |
| --- | ---: | ---: | ---: | ---: |
| the card's answer | 6,689 | 2,358 | −64.7 % | 544 |
| the 300 prompt's answer | 9,468 | 3,716 | −60.8 % | 1,393 |

v13 Task 4 measured what is reachable: 74 % of an expert's reuses inside an answer
come within 8 tokens and 97.6 % within a stack distance of 128, so recency already
does the steady-state work, **every bounded policy replayed came within about 5 % of
production**, and the route that did pay was the sweep order at the
prefill-to-decode boundary, which landed as v13 Task 5
([v13-the-turn.md](v13-the-turn.md) "## Task 4" and "## Task 5").

## Levers, ranked for these shapes (modelled from step zero)

Every prize below is stated against the card's 74.27 ms token unless another shape
is named; the arithmetic is shown, and "modelled" means it is arithmetic over the
measured rows above, not a measurement.

- **(a) Hiding the layer's fetch behind the previous layer's compute.** The prize is
  the measured miss gap: **19.68 / 19.24 / 18.03 ms per token, 26.5 / 27.4 / 26.1 %
  of the wall**, and it is the largest single term in the chapter. Two ceilings,
  both modelled from rows 1, 2 and 4:
  - *Deep enough lookahead that the drive never idles.* The bytes fit (row 4: 15.5 ms
    of drive time per token at 3.5 GB/s against 45 ms of GPU busy), so the gap goes
    to zero and the wall becomes the GPU busy plus the other gaps: 44.99 + 9.60 =
    **54.59 ms, 18.3 tok/s** (from 13.46), and 22.2 tok/s if the other gaps went too.
  - *One layer of lookahead, perfect prediction.* A missing layer needs 1.68 reads on
    average (30.7 misses over 18.3 missing layers), costing 1.08 + 0.68 x 0.61 =
    **1.49 ms** at K = 2 (row 4) against that layer's roughly 1.04 ms of GPU busy
    (44.99 ms over 40 layers plus the missing layer's own fixup), so about 0.45 ms
    per missing layer stays exposed: 8.2 ms per token, wall **62.8 ms, 15.9 tok/s**.

  **The lever's whole value is the predictor's accuracy, and specifically the rate at
  which a layer's ENTIRE absent set is named a layer ahead**, because a layer that
  still has one unfetched expert still pays the serial 1.08 ms; a partly covered
  layer only converts first-miss cost into further-miss cost. What the predictor
  would have to know is layer L+1's exact top-k, and layer L+1's router input is
  layer L's output, which does not exist yet while layer L runs. The runtime already
  ships the approximation: while encoding layer L it runs layer L+1's router on layer
  L's post-attention normalized residual and writes the speculative top-k to
  `prefetchPredictionIndices` (`RealForwardRunner.swift:3453-3471`, read back at
  `:3118-3126`, gated by `nextLayerPredictionEnabled` at `:2023-2025`), and
  `ExpertPrefetchRing` stages the predicted reads outside the authoritative cache
  (`RealForwardRunner.swift:6738-6743`), which the exact plan adopts only if the
  exact router selects them (`:6509`, `:6513`, `:6517`).

  The prior art is measured and it is negative. `architecture.md:83-89` kills a
  different predictor (same layer, previous token: 0.00 % of misses caught at 16 and
  at 128 slots). `:96-98` reports the next-layer probe recalling **64.1 % of actual
  nonresident misses at top-8, 56.4 % precision**; `:108-118` reports a later paired
  distance experiment at **recall 0.439 at k = 1 with nonresident precision 0.124**
  on the rig stream and **0.510 / 0.358** on a diverse prompt, and closes the door
  ("the miss-count lever is closed"). `:100-106` records why it is off by default:
  the interleaved run gave **−3.9 % at 4-bit** and +7.8 % at 8-bit against a +10 %
  promotion bar, diagnosed as lead time. The v10 P3 follow-on repeats the verdict
  (`docs/v10-implementation-plan.md:190-200`, an uncommitted peer edit in the working
  tree, archived at `~/.claude/handoffs/archive/shrike-ssd-split-probe/`).

  Three things step zero changes about that verdict, none of which makes it wrong,
  all of which make it re-priceable:
  1. **The bar is gone.** The +10 % promotion bar is not this chapter's rule; v13 has
     no size floor (Davor's ruling at v13 Task 1). The −3.9 % at 4-bit is still a
     measured regression and remains the thing to beat.
  2. **The lead time is now a number.** One layer's wall is 74.27 / 40 = **1.86 ms**
     on the card (1.75 / 1.73 on the other two), against a 0.95 ms read at K = 1 and
     1.08 at K = 2. Where the ring begins its reads today, the lead is much shorter:
     `begin` is called after the layer's own demand fetch has completed
     (`RealForwardRunner.swift:6739`), leaving only the fixup, the submit gap and the
     next layer's attention, and `readyBuffers` hands back only slots whose operation
     is already `.completed` (`ExpertPrefetchRing.swift:87-99`), so a correct but late
     prediction buys nothing and its bytes are read twice.
  3. **The bandwidth cost of a wrong prediction is now bounded.** Decode uses 0.73
     GB/s of 3.5 (row 4), so total reads may rise 4.8x before the drive binds:
     **precision must clear 1 / 4.8 = 0.21** for a prefetching scheme to stay off the
     drive's ceiling. The measured k = 1 precision straddles it (0.124 rig, 0.358
     diverse), which is exactly a measurement, not an argument.

  Applying the measured recall: with per-miss recall r and 1.68 misses in a missing
  layer, full-layer coverage under an independence approximation is r^1.68 = **0.26
  at r = 0.439, 0.49 at r = 0.64**, so the one-layer prize becomes p x (19.68 − 8.2)
  = **3.0 to 5.6 ms per token, 4.0 to 7.5 %**. Correlation between the misses of one
  layer pushes the true rate above the independence estimate, which is the reason to
  measure p rather than model it.

  **The offline experiment, before any code.** The archived data already contains
  everything needed for the parts that do not involve the router, and one existing
  env var supplies the rest.
  - *From the archived traces alone* (`step0-out/<shape>/route-*.trace` plus a
    recorder in the shape of `loop-files/step0-layer-misses.py`, which wraps the
    replay tool's pool without editing it): record, for every decode (position,
    layer), the identity of the absent experts (an expert is absent at plan time when
    no slot holds it, which is exactly the hit test the tool's `LayerPool.plan` runs
    over `slot_expert` at `tools/expert-pool-replay.py:477-538`). Then compute, per
    (position, layer): the size distribution of the absent set; the fraction of
    layers whose absent set is fully contained in the previous token's demanded set
    at the same layer (the predictor `architecture.md:83-89` disproved, re-checked at
    128 slots on today's pool); the same for the union of the last n tokens at that
    layer; and the same for layer L's own demanded set (a pure "next layer looks like
    this layer" baseline). These are **upper bounds for history-only predictors and
    cost nothing**, and any one of them clearing the full-coverage rate the router
    probe reaches would be a cheaper lever than the router.
  - *From one capture on the mini, no code:* `SHRIKE_PREFETCH_TRACE=<path>` already
    emits, per (position, layer), the exact route, the misses, the residency captured
    before planning, and the next-layer prediction
    (`RealForwardRunner.swift:2223-2241`; the writer states it "cannot submit I/O or
    alter cache decisions"). Joining line (position, L) to line (position, L+1) gives
    the router predictor's **per-layer full-coverage rate p and its precision on this
    build, this pool and these three shapes**, which is the number the whole lever
    hangs on. The capture's walls are not verdict rows: enabling the trace also turns
    on the second router GEMV per layer (`:2023-2025` gates both), so it taxes the
    GPU it is measuring.
  - *And a zero-code A/B:* `SHRIKE_PREDICTIVE_PREFETCH=1` (with `SHRIKE_PREFETCH_TOP_M`,
    default `min(4, topK)` = 4, and `SHRIKE_PREFETCH_PROBE_DISTANCE`, default 1) runs
    the whole scheme on the shipped binary (`RealForwardRunner.swift:908-929`). It is
    the same shape as v13 Task 4's opening move, which priced three existing policy
    values before writing a line.

- **(b) Fewer misses.** Belady at 128 slots removes 64.7 % of the card answer's
  decode misses (row 5). Through the single-feature slope that is 0.831 x (30.7 −
  10.8) = **16.5 ms per token, 74.27 to 57.7 ms, 17.3 tok/s** (modelled; the
  two-feature model cannot be applied because the profile gives Belady's window
  totals, not its per-layer distribution). What is realizable is small: v13 replayed
  every bounded eviction rule within about 5 % of production and found recency
  already at 97.6 % of an answer's reuses, so 5 % of 6,689 misses is 334, **0.28 s
  of a 16.19 s decode, 1.7 %**. The routes that are not exhausted are the ones v13
  named and did not close: the prefill-to-decode boundary (its Task 5 took the first
  cut) and a pool larger than 128 slots, which the box does not have (8G is the
  measured optimum on 16 GB, CLAUDE.md). **(a) and (b) compose:** the miss gap is
  (layers with a miss) x (that layer's serial read), and they are its two factors.

- **(c) The drive's idle-gap tax.** Decode leaves the drive idle between layers and
  pays 0.949 − 0.774 = **0.175 ms on the first read of every missing layer** (row 4),
  which at 18.3 missing layers is **3.2 ms per token, 4.3 %**. It is real and it is
  not separately collectable: the only way to stop paying it is to have a read in
  flight when the next one is issued, which is (a). Its standalone forms are worse
  than the tax (a keep-warm background stream steals shared bandwidth from the
  payload, measured NULL to negative by the peer's probes,
  `docs/v10-implementation-plan.md:170-172`).

- **(d) The miss path's host and driver windows, which no predictor is needed to
  attack.** Two measured terms:
  - The **routed submit gap**, `moe_spec_routed` to `moe_phase1_hit`: **4.13 / 4.19 /
    3.84 ms per token, 5.6 / 6.0 / 5.6 %** of the wall, over 18.18 / 18.70 / 17.39
    missing layers (0.227 / 0.224 / 0.221 ms each). Its `host_ms` (host-late time
    only, not a partition; `RealForwardRunner.swift:2082-2108`) is 2.04 / 1.97 / 1.88
    ms per token and its `queue_ms` 1.82 / 1.93 / 1.71. Between those two kernels the
    host reads back the router (`RealForwardRunner.swift:6481-6489`), plans the cache
    (`:6510-6515`), pins (`:6525`), begins the fetch (`:6533`) and encodes phase 1;
    the counters above say the planning itself is 0.16 to 0.21 ms per token, so most
    of this window is submission and queueing, not planning.
  - The **post-completion wake**, `io_fixup_wake_ms`: **3.20 / 3.14 / 2.88 ms per
    token, 4.3 / 4.5 / 4.2 %**, the interval from the read completing to the routed
    command buffer starting on the GPU (`RealForwardRunner.swift:6193-6197`). It sits
    inside (a)'s 19.68 ms, so (a) collects it if (a) works and it is collectable on
    its own if (a) does not.

  Together **7.33 / 7.33 / 6.72 ms per token, 9.9 / 10.4 / 9.7 %** of the wall, all
  measured, no prediction, no numerics change. How much of it is recoverable is not
  established: some of the submit gap may be irreducible driver work, and the wake is
  a Metal shared-event signal to GPU start.

- **(e) The LM head the server never fuses.** `head_logits` is **4.89 / 5.05 / 4.82
  ms of GPU per token, 6.6 / 7.2 / 7.0 %** of the wall, and `head_fused_ms` is
  0.000 on all three shapes. A fused greedy head exists and is the decode path's
  request (`RealForwardRunner.swift:3165-3176`, `outputMode: .greedyIfAvailable` at
  `:2301`), but the server hardcodes `forceLogitsHead: true` at
  `ServerInference.swift:661` and `:716`, so it never runs; v10 recorded the same
  observation from the other end (`docs/v10-implementation-plan.md:220-222`). The
  fused path cannot remove the head GEMV, only the full logits writeback and a
  dispatch, so the recoverable share is a fraction of 5 ms and it is a behaviour
  change (no logits means no sampling and no logprobs), not a free flip. Recorded
  with its ceiling, not priced.

**Not levers here.** The attention stack is the largest GPU term (15.57 + 9.84 =
25.4 ms per token on the card, 34 % of the wall) and it is kernel work in v11's
territory ([v11-kv-attention-inner-loop.md](v11-kv-attention-inner-loop.md)), not
scheduling. A larger expert pool: the box is 16 GB and 8G is the measured optimum
(CLAUDE.md). Speculative decoding: retired at v12 Task 17 with numbers
([v12-prefill-matrix-kernels.md](v12-prefill-matrix-kernels.md):1104-1135, acceptance
20.6 / 31.7 / 75.6 % by shape, pass 212 / 233 / 312 ms against a tie threshold of
55 / 72 / 141 ms, and greedy speculation not lossless on two of the three shapes).
The ANE (attention is 4 % of a 1.4k prefill; [ane-prefill.md](ane-prefill.md)).
Faster reads: closed on the deploy target by the peer's five probes
(`docs/v10-implementation-plan.md:156-189`).

## Method

v13's protocol carried over ([v13-the-turn.md](v13-the-turn.md) "Method"), with the
rig moved from prefill shapes to answers. **Mini-first verdicts**; the M4 Pro is
iteration signal and the check, never the verdict. One send per server lifetime for
a cold row, `settle_done` before a warm second send, a distinct prompt per arm (no
prompt a prefix of another), `SHRIKE_RUNNER_STATS=1 SHRIKE_KERNEL_STATS=1` on every
measured arm. Each task pre-registers its rule on this ledger's rows before it runs,
lands its knob either way (a measured null is a result), and reports the same rows
after. **The rule has no size floor** (Davor's ruling at v13 Task 1: ten real 1 %
wins are a 10 % chapter, and a floor drops every one of them). A default flips when
the effect is **real** (the sign holds across paired runs in both orders and the
delta exceeds the run-to-run drift, with more pairs for a smaller effect) and
**free** (no control row regresses: the other shapes, the turns, the 12k control,
hit rate, memory pressure, and golden is identical). A cost that does show is priced
in the verdict, never hidden by a bar; the code a lever leaves behind is the
review's judgment. Scheduling-only changes are golden IDENTICAL on both boxes and
both profiles; anything that reorders arithmetic is qualified as v12 did.

Two additions this chapter's shapes force. **A decode verdict row is a whole answer,
not eight tokens**: row 3 shows the rate swinging 11 to 18 tok/s inside one answer
with the local miss count, so an 8-token burst measures whichever stretch of text it
landed in. The three step-zero answers (219 / 314 / 405 tokens at `max_tokens` 512,
`temperature` 0) are the verdict rows, with the follow-up turns as controls. And
**the drift band is re-established per task on the same binary**: v13 measured about
1 % on a wall and a 0.26 tok/s spread on a long answer's decode.

## Numerics policy

Prefetch, slot policy and submission order change which expert is fetched when and
which command buffer waits on what, never what any kernel computes: **golden
identical on both boxes and both profiles** is the bar for them, and a difference is
a defect, never a recapture ([v13-the-turn.md](v13-the-turn.md) "Numerics policy"). A
predicted fetch is authoritative only after the exact router selects it
(`ExpertPrefetchRing.swift:4-11`, `RealForwardRunner.swift:6509-6517`), so it is a
scheduling change under this policy and its adoption must leave the pool's plan
sequence identical, which the replay tool checks against a capture rather than
assuming. A change that makes decode run different kernels on a token (the fused
greedy head is the one candidate here) is a numerics change and follows v12's policy
(2e-2 against the fp32 reference, golden recaptured once per box with before and
after digests in the verdict).

## Out of scope

- Kernel work on attention, GDN and the head GEMV (v11's and v10's follow-ons stay
  there), even though attention is the largest GPU term.
- Speculative decoding (v12 Task 17, retired with its numbers).
- The ANE ([ane-prefill.md](ane-prefill.md)).
- A larger expert pool or a different drive: 8G is the measured optimum on the 16 GB
  mini and the fetch-speed lever is shut on the deploy target.
- The prompt cache's interior snapshots (a cache-chapter item, recorded in v13).

## Risks

- **The predictor is the chapter's biggest lever and it has a measured failure
  behind it.** `architecture.md:100-118` records a −3.9 % end-to-end regression at
  4-bit and closes the door on distance; the v10 follow-on repeats it. This chapter
  reopens it only as far as a pricing step goes, with a named stop, and lands the
  null if the coverage rate does not clear the bar.
- **Row 1's decomposition is per-request and averaged over the answer**, while row 3
  shows the answer is not stationary. A lever measured on one stretch of text can
  read as a win that is a different stretch. Whole answers, paired, both orders.
- **The replay sees misses, not exposure.** v13 Task 5 was overruled by the box after
  the replay predicted the miss count correctly and missed a fetch-exposure cost
  entirely. Offline pricing selects candidates; the mini decides.
- **The trace and the probe tax what they measure.** `SHRIKE_PREFETCH_TRACE` also
  enables the second router GEMV per layer; `SHRIKE_ROUTE_TRACE` is a synchronous
  write per layer per token (measured nil on decode in v13 Task 4, still not a
  verdict arm).
- **The mini is production.** Every arm stops the server on 8081 and relaunches it,
  Turbo on 8080 is never touched, `memory_pressure -Q` is checked before every
  launch, one model process at a time.
