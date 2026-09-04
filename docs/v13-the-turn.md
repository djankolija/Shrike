# v13 — the turn: prefill and decode at the shapes a conversation actually runs

## The problem

v12 ([v12-prefill-matrix-kernels.md](v12-prefill-matrix-kernels.md)) closed the
long-prompt prefill: 12k tokens on the mini went 725 → 68.4 s. The shapes this box
is used for are not that. LLMBench's cards are two-turn conversations with a
56-character system prompt and 50-character questions (tools off: the model
prefills 48 tokens and decodes 600–800) or a chain of tool rounds (tools on: a
495-token opening, then 600–2,000 tokens of tool results per round, 2–7
rounds, then the answer). A conversation reaches 12k only late in a tool-heavy
card. So the chapter's ledger is measured at 300 / 1k / 2k new tokens, cold and
warm, and at the follow-up turn, and its target is the wall of a turn, not a
per-token rate.

## Step zero — the ledger (mini, v12's close build 3774ef1, production launch)

One fresh server per pair; the warm arm is a different prompt of the same
length after the prompt cache's `settle_done`; eight decode tokens per request.
Prompts: the `tools/prefill-prompts.py` ledger construction at 4 / 16 / 32
entries (≈ 62.6 tokens each); the turns append the answer and a new question.

| shape (new tokens) | cold wall | warm wall | warm prefill_s | prefill GPU | `routed→routed` host | experts fetched (warm) | prefill hit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 289 / 305 | 8.91 s | 5.46 s | 5.04 | 2.5 s (8.2 ms/tok) | 2,001 ms / 942 | 6,875 (11.8 GB) | 11 % |
| 1,069 / 1,085 | 11.44 | 8.05 | 7.62 | 5.3 s (4.9) | 1,820 / 1,107 | 8,224 (14.1 GB) | 9 % |
| 2,093 / 2,125 | 14.45 | 11.21 | 10.73 | 9.6 s (4.5) | 759 / 1,148 | 8,527 (14.6 GB) | 9 % |
| turn 2 on the 2k context (38 new, 2,118 cached) | | 2.13 | 1.66 | 0.86 s (22.6) | 355 / 387 | | |
| turn 3 (36 new, 2,149 cached) | | 1.38 | 0.91 | 0.76 s (21.0) | 26 / 359 | | |

Decode: ≈ 0.4 s per request warm (8 tokens). Outside the generation span: 2.06 s
on a fresh server's first request, 0.02–0.08 s afterwards.

**What the rows say.**

1. **A first-turn prefill costs ≈ 4.5 s + 3.0 ms per token** (the warm intercept
   from the three pairs; cold adds ≈ 3.3 s: 2.06 outside the span plus ≈ 1 s of
   slower prefill). The intercept is the expert sweep: any chunk of a few hundred
   tokens routes to 190–250 of the 256 experts in every layer, so 6.9–8.5k
   expert fetches (12–15 GB) stream from the SSD at ≈ 3 GB/s with a 0–11 % hit
   rate — v12's alternating sweep (P15) only pays from a prompt's second chunk.
   GPU busy is 51–70 % of the span at these lengths: the routed path is
   fetch-bound outright (≈ 5 ms of fetch per 8-expert tile against 0.5–2.4 ms of
   GPU), which the two-tile bank (P16) cannot hide.
2. **Follow-up chat turns hit the prompt cache** (`prompt_cache hit tier=live`,
   `settle_restore`) and cost 1.4–2.1 s for ≈ 37 new tokens: 0.9–1.7 s of prefill
   = 40 layer-chunks of fixed work (the matrix kernels' per-dispatch floor at
   tiny row counts, 21–23 ms per new token; `shared→routed` 138 ms and
   `routed→routed` 355 ms of host on turn 2) plus ≈ 0.4 s of decode. A prompt
   that diverges INSIDE a stored entry gets no prefix (`settle_reset
   reason=no_prefix_snapshot`, `cached=0`, a full re-prefill): append-only turns
   — chat, tool rounds — are served; an edited document is not.
3. For the cards, the answers (600–800 tokens at 18–21 tok/s: 30–40 s) still
   dominate; first-turn and tool-round prefills (0.5–2k tokens: 6–11 s each) are
   second; follow-up-turn overhead (1–2 s) third; the fresh-server first request
   (+3.3 s) only after a restart.

## Task 0 — the expert sweep's parity carried across requests (commit f3ede42)

The first chunk of every request used to sweep the experts ascending, into a
pool whose residents were the previous request's last-swept experts at the
other end. `SHRIKE_PREFILL_SWEEP=carry` (the new default; `alternate` is the
A/B, `fixed` P15's) remembers the direction the last chunk swept and starts the
next request's first chunk in the opposite one, alternating from there. Order
only: the comparator is the one thing the direction reaches, so golden is
identical on both boxes and both profiles (measured).

Measured on the mini, one binary (the pre-flip build with the knob as the A/B),
a fresh server per pair, the warm arm a different prompt of the same length
after `settle_done`:

| shape | alternate warm wall (hits) | carry warm wall (hits) | Δ wall | misses, GB read | `routed→routed` host |
| --- | ---: | ---: | ---: | ---: | ---: |
| 305 new tokens | 5.47 s (11.0 %) | **3.80 s (58.6 %)** | **−30.6 %** | 6,875 → 3,200 (11.8 → 5.6 GB) | 1,994 → 853 ms |
| 1,085 | 8.08 (9.2 %) | **6.88 (51.6 %)** | **−14.9 %** | 8,224 → 4,383 (14.1 → 7.6 GB) | 1,816 → 914 |
| 2,125 | 11.24 (8.9 %) | 10.80 (50.4 %) | −3.9 % | 8,527 → 4,639 (14.6 → 8.0 GB) | 782 → 486 |
| turn 2 on the 2k context (38 new) | 1.91 (52.0 %) | 1.70 (62.6 %) | −11 % | | |
| turn 3 (36 new) | 1.40 (84.8 %) | 1.37 (86.3 %) | −2 % | | |
| after a 314-token answer, 305 new | 3.77 (49.5 %) | 3.79 (49.5 %) | 0 | identical hit counts | |
| after a 405-token answer, 1,085 new | 6.62 (43.8 %) | 6.63 (43.8 %) | 0 | identical hit counts | |
| 12k, first request after launch (server wall) | 68.42 (33.6 %) | 68.58 (17.0 %) | +0.2 % | hits 9,546 → 4,817 — the C1 defect, see below | |
| 12k after a 305-token request (server wall) | 65.62 (35.0 %) | 65.55 (34.8 %) | −0.1 % | hits 9,933 → 9,869 — two C1 effects cancelling | |
| 12k, first request after launch, **after the fix** | 68.30 (33.6 %) | 68.30 (33.6 %) | 0 | hits 9,546 → 9,546 | |
| 6k (two chunks, 6,381 tokens), cold / warm, after the fix | 34.63 / 31.47 (25.4 / 49.7 %) | 34.62 / 31.43 (25.4 / 49.7 %) | 0 | hits identical (4,778; 9,340) | |

The cold first requests match across modes to ≤ 61 ms (the carry is nil on a
fresh server's first request; the model loads lazily inside it). Walls are the
server's `completed in` except where the table says curl. Walls repeat to ≤ 30
ms between runs on the single-chunk rows (turn 2 drifted 2.13 → 1.91 s between
step zero and the arms with byte-identical expert traffic), so the 5 % rule is
≈ 9× the noise on the shortest wall and the measured 300-token delta ≈ 56×.

| shape | modelled hits/layer (P = 104 / 128) | modelled warm wall | measured hit rate | measured warm wall |
| --- | ---: | ---: | ---: | ---: |
| 305 | 63.0 / 77.5 (32.6 → 40.1 %) | 4.59 → 4.29 s | **58.6 %** | **3.80 s** |
| 1,085 | 82.5 / 101.5 (36.4 → 44.8 %) | 6.77 → 6.37 | **51.6 %** | **6.88** |
| 2,125 | 87.5 / 107.7 (37.4 → 46.0 %) | 10.50 | **50.4 %** | **10.80** |

The hit rates land above the model's P = 128 ceiling and the 300-token wall
below its modelled floor; the 1k and 2k walls sit 0.5 / 0.3 s above theirs.
`routed→routed` host: 853 / 914 / 486 ms against bars of ≤ 1,400 / 800 / 350 —
✓ / ✗ / ✗ (the brief's baselines were the gap's totals; the verdict scores the
host term, the exposed part, and says so).

**Two shape facts.** After a card-length answer the pool holds decode's
working set, not a sweep's tail, and the direction is moot: both modes hit
49.5 / 43.8 % (3,824 / 3,824 and 3,964–3,966 hits) and both beat the
short-decode control — the lever pays on short-decode shapes (tool rounds,
follow-up turns) and is neutral after a long answer. Turn 2's −11 % is the
size of its own control's drift, but its sign stands on the traffic: 348 fewer
misses × 0.56 ms ≈ 195 ms against the measured 210.

**The 12k rows were a defect's footprint, found by the task review.** The first
landed `carry` re-read the carry on every chunk after overwriting it, so a
multi-chunk prompt swept `d0, d0, !d0` instead of alternating: a two-chunk
prompt got no reversal at all and a three-chunk prompt one. All four 12k rows
decompose to within 0.5 % under that model (one chunk transition ≈ 4,773 hits;
a fully aligned first chunk ≤ 5,120): alternate first request 0 + 4,773 +
4,773; carry first request 0 + 0 + 4,773; alternate after a short request 387 +
4,773 + 4,773; carry after it ≈ 5,096 + 0 + 4,773 — the lever earned ≈ 4,700
hits at chunk 0 and the defect gave ≈ 4,773 back at chunk 1. The docs had
published a readiness-prefill mechanism for the halving; the model loads lazily
inside the first request and no prefill precedes it, so that mechanism is
retracted. The fix (commit above) makes each chunk the opposite of the previous
one, across requests and within a prompt, with the composition under test; the
MTP verify path no longer participates in the carry. Re-measured after the fix:
the 12k first request reads 68.30 s with 9,546 hits in both modes, and a
two-chunk pair (6,381 tokens) reads 34.63 / 34.62 s cold and 31.47 / 31.43 s
warm with identical hit counts (4,778; 9,340). Identical is right: on a first
request the carry is nil, and a two-chunk request under `alternate` already ends
descending, so its successor's first chunk hits either way. `carry` pays after
requests with an odd chunk count — one, three — which is every short turn.

**After T0** (commit f3ede42, 2026-09-04; `SHRIKE_PREFILL_SWEEP=carry` the
default, `=alternate` the A/B; mini, warm arms, server walls):

| shape | warm wall | prefill hit rate | experts fetched |
| --- | ---: | ---: | ---: |
| 305 new tokens | 3.80 s | 58.6 % | 3,200 (5.6 GB) |
| 1,085 | 6.88 | 51.6 % | 4,383 (7.6 GB) |
| 2,125 | 10.80 | 50.4 % | 4,639 (8.0 GB) |
| turn 2 (38 new on a 2k context) | 1.70 | 62.6 % | |
| turn 3 (36 new) | 1.37 | 86.3 % | |
| 6,381 (two chunks), warm | 31.43 | 49.7 % | 9,454 |
| 12k, first request after launch | 68.30 | 33.6 % | 18,858 |

Same binary, `alternate`: 5.47 / 8.08 / 11.24 s, turn 2 1.91, turn 3 1.40.
Golden identical on both boxes and both profiles. The rig: `tools/turn-prompts.py`,
`tools/turn-rig.sh`, `tools/turn-summary.py`. Scope: measured on a launch without
an MTP sidecar; the verify path does not carry.

## Task 1 — two tile fetches in flight (commit 82608b4)

The routed prefill loop awaited each tile's expert fetch in the expression that
issued it, so between one tile's reads landing and the next tile's reads
starting the drive idled through the host's plan → encode → commit step.
`SHRIKE_PREFILL_FETCH_DEPTH=2` (the new default; `=1` the A/B) plans tile N+1
avoiding tile N's slots and the held ones, begins its fetch, then awaits N;
the next iteration awaits N+1 and begins N+2. Scheduling only: which slot an
expert lands in and when, never its bytes — golden identical on both boxes,
both profiles, at both knob values (measured).

**What the drive sees.** The drafter's reading, verified in the C reader:
`submit_batch` (`sources/ShrikeKernelsC/expert_io.c`) publishes one batch at a
time — a second caller parks on `batch_idle` until the first batch completes
and clears its pointers, and the parked caller is an `ExpertIOScheduler`
worker, not the runner. So two fetches in flight closes the inter-tile gap
(the next batch is already parked when the current clears) and does **not**
raise queue depth: bytes in flight stay capped at four experts, and after T0
a tile carries only 3.3–3.9 misses on the mean. The reader's thread count is
therefore not a lever until two batches can be published at once (T2 below).

Measured on the mini, one binary (the knob as the A/B), a fresh server per
pair, the warm arm after `settle_done`; 300 and 1k are paired means of two
runs in opposite orders (repeats agree to 11–66 ms, hit counts identical run to
run):

| shape | depth 1 warm wall (hits) | depth 2 warm wall (hits) | Δ wall | `routed→routed` host | `io_fetch_ms × 8` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 305 new tokens | 3.777 s (58.6 %) | **3.683 s (58.4 %)** | **−2.5 %** | 806 → 745 ms | 1,863 → 3,274 |
| 1,085 | 6.786 (51.6 %) | **6.667 (51.6 %)** | **−1.8 %** | 836 → 733 | 2,485 → 4,466 |
| 2,125 (one run) | 10.676 (50.4 %) | 10.451 (50.4 %) | −2.1 % | 371 → 184 | 2,680 → 4,488 |
| turn 2 on the 2k context (38 new) | 1.663 (62.5 %) | 1.656 (62.5 %) | −0.4 % | 277 → 230 | |
| turn 3 (36 new) | 1.466 (86.3 %) | 1.344 (86.3 %) | −8 %, one run per arm | 18 → 4 | |
| 12k, first request after launch | 68.342 (33.6 %) | 68.346 (33.6 %) | 0 | 73 → 33 | hits 9,546 → 9,540 |

Routed GPU and tile counts unmoved (1,413 / 2,517 / 3,907 ms; 982 / 1,147 /
1,188). Cold first requests are not verdict rows (the lazy model load sits
inside them).

**Three readings.** (1) The mechanism works as built: the fetch counter
doubles because the parked batch's wait is now counted inside the fetch, the
host term between tiles falls, and the wall falls by about the same amount.
The pre-registered inversion read — `host_ms` ÷ routed tiles, which a
batch-order inversion would raise — fell instead: 0.85 → 0.78 ms per tile at
300, 0.74 → 0.64 at 1k; no inversion signal. (2) The eight extra held slots
cost nothing measurable — hits −18 / −4 / −4 at 300 / 1k / 2k and −6 at 12k
against a modelled −349 / −360 / −363; the
`d·P/(2−d)` model's sensitivity to the readable window is wrong by an order of
magnitude and is retired for slot budgeting. (3) The gap the lever closes was
worth 61 / 103 ms per request at 300 / 1k (paired host deltas) and ≈ 190 ms
at 2k, not the modelled Σc ≈ 700–850 ms: P16's 0.72 ms per tile was an upper bound, and most of the
host's per-tile work already overlapped the drive through the GPU bank. The
drive's own time is the floor. Under depth 2 the routed stage at 300 tokens is
GPU 1,417 + gap 864 ≈ 2.28 s for 5.6 GB ≈ 2.5 GB/s effective, below the
probe's 3.6 GB/s because a tile's 3–4 misses fill at most four reader threads
for one wave and the batch ends at its slowest read. **The remaining fetch
term lives in overlapping batches** — the C reader's one-batch predicate plus
the thread count, T2 — worth up to ≈ 0.4–0.7 s at 300 tokens if the effective
rate reaches the probe's 3.0–3.6. The draft's model against the measured
routed stage (GPU + `routed→routed` total, first runs, ms), for the record:

| shape | modelled stage, depth 2 (case a) | measured, depth 1 | measured, depth 2 |
| --- | ---: | ---: | ---: |
| 305 | 2,016 | 2,336 | 2,281 |
| 1,085 | 2,685 | 3,461 | 3,355 |
| 2,125 | 3,911 | 4,388 | 4,150 |

The model expected the whole Σc to come out of the stage; a tenth of it did.

**The rule, retracted and replaced.** The task pre-registered a 5 % bar on the
300 and 1k walls. It was the controller's inherited default (Task 0's number,
copied into the drafter brief, justified after the fact by the draft's model
of a 7–8 % gain), never derived from the noise floor or the change's cost.
Davor's ruling at the verdict (2026-09-04): the chapter has no size floor —
ten real 1 % wins are a 10 % chapter — and a default flips when the effect is
**real** (paired, both orders, above drift) and **free** (no control row
regresses, golden identical). Task 1 is both. Recorded here so the bar is not
read as quietly dropped; the Method section carries the rule forward.

**After T1** (commit 82608b4, 2026-09-04; `SHRIKE_PREFILL_FETCH_DEPTH=2` the
default, `=1` the A/B; mini, warm arms, server walls):

| shape | warm wall | prefill hit rate | experts fetched |
| --- | ---: | ---: | ---: |
| 305 new tokens | 3.68 s | 58.4 % | 3,218 (5.6 GB) |
| 1,085 | 6.67 | 51.6 % | 4,387 (7.6 GB) |
| 2,125 | 10.45 | 50.4 % | 4,643 (8.0 GB) |
| turn 2 (38 new on a 2k context) | 1.66 | 62.5 % | |
| turn 3 (36 new) | 1.34 | 86.3 % | |
| 12k, first request after launch | 68.35 | 33.6 % | 18,864 |

Same binary, `=1`: 3.78 / 6.79 / 10.68 s, turn 2 1.66, turn 3 1.47. Golden
identical on both boxes and both profiles at both values. Scope: measured on a
launch without an MTP sidecar; the knob clamps to 2 (a deeper lookahead needs a
FIFO of in-flight operations and, with one batch published at a time, is
modelled at ≈ 0.8 % — deferred with T2). **The two loop paths are not the same
loop:** the original at `=1` runs the scheduler's `decide` and its
commit-before-append valve; the lookahead at `=2` runs neither — its bootstrap
re-plans after draining one pending batch and throws `expertCacheUnplaceable`
if the cache still has no room, which `fitting`'s budget makes unreachable at
128 slots (at most three batches plus one in-flight tile are held when a plan
is attempted, leaving 96 free for an 8-expert plan). The task review's
judgment: collapse them into one loop as a follow-on before the chapter merges
to main, in its own commit with its own golden pair, and first if any task
edits the loop before then; the begin/await/drain sequencing is factored into a
host-testable decision with it (today it lives only in the runner, covered by
golden).

## Task 2 — the expert reader publishing two batches at once (commit d3efdeb)

The C reader held one batch: `submit_batch` parked a second caller until the
first batch had completed and cleared its pointers, so Task 1's two fetches in
flight were served batch-serially and bytes in flight never exceeded one tile's
misses. The reader now holds a two-slot ring — each slot its own counts, error,
publication sequence and completion condvar — and workers claim from the older
batch first, spilling into the newer only when the older has no unclaimed read
left, so tile N's fetch is never delayed by N+1's. The FIFO claim and the
free-slot search are pure functions in the header, asserted from Swift; the
shutdown path now cancels a slot's unclaimed reads and signals its submitter (a
latent hang in the one-batch code); the process raises its own `RLIMIT_NOFILE`
once at the first reader's creation (the mini's soft limit is 256 against 160
descriptors at four threads and 320 at eight). Two knobs:
`SHRIKE_EXPERT_IO_BATCH_DEPTH=1|2` (**2 the new default**) and
`SHRIKE_EXPERT_IO_THREADS=1…16` (4, unchanged). Scheduling only: the same bytes
into the same planner-chosen slots — golden identical on both boxes, both
profiles, at (1, 4), (2, 8) and the new default (measured).

Measured on the mini, one binary (the two knobs as the A/B), a fresh server per
pair, the warm arm after `settle_done`; 300 and 1k are paired means of runs in
opposite orders (three runs of the baseline, two or three of each new cell;
repeats agree to ≤ 30 ms; hit counts identical run to run):

| shape | (1, 4) today | (2, 8) | **(2, 4)** | Δ at (2, 4) | `routed→routed` host (1, 4) → (2, 4) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 305 new tokens | 3.688 s (58.4 %) | 3.565 s | **3.537 s** (58.4 %) | **−4.1 %** | 756 → 630 ms |
| 1,085 | 6.672 (51.6 %) | 6.517 | **6.468** (51.6 %) | **−3.1 %** | 737 → 551 |
| 2,125 (one pair) | 10.460 (50.38 %) | 10.296 | 10.304 (50.33 %) | −1.5 % | 187 → 51 |
| turn 2 on the 2k context (38 new) | 1.637 (62.5 %) | 1.655 | 1.600 | −2 % | 229 → 174 |
| turn 3 (36 new) | 1.328 (86.3 %) | 1.354 | 1.332 | +0.3 % | 5 → 3 |
| 12k, first request after launch | 68.304 (33.6 %) | 68.256 | 68.313 | 0 | 33 → 8 |
| decode tok/s, long-decode arm, 300 / 1k | 14.09 / 14.22 | 14.01 / 14.06 | 14.17 / 14.19 | unmoved | |
| warm after the 314 / 405-token answer | 3.734 / 6.559 | 3.422 / 6.247 | 3.402 / 6.237 | −8.9 / −4.9 % | |

Cold all-miss first requests, `prefill_s`, paired means: 300 tokens 5.81 → 5.36 s
(three runs → two), 1k 8.28 → 7.87 (three → two), 2k 11.25 → 10.82 (one pair) —
the fetch term at 100 % misses gains ≈ 0.41–0.45 s at every shape. Routed GPU
and tile counts unmoved.

The four cells and what the drive sees in each — `(depth, threads)`; in flight is
concurrent `pread`s of one whole expert:

| cell | in flight | role | 300-token warm wall |
| --- | --- | --- | ---: |
| (1, 4) | ≤ min(tile misses, 4), draining between tiles | today, verdict A | 3.688 s (paired, three runs) |
| (2, 8) | ≤ 8 across two batches | verdict B | 3.565 (paired) |
| (2, 4) | ≤ 4, never draining between tiles | attribution → **the default** | 3.537 (paired) |
| (1, 8) | ≤ min(tile misses, 8) = 3–4 | attribution, predicted null | 3.699 (one run) |

**Attribution: the batch depth is the lever, the thread count is null.** At 300
tokens (2, 4) read 3.531 / 3.542 s and (2, 8) 3.550 / 3.580, while (1, 8) read
3.699 against (1, 4)'s 3.678–3.702. A tile carries 3–4 misses, so two published
batches keep four threads busy across the tile boundary and a second four buy
nothing — the probes' +5–6 % at eight in flight did not show, as their drift
qualifier allowed. Eight threads also cost a measurable tax on the hit-heavy
turn 3: +26 ms across both runs, absent at four. The reading is publication's
broadcast to every parked worker of the layer's reader — ≈ 400 one-read batches
× seven idle wakeups is the arithmetic that lands near 30 ms — and it is equally
consistent with plain mutex contention among eight workers on a one-read batch;
neither was measured directly. The refinement recorded either way: signal
`min(count, threads)` workers instead of all (sound because every worker
re-checks the claim predicate before parking; the shutdown path keeps its
broadcast) — not built here. So the default moved to (2, 4): no
extra descriptors, no extra stacks, no decode exposure.

**One control row did move.** Hits are identical to the expert at 300, 1k, the
turns, the 12k control and the long-decode arm, but the 2k warm request hit
4,709 experts at (2, 4) against 4,714 at (1, 4) (4,706 at (2, 8)) on an identical
demand of 9,357 — five experts, 0.05 points, ≈ 3 ms of residency against a 156
ms gain. The task changes no line of the planner, but it changes the planner's
inputs: a tile's plan is built at begin time against the slots then in flight,
and a second published batch changes which slots those are when the next
tile's victims are chosen. Recorded as explained, not re-measured (the 2k rows
are one pair per cell). Shutdown semantics, for the record: a batch whose reads
were all claimed before `destroy` returns success (its bytes landed); only a
batch with unclaimed reads returns `ECANCELED`.

**The model, checked.** The draft's conservative bracket (the hidden fraction of
the drive's time constant behind the GPU bank) predicted 3.51 / 6.52 / 10.40;
measured 3.54 / 6.47 / 10.30. The drive's own term (`io_fetch_ms × 8`, the parked
wait included at both cells; (1, 4) → (2, 4), paired) fell 264 / 388 / 592 ms at
300 / 1k / 2k — the realized per-expert time, that term over the misses, 1.019 →
0.937 ms, 1.019 → 0.930, 0.970 → 0.842 — and about half of it reached the wall
at 300 and 1k, the rest staying hidden; at 2k the stage is GPU-bound and the gain
is its exposed remainder. The effective rate on
the routed stage at 300 rose from ≈ 2.5 to ≈ 2.65 GB/s. What is left of the fetch
term is the per-read latency at 3–4 outstanding and the miss count itself — the
pool's size and eviction on a cached context (Task 3's re-pricing found a 38-token
turn fetching 2 GB) — not concurrency, which this task retires for the chapter.

**After T2** (commit d3efdeb, 2026-09-04; `SHRIKE_EXPERT_IO_BATCH_DEPTH=2` the
default, `=1` the A/B; mini, warm arms, server walls, paired):

| shape | warm wall | prefill hit rate | experts fetched |
| --- | ---: | ---: | ---: |
| 305 new tokens | 3.54 s | 58.4 % | 3,218 (5.6 GB) |
| 1,085 | 6.468 | 51.6 % | 4,387 (7.6 GB) |
| 2,125 | 10.30 | 50.3 % | 4,648 (8.0 GB) |
| turn 2 (38 new on a 2k context) | 1.60 | 62.5 % | 1,232 |
| turn 3 (36 new) | 1.33 | 86.3 % | 416 |
| 12k, first request after launch | 68.31 | 33.6 % | 18,864 |

Same binary, `=1`: 3.69 / 6.67 / 10.46 s, turn 2 1.64, turn 3 1.33. Golden
identical on both boxes and both profiles at every cell. Scope: measured on a
launch without an MTP sidecar; the speculative prefetch off (its default). The
deeper-lookahead follow-on, repriced now that batches overlap: two published
batches already keep four threads busy across the tile boundary at 3–4 misses per
tile, so a third batch (and the two-tile lookahead to feed it) adds reads in
flight only where the thread count would — and the thread count measured null;
still ≤ 1 %, not scheduled.

## Levers, ranked for these shapes (modelled from step zero)

- **First-chunk hit rate — LANDED as Task 0** (`SHRIKE_PREFILL_SWEEP=carry`):
  −1.7 / −1.2 / −0.4 s at 300 / 1k / 2k, neutral after a long answer. What is
  left of the intercept after it: ≈ 3.4 s at 300 tokens (3,200 misses, 5.6 GB)
  — bytes per expert and the pool's size are the remaining levers on it.
- **Bytes per expert** (the same term) — **two fetches in flight LANDED as
  Task 1** (`SHRIKE_PREFILL_FETCH_DEPTH=2`): −2.5 / −1.8 / −2.1 % at 300 / 1k /
  2k, the inter-tile gap closed, hits unmoved. The C reader publishes one batch
  at a time (`submit_batch` parks a second caller on `batch_idle`), so queue
  depth did not rise: the drive runs at ≈ 2.5 GB/s effective here because a
  tile's 3–4 misses fill at most four threads for one wave. **The reader publishing two batches at once LANDED as Task 2**
  (`SHRIKE_EXPERT_IO_BATCH_DEPTH=2`): −4.1 / −3.1 / −1.5 % at 300 / 1k / 2k,
  hits identical except five experts at 2k (plan-time inputs), decode unmoved; the batch depth is the lever and the thread
  count is null at 3–4 misses per tile (eight threads also tax hit-heavy turns
  through publication's broadcast). Concurrency is retired for this chapter:
  what is left of the term is the per-read latency at 3–4 outstanding and the
  miss count itself (the pool's size and eviction on a cached context).
- **A small-row prefill path** for follow-up turns (≤ 64 new rows): the matrix
  kernels' per-dispatch floor costs 21–23 ms per new token over 40 layer-chunks;
  a decode-style or scalar path for tiny chunks could take ≈ 0.5–1 s off a
  1.4–2.1 s turn.
- **Decode** (v12 Task 17: hit-rate-bound, 22 / 18 / 12.5 tok/s by shape):
  prefetch accuracy on a tools context, cross-layer miss queue depth, the
  drive's latency (an external NVMe on the mini is a copy-the-model experiment).

Not levers here: the GPU kernels (busy ≤ 70 % of the span below 2k tokens); the
ANE (attention is 4 % of a 1.4k prefill); speculation (retired at v12 Task 17).

## Method

The v12 protocol carried over: mini-first verdicts (the M4 Pro is the check);
one send per server lifetime for whole-chunk rows and `settle_done` before a
warm second send; a distinct prompt per arm (no prompt a prefix of another);
`SHRIKE_RUNNER_STATS=1 SHRIKE_KERNEL_STATS=1`; the gap split read before any gap
is theorised about. Each task pre-registers its rule on this ledger's rows
before it runs, lands its knob either way (a measured null is a result), and
reports the same rows after. **The rule has no size floor** (Davor's ruling at
Task 1: ten real 1 % wins are a 10 % chapter, and a floor drops every one of
them). A default flips when the effect is **real** — the sign holds across
paired runs in both orders and the delta exceeds the run-to-run drift, with
more pairs for a smaller effect — and **free** — no control row regresses (the
other shapes, the turns, the 12k control, hit rate, memory pressure) and golden
is identical. A cost that does show is priced in the verdict, never hidden by a
bar; the code a lever leaves behind is the review's judgment. Scheduling-only
changes are golden IDENTICAL on both boxes and both profiles; anything that
reorders arithmetic is qualified as v12 did.

## Numerics policy

Sweep order, prefetch and slot policy change which expert is fetched when, never
what any kernel computes: golden identical is the bar for them. A small-row
prefill path that runs different kernels on a chunk is a numerics change and
follows v12's policy (the 2e-2 bar against the fp32 reference, golden recaptured
once per box with the digests in the verdict).

## Out of scope

- Kernel work on the prefill GEMMs and attention (v12's follow-ons stay there).
- Speculative decoding (v12 Task 17).
- The prompt cache's interior snapshots (an edited document re-prefills; a
  cache-chapter item, recorded, not scheduled here).

## Risks

- The pool's contents after a long decode are the aging-LFU winners, not the
  sweep's tail; the parity lever may pay only for short answers — measured, not
  assumed.
- The mini is production: every arm restarts the server on 8081; Turbo on 8080
  is never touched; one model process at a time.
