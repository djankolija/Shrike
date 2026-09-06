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
loop** (until the close's collapse, below): the original at `=1` runs the scheduler's `decide` and its
commit-before-append valve; the lookahead at `=2` runs neither — its bootstrap
re-plans after draining one pending batch and throws `expertCacheUnplaceable`
if the cache still has no room, which `fitting`'s budget makes unreachable at
128 slots (at most three batches plus one in-flight tile are held when a plan
is attempted, leaving 96 free for an 8-expert plan). The task review's
judgment: collapse them into one loop as a follow-on before the chapter merges
to main, in its own commit with its own golden pair, and first if any task
edits the loop before then; the begin/await/drain sequencing is factored into a
host-testable decision with it (today it lives only in the runner, covered by
golden). Done at the chapter's close (the close section after Task 5).

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
wait included at both cells — a sum of overlapping windows from this task on, valid
as a delta between cells, not as elapsed time; (1, 4) → (2, 4), paired) fell 264 / 388 / 592 ms at
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

## Task 3 — the follow-up turn below the matrix kernels' row minimum (commit a1158b6)

A card's follow-up turn, measured with the answer the model actually gave chained
into the next request (the rig's `turns-live` phase, new in this task), is 21
new tokens on a 2,345-token cached context. At 21 rows the chunk fell below
three 32-row gates — the K/V and O projections' matrix dispatch, attention's
matrix path, the shared expert's — and ran a GEMV per token, the tiled
attention kernel and the per-row shared expert instead. The routed experts'
gate reads the configured chunk (4,096) and was already on the matrix path;
the GDN scan's 64-row gate is priced at ≤ 33 ms and left alone.
`SHRIKE_PREFILL_MATRIX_MIN_ROWS=16` (the new default; 3…32; `=32` the A/B,
today's fixed thresholds) lets the three matrix paths take a chunk of 16 rows
or more. The kernels already mask a partial row tile — every chunk whose length
is not a multiple of 64 runs one — so nothing new is computed; the switch moves.

Measured on the mini, one binary (the knob as the A/B), a fresh server per
chain, the live answer chained, every run after the first reusing its built
payloads so an A/B sends identical bytes:

| request | new / cached | n = 32 | n = 16 | Δ | per-role GPU at 32 → 16 (ms) |
| --- | ---: | ---: | ---: | ---: | --- |
| **turn 2 (live answer)** | **21 / 2,345** | **2.088 / 2.083 / 2.128 s (paired 2.100)** | **1.406 / 1.423 / 1.411 (1.413)** | **−32.7 %** | attention 434 → 54, GDN 340 → 164, shared 67 → 22, routed 406 → 412 |
| turn 3 | 36 / 2,359 | 1.464 / 1.468 / 1.471 | 1.455 / 1.462 / 1.456 | −0.6 % | unchanged (62 / 187 / 22 / 538) |
| turn 2, padded user turn | 55 / 2,345 | 1.803 | 1.820 | +0.9 % | unchanged (71 / 216 / 23 / 645) |
| 300 / 1k / 2k first turns (one pair) | | 3.574 / 6.464 / 10.308 | 3.635 / 6.481 / 10.307 | drift (no gate reachable at ≥ 305 rows) | unchanged |
| 12k, first request after launch | 12,285 / 0 | 68.277 | 68.173 | −0.15 % | unchanged |
| long-decode arm, decode tok/s (314 / 405 tokens) | | 14.02 / 14.17 | 14.08 / 13.99 | noise | |

Three pairs in both orders; repeats agree to ≤ 45 ms. **Every completion is
byte-identical between the two values** — turn 2 and turn 3 in all three
pairs, the 55-row chain, the six whole-chunk completions, the 12k control and
both long answers (901 and 1,023 characters) — and golden is identical on both
boxes and both profiles at 16 (the `short` profile is 13 rows, below the knob;
`long` is one chunk). The engaging turn's routed tile count moved 339 → 338 and its expert lookups
2,568 → 2,566, deterministically in all six runs at n ≤ 16 against all three
at 32 — a planner output, so the router's top-k moved by one expert on one
token as the chunk's projections and attention moved within the 2e-2
tolerance: the accepted numerics change reaching the router's margin, output
bytes unchanged. The hit counters alone are not that read: they move by ± 3
between runs where nothing can engage (the 2k control 4,707 → 4,704 at 2,125
rows, one chunk — counter jitter at fetch depth 2), and turn 3's residency
shifted the same way downstream of the changed KV (2,673 / 724 → 2,676 / 721 in
all three pairs, bytes identical). Device tests at 21 and 3 rows against the fp32
reference: attention max abs 1.5e-5 / 2.2e-4, rel 5.0e-4 / 6.2e-4; matrix vs
tiled 1.5e-5 / 2.4e-4; the shared expert (21 rows only) max abs 9.8e-4, rel 3.1e-4 — against 2e-2.

**The model landed on every role.** The draft fit the two matrix-path rows (36
and 55) per layer and predicted, for the 21-row turn at n ≤ 21: attention 54.2
ms, GDN 162.6, shared 22.0; measured 54 / 164 / 22. The modelled wall was 1.48
s and the measured 1.41: the gaps between stages shrank with the GPU, the
optimistic variant. Cells n = 8 and n = 4 equal 16 on the 21-row turn (1.413 /
1.420 s) and cost turn 3 +6 / +12 % because a chained turn's cache settle is a
restore whose remainder is 14 rows at turn 2 (29 at turn 3) — below 16 it
engages and changes the next turn's residency (hits 2,686 / 714 against 2,673 /
724, bytes identical). So 16 is the cell; the knee below 14 rows is unmeasured
(no shape exercises it). One engaging chunk at the shipped default is
unmeasured: turn 3's own restore settle re-prefills 29 rows, in [16, 31], so it
runs the matrix kernels and rewrites the stored KV within tolerance for
whatever turn 4 would follow — the chain's byte-diff is the only coverage, and
it ends at turn 3. The floor of 3 keeps the MTP verify pair (2 rows) and
a first request's rewind settle (2 rows, nine of nine archived) on today's
kernels.

**After T3** (commit a1158b6, 2026-09-04; `SHRIKE_PREFILL_MATRIX_MIN_ROWS=16`
the default, `=32` the A/B; mini, server walls):

| shape | wall | prefill hit rate | experts fetched |
| --- | ---: | ---: | ---: |
| follow-up turn, 21 new on a 2,345 cached context (after a live 219-token answer) | **1.41 s** | 67.6 % | 832 |
| turn 3, 36 new | 1.46 | 78.8 % | 721 |
| follow-up turn, 55 new | 1.82 | 64.1 % | 1,428 |
| 305 / 1,085 / 2,125 first turns | 3.54 / 6.47 / 10.30 (Task 2's paired rows; unchanged) | 58.4 / 51.6 / 50.3 % | |
| 12k, first request after launch | 68.2 | 33.6 % | 18,864 |

What is left of the 21-row turn: prefill ≈ 0.84 s = routed stage ≈ 0.48 (GPU
0.41, holding 832 misses ≈ 1.47 GB at ≈ 0.45 s of drive underneath) + attention
/ GDN / shared ≈ 0.24 + gaps ≈ 0.12; decode 0.52 for 8 tokens; outside 0.06.
The routed stage is now the largest term; Task 4 measured that its drive is
hidden under its per-tile GPU floor (a perfect pool is worth ≈ 10 ms here) and
that the miss count's price is in decode. Scope: measured on a launch without
an MTP sidecar; the GDN chunked scan's 64-row gate and the routed gate's MTP
scratch constraint are follow-ons.

## Task 4 — the expert pool's retention across the turn boundary (commit 04d4de5)

The draft opened this task on the follow-up turn's routed stage and its 832
misses on a cached context. Step zero re-priced the term before any code: the
existing `SHRIKE_EXPERT_CACHE_POLICY` values on the card's chain showed the
policy family moving both regimes in opposite directions (`lru` −502 decode
misses on the answer and +128 on the 8-token follow-up's decode; `lfu` identical
to `aging-lfu` to the counter, because the halving at 1,024 plans is per layer
and never fires inside a conversation), and the follow-up turn's prefill had
almost nothing exposed (turn 3's `lru` arm removed 304 of 721 misses for 4 ms of
`prefill_s` against a 16 ms drift: its routed stage sits at its per-tile GPU
floor with the drive underneath it). The miss count's price is in decode: a
miss costs ≈ 0.93 ms by four independent measurements and the 219-token answer
carries 7,451 of them, 41 % of its 16.8 s.

**Step 1 built the measuring instrument.** `SHRIKE_ROUTE_TRACE` gained a
request-start line and a per-tile prefill line carrying each expert's row count
and last row (`r <cached> <prompt>`, `p <position> <layer> <tile> e0…e7 | n0:l0 …`;
the settle re-prefills fall between requests and are reported apart), and
`tools/expert-pool-replay.py` replays a capture against the pool's policies at
128 slots per layer with the streamer's exact rules (hits reserved before victim
selection, the three-tile `avoidingSlots` lookback, ties by count then last use
then slot index, the halving cadence). Validated against production on six
requests across two shapes: every request within ±1 miss, prefill and decode
(the card chain 9,370 / 831 / 722 and 7,451 / 240 / 97 against 9,370 / 832 / 721
and 7,451 / 240 / 97; the short-prompt pair 7,639 / 4,014 and 10,059 / 153 against
7,639 / 4,014 and 10,059 / 154). The offline verdict on both traces (decode
misses on the answer, card chain / pair; production 7,451 / 10,059; the
captures predate the protection knob, so the tool reproduces these rows with
`--protect off`):

| candidate | card answer | pair answer | the follow-up turns |
| --- | ---: | ---: | --- |
| Belady, the ceiling | 2,485 | 3,773 | 380 / 176 prefill misses (production 831 / 722) |
| `lru` | 6,949 | 10,362 | decode +127 / +82; the warm second prefill +1,747 on the pair |
| `slru` 0.5 / 0.75, `arc`, `lru-2` | 6,919 / 7,017 / 7,146 / 7,265 | 10,074 / 9,982 / 10,403 / 10,195 | all regress the follow-ups' decode |
| aging at 256 / 64 / 16 plans | 7,451 / 7,039 / 6,958 | 9,885 / 9,626 / 10,357 | slides toward `lru` |
| prefill counts weighted by the prompt's rows | 18,383 | 13,052 | LFU pollution in its textbook form |
| the sweep ordered by the prompt's row counts | 7,257 | 9,396 | +44 / +128 prefill misses |
| **the sweep ordered by each expert's last use in the prompt** | **6,697** | **9,450** | +56 / +160 prefill misses; reversed 8,035 / 10,875 |
| production with a clairvoyant decode | 3,968 | 4,960 | |
| a clairvoyant post-sweep state with production's decode | 5,227 | 8,588 | |

The reuse structure explains the table: inside an answer 74 % of an expert's
reuses come within 8 tokens of its previous use and 97.6 % within a stack
distance of 128, so recency already does the steady-state job and no bounded
rule beats `lru`'s class there; Belady's capacity misses are ≈ 47 per layer, the
answer's 175 distinct experts minus the 128 slots, so its edge is which of the
prompt's experts it keeps across the prefill→decode boundary. Production's pool
leaves a chunk holding the last 128 experts the index-ordered sweep visited,
and every expert of a chunk carries its tile's plan clock, so the policy has no
recency to prefer the experts the prompt's final tokens used, which the answer's
first tokens reuse. Frequency was the wrong proxy for "needed soonest"; recency
in the prompt was the right one.

**Step 2 built two levers and measured them on the mini.** (1) `SHRIKE_PREFILL_SWEEP=recency`
(with `SHRIKE_PREFILL_SWEEP_TAIL`, default 96): a chunk's experts ranked by
last row, the most recent `tail` of them swept last, tiles packed inside the
head and the tail for balanced row weight (the plain order clustered light
early-used experts in the first tiles and heavy late-used ones in the last, and
the two-tile pipeline ran fetch-bound early and GPU-bound late: +0.47 s on a 2k
prefill, gone once balanced; a second +0.44 s was host time in the tile
composition and the per-tile protection set, gone once the order went through
the existing sort-key array and the protection through a per-layer `[Bool]`).
(2) `SHRIKE_EXPERT_CACHE_PROTECT=chunk`: while a chunk sweeps, slots holding an
expert the chunk still needs are ineligible as victims (the planner has the
chunk's routes before it sweeps; a graded fallback drops the protection for one
plan when too few eligible slots remain). It came out of tracing why the recency
order lost hits on a warm second prompt: each early miss evicted a resident the
same sweep needed later (240 such residents at use count ≥ 4, mean position 0.6
of the sweep).

The replay predicts the box to the miss for both: the recency order's decode
misses on the card answer 6,692 / 6,701 measured against 6,697 / 6,688
replayed, and a traced chain under both levers replayed at 6,701 / 707 / 613
against 6,701 / 707 / 612 measured. The recency order wins the first turn's
decode after a large prompt (−0.63 s on the 2k card, −0.72 s on the 300-token
prompt, −0.72 s on the 1k, +3.2 to +3.9 % tok/s) and loses everywhere a chunk
follows another, because it forfeits T0's carry benefit: the warm second prefill
of the 300 / 1k / 2k pairs +3.7 / +7.5 / +10.9 % and 12k +7.2 %, rows the
protection cannot rescue because nearly every resident is needed by the next
chunk and its fallback drops it. **Not free**; it stays a knob at default
`carry`, its refinement (a resident-first head: the chunk's needed experts that
are already resident swept first, so every hit is harvested before any
eviction, T0's trick made exact through the pool's residency, then the recency
tail) landed as Task 5 (below), interleaved after the box overruled the
grouped head. Protection alone was free on every row it ran, so the verdict round
paired it against production.

**The verdict** (mini, one binary, `SHRIKE_EXPERT_CACHE_PROTECT` as the A/B,
`off` = A today, `chunk` = C; paired in both orders where a wall is the verdict):

| row | A (today) | C (protect chunk) | Δ | misses A → C |
| --- | ---: | ---: | ---: | --- |
| the card's 21-token follow-up turn (×3) | 1.430 s | 1.385 s | −3.1 % | 832 → 705 (replayed 705) |
| turn 3, 36 new (×3) | 1.466 s (a third repeat at 1.700, a one-off stall, misses identical) | 1.458 s | −0.5 % | 721 → 613 (replayed 614) |
| the 2k first turn's answer, decode s (×3) | 16.773 | 16.794 | +0.1 % (drift) | 7,451 both |
| the 2k first turn's prefill s (×3) | 11.086 | 11.094 | +0.1 % | |
| warm 300 prompt after a 314-token answer (×2) | 3.439 s | 3.175 s | −7.7 % | 4,014 → 3,389 |
| warm 1k prompt after a 405-token answer | 6.261 s | 6.191 s | −1.1 % | 5,215 → 4,647 |
| warm 300 prompt after an 8-token answer (×2) | 3.582 s | 3.454 s | −3.6 % | 3,218 → 2,865 |
| warm 1k / 2k prompts after an 8-token answer | 6.553 / 10.323 s | 6.385 / 10.287 s | −2.6 % / −0.3 % | 4,387 → 4,066; 4,646 → 4,317 |
| 12k, first request after launch | 68.105 s | 68.209 s | +0.15 % | 18,864 → 18,457 |
| the long answers' decode tok/s | 13.06 / 14.01 / 14.03 | 13.04 / 13.82 / 14.01 | drift (one t300 repeat 22.93 s against 22.34–22.52 with identical misses) | unchanged |

Real: the follow-up turn and every warm prefill move by more than the drift, in both orders, with the miss counts the replay predicted. Free with one priced observation: in the `300` arm a fresh server's first request at 289 rows cost +0.27 s of prefill (5.48 → 5.76 s, both repeats, the routed-tile gap's host +129 ms), and the 1,069-row one +0.10 s; the SAME requests in the `d512-300` and `d512-1k` arms, same binary and launch, showed +0.017 s (gap host −61 ms) and +0.002 s. The number stands, its cause is not established (the per-tile scan explains at most half of the larger arm's delta and none of the smaller's), and it can only occur once per launch, before any resident exists to protect. Recorded as a follow-on to measure, not as a mechanism. Golden is identical on both boxes and both profiles at
`off` and at `chunk` at the landed commit, and at the two recency cells on the
M4 Pro at every amend and on the mini at the landed commit (the order changes
which experts share a tile and when they are fetched, never what any kernel
computes).
`SHRIKE_EXPERT_CACHE_PROTECT=chunk` is the default; `off` is the A/B.

**Side findings for the ledger.** After a request whose prompt has no cached
prefix, the prompt cache's settle re-prefills the whole prompt in the
background (`settle_reset reason=no_prefix_snapshot`: 967 tiles, ≈ 6 GB of
expert reads after a 3.6 s request, sweeping the pool); a follow-up turn's
restore settle costs 450–470 misses of its own (≈ 0.8 GB) off the critical
path. `expert_evictions` is arithmetically the misses minus the empty slots;
`expert_reloads` is a lifetime flag, trivially ≈ every miss after the first
sweep. The policy env parse lives in the streamer's `init`, so a bad value
fails per layer mid-request rather than at launch (recorded, not fixed here).

**After T4** (commit 04d4de5, 2026-09-05; `SHRIKE_EXPERT_CACHE_PROTECT=chunk`
the default, `=off` the A/B; `SHRIKE_PREFILL_SWEEP=carry` unchanged; mini,
server walls):

| shape | wall | prefill hit rate | notes |
| --- | ---: | ---: | --- |
| follow-up turn, 21 new on a 2,345 cached context | 1.385 s | 72.5 % | was 1.41 s at 67.6 % after T3 |
| turn 3, 36 new | 1.458 s | 82.0 % | |
| 305 / 1,085 / 2,125 first turns (warm, after an 8-token answer) | 3.454 / 6.385 / 10.287 | 62.9 / 55.1 / 53.9 % | were 3.54 / 6.47 / 10.30 after T2 |
| 12k, first request after launch | 68.209 s | 35.0 % | |
| decode on the 2k card's answer | 16.794 s for 219 tokens | | unchanged by design |

## Task 5 — the resident-first recency sweep (commit 17146ae)

Task 4 left the recency-ordered sweep as a knob: it wins the first turn's decode
after a large prompt and forfeits T0's carry benefit on every chunk that follows
another. Step zero captured two more route traces on the mini under today's
order (the 300 pair and 12k, beside Task 4's card chain and its 300-prompt
long-answer pair; every capture replays to the unit) and priced the refinement
offline: sweep the chunk's pool-resident experts first, so every hit is
harvested before any eviction, then the absent ones by recency. On all four
traces the replay of step zero's own order (resident then absent, each by
recency, tiles of eight) kept the plain recency order's decode gain whole (−754
misses on the card's answer, −609 on the 300 prompt's) and returned every row
that order had lost to at or under today's (the warm 305 after a long answer
3,389 → 3,317, 12k 18,457 → 18,271), with one warm row +41 misses of overlap
scatter (the 300 pair's warm, 2,865 → 2,906). The drafted composition packed
three row-balanced groups (resident, absent head, absent tail) and tiled the
concatenation flat at index's own tile count, within a few misses of those rows.

**Round 1 on the box overruled the composition.** With every miss count as
replayed (the card's answer 7,451 → 6,702, the 300 prompt's 10,059 → 9,485,
the warm 305 3,389 → 3,317), the follow-up turns cost +29 / +56 ms, the 50-row
turn after a long answer +130 ms and the warm 305 after a long answer **+0.64 s
(+20 %)**, with equal or fewer misses and less fetch work. The gap counters showed
where: the routed-to-routed host wait rose by 100 to 620 ms while the
shared-to-routed wait fell, scaling with the chunk's miss count. Sweeping every
hit first leaves each layer's misses in pure-miss tiles at its end, and at fetch
depth 2 a pure-miss tile's 8-expert fetch (≈ 14 MB, 4 to 5 ms) has one
preceding tile's ≈ 1.2 ms of GPU to hide under, where the index order spreads
2 to 3 misses per tile whose fetch hides under the previous tile. The replay
counts misses and cannot see exposure; a per-tile miss-density readout
(`t5-interleave.py`) ranked the landed order's cost the right way on every row
the box priced.

**The landed order interleaves.** A pure-resident head only while the residents
not yet swept exceed the slots minus six tiles' worth of misses (protection
starves inside the three-tile avoiding window otherwise: the naive interleave
gave back 195 misses on the 300 pair and 666 at 12k), then the chunk's absent
experts in recency order spread uniformly over the remaining tiles (the last
tile holds the most recent, the decode prize), each tile's free slots filled
with residents heaviest-first into the lightest tile (rows balanced), tiled
flat. The tail knob (`SHRIKE_PREFILL_SWEEP_TAIL`) no longer applies to this
mode: the interleave has no tail group, the cold-pool fallback fixes its own
tail at 96, and the residency line prints `tail=` only under `recency`.
Priced offline before the fix-up: misses at or below index on every
trace (the warm 305 after a long answer 3,325, the 300 pair's warm 2,846
against 2,865, 12k 18,321 against 18,457, the decode prizes unchanged) and the
modelled exposure below index on every chunk. Both implementations carry it
(`tools/expert-pool-replay.py --sweep-order resident-first`, the drafted order
kept as `resident-first-grouped`; `PrefillSweepOrder.residentFirstBalanced`),
and the box's traced chain replays against the tool at delta ≤ 1 while the
carry-captured chain predicts it within ±3.

**The verdict** (mini, one binary, `SHRIKE_PREFILL_SWEEP` as the A/B, `carry` =
A today, `resident` = B; paired in both orders where a wall is the verdict):

| row | A (carry) | B (resident) | Δ | misses A → B |
| --- | ---: | ---: | ---: | --- |
| the 2k card's 219-token answer, decode (×3) | 16.725 s | 16.013 s | −0.71 s, −4.3 % | 7,451 → 6,689 (replayed 6,689) |
| the 300 prompt's 314-token answer, decode (×2) | 22.456 | 21.923 | −0.53 s | 10,059 → 9,468 (replayed 9,468) |
| the 1k prompt's 405-token answer, decode | 29.078 | 28.220 | −0.86 s | 12,234 → 11,398 |
| the long-answer follow-up, turn 2 at 106 tokens (×2) | 8.664 | 8.555 | unmoved (one A repeat at 8.827) | prefill 705 → 707; decode 3,017 both |
| warm 305 after the long answer (×2) | 3.170 | 3.090 | −2.5 % | 3,389 → 3,325 (replayed 3,325) |
| warm 1,085 after the long answer | 6.176 | 5.951 | −3.6 % | 4,647 → 4,386 |
| warm 305 after an 8-token answer (×2) | 3.439 | 2.961 | **−13.9 %** | 2,865 → 2,845 |
| warm 1,085 / 2,125 after an 8-token answer | 6.467 / 10.287 | 5.889 / 10.342 | −8.9 % / +0.5 % (drift) | 4,066 → 4,084; 4,317 → 4,325 |
| the 21-token follow-up (×3) | 1.385 | 1.401 | **+16 ms, +1.2 %** | 705 → 707 |
| turn 3, 36 new (×3) / the 50-row turn (×2) | 1.461 / 1.604 | 1.474 / 1.615 | +13 / +11 ms | 613 → 616; 954 → 952 |
| 12k, first request after launch | 68.263 | 68.461 | +0.3 % (drift; prefill +0.45 s, decode −0.29) | 18,457 → 18,321 |
| the cold first requests' prefill: 2,125 rows (×3) / 289 / 1,069 / 2,093 | 11.070 / 5.473 / 7.890 / 10.931 | 11.111 / 5.492 / 7.939 / 11.085 | +0.4 % (×3); singles to +1.4 % | identical (a cold pool) |
| the same cold requests' 8-token decode | 0.754 / 0.868 / 0.889 | 0.523 / 0.527 / 0.528 | −0.3 s each | 506 → 213; 633 → 252; 640 → 242 |

Real on both verdict rows and on four of the five warm first turns after an
answer (the fifth, the 2,125-row warm, +0.5 %, within drift): the warm rows
are a second prize the replay could not predict (their misses are
equal; the interleave hides the fetch under the GPU better than the index
order does, which the plain recency order had made 4 to 11 % worse). Free
with one priced observation: the two short follow-up turns +13 to +16 ms,
consistent on all three pairs, at the drift band Task 4 recorded (±16 ms on
`prefill_s`); the gap counters put ≈ 10 ms of it in the shared-to-routed gap
that holds the route build (the residency snapshot and the composition, ≈ 0.45
ms per layer) and the rest in the first tile's fetch and two misses, a modelled
attribution because the phase timer's output never reaches the server's log (a
follow-on). The cold first requests' prefill +0.4 % on three repeats at 2k with
single runs to +1.4 %: a cold pool's tiles are all-miss under any order, so
only the packing's GPU overlap can move; recorded beside Task 4's +0.27 s as a
row to measure, not a mechanism. Golden is identical on both boxes and both
profiles at `carry`, `resident` and `recency` at every landed commit.
`SHRIKE_PREFILL_SWEEP=resident` is the default; `carry` and `recency` are the A/B.

**Side findings for the ledger.** The recency reference on the same binary
reproduced Task 4's trade (the card's answer 15.94 s, the warm 305 pair 3.675 s
against carry's 3.439). The order's cold-pool fallback (no resident, or nothing
absent) has to delegate to the grouped composition rather than a single pack:
the grouped order's flatten-then-rechunk discards each group's own bin
boundaries on chunks over 96 experts and a single pack does not reproduce it,
caught by the acceptance traces where the hand-built tests could not. The
runner's `SHRIKE_PHASES` print goes to a stdout that is fully buffered under the
server's redirected launch and is lost when the next launch kills the process.

**After T5** (commit 17146ae, 2026-09-06; `SHRIKE_PREFILL_SWEEP=resident` the
default, `carry` / `recency` the A/B; `SHRIKE_EXPERT_CACHE_PROTECT=chunk`
unchanged; mini, server walls, round 2's means):

| shape | wall | prefill hit rate | notes |
| --- | ---: | ---: | --- |
| follow-up turn, 21 new on a 2,345 cached context | 1.401 s | 72.5 % | was 1.385 after T4 (+16 ms, the priced observation) |
| turn 3, 36 new | 1.474 s | 81.9 % | was 1.458 |
| 305 / 1,085 / 2,125 first turns (warm, after an 8-token answer) | 2.961 / 5.889 / 10.342 | 63.2 / 54.9 / 53.8 % | were 3.454 / 6.385 / 10.287 after T4 |
| 305 first turn, warm after a 314-token answer | 3.090 s | 57.0 % | was 3.175 |
| 12k, first request after launch | 68.461 s | 35.5 % | was 68.209 (drift) |
| decode on the 2k card's 219-token answer | 16.013 s (13.68 tok/s) | | was 16.794 after T4 |
| decode on the 300 prompt's 314-token answer | 21.923 s (14.32 tok/s) | | was 22.456 (carry, the same round) |

## The chapter close

**The two-loop collapse** (commit c04e43b, 2026-09-06; Task 1's follow-on,
taken first at the close as its review's condition asked). `encodeRoutedMoEPrefill`
had two routed tile loops: the original at `SHRIKE_PREFILL_FETCH_DEPTH=1` and the
lookahead's at `=2`, each with its own copy of the batch bookkeeping, plus two
helpers for the lookahead's bootstrap and its successor plan. They are one loop
now, `PrefillRoutedTileSequencer` (`sources/Shrike/Kernels/Prefill/MoE/`), driven
at every depth through a small driver protocol (`PrefillRoutedTileDriver`: the
batch counts and held slots, then plan, begin, abandonPlan, encode,
commitOpenBatch, drainOldestBatch and abandonBegunFetches). The runner's
`ExpertStreamedTileDriver` owns the pool's kept plans and begun fetches by tile,
the open and pending command buffers, the slot lifetimes and the chunk's
protection; a test's recording driver answers the same calls from counters. What
the collapse reconciles, per the review: the lookahead is a predicate
(`plansLookahead` decides whether the successor is planned ahead at all,
`shouldBeginLookahead` whether its resolved plan is begun); a tile whose fetch the
lookahead began is carried into its own turn and skips planning; every other tile
is planned at its turn and issued through `decide` (the kept plan abandoned, the
oldest batch drained, the tile re-planned on `.drainBeforeIssue`); and the
commit-before-append valve runs on the actual plan availability at both depths.
The lookahead path had hardcoded that availability to `true`, so its valve was
dead and its bootstrap re-planned after one drain without committing the open
batch first; the depth-1 recovery now applies at both depths, still unreachable at
128 slots by `fitting`'s budget (Task 1's Minor 4). A begun fetch is kept by the
driver before its slot-lifetime check, so any throw after a begin is waited out by
the sequencer's one catch (Task 1's Important 1, now structural). The trace line
is written at begin, in tile order as before. The begin/await/drain order is
asserted on the host for the first time: nine traces in
`PrefillRoutedTileSequencerTests` (the successor begun before the tile is
awaited; each tile at its own turn without a lookahead; the last tile begins no
successor and zero tiles make no call; an unplaceable tile commits the open batch
and drains it before the re-plan, at depth 1 and at the lookahead depth; a
declined lookahead planned again at its own turn; the open batch counting against
the depth with the committed ones; a slotless pending batch drained and the kept
plan abandoned; a failure waiting out every begun fetch). `encodeRoutedMoEPrefill`
244 to 153 raw lines, still over the lint threshold so its baseline entry stands;
the runner's diff is −362 / +217 lines; `decide` and `batchAction` are unchanged.
**Golden identical** at `SHRIKE_PREFILL_FETCH_DEPTH` 2 (the default) and 1, short
and long, on both boxes; gates 1 to 4 green (build 0 warnings, lint 0 / 225, links 56 files 0 broken, tests 1,257 / 1,257 in 168 suites, 601.8 s). No timing row: in every
case the 128-slot budget makes reachable, the sequencer issues the same plan,
begin, await, commit and drain sequence the two loops did, so Task 1's and Task
5's rows stand as measured; the mini serves the collapsed build at the bare
launch.

**The ThreadSanitizer suite, once at the close**: 1,257 / 1,257 in 168 suites, 2,835 s,
no report (the tree before the review's fix round; the round's changes are a guard on an
invariant path, a parser that now throws, a seed constant, comments and test assertions,
so the run stands for the amended tree).

**The whole-branch review** (a fresh reviewer over 3774ef1..2f16c0d, the chapter's 22
commits): ready after the must-fix list, 0 Critical, 3 Important, 11 Minor, none of the
22 deferred items a must-fix (10 fine as they are, 12 moot). The reviewer walked every
sequencer trace by hand against the loop and found the collapse correct, its error path
strictly better than the loops it replaced, and no kept plan able to leak. The fix round,
by amend and one fix-wave commit: the docs cited two pre-amend shas for Task 5 that the
push would have garbage-collected, now the landed `17146ae` and `c99fa0e`;
the driver's in-flight avoidance fell through silently if a successor were ever planned
before its predecessor's fetch was begun, a silent-numerics class, and now throws
(`ExpertStreamedTileDriver.plan`); the error-path test asserted only that the abandon was
called, and now asserts what the driver released, with a kept-plan case beside it (commit
c04e43b, the collapse amended). The fix wave (commit e0bba79): `SHRIKE_PREFILL_SWEEP`
fails closed on an unknown value like the knob's siblings (unset and empty still take
`resident`); the ready banner prints a reader configuration the streamer will refuse as
`expert_io=invalid(...)` instead of the defaults it is not running; `packByRows`' two
overloads seed their search the same way; two tool notes. Left as they are by ruling: the
scheduler's unread `.prefetchNext` payload (v12's close ruled the same: it serves the
scheduler's tests) and `ExpertCacheProtectMode`'s home in the streamer's file. The
reviewer also asked for one measurement the collapse's golden pair does not give, a
timing row on the shipping binary, taken below. The re-review found every finding
addressed with no new breakage and one overclaim of this session's own: the second
error-path test promised a kept-plan case the sequencer cannot reach (a plan is kept only
between its planning and its begin, and nothing that throws sits between them). Its
follow-up, amended into c04e43b: the driver now abandons a kept plan whose begin throws
(the streamer's begin throws only before it executes a plan, and an unexecuted plan's
miss slots stay reserved until abandoned), and the test pins what is reachable, a failure
inside a begin waiting out the begun predecessor.

**The timing row on the shipping binary** (e0bba79's release build, the mini, two
fresh-server `pair 300` runs through the chapter's rig at the default, measured): the warm
305-token first turn **2.934 / 2.925 s** at 63.2 % hits (4,881 / 2,845) against Task 5's
landed 2.961 s at 63.2 %, inside the chapter's drift band; the cold 289-row first request
7.698 / 8.003 s (7,639 misses, as every cold row). Golden identical at fetch depth 2 and 1,
short and long, on both boxes on the fix-round build; gates 1 to 4 green on it (build 0
warnings, lint 0 / 225, links 57 files 0 broken, tests 1,259 / 1,259 in 168 suites); the
mini serves it at the bare launch. The chapter's closing table describes the binary that
ships.

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
- **The follow-up turn below the matrix kernels' row minimum — LANDED as Task 3**
  (`SHRIKE_PREFILL_MATRIX_MIN_ROWS=16`): this entry first read the turn's cost as
  "the matrix kernels' per-dispatch floor" and priced a scalar path for tiny
  chunks. The sign was backwards: a 21-token follow-up already ran the scalar
  paths, below three 32-row gates, and that was the cost — 2.10 → 1.41 s (−33 %)
  with every completion byte-identical. What was left of the follow-up turn,
  its routed stage, is mostly its per-tile GPU floor (Task 4 measured the
  drive under it at ≈ 10 ms).
- **The expert pool's retention across the turn boundary — LANDED as Task 4**
  (`SHRIKE_EXPERT_CACHE_PROTECT=chunk`): a chunk's own earlier misses no longer
  evict residents the same chunk needs later, 10 to 15 % of a warm chunk's
  prefill misses under production's sweep order (−15 % on the 21-token turn
  and the warm 300 after a long answer, −11 % on the 300 pair, −7 % at 1k and
  2k, −2 % at 12k), free on every row; the
  follow-up turn −3.1 %, the warm 300 prompt after a long answer
  −7.7 %. The replay tool and the route trace it runs on are the
  chapter's instrument for the pool from here: Belady's ceiling on the answer
  is −67 % of decode misses, the recency-ordered sweep reaches −10 % of them on
  the first turn's answer and is a knob (default `carry`) because it forfeits
  T0's carry benefit on consecutive chunks; its resident-first refinement
  landed as Task 5.
- **The resident-first recency sweep — LANDED as Task 5**
  (`SHRIKE_PREFILL_SWEEP=resident`): the chunk's pool-resident experts ahead of
  any eviction and its absent experts by recency interleaved across the tiles;
  the first turn's decode after a large prompt −0.5 to −0.9 s (the card −4.3 %),
  four of the five warm first turns after an answer −2.5 to −14 %, the fifth
  within drift (the interleave hides the fetch under the GPU better than the
  index order, a prize the replay could not see), the two short follow-up turns +13 to +16 ms at the drift band. The
  drafted grouped composition was overruled by the box (miss clustering at each
  layer's tail, +0.64 s on a warm 305): the replay counts misses, not exposure.
- **Decode** (v12 Task 17: hit-rate-bound, 22 / 18 / 12.5 tok/s by shape; Task
  4: a miss costs ≈ 0.93 ms and the answer's misses are 41 % of its decode).
  Belady at 128 slots removes 67 % of them on the traced answers; recency
  already reaches 97.6 % of an answer's reuses, so the remaining lever is the
  prefill→decode boundary (the resident-first recency sweep, landed as Task 5),
  not the eviction rule; then prefetch accuracy on a tools context and the
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
