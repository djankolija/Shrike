# v15, the miss window: hiding the misses decode cannot avoid

The design document for the chapter after v14 ([v14-decode.md](v14-decode.md)); its
implementation plan is [v15-implementation-plan.md](v15-implementation-plan.md), whose
per-task checkboxes are the status of record. Every number here is MEASURED on the
mini (the deploy target) unless marked modelled; "modelled" is arithmetic over
measured rows, never a measurement.

## The problem

Production on the mini answers the card / the 300 / the 1k prompts at **14.1 / 14.8 /
15.0 tok/s** after v14 ([v14-decode.md](v14-decode.md) "After T2"). The same box runs
warm prompts at 18 to 19 tok/s and the miss-free body is 22 modelled. The gap is the
miss count and nothing else: the rate is the miss count at every scale the v14 ledger
looked at (a 16-token window inside one answer swings 11.9 tok/s at 42 misses per
token to 18.6 at 10 to 13), a decode token's wall is a 41 to 49 ms body plus 0.83 to
0.87 ms per expert miss, 17 to 19 of 40 layers miss per token, and the GPU stands idle
between `moe_phase1_hit` and the fixup for **18.0 to 19.7 ms per token, 26 to 27 % of
the wall** ([v14-decode.md](v14-decode.md) step zero rows 1 to 3).

The misses cannot be avoided on this box. Eviction policy is exhausted (v13's replay:
recency captures 97.6 % of an answer's reuses, about 1.7 % of headroom), and a bigger
pool needs RAM the 16 GB mini does not have (`--ram-budget 8G` is the measured
optimum). Dropping a missed expert instead of reading it is recorded as the nuclear
option and is not pursued: the routing mass by rank on the card is 0.22 / 0.16 / 0.13
/ 0.12 / 0.10 / 0.09 / 0.09 / 0.08, so ranks 5 to 8 carry 37 % of the mass and
dropping is not a small perturbation; it would need a quality metric the repo does
not have. **So the misses are hidden**: read before the layer asks, under the compute
the drive would otherwise idle through. That is the redesign v14 recorded as a
candidate task with four preconditions
([v14-implementation-plan.md](v14-implementation-plan.md) "Candidate tasks"), and the
first of them was a question about the drive that this chapter's step zero answers.

## Step zero: the contention probe (2026-09-06, zero runtime code)

v14's Task 1 shipped the mechanism (the next-layer router probe, the prefetch ring,
the adoption) and measured it a loss, −2 to −3 % at top-4 and −7 % at top-8, because
the drive served the ring's reads beside the demand reads and slowed them by about
what the hidden layers saved. Whether reads can be placed so that they contend with
nothing was the unknown. The probe is a standalone C program run beside the idle
production server (`~/.claude/handoffs/archive/shrike-v15-step0-contention/`,
`probe-summary.md` with every table, the source, the scripts, 17 mini runs and 2
local): whole-expert 1,769,472 B `F_NOCACHE` preads over ornith15's 40 packed-expert
files, a demand batch of E experts through a condvar worker pool as `expert_io.c`
issues it, the next batch `gap` after the previous one completed (decode is about
1 ms), and N background reads of the same size placed three ways: `cont` (always,
the T1 ring's shape), `burst` (N issued when a demand batch completes, none more
until the next completion) and `steady` (looping only while no demand read is in
flight). p50 milliseconds, seeds 1 and 7 agreeing within 0.02 ms on every arm.

### 1. The host's state is a term the drive does not own

The first matrix, run as the archived inflight probe was (the main thread sleeping
through the gap), put a read alone at 0.941 / 0.930, v14's row 4 (0.949), and then a
read placed in the gap that finished before the demand read made the demand read
FASTER, 0.81. A drive with no memory of an idle gap cannot do that, so the controls
ran:

| cell (gap 1 ms, E = 1) | demand p50 |
| --- | ---: |
| alone, the host sleeps through the gap and spins its last 0.3 ms | 0.886 to 0.975 |
| alone, the host spins for the read but sleeps through the gap | 0.928 / 0.978 |
| alone, the host parks on the condvar for the read and SPINS through the gap | 0.796 / 0.793 |
| a thread spinning on another core through the gap, no I/O at all (five runs) | 0.795 to 0.798 |
| a whole-expert read at the gap's start, the host sleeping | 0.810 |
| a 16 KB read at the gap's start | 0.863 |
| a 256 KB read timed to end 0.05 ms before the demand read | 0.790 |

A core kept busy through the gap, by a spin or by any read, takes 0.13 to 0.15 ms off
the next demand read; the drive itself has no idle penalty. Production's host spins on
the router readback through a layer's compute (lever B's word wake) and parks on the
reader's condvar during the read (`expert_io.c:398`), which is the third row's shape:
**0.80 ms is the production-faithful lone read at decode's spacing, not 0.95.** Two
consequences for the record. The "0.175 ms on the first read of every missing layer"
that [v14-decode.md](v14-decode.md) attributed to the drive at decode's spacing (its
row 4, and the reading on line 516) measured the probe's sleeping host, and is
withdrawn; the closed chapter's document is left as written. And every drive probe in
the record that sleeps between reads overstates a read by 0.13 to 0.15 ms on the mini.
Whether production pays any of this term is MODELLED, not measured: the chapter's
ledger checks the per-read fetch time against 0.80 before anything is priced on it.

### 2. Contention, against the production-faithful baseline

The matrix re-run with the host spinning through the gap (`mini-run14`, gap 1 ms,
E = 1, 800 batches; `mini-run15` / `mini-run17` at 2 ms; `mini-run16` at E = 2):

| arm | N | demand p50 / p90 | vs alone | in flight at issue | background GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| alone | 0 | 0.796 / 0.903 and 0.800 / 0.911 | 0 | 0 | 0 |
| cont | 1 / 2 / 4 / 8 | 1.026 / 1.541 / 2.377 / 4.383 | +0.23 / +0.74 / +1.58 / +3.59 | N | 1.99 / 2.70 / 3.03 / 3.20 |
| burst | 1 | 0.785 / 0.891 | −0.01 | 0.00 | 1.00 |
| burst | 2 | 0.848 / 0.999 | +0.05 | 1.12 (bins 0: 0.783, 1: 0.851, 2: 0.861) | 1.92 |
| burst | 4 / 8 | 1.934 / 4.051 | +1.14 / +3.26 | 3.35 / 7.44 | 2.49 / 2.88 |
| steady | 1 / 2 / 4 / 8 | 1.054 / 1.343 / 2.111 / 4.212 | +0.26 / +0.55 / +1.32 / +3.42 | N | 1.77 / 2.25 / 2.56 / 2.89 |
| 2 ms gap, alone | 0 | 0.797 to 0.803 | 0 | 0 | 0 |
| 2 ms gap, burst | 1 / 2 | 0.795 / 0.794 and 0.800 / 0.797 | 0 | 0.02 / 0.04 | 0.62 / 1.23 |
| 2 ms gap, burst | 4 | 0.888 / 0.880 (bins 0: 0.767, 1: 0.878, 2: 0.922, 3: 0.920) | +0.09 | 1.69 | 2.35 |
| E = 2, alone | 0 | 1.375 / 1.369 (0.69 per expert) | 0 | 0 | 0 |
| E = 2, burst | 1 / 2 / 4 | 1.357 / 1.409 / 2.475 | −0.02 / +0.04 / +1.10 | 0.03 / 1.19 / 3.41 | 0.73 / 1.42 / 1.99 |
| E = 2, cont | 1 / 2 | 1.751 / 2.134 | +0.38 / +0.76 | N | 1.57 / 2.18 |

**Readings.**

- A background read that completes before the demand read is issued costs it
  nothing: one whole expert per 1 ms gap, two per 2 ms gap, one beside a two-expert
  batch, all within 0.02 of alone. **The redesign's precondition (2), "issued where
  it overlaps no demand read", is met on this drive, and the placement is what meets
  it.**
- A background read still in flight when the demand read is issued shares the drive
  with it until it finishes: the tail of a QD2 pair with about 0.1 ms left costs
  +0.05, a read at a random point of its life +0.26, four reads at their tails +0.09
  with the bins climbing with the count. At larger N the demand read is one fair
  share of a 3.5 GB/s drive, (N + 1) × 0.506 ms, which predicts the `cont` arms within
  0.15; a lone read gets 2.2 GB/s, not the drive's 3.5.
- The M4 Pro's external drive (`local-run1`) has no host-state term and more QD2
  headroom (a read at its tail costs the demand read nothing there); the same shape
  past N = 2. Iteration signal only.

### 3. The placement rule

What the drive hides at zero demand cost is **one whole expert per layer's compute**
(0.73 to 0.80 ms of read inside a 1.0 to 1.1 ms layer, 1.0 GB/s beside the demand
stream): up to 40 reads per token against 28 to 31 misses per token. Two per layer
(QD2, 1.10 ms each) do not fit inside a 1 ms gap and tax the following demand read
+0.05, or fit inside a 2 ms stretch (an all-hit layer and its neighbour) for free. So
the rule the redesign is built under: **issue a prefetch read the moment a layer's
demand batch completes, or at an all-hit layer's plan time; keep at most one read per
layer in flight (two where the stretch is 2 ms); never start one within 0.8 ms of a
router readback that may resolve into a miss.** Reads placed by that rule pay no
contention. T1's ring paid because it placed them beside the demand reads.

## The mechanism in the tree (every citation against `4cc6e58`)

- **The ring.** `ExpertPrefetchRing` stages predicted reads in `topM` fresh shared
  buffers outside the authoritative cache so a wrong prediction evicts nothing
  (`ExpertPrefetchRing.swift:4-9`, `:34-49`; `topM` and the stride from
  `RealForwardRunner.swift:933-948`). `begin` dedupes against the resident set and
  its own in-flight slots, claims free slots and submits one batch
  (`ExpertPrefetchRing.swift:54-98`); `readyBuffers` hands back only `.completed`
  slots and never waits (`:99-112`); `consume` adopts (`:114`); terminal slots are
  reclaimed at the next `begin` and completed-but-unadopted ones counted
  (`:126-140`). No test file covers the ring.
- **The probe.** A second router GEMV in the attention tail scores layer L + d's
  router against layer L's post-attention normalised residual and writes the top-k
  to `prefetchPredictionIndices` (`RealForwardRunner.swift:2898-2905`,
  `:3515-3533`); the host reads it back with the router readback, in top-k order
  (`:3172-3186`); `d` is `SHRIKE_PREFETCH_PROBE_DISTANCE` (`:2068-2069`). **A defect
  found at this chapter's opening:** the probe binds `effectiveScaleBuffers[L + 1]`
  and `routerLogitBias[L + 1]` (`:3526`, `:3528`) while its weights come from
  `L + d`, so every distance above one scored the wrong layer's scales and bias, and
  T1's "two layers ahead costs 0.10 of p" was measured on a mis-scaled probe. At
  distance one, the shipped and measured cell, the bindings agree.
- **The issue point, and why T1 lost.** `begin` runs right after the layer's demand
  fetch is SUBMITTED (`RealForwardRunner.swift:7011-7020`, after the `blobs` block
  at `:6963-7001`, which under the default event synchronisation does not wait for
  the bytes). The demand batch is a `.demand` work item on the shared four-worker
  scheduler and the ring's batch a `.speculative` one (`PreadExpertStreamer.swift:914`,
  `:1327`; `ExpertLoadOperation.swift:163`, `:200-211`); priority orders the queue
  and nothing else, so the speculative batch is popped by the next idle worker and
  its reads start beside the demand read on the same C reader pool. That is the
  probe's `cont` arm, and the ring submits up to `topM` reads at once: at top-8,
  eight reads at QD8 take 4 ms each and overlap the next two layers' demand reads.
  The measured cost, 12 to 13 reading layers per token slowed from 1.05 to 1.59 to
  1.71 ms each, is the `cont` row.
- **The join.** At plan time the exact router's misses are matched against the ring's
  completed slots and each match is a host `memcpy` of a full expert stride into the
  pool slot under the cache lock (`PreadExpertStreamer.swift:762-771`, reached from
  `RealForwardRunner.swift:6752-6761` through `ModelExpertIO.swift:114-127`): 0.12 ms
  per adopted expert inside the submit gap, 0.9 ms per token at top-4. A predicted
  read that is still in flight at plan time is not adopted and its expert is read
  again by the demand batch, beside its own duplicate. The GPU's residency
  classification must equal the plan's misses plus its adopted set or the layer
  fails closed (`RealForwardRunner.swift:6819-6828`); the adopted set folds into the
  fixup's partition (`:6835-6840`) and an adopted-only layer runs under its own kernel
  role (`:7098-7102`).
- **The knobs are not real by the chapter's standard.** `SHRIKE_PREDICTIVE_PREFETCH`,
  `SHRIKE_PREFETCH_TOP_M`, `SHRIKE_PREFETCH_PROBE_DISTANCE` and
  `SHRIKE_PREFETCH_TRACE` are read from the process environment inside the runner
  (`RealForwardRunner.swift:933-948`, `:2055-2069`), threaded through neither binary's
  `RuntimeConfiguration` (`ServerInference.swift:679-680`, `:736-737`;
  `Run.swift:136`, `:173` carry the other knobs) and absent from the banner
  (`RealForwardRunner.swift:348-412`).

## Task 1: the placement gate on the ring in the tree (commit d3da6f9)

Every number MEASURED on the mini unless marked modelled; the plan's Task 1 carries
the step list and the raw data lives at `~/.claude/handoffs/archive/shrike-v15-t1/`.

**What was built.** The ring's batch is issued from the layer's demand batch's
completion (`ExpertLoadOperation.onCompletion`, one hook on the operation's single
terminal transition, run on the finishing thread after the event is published) or at
an all-hit layer's plan time, and capped at an in-flight budget kept by the ring
(`inFlightBudget`; a claimed slot counts until its read terminates, so the budget
holds across the storage thread and the decode thread). `SHRIKE_PREFETCH_PLACEMENT`
(`after`, the default; `beside`, v14 T1's shape) and `SHRIKE_PREFETCH_INFLIGHT`
(1 to 8) join `SHRIKE_PREDICTIVE_PREFETCH`, `SHRIKE_PREFETCH_TOP_M`,
`SHRIKE_PREFETCH_PROBE_DISTANCE` and `SHRIKE_PREFETCH_TRACE` in `RuntimePrefetch`,
threaded into both binaries' two configurations, fail-closed on every bad value, and
printed by the banner (`prefetch=on top_m=8 inflight=1 placement=after distance=1`).
A slot `readyBuffers` hands to a plan is leased until `consume`, so a `begin` on a
storage thread never reclaims a slot the decode thread is still copying (the
review's finding, folded). Five counters join the runner line: `prefetch_deferred`
(batches issued from a completion), `prefetch_overlapped` (demand batches submitted
with a ring read in flight), `prefetch_late` (predictions still in flight when the
exact route asked for them), `prefetch_refused` (predictions dropped for lack of
budget or slot), `prefetch_hook_failed`. The probe's scales and bias now index
`L + d` with its weights.

**The arms (20 lifetimes, every answer identical, the follow-up turns and warm second
prompts unmoved, golden identical at every cell on both boxes).**

| cell (top-8) | card tok/s | the 300 | the 1k | reading layers, ms each (card / 300 / 1k) | late per token | adopted per token |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| prod (ring off) | 14.13 / 13.93 | 14.68 / 14.71 | 14.82 / 14.82 | 1.08 to 1.09 / 1.05 to 1.06 / 1.06 | 0 | 0 |
| after, B = 1 | 14.80 / 14.49 (**+4.4 %**) | 15.61 / 15.61 (**+6.2 %**) | 15.43 / 15.45 (**+4.2 %**) | 1.03 to 1.11 / 0.97 to 0.98 / 0.98 | 0.46 to 0.79 | 8.9 to 10.0 |
| after, B = 2 | 14.33 / 14.38 (+2.3 %) | 14.88 / 14.92 (+1.4 %) | 14.86 / 14.84 (+0.2 %) | 1.15 to 1.16 / 1.12 / 1.11 | 6.8 to 7.9 | 5.6 to 7.2 |
| beside, B = 8 (T1's shape, card) | 13.19 (−6.0 %) | | | 1.63 | 5.35 | 11.1 |
| beside, B = 1 (card) | 13.84 (−1.4 %) | | | 1.31 | 0.30 | 10.3 |

The percentages are the cell's mean against the two bracketing prod runs; the sign
held in both orders on every shape (+4.7 / +4.0, +6.3 / +6.1, +4.1 / +4.3 at B = 1)
against a prod drift of −1.5 / +0.2 / 0.0 %.

**Readings.**

- **The placement rule holds in production.** With one read in flight issued after the
  demand batch, the layers that still read cost production's per-read time or less:
  1.03 to 1.11 ms each on the card, 0.97 to 0.98 on the 300 and the 1k, against
  1.05 to 1.09 with the ring off. The probe's `burst` row on the box.
- **The other placements land exactly where the probe put them.** `beside` at B = 8
  reproduces T1's loss (−6.0 % against T1's −7.1 %) with the reading layers at 1.63
  ms each, the probe's `cont` row; `beside` at B = 1 pays +0.23 per reading layer
  (1.31 against 1.08), the probe's `cont` N = 1 row to the hundredth, and its
  predictions all arrive in time (late 0.30): the trade the design doc named, priced
  at −1.4 %. Two reads after the demand batch (B = 2) do not fit before the next
  plan: late 6.8 to 7.9 per token, the following reads slowed to 1.11 to 1.16 (the
  probe's tail cost), adoption down to 5.6 to 7.2.
- **The lead at distance one is enough for one read.** The lead-time risk (0.8 to
  0.9 ms before the next plan against a 0.8 ms read) resolved in the mechanism's
  favour: late predictions are 0.46 to 0.79 per token of 20 to 21 issued, so 93 to
  95 % of the correct ones are adopted, and 5.0 to 5.2 / 4.9 / 4.6 layers per token
  stop reading. Misses per token fall from 30.5 / 30.2 / 28.1 to 20.6 / 20.4 / 19.2
  and the miss window from 19.6 to 19.9 / 19.6 to 19.8 / 18.4 ms to 13.4 to 14.6 /
  13.4 to 13.5 / 12.5 to 12.6.
- **What the lever still pays, on the box.** The submit gap grows from 2.2 to 2.4 ms
  per token to 3.4 to 3.7: the adoption's host copy at 0.12 ms per adopted expert
  (8.9 to 10.0 per token) plus the begin path. That is Task 2's prize, measured. The
  probe's second router GEMV (2.0 to 2.3 ms per token of GPU in v14's measurement) is
  Task 3's.
- **The probe's index defect is inert on ornith15.** The corrected distance-2 capture
  is byte-identical to T1's: the Qwen-family runner binds one ones buffer as every
  layer's effective scale and one zeros buffer as every layer's logit bias
  (`RealForwardRunner.swift:1473-1533`), so T1's "two layers ahead costs 0.10 of p"
  stands (top-8 coverage 0.367 / 0.342 / 0.337 at precision 0.29 to 0.34). The fix
  matters for gpt-oss and Kimi, whose router bias is per layer.

**The rule.** Real (the sign in both orders on all three shapes, above the drift) and
free (the controls unmoved, golden identical): **the prefetch is on by default** at
one read in flight, placed after the demand batch, `topM` the architecture's top-k;
`SHRIKE_PREDICTIVE_PREFETCH=0` is the A/B. The per-read verdict passed, so Tasks 2 and
3 are built.

**After T1** (2026-09-07; the shipping default changed). Production on the mini
runs the prefetch at the bare launch (the banner: `prefetch=on top_m=8 inflight=1
placement=after distance=1`), and the confirmation arms on the deployed default
(prod, off, prod per shape, `~/.claude/handoffs/archive/shrike-v15-t1/t1-confirm-summary.md`)
put it at **14.44 / 14.57 then 14.81 / 14.76 on the card, 15.59 / 15.21 then 15.44 /
15.55 on the 300, 15.54 / 15.54 then 15.34 / 15.59 on the 1k** (before and after the
review's fold) against 13.92 / 13.93, 14.79 / 14.82, 14.89 / 14.94 with
`SHRIKE_PREDICTIVE_PREFETCH=0` (the off cell −4.1 to −5.8 / −3.9 to −4.4 / −3.4 to
−4.2 % against the bracketing default runs), every answer identical, the follow-ups
unmoved, golden identical at the default and off on both boxes. From the chapter's opening rows (14.1 / 14.8 / 15.0),
production is up 3 to 4 % on the card and the 1k and 3 to 5 % on the 300 with one
lever landed. What remains of the miss window is 12.5 to 15.0 ms per token (13 to 14
reading layers at production's per-read cost), the adoption copy in the submit gap
(3.2 to 4.1 ms per token against 2.0 to 2.5 off), and the probe's GPU time. The
chapter moves to the reserved-slot landing (Task 2), whose prize is now a measured
1.2 to 1.5 ms per token in the submit gap plus the 0.5 to 1.0 late predictions per
token the late join would rescue, then the fused probe (Task 3).

## Task 2: the adoption by GPU blit and the bounded late join (commit 5841078)

Every number MEASURED on the mini unless marked modelled; the plan's Task 2 carries
the step list and the raw data lives at `~/.claude/handoffs/archive/shrike-v15-t2/`.

**The pricing that changed the design.** The plan's original Task 2 landed the
predicted read straight in a pool slot. Step 0 priced it offline with the replay
tool's new speculative-fill hook on T1's captures (the baseline replay reproduces
production's decode misses exactly): one fill per layer cuts decode misses by a third
(6689 to 4309 / 9468 to 6261 / 11398 to 7666), but the wrong fills' evictions cost
about one extra miss per token (useful fills 11.8 / 11.3 / 10.1 per token against
misses saved 10.9 / 10.2 / 9.2), 0.85 ms against the 1.2 to 1.5 ms the copy costs.
Davor ruled for the alternative: the ring stays, and the copy moves to the GPU. A
per-slot buffer swap, the graphics-canonical answer, is out because v9 measured the
per-slot layout a loss (the flip to one slab halved the all-hit gap, 32.6 to 16.7 ms
per token, Metal residency over about 3,400 slot buffers); an index swap inside one
slab is that answer done properly and a later refinement.

**What was built.** The planner reserves an adopted prediction's slot as `loading`
without copying (`PrefetchAdoption.gpuBlit`), and the fixup command that computes
the adopted experts carries a blit from the ring's buffer into the slot at its head,
ahead of its event wait, so the copy runs under the demand read the command was
already waiting for (the Metal-I/O storage path's shape). The slot becomes resident
and the ring's buffer is released when that command has completed, or emptied and
released on any early exit. The blit applies only where the fixup computes the
adopted experts (the speculative modes and `gpu-residency`); elsewhere the host copy
stays and the banner says so. And the late join: a prediction still in flight when
the exact route asks for it is awaited up to a bound (`SHRIKE_PREFETCH_JOIN_US`)
instead of being read again beside its own duplicate. `SHRIKE_PREFETCH_ADOPT`
(`copy` | `blit`) and the join bound join `RuntimePrefetch`, fail-closed, printed by
the banner as `adopt=` and `join_us=`; `prefetch_joined` and `prefetch_blit_experts`
join the runner line.

**The arms (18 lifetimes, every answer identical, the follow-ups unmoved, golden
identical at every cell on both boxes and at the blit under `speculative-validate`).**

| cell | card tok/s | the 300 | the 1k | submit gap ms per token | late / joined per token |
| --- | ---: | ---: | ---: | --- | --- |
| copy (Task 1's default) | 14.65 / 14.61 | 15.44 / 15.44 | 15.41 / 15.44 | 4.19 / 3.64 / 3.38 | 0.5 / 0 |
| blit | 15.08 / 15.10 (**+3.2 %**) | 15.89 / 15.76 (**+2.5 %**) | 15.38 / 15.63 (+0.5 %) | 2.48 / 2.5 to 2.7 / 2.4 to 2.5 | 0.6 to 0.8 / 0 |
| blit, join 400 µs | 15.17 / 14.87 (**+2.7 %**) | 16.03 / 16.06 (**+3.9 %**) | 15.73 / 15.72 (**+1.9 %**) | 2.5 to 2.7 | 0.00 / 0.6 to 0.9 |

**Readings.**

- **The copy leaves the submit gap on every shape**: 1.7 / 1.2 / 1.0 ms per token,
  above the 1.2 to 1.5 modelled on the card; the adopted count is unchanged (10.0 /
  9.4 / 8.6 per token), so the blit changes nothing about what is adopted, only
  where the bytes move.
- **The join catches every late prediction.** Late falls to 0.00 on all eighteen
  lifetimes with the join on; joined 0.6 to 0.9 per token; adoption up 0.4 to 0.8
  per token and misses down 20.4 / 20.7 / 19.3 to 19.9 / 19.9 / 18.7. The wait sits
  inside the plan and is not visible in the submit gap (2.5 to 2.7 against the
  blit's 2.5).
- **The gap block's miss window is no longer the GPU's idle time under the blit.**
  The fixup command now starts with the blit ahead of its event wait, so the block
  measures the gap to the blit (0.6 to 0.7 ms per reading layer, from 1.0), and the
  wait moves inside the command. The wall and tok/s are the verdict: 68.6 to 66.2 /
  64.9 to 62.5 / 65.0 to 63.7 ms per token from copy to blit with the join.
- **The 1k is the thin shape for the blit alone** (+0.5 %, one order at −0.2) and
  the join carries it (+1.9 %, both orders); the 300 gains most (+3.9 %).

**The rule.** Real (both orders on all three shapes, +3.6 / +1.8, +3.8 / +4.0, +2.1
/ +1.8 % against a drift of −0.2 / 0.0 / +0.2) and free (the controls unmoved,
golden identical): **the defaults are the blit and a 400 µs join**;
`SHRIKE_PREFETCH_ADOPT=copy` and `SHRIKE_PREFETCH_JOIN_US=0` are the A/Bs.

**After T2** (2026-09-07; the shipping defaults changed). Production on the mini
runs the blit and the join at the bare launch (the banner: `adopt=blit
join_us=400`), and the confirmation arms on the deployed default (prod, copy, prod
per shape, `~/.claude/handoffs/archive/shrike-v15-t2/t2-confirm-summary.md`) put it
at **15.13 / 15.15 then 15.16 / 14.97 on the card, 16.01 / 16.05 then 16.04 / 16.02 on
the 300, 15.71 / 15.83 then 15.83 / 15.88 on the 1k** (before and after the review's
fold) against 14.59 / 14.76, 15.63 / 15.44, 15.45 / 15.56 with the copy (−2.0 to −3.6
/ −2.5 to −3.7 / −1.8 to −2.0 %), late predictions at zero, every answer identical,
the follow-ups unmoved. From the chapter's opening rows (14.1 / 14.8 / 15.0), production is at
15.1 / 16.0 / 15.7 to 15.8 tok/s, **+7 / +8 / +5 % with two levers landed.** What remains of the miss
window is the reading layers themselves (12.6 to 13.5 per token at production's
per-read cost) and the probe's GPU time (Task 3); the scheduled step zero on the
two-distance queue (the candidate task) asks whether the idle half of every window
can serve a layer further ahead.

## Levers, ranked (modelled from the measured rows)

Every prize is stated per token against the card's 71.3 ms (14.1 tok/s) unless
another shape is named. The predictor's measured accuracy is T1's: a layer's entire
absent set named a layer ahead at p = 0.43 to 0.46 with nonresident precision 0.41 to
0.47 at top-8, p = 0.17 at precision 0.63 to 0.69 at top-4; adopted-only layers 5.3
to 6.1 per token at top-8 and 2.4 to 2.7 at top-4, each collapsing the layer's window
below 0.12 ms ([v14-decode.md](v14-decode.md) "Task 1").

- **(a) The placement gate (Task 1).** The ring's batch is issued when the layer's
  demand batch COMPLETES (a hook on the load operation's single terminal transition,
  `ExpertLoadOperation.swift:120-155`, which every backend passes through) or at an
  all-hit layer's plan time, and capped at an in-flight budget B. Modelled at top-8:
  the slowed reads' cost (6.5 to 8.6 ms per token today) goes to zero if the rule
  holds in production, against a coverage that B = 1 trims (a layer predicting two
  or more misses keeps its best-scored one; B = 2 keeps the pair at +0.05 per
  following demand read, 0.9 ms per token). Net with today's probe and copy still
  paid: **+0.2 to +1.7 ms per token, 0 to +2.5 %**, thin on purpose; the task's first
  verdict is whether the layers that still read return to production's 1.03 to 1.07
  ms each with the ring on, which is what decides whether (b) and (c) are built.
  **Measured (Task 1): +4.4 / +6.2 / +4.2 % at B = 1, the reading layers at or below
  production's per-read cost, the default flipped.**
- **(b) The reserved-slot landing and the late join (Task 2).** The predicted read
  lands in a pool slot the planner reserves speculatively (a victim chosen as for a
  miss, the slot `loading` under a new generation, `resident` when the bytes land),
  so the GPU's classifier sees an adopted expert as a hit with no host copy, and a
  predicted read still in flight at plan time is joined (the fixup's event waits for
  it) instead of duplicated. Removes the copy (1.2 to 1.4 ms per token at top-8) and
  the duplicate reads; costs the pool a wasted fill per wrong prediction, priced
  offline by `tools/expert-pool-replay.py` before it is built. **+1.2 to +1.6 ms,
  about +2 %.** **Measured (Task 2), after the design changed to the GPU blit and the
  join: +2.7 / +3.9 / +1.9 %, the copy's 1.7 / 1.2 / 1.0 ms per token out of the
  submit gap, late predictions to zero; the defaults flipped.**
- **(c) The fused probe (Task 3).** The second router GEMV runs on the same input as
  the authoritative one and is dispatch-bound (53 µs per layer, 2.0 to 2.3 ms per
  token of GPU in the attention tail): one dispatch scoring both routers.
  **+1.5 to +2.0 ms, +2 to +3 %.**
- **(d) The prefill-to-decode boundary (Task 4, the companion).** Every answer's
  first window is its worst (42 misses per token, 11.9 tok/s) because the pool holds
  prefill's experts, not the answer's; v13 Task 5's resident-first sweep was the
  first cut and named itself unfinished. Shortens the slowest stretch of every card
  without raising the plateau; priced by the replay tool first.
- **(e) Deeper lookahead (candidate).** The "drive never idles" ceiling (18.3 tok/s
  on the cold card, 22 with the other gaps) needs reads two or more layers ahead,
  and T1's distance-2 number was measured on the mis-scaled probe. After the fix,
  one trace capture per shape prices it for nothing.

With (a) to (c) at the current predictor: **+3 to +5 ms per token, +4 to +7 %**, the
candidate task's modelled ceiling ([v14-implementation-plan.md](v14-implementation-plan.md)
"Candidate tasks"); every term above is re-priced on the box at each task, never on
this model.

## Method

v14's protocol carried over ([v14-decode.md](v14-decode.md) "Method"): mini-first
verdicts, the M4 Pro iteration signal and the check; one send per server lifetime for
a cold row, `settle_done` before a warm second send, a distinct prompt per arm,
`SHRIKE_RUNNER_STATS=1 SHRIKE_KERNEL_STATS=1` on every measured arm; a decode verdict
row is a whole answer (the three step-zero answers at `max_tokens` 512,
`temperature` 0), the turns and the warm second prompts the controls; the drift band
re-established per task on the same binary. Each task pre-registers its rule, lands
its knob either way (a measured null is a result), and reports the same rows after. A
default flips when the effect is **real** (the sign holds in paired runs in both
orders, above the drift) and **free** (no control row regresses, golden identical);
**no size floor**. A knob is real only when the banner prints it, and the banner
prints the mode in effect. The chapter's rows are attributed by the runner line's
`io_fetch_ms`, the `prefetch_*` counters, the kernel roles and the `Shrike gap` block,
so a win is credited to the window it was predicted to close, never to the wall
alone.

## Numerics policy

Placement, budgets, slot reservation and the fusion of two router GEMVs into one
dispatch change when an expert is read and which command waits on what, never what
any kernel computes on the authoritative path: **golden identical on both boxes and
both profiles**, and a difference is a defect, never a recapture. A predicted fetch
is authoritative only after the exact router selects it; under Task 2 a speculative
slot is `resident` only after its bytes land, and the residency check that already
fails closed on a mismatch between the GPU's view and the plan stays in force. The
fused probe's authoritative half must produce bit-identical top-k ids and weights
(the probe's half feeds only the prediction); a kernel change that cannot show that
follows v12's policy instead.

## Out of scope

Kernel work on attention, GDN and the head GEMV; speculative decoding; the ANE; a
larger pool or a different drive; expert dropping (recorded above, not pursued); the
prompt cache's interior snapshots.

## Risks

- **The placement rule may not survive production's timing.** The probe's gap is a
  fixed 1 ms; production's is the fixup, the submit gap and the next layer's
  attention, and a router readback can arrive before a prefetch read at its tail
  has landed. Task 1 counts every demand batch submitted with a speculative read
  in flight and every predicted expert read twice, and the per-read fetch time is the
  first verdict.
- **The budget trims coverage.** At B = 1 a layer that predicts two misses stays a
  reading layer; the A/B at B = 2 prices the trade on the box.
- **The completion hook runs on a reader thread.** Task 1 adds cross-thread
  concurrency to the load operation; its own step runs a filtered ThreadSanitizer
  pass, and the chapter's close runs the full one.
- **Speculative slots cost the pool.** A wrong prediction under Task 2 evicts a
  resident expert for nothing; precision 0.41 to 0.47 means about half the fills
  are wasted. The replay tool prices the miss-rate cost before the task is built,
  and the box decides.
- **The host-state term is modelled for production.** If production's per-read
  time is 0.93 rather than 0.80, there is a term the ledger has not named; it is
  measured before anything is built on it.
- **The mini is production.** Every arm stops the server on 8081 and relaunches it,
  Turbo on 8080 is never touched, `memory_pressure -Q` before every launch, one
  model process at a time.
