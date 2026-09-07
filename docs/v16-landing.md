# v16: the landing

v15 ([v15-miss-window.md](v15-miss-window.md)) closed with three levers landed and
production on the mini at 15.4 to 15.6 / 16.3 / 16.2 tok/s on the card / the 300 / the 1k
answers. Its closing block named two remaining terms of the miss window: the reading
layers themselves (12.6 to 13.7 per token at production's per-read cost) and one design
refinement with a prize still on the table. This chapter's step zero priced that
refinement and found the prize is not where v15 put it: not the blit's copy, but a fixup
command that runs on about five layers per token for experts the prefetch had already
delivered.

Every number in this document is MEASURED on the mini unless marked modelled; the
step zero's raw data lives at `~/.claude/handoffs/archive/shrike-v15-close/index-swap-step0/`
(the mechanism trace with its line references at v15's `3c2735b`, the replay runs, the
headroom reading, the notes).

## The problem

An expert the prefetch ring delivers is adopted at the layer's plan, after the exact
route is known. In every adoption mode (the host copy, the GPU blit, or an index exchange
inside one slab) the adopted expert is computed by the layer's fixup command, not by the
speculative phase-1 command, because the GPU's residency classifier has always run before
the plan: the plan is gated on the classifier's own readback. So a layer whose only absent
experts were predicted correctly and delivered in time still pays a fixup command: its
submit, its commit-to-kernel latency, the blit and the phase-2 launch.

Task 3's confirmation rows count those layers (a fixup command whose only work is adopted
experts, "adopted-only layers" in the arms tables):

| shape | adopted-only fixup layers per token | fixup submit per layer | fixup commit to kernel (mean over all fixups) |
| --- | ---: | ---: | ---: |
| card | 5.44 | 0.16 ms | 0.50 ms |
| the 300 | 5.1 | 0.15 ms | |
| the 1k | 4.8 | 0.14 ms | |

An adopted-only fixup carries no event wait (its experts are already resident when it is
built), so its commit-to-kernel share is nearer the driver's 0.25 ms than the event-gated
mean. Modelled at 0.5 to 0.65 ms per adopted-only layer: **2.7 to 3.5 ms per token on the
card, +4 to +5 % at the ceiling**, if every adopted expert were a phase-1 hit.

## Step zero (2026-09-07, zero runtime code)

### 1. The mechanism, traced

- The classifier (`moe_classify_expert_residency` and its speculative twin in `moe.metal`)
  reads a per-layer, per-expert residency table `{slot, state, generation}` in shared
  memory; a hit is `state == resident` with a valid slot. The table is written only by
  the host under the streamer's lock (`publishResidencyUnlocked` in
  `PreadExpertStreamer.swift`): `loading` on reservation and `empty` for the evicted expert
  at plan time, `resident` at a demand read's completion on the storage thread, and for a
  blit adoption `resident` when the fixup command has completed.
- The order for layer L in the speculative mode: the tail command (the router, the fused
  probe, the classifier) and the spec command are encoded one layer early and committed at
  the top of L's iteration; only then the host spins on the classifier's tagged readback,
  decodes the exact route and plans. Encode, commit, the GPU classifies, the host wakes on
  the classifier's output, plans, adopts. **An adoption at plan time can never reach that
  layer's classifier.** The plan counts the adopted expert a hit for its own bookkeeping,
  but the fixup partition follows the GPU's view.
- The only write that can make an adopted expert a phase-1 hit lands before classify(L)
  executes: a residency publish at the ring read's COMPLETION on the storage thread. That
  needs the bytes already in a slot of L's slab (the classifier addresses
  `expert_pool + slot * stride` within the layer's slab; the ring's buffers today are
  separate). Whether it lands in time is a race: the prediction is issued at L-1's demand
  completion (v15 Task 1's placement rule) and takes about 0.8 ms; classify(L) executes
  after fixup(L-1)'s event wait and compute and attn(L). Predictions issued at an all-hit
  layer's plan have more lead. **The code does not determine which layers win**; the
  runner already reads both timestamps (the load operation's completion and the tail
  command's kernel start), so an instrument measures it with no design change.
- No ordering primitive stands between the host's table write and the classifier's read
  (no event, no fence); the mechanism already tolerates a stale table through the plan's
  fail-closed compare against the GPU's miss set. The state word is written last and a
  32-bit store is atomic, so the classifier sees `loading` (a miss) or `resident` with its
  slot already in place, never a resident state with a stale slot.
- The slab is per layer: one streamer per layer, one `posix_memalign` slab of 128 slots of
  1,769,472 bytes (the expert stride, page-rounded) wrapped by one Metal buffer; 40 slabs,
  9.06 GB at `--ram-budget 8G`. The ring is global: nine buffers of one stride, tagged by
  layer. The reader takes any destination pointer; demand reads already land in slab
  slots, the ring's in its own buffers. A landing in the slab is a change of pointer at
  issue, plus the slot's lifecycle.

### 2. The variants and their pricing

| variant | RAM | the classifier can see the hit | the wrong prediction's cost | verdict |
| --- | --- | --- | --- | --- |
| (A) spare landing slots in every slab, exchanged with a victim at adoption | 9 x 1.77 MB x 40 = 637 MB | yes | none (the slot is reused) | **out on the mini**: after a 312-token answer on the 1k prompt the server's resident size is 10.7 GB, the box has 71 MB of free pages and 2.34 GB of its 3 GB swap in use (measured) |
| (A') the spare slots carved out of the 128 | none | yes | none, but a pool nine slots smaller | **out**: the replay below prices the carve at about 3 misses per token, the prize's size |
| (B) the landing in the pool's own slot at issue time, resident published at completion | none | yes, when the read completes first | a victim evicted per prediction; the replay says today's miss count | **the candidate** |
| (C) the exchange at plan time only, the blit's copy gone | none | no | none | a likely null: 10 x 1.77 MB per token of GPU copy already hidden under the demand wait |

The replay (v15's tool on v15 Task 1's archived captures, aging-LFU, top-8 one fill per
layer; MODELLED):

| shape | 128 slots, no fills | 128, fills | 119, no fills | 119, fills |
| --- | ---: | ---: | ---: | ---: |
| card | 6689 | 4309 (19.7 per token) | 7524 | 4922 (+2.8 per token over 128 with fills) |
| the 300 | 9468 | 6261 (19.9) | 10963 | 7340 (+3.4) |
| the 1k | 11398 | 7666 (18.9) | 13195 | 8991 (+3.3) |

Production today measures 20.0 / 20.0 / 18.8 misses per token on the same three answers,
so (B)'s modelled miss count is today's: the wrong predictions' evictions cost nothing
against the ring as it stands (v15 Task 2's fork priced them at about one miss per token
against a no-eviction ideal, and chose the blit when the landing's prize was the copy's
1.2 to 1.5 ms alone; the fixup term above was not known then). (B) is T2 Step 0's
"landing as first designed", and the ruling on it is reopened with the new pricing.

### 3. The prize and the race

The prize of (B) is the adopted-only fixups that vanish: for every layer where the
prediction's read completes before classify(L) executes, the adopted expert is a
phase-1 hit and the layer is all-hit, no fixup command at all. Modelled at 2.7 to 3.5 ms
per token on the card times the fraction of adopted experts that win the race, which
nothing in the archived data measures. A prediction that loses the race lands exactly as
today (its slot `loading` until the plan adopts it and the fixup computes it), without
the blit. The late join stays as it is.

### 4. What this chapter measures first

The race, on the box, with an instrument and no design change (Task 1): per adopted
prediction, its read's completion against the target layer's tail command's kernel
start, counted on the runner line. That number, on the three answers, prices (B) and
goes to Davor for the ruling the landing needs.

## The mechanism in the tree

Citations by file and function at `1647692`; the trace's line references at `3c2735b`
are archived with the step zero.

- The classifier and its inputs: `moe_classify_residency_body`, `moe_classify_expert_residency`
  and `_spec` in `sources/Shrike/Metal/MoE/moe.metal`; `ExpertResidencyTable.swift`; the
  table bound by `MoE.swift`'s classify encode.
- The residency's writers: `PreadExpertStreamer.publishResidencyUnlocked`, called from
  `makeExpertCachePlan` (the plan), `markPlanMissesResident` (a demand read's completion,
  the storage thread), `finalizeAdoptedSlots` (the blit's completion, the decode thread).
- The order: `RealForwardRunner.encodeLayerCommands` (the tail with the classifier, one
  layer early), `commitHeldLayerCommands`, the readback spin, `planRoutedExperts` via
  `ModelExpertIO`, the fixup partition in `DecodeExpertPartition.populate`.
- The ring: `ExpertPrefetchRing` (nine buffers, the in-flight budget, the lease, the
  join); the issue point `schedulePrefetchIssue` and `schedulePredictivePrefetch`; the
  reader's destinations in `ModelExpertIO.beginPrefetch` and `expert_io.c`'s `read_one`.
- The slab: the streamer's `.pool` layout (`slotBuffers`, `slotBufferOffsets`,
  `poolSlotStride`), the per-slot arrays, `selectVictimSlots` and `shouldEvictSlot`, the
  pin lease.

## Task 1: the instrument (commit c9b79cb)

Every number MEASURED on the mini; the plan's Task 1 carries the step list and the raw
data lives at `~/.claude/handoffs/archive/shrike-v16-t1/` (the four cuts, `race-cut1`
to `race-cut4`, the fourth the reading).

**What was built.** At a layer's plan, every adopted prediction's read completion (the
load operation's `CLOCK_UPTIME_RAW` stamp, through `ExpertPrefetchRing.completionNanos`)
is set against the GPU window of the command whose last kernel is the classifier (in
production the attention command, into which the tail is folded): `before` its GPU start,
`during` it, binned by the margin before its end at 50 and 150 µs, `after` its end,
`unknown` when the GPU times were never reported. Under the word wake the plan runs
before the command reports its times, so the race rides the runner's deferred GPU
records and is settled by their drain once the command has completed. Seven counters
on the server's runner line and in `decode-rows.py`; the split is a pure function with
its tests. Three cuts preceded the reading, each archived: the first compared against
`kernelStartTime`, the driver's scheduling start, set at the top of the layer's
iteration before the previous layer's demand read completes, so every prediction landed
after by construction; the second counted at the plan and found the GPU times unreported
for 91 % of adoptions; the third settled every race but could not place the during
bucket without the bins.

**The reading** (six lifetimes at production's default, every answer identical, golden
identical at the default on both boxes):

| shape | adopted per token | before | during: under 50 / 50 to 150 / over 150 µs before the end | after | surely won | likely won | at most |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| card | 10.4 | 326 | 613 / 613 / 1705 | 1297 | 0.072 | **0.446** | 0.581 |
| the 300 | 10.1 | 464 | 1094 / 1092 / 1167 | 2553 | 0.073 | **0.256** | 0.427 |
| the 1k | 9.2 | 514 | 996 / 1231 / 2335 | 2355 | 0.069 | **0.383** | 0.549 |

Counts over both lifetimes. "Surely won" is the before share; "likely won" adds the
completions more than 150 µs before the command's end (the command is 0.4 to 0.5 ms wide
and the classifier its last tens of microseconds); "at most" adds the 50 to 150 µs bin,
which the instrument cannot place.

**Readings.**

- **The race is a real race, mostly lost on the 300 and split on the other two.** A
  prediction is issued at the previous layer's demand completion and takes about 0.8 ms;
  the classifier executes after that layer's fixup and the next layer's attention, about
  as long. Only 7 % of adoptions beat the attention command outright.
- **The shape decides**: the 300 loses most (its fixups are shorter, so the classifier
  comes sooner), the card wins most.
- **The joined predictions sit in "after" by construction** (1.2 to 1.9 per token: a join
  completes inside the plan's bounded wait, after the classifier), and they are lost for
  the landing too; "after" less `prefetch_joined` is the count lost outright. An adopted
  prediction without a completion stamp would count as unknown, so the seven buckets
  total the adoptions (the review's finding; none occurred).
- **Nothing about production changed**: adopted, misses, the adopted-only fixup layers
  and tok/s are Task 3's rows, the counters cost nothing visible.

**After T1: the named stop.** The landing's priced prize, MODELLED as the adopted-only
fixup layers (5.4 / 5.1 / 4.8 per token) times the likely-won share times 0.5 to 0.65 ms
per such layer: **1.2 to 1.6 / 0.7 to 0.9 / 0.9 to 1.2 ms per token on the card / the 300
/ the 1k, about +2 / +1.3 / +1.7 %**; at most 1.6 to 2.0 / 1.1 to 1.4 / 1.3 to 1.7 ms if
the middle bin wins too; the surely-won share alone 0.2 ms, negligible. The estimate
assumes one adoption per adopted-only layer and the replay's finding that the landing
keeps today's miss count. The smallest of the chapter's candidates against v15's levers
(+2 to +6 % each); Davor rules on Task 2.

## Levers, ranked (modelled from the measured rows)

- **(a) The instrument (Task 1).** No prize of its own; it prices (b). **Measured: the
  likely-won share 0.45 / 0.26 / 0.38, the prize of (b) +2 / +1.3 / +1.7 % modelled.**
- **(b) The landing (Task 2, built as the merge, variant D below).** Up to 2.7 to 3.5 ms
  per token on the card, +4 to +5 %, times the race's fraction; no RAM; today's miss count
  on the replay; the ring's buffers (15.9 MB) and the blit retired when it lands. **Ruled
  2026-09-07: built, as the merge, with the "at most" share: +2.5 to +3 / +2 / +2.5 %
  modelled. Measured: the mechanism real (the adopted-only fixups down 56 to 69 %), the wall
  flat; the merge kept as a subtraction (Task 2).**
- **(c) The exchange at plan time.** The blit's copy alone; not scheduled (a likely null).
- **(d) The reading layers themselves.** 12.6 to 13.7 per token at production's per-read
  cost, 7.7 to 9.0 ms of GPU idle per token; no lever named in this chapter.

## Task 2's step zero, part 3: the merge (2026-09-07, zero runtime code)

The ruling of 12:40 was to build the landing and let the arms decide. The discussion
that preceded and followed it reframed the task: the fixup is a host-built second
command for the layer, not a fetch; the classifier is the GPU's one-read directory
lookup while the host keeps the tagged side for eviction; the ring beside the cache is
where the chapter's accretion lives (adoption, copy, blit, join, guard), and the landing
as first designed would extend it. Davor's direction: a consolidation series, the ring
merged into the cache's address space as its first step, sequenced so it answers the
landing's question. This part prices the merge; the raw data lives at
`~/.claude/handoffs/archive/shrike-v16-t2-step0/` (`merge-step0-notes.md`,
`boundary-probe/`, `replay/`).

**The kernel-boundary probe (MEASURED, both boxes).** The instrument's likely-won share
assumed the classifier, a later dispatch of the attention command's single compute
encoder, sees a residency write the host makes while that command runs; v14's probe had
shown only that a running kernel never does. A standalone Metal program (one command,
one encoder, dispatch A then dispatch B reading a flag once; the host writes the flag
during A; every write classified against the command's own GPU window and binned by its
margin before the end) ran 300 iterations per cell:

| cell, the mini | seen by the later dispatch |
| --- | --- |
| write before the commit / no write | 300/300, 0/300 |
| write after the commit, before the GPU start | 206/206 |
| A pure ALU, no memory traffic (same line, fresh page-strided line, atomic load, second encoder, memory barrier) | 0/264, 0/274, 0/275, 0/270, 0/284 |
| A streams 1 / 4 / 16 MB after the write | 274/274, 285/285, 281/281 |
| A streams 64 MB before the write, then quiet, by margin under 50 / 50 to 150 / 150 to 300 / 300 to 1000 µs | 20/20, 37/37, 47/47, 315/316 |
| the write inside a 256 MB stream, every bin | 898/898 |
| the write after the command's end | 0/142 |

A later dispatch sees the host's write immediately, the under 50 µs bin included, once
the command has moved 1 MB through memory (4 MB on the M4 Pro); never when it has not.
Production's attention command streams the attention projections and two router GEMVs
on both sides of any write, so the winnable share is the instrument's "at most" row,
0.581 / 0.427 / 0.549, and the prize +2.5 to +3 / +2 / +2.5 % modelled. The mechanism was
not chased.

**Retention (MEASURED offline, deterministic over the recorded routes).** The replay
gained `--fill-mode pool|ring|ring-retain`; v15 T1's captures, 128 slots, aging-lfu,
one top-8 fill per layer per position; decode misses per token:

| mode | card | the 300 | the 1k |
| --- | ---: | ---: | ---: |
| no fills | 30.5 | 30.2 | 28.1 |
| pool: a fill evicts a victim at issue and stays (variant B) | 19.7 | 19.9 | 18.9 |
| ring: a fill beside the pool until its plan, nothing retained | 24.1 | 22.4 | 21.5 |
| ring-retain: a hit fill then takes a victim slot (production today) | 19.9 | 19.8 | 18.7 |
| production, measured (Task 1's cut 4) | 20.0 | 20.0 | 18.8 |

The retain row reproduces production within 0.2, so the model holds. A ring that keeps
nothing costs 2.6 to 4.4 misses per token, more than the landing's prize: the merge must
retain. A victim per right prediction (retain) and a victim per prediction (pool) cost
the same on these routes.

**Retention without a copy (MEASURED: the device limit).** Retained without the blit, a
landed cell must become the layer's cell and the victim's cell the ring's, an index swap,
which needs every cell the classifier can name in one Metal buffer (the kernels address
one base plus a cell times the stride). The mini's `maxBufferLength` is 8.88 GiB (the
M4 Pro's 28.08); the pool at the 8G budget is 128 x 40 x 1,769,472 bytes, 8.438 GiB,
8.453 with the ring's nine cells: it fits with 0.43 GiB of headroom. A larger budget on
the mini would not; the kernels' second base is that day's fallback (a candidate).

**The merge (variant D, the design).** One allocation and one buffer for every cell, 40 x
128 owned by the layers and 9 by the ring, one stride; a layer owns a set of cells, not
a range, and a layer's residency table names a global cell, so the classifier and the
kernels are unchanged. Issue claims a free ring cell, no victim; the read lands there;
the storage thread publishes `resident` at that cell. The layer's plan resolves its
landings: wanted (a GPU hit, or a GPU miss the fixup computes from the cell, `adopted`,
no blit), the cell joins the layer and an evicted victim cell joins the ring; unwanted,
the entry is emptied and the cell stays the ring's; still loading, joined to completion.
Deleted: the copy and blit adoption modes, the transfer, the guard's transfer branch, the
blit counter, the nine standalone buffers. The wrong prediction evicts nothing, the right
one evicts at the plan as today. The miss profile is the retain row's; the prize is the
landing's; three concepts leave. The merge carries no knob (its point is the
alternative's deletion): its A/B is build against build and its rollback the previous
deploy, a recorded deviation from the chapter's rule.

The variants table above gains its fourth row:

| variant | RAM | the classifier can see the hit | the wrong prediction's cost | verdict |
| --- | --- | --- | --- | --- |
| (D) the ring's cells in the pool's address space, the swap at the plan | none (the ring's 15.9 MB move into the one buffer) | yes, when the read completes first | none | **built as Task 2** |

## Task 2: the merge (commit 7652fb6)

Every number MEASURED on the mini; the plan's Task 2 carries the step list, the raw data
lives at `~/.claude/handoffs/archive/shrike-v16-t2/` (the arms, the ledger, the scripts).

**What was built.** One allocation and one Metal buffer for every expert cell
(`ExpertCellArena`, 40 x 128 pool cells and the ring's nine at the page-rounded stride,
8.45 GiB at the 8G budget under the mini's 8.88 GiB limit), allocated at the first layer's
opening; a layer's pool is a set of the arena's cells and its residency table names
global cells, so the classifier and the kernels are unchanged. A prediction's read is a
landing: claimed into a free ring cell with the table entry `loading`, published
`resident` from the storage thread when the bytes land, discarded if the pool's own read
of the expert overtook it. The plan that wants a landing swaps its cell into a pool slot
(the slot's old cell returns to the ring, the entry republished at the same cell under
the slot's next generation, no copy) and reports it `adopted` only when the classifier had missed it, so
the fixup computes it from the cell; an unwanted landing stays resident until the ring
reclaims the cell and drops it from the table. A wanted prediction still in flight is
awaited to completion, since its cell is claimed. Deleted: the copy and blit adoption
modes and `SHRIKE_PREFETCH_ADOPT` (set, the launch fails by name), the transfer, the
guard, the blit counter, the nine standalone buffers. `prefetch_landed_hits` counts the
landings the classifier saw resident. The merge carries no knob: its A/B is build against
build (T1's rows at c9b79cb against the merge's) and its rollback the previous deploy,
the recorded deviation from the chapter's rule. Golden byte-identical on both boxes at the
default and with the prefetch off, and locally under `speculative-validate`,
`gpu-residency` and the per-slot layout (where the prefetch is off and the banner says so).

**The arms** (two lifetimes per shape at the bare launch, T1's cut 4 as the reference,
every answer identical; the prefetch-off control one lifetime):

| shape | arm | tok/s | adopted per token | landed hits per token | adopted-only fixup layers | fixup commands per token | misses per token |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| card | T1 at c9b79cb | 15.56 / 15.05 | 10.5 / 10.3 | | 5.43 / 5.28 | 18.2 | 20.0 / 20.3 |
| card | the merge | 15.34 / 15.35 | 10.6 / 10.6 | 7.38 / 7.23 | 1.68 / 1.76 | 14.4 | 20.0 / 20.0 |
| card | the merge, prefetch off | 14.01 | 0 | 0 | 0 | 18.2 | 30.5 |
| the 300 | T1 at c9b79cb | 16.54 / 16.54 | 10.2 / 10.1 | | 5.14 / 5.12 | 18.7 | 20.0 / 20.0 |
| the 300 | the merge | 16.46 / 16.48 | 10.2 / 10.2 | 6.05 / 5.91 | 2.18 / 2.25 | 15.8 | 20.0 / 20.0 |
| the 300 | the merge, prefetch off | 14.46 | 0 | 0 | 0 | 18.7 | 30.2 |
| the 1k | T1 at c9b79cb | 16.01 / 16.39 | 9.0 / 9.3 | | 4.62 / 4.77 | 17.4 | 19.1 / 18.8 |
| the 1k | the merge | 16.12 / 16.14 | 9.3 / 9.4 | 6.39 / 6.44 | 1.53 / 1.52 | 14.1 | 18.8 / 18.8 |
| the 1k | the merge, prefetch off | 14.59 | 0 | 0 | 0 | 17.4 | 28.1 |

**Readings.**

- **The mechanism works, above the instrument's share.** The classifier saw 70 / 59 /
  69 % of the landed predictions resident (the instrument's "at most" was 58 / 43 / 55 %,
  its bins placed against the command's end rather than the classifier's own read); the
  adopted-only fixup layers fell 66 to 69 % on the card, 56 to 58 % on the 300 and 67 to
  68 % on the 1k, 3.0 to 3.8 host-built commands per token gone; misses per token unchanged to the tenth, as the replay said; the
  prefetch-off control's misses are the replay's no-fills rows to the tenth (30.5 / 30.2 /
  28.1), the model validated a third time.
- **The wall did not move.** tok/s within the repeats' drift on all three shapes, no
  loss. The runner line says why: the read terms are unchanged (the card's io 14.69 to
  14.67 ms per token, fetch 56.7 to 56.6, hidden 32.7 to 33.4 %), the fixup submit per
  token fell 2.55 to 1.91 ms and the fixup's commit-to-kernel 0.51 to 0.34 (fewer
  commands), but the window on each reading layer, from the hit-split command's end to
  the fixup's first kernel, grew from 0.61 / 0.59 / 0.60 to 1.04 / 0.97 / 0.98 ms. The GPU
  reaches every reading layer sooner and waits longer for the same read. The removed
  round trips were paid under the drive: the token's pace is the reading layers' SSD
  chain (fetch 56.6 / 38.7 / 36.0 ms of a 65 / 61 / 62 ms token), and the model that
  priced the landing at 0.5 to 0.65 ms per adopted-only layer assumed those commands sat
  on the critical path. They did not. The fixup's wake after the I/O event also grew,
  1.23 to 2.41 ms per token; its mechanism was not chased.
- **The merge stays, by the rule.** Real (the mechanism on all three shapes, misses at
  today's) and free (golden identical everywhere, tok/s within drift): a subtraction of
  three concepts and a knob at no cost, and the chapter's second measured result on the
  landing: it can be won, and winning it buys nothing on this box because the drive sets
  the pace.

**After T2.** Production on the mini at the merge: 15.34 / 15.35, 16.46 / 16.48, 16.12 /
16.14 tok/s on the card / the 300 / the 1k, flat against 15.56 / 15.05, 16.54 / 16.54,
16.01 / 16.39 at c9b79cb; the same answers; the same misses. The landing's question is
closed both ways: the race can be won on the hardware (the kernel-boundary probe, the
landed hits), and the prize is not where the model put it. What remains is what every
chapter since v13 has circled: the reading layers themselves, 12.6 to 13.6 per token at
production's per-read cost, the drive's chain. The consolidation series (the plan's
Candidates) continues from a tree with one address space and no adoption.

## Method

v15's, unchanged: one measured lever at a time behind a fail-closed knob threaded into
both binaries' two configurations and printed by the banner; host tests RED first; the
four gates per code commit; golden identical on both boxes and both profiles; the
mirrored arms on the mini (`tools/decode-rig.sh`, `tools/decode-rows.py`), real and free
with no size floor, a null a result; a fresh reviewer per task, fixes folded into the
owning commits; the full suite under ThreadSanitizer once at the close. The mini decides.

## Numerics policy

The landing changes which experts are RESIDENT (a victim evicted at issue instead of at
adoption), never which experts are COMPUTED for a token: the exact route decides that,
and a resident prediction is the same bytes the demand read would have brought. Golden
must stay byte-identical at every knob value on both boxes and both profiles; a mismatch
is a defect of the task, never a reason to recapture.

## Out of scope

The reading layers' per-read cost; the token-boundary window; the deeper lookahead; the
expert dropping recorded as the nuclear option in v15; the per-slot cache layout (v9's
measured loss).

## Risks

- The race is lost on most layers: then (b) is a null at Task 1's named stop and nothing
  is built. The instrument is cheap; the answer is the chapter's first result either way.
- A landing slot claimed at issue is a `loading` slot of the target layer; the same
  layer's plan must neither pick it as a victim nor count its expert a hit before the
  bytes land. The existing rules cover both (loading slots are skipped by the victim
  scan and are not `resident`), and a failed read must return the slot to `empty`.
- The storage thread publishing `resident` while the classifier reads the table: the
  state word is written last, atomically, with the slot already in place, as today's
  demand completion does.
- A landed prediction the exact route does not want is an ordinary resident expert the
  aging-LFU evicts in its turn (the replay's "wasted fills"); the replay prices it at
  today's miss count, the arms confirm or refute it.
- The ring's lease and Task 2's guard collapse when nothing is copied; the review looks
  for what they guarded.

## The chapter's close (2026-09-07)

The full suite under ThreadSanitizer on the final tree: 1327 tests in 173 suites, zero
reports, 41 minutes. The whole-branch review by a fresh reviewer verified every fold of
the two per-commit reviews and found no code defect blocking the merge; its findings were
folded into the merge commit by amend and into the docs.

The landing's question, opened from v15's closing pointer, is closed both ways, and the
tree it leaves is smaller than the one it found.

- **The race can be won on the hardware.** The instrument (Task 1) measured it lost on
  most layers by the tail command's window; the kernel-boundary probe showed a later
  dispatch of a running command sees a host write at every margin once the command has
  streamed 1 MB, so the whole "during" bucket was winnable; the merge (Task 2) measured
  the classifier seeing 70 / 59 / 69 % of landed predictions resident, above the
  instrument's "at most" row.
- **Winning it buys nothing on this box.** With 56 to 69 % of the adopted-only fixup
  commands gone, three to four host-built commands per token, tok/s stayed flat within
  the repeats' drift on all three shapes: the round trips had been paid under the drive.
  The token's pace is the reading layers' SSD chain, 56.6 / 38.7 / 36.0 ms of reads in a
  65 / 61 / 62 ms token. Every model that priced the landing (2.7 to 3.5 ms per token at
  the ceiling, then +2.5 to +3 %) assumed those commands sat on the critical path.
- **The merge stays as a subtraction.** One address space for every expert cell, the
  ring's included; a landing published from the storage thread; an index swap at the
  plan with no copy; the copy and blit adoption modes, the transfer, the guard, a knob
  and nine standalone buffers gone; misses per token unchanged to the tenth; golden
  byte-identical on both boxes in every mode. The consolidation series continues from it.
- **The replay is a trustworthy instrument for misses, and only for misses.** It priced
  the pool landing and the retaining ring within 0.2 per token of production three times
  running, the non-retaining ring as a loss, and the no-prefetch control to the tenth; it
  cannot see a millisecond, and the wall's null is the third time the box overruled a
  timing model it had no way to check.

**Production on the mini at the close:** 15.34 / 15.35, 16.46 / 16.48, 16.12 / 16.14 tok/s
on the card / the 300 / the 1k at the bare launch (v15's close: 15.4 to 15.6 / 16.3 / 16.2),
the same answers, the same misses.

**What remains.** The reading layers themselves: 12.6 to 13.6 per token at production's
per-read cost, the one term every chapter since v13 has circled and none has moved. The
consolidation series (the plan's Candidates: the architecture document, the knobs pruned
by measured status, one residency publish path, the runner decomposed) is the next
chapter, on Davor's direction of 2026-09-07; the reading layers come after it on a
smaller tree.
