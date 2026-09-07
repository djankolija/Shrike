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
- **(b) The landing in the pool's own slot (Task 2).** Up to 2.7 to 3.5 ms per token on
  the card, +4 to +5 %, times the race's fraction; no RAM; today's miss count on the
  replay; the ring's buffers (15.9 MB) and the blit retired when it lands. Built only on
  Davor's ruling with the race's number.
- **(c) The exchange at plan time.** The blit's copy alone; not scheduled (a likely null).
- **(d) The reading layers themselves.** 12.6 to 13.7 per token at production's per-read
  cost, 7.7 to 9.0 ms of GPU idle per token; no lever named in this chapter.

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
