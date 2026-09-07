# Architecture

How the engine is built and why, written from the tree at `e959d55` (2026-09-07, v16
merged) in the v17 consolidation chapter ([v17-consolidation.md](v17-consolidation.md)),
replacing the v4-era document. Every piece below carries the measurement that keeps it
and the chapter that measured it; every candidate for removal carries its measured status.
Numbers are measured on the Mac mini (M1, 16 GB, the deploy target) unless marked
modelled or counted; line references are at `e959d55` and will move as the chapter
deletes code.

The product claim is bounded memory: run a mixture-of-experts model larger than RAM by
streaming routed experts from SSD into a cache whose size is declared, not discovered.
Production on the mini at v16's close answers the card / the 300-token / the 1k-token
shapes at 15.34 / 16.46 / 16.12 tok/s with 20.0 / 20.0 / 18.8 expert misses per token, the
token's pace set by the reading layers' SSD chain (56.6 / 38.7 / 36.0 ms of reads in a
65 / 61 / 62 ms token).

## The decode path, token by token

`RealForwardRunner.produceToken` (`RealForwardRunner.swift:3153`) runs one token: the
embed, a loop over the layers, the head. The loop pipelines each layer's GPU commands
against the host's work for the previous layer, and the routed experts' reads against
both.

### The layer's held command

`encodeLayerCommands(layer:position:)` (`:3051`) encodes one layer into a
`HeldLayerCommands`: the attention command (`attnCB`: input norm, attention, o_proj), a
tail that folds into it on GDN, MLA and gated-attention layers (one submission and one
boundary per layer; gpt-oss and the plain path keep a separate softmax command and a
separate `tailCB`, because their o_proj must follow the softmax), and either a
shared-expert command or the speculative command. The tail (`encodeDecodeTailStage`,
`:3639`) is the post-attention norm, the layer's router, the next layer's router run on
the same hidden state (the prefetch's probe, fused into the tail since v15), and the
residency classifier.

`commitHeldLayerCommands` (`:3040`) commits attention, softmax, tail, shared and the
speculative command together, before the host waits on the router, so the GPU runs the
shared expert and the whole speculative routed layer while the host waits for the routing.

**Kept by:** the encoder merge and the fused tail, v10 ("no per-layer round trip" made
concrete: one boundary per layer); the fused probe, v15 (the fusion took 1.3 to 2.7 ms
per token off the wall; the separate probe as an arm measured −3.5 / −5.0 / −4.4 % tok/s,
`v15-miss-window.md`).

### The speculative command

`encodeSpeculativeRouted` (`:6799`) encodes, a layer ahead of the host's knowledge of the
route, the shared-expert chain and a pool-addressed phase 1 and phase 2 whose grids size
themselves from the classifier's indirect arguments
(`moe_phase1_gate_up_act_spec_u16load`, `moe_phase2_down_reduce_spec_k8` in
`Metal/MoE/moe.metal`). The classifier (`moe_classify_expert_residency_spec`, `:218`) runs
as the tail's last kernel: for each of the router's top-k experts it reads the layer's
residency table, counts hits and misses, writes the hit positions and their resolved cells
for the speculative phase 1, the miss list for the host, and a tagged host readback word.
On a layer where every expert is resident, the speculative command is the routed command:
nothing more is built for the layer (`encodeDecodeRoutedMoE` stage 21, `:7232`). On a
miss layer it computes the hits and the host builds a fixup for the misses.

The runner encodes layer L+1's held command while layer L's commands run (`:3301`), so
the encode cost is off the critical path.

**Kept by:** v9 (the speculative routed dispatch; the per-slot to pool flip alone halved
the GPU idle gap, 32.6 to 16.7 ms per token on the rig, `v9-speculative-routed-dispatch.md`);
the decode chapter's arithmetic across v9 to v11, rig 111.6 to 38.1 ms per token
(`v10-implementation-plan.md`). The classic path (`barrier`), the CPU-planned partition
(`hit-fixup`), the readback-authoritative mode (`gpu-residency`) and the cross-checking
mode (`speculative-validate`) survive at `e959d55` only as A/B arms of
`SHRIKE_DECODE_EXPERT_EXECUTION`; v17 Task 2 deletes them.

### The word wake

The host does not wait for the tail command to complete. `waitForRouterReadback`
(`:4244`) spins on the classifier's tagged readback word, which lands 0.063 ms after the
router command's GPU end while the driver's completion mark comes 0.16 ms after it on
every layer (v14, measured), with a one-second fallback to the status wait so a failed
command still surfaces.
The readback (`RouterHostReadback.swift`) carries the top-k ids and weights, the hit and
miss positions and the probe's predictions in 32-bit words, each a 16-bit tag over a
16-bit value, read with an acquire load (`shrike_load_acquire_u32`,
`ShrikeKernelsC/include/shrike_atomics.h`).

**Kept by:** v14 lever B, the shipping default: +2.8 to +3.6 % tok/s on the three shapes,
the router wake's 4.0 ms of stat and 2.1 to 2.5 ms of wall per token recovered
(`v14-decode.md`). The status wake (`SHRIKE_ROUTER_WAKE=status`) and the parked host
wait (`SHRIKE_HOST_WAIT=wait`, the spin the T5 default of v10) are its A/B arms; v17
Task 2 deletes both.

### The routed stage: the plan, the swap, the fixup

`encodeDecodeRoutedMoE` (`:6878`, 389 lines, 25 stages) runs once per routed layer after
the wake, with the previous layer's routed command still in flight:

1. The route is read from the readback; the ring's landed predictions for this layer are
   collected (`readyCells`, joining a read still in flight for up to 400 us).
2. The plan (`PreadExpertStreamer.makeExpertCachePlan`, `PreadExpertStreamer.swift:707`)
   decides, under the layer's cache lock, which experts are hits, which landed
   predictions the layer swaps in (a leased ring cell becomes the slot's, the slot's old
   cell goes back to the ring, no bytes move), and which are misses needing a slot and a
   read; victims come from the aging-LFU with chunk protection (v13).
3. The misses are submitted at once to the storage threads (immediate submission, v10
   T5) with a shared-event token the GPU will wait on (event sync, v10 T5).
4. The classifier's hit and miss sets are cross-checked against the plan's, fail-closed
   (`:7037`); a landing the classifier missed but the plan swapped in is reported
   "adopted" and computed by the fixup.
5. The fixup (`buildAndCommitMissFixupCommand`, `:4127`) is built and committed only on a
   miss layer: the event wait, phase 1 for the misses (`moe_phase1_gate_up_act_subset_u16load`),
   the phase-2 reduce (`moe_phase2_down_reduce_k8`) over hits and misses, the residual.
   The host never waits for the read: the GPU does, on the event.
6. A `PendingRoutedCommand` records the layer's routed command and its lease;
   `finishPendingRoutedCommand` (`:6520`) releases the lease at the next layer's wake.

**Kept by:** the miss window chapter, v15 (the placement gate, the 400 us join and the
fused probe: 14.1 / 14.8 / 15.0 to 15.4 to 15.6 / 16.3 / 16.2 tok/s, `v15-miss-window.md`);
the landing, v16 (the swap at the plan: the classifier saw 70 / 59 / 69 % of landed
predictions resident, the adopted-only fixup commands per token down 56 to 69 %, tok/s
flat within the drift, misses unchanged; kept as a subtraction, `v16-landing.md`). The
rdadvise stage (`:7145`) runs only under deferred submission, an arm production never
takes; v17 Task 2 deletes it.

### The deferred GPU records

Under the word wake the host runs ahead of the driver's completion marks, so a command's
GPU timestamps and its error are not yet readable when the host would record them.
`deferredGPURecords` (`:4340`, `drainDeferredGPURecords`) holds the kernel records, the
router wake, the routed command's timings and v16's prefetch race until the driver marks
the commands complete; a failed command throws from the drain with its name, since the
immediate error checks ran before the mark existed. The drain runs after each wake and
once, waiting, at the token's end.

**Kept by:** the word wake needs it; the stats it settles are the runner line every
chapter's rows are read from (`tools/decode-rows.py`).

### The head

`produceToken`'s tail (`:3371`): the final norm and the lm_head GEMV, or the fused greedy
head that writes the argmax token directly when the sampler is greedy (`useFusedGreedyHead`);
the sampler's tiled top-k kernel otherwise (`Kernels/Sampling`).

### The dense layers

Layers below `numLeadingDenseLayers` (Kimi's layer 0) take an inline path in the loop
(`:3232`): norm, attention, the dense SwiGLU, three commits and a status wait, no
classifier and no routed stage. v17 Task 4 lifts it into `produceDenseLayer`.

## Residency

### The table

One table per routed layer, owned by the layer's `PreadExpertStreamer`
(`ExpertResidencyTable.swift:4`): `ExpertResidencyEntry { slot: UInt32, state: UInt32,
generation: UInt64 }`, 16 bytes per expert, a shared Metal buffer allocated at the layer's
first touch (`PreadExpertStreamer.swift:438`) and bound to the classifier at buffer index
1 (`MoE.encodeResidencyClassification`, `Kernels/MoE/MoE.swift:506`). States: `empty`,
`loading`, `resident`. Under the pool layout `slot` is the expert's global arena cell.
The GPU hit test is `state == resident && slot != notResidentSlot` (`moe.metal:141`).

### The writers

Fourteen at `e959d55`, every one a 16-byte struct store: writers 2 to 11 through
`publishResidencyUnlocked` (`PreadExpertStreamer.swift:1573`) and writers 12 to 14
directly, both into `writeResidencyEntryUnlocked` (`:1581`) under the streamer's
`cacheLock`; the init's fill (writer 1) runs before any reader exists:

| # | writer | thread | trigger and what is written |
| ---: | --- | --- | --- |
| 1 | `init` (`:447`) | the streamers queue | the layer's first touch: every entry `empty` |
| 2 to 5 | `loadExpertUnlocked` (`:621`, `:631`, `:642`, `:652`) | the caller's | the single-expert round-robin load: evict, reserve, complete, fail; no production caller (tests only) |
| 6 | `makeExpertCachePlan`, the victim (`:789`) | the planner's | the evicted expert `empty` at the slot's next generation |
| 7 | `makeExpertCachePlan`, the swap (`:795`, the store at `:806`) | the planner's | a leased landing is resident: the slot takes its cell, republished `resident` at the slot's bumped generation |
| 8 | `makeExpertCachePlan`, the reservation (`:816`) | the planner's | a miss: `loading` at the slot's cell |
| 9 | `markPlanMissesResident` (`:1501`) | the storage thread | the demand read completed: `resident`, the generation re-validated first |
| 10 | `markStagedMetalPlanResident` (`:1527`) | the runner's | the Metal backend's staged blit completed (never in production) |
| 11 | `resetLoadingMissesUnlocked` (`:1549`) | the failed read's, the abandoned prefill plan's, the failed staged plan's | `loading` back to `empty` |
| 12 | `claimLanding` (`:1364`) | the issuing thread (decode or storage) | a ring cell claimed: `loading` at the ring cell, the landing's own generation |
| 13 | `completeLanding` (`:1382`) | the storage thread | the speculative read landed: `resident` at the ring cell |
| 14 | `dropLanding` / `failLanding` (`:1399`) | the storage thread, the issuing thread, the ring's reclaim | `empty`, only if the pool does not own the expert |

The planners are the decode runner (`RealForwardRunner.swift:6949`), prefill's union and
tile planners (`:5400`, `:6087`) and `Model.fetchRoutedExperts(layer:experts:)`
(`ModelExpertIO.swift:284`, through the streamer's `loadExpertsCached`, `:661`). Only the
decode planner holds a ring lease, so only it swaps; every other planner reads a landed
expert into the pool and the landing is dropped at the ring's reclaim (v16's review fold).

### The invariants

- **The store is the publish.** There is no memcpy, no encode-time copy and no publish
  step: the host's struct store into the shared buffer is what the GPU reads at its next
  dispatch. v16's kernel-boundary probe measured when a later dispatch of a running
  command sees a host write: always, once the command has streamed 1 MB, at every margin;
  never without memory traffic. Production's attention command streams tens of MB on
  both sides of any write.
- **A torn entry reads as a miss, with one exception that ordering covers.** The 16-byte
  store is not atomic and Swift does not order its fields. For thirteen of the fourteen
  writers no transition changes `slot` while `state` stays `resident` (a swap keeps the
  cell, an eviction goes through `empty`), and the hit test needs both fields, so a read
  that mixes the old and the new entry is a miss. The exception is the reservation
  (writer 8, `:819`) over an unleased resident landing: an expert the ring delivered to
  cell C and no decode plan leased is reserved at a pool slot in one store of `{S',
  loading}` over `{C, resident}`, and a reader mixing the new slot with the old state would
  see a hit at bytes not yet landed. That transition is taken by prefill's planners, the
  loader's fetch, and the decode planner for a landing that completed past the join bound;
  in every case the layer's classifier for this token has already run and the next reader
  of the table is the next token's, so the tear is unreachable by ordering, not by the
  argument. A miss is always safe: the plan fails closed on the classifier's miss set.
  v17 Task 3 makes the pair one 64-bit release store, which removes the exception.
- **The generation is host bookkeeping.** It guards a stale completion against publishing
  over a newer occupant (`:1506`). The classifier writes `resolved_generations`
  (`Kernels/MoE/MoE.swift:546`) and nothing in `sources/` reads it; one test does
  (`GPUExpertResidencyTests.swift:207`). Two generation spaces land in the field, the
  slot's (`slotGeneration`) and the landing's (`landingGeneration`, `:1371`), because a
  ring cell belongs to no slot until the swap. v17 Task 3 unifies them per cell and drops
  the field from the entry.
- **Lock order.** The ring's lock, then a layer's cache lock, never the reverse
  (`ExpertPrefetchRing.swift:35`). `beginPrefetch` takes the cache lock once per claim, so
  a batch's claims are not atomic as a group.

### The arena and the ring

`ExpertCellArena` (`Infrastructure/Streaming/ExpertCellArena.swift`) is one allocation
and one Metal buffer for every expert cell the classifier can name: the layers' slots
(the slot count times the routed layers) plus the ring's nine, at the page-rounded expert
stride. On the mini the 8G budget snaps to 128 slots, 8.45 GiB with the ring, under the
device's 8.88 GiB `maxBufferLength` with 0.43 to spare (v16); a larger budget there needs
the kernels given a second base (v16's candidate). A cell changes owner at a swap without
a byte moving; that is the whole reason for one address space.

`ExpertPrefetchRing` (`Runtime/Inference/ExpertPrefetchRing.swift`) owns nine cells and
at most one read in flight across all layers (v15 step zero: a read still in flight shares
the drive with the next demand read). At layer L's wake the probe's top-k for layer L+1 is
issued after the demand submission (the placement gate, v15); a prediction lands in a
ring cell and is published `resident` from the storage thread, so layer L+1's classifier
can hit it; the plan swaps a wanted landing in and returns the freed cell; the reclaim
drops an unwanted one when the ring needs the cell. The probe distance above one and the
in-flight budget above one are closed levers. The distance's record is the recall curve
of 2026-08-31 (paired fresh-server runs on identical deterministic streams, `ornith15`):
miss recall 0.439 / 0.322 / 0.256 / 0.223 at k = 1 to 4 on the rig stream (nonresident
precision 0.124 to 0.036) and 0.510 (k = 1) to 0.305 (k = 3) on a diverse prompt
(precision 0.358 to 0.127); the hidden state drifts 20 to 27 % per layer of lookahead, so
any distance that buys the drive useful lead time (k of 3 or more) catches at most a
third of misses while issuing 8 to 28 wasted fetches per useful one. The in-flight
budget's record is v15's step zero (a read still in flight shares the drive with the next
demand read) and its two-distance queue, measured null.

**Kept by:** the prefetch-off control in v16's arms: 14.0 / 14.5 / 14.6 tok/s at 30.5 /
30.2 / 28.1 misses per token against production's 15.3 / 16.5 / 16.1 at 20.0 / 20.0 /
18.8; `tools/expert-pool-replay.py --fill-mode ring-retain` reproduces production's misses
within 0.2 per token and the no-fills control to the tenth, so the control survives in the
instrument. The off switch, the placement, probe, distance, in-flight and join knobs are
v17 Task 2's deletions.

## The demand path

A miss is read by `PreadExpertStreamer.beginExpertCachePlan` (`:890`) on
`ExpertIOScheduler` (`ExpertLoadOperation.swift:199`): four `.userInitiated` worker
queues, demand batches queued ahead of speculative ones. The read itself is the bounded
pread reader in C (`ShrikeKernelsC/expert_io.c`, `shrike_expert_io.h`): fixed reader
threads (four: v4's knee, and v13 measured that a second four buys nothing at a tile's
three to four misses), `F_NOCACHE` reads, at most two published batches (v13's winner). A batch carries an `ExpertIOCompletionToken`
(`ExpertIOEventCoordinator.swift`): one shared Metal timeline for the model, a status word
per batch (loading, complete, failed) the GPU's fixup waits on; out-of-order completions
are held until every preceding value is terminal, since advancing the timeline past an
unfinished batch would release its GPU wait early. The completion publishes `resident`
on the storage thread (writer 9) and wakes the ring's deferred issue.

Production's per-read cost on the mini is 0.73 to 0.80 ms at p50 inside a 1.0 to 1.1 ms
reading layer (v15's ledger), 12.6 to 13.6 reading layers per token: the term no chapter
since v13 has moved and the next chapter's object.

The legacy cached-pread path (`ParallelExpertReader.swift`, `SHRIKE_BOUNDED_IO=0`) and
the Metal IO backend (`MetalExpertReader.swift`, `MetalExpertStagingPool.swift`,
`SHRIKE_EXPERT_IO_BACKEND=metal`; its A/B of 2026-09-01: rig wait 43.29 ms sd 9.2 %
against pread's 38.68 sd 2.2 %, and the server died mid-prefill,
`v10-implementation-plan.md`) are v17 Task 2's deletions.

## Prefill and the turn

Prefill (`executePrefillChunk`, `RealForwardRunner.swift:2683`) runs the prompt in chunks:
per layer the attention on the matrix path (`Metal/Prefill/attention_matrix.metal`, the
causal-matrix tile `g2k256d`, matrix min rows 16), the router over the chunk, then the
routed experts as tiles over the union of the chunk's experts, fetched two tiles deep
through the same streamer with a sweep order that starts from what is resident
(`SHRIKE_PREFILL_SWEEP=resident`, v13). The record is
[v12-prefill-matrix-kernels.md](v12-prefill-matrix-kernels.md): the mini's 12k-token
prefill 725 to 68.4 s (5.57 ms per token), the 3.7k 110.7 to 20.6 s. The ANE prefill
attention ([ane-prefill.md](ane-prefill.md), `SHRIKE_PREFILL_ANE=on`) is an opt-in
experiment with its own record, the one switch v17 keeps.

The turn ([v13-the-turn.md](v13-the-turn.md)) is the server's prompt cache
(`ServerPromptCache.swift`, [v6-dialect-normalized-cache.md](v6-dialect-normalized-cache.md):
the cached KV equals what the dialect's template renders, so a match is a byte comparison)
plus the settle and the rewrite after an answer, the resident sweep and fetch depth 2:
the 305-token first turn 5.47 to 3.54 s on the mini, the 1,085 8.08 to 6.47, the 2,125
11.24 to 10.30, warm first turns after an answer 3.45 / 6.39 / 10.29.

The v12 kernel variants and tuning knobs (the tile families, the block router, the
per-expert routed GEMM, the tiled attention path, the losing sweep orders, the serial GDN
scan) survive at `e959d55` as A/B arms of twenty-one knobs; v17 Task 2 deletes them at
their defaults. MTP (`StreamingMTP.swift`, `encodeRoutedMoEVerifyPair`) was retired at
v12's P17 (rig acceptance 20.6 %) and goes with them.

## The serving layer

`ShrikeServer` serves an OpenAI-compatible API on loopback over `ServerModelSession`
(`ServerInference.swift`): the tokenizer and its dialect (ChatML, Harmony), the prompt
cache, the structured decoders for thinking and tool calls
([v7-reasoning-effort.md](v7-reasoning-effort.md),
[v8-emission-form-tool-calls.md](v8-emission-form-tool-calls.md)), the runner counters on
the runner line. `ShrikeCLI` drives one generation for the golden baseline; `ShrikeMac`
spawns `ShrikeDecodeService` out of process over a socket. Multi-model serving is recorded
in [multi-model-serving.md](multi-model-serving.md); the channel-faithful turn design in
[channel-faithful-turns.md](channel-faithful-turns.md).

## The four invariants of v4, re-verified at `e959d55`

1. **RAM budget is an input, not an outcome.** Still true. `--ram-budget` (`ServerArguments.swift:312`,
   `RuntimeConfiguration.parseBudgetBytes`) defaults to 8 GiB
   (`defaultExpertCacheBudgetBytes`, `RuntimeConfiguration.swift:342`); the slot count is
   the ladder value (8 to 128) nearest budget over stride times routed layers
   (`expertCacheSlots`, `:375`); the arena is sized from that count plus the ring's nine.
   The load path's comment at `ServerInference.swift:700` still describes a 1 GiB default
   with 16 slots at 4-bit; that is stale and v17 Task 4 removes it with the function's
   decomposition. The mini runs `--ram-budget 8G`, 128 slots, about 9.06 GB allocated.
2. **Streaming that genuinely uses the disk.** Still true, with the figures replaced.
   The v4 rates (2.83 GB/s at prefill, 2.6 at decode against a 3.92 ceiling) were rig-era
   and described the old reader. Production's term is per read: 0.73 to 0.80 ms at p50 on
   the mini, 12.6 to 13.6 reading layers per token, 56.6 / 38.7 / 36.0 ms of reads in a
   65 / 61 / 62 ms token (v15, v16). Splitting one read across several preads is null on
   the mini's internal drive (2026-09-04, `v10-implementation-plan.md`, the P3 follow-on);
   the mini needs many experts in flight to saturate. The I/O threads remain the one CPU load safe beside decode: they block in
   the kernel rather than burn ALU (v4's measurement, unchanged in kind).
3. **No per-layer CPU round trip in decode.** Still true, and now the mechanism above:
   the classifier on the GPU, the speculative command a layer ahead, the host woken by a
   word, the fixup gated on the storage event. The v4 text's per-layer sync stall at an
   8 % miss rate is gone; what remains per reading layer is the read itself.
4. **C99 for hot loops, Swift for structure.** Still true; the target grew.
   `ShrikeKernelsC` is 577 lines of C in two files (`expert_io.c` 442, `int4_affine_gemv.c`
   135) plus 173 of headers, counted at `e959d55`, against v4's 439. The int4 GEMV's 2.9x
   and the reader's threads are v4's measurements; the acquire load the word wake uses is
   the third thing the target owns.

## The knobs

Sixty-six `SHRIKE_*` names read under `sources/` at `e959d55`, counted from the tree.
"Users" are references outside the reading code (`tools/`, `docs/`, `tests/`, `CLAUDE.md`,
`README.md`); "class" is diagnostic (output only), mode (selects a code path), config (a
size, a path, a count) or gate (a retired name refused); "fails open" marks a knob whose
unrecognised value silently takes the default instead of failing the launch (28 of the
66; the enum knobs read through `environmentValue` throw). The disposition column is v17
Task 2's, from the rule that a knob is useful only if something uses it; the citation is
the record that measured the default.

| knob | read at | class | default and values | users | measured record | disposition |
| --- | --- | --- | --- | --- | --- | --- |
| `SHRIKE_DECODE_EXPERT_EXECUTION` | `RuntimeConfiguration.swift:42` | mode | `speculative`; `hit-fixup`, `barrier`, `gpu-residency`, `speculative-validate` | v9, v12, v14 docs; tests | v9 the speculative dispatch, v10 T5 the default | delete; one path |
| `SHRIKE_SPEC_PHASE1` | `:77` | mode | `all-hit`; `hits` | v14 docs; tests | v14 lever A, a null | delete |
| `SHRIKE_ROUTER_WAKE` | `:97` | mode | `word`; `status` | v14 docs; tests | v14 lever B, +2.8 to +3.6 % | delete |
| `SHRIKE_HOST_WAIT` | `RealForwardRunner.swift:483` | mode, fails open | spin; `wait` | CLAUDE.md, v9, v14 docs | v10 T5 | delete |
| `SHRIKE_EXPERT_IO_SYNC` | `RuntimeConfiguration.swift:59` | mode | `event`; `host` | v9, v10, v14 docs; tests | v10 T5 | delete |
| `SHRIKE_EXPERT_IO_SUBMISSION` | `:238` | mode | `immediate`; `deferred` | tests | v10 T5 | delete |
| `SHRIKE_RDADVISE_POLICY` | `ServerInference.swift:740` | mode, fails open | default; `off`, `bounded`, `adaptive` | none | dead under immediate submission | delete with the stage |
| `SHRIKE_EXPERT_CACHE_LAYOUT` | `PreadExpertStreamer.swift:228` | mode | `pool`; `per-slot` | CLAUDE.md, v9, v12 docs; tests | v9's loss, v10 T5 | delete; the arena |
| `SHRIKE_EXPERT_IO_BACKEND` | `:212` | mode | `pread`; `metal` | v10 docs; tests | the 2026-09-01 A/B, a loss | delete |
| `SHRIKE_BOUNDED_IO` | `:539` | mode, fails open | on; `0` | v12 docs | the bounded reader, v12 | delete the legacy path |
| `SHRIKE_PARALLEL_IO` | `:875` | mode, fails open | parallel; `0` (legacy path only) | none | | delete |
| `SHRIKE_EXPERT_IO_THREADS` | `:250` | config | 4, `1...16` | v13 docs; tests | four the knee | constant |
| `SHRIKE_EXPERT_IO_BATCH_DEPTH` | `:251` | config | 2, `1...2` | v13 docs; tests | v13's two batches | constant |
| `SHRIKE_EXPERT_CACHE_POLICY` | `:389` | mode | `aging-lfu`; `lfu`, `lru` | v13 docs | v13 | delete the losers |
| `SHRIKE_EXPERT_CACHE_PROTECT` | `:196` | mode | `chunk`; `off` | v13, v14 docs; the replay; tests | v13's chunk protection | delete; always on |
| `SHRIKE_NO_PIN` | `ResidentBuffer.swift:65` | mode | pins; presence skips `mlock` | none | | delete |
| `SHRIKE_PREDICTIVE_PREFETCH` | `RuntimeConfiguration.swift:168` | mode | on; `0` | architecture, v13 to v16 docs; the rig; tests | the off control, v16 | delete; the replay keeps the control |
| `SHRIKE_PREFETCH_TOP_M` | `:175` (the `1...topK` bound in the runner, `RealForwardRunner.swift:959`) | config | top-k, `1...topK` | v14, v15 docs; the rig; tests | v14 top-4 vs top-8 | constant |
| `SHRIKE_PREFETCH_INFLIGHT` | `:176` | config | 1, `1...8` | v15 docs; tests | v15's queue, a null | constant |
| `SHRIKE_PREFETCH_PROBE_DISTANCE` | `:188` | config | 1, `1...8` | architecture, v14, v15 docs; the rig; tests | the recall curve, closed | constant; the reclaim window goes |
| `SHRIKE_PREFETCH_JOIN_US` | `:196`, `:202` | config and gate | 400, `1...2000`; `0` refused | v15 docs; tests | v15's 400 us join | constant |
| `SHRIKE_PREFETCH_PLACEMENT` | `:179` | mode | `after`; `beside` | v15 docs; tests | v15's placement gate | delete |
| `SHRIKE_PREFETCH_PROBE` | `:205` | mode | `fused`; `separate` | v15 docs; tests | v15's fused probe | delete |
| `SHRIKE_PREFETCH_ADOPT` | `:191` | gate | any value refused | v15, v16 docs; tests | removed by v16's merge | delete; the tripwire |
| `SHRIKE_PREFETCH_TRACE` | `:190` | diagnostic | a JSONL path | the rig, the replay, the coverage tool | | stays |
| `SHRIKE_ATTN_MATRIX_TILE` | `PrefillAttention.swift:91` | mode, fails open | `g2k256d`; six others | v12 docs | v12's tile arms | delete the losers |
| `SHRIKE_MPP_TILE_N`, `_TILE_K`, `_DEQUANT_BUFFERS` | `MPPPrefillInt4QMM.swift:44` to `:46` | mode, fails open | the `n32k256b1` variant | v12 docs | v12's tensor-core arms | delete the losers |
| `SHRIKE_MPP_WEIGHT_LOADS` | `:52` | mode, fails open | `vector`; `byte` | v12 docs | v12 | delete |
| `SHRIKE_PREFILL_ATTENTION` | `RealForwardRunner.swift:624` | mode, fails open | `matrix`; `tiled` | v12 docs | v12's matrix path | delete the tiled path |
| `SHRIKE_PREFILL_ROUTER` | `PrefillRouter.swift:55` | mode, fails open | `tiled`; `block` | v12 docs | v12 | delete the block router |
| `SHRIKE_PREFILL_ROUTER_TOKENS` | `:61` | config, fails open | 12, `4...24` | v12 docs | v12 | constant |
| `SHRIKE_PREFILL_ROUTED_GEMM` | `RealForwardRunner.swift:829` | mode, fails open | grouped; `per-expert` | v12 docs | v12's grouped dispatch | delete the per-expert path |
| `SHRIKE_PREFILL_ROUTE_OVERLAP` | `:831` | mode, fails open | on; `off` | v12 docs | v12 | delete the knob only |
| `SHRIKE_PREFILL_POOL_RESIDENCY` | `:677` | mode, fails open | on; `none` | v12 docs | v12 | delete the knob only |
| `SHRIKE_PREFILL_TAIL_TILE` | `:647` | mode, fails open | `32`; `off` | v12 docs | v12 | delete the knob only |
| `SHRIKE_PREFILL_TILE_BATCH` | `:688` | config, fails open | 1, `1...16` | v12 docs; the replay | v12 | constant |
| `SHRIKE_PREFILL_TILE_DEPTH` | `:708` | config, fails open | 2, `1...8` | v12 docs; the replay | v12 | constant |
| `SHRIKE_PREFILL_FETCH_DEPTH` | `:726` | config, fails open | 2, `1...2` | v13 docs; the replay | v13's fetch depth 2 | constant |
| `SHRIKE_PREFILL_MATRIX_MIN_ROWS` | `:672` | config, fails open | 16, `3...32` | v13 docs | v13's min rows 16 | constant |
| `SHRIKE_PREFILL_SWEEP` | `:618` | mode | `resident`; `alternate`, `fixed`, `carry`, `recency` | v12 to v14 docs; the replay; the turn rig | v13's resident sweep | delete the losers |
| `SHRIKE_PREFILL_SWEEP_TAIL` | `:642` | config | `min(96, upper)`, `8...expertCount` (recency only) | v13 docs; the replay | v13 | delete with `recency` |
| `SHRIKE_GDN_PREFILL_SCAN` | `:823` | mode, fails open | chunked; `serial` | v12 docs | v12's chunked scan | delete the serial path |
| `SHRIKE_ATTN_FULL_CHUNKS` | `Kernels/Attention/Attention.swift:102` | config, fails open | 16, `1...max` | v10 docs | v10 | constant |
| `SHRIKE_SAMPLER_PATH` | `Sampler.swift:107` | mode | `tiled`; `generic` | tests | the tiled top-k | delete the generic path |
| `SHRIKE_PREFILL_ANE` | `ANEPrefillAttention.swift:21` | mode | `off`; `on` | README, ane-prefill.md, the probes; tests | an open candidate | stays |
| `SHRIKE_MTP_VERIFY` | `StreamingMTP.swift:238` | mode | `pair`; `tile` | v12 docs; tests | MTP retired at v12 P17 | delete with MTP |
| `SHRIKE_MTP_EXPERT_SLOTS` | `:30` | config, fails open | 8, of the ladder | tests | | delete with MTP |
| `SHRIKE_THINKING_MODE` | `Tokenizer.swift:50` | mode, fails open | off; on, `adaptive` | tests | product | stays |
| `SHRIKE_REASONING_EFFORT` | `:68` (the server validates at `ServerArguments.swift:161`) | config | medium; `low`, `high` | v7 docs; tests | product | stays |
| `SHRIKE_REASONING_RETENTION` | `:86` (the server validates at `ServerArguments.swift:167`) | mode | `as-generated`; `stripped` | v6.1 docs | product | stays |
| `SHRIKE_STRIP_CLI_PROMPT` | `CLIStrip.swift:34` | mode, fails open | off; on | multi-model-serving.md; tests | product | stays |
| `SHRIKE_STRIP_TAGS` | `:44` | config | `system-reminder` | tests | product | stays |
| `SHRIKE_CONCISE_MODE` | `ServerInference.swift:912` | mode, fails open | off; on | none | product | stays |
| `SHRIKE_TOKENIZER_DIR` | `Tokenizer.swift:184` | config | unset | none | product | stays |
| `SHRIKE_MODEL` | `AppModelInstallDescriptor.swift:119` | config, fails open | Ornith 1.5 8-bit; four names | none | the app's selector | stays |
| `SHRIKE_EXPERT_CACHE_SLOTS` | `ServerInference.swift:697` | config, fails open | unset; the ladder | the flag's help | duplicates `--expert-cache-slots` | delete |
| `SHRIKE_RUNNER_STATS` | `RealForwardRunner.swift:2169` | diagnostic | off | CLAUDE.md; the rigs, the deploy script, the parsers | | stays |
| `SHRIKE_KERNEL_STATS` | `:2167` | diagnostic | off | CLAUDE.md; the rigs, the deploy script, the ledger | | stays |
| `SHRIKE_ROUTE_TRACE` | `:2211` | diagnostic | a path | the rig, the replay, the coverage tool | | stays |
| `SHRIKE_LAYER_TRACE` | `:2171` | diagnostic | off | none | | delete |
| `SHRIKE_GPU_CAPTURE_DIR` | `:2176` | diagnostic | unset | v10 docs | | delete |
| `SHRIKE_CACHE_DIAG` | `ServerPromptCache.swift:489` | diagnostic | off | v7 docs | | delete |
| `SHRIKE_GEN_DIAG` | `ServerInference.swift:72` | diagnostic | off | v7, v8 docs | | delete |
| `SHRIKE_PHASES` | `RealForwardRunner.swift:2812`, `Run.swift:206` | diagnostic | off | v12, v13 docs | | delete |

The surviving set, thirteen names: `SHRIKE_THINKING_MODE`, `SHRIKE_REASONING_EFFORT`,
`SHRIKE_REASONING_RETENTION`, `SHRIKE_STRIP_CLI_PROMPT`, `SHRIKE_STRIP_TAGS`,
`SHRIKE_CONCISE_MODE`, `SHRIKE_TOKENIZER_DIR`, `SHRIKE_MODEL`, `SHRIKE_PREFILL_ANE`,
`SHRIKE_RUNNER_STATS`, `SHRIKE_KERNEL_STATS`, `SHRIKE_ROUTE_TRACE`, `SHRIKE_PREFETCH_TRACE`.
Checked against what sets a knob outside the code: `tools/mini-deploy.sh`'s launch line
sets `SHRIKE_RUNNER_STATS` and `SHRIKE_KERNEL_STATS`; `tools/decode-rig.sh`'s every launch
sets those two and `SHRIKE_ROUTE_TRACE`; the rig's `PREFETCH_TRACE=1` sets `SHRIKE_PREFETCH_TRACE`; its
`SERVER_ENV` examples name `SHRIKE_PREDICTIVE_PREFETCH`, `SHRIKE_PREFETCH_TOP_M` and
`SHRIKE_PREFETCH_PROBE_DISTANCE`, which Task 2 rewrites when those go. Four names in
`docs/` have no reader at all (`SHRIKE_ATTN_DECODE_LOOP`, `SHRIKE_EXPERT_IDLE_REFILL`,
`SHRIKE_MPP_ROW_TILE`, `SHRIKE_MTP_REJECT_KEEP`): history. `SHRIKE_IO_MAX_THREADS` and
`SHRIKE_IO_MAX_BATCHES` are C compile-time bounds in `shrike_expert_io.h`, not knobs.

After Task 2 one tripwire replaces the refused names: any `SHRIKE_*` variable in the
environment outside the surviving set fails the launch by name.

## The long functions

The swiftlint baseline's eighteen entries at `e959d55`, the map v17 Task 4 decomposes
from (stages from the design doc's step zero):

| function | lines | shape and the first seam |
| --- | ---: | --- |
| `RealForwardRunner.encodeDecodeRoutedMoE` (`:6878`) | 389 | 25 stages under 7 mode knobs; the classification, the hit-split encode, the I/O acquisition |
| `RealForwardRunner.produceToken` (`:3153`) | 229 | the embed, the layer loop (7 sub-stages), the head; the dense-layer body first |
| `RealForwardRunner.executePrefillChunk` (`:2683`) | 225 | 12 stages; the validation, the token buffer, the embed-or-blit, the ANE probe, the close-out |
| `RealForwardRunner.encodeRoutedMoEVerifyPair` (`:5340`) | 214 | MTP's width-2 verify; deleted in Task 2 |
| `RealForwardRunner.encodeFullAttentionPrefill` (`:4997`) | 204 | 9 stages, no `self` mutation; the RoPE epilogue, the causal dispatch |
| `ServerInference.generate` (`:1112`) | 284 | 21 stages; a 60-line counter snapshot the app's client already has as an init |
| `ServerInference.load` (`:642`) | 189 | 13 stages; the slot derivation, the runner, the cache domain, the cache |
| `ServerArguments.parse` (`:140`) | 241 | a flag switch and 40 lines of validation |
| `RemoteStreamingRepacker.runPrepared` (`:197`) | 231 | the install pipeline: resume, copy ranges, finalize |
| `RawCompletion.runRawCompletion` (`:89`) | 180 | prefill then the decode loop |
| `ShrikeDecodeService` `Entry.main` (`:12`) | 177 | a command switch; one handler per case |
| `ShrikeCLI` `Run.run` (`:50`) | 176 | the driver: the prompt, the runtime, the footer |
| `ShrikeCLI` `Args.parse` (`:139`) | 170 | a flag switch and validation |
| `PreadExpertStreamer.init` (`:375`) | 167 | resource acquisition; the layout and reader branches go in Task 2 |
| `Model.load` (`:607`) | 149 | a verify-then-map pipeline already sectioned by comments |
| `RealInferenceClient.run` (`:288`) | 126 | one `do` with three `catch` arms |
| `ShrikeBench.runMoE` (`:192`), `runGDN` (`:407`) | 179, 175 | deleted with the target in Task 2 |

## The instruments

- `tools/golden-baseline.sh --check`: the only check that runs real inference; greedy,
  byte-identical, two profiles per machine tag under `baselines/`.
- `tools/decode-rig.sh` with `tools/decode-rows.py`: the three request shapes on the
  mini, a fresh server per shape, every token's arrival streamed, one row per request.
- `tools/turn-rig.sh` with `tools/turn-summary.py`: the turn's shapes (a pair, a suffix,
  the multi-turn chain).
- `tools/expert-pool-replay.py`: a route trace replayed against the pool's policy and
  the ring's fills; trustworthy for misses, blind to milliseconds (v16).
- `tools/prefetch-coverage.py`: a prediction priced offline against a trace.
- `tools/prefill-ledger.py`, `tools/parse-kernel-stats.py`, `tools/parse-runner-stats.py`:
  the stats lines read into ledgers.
- `tools/mini-deploy.sh`: the release binaries and their bundles to the mini, optionally
  a restart at the production launch.

## History

Each chapter's design doc sits beside its plan under `docs/`; the plan's checkboxes are
the status of record.

- v5 ([v5-second-architecture-objective.md](v5-second-architecture-objective.md)): the
  second architecture.
- v6, v6.1 ([v6-dialect-normalized-cache.md](v6-dialect-normalized-cache.md),
  [v6.1-reasoning-retention.md](v6.1-reasoning-retention.md)): the prompt cache as a byte
  comparison; reasoning retention.
- v7, v8 ([v7-reasoning-effort.md](v7-reasoning-effort.md),
  [v8-emission-form-tool-calls.md](v8-emission-form-tool-calls.md)): reasoning effort;
  the emission form and tool calls.
- v9 to v11 ([v9-speculative-routed-dispatch.md](v9-speculative-routed-dispatch.md),
  [v10-decode-kernel-mergers.md](v10-decode-kernel-mergers.md),
  [v11-kv-attention-inner-loop.md](v11-kv-attention-inner-loop.md)): the decode chapter,
  rig 111.6 to 38.1 ms per token, the depth tax +8.0 to +2.19 ms per 1k of context.
- v12 ([v12-prefill-matrix-kernels.md](v12-prefill-matrix-kernels.md)): prefill, 725 to
  68.4 s at 12k tokens; MTP audited and retired.
- v13 ([v13-the-turn.md](v13-the-turn.md)): the turn, the 305-token first turn 5.47 to
  3.54 s.
- v14 ([v14-decode.md](v14-decode.md)): decode pass II, the word wake +2.8 to +3.6 %.
- v15 ([v15-miss-window.md](v15-miss-window.md)): the miss window, three levers,
  15.4 to 15.6 / 16.3 / 16.2 tok/s.
- v16 ([v16-landing.md](v16-landing.md)): the landing, the ring merged into the pool's
  address space, the wall flat.
- v17 ([v17-consolidation.md](v17-consolidation.md)): this document, the knobs, one
  residency path, the runner decomposed.
