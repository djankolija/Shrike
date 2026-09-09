# Architecture

How the engine is built and why, written from the tree at v17's close (`0dbeba0`,
2026-09-08) to replace the v4-era document, and brought to the tree at v18's close
(2026-09-09). Every piece below carries the measurement that keeps it and the chapter
that measured it. Numbers are measured on the Mac mini (M1, 16 GB, the deploy target)
unless marked modelled or counted; line references are at v18's closing tree. Each
chapter's own record, what its tasks built, deleted and counted, is its design doc
beside its plan: [v17-consolidation.md](v17-consolidation.md) for the consolidation,
[v18-quiet-host.md](v18-quiet-host.md) for the decode path's command structure below;
git history holds the paths the chapters removed.

The product claim is bounded memory: run a mixture-of-experts model larger than RAM by
streaming routed experts from SSD into a cache whose size is declared, not discovered.
Production on the mini at v16's close answered the card / the 300-token / the 1k-token
shapes at 15.34 / 16.46 / 16.12 tok/s with 20.0 / 20.0 / 18.8 expert misses per token, the
token's pace set by the reading layers' SSD chain (56.6 / 38.7 / 36.0 ms of reads in a
65 / 61 / 62 ms token); v18 took the three shapes to about 16.2 / 16.9 / 16.8 tok/s with
the misses per token within 0.2 of those and the reads' term untouched
(`v18-quiet-host.md`).

## The decode path, token by token

`RealForwardRunner.produceToken` (`RealForwardRunner.swift:2052`) runs one token: the
embed when the caller passes a token, or the held layer 0 when the previous token's
boundary command already embedded it (the continued pass); then a loop that calls
`produceDenseLayer` (`:2233`) or `produceRoutedLayer` (`:2296`) per layer; then the
token boundary (`emitBoundary`, `:2195`) or the synchronous head (`emitHead`, `:2368`).
The loop pipelines each layer's GPU command against the host's work for the previous
layer, and the routed experts' reads against both.

### The layer's held command

`encodeLayerCommands(layer:position:)` (`:1956`) encodes one layer into a
`HeldLayerCommands` (`:1923`): the attention command (`attnCB`: input norm, attention,
o_proj), a tail that folds into it on GDN, MLA and gated-attention layers (gpt-oss and
the plain path keep a separate softmax command and a separate `tailCB`, because their
o_proj must follow the softmax), and the speculative routed work, the shared-expert
chain at its head, encoded behind the tail in `attnCB` on every folded layer, which is
every layer of the served model, so the layer is one command and one submission; the
split-tail path gives it a separate `specCB` behind its tail. `routerCB` names the
command whose word publishes the route, `routedCB` the one carrying the routed work. The
tail (`encodeDecodeTailStage`, `:2630`) is the residual fold, the post-attention norm,
the layer's router, the next layer's router run on the same hidden state (the prefetch's
probe, fused into the tail since v15), and the residency classifier, the tail's last
kernel.

`commitHeldLayerCommands` (`:1946`) commits whichever of attention, softmax, tail and
separate speculative command exist, before the host waits on the router, so the GPU runs
the shared expert and the whole speculative routed layer while the host waits for the
routing.

**Kept by:** the encoder merge and the fused tail, v10 ("no per-layer round trip" made
concrete: one boundary per layer); the fused probe, v15 (the fusion took 1.3 to 2.7 ms
per token off the wall; the separate probe as an arm measured −3.5 / −5.0 / −4.4 % tok/s,
`v15-miss-window.md`); one command per layer, v18 Task 3 (the forty tail-to-speculative
command boundaries gone, about 1.4 ms of gaps, the merged commands up by about 1.65 with
the drain now an encoder boundary inside the command; the wall +0.4 to +0.8 % on the 300
and the 1k, about 0.3 to 0.5 ms per token, a third of the modelled 0.9; kept as simpler
and non-negative, `v18-quiet-host.md`).

### The speculative routed work

`encodeSpeculativeRouted` (`:5260`) encodes, a layer ahead of the host's knowledge of the
route, five dispatches on one serial compute encoder: the shared-expert chain (the gate
and up INT4 GEMVs as one grid, `dequant_int4_shared_gate_up_gemv_simd` at
`Metal/Quant/dequant_int4.metal:206` through `FusedGateUpGEMV`
(`Kernels/Fusions/FusedGateUpGEMV.swift`), dispatched by `SharedExpertInt4.encodeGateUp`
(`Kernels/MoE/SharedExpertInt4.swift:91`); the scalar gate; the fused silu-mul-down with
the sigmoid gate), then a pool-addressed phase 1 and phase 2 whose grids size themselves
from the classifier's two indirect arguments (`moe_phase1_gate_up_act_spec_u16load` at
`Metal/MoE/moe.metal:923`, `moe_phase2_down_reduce_spec_k8` at `:975`, whose epilogue
finishes each element with the residual add, `moe_phase2_finish` at `:865`, shared with
the fixup's phase 2 and the affine twin). One serial encoder, never a `.concurrent` one:
a `.concurrent` encoder followed by an indirect dispatch on the same command segfaults
the AGX driver (v9's trap, named at `:5275`). The classifier
(`moe_classify_expert_residency_spec`, `moe.metal:187`) runs as the tail's last kernel:
for each of the router's top-k experts it reads the layer's residency table, counts hits
and misses, writes the hit positions and the resolved cells (a miss's position carries
the `0xffffffff` sentinel, `:141`), the miss list for the host, a tagged host readback
word, and the two grids (`MoESpecDispatchArgs`, `:109`): phase 1's full grid on every
layer, since the kernel's rows skip a sentinel position (`:957`), so the hits are
computed inside the layer's command on a miss layer too; phase 2's full grid only when
every expert is resident, zero otherwise. On an all-hit layer the layer's command is the
routed command: nothing more is built (`handOffDecodeSpeculativeAllHit`, `:5584`). On a
miss layer the host builds a fixup for the misses alone.

`produceRoutedLayer` encodes layer L+1's held command while layer L's command runs
(`:2312`), so the encode cost is off the critical path.

**Kept by:** v9 (the speculative routed dispatch; the per-slot to pool flip alone halved
the GPU idle gap, 32.6 to 16.7 ms per token on the rig, `v9-speculative-routed-dispatch.md`);
the decode chapter's arithmetic across v9 to v11, rig 111.6 to 38.1 ms per token
(`v10-implementation-plan.md`). It is the only decode execution path: the classic
`barrier`, the CPU-planned `hit-fixup`, the readback-authoritative `gpu-residency` and the
cross-checking `speculative-validate` arms went with their knob in v17. The hits on
every layer, v18 Task 1: a measured null on the wall kept as a simplification (the
separate hit command and its 1.8 ms submit gap sat inside the read's flight, so removing
them moved the window's start and not the read's landing; one command buffer and one
host path fewer per miss layer, golden identical, `v18-quiet-host.md`). The one
encoder, v18 T6.0 (the speculative work's and the fixup's dispatches on one encoder
each): 3.3 ms per token of GPU role time over the 147 encoder boundaries that went,
about 22 µs an encoder boundary around these indirect dispatches, the wall +1.6 to
+2.8 % on the 300 and the 1k, about 1.0 to 1.7 ms per token. The gate and up grid
(T6.1) and the residual in phase 2's epilogue (T6.3): class 1, bit-identical against
the kernels they replaced, the wall flat on same-box A/Bs, kept as smaller; the
dispatch wall between two small independent GEMVs measured at most about 3 µs and the
one after an indirect kernel before a tiny one about 5.5 (T6.3's 0.22 ms per token of
role time, every A/B lifetime separated), which is why the four remaining small-kernel
merges were not built.

### The word wake

The host does not wait for the layer's command to complete. `waitForRouterReadback`
(`:3197`) spins on the classifier's tagged readback word, with a one-second fallback to
the status wait so a failed command still surfaces. Under v14's separate tail command
the word landed 0.063 ms after that command's GPU end while the driver's completion mark
came 0.16 ms after it, on every layer (v14, measured); with the routed work behind the
classifier in the same command the word lands mid-command, tens of microseconds after
the kernel writes it (`MidCommandVisibilityTests`: 42 to 45 µs after the command's GPU
start, 29 to 31 ms before its end, three runs), so the runner line's
`path_router_wake_ms`, which counts the wake past the command's GPU end (`:3257`), reads
zero by design. The token boundary takes the same wake on the sampler's token word
(`awaitBoundaryToken`, `:1487`), with the same one-second fallback, counted as
`boundary_wake_fallbacks`.
The readback (`RouterHostReadback.swift`) carries the top-k ids and weights, the hit and
miss positions and the probe's predictions in 32-bit words, each a 16-bit tag over a
16-bit value, read with an acquire load (`shrike_load_acquire_u32`,
`ShrikeKernelsC/include/shrike_atomics.h:7`).

**Kept by:** v14 lever B, the shipping default: +2.8 to +3.6 % tok/s on the three shapes,
the router wake's 4.0 ms of stat and 2.1 to 2.5 ms of wall per token recovered
(`v14-decode.md`). It is the only wake the routed path takes: the status wake and the
parked host wait went with their knobs in v17. `waitForRouterCompletion` (`:3181`), the
spinning status wait of v10 T5, survives as this wake's one-second fallback and as the
wait on a layer whose classifier did not run.

### The routed stage: the plan, the swap, the fixup

`encodeDecodeRoutedMoE` (`:5347`) runs once per routed layer after the wake, with the
previous layer's routed command still in flight. Its body is nine private stage methods
over one `DecodeRoutedLayerContext` (`:5312`), which carries the layer's locals from stage
to stage, in the order the work runs:

1. `readDecodeRouterReadback` (`:5396`) reads the route from the readback (or from the raw
   index buffer when no classifier ran) and records the route trace.
2. `joinDecodePrefetch` (`:5436`) collects the ring's landed predictions for this layer
   (`readyCells`, joining a read still in flight for up to 400 us) and takes the
   classifier's miss set as it stood before this plan.
3. `planDecodeRoutedExperts` (`:5448`) runs the plan
   (`PreadExpertStreamer.makeExpertCachePlan`, `PreadExpertStreamer.swift:382`), which
   decides, under the layer's cache lock, which experts are hits, which landed predictions
   the layer swaps in (a leased ring cell becomes the slot's, the slot's old cell goes back
   to the ring, no bytes move), and which are misses needing a slot and a read; victims
   come from the aging-LFU with chunk protection (v13). The stage then consumes the ring's
   leases, handing the freed cells back, and writes the prefetch trace.
4. `pinAndSubmitDecodeRoutedExperts` (`:5481`) pins the plan's slots and submits the misses
   at once to the storage threads (immediate submission, v10 T5) with a shared-event token
   the GPU will wait on (event sync, v10 T5).
5. `partitionDecodeRoutedExperts` (`:5498`) splits the top-k into hit and miss positions
   and cross-checks the classifier's miss set against the plan's, fail-closed (`:5519`); a
   landing the classifier missed but the plan swapped in is reported "adopted" and computed
   by the fixup.
6. `acquireDecodeRoutedIO` (`:5529`) takes the miss batch's buffers without waiting on the
   read (the GPU waits, on the event) and issues the next layer's prediction.
7. `handOffDecodeSpeculativeAllHit` (`:5584`) returns on an all-hit layer: the layer's
   command already is the routed command, and it becomes the `PendingRoutedCommand`
   (`:5039`) with no role of its own, since the layer's record covers it.
8. `buildDecodeFixup` (`:5616`) builds and commits the fixup on a miss layer
   (`buildAndCommitMissFixupCommand`, `:3097`): the event wait, then one compute encoder
   (the wait sits between the command's start and the encoder, never inside one) holding
   phase 1 for the misses alone (`moe_phase1_gate_up_act_subset_u16load`,
   `moe.metal:808`; the hits' activations already stand in `moeActs` from the layer's
   command) and the phase-2 reduce (`moe_phase2_down_reduce_k8`, `:873`) over hits and
   misses, the residual add in its epilogue. A layer that ran no classifier takes the full
   phase 1 instead.
9. `handOffDecodeFixup` (`:5641`) records that command and its lease as the
   `PendingRoutedCommand`; `finishPendingRoutedCommand` (`:5085`) releases the lease at
   the next layer's wake.

**Kept by:** the miss window chapter, v15 (the placement gate, the 400 us join and the
fused probe: 14.1 / 14.8 / 15.0 to 15.4 to 15.6 / 16.3 / 16.2 tok/s, `v15-miss-window.md`);
the landing, v16 (the swap at the plan: the classifier saw 70 / 59 / 69 % of landed
predictions resident, the adopted-only fixup commands per token down 56 to 69 %, tok/s
flat within the drift, misses unchanged; kept as a subtraction, `v16-landing.md`). The
stages are v17 Task 4's shape, not a behaviour change: the sixty-odd stage calls a token
makes are invisible against its 60 ms (`v17-consolidation.md`).

### The deferred GPU records

Under the word wake the host runs ahead of the driver's completion marks, so a command's
GPU timestamps and its error are not yet readable when the host would record them.
`deferredGPURecords` (`:308`, `drainDeferredGPURecords` at `:3290`) holds the kernel records, the
router wake, the routed command's timings and v16's prefetch race until the driver marks
the commands complete; a failed command throws from the drain with its name, since the
immediate error checks ran before the mark existed. The drain runs after each wake and
once, waiting, at the token's end. The boundary command is not among the records: the
next pass waits on it directly at its own end (`finishPreviousBoundary`, `:1529`), when
it has long completed, and records it as `head_logits`.

**Kept by:** the word wake needs it; the stats it settles are the runner line every
chapter's rows are read from (`tools/decode-rows.py`).

### The token boundary

The pass ends with one command on one compute encoder (`emitBoundary`, `:2195`): the
final norm, the lm_head GEMV, the caller's sampler (`Runtime/Generation/Sampler.swift`:
the tiled softmax and the top-k-64 kernel; the generic kernels stay the path for greedy,
for k above 64 and for top-k disabled) and the next token's embed, which reads the
sampled id from a token word the sampler kernel writes (the two embed kernels'
`tokenBuffer:` overloads, `Kernels/Quant/EmbedLookupInt4.swift:89` and
`AffineQuant.swift:141`, bind the word at the kernel's constant argument; no Metal
change). The host writes a sentinel (`0xFFFFFFFF`) into the word before the commit and
does not wait on the command: it advances the cursor and encodes layer 0 of the next pass
into the held slot (`holdLayerZero`, `:1536`) while the head runs (a model whose layer 0
is dense holds nothing; that layer encodes itself in the pass). The loop
(`Runtime/Generation/RawCompletion.swift:279`) then spins on the word
(`awaitBoundaryToken`, `:1487`, the router wake's one-second fallback behind it), checks
the stop token, detokenises, runs the stop-string matcher, the max-tokens check and the
progress callback, and only then calls the continued pass, which commits the held layer
0; on a stop nothing is committed, so the stop check stays on time and no pass runs
late. `BoundaryLogitProducer` (`Runtime/Generation/LogitProducer.swift:16`) is the
two-step shape the runner conforms to. The loop chooses the path once per generation and
keeps the synchronous head (`emitHead`, `:2368`: the final norm and the lm_head GEMV on
one encoder, or the fused greedy head that writes the argmax token directly,
`useFusedGreedyHead`, `:305`) for a repetition penalty other than 1.0, for the fused
greedy head and for the first token after prefill.

**Kept by:** v18 Task 4 (the three boundary gaps, 0.83 to 0.90 ms per token, to one of
0.25 to 0.27, the sample and embed roles folded into `head_logits`; +0.9 % on the 300
and, on a same-box interleaved A/B on the 1k, four lifetimes each, Task 4 winning every
pair at +1.4 % on the clean lifetimes, about 0.8 ms per token; the misses per token and
the answers identical, `v18-quiet-host.md`); the one encoder, T6.0b (`head_logits` down
0.06 to 0.09 ms per token over eight boundaries, about 10 µs a boundary between the
sampler's small kernels, the wall inside the drift, kept as simpler and non-negative).

### The dense layers

Layers below `numLeadingDenseLayers` (Kimi's layer 0) take `produceDenseLayer` (`:2233`):
norm, attention, the dense SwiGLU, three commits and a status wait, no classifier and no
routed stage.

## Residency

### The table

One table per routed layer, owned by the layer's `PreadExpertStreamer`
(`ExpertResidencyTable.swift:4`): `ExpertResidencyEntry { slot: UInt32, state: UInt32 }`,
eight bytes per expert, one word, a shared Metal buffer allocated at the layer's first
touch (`PreadExpertStreamer.swift:280`, from `Model.ensureLayerOpened`) and bound to the
classifier at buffer index 1 (`MoE.encodeResidencyClassification`,
`Kernels/MoE/MoE.swift:526`). States: `empty`, `loading`, `resident`. `slot` is the
expert's global arena cell. The classifier takes the table as `device const ulong*` and
unpacks each entry from one 64-bit load (`moe.metal:132`); the hit test is
`state == resident && slot != notResidentSlot` (`:134`).

### The writers

One function writes the table: `PreadExpertStreamer.publish(expert:cell:state:)`
(`PreadExpertStreamer.swift:928`), which packs the state above the slot and stores the word
once with `shrike_store_release_u64` (`ShrikeKernelsC/include/shrike_atomics.h:12`). No
caller writes the buffer. It has eight call sites, all but the init's fill under the
streamer's `cacheLock`; the init's fill runs before any reader exists:

| # | call site | thread | trigger and what is written |
| ---: | --- | --- | --- |
| 1 | `init` (`:332`) | the streamers queue | the layer's first touch: every entry `empty` |
| 2 | `makeExpertCachePlan`, the victim (`:465`) | the planner's | the evicted expert `empty`, after the cell's generation is bumped |
| 3 | `makeExpertCachePlan`, the reservation (`:485`) | the planner's | a miss: `loading` at the slot's cell |
| 4 | `markPlanMissesResident` (`:892`) | the storage thread | the demand read completed: `resident`, every miss's cell generation re-validated first |
| 5 | `resetLoadingMissesUnlocked` (`:914`) | the failed read's, the abandoned plan's | `loading` back to `empty` |
| 6 | `claimLanding` (`:766`) | the issuing thread (decode or storage) | a ring cell claimed: `loading` at the ring cell, whose generation the claim bumps |
| 7 | `completeLanding` (`:785`) | the storage thread | the speculative read landed: `resident` at the ring cell |
| 8 | `dropLanding` / `failLanding` (`:801`) | the storage thread, the issuing thread, the ring's reclaim | `empty`, only if the pool does not own the expert |

The swap writes nothing (`:467` to `:481`): a landing already stands `{cell, resident}` in
the table, so the slot takes the landing's cell and only the host's bookkeeping moves.

The planners are the decode runner (`planDecodeRoutedExperts`,
`RealForwardRunner.swift:5448`), prefill's tile fetches
(`PrefillGroupedRoutedMoE.swift:579`, `:631`) with the tile scheduler's lookahead
(`RealForwardRunner.swift:271`), and `Model.fetchRoutedExperts(layer:experts:)`
(`ModelExpertIO.swift:230`, through the streamer's `loadExpertsCached`, `:336`). Only the
decode planner passes a ring lease, so only it swaps; every other planner reads a landed
expert into the pool and the landing is dropped at the ring's reclaim (v16's review fold).

### The invariants

- **The store is the publish.** There is no memcpy, no encode-time copy and no publish
  step: the host's store into the shared buffer is what the GPU reads at its next
  dispatch. v16's kernel-boundary probe measured when a later dispatch of a running
  command sees a host write: always, once the command has streamed 1 MB, at every margin;
  never without memory traffic. Production's attention command streams tens of MB on
  both sides of any write.
- **The pair is never torn, on either side, by construction.** The host writes the slot
  and the state as one aligned 64-bit release store; the classifier reads them back as one
  `ulong` and unpacks the halves in registers. Neither side can observe half an entry, so
  the reservation over an unleased resident landing (v16's one exception, closed by ordering
  rather than by construction) is no longer an exception at all. A miss stays safe in any
  case: the plan fails closed on the classifier's miss set (`RealForwardRunner.swift:5519`).
- **The generation is host bookkeeping, one word per arena cell.** It lives on
  `ExpertCellArena` (`ExpertCellArena.swift:60`, bumped at `:66`), not in the table, and
  the classifier neither reads nor writes it. Every value comes from one atomic clock, so
  no two cells' bumps can coincide and a stale plan's recorded generation can never equal a
  different cell's by chance. It is read and written under the owning layer's cache lock,
  and it guards a stale completion against publishing over a newer occupant
  (`PreadExpertStreamer.swift:882`). The swap moves the landing's cell under the slot and
  its generation with it.
- **Lock order.** The ring's lock, then a layer's cache lock, never the reverse
  (`ExpertPrefetchRing.swift:38`). `beginPrefetch` takes the cache lock once per claim, so
  a batch's claims are not atomic as a group.

### The arena and the ring

`ExpertCellArena` (`Infrastructure/Streaming/ExpertCellArena.swift`) is one allocation
and one Metal buffer for every expert cell the classifier can name: the layers' slots
(the slot count times the routed layers) plus the ring's nine, at the page-rounded expert
stride. On the mini the 8G budget snaps to 128 slots, 8.45 GiB with the ring, under the
device's 8.88 GiB `maxBufferLength` with 0.43 to spare (v16); a larger budget there needs
the kernels given a second base (v16's candidate). A cell changes owner at a swap without
a byte moving; that is the whole reason for one address space.

`ExpertPrefetchRing` (`Runtime/Inference/ExpertPrefetchRing.swift:39`) owns the top-k plus
one cells, nine on this model (`RealForwardRunner.swift:421`), and at most one read in
flight across all layers (`inFlightBudget`, `ExpertPrefetchRing.swift:46`; v15 step zero: a
read still in flight shares the drive with the next demand read). At layer L's wake the
probe's top-k for layer L+1 is issued after the demand submission, one layer ahead (the
placement gate and distance one, v15's constants); a prediction lands in a
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
instrument, not in the binary. The ring is always built: the off switch and the placement,
probe, distance, in-flight and join knobs went in v17, each at its measured winner.

## The demand path

A miss is read by `PreadExpertStreamer.beginExpertCachePlan` (`:544`) on
`ExpertIOScheduler` (`ExpertLoadOperation.swift:175`): four `.userInitiated` worker
queues, demand batches queued ahead of speculative ones. The only reader is the bounded
pread reader in C (`ShrikeKernelsC/expert_io.c`, `shrike_expert_io.h`) behind
`ParallelExpertReader` (`ParallelExpertReader.swift`): fixed reader threads (four: v4's
knee, and v13 measured that a second four buys nothing at a tile's three to four misses),
`F_NOCACHE` reads, at most two published batches (v13's winner), both constants in
`BoundedReaderConfiguration` (`PreadExpertStreamer.swift:146`). A batch carries an
`ExpertIOCompletionToken` (`ExpertIOEventCoordinator.swift`): one shared Metal timeline
for the model, a status word per batch (loading, complete, failed) the GPU's fixup waits
on; out-of-order completions are held until every preceding value is terminal, since
advancing the timeline past an unfinished batch would release its GPU wait early. The
completion publishes `resident` on the storage thread (`markPlanMissesResident`) and wakes
the ring's deferred issue. The gate on the GPU's side is a function constant: the runner
builds its `MoE` with `eventGatedIO: true` (`RealForwardRunner.swift:648`), and the
parameter's `false` default (`Kernels/MoE/MoE.swift:90`) exists only so the kernel tests
can build a `MoE` without a coordinator; the un-gated arm of `moe_io_ready` is the tests'
path, not a mode.

Production's per-read cost on the mini is 0.73 to 0.80 ms at p50 inside a 1.0 to 1.1 ms
reading layer (v15's ledger), 12.6 to 13.6 reading layers per token: the term no chapter
since v13 has moved and the next chapter's object.

Two other readers stood beside it until v17 and are gone with their knobs: the legacy
cached-pread path, and the Metal IO backend (its A/B of 2026-09-01: rig wait 43.29 ms sd
9.2 % against pread's 38.68 sd 2.2 %, and the server died mid-prefill,
`v10-implementation-plan.md`).

## Prefill and the turn

Prefill (`executePrefillChunk`, `RealForwardRunner.swift:1618`) runs the prompt in chunks:
per layer the attention on the matrix path (`Metal/Prefill/attention_matrix.metal`, the
causal-matrix tile `g2k256d` at `PrefillAttention.swift:88`, matrix min rows 16 at
`RealForwardRunner.swift:282`), the router over the chunk, then the routed experts as tiles
over the union of the chunk's experts, fetched two tiles deep through the same streamer
(`prefillRoutedTileSchedulerConfig`, `:268`) with a sweep order that starts from what is
resident (v13). The record is
[v12-prefill-matrix-kernels.md](v12-prefill-matrix-kernels.md): the mini's 12k-token
prefill 725 to 68.4 s (5.57 ms per token), the 3.7k 110.7 to 20.6 s. The ANE prefill
attention ([ane-prefill.md](ane-prefill.md), `SHRIKE_PREFILL_ANE=on`,
`ANEPrefillAttention.swift:21`) is an opt-in experiment with its own record, the one code
path v17 left behind an environment switch.

The turn ([v13-the-turn.md](v13-the-turn.md)) is the server's prompt cache
(`ServerPromptCache.swift`, [v6-dialect-normalized-cache.md](v6-dialect-normalized-cache.md):
the cached KV equals what the dialect's template renders, so a match is a byte comparison)
plus the settle and the rewrite after an answer, the resident sweep and fetch depth 2:
the 305-token first turn 5.47 to 3.54 s on the mini, the 1,085 8.08 to 6.47, the 2,125
11.24 to 10.30, warm first turns after an answer 3.45 / 6.39 / 10.29.

v12's losing kernel variants and its twenty-one tuning knobs went in v17 at their measured
defaults: the alternate attention tiles and the tensor-ops 2D path, the block router, the
per-expert routed GEMM with its gather and scatter, the four losing sweep orders and the
carry plumbing, three MPP instantiations. The tiled attention kernel and the serial GDN
scan stay as the winners' own fallbacks: the tiled kernel for a chunk the matrix tile
refuses (`matrixPathAccepts`, `PrefillAttention.swift:256`), the serial scan for a chunk
under the chunk size. Speculative decode (MTP) went with them: it was retired at v12's P17
(rig acceptance 20.6 %) and v17 deleted the code. The `.gturbo` format keeps its MTP
family and the roster still excludes a sidecar bundle, which the loader refuses by family
before any tensor check (`Model.swift:635`, `:825`).

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

## The four invariants of v4, re-verified at v17's close

1. **RAM budget is an input, not an outcome.** Still true. `--ram-budget`
   (`ServerArguments.swift:304`, `RuntimeConfiguration.parseBudgetBytes`) defaults to 8 GiB
   (`defaultExpertCacheBudgetBytes`, `RuntimeConfiguration.swift:98`); the slot count is
   the ladder value (8 to 128) nearest budget over stride times routed layers
   (`expertCacheSlots`, `:131`, resolved at `ServerInference.swift:798`); the arena is
   sized from that count plus the ring's nine. The mini runs `--ram-budget 8G`, 128 slots,
   about 9.06 GB allocated.
2. **Streaming that genuinely uses the disk.** Still true, with the figures replaced.
   The v4 rates (2.83 GB/s at prefill, 2.6 at decode against a 3.92 ceiling) were rig-era
   and described the old reader. Production's term is per read: 0.73 to 0.80 ms at p50 on
   the mini, 12.6 to 13.6 reading layers per token, 56.6 / 38.7 / 36.0 ms of reads in a
   65 / 61 / 62 ms token (v15, v16). Splitting one read across several preads is null on
   the mini's internal drive (2026-09-04, `v10-implementation-plan.md`, the P3 follow-on);
   the mini needs many experts in flight to saturate. The I/O threads remain the one CPU load safe beside decode: they block in
   the kernel rather than burn ALU (v4's measurement, unchanged in kind).
3. **No per-layer CPU round trip in decode.** Still true, and now the mechanism above:
   the classifier on the GPU, the speculative routed work a layer ahead in the layer's
   own command, the host woken by a word, the fixup gated on the storage event, the token
   boundary one command. The v4 text's per-layer sync stall at an
   8 % miss rate is gone; what remains per reading layer is the read itself.
4. **C99 for hot loops, Swift for structure.** Still true; the target grew.
   `ShrikeKernelsC` is 577 lines of C in two files (`expert_io.c` 442, `int4_affine_gemv.c`
   135) plus 178 of headers, counted from the tree, against v4's 439. The int4 GEMV's 2.9x
   and the reader's threads are v4's measurements; the atomics the word wake and the
   residency publish use are the third thing the target owns (`shrike_atomics.h`, an
   acquire load of a 32-bit word and a release store of a 64-bit one).

## The knobs

Thirteen `SHRIKE_*` names are read under `sources/`, counted from the tree: eight product
settings, four instruments and the ANE prefill switch. Nothing else selects a code path.
Every performance choice the chapters measured is a constant at its winner, and the losing
arm is deleted; git history and each chapter's design doc are the record of what the arms
were.

| knob | read at | what it does |
| --- | --- | --- |
| `SHRIKE_THINKING_MODE` | `Tokenizer.swift:50` | off, on or `adaptive` thinking for a dialect that has it |
| `SHRIKE_REASONING_EFFORT` | `Tokenizer.swift:68` (the server validates at `ServerArguments.swift:172`) | low, medium or high, the default medium ([v7-reasoning-effort.md](v7-reasoning-effort.md)) |
| `SHRIKE_REASONING_RETENTION` | `Tokenizer.swift:86` (validated at `ServerArguments.swift:178`) | `as-generated` or `stripped` reasoning in the turn's history ([v6.1-reasoning-retention.md](v6.1-reasoning-retention.md)) |
| `SHRIKE_TOKENIZER_DIR` | `Tokenizer.swift:184` | an override tokenizer folder, unset by default |
| `SHRIKE_MODEL` | `AppModelInstallDescriptor.swift:120` | the app's model selector, one of the roster's names |
| `SHRIKE_STRIP_CLI_PROMPT` | `CLIStrip.swift:34` | drop a coding CLI's system and developer boilerplate from the prompt |
| `SHRIKE_STRIP_TAGS` | `CLIStrip.swift:44` | the block tags that strip removes, `system-reminder` by default |
| `SHRIKE_CONCISE_MODE` | `ServerInference.swift:950` | the per-quant concise instruction, off by default |
| `SHRIKE_RUNNER_STATS` | `RealForwardRunner.swift:1232` (the server's footer at `ServerInference.swift:1952`) | the runner line: the per-stage split every chapter's rows are read from |
| `SHRIKE_KERNEL_STATS` | `RealForwardRunner.swift:1230` (the footer at `ServerInference.swift:1956`) | the per-kernel GPU timeline |
| `SHRIKE_ROUTE_TRACE` | `RealForwardRunner.swift:1238` | a path: every layer's top-k, what the replay and the coverage tool read |
| `SHRIKE_PREFETCH_TRACE` | `RuntimeConfiguration.swift:188` | a JSONL path: the ring's predictions, landings and misses per layer |
| `SHRIKE_PREFILL_ANE` | `ANEPrefillAttention.swift:21` | `off` or `on`: the ANE prefill attention experiment ([ane-prefill.md](ane-prefill.md)) |

The two stats names are what `tools/mini-deploy.sh` sets at the production launch and what
`tools/decode-rig.sh` and `tools/turn-rig.sh` set on every launch of theirs, the rig adding
the route trace and, under `PREFETCH_TRACE=1`, the prefetch trace.

One tripwire guards the set. `RuntimeConfiguration.refuseUnknownEnvironment`
(`RuntimeConfiguration.swift:200`) scans the environment for any `SHRIKE_*` name outside
`knownEnvironmentNames` (`:191`) and fails the launch by name, listing the offenders
sorted and naming the chapter that removed them (`:56`). It runs first at the server's
launch (`ShrikeServer/Command/main.swift:18`) and again in the session's load
(`ServerInference.swift:692`), in the CLI's run (`ShrikeCLI/Run.swift:64`) and in the app
client's load (`RealInferenceClient.swift:76`), all before any model load, so a stale
launch script fails loudly instead of quietly taking a default. The 53 names v17 removed,
each with the measurement that closed it, are in
[v17-consolidation.md](v17-consolidation.md).

## The long functions

There are none: no function body is over 120 lines, there is no swiftlint baseline, and
the gate is a bare `swiftlint lint --strict` over `force_cast`, `force_try` and
`function_body_length` (warn 120, error 400). The shape the chapter settled on, and the
one CLAUDE.md now asks for as code is written, is a sequence of named stage methods over a
small context struct, in the order the work runs, with the caller reading as the stage
list: `encodeDecodeRoutedMoE`'s nine stages over `DecodeRoutedLayerContext` are the
worked example, above.

Where the headroom is thin, so a reader knows what a new branch costs:

| function | body lines |
| --- | ---: |
| `ServerArguments.ParseContext.apply(flag:value:)` (`ServerArguments.swift:213`) | 110, an exhaustive flag switch; the next two or three flags put it over, and the honest split then is by option group |
| `RealInferenceSession.run` (`RealInferenceClient.swift:285`) | 101 |
| `RealForwardRunner.executePrefillChunk` (`:1618`) | 100 |

The per-function record, the fourteen bodies that were over the bar and what each became,
is [v17-consolidation.md](v17-consolidation.md)'s Task 4 table.

## The instruments

- `tools/golden-baseline.sh --check`: the only check that runs real inference; greedy,
  byte-identical, two profiles per machine tag under `baselines/`.
- `tools/decode-rig.sh` with `tools/decode-rows.py`: the three request shapes on the
  mini, a fresh server per shape, every token's arrival streamed, one row per request.
  Every launch carries `SHRIKE_RUNNER_STATS=1 SHRIKE_KERNEL_STATS=1` and a
  `SHRIKE_ROUTE_TRACE` path; `PREFETCH_TRACE=1` adds a `SHRIKE_PREFETCH_TRACE` path. Its
  miss-window row reads from the layer's merged role, `layer_linear` or `layer_kv`, to
  the fixup's.
- `tools/turn-rig.sh` with `tools/turn-summary.py`: the turn's shapes (a pair, a suffix,
  the multi-turn chain), the same two stats names on every launch.
- `tools/expert-pool-replay.py`: a route trace replayed against the pool's policy and
  the ring's fills; trustworthy for misses, blind to milliseconds (v16). It keeps the
  fill-mode controls, including no fills, that the runtime no longer has.
- `tools/prefetch-coverage.py`: a prefetch trace's predictions priced offline against a
  route trace.
- `tools/prefill-ledger.py`, `tools/parse-kernel-stats.py`, `tools/parse-runner-stats.py`:
  the runner and kernel stats lines read into ledgers. The kernel line records a folded
  layer once, as `layer_linear` or `layer_kv`, and the boundary command as `head_logits`;
  the runner line counts the two word wakes' fallbacks as `path_router_wake_fallbacks`
  and `boundary_wake_fallbacks`.
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
- v17 ([v17-consolidation.md](v17-consolidation.md)): the consolidation. This document,
  then the knobs 66 to 13 behind a tripwire (the decode execution enum to one path, two
  cache layouts to one, three expert readers to one, Metal kernels 82 to 65, source files
  250 to 235, +908 −12621 lines, MTP and the bench target gone), one residency publish
  path (the entry 16 bytes to 8 in one release store, fourteen writer sites to eight calls
  of one function, two generation spaces to one per cell), and the runner decomposed (the
  swiftlint baseline 18 entries to none and the file gone, the longest body 289 lines to
  110). The wall flat within the drift on all three shapes at every task's end, every
  answer identical, the misses to the tenth.
- v18 ([v18-quiet-host.md](v18-quiet-host.md)): the quiet host. The hits in the
  speculative command (a null kept as a simplification), the token boundary as one
  command (+1.4 % on a same-box A/B), one command per layer (+0.4 to +0.8 %), the walls
  (the speculative command and the fixup on one encoder each, +1.6 to +2.8 %; the
  boundary command on one encoder, the gate and up GEMVs as one grid and the residual
  folded into phase 2 kept as class 1 nulls; the other four merges not built by the
  floor rule; the fold deferred to v20 as structure). The wall on the mini 15.6 / 16.4 /
  16.2 to about 16.2 / 16.9 / 16.8 tok/s on the card, the 300 and the 1k, about +3.5 to
  +4 %, golden identical at every commit; one Metal kernel added, the gate and up grid,
  66 in the tree; no knob added or removed; the measurement-grading rule (every cost a
  grade and a range, tasks ranked by the floor).
