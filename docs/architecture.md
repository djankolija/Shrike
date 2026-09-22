# Architecture

How the engine is built and why, written from the tree at v17's close (`0dbeba0`,
2026-09-08) to replace the v4-era document, brought to the tree at v18's close
(2026-09-09), to v19's (2026-09-17) and to v20's (2026-09-18). Every piece below carries the measurement that
keeps it and the chapter that measured it. Numbers are measured on the Mac mini (M1,
16 GB, the deploy target) unless marked modelled or counted; line references are at
v20's closing tree. Each chapter's own record, what its tasks built, deleted and
counted, is its design doc beside its plan: [v17-consolidation.md](v17-consolidation.md)
for the consolidation, [v18-quiet-host.md](v18-quiet-host.md) for the decode path's
command structure below, [v19-scan-rewrite.md](v19-scan-rewrite.md) for the attention
scan, [v20-ssd-mechanism.md](v20-ssd-mechanism.md) for the pool's allocation and the
fold of the miss path into the token's command; git history holds the paths the
chapters removed.

The product claim is bounded memory: run a mixture-of-experts model larger than RAM by
streaming routed experts from SSD into a cache whose size is declared, not discovered.
Production on the mini at v16's close answered the card / the 300-token / the 1k-token
shapes at 15.34 / 16.46 / 16.12 tok/s with 20.0 / 20.0 / 18.8 expert misses per token, the
token's pace set by the reading layers' SSD chain (56.6 / 38.7 / 36.0 ms of reads in a
65 / 61 / 62 ms token); v18 took the three shapes to about 16.2 / 16.9 / 16.8 tok/s with
the misses per token within 0.2 of those and the reads' term untouched
(`v18-quiet-host.md`); v19 added the 7k shape, the context a coding session runs at,
and took it from 13.7 to 17.05 tok/s by the attention scan alone, the card to about
17.2 (`v19-scan-rewrite.md`); v20 gave the pool a per-layer allocation and segmented
LRU (+4.4 to +7.8 % on the four shapes, the misses per token 19 to 20 down to 14 to
17) and folded the miss path into one command per token committed ahead of the
sampler, so production under its configuration answers the card / the 300 / the 1k /
the 7k at 18.4 to 18.5 / 18.2 to 18.4 / 18.8 to 18.9 / 18.2 tok/s
(`v20-ssd-mechanism.md`).

## The decode path, token by token

`RealForwardRunner.produceToken` (`RealForwardRunner.swift:2236`) runs one token as
one Metal command (v20 T3.2): the embed when the caller passes a token, every layer as
encoders of that command (`encodeLayers`, `:2128`, calling `encodeDenseLayer`, `:2454`,
or `encodeRoutedLayer`, `:2150`), then the token boundary (`encodeBoundary`, `:2419`)
or the synchronous head (`encodeHead`, `:2501`). A continued pass takes instead the
command the previous token encoded ahead a layer per word and, since v20 T3.3,
committed after its own last word (`takeHeldToken`, `:1685`). The host's loop over the
routed layers then does, per layer: the next token's same layer encoded into the next
command, the layer's word awaited, the previous layer's batch checked and its plan run,
the layer serviced (its reads issued); after the last word the next token's remaining
layers and its boundary are encoded and the next command is committed, so the GPU runs
from this token's embed into the next token's layer 0 with nothing of the host between
them.

### The token's command

`TokenCommand` (`:2098`) holds the command buffer, made from a
`MTLCommandBufferDescriptor` with `encoderExecutionStatus` (`makeTokenCommand`,
`:2117`) so a fault names the encoder that faulted and counts the affected
(`describeCommandBufferError`, `:3269`); the GDN parity its layers read; the routed
layers' `TokenLayer` records (the classifier's readback tag and the layer's agreed
value); and, on the boundary path, the word its sampler writes. A routed layer is three
encoders in order: `layer L attention` (the input norm, the attention and the tail with
the classifier on one serial encoder on GDN and gated layers), `layer L routed` (the
shared-expert chain and the speculative routed phases) and `layer L fixup` (the wait on
the layer's value and the agreed fixup). A fresh token is committed once its boundary is
encoded (`commitToken`, `:2357`); the previous token's completion is waited on behind
that commit (`finishToken`, `:1697`), where it is free, and recorded under the `token`
role. The per-encoder error option is always on: it measured free on the four shapes
(v20 T3.2's third arm, a variable that lived for those lifetimes only).

**Kept by:** v20 T3.2 (`v20-ssd-mechanism.md`): the forty per-layer command boundaries
gone, the token 0.2 to 0.6 ms faster on most rows of both arms and level on the rest,
the misses and the answers unchanged; the boundary between tokens measured directly
for the first time as the gap between consecutive `token` commands, 0.26 to 0.34 ms
per token. v20 T3.3 took that row to 0.033 to 0.038 (below).

### The drain invariant

Every timeline value reserved at a routed layer's encode is published exactly once:
by the layer's batch when its reads land, by the host at the word when nothing was
read, or by the drain as failed on every other exit. The armed values live in
`armedAgreedTokens` (`:5069` drains them); the drain runs on a pass's throw path,
in `discardBoundaryState` (`:1658`) from every entry point that submits GPU work
after a stop (`reset`, `prepareForContinuation`, `rewind`, `restoreInferenceState`,
`prefillChunked`, a fresh `produce`, `settle`) and, since T3.3, at the loop's exit
through `releasePassAhead` (`:1649`). A command whose values are published failed
runs through with its fixups skipped (`moe_io_ready`, `moe.metal:36`), so a drained
command always completes. `awaitCompletion` (`:3243`) bounds every completion wait
at ten seconds and names what was waited on; the boundary wake falls back to it
after a second, and the word wake keeps polling its word to the same deadline (the
token's command cannot complete before the host has serviced every later layer, so
a completion wait there would deadlock until the deadline), so a wait nothing will
publish ends as `commandBufferFailed` naming the layer or the boundary, never a
hang. A failed read
names its layer (`ModelError.expertReadFailed`).

**Kept by:** v20 T3.1 and T3.2 (the tests: a throw at layer k completes the command,
names the layer, hangs nothing and the next request runs; a wait on a command that
never completes ends at its deadline naming the layer).

### The speculative routed work

`encodeSpeculativeRouted` (`:5418`) encodes, a layer ahead of the host's knowledge of
the route, five dispatches on one serial compute encoder: the shared-expert chain
(the gate and up INT4 GEMVs as one grid, `dequant_int4_shared_gate_up_gemv_simd`
through `FusedGateUpGEMV` (`Kernels/Fusions/FusedGateUpGEMV.swift`), dispatched by
`SharedExpertInt4.encodeGateUp` (`Kernels/MoE/SharedExpertInt4.swift:91`); the scalar
gate; the fused silu-mul-down with the sigmoid gate), then a pool-addressed phase 1
and phase 2 whose grids size themselves from the classifier's indirect arguments
(`moe_phase1_gate_up_act_spec_u16load` at `Metal/MoE/moe.metal:937`,
`moe_phase2_down_reduce_spec_k8` at `:996`, whose epilogue finishes each element
with the residual add, `moe_phase2_finish` at `:876`). One serial encoder, never a
`.concurrent` one: a `.concurrent` encoder followed by an indirect dispatch on the
same command segfaults the AGX driver (v9's trap). The classifier
(`moe_classify_expert_residency_spec`, `moe.metal:194`) runs as the tail's last
kernel: for each of the router's top-k experts it reads the layer's residency table,
counts hits and misses, writes the hit positions and the resolved cells (a miss's
position carries the `0xffffffff` sentinel), the miss list for the host, a tagged host
readback word, and four grids (`MoESpecDispatchArgs`, `:111`): the speculative
phase 1's full grid on every layer, since the kernel's rows skip a sentinel position,
so the hits are computed inside the layer's command on a miss layer too; the
speculative phase 2's full grid only when every expert is resident; and the agreed
fixup's two grids, full only when an expert missed.

The fixup is the same two kernels encoded with the layer, before its router has run
(`encodeAgreedFixup`, `:5021`): behind `encodeWaitForEvent` on the layer's value, the
phase 1 over the host's per-layer row of agreed cells with the sentinel at the hits,
the phase 2 resolving a sentinel through that row; both gated by the value's status
word, and the speculative pair binds an always-ready word and the classifier's array
in their place. On an all-hit layer the value is published at the word and the fixup's
grids are zero; on a miss layer the host names the cells and the batch publishes the
value when the reads land. No command is built on the host on decode.

**Kept by:** v9 (the speculative routed dispatch; the per-slot to pool flip alone
halved the GPU idle gap, 32.6 to 16.7 ms per token on the rig,
`v9-speculative-routed-dispatch.md`); the decode chapter's arithmetic across v9 to
v11, rig 111.6 to 38.1 ms per token (`v10-implementation-plan.md`); the hits on every
layer, v18 Task 1; the one encoder, v18 T6.0 (3.3 ms per token of GPU role time over
the 147 encoder boundaries that went; the wall +1.6 to +2.8 % on the 300 and the 1k);
the agreed cells, v20 T3.1 (the fixup encoded ahead: the token flat within the drift,
the misses within 0.2 of the host-built fixup's, the io 0.3 to 0.5 ms lower with the
demand batch reaching the drive a few tens of microseconds earlier; the host-built
fixup, `DecodeExpertPartition`, the pending routed command and the completion clock
retired, `v20-ssd-mechanism.md`). The pool victim on the path (`agreed_overflow`) is
the fallback when the ring has no free cell for a miss; it never fired on the arms.

### The word wake

The host does not wait for a layer's command to complete. `waitForWord` (`:3213`) spins
on the classifier's tagged readback word for a second, then polls it gently to the
ten-second deadline, checking the command's status as it goes so a failed command
still surfaces (the slow phase counted as `path_router_wake_fallbacks`). With the routed work behind the classifier in the same
command the word lands mid-command, tens of microseconds after the kernel writes it
(`MidCommandVisibilityTests`: 42 to 45 µs after the command's GPU start, three runs).
The token boundary takes the same wake on the sampler's token word
(`awaitBoundaryToken`, `:1607`, `boundary_wake_fallbacks`). Since v20 T3.3 there are
two boundary words, the runner's, by token parity (`decodeScratch.boundaryWords`,
`:249`): a token's sampler writes its own, so the pass committed ahead never
overwrites a word the host has yet to read; the sentinel (`0xFFFFFFFF`) goes into a
token's word at its commit. Every word's arrival is recorded by the word clock
(`DecodeWordClock.swift`), the per-layer instrument that replaced the per-layer GPU
rows when the token became one command: a layer's wall is the gap between
consecutive words, the first routed layer's from the commit or, for a token
committed ahead, from the previous token's word, the boundary's from the last word to
the token word.
The readback (`RouterHostReadback.swift`) carries the top-k ids and weights, the hit
and miss positions and the probe's predictions in 32-bit words, each a 16-bit tag over
a 16-bit value, read with an acquire load (`shrike_load_acquire_u32`,
`ShrikeKernelsC/include/shrike_atomics.h:7`).

**Kept by:** v14 lever B, the shipping default: +2.8 to +3.6 % tok/s on the three
shapes, the router wake's 4.0 ms of stat and 2.1 to 2.5 ms of wall per token
recovered (`v14-decode.md`). It is the only wake the routed path takes.

### The agreed cells: the word, the batch, the plan at the next wake

`serviceAgreedLayer` (`:5113`) runs at layer L's word, with layer L's command still
running its speculative work and waiting at its fixup:

1. the route is read from the readback (`readDecodeRouterReadback`) and the route
   trace recorded;
2. the ring's landed predictions for this layer are leased (`readyCells`,
   `ExpertPrefetchRing.swift:171`, joining a read still in flight for up to 400 µs);
3. every miss is given a cell: a landing's, a free ring cell claimed as a landing
   through the ring's `claimDemand` (`:257`) and the streamer's `claimLanding`
   (`PreadExpertStreamer.swift:803`), or, when the ring has none, a pool victim from
   `reserveOverflowSlot` (`:852`), counted as `agreed_overflow`;
4. the host's row of agreed cells is written (one row per layer, the hits at the
   sentinel), the batch is submitted through `beginAgreedReads` (`:887`) on the demand
   lane with the layer's value (an empty batch publishes the value at once), the ring's
   demand cells are attached, and the next layer's prediction is issued.

At the next wake `finishPendingAgreedLayer` (`:5257`) checks the previous layer's
command and its batch (a failed read throws `expertReadFailed` naming its layer), takes
the io rows, then runs the plan off the path (`planAgreedLayer`, `:5280`):
`Model.planRoutedExperts` (`ModelExpertIO.swift:96`) with the predictions and the
demand cells as its leased landings and the misses counted as the reads issued, so
`expert_misses_decode` keeps its meaning; under the layer's cache lock the plan
decides the hits' use counts and promotions and swaps every leased cell into the pool
by index against a victim chosen now (the streamer's `makeExpertCachePlan`, `:395`,
under the pool's policy: the aging-LFU with chunk protection by default, segmented
LRU when configured, v20 Task 1); the freed cells go back to the ring; the prefetch
trace rows are written. The last layer's plan runs at the token's end on every exit.

**Kept by:** the miss window chapter, v15 (the placement gate, the 400 µs join and the
fused probe: 14.1 / 14.8 / 15.0 to 15.4 to 15.6 / 16.3 / 16.2 tok/s,
`v15-miss-window.md`); the landing, v16 (the swap at the plan; kept as a
subtraction, `v16-landing.md`); the agreed cells, v20 T3.1 (the plan off the path:
`cache_plan_ms` 0.14 to 0.21 per token now spent at the next wake).

### The token boundary

The boundary is the token's last encoder (`encodeBoundary`, `:2419`): the final norm,
the lm_head GEMV, the caller's sampler (`Runtime/Generation/Sampler.swift`: the tiled
softmax and the top-k-64 kernel; the generic kernels stay the path for greedy, for k
above 64 and for top-k disabled) writing the token's word, and the next token's embed,
which reads the sampled id from that word (the two embed kernels' `tokenBuffer:`
overloads, `Kernels/Quant/EmbedLookupInt4.swift:89` and `AffineQuant.swift:141`).
`BoundaryLogitProducer` (`Runtime/Generation/LogitProducer.swift:20`) is the shape the
runner conforms to: the caller's sampler closure is given the position of the pass it
ends and the word its token goes into, and the runner encodes it for this pass when
the pass is fresh and for the next pass at the end of every pass, unless the caller
passed `last`.

Under v20 T3.3 the next token's command is committed right after the current token's
last routed word, before the current sampler has run, so the GPU flows from this
token's embed into the next token's layer 0. The loop
(`Runtime/Generation/RawCompletion.swift:263`) then spins on the word, checks the stop
token, detokenises, runs the stop-string matcher, the external stop and the max-tokens
check: every stop but max tokens is therefore seen one pass late, with the next pass
already running as the extra pass; max tokens never is, since the loop passes `last`
on the pass whose boundary sample would reach it and that pass commits nothing ahead.
At the loop's exit on every path, the stops and a disconnect's cancellation alike
(`:288`), `releasePassAhead` publishes the extra pass's forty values as failed, so it
runs through with its fixups skipped during the answer's finish frames and the
client's turnaround; the wait for it is the next entry point's (the drain, above),
counted on the runner line as `drained_passes` and `drain_ms` on the line after the
submission that waited it out. What the extra pass touches survives the stop by
construction: the gated-DeltaNet recurrent state and conv tail of every linear layer
are held in two parities (`GDNStateManager`, `[parity][layer]`, 61.4 MiB more on the
served model; the decode kernels `gdn_conv_mix_decode`, `gdn.metal:250`, and the
`gdn_delta_step_decode` pair, `:503` and `:524`, take the state entering the step and
the state leaving it, the arithmetic unchanged), a pass reads the parity holding the
state at the cursor (`gdnStateParity`, `RealForwardRunner.swift:253`) and writes the
other, and the cursor's advance flips the parity, so the extra pass writes the parity
the stop's state is not in; prefill, the snapshot and the restore work in place on
the cursor's parity. The KV row the extra pass writes sits past the cursor, which
its pass never advances, so the cursor after a stop is where the stop left it and no
rewind is needed. The prompt cache's settle takes its snapshot without waiting for
the drain: it reads the cursor's parity and the rows below the cursor, neither of
which the extra pass writes.

The loop chooses the path once per generation and keeps the synchronous head
(`encodeHead`, `:2501`: the final norm and the lm_head GEMV on one encoder, or the
fused greedy head that writes the argmax token directly, `useFusedGreedyHead`,
`:335`) for a repetition penalty other than 1.0, for the fused greedy head, for the
forced-token and logits-sink instruments and for the first token after prefill.

**Kept by:** v18 Task 4 (the three boundary gaps, 0.83 to 0.90 ms per token, to one of
0.25 to 0.27; +0.9 % on the 300 and +1.4 % on a same-box A/B on the 1k,
`v18-quiet-host.md`); v20 T3.3 and T3.4 (`v20-ssd-mechanism.md`): the boundary gap
0.26 to 0.34 ms per token at T3.2 to 0.033 to 0.038 on every arm and shape, the
`token` row's GPU span equal to the token within 0.2 ms, the token faster by about
that or more on every configured row (0.4 to 1.2 ms) and on two of the four bare
rows, level on the other two; the misses, the io and all twenty answers' bytes
unchanged; the drain's wait on the request after a stop 0.000 ms, the pass having
run through during the finish frames and the cache's capture; the golden identical
on all five profiles, the fifth the two-turn continuation gate (`turns-lh`).

### The dense layers

Layers below `numLeadingDenseLayers` (Kimi's layer 0) take `encodeDenseLayer` (`:2454`)
as encoders of the token's command: norm, attention, the dense SwiGLU, no classifier
and no routed stage.

### The attention scan

The ten full-attention layers' decode attention is two passes on the layer's encoder: a
partial kernel writes sixty-four partials per query head (a running max, a denominator
and an unnormalised output row each) and `attention_decode_combine`
(`attention.metal:861`) reduces them. The KV cache holds K and V at int8 in production:
one 544-byte row per position per layer for each, both KV heads' 256 values packed and
eight fp16 scales and eight fp16 biases inside the row, in groups of 64
(`KVCacheManager.rowLayout`, `:476`).

The partial on the served shape is `attention_decode_partial_stream`
(`attention.metal:533`, v19): a threadgroup per KV head and chunk, its eight simdgroups
four position streams by two head sets of four query heads; each lane loads its eight
contiguous bytes of the half-row from device straight into registers, dequantizes in
fp32, dots its four heads, reduces once per head with `simd_sum`, runs the online
softmax per head and accumulates V the same way; no threadgroup memory and no barrier
in the loop; each stream writes its own partial, so sixteen chunks are dispatched per
KV head and the combine's contract is unchanged. The wrapper takes it only for the
served shape on int8 rows (`Attention.streamServes`, `Attention.swift:83`: head dim 256,
eight query heads per KV head), with a specialized pipeline per shape key (`:676`);
every other shape keeps the v11 shared partial `attention_decode_partial_shared`
(`attention.metal:377`), which stages four positions through threadgroup memory per
barrier pair. The runner refuses a Qwen-family model outside the streaming shape at
load (`RealForwardRunner.swift:577`) rather than serve it slower without a word;
`RuntimeConfiguration.attentionFallbackAllowed`, off in production, lets the runner
tests load their toy shape.

**Kept by:** v19 ([v19-scan-rewrite.md](v19-scan-rewrite.md)). The chapter's ladder on
the mini, through what is now `shrike bench attention`, found the shipped kernel bound by
its loop form, not by memory: a static trip count with the explicit fused multiply was
2.7× on the kernel and bit for bit the shipped output (Task 2, class 1; the slope 2.23
to 0.72 ms per 1,000 context tokens per decoded token); the streaming structure a
further 1.85× (Task 3, class 2 under the forced-token instrument and the read; the slope
to 0.39). The 7k token on the mini 73.0 to 58.6 ms, 13.7 to 17.05 tok/s. What binds now
is the per-position chain under a register-limited occupancy; the reference's 0.2 to
0.3 per 1,000 is 1.3 to 2× away.

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
(`PreadExpertStreamer.swift:1122`), which packs the state above the slot and stores the word
once with `shrike_store_release_u64` (`ShrikeKernelsC/include/shrike_atomics.h:12`). No
caller writes the buffer. It has twelve call sites in ten functions, all but the init's
fill under the streamer's `cacheLock`; the init's fill runs before any reader exists:

| # | call site | thread | trigger and what is written |
| ---: | --- | --- | --- |
| 1 | `init` (`:340`) | the streamers queue | the layer's first touch: every entry `empty` |
| 2 | `makeExpertCachePlan`, the victim (`:485`) | the planner's | the evicted expert `empty`, after the cell's generation is bumped |
| 3 | `makeExpertCachePlan`, the reservation (`:505`) | the planner's | a miss on prefill: `loading` at the slot's cell |
| 4 | `markPlanMissesResident` (`:1086`) | the storage thread | a plan's read completed: `resident`, every miss's cell generation re-validated first |
| 5 | `resetLoadingMissesUnlocked` (`:1108`) | the failed read's, the abandoned plan's | `loading` back to `empty` |
| 6 | `claimLanding` (`:812`) | the issuing thread (decode or storage) | a ring cell claimed, for a prediction or for a route's agreed read: `loading` at the ring cell, whose generation the claim bumps |
| 7 | `completeLanding` (`:831`) | the storage thread | the read into the ring cell landed: `resident` |
| 8 | `dropLanding` / `failLanding` (`:995`) | the storage thread, the issuing thread, the ring's reclaim | `empty`, only if the pool does not own the expert |
| 9 | `reserveOverflowSlot` (`:871`, `:876`) | the decode thread, at the word | a miss the ring could not cell (v20 T3.1): the victim `empty` after its generation is bumped, the miss `loading` at the victim's cell |
| 10 | `markOverflowResident` (`:955`) | the storage thread | the agreed read into a pool cell landed: `resident` |
| 11 | `emptyOverflowSlot` (`:971`) | the failed batch's, the abandoned pass's | `loading` back to `empty` |

The swap writes nothing (`:467` to `:481`): a landing already stands `{cell, resident}` in
the table, so the slot takes the landing's cell and only the host's bookkeeping moves.

The planners are the decode runner (`planAgreedLayer`, `RealForwardRunner.swift:5280`,
through `Model.planRoutedExperts`, at the wake after the layer's reads), prefill's tile
fetches
(`PrefillGroupedRoutedMoE.swift:579`, `:631`) with the tile scheduler's lookahead
(`RealForwardRunner.swift:298`), and `Model.fetchRoutedExperts(layer:experts:)`
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
  case: on decode the plan runs after the reads, with the classifier's miss set, and a
  route it cannot place throws rather than compute over an unread cell
  (`planAgreedLayer`, `RealForwardRunner.swift:5280`).
- **The generation is host bookkeeping, one word per arena cell.** It lives on
  `ExpertCellArena` (`ExpertCellArena.swift:60`, bumped at `:66`), not in the table, and
  the classifier neither reads nor writes it. Every value comes from one atomic clock, so
  no two cells' bumps can coincide and a stale plan's recorded generation can never equal a
  different cell's by chance. It is read and written under the owning layer's cache lock,
  and it guards a stale completion against publishing over a newer occupant
  (`markPlanMissesResident`, `PreadExpertStreamer.swift:1086`). The swap moves the landing's
  cell under the slot and
  its generation with it.
- **Lock order.** The ring's lock, then a layer's cache lock, never the reverse
  (`ExpertPrefetchRing.swift:38`). The ring's `begin` takes the cache lock once per claim,
  so a batch's claims are not atomic as a group.

### The arena and the ring

`ExpertCellArena` (`Infrastructure/Streaming/ExpertCellArena.swift`) holds every expert
cell the classifier can name, numbered globally: each routed layer's slots, the uniform
count by default or the per-layer table (`SHRIKE_EXPERT_SLOT_TABLE`, v20 Task 1:
prefix-sum cell ranges, a table refused at load unless its count, its floor of 8, its
dense zeros and its total against the budget hold; the served model's table is blend
0.3 of its production miss profile scaled to 6,400, 130 to 256 slots by layer), plus the
ring's nine, at the page-rounded expert stride. Since v22 Task 2 the cells live in as
many Metal buffers as the device's `maxBufferLength` needs (chunks of equal cell
count except the last, which takes the remainder,
at most eight; one on the M4 Pro, two on the mini, whose limit is 8.88 GiB), and the
arena publishes `PoolBases`, the chunks' GPU addresses and the cells per chunk, which the
two speculative kernels take at buffer 0 and address a cell through (`pool_cell_base` in
moe.metal); every other path names a cell by its global index and reaches its bytes
through the arena's `buffer(cell:)` and `bufferOffset(cell:)`, the streamer keeping the
global offset as a slot's identity. A cell changes owner at a swap without a byte moving,
across chunks as within one; that is the whole reason for one numbering. The pool's
eviction is the aging-LFU with chunk protection by default and segmented LRU under
`SHRIKE_EXPERT_POLICY=slru` (the protected share 0.5), the served model's production
configuration. On the mini the budget is 160 slots per layer since v22 (11.33 GB of
cells; the allowed counts run to 256), with oMLX's models unloaded and the prefill
scratch released between requests (v22 Task 1: the chunk's private buffers, about
600 MB, held only while a prefill runs).

`ExpertPrefetchRing` (`Runtime/Inference/ExpertPrefetchRing.swift:39`) owns the top-k plus
one cells, nine on this model (`makePredictivePrefetch`, `RealForwardRunner.swift:449`),
and at most one read in flight across all layers (`inFlightBudget`,
`ExpertPrefetchRing.swift:49`; v15 step zero: a
read still in flight shares the drive with the next demand read). At layer L's wake the
probe's top-k for layer L+1 is issued after the demand submission, one layer ahead (the
placement gate and distance one, v15's constants); a prediction lands in a
ring cell and is published `resident` from the storage thread, so layer L+1's classifier
can hit it; the plan swaps a wanted landing in and returns the freed cell; the reclaim
drops an unwanted one when the ring needs the cell. Since v20 T3.1 the ring also lends a
free cell to a route's agreed read at the word (`claimDemand`, `:257`), consumed at the
plan like a landing and never counted adopted; its `leasedPeak` is on the runner line. The probe distance above one and the
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

A miss is read by `PreadExpertStreamer.beginAgreedReads` (`:887`) on decode and by
`beginExpertCachePlan` on prefill, both on `ExpertIOScheduler` (`ExpertLoadOperation.swift:175`): four `.userInitiated` worker
queues, demand batches queued ahead of speculative ones. The only reader is the bounded
pread reader in C (`ShrikeKernelsC/expert_io.c`, `shrike_expert_io.h`) behind
`ParallelExpertReader` (`ParallelExpertReader.swift`): fixed reader threads (four: v4's
knee, and v13 measured that a second four buys nothing at a tile's three to four misses),
`F_NOCACHE` reads, at most two published batches (v13's winner), both constants in
`BoundedReaderConfiguration` (`PreadExpertStreamer.swift:146`). A batch carries an
`ExpertIOCompletionToken` (`ExpertIOEventCoordinator.swift`): one shared Metal timeline
for the model and a status word per value (loading, complete, failed) the GPU's fixup
waits on; since v20 T3.1 every routed layer reserves a value at its encode, forty per
token, and the words are one ring of 4,096 recycled by value (`:35`), a token's command
draining within a few tokens; out-of-order completions are held until every preceding
value is terminal, since advancing the timeline past an unfinished batch would release
its GPU wait early. The completion publishes `resident` on the storage thread
(`markPlanMissesResident`, `markOverflowResident`) and wakes the ring's deferred issue. The gate on the GPU's side is a function constant: the runner
builds its `MoE` with `eventGatedIO: true` (`RealForwardRunner.swift:625`), and the
parameter's `false` default (`Kernels/MoE/MoE.swift:90`) exists only so the kernel tests
can build a `MoE` without a coordinator; the un-gated arm of `moe_io_ready` is the tests'
path, not a mode.

Production's per-read cost on the mini is 0.73 to 0.80 ms at p50 inside a 1.0 to 1.1 ms
reading layer (v15's ledger). The term's other factor moved in v20: the misses per token
19 to 20 down to 14 to 17 by the pool's allocation, the io 13.5 to 15.1 ms per token to
10.6 to 12.3 (`v20-ssd-mechanism.md`, Task 1); what a read costs is untouched.

Two other readers stood beside it until v17 and are gone with their knobs: the legacy
cached-pread path, and the Metal IO backend (its A/B of 2026-09-01: rig wait 43.29 ms sd
9.2 % against pread's 38.68 sd 2.2 %, and the server died mid-prefill,
`v10-implementation-plan.md`).

## Prefill and the turn

Prefill (`executePrefillChunk`, `RealForwardRunner.swift:1781`) runs the prompt in chunks:
per layer the attention on the matrix path (`Metal/Prefill/attention_matrix.metal`, the
causal-matrix tile `g2k256d` at `PrefillAttention.swift:88`, matrix min rows 16 at
`RealForwardRunner.swift:312`), the router over the chunk, then the routed experts as tiles
over the union of the chunk's experts, fetched two tiles deep through the same streamer
(`prefillRoutedTileSchedulerConfig`, `:298`) with a sweep order that starts from what is
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
before any tensor check (`Model.swift:676`, `:858`).

## The serving layer

`shrike serve` serves an OpenAI-compatible API on loopback over `ServerModelSession`
(`ServerInference.swift`): the tokenizer and its dialect (ChatML, Harmony), the prompt
cache, the structured decoders for thinking and tool calls
([v7-reasoning-effort.md](v7-reasoning-effort.md),
[v8-emission-form-tool-calls.md](v8-emission-form-tool-calls.md)), the runner counters on
the runner line. `shrike generate` drives one generation for the golden baseline.
Multi-model serving is recorded
in [multi-model-serving.md](multi-model-serving.md); the channel-faithful turn design in
[channel-faithful-turns.md](channel-faithful-turns.md).

## The four invariants of v4, re-verified at v17's close

1. **RAM budget is an input, not an outcome.** Still true. `--ram-budget`
   (`ShrikeServerCommand.swift:89`, the option's own transform over
   `RuntimeConfiguration.parseBudgetBytes`) defaults to 8 GiB
   (`defaultExpertCacheBudgetBytes`, `RuntimeConfiguration.swift:104`); the slot count is
   the ladder value (8 to 256 since v22) nearest budget over stride times routed layers
   (`expertCacheSlots`, `:137`, resolved at `ServerInference.swift:786`), or, under
   `SHRIKE_EXPERT_SLOT_TABLE`, the per-layer table whose total must equal that count
   times the routed layers; the arena is sized from the sum plus the ring's nine, in as
   many chunks as the device's `maxBufferLength` needs. The mini runs a budget of
   11,324,620,800 bytes, 160 slots as 6,400 cells split 130 to 256 by layer, 11.33 GB in
   two chunks (v22; 8G and 128 slots in one chunk before it).
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
   own command, the host woken by a word, the fixup encoded with the layer and gated on
   the storage event, the token one command with the next committed behind it. The v4 text's per-layer sync stall at an
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
| `SHRIKE_REASONING_EFFORT` | `Tokenizer.swift:68` (the server validates at `ShrikeServerCommand.swift:182`, at launch and again when it assembles its config, not while parsing) | low, medium or high, the default medium ([v7-reasoning-effort.md](v7-reasoning-effort.md)) |
| `SHRIKE_TOKENIZER_DIR` | `Tokenizer.swift:176` | an override tokenizer folder, unset by default |
| `SHRIKE_STRIP_CLI_PROMPT` | `CLIStrip.swift:34` | drop a coding CLI's system and developer boilerplate from the prompt |
| `SHRIKE_STRIP_TAGS` | `CLIStrip.swift:44` | the block tags that strip removes, `system-reminder` by default |
| `SHRIKE_CONCISE_MODE` | `ServerInference.swift:935` | the per-quant concise instruction, off by default |
| `SHRIKE_EXPERT_SLOT_TABLE` | `RuntimeConfiguration.swift:213` (the server and the CLI pass it at load) | a comma list or a JSON path of expert cache slots per layer, refused unless the count, the floor of 8, the dense layers' zeros and the budget's total hold; the uniform pool by default (v20 Task 1) |
| `SHRIKE_EXPERT_POLICY` | `RuntimeConfiguration.swift:245` | `aging-lfu` (the default), `slru` or `slru:<share>`: the pool's eviction policy (v20 Task 1) |
| `SHRIKE_RUNNER_STATS` | `ServerInference.swift:1937` (the runner's counters are always kept) | the runner line: the per-stage split every chapter's rows are read from |
| `SHRIKE_KERNEL_STATS` | `RealForwardRunner.swift:1227` (the footer at `ServerInference.swift:1941`) | the per-kernel GPU timeline |
| `SHRIKE_ROUTE_TRACE` | `RealForwardRunner.swift:1235` | a path: every layer's top-k, what the replay and the coverage tool read |
| `SHRIKE_PREFETCH_TRACE` | `RuntimeConfiguration.swift:203` | a JSONL path: the ring's predictions, landings and misses per layer |
| `SHRIKE_PREFILL_ANE` | `ANEPrefillAttention.swift:21` | `off` or `on`: the ANE prefill attention experiment ([ane-prefill.md](ane-prefill.md)) |

The two pool names and the two stats names are what the mini's production launch sets
(`tools/mini-production.sh`, copied in `CLAUDE.md`); a bare launch runs the uniform pool
and the aging-LFU. `tools/decode-rig.sh` and `tools/turn-rig.sh` launch production with an
arm's env layered on top, the rig adding the route trace and, under `PREFETCH_TRACE=1`,
the prefetch trace.

One tripwire guards the set, and it is an **allow**-list, which is the one way it can
fail quietly: a name left in `knownEnvironmentNames` after its reader is deleted is
silently accepted and ignored rather than refused. v24 found exactly that and fixed it —
`SHRIKE_MODEL`'s reader went with the Mac app in `8e50806` and the entry stayed, so
removing it is what makes the tripwire cover it.
`RuntimeConfiguration.refuseUnknownEnvironment` scans the environment for any `SHRIKE_*`
name outside `knownEnvironmentNames` and fails the launch by name, listing the offenders
sorted and naming the chapter that removed them. It runs first at the server's launch
(the first line of `ShrikeServerCommand+Run.swift`'s `run()`) and again in the session's
load (`ServerInference.swift`) and in the CLI's run (`ShrikeCLI/Run.swift`), all before
any model load, so a stale launch script fails loudly instead of quietly taking a
default. The 53 names v17 removed, each with the measurement that closed it, are in
[v17-consolidation.md](v17-consolidation.md); v24 removed two more,
`SHRIKE_REASONING_RETENTION` with its flag and `SHRIKE_MODEL` with the app that read
it, and the suite refuses all 55 by name.

## The long functions

There are none: no function body is over 120 lines, there is no swiftlint baseline, and
the gate is a bare `swiftlint lint --strict` over `force_cast`, `force_try` and
`function_body_length` (warn 120, error 400). The shape the chapter settled on, and the
one CLAUDE.md now asks for as code is written, is a sequence of named stage methods over a
small context struct, in the order the work runs, with the caller reading as the stage
list: `serviceAgreedLayer`'s stages over `AgreedLayerContext` (`RealForwardRunner.swift:5079`)
and `produceToken`'s word loop are the worked examples, above.

Where the headroom is thin, so a reader knows what a new branch costs:

| function | body lines (swiftlint's count, comments and blank lines excluded) |
| --- | ---: |
| `Attention.encodeSplit` (`Attention.swift:506`) | 113 |
| `PreadExpertStreamer.makeExpertCachePlan` (`PreadExpertStreamer.swift:395`) | 112, SLRU's promotion beside the aging-LFU's victim (v20 Task 1) |
| `OpenAIChatRequest.validate` (`OpenAIModels.swift:328`) | 109 |
| `MoE.init` (`MoE.swift:94`) | 108, the pipelines by variant (v20 T3.1 added the event gate to the generic speculative pair) |
| `runDecodeLoop` (`RawCompletion.swift:263`) | 106 |
| `RealForwardRunner.produceToken` (`:2236`) | about 100 |

The per-function record, the fourteen bodies that were over the bar and what each became,
is [v17-consolidation.md](v17-consolidation.md)'s Task 4 table.

## The instruments

- `tools/golden-baseline.sh --check`: the only check that runs real inference; greedy,
  byte-identical, five profiles per machine tag under `baselines/`: `short` and `long` on
  the CLI's fused greedy head, their `-lh` twins on the server's logits head (v19), and
  `turns-lh`, a chat turn answered to its stop token then a follow-up generated from the
  state the stop left (the CLI's `--follow-up`, v20 T3.3: the pass committed ahead of a
  stop and drained must leave that state exactly as a run without it). `CLI_EXTRA_ARGS`
  appends to every run, e.g. `--expert-cache-slots 160` for the mini's two-chunk arena
  (v22).
- `shrike generate --dump-logits <file>` with `tools/logit-compare.py`: the class-2
  gate's instrument (v19), every position's logits dumped and the comparison listing
  each argmax flip against the old build's top-2 margin and a band from the median
  logit difference. v24 retired `--force-tokens`, which held the two builds on one
  token sequence by construction; `--temperature 0 --seed <n>` holds them on one
  sequence as long as they agree, and `GenerationConfig.forcedTokens` remains for a
  chapter that needs the stronger form back.
- `shrike generate --dump-hidden <file>`: every position's fp16 residual before the final
  norm, the prompt's rows then the answer's, with a JSON sidecar of the positions; a
  `HiddenSink` beside the logits sink that forces the plain pass, fed from the prefill
  chunk (a blit out of the private scratch) and from each decode pass after its command
  completes (the Q3 close, 2026-09-18). With `tools/q3-drafter-routes.py` it replays the
  MTP drafter over a run and scores route predictors against the route trace.
- `shrike bench attention`: the decode attention scan on synthetic rows at the served
  shape, the production pipeline through the wrapper, the shipped kernel's copy with one
  switch per function constant, and the streaming prototype. It runs on the mini, which
  has no toolchain, so it reaches that box only inside the one deployed binary — which
  is why v24 withdrew its plan to compile the `bench` verb out of release (v19).
- `shrike bench expert`: the decode phase-1 gate/up kernel on eight real experts of a
  layer read from the `.gturbo`, the production pipeline itself as the plain arm and any
  variant held to bit-identity against it, timed with the GPU kept busy by a batch of
  dispatches per command buffer (v21's step zero, which closed the lossless-compression
  avenue on its numbers).
- `tools/decode-rig.sh` with `tools/decode-rows.py`: the four request shapes on the
  mini (the card, the 300, the 1k and, since v19, the 7k), a fresh server per shape,
  every token's arrival streamed, one row per request.
  Every launch is production's plus an arm's `SERVER_ENV` and a `SHRIKE_ROUTE_TRACE`
  path; `PREFETCH_TRACE=1` adds a `SHRIKE_PREFETCH_TRACE` path. Its
  rows since v20: the misses, the io, `agreed_overflow` and `cells_leased_peak` per
  request, the word clock's layers sum, slowest layer and boundary, and `drained_passes`
  with `drain_ms`; the miss-window rows of v15 to T3.1 read n/a since the fixup rides in
  the layer's command.
- `tools/turn-rig.sh` with `tools/turn-summary.py`: the turn's shapes (a pair, a suffix,
  the multi-turn chain), launched as production like the decode rig.
- `tools/expert-pool-replay.py`: a route trace replayed against the pool's policy and
  the ring's fills; trustworthy for misses, blind to milliseconds (v16). It keeps the
  fill-mode controls, including no fills, that the runtime no longer has, and since v20
  step zero the table mode (`--slots-json`, SLRU, the per-source fill budgets, the `q` and
  `t` line kinds) that priced the pool's allocation and the predictor the chapter did not
  build.
- `tools/prefetch-coverage.py`: a prefetch trace's predictions priced offline against a
  route trace, the rankings past eight and `--distance 1|2|3` (v20 S0.5b).
- `tools/prefill-ledger.py`, `tools/parse-kernel-stats.py`, `tools/parse-runner-stats.py`:
  the runner and kernel stats lines read into ledgers. The kernel line records one
  `token` row per token (the layers and the boundary are its encoders) and the boundary
  between tokens as `gap token->token`, the row v20 T3.3 took from 0.26 to 0.34 ms per
  token to 0.033 to 0.038; the word clock prints under it as `Shrike word_clock`. The
  runner line counts the two word wakes' fallbacks as `path_router_wake_fallbacks` and
  `boundary_wake_fallbacks`, the pool victims taken on the path as `agreed_overflow`, the
  ring's most cells leased as `cells_leased_peak`, and the passes committed ahead of a
  stop and waited out as `drained_passes` and `drain_ms`, charged to the line after the
  submission that waited them out; `cb2_ms`, `io_hidden_pct`, `io_fixup_wake_ms`,
  `path_pin_ms`, `path_fixup_build_ms`, `path_fixup_commit_to_kernel_ms`,
  `io_host_waits_avoided` and `path_router_wake_ms` went with the paths they measured.
- `tools/mini-deploy.sh`: the one release binary and its bundles to the mini, anything
  else in `bin/` removed, optionally a restart at the production launch. That launch is
  `tools/mini-production.sh`, shared with both rigs' `restore`, because every copy had
  drifted from CLAUDE.md's line since v20 T1: the deploy's, found at v24's close, and
  the rigs', found in the review after it.

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
- v19 ([v19-scan-rewrite.md](v19-scan-rewrite.md)): the scan rewrite. Step zero's
  ladder on the mini, through the new `ShrikeAttnBench`, named the shipped scan's
  constraint as its loop form, not memory; the loop form fixed (class 1, 2.7× on the
  kernel, the slope 2.23 to 0.72 ms per 1,000, the 7k token 73.0 to 61.7 ms); the
  forced-token instrument built once (`--force-tokens`, `--dump-logits`,
  `tools/logit-compare.py`, the logits-head golden profiles on both boxes); the
  streaming scan (class 2 under the gate, the slope to 0.39, the 7k token to 58.6 ms and
  17.05 tok/s); the hardening (the load-time refusal, the v11 simdgroup variant
  retired, the fp64 arm at 7 to 9e-8 for both kernels). The 7k token +25 % over the
  chapter, the card +6 %, the 300 and the 1k inside the noise; one Metal kernel added
  and one retired, 66 in the tree; no env knob added; the 7k rig shape; the lesson that
  a public struct's layout change wants a clean build before a crash is believed.
- v20 ([v20-ssd-mechanism.md](v20-ssd-mechanism.md)): the SSD mechanism. Step zero
  priced the board by replay and two captures on the mini: the pool's split is the
  lever, the token-id table small and not built, the width closed at distance one and,
  at distance two, by a number (recall 0.40 to 0.45 at precision 0.08 to 0.09), the
  attention row's fixed part to a chapter of its own. Task 1 the pool's allocation: a
  per-layer slot table from the production miss profile and segmented LRU, +4.4 to
  +7.8 % tok/s on the four shapes, the misses per token 19 to 20 to 14 to 17, both as
  the production launch's two variables (the known names 13 to 15). Task 3 the fold, in
  three commits: the agreed cells (the fixup encoded with the layer behind the event
  wait over a host-named cell row, the plan off the path; flat, the io 0.3 to 0.5 ms
  lower), one command per token (the forty command boundaries gone, the drain invariant
  with a deadline and a failed command naming its encoder, the word clock in place of
  the per-layer GPU rows; 0.2 to 0.6 ms faster on most rows) and the commit ahead of
  the sampler under Shape B (the GDN state in two parities, the pass ahead released at
  the stop and drained at the next submission, the boundary gap 0.26 to 0.34 ms per
  token to 0.033 to 0.038, the drain's wait 0.000 ms; the two-turn continuation gate as
  the golden's fifth profile). No Metal kernel added or retired, 66 in the tree; the
  golden byte-identical at every commit on both boxes; production on the mini 17.0 to
  17.4 tok/s to 18.2 to 18.9 over the chapter.
- Q3 ([v18-avenues.md](v18-avenues.md), A9): the MTP drafter's hidden state through the
  forty main routers as the next pass's routes, measured and closed 2026-09-18: the
  drafter replayed in fp32 over the four shapes (the head 64 of 64, the token guess 82
  to 86 %), its vector through each layer's post-norm and router overlaps the real
  top-8 at 0.10 to 0.14, the same as the main model's own true final residual does; the
  routers read their own layer's features. The `--dump-hidden` instrument stays.
- v21 ([v21-compression.md](v21-compression.md)): lossless compression, closed at step
  zero the day it opened. The palette per group closed (every group uses all sixteen
  levels), the aux table 11 bits, the entropy code the only path and its per-lane
  framing 4.7 % of a row; the drive scales a read with its size (0.16 ms fixed, 0.35 ms
  per MB); the pool's misses fall twice as fast as its capacity grows (22 to 29 % at a
  0.88 stride); and the gate: the production phase-1 kernel at the roof on the mini
  (60.5 to 63.0 GB/s), the in-lane decoder 5.3 to 6.9× slower at bit-identical
  arithmetic, the bytes 4 % of the stride rather than 11. No runtime code changed;
  `ShrikeExpertBench` stays as the instrument.
- v22 ([v22-pool-capacity.md](v22-pool-capacity.md)): the pool's capacity. The RAM
  ledger of the mini under load found 0.6 GB of prefill scratch held for the process's
  life and 2.9 GB in an idle oMLX; the replay priced a slot at twice its share in misses;
  v16's device limit stood in the way. Task 1 the prefill scratch released after every
  prefill; Task 2 the arena in chunks under `maxBufferLength` with `PoolBases` for the
  two speculative kernels, cells numbered globally, the allowed counts to 256; Task 3
  oMLX restarted empty and the mini at 160 slots per layer: +9.9 to +15.6 % tok/s and
  40 to 52 % fewer misses on the four shapes (the io 10.5 to 13.0 → 5.8 to 6.8 ms per
  token, the token 54.6 to 56.3 → 48.5 to 50.8 ms), the golden byte-identical on both
  boxes, the mini's at two chunks.
- v23 ([v23-argument-parsing.md](v23-argument-parsing.md)): the argument surface. Five
  hand-rolled parsers, 1,212 lines of `switch` over `case "--flag":` beside a `usage`
  string wrapped by hand to 80 columns beside a `main.swift` that caught the parse error
  and picked an exit code, retired for `swift-argument-parser`. Each binary's argument
  type is now a `ParsableCommand` in a library target under a three-line
  `@main extension` shim; help is generated and wrapped to the terminal and `-h` works
  everywhere. `ShrikeRepack`'s four mutually exclusive mode flags became four
  subcommands (`install`, `import-snapshot`, `verify-install`, `discard-partial`), which
  deleted the mode-validation block, its silent `return 2`, and the per-mode guard chains
  that existed only to reject another mode's flags. Sentinels became types: `--top-k`'s
  `0`, `--prefill-chunk`'s `auto`, `--seed`'s hex-or-decimal across the two benches, and
  the three `WasSet` booleans that told "unset" from "typed". Conformances shared by more
  than one parser live in `ShrikeArgumentSupport`, since two modules conforming the same
  type collide the moment anything links both. Five drifts fixed, each now pinned by a
  test: the server's `--max-context` help described a contiguous range where the code
  enforces a seven-value set, so `50000` read as legal and was refused naming nothing;
  `ShrikeServer --bogus` reported a missing value because the value guard ran before the
  unknown-flag check; `--seed` took hex in one bench and decimal in the other; `-h`
  reached only three of the five; and Repack could exit 2 with nothing on stderr. Argv is
  a production contract, so the invocations that exist were pinned first, before any
  parser changed, and they still parse unedited. No runtime or kernel code changed.
- v24 ([v24-unified-cli.md](v24-unified-cli.md)): one shrike, one verb tree. The Mac app
  and its decode service went first as a leaf island (five targets, 12,248 lines), then
  the five remaining executables became one `shrike` over `ShrikeRootCore`, with
  generation reached bare through ArgumentParser's `defaultSubcommand` and `serve`,
  `repack` and `bench` as verbs. Three designs died on measured behaviour before that
  one: a root carrying a required option refuses to dispatch, a root's `validate()` runs
  for every subcommand, and a flag name shared between root and subcommand binds to the
  **root**, so `shrike serve --model X` reached serve with `model` nil and silently
  ignored six flags including the mini's launch line. The model resolves from
  `~/.shrike/config.json` instead of being named on every invocation, `ServerConfig`
  becoming `ShrikeConfig` in a new `ShrikeCatalog` target that both the server and the
  CLI depend on. The flag surface was trimmed against a three-rule test (a launch line,
  gate or rig that runs on the mini; naming what to operate on; a per-invocation
  product choice): **16 distinct flags across 19 slots**, leaving generate 21, serve 10,
  repack 5, bench 7, 36 in all. Where a flag was the only way to reach working code the
  argument went and the code stayed, as a defaulted parameter the call site stops
  passing, so the prompt cache's modes, the disk cache, the prefill chunk and reasoning
  retention are all still constructible and are now pinned by a test where none existed
  before. Two environment names left the registry with them, 15 to 13, one of them
  `SHRIKE_MODEL`, whose reader had died with the Mac app: `refuseUnknownEnvironment`
  reads the registry as an allow-list, so a stale entry is silently accepted rather than
  refused, which is the one way that tripwire can fail quietly. No runtime or kernel
  code changed, and the golden baseline was byte-identical on all five profiles at every
  task.
