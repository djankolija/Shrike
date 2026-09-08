# v17: the consolidation

v16 ([v16-landing.md](v16-landing.md)) closed with the ring merged into the pool's address
space, the copy and blit adoption deleted, and production on the mini at 15.34 / 16.46 /
16.12 tok/s on the card / the 300 / the 1k answers, the wall flat within the repeats' drift
because the token's pace is the reading layers' SSD chain. Its closing block named the
next chapter on Davor's direction of 2026-09-07: the consolidation series, four steps that
each leave fewer states in the tree at no cost to the wall, before the reading layers are
taken on from a smaller tree.

Every number in this document is COUNTED from the tree at `e959d55` or MEASURED on the mini
in the chapter it cites, unless marked modelled. The step zero's maps (the knob inventory,
the residency writers, the stage maps) were taken from the tree on 2026-09-07 with zero
runtime code.

## The problem

Sixteen chapters of measured levers left the tree carrying every arm they measured. At
`e959d55`:

| what | count |
| --- | ---: |
| `SHRIKE_*` environment knobs read under `sources/` | 66 |
| of which retired names kept only to refuse their old value | 2 |
| of which fail open (an unrecognised value silently takes the default) | 28 |
| cases of the decode expert execution enum | 5 (production runs 1) |
| expert cache layouts | 2 (production runs `pool`) |
| expert readers | 3 (bounded pread, the legacy cached pread, the Metal IO backend) |
| writers of the residency table | 14 |
| generation spaces landing in the table's one field | 2 |
| functions over 120 lines in the swiftlint baseline | 18 |
| stale knob names in `docs/` with no reader | 4 |

Each state has a price the chapters kept paying: a golden cell per mode on both boxes, a
reviewer's attention on branches production never takes, the ThreadSanitizer run's 41
minutes at v16's close, and a launch that silently takes a default when a stale script names
a knob the tree no longer reads (the failure `CLAUDE.md` records for the `NVMAI_*` names).
The chapter's object is the count in that table, driven down under one rule, with the wall
and the answers unchanged.

**The rule (Davor, 2026-09-07).** A knob is useful only if something actually uses it. A
knob that exists as a fallback because there was uncertainty during implementation belongs
in git history. The same rule reaches what a knob gates: a losing path deleted with its
knob, a retired subsystem deleted with its switch.

## Step zero (2026-09-07, zero runtime code): the three maps

### 1. The knobs

Sixty-six distinct names, read through `RuntimeConfiguration.environmentValue` and the same
static shape in the streamer, the sampler, the MTP decoder and the ANE sidecar. The
enum-valued knobs fail closed (an unknown value throws and the launch stops); twenty-eight
knobs fail open (an unrecognised value silently takes the default; the per-knob table in
`architecture.md` marks each). Two names are refused by name (`SHRIKE_PREFETCH_ADOPT`,
`SHRIKE_PREFETCH_JOIN_US=0`). The banner prints the modes in effect from
`RealForwardRunner.prefillGapLeversDescription`.

By family, with the measured status that decides each disposition (the per-knob table with
its read site, its users and its citation is Task 1's deliverable in
[architecture.md](architecture.md)):

| family | knobs | measured status | disposition |
| --- | ---: | --- | --- |
| decode execution: `SHRIKE_DECODE_EXPERT_EXECUTION`, `SHRIKE_SPEC_PHASE1`, `SHRIKE_HOST_WAIT`, `SHRIKE_ROUTER_WAKE`, `SHRIKE_EXPERT_IO_SYNC`, `SHRIKE_EXPERT_IO_SUBMISSION`, `SHRIKE_RDADVISE_POLICY` | 7 | v10 T5 made speculative, event, immediate and spin the defaults (f169f51); v14 lever B made the word wake the default and lever A (`hits`) a null; the rdadvise stage runs only under deferred submission | delete all seven; the enum becomes one case, the losing paths go, the rdadvise stage with its adaptive state goes |
| the streamer: `SHRIKE_EXPERT_CACHE_LAYOUT`, `SHRIKE_EXPERT_IO_BACKEND`, `SHRIKE_BOUNDED_IO`, `SHRIKE_PARALLEL_IO`, `SHRIKE_EXPERT_IO_THREADS`, `SHRIKE_EXPERT_IO_BATCH_DEPTH`, `SHRIKE_EXPERT_CACHE_POLICY`, `SHRIKE_EXPERT_CACHE_PROTECT`, `SHRIKE_NO_PIN` | 9 | per-slot is v9's measured loss; the Metal IO backend lost its A/B on 2026-09-01 (rig wait 43.29 sd 9.2 % against pread's 38.68 sd 2.2 %, and the server died mid-prefill; `v10-implementation-plan.md`); four threads the knee and two batches v13's winner; aging-LFU and chunk protection v13's defaults | delete all nine; one layout, one reader, the thread and batch counts constants, one policy, protection always on |
| the prefetch: `SHRIKE_PREDICTIVE_PREFETCH`, `SHRIKE_PREFETCH_TOP_M`, `SHRIKE_PREFETCH_INFLIGHT`, `SHRIKE_PREFETCH_PROBE_DISTANCE`, `SHRIKE_PREFETCH_JOIN_US`, `SHRIKE_PREFETCH_PLACEMENT`, `SHRIKE_PREFETCH_PROBE`, `SHRIKE_PREFETCH_ADOPT` | 8 | the ring won in v15 and v16 (the off control 14.0 / 14.5 / 14.6 tok/s against 15.3 / 16.5 / 16.1, misses 30.5 / 30.2 / 28.1 against 20.0 / 20.0 / 18.8; the replay reproduces the control to the tenth); placement after and the fused probe v15's winners; distance above one closed by the recall curve (`architecture.md`, 2026-08-31); the adopt knob already refused by name | delete all eight; top-m, in-flight, distance and the join bound become constants at their measured values (top-k, 1, 1, 400 us), the reclaim's distance window goes with the distance |
| prefill and the kernels: `SHRIKE_ATTN_MATRIX_TILE`, `SHRIKE_MPP_TILE_N`, `SHRIKE_MPP_TILE_K`, `SHRIKE_MPP_DEQUANT_BUFFERS`, `SHRIKE_MPP_WEIGHT_LOADS`, `SHRIKE_PREFILL_ATTENTION`, `SHRIKE_PREFILL_ROUTER`, `SHRIKE_PREFILL_ROUTER_TOKENS`, `SHRIKE_PREFILL_ROUTED_GEMM`, `SHRIKE_PREFILL_ROUTE_OVERLAP`, `SHRIKE_PREFILL_POOL_RESIDENCY`, `SHRIKE_PREFILL_TAIL_TILE`, `SHRIKE_PREFILL_TILE_BATCH`, `SHRIKE_PREFILL_TILE_DEPTH`, `SHRIKE_PREFILL_FETCH_DEPTH`, `SHRIKE_PREFILL_MATRIX_MIN_ROWS`, `SHRIKE_PREFILL_SWEEP`, `SHRIKE_PREFILL_SWEEP_TAIL`, `SHRIKE_GDN_PREFILL_SCAN`, `SHRIKE_ATTN_FULL_CHUNKS`, `SHRIKE_SAMPLER_PATH` | 21 | each default is the winner of a v12 or v13 arm (the matrix path, the grouped routed GEMM, the resident sweep, fetch depth 2, matrix min rows 16, the tiled sampler); the losing variants are kernels with reference suites | delete all twenty-one; the default's value becomes a constant; where the knob selected a code path, the losing path and its reference tests go |
| `SHRIKE_PREFILL_ANE` | 1 | off by default, an open candidate with its own record ([ane-prefill.md](ane-prefill.md)) | stays: the one switch the rule keeps, an A/B for a lever still open |
| MTP: `SHRIKE_MTP_VERIFY`, `SHRIKE_MTP_EXPERT_SLOTS` | 2 | speculative decode retired at v12's P17 (rig acceptance 20.6 %, the verify pass at width 2) | delete with the subsystem: the draft runner, the verify pair, the sidecar load, the server's prompt-cache forcing |
| product configuration: `SHRIKE_THINKING_MODE`, `SHRIKE_REASONING_EFFORT`, `SHRIKE_REASONING_RETENTION`, `SHRIKE_STRIP_CLI_PROMPT`, `SHRIKE_STRIP_TAGS`, `SHRIKE_CONCISE_MODE`, `SHRIKE_TOKENIZER_DIR`, `SHRIKE_MODEL`, `SHRIKE_EXPERT_CACHE_SLOTS` | 9 | selects product behaviour per launch, not an implementation fallback | eight stay; `SHRIKE_EXPERT_CACHE_SLOTS` goes, `--expert-cache-slots` already carries it |
| diagnostics: `SHRIKE_RUNNER_STATS`, `SHRIKE_KERNEL_STATS`, `SHRIKE_ROUTE_TRACE`, `SHRIKE_PREFETCH_TRACE`, `SHRIKE_LAYER_TRACE`, `SHRIKE_GPU_CAPTURE_DIR`, `SHRIKE_CACHE_DIAG`, `SHRIKE_GEN_DIAG`, `SHRIKE_PHASES` | 9 | the first four are read by `tools/decode-rig.sh`, `tools/turn-rig.sh`, `tools/expert-pool-replay.py`, `tools/prefetch-coverage.py` and the parsers; the other five have no reader outside `docs/` | four stay; five go |

Modelled from the table: 66 knobs become 13 (eight product, four diagnostic, the ANE
switch). Task 1's per-knob table confirms or corrects each row with its citation before
Task 2 deletes anything.

The two refused names are replaced by one tripwire: any `SHRIKE_*` variable in the
environment outside the surviving set fails the launch by name. A stale script can then
never silently take defaults; the failure is loud on the mini and in the rig.

### 2. The residency writers

One table per layer, `ExpertResidencyEntry { slot, state, generation }` at 16 bytes,
allocated shared in `PreadExpertStreamer.init` and read by `moe_classify_expert_residency`
(and its speculative twin) at buffer index 1. The hit test is `state == resident &&
slot != notResidentSlot`. Fourteen writers at `e959d55`, every one a single struct store
under the streamer's `cacheLock` through `writeResidencyEntryUnlocked`:

| # | writer | thread | trigger |
| ---: | --- | --- | --- |
| 1 | `init` | the streamers queue | the layer's first touch, the whole table `empty` |
| 2 to 5 | `loadExpertUnlocked` (evict, reserve, complete, fail) | the caller's | the round-robin single-expert load; no production caller, tests only |
| 6 | `makeExpertCachePlan`, the victim | the planner's (decode, prefill's union and tile planners, `loadExpertsCached`) | a miss needs a slot: the evicted expert `empty` at the slot's next generation |
| 7 | `makeExpertCachePlan`, the cell swap | the planner's | a leased landing is resident: the slot takes the landing's cell, republished `resident` at the slot's bumped generation |
| 8 | `makeExpertCachePlan`, the miss reservation | the planner's | `loading` at the slot's cell |
| 9 | `markPlanMissesResident` | the storage thread (or the Metal completion thread, or inline) | the demand read completed: `resident` at the plan's assigned generation, re-validated |
| 10 | `markStagedMetalPlanResident` | the runner's | the staged blit completed (Metal backend under event sync only) |
| 11 | `resetLoadingMissesUnlocked` | three callers: the failed read, the abandoned prefill plan, the failed staged plan | `loading` back to `empty` |
| 12 | `claimLanding` | the issuing thread (decode or storage) | a ring cell claimed: `loading` at the ring cell at the landing's own generation |
| 13 | `completeLanding` | the storage thread | the speculative read landed: `resident` at the ring cell |
| 14 | `dropLanding` / `failLanding` | the storage thread, the issuing thread, or the ring's reclaim | `empty`, only if the pool does not own the expert |

Three facts the map settled, which v16's design doc had by assumption:

- **The store is the publish.** There is no separate publish step, no memcpy and no
  encode-time copy: the host writes the shared buffer and the GPU reads it at its next
  dispatch (the kernel-boundary probe of v16 measured when).
- **The "state word written last" ordering is not enforced, and does not need to be.**
  The entry is stored as one 16-byte struct assignment; Swift does not order its fields.
  The hit test needs both `state == resident` and `slot != notResidentSlot`, and no
  transition changes the slot while the state stays resident (a swap keeps the cell, an
  eviction goes through `empty`), so a torn read of any writer's store reads as a miss,
  never as a hit at a wrong cell. A miss is always safe: the plan fails closed on it.
- **The generation the classifier writes back is consumed by nobody.** The kernel emits
  `resolved_generations`; neither the host nor the phase-1 and phase-2 kernels read it.
  The generation is host bookkeeping (a stale completion must not publish over a newer
  occupant) that happens to live in the GPU's buffer.

Two generation spaces land in the same field: the slot's (`slotGeneration`, bumped at every
plan) and the landing's (`landingGeneration`, a per-streamer counter), because a ring cell
belongs to no slot until the swap. The swap republishes the swapped-in expert at the slot's
generation to keep the eviction bookkeeping consistent, the v16 review's fold.

### 3. The long functions

The baseline's 18 entries, by stage count from the map (a stage is a sequential phase of
the body producing local state a later phase consumes):

| function | file | lines | stages | of which inline |
| --- | --- | ---: | ---: | ---: |
| `encodeDecodeRoutedMoE` | `RealForwardRunner.swift:6876` | 389 | 25 | 20 |
| `produceToken` | `RealForwardRunner.swift:3151` | 229 | 9 (the layer loop 7 sub-stages) | 8 |
| `executePrefillChunk` | `RealForwardRunner.swift:2681` | 225 | 12 | 8 |
| `encodeRoutedMoEVerifyPair` | `RealForwardRunner.swift:5338` | 214 | 12 | 7 (MTP: deleted in Task 2) |
| `encodeFullAttentionPrefill` | `RealForwardRunner.swift:4995` | 204 | 9 | 7 (no `self` mutation) |
| `ServerInference.generate` | `ServerInference.swift:1112` | 284 | 21 | 13 (60 lines are one counter snapshot) |
| `ServerInference.load` | `ServerInference.swift:642` | 189 | 13 | 9 |
| `ServerArguments.parse` | `ServerArguments.swift:140` | 241 | a flag switch and 40 lines of cross-flag validation | |
| `RemoteStreamingRepacker.runPrepared` | `RemoteStreamingRepacker.swift:197` | 231 | a sequential install pipeline | |
| `RawCompletion.runRawCompletion` | `RawCompletion.swift:89` | 180 | prefill then the decode loop | |
| `Entry.main` | `ShrikeDecodeService/Entry.swift:12` | 177 | a command switch with four cases | |
| `Run.run` | `ShrikeCLI/Run.swift:50` | 176 | the CLI driver | |
| `Args.parse` | `ShrikeCLI/Args.swift:139` | 170 | a flag switch and validation | |
| `PreadExpertStreamer.init` | `PreadExpertStreamer.swift:375` | 167 | resource acquisition, a two-way layout branch, a three-way reader branch (Task 2 deletes both branches) | |
| `Model.load` | `Model.swift:607` | 149 | a verify-then-map pipeline already sectioned by comments | |
| `RealInferenceClient.run` | `RealInferenceClient.swift:288` | 126 | one `do` with three `catch` arms | |
| `ShrikeBench.runMoE` | `ShrikeBench.swift:192` | 179 | a fixture then a kernel switch (Task 2 deletes the target) | |
| `ShrikeBench.runGDN` | `ShrikeBench.swift:407` | 175 | the same shape | |

The routed stage's 25 stages branch on seven mode knobs; Task 2 removes the three-way
classification, the rdadvise stage and three of the five arms of the I/O acquisition
before Task 4 decomposes what is left. The `lint:allow-long` marker some of these carry is
prose only: `.swiftlint.yml` reads no such marker.

### 4. What the chapter measures

Real is counted: the table in "The problem" before and after each task. Free is measured
on the mini: golden byte-identical on both boxes at the default (the only mode left), and
tok/s and misses per token on the three shapes within the repeats' drift of v16's close
(15.34 / 15.35, 16.46 / 16.48, 16.12 / 16.14 tok/s; 20.0 / 20.0 / 18.8 misses; drift 0 to
3.3 % across v14 to v16's lifetimes). A loss outside the drift is a defect of the task,
found and fixed, never a trade.

## The four subsystems the rule reaches

Named here because each is a whole subsystem, not a fallback; Davor's go of 2026-09-07
covers all four with the design as presented.

1. **MTP** (the draft runner `StreamingMTP`, `encodeRoutedMoEVerifyPair`, the sidecar load
   in `ServerInference.load` and `Model.load`, `--mtp-model-dir`, the prompt cache forced to
   single-prefix under MTP): retired at v12's P17 by measurement.
2. **ShrikeBench** (the target and its eight files): its MoE mode measures a dispatch
   production never uses (the hand-stuffed argument buffer, no `useResource`, no constants;
   Davor's note of 2026-08-30), and nothing in `tools/` runs it. A microbench the reading
   layers' chapter needs will be written against production's dispatch.
3. **The prefetch's off switch**: the lever won three times; the replay's no-fills row
   reproduces the off control to the tenth of a miss, so the control survives in the
   instrument, not in the binary.
4. **The Metal IO backend** (`MetalExpertReader`, the staging blit, `ExpertIOEventCoordinator`'s
   staging path, `markStagedMetalPlanResident`, `failStagedMetalPlan`): lost its A/B on
   2026-09-01 and killed the server.

## Tasks

### Task 1: the document

[architecture.md](architecture.md) rewritten in place from the tree at `e959d55`: the
decode path as the runner runs it (embed; per layer the held command, the speculative
lookahead a layer ahead, the word wake on the classifier's readback, the routed stage with
the plan, the landing's swap, the fixup and the deferred GPU records; the head), the
residency table and its writers (the list above), the arena and the ring, the demand path's
storage threads and the event-gated sync, prefill and the turn in summary with pointers to
[v12-prefill-matrix-kernels.md](v12-prefill-matrix-kernels.md) and
[v13-the-turn.md](v13-the-turn.md). Each surviving piece carries the number that keeps it
and the chapter it came from; each candidate for removal carries its measured status. The
four v4 invariants re-verified against the tree: the budget as input (still true), the
streaming figures re-cited from production's per-read cost (the 2.83 GB/s and 66 to 72 %
figures are rig-era), the round-trip section rewritten for the classifier and the fixup as
they stand, the C99 line count re-counted. The prefetch section rewritten from v16's state,
its history condensed to the two measured verdicts.

The document holds the three tables Tasks 2 to 4 execute from: the per-knob table (read
site, binaries, users, class, citation, disposition), the writer list, the stage maps.

Real: one document that contradicts the tree becomes none. Free: no code; the link check is
the gate.

### Task 2: the knobs

The dispositions of the family table applied, one commit per family (the decode modes, the
streamer, the prefetch, prefill and the kernels with MTP and ShrikeBench, the product and
diagnostic knobs with the tripwire). Per commit: the deleted knob's reader, its enum cases,
the losing path and its tests gone; the winner's value inlined where the knob carried a
number; the banner printing only what remains; the four gates; golden identical on both
boxes and both profiles at the default. The arms on the mini once at the task's end.

The tripwire: at launch, both binaries scan the environment for `SHRIKE_*` names outside
the surviving set and fail by name with the list. The test names every deleted knob and
expects the refusal; the refusal's message names the chapter that removed it.

Real: the count of knobs, enum cases, layouts, readers, backends and kernels. Free: by
construction for a path production never took, confirmed by the golden and the arms.

**Built (2026-09-08), seven commits on `refactor/v17-consolidation`, each with the four
gates and the local golden identical on both profiles:**

| commit | family | what went | files | lines | tests after |
| --- | --- | --- | ---: | ---: | ---: |
| 28fd4a6 | the decode modes | the five mode enums and their knobs; the hit-fixup, barrier, gpu-residency and speculative-validate arms; the plain classifier kernel and the hits-only phase 1; the parked wait and the status wake; host sync and deferred submission with the I/O acquisition's two host-wait arms (a fail-closed guard in their place); the standalone shared-expert command; five counters only those modes fed | 13 | +213 −710 | 1319 |
| eef39e1 | rdadvise | the policy engine dead at the only path: the policy, its adaptive state, the stage, the CLI flag, the app's option, picker, protocol fields and diagnostics rows, the runner line's three fields; the load-time warm kept | 29 | +70 −672 | 1308 |
| 9a70a1e | the streamer | one layout (the arena required), one reader (the bounded C pread at four threads and two batches), one policy (aging-LFU), chunk protection always, the pin always; the Metal IO backend with its staging and finalize paths, the legacy cached-pread path | 28 | +132 −1295 | 1299 |
| 3597ad1 | the prefetch | the seven knobs and `RuntimePrefetch`; the ring always built with the top-k plus one cells, one in flight, distance one, the 400 us join, the after placement, the fused probe; the trace path kept | 13 | +223 −430 | 1296 |
| 25ccdf7 | MTP and ShrikeBench | the runtime's speculative decode (the draft runner, the verify pair, the sidecar load, the two-row kernels, the GDN speculative checkpoint, the server's flags and session plumbing, the prompt-cache forcing, the tool); the bench target, four library helpers and the bench-only Metal variants; the format's MTP family kept and the roster's exclusion kept | 49 | +185 −5943 | 1281 |
| ceeed38 | prefill and the kernels | the twenty-one knobs to constants; the r32s4, r64s8, g4k128d and flash attention tiles, the tensor-ops 2D path, the block router, the per-expert routed GEMM with its gather and scatter, the four losing sweep orders and the carry plumbing, the MPP n32b2, n64b1 and n64b2 instantiations; the tiled attention and serial GDN kernels kept as the default's own fallbacks; the banner one line | 30 | +322 −3439 | 1221 |
| f373569 | diagnostics and the tripwire | the slot-count override and the five diagnostics without a reader; `refuseUnknownEnvironment` at every launch, 53 names refused by test, verified end to end on the CLI and the server | 10 | +187 −214 | 1223 |

The count, at `f373569` against `e959d55`:

| what | before | after |
| --- | ---: | ---: |
| `SHRIKE_*` knobs read under `sources/` | 66 | 13 |
| decode execution enum cases | 5 | 0 (one path) |
| expert cache layouts | 2 | 1 |
| expert readers | 3 | 1 |
| Metal kernels | 82 | 67 (65 after the close's fold) |
| source files | 250 | 235 |
| lines under `sources/`, `tests/`, `tools/` | | +1109 −12720 |
| `RealForwardRunner.swift` | 7326 lines | 5467 |
| `PreadExpertStreamer.swift` | 1716 lines | 1174 |
| swiftlint baseline entries | 18 | 14 |
| the serial suite | 1327 tests, 604 s | 1229 tests, 205 s |

Two things the deletions surfaced that the plan did not name: the prompt cache's runtime
identity lost two knob names (rdadvise, the policy), so persisted prompt-cache entries
re-key once after the deploy, a one-time miss with no numerics involved; and the GDN
delta-step kernels keep a checkpoint parameter whose only writer was MTP (a production
kernel signature, left for a decision of its own; the review's fold takes it out).

**The arms (2026-09-08, the mini at `f373569`'s build before the review's fold, deployed
at the bare launch; golden identical on both profiles there; two production lifetimes per
shape through the rig, beside v16's close read the same way):**

| shape | v16 close tok/s | v17 T2 tok/s | misses per token | landed hits per token | reading layers per token | answer |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| the card | 15.34 / 15.35 | 15.62 / 15.58 | 20.0 / 20.0 | 7.38, 7.23 / 7.14, 7.37 | 12.73, 12.70 / 12.74, 12.73 | identical |
| the 300 | 16.46 / 16.48 | 16.23 / 16.11 | 20.0 / 20.0 | 6.05, 5.91 / 6.06, 6.12 | 13.55, 13.55 / 13.53, 13.55 | identical |
| the 1k | 16.12 / 16.14 | 16.11 / 16.20 | 18.8 / 18.8 | 6.39, 6.44 / 6.33, 6.37 | 12.61, 12.59 / 12.57, 12.57 | identical |

Free: +1.6 / −1.9 / 0 % on the three shapes, inside the repeats' drift of 0 to 3.3 %, with
every answer identical to v16's, the misses to the tenth, and the ring's issued, adopted,
landed, joined and late counts, the race split and the fixup layers per token within the
noise of the two lifetimes. The turn rig's 300-token pair on the same build: the warm
second turn's wall 3.19 s (prefill 2.82 s) against v13's close at 3.54 s; the cold first
turn on the fresh server 7.69 s against v13's step zero at 8.9 s. Prefill is at or under
the record on the surviving kernels.

**The review (2026-09-08, a fresh reviewer over the seven commits against `e959d55`)**
found no surviving-path behaviour change: the 32 values of the pre-task banner all matched
their constant or sole remaining path, the routed stage, the streamer, the ring, the
attention gate and the grouped dispatch read identical under the defaults, and none of
the 346 surviving tests was left vacuous. Its findings and their disposition, folded into
the owning commits by autosquash: HIGH, two coverage losses (the kept read-advice
primitive's two tests restored as `RDAdviceCallTests`; the tiled router's only 4-bit,
weight-offset and padded-stride cases restored as one parameterised test against the
scalar reference); MEDIUM, the loader still validated the MTP sidecar's private tensors
(it now refuses the family before any tensor check, with a test), the GDN prefill
kernels' checkpoint parameter with no writer left (deleted, the golden the witness), the
MPP wide tile's 8-bit and irregular-scale coverage (two tests on the default instance),
`tools/prefill-ledger.py` parsing the deleted gen_diag line (the request line's completion
count instead), the replay's prose naming deleted knobs as settable (reworded), stale
comments in the streamer, the reader and the prefill (trimmed), two commit messages
(reworded); LOW, an error case named for the deleted tensor-ops path (`matrixPathUnavailable`),
the prefetch error's description, a benchmark-era doc line. The fold's tree: the release
build clean, swiftlint's 14 entries unchanged, 1229 tests, golden identical on both boxes,
the mini at the final build with the pair at 3.17 s warm.

### Task 3: one residency publish path

After Task 2 the writers shrink by the test-only round-robin load (writers 2 to 5 go with
`Model.routedExpert(layer:expert:)`, the tests moved to the plan path), the staged Metal
publish and its failure clear (10 and one caller of 11). The design for the rest:

- One state machine per arena cell (`empty`, `loading`, `resident`) with one generation
  counter per cell, in place of the slot's and the landing's two spaces. A landing bumps
  its cell's generation; the swap moves the cell and its generation under the slot with no
  republish; the plan's re-validation at completion compares the cell's generation.
- The table entry shrinks to what the GPU reads: `{ slot: UInt32, state: UInt32 }`, 8
  bytes, written as one 64-bit atomic release store so the pair is never torn. The
  generation stays host-side. The classifier's `resolved_generations` output and its buffer
  go.
- One function publishes: `publish(expert:cell:state:)` under the lock, called by the plan
  (victim, swap, reservation), the completion (demand and landing), the reset and the
  drop. No caller writes the buffer directly.

Tests RED first on the transitions (a landing overtaken by the pool, a stale completion
against a bumped generation, a swap followed by an eviction, a torn-store witness on the
8-byte entry). The fail-closed cross-checks at the plan (the classifier's miss set against
the plan's) stay as the tripwires.

Real: the writers, the generation spaces, the entry's bytes. Free: golden by construction
(the same experts are computed), the arms confirm.

**Built (2026-09-08), one commit `29e152e` on `refactor/v17-consolidation`, tests RED
first, the four gates and the golden identical on both boxes:** the entry is
`{ slot: UInt32, state: UInt32 }`, eight bytes, and `PreadExpertStreamer.publish(expert:cell:state:)`
is the only writer: it packs the state above the slot and stores the word once through
`shrike_store_release_u64` into the table bound once as words; the init's fill, the plan's
victim and reservation, the demand completion, the reset, the claim, the landing's
completion and the drop are its eight call sites, and the swap stores nothing, since the
landing's `{C, resident}` already stands and only the host's bookkeeping moves. The
classifier takes the table as `device const ulong*` and unpacks each entry from one 64-bit
load, so the pair is torn-free on both sides by construction, not by the compiler's choice
for a two-`uint` struct; `ExpertResidencyGPU`, the `resolved_generations` output and its
buffer are gone. The generation lives on the arena, one word per cell, every value drawn
from one atomic clock so no two bumps ever coincide; the plan's victim bumps the cell under
the slot (the reservation takes that value), `claimLanding` bumps the ring cell, and `pin`,
`unpin`, `markPlanMissesResident`, `resetLoadingMissesUnlocked` and `completeLanding`
compare against the cell now under the slot. The round-robin load went whole:
`Model.routedExpert(layer:expert:)`, both `loadExpert` entry points,
`loadExpertUnlocked`, `readFull`, the descriptor the streamer held idle after it, and two
error cases; its tests moved to the plan path with their assertions kept.

The count, at `29e152e` against `2ab2de4`:

| what | before | after |
| --- | ---: | ---: |
| functions that write the residency table | 2 (`publishResidencyUnlocked`, `writeResidencyEntryUnlocked`) | 1 (`publish`) |
| writer sites | 13 (14 at `e959d55`) | 8 calls of the one function |
| generation spaces | 2 (the slot's, the landing's) | 1 (the cell's, one clock) |
| the entry | 16 bytes, a struct store | 8 bytes, one release store |
| the classifier's reads of an entry | a two-field struct | one `ulong` |
| the classifier's buffers | 17 | 16 |
| descriptors held idle per routed layer | 1 | 0 |
| `PreadExpertStreamer.swift` | 1174 lines | 1006 |
| `ExpertCellArena.swift` | 62 lines | 87 |
| lines under `sources/` and `tests/` | | +287 −332 |
| swiftlint baseline entries | 14 | 14 |
| the serial suite | 1229 tests, 205 s | 1234 tests, 201 s |

Two things the work surfaced. The design's "never torn" claim needs the reader's side as
much as the writer's: the store was one aligned word, but a two-`uint` struct read in MSL
is one load only by Apple's codegen, so the kernel now reads a `ulong` (the implementer's
concern, ruled a fix before the review). And per-cell counters that start at zero can
coincide across cells: after a swap put a different cell under a slot, a stale plan's
recorded generation could equal the new cell's by chance. The trace found it unreachable
on the surviving path (a loading or pinned slot is never a victim, and only the decode
planner swaps, planning and pinning on one thread), but the tripwire exists for a caller-
ordering bug, so the values come from one clock and the compare is exact again.

**The arms (2026-09-08, the mini at `29e152e`'s build, deployed at the bare launch;
golden identical on both profiles there; production lifetimes per shape through the rig,
beside Task 2's arms read the same way):**

| shape | v17 T2 tok/s | v17 T3 tok/s | misses per token | landed hits per token | reading layers per token | answer |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| the card | 15.62 / 15.58 | 15.32 / 15.02, then 15.59 / 15.61 | 20.0 / 20.0 | 7.14, 7.37 / 7.48, 7.24, 7.21, 6.99 | 12.74, 12.73 / 12.74, 12.77, 12.75, 12.74 | identical |
| the 300 | 16.23 / 16.11 | 16.08 / 16.44 | 20.0 / 20.0, 19.9 | 6.06, 6.12 / 5.89, 6.09 | 13.53, 13.55 / 13.55, 13.52 | identical |
| the 1k | 16.11 / 16.20 | 16.30 / 16.28 | 18.8 / 18.8 | 6.33, 6.37 / 6.43, 6.38 | 12.57, 12.57 / 12.58, 12.57 | identical |

Free: the 300 and the 1k flat to +1 %, the card's first two lifetimes at −1.9 and −3.6 %
against Task 2's pair with the wake and submit terms up 0.4 and 0.3 ms per token and the
second one's fetch up 1.4 ms, then two more lifetimes at 15.59 / 15.61 with every term at
Task 2's values: slow lifetimes right after the deploy and the golden cell, not a cost of
the change, which has no per-token work (one release store in place of a struct store,
one atomic add per bump, one load in place of two). Every answer identical to Task 2's,
the misses to the tenth, the ring's issued, adopted, landed and joined counts and the race
split within the noise. The turn rig's 300-token pair: the warm second turn 3.09 to 3.20 s
against Task 2's 3.17, the cold first turn 7.75 against 7.61.

**The review (2026-09-08, a fresh reviewer on the working tree before the commit)**
found the diff spec-compliant with the fourteen writer sites of `e959d55` accounted for
one by one, the six tests asserting what their names say, the torn-read argument closed
on both sides, the lock order (ring, then cache) verified at every ring call site, and
no Critical or Important finding. Its eight Minor findings and their disposition: six
folded before the commit (the descriptor closed right after `fstat`, the table bound
once instead of `assumingMemoryBound`, the clock, the lease's doc word, the error text
naming the cell, the stale-plan test pinned to its detail) and verified by a scoped
re-review with no new breakage; the C header's doc line kept by ruling (its sibling has
one); two pre-existing test-target warnings outside the diff deferred to the close.

### Task 4: the runner decomposed

Each function over 120 lines becomes a sequence of named stage calls over a small context
struct carrying the locals the stages share, in the order the stage maps give. No new
abstraction beyond the stages, no behaviour change, no reordering of commits or waits. The
seams the map names first: the dense-layer body in `produceToken`; the classification, the
hit-split encode and the I/O acquisition in the routed stage; the counter snapshot in the
server's `generate` (a convenience init the app's client already has); the post-loop
validation in both argument parsers; `Model.load`'s comment sections; the decode service's
one handler per command. One commit per file; the baseline regenerated and diffed against
HEAD's entry set at every commit until it is empty, then the file, the gate's `--baseline`
flag and `CLAUDE.md`'s gate text go.

Real: the baseline's entries, to zero. Free: golden per commit, the arms at the end.

**Built (2026-09-08), twelve commits on `refactor/v17-consolidation`, one per file, one
for the baseline and one for the fold's last marker (`4de6bd1`), each with the four gates
and the local golden identical on both profiles:** the fourteen functions became stage methods, every statement moved once in
its order and every commit, wait, event, counter, timing mark, log line, error text, lock
span, early exit and `defer` where it was, each move checked by the implementer as a line
multiset against the previous tree and the decode path's four by a step-scoped review that
walked the originals beside the stages.

| commit | file | function, body lines before to after | the stages |
| --- | --- | --- | --- |
| `89f5185` | `RealForwardRunner.swift` | `encodeDecodeRoutedMoE` 289 to 36; `produceToken` 220 to 68; `executePrefillChunk` 165 to 100; `encodeFullAttentionPrefill` 199 to 96 | the routed stage's ten over `DecodeRoutedLayerContext` (the readback, the join, the plan, the pin, the partition, the hit split, the I/O acquisition, the speculative hand-off, the fixup build, the pending hand-off); the dense layer, the routed layer, the head; the validation, the token buffer, the embed, the ANE probe, the close-out; the RoPE epilogue, the causal dispatch |
| `ac9bf94` | `ServerInference.swift` | `generate` 256 to 87; `load` 140 to 91 | the snapshot init, a `StreamingSink`, the decode, the structured finish, the cache settlement; the slots, the runner, the cache domain, the cache |
| `091db14` | `Model.swift` | `load` 148 to 68 | eight, one per comment section, the stats `inout` |
| `01199ef` | `RawCompletion.swift` | `runRawCompletion` 168 to 87 | the prefill, the decode loop |
| `79e693d` | `RealInferenceClient.swift` | `run` 126 to 101 | the prompt rendering, the cancellation diagnostics |
| `bc362b3` | `ShrikeCLI/Args.swift` | `parse` 162 to 5 | the flag loop, the validation, the construction over a parse context; two typed helpers |
| `2eaf680` | `ServerArguments.swift` | `parse` 218 to 4 | the loop, the switch, the validation, the construction; one emptiness helper |
| `8542586` | `ShrikeCLI/Run.swift` | `run` 148 to 74 | the arch resolution, the prompt, the runtime, the footer, the exits as a stage outcome |
| `c0367d3` | `ShrikeDecodeService/Entry.swift` | `main` 177 to 66 | the load and generate handlers |
| `ecc6613` | `RemoteStreamingRepacker.swift` | `runPrepared` 231 to 10 | the resume, the output reservation, the ranges, the finalize over `PreparedInstall` |
| `1184875` | `.swiftlint-baseline.json`, `CLAUDE.md` | | the empty file deleted, the gate `swiftlint lint --strict`, the gate text rewritten |

The count, at `4de6bd1` against `62edee3`:

| what | before | after |
| --- | ---: | ---: |
| swiftlint baseline entries | 14 | 0, the file gone |
| the longest function body | 289 lines | 110 (the server parser's switch, ten from the bar: the next two or three flags put it over, and the honest split then is by option group) |
| `lint:allow-long` doc paragraphs (prose only) | 18 | 0 |
| the ten files | 11830 lines | 12472 (+642: signatures, structs, calls, returns) |
| lines under `sources/` | | +1792 −1152 |
| the serial suite | 1234 tests, 201 s | 1234 tests, 201 s |

What the work settled that the plan left open: a stage beyond the named seams is right when the named ones leave a function over the bar or an exit belongs with its block (the server's structured finish and cache settlement, the CLI driver's arch resolution, the repacker's output reservation, the server parser's switch); a seam the code no longer has is dropped (`selectProducer`, after Task 2's MTP deletion); the swap between a context struct and plain parameters follows whether the stages hand state forward (the routed stage, the parsers, the repacker) or only share inputs (the token producer, the prefill, the loader, the raw completion, the decode service, whose load case's escaping task cannot capture an `inout` context); and Swift 6 shapes two seams, the server's sink built inside the decode stage because region isolation refuses to send non-Sendable closures whose capture is a parameter, and one `let` copy in the repacker's copy stage because an `inout` cannot be captured by an escaping `Sendable` closure.

**The arms (2026-09-08, the mini at Step 5's build before the review's fold, deployed at
the bare launch; the fold's commits differ from it by comment lines alone, so the code is
the final tree's and only the embedded line numbers moved; golden identical on both
profiles there; two production lifetimes per shape through the rig, beside Task 3's arms
read the same way):**

| shape | v17 T3 tok/s | v17 T4 tok/s | misses per token | landed hits per token | reading layers per token | answer |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| the card | 15.32 / 15.02 / 15.59 / 15.61 | 15.57 / 15.62 | 20.0 / 20.0 | 7.48, 7.24, 7.21, 6.99 / 7.25, 7.16 | 12.74, 12.77, 12.75, 12.74 / 12.74, 12.76 | identical |
| the 300 | 16.08 / 16.44 | 16.43 / 16.30 | 20.0, 19.9 / 19.9, 20.0 | 5.89, 6.09 / 6.19, 6.04 | 13.55, 13.52 / 13.52, 13.53 | identical |
| the 1k | 16.30 / 16.28 | 16.14 / 16.16 | 18.8 / 18.8 | 6.43, 6.38 / 6.40, 6.47 | 12.58, 12.57 / 12.59, 12.59 | identical |

Free: flat within the drift on all three shapes, the sixty-odd stage calls a token now
makes invisible against its 60 ms; every answer identical to Task 3's, the misses to the
tenth, the ring's counts and the reading layers per token within the noise. The turn rig's
300-token pair: the warm second turn 3.09 to 3.13 s against Task 3's 3.09 to 3.20, the cold
first turn 7.67 against 7.75.

**The reviews (2026-09-08).** The runner's commit had its own review before it landed
(order preserved in all four functions with no deviation, five Minor findings, two folded
before the commit, three carried to the task's review). The task's review by a fresh
reviewer over the eleven commits found the spec met (every function under the bar, every
plan stage present under its name or a narrower justified one, the five unplanned stages
each needed and faithful, the dropped seam right, the baseline gone and the gate text
accurate, no type beyond context structs, the plan's sink and return shapes) and the
order preserved with no deviation (the nine non-runner functions walked statement by
statement against the base, `produceToken` end to end, the routed stage's diff in full,
the parsers' typed helpers and the 51-field snapshot init exact folds); no Critical or
Important finding, five Minor, approved; the three carried minors judged fit to stay. The
fold, comment lines only, into the owning commits by fixup and autosquash: a one-line role
summary restored on the eight functions whose only doc line went with their length
paragraph (the runner's fold had kept one), the moved comment that said "the load below"
now naming `Model.load`, and the five length paragraphs still standing on functions that
were never long retired (the runner's four and the OpenAI models' one), since the gate
text now says decompose as you write. Kept by ruling: the raw completion's result as a
struct where three files used labelled tuples. The fold's tree: the release build clean,
the flagless lint clean, 1234 tests, the local golden identical on both profiles.

## Method

v16's, with the chapter's own gates: the four gates per code commit; golden identical on
both boxes and both profiles at the default; the turn rig's pair beside the golden at each
task's end (the golden is single-turn; a decode followed by a prefill is where v16's review
found its HIGH); the arms on the mini once per task (`tools/decode-rig.sh`, two production
lifetimes per shape); a fresh reviewer per task, the fixes folded into the owning commits by
amend; the full suite under ThreadSanitizer once at the close. The mini decides. Deploy
leave is per session and asked for before any deploy.

## Numerics policy

Nothing in the chapter changes which experts are computed, which kernels compute them at
the default, or the order of any reduction. Golden must stay byte-identical at every commit
on both boxes and both profiles; a mismatch is a defect of the commit, never a reason to
recapture. The one deliberate byte change is the residency entry's shrink in Task 3, which
the classifier reads and whose output (hit and miss sets) must be identical; the golden and
the plan's cross-checks are the witnesses.

## Out of scope

The reading layers' per-read cost (the next chapter); any new lever; the ANE prefill's
promotion or removal; the second base for cells beyond one buffer (v16's candidate); the
app targets' own structure beyond what a deleted path forces.

## Risks

- A deleted path load-bearing for a caller the golden never runs: the app targets
  (`ShrikeMac`, `ShrikeDecodeService`) link the library and reach every knob; the turn's
  second request; the CLI's messages-file path. Every deletion's callers are enumerated
  before the delete, the app targets build in the release gate, the turn rig runs per task.
- The swiftlint baseline's entries go stale whenever a long function's length changes
  (v16's lesson): regenerated per commit and diffed against HEAD's set; a new entry means
  decompose, not regenerate.
- A constant inlined at the wrong value: the banner at `e959d55` prints every mode in
  effect; the first commit of Task 2 records the bare launch's banner on both boxes, and
  every later banner must print the same values for what remains.
- The tripwire refusing a name the mini's launch or a tool sets: the surviving set is
  checked against `tools/*.sh`, `tools/mini-deploy.sh`'s launch line and the rig's
  `SERVER_ENV` before the tripwire lands.
- Task 3's atomic store: Swift has no 64-bit atomic store on a raw pointer without the
  `Synchronization` module or C; the C99 target already exists (`ShrikeKernelsC`) and takes
  a one-line release store, matching the acquire loads the readback already uses.
- The suite shrinks as modes and kernels go; a test deleted for a losing kernel must be
  the kernel's own reference test, never a test of the surviving path that happened to
  run under the knob.
