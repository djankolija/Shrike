# v17 implementation plan: the consolidation

The design record is [v17-consolidation.md](v17-consolidation.md). The checkboxes here are
the status of record. Every number is counted from the tree or measured on the mini unless
marked modelled.

## Ground rules

- Branch `refactor/v17-consolidation` off `main` at `e959d55` (v16 merged).
- The rule: a knob is useful only if something actually uses it; a fallback born of
  implementation uncertainty belongs in git history, and so does the path it gated.
- Each task: the four gates per code commit (release build with zero warnings, swiftlint
  strict against the baseline, the markdown link check, the serial suite); golden identical
  on both boxes and both profiles at the default; the turn rig's `pair 300` beside the golden
  at the task's end; the arms on the mini once per task (`tools/decode-rig.sh`, two
  production lifetimes per shape, the rows against v16's close); a fresh reviewer per task,
  the fixes folded into the owning commits by amend; the docs commit last. The full suite
  under ThreadSanitizer once at the close.
- Real is counted (the table in the design doc's "The problem", before and after each
  task). Free is golden byte-identical plus tok/s and misses per token within the repeats'
  drift of v16's close (15.34 / 15.35, 16.46 / 16.48, 16.12 / 16.14 tok/s; 20.0 / 20.0 /
  18.8 misses per token; drift 0 to 3.3 %). A loss outside the drift is a defect, fixed.
- The swiftlint baseline is regenerated per commit and its entry set diffed against
  HEAD's; a new entry means decompose, not regenerate.
- Never two model processes; the mini's 8081 is production, Turbo on 8080 is never touched;
  a deploy needs the session's leave.

## Tasks

### Task 1: the document

- [x] **T1: `docs/architecture.md` rewritten in place from the tree at `e959d55`.**
  **DONE 2026-09-07**: 504 lines from the tree, every piece with its keeping measurement,
  the four v4 invariants re-verified and all holding, the 66-row knob table with the
  13-name surviving set, the writer list, the stage maps; reviewed against the tree and
  the fixes folded.

  **Steps.**
  - [x] Step 0 (zero code): the three maps taken from the tree (the knob inventory with
        read sites, binaries, users and class; the fourteen residency writers with thread
        and trigger; the stage maps of the eighteen long functions), recorded in the design
        doc's step zero. **DONE 2026-09-07.**
  - [x] Step 1: the decode path written from `produceToken` and `encodeDecodeRoutedMoE`
        as they stand (the embed, the held command per layer, the speculative lookahead,
        the word wake, the routed stage's plan, swap, fixup and deferred records, the
        head), each piece with the measurement that keeps it and its chapter: the
        speculative command (v9), the event sync and immediate submission (v10 T5), the
        word wake (v14 lever B, +2.8 to +3.6 %), the fused probe and the placement gate
        (v15), the merge (v16, the classifier seeing 70 / 59 / 69 % of landed predictions).
        **DONE 2026-09-07** (the sections "The decode path, token by token" through "The
        dense layers"; the word wake's timing corrected against v14's text: the word lands
        0.063 ms after the router command's GPU end, the driver's mark 0.16 ms after).
  - [x] Step 2: the residency table and its writers (the design doc's list), the arena
        and the ring, the demand path (the four storage queues, the event coordinator, the
        bounded pread reader at four threads and two batches), with the three step-zero
        facts stated as invariants: the store is the publish; a torn entry reads as a miss;
        the generation is host bookkeeping. **DONE 2026-09-07** (the sections "Residency"
        and "The demand path", the lock order as a fourth invariant).
  - [x] Step 3: prefill and the turn in summary (the matrix path, the routed tiles, the
        resident sweep, the prompt cache's settle and rewrite), each with its pointer to
        the chapter doc and the number that keeps it. **DONE 2026-09-07** (the sections
        "Prefill and the turn" and "The serving layer").
  - [x] Step 4: the four v4 invariants re-verified: the budget as input (the flag, the
        slot derivation, the arena sized from it); the streaming section re-cited from
        production's per-read cost (v15 and v16's `fetch_ms` per token and the reading
        layers' 12.6 to 13.6 per token) with the rig-era rates retired; the round-trip
        section rewritten for the classifier; the C99 line count re-counted with `wc -l`
        on `sources/ShrikeKernelsC`. **DONE 2026-09-07**: all four still hold; the budget
        default is 8 GiB (`defaultExpertCacheBudgetBytes`) against a stale 1 GiB comment
        in `ServerInference.load` noted for Task 4; the C target is 577 lines of C in two
        files plus 173 of headers against v4's 439; the rig-era rates retired.
  - [x] Step 5: the per-knob table (name, read site, binaries, users, class, citation,
        disposition), every disposition of the design doc's family table confirmed or
        corrected against the cited chapter doc; the four stale names in `docs/` noted;
        the surviving set written out (modelled at 13) and checked against `tools/*.sh`,
        `tools/mini-deploy.sh`'s launch line and `tools/decode-rig.sh`'s `SERVER_ENV`.
        **DONE 2026-09-07**: 66 rows, the surviving set 13 names; the tools set only the
        four diagnostics that stay, the rig's `SERVER_ENV` examples name three knobs Task 2
        rewrites; `SHRIKE_IO_MAX_*` recorded as C compile-time bounds, not knobs.
  - [x] Step 6: the link check (gate 3), a fresh reviewer on the document against the
        tree (every claim a line reference or a citation), the fixes folded; committed.
        **DONE 2026-09-07.** The review (131 line references checked, 125 right; three
        HIGH, eight MEDIUM, eight LOW) and the disposition: HIGH, the "torn entry reads as
        a miss" argument had one uncovered transition, the reservation over an unleased
        resident landing storing `{slot', loading}` over `{cell, resident}` in one 16-byte
        store (the invariant rewritten: thirteen writers by the argument, that one by
        ordering, since no classifier reads the layer's table between the plan and the
        next token; Task 3's single store removes it); `tools/mini-deploy.sh` does not set
        the route trace (corrected); the fourth planner is `Model.fetchRoutedExperts`, not
        a `Model.loadExpertsCached` (corrected). MEDIUM: three declarations cited at a line
        inside their body (`encodeDecodeTailStage` 3639, `makeExpertCachePlan` 707,
        `beginExpertCachePlan` 890, corrected); the recall curve's numbers had no record
        left once the old text went (restored in the ring section); the fused probe's
        figure cited from v15's rows; writer 1 writes before any lock and writers 2 to 11
        go through `publishResidencyUnlocked` (corrected); the split-read null cited to
        the v10 plan; "fails open" marked by one stated rule (28 of 66, the design doc's
        15 corrected by amend into the opening commit). LOW: two more line references,
        the readback's 32-bit words and its hit positions, the softmax command in the held
        set, the one test that reads `resolved_generations`, the `Attention.swift` path,
        the classifier overload's line, the top-m bound's site: all folded.

### Task 2: the knobs

- [x] **T2: 66 knobs to the surviving set, the losing paths deleted, one tripwire.** One
  commit per family; the order below so each commit shrinks the next one's surface.
  **DONE 2026-09-08**: seven commits, 66 knobs to 13, +1109 −12720 across 114 files, 19
  files gone, Metal kernels 82 to 67, the baseline 18 to 14, the suite 1327 to 1229 tests
  and 604 to 205 s; golden identical on both boxes at every commit; the arms on the mini
  flat within the drift with every answer identical; reviewed and folded.

  **Steps.**
  - [x] Step 1, the record: the bare launch's banner captured on both boxes at `e959d55`
        (the modes in effect) and kept in the ledger; every later commit's banner must
        print the same values for what remains. **DONE 2026-09-07**: the mini's banner
        from the archived v16 arms log at the final build (its own log had rotated; no
        restart taken); the CLI prints no banner, the server's line is the record.
  - [x] Step 2, the decode modes. **DONE 2026-09-08 as two commits**, 28fd4a6 (the five
        mode enums and their knobs with the losing code: one classifier kernel, the
        speculative command on every routed layer, the word wake and the spin the only
        waits, the I/O acquisition's host-wait arms gone with a fail-closed guard, the five
        counters only the deleted modes fed gone; 13 files, +213 −710; 1319 tests) and
        eef39e1 (the rdadvise engine, dead at the only remaining path: the policy, its
        state, the stage, the CLI flag, the app's option and picker, the protocol fields,
        the runner line's three fields; 29 files, +70 −672; 1308 tests); golden identical
        on both profiles at each; the baseline 18 entries throughout, the routed stage
        389 to 304 lines.
        As planned: `RuntimeDecodeExpertExecution` reduced to
        `speculative` and then removed as an enum (a mode with one value is not a mode):
        the `hitFixup`, `barrier`, `gpuResidency` and `speculativeValidate` arms in
        `encodeLayerCommands`, `encodeDecodeRoutedMoE`'s classification (the three-way
        becomes the speculative partition), `buildAndCommitMissFixupCommand`'s full-phase-1
        `else`, `crossCheckSpeculativeScratch` and the spec scratch's validate flag go.
        `RuntimeSpecPhase1Coverage` (`all-hit` only), `RuntimeRouterWake` (`word` only),
        `RuntimeExpertIOSynchronization` (`event` only), `RuntimeExpertIOSubmission`
        (`immediate` only) and `hostWaitSpin` (always) go the same way, each with its
        losing branch: the parked wait in `waitForRouterCompletion`, the status wake, the
        host-side I/O wait arms, the deferred `beginFetch`. The rdadvise stage
        (`shouldSkipRDAdvice`, `adviseRoutedExperts`, `recordRDAdvice`,
        `updateRDAdvicePolicy`, `rdadviseAdaptiveState`, `RDAdvicePolicyMode`) goes with
        deferred submission. Tests: `RuntimeConfigurationTests`' cases for the deleted
        knobs go; the speculative path's tests stay; the golden's speculative-validate and
        gpu-residency cells are no longer cells. Gates, golden, the banner diffed.
  - [x] Step 3, the streamer. **DONE 2026-09-08 as 9a70a1e**: one layout (the arena, the
        pool buffer required), one reader (the bounded C pread reader at four threads and
        two batches), one policy (aging-LFU, the lru arm gone), chunk protection always
        built, the pin always attempted; the Metal IO backend and the legacy cached-pread
        path deleted with their staging and finalize paths (three files gone); 28 files,
        +132 −1295; the baseline 18 to 17 (the streamer's init under 120 lines), none
        added; 1299 tests; golden identical on both profiles; CLAUDE.md's per-slot
        fallback sentence replaced.
        As planned: `SHRIKE_EXPERT_CACHE_LAYOUT` and the per-slot
        allocation branch in `PreadExpertStreamer.init` (`posix_memalign` per slot, the
        per-slot `cellIndexUnlocked` arm, the per-slot notice, the `prefetchCells` empty
        case) go; the arena is the layout. `SHRIKE_EXPERT_IO_BACKEND` and
        `MetalExpertReader`, the staging path of `ExpertIOEventCoordinator`,
        `markStagedMetalPlanResident`, `failStagedMetalPlan`,
        `Model.finalizeRoutedExpertStagingTransfer`, `requiresGPUFinalization` go; pread
        is the reader. `SHRIKE_BOUNDED_IO`, `SHRIKE_PARALLEL_IO` and the legacy cached
        pread path go; the bounded reader at `threads = 4`, `batches = 2` as constants
        (the C header's `SHRIKE_IO_MAX_*` bounds stay as the C reader's own limits).
        `SHRIKE_EXPERT_CACHE_POLICY` (aging-LFU only, the `lfu` and `lru` arms deleted),
        `SHRIKE_EXPERT_CACHE_PROTECT` (chunk protection always), `SHRIKE_NO_PIN` go. The
        tests of the deleted arms go; `PreadExpertStreamerTests+CachePlanning`'s knob
        cases become plain cases. Gates, golden.
  - [x] Step 4, the prefetch. **DONE 2026-09-08 as 3597ad1**: the seven knobs and
        `RuntimePrefetch` gone, the ring always built with the top-k plus one cells, one
        read in flight, distance one, the 400 us join, the after placement, the fused
        probe; the trace path the survivor; the ring's tests adapted to the constant
        budget, two deleted; 13 files, +223 −430; the baseline 17 throughout; 1296
        tests; golden identical on both profiles.
        As planned: `RuntimePrefetch` loses `enabled`, `topM`,
        `inflight`, `probeDistance`, `joinMicros`, `placement`, `probe`: the ring is
        always built (nine cells, top-k predictions, one in flight, distance one, the
        400 us join, placement after the demand submission, the fused probe); the
        separate probe's dispatch, the beside placement's issue point, the reclaim's
        distance window (`(issuing, target]` collapses to the next layer), the in-flight
        queue above one go. `SHRIKE_PREFETCH_ADOPT`'s and `SHRIKE_PREFETCH_JOIN_US=0`'s
        refusals go (the tripwire in Step 6 covers every deleted name).
        `tools/decode-rig.sh`'s `SERVER_ENV` examples updated. Tests: the ring's and the
        configuration's cases for the deleted knobs go. Gates, golden; the golden's
        prefetch-off cell is no longer a cell.
  - [ ] Step 5, prefill and the kernels, MTP and ShrikeBench, as two commits. **The first
        DONE 2026-09-08 as 25ccdf7**: MTP (the runtime's, 5943 lines with its kernels,
        scratch, checkpoint, server plumbing and tools; the format's family kept, the
        roster's exclusion kept) and ShrikeBench (the target, four library helpers only it
        used, the bench-only Metal variants); the baseline 17 to 14, none added; 1281
        tests; golden identical on both profiles. **The second DONE 2026-09-08 as
        ceeed38**: the twenty-one prefill and kernel knobs to constants, the losing tile,
        tensor-ops, block-router, per-expert GEMM, sweep and MPP variants deleted with
        their kernels and reference tests, the tiled attention and serial GDN kernels
        kept as the default's own fallbacks, the banner one line; 30 files, +322 −3439;
        the baseline 14 throughout; 1221 tests in 194 s (the suite's wall 604 s before);
        golden identical on both profiles.
        As planned (one commit, or two if the review wants the kernels apart). The twenty-one prefill knobs inlined at their
        defaults; where a knob selected a kernel variant, the losing variants
        (`SHRIKE_ATTN_MATRIX_TILE`'s six, the MPP tile family's, the block router, the
        per-expert routed GEMM, the tiled prefill attention path, the serial GDN scan, the
        generic sampler, the four losing sweep orders) and their reference tests go; where
        a knob only disabled a path the default uses (`SHRIKE_PREFILL_TAIL_TILE`,
        `SHRIKE_PREFILL_ROUTE_OVERLAP`, `SHRIKE_PREFILL_POOL_RESIDENCY`), only the knob
        goes. MTP: `StreamingMTP.swift`, `encodeRoutedMoEVerifyPair`, the sidecar load in
        `Model.load` and `ServerInference.load`, `--mtp-model-dir` and `--mtp-memory-mib`,
        the prompt cache's MTP forcing, `tools/prepare_ornith_mtp.py`, the MTP tests go.
        ShrikeBench: the target, `sources/ShrikeBench/`, its `Package.swift` product,
        `README.md`'s mention go. Gates, golden, and the prefill ledger's shapes (300 / 1k
        / 2k pairs through `tools/turn-rig.sh pair`) within v13's numbers.
  - [x] Step 6, product, diagnostics and the tripwire. **DONE 2026-09-08 as f373569**: the
        slot-count override and the five diagnostics without a reader gone; the tripwire
        at every launch (the server's main, the session's load, the CLI, the app client),
        the 53 deleted names refused by test and verified end to end on both binaries; the
        read set under `sources/` exactly the thirteen; 10 files, +187 −214; the baseline
        14 throughout; 1223 tests; golden identical on both profiles. The task's seven
        commits: 114 files, +1109 −12720, 19 files deleted, Metal kernels 82 to 67, the
        suite 1327 to 1229 tests and 604 to 205 s.
        As planned (one commit).
        `SHRIKE_EXPERT_CACHE_SLOTS` goes (`--expert-cache-slots` carries it);
        `SHRIKE_LAYER_TRACE`, `SHRIKE_GPU_CAPTURE_DIR` (and the capture window),
        `SHRIKE_CACHE_DIAG`, `SHRIKE_GEN_DIAG` (`ShrikeGenDiag`), `SHRIKE_PHASES` (the
        prefill phase counters and the CLI's footer) go. The tripwire:
        `RuntimeConfiguration.refuseUnknownEnvironment(_:)` scans the environment for
        `SHRIKE_*` names outside the surviving set and throws with the names, called first
        in `ServerInference.load` and `Run.run` (and the app's session start); a test
        passes every deleted name and expects the refusal, and passes the surviving set
        and expects none. The banner prints only what remains. The four stale names in
        `docs/` left as history. Gates, golden.
  - [x] Step 7: the arms on the mini (deploy leave asked first): the golden at the default
        on the mini's build, two production lifetimes per shape through the rig, the rows
        beside v16's close; the turn rig's `pair 300`. Real: the before-and-after table.
        Free: within the drift. **DONE 2026-09-08** (Davor's leave; `f373569`'s build
        before the review's fold deployed at the bare launch 00:55, golden identical on
        both profiles on the mini;
        the card 15.62 / 15.58 against 15.34 / 15.35, the 300 16.23 / 16.11 against
        16.46 / 16.48, the 1k 16.11 / 16.20 against 16.12 / 16.14 tok/s, every answer
        identical, misses 20.0 / 20.0 / 18.8 to the tenth, the ring's counts and the
        reading layers per token within the noise; the pair's warm 300-token turn 3.19 s
        against v13's 3.54, the cold 7.69 against v13's step zero at 8.9; production
        restored 01:02).
  - [x] Step 8: a fresh reviewer over the task's commits (the deletions' callers, the
        inlined constants against the banner of Step 1, the tripwire's set against the
        tools), the fixes folded, the docs commit (the design doc's Task 2 section, this
        task's boxes). **DONE 2026-09-08**: no surviving-path change found (32 banner
        values matched, the default paths identical at every site read, 0 vacuous tests);
        two HIGH coverage losses restored (the read-advice primitive's tests, the tiled
        router's 4-bit, offset and stride cases), the sidecar family refused at load with a
        test, the GDN checkpoint parameter deleted, the MPP wide tile's 8-bit and irregular
        coverage added, the prefill ledger's parse and the replay's prose fixed, stale
        comments trimmed, an error case renamed, two messages reworded; folded by
        autosquash (four conflicts by hand), the final tree's gates and golden clean on
        both boxes, 1229 tests.

### Task 3: one residency publish path

- [ ] **T3: one state machine per cell, one generation space, an 8-byte entry, one
  publish function.**

  **Steps.**
  - [ ] Step 1 (tests RED first): `ExpertResidencyEntry` as `{ slot: UInt32, state:
        UInt32 }` with `MemoryLayout.size == 8`; `PreadExpertStreamer.publish(expert:cell:
        state:)` the only writer; `cellGeneration` per arena cell replacing
        `slotGeneration` and `landingGeneration`. Tests: a landing overtaken by the pool
        is discarded at completion; a demand completion against a bumped cell generation
        throws; a swap followed by an eviction of the same slot publishes `empty` once at
        the cell; a stale completion after a drop publishes nothing; the whole-table read
        after each transition matches the expected entries. RED as a compile failure on
        the missing API.
  - [ ] Step 2: the entry's shrink through the Metal side: `ExpertResidencyGPU` in
        `moe.metal` to two fields, the `resolved_generations` output and its buffer
        removed from `moe_classify_expert_residency`, its spec twin and
        `MoE.encodeResidencyClassification`; the argument layout's stride 16 to 8.
  - [ ] Step 3: the atomic store: `shrike_store_release_u64` beside the existing
        `shrike_load_acquire_u32` in `ShrikeKernelsC`, the entry packed as
        `UInt64(state) << 32 | UInt64(slot)` and stored once; `writeResidencyEntryUnlocked`
        becomes the one call site.
  - [ ] Step 4: the generation unified: `ExpertCellArena` owns `cellGeneration`; the
        plan's victim, swap and reservation bump the cell's; `claimLanding` bumps the ring
        cell's; `markPlanMissesResident` and `completeLanding` validate against the cell's;
        the swap moves the cell under the slot with no republish; `assignedGenerations`
        keyed by cell. `Model.routedExpert(layer:expert:)` and `loadExpertUnlocked` go, their
        tests moved to the plan path.
  - [ ] Step 5: the targeted suites GREEN, the four gates, golden on both boxes and both
        profiles at the default, the turn rig's pair; the arms on the mini (deploy leave
        asked first). Free: within the drift, misses to the tenth.
  - [ ] Step 6: a fresh reviewer (the torn-read argument, the lock order ring then cache,
        every former writer's call site), the fixes folded, the docs commit.

### Task 4: the runner decomposed

- [ ] **T4: the baseline to zero entries, the file and the gate's flag with it.** One
  commit per file, in this order so the decode path's functions go first.

  **Steps.**
  - [ ] Step 1: `RealForwardRunner.swift`. `encodeDecodeRoutedMoE` over a
        `DecodeRoutedLayerContext` struct (the layer, the position, the command buffers,
        the readback, the plan, the partition, the timing marks) with the stages as private
        methods in the map's order: the router readback, the prefetch join, the plan and
        the consume, the pin and the submission, the partition, the hit-split encode and
        commit, the I/O acquisition (event-gated or all-hit), the speculative all-hit
        return, the fixup build, the pending hand-off. `produceToken`: the dense-layer body
        as `produceDenseLayer`, the head as `emitHead`, the per-layer routed sequence as
        `produceRoutedLayer`. `executePrefillChunk`: the validation, the token buffer, the
        embed-or-blit, the ANE probe, the close-out as methods. `encodeFullAttentionPrefill`:
        the RoPE epilogue and the causal attention dispatch as methods. Each commit: gates,
        golden, the baseline regenerated and diffed (the runner's entries gone, none
        added).
  - [ ] Step 2: `ServerInference.swift`. `RunnerCounterSnapshot.init(_ runner:)` (the
        app's client already has it) replaces the 60-line literal in `generate`; a
        `StreamingSink` holds the content, reasoning, calls, stop matcher and `publish`;
        `selectProducer` picks the runner. `load`: `resolveExpertCacheSlots`, `makeRunner`,
        `makePromptCacheDomain`, `makePromptCache`. Gates, golden.
  - [ ] Step 3: `PreadExpertStreamer.init` (already shrunk by Task 2's deletions;
        `allocateCells` and `makeReader` if still over), `Model.load` (one method per
        comment section, `ModelLoadStats` threaded), `RawCompletion` (`runPrefill` and
        `runDecodeLoop`), `RealInferenceClient.run` (`failureDiagnostics`,
        `renderPrompt`). Gates, golden.
  - [ ] Step 4: `Args.parse` and `ServerArguments.parse` (`validate()` for the post-loop
        block, typed value helpers), `Run.run` (`buildPrompt`, `buildRuntime`, the footer),
        `Entry.main` (`handleLoad`, `handleGenerate`), `RemoteStreamingRepacker.runPrepared`
        (`validateResume`, `copyRanges`, `finalizeInstall`). Gates.
  - [ ] Step 5: the baseline empty: `.swiftlint-baseline.json` deleted, the gate becomes
        `swiftlint lint --strict`, `CLAUDE.md`'s gate 2 text updated (the 18 functions'
        sentence gone). Gates, golden on both boxes, the turn rig's pair, the arms on the
        mini (deploy leave asked first).
  - [ ] Step 6: a fresh reviewer over the task (no behaviour change: every commit, wait
        and counter in the same order), the fixes folded, the docs commit.

## Candidates (not scheduled)

- **The reading layers themselves.** 12.6 to 13.6 per token at production's per-read cost;
  the next chapter, from this tree.
- **The ANE prefill's promotion or removal.** The one switch left; its own record decides.
- **A second base for cells beyond one buffer.** v16's candidate, unchanged.
- **The stale names in `docs/`** (`SHRIKE_ATTN_DECODE_LOOP`, `SHRIKE_EXPERT_IDLE_REFILL`,
  `SHRIKE_MPP_ROW_TILE`, `SHRIKE_MTP_REJECT_KEEP`): history in the chapter docs, left.

## Follow-ons (not scheduled)

- v16's follow-ons (`v16-implementation-plan.md`, "Follow-ons"), less those Task 2 deletes.

## Close

- [ ] The full suite under ThreadSanitizer, a whole-branch review by a fresh reviewer, the
  fixes folded into their owning commits, the design doc's closing block with the
  before-and-after table, Davor's go, the fast-forward merge to `main` and the push.
