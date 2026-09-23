# v16 implementation plan: the landing

The design record is [v16-landing.md](v16-landing.md). The checkboxes here are the status of
record. Every number is measured on the mini unless marked modelled.

## Ground rules

- Branch `perf/v16-landing` off `main` at `1647692` (v15 merged).
- Each task: host tests RED first, the four gates per code commit (release build with
  zero warnings, swiftlint strict against the baseline, the markdown link check, the
  serial suite), golden identical on both boxes and both profiles at every knob value, the
  mirrored arms on the mini for a lever, a fresh reviewer, the fixes folded into the owning
  commits, the docs commit last. The full suite under ThreadSanitizer once at the close.
- Every knob fail-closed (`RuntimeConfiguration.environmentValue`'s pattern), threaded into
  both binaries' two configurations (`ServerInference.swift`, `Run.swift`) and printed by
  the banner as the mode in effect.
- Never two model processes; the mini's 8081 is production, Turbo on 8080 is never touched;
  a deploy needs the session's leave.

## Tasks

### Task 1: the instrument (the race measured)

- [x] **T1: how many adopted predictions complete before their layer's classifier
  runs.** The prize of the landing is the adopted-only fixup layers (5.44 / 5.1 / 4.8 per
  token on the card / the 300 / the 1k, 2.7 to 3.5 ms per token modelled on the card) times
  the fraction of adopted predictions whose read completed before classify(L) executed.
  Nothing in the archived data measures it; the runner reads both timestamps.

  **Steps.**
  - [x] Step 0 (zero code): the step zero in the design doc (the trace, the replay, the
        headroom). **DONE 2026-09-07** (`~/.claude/handoffs/archive/shrike-v15-close/index-swap-step0/`).
  - [x] Step 1 (tests RED first): two runner-line counters, `prefetch_before_classify`
        (an adopted or joined prediction whose load operation's completion nanos precede
        the target layer's tail command's kernel start) and `prefetch_after_classify` (the
        rest), and `decode-rows.py` printing them beside `adopted`. The tail command's
        kernel start is what the runner already reads for the kernel stats; the
        prediction's completion is `ExpertLoadOperation`'s completion stamp. A test on the
        classification helper with synthetic stamps; a rows-tool row carrying the two.
        **DONE 2026-09-07:** `ExpertPrefetchRing.completionNanos(layer:experts:)` (the
        completed predictions' stamps only) and `RealForwardRunner.prefetchRaceSplit`,
        both RED before the code (the build failing on the missing members). **The first
        cut compared against the wrong event**: `kernelStartTime` is the driver's
        scheduling start (the runner's own "commit to kernel" reading), set at the top of
        the layer's iteration before the previous layer's demand read completes, so every
        prediction landed "after" by construction (0 of 4604 / 6353 / 7537 on the three
        shapes, archived as `shrike-v16-t1/race-cut1`). The split now stands against the
        tail command's GPU window: `before` its `gpuStartTime` (surely beat the classifier,
        the command's last kernel), `during` (between its start and end), `after` its
        `gpuEndTime`, `unknown` when the GPU times are not yet reported at the plan.
  - [x] Step 2 (the code): the helper and the counters, no knob (a counter changes
        nothing); the four gates; golden identical on both boxes (the default only; the
        counters do not touch the numerics path). **DONE 2026-09-07:** the four counters
        recorded at the plan for the adopted predictions (the completion stamp and the
        GPU times share `CLOCK_UPTIME_RAW`, the conversion the fixup's readings use); on
        the server's runner line as `prefetch_before_classify`, `prefetch_during_tail`,
        `prefetch_after_classify`, `prefetch_race_unknown` and in `decode-rows.py`; the
        four gates, golden identical at the default on both boxes. **The second cut
        counted at the plan and found the tail command's GPU times unreported there for
        about 91 % of adopted predictions** (the word wake fires on the classifier's
        readback word before the command is marked complete; archived as
        `shrike-v16-t1/race-cut2`), so the race now rides the deferred GPU records: a
        `prefetchRace` record with the tail command as its buffer, settled by the drain
        once the command has completed, and at once when the plan already finds it
        complete (the status wake).
  - [x] Step 3 (the measurement, mini): the three answers at production's default, two
        lifetimes each; the reading: `before_classify / (before + after)` per shape, and
        the same split for predictions issued at an all-hit layer's plan against those
        issued at a demand completion if the counters can tell them apart (a third
        counter if cheap). **DONE 2026-09-07, the fourth cut** (golden identical at the
        default, six lifetimes, every answer identical, 15.6 / 15.1, 16.5 / 16.5, 16.0 /
        16.4 tok/s; `~/.claude/handoffs/archive/shrike-v16-t1/race-cut4-summary.md`;
        the issue-point split not taken, the counters do not tell the placements apart):

        | shape | adopted per token | before | during: under 50 / 50 to 150 / over 150 µs before the end | after | surely won | likely won (before + over 150) | at most (+ 50 to 150) |
        | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
        | card | 10.4 | 326 | 613 / 613 / 1705 | 1297 | 0.072 | **0.446** | 0.581 |
        | the 300 | 10.1 | 464 | 1094 / 1092 / 1167 | 2553 | 0.073 | **0.256** | 0.427 |
        | the 1k | 9.2 | 514 | 996 / 1231 / 2335 | 2355 | 0.069 | **0.383** | 0.549 |

        (Counts over both lifetimes; the attention command, into which production folds
        the tail, is 0.4 to 0.5 ms wide with the classifier in its last tens of
        microseconds, so a completion more than 150 µs before its end beat the classifier
        and one in the last 50 µs did not; the middle bin is uncertain.) The earlier cuts
        are archived beside it: the first compared against the driver's scheduling start
        (every prediction after, by construction), the second counted at the plan (91 %
        unknown), the third settled every race but could not place the during bucket.
  - [x] Step 4 (the named stop): the fraction times the modelled prize is the landing's
        priced prize. **Davor rules** on Task 2 with that number: build, or record the
        landing as a measured null and close the chapter on the instrument alone.
        **REACHED 2026-09-07.** The landing's priced prize, MODELLED as the adopted-only
        fixup layers (5.4 / 5.1 / 4.8 per token) times the likely-won share times 0.5 to
        0.65 ms per such layer: **1.2 to 1.6 / 0.7 to 0.9 / 0.9 to 1.2 ms per token on the
        card / the 300 / the 1k, about +2 / +1.3 / +1.7 %**; at most (the middle bin won
        too) 1.6 to 2.0 / 1.1 to 1.4 / 1.3 to 1.7 ms, +2.5 to +3 / +1.8 to +2.2 / +2.1 to
        +2.7 %; the surely-won share alone 0.2 ms per token, negligible. The prize assumes
        an adopted-only layer has one adoption (10 adoptions over about 10 layers with
        any), no RAM, and the replay's finding that the landing keeps today's miss count.
        Against v15's levers (+2 to +6 % each) it is the smallest, and the arms decide it
        for real if built. **Ruling pending.** **Done: ruled 2026-09-07 to build it as the merge, `v16-landing.md:212-214` (noted 2026-09-23).**
  - [x] The review (2026-09-07, a fresh reviewer on the instrument): one MEDIUM, the
        swiftlint baseline had gained a new entry (the server's runner-line function
        crossed 120 lines with the seven new arguments and the baseline was regenerated
        without checking the entry was new); fixed by decomposition, the eighteen prefetch
        fields and their arguments in a helper spliced by one `%@`, the output byte for
        byte the same, the baseline back to main's 18 entries. Two LOWs folded: a
        prediction without a completion stamp now counts as unknown so the seven buckets
        total the adoptions; the joined predictions' place in "after" recorded in the
        design doc. A name nit left, a comment trimmed; the boundary tests and the ring's
        exclusion tests added.

### Task 2: the merge (the ring's cells in the pool's address space, on the ruling)

- [x] **T2: one address space for every expert cell, the ring's nine included, so a
  prefetched read lands where the classifier can already see it and is retained by an
  index swap instead of a copy.** Today a layer's pool is its own slab and Metal buffer,
  the ring nine standalone buffers beside every slab; a right prediction is adopted at
  the plan by a GPU blit into a victim slot and published resident when the fixup
  command completes, always after the classifier ran. Under the merge one allocation and
  one Metal buffer hold 40 x 128 cells owned by the layers and 9 owned by the ring, one
  stride; a layer's residency table names a GLOBAL cell, so the classifier and the
  kernels are unchanged (the same base, a wider index). Issue claims a free ring cell
  (no victim); the read lands there; the storage thread publishes `resident` for (layer,
  expert) at that cell; the layer's plan resolves its landings: wanted and classified a
  hit, or wanted and classified a miss (the lost race, the fixup computes it from the
  cell, `adopted`, no blit), the cell joins the layer and an evicted victim cell joins
  the ring; not wanted, the entry is emptied and the cell stays the ring's; still
  loading, joined (the wait now runs to completion or failure, `prefetch_late` counting
  the waits past the bound). Only the landings the ring leased to the plan are swapped,
  so the lease and the cell exchange are one transaction; the swap republishes the entry
  at the same cell under the slot's next generation (the review's fold). Deleted: the copy and blit adoption modes and
  `SHRIKE_PREFETCH_ADOPT` (set, it fails the launch by name), `PrefetchAdoptionTransfer`,
  the guard's transfer branch, `prefetch_blit_experts`, the nine standalone buffers. The
  prefetch requires the pool layout; under `per-slot` it is off and the banner says so.
  **No knob**: the merge's point is the alternative's deletion, so its A/B is build
  against build (T1's production rows, six lifetimes at c9b79cb, against the merge's at
  the bare launch, plus the merge with the prefetch off as the control) and its rollback
  is the previous deploy; the deviation from the chapter's knob rule is deliberate and
  recorded here. Step zero's pricing: retention is required (a non-retaining ring costs
  2.6 to 4.4 misses per token, the replay), the swap's miss profile is today's (19.9 /
  19.8 / 18.7 against 20.0 / 20.0 / 18.8 measured), the single buffer fits the mini's
  8.88 GiB limit at 8.45 GiB with 0.43 GiB of headroom, and the classifier sees a
  mid-command publish (the kernel-boundary probe, 100 % at every margin bin once the
  command has streamed 1 MB). Modelled prize: the adopted-only fixups times the race's
  "at most" share, +2.5 to +3 / +2 / +2.5 %.

  **Files.** New `Sources/Shrike/Infrastructure/Streaming/ExpertCellArena.swift` (one
  allocation, one buffer, `cellCount`, `stride`, `buffer`, `pointer(cell:)`,
  `offset(cell:)`, `cell(atOffset:)`); `PreadExpertStreamer.swift` (the `.pool` layout
  takes its cells from the arena; `slotBuffers`, `slotBufferOffsets`, `slotPointers`
  rewritten at a swap so every reader stays as it is; `publishResidencyUnlocked` writes
  the global cell, `slotBufferOffsets[slot] / poolSlotStride`; the landings:
  `claimLanding(expert:cell:)`, `completeLanding(expert:cell:)`, `failLanding`,
  `dropLanding`, the resolution inside `makeExpertCachePlan` given the GPU's missed
  experts, `ExpertCachePlan.freedCells`); `Model.swift` (the arena created once at the
  first layer's opening from the budget's slot count and the ring's cell count, the ring
  cells handed to the runner); `ExpertPrefetchRing.swift` (a slot carries a cell, not a
  buffer; `readyCells`, `consume(layer:experts:freedCells:)`, the join to completion, the
  reclaim's drop callback); `ModelExpertIO.swift` (`beginRoutedExpertPrefetch` by cells;
  `planRoutedExperts(..., gpuMissedExperts:)`; the adoption bridge and the finalize and
  fail hooks removed); `RealForwardRunner.swift` (the issue by cells, the plan with the
  readback's missed experts, the consume with the freed cells, the transfer and blit
  removed from the fixup build, `prefetch_landed_hits`); `RuntimeConfiguration.swift`
  (`RuntimePrefetchAdoption` and `SHRIKE_PREFETCH_ADOPT` removed, the name refused);
  `ServerInference.swift`, `Run.swift`, `decode-rows.py` (the counters); the tests below;
  `PrefetchAdoptionTransfer.swift`, `PrefetchAdoptionGuard.swift` and their tests deleted
  (the guard's remaining duty, the ring lease returned when the plan throws, is
  `ring.unlease`).

  **Steps.**
  - [x] Step 0 (zero code): the replay's pricing at 128 and 119 slots, the headroom under
        load. **DONE 2026-09-07** (the design doc's step zero, part 2). The kernel-boundary
        probe and the merge's pricing (retention by the replay, the device's buffer limit).
        **DONE 2026-09-07** (the design doc's step zero, part 3; the archive
        `~/.claude/handoffs/archive/shrike-v16-t2-step0/`).
  - [x] Step 1 (tests RED first). `ExpertCellArenaTests`: cells non-overlapping at the
        stride, one buffer, `offset(cell:)` and `cell(atOffset:)` inverse, the pointer at
        the offset. `PreadExpertStreamerTests+Landing`: a claimed landing publishes
        `loading` at its global cell and is neither a hit nor a victim for the plan; a
        completed landing publishes `resident` at the cell and the next plan hits it with
        the swap (the expert's plan buffer at the landing's cell, the victim's cell in
        `freedCells`, the victim's expert `empty`, the entry republished under the slot's
        next generation); a
        completed landing the GPU missed is `adopted`, not a miss, and swapped the same;
        an unwanted completed landing is dropped at the plan (`empty`, its cell still the
        ring's and absent from `freedCells`); a
        landing still loading at the plan is joined and then swapped; a failed landing
        is `empty` and counts nothing resident; a landing for an expert the pool already
        holds is discarded at completion; `dropLanding` publishes `empty`; every
        residency entry's slot is the global cell. `ExpertPrefetchRingTests`: a slot
        carries a cell; `consume` with `freedCells` moves the entry to the freed cell;
        `readyCells` waits to completion and counts past-bound waits as late; the reclaim
        calls the drop callback before a cell is reused. `RuntimeConfigurationTests`:
        `SHRIKE_PREFETCH_ADOPT` set is refused by name; the banner without `adopt=`.
        The two deleted suites removed. Run the touched suites: RED. **DONE 2026-09-07**
        (RED as a compile failure: the API absent, the deleted sources still referenced by
        the runner; the loader test's offset expectation, a property of the per-layer slab,
        rewritten to the arena's cell in Step 2).
  - [x] Step 2 (the code): the arena, the streamer, the ring, the model, the runner, the
        configuration, the counters, the deletions; the touched suites GREEN; the four
        gates (the swiftlint baseline must not gain an entry: decompose). **DONE 2026-09-07**
        (7652fb6; 121 targeted tests GREEN after two counting fixes, a joined batch counting
        every prediction it carried and the banner's second expectation; the release build
        clean, swiftlint strict clean with the baseline regenerated for two grown entries,
        the streamer's init and the decode routed function, 18 entries before and after, none
        added; 67 markdown files 0 broken links; the full suite 1322 tests GREEN).
  - [x] Step 3 (numerics): golden IDENTICAL on both boxes and both profiles at the
        default, with `SHRIKE_PREDICTIVE_PREFETCH=0`, under `speculative-validate` and
        under `gpu-residency`; the per-slot layout's golden with the prefetch refused.
        **DONE 2026-09-07** (IDENTICAL on both profiles: locally at the default, prefetch off,
        speculative-validate, gpu-residency and per-slot under hit-fixup; on the mini at the
        default and prefetch off).
  - [x] Step 4 (the arms, mini): the merge at the bare launch, two lifetimes per shape
        through the rig (`v16t2-prod-*`), against T1's cut 4 rows at c9b79cb; the merge
        with the prefetch off, one lifetime per shape, as the control; the readings: the
        adopted-only fixup layers per token (expected to fall by the race's share),
        `prefetch_landed_hits` and the race counters, misses per token (today's 20.0 /
        20.0 / 18.8), the wall and tok/s. **DONE 2026-09-07** (the design doc's Task 2 table:
        landed hits 7.3 / 6.0 / 6.4 of 10.6 / 10.2 / 9.4 adopted per token, the adopted-only
        fixup layers 5.4 / 5.1 / 4.7 to 1.7 / 2.2 / 1.5, misses unchanged, tok/s 15.34 / 15.35,
        16.46 / 16.48, 16.12 / 16.14 against T1's, every answer identical; the prefetch-off
        control 14.01 / 14.46 / 14.59 at the replay's no-fills misses).
  - [x] Step 5 (the rule): real (the adopted-only fixup layers fall on all three shapes,
        the misses per token at today's within 0.3, tok/s above T1's rows beyond the
        repeats' drift or within it with the fixup layers' fall as the mechanism's proof)
        and free (golden identical everywhere above) keeps the merge; a loss on any shape
        reverts by deploy and the chapter closes on the instrument and the step zero.
        **DONE 2026-09-07: the merge STAYS.** Real by the mechanism (the fixup layers' fall on
        all three shapes, misses at today's) and free (golden, no loss); the wall flat within
        the drift, the removed round trips paid under the drive (the design doc's readings).
  - [x] Step 6 (design doc, review). **DONE 2026-09-07** (the design doc's Task 2 section
        and After T2 block; two fresh reviewers on 7652fb6, a correctness reviewer and a
        silent-failure hunter, archived at `~/.claude/handoffs/archive/shrike-v16-t2/`: one
        HIGH, the swap ran for every planner of the layer while only the decode path returned
        the freed cell, so a prefill plan could leave the ring a cell the pool owned, folded by
        swapping only the landings the ring leased to the plan; MEDIUMs folded: the pool
        layout's unwind, the reclaim window at probe distances above one, JOIN_US=0 refused,
        the arena sized by the routed layers, refused claims no longer counted as adoptions,
        prefetch_failed; LOWs folded: the swap's generation and republish, landed hits from
        the plan's swaps, a pool-owned cell refused, consume dropping what it does not
        exchange, the comments; the whole-branch review verified every fold, found no code
        defect blocking the merge, and its three documentation errors and five naming
        cleanups were folded; every fold amended into 7652fb6, the four gates and the golden
        cells rerun on the final tree).

## Candidates (not scheduled)

- **The landing in the pool's own slot (variant B, Task 2 as first designed).** A victim
  claimed at issue for every prediction, wrong ones included; the replay prices its misses
  at the swap's (19.7 / 19.9 / 18.9 against 19.9 / 19.8 / 18.7). Superseded by the merge,
  which evicts only for a right prediction and deletes the copy; kept as the record. **Superseded: by the merge, 7652fb6 (noted 2026-09-23).**
- **The exchange at plan time (variant C).** The blit's copy alone; retired with the blit
  by the merge. **Superseded: by the merge, 7652fb6 (noted 2026-09-23).**
- **A second base for cells beyond one buffer.** The merge's single buffer fits the mini's
  8.88 GiB limit only up to about 128 slots at the production stride; a larger budget
  there needs the kernels given a second base and the cell index split across the two. **Done: 1dcf745, v22 Task 2, the expert cell arena in chunks (noted 2026-09-23).**
- **The consolidation series (a following chapter, Davor's direction of 2026-09-07).**
  After the merge: the architecture document the repo lacks (the decode path as it stands,
  each surviving piece annotated by the measurement that keeps it); the knobs pruned by
  measured status (losers and nulls deleted with git as their record, winners as defaults
  without a switch, an A/B only for a lever still open); one residency publish path; the
  runner decomposed into explicit stages (the baseline's eighteen long functions). Each
  step real (fewer states) and free (golden identical, the arms unmoved). **Done: as v17, [v17-consolidation.md](v17-consolidation.md) (noted 2026-09-23).**
- **The reading layers themselves.** 12.6 to 13.7 per token at production's per-read
  cost; no lever named. **Done: under other names, v18's surface A, then v20, v21 and v22 (noted 2026-09-23).**

## Follow-ons (not scheduled)

- v15's follow-ons (`v15-implementation-plan.md`, "Follow-ons"): `prefetch_begin_ms`'s
  thread, the draft runner's prefetch, the ring's test coverage, the adoption reload
  counter.

## Close

- [x] The full suite under ThreadSanitizer, a whole-branch review by a fresh reviewer, the
  fixes folded into their owning commits, the design doc's closing block, Davor's go, the
  fast-forward merge to `main` and the push.

  **DONE 2026-09-07**: the full suite under ThreadSanitizer on the final tree 1327 tests, zero
  reports, 41 minutes; the whole-branch review (archived at `~/.claude/handoffs/archive/shrike-v16-t2/branch-review.md`)
  verified every fold and found no code defect blocking the merge, its three documentation
  errors and five naming cleanups folded; the fixes amended into 7652fb6; the four gates and
  five golden cells on the final tree, the mini at the final build (245d62e5a09420bf) with
  golden identical at the default and with the prefetch off; the closing block written;
  Davor's go and the merge pending. **Done: merged to `main`, e959d55 (noted 2026-09-23).**