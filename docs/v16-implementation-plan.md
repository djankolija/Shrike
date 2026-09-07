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

- [ ] **T1: how many adopted predictions complete before their layer's classifier
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
        for real if built. **Ruling pending.**
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

### Task 2: the landing in the pool's own slot (on the ruling)

- [ ] **T2: the ring's read lands in a victim slot of the target layer's slab at issue,
  the slot `loading` from the issue, `resident` at the read's completion from the storage
  thread, so the classifier can count the expert a hit.** The ring keeps its budget and
  placement (v15 Task 1) and its join (Task 2), but its slots become (layer, slot index)
  claimed through the target layer's streamer under its lock instead of nine separate
  buffers; the adoption becomes either a hit (the classifier saw it) or today's path
  without a blit (the slot is already the expert's; the plan reserves it as `loading`
  and the fixup computes it). A failed read returns the slot to `empty`. The knob:
  `SHRIKE_PREFETCH_LANDING` (`ring` | `slot`, the default `ring` until the arms; `slot`
  retires the blit's copy on that path), fail-closed, on the banner as `landing=`.
  Modelled prize: the adopted-only fixups times the race's fraction, no RAM, today's miss
  count on the replay (`~/.claude/handoffs/archive/shrike-v15-close/index-swap-step0/swap-replay-*`).

  **Steps.**
  - [x] Step 0 (zero code): the replay's pricing at 128 and 119 slots, the headroom under
        load. **DONE 2026-09-07** (the design doc's step zero, part 2).
  - [ ] Step 1 (tests RED first): the streamer claims a landing slot for a prediction
        (a victim by the same rule as a demand miss, `loading`, its expert published at
        the claim, not a victim for the same layer's plan, not a hit before the bytes
        land); a completed landing is `resident` and a hit at the next plan with no
        adoption; a landing still in flight at plan time is adopted as `loading` (the
        fixup path, no blit) or joined within the bound; a failed read frees the slot
        (`empty`) and counts `prefetch_failed`; the ring's slots carry (layer, slot) and
        `readyBuffers` becomes `readySlots`; the reclaim of a completed, unadopted
        landing leaves the expert resident (the LFU evicts it in its turn) and frees the
        ring entry; the knob's parse and the banner; the guard's `abandon` under the
        landing (nothing to release but the ring entry).
  - [ ] Step 2 (the code): the ring keyed by (layer, slot), the streamer's claim and the
        completion publish on the storage thread (the state word last), the plan's hit
        scan unchanged, `PrefetchAdoption.landed` beside the copy and the blit, the
        reader's destination the slab slot's pointer; the four gates.
  - [ ] Step 3 (numerics): golden IDENTICAL on both boxes and both profiles at `slot` and
        `ring`, and at `slot` under `speculative-validate` and `gpu-residency`.
  - [ ] Step 4 (the arms, mini): `ring` against `slot` at production's defaults, mirrored
        (prod, slot, slot, prod per shape, the follow-ups as controls); the readings: the
        adopted-only fixup layers per token, `prefetch_before_classify`, misses per token,
        the wall and tok/s.
  - [ ] Step 5 (the rule): real (both orders on all three shapes above the repeats' drift)
        and free (the controls unmoved, golden identical) flips the default to `slot`.
  - [ ] Step 6 (design doc, review).

## Candidates (not scheduled)

- **The exchange at plan time (variant C).** The blit's copy alone, 10 x 1.77 MB per token
  of GPU under the demand wait; a likely null. Priced only if Task 2 is ruled out and the
  blit's own GPU time is ever isolated in the kernel stats.
- **The reading layers themselves.** 12.6 to 13.7 per token at production's per-read
  cost; no lever named.

## Follow-ons (not scheduled)

- v15's follow-ons (`v15-implementation-plan.md`, "Follow-ons"): `prefetch_begin_ms`'s
  thread, the draft runner's prefetch, the ring's test coverage, the adoption reload
  counter.

## Close

- [ ] The full suite under ThreadSanitizer, a whole-branch review by a fresh reviewer, the
  fixes folded into their owning commits, the design doc's closing block, Davor's go, the
  fast-forward merge to `main` and the push.
