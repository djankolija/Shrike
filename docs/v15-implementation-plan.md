# v15 implementation plan: the miss window

The plan for [v15-miss-window.md](v15-miss-window.md). The per-task checkboxes below
are the status of record; the chapter's ledger of rulings and raw rows is
`.superpowers/sdd/v15-implementation-plan/progress.md` (gitignored), the raw data
under `~/.claude/handoffs/archive/shrike-v15-*/`. Every number is MEASURED on the mini
unless marked modelled.

**Where the chapter starts (2026-09-06).** Production on the mini: 14.1 / 14.8 / 15.0
tok/s on the card / 300 / 1k answers at the bare launch of `4cc6e58`. The miss window
is 18.0 to 19.7 ms per token, 26 to 27 % of the wall, over 17 to 19 missing layers per
token; the prefetch ring, its probe and its adoption are in the tree behind
`SHRIKE_PREDICTIVE_PREFETCH=1`, output-identical, and a measured loss because the
ring's reads are placed beside the demand reads. Step zero, the contention probe,
measured that a background read which completes before the demand read is issued
costs it nothing on this drive and that one still in flight shares the drive with it;
the drive has no idle penalty of its own (the 0.13 to 0.15 ms that v14's row 4 saw was
the probe's sleeping host). The design doc carries the tables and the placement rule.

## Global constraints

- The chapter's branch is `perf/v15-miss-window` off `main` at `4cc6e58` (v14 merged
  and pushed). The peer session's uncommitted edit to
  `docs/v10-implementation-plan.md` rides in the working tree and is never committed,
  stashed or touched here.
- macOS 26+, Swift 6.3+; never two model processes (`pgrep` first); the mini's server
  on 8081 is production and Turbo on 8080 is never touched. Deploy leave (kill the
  server, relaunch) was granted to session decode-pass-2-leg-3 on 2026-09-06 and is
  per session. `tools/mini-deploy.sh --restart` deploys; `memory_pressure -Q` before
  every launch.
- Four gates per code commit: release build with zero warnings; `swiftlint lint
  --strict --baseline .swiftlint-baseline.json`; the markdown link check; `swift test
  --no-parallel`. ThreadSanitizer on the full suite once at the chapter's close; a
  task whose own code is cross-thread (Task 1's completion hook) runs a filtered
  sanitizer pass in its own step.
- Numerics: every task here is a scheduling change, golden IDENTICAL on both boxes
  and both profiles (`tools/golden-baseline.sh --check`; on the mini the cell scripts
  in the session's `loop-files/`), a mismatch a defect, never a recapture. A change
  that cannot show bit-identical authoritative output follows v12's policy.
- Ledger protocol per task: the rig on the mini (`tools/decode-rig.sh`,
  `tools/decode-rows.py`, the arms driver in the shape of
  `~/.claude/handoffs/archive/shrike-v14-t2/step2-leverB/t2-leverB-arms.sh`), a
  verdict row a whole answer, the three answers (the card, the 300, the 1k at
  `max_tokens` 512, `temperature` 0), the turns and the warm second prompts the
  controls, paired runs in both orders, the drift band re-established per task on the
  same binary. **No size floor**: real (the sign holds in both orders above the
  drift) and free (no control regresses, golden identical).
- A knob is real only when the banner prints it, and both binaries build two
  `RuntimeConfiguration` values (load-time and generation-time): a knob is threaded
  into both (`ServerInference.swift:679-680` and `:736-737`; `Run.swift:136` and
  `:173`) and fails closed on a bad value.
- Comments: none unless a genuinely non-obvious why (repo rule).
- Every `file:line` in this plan was verified against `4cc6e58` on 2026-09-06 when
  the plan was written.

## Tasks

### Task 1: the placement gate on the ring in the tree

- [ ] **T1: the ring's reads are issued beside the demand reads, and the probe says
  that is the whole loss.** `begin` runs right after the demand fetch is submitted
  (`RealForwardRunner.swift:7011-7020`), the demand batch and the ring's batch are
  both work items on the shared four-worker scheduler whose priority orders the queue
  and nothing else (`PreadExpertStreamer.swift:914`, `:1327`;
  `ExpertLoadOperation.swift:163`, `:200-211`), and the ring submits up to `topM`
  reads at once (`ExpertPrefetchRing.swift:54-98`). Measured in v14: the layers that
  still read slow from 1.03 to 1.07 ms each to 1.15 to 1.23 at top-4 and 1.59 to 1.71
  at top-8 ([v14-decode.md](v14-decode.md) "Task 1"), the probe's `cont` row. The
  probe's `burst` row says a read issued at the demand batch's completion and kept to
  one per layer costs the next demand read nothing. **The task moves the issue point
  to the demand batch's completion, caps the reads in flight, and measures whether
  the reading layers return to production's per-read cost with the ring on.** That
  per-read reading is the verdict that decides Tasks 2 and 3; tok/s is the rule for
  the default. Modelled net at the current probe and copy: 0 to +2.5 %. **The mini
  decides**, and a measured null is a result.

  **The mechanism, verified in the tree** (against `4cc6e58`).
  - Every backend's demand batch ends in one `ExpertLoadOperation.finish`
    (`ExpertLoadOperation.swift:120-155`), which publishes the event, wakes the
    awaiters and is the single terminal transition; a completion hook registered
    there runs on the finishing thread (a scheduler worker or a reader callback)
    the moment the bytes land, which is exactly the `burst` placement, with no host
    wake. On an all-hit layer there is no batch (`PreadExpertStreamer.swift:852-858`
    finishes an empty plan at once) and `begin` runs at plan time as today.
  - The predicted ids arrive in top-k order (`RealForwardRunner.swift:3172-3186`) and
    `begin` keeps the first `count` after the dedupe (`ExpertPrefetchRing.swift:60-70`),
    so an in-flight budget keeps the best-scored predictions.
  - The probe's scales and bias are bound at `L + 1` while its weights come from
    `L + prefetchProbeDistance` (`RealForwardRunner.swift:3526`, `:3528` against
    `:2898-2905`): correct at distance one, wrong above it. The fix is two index
    expressions and is golden identical at distance one.
  - The prefetch knobs are read from the environment inside the runner
    (`RealForwardRunner.swift:933-948`, `:2055-2069`), absent from
    `RuntimeConfiguration` and from the banner (`:348-412`): not real by the
    chapter's standard.

  **Steps.**
  - [ ] Step 0 (zero code): the production per-read baseline from the archived v14
        arms' runner lines (`io_fetch_ms` per reading layer on the `prod` arms of
        `~/.claude/handoffs/archive/shrike-v14-t2/step2-leverB/`, and on T1's
        prefetch arms in `shrike-v14-t1/step2-fix/`), so the verdict has its two
        anchors before the first arm runs: what a reading layer costs with the ring
        off, and what it cost with the ring beside the demand reads.
  - [ ] Step 1 (tests RED first): `RuntimePrefetch` in `RuntimeConfiguration.swift`
        (enabled, `topM`, `inFlight`, `placement` in `after` / `beside`, `distance`,
        `trace`) with `environmentValue()` fail-closed on every bad value, production
        off, both binaries threaded; the banner's `prefetch=off` and `prefetch=on
        top_m=8 inflight=1 placement=after distance=1` forms with the existing
        expectations updated (`PrefillRoutedTileSchedulerTests.swift:328-384`,
        `RuntimeConfigurationTests.swift:96-111` the patterns);
        `ExpertLoadOperation.onCompletion` (called once, after the terminal
        transition, at once if already terminal, never on the calling thread's lock)
        in `ExpertLoadOperationTests.swift`; a new `ExpertPrefetchRingTests.swift`
        (the budget counts `.submitted` and `.inFlight` slots, the best-scored
        predictions are kept, the dedupe against resident and active, the reclaim
        count); the runner's placement selection on a fake plan (a batch present
        defers, an all-hit layer issues at once).
  - [ ] Step 2 (the code): the configuration and the banner; the ring's
        `inFlightBudget`; `placement=after` registers `begin` on the layer's demand
        operation's completion with the predicted ids and the target layer captured
        and the resident set read at completion (the demand batch's own experts are
        resident by then and dedupe out); `placement=beside` is today's issue point;
        the probe's distance fix; three counters on the runner line beside the
        `prefetch_*` family: `prefetch_deferred` (batches issued from a completion),
        `prefetch_overlapped` (demand batches submitted while a ring read was in
        flight), `prefetch_late` (predicted experts in flight at plan time that the
        plan then read again). Four gates; the filtered ThreadSanitizer pass on the
        streaming and runtime suites.
  - [ ] Step 3 (numerics): golden IDENTICAL on both profiles on the M4 Pro at the
        default, at `after` B = 1 and B = 2 at top-8, and at `beside` top-8; on the
        mini at the default and the candidate cells at each amend and at all cells at
        the landed commit.
  - [ ] Step 4 (the arms, mini): per shape the mirrored order prod, after-B1,
        after-B2, after-B2, after-B1, prod at top-8 (18 lifetimes), plus one
        `beside` top-8 lifetime on the card as the control that reproduces the
        slowed reads on this binary. Readings per arm: `io_fetch_ms` per reading
        layer, `prefetch_issued / adopted / reclaimed / deferred / overlapped /
        late`, adopted-only layers per token (the kernel role), the miss window from
        the `Shrike gap` block, the submit gap, and tok/s on the three answers with
        the turns and the warm second prompts as controls.
  - [ ] Step 5 (the rule, pre-registered): **the verdict on the rule** is the reading
        layers' per-read cost with the ring on at `after`: within the drift of the
        prod arms' 1.03 to 1.07 ms means the placement rule holds in production and
        Tasks 2 and 3 are built; above it by more than the drift means the drive's
        contention is not placement alone, the term is named, and Tasks 2 and 3 are
        re-priced on that number before either is written. **The rule for the
        default**: real and free on the three answers flips
        `SHRIKE_PREDICTIVE_PREFETCH` on at the winning cell; otherwise the knobs land
        at their measured defaults with the prefetch off, as today.
  - [ ] Step 6 (zero code beyond the fix): one trace capture per shape at distance 2
        on the corrected probe (`tools/decode-rig.sh` with `PREFETCH_TRACE=1` and
        `SERVER_ENV="SHRIKE_PREFETCH_PROBE_DISTANCE=2"`, `tools/prefetch-coverage.py`),
        the coverage and precision two layers ahead re-measured; recorded against the
        candidate (e) in the design doc.
  - [ ] Step 7 (design doc): the Task 1 section, the After T1 block, the lever
        entries updated with what was measured. Task review by a fresh reviewer,
        fixes folded into the owning commits.

  **Risks and what falsifies the model.**
  - **Production's gap is not the probe's.** The prefetch read must land before the
    next router readback resolves into a miss; `prefetch_overlapped` and
    `prefetch_late` say how often it does not, and a per-read cost above production's
    with both counters near zero would mean a term the probe did not see.
  - **The budget trims coverage.** A layer that predicts two misses stays a reading
    layer at B = 1; the B = 2 arm prices the trade, and the late counter says whether
    its tail cost is the probe's +0.05.
  - **The hook's thread.** `begin` from a reader thread takes the ring's lock and the
    streamer's cache lock for the resident set; neither is held by the finishing
    path, and the filtered sanitizer pass is the check.
  - **The copy and the probe are still paid.** Task 1's tok/s is bounded by them; a
    null on tok/s with the per-read verdict passed is the expected outcome the plan
    is written to survive.

### Task 2: the reserved-slot landing and the late join

- [ ] **T2: adoption is a host copy of a full expert stride inside the submit gap,
  0.12 ms per adopted expert (0.9 ms per token at top-4, 1.2 to 1.4 at top-8), and a
  correct prediction still in flight at plan time is read twice.** The copy sits at
  `PreadExpertStreamer.swift:762-771` under the cache lock because the ring's buffers
  are not cache slots (`:1309-1311`); the plan's `adopted` set exists so the GPU's
  residency snapshot and the plan can disagree by exactly that set
  (`RealForwardRunner.swift:6819-6828`). The task lands the predicted read in a pool
  slot the planner reserves speculatively (a victim chosen as for a miss, `loading`
  under a new generation, `resident` when the bytes land, the same path a demand
  miss takes at `:744-761`), so the classifier sees an adopted expert as a hit and
  the copy and the `adopted` fold disappear; a predicted read still in flight at
  plan time is joined by the fixup's event wait instead of duplicated. Costs the pool
  a wasted fill per wrong prediction. Modelled +1.2 to +1.6 ms per token. **Built
  only if Task 1's per-read verdict passes.**

  **Steps.**
  - [ ] Step 0 (zero code): the replay tool prices the wasted fills. Extend
        `tools/expert-pool-replay.py` with a speculative-fill input (the archived
        `prefetch-*.jsonl` traces from T1 carry the predicted top-8 per decode layer)
        that fills predicted nonresident experts into victim slots at the placement
        rule's budget, and report the miss count against production's replay on the
        four archived traces. A miss-rate cost above what the copy's removal buys
        lands the task as a candidate with its number.
  - [ ] Step 1 (tests RED first): the planner's speculative reservation (a slot
        claimed `loading` for a predicted expert, released or promoted, never
        double-claimed, protected experts and chunk protection respected, the
        generation rule for a pinned slot); the classifier and plan agreement with
        no `adopted` set; the late join's event accounting (the fixup waits on the
        speculative operation's token when its expert is in the plan's misses).
  - [ ] Step 2 (the code): the reservation API on `PreadExpertStreamer`, the ring
        retargeted at reserved slots (its own buffers retired behind the knob),
        `beginPrefetch` given a completion token, the late join in the runner's fixup
        build, the `adopted` fold retired on the new path, the counters
        (`prefetch_landed`, `prefetch_joined`, `prefetch_evicted_unused`). Four gates.
  - [ ] Step 3 (numerics): golden IDENTICAL on both boxes and profiles, the knob off
        and on, plus `speculative-validate` on the M4 Pro; one `SHRIKE_ROUTE_TRACE`
        capture replayed against production's miss counts to bound the wasted fills'
        effect on the plan sequence.
  - [ ] Step 4 (the arms, mini): the Task 1 winning cell with and without the landing,
        mirrored, the three answers; readings: the submit gap (the copy's 0.12 per
        adopted expert should leave it), `prefetch_late` (should go to zero),
        `prefetch_evicted_unused`, the hit rate, tok/s.
  - [ ] Step 5 (the rule): real and free flips the default to the landing; a null
        lands the knob at its measured default.
  - [ ] Step 6 (design doc, review).

### Task 3: the fused probe

- [ ] **T3: the next-layer probe is a second router GEMV per layer on the same
  input as the authoritative router, 53 µs of GPU per layer, 2.0 to 2.3 ms per token
  in the attention tail, and it is dispatch-bound.** Both dispatches read `routedX`
  and differ only in the weights, scales, bias and output buffers
  (`RealForwardRunner.swift:3503-3533`). One dispatch scoring both routers removes
  the second launch; the authoritative half's ids and weights must be bit-identical.
  Modelled +1.5 to +2.0 ms per token. **Built only if Task 1's per-read verdict
  passes.**

  **Steps.**
  - [ ] Step 0 (zero code): the probe's GPU cost re-measured on the Task 1 binary
        from the kernel stats (the role's per-layer time with the ring on against
        off), so the prize is the box's number.
  - [ ] Step 1 (tests RED first): a kernel test that the fused router's authoritative
        outputs equal the single router's bit for bit on the toy models, and that
        the probe half equals the separate probe's.
  - [ ] Step 2 (the code): the fused kernel in `moe.metal` and its encode in
        `MoE`, selected by the ring's presence, the separate probe retained behind
        the knob for the A/B. Four gates.
  - [ ] Step 3 (numerics): golden IDENTICAL on both boxes and profiles with the ring
        on and off.
  - [ ] Step 4 (the arms, mini): fused against separate at the Task 1 (and Task 2)
        winning cell, mirrored; readings: the tail's GPU time per layer, tok/s.
  - [ ] Step 5 (the rule): real and free flips the default.
  - [ ] Step 6 (design doc, review).

### Task 4: the prefill-to-decode boundary (the companion)

- [ ] **T4: every answer's first window is its worst, 42 misses per token at 11.9
  tok/s ([v14-decode.md](v14-decode.md) step zero row 3), because the pool holds
  prefill's experts when the answer starts.** v13 Task 5's resident-first sweep was
  the first cut (0.5 to 0.9 s off the first turn's decode) and named itself
  unfinished ([v13-implementation-plan.md](v13-implementation-plan.md) Task 5). The
  task prices what the boundary has left with the replay tool on the four archived
  traces first (the state the prompt hands to the answer: which of prefill's
  residents the answer's first 64 tokens reuse, and what a boundary-time fill from
  the prompt's own route history would have caught), then builds the winner behind
  a knob. It shortens the slowest stretch of every card without raising the plateau.

  **Steps.**
  - [ ] Step 0 (zero code): the replay pricing above, a named stop (a boundary
        policy must remove a measured fraction of the first window's misses above
        what the resident-first sweep already takes).
  - [ ] Step 1 (tests RED first), Step 2 (the code), Step 3 (numerics), Step 4 (the
        arms: the first 64 tokens of each cold answer as the verdict window beside
        the whole answer, the turns as controls), Step 5 (the rule), Step 6 (design
        doc, review).

## Candidate tasks (not scheduled)

- **(e) Deeper lookahead.** The "drive never idles" ceiling (18.3 tok/s on the cold
  card, 22 with the other gaps) needs reads two or more layers ahead; T1's
  distance-2 coverage was measured on the mis-scaled probe and Task 1 Step 6
  re-measures it for nothing. Scheduled only if the corrected number clears the
  candidate bar (coverage above 0.10, precision above 0.21) and Task 1's rule holds.
- **The host-state term in production.** If the chapter's ledger finds production's
  per-read time nearer 0.93 than 0.80, the term is named and priced (a busy core
  through the miss window); not built on the probe's number alone.

## Follow-ons (not scheduled)

- The ring's own test coverage beyond Task 1's (slot lifecycle, error rollback, the
  probe wiring in the runner).
- `recordPrefetchAdoptionsUnlocked` counts adoptions as reloads
  (`PreadExpertStreamer.swift:1454-1466`); a distinct counter once Task 2 retires
  the copy.

## Close

- [ ] The full suite under ThreadSanitizer, a whole-branch review by a fresh
  reviewer, the fixes folded into their owning commits, the design doc's After
  block, Davor's go, the fast-forward merge to `main` and the push.
