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
  - [x] Step 0 (zero code): the production per-read baseline from the archived v14
        arms, so the verdict has its two anchors before the first arm runs.
        **DONE 2026-09-06.** The measure is the `Shrike gap` block's miss window per
        token over the reading layers per token (`hit_fixup_layers` less the
        adopted-only layers from the kernel roles; the runner line's `io_fetch_ms` is
        cumulative over the request and mixes prefill's fetches,
        `ServerInference.swift:2078`). Ring off (the T1 A/B's prod arms and the
        lever B prod arms on `4cc6e58`'s build): 18.1 to 20.3 ms over 17.4 to 18.7
        layers, **1.03 to 1.07 ms per reading layer**, at 30.5 misses per token on
        the card (`expert_misses_decode` 6689 over 219 tokens, `hit_fixup_layers`
        3981). Ring beside the demand reads (T1's instrumented arms): **1.15 to 1.23
        at top-4, 1.59 to 1.71 at top-8** ([v14-decode.md](v14-decode.md) "Task 1").
        The drift on prod's tok/s across the T1 A/B: −1.2 / +0.3 / +0.6 % on the
        three answers.
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
  - [x] Step 3 (numerics): golden IDENTICAL on both profiles on the M4 Pro at the
        default, at `after` B = 1 and B = 2 at top-8, and at `beside` top-8; on the
        mini at the default and the candidate cells at each amend and at all cells at
        the landed commit. **DONE 2026-09-06 on the task's build:** five cells
        (default, after B = 1, after B = 2, beside B = 8, beside B = 1, all at
        top-8), short and long, identical on the mini
        (`ledger/t1-mini-golden.log`) and on the M4 Pro (`ledger/t1-local-golden.log`);
        repeated at the landed commit if it is amended.
  - [x] Step 4 (the arms, mini): per shape the mirrored order prod, after-B1,
        after-B2, after-B2, after-B1, prod at top-8 (18 lifetimes), plus on the
        card one `beside` at B = 8 (T1's shape, the control that reproduces the
        slowed reads on this binary) and one `beside` at B = 1 (the bounded form
        of T1's placement: every prediction in time, at the `cont` row's +0.23 on
        the missing layer's own read). Readings per arm: the miss window per
        reading layer from the `Shrike gap` block (`tools/decode-rows.py`), the
        adopted-only layers per token (the `_adopted` role's gap count),
        `prefetch_issued / adopted / reclaimed / deferred / overlapped / late /
        hook_failed` per token, the submit gap, and tok/s on the three answers with
        the turns and the warm second prompts as controls.
        **DONE 2026-09-06, 20 lifetimes, every answer identical** (the ledger's
        23:58 entry, `~/.claude/handoffs/archive/shrike-v15-t1/t1-arms-summary.md`):

        | cell (top-8) | card tok/s | the 300 | the 1k | reading layers, ms each (card / 300 / 1k) | late per token | adopted per token |
        | --- | ---: | ---: | ---: | --- | ---: | ---: |
        | prod (ring off) | 14.13 / 13.93 | 14.68 / 14.71 | 14.82 / 14.82 | 1.08 to 1.09 / 1.05 to 1.06 / 1.06 | 0 | 0 |
        | after, B = 1 | 14.80 / 14.49 (+4.4 %) | 15.61 / 15.61 (+6.2 %) | 15.43 / 15.45 (+4.2 %) | 1.03 to 1.11 / 0.97 to 0.98 / 0.98 | 0.46 to 0.79 | 8.9 to 10.0 |
        | after, B = 2 | 14.33 / 14.38 (+2.3 %) | 14.88 / 14.92 (+1.4 %) | 14.86 / 14.84 (+0.2 %) | 1.15 to 1.16 / 1.12 / 1.11 | 6.8 to 7.9 | 5.6 to 7.2 |
        | beside, B = 8 (T1's shape, card) | 13.19 (−6.0 %) | | | 1.63 | 5.35 | 11.1 |
        | beside, B = 1 (card) | 13.84 (−1.4 %) | | | 1.31 | 0.30 | 10.3 |

        The reading layers cost production's per-read time or less with the ring on
        at B = 1 (the placement rule holds on the box); the pair at B = 2 arrives
        late and slows the reads it overlaps; `beside` at B = 8 reproduces T1's loss
        and at B = 1 pays the probe's +0.23 per reading layer exactly. Misses per
        token 30.5 / 30.2 / 28.1 to 20.6 / 20.4 / 19.2 at B = 1; the miss window 19.6
        to 19.9 / 19.6 to 19.8 / 18.4 to 13.4 to 14.6 / 13.4 to 13.5 / 12.5 to 12.6 ms;
        the submit gap 2.2 to 2.4 to 3.4 to 3.7 (the adoption copy, Task 2's prize);
        the follow-up walls unmoved.
  - [x] Step 5 (the rule, pre-registered): **the verdict on the rule** is the reading
        layers' per-read cost with the ring on at `after`: within the drift of the
        prod arms' 1.03 to 1.07 ms means the placement rule holds in production and
        Tasks 2 and 3 are built; above it by more than the drift means the drive's
        contention is not placement alone, the term is named, and Tasks 2 and 3 are
        re-priced on that number before either is written. **The rule for the
        default**: real and free on the three answers flips
        `SHRIKE_PREDICTIVE_PREFETCH` on at the winning cell; otherwise the knobs land
        at their measured defaults with the prefetch off, as today.
        **DONE 2026-09-06: both verdicts pass.** The per-read verdict: at or below
        production's on all three shapes, so Tasks 2 and 3 are built. The default:
        real (4.0 to 6.3 % in both orders on every shape against a prod drift of
        −1.5 to +0.2 %) and free (the turns and the warm second prompts unmoved,
        golden identical at every cell on both boxes), so the prefetch is ON by
        default at one read in flight, placed after the demand batch, topM the
        architecture's top-k; `SHRIKE_PREDICTIVE_PREFETCH=0` is the A/B. Landed as
        ff4cfc6; the flipped build golden identical at the default and off on both
        boxes (and at B = 2 on the M4 Pro); the confirmation arms on the deployed
        default (prod, off, prod per shape,
        `~/.claude/handoffs/archive/shrike-v15-t1/t1-confirm-summary.md`): **14.44 /
        14.57, 15.59 / 15.21, 15.54 / 15.54 tok/s** against 13.92 / 14.79 / 14.89 off
        (−4.1 / −3.9 / −4.2 %), every answer identical, the follow-ups unmoved.
  - [x] Step 6 (zero code beyond the fix): one trace capture per shape at distance 2
        on the corrected probe (`tools/decode-rig.sh` with `PREFETCH_TRACE=1` and
        `SERVER_ENV="SHRIKE_PREFETCH_PROBE_DISTANCE=2"`, `tools/prefetch-coverage.py`),
        the coverage and precision two layers ahead re-measured; recorded against the
        candidate (e) in the design doc. **DONE 2026-09-07:** the corrected capture is
        BYTE-IDENTICAL to T1's archived distance-2 capture on the card and gives the
        same join on all three shapes (top-8 coverage p 0.367 / 0.342 / 0.337,
        precision 0.341 / 0.301 / 0.289, wasted reads 24 to 28 per token; distance 1:
        0.462 / 0.442 / 0.428 at 0.471 / 0.423 / 0.407). The defect is real in the
        code and inert on ornith15: the Qwen-family runner binds one shared ones
        buffer as every layer's effective scale and one zeros buffer as every layer's
        logit bias (`RealForwardRunner.swift:1473-1533`), so `L + 1` and `L + d`
        index the same bytes; the fix matters for gpt-oss (a per-layer router bias)
        and Kimi (a per-layer correction bias). T1's "two layers ahead costs 0.10 of
        p" stands as measured; the candidate (e) is priced at that number.
  - [ ] Step 7 (design doc): the Task 1 section, the After T1 block, the lever
        entries updated with what was measured. Task review by a fresh reviewer,
        fixes folded into the owning commits. **The review (2026-09-07) found one
        real defect of this task and it was folded:** a storage-thread `begin`'s
        reclaim could free a completed slot of another layer while the decode thread
        was still copying its bytes into the cache (between `readyBuffers` and
        `consume`; at distance 2 the ordinary case, at distance 1 a delayed hook);
        fixed by a lease on the slots `readyBuffers` hands out, cleared by `consume`,
        skipped by the reclaim, with its test. Also folded: a `prefetch_refused`
        counter for predictions dropped for lack of budget or slot, a one-time log of
        the first refused speculative read, a cross-thread exactly-once test on the
        hook, and the comment trims. Left as noted: a one-count overcount of
        `deferred` when the batch finishes between the state read and the hook's
        registration; the banner's `placement=after` under the host-wait and deferred
        submission modes, where the batch is already awaited.

  **Risks and what falsifies the model.**
  - **The lead at distance one after a demand completion is thin.** A read issued
    when layer L's demand batch completes has L's fixup and L + 1's attention,
    about 0.8 to 0.9 ms, before L + 1's plan asks for it, against a 0.8 ms read
    (p90 0.9); on an all-hit layer it has the layer's whole compute. So under
    `after` a share of the missing layers' predictions arrive late and are read
    again by the demand batch (`prefetch_late`), which is the in-time fraction the
    task measures; the remedies are the late join (Task 2) and distance two after
    Step 6's re-measure, and the bounded `beside` arm prices the other trade.
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
