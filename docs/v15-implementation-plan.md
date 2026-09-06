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
        d3da6f9; the flipped build golden identical at the default and off on both
        boxes (and at B = 2 on the M4 Pro); the confirmation arms on the deployed
        default (prod, off, prod per shape, run before and again after the
        review's fold, `~/.claude/handoffs/archive/shrike-v15-t1/t1-confirm-summary.md`
        and `t1-confirm-review-summary.md`): **14.44 / 14.57 then 14.81 / 14.76 on the
        card, 15.59 / 15.21 then 15.44 / 15.55 on the 300, 15.54 / 15.54 then 15.34 /
        15.59 on the 1k** against 13.92 / 13.93, 14.79 / 14.82, 14.89 / 14.94 off
        (the off cell −4.1 to −5.8 / −3.9 to −4.4 / −3.4 to −4.2 %), every answer
        identical, the follow-ups unmoved; the fixed build's full suite 1294 tests,
        its filtered sanitizer pass 50 tests, golden identical at the default and off
        on both boxes.
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
  - [x] Step 7 (design doc): the Task 1 section, the After T1 block, the lever
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

### Task 2: the adoption by GPU blit and the bounded late join

- [ ] **T2: adoption is a host copy of a full expert stride inside the submit gap,
  0.12 ms per adopted expert (1.2 to 1.5 ms per token at Task 1's default, the
  submit gap 2.2 to 2.4 to 3.4 to 3.7 ms per token measured), and a correct
  prediction still in flight at plan time (0.5 to 1.0 per token) is read twice.**
  The copy sits at `PreadExpertStreamer.swift:762-771` under the cache lock because
  the ring's buffers are not cache slots (`:1309-1311`). Step 0 priced the plan's
  original design, a speculative landing straight into a pool slot, and found the
  wrong fills' evictions cost about one miss per token, most of the prize; Davor's
  ruling (2026-09-07) took the alternative: **the ring stays, the planner reserves
  the adopted slot as `loading` without copying, and the fixup command carries a GPU
  blit from the ring's buffer into the slot ahead of its event wait**, the shape the
  Metal-I/O storage path already uses (`MetalExpertStagingTransfer.encodeCopy`,
  `RealForwardRunner.swift:4037-4042`, the slot resident only when that command has
  completed, `:6404-6432`). No host copy, no speculative eviction, the pool's
  accounting untouched; the GPU pays a blit of 1.77 MB per adopted expert under the
  demand read it was already waiting for. And **the late join**: a prediction still
  in flight when the exact route asks for it is awaited up to a bound (the residual
  of a read that is nearly done) instead of being read again beside its own
  duplicate. Modelled: +1.2 to +1.5 ms per token from the copy and +0.3 to +0.6 from
  the join, +2 to +3 %. **The mini decides**, and a measured null is a result.

  **The mechanism, verified in the tree.**
  - The fixup command is built once for every mode that computes the adopted
    experts (`buildAndCommitMissFixupCommand`, `RealForwardRunner.swift:4005`); its
    event wait is encoded first, so a blit encoded before it runs the moment the
    command starts, under the demand read; Metal orders the blit before the compute
    on the same slab.
  - The plan's `adopted` indices already ride the fixup's partition
    (`:6886`, `:6899`) and the GPU's classifier already sees them as misses; nothing
    in the kernels changes. The slot is pinned by the plan's lease until the pending
    command is finished (`pinRoutedExperts`, `finishPendingRoutedCommand`).
  - The ring's leased slot (Task 1's lease) is held until the command completes and
    released there, not at the plan; the ring is sized for the slots in flight,
    the completed ones awaiting a plan and the adopted ones awaiting their blit.
  - `ExpertLoadOperation.wait()` is unbounded; the join needs a bounded wait.

  **Steps.**
  - [x] Step 0 (zero code): the replay tool prices the wasted fills. Extend
        `tools/expert-pool-replay.py` with a speculative-fill input (the archived
        `prefetch-*.jsonl` traces from T1 carry the predicted top-8 per decode layer)
        that fills predicted nonresident experts into victim slots at the placement
        rule's budget, and report the miss count against production's replay on the
        four archived traces. A miss-rate cost above what the copy's removal buys
        lands the task as a candidate with its number. **DONE 2026-09-07 (the tool's
        `--speculative-fills`, `~/.claude/handoffs/archive/shrike-v15-t2/step0/`),
        MODELLED from T1's measured distance-1 captures (route and prefetch traces of
        the same lifetimes) at production's pool (128 slots, aging-lfu; the baseline
        replay reproduces production's decode misses exactly, 6689 / 9468 / 11398):**

        | cell | card misses (per token) | the 300 | the 1k | useful / wasted fills per token (card) |
        | --- | ---: | ---: | ---: | --- |
        | production, no fills | 6689 (30.5) | 9468 (30.2) | 11398 (28.1) | |
        | top-8, one fill per layer | 4309 (19.7) | 6261 (19.9) | 7666 (18.9) | 11.8 / 7.0 |
        | top-8, two per layer | 4151 (19.0) | 6150 (19.6) | 7665 (18.9) | 12.7 / 13.6 |
        | top-4, one per layer | 5258 (24.0) | 7545 (24.0) | 9090 (22.4) | 6.9 / 1.9 |

        The misses saved per token at the ring's cell (10.9 / 10.2 / 9.2) fall short of
        the useful fills (11.8 / 11.3 / 10.1) by **about one miss per token: the wrong
        fills' evictions**, 0.85 ms per token at step zero's slope, against the copy's
        1.2 to 1.5 ms. So the landing as first designed nets 0.3 to 0.6 ms per token
        before the late join, about +1 to +2 % in all. The fork was Davor's call:
        (i) the landing as designed; (ii) the adoption by GPU blit from the ring's
        buffer (no host copy, no speculative eviction); (iii) a buffer swap under the
        per-slot cache layout, which v9 measured a loss (the per-slot to pool flip
        halved the all-hit gap, 32.6 to 16.7 ms per token, Metal residency over about
        3,400 slot buffers). **Ruled (ii)**; an index swap inside one slab is the
        swap done properly and a later refinement.
  - [x] Step 1 (tests RED first): the streamer's blit adoption (a plan with an
        adoptable expert reserves its slot `loading` with the bytes untouched, lists
        it in `adopted`, excludes it from the resident sweep until
        `finalizeAdoptedSlots` marks it resident under the generation guard, and
        `failAdoptedSlots` empties it); the adoption transfer (a Metal blit from
        tagged source buffers into slab regions lands the bytes, an out-of-bounds
        range throws before encoding, `release` runs once); the operation's bounded
        wait (true when finished before the deadline from another thread, false at
        the deadline); the ring's join (an in-flight prediction finishing within the
        budget is adopted and counted `joined`, one that does not is `late`);
        `RuntimePrefetch.adoption` (`copy` | `blit`) and `joinMicros` (0 to 2000)
        fail-closed with the banner's `adopt=` and `join_us=` fields.
  - [x] Step 2 (the code): the streamer's `PrefetchAdoption` (`hostCopy` /
        `gpuBlit`) through the model's plan entry point, the finalize and fail paths
        beside the Metal staging ones; `PrefetchAdoptionTransfer` built by the runner
        from the ring's buffers and the plan's expert views, encoded at the head of
        the fixup command, carried on the pending command, finalized and released
        when it finishes (and failed on every early exit); the ring's slots released
        at that point in blit mode; the bounded join in `readyBuffers`; the knobs in
        both binaries and the banner; `prefetch_joined` and `prefetch_blit_experts`
        on the runner line and in `tools/decode-rows.py`. Four gates; the filtered
        sanitizer pass.
  - [x] Step 3 (numerics): golden IDENTICAL on both boxes and both profiles at
        `copy` and `blit`, join off and on, plus `speculative-validate` on the M4 Pro.
        **DONE 2026-09-07:** copy, blit, blit with the join at 400 µs and off on the
        mini; those plus blit under `speculative-validate` on the M4 Pro; the filtered
        sanitizer pass 57 tests, no report.
  - [x] Step 4 (the arms, mini): Task 1's default (copy, no join) against blit,
        against blit with the join, mirrored per shape; readings: the submit gap
        (the copy's 0.12 per adopted expert should leave it), the miss window,
        `prefetch_late` and `prefetch_joined`, the fixup's GPU time (the blit),
        tok/s, the follow-ups as controls. **DONE 2026-09-07, 18 lifetimes, every
        answer identical** (`~/.claude/handoffs/archive/shrike-v15-t2/t2-arms-summary.md`):

        | cell | card tok/s | the 300 | the 1k | submit gap ms per token | late / joined per token |
        | --- | ---: | ---: | ---: | --- | --- |
        | copy (Task 1's default) | 14.65 / 14.61 | 15.44 / 15.44 | 15.41 / 15.44 | 4.19 / 3.64 / 3.38 | 0.5 / 0 |
        | blit | 15.08 / 15.10 (+3.2 %) | 15.89 / 15.76 (+2.5 %) | 15.38 / 15.63 (+0.5 %) | 2.48 / 2.5 to 2.7 / 2.4 to 2.5 | 0.6 to 0.8 / 0 |
        | blit, join 400 µs | 15.17 / 14.87 (+2.7 %) | 16.03 / 16.06 (+3.9 %) | 15.73 / 15.72 (+1.9 %) | 2.5 to 2.7 | 0.00 / 0.6 to 0.9 |

        The blit takes the copy out of the submit gap on every shape (1.7 / 1.2 / 1.0
        ms per token); the join catches every late prediction and lifts adoption by
        0.4 to 0.8 per token. Under the blit the gap block's miss window no longer
        measures the GPU's idle time (the command starts with the blit ahead of its
        event wait), so the wall and tok/s are the verdict: 68.6 to 66.2 / 64.9 to
        62.5 / 65.0 to 63.7 ms per token from copy to blit with the join.
  - [x] Step 5 (the rule): real and free flips the defaults to blit and the winning
        join bound; a null lands the knobs at their measured defaults. **DONE
        2026-09-07:** blit with the join is real (both orders on all three shapes,
        +3.6 / +1.8, +3.8 / +4.0, +2.1 / +1.8 % against a drift of −0.2 / 0.0 / +0.2)
        and free (the follow-ups unmoved, golden identical everywhere): the defaults
        are `adopt=blit join_us=400`; `SHRIKE_PREFETCH_ADOPT=copy` and
        `SHRIKE_PREFETCH_JOIN_US=0` are the A/Bs.
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

- **The two-distance queue (SCHEDULED as a step zero after Task 2 lands; Davor's
  ruling 2026-09-07 ≈ 01:50 on a peer session's proposal, "agreed with your
  recommendations").** The proposal's diagnosis holds on the deployed default's own
  counters: per token on the card the ring issues 20.2 predictions, adopts 10.0 and
  refuses 15.0 for want of the one read in flight, and the refused ones are mostly
  layer L + 1's further absent experts, which no placement can serve inside the
  0.9 ms lead a read has before L + 1's plan (Task 1's B = 2 arm measured exactly
  that); the idle half of every window can only be spent on layers further ahead.
  The proposal's forecast, the previous token's route, is measured dead in the record
  (`docs/architecture.md`, "What the proof disproves": 0.00 % of misses at 16 and
  128 slots, the predictable set and the miss set disjoint by construction), so it is
  not run. What survives is a queue fed by the router probe at distances 1 and 2 (the
  corrected probe's distance-2 coverage 0.34 to 0.37 at precision 0.29 to 0.34, Task 1
  Step 6): a window with nothing useful for L + 1 serves L + 2's prediction with
  2.6 ms of lead, and under Task 2's blit a wrong one costs a ring slot and drive time
  the placement rule already makes free. **The step zero, zero runtime code, on T1's
  archived captures:** (i) join the distance-1 and distance-2 captures by position and
  target layer and report the union's full-layer coverage and precision per layer
  against 0.46 / 0.47; (ii) price the queue with the replay tool's fill hook (a
  target's fill from the d1 line at T − 1 or the d2 line at T − 2, one per window)
  against Task 1's modelled 4309 / 6261 / 7666 misses. **The named stop:** a union
  coverage near 0.46 or misses near 4309 means the chapter proceeds as written. If it
  survives, it is a task with a bounded prize (the misses the current mechanism
  cannot reach, about 20 per token) and Task 3 becomes "fuse both distances into one
  dispatch". A smaller separate item from the same proposal, the token-boundary window
  (the head and sampling, about 5 ms of idle drive, spent on layers 0 to 4 if their
  routes follow from the sampled token), is priced later.
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
