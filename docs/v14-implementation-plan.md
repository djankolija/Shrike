# v14 implementation plan: decode pass II

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Status of record for [v14-decode.md](v14-decode.md). Checkboxes here are the only
status tracking. Commit SHAs in the verdicts are the branch's current ones; every
review fix is folded into its owning commit (rebase and amend, never fixup commits),
so the SHAs settle only once the branch stops being rebased: the subjects are the
stable names.

**Goal:** Cut the wall of a card's answer on the mini. Step zero measures 69 to 74 ms
per decoded token on three answers of 219 / 314 / 405 tokens, of which 45 ms is GPU
work and 26 ms is the GPU standing still, 18 to 20 ms of it in the single window
where the routed stage waits for an expert to arrive from the SSD. One measured
lever at a time, each landing as a knob whose default is the measured winner.

**Architecture:** The runtime is v13's (the matrix prefill kernels, the collapsed
routed tile sequencer, the resident-first sweep, chunk-aware protection, the
two-batch expert reader, the prompt cache). Decode's own path is unchanged since
v9's speculative routed dispatch: one cache plan per layer per token, the fetch
begun immediately, the routed command buffer event-gated on it. Each task changes
scheduling or policy behind a `SHRIKE_*` knob, is measured on the step-zero rig (a
fresh server per arm, whole answers), passes the four per-commit gates, and keeps
golden identical unless it runs different kernels on a token.

**Tech Stack:** Swift 6.3, Metal 4, swift-testing, the `SHRIKE_RUNNER_STATS` /
`SHRIKE_KERNEL_STATS` counters and their `tools/parse-*-stats.py` parsers,
`SHRIKE_ROUTE_TRACE` with `tools/expert-pool-replay.py`, `SHRIKE_PREFETCH_TRACE`,
and the step-zero rig (`tools/turn-rig.sh` / `tools/turn-prompts.py` /
`tools/turn-summary.py` plus the streaming client archived at
`~/.claude/handoffs/archive/shrike-v14-step0/loop-files/step0-stream*.{sh,py}`, to be
moved under `tools/` by the first task that needs it in the tree).

**Spec:** [v14-decode.md](v14-decode.md)

## Global constraints

- **The chapter's code starts on a fresh branch off `main` after v13 merges, never
  on `perf/v13-the-turn`.** v13's branch is under its close (ThreadSanitizer, a
  whole-branch review, then Davor's fast-forward merge and push); nothing in this
  chapter is committed until that merge has landed and the new branch is cut from
  the merged `main`.
- macOS 26+, Swift 6.3+; never two model processes (`pgrep` check first); the mini's
  server on 8081 is production and Turbo on 8080 is never touched. Restarts and
  deploy actions are approved per session; `tools/mini-deploy.sh` copies binaries
  plus `*.bundle` directories by default and restarts the server only when passed
  `--restart`. `memory_pressure -Q` before every launch.
- Four gates per code commit: release build with zero warnings; `swiftlint lint
  --strict --baseline .swiftlint-baseline.json`; markdown link check; `swift test
  --no-parallel`. The same suite under ThreadSanitizer (`env
  TSAN_OPTIONS=suppressions=tsan-suppressions.txt`) runs **once per chapter at its
  close**, before the merge to main (Davor's ruling at v13 Task 1, 2026-09-04). A
  task whose own code is pthread or Metal-event concurrency runs a **filtered**
  sanitizer pass in its own step (v13 Task 2's precedent), which is minutes.
- Numerics: a scheduling or policy change is golden IDENTICAL on both boxes and both
  profiles (`tools/golden-baseline.sh --check`); a change that runs different kernels
  on a token follows v12's policy (2e-2 against the fp32 reference, golden recaptured
  once per box with before and after digests in the verdict). Never recapture for an
  unexplained mismatch.
- Ledger protocol per task: the step-zero rig on the mini, and **a decode verdict row
  is a whole answer**, not an 8-token burst (the rate swings 11 to 18 tok/s inside
  one answer, [v14-decode.md](v14-decode.md) step zero row 3). The three step-zero
  answers (2,125 / 289 / 1,069-row prompts at `max_tokens` 512, `temperature` 0) are
  the verdict rows; the card's turn 2 and turn 3, the warm same-length second
  prompts, and the 12k first request are the controls. One send per server lifetime
  for a cold row, `settle_done` before a warm second send, a distinct prompt per arm.
  Rows appended to the design's ledger. The M4 Pro is the check.
- **No size floor:** a task's rule tests only that the effect is real (paired runs in
  both orders, above the run-to-run drift) and free (no control row regresses, golden
  identical). Davor's ruling at v13 Task 1.
- Comments: none unless a genuinely non-obvious why (repo rule).
- **Every `file:line` in this plan was re-verified against this branch's base,
  commit `5316d2a`** (the merged v13 close), when the plan was copied onto the branch
  on 2026-09-06; the citations into `RealForwardRunner.swift` and
  `tools/expert-pool-replay.py` had drifted since the draft and were re-anchored.
  The exception is `docs/v10-implementation-plan.md`, cited against the working-tree
  copy carrying the peer's uncommitted P3 follow-on (a 50-line insertion after its
  line 155); those resolve exactly once that edit lands. The symbol names are the
  stable part.

## Tasks

### Task 1: T1, the layer's fetch under the previous layer's compute, priced before it is built

- [x] **T1: the biggest single term in the chapter is the GPU standing idle between
  `moe_phase1_hit` and `moe_phase1_miss_fixup_phase2`, waiting for the layer's absent
  experts: 19.68 / 19.24 / 18.03 ms per token on the three answers, 26.5 / 27.4 /
  26.1 % of a 74.27 / 70.15 / 69.08 ms token, with `host_ms=0.0` on all three so the
  host is never late (`step0-out/<shape>/server-mini-step0-*.log`, the `Shrike gap`
  lines). Its two ceilings, modelled from step zero's own rows: a lookahead deep
  enough that the drive never idles removes the window entirely (the bytes fit,
  0.73 GB/s used of a measured 3.5 GB/s ceiling), 74.27 to 44.99 + 9.60 = 54.59 ms,
  13.46 to 18.3 tok/s, and a 700-token card answer 52.0 to 38.2 s; a single layer of
  lookahead with perfect prediction leaves the layer's own 1.49 ms of reads against
  its 1.04 ms of compute, 8.2 ms per token exposed, 62.8 ms and 15.9 tok/s (44.0 s).
  Applying the only measured predictor accuracy on record (`architecture.md:96-98`
  recall 0.64 at top-8, `:108-118` recall 0.439 at k = 1) through 1.68 misses per
  missing layer gives a full-layer coverage rate of 0.26 to 0.49 under an
  independence approximation, so 3.0 to 5.6 ms per token, 4.0 to 7.5 %. **The spread
  between 4 % and 36 % is one number the archived instruments can measure without
  writing a line of code**, and the same passage records a −3.9 % end-to-end
  regression at 4-bit when this was last tried (`architecture.md:100-106`). So the
  task is a pricing task first: Step 1 measures the predictor's per-layer coverage
  and precision offline and A/Bs the shipped `SHRIKE_PREDICTIVE_PREFETCH` path with
  zero code, and it has a named stop that lands the null. Step 2 is written only if
  Step 1 clears it. **The mini decides**, and a measured null is a result
  ([v14-decode.md](v14-decode.md) "Method").**

  **Alternative first task (the owner's other option): the miss path's host and
  driver windows.** Two measured terms that need no predictor, no numerics change and
  no closed door reopened: the routed submit gap `moe_spec_routed` to
  `moe_phase1_hit`, **4.13 / 4.19 / 3.84 ms per token (5.6 / 6.0 / 5.6 %)** over
  18.18 / 18.70 / 17.39 missing layers, whose host-late component is 2.04 / 1.97 /
  1.88 ms per token while the planning inside it is only 0.16 to 0.21
  (`cache_plan_ms`); and the post-completion wake `io_fixup_wake_ms`, **3.20 / 3.14 /
  2.88 ms per token (4.3 / 4.5 / 4.2 %)**, the interval from the expert read
  completing to the routed command buffer starting on the GPU
  (`RealForwardRunner.swift:6193-6197`), which is dead by construction. Together
  **7.33 / 7.33 / 6.72 ms per token, 9.9 / 10.4 / 9.7 % of the wall**, or 5.1 s of a
  52.0 s card answer if all of it came off. Against it: how much is recoverable is
  not established (part of the submit gap is driver work and the wake is a Metal
  shared-event signal to GPU start), where T1's prize is bounded above by a number
  three times larger and its uncertainty is a measurement T1's own Step 1 makes. In
  favour of it: the wake sits inside T1's window, so T1 collects it if T1 works and
  this task collects it if T1 dies; it has no prior art arguing against it; and its
  arithmetic is entirely the box's own counters with no independence approximation.

  **Ruling (Davor, 2026-09-06): the drafted T1 is the chapter's first task; the
  alternative is the fallback the named stop moves to.** Step 1 runs first, zero
  code, and decides whether Step 2 is written.

  **Step zero: the term as measured** (mini, the deployed `b13fbfe` binary, the
  pre-amend form of v13's close commit `c04e43b`, see the design doc's step zero; the bare
  production launch plus `SHRIKE_RUNNER_STATS=1 SHRIKE_KERNEL_STATS=1` and
  `SHRIKE_ROUTE_TRACE`, zero code; the cold request streamed so each token's arrival
  is recorded. Logs and traces at
  `~/.claude/handoffs/archive/shrike-v14-step0/step0-out/<shape>/`, the rows by
  `t4-rows.py` in `ledger/step0-stream.out`).

  | per token (ms) | the 2k card (219 tok) | the 300 prompt (314) | the 1k prompt (405) |
  | --- | ---: | ---: | ---: |
  | wall (streamed arrivals) | 74.27 | 70.15 | 69.08 |
  | decode GPU busy (roles summed) | 44.99 | 41.52 | 42.20 |
  | **`moe_phase1_hit` to `moe_phase1_miss_fixup_phase2`** | **19.68** | **19.24** | **18.03** |
  | ... of which `io_fixup_wake_ms` (after the bytes land) | 3.20 | 3.14 | 2.88 |
  | `moe_spec_routed` to `moe_phase1_hit` | 4.13 | 4.19 | 3.84 |
  | other listed decode gaps | 2.52 | 2.74 | 2.80 |
  | unaccounted (below the log's top-8 gap cut) | 2.95 | 2.46 | 2.21 |
  | layers with a miss (`hit_fixup_layers`) | 18.18 | 18.70 | 17.39 |
  | the gap per missing layer | 1.082 | 1.029 | 1.037 |

  The same quantity from two other instruments: the regression's first-miss
  coefficient over the replayed per-layer misses is **1.19 / 1.10 / 1.10 ms**
  (`ledger/step0-fit-aligned.txt`) and `io_ms` divided by `hit_fixup_layers` is
  **1.19 / 1.16 / 1.16 ms**. The drive's own whole-expert read at decode's roughly
  1 ms spacing is **0.949 ms** and back-to-back **0.774**
  (`ledger/step0-inflight-run1.txt`), so the layer pays a read plus 0.13 to 0.24 ms,
  and 0.165 to 0.176 of that 0.24 is the post-completion wake.

  **The mechanism, verified in the tree.**
  - A decode layer plans once and fetches once. `encodeDecodeRoutedMoE`
    (`RealForwardRunner.swift:6463`) reads the exact top-k back from the router
    (`:6481-6489`), writes the route trace (`:6502`), plans the cache
    (`:6510-6515`), pins (`:6525`) and begins the fetch immediately (`:6529-6537`,
    because the shipped `SHRIKE_EXPERT_IO_SYNC` default is `.event`,
    `RuntimeConfiguration.swift:52-59`). The routed command buffer waits on the
    shared event rather than the host (`:6692-6700`, `io_host_waits=0` and
    `io_host_waits_avoided` in every step-zero row).
  - The prediction already exists and is already computed on the GPU. While encoding
    layer L's tail the runner runs layer **L+1's** router on layer L's
    post-attention normalized residual and writes the speculative top-k to
    `prefetchPredictionIndices` (`RealForwardRunner.swift:3453-3471`, the next
    router selected at `:2857-2862`), the host reads it back at `:3118-3126` and
    passes it as `predictedNextLayer` (`:3132-3142`), and both the probe and the ring
    are gated by `nextLayerPredictionEnabled` (`:2023-2025`), which is true when
    either `SHRIKE_PREFETCH_TRACE` names a file or `SHRIKE_PREDICTIVE_PREFETCH=1`
    (`:908-929`). It is an approximation by construction: layer L+1's exact router
    input is layer L's output, which does not exist while layer L runs.
  - The staging path exists too. `ExpertPrefetchRing`
    (`sources/Shrike/Runtime/Inference/ExpertPrefetchRing.swift`, 125 lines) holds
    `topM` raw slots outside the authoritative cache (`:4-11`), `begin` stages only
    the predicted experts that are absent and not already queued (`:43-85`),
    `readyBuffers` hands the exact plan whatever has **completed** (`:87-100`), the
    plan adopts those bytes (`RealForwardRunner.swift:6509`, `:6513` into
    `planRoutedExperts(prefetched:)`) and `consume` frees the slots (`:6517`).
  - **Two structural facts limit what the shipped path can collect**, and both are
    Step 2's subject. `begin` is called at `RealForwardRunner.swift:6738-6743`,
    **after** the layer's own demand fetch has completed and its blobs are in hand,
    so the lead is only the rest of layer L plus layer L+1's attention, against one
    layer's whole wall of 1.86 / 1.75 / 1.73 ms if it were issued at plan time. And
    `readyBuffers` returns only `.completed` slots (`:87-100`, the comment says so),
    so a correct but late prediction buys nothing **and its bytes are read a second
    time** by the demand path.
  - The ring is sized `slotCount: topM` (`:908-929`), default `min(4, topK)` = 4,
    while a missing layer's absent set is up to 8 (mean 1.68, `max/layer` reaching
    4 to 5 in the worst windows, `ledger/step0-fit.txt`).
  - The pool is untouched by all of this: one `PreadExpertStreamer` per layer at 128
    slots, `aging-lfu` (`PreadExpertStreamer.swift:164-168`, `:280`), the reader four
    threads and two published batches (`:223-226`, the v13 T2 winner), the plan the
    same `planExpertsCached` (`:634`) with or without prefetched bytes.

  **Prior art, placed rather than re-derived.** `docs/architecture.md:76-119` holds
  two predictors and only one is disproven: `:83-89` kills same-layer previous-token
  prediction (0.00 % of misses caught at 16 and 128 slots), while `:96-98` measures
  the next-layer probe at 64.1 % recall of nonresident misses at top-8 (56.4 %
  precision) and `:108-118` measures it again at recall 0.439 / 0.322 / 0.256 / 0.223
  for k = 1..4 with nonresident precision 0.124 falling to 0.036 (0.510 to 0.305 and
  0.358 to 0.127 on a diverse prompt) and closes the door: "the miss-count lever is
  closed; the surviving miss levers are cost-side (event gating, free-running) and
  policy-side (cache)". `:100-106` records the end-to-end failure (−3.9 % at 4-bit,
  +7.8 % at 8-bit, against a +10 % bar) and diagnoses lead time. The v10 P3 follow-on
  repeats the verdict from the drive's side (`docs/v10-implementation-plan.md:190-200`,
  an uncommitted peer edit in the working tree, archived at
  `~/.claude/handoffs/archive/shrike-ssd-split-probe/`): the single 1.77 MB miss is
  at its 0.77 ms floor, the only throughput lever is more experts in flight, and
  "every wasted prefetch now measurably steals bytes-in-flight from the real miss".
  **The two numbers step zero adds against that verdict are the lead time (one
  layer's wall is 1.73 to 1.86 ms against a 0.95 to 1.08 ms read) and the bandwidth
  headroom (0.73 GB/s used of 3.5, so total reads may rise 4.8x before the drive
  binds, which puts the break-even precision at 0.21 and the measured k = 1
  precision on both sides of it).** Neither makes the old verdict wrong; both make it
  a measurement this chapter can redo in an afternoon of mini time.

  **The offline experiment and its named stop.** Nothing here writes runtime code.
  - **(i) History-only baselines, from the archived route traces alone.** A recorder
    in the shape of `loop-files/step0-layer-misses.py` (which imports
    `tools/expert-pool-replay.py` rather than editing it) records, for every decode
    (position, layer), the identity of the absent experts, since an expert is absent
    at plan time exactly when no slot holds it (`LayerPool.plan`, `:477-538`, over
    `slot_expert`). From that: the absent-set size distribution per layer; and the
    full-coverage rate of three predictors that need no router at all, namely the
    previous token's demanded set at the same layer (the predictor `:83-89`
    disproved, re-checked at 128 slots on today's pool), the union of the last n
    tokens at that layer, and layer L's own demanded set. Any of these clearing the
    router's rate would be a cheaper lever than the router.
  - **(ii) The router predictor's real accuracy, from one capture per shape on the
    mini, zero code.** `SHRIKE_PREFETCH_TRACE=<path>` emits one JSONL line per
    (position, layer) carrying `experts`, `misses`, `resident` and
    `next_layer_prediction` (`RealForwardRunner.swift:2223-2241`; `misses` is the
    demand plan's own miss identities, `:6519-6521`, and `resident` is captured
    before planning, `:6507-6508`). Joining line (position, L) to line (position,
    L+1) yields, on this build, this pool and these three shapes: per-miss recall,
    nonresident precision, and the quantity the lever actually hangs on, **the rate
    p at which layer L+1's ENTIRE absent set is named at layer L**. A partly covered
    layer still pays the serial 1.08 ms, so p, not recall, is the prize's multiplier.
    Also report p at `SHRIKE_PREFETCH_PROBE_DISTANCE` 1 and 2 (`:2030-2031`) and at
    top-M 4 and 8 (`:916-917`).
  - **(iii) The zero-code A/B.** `SHRIKE_PREDICTIVE_PREFETCH=1` at top-M 4 and 8
    against the production launch, on the three answers, paired in both orders. This
    is the shipped scheme with its late `begin` and its completed-only join, so it
    measures the floor of the idea, not its ceiling; if it already pays, the task
    lands a default flip and writes nothing.
  - **The named stop.** After (i) to (iii): if the best predictor's **full-layer
    coverage p < 0.10 on all three shapes**, or its **precision on the fetches the
    scheme would issue is below 0.21** (the drive's headroom, step zero row 4), the
    lever is dead at this box's numbers. The task then lands the pricing tool, the
    measured table, and the null, and the chapter moves to the alternative above. If
    p clears 0.10 and precision clears 0.21, Step 2 is written, and the modelled
    prize it must beat is p x 11.5 ms per token.

  **The decision rule, under the chapter's real-and-free rule.** The default flips
  only if the effect is **real** (the sign holds on the three answers' `decode_tok_s`
  across paired runs in both orders, three pairs, a fourth where a row sits inside
  twice its drift) and **free** (the card's turn 2 and turn 3 walls, the warm
  same-length second prompts, the 12k first request and `memory_pressure -Q` all
  unmoved, and golden IDENTICAL on both boxes and both profiles). No percentage bar,
  and a knob lands either way.

  **Numerics: nothing moves.** A prefetch changes which bytes are where and when,
  never what a kernel computes: predicted bytes become authoritative only when the
  exact router selects them and the ordinary planner adopts them
  (`ExpertPrefetchRing.swift:4-11`, `RealForwardRunner.swift:6509-6517`). Golden
  identical on both boxes and both profiles is the bar and a difference is a defect,
  never a recapture. The adoption must also leave the **pool's** plan sequence
  identical, which is checked, not assumed: a `SHRIKE_ROUTE_TRACE` capture under the
  candidate replays to the same miss counts as production within v13's +/-1
  ([v13-the-turn.md](v13-the-turn.md) "## Task 4").

  **The knob.** `SHRIKE_PREDICTIVE_PREFETCH` already exists and already fails closed
  on anything but `1` (`RealForwardRunner.swift:912-913`), as do `SHRIKE_PREFETCH_TOP_M`
  (`:916-921`) and `SHRIKE_PREFETCH_PROBE_DISTANCE` (`:2030-2031`). Step 2's two
  changes take sub-knobs in the same family: the ring's slot count and the in-flight
  join. The effective mode is **not printed anywhere today**:
  `prefillGapLeversDescription` (`:377-408`) prints `overlap=`, `residency=`,
  `sweep=`, `cache_layout=`, `expert_io=` and `protect=`, so a `prefetch=` field
  rides with the task exactly as `protect=` rode with v13 Task 4, read from the same
  static parse the runner uses.

  **The rig and the rows.** The step-zero streaming rig
  (`loop-files/step0-stream.sh` plus `step0-stream-client.py`) is what produces a
  per-token wall, and it belongs under `tools/` the moment this task needs it in the
  tree. Rows per arm, exactly the fields step zero used: `prefill_s`, `decode_s`,
  `decode_tok_s`, `expert_hit_rate_decode`, `expert_misses_decode`,
  `hit_fixup_layers`, `io_ms`, `io_fixup_wake_ms`, `io_fetch_ms`, `io_hidden_pct`,
  plus the `Shrike kernel role=` and `Shrike gap` blocks, which are the only place
  the miss window is visible. **No new counter is needed for the verdict**; a
  prefetch hit rate (predictions issued, adopted, wasted) is needed for the
  attribution and is the one counter Step 2 may add.

  **Tests (RED first, host-only; no device suite for a scheduling-only change).**
  - `expertPrefetchRingStagesOnlyAbsentUnqueuedExperts`: `begin`'s filter
    (`ExpertPrefetchRing.swift:43-52`) drops residents, drops experts already staged
    for that layer, dedupes, and caps at the free slot count.
  - `expertPrefetchRingGeometryFollowsTheTopK`: the ring's slot count from
    `makePredictivePrefetch` (`:908-929`) at the new default and under the override,
    failing closed on a bad value with the allowed range in the message.
  - `expertPrefetchRingJoinsAnInFlightPrediction`: the new awaiting variant of
    `readyBuffers` returns bytes for a submitted-but-not-completed operation, leaves
    another layer's slots alone, and never awaits on an all-hit layer.
  - `expertPrefetchRingReclaimKeepsInFlightSlots`: `reclaimTerminalSlotsUnlocked`
    (`:113-124`) frees completed, failed and empty slots and never a submitted or
    in-flight one; `consume` (`:102-111`) clears exactly the named experts.
  - `prefillGapLeversDescriptionReportsThePrefetchMode`: the existing assertion
    (`tests/Shrike/Core/Infrastructure/Streaming/PreadExpertStreamerTests+CachePlanning.swift`, the shape v13
    Task 4 used) plus the new field.
  - The pricing tool ships its own `--self-test` on a tiny synthetic trace, run from
    the tool and not from `swift test`, as `tools/expert-pool-replay.py` does.

  **Files.** Step 1: `tools/prefetch-coverage.py` (new: the `SHRIKE_PREFETCH_TRACE`
  join, the history-only baselines over a route trace, the modelled prize through
  step zero's coefficients, `--self-test`), and the step-zero streaming rig moved
  under `tools/`. Step 2, only if the stop passes:
  `sources/Shrike/Runtime/Inference/ExpertPrefetchRing.swift` (the awaiting join at
  `:87-100`, the geometry), `sources/Shrike/Runtime/Inference/RealForwardRunner.swift`
  (`makePredictivePrefetch` `:908-929`, the `begin` call site moved from `:6738-6743`
  to just after the demand fetch is begun at `:6529-6537`,
  `prefillGapLeversDescription` `:377-408`, the prefetch counters if any).
  **Unchanged:** every `.metal` file and kernel body, `expert_io.c` and the reader,
  `PreadExpertStreamer` and the pool's policy, the planner's placement logic, the
  prompt cache. **Lint:** `encodeDecodeRoutedMoE` is already over the
  `function_body_length` threshold, so its baseline entry stands; check
  `swiftlint lint --write-baseline` if any baselined reason string embeds a line
  count that this task's edits move (v13 hit this on a different function five times).

  Steps:

  - [x] Step 1 (measurement, no code in `sources/`): the three offline arms above,
        (i) the history-only baselines on the three archived traces, (ii) one
        `SHRIKE_PREFETCH_TRACE` capture per shape on the mini and the coverage join,
        (iii) the zero-code `SHRIKE_PREDICTIVE_PREFETCH` A/B at top-M 4 and 8 on the
        three answers, paired in both orders. Deliverable: the tool, one table per
        shape (recall, nonresident precision, full-layer coverage p at distance 1 and
        2, the modelled prize), and the stop applied in writing.
        **DONE 2026-09-06.** (i) The router-free predictors are dead at 128 slots on
        all three traces: full-layer coverage 0.000 for the previous token's set,
        0.018 to 0.040 for the last-8 union, 0.019 to 0.023 for the previous layer's
        set (the replay's decode misses 6,689 / 9,468 / 11,398 match the box).
        (ii) MEASURED on the mini (the v13 close binary, `prefix(M)` exactly as the
        runtime applies it; the probe costs ≈ 2 ms per token of wall and all six
        answers were byte-identical to step zero's), card / 300 / 1k:

        | distance, M | full-layer coverage p | per-miss recall | nonresident precision | fetches per token |
        | --- | --- | --- | --- | --- |
        | d = 1, M = 8 | **0.462 / 0.442 / 0.428** | 0.583 / 0.561 / 0.554 | 0.471 / 0.423 / 0.407 | 35.3 / 37.8 / 36.0 |
        | d = 1, M = 4 | 0.169 / 0.164 / 0.158 | 0.298 / 0.279 / 0.283 | 0.692 / 0.625 / 0.625 | 12.3 / 12.7 / 12.0 |
        | d = 2, M = 8 | 0.367 / 0.342 / 0.337 | 0.475 / 0.445 / 0.453 | 0.341 / 0.301 / 0.289 | 37.0 / 39.8 / 38.9 |
        | d = 2, M = 4 | 0.153 / 0.143 / 0.138 | 0.253 / 0.233 / 0.240 | 0.525 / 0.454 / 0.450 | 12.8 / 13.8 / 13.3 |

        Missing layers per token 18.3 / 18.8 / 17.4; absent-set sizes on the card
        1: 2,401, 2: 919, 3: 373, 4: 176, 5 or more: 113. **The stop clears**: p is
        above 0.10 and precision above 0.21 on all three shapes at both M and both
        distances. The modelled prize (p x 11.5 ms, MODELLED) is 4.9 to 5.3 ms per
        token at d = 1, M = 8 (7.1 to 7.2 % of the wall), 1.8 to 1.9 at M = 4; the
        4-to-36 % spread of the draft closes at ≈ 7 %. (iii) carries **no verdict**:
        every `SHRIKE_PREDICTIVE_PREFETCH=1` arm answered a different text (the
        defect below), so its rows are the shipped scheme's timing floor only: on
        the three answers top-4 cut the window 1 to 3 ms per token at −1 to +2 %
        tok/s, top-8 added 1 to 4 ms at −7 to −10 % (its 37 ring reads per token
        steal bandwidth from the demand reads); production repeated to 0.1 tok/s
        with identical miss counts in both orders. Tool: `tools/prefetch-coverage.py`
        (`join` and `history`, `--self-test`); the rig: `tools/decode-rig.sh`,
        `tools/decode-stream-client.py`, `tools/decode-rows.py`. Raw captures, rows
        and golden logs: `~/.claude/handoffs/archive/shrike-v14-t1/`.

        **Step 1's finding: the shipped path is not output-identical.** Under the
        default `SHRIKE_DECODE_EXPERT_EXECUTION=speculative`, `SHRIKE_PREDICTIVE_PREFETCH=1`
        changes the answer on the mini (four arms, four texts, the route traces first
        differing at the first decode position, layer 12) and fails
        `tools/golden-baseline.sh --check` on the M4 Pro on both profiles, under the
        pool and the per-slot cache layouts alike. Three golden runs pin the cause:
        `hit-fixup` execution with the prefetch is IDENTICAL, `hit-fixup` alone is
        IDENTICAL, and `speculative-validate` with the prefetch throws the runtime's
        own cross-check, "speculative routed output diverged from the classic path".
        The GPU classifies the layer's hits and misses from the residency table in
        the attention tail (`encodeResidencyClassification`,
        `RealForwardRunner.swift:3473-3487`) BEFORE the host's plan adopts the ring's
        bytes (`PreadExpertStreamer.swift:756-763`, a memcpy at plan time); the host
        then sets `specAllHit` from its own post-adoption miss count (`:6661`,
        `:6806`) and takes the speculative command's result, which computed the
        layer without the adopted expert. `gpu-residency` guards exactly this
        disagreement and fails closed (`:6583-6586`); the speculative modes do not.
        The bytes themselves are right (the offsets, the C reader's blocking
        completion and the victim selection were each read and ruled out).
        **Ruling (Davor, 2026-09-06): Step 2 opens with the fix, the fixup follows
        the GPU's classification.**
  - [ ] Step 2 (code; the stop passed): **first the fix**: in the speculative
        modes the fixup's partition follows the GPU's classification (its misses are
        the plan's misses plus the adopted experts, which the fixup computes from
        their now-resident slots with no storage read; `specAllHit` only when the
        GPU saw all hits; a fail-closed check that the two views differ by exactly
        the adopted set, as `gpu-residency` already does), the plan carrying its
        adopted indices, and the prize becoming the hidden read rather than the
        speculative shortcut on adopted layers. RED first: a host-only test of the
        reconciliation rule; the failing golden under the knob is the device-level
        RED and turns IDENTICAL on both boxes. Then the ring sized to the layer's
        top-k, `begin` moved to plan time so the predicted reads run beside the
        demand read (K = 2, the probe's own arm), and the in-flight join so a
        correct-but-late prediction is awaited instead of re-read. Host tests RED
        first. Gates 1 to 4, plus a filtered ThreadSanitizer run over the ring's suite
        (the ring is lock-guarded shared state across the reader's threads, v13 Task
        2's precedent).
        **The fix LANDED** (commit "decode: the fixup follows the GPU's residency
        classification"): `ExpertCachePlan.adopted` (the indices the planner
        counted as hits from prefetched bytes), `DecodeExpertPartition.populate`
        taking them as fixup misses, and in the routed encoder the speculative
        modes reading the GPU's miss list back as a fail-closed check, the hit
        split running whenever the fixup has work, and the partition's miss count
        driving the all-hit branch and `specAllHit` while the storage miss count
        keeps the I/O bookkeeping; `gpu-residency`'s guard expects the plan's misses
        plus the adopted set. Two host-only tests RED then GREEN
        (`adoptedPrefetchesJoinTheFixupMisses`,
        `plannedCacheReportsAdoptedPrefetchesBesideItsMisses`). Golden on the M4
        Pro: speculative with the prefetch at top-8 short + long IDENTICAL (was a
        mismatch on both), the default launch IDENTICAL, speculative-validate with
        the prefetch IDENTICAL (was the cross-check throw). The speculative modes,
        the shipping default included, now carry the fail-closed residency check
        `gpu-residency` always had: a disagreement between the GPU's classification
        and the plan beyond the adopted set fails the generation instead of running
        a different partition. The mini's golden with the knob off and on is the
        deploy's gate.
        Landed 77dd587; the mini's golden IDENTICAL with the knob off and on.
        **The zero-code A/B on the fixed binary (mini, 18 lifetimes, every answer
        identical, the follow-ups unmoved): production 13.70 / 14.13 / 14.48 tok/s on
        card / 300 / 1k, top-4 −2.4 / −1.9 / −3.0 %, top-8 −7.1 / −6.6 / −7.5 %, the
        sign in both orders on every shape: real, free, and a loss.** Davor's ruling:
        instrument before deciding. **The instrument** (the fixup's kernel role
        `moe_phase1_miss_fixup_phase2_adopted` on adopted-only layers so the gap block
        splits the window by class, the block's cut raised to 12, the ring's issued /
        adopted / reclaimed-unadopted counters and the begin path's host time on the
        runner line as `prefetch_*`) answers the question the A/B could not, per token
        on the mini (measured):

        | shape, arm | tok/s | adopted-only layers | storage-miss layers, ms each | ring reads issued / adopted / wasted | plan ms | begin ms |
        | --- | ---: | ---: | --- | --- | ---: | ---: |
        | card, prod | 13.67 | 0 | 18.2 at 1.07 | 0 | 0.21 | 0 |
        | card, top-4 | 13.17 | 2.7 | 15.5 at 1.23 | 12.1 / 7.3 / 4.8 (40 %) | 1.09 | 0.39 |
        | card, top-8 | 12.94 | 6.1 | 12.1 at 1.64 | 34.9 / 11.6 / 23.3 (67 %) | 1.66 | 0.47 |
        | 300, prod | 14.39 | 0 | 18.7 at 1.03 | 0 | 0.21 | 0 |
        | 300, top-4 | 13.85 | 2.6 | 16.1 at 1.16 | 12.5 / 6.7 / 5.8 (46 %) | 0.96 | 0.39 |
        | 300, top-8 | 12.89 | 5.7 | 13.0 at 1.71 | 37.3 / 10.5 / 26.8 (72 %) | 1.42 | 0.50 |
        | 1k, prod | 14.43 | 0 | 17.4 at 1.04 | 0 | 0.21 | 0 |
        | 1k, top-4 | 14.04 | 2.4 | 15.0 at 1.15 | 11.9 / 6.6 / 5.3 (45 %) | 0.85 | 0.38 |
        | 1k, top-8 | 13.41 | 5.3 | 12.0 at 1.59 | 35.6 / 9.9 / 25.7 (72 %) | 1.20 | 0.46 |

        **An adopted-only layer's window does collapse** (its gap total falls below
        the block's twelfth entry, 65.9 ms over the card's 587 such layers: below 0.12
        ms per layer against 1.03 to 1.07 for a
        read), **and the drive takes it back**: the layers that still read slow from
        1.03 to 1.07 ms to 1.15 to 1.23 at top-4 and 1.59 to 1.71 at top-8, in
        proportion to the ring's reads in flight beside the demand reads (step zero's
        probe: 0.77 ms alone, 1.08 each for two overlapped). The card's ledger at
        top-4, per token, every hidden layer priced at production's 1.07 ms: 2.7
        hidden layers save 2.9 ms, 15.5 slowed reads cost 2.5, the probe's router pass
        costs 2.0 to 2.3 of GPU time in the attention tail (`attn_layer_linear` plus
        `attn_layer_kv`, 15.4 to 17.8 ms across the nine rows), and the submit gap
        grows a measured 1.0 (the adoption copy's 0.9 and the begin path's 0.4 sit
        inside it, partly absorbed by slack): +2.6 to +2.9 modelled against +2.8
        measured. At top-8 the hidden 6.1 layers save 6.5 and the 12.1 slowed reads
        cost 6.9. Under the knob `hit_fixup_layers` counts the adopted-only layers too
        (they run the hit split for the first time), so a storage-miss layer count is
        the gap block's `count`, never that field.
        So on this drive a hidden read costs the neighbouring demand reads about what
        it saves; the lever pays only if its reads stop overlapping demand reads
        (issued at plan time on all-hit layers they overlap nothing, on miss layers
        they contend with the layer's own read), the probe is fused into the router
        dispatch it duplicates (53 µs per layer today), the copy goes (a read into a
        reserved slot), and the wasted reads fall. Modelled ceiling with all four:
        +3 to +7 % tok/s, the contention model the risk. Ruling on the continuation:
        Davor's.
  **T1 CLOSED 2026-09-06 as a measured result** (Davor's ruling after the instrument):
  the fix 77dd587 and the instrument 71fc47f stay, the shipping default is unchanged,
  the redesign is a candidate task below with its four preconditions, and the chapter
  moves to the alternative (Task 2). Steps 3 to 5 did not run: the lever closed at
  Step 2's measurement. Golden identical on both boxes with the knob off and on is on
  the record for both landed commits.
  - [ ] Step 3 (numerics, not run): golden IDENTICAL on both profiles at every knob cell on
        the M4 Pro at each amend, and on the mini at the default and the candidate
        cells at each amend and at all cells at the landed commit. Plus one
        `SHRIKE_ROUTE_TRACE` capture under the candidate replayed against production's
        miss counts to +/-1, proving the pool's plan sequence did not move.
  - [ ] Step 4 (the arms, not run): the three answers as
        verdict rows, the turns and the warm second prompts and 12k as controls, each
        arm reporting the `Shrike gap` block so the prize is attributed to the window
        it was predicted to close, not just to the wall.
  - [ ] Step 5 (the rule, not run): real and free applied in writing; the default flipped by
        amend if it passes, the knob landed at its measured default either way.
  - [x] Step 6 (design doc): the Task 1 section, the After T1 block, the lever
        entries updated with what was measured, Follow-ons gained. Task review by a
        fresh reviewer, fixes folded into the owning commit.

  **Risks and what falsifies the model.**
  - **The prior art may simply be right.** `architecture.md:108-118` closed this door
    with paired measurements, and `docs/v10-implementation-plan.md:190-200` closed it
    again from the drive's side. If Step 1's coverage rate comes in under 0.10, or
    precision under 0.21, the model in the opening paragraph is wrong and the task
    lands the null. That is the expected outcome the plan is written to survive.
  - **Full-layer coverage is the multiplier, not recall, and the independence
    approximation used to get 0.26 to 0.49 from a recall of 0.44 to 0.64 is not a
    measurement.** Correlation between one layer's misses pushes the true rate up;
    a predictor that is right about the easy expert and wrong about the rare one
    pushes it down. Step 1 measures p directly and the modelled range is discarded
    the moment it does.
  - **A wasted prefetch is not free.** At precision below 0.21 the extra reads exceed
    the drive's measured headroom and every demand read slows down; the K sweep
    (1.076 ms at K = 2 against 0.949 at K = 1) is the cost even when the prediction
    is right. Any arm that improves `decode_tok_s` while `io_fetch_ms` per expert
    rises is reporting a trade, not a win.
  - **The lead time may still be short.** Moving `begin` to plan time buys one demand
    fetch of lead (about 1.08 ms) at the price of contending with that same fetch. If
    the join still finds most predictions in flight rather than complete, the scheme
    is paying for reads it cannot use, which is the k = 1 demand-contention failure
    `architecture.md:102-104` named. The prefetch counters in Step 2 are what make
    that visible instead of inferred.
  - **The capture taxes what it measures.** `SHRIKE_PREFETCH_TRACE` also enables the
    second router GEMV per layer per token (`RealForwardRunner.swift:2023-2025` gates
    both), so a traced run's walls are not comparable to production's; only the
    predictor's accuracy comes out of that arm. Likewise the trace's write path is a
    synchronous `write(2)` per layer per token, measured nil on decode in v13 Task 4
    and still not run on a verdict arm.
  - **The replay sees misses, not exposure.** v13 Task 5 was overruled by the box
    after the replay predicted the miss count exactly and missed a fetch-exposure
    cost entirely. Here the replay's job is narrower (the plan sequence is unchanged)
    and the exposure question is answered by the `Shrike gap` block on the box.
  - **Memory and the box.** The ring at 8 slots is 8 x 1,769,472 B = 14.2 MB of
    shared storage on top of the pool's 9.06 GB; `--ram-budget 8G` stays and
    `memory_pressure -Q` is checked before every launch. The mini is production: one
    model process at a time, every arm relaunches the server on 8081, Turbo on 8080
    is never touched.

### Task 2: T2, the miss path's host and driver windows

- [ ] **T2: a missing layer pays two windows that have nothing to do with the bytes:
  the routed submit gap, `moe_spec_routed` to `moe_phase1_hit`, **3.86 / 3.95 / 3.91
  ms per token** on the card / 300 / 1k answers (0.21 ms per missing layer), of which
  host-late 2.01 / 1.98 / 1.90 (the hit split's command committed after the spec
  command's GPU end), driver 0.30 / 0.30 / 0.27 and queue 1.57 / 1.67 / 1.75; and the
  post-completion wake `io_fixup_wake_ms`, **2.77 / 2.85 / 2.91 ms per token** (0.15
  per missing layer), the interval from the expert read landing to the fixup command
  starting on the GPU, which the fixup's encoded event wait spends by construction.
  Together **6.6 to 6.8 ms per token, 9 % of the wall**, all MEASURED on the
  instrumented build's production arms (b39d937, `step2-instrument/`, the runner line
  and the twelve-transition gap block), none of it bytes, none of it numerics.
  Planning inside the submit gap is 0.21 ms per token (`cache_plan_ms`), the top-k
  readback 0.009, the storage submission-to-start 0.55, so what fills the host-late
  2.0 ms is the rest of the missing layer's host path between the readback and the
  hit split's commit: the pin, the fetch's submission to the reader, the hit split's
  argument buffer (`MoE.makeRoutedArgumentBuffer` allocates a fresh `MTLBuffer` per
  missing layer where the fixup reuses one, `makeReusedRoutedArgumentBuffer`), the
  active-slot write, the phase-1 subset encode and the commit. Which of those is the
  0.11 ms is the first measurement. **The mini decides**, and a measured null is a
  result.**

  **The mechanism, verified in the tree** (every citation against the branch).
  - A missing layer costs four command buffers where an all-hit layer costs two: the
    attention tail (router and classification), the speculative routed command, the
    hit split (`encodeRoutedPhase1Subset` on a fresh command buffer, committed
    before the fetch is awaited) and the fixup (`buildAndCommitMissFixupCommand`,
    which encodes the event wait, `encodeWaitForEvent`, then phase 1 for the misses
    and phase 2). Each commit pays the driver's commit-to-start latency once: the
    submit gap's driver plus queue terms are 1.9 ms per token, about 0.1 ms per
    missing layer, the price of the hit split's own command.
  - The host path between the readback and the hit split's commit runs on the layer's
    critical path: `planRoutedExperts`, `pinRoutedExperts`, `beginFetchRoutedExperts`
    (the submission into the C reader's batch slot, `submit_batch`, which blocks
    when both published batches are busy), `routedExpertBuffers`, the argument
    buffer allocation, `writeActiveSlots`, the encode, the commit. Only the plan and
    the readback are timed today.
  - The wake is the event: the reader thread signals the shared event when the batch
    lands, the driver wakes the parked fixup command, the GPU starts it 0.15 ms
    later. The host never waits (`io_host_waits` 0, `io_host_waits_avoided` every
    layer). `SHRIKE_EXPERT_IO_SYNC=host` is the shipped A/B for the other way round
    (the host awaits the read and commits the fixup then; v10 T3 measured a host-spin
    late commit null on the M1, not on the mini). The phase-1 kernels carry an
    `io_status` early-return guard (`moe_io_ready`), not a poll.

  **Steps.**
  - [x] Step 0 (zero code): the sync mode A/B on the mini, `SHRIKE_EXPERT_IO_SYNC=host`
        against the default `event`, the three answers paired in both orders, the
        wake and the submit gap per token the readings, tok/s the verdict. Answers
        whether the event's signal-to-start latency is the floor or the host's
        commit-to-start is lower on this box.
        **DONE 2026-09-06, a measured NULL** (12 lifetimes, every answer identical,
        `~/.claude/handoffs/archive/shrike-v14-t2/step0/`): host against event, means,
        card −1.0 %, 300 −0.5 %, 1k +0.2 %, inside the event pairs' own drift (0.3 to
        1.0 %). The terms move and cancel: under `host` the wake disappears (it is the
        host's wait now, unmeasured by that counter) but the miss window grows 0.7 to
        0.85 ms per token on the card and the 300 (the host's wake plus commit-to-start
        replacing the event's signal-to-start, 1.07 to 1.13 ms per missing layer) and
        the submit gap's host-late drops 0.45 (2.0 to 1.55: the event-driven fetch
        submission's own host cost, 0.025 ms per missing layer). The event's wake is
        not recoverable by the sync mode on this box (v10 T3's M1 null repeats on the
        mini); the 0.45 ms per token of event-driven submission cost is a named term
        for Step 1's split.
  - [x] Step 1 (instrument, no scheduling change): per-stage host timers on the missing
        layer's path (pin, fetch submission, argument buffer, encode-and-commit of the
        hit split, the fixup's encode-and-commit) on the runner line, one rig pass, the
        0.11 ms per missing layer split into named terms. The tools read them.
        **DONE 2026-09-06 in three rounds** (the host stages; the two commands' commit
        stamps against their kernel and GPU starts; the router wake), each a
        production lifetime per shape on the mini
        (`~/.claude/handoffs/archive/shrike-v14-t2/step1/`). Per token, card / 300 /
        1k, MEASURED:

        | term | card | 300 | 1k | per layer |
        | --- | ---: | ---: | ---: | --- |
        | the router wake (`path_router_wake_ms`: the status spin's return after the router command's GPU end, every layer) | 6.46 | 6.32 | 6.45 | 0.16 on all 40 |
        | the timed host stages (readback 0.009, plan 0.21, pin 0.046, submit 0.09, argument buffer 0.14, hit split encode and commit 0.14, fixup build 0.11) | 0.75 | 0.77 | 0.75 | 0.041 per missing layer |
        | the hit split's commit to kernel start (the driver's pickup) | 0.58 | 0.56 | 0.53 | 0.031 per missing layer |
        | the hit split's kernel start to GPU start (the launch) | 1.89 | 2.18 | 1.94 | 0.10 to 0.12 per missing layer |
        | the fixup's commit to kernel start | 0.71 | 0.70 | 0.65 | 0.038 per missing layer |
        | the fixup's wake (`io_fixup_wake_ms`, the read landing to the GPU start) | 2.85 | 3.38 | 2.81 | 0.15 per missing layer |
        | the submit gap, for reference (Step 0's rows: host-late 2.0 + driver 0.3 + queue 1.6 to 1.75) | 3.95 | 4.17 | 3.86 | 0.22 per missing layer |

        The missing layer's timeline on the card, from the router command's GPU end:
        the host wakes 0.16 ms late (the driver marks the command complete that long
        after the GPU finishes; v9's spin removed the thread-park wake of ≈ 0.175,
        this is what the spin cannot see past), spends 0.041 in the timed stages and
        about 0.08 elsewhere (the previous layer's pending command finished, the
        partition and its classification check, the scratch handling), commits the
        hit split ≈ 0.28 after the router's end; the driver picks it up 0.03 later and
        the GPU launches it 0.10 to 0.12 after that, the speculative command's 0.23
        overlapping the first part: the submit gap's 0.22. The fixup is committed with
        its event wait 0.11 after the hit split, picked up 0.04 later, and started by
        the GPU 0.15 after the read lands. So of a missing layer's ≈ 0.37 ms of
        windows (the 0.22 submit gap and the 0.15 wake), host code is 0.04 and 0.33 is
        driver and launch latency around four commands.
        **The levers, sized from the split (MODELLED, per token):** (A) the hit split
        folded into the speculative command, which already computes the GPU's hits on
        an all-hit layer and can compute them on every layer while the fixup keeps the
        misses and phase 2: removes its argument buffer, encode and commit (0.28), its
        pickup (0.55) and its launch (1.9 to 2.2), ≈ 2.8 to 3.1 ms, 4 %, no visibility
        risk, the T1 fix's partition rule already in place; (B) the router wake: the
        host polls a word the router kernel writes last instead of the command's
        status, up to 18 x 0.16 = 2.9 ms on the missing layers (the all-hit layers'
        wake overlaps the speculative command), the risk a CPU's view of an in-flight
        kernel's write on shared memory, to be probed on the mini with a test kernel
        before it is built; (C) the fixup's wake: a bounded GPU-side spin on the
        `io_status` word inside a command that is already running, so the read's
        landing costs no launch, up to ≈ 2 ms, the same visibility risk in the other
        direction (the runtime's `moe_io_ready` guard relies on it after a command
        boundary, not during one); (D) the argument buffer reused like the fixup's,
        0.1, subsumed by A. Order: A, then B behind a probe, then C behind a probe.
  - [ ] Step 2 (code, by measured size): lever A first, the hit split folded into the
        speculative command (phase 1 for the GPU-classified hits on every layer, the
        fixup computing the misses and phase 2 as it does today; a knob; host tests
        RED first; golden identical on both boxes; the arms). Then lever B behind a
        mini probe of a kernel-written word's visibility latency to a polling host,
        then lever C behind the reverse probe. Gates 1 to 4 per commit.
  - [ ] Step 3 (numerics): golden IDENTICAL on both boxes and both profiles at every
        knob cell; a difference is a defect, never a recapture.
  - [ ] Step 4 (the arms, the mini, one binary per round): the three answers as verdict
        rows, the turns and the warm second prompts as controls, the gap block
        attributing the saving to the window it was predicted to close.
  - [ ] Step 5 (the rule): real and free applied in writing; defaults flipped by amend
        if they pass, the knobs landed at their measured defaults either way.
  - [ ] Step 6 (design doc): the Task 2 section, the After T2 block, the lever entries.
        Task review by a fresh reviewer, fixes folded into the owning commits.

  **The decision rule.** Real: the sign holds on the three answers' `decode_tok_s`
  across paired runs in both orders. Free: the turns, the warm second prompts and
  `memory_pressure -Q` unmoved, golden IDENTICAL. No size floor. **Numerics: nothing
  moves**; every change here is when a command is committed and what buffer it
  reads its arguments from, never what a kernel computes.

  **Files.** Step 1: `sources/Shrike/Runtime/Inference/RealForwardRunner.swift` (the
  routed encoder's timers and the runner's totals), `sources/ShrikeServer/Core/ServerInference.swift`
  (the runner line), `tools/decode-rows.py`. Step 2: the same encoder,
  `sources/Shrike/Kernels/MoE/MoE.swift` (a second reusable argument buffer).
  **Unchanged:** every `.metal` file, the reader, the pool, the planner, the prompt
  cache. **Lint:** `encodeDecodeRoutedMoE` is baselined; regenerate on growth.

## Candidate tasks (not scheduled)

- **The prefetch redesign (T1's candidate, re-price before building).** T1 measured
  the mechanism sound (an adopted-only layer's window collapses below 0.11 ms) and
  the shipped form a loss (−2 to −3 % at top-4, −7 % at top-8) because the drive
  serves the ring's reads beside the demand reads and slows them by about what the
  hidden layers save, with 40 to 72 % of the ring's reads wasted. It pays only if
  all four hold: (1) the predicted read lands in a reserved pool slot, no host copy
  (today 0.12 ms per adopted expert in the submit gap); (2) it is issued where it
  overlaps no demand read, at plan time on all-hit layers, never beside a miss
  layer's own read; (3) the probe is fused into the router dispatch it duplicates
  (53 µs per layer of GPU time today, 2.1 ms per token); (4) the wasted reads fall
  well below half (top-4's precision 0.63 to 0.69 is the better start). Modelled
  ceiling with all four: +3 to +7 % of tok/s; the contention model is the risk and
  the zero-code distance-2 probe (the same ring with `SHRIKE_PREFETCH_PROBE_DISTANCE=2`,
  three lifetimes) tests it before any line is written. Re-price on the box as it
  stands after Task 2, never on this model.

- **What the pool has left at the prefill-to-decode boundary.** Belady removes 64.7 %
  of the card answer's decode misses and 60.8 % of the 300 answer's
  (`ledger/step0-profile.txt`), which is 16.5 ms per token modelled through step
  zero's slope, but v13 replayed every bounded eviction rule within about 5 % of
  production and found recency already at 97.6 % of an answer's reuses. What is not
  exhausted is the state the prompt hands to the answer: v13 Task 5's resident-first
  sweep took the first cut of it and was worth 0.5 to 0.9 s of the first turn's
  decode. The replay tool and the four archived traces price any successor offline
  before a line is written.

- **The LM head the server never fuses.** `head_logits` is 4.82 to 5.05 ms of GPU per
  token (6.6 to 7.2 % of the wall) and `head_fused_ms` is 0.000 on every step-zero
  row. The fused greedy head exists and decode already asks for it
  (`RealForwardRunner.swift:3165-3176`, `:2301`), but the server hardcodes
  `forceLogitsHead: true` (`ServerInference.swift:661`, `:716`); v10 recorded the
  same observation (`docs/v10-implementation-plan.md:220-222`). The fused path cannot
  remove the head GEMV, only the full logits writeback and a dispatch, and it is a
  behaviour change (no logits means no sampling and no logprobs), so the task is
  "measure the fused path's saving on a greedy request, then decide whether the
  server can pick per request", not a flip.

- **The drive's idle-gap tax as a standalone.** 0.175 ms on the first read of every
  missing layer, 3.2 ms per token, 4.3 %. Not separately collectable (the only way
  to stop paying it is to have a read already in flight, which is T1), and the
  keep-warm variants are measured NULL to negative on the mini
  (`docs/v10-implementation-plan.md:170-172`). Recorded so it is not re-proposed.

## Follow-ons (not scheduled)

- DONE in 71fc47f: the `Shrike gap` log prints twelve transitions instead of eight
  (`ServerInference.swift`, the `prefix` in `emitKernelDiagnostics`); step zero's 2.2
  to 3.0 ms per token below the old cut is now listed.
- From T1's review: `hit_fixup_layers` counts adopted-only layers under the prefetch
  knob (they run the hit split), so `io_ms / hit_fixup_layers` mixes classes there;
  a per-class counter would let the estimator mean one thing. The ring's counters
  have no host test (`begin` needs a `Model`; a seam would allow one). The
  speculative modes' fail-closed residency check is symmetric; a directional form
  (the GPU's misses a superset of the plan's, equal beyond the adopted set) would
  keep the benign direction alive if it ever occurred.
- The runner's own overlap accounting and the regression disagree about how much of
  the fetch is exposed: `io_hidden_pct` reads 32.2 / 33.8 / 33.2 % hidden (so about
  14.5 ms per token exposed) while the kernel gap and the regression both put the
  whole 18 to 20 ms on the wall. One of the two definitions is not measuring what its
  name says; worth settling before either is used in a verdict.
- The prompt cache's settle after a request whose prompt has no cached prefix
  re-prefills the whole prompt in the background (v13 Task 4's finding, visible again
  in step zero's `settle chunk` rows at 2,706 to 2,982 misses); a cache-chapter item.
- v13's open follow-ons stay in [v13-implementation-plan.md](v13-implementation-plan.md):
  the GDN chunked scan below its 64-row gate, the routed matrix gate's `> 32`, the
  reader's `min(count, threads)` publication signal, the resident sweep's route-build
  host on a tiny chunk, the cold first request's prefill under the balanced recency
  composition, the expert-cache policy parse in `PreadExpertStreamer.init`, the
  protection-fallback counter, the cold first request's +0.27 s observation, and the
  prompt cache's interior snapshots.
- v12's prefill kernel follow-ons stay in [v12-prefill-matrix-kernels.md](v12-prefill-matrix-kernels.md);
  v11's attention work stays in [v11-kv-attention-inner-loop.md](v11-kv-attention-inner-loop.md).
