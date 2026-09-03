# v13 implementation plan — the turn

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Status of record for [v13-the-turn.md](v13-the-turn.md). Checkboxes here are
the only status tracking. Commit SHAs in the verdicts are the branch's current
ones; every review fix is folded into its owning commit (rebase and amend,
never fixup commits), so the SHAs settle only once the branch stops being
rebased — the subjects are the stable names.

**Goal:** Cut the wall of a conversation turn on the mini at the shapes it runs
— first-turn and tool-round prefills of 300–2k tokens (5.5 / 8.1 / 11.2 s warm
at step zero), follow-up turns (1.4–2.1 s), and decode on a tools context (12.5
tok/s) — one measured lever at a time, each landing as a knob whose default is
the measured winner.

**Architecture:** The runtime is v12's (matrix prefill kernels, the grouped
routed GEMM, the two-tile pipeline, the alternating expert sweep, the prompt
cache). Each task changes scheduling or policy behind a `SHRIKE_*` knob, is
measured on the step-zero rig (a fresh server per pair, `settle_done` before a
warm send), passes the five gates, and keeps golden identical unless it runs
different kernels on a chunk.

**Tech Stack:** Swift 6.3, Metal 4, swift-testing, the `SHRIKE_RUNNER_STATS` /
`SHRIKE_KERNEL_STATS` counters, the step-zero rig (this session's
`turn-prompts.py` / `turn-rig.sh` / `turn-summary.py`, to be moved under
`tools/` by the first task that needs them in the tree).

**Spec:** [v13-the-turn.md](v13-the-turn.md)

## Global constraints

- macOS 26+, Swift 6.3+; never two model processes (`pgrep` check first); the
  mini's server on 8081 is production. Restarts and deploy actions are
  approved; `tools/mini-deploy.sh` copies binaries + `*.bundle` directories by
  default and restarts the server only when passed `--restart`.
- Five gates per code commit: release build with zero warnings; `swiftlint lint
  --strict --baseline .swiftlint-baseline.json`; markdown link check;
  `swift test --no-parallel`; the same under ThreadSanitizer with
  `env TSAN_OPTIONS=suppressions=tsan-suppressions.txt`.
- Numerics: a scheduling or policy change is golden IDENTICAL on both boxes and
  both profiles (`tools/golden-baseline.sh --check`); a change that runs
  different kernels on a chunk follows v12's policy (2e-2 against the fp32
  reference, golden recaptured once per box with before/after digests in the
  verdict). Never recapture for an unexplained mismatch.
- Ledger protocol per task: the step-zero rig on the mini — the 300 / 1k / 2k
  warm pairs (the verdict rows), the turn-2/turn-3 rows, and the arm the task
  names as its risk (e.g. a long answer between requests); one send per server
  lifetime for whole-chunk rows, `settle_done` before a warm second send, a
  distinct prompt per arm. Rows appended to the design's ledger. The M4 Pro is
  the check.
- Comments: none unless a genuinely non-obvious why (repo rule).

## Tasks

### Task 0: T0 — the expert sweep's parity carried across requests

- [ ] **T0: below 4,096 tokens P15's alternating sweep never fires, so every
  request's expert sweep starts ascending into a pool the last request left at
  the other end.** The parity is the chunk index within one prompt —
  `prefillChunkSweepIsDescending(startPosition:chunkTokens:)` is `(startPosition /
  chunkTokens) % 2 == 1`
  (`sources/Shrike/Runtime/Inference/RealForwardRunner.swift:493-497`), computed
  per chunk at `:2324-2326` and passed through `buildPrefillRoutes` (`:4898-4941`)
  to `PrefillMoEGrouping.groupTokenExpertPairs(… descending:)`, which flips only
  the expert-level sort key
  (`sources/Shrike/Kernels/Prefill/MoE/PrefillMoEGrouping.swift:141-148`). A
  qwen36 session takes `chunkTokens = 4_096`
  (`sources/ShrikeServer/Core/ServerInference.swift:710-713`;
  `RuntimeConfiguration.swift:218`), so **a 300 / 1k / 2k prompt is one chunk,
  index 0, always ascending** — while the pool holds the previous request's
  last-swept experts, its highest physical offsets. Step zero measured the cost:
  first-chunk hit rate **11.0 / 9.2 / 8.9 %** with 6,875 / 8,224 / 8,527 misses
  per warm request, 11.8–14.6 GB of `F_NOCACHE` reads — the ≈ 4.5 s intercept
  under every first-turn and tool-round prefill
  ([v13-the-turn.md](v13-the-turn.md):25-27, `:36-44`). This task
  carries the parity **across** requests: remember the direction the last chunk
  swept, start the next request's first chunk in the opposite one. First because
  it is the largest term at these shapes, it is a comparator flip behind an
  existing knob, and it changes no arithmetic. **The mini decides.**

  **The decision rule, in two lines.** `SHRIKE_PREFILL_SWEEP=alternate|fixed|carry`
  lands either way — a third value on the existing knob, not a second knob, so
  `fixed` (P15's A/B) and `alternate` (today's default) stay exactly what they
  are and no combination is contradictory; it prints in the projection-path
  line's `sweep=` field (`prefillGapLeversDescription`, `:271-290`, emitted at
  `ServerInference.swift:821-824`). `carry` becomes the code default **only if**
  the mini's 300 and 1k warm walls each improve by **≥ 5 %** on the same
  binary and 2k does not regress (≥ −1 %; its gain is reported, not required —
  the model puts it near an exposed-fetch floor), with golden IDENTICAL, turn 2 /
  turn 3 and the 12k control unmoved (± 1 %); otherwise the default stays `alternate` and the rows are recorded
  (P8's precedent — a measured null is a result).

  **The step-zero rows this task is anchored on** (mini, build 3774ef1,
  production launch, the warm arm of each pair; walls and hit rates from
  [v13-the-turn.md](v13-the-turn.md):23-29, the rest read off the
  same server logs with `turn-summary.py` and the `Shrike kernel role=` lines).

  | warm arm | wall | prefill_s | routed tiles | routed GPU | prefill hits / misses | fetch total | ms / expert | `routed→routed` (host) |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 305 new | 5.459 s | 5.037 | 982 | 1,516.5 ms | 851 / 6,875 (11.0 %) | 3,916.3 ms | 0.560 | 2,246 (2,001) |
  | 1,085 | 8.054 | 7.616 | 1,147 | 2,505.8 | 833 / 8,224 (9.2 %) | 4,660.5 | 0.559 | 2,016 (1,820) |
  | 2,125 | 11.209 | 10.729 | 1,188 | 3,911.4 | 830 / 8,527 (8.9 %) | 4,830.2 | 0.560 | 880 (759) |
  | turn 2 (38 new) | 2.128 | 1.659 | 427 | 567.4 | 1,708 / 1,579 (**52.0 %**) | 1,006.2 | — | 406 (355) |
  | turn 3 (36 new) | 1.376 | 0.910 | 399 | 482.6 | 2,582 / 462 (**84.8 %**) | 410.7 | — | 52 (26) |

  Fetch total is `io_fetch_ms × 8` (the P16 convention, `ServerInference.swift:2023`,
  divisor `result.newTokens` = 8 at `:1957`); ms/expert divides it by prefill +
  decode misses. **Three lengths give 0.560 / 0.559 / 0.560 ms per 1.6875 MiB
  expert** (3.02 GB/s at the bounded reader's four threads) — an independent
  confirmation of P16's marginal 0.630 ms at 12k, and the price of every hit this
  task buys. Two facts fall out. **The exposed fetch is the fetch above the routed
  GPU:** `fetch − routed GPU` = 2,400 / 2,155 / 919 ms against a measured
  `routed→routed` of 2,246 / 2,016 / 880 — ratios 0.936 / 0.936 / 0.958, so
  `exposed ≈ 0.94 × (fetch − routed GPU)` and the span model closes to ≤ 7 %.
  **The pool does hold a useful working set across requests:** turn 2 needs only
  82 experts per layer (3,287 / 40) against 128 slots, never overflows, and hits
  **52.0 %** — almost exactly the 128/256 resident fraction. The 9–11 % is not an
  empty pool; it is the sequential-scan pathology, and it appears only once demand
  (193–234 per layer) exceeds the pool.

  **What is in the pool when a request ends.** Each layer has its own 128-slot
  streamer (`ModelExpertIO.swift:99-103`; `--ram-budget 8G` snaps to 128,
  `RuntimeConfiguration.swift:142`, `:196-213`) under aging-LFU with an LRU
  tiebreak (`PreadExpertStreamer.swift:221`, `:1138-1151`). A sweep gives every
  expert of a layer-chunk the same use count (`:657-658`, one increment per plan),
  so the tiebreak is exactly recency and **the residents are the last 128 the
  sweep touched** — its tail. Decode displaces a little of it: measured 123 / 111 /
  103 decode misses over 40 layers = **3.1 / 2.8 / 2.6 evictions per layer, 2.0–2.4 %
  of the pool**, so the eight-token step-zero shape leaves ≥ 97.6 % intact. A
  card-length answer does not. At the warm decode hit rate 0.945–0.958 a step
  misses 0.44 experts per layer, so **512 steps evict ≈ 225 per layer = 1.8
  pool-fulls** (3.8 at v12 P17's tools-shape 0.88), and the count term then decides
  rather than recency: a decode trajectory touches its working set ≈ 4,096 times
  per layer against the sweep's one, so the tail goes first and decode's high-count
  set remains. Counts halve every 1,024 plans per streamer (`:624-627`) — roughly
  every second card — which damps that disparity, not inverts it. **This is the
  risk arm, and it is measured, not argued.**

  **Where the state lives.** A private `var prefillLastChunkDescending: Bool?` on
  the runner, written at `:2324-2326` where the direction is already computed —
  every prefill path (chunked, MTP, a re-prefill after a settle) then updates it
  with no `spanIndex == spans.count - 1` bookkeeping, and "the last chunk" is
  simply the last one executed. The read is the next request's first chunk:
  `direction(chunkIndex:) = (carried == false) != (chunkIndex % 2 == 1)` — start
  opposite the carry, alternate from there — so a multi-chunk predecessor needs
  nothing extra: only its **last** chunk's direction is remembered, which is what
  left the tail. `nil` gives ascending, identical to today. Concurrency:
  `ServerModelSession` is an `actor` (`ServerInference.swift:509`) and the runner
  is single-flight per generation (`RealForwardRunner.swift:401-403`, enforced by
  `prefillChunkState.requireClean` at `:2003`, `:2017`), so the var needs no lock.
  **`reset()` must not clear it** (`:1211-1215`): it clears KV, GDN and transient
  chunk state, all runner-owned, whereas the expert pool lives on
  `ModelExpertIO`'s per-layer streamers and survives — including the `settle_reset
  reason=no_prefix_snapshot` path (`ServerInference.swift:1626-1628`). A model
  reload rebuilds both and the carry is `nil`, correct because the pool is then
  cold. **A request served with no prefill at all** (`guard !tokens.isEmpty`,
  `:2035-2037`; `:2202`) leaves the carry alone: the pool holds decode's residue,
  which has no direction, so flipping is a coin toss and resetting to ascending is
  a hidden fourth mode the A/B cannot separate. Keeping costs nothing.

  **Order only — golden IDENTICAL is the bar.** `descending` reaches exactly one
  expression, the expert-level comparator (`PrefillMoEGrouping.swift:141-148`);
  the tie-breaks below it (`$0.token < $1.token`, `$0.rank < $1.rank`) are **not**
  reversed, so within an expert block the row order is unchanged. Each pair's
  output lands at its own slot, `route_partials[(pair.token * top_k + pair.rank)
  * D + d]` (`sources/Shrike/Metal/Prefill/prefill.metal:946`), and
  `prefill_moe_reduce_token_major` folds `r = 0..<top_k` in rank order
  (`:743-764`), so no tile order can change a value — P15's argument, and P15
  measured it: completions byte-identical between the `alternate` and `fixed`
  arms at both sizes, golden identical on both boxes and both profiles
  ([v12-implementation-plan.md](v12-implementation-plan.md), Task
  15 LANDED). Second order and named: a different expert order repacks the
  1,024-row staging waves, so a block may split elsewhere and the padded-row count
  moves — `prefill_routed_tile` GPU shifts, no output value does. P16's 24 held
  slots (`inFlightBatches = maxPendingDepth + 1` = 3 × 1 × 8,
  `PrefillRoutedTileScheduler.swift:47-48`, `:81-82`) are orthogonal: they hold
  experts of the **current** sweep and are empty at a layer-chunk's first tile;
  they only shrink the window this task reads from, which is why the gain is
  modelled at both 104 and 128 usable slots.

  **The modelled gain, and its arithmetic.** Let `d` = experts a chunk touches
  per layer ÷ 256 and `P` = usable slots. The previous ascending sweep leaves its
  last `P` experts resident, spanning `W = P/d` ranks of physical offset. A
  descending sweep entering that window hits with probability `d` per needed
  expert; each miss evicts the least-recently-used resident, which is the
  window's lowest-offset member, so the floor climbs `(1 − d)` ranks per rank the
  sweep descends. The fronts meet after `x = W/(2 − d)` ranks, by which point the
  sweep has taken `d²x` hits: **hits per layer = d·P / (2 − d)**. With d =
  193.15/256, 226.4/256, 233.9/256 (the measured requested-expert counts ÷ 40
  layers):

  | shape | d | hits/layer at P = 104 | at P = 128 | first-chunk hit rate | Δ misses | Δ fetch | modelled warm wall |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 305 | 0.754 | 63.0 | 77.5 | 32.6 → 40.1 % | 1,669–2,249 | 935–1,259 ms | 4.59 → 4.29 s (−16 to −21 %) |
  | 1,085 | 0.884 | 82.5 | 101.5 | 36.4 → 44.8 % | 2,467–3,227 | 1,382–1,807 | 6.77 → 6.37 (−16 to −21 %) |
  | 2,125 | 0.914 | 87.5 | 107.7 | 37.4 → 46.0 % | 2,670–3,478 | 1,495–1,948 | 10.50 (−6.4 %) |

  Δ fetch is Δ misses × 0.560 ms; the wall follows `exposed = 0.94 × (fetch −
  routed GPU)`. **2k is the arm the rule does not bind on**: its fetch (4,830 ms) is
  already close to its routed GPU (3,911 ms), so removing 1.5–1.9 s drives the
  exposed term to a floor and no further — that floor being an almost-all-hit
  `routed→routed`, measured on turn 3 at 0.145 ms per boundary (52 / 359) = 166 ms
  over 2k's 1,148 boundaries. The model gives 2k **−6.4 %** against a 5 %
  threshold; a partial capture there fails the rule and the default stays
  `alternate`. **The decode-churn arm's expected loss:** after a 512-token answer
  the tail is gone, so the carry arm's advantage collapses toward zero — but so
  may the *control's* deficit, since a count-protected decode set resists the
  scan's own evictions where a uniform-count tail cannot. The prediction is a
  **smaller delta on a higher pair of baselines**; the rows decide.

  **Bars (mini, warm arm, the rows above).** Warm wall **5.46 → ≤ 5.19 s**, **8.05
  → ≤ 7.65**, **11.21 → ≤ 10.65** (the rule's 5 %); the modelled 4.59 / 6.77 /
  10.50 carried as the stretch — report which was met. First-chunk
  `expert_hit_rate_prefill` **11.0 → ≥ 30 %**, **9.2 → ≥ 34 %**, **8.9 → ≥ 35 %**
  (90 % of the conservative P = 104 model). `routed→routed` **2,246 → ≤ 1,400 ms**,
  **2,016 → ≤ 800**, **880 → ≤ 350**. Long-decode arm: absolute rows both modes;
  the lever survives it if the carry arm still leads by ≥ 5 %. Turn 2 / turn 3
  walls **unmoved** (2.13 / 1.38 s ± 1 %). **12k control: byte-identical by
  construction** — on a fresh server with one send the carry is `nil`, so `carry`
  and `alternate` compute the same directions; wall **68.38 s** and
  `expert_hits_prefill` **9,546** (P16's landed rows) must not move, and a move
  means carried state leaked into a first request. Golden **IDENTICAL** both
  boxes both profiles; `memory_pressure -Q` read before and during every mini
  arm. The M4 Pro is the check (its first-chunk hit rate is low too, so it should
  move the same way); it decides nothing.

  **Tests (RED first, host-only — no Metal, no model).** In
  `tests/Shrike/Core/Kernels/Prefill/PrefillMoEGroupingTests.swift` beside
  `chunkSweepParityAlternatesByChunkIndex` (`:191-207`):
  `carriedSweepStartsOppositeThePreviousRequestsLastChunk` (carried `true` → chunk
  0 ascending, `false` → descending); `carriedSweepAlternatesFromTheCarriedStart`
  (carried `false` → desc / asc / desc over chunk indices 0,1,2);
  `carriedSweepWithNoHistoryIsAscending` (`nil` → ascending, matching
  `alternate`'s first chunk); `alternateModeIgnoresTheCarriedDirection` — for
  every carried value `.alternate` equals
  `prefillChunkSweepIsDescending(startPosition:chunkTokens:)`, so P15's A/B stays
  bit-identical; `fixedModeIsAscendingAtEveryChunkAndCarry`. In
  `PrefillRoutedTileSchedulerTests.swift` beside the existing knob tests
  (`:307-345`): `sweepModeParsesItsThreeValues` (`"alternate"`, `"fixed"`,
  `"carry"`, unset and unknown → the default) and
  `prefillGapLeversDescriptionReportsTheSweepMode` extended for `sweep=carry`.

  **Files.** `sources/Shrike/Runtime/Inference/RealForwardRunner.swift`:
  `:430-446` (`prefillSweepAlternate: Bool` → a `PrefillSweepMode` enum and its
  parser, replacing `environmentPrefillSweepAlternate`), `:493-497` (the carried
  overload beside the existing parity function), `:547-554` (init), `:269`,
  `:271-290` (`prefillGapLeversDescription` takes the mode), `:2324-2326` (the
  direction and the carry write), plus the new `var`; the two test files above
  (the description helper's signature change touches
  `PrefillRoutedTileSchedulerTests.swift:328-345`). **Lint:**
  `executePrefillChunk` (`:2191`) is already in `.swiftlint-baseline.json` at
  "currently spans 217 lines" — the reason string embeds the count, so **any**
  edit to that body makes the entry stale and the baseline must be regenerated
  (`swiftlint lint --write-baseline .swiftlint-baseline.json`) in Step 1. **The
  rig moves into the tree — yes**, every v13 task uses it: `tools/turn-prompts.py`,
  `tools/turn-rig.sh`, `tools/turn-summary.py` from this session's scratchpad,
  their hardcoded paths turned into arguments the way
  `tools/prefill-measure.sh:1-10` takes `<host> <port> <promptdir> <outdir>
  <tag>`, linked from the design doc. **Unchanged:** every kernel and `.metal`
  file, `PrefillMoEGrouping` (it already takes `descending:`),
  `PrefillRoutedTileScheduler`, `PreadExpertStreamer` and its eviction policy,
  the prompt cache, decode.

  Steps:

  - [ ] Step 1 (an implementer): the seven failing tests, then the mode enum, the
        parser, the carried parity function, the runner var and the `sweep=`
        field. `swift test --no-parallel --filter PrefillMoEGrouping` and
        `--filter PrefillRoutedTileScheduler` → FAIL then PASS. Five gates
        (release build 0 warnings; `swiftlint lint --strict --baseline`,
        regenerating the baseline if `executePrefillChunk`'s span moved;
        `tools/check-md-links.py`; `swift test --no-parallel`; the same under
        TSAN with `TSAN_OPTIONS=suppressions=tsan-suppressions.txt`). Move the
        three rig scripts under `tools/` in the same commit.
  - [ ] Step 2: `tools/golden-baseline.sh --check` on the M4 Pro — short and long
        **IDENTICAL**; a difference is a defect and never a recapture. Then
        `tools/mini-deploy.sh --restart` and the mini's golden check.
  - [ ] Step 3 (the arms, controller-run, **one binary**, `SHRIKE_PREFILL_SWEEP`
        as the A/B; `pgrep -fl 'ShrikeServer|ShrikeMac|ShrikeDecodeService|ShrikeCLI'`
        before every launch; a fresh server per pair, `settle_done` before the
        warm send, a distinct tag per arm and per box — P10's
        `resp-<tag>-<label>.json` collision): **A** `alternate` and **B** `carry`,
        `tools/turn-rig.sh pair 300|1k|2k`, tags `t0-alt-mini-<len>` /
        `t0-carry-mini-<len>`. **C** the long-decode arm — the same pair with the
        cold request's `max_tokens` raised to 512, tags `t0-<mode>-mini-<len>-d512`:
        does the gain survive a card-length answer. **D** the turns arm
        (`turn-rig.sh turns`, tX → tXturn2 → tXturn3), tags `t0-<mode>-mini-turns`.
        **E** the 12k control, `tools/prefill-measure.sh macmini 8081 <promptdir>
        <outdir> t0-<mode>-mini-12k 6k` — label **`6k`**, which is
        `tools/prefill-prompts.py`'s 12,285-token prompt (`:10-11`); its `12k`
        label is the 25,245-token one, and P16 used `2k 6k` for exactly this
        reason. Read per arm: wall, `prefill_s`,
        `expert_hits_prefill` / `expert_misses_prefill` / `expert_hit_rate_prefill`,
        `expert_read_mib`, `io_fetch_ms`, `expert_evictions` / `expert_reloads`,
        the `prefill_routed_tile` role and count, `routed→routed` total / host /
        count, `shared→routed`, busy, span, and the decode counters.
  - [ ] Step 4 (the rule): apply it to the three warm walls. `carry` becomes the
        default with `=alternate` as the A/B, or the default stays `alternate`.
        **Record the verdict either way**, with the first-chunk hit rate and the
        exposed-fetch term per arm — the direct read of whether the hits arrived
        and whether they were on the critical path. Then five gates on the landed
        tree, golden both boxes both profiles IDENTICAL, and
        `tools/mini-deploy.sh --restart` with the mini golden check.
  - [ ] Step 5: design doc — a "Task 0 — the carried sweep parity" section with
        the hit-rate model, its measured-against-modelled table and an "**After
        T0**" ledger block; the rig scripts linked. Plan: Task 0 `[x]` with the
        landed paragraph. Task review by a fresh reviewer; fixes folded into the
        owning commit (rebase and amend, never a fixup commit).

  **Risks and what falsifies the model.**
  - **The hits do not rise.** Then the pool after a request is not the sweep's
    tail, and the next measurement is what it actually is: one instrumented build
    printing, per layer, the first tile's hit count and the resident set's offset
    range at a request's start (P15 Step 1's precedent — instrument, measure,
    revert). Turn 2's 52 % says the pool is not empty, so a null here is a
    statement about *ordering*, not residency.
  - **The hits rise and the wall does not.** Then the fetch was not on the
    critical path at that length: check `routed→routed` against `0.94 × (fetch −
    routed GPU)`. At 2k the model already says the exposed term hits a 166 ms
    floor, so a 2k null beside a 300/1k gain is **predicted**, not a surprise —
    and the rule reports it without binding on it.
  - **The long-decode arm loses the gain.** Then the lever is short-answer and
    tool-round only; the verdict says so with both arms' absolute rows, and the
    default follows the short-decode arms only if the long arm does not regress.
  - **The 12k control moves.** It cannot, on a fresh server with one send: the
    carry is `nil` and `carry` ≡ `alternate`. A move is carried state leaking into
    a first request — a defect in where the var is written or reset.
  - **Golden moves.** Then the comparator is not the only thing `descending`
    reaches. The default stays `alternate` until it is found; never a recapture.
  - **The mini is production.** Every arm stops the server on 8081 and relaunches
    it; Turbo on 8080 is never touched. One model process at a time — `pgrep`
    first, every time.

## Follow-ons (not scheduled)

- The prompt cache's interior snapshots (a prompt that diverges inside a stored
  entry re-prefills in full; append-only turns are served).
- v12's prefill kernel follow-ons stay in [v12-prefill-matrix-kernels.md](v12-prefill-matrix-kernels.md).
