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
warm send), passes the four per-commit gates, and keeps golden identical unless it runs
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
- Four gates per code commit: release build with zero warnings; `swiftlint lint
  --strict --baseline .swiftlint-baseline.json`; markdown link check;
  `swift test --no-parallel`. The same suite under ThreadSanitizer
  (`env TSAN_OPTIONS=suppressions=tsan-suppressions.txt`) runs **once per
  chapter at its close**, before the merge to main (Davor's ruling at Task 1,
  2026-09-04: ≈ 50 minutes per run, most of it kernel-reference suites with no
  threads to check; a report found at close is fixed then).
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
  the check. **No size floor:** a task's rule tests only that the effect is
  real (paired runs in both orders, above the run-to-run drift) and free (no
  control row regresses, golden identical) — Davor's ruling at Task 1.
- Comments: none unless a genuinely non-obvious why (repo rule).

## Tasks

### Task 0: T0 — the expert sweep's parity carried across requests

- [x] **T0: below 4,096 tokens P15's alternating sweep never fires, so every
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

  **LANDED f3ede42 (2026-09-04): measured on the mini on one binary (the pre-flip
  build, the knob as the A/B; the flip folded by amend — the same value by a
  different route), a fresh server per pair, `settle_done` before the warm send:
  warm walls 300 tokens 5.47 → 3.80 s (−30.6 %; hits 11.0 → 58.6 %, misses 6,875
  → 3,200, `routed→routed` host 1,994 → 853 ms), 1k 8.08 → 6.88 (−14.9 %; 9.2 →
  51.6 %, 8,224 → 4,383, 1,816 → 914), 2k 11.24 → 10.80 (−3.9 %; 8.9 → 50.4 %,
  8,527 → 4,639, 782 → 486); cold first requests identical across modes to ≤ 60
  ms. Turn 2 1.91 → 1.70 s (52.0 → 62.6 %), turn 3 1.40 → 1.37. Long-decode arm
  (a 314 / 405-token answer before the warm request): alternate 3.77 / 6.62 s,
  carry 3.79 / 6.63, identical hit counts (49.5 / 43.8 %) — neutral, no
  regression. 12k control on the FIRST landed code (server walls): as the first
  request after launch 68.42 → 68.58 s (+0.23 %) with hits 9,546 → 4,817; after
  a real 300-token request 65.62 → 65.55 s with hits 9,933 → 9,869 — **the task
  review found the cause: that code re-read the carry per chunk after writing
  it, so multi-chunk prompts swept d0, d0, !d0 (all four rows decompose to
  within 0.5 %: one transition ≈ 4,773 hits); the readiness-prefill mechanism the
  first docs gave is retracted (the model loads lazily inside the first request).
  Fixed in the commit above (each chunk opposite the previous, the composition
  under test; the MTP verify path excluded from the carry); re-measured:
  the 12k first request 68.30 / 68.30 s with hits 9,546 in both modes; a
  two-chunk pair (6,381 tokens) 34.63 / 34.62 s cold (4,778 hits both) and 31.47 /
  31.43 s warm (9,340 both) — identical between modes, as they must be (a nil
  carry on a first request; a two-chunk request ends descending under either
  mode).** Bars: 300 and 1k ≥ 5 % ✓✓; 2k no regression ✓; hit
  rates ≥ 30 / 34 / 35 % ✓ (58.6 / 51.6 / 50.4, above the P = 128 model);
  `routed→routed` host ≤ 1,400 / 800 / 350 ms ✓ / ✗ / ✗ (853 / 914 / 486 — the
  brief's baselines were the gap totals, the verdict scores the exposed host
  term); turn 2/3 not regressed ✓ (turn 2 −11 %, the size of its control's own
  drift; the sign stands on 348 fewer misses); 12k wall ± 1 % ✓; the 12k
  hit-count clause: ✓ after the fix (9,546 in both modes; the first
  landed code's ✗ is the defect's footprint above). Golden IDENTICAL on both boxes and both
  profiles. Default `carry`; `=alternate` the A/B. The rig lives in `tools/`
  (`turn-prompts.py`, `turn-rig.sh`, `turn-summary.py`); the turn-2/3 payloads are
  hand-built from a saved answer (a follow-up: generate them). Gate counts on the
  SDD ledger.**

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
  decode misses. **Caveat found at Task 3:** `io_fetch_ms` sums each `executePlan`
  call's elapsed window, and from Task 2 on two plans execute concurrently, so the
  sum exceeds elapsed drive time — valid as a delta between cells on one shape,
  not as elapsed time; the drive is priced from bytes ÷ the measured rate
  thereafter. **Three lengths give 0.560 / 0.559 / 0.560 ms per 1.6875 MiB
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
  means carried state leaked into a first request. **FALSIFIED as written —
  the move came from a defect in the direction rule, not from leaked state; see
  the LANDED paragraph.** Golden **IDENTICAL** both
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

  - [x] Step 1 (an implementer): the seven failing tests, then the mode enum, the
        parser, the carried parity function, the runner var and the `sweep=`
        field. `swift test --no-parallel --filter PrefillMoEGrouping` and
        `--filter PrefillRoutedTileScheduler` → FAIL then PASS. Five gates
        (release build 0 warnings; `swiftlint lint --strict --baseline`,
        regenerating the baseline if `executePrefillChunk`'s span moved;
        `tools/check-md-links.py`; `swift test --no-parallel`; the same under
        TSAN with `TSAN_OPTIONS=suppressions=tsan-suppressions.txt`). Move the
        three rig scripts under `tools/` in the same commit.
  - [x] Step 2: `tools/golden-baseline.sh --check` on the M4 Pro — short and long
        **IDENTICAL**; a difference is a defect and never a recapture. Then
        `tools/mini-deploy.sh --restart` and the mini's golden check.
  - [x] Step 3 (the arms, controller-run, **one binary**, `SHRIKE_PREFILL_SWEEP`
        as the A/B; `pgrep -fl 'ShrikeServer|ShrikeMac|ShrikeDecodeService|ShrikeCLI'`
        before every launch; a fresh server per pair, `settle_done` before the
        warm send, a distinct tag per arm and per box — P10's
        `resp-<tag>-<label>.json` collision): **A** `alternate` and **B** `carry`,
        `tools/turn-rig.sh <host> <port> <promptdir> <outdir> <tag> pair 300|1k|2k`
        (the landed argument order), tags `t0-alt-mini-<len>` /
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
  - [x] Step 4 (the rule): apply it to the three warm walls. `carry` becomes the
        default with `=alternate` as the A/B, or the default stays `alternate`.
        **Record the verdict either way**, with the first-chunk hit rate and the
        exposed-fetch term per arm — the direct read of whether the hits arrived
        and whether they were on the critical path. Then five gates on the landed
        tree, golden both boxes both profiles IDENTICAL, and
        `tools/mini-deploy.sh --restart` with the mini golden check.
  - [x] Step 5: design doc — a "Task 0 — the carried sweep parity" section with
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
    a first request — a defect in where the var is written or reset. **It moved
    (FALSIFIED): the defect was in the direction rule itself (the per-chunk
    re-read), caught by the task review — see the LANDED paragraph.**
  - **Golden moves.** Then the comparator is not the only thing `descending`
    reaches. The default stays `alternate` until it is found; never a recapture.
  - **The mini is production.** Every arm stops the server on 8081 and relaunches
    it; Turbo on 8080 is never touched. One model process at a time — `pgrep`
    first, every time.

### Task 1: T1 — two tile fetches in flight

- [x] **T1: the routed prefill loop awaits each tile's fetch in the expression
  that issues it, so between one tile's reads landing and the next tile's reads
  starting the drive is idle for the host's whole plan → encode → commit step.**
  `PrefillStreamedTileBinding.fetchBindingForTile`
  (`Sources/Shrike/Kernels/Prefill/MoE/PrefillGroupedRoutedMoE.swift:562-606`)
  does `views = try await model.fetchRoutedExperts(plan:)` at `:584` =
  `beginFetchRoutedExperts(plan:).completion()`
  (`Sources/Shrike/Runtime/Inference/ModelExpertIO.swift:208-210`), and the
  per-tile `for` (`RealForwardRunner.swift:5282`, in `encodeRoutedMoEPrefill`
  `:5167`) suspends on it at `:5342`. P16 found this and named the lever it did
  not take — `TILE_DEPTH` banks committed **GPU** tiles, not fetches, "hence
  `TILE_DEPTH` … not `FETCH_DEPTH`"
  ([v12-implementation-plan.md](v12-implementation-plan.md):5194-5201) — and
  priced two fetches in flight as a follow-on
  ([v12-prefill-matrix-kernels.md](v12-prefill-matrix-kernels.md):1271-1276).
  This task takes it: plan tile N+1 avoiding N's slots and the held ones, begin
  its fetch, **then** await N, encode N, commit; the next iteration awaits N+1
  and begins N+2. The primitive exists — `beginFetchRoutedExperts` returns a
  `RoutedExpertLoadOperation` (`ModelExpertIO.swift:212`, `:35`; `wait()` `:50`,
  `completion()` `:55`), used deferred by decode already
  (`RealForwardRunner.swift:4838`, `:6336`). This next because after T0 it is the
  largest term left here: `routed→routed` host is **853 / 914 / 486 ms** of the
  3.80 / 6.88 / 10.80 s warm walls ([v13-the-turn.md](v13-the-turn.md):135-150),
  and the host cost the drive idles through is ≈ 0.72 ms × 982 / 1,147 / 1,188
  tiles = **707 / 826 / 855 ms** (P16's upper bound). Scheduling only. **The mini
  decides.**

  **LANDED 82608b4 (2026-09-04): measured on the mini on one binary (the knob as the
  A/B; the flip folded by amend), a fresh server per pair, `settle_done` before
  the warm send; 300 and 1k as paired means of two runs in opposite orders:
  warm walls 300 tokens 3.777 → 3.683 s (−2.5 %; hits 58.6 → 58.4 %, misses
  3,200 → 3,218, `routed→routed` host 806 → 745 ms), 1k 6.786 → 6.667 (−1.8 %;
  51.6 → 51.6, 4,383 → 4,387, 836 → 733), 2k 10.676 → 10.451 (−2.1 %; 50.4 →
  50.4, 4,639 → 4,643, 371 → 184); turn 2 1.663 → 1.656, turn 3 1.466 → 1.344
  (one run per arm; turn 2 hits 2,056 → 2,055, turn 3 identical); 12k control
  68.342 → 68.346 s, hits 9,546
  → 9,540; routed GPU and tile counts unmoved; golden IDENTICAL both boxes both
  profiles at both values. `io_fetch_ms × 8` rose 1,863 → 3,274 / 2,485 → 4,466
  / 2,680 → 4,488 — the parked batch's wait counted inside the fetch, the
  predicted signature of the gap closing. The model's Σc ≈ 700–850 ms was wrong
  by 8×: the exposed host gap was 61 / 103 ms at 300 / 1k (≈ 190 at 2k), and the
  held slots cost no hits (modelled −349 / −360 / −363, measured −18 / −4 / −4).
  The inversion read (`host_ms` ÷ routed tiles) fell 0.85 → 0.78 ms at 300 and
  0.74 → 0.64 at 1k — no inversion signal.
  The knob clamps to 2 (the loop looks exactly one tile ahead; arm C dropped by
  ruling). The 5 % bar below is RETRACTED (an inherited default, never derived);
  under the chapter's real-and-free rule the default is 2. The design doc's
  Task 1 section carries the rows, the C-reader finding and T2.**

  **The decision rule, in two lines (retracted at the verdict, see LANDED).**
  `SHRIKE_PREFILL_FETCH_DEPTH=<n>` (1…4,
  parsed like `parsePrefillTileDepth`, `RealForwardRunner.swift:501-512`; 1 =
  today) lands either way, printed as `fetch=` beside `depth=` in the
  projection-path line's tile field (`prefillTileBatchDescription`, `:226-241`).
  **2 becomes the default only if** the mini's 300 and 1k warm walls each improve
  by **≥ 5 %** on the same binary with 2k not regressing (≥ −1 %; its gain
  reported, not required), turn 2 / turn 3 not regressing beyond their control
  drift, the 12k control unmoved (± 1 %), golden **IDENTICAL** and
  `memory_pressure -Q` acceptable; otherwise the default stays 1 and the rows are
  recorded (P8's precedent). The model gives −7.4 % and −8.5 % at 300 and 1k, so
  the 5 % bar is supported, not aspirational.

  **What the drive sees, and why this is not "queue depth 2".** Production takes
  the bounded branch (`ParallelExpertReader(threads: 4, bypassCache: true)`,
  `PreadExpertStreamer.swift:451-454` — a literal with no knob, **one reader per
  layer**, `Model.swift:552`); `beginExpertCachePlan` hands the plan to
  `ExpertIOScheduler.shared` (4 workers, `ExpertLoadOperation.swift:168-169`;
  `PreadExpertStreamer.swift:837-845`), whose worker runs `executeBoundedReads`
  (`:977-988`) → `submit_batch` (`sources/ShrikeKernelsC/expert_io.c:237-277`).
  **`submit_batch` publishes exactly one batch at a time**: a second caller waits
  on `batch_idle` until the first completed *and* cleared the published pointers
  (`:247-250`, `:270-274`; the predicate exists so a second caller cannot
  overwrite them). Two operations outstanding are therefore served
  **batch-serially** — tile N's reads all complete before N+1's, which is what we
  want — and the four worker threads cap bytes in flight at **four experts
  however many operations are queued**. T1 removes the inter-tile idle and the
  submission latency (the next batch is already parked on `batch_idle` when the
  current clears) and **does not raise queue depth**. Bounded caveat: which
  worker wins the mutex is a race, so tile N is occasionally served after N+1 —
  priced in the risks below: each inversion re-exposes one host step.

  **The reader's thread count is NOT in this task, for a measured reason.** After
  T0 a tile carries **3.26 / 3.82 / 3.90 misses on the mean**, so its fetch is
  one wave that does not fill the four threads it has: there are never eight
  reads for eight threads. `threads` pays only once **two** batches can be
  published at once — a change to `expert_io.c`'s one-batch predicate with its
  own correctness surface. The peer's mini probes price that pair together and
  small: four whole experts in flight **3.57 / 3.61 GB/s** (0.496 / 0.490 ms per
  expert), eight **3.76 / 3.83** (0.470 / 0.462) — **+5–6 %**, inside the probe's
  own drift (four in flight read 2.88 / 2.93 / 3.30 in earlier runs);
  `~/.claude/handoffs/archive/shrike-ssd-split-probe/mini-probe-run3.txt:11-12`,
  `:26-27`, `run1.txt:4-8`, `run2.txt:5,21`. **T1 alone here; the paired C-reader
  + thread change is T2, with T1's rows as its prior.**

  **The slot budget.** `fitsSlotBudget` is `maxInFlightTiles × tileExperts +
  reservedHits ≤ slotCount`, `maxInFlightTiles = (maxPendingDepth + 1) ×
  tilesPerCommandBuffer` (`PrefillRoutedTileScheduler.swift:55-58`, `:81-82`);
  production is depth 2, width 1, `tileExperts` 8
  (`RealForwardRunner.swift:590-592`) = **24 of the mini's 128 slots**. An
  in-flight fetch holds its slots from plan time until its tile's GPU work
  completes, so one lookahead adds one tile — **32 of 128**, width-1 ceiling
  (D + 1 + 1) × 8 ≤ 128 → **D = 14** (was 15). `fitting` (`:63-79`) must pass the
  lookahead through unchanged in both returns as it already does
  `maxPendingDepth`: 128 returns the config untouched, 16 narrows `tileExperts`
  5 → 4, `nil` only below four tiles' worth. **`reservedHits` is dead in
  production** — the only two call sites take the default 0
  (`RealForwardRunner.swift:233`, `:4977`). The readable window shrinks 128 − 24
  = **104 → 96**; T0's model has hits/layer = `d·P / (2 − d)`, **linear in P**,
  so 8 more held slots costs **7.7 % of the hits at every shape**: measured hits
  4,529 / 4,673 / 4,714 (backed out of the After-T0 misses and rates) → **−349 /
  −360 / −363 hits = +195 / +202 / +203 ms of fetch** at 0.560 ms per expert.

  **Planning N+1 while N is in flight changes only that window.**
  `makeExpertCachePlan` (`PreadExpertStreamer.swift:600-700`) reserves every
  `.loading` slot before matching hits (`:632-634`) and `selectVictimSlots` skips
  them (`:1111-1120`), so an in-flight tile's slots are already safe from
  eviction and from counting as a hit; within a layer-chunk tiles hold
  **disjoint** expert sets (one group per distinct expert), so no tile can want
  an expert another is loading. Aging-LFU's count term is untouched
  (`expertUseCount`, `:657-658`, same order) and the tiebreak sees the same
  `useClock`. Only `avoidingSlots` grows: **hits are expected to fall ≈ 7.7 %**,
  a verdict row either way.

  **Layer boundaries — the floor this lever cannot reach.** Layer L+1's routes
  come from its router over L's output: `waitForCompletion(cb)`
  (`RealForwardRunner.swift:5226`) then `buildPrefillRoutes` (`:4952-4995`,
  called at `:5239`) reads `scratch.routeIDs` back, so **the first tile of L+1
  cannot be planned earlier** — its expert IDs do not exist yet. **40 boundaries
  per chunk**, each an un-overlapped first-tile fetch of F̄ = ΣF/tiles = **1.83 /
  2.14 / 2.19 ms → 73 / 86 / 87 ms per request**. A different lever on that term
  is deliberately out of scope: routes are built before
  `try waitForCompletion(sharedCB)` (`:5250`), so tile 0's fetch could ride the
  shared expert's GPU — P16's own follow-on
  ([v12-prefill-matrix-kernels.md](v12-prefill-matrix-kernels.md):1276-1280);
  folding it in would make this A/B measure two things.

  **Byte-identity — scheduling only.** The change decides which slot an expert
  lands in and when, never its bytes; commit order is unchanged, each pair still
  writes `route_partials[(token · top_k + rank) · D + d]`
  (`Sources/Shrike/Metal/Prefill/prefill.metal:946`) and
  `prefill_moe_reduce_token_major` folds ranks in order (`:743-764`). P16's
  precedent: golden IDENTICAL both boxes both profiles. **A slot handed to the
  wrong tile is a correctness bug, not a numerics change**, and two guards keep
  it exact: (1) `avoidingSlots` becomes `openBatch ∪ pendingBatches ∪ the
  in-flight plan's assignedSlots` (today `:5347-5349` unions the first two); (2)
  `tileLifetime.begin` moves from after the fetch (`:5353-5356`) to **plan
  time**, so the overlap check (`PrefillGroupedRoutedMoE.swift:387-399`) covers
  the in-flight tile, `complete` still at drain (`:5150-5152`). The assertions
  are that overlap rejection with two tiles in flight, and
  `streamedTileFetchBindingAvoidsInFlightPlannedSlots`
  (`tests/…/PrefillGroupedRoutedMoETests+Binding.swift:196`) extended to begin
  the first fetch without awaiting it.

  **Error paths.** A begun fetch must be **awaited, never abandoned**:
  `abandonRoutedExpertPlan` (`ModelExpertIO.swift:143`) →
  `resetLoadingMissesUnlocked` (`PreadExpertStreamer.swift:1324-1337`) marks the
  slots `.empty` while the C reader is still writing into those pointers, after
  which a later plan could pick the same slot as a victim. So the loop's catch
  path drains the in-flight operation first; `RoutedExpertLoadOperation.wait()`
  (`ModelExpertIO.swift:50`) is synchronous, so a non-async `defer` can do it.
  `drainBeforeIssue` (`PrefillRoutedTileScheduler.swift:99-107`) keeps abandoning
  only plans not yet begun; a plan made one iteration early stays valid because
  the held set only shrinks between iterations. The tail (`:5391-5395`) is
  unchanged — the last tile has no successor, so none is begun.

  **The expected gain, modelled per shape.** The drive is busy ΣF = misses ×
  0.560 ms while the host is serially in front of it for Σc = tiles × 0.72 ms;
  today the chain is ΣF + Σc and the routed stage is `max(chain, routed GPU)`.
  Under T1 the chain becomes ΣF + 40 × c (one un-overlapped host step per layer
  boundary, ≈ 29 ms) plus the held slots' +195 / +202 / +203 ms.

  | shape | tiles | misses | ΣF | Σc | chain today | routed GPU | modelled stage | measured (GPU + `r→r` host) |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 305 | 982 | 3,200 | 1,792 ms | 707 | 2,499 | 1,516 | 2,499 | **2,369** |
  | 1,085 | 1,147 | 4,383 | 2,454 | 826 | 3,280 | 2,506 | 3,280 | **3,420** |
  | 2,125 | 1,188 | 4,639 | 2,598 | 855 | 3,453 | 3,911 | 3,911 | **4,397** |

  The model lands within +5.5 / −4.1 / −11.1 % of measured. Three cases, all
  **modelled**; the bars come from (a):

  | case | 305 | 1,085 | 2,125 |
  | --- | ---: | ---: | ---: |
  | (a) gap only, held slots paid, 80 % capture | 3.52 s (−7.4 %) | 6.29 (−8.5 %) | 10.41 (−3.6 %) |
  | (b) + T2's rate 0.470 ms/expert | 3.26 (−14.2 %) | 6.15 (−10.6 %) | 10.41 (−3.6 %) |
  | (c) layer-boundary floor at today's rate | 3.30 | 6.05 | 10.40 |

  (a) is stage(T1) = max(ΣF + 349·0.560 + 29, GPU) = 2,016 / 2,685 / 3,911 ms,
  Δ = 353 / 735 / 486 against the measured stage, taken at 80 % (P16 captured
  90 % of host-late at 12k). **2k is the arm the rule does not bind on**: its
  chain (2,830 ms) already sits below its routed GPU (3,911), so after T1 it is
  GPU-bound on the routed path — a small 2k gain beside a large 300/1k one is
  **predicted**. (b) is what T2 would add; 1k and 2k cannot take it, going
  GPU-bound first, which is the chapter's next signpost (kernels, not I/O).

  **Bars (mini, warm arm, on the After-T0 rows).** Warm wall **3.80 → ≤ 3.61 s**,
  **6.88 → ≤ 6.54**, **10.80 → ≤ 10.91**; the (a) model 3.52 / 6.29 / 10.41 as
  the stretch, the (c) floor 3.30 / 6.05 / 10.40 as what this lever cannot reach.
  `routed→routed` host **853 → ≤ 600 ms**, **914 → ≤ 400**, **486 → ≤ 350**
  (modelled residual stage − GPU = 500 / 179 / ≈ 0 plus variance).
  `expert_hit_rate_prefill` **58.6 → ≥ 52 %**, **51.6 → ≥ 46**, **50.4 → ≥ 45**
  (the model's −7.7 % relative gives 54.1 / 47.6 / 46.5, two points of slack);
  any larger loss priced at 0.560 ms of drive per lost hit against the wall gain.
  `prefill_routed_tile` count and GPU **unmoved**. **`io_fetch_ms × 8` is
  predicted to RISE, not fall** — the `batch_idle` wait now sits inside
  `reader.fetch`, so a rise beside a falling wall is the signature that the gap
  closed; the invariant drive-busy term is `expert_misses_prefill × 0.560`. 12k
  control **68.30 s ± 1 %**, `expert_hits_prefill` **≥ 9,070**. Turn 2 **1.70 s**
  / turn 3 **1.37 s** ± **5 %** — not T0's ± 1 %, because turn 2's own control
  drifted 2.13 → 1.91 s at byte-identical traffic
  ([v13-the-turn.md](v13-the-turn.md):108-112). Golden **IDENTICAL** both boxes
  both profiles; `memory_pressure -Q` before and during every mini arm. The M4
  Pro is the check (F̄ / G = 1.80): it should move the same way or more, and
  decides nothing.

  **The measurement.** One binary, the knob as the A/B: `tools/turn-rig.sh <host>
  <port> <promptdir> <outdir> <tag> pair 300|1k|2k` (header `:1-33`; `SERVER_ENV`
  carries the knob, a fresh server per pair, the warm arm after `settle_done`),
  the `turns` phase, and the 12k control via `tools/prefill-measure.sh … 6k` —
  label **`6k`** is `tools/prefill-prompts.py`'s 12,285-token prompt, its `12k`
  label the 25,245-token one (Task 0 Step 3, `:317-336`). A distinct tag per arm
  and mode (`t1-f1-mini-<len>` / `t1-f2-mini-<len>`; P10's collision). Rows per
  arm: wall, `prefill_s`, `expert_hits_prefill` / `expert_misses_prefill` /
  `expert_hit_rate_prefill`, `expert_read_mib`, `io_fetch_ms`, `expert_evictions`
  / `expert_reloads`, the `prefill_routed_tile` role GPU and count,
  `routed→routed` total / `host_ms` / `queue_ms` / count, `shared→routed`, busy,
  span, `memory_pressure -Q`. **No new counter:** F̄ per tile is `io_fetch_ms × 8`
  ÷ the routed role's count (`ServerInference.swift:2023`, divisor
  `result.newTokens` = 8 at `:1957`), host-late is `host_ms` ÷ count, both on the
  stats line; `io_queue_ms` is not accumulated on the prefill path
  (`totalIOQueueNanos` only at `RealForwardRunner.swift:5810`, `:6330`, `:6339`,
  all decode) and would measure the `ExpertIOScheduler` wait, not `batch_idle`.
  `tools/` needs no change.

  **Tests (RED first, host-only)** in
  `tests/Shrike/Core/Kernels/Prefill/PrefillRoutedTileSchedulerTests.swift`
  beside the depth tests (`:266-345`):
  `theLookaheadCountsOneMoreTileInTheSlotBudget` (lookahead 1, depth 2, width 1,
  `tileExperts` 8 → `fitsSlotBudget(128)` true, `(31)` false; ceiling D = 14, so
  `(15)` false at 128); `fittingKeepsTheLookaheadAndNarrowsTheTile` (128 →
  itself, 16 → 4, 8 → 2, 3 → `nil`);
  `schedulerBeginsTheLookaheadOnlyWhileTheBudgetAllows` — the decision as a pure
  function over a short tile list: what to begin, await and drain, and that the
  last tile begins no successor; `theLookaheadIsNotBegunWithoutAnAvoidingSlotPlan`;
  `parsePrefillFetchDepthClampsToOneThroughFour` (unset → default, `"0"` → 1,
  `"9"` → 4, garbage → default); `prefillTileBatchDescriptionReportsTheFetchDepth`
  (asserted as `prefillTileDepthDescription` is at `:320-327`); and beside
  `slotLifetimeRejectsReuseInsideAnOpenBatch` (`:363-374`) the same rejection
  with the in-flight tile begun at plan time. Plus the binding test above. **The
  runner loop is not host-testable** (device, model, streamer); the arms verify
  it and golden IDENTICAL is the correctness stop — a mis-handed slot changes
  output.

  **Files.** `PrefillRoutedTileScheduler.swift`: `fetchLookahead` on the config
  (`:42-51`), the budget (`:55-58`), `fitting` (`:63-79`), `maxInFlightTiles`
  (`:81-82`), the lookahead predicate beside `decide` (`:92-110`).
  `RealForwardRunner.swift`: the parser beside `environmentPrefillTileDepth`
  (`:499-512`), the config build (`:590-592`), `prefillTileBatchDescription`
  (`:226-241`), and the loop (`:5282-5395`) — the begin/await split,
  `avoidingSlots` gaining the in-flight plan, `tileLifetime.begin` at plan time,
  the error-path drain, the tail. `PrefillGroupedRoutedMoE.swift`: split
  `fetchBindingForTile` (`:562-606`) into plan-and-begin and
  binding-from-completed-views halves, keeping the existing entry point for the
  `fetch=1` path so depth 1 stays structurally today's code. Tests: the two files
  above. **Lint:** `encodeRoutedMoEPrefill` (`:5167`) sits in
  `.swiftlint-baseline.json` at "currently spans 231 lines" — the reason string
  embeds the count, so **any** edit to that body makes the entry stale and the
  baseline must be regenerated in Step 1 (T0 hit this on `executePrefillChunk`).
  **Unchanged:** every kernel and `.metal` file, `PrefillMoEGrouping` and the
  sweep, `PreadExpertStreamer` and its eviction policy, `ParallelExpertReader`
  and `expert_io.c` (**explicitly** — the thread count and the
  one-published-batch predicate are T2's), the prompt cache, decode, `tools/`.

  Steps:

  - [x] Step 1 (an implementer): the eight failing tests, then the config field
        and its budget arithmetic, the lookahead predicate, the knob and the
        `fetch=` field, and the loop restructure. `swift test --no-parallel
        --filter PrefillRoutedTileScheduler` and `--filter PrefillGroupedRoutedMoE`
        → FAIL then PASS. Five gates (build 0 warnings; `swiftlint lint --strict
        --baseline`, regenerating it for `encodeRoutedMoEPrefill`;
        `tools/check-md-links.py`; `swift test --no-parallel` — the **full**
        suite, since `fitting` now narrows small caches further; the same under
        TSAN).
  - [x] Step 2: `tools/golden-baseline.sh --check` on the M4 Pro — short and long
        **IDENTICAL**; a difference is a defect, never a recapture. Then
        `tools/mini-deploy.sh --restart` and the mini's golden check.
  - [x] Step 3 (the arms, controller-run, **one binary**,
        `SHRIKE_PREFILL_FETCH_DEPTH` as the A/B; `pgrep -fl
        'ShrikeServer|ShrikeMac|ShrikeDecodeService|ShrikeCLI'` before every
        launch; a fresh server per pair, `settle_done` before the warm send, a
        distinct tag per arm and box): **A** `=1` and **B** `=2` via
        `tools/turn-rig.sh macmini 8081 <promptdir> <outdir> t1-f<n>-mini-<len>
        pair 300|1k|2k`; **C** `=3` at 300 and 1k, the plateau check (P16's
        precedent) priced against eight more held slots; **D** the turns arm
        (`turn-rig.sh … turns`); **E** the 12k control
        (`tools/prefill-measure.sh macmini 8081 … t1-f<n>-mini-12k 6k`).
  - [x] Step 4 (the rule): apply it to the three warm walls; `2` becomes the
        default with `=1` as the A/B, or the default stays 1. **Record the
        verdict either way**, with `host_ms` per boundary and `io_fetch_ms × 8`
        per arm — whether the gap closed, and whether the drive's busy time moved
        inside the fetch measure. Then the four per-commit gates on the landed tree, golden both
        boxes both profiles IDENTICAL, `tools/mini-deploy.sh --restart` and the
        mini golden check.
  - [x] Step 5: design doc — a "Task 1" section with the ΣF / Σc / stage table,
        the arms' rows and an "**After T1**" ledger block; the "Bytes per expert"
        lever entry rewritten with the C reader finding (one batch published at a
        time, four threads, ≈ 3.3 misses per tile) and T2 named. Plan: Task 1
        `[x]` with the landed paragraph. Task review by a fresh reviewer; fixes
        folded into the owning commit.

  **Risks and what falsifies the model.**
  - **The wall does not move though `routed→routed` host falls.** Then Σc was not
    on the critical path — the drive was already back-to-back and the idle was
    the GPU's; `io_fetch_ms × 8` unchanged rather than risen says so, and the
    remaining term is the drive's rate: T2 (two published batches + `threads` 8),
    priced at the probe's +5–6 %, with these rows as its prior. A verdict
    claiming a bandwidth gain from **this** knob would be misattributed: the C
    reader serializes the two operations by construction.
  - **Hits fall more than the fetch gains.** The 8 held slots, ≈ 200 ms per shape
    against a modelled 353–735. If the loss exceeds the gain the arm is **fetch
    depth 2 with tile depth 1** — 24 held, today's budget, trading GPU bank for
    fetch overlap — not a redesign.
  - **Tile N served after N+1.** The workers race for `submit_batch`'s mutex.
    The parked waiter wakes on `batch_idle`'s broadcast before the next
    submission arrives, so inversions should be rare — but each one re-exposes
    one host step c (the drive idles while the host encodes the late tile), so
    an inversion rate r costs r × Σc, and at r = 0.5 it is the whole modelled
    gain. If `host_ms` per tile shows it, the fix is a sequence number or a
    single-worker prefill submission path — recorded (ruling on the ledger).
  - **The 12k control moves.** At F̄ / G = 0.58 it is GPU-bound and the
    prediction is flat; a move means the extra held slots cost hits at a shape
    with more sweep passes to lose, and `expert_hits_prefill` (≥ 9,070) is the
    read.
  - **Golden moves.** A slot was handed to the wrong tile — a defect in
    `avoidingSlots` or in the lifetime's move to plan time, not a numerics
    change. The default stays 1 until it is found; never a recapture.
  - **Small-cache models.** `fitting` reserves one more tile, so a 16-slot cache
    drops `tileExperts` 5 → 4 and the `nil` threshold moves. Run the full suite
    (P15b's ragged-K gate failure came from this class of path).
  - **The mini is production.** Every arm stops the server on 8081 and relaunches
    it; Turbo on 8080 is never touched. One model process at a time.

### Task 2: T2 — the expert reader publishing two batches at once

- [x] **T2: the C reader publishes exactly one batch at a time, so Task 1's two
  tile fetches in flight are served batch-serially and bytes in flight never
  exceed one tile's misses — 3.3–3.9 on the mean after T0.** `submit_batch`
  (`sources/ShrikeKernelsC/expert_io.c:237-277`) parks a second caller on
  `batch_idle` (`:248-250`) until the first batch has completed *and* cleared its
  published pointers (`:264-275`); the struct holds exactly one batch — `count`,
  `next_index`, `outstanding`, `first_errno`, `generation` and the three pointer
  arrays, under "Current batch. Valid only while outstanding > 0" (`:27-36`) —
  and `worker_main` claims reads from it by `next_index` (`:79-100`, claim at
  `:80-86`). Production runs `ParallelExpertReader(threads: 4, bypassCache: true)`,
  a literal with no knob, **one reader per layer**
  (`PreadExpertStreamer.swift:451-454`; `Model.swift:552`), so a tile's 3–4 misses
  are one wave that never fills its four threads, and the drive drains and refills
  between tiles. Task 1 measured the cost: `io_fetch_ms × 8` at depth 1 is
  1,863 / 2,485 / 2,680 ms against 3,200 / 4,383 / 4,639 misses
  ([v13-the-turn.md](v13-the-turn.md):178-185) = **0.582 / 0.567 / 0.578 ms per
  expert**, ≈ **3.08 GB/s** at a 1,769,472-byte stride, against the peer's mini
  probes' **3.57–3.61 GB/s at four whole experts in flight, 3.76–3.83 at eight**
  (`~/.claude/handoffs/archive/shrike-ssd-split-probe/mini-probe-run3.txt:11-12`,
  `:26-27`). This task lets the reader hold **two published batches**: a second
  caller's batch is accepted while the first is outstanding, workers claim from
  the older batch first and spill into the newer only when the older has no
  unclaimed read left, and each caller waits for its own batch. Tiles N and N+1
  then read concurrently and the thread count becomes a live knob. Next because it
  is the term Task 1 exposed and could not reach
  ([v13-the-turn.md](v13-the-turn.md):205-212, `:266-277`), and because it is
  scheduling only: the same bytes in the same slots, chosen by a planner this task
  does not touch. **The mini decides.**

  **LANDED d3efdeb (2026-09-04): measured on the mini on one binary (the two knobs
  as the A/B; the flip folded by amend), a fresh server per pair, `settle_done`
  before the warm send; 300 and 1k as paired means in opposite orders (three
  baseline runs): warm walls 300 tokens 3.688 → 3.537 s at (2, 4) (−4.1 %;
  hits 58.4 → 58.4 %, `routed→routed` host 756 → 630 ms), 1k 6.672 → 6.468
  (−3.1 %; 51.6 → 51.6; 737 → 551), 2k 10.460 → 10.304 (−1.5 %; 50.4 → 50.3;
  187 → 51); turn 2 1.637 → 1.600, turn 3 1.328 → 1.332; 12k control 68.304 →
  68.313, hits 9,540 both; decode tok/s on the long-decode arm 14.09 / 14.22 →
  14.17 / 14.19 (unmoved); memory pressure 83–91 % free before every launch;
  golden IDENTICAL both boxes both profiles at (1, 4), (2, 8) and (2, 4). **The
  verdict cell moved from (2, 8) to (2, 4) on the attribution arms:** at 300
  (2, 8) 3.550 / 3.580, (2, 4) 3.531 / 3.542, (1, 8) 3.699 vs (1, 4) 3.678–3.702
  — the batch depth is the lever, the thread count null (arm D's predicted
  null held), and eight threads cost turn 3 +26 ms across two runs (absent at
  four; publication's broadcast to every parked worker — a per-read signal
  recorded as a refinement). Real (the sign held in every pair, deltas 5–7× the
  drift) and free (no control row regressed beyond five experts of residency at
  2k, 50.38 → 50.33 %, a plan-time-input effect worth ≈ 3 ms — the design doc
  names it) → the default is batch depth 2 with four threads. The realized
  drive term (`io_fetch_ms × 8`, the parked wait included; (1, 4) → (2, 4)
  paired) fell 264 / 388 / 592 ms — per expert 1.019 → 0.937, 1.019 → 0.930,
  0.970 → 0.842 ms — and about half reached the wall at 300 / 1k, the model's
  conservative bracket. The design doc's Task 2 section
  carries the rows, the attribution and the herd reading.**

  **The decision rule, under the chapter's real-and-free rule**
  ([v13-the-turn.md](v13-the-turn.md):289-303). Two knobs land either way:
  `SHRIKE_EXPERT_IO_BATCH_DEPTH=<1|2>` (1 = today) and `SHRIKE_EXPERT_IO_THREADS=<1…16>` (4 =
  today). **No percentage bar.** The defaults move to the best measured cell only
  if the effect is **real** — the sign holds across paired runs in both orders at
  300 and 1k and the delta exceeds the rig's shown drift (11–66 ms on the
  single-chunk walls, [v13-the-turn.md](v13-the-turn.md):176-177), a third pair if
  it sits inside twice that — and **free**: no control row regresses. Those rows
  are the other two shapes, turn 2 / turn 3 (± 5 %, their own drift), the 12k
  control (± 1 % wall, `expert_hits_prefill` ≥ 9,070), `expert_hit_rate_prefill`
  unmoved (this task changes no plan, so a move is a defect and not a cost),
  **`decode_tok_s` on the long-decode arm not regressed**, `memory_pressure -Q`
  acceptable before and during every arm, and golden **IDENTICAL** on both boxes
  and both profiles at both cells. Otherwise the defaults stay at (1, 4) and the
  rows are recorded (P8's precedent); a measured null is a result.

  **What changes in the C reader.** One batch becomes a two-slot ring, `depth`
  fixed at create and clamped 1…`SHRIKE_IO_MAX_BATCHES` (2) as `threads` is
  clamped 1…`SHRIKE_IO_MAX_THREADS` (`:14`, `:119-120`). Each slot carries today's
  `expert_ids` / `offsets` / `destinations` / `count` / `next_index` /
  `outstanding` / `first_errno` (`:27-35`) plus an `active` flag (published, not
  yet reaped) and a `sequence` stamped at publication. `work_done` (`:24`) becomes
  **per slot**, so a completion never wakes the wrong submitter; `work_ready`
  (`:23`) and `batch_idle` (`:25`) stay reader-wide — "some slot has claimable
  work", "a slot was freed". **`generation` (`:36`) is deleted for `sequence`**:
  today it is incremented at `:262` and read nowhere, while what stops a worker
  re-running a finished batch is `next_index >= count` plus the cleared pointers;
  `sequence` does a real job as the FIFO claim key. **Claim order**: the wait
  predicate becomes "no slot is active with `next_index < count`", and the claim
  takes the active slot with the **lowest `sequence` that still has an unclaimed
  read** — so no read of N+1 is claimed while an unclaimed read of N exists and
  **tile N's fetch is never delayed by N+1's**. Completion order can still invert
  by one read's latency jitter; harmless, since each caller waits on its own slot
  and the runner awaits tile N's own operation (`RealForwardRunner.swift:5568`). A
  worker captures its slot index before unlocking for `read_one` and uses it after
  re-locking; the slot cannot be reused underneath it (next paragraph). The claim
  rule and the free-slot search are factored out as **pure functions over the slot
  table** (no locks, no I/O) in the C header, used by `worker_main` and asserted
  from Swift — that is how FIFO gets a deterministic test, not a timing race.

  **Pointer ownership.** The comment at `:242-246` gives the reason one batch
  exists: a second caller must not overwrite the published arrays while the first
  waits. With two slots the rule holds per slot. A submitter finds a slot with
  `active == 0`, fills it, publishes, and **owns that slot's arrays until its own
  `outstanding` reaches zero**; only then does it read `first_errno`, clear the
  count, index and three pointers, set `active = 0`, broadcast `batch_idle` and
  return (today's `:267-276`, now per slot). A **third** caller parks exactly as a
  second does today — `while (no free slot && !shutting_down) wait(batch_idle)` —
  because the depth is 2. The Swift caller's arrays outlive the call for today's
  reason: `fetch(offsets:into:)` holds them in `withUnsafeBufferPointer` for the
  whole blocking call (`ParallelExpertReader.swift:122-140`).

  **Errors and shutdown.** `first_errno` moves into the slot, so one batch's
  failing read cannot fail the other caller — today's single field would leak an
  `EIO` from tile N+1 into tile N's return. `shutting_down` stays reader-wide, and
  `destroy` (`:216-234`) gains the accounting that fixes a latent hang: today a
  shutdown raised while a batch has **unclaimed** reads leaves its submitter
  waiting forever (workers break at `:83-85`, `outstanding` never reaches zero).
  Under T2 `destroy` sets `shutting_down`, then per active slot sets `first_errno
  = ECANCELED` if unset, subtracts the unclaimed reads (`outstanding -= count -
  next_index; next_index = count`) and signals that slot — **both a published and
  a parked caller return `ECANCELED`**, shutdown bounded by the reads already in
  the kernel (claimed reads still decrement; each read counted once).

  **The Swift side and the knobs.** `shrike_expert_reader_create` gains
  `batch_depth` beside `threads` (header `:32-38`), with
  `shrike_expert_reader_batch_depth` mirroring `shrike_expert_reader_threads`
  (`:66-67`, `expert_io.c:302-304`). `ParallelExpertReader.init` (`:68-84`) takes
  `batchDepth: Int = 1` and exposes it beside `threadCount` (`:45`);
  `PreadExpertStreamer`'s bounded branch (`:448-460`) drops the literal `threads:
  4` for a parser next to `ExpertCacheLayout.environmentValue` (`:185-198`) —
  unset → the defaults, out of range or unparseable → a thrown
  `ModelError.internalInconsistency` naming the range, as the layout knob throws.
  **Nothing prints the reader's shape today**: `threadCount` is stored
  (`ParallelExpertReader.swift:45`, `:83`) and read nowhere outside the class, and
  the projection-path line (`ServerInference.swift:821-824`) carries
  `prefill_tile_batch=` and `prefill_gap_levers=` (`RealForwardRunner.swift:226-247`,
  `:288-305` — residency allocations, the sweep and `cache_layout` since v12's
  close) but nothing about the reader. `prefill_gap_levers` gains
  `expert_io=threads=N batch_depth=D`, from the **parsed configuration** and not a
  live reader: that line is emitted at session construction, where reaching a
  streamer would force a layer open ahead of the lazy load Task 0's review
  documented.

  **The runner needs no change, and the scheduler's four workers are enough.**
  Task 1's lookahead loop begins tile N+1's fetch
  (`RealForwardRunner.swift:5544-5555`) through
  `PrefillStreamedTileBinding.beginFetchForTile`
  (`PrefillGroupedRoutedMoE.swift:626`) → `beginFetchRoutedExperts`
  (`ModelExpertIO.swift:212-224`) → `beginExpertCachePlan` →
  `ExpertIOScheduler.shared.submit` (`PreadExpertStreamer.swift:837-845`), whose
  worker runs `executeBoundedReads` (`:977-988`) → `reader.fetch(offsets:into:)`.
  Today that worker parks inside `submit_batch`; at depth 2 it publishes instead
  and returns when its own batch lands; the loop still awaits tile N's operation
  at `:5568`. Nothing in the loop, `PrefillRoutedTileScheduler`, the planner
  (`makeExpertCachePlan`, `:600-700`) or eviction (`selectVictimSlots`,
  `:1111-1120`) moves. The scheduler runs 4 workers
  (`ExpertLoadOperation.swift:168-169`) and prefill needs **2**: the loop holds at
  most one `inFlight` begin (`:5513`, `:5550`) beside the tile being awaited,
  requests do not overlap (`ServerModelSession` is an actor,
  `ServerInference.swift:509`), and prefill precedes decode within one. The only
  other user is the speculative prefetch (`PreadExpertStreamer.swift:1241`, via
  `beginRoutedExpertPrefetch`, `ModelExpertIO.swift:186-194`), **off unless
  `SHRIKE_PREDICTIVE_PREFETCH=1`** (`RealForwardRunner.swift:743-746`) and, when
  on, targeting layer `L + prefetchProbeDistance` (`:6569-6575`) — a different
  streamer, hence a different reader; demand runs ahead of speculative anyway
  (`ExpertLoadOperation.swift:189-210`).

  **Decode.** The same reader serves decode's demand fetches (≈ one miss per layer
  at 128 slots; v12 Task 17 read decode as hit-rate-bound at 22 / 18 / 12.5 tok/s
  by shape) and, when enabled, its prefetch. **Depth 2 changes nothing there by
  construction** — one caller per reader at a time, verified above — and 8 threads
  change nothing for a one-read batch, which uses one thread either way. The cost
  is countable, which is why decode is a control row: readers are per layer and ≈
  40 layers open, so `threads: 8` means **160 extra worker pthreads and 160 extra
  descriptors** (160 → 320 for the process). No per-thread I/O buffer is added —
  `read_one` writes into the caller's destination, the cache slot pointer
  (`expert_io.c:40-58`, `PreadExpertStreamer.swift:977-987`), and `F_NOCACHE` is a
  descriptor flag (`:139-150`) — so the cost is stack reservation: macOS's default
  512 KiB per pthread, ≈ 80 MiB of address space, kilobytes of RSS while parked.
  **Nothing in the tree calls `setrlimit`, and the mini's soft limit is 256**
  (`ulimit -n`; hard unlimited; `launchctl limit maxfiles 256 unlimited`) against
  ≈ 160 descriptors today and ≈ 320 at 8 threads — a bare launch at (2, 8) would
  fail with `EMFILE`. So Step 1 raises the process's soft limit **once, at the
  first reader's creation** (`setrlimit(RLIMIT_NOFILE)` to the hard limit, capped
  at `OPEN_MAX`), and arm B's launch is the check that the production launch
  works without a manual `ulimit` (`lsof -p <pid> | wc -l` the read).

  **Bytes in flight per cell** — `(depth, threads)`; in flight is concurrent
  `pread`s of one whole expert (1.6875 MiB). Arm D falsifies the whole model: if
  it moves the wall, the effect was never batch size and Task 1's reading is wrong.

  | cell | in flight | role |
  | --- | --- | --- |
  | (1, 4) today | ≤ min(tile misses, 4); mean **3.26 / 3.82 / 3.90**, draining between tiles | **verdict**, arm A |
  | (2, 8) | ≤ 8 across two batches, the drive fed continuously | **verdict**, arm B |
  | (2, 4) | ≤ 4, but never draining between tiles | attribution, arm C |
  | (1, 8) | still ≤ min(tile misses, 8) = 3.3–3.9 — **predicted null** | attribution, arm D |

  **The expected gain, modelled per shape.** All **modelled**; the measured inputs
  are Task 1's rows ([v13-the-turn.md](v13-the-turn.md):178-189, `:213-219`,
  `:231-241`). Write the routed stage as GPU + ΣF − hidden, ΣF = misses × the
  realized per-expert time, `hidden` the part of the drive the two-tile GPU bank
  already covers:

  | shape | misses | per-expert (measured) | ΣF | routed GPU | measured stage | hidden | hidden / ΣF |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 305 | 3,218 | 0.582 ms | 1,874 ms | 1,413 | 2,281 | 1,006 | 54 % |
  | 1,085 | 4,387 | 0.567 | 2,487 | 2,517 | 3,355 | 1,649 | 66 % |
  | 2,125 | 4,643 | 0.578 | 2,682 | 3,907 | 4,150 | 2,439 | 91 % |

  T2 lowers the per-expert time and nothing else. Two cases from probe run 3 (both
  seeds): **(a)** the per-batch drain removed but the rate only reaching four in
  flight, **0.493 ms**; **(b)** eight in flight, **0.466 ms**. ΔΣF = misses ×
  (0.575 − rate): (a) **287 / 324 / 393 ms**, (b) **374 / 443 / 518**. Two
  bracketing readings of how much of that reaches the wall — which one it lands
  near is the task's real question:

  | | 305 | 1,085 | 2,125 |
  | --- | ---: | ---: | ---: |
  | (b) optimistic — hidden ms constant, stage floored at GPU | −374 ms → **3.31 s** | −443 → **6.22** | −243 (floor) → **10.21** |
  | (b) conservative — hidden *fraction* constant | −173 → 3.51 | −149 → 6.52 | −47 → 10.40 |
  | (a) optimistic / conservative | −287 → 3.40 / −133 → 3.55 | −324 → 6.34 / −109 → 6.56 | −243 → 10.21 / −36 → 10.42 |
  | (c) null — the probes' +5–6 % inside their own drift (four in flight read 2.88 / 2.93 / 3.30 GB/s in earlier runs) | 0 | 0 | 0 |

  Optimistic assumes every millisecond off the drive comes straight off the stage;
  conservative assumes the same fraction stays hidden. **2k is capped either way at
  its exposed 243 ms** (stage 4,150 − GPU 3,907): its drive term is already under
  its GPU, so a small 2k gain beside a larger 300 one is *predicted*, as at Task 1.
  1k's cap is its exposed **838 ms**, which neither case reaches. **The
  layer-boundary floor this lever cannot cross:** layer L+1's routes come from L's
  output, so each layer's first tile has no predecessor to overlap with — 40
  boundaries × F̄ (= ΣF ÷ tiles = 1.91 / 2.17 / 2.26 ms) = **76 / 87 / 90 ms per
  request today**, ≈ 61 / 70 / 73 at (b); at 24.6 / 28.7 / 29.7 tiles per layer
  (982 / 1,147 / 1,188 ÷ 40) the mechanism reaches ≈ 96 % of tiles and the other
  4 % is that floor. **The Swift header's bench note is not evidence against this**
  — `ParallelExpertReader.swift:9-11` ("4 threads 3.92 GB/s, 8 threads 3.92
  (saturated)", "four readers is the knee") and `:21-23` (at batch 1 the pool
  matches a plain `pread`) are **sustained full-batch** measurements over a 16.88
  GiB working set, where every thread always has a read to take. At 3.3–3.9 misses
  per batch it is **the batch size, not the thread count**, that starves the drive
  — the pool's own table says so (batch 1 → 3.41 GB/s, batch 4 → 5.36, batch 8 →
  5.95 there), which is why arm D is predicted null.

  **Byte-identity — scheduling only.** Destinations come from the planner's
  `assignedSlots`, untouched here; the reader only decides which thread reads which
  offset when, so golden **IDENTICAL** on both boxes and both profiles at both cells
  is the stop. **A slot's arrays reused before its batch completes is a correctness
  bug, not a numerics change** — one tile's expert in another's slot, which golden
  catches; the guard is `active` plus the ownership window above.

  **Tests (RED first, host-only, against a temp file through
  `ParallelExpertReader`)** in a new
  `tests/…/Streaming/ParallelExpertReaderTests+BatchDepth.swift` beside the existing
  suite (`ParallelExpertReaderTests.swift`: `makeFixture` / `withDestinations`
  `:16-34`, `concurrentCallersCannotOverwritePublishedBatch` `:96-125`), so
  `--filter ParallelExpertReader` picks up both:
  - `twoConcurrentFetchesBothLandTheirOwnBytes` — depth 2, threads 4, two tasks ×
    50 rounds, disjoint destinations, every byte checked (`:96-125` at depth 2).
  - `claimTakesTheOlderBatchBeforeTheNewer` — the FIFO rule as a **pure function**
    over a slot table: both active with unclaimed reads → the lower `sequence`; the
    older exhausted → the newer; neither claimable → none; an inactive slot never
    claimed. Deterministic by construction, no timing.
  - `publishFindsNoSlotWhileBothAreActive` — the same table, the free-slot search:
    both active → none (the third caller parks); one reaped → that index.
  - `oneBatchsReadErrorDoesNotFailTheOtherCaller` — depth 2, threads 2; one task
    reads valid ids, the other an id past EOF, 50 rounds: the first never throws
    and its bytes are right, the second always throws `readFailed`.
  - `cancellingAPublishedBatchAccountsOnlyItsUnclaimedReads` — the shutdown
    accounting as a pure function: `first_errno` → `ECANCELED`, `outstanding` drops
    by exactly `count − next_index`, `next_index == count`, a slot with no unclaimed
    read untouched. (End to end it is unreachable from Swift — a caller inside
    `fetch` retains the reader, `:86-88` — stated, not simulated.)
  - `depthTwoReadsTheSameBytesAsDepthOne` — one out-of-order id list with a repeat
    through (1, 4) and (2, 8), byte-identical (extends `:59-74`).
  - `batchDepthIsClampedToTheSupportedRange` — 0 → 1, 9 → 2, mirroring
    `threadCountIsClampedToTheSupportedRange` (`:157-166`).
  - `boundedReaderConfigurationParsesThreadsAndBatchDepth` — unset → the defaults;
    `"8"` / `"2"` → (8, 2); `"0"`, `"99"`, `"x"` → the thrown error naming the
    range, via `setenv` / `unsetenv` as
    `PreadExpertStreamerTests+CachePlanning.swift:143-144` does.
  - `prefillGapLeversDescriptionReportsTheBoundedReaderShape` — the printed field,
    asserted through the static description as `prefillTileDepthDescription` is
    (`PrefillRoutedTileSchedulerTests.swift:320-327`).

  **Files.** `sources/ShrikeKernelsC/expert_io.c`: the two-slot ring, the claim and
  free-slot rules, per-slot `done` / `first_errno` / `sequence`, `depth` at create,
  the shutdown accounting, and the `RLIMIT_NOFILE` raise (in `create`, or in
  `ParallelExpertReader.init` under a static once — the implementer picks; one
  call, before any reader opens its descriptors). `include/shrike_expert_io.h`: `batch_depth` on `create`,
  `shrike_expert_reader_batch_depth`, `SHRIKE_IO_MAX_BATCHES`, the slot-table view
  and the two pure rules, and the doc comment that today says four saturates the
  device (`:32-38`). `ParallelExpertReader.swift`: the parameter and property, the
  header note reconciled (`:4-36`, `:68-84`). `PreadExpertStreamer.swift`: the
  parser beside `ExpertCacheLayout` (`:185-198`) and the bounded branch (`:448-460`).
  `RealForwardRunner.swift`: `prefillGapLeversDescription` (`:279-305`) gains the
  `expert_io=` field. Tests: the four files above. **Lint:**
  `PreadExpertStreamer.init` is in `.swiftlint-baseline.json` at "currently spans
  160 lines" (`:291`) — the reason string embeds the count, so **any** edit to that
  body makes the entry stale and the baseline must be regenerated in Step 1 (T0 and
  T1 both hit it). **Unchanged:** Task 1's two routed tile loops and
  `PrefillRoutedTileScheduler`, `PrefillGroupedRoutedMoE`'s binding helpers, every
  kernel and `.metal` file, `PreadExpertStreamer`'s planning and eviction,
  `ExpertIOScheduler`, the prompt cache, `tools/`.

  Steps:

  - [x] Step 1 (an implementer): the nine failing tests, then the C ring and its
        two pure rules, the `batch_depth` plumbing, the one-time `RLIMIT_NOFILE`
        raise at the first reader's creation, the two env knobs and the
        `expert_io=` field. `swift test --no-parallel --filter ParallelExpertReader`
        and `--filter PreadExpertStreamer` → FAIL then PASS. The four per-commit
        gates (release build, 0 warnings; `swiftlint lint --strict --baseline`,
        regenerating it for `PreadExpertStreamer.init`; `tools/check-md-links.py`;
        `swift test --no-parallel`, the **full** suite) **plus a task-specific
        ThreadSanitizer check**: `env
        TSAN_OPTIONS=suppressions=tsan-suppressions.txt swift test --no-parallel
        --sanitize=thread --filter ParallelExpertReader` — minutes, not the
        chapter-close hour. Justified: this is the only pthread code the chapter
        touches, the new tests run 2–3 callers against 8 threads, and TSAN
        instruments the C target too. The full run stays the close gate, and a
        report here is real (the suppressions file's family is swift-nio's future
        bridge, unrelated to this mutex).
  - [x] Step 2: `tools/golden-baseline.sh --check` on the M4 Pro at **(1, 4)** and
        **(2, 8)** — short and long **IDENTICAL** at both; a difference is a defect,
        never a recapture. Then `tools/mini-deploy.sh --restart` and the mini's
        golden check at both cells.
  - [x] Step 3 (the arms, controller-run, **one binary**, the two knobs as the A/B
        through the rig's `SERVER_ENV`; Task 1's `t1-arms.sh` is the pattern — a
        distinct tag per cell, arm and order; `pgrep -fl
        'ShrikeServer|ShrikeMac|ShrikeDecodeService|ShrikeCLI'` and
        `memory_pressure -Q` before every launch (the process raises its own
        descriptor limit — Step 1; arm B's launch is its check); a fresh server per pair, `settle_done` before the warm send):
        **A** (1, 4) and **B** (2, 8) via `tools/turn-rig.sh macmini 8081
        <promptdir> <outdir> t2-<cell>-mini-<len> pair 300|1k|2k`, **paired in both
        orders** at 300 and 1k, one pair at 2k; **C** (2, 4) and **D** (1, 8) at
        300, and at 1k if the 300 rows separate; **E** the turns arm at A and B;
        **F** the 12k control (`tools/prefill-measure.sh macmini 8081 … 6k` after
        the rig's `restore`; label `6k` = the 12,285-token prompt); **G** the
        long-decode arm at A and B, `MAX_TOKENS=512` (`turn-rig.sh:24-26`), for
        the completion count. Rows per arm exactly as Task 1 read them (`t1-rows.py`:
        wall, `prefill_s`, hits / misses / rate, `expert_read_mib`, `io_fetch_ms × 8`,
        the `prefill_routed_tile` role GPU and count, `routed→routed` total /
        `host_ms` / `queue_ms` / count, `shared→routed`, busy, span) **plus `decode_s`
        / `decode_tok_s`** from the `Shrike generation` line
        (`ServerInference.swift:1936-1941`) and the new `expert_io=` field off the
        projection line. **No new counter:** `io_fetch_ms × 8` ÷ misses is the
        realized per-expert time and the row that decides the model — it should
        **fall** at B, the parked wait Task 1 pushed inside it having disappeared.
  - [x] Step 4 (the rule): apply it to the three warm walls and every control row.
        The defaults move to the winning cell, or stay at (1, 4). **Record the
        verdict either way**, with the realized per-expert time per arm and the
        arm-D null stated explicitly. Then the four per-commit gates on the landed
        tree, golden both boxes both profiles IDENTICAL, `tools/mini-deploy.sh
        --restart` and the mini golden check.
  - [x] Step 5: design doc — a "Task 2" section with the ΣF / hidden / stage table,
        the cell table, the arms' rows and an "**After T2**" ledger block; the
        "Bytes per expert" lever entry ([v13-the-turn.md](v13-the-turn.md):266-277)
        rewritten with the measured per-expert time and whatever is left of the
        term; the deeper-lookahead follow-on repriced now that batches overlap.
        Plan: Task 2 `[x]` with the landed paragraph. Task review by a fresh
        reviewer; fixes folded into the owning commit.

  **Risks and what falsifies the model.**
  - **The realized rate does not rise.** `io_fetch_ms × 8` ÷ misses stays at
    0.57–0.58 ms with two batches published and eight threads. Then per-read latency
    at 3–4 outstanding is the floor, the probes' 3.6–3.8 GB/s was a
    synthetic-pattern number a real interleaved plan cannot reach, and the next
    lever is the **miss count** itself (the pool's size, bytes per expert), not
    concurrency — a null here retires concurrency for this chapter.
  - **Decode regresses at 8 threads.** The read is `decode_tok_s` on the
    long-decode arm; the fix is the split knobs — depth 2 with threads 4 (arm C),
    costing no descriptors and no stacks. That is why the knobs are separate rather
    than one "wider I/O" switch.
  - **A completion inversion (tile N after N+1).** The FIFO claim order is the
    guard, asserted as a pure function rather than measured. The residual race is
    *publication* order: two `ExpertIOScheduler` workers can reach `submit_batch`'s
    mutex out of submission order, as they can today. `host_ms` ÷ routed tile count
    is the read — it fell 0.85 → 0.78 and 0.74 → 0.64 ms at Task 1
    ([v13-the-turn.md](v13-the-turn.md):191-198); a rise at B with the wall flat is
    the signature, and the fix is a submission sequence number from the scheduler —
    recorded, not built here.
  - **The scheduler's four workers become the cap.** Two are needed and four exist,
    but a task raising the lookahead past one tile hits it — named so the next lever
    prices it, not discovers it.
  - **A batch's arrays reused before it completes.** Wrong expert bytes in a slot:
    golden catches it and it is a **correctness stop**, not a numerics change. The
    defaults stay at (1, 4) until it is found; never a recapture.
  - **Descriptors or stacks.** 320 open descriptors at (2, 8) against the mini's
    soft limit of 256: the process raises its own limit at the first reader's
    creation (Step 1); a launch that still fails with `EMFILE` is a defect, and
    `memory_pressure -Q` during arm B reads the stacks' cost.
  - **The 12k control moves.** It is GPU-bound (F̄ / G = 0.58 at Task 1) and flat
    is the prediction; a move means the reader's shape reaches something this model
    does not describe, and `expert_hits_prefill` (≥ 9,070) is the read.
  - **The mini is production.** Every arm stops the server on 8081 and relaunches
    it; Turbo on 8080 is never touched. One model process at a time.

### Task 3: T3 — the follow-up turn below the matrix kernels' row minimum

- [x] **T3: a card's follow-up turn is 21 new tokens, and at 21 rows the chunk
  falls off the matrix attention path, the matrix projection path and the matrix
  shared expert onto the scalar ones — the same chunk costs 43.2 ms per attention
  layer against 6.2 at 36 rows.** The design doc's lever entry reads the turn's
  cost as "the matrix kernels' per-dispatch floor at tiny row counts, 21–23 ms per
  new token over 40 layer-chunks" and prices "a decode-style or scalar path for
  tiny chunks" at ≈ 0.5–1 s
  ([v13-the-turn.md](v13-the-turn.md):390-393). **The sign is
  backwards: the scalar path is what a small chunk gets today, and it is the
  cost.** Three of the four thresholds are 32 rows, the fourth is 64, and a
  21-token user turn is below all of them; a 36-token one is above three. The
  lever is to keep the small chunk **on** the matrix kernels — they already
  accept a partial row tile by masking, which every chunk whose length is not a
  multiple of 64 already exercises — behind
  `SHRIKE_PREFILL_MATRIX_MIN_ROWS=<n>` (32 = today; the default moves on the
  verdict). Modelled at ≈ **0.60 s off a 1.50 s prefill and a 2.08 s wall** on
  the mini's measured 21-row turn, after which the routed stage (591 ms, holding
  831 misses / 1.47 GB) is 65 % of what is left — T4's term, not this one's.
  **The mini decides.**

  **LANDED a1158b6 (2026-09-04): measured on the mini on one binary (the knob as the
  A/B; the flip folded by amend), the live answer chained (`turns-live`, REUSE on
  every run after the first): the 21-row follow-up turn 2.088 / 2.083 / 2.128 →
  1.406 / 1.423 / 1.411 s, paired 2.100 → 1.413 (−32.7 %), three pairs in both
  orders, repeats ≤ 45 ms; per-role GPU attention 434 → 54 ms (modelled 54.2),
  GDN 340 → 164 (162.6), shared 67 → 22 (22.0), routed unchanged; turn 3 (36 rows)
  1.468 → 1.458, the 55-row turn 1.803 → 1.820, 300 / 1k / 2k 3.574 / 6.464 /
  10.308 → 3.635 / 6.481 / 10.307 (one pair each; no gate reachable at ≥ 305 rows;
  roles identical, the 2k hit counters ± 3 by jitter), 12k
  68.277 → 68.173, decode tok/s 14.02 / 14.17 → 14.08 / 13.99; **every completion
  byte-identical** (the three live pairs, the 55-row chain, the six whole-chunk
  completions, 12k, both long answers); golden IDENTICAL both boxes both profiles
  at 32 and at 16 (short is 13 rows — measured, Step 1; long one chunk); no
  recapture. Cells 8 / 4 equal 16 on the engaging turn and cost turn 3 +6 / +12 %
  (a chained turn's settle is a RESTORE of 14 / 29 rows, not the draft's 2 — that
  is the rewind kind — and engages below 16). The engaging turn's expert traffic
  moved by one tile (339 → 338, hits 1,737 → 1,734) as the router's top-k moved
  within the tolerance, bytes unchanged — ruled the accepted numerics change, not
  a planner defect. Real (−32.7 %, 15× the drift) and free → the default is 16.**

  **Step zero — the follow-up shape at three row counts** (mini, d3efdeb, the
  shipped defaults; a fresh server per chain, the answer chained live into the
  next turn, `temperature: 0` so the chain repeats; logs
  `~/.claude/handoffs/archive/shrike-v13-t0/t3-out/server-mini-t3-live-mini-turns-d512.log`
  (A) and `…-d512-long.log` (B), also in the session scratchpad `t3-out/`;
  `tools/turn-summary.py` prints the split, the `Shrike kernel role=` lines the
  per-role GPU):

  | request | new / cached | wall | `prefill_s` | prefill GPU | attn (10) | GDN (30) | shared (40) | routed GPU / tiles | hits / misses | `r→r` total (host) | `s→r` total (host) | source |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
  | tX (A), answered 219 tokens at 13.1 tok/s | 2,125 / 0 | 29.80 s | 10.970 | 9,616 | 1,322 | 3,905 | 373 | 3,930 / 1,187 | 0 / 9,370 | 581 (474) | 268 (244) | A:5-32 |
  | **turn 2 (A)** | **21 / 2,345** | **2.077** | **1.500** | **1,244 (59 ms/tok)** | **432.1** | **335.8** | **66.4** | **408.5 / 339** | **1,737 / 831 (67.6 %)** | **76.5 (54.2)** | **106.4 (96.5)** | **A:40-66** |
  | turn 3 (A) | 36 / 2,359 | 1.467 | 0.993 | 811 (22.5) | 61.7 | 186.1 | 22.5 | 539.0 / 443 | 2,673 / 724 (78.7 %) | 31.9 (13.2) | 70.4 (60.8) | A:74-100 |
  | **turn 2 (B), user turn padded** | **55 / 2,345** | **1.804** | **1.222** | **955 (17.4)** | **71.2** | **215.9** | **22.9** | **642.7 / 515** | **2,546 / 1,428 (64.1 %)** | **140.0 (97.3)** | **93.8 (83.8)** | **B:40-66** |
  | turn 3 (B) | 36 / 2,393 | 1.459 | 0.948 | 772 (21.4) | 61.7 | 186.4 | 22.3 | 500.1 / 413 | 2,524 / 647 (79.6 %) | 37.5 (19.1) | 59.5 (49.3) | B:74-100 |

  **The discriminating row is B's turn 2: 2.6× the tokens of A's turn 2 and 0.27 s
  faster.** Per layer the regime is unmistakable — attention **43.21** ms/layer at
  21 rows against **6.17** at 36 and **7.12** at 55; GDN **11.19** against 6.20 /
  7.20; the shared expert **1.660** ms/call against 0.561 / 0.573. The two
  independent 36-row rows agree to ≤ 0.03 ms/layer, so the 21-row row is not
  drift. **The routed GEMM does not switch**: 1.205 ms/tile at 21 rows against
  1.211–1.217 at 36 and 1.248 at 55 — flat, because its gate reads the
  *configured* chunk, not the rows (below).

  **The four thresholds, verified in the tree.** Every gate is a bare literal 32
  (the GDN scan's is 64); none has a knob today.

  | # | stage (layers) | gate | site | takes 21 rows today? | this task |
  | --- | --- | --- | --- | --- | --- |
  | 1 | q / kv / o projections (40) | `tokenCount >= 32` → the MPP matrix dispatch; else `selectedDispatch`, `chunkTokens >= 32`, → `PrefillInt4QMM`; else **one GEMV per row** | `RealForwardRunner.swift:3827`; `:113-126` (`:116`); the loop `:3895-3896` | **yes, masked** — the MPP body declares the activation tensor with extent `M` (`tensorops.metal:182-186`) and bounds its stores by `globalM < rowEnd` (`:239-247`); `tileM = 64` (`MPPPrefillInt4QMM.swift:25`) and `encode` requires only `m > 0` (`:193-207`), so a 64-row tile holding 13 or 49 valid rows is what **every** chunk whose length is not a multiple of 64 already pays | **lowers** |
  | 2 | full attention (10) | `queryCount >= matrixPathMinimumQueries` | `PrefillAttention.swift:293`, used at `:308` in `matrixPathAccepts`, gate at `:160-161` | **yes, masked** — `if (q0 >= p.queryCount) return;` and `queries_valid = min(Rq, queryCount − q0)` (`attention_matrix.metal:297-298`); `Rq = 2` for `g2k256d` (`:447`), so every odd-length chunk already runs a half tile | **lowers** |
  | 3 | shared expert (40) | `queryCount >= matrixPathMinimumRows`, and `encodeChunk` throws `chunkTooShort` below it | `PrefillSharedExpert.swift:15`, `:33`, `:114-116` | **yes** — its three projections go straight to `mpp.encode` (`:159-180`), threshold 1's kernel and masking | **lowers** |
  | 4 | routed experts (40) | `chunkTokens > matrixPathMinimumRows` on the **configured** chunk | `PrefillGroupedRoutedMoE.swift:735`; `PrefillChunkScratch.swift:124-130`; `RealForwardRunner.swift:5698` | **already on the matrix path at 21 rows** — the predicate reads the scratch layout's configured chunk (4,096 in production, `RuntimeConfiguration.swift:308`), not the rows the chunk carries; the strict `>` exists to keep the **32-token MTP draft scratch** off the matrix path for its hard memory budget (`PrefillChunkScratch.swift:124-127`), not to gate a short chunk. Measured above: 1.205 ms/tile at 21 rows | **untouched** |
  | 5 | GDN delta scan (30) | `t >= GDN.chunkTokens` (64) | `GDN.swift:417`; `RealForwardRunner.swift:4076-4077` | **yes, by padding** — the chunked kernel computes `paddedRows = chunkCount × 64` and zeroes the pad rows itself (`GDN.swift:451-453`, `:471-472`), and the factors scratch is sized from the configured 4,096-token chunk (`PrefillChunkScratch.swift:68-73`), so it exists at any row count | **untouched, priced below** |

  **What the rows attribute to which threshold.** Fit the two matrix-path rows
  (36 and 55) per layer and read the intercept as the row-independent part:
  attention 4.37 + 0.0500·rows, GDN 4.32 + 0.0524·rows, shared 0.534 +
  0.00071·rows (all ms). At 21 rows that predicts **5.42 / 5.42 / 0.549** against
  the measured **43.21 / 11.19 / 1.660**. A GDN layer has no attention kernel and
  its scan does not switch, so its whole **+5.77 ms/layer** is the projections;
  an attention layer's projections carry 0.77× a GDN layer's weight (0.055
  against 0.0717 GFLOP/token/layer,
  [v12-prefill-matrix-kernels.md](v12-prefill-matrix-kernels.md):32-35),
  so ≈ 4.4 of its **+37.79** is projections and ≈ **33.4 ms/layer is the
  attention kernel itself**. The split of the 596 ms: **attention kernel 334 ms,
  projections 218 ms (44 in the attention layers, 174 in the GDN ones), shared
  expert 44 ms.**

  **Threshold 5 is not the lever, and the rows say so.** The GDN role's whole
  row-proportional term is 0.0524 ms/row/layer, an upper bound on the serial
  scan's share of it: **≤ 33 ms at 21 rows** and ≤ 86 at 55, against the ≈ 174 ms
  its *projections* cost at 21. A free chunked scan below 64 rows would buy ≤ 33
  ms and would change the numerics of every 32–63-row chunk on top. Priced,
  recorded as a follow-on, not taken. Threshold 4 is not taken either: lowering
  it would size `routedExpertStagingRows` on the 32-token MTP scratch and raise
  the draft path's hard memory budget for no measured gain (the routed GEMM is
  already flat per tile at 21 rows).

  **The knob.** `SHRIKE_PREFILL_MATRIX_MIN_ROWS=<n>`, parsed beside
  `environmentPrefillTailTile` (`RealForwardRunner.swift:484-490`) in the shape of
  `parsePrefillFetchDepth` (`:529-545`), **clamped 3…32**, default 32 at land.
  Thresholds 1–3 read it; nothing else does. Never above 32: raising it would push
  chunks *off* the paths golden was captured with, which is a different change.
  **Why the floor is 3, not 1:** the MTP verify pair prefills exactly two rows with
  `useTwoRowProjection: true` (`RealForwardRunner.swift:1444-1466`, `:3847-3849`),
  and the prompt cache's settle re-prefilled **two** rows on every measured turn
  (`settled 2345 − rewind 2343`, A:31; `settleLiveRegion`,
  `ServerInference.swift:1479-1540`). A floor of 3 leaves both on today's kernels
  exactly, so neither the speculative path nor the cache's rewrite becomes a
  numerics change. Printed as `prefill_matrix_min_rows=<n>` on the residency line
  (`ServerInference.swift:821-824`) through a static description, asserted in a
  host test as `prefillTileDepthDescription` is
  (`PrefillRoutedTileSchedulerTests.swift:320-327`).

  **Candidate values and the crossover.** **32** (today, the A/B), **16**, **8**,
  **4**. n = 16 is the value that takes the 21-row turn onto the matrix paths
  while leaving the golden `short` profile (≈ 12 rows) and the two-row settle
  alone — golden IDENTICAL, no recapture. Lower values probe the crossover: the
  tiled attention kernel dispatches `queryCount × numQHeads` threadgroups each
  walking the whole KV (`PrefillAttention.swift:222-231`), so it is linear in
  rows — 33.4 ms/layer at 21 rows = **1.59 ms per row per layer** — against the
  matrix path's row-independent **4.37 ms/layer**, putting the crossover at
  ≈ **2.7 rows** on a 2.3k KV (modelled; both terms scale with `kvValidCount`, so
  it is roughly KV-independent). Cells C and D measure it.

  **The decision rule, under the chapter's real-and-free rule**
  ([v13-the-turn.md](v13-the-turn.md):401-418). The knob lands
  either way. **No percentage bar.** The default moves to the best measured n only
  if the effect is **real** — the 21-row turn-2 wall improves with the sign
  holding across **paired runs in both orders**, three pairs at the leading
  candidate against n = 32, a fourth if a shape sits inside twice its drift. The
  drift on this rig: the two independently-run 36-row rows read 1.467 / 1.459 s,
  the three archived post-T0 turn-2 rows repeat within 42 ms, and turn 3 spreads
  122 ms across five archived rows; the modelled effect is ≈ 600 ms, 5–15× that.
  And **free**: no control row regresses. Those rows are the **36-row turn 3 and
  the 55-row turn 2 unmoved and their completions byte-identical** (both above
  every candidate n — if either moves, the gate engaged where it must not), the
  300 / 1k / 2k warm walls and completions **byte-identical**, the 12k control
  ± 1 % with `expert_hits_prefill` ≥ 9,070, `expert_hit_rate_prefill` on the
  engaging turn **unmoved** (this task changes no plan and fetches no expert
  differently, so a move is a defect, not a cost), `decode_tok_s` on the
  long-decode arm not regressed, `memory_pressure -Q` acceptable before every
  launch, and the numerics qualification below passed with golden recaptured only
  where the policy says. Otherwise the default stays 32 and the rows are recorded
  (P8's precedent); a measured null is a result.

  **Numerics — exactly what moves.** This runs different kernels on a chunk, so
  v12's policy applies: the **2e-2** bar against the reference and golden
  recaptured once per box with before/after digests in the verdict
  ([v13-the-turn.md](v13-the-turn.md):420-426). **Changes:** every
  chunk of n…31 rows — the follow-up turn and short tool round (the target), the
  tail chunk of any prompt whose length mod 4,096 lands in [n, 31], the prompt
  cache's settle re-prefill if its remainder lands there — a first request's
  rewind settle is 2 rows (never); a chained turn's restore settle is 14 rows at
  turn 2 (not at 16) and 29 at turn 3 (**engages at the default**, shaping an
  unmeasured turn 4; corrected at the verdict), and golden `short` if its chunk is ≥ n. **Unchanged and
  byte-identical:** the 305 / 1,085 / 2,125 first turns, 6,381 (4,096 + 2,285),
  12k (4,096 + 4,096 + 4,093), turn 3 at 36, turn 2 at 55, golden `long` (≈ 2k in
  one chunk), decode, the MTP verify pair at 2 rows, and the routed GEMM at every
  size. **Golden `short` is a 51-byte prompt** run through `ShrikeCLI` with no
  chat template (`tools/golden-baseline.sh:47`, `:81`, `:92-93`) — ≈ 12 rows,
  **modelled, not measured**: Step 1 reads the real count off a non-`--quiet` run
  before any n is chosen, and if it is ≥ n then `short` **will** move and is
  recaptured with digests. A `long`-profile difference is a defect, never a
  recapture. **One gap named, not discovered:** golden never exercises the
  settle, because it runs the CLI and not the server — the settle's check is the
  rig's byte-diff of the turn chain plus the `settled − rewind` remainder read
  per arm.

  **The qualification, extended to sub-32 rows.** Three existing suites already
  compare these kernels against a reference; each gains small-row cases.
  `PrefillAttentionMatrixTests.swift` holds an fp32 CPU reference with
  `tolerance = 2e-2` on `RelError.maxAbsDiff` and `RelError.compute` (`:13`,
  `:58-71`) over chunks 64 / 130 / 40 and 96 / 64 / 130 / 37 / 64 (`:16-27`),
  **none below 32** — and **the trap**: `run` calls `encodeCausal(…, path:)`
  (`:348-353`), which re-tests `matrixPathAccepts`, so a sub-32 case added
  naively runs the *tiled* kernel and passes vacuously; the new cases must pass
  the lowered minimum through and assert the gate accepted, and
  `gateAcceptsOnlyTheMatrixShape`'s `queryCount = 8` rejection (`:250-252`)
  becomes "rejected at 32, accepted at 8". `MPPPrefillInt4QMMTests.swift` has
  `cpuReference` (`:104-137`) with the same bars (`:212-218`) over `variantShapes`
  m = 64 / 33 / 128 (`:223-227`) — add m = 21 and m = 3.
  `PrefillSharedExpertTests.swift`'s
  `matrixPathSelectionRequiresAvailableMatchingMPP` (`:230-245`) uses `rows - 1`
  as its reject case and takes the minimum as a parameter.

  **The expected gain, modelled from the fits above.** Holding the routed stage
  and every gap at their measured values:

  | term (turn 2, 21 rows) | today | modelled at n ≤ 21 | source |
  | --- | ---: | ---: | --- |
  | attention role GPU (10) | 432.1 | 54.2 | A:42; fit |
  | GDN role GPU (30) | 335.8 | 162.6 | A:44; fit |
  | shared expert GPU (40) | 66.4 | 22.0 | A:47; fit |
  | routed tile GPU (339) | 408.5 | 408.5 | A:43, untouched |
  | `prefill_moe_reduce` (40) | 0.9 | 0.9 | A:53 |
  | **prefill GPU** | **1,243.7** | **648.2** | sum |
  | gaps (`r→r` 76.5, `s→r` 106.4, reduce→gdn 37.9, reduce→attn 11.4, routed→reduce 14.5) | 246.7 | 246.7 | A:57-62 |
  | remainder (chunk setup) | 9.6 | 9.6 | `prefill_s` − the above |
  | **`prefill_s`** | **1,500** | **≈ 905** | A:40 |
  | **server wall** | **2.077 s** | **≈ 1.48 s** | A:66 |

  **Then max(GPU, fetch), and the GPU still wins.** 831 misses × 1,769,472 bytes
  = **1.470 GB**; at T2's realized 3.08 GB/s and the peer's mini probes' 3.57
  GB/s that is **0.41–0.48 s** of drive
  ([v13-implementation-plan.md](v13-implementation-plan.md):742-745).
  All of it lives inside the routed stage (the speculative prefetch is off by
  default, and no cross-stage prefetch runs), and that stage measures **591.4 ms**
  — GPU 408.5 + `r→r` 76.5 + `s→r` 106.4 — which already covers it. So
  max(648.2 GPU, ≈ 450 fetch) is the GPU, the cut reaches the wall, and the
  optimistic variant is only that the `r→r` / `s→r` host terms shrink with the GPU
  as they do between the 21- and 36-row rows (183 → 102 ms), taking the wall
  toward ≈ **1.40 s**. **Cross-check:** the modelled 21-row turn (905 ms prefill,
  1.48 s wall, 831 misses) sits between the two measured matrix-path rows — 36
  rows at 948–993 ms / 1.459–1.467 s with 647–724 misses, and 55 rows at 1,222 ms
  / 1.804 s with 1,428. A 21-row turn should cost no more than a 36-row one;
  today it costs 42 % more. **What is left afterwards** is the routed stage,
  ≈ 591 of ≈ 905 ms (**65 %**), holding 831 misses at 67.6 % hits against turn
  3's 78.7–79.6 % on a nearly identical context — T4's term, not this one's.

  **The rig.** The live-answer phase moves under `tools/`: `turn-rig.sh` gains
  **`turns-live [answer_max_tokens]`** (the session's `t3-turns-live.sh` is the
  prototype) — tX with the answer budget, `wait_settle 1`; turn 2 built from tX's
  **actual response content** plus the last user message of `tXturn2.json`,
  `max_tokens` 8, `wait_settle 2`; turn 3 built the same way from turn 2. **The
  built payloads are written beside the responses in `<outdir>`, and a
  `REUSE=<dir>` env sends the stored ones instead of rebuilding** — mandatory,
  not a convenience: once the path engages, arm B's turn-2 completion differs from
  arm A's, so a rebuilt turn 3 would send different bytes and the pair would not
  be an A/B. (tX is unaffected by the knob at 2,125 rows, so turn 2's payload is
  safe to rebuild.) A `USER_TURN2=<payload>` override selects the user turn and
  therefore the row count — that is how the 55-row row above was produced.
  **Rows** exactly as `t2-rows.py` reads them
  (wall, `prefill_s`, new, hits / misses / rate, `expert_read_mib`,
  `io_fetch_ms × 8`, routed tile GPU and count, `r→r` total / host / queue /
  count, `s→r` host, busy / span, `decode_s` / `decode_tok_s` / completion) **plus
  the `prefill_attn_router` / `prefill_gdn_router` / `prefill_shared_expert` /
  `prefill_moe_reduce` role GPU** and the `settled` / `rewind` pair per request.
  No new counter. **The `suffix` phase must not be used for threshold edges:**
  `turn-prompts.py` inserts entries *before* the trailing "Summarize:" line
  (`tools/turn-prompts.py:23-33`), so `tXp4` / `tXp16` diverge inside the stored
  entry and re-prefill in full — `settle_reset reason=no_prefix_snapshot`,
  `cached=0`, `new=2385` and `new=3165` at
  `~/.claude/handoffs/archive/shrike-v13-t0/server-mini-turn-suffix.log:64-65`,
  `:97-98`. The turns chain appends correctly (`settle_restore`, `cached=2345`).

  **Tests (RED first, host-only unless noted).**
  - `matrixPathAcceptsHonoursALoweredMinimum` — `PrefillAttention.matrixPathAccepts`
    as a pure function: `queryCount = 21` rejected at 32, accepted at 16, rejected
    at 22; every other clause of the gate (`:299-310`) still rejecting at the
    lowered minimum.
  - `sharedExpertMatrixPathHonoursALoweredMinimum` — the same for
    `PrefillSharedExpert.matrixPath`, plus `encodeChunk` no longer throwing
    `chunkTooShort` at a row count the lowered minimum admits.
  - `projectionDispatchPolicyHonoursALoweredMinimum` — `selectedDispatch` as a
    pure function over (family, chunkTokens, minimumRows): below the minimum
    every family is `.repeatedGEMV`; at or above it `.kv` / `.o` are `.qmm` and
    `.q` stays `.repeatedGEMV`.
  - `parsePrefillMatrixMinRowsClampsToTheSupportedRange` — unset → 32, `"16"` →
    16, `"0"` / `"2"` → 3, `"99"` → 32, garbage → 32 (mirroring
    `parsePrefillFetchDepth`'s test shape).
  - `prefillMatrixMinRowsDescriptionReportsTheThreshold` — the printed field,
    through the static description.
  - **Device, numerics:** the three suites above extended —
    `matrixMatchesReferenceOnFP16Cache` and
    `matrixMatchesTiledOnQuantizedCache` gaining chunks **21** and **3** with the
    minimum passed through and the gate asserted; `MPPPrefillInt4QMM` gaining
    m = 21 and m = 3 against `cpuReference`; the shared expert's chunk path at 21
    rows against its own `encodeBlock` reference. `RelError.maxAbsDiff` and
    `RelError.compute` ≤ 2e-2 on all of them.

  **Files.** `sources/Shrike/Kernels/Attention/PrefillAttention.swift`:
  `matrixPathAccepts` and `encodeCausal` gain a defaulted `minimumQueries:`
  (default `matrixPathMinimumQueries`, which stays 32).
  `sources/Shrike/Kernels/Prefill/MoE/PrefillSharedExpert.swift`: `matrixPath` and
  `encodeChunk` gain a defaulted `minimumRows:`.
  `sources/Shrike/Runtime/Inference/RealForwardRunner.swift`: the parser and its
  default (`:484-490`), `PrefillProjectionDispatchPolicy.selectedDispatch`
  (`:113-126`), the inline gate in `encodeAffineProjection` (`:3827`), the
  attention call site in `encodeFullAttentionPrefill` (`:4401-4620`), the shared
  expert's at `:5039-5044`, and the printed description beside
  `prefillProjectionPath` (`:194-198`). `sources/ShrikeServer/Core/ServerInference.swift`:
  the new field on the residency line (`:821-824`). `tools/turn-rig.sh` and
  `tools/turn-prompts.py`: the `turns-live` phase and its payload reuse. Tests:
  the five files above. **Lint:** `encodeFullAttentionPrefill` is in
  `.swiftlint-baseline.json` at "currently spans 203 lines"
  (`RealForwardRunner.swift:4401`) — the reason string embeds the count, so any
  edit to that body makes the entry stale and the baseline must be regenerated
  (T0, T1 and T2 each hit this on a different function). `encodeAffineProjection`
  is 91 lines and not in the baseline; keep it there.
  **Unchanged:** every `.metal` file and every kernel body; `PrefillGroupedRoutedMoE`,
  `PrefillRoutedTileScheduler` and **both routed tile loops** — so the Task 1
  collapse follow-on is *not* forced by this task and the baseline entry for
  `encodeRoutedMoEPrefill` ("currently spans 244 lines", `:5203`) stays valid;
  `GDN`; `PrefillChunkScratch`; `PreadExpertStreamer`; `expert_io.c`; the prompt
  cache; decode; `PrefillInt4QMM` (the `selectedDispatch` fallback is unreachable
  on both boxes today — both print
  `prefill_projection_path=affine-threadgroup-f16`, A:3 — so it is covered by the
  host test only).

  Steps:

  - [x] Step 1 (controller, **no code**, the two facts the design needs before an
        implementer starts): `pgrep -fl
        'ShrikeServer|ShrikeMac|ShrikeDecodeService|ShrikeCLI'` and
        `memory_pressure -Q` first. (a) The golden `short` profile's real chunk
        rows, from one non-`--quiet` `ShrikeCLI` run of `SHORT_PROMPT` on the M4
        Pro — the ≈ 12 above is modelled, and it decides whether `short` is
        recaptured. (b) The settle remainder (`settled` − `rewind`) across the
        archived turn logs, confirming the 2 rows the measured chain shows is the
        shape and not a coincidence. Neither is a model run on the mini.
  - [x] Step 2 (an implementer): the `turns-live` phase and its payload reuse in
        `tools/turn-rig.sh` first — every later task on this shape needs it — then
        the failing tests, then the parser, the three thresholds' parameters, the
        printed field. `swift test --no-parallel --filter PrefillAttentionMatrix`,
        `--filter MPPPrefillInt4QMM`, `--filter PrefillSharedExpert` → FAIL then
        PASS. The four per-commit gates (release build, 0 warnings; `swiftlint
        lint --strict --baseline`, **regenerating it** for
        `encodeFullAttentionPrefill`; `tools/check-md-links.py`; `swift test
        --no-parallel`, the **full** suite).
  - [x] Step 3 (numerics qualification): on the M4 Pro, `tools/golden-baseline.sh
        --check` at `SHRIKE_PREFILL_MATRIX_MIN_ROWS=32` — short and long
        **IDENTICAL**, proving the knob at today's value is today's build. Then at
        each candidate n: **long IDENTICAL** always; **short** identical while
        n > its measured rows, expected to differ once n ≤ them — recapture once,
        digests before and after kept for the verdict. Then `tools/mini-deploy.sh
        --restart` and the same pair on the mini. A long-profile difference is a
        defect, never a recapture.
  - [x] Step 4 (the arms, controller-run, **one binary**, the knob as the A/B
        through the rig's `SERVER_ENV`; a fresh server per phase, `settle_done`
        before each send, a distinct tag per cell / arm / order, `REUSE` set on
        every arm after the first): **A** n = 32 and **B** n = 16 via `turns-live`,
        **paired in both orders, three pairs**; **C** n = 8 and **D** n = 4, one
        pair each, for the crossover; **E** the 55-row control (`USER_TURN2` the
        padded turn) at A and the winner, completions diffed byte for byte; **F**
        the whole-chunk controls `pair 300|1k|2k` at A and the winner, completions
        diffed byte for byte; **G** the 12k control (`tools/prefill-measure.sh
        macmini 8081 … 6k`); **H** the long-decode arm (`MAX_TOKENS=512`) at A and
        the winner for `decode_tok_s`. Read the **per-role GPU split**, not the
        wall alone: if the attention and GDN roles do not fall at B, the model is
        wrong and the task stops before the rule is applied.
  - [x] Step 5 (the rule): apply it to the 21-row turn-2 wall and every control
        row. The default moves to the winning n, or stays 32. **Record the verdict
        either way**, with the per-role split per arm and the crossover the C / D
        cells show, so the reader can see which threshold paid. Then the four
        per-commit gates on the landed tree, golden both boxes both profiles (long
        IDENTICAL, short at the recaptured digests where the policy said so),
        `tools/mini-deploy.sh --restart` and the mini golden check.
  - [x] Step 6: design doc — a "Task 3" section with the three-row-count table,
        the four thresholds, the per-role attribution and an "**After T3**" ledger
        block; the "A small-row prefill path" lever entry
        ([v13-the-turn.md](v13-the-turn.md):390-393) rewritten with
        the sign corrected and the measured mechanism; the GDN chunked scan below
        64 rows and the routed gate's MTP memory constraint added to Follow-ons.
        Plan: Task 3 `[x]` with the landed paragraph. Task review by a fresh
        reviewer; fixes folded into the owning commit.

  **Risks and what falsifies the model.**
  - **The attention kernel is not the term.** The 334 ms attributed to it rests on
    one modelled step: that an attention layer's scalar projections cost 0.77× a
    GDN layer's, scaled by v12's per-layer GFLOP table. If arm B's **GDN** role
    falls to ≈ 163 ms but its **attention** role does not fall to ≈ 54, the split
    is wrong and the remaining term is the attention kernel's occupancy at 11
    threadgroups (`queryCount / 2 × 2` KV heads), not its path — which no
    threshold reaches and which would retire T3 with the projections' ≈ 218 ms
    banked.
  - **The scalar path is faster at some row count.** The crossover is modelled at
    ≈ 2.7 rows from a two-point fit, so cells C and D exist to measure it. If
    n = 4 is worse than n = 8, the clamp's floor of 3 is decoration and the
    verdict records the measured knee.
  - **The gate engages where it must not.** A prompt whose length mod 4,096 lands
    in [n, 31] has a tail chunk that engages — a real widening of the blast
    radius, named here rather than discovered. The 12k (4,093) and 6k (2,285)
    controls do not have such a tail; the byte-diff of the 300 / 1k / 2k
    completions and the 36- and 55-row turns is the check. The conservative
    variant, if a control moves, is to gate on the request's uncached token count
    rather than the chunk's rows, at the cost of leaving a long prompt's tail
    chunk on the scalar path.
  - **The settle changes and nothing catches it.** Golden runs the CLI, which
    never settles, so a settle whose remainder lands in [n, 31] changes the stored
    KV and therefore the *next* turn's output with no golden signal. The floor of
    3 keeps the measured 2-row settle exact; the rig's byte-diff of the turn chain
    and the `settled − rewind` row per arm are the only other checks, and they are
    named as the coverage, not assumed.
  - **Numerics drift beyond 2e-2 at small rows.** The device tests are the stop
    before the arms. The masking is already exercised in production — a
    2,125-token chunk's last projection tile carries 13 valid rows of 64, and
    every odd chunk runs a half attention tile — so a failure at 21 rows would be
    a bug in the *gate*, not a tolerance question.
  - **`expert_hit_rate_prefill` moves on the engaging turn.** This task changes no
    plan and fetches no expert differently; a move is a defect (a changed tile
    composition or sweep parity), and the verdict does not proceed.
  - **The mini is production.** Every arm stops the server on 8081 and relaunches
    it; Turbo on 8080 is never touched. One model process at a time.

### Task 4: T4 — the pool's retention across the turn boundary, and where its miss count is actually exposed

- [x] **T4: the expert pool's eviction policy is the chapter's last untouched
  term, and step zero re-prices it. Task 3 left the follow-up turn saying "the
  routed stage is now the largest term and its drive is the miss count on a cached
  context" ([v13-the-turn.md](v13-the-turn.md):450-452). Measured, that overstates
  it: on turn 3 the `lru` policy removed 304 of 721 prefill misses (42 %) and
  `prefill_s` moved 0.985 → 0.981 s, inside a 16 ms drift, because that turn's
  routed stage sits at its GPU floor (412 ms for 338 tiles) and at the drive's
  measured ceiling (832 misses = 1.472 GB in 412 ms = 3.57 GB/s, the peer's probe
  number, [v13-implementation-plan.md](v13-implementation-plan.md):747) at the same
  time. A perfect pool buys ≈ 10 ms there. **The miss count is exposed in decode,
  not in a follow-up prefill**: four independent measurements put a decode miss at
  ≈ 0.93 ms of wall and tX's 219-token answer carries 7,451 of them, **6.9 s of a
  16.83 s decode, 41 %**. A clairvoyant policy at 128 slots per layer, replayed on
  the measured route trace, removes 1,846 (**−1.7 s, −10 % of decode, 13.01 → 14.50
  tok/s modelled**); `lru` realizes 502 of that on the box (+2.8 % tok/s) and pays
  for it on the 8-token turns. So the task lands the trace's missing prefill line
  and an offline replay first, prices every candidate against Belady before an arm
  runs, and lands a policy behind `SHRIKE_EXPERT_CACHE_POLICY` either way. **The
  mini decides**, and a measured null is a result
  ([v13-the-turn.md](v13-the-turn.md):488-505).

  **LANDED 04d4de5 (2026-09-05): the eviction policy was not the lever and the plan's Step
  2 as drafted (a policy case) was superseded by Step 1's offline verdict; the task landed
  the route trace's prefill and request lines, `tools/expert-pool-replay.py` (validated
  against production at ±1 miss on six requests across two shapes; Belady 2,485 / 3,773
  against production's 7,451 / 10,059 on the two answers; every steady-state rule in
  `lru`'s class; the recency-ordered sweep 6,697 / 9,450), two scheduling-only levers
  behind knobs, and measured them in three rounds on the mini: **`SHRIKE_EXPERT_CACHE_PROTECT=chunk`
  is the default** (chunk-aware victim selection; paired A/C: the 21-token follow-up
  1.430 → 1.385 s (−3.1 %), the warm 300 prompt after a long
  answer 3.439 s → 3.175 s (−7.7 %), the warm 300 pair
  3.582 s → 3.454 s (−3.6 %), 12k 68.105 s → 68.209 s
  (+0.15 %), decode unchanged, no row worse; golden IDENTICAL both boxes both profiles
  at every cell); **`SHRIKE_PREFILL_SWEEP=recency` stays a knob at default `carry`** (with
  `SHRIKE_PREFILL_SWEEP_TAIL`, default 96): it takes 0.6–0.8 s off the first turn's decode
  after a large prompt exactly as replayed (6,692 / 6,701 measured against 6,697 / 6,688)
  and forfeits T0's carry benefit on consecutive chunks (the 300 / 1k / 2k pairs' warm
  prefill +3.7 / +7.5 / +10.9 %, 12k +7.2 %), so it is not free; its resident-first
  refinement is the next task. Two costs the replay could not see were found and fixed on
  the box before the verdict (tile balance, +0.47 s; host time in the composition and the
  per-tile protection, +0.44 s). Real and free → protection is the default.**

  **Step zero: the term as measured** (mini, the deployed a1158b6 binary, bare
  launch, zero code). The card's follow-up chain: `turns-live 512`, tX answered
  live at 219 tokens, `temperature: 0`, turns 2 and 3 REUSEing the default chain's
  payloads (`t3-out/t3-default-mini-live/`) so every arm sent identical bytes.
  **Every completion is byte-identical across all four arms below** (tX 672 chars,
  turn 2 27, turn 3 32; verified by diffing the responses). Logs at
  `~/.claude/handoffs/archive/shrike-v13-t0/t4-out/<tag>/`, also in the
  controller's scratchpad; rows from `t4-rows.py`.

  | request | new / cached | wall | `prefill_s` | prefill hits / misses | lookups / layer | MiB read | routed GPU / tiles | busy / span (ms) | decode s / tok / tok/s | decode hits / misses | evict / reload |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | tX (2,125, answered 219) | 2,125 / 0 | 29.72 s | 11.148 | 0 / 9,370 (0 %) | 234 | 28,385 | 3,943 / 1,187 | 19,740 / 27,942 | 16.828 / 219 / **13.01** | 62,309 / **7,451** (89.32 %) | 11,701 / 6,907 |
  | **turn 2** | **21 / 2,345** | **1.422** | 0.844 | 1,734 / **832** (67.6 %) | 64 | 1,809 | 412 / 338 | 955 / 1,361 | 0.518 / 8 / 15.46 | 2,000 / 240 (89.29 %) | 1,072 / 1,060 |
  | turn 3 | 36 / 2,359 | 1.461 | 0.985 | 2,676 / **721** (78.8 %) | 85 | 1,380 | 538 / 443 | 1,108 / 1,359 | 0.413 / 8 / 19.39 | 2,143 / 97 (95.67 %) | 818 / 809 |

  **The zero-code A/B: the three existing `SHRIKE_EXPERT_CACHE_POLICY` values on
  the same chain**, one launch each. The miss counts are exact (identical routes);
  the walls are single runs.

  | policy | tX tok/s | tX decode misses | turn 2 wall / `prefill_s` | t2 prefill / decode misses | turn 3 wall / `prefill_s` | t3 prefill / decode misses |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | `aging-lfu` (default) | 13.014 | 7,451 | 1.422 / 0.844 | 832 / 240 | 1.461 / 0.985 | 721 / 97 |
  | `lru` | **13.369** | **6,949** | 1.606 / 0.870 | 869 / **368** | 1.512 / 0.981 | **417** / 179 |
  | `lfu` | 12.941 | 7,451 | 1.429 / 0.854 | 832 / 240 | 1.474 / 1.001 | 721 / 97 |

  Drift on the default across four launches (the three above plus the trace arm):
  turn 2 wall 1.422 / 1.429 / 1.456 and T3's own 1.406 / 1.423 / 1.411; tX 13.014
  / 12.941 / 13.099 / 13.20; turn 3 `prefill_s` 0.985 / 1.001 / 0.993. So ≈ ±1 %
  on a wall and ≈ ±16 ms on `prefill_s`. LRU's +2.8 % on tX is above it (single
  run), its +184 ms on turn 2 far above it, and every miss count is exact.

  **`lfu` equals `aging-lfu` to the counter on every request, and the arithmetic
  says why.** The two policies differ in exactly one place: the halving of every
  count when `statisticsPlans` is a multiple of 1,024
  (`PreadExpertStreamer.swift:664-670`); the victim comparator has no other policy
  branch (`:1179-1192`, only `if cachePolicy == .lru`). `statisticsPlans` is a
  per-instance field (`:314`) and there is one streamer per layer
  (`Model.swift:552`), so the period is per layer. This chain makes, per layer, tX
  30 prefill tiles (1,187 / 40) + 218 decode plans, turn 2 8 + 7 and turn 3 11 + 7
  = **281 plans**. **The halving never fires**, and the `lfu` row proves it to the
  unit: the pool runs **plain lifetime LFU** for any conversation under ≈ 1,000
  decode tokens. `expertUseCount` is written only at `:514` (allocation), `:668`
  (halve) and `:699` (increment), with no reset anywhere, so the counts are a
  **global popularity prior since process launch**. Whether that is the design is
  a question for the verdict, not an assumption here. On a card the halving first
  fires inside the second or third answer (a 600-token answer is ≈ 630 plans per
  layer). After tX's sweep every routed expert has count ≈ 1 (234 lookups per
  layer over ≈ 234 distinct experts); the 219 decode tokens then add 1,744 uses
  per layer over a mean of 177 distinct experts (measured on the trace), so
  decode's set carries counts an order of magnitude above any prefill's.

  **The policy family moves BOTH regimes, in opposite directions.** LRU takes 502
  misses off a 219-token decode and 304 off turn 3's prefill, and puts 128 onto
  turn 2's 8-token decode, 82 onto turn 3's and 37 onto turn 2's prefill: a
  follow-up prefill's 64 experts per layer are the most recent, so under LRU they
  displace decode's set, while under aging-LFU they carry count 1 and are the first
  victims. **Neither policy holds the union.** The replay below shows it is
  holdable in principle.

  **Why 832 misses on turn 2 and 721 on turn 3.** A 21-token chunk's 168 routed
  slots per layer collapse to **64 distinct experts** (2,566 lookups / 40, T3's
  measured figure) and turn 3's 36 tokens to **85**: both fit inside 128 slots, so
  neither is a capacity failure of the chunk. They miss because the pool holds the
  *previous* request's decode winners. That the hit rate is nonetheless 67.6 % and
  78.8 % says the answer's high-count residents already cover two thirds of what a
  new user turn routes to, which is what a shared prefix plus the chat template's
  recurring role tokens would give. **Which of the chunk's experts decode had
  evicted is exactly what the trace cannot answer today**, because prefill's plans
  are not recorded: that is Step 1's line, and the replay's failure on turn 3 below
  is the measurement of the gap.

  **The pool's mechanics, verified in the tree.**
  - One `PreadExpertStreamer` per layer (`Model.swift:552`), built lazily on that
    layer's first routed touch. `slotCount` comes from `--ram-budget`
    (`ServerArguments.swift:113-124`) through
    `RuntimeConfiguration.expertCacheSlots` (`:202-213`) at
    `ServerInference.swift:687-692`: `perSlot` = stride 1,769,472 B × 40 layers =
    70,778,880 B, `wanted` = 8 GiB / that = 121.4, snapped to the nearest of
    `[8, 16, 24, 32, 64, 96, 128]` (`:142`) = **128 slots per layer** for 256
    experts at top-k 8, which is the 9.06 GB CLAUDE.md records.
  - Policy `ExpertCachePolicy` (`PreadExpertStreamer.swift:164-168`), default
    `.agingLFU` (`:260`), env parse at `:342-349` (fails closed; the error text
    lists `lfu, lru, aging-lfu`). **The parse lives in the streamer's `init`**, run
    per layer, so a bad value throws mid-request forty times rather than at launch,
    and it never reaches `RuntimeConfiguration.expertCachePolicy`
    (`ServerInference.swift:702` passes the CLI value).
  - A **plan** is one `makeExpertCachePlan` (`:641`): decode makes one per layer
    per token (`RealForwardRunner.swift:6397` → `ModelExpertIO.swift:113`, `:123`),
    prefill one per tile of 8 experts (`RealForwardRunner.swift:5378`, `:5594`,
    `:5678` → `ModelExpertIO.swift:127`, `:134`). Hit slots are reserved before
    victim selection (`:686`), `useClock` advances per plan (`:663`, `:697`) and
    `expertUseCount[e] += 1` per expert per plan (`:698-699`).
  - Victim order (`selectVictimSlots` `:1152-1177`, `shouldEvictSlot` `:1179-1192`):
    the LFU family takes the lowest count first with ties by oldest `slotLastUse`,
    LRU takes `slotLastUse` alone, loading and pinned slots are ineligible
    (`:1156-1157`) and ties resolve to the lower slot index. **No host test covers
    any of this today** (nothing in `tests/` names `ExpertCachePolicy` outside two
    `RuntimeConfiguration` round-trips).
  - Counters (`ServerInference.swift:1986-2005`). `expert_evictions`
    (`PreadExpertStreamer.swift:710`) is **arithmetically redundant**: tX's 16,821
    misses minus the 5,120 initially empty slots (128 × 40) = **11,701, the
    measured value exactly**, turn 2 832 + 240 = **1,072**, turn 3 721 + 97 =
    **818**. `expert_reloads` (`:1411`, `:1418`) is a **lifetime** flag, so once
    tX's sweep has loaded 9,914 distinct (layer, expert) pairs of 10,240 every
    later miss counts as a reload: it never says when the eviction happened.
    `expert_rank_mass` (`:1976-1984`) is router-weight mass by rank, not frequency
    headroom.
  - **A speculative plan still mutates the pool.** `abandonExpertCachePlan`
    (`:1358-1377`) resets only the loading slots, rolling back neither
    `expertUseCount` nor `slotLastUse` nor the statistics. The path is reachable
    only from the depth-1 tile loop (`RealForwardRunner.swift:5421-5423`), which
    the shipped fetch depth 2 does not run, so production plans each tile exactly
    once and a `SHRIKE_PREFILL_FETCH_DEPTH=1` capture would not replay cleanly.

  **Prior art, placed rather than re-derived.** v12 Task 17's side finding that
  plain decode is hit-rate-bound
  ([v12-prefill-matrix-kernels.md](v12-prefill-matrix-kernels.md):1163-1166: body
  39.6 / 48.2 / 72.5 ms, 22.0 / 18.4 / 12.5 tok/s as the hit rate falls 0.99 →
  0.88) is this task's term from the other end. v10's P3 entry and its 2026-09-04
  follow-on (`docs/v10-implementation-plan.md`; the follow-on is an uncommitted
  peer edit in the working tree, archived at
  `~/.claude/handoffs/archive/shrike-ssd-split-probe/`) closed the fetch-speed
  lever (the mini's 1.77 MB read is at its ≈ 0.77 ms floor; splitting, padding and
  file-splitting are all null on the deploy target) and named what survives in
  this task's words: "policy-side (reduce miss COUNT via residency / cache)". Its
  ramp probe also found that a 5 ms idle gap doubles the next read (`mini-ramp-run1.txt`
  rows B and E: 64 KB p50 0.123 ms back-to-back, 0.278 ms after the gap), a caveat
  for anything that fetches between turns. **`docs/architecture.md:76-119` holds
  two different predictors and only one is disproven**: the proof (`:78-89`) kills
  *same-layer, previous-token* prediction (0.00 % of misses caught at both 16 and
  128 slots), while `ExpertPrefetchRing.swift` / `SHRIKE_PREDICTIVE_PREFETCH`
  implement *next-layer* prediction, which predicts well (64.1 % recall) but did
  not pay end to end and whose door the distance experiment closed (`:91-118`).
  **T4 is neither**: eviction order is not prediction, and that passage ends "the
  surviving miss levers are cost-side (event gating, free-running) and policy-side
  (cache)" (`:117-118`). Candidate (c) is the one part of T4 that touches
  prediction and inherits that bar.

  **Where the miss count is exposed: two regimes, two answers.**

  1. **The follow-up turn's prefill: ≈ nothing.** Turn 2's `prefill_s` is 844 ms,
     of which the role GPU is 654 (attention 54 + GDN 165 + shared 22 + routed 412
     + reduce 0.9), leaving 190 ms of idle: the two named host gaps are 128 (`s→r`
     77, `r→r` 51) and the three smaller ones ≈ 62. Nothing is left over for
     exposed fetch. The drive's occupancy (`io_fetch_ms × 8` = 952 ms over 1,072
     misses = 0.888 ms per expert, T2's 0.84-1.02 ms) needs mean concurrency ≥ 1.79
     to fit under the routed stage's 412 ms of GPU, which 4 threads over 2
     published batches supply; in bandwidth terms 832 × 1.6875 MiB = 1.472 GB in
     412 ms is 3.57 GB/s, exactly the peer's probe ceiling. **Turn 2's routed stage
     is at its GPU floor and at the drive's ceiling at the same time.** Modelled, a
     perfect pool takes **X = 0 to 66 ms** (66 if the drive only realizes T2's 3.08
     GB/s: 478 − 412). Measured, turn 3's LRU arm removed 304 of 721 prefill misses
     and `prefill_s` moved 4 ms against a 16 ms drift, so **X ≈ 10 ms** scaled to
     all 721, and halving turn 2's 832 buys **Y ≈ 5 ms**. Both are twenty times
     under 0.1 s. **The follow-up turn is not where this lever pays**, and Task 3's
     closing sentence ([v13-the-turn.md](v13-the-turn.md):450-452) overstated it;
     what is left of that stage is its per-tile GPU (1.22 ms × 338 tiles at 21
     rows), a kernel question recorded as a follow-on.
  2. **Decode: the chapter's largest single term.** tX answers 219 tokens at 13.01
     tok/s with 7,451 misses = 34.0 per token = 0.85 per layer-token of 8 lookups
     (hit rate 0.8932). Four independent measurements of what one miss costs: the
     LRU arm's decode deltas on tX 0.447 s ÷ 502 = **0.890 ms**, turn 2 0.160 ÷ 128
     = **1.250**, turn 3 0.056 ÷ 82 = **0.683**, pooled 0.663 ÷ 712 = **0.931**;
     and independently v12 T17's 3.0 ms per token per point of hit rate over 3.2
     misses per point = **0.94 ms**. At 0.93 ms, tX's 7,451 misses are **6.93 s of
     a 16.83 s decode: 41 %**, and a card's 600-800-token answer (46-61 s) carries
     the same 41 %: **19-25 s per card**.

  **The bound: the decode route trace, replayed.** `SHRIKE_ROUTE_TRACE` produced
  `t4-out/t4-probe-trace/route-t4-probe-trace.trace`, 9,280 lines of
  `position layer e0 … e7` (`RealForwardRunner.swift:2027-2042`, written only from
  `encodeDecodeRoutedMoE` at `:6387`, before planning, so it records demand and not
  residency): 232 positions × 40 layers in three contiguous runs of 218 / 7 / 7,
  which matches the decode counters **exactly** (tX 62,309 + 7,451 = 8 × 40 × 218;
  turn 2 2,240 = 8 × 40 × 7, because the first generated token comes out of
  prefill's last row, so an N-token decode traces N − 1 positions). Replayed at 128
  slots per layer with hits reserved, cold pools, one continuous pass (the draft's
  probe `t4-draft-replay-probe.py` in the controller's scratchpad: **modelled, and
  superseded by Step 1's real tool**):

  | policy | run 1 (tX, 218 tokens) | run 2 (turn 2, 7) | run 3 (turn 3, 7) |
  | --- | ---: | ---: | ---: |
  | compulsory (first touch, cold pool) | 7,004 | | |
  | `lru` | 8,580 | 376 | 119 |
  | `lfu` = `aging-lfu` | 9,006 | 245 | 112 |
  | aging period 32 / 64 / 128 | 8,631 / 8,746 / 8,878 | 362 / 311 / 287 | 112 / 112 / 99 |
  | **Belady (clairvoyant)** | **7,160** | **112** | **34** |

  Three readings. (1) **The replay tracks the box**: its aging-minus-LRU delta is
  −426 on run 1 against a measured −502, and −131 on run 2 against a measured −128.
  (2) **The one place it fails is the one the prefill lines would fix**: run 3
  replays −7 where the box measured **+82**, and turn 3 is the request with the
  most prefill lookups per layer (85 against turn 2's 64), which under LRU displace
  decode's set where the replay cannot see them. That is the argument for Step 1's
  trace line, as a measurement rather than a hunch. (3) **A faster decay does not
  escape the trade-off, it slides along it**: the aging period sweeps aging-LFU
  toward LRU (period 32 lands at 8,631 / 362 against LRU's 8,580 / 376). But
  **Belady beats both policies on both regimes at once** (7,160 and 112 and 34), so
  the union *is* holdable and the trade-off is an artifact of these two rules, not
  a law. Capacity misses, the part a policy can move: aging 2,002, LRU 1,576,
  Belady **156**, so Belady removes 92 % of aging's and LRU 21 %.

  **What that prices.** Belady's 1,846-miss saving on the answer at 0.93 ms is
  **−1.72 s off a 16.83 s decode (−10.2 %), 13.01 → 14.50 tok/s**, and on a card's
  600-token answer 46.1 → 41.4 s (modelled: the cold-pool replay's delta carried
  onto the warm measured count; Step 1's prefill-inclusive replay replaces it). That is the **ceiling** for any eviction policy
  at 128 slots on this trace and the largest single number in the chapter (T3 took
  0.69 s off a follow-up turn). LRU realizes 23 % of it in the replay and 27 % on
  the box; a segmented policy capturing half the gap would still be ≈ 2.4 s per
  card. If Step 1's replay puts the best implementable policy inside aging-LFU's
  drift on both regimes, the task lands the tooling and records the null.

  **Candidate levers, priced from the rows and the replay.**
  - (a) **A default flip to `lru`.** Measured: +2.8 % tok/s and −502 misses on
    tX, −304 prefill misses on turn 3, against **+184 ms on turn 2's wall and +51
    ms on turn 3's**. On the chapter's rule that is not free today. **But the
    penalty is a burst transient, not a rate**: the replay's LRU penalty is +131
    misses over run 2's 7 tokens, the pool re-converging after the policy's
    recency set was rebuilt by prefill; it is ≈ 131 × 0.93 = **122 ms fixed**,
    while LRU's steady-state advantage on a real answer is 2.8 % (≈ 1.1 s on a 40
    s turn-2 answer). Modelled, LRU **wins by ≈ 1.0 s once the follow-up turn
    decodes a card's answer instead of 8 tokens**. Arm C measures exactly that,
    and it is why the rig needs a long-answer turn 2.
  - (b) **A policy that holds the union**, which Belady says exists. Worth
    replaying, cheapest first: a **protected segment** (a share of the 128 slots
    held for experts with two or more uses, the rest LRU: SLRU / S3-FIFO), a
    **per-plan decay** instead of the 1,024-plan halving, and **ARC-like
    adaptation**. The aging-period sweep says a decay alone will not do it. Replay
    first; the winner lands as a new `ExpertCachePolicy` case behind the knob.
  - (c) **An idle-time refill between turns.** After `settle_done`
    (`ServerInference.swift:1714-1716`, emitted from the post-response `Task`
    `startRewrite` launches at `:1552-1558`: the precedent for work off the
    critical path) the drive idles while the human reads, and turn 2's 240 decode
    misses are ≈ 0.22 s of its 1.42 s wall, the largest removable term on that
    shape. Against it: the pool has no spare slots, so a refill evicts something
    decode would have hit; the idle drive's first reads cost ≈ 2.3× (the ramp
    probe); and a request arriving mid-refill must cancel it as a rewrite is
    cancelled. **Price it only if the replay says the next turn's set is
    predictable from the conversation's history** (Belady still leaves 112 misses
    on run 2, so the headroom above eviction is real but small). It is the one
    candidate that is prediction and inherits `architecture.md`'s bar.
  - **Not levers**: a larger pool (16 GB box; 8G is the measured optimum,
    CLAUDE.md) and next-token expert prediction (v4.3's closed territory).

  **The decision rule, under the chapter's real-and-free rule**
  ([v13-the-turn.md](v13-the-turn.md):488-505). The knob lands either way. **No
  percentage bar.** The pool's policy is global, so **both regimes are verdict rows
  and neither may regress**: (i) `decode_tok_s` on tX's 219-token answer and on the
  long-decode arm, and (ii) the follow-up turn's wall **at a card's answer length**
  (turn 2 with `max_tokens` 512), with the 8-token turn 2 and turn 3 walls beside
  them. The default moves only if the effect is **real** (the sign holds across
  paired runs in both orders, three pairs at the leading candidate, a fourth if a
  row sits inside twice its drift; drift is ±1 % on a wall, ±16 ms on `prefill_s`,
  a 0.26 tok/s spread on tX's decode) and **free**: the 300 / 1k / 2k warm pairs
  unmoved (their 58.4 / 51.6 / 50.3 % prefill hit rates depend on what the previous
  request left resident, which Task 0's carry parity created, so a move there is a
  cost, not a defect), the 12k control ± 1 %, `memory_pressure -Q` acceptable before
  every launch, and **golden IDENTICAL on both boxes and both profiles**.

  **Numerics: nothing moves.** Slot policy changes which expert is fetched when,
  never what any kernel computes ([v13-the-turn.md](v13-the-turn.md):507-513), and
  step zero measured it: all four arms' completions byte-identical. Golden
  identical is the bar and a difference is a defect, never a recapture. The trace
  line is diagnostic and off unless the env names a file.

  **The knob.** The existing `SHRIKE_EXPERT_CACHE_POLICY` gains the new case, and
  its allowed list and error text (`PreadExpertStreamer.swift:344-346`) gain the
  value. Two shape fixes ride with it, both in family with T2's work: hoist the
  parse out of `init` into a static `ExpertCachePolicy.environmentValue(_:)` in the
  shape of `ExpertIOBackend.environmentValue` (`:170-181`), called once when the
  session is built so a bad value fails the launch rather than the first routed
  layer of the first request; and print the **effective** policy on the residency
  line, which does not carry it today (`prefillGapLeversDescription`,
  `RealForwardRunner.swift:293-330`, prints `sweep=`, `cache_layout=`, `expert_io=`
  and, since T3, `prefill_matrix_min_rows=`), reading that same static parse rather
  than `RuntimeConfiguration.expertCachePolicy`, which the env override never
  reaches (`ServerInference.swift:702`). `cachePolicyDefault` (`:260`) moves on the
  verdict. If (c) wins instead, the refill takes its own knob
  (`SHRIKE_EXPERT_IDLE_REFILL`) and the policy knob is untouched.

  **The trace's prefill line.** One line per **tile**, not per row: the pool sees
  one plan per tile, and a per-row line would need a row→expert map the pool never
  has. Grammar: a marked variant `p <chunkFirstPosition> <layer> <tileIndex> <e0 … e7>`,
  so the bare `position layer e…` decode line and its consumers keep working and a
  replay splits on the first field. Emitted from the three sites that already hold
  the tile's expert IDs (`RealForwardRunner.swift:5371`, `:5589`, `:5675`, each a
  `PrefillStreamedTileBinding.expertIDs(forTile:routes:)`) through the same
  `routeTraceFD` (`:1887-1891`, opened `O_TRUNC` once per process, so a capture is
  exactly one server's history). The formatting becomes a pure static function with
  a host test; the write path is unchanged.

  **The replay tool, `tools/expert-pool-replay.py`.** Inputs: a trace, slots per
  layer, a policy, optionally a layer filter. Outputs: hits / misses per phase per
  request (per layer on request) and the capacity / compulsory split. Policies:
  `lru`, `lfu`, `aging-lfu` at a settable period, `belady`, and each candidate from
  (b). **Fidelity list**, each verified above and each a line of the tool: hits
  reserved before victim selection (`:686`), loading and pinned slots ineligible
  (`:1156-1157`), ties by count then `slotLastUse` then lower slot index,
  `expertUseCount` incremented per expert per plan and never reset, the halving at
  multiples of 1,024 per layer, prefill's `avoidingSlots` (the held slots of the
  open and pending batches), and one plan per tile at fetch depth 2. **Validation,
  the step's own stop:** replaying the same trace under `aging-lfu` must reproduce
  the measured **7,451 / 240 / 97** decode misses and, once prefill is traced,
  **832 / 721** prefill misses, within Task 3's ± 3 counter jitter
  ([v13-the-turn.md](v13-the-turn.md):410-414). The draft's probe reaches 9,006 /
  245 / 112 from a cold pool with no prefill lines: the run-1 gap is prefill's
  residency covering 1,555 first touches and the run-3 gap is the missing prefill
  plans, so both should close, and if they do not the model is wrong. A self-test
  on a tiny synthetic trace ships with it.

  **The rig and the rows.** `turns-live` already gives the chain; the long-answer
  turn 2 needs **one token**. `tools/turn-rig.sh:179` becomes
  `send "turn2" "${TURN2_MAX_TOKENS:-}" "$turn2_payload"`, reusing the `send`
  override that rewrites `max_tokens` into a temp copy (`:75-82`) instead of
  touching the stored payload, so `REUSE` still sends byte-identical bodies in both
  arms; the header comment (`:19-27`, `:35-36`) gains it. `USER_TURN2` cannot do
  this: it selects which user turn is appended, not the budget (`:47-50`, `:177`).
  The long-answer arm builds its own REUSE directory on arm A and reuses it on arm
  B as T3 did, and its turn 3 is reported but is not a verdict row (that payload
  carries the 8-token assistant turn while the server just produced a long one, so
  its cached fraction differs). **Rows** exactly as `t4-rows.py` reads them (Task
  3's fields) **plus the pool counters folded into the row**: `expert_evictions`,
  `expert_reloads`, `expert_hit_rate_decode`, `expert_hits_decode`,
  `expert_misses_decode`, which `t4-probe.sh` greps separately today and which all
  already print (`ServerInference.swift:1991`, `:2002-2003`). **No new counter.**
  Caveat to carry: `io_fetch_ms × 8` is the request's total only when the
  completion is 8 tokens; on tX read it as `io_fetch_ms × 219`.

  **Tests (RED first, host-only; no device suite for a scheduling-only change).**
  - `expertVictimOrderFollowsThePolicy`: the comparator extracted from
    `shouldEvictSlot` (`:1179-1192`) as a pure function over (policy, lhs count,
    lhs last use, rhs count, rhs last use). LRU by last use alone, the LFU family
    by count then last use, the new policy's own rule, the negative-expert slot
    case (`:1183-1185`). The extraction is the point: none of it needs a device.
  - `expertCachePolicyEnvironmentDefaultsAndFailsClosed`: the hoisted parse in the
    shape of `expertIOBackendEnvironmentDefaultsAndFailsClosed`
    (`PreadExpertStreamerTests+CachePlanning.swift:9-19`). Unset gives the default,
    each allowed value round-trips, an unknown value throws with the allowed set.
  - `prefillGapLeversDescriptionReportsTheCachePolicy`: the existing assertion
    (`PreadExpertStreamerTests+CachePlanning.swift:60-67`) plus the new field.
  - `routeTraceLineFormatsPrefillAndDecode`: the pure formatter, beside T3's parser
    tests (`PrefillRoutedTileSchedulerTests.swift:497-513`). A decode line stays
    bare `position layer e…`, a prefill line is `p position layer tile e…`, both
    round-tripping through the replay's parser. Plus the replay's own
    `--self-test`, run from the tool and not from `swift test`.

  **Files.** `sources/Shrike/Infrastructure/Streaming/PreadExpertStreamer.swift`:
  the new `ExpertCachePolicy` case and its static `environmentValue` (`:164-181`),
  the `init` parse delegating to it (`:342-349`), `cachePolicyDefault` (`:260`) on
  the verdict, `shouldEvictSlot` delegating to the pure comparator (`:1179-1192`),
  and the new policy's per-slot state if (b) wins.
  `sources/Shrike/Runtime/Inference/RealForwardRunner.swift`: `recordRouteTrace`
  and its formatter (`:2027-2042`), the three prefill tile sites (`:5371`, `:5589`,
  `:5675`), `prefillGapLeversDescription` (`:293-330`). `ServerInference.swift`:
  nothing, the field rides the gap-levers string already on the residency line
  (`:822-826`). `tools/expert-pool-replay.py` (new), `tools/turn-rig.sh:179` and
  its header. Tests: the two files above. **Lint:** two baseline entries embed a
  line count in their reason string and go stale on any edit to their body,
  `PreadExpertStreamer.init` ("currently spans 162 lines") and
  `encodeRoutedMoEPrefill` ("currently spans 244 lines"), so
  `swiftlint lint --write-baseline` is expected (T0 to T3 each hit this on a
  different function). **Unchanged:** every `.metal` file and kernel body,
  `expert_io.c` and the reader, the planner's placement logic, the prompt cache.

  Steps:

  Steps (as run; the drafted Step 2 was a policy case and was superseded by Step 1's
  verdict):

  - [x] Step 1 (an implementer, three fix-up rounds): the trace's prefill line (per tile,
        row counts and last rows added when the verdict needed them) and the request-start
        line; `tools/expert-pool-replay.py` with the fidelity list, `--self-test`, `--expect`,
        `--avoid-lookback` (3, the production bound), `--prefill-weight`, `--sweep-order`
        (index / rows / last, with the carry alternation), `slru` / `arc` / `lru-2`,
        `--phase-policy`, `--profile`; three captures of two shapes on the mini (each
        recapture after the trace grew a field); validation ±1 on every request; the offline
        verdict over every candidate on both traces (the tables in the design doc). The
        named stop was reached for every eviction rule and passed by the recency-ordered
        sweep.
  - [x] Step 2 (an implementer, three fix-up rounds): `PrefillSweepMode.recency` with the
        balanced tiles and `SHRIKE_PREFILL_SWEEP_TAIL`; `ExpertCacheProtectMode` with
        `SHRIKE_EXPERT_CACHE_PROTECT`, the chunk's remaining experts carried from the tile
        planner to `selectVictimSlots`; the rig's `TURN2_MAX_TOKENS`; the host costs of both
        removed after round 2 measured them; the default flip and the replay's `--protect`
        in the final amend. Gates 1–4 green on every amend, the landed tree's run 1239 / 1239.
  - [x] Step 3 (numerics): golden IDENTICAL on both profiles at the four knob cells
        (carry / recency × protect off / chunk) on the M4 Pro at every amend, and on the
        mini at the default and the candidate cells at every amend and at all four cells
        at the landed commit.
  - [x] Step 4 (the arms, three rounds on the mini, one binary each): round 1 (the plain
        recency order) found the tile imbalance; round 2 (balanced + protect, with the
        attribution cells C = protect alone and D = order alone) found the host cost and
        attributed every gain and cost to its lever, and its traced chain replayed exactly;
        round 3 paired A against C (the tables above and in the design doc).
  - [x] Step 5 (the rule): protection alone is real (paired in both orders, above drift)
        and free (no control row regresses, golden identical) → `chunk` is the default; the
        order is real on the first turn's decode and not free on consecutive chunks → a knob.
  - [x] Step 6 (design doc): the Task 4 section, the After T4 block, the lever entries and
        the Task 3 close-out sentence corrected; Follow-ons gained the resident-first sweep,
        the settle re-prefill after a no-prefix request, the policy parse in the streamer's
        `init`, and the protection fallback counter. Task review by a fresh reviewer, fixes
        folded into the owning commit.

  **Risks and what falsifies the model.**
  - **The replay's initial state.** A traced request starts on whatever the
    previous one left, so a capture must begin at a fresh server and cover every
    plan from launch. Prefill's plans are what is missing today, and the draft's
    probe shows the cost: run 3 replays −7 where the box measured +82. If Step 1's
    replay still cannot reproduce 7,451 / 240 / 97 within ± 3 once the prefill line
    lands, something else mutates the pool that this task has not found, and Step 1
    stops there.
  - **The trace's own cost.** `recordRouteTrace` is a synchronous `write(2)` per
    layer per token (40 per decode token) and the prefill line adds one per tile
    (338 on a 21-row turn, 18,864 on 12k). The decode cost is measured nil on the
    trace arm (tX 13.099 tok/s against 13.014, turn 2 1.456 s against 1.422, both
    inside drift); the prefill line's is **modelled**, ≈ 40-95 ms on a 68 s 12k
    request at a few µs per write. It is off unless the env names a file and no
    verdict arm runs with it on.
  - **A policy that trades the regimes** is the expected failure, and the A/B
    already shows its shape: LRU buys the answer and sells the turns. Both regimes
    are verdict rows, so a policy inside aging-LFU's drift on one and above it on
    the other records the trade and does not flip. **And the turn-2 penalty may not
    amortise**: reading LRU's +184 ms as a fixed 122 ms burst transient rests on
    the replay's per-run attribution, not on a measured long-answer turn 2. Arm C
    is that measurement; if the penalty scales with the turn's token count, (a) is
    dead and this draft is wrong.
  - **The first-turn and 12k controls' hits move.** The 300 / 1k / 2k warm rows hit
    58.4 / 51.6 / 50.3 % because Task 0's carry parity leaves the previous
    request's experts resident, and which of them survives is exactly what this
    task changes. A move there is a **cost** priced in the verdict, not a defect
    (the opposite of T3's rule). The 12k control at 33.6 % is the large-shape check.
  - **A policy fitted to one trace.** Every eviction rule is a bet on the access
    pattern, and one chain is one sample of it. The offline verdict runs on two
    captures of different shape (Step 1) and the winner must lead on both; a
    policy that wins the 2k card and loses the short-prompt pair is recorded as
    a trade, not flipped.
  - **LFU's counts alias across the sweep.** Every routed expert leaves tX's
    2,125-token prefill with count ≈ 1, so the tie-break (oldest `slotLastUse`)
    decides most early evictions and a policy that changes how prefill increments
    counts moves far more than its own regime. Every candidate in (b) is replayed
    on the **prefill-inclusive** trace before it is built.
  - **Memory and the box.** A segmented policy's extra per-slot state is a few
    bytes × 256 experts × 40 layers; the pool's 9.06 GB does not move and
    `--ram-budget 8G` stays. `memory_pressure -Q` before every launch. The mini is
    production: every arm stops the server on 8081 and relaunches it, Turbo on 8080
    is never touched, one model process at a time.

### Task 5: T5 — the resident-first recency sweep

- [ ] **T5: Task 4 left one lever measured and unlanded. `SHRIKE_PREFILL_SWEEP=recency`
  takes 0.6 to 0.8 s off the first turn's decode after a large prompt on the mini
  (−0.63 s on the 2k card, −0.72 s on the 300-token prompt, −0.72 s on the 1k,
  +3.2 to +3.9 % tok/s) and forfeits T0's carry benefit on every chunk that follows
  another (the 300 / 1k / 2k pairs' warm prefill +3.7 / +7.5 / +10.9 %, 12k +7.2 %),
  so it stayed a knob at default `carry`
  ([v13-the-turn.md](v13-the-turn.md):531-537). The forfeit has one cause: an
  ascending-by-last-row sweep visits the chunk's experts in an order unrelated to what
  the pool already holds, so its early misses evict residents the same chunk needs
  later. **Sweeping the chunk's resident experts first fixes exactly that**: every hit
  is harvested before any eviction (T0's parity trick, which alternates direction to
  approximate the same thing, made exact through the pool's own residency), and the
  absent experts keep the recency order that leaves the prompt's last-used experts in
  the pool for decode. Step zero replayed it on four captures at 128 slots with the
  streamer's exact rules: **the first turn's decode gain is kept in full on every trace**
  (a cold pool has no residents, so a first chunk's order is identical to today's
  `recency`: −762 misses on the card's answer, −591 on the 300 prompt's 512-token
  answer, −293 on the 8-token pair, −26 at 12k, a control), **and every row the plain recency
  order lost comes back at or under today's** (the warm prefill after a long answer
  −2.1 %, the chain's turn 3 −3.7 %, 12k −0.8 %, and the second 12k chunk's hits become
  exactly 5,120 = 128 slots × 40 layers, the arithmetic maximum). One row costs: a
  warm second prompt after a short answer, +30 prefill misses (+1.0 %) against that
  same request's −12 decode misses, a wash of ≈ 11 ms either way. **The prize is the
  first turn's decode after a large prompt, ≈ −0.6 s, without the multi-chunk and
  warm-prefill losses that kept `recency` a knob.** The task lands a fifth
  `PrefillSweepMode` behind `SHRIKE_PREFILL_SWEEP=resident`, default `carry` in the
  commit, flipped by amend on the mini's verdict, either way
  ([v13-the-turn.md](v13-the-turn.md):640-658).**

  **Step zero: two more captures, zero Swift code** (the controller, 2026-09-06; mini,
  the deployed 04d4de5 binary, bare launch so `sweep=carry protect=chunk` and
  `aging-lfu`, with `SHRIKE_ROUTE_TRACE` naming a file; driver
  `loop-files/t5-capture.sh`, archived at
  `~/.claude/handoffs/archive/shrike-v13-t0/t5-out/`; rows by `t4-rows.py` off the
  server's own counters):

  | capture | request | new rows | wall | `prefill_s` | prefill hits / misses | routed GPU / tiles | decode misses (8 tokens) |
  | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | `t5-cap-pair300` | t300 (cold) | 289 | 7.899 s | 5.461 | 0 / 7,639 (0 %) | 1,522 ms / 968 | 506 |
  | `t5-cap-pair300` | t300b (warm) | 305 | 3.443 | 3.017 | 4,861 / 2,865 (62.9 %) | 1,400 / 982 | 123 |
  | `t5-cap-12k` | the 12k prompt | 12,285 | 68.272 | 64.95 | 9,947 / 18,457 (35.0 %) | 19,797 / 3,605 | 729 |

  Both validate against the replay to the unit (delta 0 on every counter, the trace's
  own `p` lines equal to the logged tile counts). The 12k prompt is three chunks at
  positions 0 / 4,096 / 8,192. Both captures re-confirm Task 4's no-prefix finding: the
  prompt cache's settle after a request with no cached prefix re-prefills the whole
  prompt in the background (954 tiles after t300; 3,595 tiles and ≈ 64 s after 12k), a
  cache-chapter follow-on and not this task's. The other two traces are Task 4's round-3
  captures `t4-out/t4-cap-aging-lfu-chain-r3/` (the card: a 2,125-row first turn answered
  live at 219 tokens, then the 21- and 36-token follow-ups) and
  `t4-out/t4-cap-aging-lfu-d512-300-r3/` (a 289-row prompt answered at 512 tokens, then a
  warm 305-row prompt), captured with protection off and replayed here at `--protect
  chunk`, which reproduces round 3's measured rows (705 / 614 on the follow-ups, 3,389 on
  t300b against the box's 705 / 613 / 3,389).

  **The offline verdict** (`tools/expert-pool-replay.py` at `--policy aging-lfu --slots
  128 --protect chunk`; prefill / decode misses per request, the settle chunks summed;
  every cell in this table re-run by this draft):

  | trace | row | index (today's `carry` tiles as recorded) | `last-asc` (the plain recency order) | resident-first, plain tiles | **the design: resident-first, balanced groups, flat tiles** | Belady |
  | --- | --- | ---: | ---: | ---: | ---: | ---: |
  | chain r3 | tX (2,125 rows, 219-token answer) | 9,370 / 7,451 | 9,370 / 6,697 | 9,370 / 6,697 | 9,370 / **6,689** | 9,370 / 2,485 |
  | chain r3 | turn 2 (21 new, 8 tokens) | 705 / 238 | 707 / 242 | 707 / 234 | 707 / **230** | 380 / 95 |
  | chain r3 | turn 3 (36 new, 8 tokens) | 614 / 98 | 602 / 104 | 593 / 92 | **591** / 96 | 176 / 31 |
  | chain r3 | settle (background) | 897 | 852 | 708 | **689** | 96 |
  | d512-300 | t300 (289 rows, 512-token answer) | 7,639 / 10,059 | 7,639 / 9,450 | 7,639 / 9,450 | 7,639 / **9,468** | 7,639 / 3,773 |
  | d512-300 | t300b (305 warm, 8 tokens) | 3,389 / 157 | 3,596 / 125 | 3,317 / 171 | **3,317** / 205 | 3,087 / 45 |
  | pair300 | t300 (289 rows, 8 tokens) | 7,639 / 506 | 7,639 / 231 | 7,639 / 231 | 7,639 / **213** | 7,639 / 111 |
  | pair300 | t300b (305 warm, 8 tokens) | 2,865 / 123 | 3,759 / 125 | 2,906 / 108 | **2,895** / 111 | 2,717 / 96 |
  | 12k | 12,285 rows, 8 tokens | 18,457 / 729 | 20,437 / 312 | 18,271 / 427 | **18,315** / 703 | 18,294 / 56 |
  | 12k | prefill per chunk (0 / 4,096 / 8,192) | 9,426 / 4,383 / 4,648 | 9,426 / 5,395 / 5,616 | 9,426 / **4,202** / 4,643 | | |

  **Three readings.**

  1. **The prize is the plain recency order's, kept whole.** On a first request the pool
     is cold, the resident set is empty, and the order degrades to `recencyBalanced`
     exactly (`PrefillMoEGrouping.swift:116`), so the decode gain is the one already
     measured on the box: −762 misses on the card's answer against the −754 that
     measured −0.63 s, −591 on the 300 prompt's 512-token answer against a measured
     −0.72 s ([v13-the-turn.md](v13-the-turn.md):531-533). At Task 4's 0.93 ms per decode
     miss ([v13-implementation-plan.md](v13-implementation-plan.md):1727-1731) that is
     −0.71 s and −0.55 s modelled, bracketing the measured pair. **This gain is a fixed
     transient, not a rate**, which the replay's `--profile` shows directly: on the card's
     answer the per-32-token decode misses run 1,830 / 1,024 / 529 / 1,162 / 958 / 1,041 /
     907 at index against 1,227 / 896 / 528 / 1,133 / 964 / 1,041 / 908 resident-first, so
     −603 of the −754 is in the first 32 tokens and −731 in the first 64; on the 512-token
     answer the per-64-token windows are 3,011 / 1,888 / 1,674 / 1,933 / 1,553 against
     2,410 / 1,873 / 1,681 / 1,933 / 1,553, so −601 of −609 is in the first 64. The
     boundary between a prompt's sweep and its answer's first tokens is the whole term.
     A card's 600 to 800-token answer therefore gains the same ≈ 0.6 s absolute (1.2 to
     1.5 % of a 46 to 61 s answer), and the 219-token answer 3.7 % of 16.8 s.
  2. **Every row the plain order lost comes back.** The warm prefill after a long answer
     3,389 → 3,317 (−2.1 %), the chain's turn 3 614 → 591 (−3.7 %), 12k 18,457 → 18,315
     (−0.8 %), and the second 12k chunk's hits 4,939 → **5,120**, which is 128 slots × 40
     layers: every slot a hit, the arithmetic maximum for that chunk and the exact form of
     what T0's alternation approximates. The one exception is pair300's warm second
     prompt, +30 prefill misses (+1.0 %; the plain order costs +41, and step zero's
     `t5-layers.py` found that scatter to be 11 layers better, 6 the same, 23 worse, none
     beyond +7, against `last-asc` losing +9 to +28 on all 40). At the short shapes'
     measured ≈ 0.4 ms per exposed prefill miss (Task 4's verdict rows: 127 misses for 45
     ms on the 21-token turn, 625 for 264 ms on the warm 300 after a long answer, 353 for
     128 ms on the warm 300 pair, [v13-the-turn.md](v13-the-turn.md):550-556) that is
     +12 ms of prefill against that same request's −12 decode misses, ≈ −11 ms: a wash
     inside the drift of a 3.44 s wall.
  3. **12k's 8-token decode is a control, not a verdict row.** Step zero's `t5-debug.py`,
     re-run here: on the last 12k chunk 219 to 255 of the 256 experts are needed per
     layer and 118 to 128 of the 128 slots hold a hit resident, so almost nothing survives
     the chunk except the last fresh loads, and which ones is the victim comparator's slot
     tie inside one tile. The row moves 427 to 708 across compositions that differ nowhere
     else. Read 12k on its prefill and its wall.

  **Tile composition: three balanced groups, tiled flat.** Task 4's fix round 1 measured
  that an unbalanced order costs GPU (+0.47 s on a 2k prefill, gone once the head and tail
  were packed by row weight, [v13-the-turn.md](v13-the-turn.md):511-518), and the plain
  resident-first order is as unbalanced as `last-asc` (per-tile row-weight stdev 373
  against index's 317 on the chain, 844 against 467 at 12k). Packing each of the three
  groups (resident, absent head, absent tail of the `tail` most recent absent) by row
  weight as `recencyBalanced` packs restores the balance, but giving each group its own
  tiles costs partial tiles at two group boundaries: the chain 2,767 → 2,859 (+3.3 %), the
  warm 300 chunk 982 → 1,000, the 21-token turn 338 → 358, turn 3 443 → 462. At the
  measured ≈ 1.2 ms per routed tile on a 21-row chunk
  ([v13-implementation-plan.md](v13-implementation-plan.md):1722) that is +24 ms on the
  follow-up turn, which eats its own prefill gain. **Concatenating the three balanced
  group orders and tiling the concatenation flat in runs of 8 keeps index's tile count
  exactly** (chain 2,767, 12k 7,200, the 300 traces 2,997 and 3,871, verified per chunk)
  with the balance intact and the misses unchanged to a few counts:

  | variant | chain: tX decode / turn 2 / turn 3 | chain tiles, spread | d512-300: t300 decode / t300b | tiles, spread | pair300: t300 decode / t300b | tiles, spread | 12k: prefill / decode | tiles, spread |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | index (today) | 7,451 / 705·238 / 614·98 | 2,767, 317 | 10,059 / 3,389·157 | 2,997, 51 | 506 / 2,865·123 | 3,871, 49 | 18,457 / 729 | 7,200, 467 |
  | rf-plain (tiles of 8 across the concatenation) | 6,697 / 707·234 / 593·92 | 2,767, 373 | 9,450 / 3,317·171 | 2,997, 77 | 231 / 2,906·108 | 3,871, 76 | 18,271 / 427 | 7,200, 844 |
  | rf-bal:96 (three groups, each its own tiles) | 6,688 / 707·229 / 581·98 | 2,859, 283 | 9,467 / 3,317·203 | 3,043, 40 | 213 / 2,893·108 | 3,919, 32 | 18,315 / 708 | 7,232, 475 |
  | **rf-bal-flat:96 (three balanced groups, tiled flat)** | **6,689 / 707·230 / 591·96** | **2,767, 290** | **9,468 / 3,317·205** | **2,997, 47** | **213 / 2,895·111** | **3,871, 42** | **18,315 / 703** | **7,200, 484** |

  The adaptive tail (`tail` clamped to 128 − |resident|) and a strict last-asc tail move
  nothing outside 12k's tie-break decode row (695 and 662 against 708) and cost more
  partial tiles; both are recorded and not built. A tile that straddles a group boundary
  is safe: a plan reserves its hit slots before `selectVictimSlots` runs
  (`PreadExpertStreamer.swift:704-711` before `:716-722`), so a resident sharing a tile
  with misses cannot be their victim, and every later tile of the chunk is protected by
  `SHRIKE_EXPERT_CACHE_PROTECT=chunk` (`:1185-1195`).

  **The mechanism, verified in the tree.**
  - **The mode.** `PrefillSweepMode` (`RealForwardRunner.swift:143`, four cases today),
    its parser `parsePrefillSweepMode` (`:546`, unknown values fall back to the default
    rather than throwing) and env read (`:553`), the tail parser `parsePrefillSweepTail`
    (`:557`, fails closed on anything outside 8...expertCount, since fix round 1), the
    printed residency field (`prefillGapLeversDescription:357`, the `sweep=recency
    tail=96` branch at `:376-378`), `prefillChunkUsesRecencyBalance` (`:693`: `.recency`
    only, and only when the chunk participates in carry, so the verify / MTP sidecar's
    chunks keep index tiling), and the carry write left alone under `recency`
    (`:2593`).
  - **The order.** `buildPrefillRoutes` (`:5173-5242`, one call site, `:5492`, once per
    layer per chunk) builds `rowsByExpert` / `lastRowByExpert` from the chunk's pairs
    (`:5212-5220`), calls
    `PrefillSweepOrder.recencyBalanced(rowsByExpert:lastRowByExpert:tail:tileWidth:)`
    (`PrefillMoEGrouping.swift:116`; `recency` at `:103`, `packByRows` at `:143`,
    heaviest expert first into the lightest open tile), then
    `expertSortKeys(forOrder:numExperts:)` (`:135`, one array read per comparison side;
    round 2's lesson was that a dictionary-keyed pair sort cost +0.44 s of host on a 2k
    chunk) and hands both to `groupTokenExpertPairs`
    (`:177`), whose `expertTileCounts` branch (`:262-273`) slices the order into the
    supplied group tiles. **Passing `expertTileCounts: nil` instead makes it slice flat
    in runs of `tileExpertCount` (`:274-288`), which is index's own tiling**: the flat
    design needs no tile-count vector at all, and the order function returns a plain
    `[UInt32]`.
  - **The residency the order needs is not the runner's `poolResidency`**, which is a
    Metal `MTLResidencySet` of GPU buffers (`ExpertPoolResidency.swift`). The expert
    residency lives per layer in `PreadExpertStreamer` (`slotExpert` / `slotState` under
    `cacheLock`, `:318-348`) and is already exposed by `residentExperts()` (`:1293`:
    `.resident` slots only, sorted, under the lock; its doc comment says why `.loading`
    slots are omitted), forwarded per layer by `ModelExpertIO.routedExpertResidentIDs`
    (`:178-182`). Today that forwarder has only diagnostic callers, both off by default
    (`RealForwardRunner.swift:6632`, gated on `SHRIKE_PREFETCH_TRACE` at `:1986-1987`,
    and `:6862` under predictive prefetch), and its doc comment says so; this task gives
    it a hot-path caller and the comment is corrected with it.
  - **The snapshot is exact under the shipped loop.** It is taken once per layer per
    chunk at the top of `buildPrefillRoutes`, before any tile of that layer is planned.
    At that moment layer L has no `.loading` slots: the lookahead loop commits its open
    batch and drains every pending batch before returning (`:5885-5887`), and each tile's
    fetch is awaited inside the loop (`:5849`), so the previous chunk's loads for layer L
    are published (`PreadExpertStreamer.swift:1366`) before the next chunk reaches that
    layer. If a `.loading` slot ever appeared it would count as absent and land in the
    absent group by recency, where the plan either finds it resident by then (a hit) or
    re-fetches it: `makeExpertCachePlan` reserves loading slots and never counts them as
    hits (`:696-699`), so the order can only be pessimistic, never wrong.
  - **The protection's interplay.** Under resident-first every resident of the chunk is
    swept before any eviction happens, so `protectedExperts` has nothing resident left to
    protect on the head and does its work only across the absent group's later tiles. The
    starvation fallback (the plan retried with `nil`, `:718-722`) therefore fires on
    strictly fewer plans than today, and the Follow-ons entry asking for a counter on it
    matters less after this task, not more.
  - **The settle's chunks take the same order** (`ServerPromptCache`'s rewrite calls
    `prefillChunked`, `RealForwardRunner.swift:2273`, which is the same chunk path;
    `recordRouteTraceRequestStart` is deliberately not called from it, `:2125`). The
    replay assumes so and its settle rows move with the order (the chain 897 → 689).
  - **Prior art placed.** T0's carry is the same idea with one bit of state instead of
    the pool's; T4's `recency` order and its balanced tiles are the head this task
    reorders; the route trace and `tools/expert-pool-replay.py` are the instrument
    ([v13-the-turn.md](v13-the-turn.md):474-487). The mini's current rows are the After
    T4 table ([v13-the-turn.md](v13-the-turn.md):578-589): the 21-token follow-up
    1.385 s, the warm 300 / 1k / 2k first turns 3.454 / 6.385 / 10.287 s, 12k 68.209 s.

  **The knob.** A **fifth `PrefillSweepMode` case, `resident`**, behind
  `SHRIKE_PREFILL_SWEEP=resident`, beside `recency` rather than replacing its body, so
  one binary runs `carry` (A), `resident` (B) and the already-measured `recency` (C) and
  the arms can show that B reproduces C's decode prize while C's losses are gone.
  `parsePrefillSweepMode` (`:546`) gains the value through the enum's `RawValue`, so its
  fallback behaviour is unchanged and the four existing cases keep their strings. The
  tail knob is shared unchanged (`SHRIKE_PREFILL_SWEEP_TAIL`, default 96, `:531`); the
  printed field reads `sweep=resident tail=96`. `prefillChunkUsesRecencyBalance` (`:693`)
  is renamed `prefillChunkUsesComputedSweepOrder` over a new
  `PrefillSweepMode.usesComputedOrder` (`.recency` or `.resident`), and the carry write
  (`:2593`) tests the same property so a computed order never poisons the carry state.
  Default `carry` in the landed commit; the flip by amend on the verdict, Task 4's
  pattern.

  **The order as a pure function.**
  `PrefillSweepOrder.residentFirstBalanced(rowsByExpert:lastRowByExpert:resident:tail:tileWidth:)
  -> [UInt32]` beside `recencyBalanced` (`PrefillMoEGrouping.swift:116`): rank the
  chunk's experts by last row ascending with ties by expert id (`recency`, `:103`);
  partition by `resident` (a `[Bool]` of `numExperts` entries, `true` where the pool holds
  that expert); take the last `min(tail, absent.count)` of the absent group as the tail;
  return `packByRows(resident) + packByRows(head) + packByRows(tail)` concatenated, each
  group's own `packByRows` binning into `ceil(group.count / tileWidth)` tiles as today
  (`:143`). The caller passes the result through `expertSortKeys` (`:135`) and leaves
  `expertTileCounts` nil, so `groupTokenExpertPairs` tiles the concatenation flat. **With
  an empty resident set the function returns `recencyBalanced(...).order` exactly**,
  which is the cold-pool case (a fresh launch's first chunk, and every first request's
  first chunk) and the reason the first-turn prize is unchanged. `packByRows` is
  O(group × tiles-in-group) and the three groups partition the chunk, so splitting into
  three costs no more than today's two.

  **The residency snapshot's cost.** One `routedExpertResidentIDs` per layer per chunk:
  one `NSLock`, a 128-slot scan, a sort of at most 128 `Int`s, then a fill of a
  runner-owned `[Bool]` scratch cleared in place in the shape of `routeIDScratch`
  (`RealForwardRunner.swift:5183-5188`). Budget: ≈ 10 µs per layer, ≈ 0.4 ms per chunk
  across 40 layers, against a 21-row turn's 844 ms and a 2k chunk's 11.1 s. The chunk
  already does far more per layer (the route copy and pair build at `:5189-5196` walk
  `t × topK` pairs). `prefillRouteNanos` (`:2583`, accumulated at `:5498`) is the
  pre-registered check under `SHRIKE_PHASES`: if it moves by more than a few ms per
  chunk, the fix is an allocation-free `residentExpertMask(into:)` on the streamer
  filling the `[Bool]` directly under the lock, and this draft is wrong about the cost.

  **The modelled gain per regime**, replayed misses priced at 0.93 ms per decode miss and
  ≈ 0.4 ms per exposed prefill miss on the short warm shapes (≈ 0.1 ms at 2k, ≈ 0 at 12k,
  where the routed stage is GPU-bound):

  | row | replayed miss delta | modelled Δ | measured anchor |
  | --- | --- | ---: | --- |
  | tX decode, 2,125-row card, 219-token answer | decode −762 | **−0.71 s** (16.79 → 16.08 s, 13.0 → 13.6 tok/s) | `recency` measured −0.63 s for −754 |
  | the 300 prompt's 512-token answer (d512-300) | decode −591 | **−0.55 s** | `recency` measured −0.72 s |
  | the long-answer follow-up (turn 2 at 512 tokens) | decode −8 over the 7 traced positions | −10 to −60 ms, modelled from the transient's shape | to be measured; the row exists to show it does not sell |
  | the 21-token follow-up turn | prefill +2, decode −8 | ≈ 0, tile count identical | 1.385 s, unmoved |
  | turn 3, 36 new | prefill −23, decode −2 | −10 ms | 1.458 s |
  | warm 300 after a long answer (d512-300 t300b) | prefill −72, decode +48 | −29 + 45 = **+16 ms** (+0.5 % of 3.175 s) | the row most likely to move the wrong way |
  | warm 300 pair (pair300 t300b) | prefill +30, decode −12 | +12 − 11 ≈ **0** | 3.454 s |
  | 12k, first request after launch | prefill −142, decode −26 | ≈ 0 | 68.1 to 68.3 s across launches |
  | the settle's background re-prefill | chain −208, 12k −80 | off the critical path | ≈ 0.4 GB less read |

  **The decision rule, under the chapter's real-and-free rule**
  ([v13-the-turn.md](v13-the-turn.md):640-658). The knob lands either way. **No
  percentage bar.** **Verdict rows:** (i) `decode_tok_s` and `decode_s` on tX's answer in
  the live chain, and the same on the `d512-300` / `d512-1k` arms' 512-token answers, and
  (ii) the long-answer follow-up turn's wall (turn 2 at `TURN2_MAX_TOKENS=512`). The
  default moves only if the effect is **real** (the sign holds across paired runs in both
  orders, three pairs, a fourth where a row sits inside twice its drift; drift is ±1 % on
  a wall, ±16 ms on `prefill_s`, a 0.26 tok/s spread on tX's decode) and **free**: the
  300 / 1k / 2k warm pairs' prefill back within drift of `carry` (`recency`'s +3.7 /
  +7.5 / +10.9 % must be **gone**, which is the whole point of the task), the
  `d512-300` warm request within drift (+16 ms modelled), the 21-token follow-up and
  turn 3 unmoved or better, the 12k control within its launch-to-launch drift, the cold
  first request unmoved (Task 4's unexplained +0.27 s observation on one arm of two is
  read again here, not explained here), `memory_pressure -Q` acceptable before every
  launch, and **golden IDENTICAL on both boxes and both profiles at `carry`, `resident`
  and `recency`**.

  **Numerics: nothing moves.** Sweep order changes which experts share a tile and when
  they are fetched, never what any kernel computes
  ([v13-the-turn.md](v13-the-turn.md):659-666); Task 4 measured golden identical at both
  `recency` cells on both boxes. Golden identical is the bar and a difference is a defect,
  never a recapture. The route trace stays off unless the env names a file and no verdict
  arm runs with it on.

  **Tests (RED first, host-only; no device suite for a scheduling-only change).**
  - `PrefillMoEGroupingTests.swift`, in the style of the existing `recencyBalanced` cases
    (`:438`, `:449`, `:460`) and `groupingReproducesRecencyBalancedTilesEndToEnd`
    (`:538`): the resident group leads regardless of its experts' last rows; within each
    group the order is `recency`'s and the packing is `packByRows`'s; an expert resident
    but not routed by this chunk is absent from the order entirely; an empty resident set
    returns `recencyBalanced(...).order` exactly; the tail is taken from the absent group
    only; a `resident` array shorter or longer than `numExperts` is handled without a
    trap (the shape `expertSortKeysSkipsAnOutOfRangeExpertRatherThanTrapping` `:431`
    already sets); and one end-to-end case through `groupTokenExpertPairs` with
    `expertTileCounts` nil asserting the flat tiles are `ceil(n / 8)` with contiguous pair
    ranges.
  - `PrefillRoutedTileSchedulerTests.swift`: `sweepModeParsesItsFourValues` (`:351`)
    becomes five and keeps the fallback assertions;
    `prefillGapLeversDescriptionReportsTheSweepMode` (`:361`) gains `sweep=resident
    tail=96`; `sweepTailDefaultsAndFailsClosed` (`:384`) is unchanged and re-asserted
    against the new mode; `recencyBalanceRequiresBothRecencyModeAndCarryParticipation`
    (`:283`, in `PrefillMoEGroupingTests.swift`) becomes the renamed predicate over all
    five modes plus `participatesInCarry` false.
  - No new streamer API and so no new streamer test: `residentExperts()`'s behaviour is
    already covered by `residentSnapshotExcludesLoadingEntries`
    (`PreadExpertStreamerTests+CachePlanning.swift:167-183`), which is the precedent this
    task relies on. If the route timer forces the mask accessor, its test goes there.
  - The replay's own `--self-test` (dataset 8c is in the tree; Step 1 adds a composition
    dataset), run from the tool and not from `swift test`.

  **The rig and the rows.** Task 4's `t4-arms.sh` / `t4-arms-run3a.sh` shape with a third
  mode (`resident` = sweep resident, protect chunk) beside `carry` and `recency`, one
  binary, each arm relaunching the server, `SERVER_ENV` carrying the knobs: the live chain
  ×3 in both orders (A B B A A B) with REUSE of the default chain's payloads so every arm
  sends identical bytes; the long-answer follow-up ×2 (`TURN2_MAX_TOKENS=512`, the rig's
  existing override, `tools/turn-rig.sh:37-39`); the `d512-300` and `d512-1k` long-decode
  pairs ×2 in both orders; the 300 pair ×2 in both orders and 1k / 2k once each; 12k once
  per arm; one `recency` cell on the live chain and the 300 pair to re-confirm the prize
  and the loss on the same binary; a traced `resident` chain last so the replay's
  prediction (9,370 / 6,689 | 707 / 230 | 591 / 96) is checked against the box; the cold
  first requests read off the same arms; `restore` at the end. Rows exactly as
  `t4-rows.py` reads them, the pool counters included. **No new counter.**

  **Files.** `sources/Shrike/Runtime/Inference/RealForwardRunner.swift`: the fifth
  `PrefillSweepMode` case and `usesComputedOrder` (`:143-148`), the renamed chunk
  predicate (`:693`), the sweep field in `prefillGapLeversDescription` (`:376-378`), the
  carry write (`:2593`), `buildPrefillRoutes` (`:5173-5242`) and its new per-chunk
  `[Bool]` scratch beside `routeIDScratch` (`:5183-5188`).
  `sources/Shrike/Kernels/Prefill/MoE/PrefillMoEGrouping.swift`: `residentFirstBalanced`
  beside `recencyBalanced` (`:116`), reusing `recency` (`:103`) and `packByRows` (`:143`).
  `sources/Shrike/Runtime/Inference/ModelExpertIO.swift`: no new API, the doc comment on
  `routedExpertResidentIDs` (`:175-177`) corrected now that it has a hot-path caller.
  `tools/expert-pool-replay.py`. Tests: the two files above. Docs: the design doc's Task 5
  section and its After T5 table, this plan's checkboxes, the Follow-ons entry retired.
  **Lint:** `encodeRoutedMoEPrefill`'s baseline entry records line 5418 and this task adds
  lines above it in the same file, so `swiftlint lint --write-baseline
  .swiftlint-baseline.json` is expected if the gate reports a stale entry (T0 to T4 each
  hit this on a different function); `buildPrefillRoutes` itself must stay under 120 body
  lines. **Unchanged:** every `.metal` file and kernel body, `expert_io.c` and the reader,
  the planner's placement logic, the victim comparator, the prompt cache.

  Steps:

  - [ ] Step 1 (the replay tool, its own commit as Task 4 kept its instrument): the
        uncommitted `--sweep-order resident-first` in the tree today is the **plain**
        order, which is not what this task ships. Step 1 makes `resident-first` the
        shipped composition (three balanced groups, tiled flat) behind a new
        `--sweep-tail K` (default 96, the knob's mirror), keeps the plain order reachable
        as `resident-first-plain` so step zero's rows stay reproducible, extends
        `--self-test` (dataset 8c plus a composition dataset asserting the group order,
        the packing and the flat tile count), and reproduces every cell of the two tables
        above on the four archived traces. Gates 1 to 4.
  - [ ] Step 2 (the Swift order behind the knob): tests RED first, then
        `PrefillSweepOrder.residentFirstBalanced`, the residency snapshot and its scratch,
        the fifth mode with the renamed predicate and the printed field, the carry write.
        Default `carry` in the commit. Gates 1 to 4 on every amend.
  - [ ] Step 3 (numerics): golden IDENTICAL on both profiles at `carry`, `resident` and
        `recency` on the M4 Pro at every amend, and on the mini at the default and the
        candidate cell at every amend and at all three cells at the landed commit.
  - [ ] Step 4 (the arms on the mini, one binary): the rig above, `pgrep` and
        `memory_pressure -Q` before every launch, production restored at the end. The
        traced `resident` chain replayed against the box before the verdict is written.
  - [ ] Step 5 (the rule): real and free by the rows above → `resident` becomes the
        default by amend; real and not free → it lands as a knob beside `recency` and the
        cost is priced in the verdict, never hidden by a bar.
  - [ ] Step 6 (docs): the design doc's Task 5 section, the After T5 table, the lever
        entries updated (the sweep-order entry closes), this plan's checkboxes, the
        Follow-ons entry retired. Task review by a fresh reviewer, fixes folded into the
        owning commit, re-review.

  **Risks and what falsifies the model.**
  - **The snapshot's timing.** The replay reads residency from the pool's `slot_expert` at
    the chunk's first tile, and the box reads it one step earlier, at the layer's route
    build. Between those two points nothing touches layer L's pool in the shipped loop
    (the previous chunk's batches are drained at `:5885-5887`), but the loop has two
    paths and only fetch depth 2 ships; a `SHRIKE_PREFILL_FETCH_DEPTH=1` capture would not
    replay cleanly, which Task 4 already recorded. If the traced `resident` chain does not
    replay to within Task 3's ±3 counter jitter, the residency the order sees is not the
    residency the replay modelled and Step 4 stops there.
  - **The host cost.** Round 2's +0.44 s of host in the tile composition is the warning.
    The budget above is ≈ 0.4 ms per chunk for the snapshot and no new cost for the
    packing; `prefillRouteNanos` under `SHRIKE_PHASES` is the pre-registered read, and an
    allocation-free mask accessor is the fix if it moves.
  - **The tile count.** The flat tiling is load-bearing: the grouped tiling costs +20
    tiles on the 21-token turn and +18 on the warm 300 chunk, ≈ +24 ms each, which is
    larger than those rows' prefill gains. If the shipped composition is ever changed to
    per-group tiles, the follow-up turn's row is where it shows.
  - **12k's decode row is a tie-break**, not a lever: 219 to 255 experts needed against
    128 slots holding hit residents, so the 8-token decode moves 427 to 708 between
    compositions that differ nowhere else. It is reported and is not a verdict row.
  - **pair300's +30 and d512-300's +48.** The two rows that go the wrong way are both a
    different prompt's overlap with what the previous request left, and both are inside a
    3.2 to 3.4 s wall's ±1 % drift as modelled. If either moves more than modelled on the
    box, the lever is real and not free and lands as a knob.
  - **The prize is a fixed transient.** If the arms show the decode gain scaling with the
    answer's length rather than sitting in its first 32 to 64 tokens, the profile above is
    wrong and the modelled per-card saving (≈ 0.6 s, not 41 % of decode) is wrong with it.
  - **The settle's chunks take the order too**, and their misses fall (the chain 897 →
    689). That is background work whose only visible effect is the pool state the next
    request starts from, which is already inside every row above; a settle that is
    cancelled mid-chunk by an arriving request leaves a different state than the replay
    models, and the traced chain is the check.
  - **Memory and the box.** No new per-slot state and no allocation per tile; the pool's
    9.06 GB does not move and `--ram-budget 8G` stays. The mini is production: every arm
    stops the server on 8081 and relaunches it, Turbo on 8080 is never touched, one model
    process at a time.

## Follow-ons (not scheduled)

- The GDN chunked scan below its 64-row gate (`GDN.chunkTokens`): ≤ 33 ms on a
  21-row turn, and a numerics change for every 32–63-row chunk (Task 3).
- The routed experts' matrix gate reads the configured chunk and its strict
  `> 32` protects the 32-token MTP draft scratch's memory budget; lowering it
  needs that budget priced (Task 3; the routed path is already the matrix path at
  21 rows in production).
- The expert reader's publication signals `min(count, threads)` workers instead
  of broadcasting to all (Task 2 review): sound because every worker re-checks
  the claim predicate before parking; the shutdown path keeps its broadcast. The
  read is turn 3's +26 ms at eight threads; at the default four it is inside
  noise, so this rides on any task that raises the thread count.
- Collapse Task 1's two routed tile loops into one (the lookahead as a
  predicate; the scheduler's `decide` and the commit-before-append valve
  reconciled; the begin/await/drain sequencing factored into a host-testable
  decision) — **before the chapter merges to main**, in its own commit with its
  own golden pair, and first if any task edits `encodeRoutedMoEPrefill`'s loop
  before then (Task 1 review).
- **The resident-first recency sweep** (Task 4's next step, a task of its own): the
  chunk's needed experts that are already resident swept first in recency order (every
  hit harvested before any eviction, T0's carry trick made exact through the pool's
  residency), then the absent ones with the recency tail last, tiles balanced by row
  weight. Step zero: traces of the 300 / 1k / 2k pairs and 12k under today's order,
  the replay's verdict on both regimes, then the mini. The prize is the first turn's
  decode (−0.6 to −0.8 s measured for the plain recency order) without the warm-prefill
  and multi-chunk losses that kept it a knob.
- The prompt cache's settle after a request whose prompt has no cached prefix
  re-prefills the whole prompt in the background (`settle_reset reason=no_prefix_snapshot`,
  ≈ 6 GB of expert reads after a 300-token request, the pool swept): a cache-chapter
  item found by Task 4's trace.
- The expert-cache policy env parse lives in `PreadExpertStreamer.init`, so a bad value
  fails per layer mid-request instead of at launch, and never reaches
  `RuntimeConfiguration.expertCachePolicy` (Task 4 draft; a hoist to a static
  `environmentValue` in the shape of `ExpertIOBackend`'s).
- A counter for how often chunk-aware protection's starvation fallback fires (Task 4
  review note; the replay models the fallback, the box does not report it).
- The cold first request's +0.27 s under protection on one of two same-shape arms
  (Task 4 round 3: the `300` arm's 289-row first request 5.48 → 5.76 s, the `d512-300`
  arm's identical request 5.47 → 5.49): measure before explaining; a per-plan early-out
  when no resident is protected is the candidate only if the measurement points at
  the scan.
- The prompt cache's interior snapshots (a prompt that diverges inside a stored
  entry re-prefills in full; append-only turns are served).
- v12's prefill kernel follow-ons stay in [v12-prefill-matrix-kernels.md](v12-prefill-matrix-kernels.md).
