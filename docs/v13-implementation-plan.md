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
  (`:977-988`) → `submit_batch` (`Sources/ShrikeKernelsC/expert_io.c:237-277`).
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

## Follow-ons (not scheduled)

- Collapse Task 1's two routed tile loops into one (the lookahead as a
  predicate; the scheduler's `decide` and the commit-before-append valve
  reconciled; the begin/await/drain sequencing factored into a host-testable
  decision) — **before the chapter merges to main**, in its own commit with its
  own golden pair, and first if any task edits `encodeRoutedMoEPrefill`'s loop
  before then (Task 1 review).
- The prompt cache's interior snapshots (a prompt that diverges inside a stored
  entry re-prefills in full; append-only turns are served).
- v12's prefill kernel follow-ons stay in [v12-prefill-matrix-kernels.md](v12-prefill-matrix-kernels.md).
