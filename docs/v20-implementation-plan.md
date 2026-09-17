# v20 implementation plan: the SSD mechanism

Companion to [v20-ssd-mechanism.md](v20-ssd-mechanism.md). Checkboxes here are the
only status tracking. Branch: `perf/v20-ssd-mechanism` off `main` at `c4a96d6`.

Protocol per task: the four gates per commit (release build with zero warnings,
`swiftlint lint --strict`, `tools/check-md-links.py`, `swift test --no-parallel`);
the golden byte-identical on all four profiles on both boxes for every commit (the
chapter is class 1 throughout); deploy to the mini with `tools/mini-deploy.sh`
(binary plus the `*.bundle` directories); the arms rig, two production lifetimes
per shape on four shapes, read against the previous task's arms, every arm carrying
misses per token, reads per token, cells held, the io and the token; the rows the
task pre-registered, moved or not, recorded in the design document's task record;
ThreadSanitizer once at the close. Every model run on the mini stops production and
runs under the session's deploy leave (given 2026-09-17); production is restored and
verified against the golden before the session ends.

Order: step zero's replay arms first (S0.1 to S0.4, no model), the two model runs
(S0.5, S0.6), the ruling (S0.7); Task 1 as ruled; Task 2 only on S0.4's number;
Task 3 last, its design note before its build; Task 4 only by ruling; the close.

## Step zero: the board priced on the current tree

- [x] **S0.1 The replay's table mode.** DONE 2026-09-17: `tools/expert-pool-replay.py`
      with `--table-fills`, `--table-layers`, `--table-width`, `--table-source
      last|last2|last3|freq|none`, `--table-cells`, `--union-previous`, `--draft
      prompt-lookup:N`, `--draft-layers`, `--prompt-pieces`, `--table-seed prefill`,
      `--table-future`, `--table-protect`, `--slots-json`; fills as per-source
      `(candidates, budget)` pairs; the `q` and `t` line kinds; the report by layer
      group with reads per position and the cells at the pass start; the self-tests.
      `ShrikeCLI --tokenize <path>` (the tokenizer only); the four rig prompts
      tokenize to the traces' exact counts. The four gates on the CLI change: 1,259
      tests in 174 suites, zero warnings, lint and links clean. Scripts and outputs
      at `~/.claude/handoffs/archive/shrike-v20-step0/`.
- [x] **S0.2 The table as fills.** DONE 2026-09-17: over the pool 0.33 (layer 0) to
      1.34 (all forty) misses saved per position; over the probe 0.31 to 0.85, 0.24 to
      0.71 ms modelled, for 0.4 to 4.2 more reads; the previous position a null by
      construction; the record in the design document.
- [x] **S0.3 The draft.** DONE 2026-09-17: prompt lookup proposes on 0.44 of positions
      at a 0.39 hit rate (n = 2); the table keyed on it saves 0.05 misses per position
      at layer 0 against 0.33 keyed on the real token; closed for the chapter; the
      record.
- [x] **S0.4 The policy and the split.** DONE 2026-09-17: knowledge in the policy at
      a horizon of one is null (0.06 per position at the ceiling); the Belady-on-the-
      table form leaks the future tokens and is recorded as a bound; the split is the
      lever: one allocation from the 300's profile (94 to 210 slots per layer, the
      same 5,120) saves 2.4 to 3.7 misses per position over the probe out of sample,
      1.8 to 2.6 ms modelled; SLRU about a miss on the longer shapes; the record.
- [x] **S0.5 The wide capture.** DONE 2026-09-17: the diagnostic (the probe's scores
      per layer in two banks by position parity, the rankings to width 32 as JSON
      lines in `SHRIKE_PREFETCH_TRACE`, the `t` and `q` line kinds in
      `SHRIKE_ROUTE_TRACE`), golden identical on both boxes, deployed; two runs on the
      mini (the first without rankings, its bug fixed and folded into the commit), one
      lifetime per shape, four shapes, archived under
      `~/.claude/handoffs/archive/shrike-v20-step0/capture/`; zero stale-slot rows;
      the width priced: coverage of the remaining miss layers 0.21 / 0.47 / 0.62 /
      0.76 / 0.84 at widths 8 / 12 / 16 / 24 / 32 at precision 0.15 down to 0.04 and
      reads 30 to 340 per token, closed at distance one on a bandwidth-bound window;
      the prompt-seeded table closed; the record.
- [x] **S0.5b The ranking at distance two and three.** DONE 2026-09-17 (Davor's
      ruling that the width is not closed while lead could make the low teens free):
      the routers two and three layers ahead on the diagnostic path
      (`MoE.encodeRouterScores`, `probe_ranking_d2`/`_d3`, the coverage tool's
      `--distance`), golden identical on both boxes, one lifetime per shape at
      `3060034`, zero stale rows; the bar half cleared: width twelve at distance two
      recalls 0.40 to 0.45 of the remaining misses (bar 0.32) at precision 0.08 to
      0.09 (bar 0.13), ranks nine to twelve at 0.076; two more useful reads per token
      for 37 more reads; the width closed for the chapter with the number, distance
      two at width eight noted for a later chapter; the record.
- [x] **S0.6 The attention row's fixed part.** READ 2026-09-17, no run: the kernel
      stats cannot split a held command and the GPU counters sample per encoder, so
      the arm needs an instrument that does not exist; B3 and B4 to a chapter of
      their own by Davor's ruling (S0.7).
- [x] **S0.7 The record and the ruling.** DONE 2026-09-17: the record and the
      recommendation in the design document's step-zero record; **Davor's ruling:
      proceed with what the data says**: Task 1 the pool's allocation (the split with
      SLRU as its policy), Task 2 folded in, the table not built, Task 3 as planned,
      Task 4 to its own chapter, the width closed by S0.5b's number, the read budget
      untouched.

## Task 1: the pool's allocation (class 1; the shape ruled at S0.7)

The split with SLRU as its policy. The predictor, the batch, the wider probe and
the draft that this task carried before the ruling are not built (S0.2, S0.3,
S0.5, S0.5b); the table's design and the replay's mode stay on record.

- [x] **T1.1 The pre-registration.** DONE 2026-09-17: the production miss profile
      from the four S0.5 captures' plan rows (51,160), eleven to one between layer 0
      and the quietest layer; blends 0.2 to 0.5 and S0.4b's table re-priced by
      replay over the probe on the current tree's captures and the v14 captures;
      blend 0.3 the reference (103 to 240 slots, 3.08 misses per position saved on
      the current tree, 2.24 ms modelled; 0.25 to 0.35 within a tenth); SLRU on the
      pool basis zero to three misses by shape; the rows per shape in the design
      document's Task 1 record; the driver and tables at
      `~/.claude/handoffs/archive/shrike-v20-t1/`.
- [x] **T1.2 Per-layer slots.** DONE 2026-09-17 (`bc3e25c`): the streaming mode's
      optional per-layer table, the model's construction with prefix-sum cell ranges
      and the arena from their sum, a wrong-length table refused at load, the two
      prefill sites per layer, `SHRIKE_EXPERT_SLOT_TABLE` (a comma list or a JSON
      path, refused unless the count, the floor of 8, the dense zeros and the
      budget's total hold), the uniform table the default; the server passes it.
- [x] **T1.3 SLRU.** DONE 2026-09-17 (`f0e056c`): `ExpertEvictionPolicy` as the
      mode's third value, the streamer's SLRU on the replay's rule, `SHRIKE_EXPERT_POLICY`
      (aging-lfu | slru | slru:<share>), the known names fifteen; the server and the
      CLI pass both variables so the golden covers the configured pool; the load
      description names the configuration.
- [x] **T1.4 Tests.** DONE 2026-09-17: the table parser's accept and refuse cases and
      the file form, the policy parser and the capacity rule, the toy model under a
      table placing layer 1's cells after layer 0's three and refusing a wrong-length
      table, SLRU against aging-LFU on the sequence A B C A B D E, the description's
      two forms; 1,268 tests in 175 suites.
- [x] **T1.5 The gates, the golden, the deploy.** DONE 2026-09-17: the four gates on
      every commit; the golden identical on all four profiles on the dev box bare and
      configured (the reference table scaled to the CLI's 64 slots plus SLRU) and on
      the mini; deployed.
- [x] **T1.6 The arms.** DONE 2026-09-17: three arms per shape interleaved, two
      lifetimes each, the configuration confirmed in every arm's server log; the
      split alone +4.4 to +5.3 % tok/s on the four shapes (misses 19 to 20 down to 14
      to 17 per token, io 13.5 to 15.1 down to 10.9 to 12.9 ms), with SLRU +4.4 to
      +7.8 %; the pre-registered rows met or beaten on every shape; the record in
      the design document's Task 1 section; the arms archived at
      `~/.claude/handoffs/archive/shrike-v20-t1/arms/`. Both ship as the mini's
      launch configuration.

## Task 2: folded into Task 1 (Davor's ruling, 2026-09-17)

SLRU is T1.3; the predicted-future eviction was null at a legitimate horizon
(S0.4) and is not built.

## Task 3: the agreed cells and the fold (class 1; structure)

- [ ] **T3.0 The design note**: the four edges (the stop path against the GDN state a
      committed pass mutates in place; the error surfacing per layer when a token is
      one command; the agreed-cell contract between the host and the kernels, encoded
      before the router has run, with the fallback when a layer's misses exceed its
      free cells; the cancel with two in flight, including the batch's reads in
      flight), written in the design document for Davor's ruling before T3.1.
      WRITTEN 2026-09-17 (`7596f86` read): the design document's Task 3 section,
      the tree on the edges, the four edges designed, the stop path as two shapes
      (A, encoded ahead and committed on the word, recommended; B, committed ahead
      with a GDN parity and a drain, designed), the fold's drain invariant, the
      per-encoder error naming, the overflow as a victim on the path, the
      instruments the fold retires, the pre-registration, five points for the
      ruling. **Davor's ruling pending; T3.1 waits on it.**
- [ ] **T3.1 The agreed cells** (v18's T2.1 to T2.5 as written): the read of the
      ring's cell leases and the index swap; the fixup encoded before the route with
      an indirect phase 1 over the classifier's miss list, the reduce, the residual,
      behind the event wait; the host's on-word path reduced to the reads' issue into
      agreed cells; the plan moved to the next wake; the fallback to the host-built
      fixup when misses exceed free cells, counted; tests; gates and golden; deploy
      and arms (the host's path fields off the path, misses per token recorded for
      drift).
- [ ] **T3.2 One command per token**: the forty layers' held commands as one, the
      token boundary inside it, the host feeding reads and signalling events; tests;
      gates and golden; arms expected flat.
- [ ] **T3.3 Two in flight and the cancel**: the next token's command encoded while
      the current runs; the cancel path per the design note; the stop path; the
      error surfacing per layer; tests for each edge.
- [ ] **T3.4 The gates, the golden, the deploy, the arms, the record.**

## Task 4: to a chapter of its own (Davor's ruling, 2026-09-17)

B3 and B4, the attention row's fixed part, leave this chapter: S0.6 found the
per-kernel instrument does not exist, and that chapter's step zero builds it once.

## Close

- [ ] ThreadSanitizer once on the whole suite at the final tree
      (`env TSAN_OPTIONS=suppressions=tsan-suppressions.txt swift test --no-parallel --sanitize=thread`).
- [ ] The whole-branch review; the fixes folded into their owning commits.
- [ ] `docs/architecture.md` brought to the tree: the table, the batch, the cells,
      the fold, the instruments, the v20 history entry, references re-anchored.
- [ ] The design document's closing block: the tally from S0.5's ledger to the last
      task on four shapes, what the chapter settled, what remains and where it went.
- [ ] Production on the mini at the close's build, golden verified.
- [ ] The merge to `main` on Davor's go.
