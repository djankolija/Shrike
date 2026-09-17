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
- [ ] **S0.5 The wide capture** (**a model run on the mini**): the diagnostic behind
      `SHRIKE_PREFETCH_TRACE` (the probe's full scores per position and layer; the
      prefill's per-token top-8 per layer as a new trace line kind), golden identical
      on both boxes with the var unset; one production lifetime per shape, four
      shapes, the captures archived; `tools/prefetch-coverage.py --top-m 8 12 16 24`
      and the table's coverage with the prompt seeded; the lifetimes as the opening
      ledger; the record.
- [ ] **S0.6 The attention row's fixed part** (**a model run on the mini**, the rig
      with kernel stats, the 300): the attention layers' command by kernel; B3 and B4
      priced on the current tree; the record.
- [ ] **S0.7 The record and the ruling**: the read budget allocated; Task 1's shape;
      Task 2's go or no-go; the fold's design note scheduled; B3's home. Davor's
      ruling on each, recorded.

## Task 1: the predictor (class 1; the shape from S0.7)

- [ ] **T1.1 The pre-registration**: the rows per shape (misses per token by layer
      group, reads per token, cells held, io, the token) graded T with a range from
      the replay; the answers expected identical.
- [ ] **T1.2 The table**: per layer, keyed by token id, the source S0.2 named; filled
      from every decoded token's route as the classifier reports it; seeded from the
      prefill if S0.5 earned it; its memory bounded and stated.
- [ ] **T1.3 The batch**: at the token boundary, after the sampled id's readback, the
      table's predictions for the served layers issued into ring cells on the ring's
      landing path; the in-flight budget for the batch; the cells held until the
      layer's classifier has run or the reclaim needs them; a counter for each of
      issued, landed before the classifier, late, wrong, refused.
- [ ] **T1.4 The wider probe** (only if S0.5 earned it): the readback widened on a
      diagnostic-free path or the probe's list lengthened where it is issued, at the
      width named; the same counters.
- [ ] **T1.5 The draft for layer 0** (only if S0.3 earned it): prompt lookup over the
      request's ids on the host at the token boundary, its proposal through the
      table for layer 0 a pass ahead; its hit rate counted.
- [ ] **T1.6 Tests**: the table's fill and lookup; the batch's issue and its cell
      accounting; the counters; the draft's match; the runner tests' toy shape
      unchanged.
- [ ] **T1.7 The gates, the golden, the deploy.**
- [ ] **T1.8 The arms** against S0.5's ledger, two lifetimes per shape, interleaved
      with the previous build if the box drifts; the record in the design document
      with the pre-registered rows, moved or not.

## Task 2: the policy and the split (class 1; only on S0.4's number)

- [ ] **T2.1 The predicted-future eviction**: the plan's victim choice steered by the
      table's predictions for the next tokens; its own commit, gates, golden, arms.
- [ ] **T2.2 The slot split**: the per-layer slot count from S0.4, the total held;
      its own commit, gates, golden, arms.
- [ ] **T2.3 The record.**

## Task 3: the agreed cells and the fold (class 1; structure)

- [ ] **T3.0 The design note**: the four edges (the stop path against the GDN state a
      committed pass mutates in place; the error surfacing per layer when a token is
      one command; the agreed-cell contract between the host and the kernels, encoded
      before the router has run, with the fallback when a layer's misses exceed its
      free cells; the cancel with two in flight, including the batch's reads in
      flight), written in the design document for Davor's ruling before T3.1.
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

## Task 4, held: the attention row's fixed part (B3, B4)

- [ ] **T4.0** Only on S0.6's number and Davor's ruling: the folds into neighbours
      (the combine into the o-projection's prologue, RoPE and the KV append into the
      projection's epilogue) with the volatile slot where a fused kernel would elide
      a rounding, and one pass at short context if B4 earns it; class 1, golden
      identical.

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
