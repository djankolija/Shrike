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

- [x] **T3.0 The design note.** DONE 2026-09-17 (`8ee535f`): the design document's
      Task 3 section, the tree read on the edges at `7596f86`, the four edges
      designed, the stop path as two shapes, the fold's drain invariant, the
      per-encoder error naming, the overflow as a victim on the path, the
      instruments the fold retires, the pre-registration, five points for the
      ruling. **Davor's ruling (2026-09-17): Shape B for the stop path (the drain's
      cost hidden by the client's turnaround; the unguarded drain, no cancel word);
      the overflow as the on-the-spot eviction; the word clock in place of the
      per-layer GPU rows; the per-encoder error option on and measured on the first
      T3.2 build; the order T3.1, T3.2, T3.3.**
- [x] **T3.1 The agreed cells.** DONE 2026-09-17 (`2a0f7d9`): the fixup encoded
      with the layer behind the event wait as the speculative kernels over the
      host's per-layer cell row, `MoESpecDispatchArgs` at four grids, the
      speculative pair gaining the status word and the fallback cells; a timeline
      value per routed layer reserved at the encode; the host on the word (the
      readback, the landed predictions leased, every miss's cell from a landing,
      the ring's `claimDemand` or `reserveOverflowSlot` counted as
      `agreed_overflow`, the batch with the layer's value, the next prediction);
      the plan at the next wake with the misses counted as the reads; the drain in
      its T3.1 form; the coordinator's status words recycled; the host-built fixup,
      `DecodeExpertPartition`, the pending routed command, the completion clock and
      seven dead rows retired, `agreed_overflow` and `cells_leased_peak` added,
      `tools/decode-rows.py` following; tests, 1,275 in 174 suites; the four
      gates; the golden identical bare and configured on both boxes; deployed; the
      arms flat against Task 1's on all four shapes (misses within 0.2, the token
      within 0.4 ms, no overflow, the answers identical), the record in the design
      document; the scripts, logs and rows at
      `~/.claude/handoffs/archive/shrike-v20-t31/`.
- [x] **T3.2 One command per token.** DONE 2026-09-17 (`51fc54f`): `TokenCommand`
      with the descriptor's `encoderExecutionStatus` always on (measured free on
      the mini's four shapes through a variable that lived for the arms only; the
      known names stay fifteen), the layers and the boundary as its encoders,
      encoded a layer per word during the previous token and committed on the
      boundary word after the stop check; the drain invariant in full with
      `awaitCompletion`'s ten-second deadline behind the word and boundary wakes;
      `describeCommandBufferError` naming the faulted encoder; `DecodeWordClock`
      in place of the per-layer GPU rows, read by `tools/decode-rows.py`; the
      deferred records, the race split and `path_router_wake_ms` retired; tests
      (the faulted encoder, the deadline, the word clock, the toy runner's failed
      read at layer 1 naming its layer and the runner reusable); 1,280 tests in
      175 suites; the four gates; the golden identical bare and configured on both
      boxes at the final tree; deployed; the arms flat within the drift and 0.2
      to 0.6 ms faster on most rows, the misses and the answers identical, the
      boundary gap measured directly at 0.26 to 0.34 ms per token; the record in
      the design document; the scripts, logs and rows at
      `~/.claude/handoffs/archive/shrike-v20-t32/`.
- [x] **T3.3 Committed ahead (Shape B).** DONE 2026-09-17 (`4df0be3` the gate,
      `d356a8e` the runtime): the GDN state and conv tail of every linear layer in
      two parities, the decode kernels taking the state entering the step and the
      state leaving it, a pass reading the cursor's parity and writing the other,
      prefill, the snapshot and the restore on the cursor's parity; the next
      token's command committed after the current token's last word, its boundary
      from the caller's sampler closure given the pass's position and the word of
      its parity (two boundary words, the runner's), the loop's `last` on the pass
      before max tokens so it commits nothing ahead; the stop token, the stop
      strings, the external stop and a disconnect seen one pass late and the pass
      ahead released at the loop's exit (its forty values published failed, no
      reads, no cells) so it runs through during the finish frames and the client's
      turnaround, the wait at the next entry point counted as `drained_passes` and
      `drain_ms`; the parity left where it was; the extra pass's trace rows and
      counters never written; two deviations on the tree's evidence, recorded: the
      cursor needs no rewind (a pass's advance sits at the end of its own word
      loop, which the extra pass never runs) and the settle needs no drain (the
      snapshot reads what the extra pass never writes); the two-turn continuation
      gate as the golden's `turns-lh` profile (the CLI's `--follow-up`), its
      reference captured before the commit ahead on both boxes; the cancel's tests
      on the Qwen toy and the loop; 1,289 tests in 176 suites; the four gates; the
      golden identical on all five profiles bare and configured on both boxes;
      deployed, production under the v20 configuration; the scripts and logs at
      `~/.claude/handoffs/archive/shrike-v20-t33/`.
- [x] **T3.4 The arms and the record.** DONE 2026-09-18: two lifetimes per shape
      on four shapes, bare and configured, against T3.2's arms: the boundary gap
      0.26 to 0.34 ms per token at T3.2 is 0.033 to 0.038 at T3.3 on every arm and
      shape, the token faster by about that or more on every configured row (0.4
      to 1.2 ms, the larger differences inside the drift) and on two bare rows,
      level on the other two, the misses, the io and the answers' bytes unchanged
      (all twenty responses identical to T3.2's), no overflow; the `token` row's
      GPU span equal to the token within 0.2 ms; the word clock's first row 0.57
      to 0.59 ms from the previous token's word; the drain's wait on the request
      after a stop 0.000 ms (one drained pass on a card lifetime's second line);
      the record in the design document; the scripts, logs, rows and instruments
      at `~/.claude/handoffs/archive/shrike-v20-t33/`.

## Task 4: to a chapter of its own (Davor's ruling, 2026-09-17)

B3 and B4, the attention row's fixed part, leave this chapter: S0.6 found the
per-kernel instrument does not exist, and that chapter's step zero builds it once.

## Close

- [x] ThreadSanitizer once on the whole suite at the final tree. DONE 2026-09-18:
      twice, clean both times, at `a4c6431`'s tree before the review's folds (1,289
      tests, 852 s) and at the final tree after them (1,292 tests, 850 s); the logs at
      `~/.claude/handoffs/archive/shrike-v20-t33/close-tsan*.log`.
- [x] The whole-branch review; the fixes folded into their owning commits. DONE
      2026-09-18: five findings (the report at `shrike-v20-t33/close-review.md`), each
      verified and folded: the KV growth under a running command and the drain's
      swallowed fault into T3.3's commit, the word wake's completion fallback into
      T3.2's, the overflow victim among the route's hits and SLRU's overflow placement
      into T3.1's; three tests added, 1,292 in 176 suites; the four gates and the
      golden on both boxes rerun at the folded tree; the record in the design
      document's close.
- [x] `docs/architecture.md` brought to the tree. DONE 2026-09-18: the decode path
      rewritten for the token's command, the drain invariant, the agreed cells and the
      commit ahead; the residency writers' table at twelve sites; the arena and the
      ring under the table and SLRU; the demand path's status ring; the knobs at
      fifteen; the long functions by swiftlint's count; the instruments; the v20
      history entry; every line reference re-anchored at the close's tree.
- [x] The design document's closing block. DONE 2026-09-18: the tally from the
      opening ledger to T3.4 on four shapes, the count, what the chapter settled, the
      review's folds, what remains and where it went.
- [x] Production on the mini at the close's build, golden verified. DONE 2026-09-18:
      the final build deployed, the golden identical on all five profiles bare and
      configured on both boxes, production relaunched under the v20 configuration and
      its load line confirmed.
- [x] The merge to `main` on Davor's go. DONE 2026-09-18: main fast-forwarded from
      `c4a96d6` and pushed; the branch and its backups deleted.
