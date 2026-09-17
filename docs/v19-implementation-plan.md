# v19 implementation plan: the scan rewrite

Companion to [v19-scan-rewrite.md](v19-scan-rewrite.md). Checkboxes here are the
only status tracking. Branch: `perf/v19-scan-rewrite` off `main` at `2d357c6`.

Protocol per task: the four gates per commit (release build with zero warnings,
`swiftlint lint --strict`, `tools/check-md-links.py`, `swift test --no-parallel`);
the golden byte-identical on both boxes for every class-1 commit (from T1.5 on, the
two logits-head profiles too); deploy to the mini with `tools/mini-deploy.sh`
(binary plus the `*.bundle` directories); the arms rig, two production lifetimes per
shape on four shapes, read against the previous task's arms; the rows the task
pre-registered, moved or not, recorded in the design document's task record;
ThreadSanitizer once at the close. Every bench run and every arm on the mini stops
production and needs the session's deploy leave; production is restored and
verified against the golden before the session ends.

Order: step zero and Task 1 in parallel (Task 1 is class 1 and touches no kernel);
S0.5's ruling decides Task 2; Task 3 after the ruling; Task 4 only on S0.4's
no-load twin; the close.

## Step zero: the constraint named and the prototype priced (no runtime code)

- [x] **S0.1 The 7k shape and the four-shape baseline.** DONE 2026-09-17.
  - [x] `tools/turn-prompts.py`: the `t7k` / `t7kb` pair (112 entries at offsets 900
        and 1,000), generated into the rig's prompt directory; 7,463 prompt tokens
        measured.
  - [x] `tools/decode-rig.sh`: `d512-7k` in the shape list and the case.
  - [x] Two production lifetimes per shape, four shapes, at the v18 close's build;
        the ledger in the design document's step-zero record. The role on the
        current tree is `layer_kv` (the layer's whole held command since v18), 8.4 ms
        at the 300 and 24.4 at 7,463 context, the slope 2.23 ms per 1,000; the
        pre-registered table restated on it. Scripts and arms at
        `~/.claude/handoffs/archive/shrike-v19-step0/`.
- [x] **S0.2 The bench executable.** DONE 2026-09-17: `ShrikeAttnBench` at
      `sources/ShrikeAttnBench` (`Package.swift`), `Metal/ladder.metal` as a copied
      resource compiled at run time, `--arms`, `--positions`, `--repeats`,
      `--warmup`, `--seed`, `--list`; the production arm through the wrapper
      (`Attention` and `KVCacheQuantizer` made public for it); the fidelity check
      built into every run (the partials' hash per arm, the CPU-combined output
      against production's, 3.8e-6) rather than a separate test, since the ladder
      is a measurement tool; deployed to the mini with its bundle. Release build
      with zero warnings; the full gates run with the step-zero commit.
- [x] **S0.3 The ablation ladder (B7) on the mini.** DONE 2026-09-17 in both arm
      orders; the table and the reading in the design document. The constraint is
      the loop form: the layout streams at the roof, every occupancy and latency
      remedy is null or slower, the V half costs 64 % of the time. S0.3b: the
      static trip count is 3.07× faster on the kernel (539 against 1,654 µs at 8k),
      not bitwise the copy (FMA contraction), the combined output equal to 3.8e-6.
- [x] **S0.3c The form search.** DONE 2026-09-17: the static loops with the V
      accumulate as `fma(o, alpha, p * v)` reproduce the shipped kernel's partials
      bit for bit at 613 µs against 1,653 (2.70×); the denominator stays as written.
      Task 2 is class 1.
- [x] **S0.4 The streaming prototype on the mini.** DONE 2026-09-17 (Davor's go,
      "be thorough"), after Task 2: `Metal/stream.metal`, the sweep over heads per
      simdgroup (2, 4, 8; 8 spills), the no-load twins, then the levers (the lazy
      rescale, two positions per iteration) interleaved with the fix's form; the
      full-row variant dropped on the ladder's evidence. Four heads per simdgroup is
      1.7× the fix's form on the kernel (M, interleaved), correct to one fp16 ulp;
      the levers a loss and a null; the chain under register-limited occupancy is
      what binds. Transferred: about 2 ms per token at 7k (3.4 %, T). The record in
      the design document's step-zero section.
- [x] **S0.5 The record and the ruling.** DONE 2026-09-17: the step-zero record
      complete (the ladder, the repair built as Task 2, the prototype priced at
      about 2 ms per token at 7k). Davor's ruling: build it ("small things, bit by
      bit, they accumulate"); Task 1 the instrument first, then Task 3 the rewrite
      under the class-2 gate.

## Task 1: the instrument (class 1; beside step zero)

- [x] **T1.1 The forced-token mode.** DONE 2026-09-17: `GenerationConfig.forcedTokens`
      (`Sampler.swift`), the loop in `RawCompletion.swift` taking the list's next
      id from the first generated position on the synchronous logits path (the
      boundary producer bypassed, a fused greedy head refused), stopping when the
      list ends; validated non-empty. Tests in
      `RawCompletionLoopTests+Instrument.swift` with the scripted producer.
- [x] **T1.2 The logits dump.** DONE 2026-09-17: `LogitsSink` (`LogitProducer.swift`)
      with `record(position:logits:)` over the fp16 buffer and
      `chose(position:token:)`; `GenerationConfig.logitsSink`; the loop records
      before the choice and reports the choice; `FileLogitsSink` in the CLI
      (`LogitsDump.swift`) writing raw fp16 rows and the JSON sidecar (vocab,
      positions, chosen, forced, the binary's SHA-256). Tested with the scripted
      producer: the rows' argmaxes are the tokens chosen.
- [x] **T1.3 The CLI flags.** DONE 2026-09-17: `--force-tokens`, `--dump-logits`,
      `--logits-head` in `Args.swift`; `Run.swift` reads the ids, caps the token
      count to the list, installs the sink, and takes the logits head whenever a
      sampler, the flag, forced tokens or a dump asks for it; the usage block and
      its inventory test updated; `CLIArgumentsTests+Instrument.swift`.
- [x] **T1.4 The comparison.** DONE 2026-09-17: `tools/logit-compare.py` (pure
      Python); the band three times the median of the per-position max |Δ| and an
      outlier rule at ten times the median (the board's run-maximum band hid every
      flip behind one bad position in the self-test); `--self-test` green; on the
      dev box a free run's dump against the forced replay of its own ids: KL and
      |Δ| zero at every position, the texts identical, a shifted list stopping
      where it ends.
- [x] **T1.5 The logits-head golden.** DONE 2026-09-17: `HEAD=logits` in
      `tools/golden-baseline.sh` (`--logits-head` on the CLI line, the `-lh` suffix);
      the two profiles captured on both boxes
      (`baselines/ornith15-int4-{short,long}-lh.{Mac167,mini}.txt`) and checked;
      `--check` with `HEAD=logits` covers them from here on.
- [x] **T1.6 The calibration run and the record.** DONE 2026-09-17: on the dev box
      the free dump against the forced replay, zero everywhere; the fused head
      against the logits head at temperature zero, character-identical on both
      profiles on both boxes; the stale-build segfault chased to a clean rebuild;
      the four gates on clean builds (1,254 tests); the golden identical on all
      four profiles on both boxes; production on the mini at `beeba44a30bbeb12`;
      the record in the design document.

## Task 2: the loop form (class 1; ruled first by Davor, 2026-09-17)

- [x] **T2.1 The pre-registration.** DONE 2026-09-17: the expected rows, graded T
      from the bench's 2.70×, in the design document's Task 2 section before the
      kernel changed.
- [x] **T2.2 The kernel.** DONE 2026-09-17: `attention_decode_partial_shared`
      (`attention.metal:509`), the three per-lane loops with a static trip count
      over `kPerLane` and a guard on the head dimension, the V accumulate as
      `fma(o, alpha, p * v)`; nothing else moved.
- [x] **T2.3 The arm.** DONE 2026-09-17 as three pieces rather than a new test: the
      bench's ladder (the v18-close kernel's copy against the production form,
      partials hashed identical at 1k and 8k on the mini, S0.3c); the existing
      bitwise arms of `AttentionTests` (specialized against unspecialized, so the
      guard folds the same on both paths) and the CPU-reference arms, 54 tests in
      10 suites green; the golden identical on both local profiles. The
      production pipeline in the bench: 246 µs at 8k on the M4 Pro against about
      650 before, within 10 % of the bench's form plus the combine.
- [x] **T2.4 The gates, the golden, the arms.** DONE 2026-09-17: the four gates
      (1,250 tests, 203 s); the golden identical on both boxes, the mini's with the
      server down; deployed (`ca2d3bab87469ef2`), two lifetimes per shape: the slope
      2.23 to 0.72 ms per 1,000, the 7k token 73 to 61.7 ms (13.7 to 16.2 tok/s), the
      card 62 to 59, the eight answers character-identical to S0.1's; the record in
      the design document's Task 2 section; production on the mini at this build.

## Task 3: the streaming scan (class 2)

- [x] **T3.1 The pre-registration.** DONE 2026-09-17: the rows from the bench's
      1.7× and the kernel arm's band, in the design document's Task 3 section
      before the kernel landed (committed with Task 1, d440004, so the order is
      in the history).
- [x] **T3.2 The kernel.** DONE 2026-09-17: `attention_decode_partial_stream` in
      `attention.metal` on the V4.1 function constants, the same buffers and
      partial contract, four heads per simdgroup by four streams, each stream
      writing its own partial (no merge: sixteen chunks dispatched, sixty-four
      partials per head, the combine untouched); `Attention.swift`: the `.stream`
      variant, `streamApplicable` (the served shape on int8 rows), its cached
      specialized pipeline, the geometry; the runner on `.stream`.
- [x] **T3.3 The kernel arm.** DONE 2026-09-17: `AttentionStreamTests` (nine tests):
      the reference at 0.02 on int8 rows at 3, 17, 96, 500 and 1,100 positions; the
      stream against the shared kernel, max |Δ| 6e-8 to 3.8e-6 on the fp16 output;
      the pipeline gate (served shape only, cached, fp16 rows and other shapes fall
      back). The int4 and fp16 rows keep the shared kernel by the gate, so their
      existing arms are the coverage.
- [x] **T3.4 The gates and the deploy.** DONE 2026-09-17: the four gates (1,257
      tests in 174 suites, zero warnings); the golden on the dev box identical on
      the short profile on both heads and a mismatch on the long profile from its
      twelfth token (the diff kept in the t3 archive); on the mini identical on
      all four profiles; a clean release build deployed (`t3-mini-new.sh`).
- [x] **T3.5 The instrument.** DONE 2026-09-17: both golden prompts forced with the
      old build's own tokens through old and new on both boxes; the dev box two
      near-tie flips (margins 0.031 and 0.047) inside the band on the long prompt,
      none on the short; the mini none on either; no defect anywhere; the tables in
      the design document's Task 3 record; dumps and logs in
      `~/.claude/handoffs/archive/shrike-v19-t3/`.
- [x] **T3.6 The read.** DONE 2026-09-17: the four profiles free-run on both boxes;
      only the dev box's long profile differs (from token 12: "I need to: 1. Count
      the number of entries…" for "First, let me count the entries…", the same plan
      in a different order, the same shelves listed after). Davor's ruling:
      variance ("Agreed"); the kernel accepted.
- [x] **T3.7 The golden re-captured.** DONE 2026-09-17: only the dev box's two long
      profiles changed and were re-captured (423 bytes, the same on both heads); the
      mini's four and the dev box's short profiles identical, untouched; `--check`
      identical on all four on both boxes.
- [x] **T3.8 The arms.** DONE 2026-09-17 against Task 2's ledger (the chapter's
      current baseline): `layer_kv` 12.8 to 10.25 on the 7k, 9.0 to 8.1 on the card,
      the slope 0.72 to 0.39 ms per 1,000, the 7k token 61.7 to 58.6 ms (17.05
      tok/s), the other rows flat, every pre-registered row met; the 7k's move is
      five times the drift, no A/B needed; the record in the design document;
      production on the mini at `3042665f2fb11370`; the commit.
- [ ] **T3.9 The hardening (Davor's ruling, 2026-09-17: worth doing without a gain).**
      A thrown load error in the runner when the served model's shape or KV precision
      is not the streaming kernel's, in place of the silent fallback (the wrapper's
      fallback stays for the tests and the bench); the head-dim-256 assumption
      stated at the eight-byte load; the v11 simdgroup variant retired
      (`attention_decode_partial_sg`, `.simdgroup`, its tests); an fp64 reference arm
      reporting both kernels' error against the exact value. The four gates, the
      golden identical on both boxes (class 1), the commit.

## Task 4, held: the matrix-unit tile (B6)

- [ ] **T4.0** Only if S0.4's no-load twin puts Approach A at the ALU wall short of
      the roof: the pre-registration (including the wider band the half-precision
      inputs imply), then T3.2 to T3.8 for the tile kernel.

## Close

- [ ] ThreadSanitizer once on the whole suite
      (`env TSAN_OPTIONS=suppressions=tsan-suppressions.txt swift test --no-parallel --sanitize=thread`).
- [ ] The whole-branch review; fixes folded into their owning commits.
- [ ] `docs/architecture.md`: an attention section (the decode scan, the KV row
      format, the two-pass contract) at the final tree, references verified.
- [ ] The design document's closing block: the tally from S0.1's ledger to the last
      task on four shapes, what the chapter settled, what remains and where it went.
- [ ] Production on the mini at the close's build, the golden (re-captured) identical
      on both boxes.
- [ ] The merge to `main` on Davor's go.
