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

- [ ] **S0.1 The 7k shape and the four-shape baseline.**
  - [ ] `tools/turn-prompts.py:45-50`: a `t7k` / `t7kb` pair (about 112 entries at
        a fresh offset so no prompt is a prefix of another), generated into the rig's
        prompt directory beside the existing three; the prompt token count measured
        from the server log and recorded (target about 7,000).
  - [ ] `tools/decode-rig.sh`: `d512-7k` in the shape list (`:3`) and the case
        (`:129`), the same 512-token cold answer and eight-token follow-ups.
  - [ ] Two production lifetimes per shape on the current tree, four shapes; the
        runner and kernel stats parsed; `attn_layer_kv`, the wall, the misses and the
        miss window per shape recorded in the design document as the chapter's
        ledger. The 7k row is expected near the live log's 21.3 ms; a large
        difference is investigated before anything is built on the number.
- [ ] **S0.2 The bench executable.**
  - [ ] `Package.swift`: an executable target `ShrikeAttnBench` at
        `sources/ShrikeAttnBench`, depending on `Shrike`, its kernel sources under
        `sources/ShrikeAttnBench/Metal/` shipped as a copied resource and compiled
        with `makeLibrary(source:)` at run time, the project's own pattern
        (`Package.swift:48-50`).
  - [ ] `sources/ShrikeAttnBench/main.swift` and `Bench.swift`: arguments
        `--arm <name>` (repeatable), `--positions <n>` (repeatable; default 1024,
        4096, 8192), `--repeats <n>` (default 5), `--kv-bits 8`; synthetic K, V and Q
        at the served shape (two KV heads, sixteen query heads, head dim 256, int8
        rows of 544 bytes with random values, scales and biases; Q random fp16);
        timing by the command buffer's `gpuStartTime` and `gpuEndTime`, the median
        of the repeats; one line per arm and position count: µs per position, ns per
        KB scanned, GB/s against the roof, plus a checksum of the output so a broken
        arm cannot post a fast number.
  - [ ] The baseline arm dispatches the production kernel through the library's own
        wrapper (`Attention.swift`) so the bench and the runner agree on the
        geometry; the ablation arms are the kernel's source under the bench's
        `Metal/` with one function constant per switch.
  - [ ] The four gates; a test that the bench's baseline arm and the library's
        kernel produce the same partials on the same synthetic rows (bitwise); the
        bench built release and deployed to the mini beside the CLI (the binary and
        `Shrike_ShrikeAttnBench.bundle`).
- [ ] **S0.3 The ablation ladder (B7) on the mini, Shrike stopped.** The eight arms
      of the design document at 1k, 4k and 8k positions: as shipped; Q in registers;
      the eight-position block; double-buffered staging; the softmax removed; V
      removed; a pure load at the same layout; a pure load over the full row. The
      shipped arm calibrated against S0.1's `attn_layer_kv` slope (the bench's µs per
      position times ten layers against the rig's ms per 1,000). The table in the
      design document's step-zero record, the constraint named.
- [ ] **S0.4 The streaming prototype on the mini.** Approach A as a bench kernel,
      the sweep over heads per simdgroup (1, 2, 4, 8), positions per simdgroup per
      iteration (1, 2) and one or both KV heads per threadgroup, at 1k, 4k and 8k
      positions; each configuration against the shipped arm and the pure-load floor;
      the no-load twin of the best configuration (the loads replaced by constants) to
      say whether it is ALU-bound. The outputs checked against the shipped kernel's
      partials at a tolerance (a wrong tile shows as a delta orders above rounding).
      The table and the chosen configuration in the design document.
- [ ] **S0.5 The record and the ruling.** The design document's step-zero record
      complete (the ladder, the sweep, the repair priced from arms 2 and 4, the
      rewrite priced from the prototype, both graded); production restored on the
      mini and verified golden-identical; Davor's ruling recorded here: the repair
      first then the rewrite, the rewrite alone, or neither.

## Task 1: the instrument (class 1; beside step zero)

- [ ] **T1.1 The forced-token mode.** `GenerationConfig` (`Sampler.swift:18`) gains
      `forcedTokens: [Int32]?`; `RawCompletion.swift:275-296` takes the list's next
      id in place of the sampler's from the first generated position, through the
      synchronous logits path (`useBoundary` false and the fused greedy head off
      whenever the list is set), and stops when the list ends. A test with the
      `ScriptedLogitProducer` fixture
      (`ShrikeValidation/Support/Fixtures/ScriptedLogitProducer.swift`) asserts the
      fed ids are the list's, in order, whatever the scripted logits say.
- [ ] **T1.2 The logits dump.** `GenerationConfig` gains `logitsSink:
      LogitsSink?`, a protocol with `func record(position: Int, logits:
      UnsafeBufferPointer<Float>)`; the loop calls it after each position's logits
      are on the host (a synchronous read-back of the logits buffer in this mode
      only); a file sink writes raw fp32 rows to `<file>` and a JSON sidecar
      `<file>.json` (vocabulary size, position count, the tokens fed, the build's
      commit). A test with the scripted producer asserts the rows written equal the
      logits scripted.
- [ ] **T1.3 The CLI flags.** `Args.swift`: `--force-tokens <file>` (one id per
      line), `--dump-logits <file>`, `--logits-head`; `Run.swift:218` and `:247`
      pass `forceLogitsHead: !config.isPureGreedy || logitsHead || forcedTokens != nil`.
      The usage block updated; a parser test per flag.
- [ ] **T1.4 The comparison.** `tools/logit-compare.py <old> <new>`: per position
      the KL divergence old to new in fp64, max |Δ logit|, both argmaxes, the old
      build's top-2 margin; the band three times the run's max |Δ|; every flip
      listed with its margin and the verdict variance or defect; a summary line.
      Its own test: old against old reports zero everywhere (run on a real dump at
      T1.6 and kept as a `--self-test` mode over a synthetic pair).
- [ ] **T1.5 The logits-head golden.** `tools/golden-baseline.sh`: a `HEAD=logits`
      mode adding `--logits-head` to the CLI line and a `-lh` suffix to the profile
      name; the two profiles captured on both boxes on the current tree
      (`baselines/ornith15-int4-{short,long}-lh.{Mac167,mini}.txt`); `--check`
      covers all four from here on. The mini's capture through the archived
      `mini-golden.sh` pattern (Shrike stopped, deploy leave).
- [ ] **T1.6 The calibration run and the record.** On each box: both golden prompts
      forced with the golden's own tokens through the same build twice (old against
      old: zero), then the fused greedy head against the logits head at temperature
      zero (a known class-1 pair: agreement everywhere but exact ties); the tables in
      the design document's Task 1 record; the four gates; the golden identical on
      all four profiles on both boxes; the commit.

## Task 2: the class-1 repair (only if S0.5 rules it)

- [ ] **T2.1 The pre-registration.** The rows from S0.3's arms 2 and 4 (Q in
      registers, double-buffered staging) and the load width, the share of the 7k
      row each is expected to move, graded, in the design document before the
      kernel changes.
- [ ] **T2.2 The kernel.** `attention_decode_partial_shared` (`attention.metal:509`)
      with Q in per-lane registers, the staging double-buffered and the loads
      widened, in the order S0.3 priced; the chain, the `simd_sum` and the softmax
      untouched so the output is bitwise the shipped kernel's.
- [ ] **T2.3 The arm.** The bitwise arms of `AttentionTests` (`:279-357`, the V4.1
      pattern) extended: repaired against shipped at every shape class, raw output
      bytes equal.
- [ ] **T2.4 The gates, the golden, the arms.** The four gates; the golden identical
      on all four profiles on both boxes; deploy; the arms on four shapes against
      S0.1's ledger; the record in the design document; the commit.

## Task 3: the streaming scan (class 2)

- [ ] **T3.1 The pre-registration.** S0.4's chosen configuration, the expected rows
      (the design document's table re-graded on S0.4's measured rate) and the band
      the kernel arm will accept, in the design document before the kernel lands.
- [ ] **T3.2 The kernel.** `attention_decode_partial_stream` beside the shipped one
      in `attention.metal`, on the V4.1 function constants, the same buffers and the
      same partial contract; the threadgroup's simdgroups merged once at the chunk's
      end; `Attention.swift`: the variant selected by the shape gate
      (`:50-60`), the chunk budget kept at 64 unless S0.4 chose otherwise, the
      combine untouched.
- [ ] **T3.3 The kernel arm.** `AttentionTests` and `KVCacheQuantizedAttentionTests`
      extended to the new kernel: against the CPU reference at 1e-2 and the
      quantized cache at 0.02 at the small, the straddling and the served shapes at
      int8, int4 and fp16 rows; a new-against-shipped arm at the served shape
      recording max |Δ| against the pre-registered band; the PSO engagement test.
      Red first with the variant absent.
- [ ] **T3.4 The gates and the deploy.** The four gates; the golden NOT expected
      identical (recorded as the class-2 exception, with the diff kept); deploy to
      the mini.
- [ ] **T3.5 The instrument.** Both golden prompts forced through the shipped build
      and the new build on both boxes (`--force-tokens` with the golden's tokens,
      `--dump-logits`), `tools/logit-compare.py` on each pair; the tables in the
      design document's Task 3 record; every flip inside the band, or the kernel is
      fixed before anything else runs.
- [ ] **T3.6 The read.** The four golden profiles free-run on the new build on both
      boxes, the answers in the record, read by Davor for route and language; the
      ruling recorded here.
- [ ] **T3.7 The golden re-captured** on both boxes (four profiles), once, after the
      read; the commit carries the new baselines and the design document's note of
      the class-2 acceptance.
- [ ] **T3.8 The arms.** Four shapes, two lifetimes each, against S0.1's ledger; a
      same-box interleaved A/B on the 7k if the wall's move is inside the drift; the
      pre-registered rows moved or not in the record; the commit.

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
