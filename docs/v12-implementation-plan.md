# v12 implementation plan — prefill on the matrix path

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Status of record for [v12-prefill-matrix-kernels.md](v12-prefill-matrix-kernels.md).
Checkboxes here are the only status tracking. Commit SHAs in the verdicts are
the branch's current ones; every review fix is folded into its owning commit
(rebase and amend, never fixup commits), so the SHAs settle only once the
branch stops being rebased — the subjects are the stable names.

**Goal:** Bring ornith prefill from 7.3 ms/token (M4 Pro, 4k) and 28 ms/token
(M1 mini, 4k) to the design's target of 1.4 / 5.6 ms/token by moving the three
scalar prefill kernels (shared expert, attention core, routed experts) onto the
SIMD-group matrix path, in ledger order.

**Architecture:** Each step replaces one kernel family behind the same
shape/size gates the runner already uses, is qualified against the fp32
reference in `ShrikeValidation`, passes the five repo gates, then triggers a
golden-baseline recapture (deliberate numerics change) and a ledger re-measure
on both boxes. Decode, the KV layout, chunk size and the expert streamer are
untouched.

**Tech Stack:** Swift 6.3, Metal 4 (`mpp::tensor_ops::matmul2d` via the
runtime-compiled `tensorops` module), swift-testing, ShrikeBench, the
`SHRIKE_KERNEL_STATS` role timers.

**Spec:** [v12-prefill-matrix-kernels.md](v12-prefill-matrix-kernels.md)

## Global constraints

- macOS 26+, Swift 6.3+; never two model processes (`pgrep` check first);
  the mini's server on 8081 is production. The owner approved restarts and
  deploy actions on 2026-09-01; `tools/mini-deploy.sh` copies binaries +
  `*.bundle` directories by default and restarts the server only when passed
  `--restart`.
- Five gates per commit: release build with zero warnings; `swiftlint lint
  --strict --baseline .swiftlint-baseline.json`; markdown link check;
  `swift test --no-parallel`; the same under ThreadSanitizer with
  `env TSAN_OPTIONS=suppressions=tsan-suppressions.txt`.
- Numerics: tolerance against the fp32 reference (`maxAbs ≤ 2e-2`, `rel ≤ 2e-2`,
  the bar `PrefillAttentionTests` already uses); `tools/golden-baseline.sh
  --check` is *expected* to differ after each step, and both boxes' baselines
  are recaptured once per step with the before/after greedy digests recorded
  in that task's verdict line. Never recapture for an unexplained mismatch.
- Ledger protocol per step: `tools/prefill-measure.sh` at the 3.7k and 12k
  prompts on both boxes, GPU ms per prompt token by role, appended to the
  design's ledger table. One send per prompt per server lifetime: a repeated
  prompt hits the multi-prefix prompt cache and prefills only a suffix, and
  its wall time absorbs the previous request's settle. Re-measure on a fresh
  server (or a fresh-salt prompt), never by resending.
- Comments: none unless a genuinely non-obvious why (repo rule).

## Tasks

### Task 0: P0 — harness

- [x] **P0: harness** — the measurement kit the other tasks gate on. **LANDED
  49f57f4 (2026-09-02): both boxes print
  `prefill_projection_path=affine-threadgroup-f16` (M4 Pro 2026-09-01, M1
  2026-09-02). The log line is a standalone `ServerLog.residency` call right
  after session creation, not the runtime-identity array the step text named
  (that array is the prompt-cache domain digest and was left untouched).
  `tools/mini-deploy.sh` was added to the harness. The swiftlint baseline was
  regenerated for the moved file (20 entries before and after).**

  **Files:**
  - Modify: `sources/ShrikeBench/GEMMBench.swift` (already written, uncommitted),
    `sources/ShrikeBench/ShrikeBench.swift` (renamed from `main.swift`, gemm dispatch added)
  - Create: `tools/prefill-prompts.py`, `tools/prefill-ledger.py`, `tools/prefill-measure.sh`
  - Modify: `Sources/ShrikeServer/Core/ServerInference.swift:752` (the runtime line)

  **Interfaces:**
  - Produces: `ShrikeBench gemm [iterations]` printing
    `kernel=gemm_<label> m= k= n= launches= per_launch_ms= achieved_tflops=`;
    `tools/prefill-prompts.py <outdir> [model]` writing `prompt-{2k,6k,12k,2kb}.json`
    (3,756 / 12,285 / 25,245 / 4,305 tokens on ornith); `tools/prefill-ledger.py
    <server.log> [lastN] [prompt_tokens]` printing per-request role ms per prompt
    token; `tools/prefill-measure.sh <host> <port> <promptdir> <outdir> <tag>
    <label>...` (the `decode-measure.sh` twin: waits for `/v1/models`, posts each
    prompt once, prints wall + usage, sleeps `PAUSE` seconds between prompts).

  - [x] Step 1: `tools/prefill-prompts.py`, `tools/prefill-ledger.py`,
        `tools/prefill-measure.sh` written (2026-09-01, uncommitted), alongside
        the `gemm` bench mode and the `ShrikeBench.swift` rename (the
        `.swiftlint-baseline.json` was regenerated for the moved path: 20 entries
        before and after).
  - [x] Step 2: extend the runtime diag line at `ServerInference.swift:752`
        (the array that already carries `String(runtime.prefillChunkTokens)`)
        with `"prefill_projection_path=" + runtime.prefillProjectionPath`,
        where `RealForwardRunner` exposes

        ```swift
        public var prefillProjectionPath: String {
            prefillMPPAffineInt4 == nil ? "unavailable" : "affine-threadgroup-f16"
        }
        ```

        (`prefillMPPAffineInt4` is `RealForwardRunner.swift:176`; the plumbing
        from runner to `ServerInference.runtime` follows `prefillChunkTokens`
        at `:526`/`:842`.)
  - [x] Step 3: run `swift build -c release --product ShrikeServer` and
        `--product ShrikeBench`; expect zero warnings. `swiftlint lint --strict
        --baseline .swiftlint-baseline.json`; `python3 tools/check-md-links.py`.
  - [x] Step 4: launch a local server (`--port 8082 --ram-budget 20G --thinking
        off`, env `SHRIKE_KERNEL_STATS=1 SHRIKE_RUNNER_STATS=1`), confirm the
        runtime line prints `prefill_projection_path=affine-threadgroup-f16`,
        stop it. Deploy `ShrikeServer` + bundles to the mini only at the next
        agreed restart and read the same line there — this settles the design's
        open risk about the M1.
  - [x] Step 5: commit — `bench+tools: prefill ledger harness (v12 P0)`.
        Verdict line: the two boxes' projection paths.

### Task 1: P1 — shared expert on the matrix path

- [x] **P1: shared expert on the matrix path** — target 0.83 → ~0.06 ms/token
  (M4 Pro), 3.9 → ~0.3 (M1). **LANDED 1450c1c (2026-09-02): measured 0.83 →
  0.058 (M4 Pro) and 3.92 → 0.29 (M1) ms/prompt-token, on target. Wall:
  M4 Pro 3.7k 8.74 → 7.32 ms/tok; M1 3.7k 29.5 → 25.7, 12k 59.0 → 55.0.
  Chunk-vs-row-loop maxAbs 1.95e-3 (64 rows), 9.8e-4 (33 rows); the T-row
  scalar gate is bit-identical to the per-row GEMV at D=2048. Golden: M4 Pro
  identical on both profiles; M1 short identical, long diverged at the
  thinking block's second sentence (near-tie flip, both continuations
  coherent) → M1 long baseline recaptured. Two review rounds: doc block
  moved back onto `encodeRoutedMoEPrefill` (now 276 lines, was 325), gate
  on `MPPPrefillInt4QMM.isAvailable`, scalar-gate test at D=2048, `weightBits`
  retained by the type. M1 qualification is by construction (no toolchain
  there) plus the coherent golden continuation.**

  **Files:**
  - Modify: `Sources/Shrike/Kernels/Prefill/MoE/PrefillSharedExpert.swift`
  - Modify: `Sources/Shrike/Runtime/Prefill/PrefillChunkScratch.swift:104,121`
    (`sharedExpertScratchElements` becomes `chunkTokens * sharedIntermediate`)
  - Modify: `Sources/Shrike/Metal/MoE/moe.metal` (two new kernels),
    `Sources/Shrike/Kernels/Primitives/Elementwise.swift` (two encoders)
  - Modify: `Sources/Shrike/Runtime/Inference/RealForwardRunner.swift:4738-4790`
  - Test: `Tests/Shrike/Core/Kernels/Prefill/PrefillSharedExpertTests.swift`

  **Interfaces:**
  - Consumes: `MPPPrefillInt4QMM.encode(commandBuffer:weights:weightsOffset:
    scales:scalesOffset:biases:biasesOffset:x:y:m:n:k:) -> Path`
    (`MPPPrefillInt4QMM.swift:~80`), `SharedExpertProjection` (weights, scales,
    biases buffers + `scalesOffset`, `biasesOffset`, `rows`, `cols`),
    `Elementwise.encodeSigmoidScalarMul` (`Elementwise.swift:89`), the
    `silu_mul_fp16` kernel (`SharedExpertInt4.swift:46`).
  - Produces:

    ```swift
    extension PrefillSharedExpert {
        /// Whole-chunk form: three GEMMs + one silu_mul + one T-row scalar gate.
        /// Requires queryCount >= PrefillSharedExpert.matrixPathMinimumRows (32).
        func encodeChunk(commandBuffer: MTLCommandBuffer,
                         mpp: MPPPrefillInt4QMM,
                         x: MTLBuffer, y: MTLBuffer,
                         gate: SharedExpertProjection, up: SharedExpertProjection,
                         down: SharedExpertProjection,
                         scratchGate: MTLBuffer, scratchUp: MTLBuffer,   // T × F each
                         queryCount: Int, d: Int, intermediate: Int) throws
        static let matrixPathMinimumRows = 32
    }
    enum PrefillSharedExpertError: Error, Equatable { case chunkTooShort(Int) }
    extension Elementwise {
        /// y[row * d ..< row * d + d] *= sigmoid(gate[row]) for row in 0..<rows.
        func encodeSigmoidScalarMulRows(commandBuffer:, y:, gate:, rows: Int, d: Int) throws
        /// gate[row] = dot(dequant_int8(w), x[row]) + bias for row in 0..<rows.
        func encodeScalarGateRows(commandBuffer:, weights: TensorView, x:, gate:, rows: Int, d: Int) throws
    }
    ```

    Metal: `shared_scalar_gate_rows` (one threadgroup of 256 threads per row,
    `simd_sum` reduction, int8 affine dequant per group of 64 exactly as
    `dequant_int8_gemv_simd` does) and `sigmoid_scalar_mul_rows_fp16`
    (`dispatchThreads(w: d, h: rows)`), both in `moe.metal`.

  - [ ] Step 1: write the failing test in `PrefillSharedExpertTests`:

        ```swift
        @Test func chunkSharedExpertMatchesRowLoop() throws {
            var rng = SeedTree(0xC0FFEE).key("prefill-shared-expert-chunk")
            let ctx = try MetalContext()
            let prefill = try PrefillSharedExpert(context: ctx, weightBits: 4, siluActivation: true)
            let mpp = MPPPrefillInt4QMM(context: ctx, weightBits: 4)
            let rows = 64, d = 2048, f = 512          // production shape, ≥ matrixPathMinimumRows
            let x = Self.makeInputBlock(rows: rows, d: d, rng: &rng)
            let gate = Self.makeWeights(rows: f, cols: d, rng: &rng)
            let up = Self.makeWeights(rows: f, cols: d, rng: &rng)
            let down = Self.makeWeights(rows: d, cols: f, rng: &rng)
            // yLoop via encodeBlock (existing), yChunk via encodeChunk (new), same inputs
            // ...allocate with Fp16Buffer.make as the existing tests do...
            let maxAbs = RelError.maxAbsDiff(yChunk, yLoop)
            let rel = RelError.compute(actual: yChunk, reference: yLoop)
            #expect(maxAbs <= 2e-2, "chunk shared expert maxAbs=\(maxAbs) rel=\(rel)")
            #expect(rel <= 2e-2)
        }
        ```

        The file's `makeInputBlock(rng:)` uses the suite's static 4×128 shape;
        add `makeInputBlock(rows:d:rng:)` beside it (`makeWeights(rows:cols:rng:)`
        already takes its shape). Plus `chunkSharedExpertRejectsShortChunks`:
        `queryCount: 8` must throw `PrefillSharedExpertError.chunkTooShort(8)`.
  - [ ] Step 2: `swift test --no-parallel --filter PrefillSharedExpertTests` —
        expect FAIL: `encodeChunk` undefined.
  - [ ] Step 3: implement `encodeChunk`: `mpp.encode(... x: x, y: scratchGate,
        m: queryCount, n: intermediate, k: d)` with `gate`'s buffers/offsets;
        same into `scratchUp` with `up`; `silu_mul_fp16` over
        `queryCount * intermediate` elements (gate := silu(gate) * up, in place);
        `mpp.encode(... x: scratchGate, y: y, m: queryCount, n: d, k: intermediate)`
        with `down`. Throw `chunkTooShort` below 32 rows. Add the two Metal
        kernels and their `Elementwise` encoders.
  - [ ] Step 4: grow the scratch: `sharedExpertScratchElements = chunkTokens *
        sharedIntermediate` (and the `3 *` multiplier at `:121` stays: gate, up,
        act). Update the call site at `RealForwardRunner.swift:4740`: if
        `t >= PrefillSharedExpert.matrixPathMinimumRows`, `prefillMPPAffineInt4`
        is non-nil and `model.sharedExpertWeightBits == 4` (the value the
        runner already passes to `PrefillSharedExpert` at `:660`), call `encodeChunk`
        then, when `cfg.sharedExpertGated`, `encodeScalarGateRows` +
        `encodeSigmoidScalarMulRows` once each; else the existing row loop.
  - [ ] Step 5: `swift test --no-parallel --filter PrefillSharedExpertTests` —
        PASS; then the five gates.
  - [ ] Step 6: `tools/golden-baseline.sh --check` — record the mismatch (it is
        the expected numerics change); recapture on this box; commit —
        `prefill: shared expert as three chunk GEMMs (v12 P1)`.
  - [ ] Step 7: ledger on this box (3.7k, 12k); deploy + mini ledger at the next
        agreed restart; mini baseline recapture. Verdict line: role ms/token
        before → after on both boxes, digests before/after.

### Task 2: P2 — attention core on the matrix path

- [x] **P2: attention core on the matrix path** — target 3.39 → ~0.2 ms/token
  at 3.7k, 22.2 → ~0.8 at 25k (M4 Pro). **LANDED 066fe67 (2026-09-02):
  measured 3.40 → 0.27 (3.7k), 13.53 → 0.47 (12k), 22.19 → 0.85 (25k) on
  the M4 Pro; 13.82 → 1.36 (3.7k), 43.58 → 2.40 (12k) on the M1. Wall: M4 Pro
  12k 207.8 → 46.1 s, 25k 666.8 → 114.0 s; M1 12k 675.9 → 166.0 s (chapter
  start 725.2). Design refinement by ruling: a per-layer fp16 shadow of the
  cache (`attention_prefill_kv_dequant`, `prefill_load_kv`'s formula) feeds
  `matmul2d` device tensors instead of per-tile threadgroup dequant (a 64×256
  fp16 tile alone is the 32 KB budget). Spike on the ledger, not a bench:
  r32s4 serial softmax 6.15 s → lane-parallel 5.39 s → r64s8 5.91 s at 12k;
  r32s4 shipped, ≈ 36 % of the M4 Pro ceiling (bar was 40 %; accepted, the
  eightfold GQA tile re-read is the documented follow-on). Tests: 3 fp32
  reference cases (fp16 cache), 5 matrix-vs-tiled cases on int8/int4
  `KVCacheManager` views, gate cases, rejected-shape end-to-end. Golden: both
  boxes flipped the long profile (near-tie), short identical → the M1's long
  baseline recaptured at P2; the M4 Pro's was recaptured once at P3 (covering
  P2 + P3). Review approved; fix round: shadow grows in 8 MiB
  quanta, header comment corrected, self-contained includes, tile
  `static_assert`s, gate threshold pinned at 32, nonzero test offsets.
  `RuntimeConfiguration` default is now `.causalMatrix`;
  `SHRIKE_PREFILL_ATTENTION=tiled|matrix` A/B. Fix wave (2026-09-02):
  `.causalMatrix` falls through to the tensor-ops shape test when
  `matrixPathAccepts` rejects it, so the 512/16/2 fp16-KV kernel selection is
  unchanged under the new default. Memory check on the mini (P3 build, 8G
  budget, 32k max context, 25,245-token prompt): `memory_pressure -Q` free
  83 % → 25 % across the request (the expert pool becoming resident dominates;
  the attention shadow at 25k is ~50 MB), server RSS 10.7 GB, no pressure
  warning, wall 320 s.**

  **Files:**
  - Create: `Sources/Shrike/Metal/Prefill/attention_matrix.metal` (new module:
    append `"attention_matrix"` to `shaderModules` at `MetalContext.swift:86`
    and `"attention_matrix": "Metal/Prefill"` to `shaderSubdirectories` at
    `:104`)
  - Modify: `Sources/Shrike/Kernels/Attention/PrefillAttention.swift:112-135`
    (selection gate), `:160-175` (dispatch)
  - Modify: `sources/ShrikeBench/` — add `AttentionBench.swift` (`attn` mode)
  - Test: `Tests/Shrike/Core/Kernels/Attention/PrefillAttentionTests.swift`

  **Interfaces:**
  - Consumes: `PrefillAttentionParams` unchanged (`PrefillAttention.swift:4`);
    the int8 cache layout and `prefill_load_kv` / `prefill_kv_slot` helpers in
    `prefill.metal`; `PrefillAttentionRef.apply(_:) -> [Float]` and
    `PrefillAttentionRef.Inputs` (q, k, v, strides, headDim, qHeads, kvHeads,
    start, chunk, kvValid, window, scale, sinks).
  - Produces: kernel `attention_prefill_causal_matrix` with the same buffer
    binding as `attention_prefill_causal_tiled` (Q 0, K 1, V 2, O 3, params 4,
    sinks 5 unused); function constants `FC_ATTN_ROWS` (queries per
    threadgroup), `FC_ATTN_KEYS` (key tile), `FC_ATTN_KV_BITS`; a new
    `RuntimePrefillAttentionPath.causalMatrix` selected when `headDim == 256 &&
    numQHeads == 16 && numKVHeads == 2 && kvRingCapacity == 0 && !hasSinks &&
    (slidingWindow == 0 || slidingWindow >= kvValidCount) && queryCount >= 32
    && kvValidCount <= 65_536` (the runner passes `kvValidCount` as the window
    for full layers, so that clause is full-visibility, not literally "no
    sliding window"; kvBits is not gated at all — 4, 8, and 16 all reach the
    matrix path, since `prefill_load_kv` handles 4-bit and the int4 case is
    tested); everything else keeps `attention_prefill_causal_tiled`.

  - [ ] Step 1 (spike, throwaway numbers, kept harness): add `ShrikeBench attn
        <variant> [iterations]` that builds synthetic Q (T × 4096 fp16), an int8
        K/V cache with `kvValid = start + T`, and dispatches either
        `attention_prefill_causal_tiled` or the new kernel at
        `(start, T) ∈ {(0, 3756), (8192, 4096)}`; prints per-launch ms and
        TFLOPS = `4 · 16 · 256 · pairs / seconds`. Record the scalar kernel's
        numbers first (expected ≈ 1 % of the gemm ceiling).
  - [ ] Step 2 (spike): write the kernel with rows-per-threadgroup and key-tile
        as function constants and try the candidates `(rows, keys) ∈ {(16, 32),
        (32, 32), (32, 64), (64, 32)}` for one KV-head group (all 8 query heads
        of `rows` queries = `8 · rows` matmul rows). Structure, in order per key
        tile: cooperative load of the K tile (`keys × 256`) and V tile from the
        int8 cache into threadgroup fp16 with the group-64 scale/bias dequant
        (`kvBits == 16` copies); `matmul2d` S = Q·Kᵀ with Q read as a device
        tensor (per head, rows strided by `qTokenStrideElements`); on the
        diagonal tile mask `key > query` to `-INFINITY`; per-row running max and
        sum (fp32) with the rescale of the O accumulator; `matmul2d` O += P·V;
        after the last tile divide by the row sum and store fp16. Accept the
        first candidate that reaches ≥ 40 % of the gemm ceiling at both shapes;
        if none does within 32 KB of threadgroup memory, fall back to
        `simdgroup_matrix` fragments (the design's risk item) before proceeding.
        Verdict line: the table of candidates.
  - [ ] Step 3: write the failing tests (production dims, the shapes the gate
        accepts):

        ```swift
        @Test(arguments: [
            (label: "single-chunk-64",  start: 0,    chunk: 64,   kvBits: 8),
            (label: "single-chunk-4096", start: 0,   chunk: 4096, kvBits: 8),
            (label: "second-chunk-history", start: 4096, chunk: 4096, kvBits: 8),
            (label: "ragged-tail", start: 8192, chunk: 209, kvBits: 8),
            (label: "fp16-cache", start: 4096, chunk: 256, kvBits: 16),
        ]) func matrixPrefillAttentionMatchesReference(c: (label: String, start: Int, chunk: Int, kvBits: Int)) throws {
            let fixture = Self.makeFixture(start: c.start, chunk: c.chunk, window: 0,
                                           seed: 0xB120, headDim: 256, qHeads: 16, kvHeads: 2)
            let reference = PrefillAttentionRef.apply(fixture)
            let candidate = try Self.runKernel(fixture, path: .causalMatrix, kvBits: c.kvBits)
            let maxAbs = RelError.maxAbsDiff(candidate, reference)
            let rel = RelError.compute(actual: candidate, reference: reference)
            #expect(maxAbs <= 2e-2, "\(c.label) maxAbs=\(maxAbs) rel=\(rel)")
            #expect(rel <= 2e-2, "\(c.label) rel=\(rel)")
        }
        @Test func matrixPathGateFallsBackOffShape() throws {
            // For each of: headDim 512; sinks present; window 1024; queryCount 8 —
            // PrefillAttention.selectPath(params:sinks:kvRingCapacity:) == .causalTiled
        }
        ```

        `makeFixture(start:chunk:window:seed:headDim:qHeads:kvHeads:)` and
        `runKernel(_:kvRingCapacity:path:)` (`PrefillAttentionTests.swift:199,
        255`) exist; add a `kvBits: Int = 16` parameter to `runKernel` that,
        when 8, quantizes the fixture's K/V buffers through
        `KVCacheQuantizer.encode(commandBuffer:…)` (`KVCacheQuantizer.swift:13`,
        the runner's own path) and passes `kvBits`/`kvTokenStrideBytes`/
        `kvValueBytes` in the params exactly as `RealForwardRunner` does at the
        `attention_prefill_causal_tiled` dispatch (`:4221`). Expose the selection
        as `PrefillAttention.selectPath(params:sinks:kvRingCapacity:) ->
        RuntimePrefillAttentionPath` so the fallback test needs no GPU run.
  - [ ] Step 4: `swift test --no-parallel --filter PrefillAttentionTests` —
        FAIL: `.causalMatrix` undefined.
  - [ ] Step 5: land the kernel with the spike's constants, the selection gate,
        and the dispatch (`threadgroups = (ceil(queryCount / rows), numKVHeads)`,
        `threadsPerThreadgroup = 128`). Tests PASS; five gates.
  - [ ] Step 6: `tools/golden-baseline.sh --check` (expected to differ),
        recapture, commit — `prefill: matrix-path causal attention for the
        256/16/2 shape (v12 P2)`.
  - [ ] Step 7: ledger on both boxes (3.7k, 12k, and 25k here); mini recapture.
        Verdict line with the attention role before → after and the achieved
        share of the ceiling.

### Task 3: P3 — routed experts as per-expert GEMMs

- [x] **P3: routed experts as per-expert GEMMs** — target 2.03 → ~0.6 ms/token
  (M4 Pro). **LANDED 9323fa8 (2026-09-02): measured 2.03 → 1.00 ms/prompt-token
  (M4 Pro, 12k; 1.04 at 3.7k, 1.21 at 25k) and 5.82 → 3.35 (M1, 12k) — half
  the target's cut, at ≈ 35 % of the 128-row expert-shape ceiling. Wall: M4 Pro
  12k 46.1 → 34.3 s, 25k 114.0 → 92.1 s; M1 12k 166.0 → 135.6 s. GEMM vs scalar
  maxAbs 7.6e-6, rel 6e-4. Golden: both boxes' long profiles flipped (near-tie)
  → M1 long recaptured; M4 Pro long + short(header) recaptured once for P2 + P3.
  Gates all green incl. TSAN (1121 tests; 1123 after the fix round). Review
  approved; fix round: staging scratch gated on one layout predicate shared
  with the runtime branch (dense layers and the 32-token MTP draft chunk no
  longer allocate it; the T=32 scratch total is back at its pre-P3 bytes),
  `k % 64` guard, throwing bounds check on tile ranges, below-threshold
  fallback test. Expert sub-tensor `setBuffer(offset:)` at 2 KiB multiples
  proven by the golden runs. Decision point: M4 Pro 3.7k GPU busy
  2.42 ms/tok > 2.0 → P4 scheduled by the rule.**

  **Files:**
  - Modify: `Sources/Shrike/Kernels/Prefill/MoE/PrefillGroupedRoutedMoE.swift`
    (new `encodeExpertGEMMs`), `PrefillMoEGrouping.swift:171-183` (expose
    per-expert `[pairStart, pairCount)` ranges in `sortedPairs` order),
    `Sources/Shrike/Metal/Prefill/prefill.metal` (gather + scatter kernels),
    `Sources/Shrike/Runtime/Prefill/PrefillChunkScratch.swift:105-106`
    (staging `maxRowsPerExpert × 2048` and `× 512 × 2`),
    `RealForwardRunner.swift:4823-4944` (tile loop)
  - Test: `Tests/Shrike/Core/Kernels/Prefill/PrefillGroupedRoutedMoETests+Execution.swift`

  **Interfaces:**
  - Consumes: the packed-expert offsets (`gateW/gateS/gateB/upW/…/downB`, the
    `ExpertOffsets` mirror in `ShrikeBench.swift:24`), `sortedPairs` (token,
    expert, weight-slot) and `routePartials` (per-pair `D`-wide rows summed by
    `prefill_moe_reduce_token_major`), `MPPPrefillInt4QMM.encode`.
  - Produces:

    ```swift
    extension PrefillGroupedRoutedMoE {
        /// Per expert in the tile: gather its pair rows, gate/up GEMMs, silu_mul,
        /// down GEMM, scatter to routePartials. Experts with fewer than
        /// `minimumRows` pairs are returned untouched for the scalar path.
        func encodeExpertGEMMs(commandBuffer:, mpp: MPPPrefillInt4QMM,
                               hidden:, sortedPairs:, routePartials:,
                               binding: PrefillStreamedTileBinding,
                               ranges: [PrefillExpertPairRange],   // (expertIndex, pairStart, pairCount)
                               staging: PrefillExpertStaging,     // rows×2048, rows×512, rows×512
                               minimumRows: Int) throws -> [PrefillExpertPairRange]  // the leftovers
        static let matrixPathMinimumRows = 32
    }
    ```

    Metal: `prefill_routed_gather_rows` (`dispatchThreads(w: 2048, h: rows)`:
    `staging[r] = hidden[sortedPairs[pairStart + r].token]`) and
    `prefill_routed_scatter_rows` (`routePartials[pairStart + r] = down[r]`,
    same shape; the router weight is applied in the existing reduce).

  - [ ] Step 1: failing test in `+Execution`: one synthetic tile of 3 experts
        with 40 / 32 / 5 pairs; run the scalar path and the GEMM path on the
        same inputs; expect `RelError` within `2e-2` on `routePartials` for the
        first two experts and the third returned as leftover.
  - [ ] Step 2: run it — FAIL: `encodeExpertGEMMs` undefined.
  - [ ] Step 3: implement gather → `mpp.encode` (gate: `n: 512, k: 2048`, weights
        at the expert view's `gateW/gateS/gateB`; up likewise into the second
        staging block) → `silu_mul_fp16` over `rows × 512` → `mpp.encode` (down:
        `n: 2048, k: 512`) → scatter. Leftovers go through
        `encodeStreamedBatched` unchanged.
  - [ ] Step 4: wire the tile loop: after `makeStreamedMetadataBuffers`, call
        `encodeExpertGEMMs` when `prefillMPPAffineInt4 != nil`, then the scalar
        path for the leftovers. Tests PASS; five gates.
  - [ ] Step 5: golden check (differs), recapture, commit — `prefill: per-expert
        GEMMs on the matrix path (v12 P3)`.
  - [ ] Step 6: ledger both boxes; mini recapture. Verdict line. **Decision
        point:** if the M4 Pro 3.7k ledger is ≤ 2.0 ms/token, P4 is optional.

### Task 4: P4 — GDN chunked scan

- [x] **P4: GDN chunked scan** — scheduled by the P3 decision rule (M4 Pro 3.7k
  GPU busy 2.42 ms/tok > 2.0). Target `prefill_gdn_router` 1.00 → ≈ 0.45
  ms/prompt-token (M4 Pro, 12k). **LANDED 9b9374d (2026-09-02): measured
  `prefill_gdn_router` 1.01 → 0.65 ms/prompt-token (M4 Pro 3.7k; 0.99 → 0.64
  at 12k, 1.16 → 0.70 at 25k) and 4.39 → 3.44 (M1 3.7k; 4.37 → 3.43 at 12k).
  The scan itself: 51.7 → 9.5 ms per 4,096-row layer call on the M4 Pro
  (`ShrikeBench gdn_scan`, 5.4×, 2.27 TFLOPS ≈ 30 % of the ceiling). The 0.45
  target was not reached because the serial scan was ≈ 0.38 of the role, not
  the 0.6 the estimate assumed; the remaining ≈ 0.57 ms/token is the role's
  projections, conv and norms, already on the matrix path — the role is now
  GEMM-bound. Same-binary A/B at 3.7k (M4 Pro, two pairs): GDN 0.65 / 0.69
  chunked against 0.96 / 0.99 serial, GPU busy 2.11 / 2.18 against 2.37 /
  2.41, walls 12.98 / 11.51 against 12.49 / 12.52 s (the first pair's
  inversion was 1.5 s of tile-boundary gap noise). Wall: M4 Pro 12k 34.3 →
  32.6 s, 25k 92.1 → 81.2 s; M1 3.7k 40.6 → 39.1 s, 12k 135.6 → 126.3 s.
  Untouched roles moved < 2 % on the M1 and ≤ 8 % on the M4 Pro (its
  documented drift). Numerics: chunked against the serial kernel at the
  ornith shape maxAbs 3.1e-5 (rel 5.8e-4), state rel 2.2e-5; against
  `GDNReference` at T = 200 through the full chain within the 2e-2 bar; the
  CPU chunk model against the serial reference within 1e-3. Golden: M4 Pro
  short identical (its prompt is under 64 rows → serial kernel), long flipped
  at a near-tie in the thinking block with a coherent continuation →
  recaptured; M1 short and long identical (no tie crossed) → not recaptured.
  Gates all green incl. TSAN (1129 tests, 0 reports). Review (opus) approved
  with two Important findings — the availability probe did not check
  `maxTotalThreadsPerThreadgroup ≥ 128` before the fixed 128-thread
  dispatches, and no test could fail without the pad-row blit — plus minors;
  two fix rounds folded into the commit (probe guard; a NaN sentinel in the
  test's pad rows plus explicit finiteness assertions, because the kernel
  masks every pad-row write and only `0 × NaN` in the state contraction can
  leak, which `RelError`'s `max`-based helpers swallow — with the blit
  disabled the T = 200 case now fails on the state assertion; the class doc
  comment moved off the inserted enum, `usesGDNChunkedScan` removed as
  unread, the parameter renamed `prefillChunkTokens`, the bench's serial
  FLOP count 3 → 4 passes); scoped re-review clean. Deferred
  minors, ledgered: the path description reads `chunked` when a sub-64-row
  chunk configuration runs serial; the env knob has no typo guard; the
  factors scratch is allocated under the serial A/B knob.** Scope: the scalar
  per-head decay shape only (ornith: `in_proj_a` has `Hv = 32` rows). The
  per-channel variant (`gdn_delta_step_prefill_vec`, Kimi KDA), any shape
  other than `Dk = Dv = 128`, chunks under 64 rows and the 32-token MTP draft
  chunk keep the serial kernel.

  **Math.** Per value head `h` (KV head `hk = h / (Hv/Hk)`), `S ∈ ℝ^{Dv×Dk}` is
  the fp32 state stored row-major as `state[h][dv][dk]`; the row vectors
  `q_t, k_t ∈ ℝ^{Dk}`, `v_t ∈ ℝ^{Dv}` are the normed conv rows the serial
  kernel reads. The serial rule `GDNReference.step` implements, per token:

  ```
  α_t = exp(−exp(A_log[h]) · softplus(a_t + dt_bias[h]))     β_t = σ(b_t)
  u_t = β_t (v_t − α_t S_{t−1} k_t)      S_t = α_t S_{t−1} + u_t k_tᵀ      o_t = S_t q_t
  ```

  Over a chunk of `C = 64` rows with incoming state `S_0`, cumulative
  log-decay `ℓ_t = Σ_{j≤t} log α_j` and `γ_t = exp ℓ_t`, unrolling gives
  `S_t = γ_t S_0 + Σ_{i≤t} exp(ℓ_t − ℓ_i) u_i k_iᵀ`; substituting into `u_t`
  and stacking rows (`K, Q ∈ ℝ^{C×Dk}`, `V, U, O ∈ ℝ^{C×Dv}`):

  ```
  (I + A) U = diag(β) (V − diag(γ) K S_0ᵀ)     A[t,i] = β_t exp(ℓ_t − ℓ_i) k_t·k_i   (i < t, else 0)
  O = diag(γ) Q S_0ᵀ + M U                     M[t,i] = exp(ℓ_t − ℓ_i) q_t·k_i       (i ≤ t, else 0)
  S_C = γ_{C−1} S_0 + (diag(λ) U)ᵀ K           λ_i = exp(ℓ_{C−1} − ℓ_i)
  ```

  `I + A` is unit lower triangular, so `T⁻¹ = (I + A)⁻¹` comes from forward
  substitution and `U = T⁻¹ diag(β)(V − diag(γ) K S_0ᵀ)`. Every product except
  `T⁻¹` and `M` is independent per `dv` column, which is what lets the
  sequential pass run on `Dv/32` column blocks per head. The speculative
  checkpoint (state after the chunk's first row) is `α_0 S_0 + u_0 k_0ᵀ` with
  `u_0 = U[0,:]`. A partial last chunk pads rows `t ≥ rows` with `α = 1`,
  `β = 0` and zeroed `q, k, v` (a blit fill of the `conv_out` rows
  `[rows, ⌈rows/64⌉·64)` before the scan), so the pads add nothing and
  `γ_{C−1}` is the last real token's decay. `ℓ_t ≤ ℓ_i` for `i ≤ t`, so every
  ratio `exp(ℓ_t − ℓ_i) ≤ 1` and nothing overflows; `γ_t` underflowing to 0 is
  the same fully-forgotten state the serial product of `g` reaches.

  **Kernels** — new module `gdn_chunked`
  (`Sources/Shrike/Metal/GDN/gdn_chunked.metal`), listed after `gdn` in
  `MetalContext.shaderModules` because it uses `gdn_softplus`, guarded by
  `__HAVE_TENSOR__` like `attention_matrix.metal`, both 128 threads
  (`execution_simdgroups<4>`):

  1. `gdn_chunk_factors`, grid `(⌈rows/64⌉, Hv)`. Per (chunk, head): `ℓ` and
     `β` for the 64 rows (pads: `log α = 0`, `β = 0`); `K Kᵀ` and `Q Kᵀ` from
     one `matmul2d_descriptor(64, 64, 128, false, true)` op over device
     tensors on `conv_out` (row stride `C`); `A` into a 16 KB fp32 threadgroup
     tile; `T⁻¹` by forward substitution, one thread per column
     (`x_j = 1; x_t = −Σ_{i=j}^{t−1} A[t,i] x_i`), stored as fp16; `M` stored
     as fp16; `β, γ, λ` and `γ_{C−1}` as fp32. Output: the *factors* scratch,
     `17,408` bytes per (head, chunk), head-major (`((h · chunkCount) + c) ·
     17,408`): `tinv: half[64·64]` @0, `m: half[64·64]` @8192, `beta, gamma,
     lambda: float[64]` each from @16384, `gammaChunk: float` @17152, padded
     to a 256-byte multiple. fp16 factors are the precision
     flash-linear-attention ships (bf16 there); the products accumulate fp32.
  2. `gdn_chunk_scan`, grid `(Dv/32, Hv)`. One threadgroup owns a 32-column
     block of one head's state for the whole chunk sequence. Threadgroup
     memory: `s_tile: float[32·128]` (16 KB, the state block), `u_tile:
     float[64·32]` (8 KB). Per chunk, in this order: (1) `P = K S_blkᵀ`
     (`matmul2d_descriptor(64, 32, 128, false, true)`, left device half, right
     threadgroup float), `u_tile[t,dv] = t < valid ? β_t (v[t,dv] − γ_t P[t,dv])
     : 0`; (2) `U = T⁻¹ X` (`matmul2d_descriptor(64, 32, 64, false, false)`,
     left device half `tinv`, right `u_tile`), written back over `u_tile`
     after a barrier; (3) chunk 0 with `checkpointEnabled`: `checkpointState
     = γ_0 s_tile + u_tile[0,·] ⊗ k_0`; (4) `O₁ = Q S_blkᵀ` (the op from (1))
     and `O₂ = M U` (the op from (2)) into two cooperative tensors; (5)
     `u_tile[t,·] *= λ_t`; (6) `S_blk = γ_{C−1} S_blk + Uλᵀ K`
     (`matmul2d_descriptor(32, 128, 64, true, false)`, left `u_tile`
     transposed, right device half `K`), read-modify-write per element into
     `s_tile`; (7) `u_tile = γ_t O₁ + O₂` across two barriers, then `y[t] =
     half(u_tile[t])` for `t < valid`. After the last chunk `s_tile` is written
     back to `state`. Cooperative tensors of different ops are never combined
     by element index — everything crosses through `u_tile` / `s_tile`.

  **Files:**
  - Create: `Sources/Shrike/Metal/GDN/gdn_chunked.metal` (both kernels, a
    `GDNChunkParams` struct);
    `Sources/ShrikeValidation/Support/Reference/GDN/GDNChunkedReference.swift`
    (fp32 CPU model of the chunk math — the oracle for the kernel's
    intermediates); `Tests/Shrike/Core/Kernels/GDN/GDNChunkedScanTests.swift`.
  - Modify: `Sources/ShrikeValidation/Support/Reference/GDN/GDNReference.swift`
    (split `step` into `normalize`, `deltaRule`, `gatedNorm`; `step` composes
    them, behaviour unchanged); `Sources/Shrike/Kernels/GDN/GDN.swift` (chunked
    pipelines, `encodeDeltaStepPrefillChunked`, the shape predicate, scratch
    sizing); `Sources/Shrike/Infrastructure/Metal/MetalContext.swift:88-123`
    (register the module);
    `Sources/Shrike/Runtime/Prefill/PrefillChunkScratch.swift`
    (`usesGDNChunkedScan`, `gdnChunkFactorBytes`, the buffer);
    `Sources/Shrike/Runtime/Inference/RealForwardRunner.swift` (~185-195 the
    path description, ~361 the env knob, ~3866 the delta-step call — extracted
    into a private helper so the baselined layer function does not grow);
    `Sources/ShrikeServer/Core/ServerInference.swift:527-528, 816-817, 848-849`
    (`prefill_gdn_scan=` on the residency line);
    `Sources/ShrikeBench/ShrikeBench.swift` (`gdn_scan` mode);
    `docs/v12-prefill-matrix-kernels.md` (Step 4, ledger rows, the GDN row of
    "Where the time goes").

  **Interfaces:**
  - Consumes: `GDN.encodeDeltaStepPrefill`'s buffer contract (`conv_out
    [T, C]` half with normed q/k and raw v; `a_proj`/`b_proj [T, Hv]` half;
    `A_log`, `dt_bias [Hv]` bfloat; `state [Hv, Dv, Dk]` fp32; `y [T, Hv·Dv]`
    half; optional `checkpointState`), `MetalContext.pipeline`,
    `GDNReference`, `RelError`, `Fp16Buffer`, `SeedTree`.
  - Produces:

    ```swift
    extension GDN {
        static let chunkTokens = 64
        static let chunkedScanHeadDim = 128
        static let chunkedScanValueBlock = 32
        static let chunkFactorsBytesPerChunk = 17_408
        /// The shape the chunked kernels are compiled for: scalar per-head decay, Dk = Dv = 128.
        static func chunkedScanSupports(config: LinearAttentionConfig, perChannelDecay: Bool) -> Bool
        static func chunkFactorsBytes(config: LinearAttentionConfig, prefillChunkTokens: Int) -> Int  // Hv · ⌈prefillChunkTokens/64⌉ · 17_408
        var chunkedScanAvailable: Bool             // shape supported, both pipelines compiled, 128 threads per threadgroup allowed
        var chunkedScanUnavailableReason: String?  // nil when available
        /// Same contract as `encodeDeltaStepPrefill` plus the factors scratch. Throws
        /// `GDNChunkedScanError.tooFewRows` below 64 rows, `.unavailable(reason)`,
        /// `.factorsTooSmall(needed:have:)`, `.convOutTooSmall(needed:have:)` when
        /// `convOut` does not cover the 64-row multiple the pad blit writes.
        func encodeDeltaStepPrefillChunked(commandBuffer:, convOut:, convOutOffset:,
                                           aProj:, aProjOffset:, bProj:, bProjOffset:,
                                           aLog:, aLogOffset:, dtBias:, dtBiasOffset:,
                                           state:, checkpointState:, y:, yOffset:,
                                           rows: Int, factors: MTLBuffer) throws
    }
    enum GDNChunkedScanError: Error, Equatable {
        case tooFewRows(Int), unavailable(String)
        case factorsTooSmall(needed: Int, have: Int), convOutTooSmall(needed: Int, have: Int)
    }

    public struct GDNChunkedReference {          // ShrikeValidation
        public init(cfg: LinearAttentionConfig, aLog: [Float], dtBias: [Float])
        /// `normed`: [T][C] rows as `GDNReference.normalize` returns them; `state` is
        /// consumed and replaced. Returns fp32 y rows [T][Hv·Dv] (unrounded) and the
        /// state after row 0.
        public func run(normed: [[Float]], a: [[Float]], b: [[Float]],
                        state: inout [Float]) -> (y: [[Float]], checkpoint: [Float])
    }
    extension GDNReference {
        public mutating func normalize(qkvRaw: [Float]) -> [Float]   // conv + SiLU + tail carry + q/k norm
        public mutating func deltaRule(normed: [Float], a: [Float], b: [Float]) -> [Float]  // one token; fp16-rounded y
        public func gatedNorm(y: [Float], z: [Float]) -> [Float]
    }
    ```

    Runner: `SHRIKE_GDN_PREFILL_SCAN=serial|chunked` (default: chunked when
    available); `RealForwardRunner.prefillGDNScanPathDescription` ∈
    {`chunked`, `serial`, `serial(unavailable: …)`}, logged as
    `prefill_gdn_scan=` on the server's residency line beside
    `prefill_attention_path=`. Scratch:
    `PrefillChunkScratchLayout.gdnChunkFactorBytes`, non-zero only when
    `gdnQKVDim > 0 && !perChannel && Dk == Dv == 128 && chunkTokens >= 64`
    (the 32-token draft chunk allocates nothing), counted in
    `devicePrivateBytes` (34 MB at 4,096-token chunks for ornith);
    `PrefillChunkScratchBuffers.gdnChunkFactors: MTLBuffer?` — nil is the
    runner's "serial" signal. Bench:
    `ShrikeBench gdn_scan [iterations]` — ornith shape, `T = 4096`, serial vs
    chunked ms per call and µs per token, chunked TFLOPS against the `gemm`
    ceiling.

  Steps (TDD; the CPU model first so the math is verified before any Metal):

  - [x] Step 1: `GDNReference` split — `normalize`, `deltaRule`, `gatedNorm`;
        `step` = the three in sequence. `swift test --no-parallel --filter
        GDNKernelTests` → still green.
  - [x] Step 2: failing test `GDNChunkedScanTests.chunkedReferenceMatchesSerialReference`:
        cfg `(Hk 1, Hv 2, Dk 128, Dv 128, conv 4)`; seeded normed rows (q/k unit
        vectors scaled like the kernel's norm, then fp16-rounded; v fp16 in
        [−1, 1]), `a, b` in [−1, 1], `A_log` in [−1, 1.5], `dt_bias` in
        [−0.5, 0.5], a random non-zero incoming state in [−0.5, 0.5]; T ∈ {64,
        200}; serial = `deltaRule` per row on a copy of the state; expect
        `RelError.maxAbsDiff(y) ≤ 1e-3`, state `≤ 1e-3`, checkpoint == serial
        state after row 0 within 1e-5. FAIL: `GDNChunkedReference` undefined.
  - [x] Step 3: implement `GDNChunkedReference.run` exactly as the math block
        (fp32, `T⁻¹` by forward substitution). PASS.
  - [x] Step 4: failing tests: `chunkedKernelMatchesSerialKernel` — same
        fixture at T ∈ {64, 200, 4096}: `conv_out` (normed rows as half),
        `a_proj`/`b_proj` half, bfloat `A_log`/`dt_bias`, two copies of the
        state; serial `encodeDeltaStepPrefill` with a checkpoint buffer vs
        `encodeDeltaStepPrefillChunked` with its own; expect y `maxAbs ≤ 2e-2`
        and `RelError.compute ≤ 2e-2`, state `RelError.compute ≤ 2e-2`,
        checkpoint `maxAbs ≤ 2e-2`. `chunkedKernelMatchesReference` — T = 200
        through the real chain (conv → tail → qk norm → chunked scan → gated
        norm) against `GDNReference.step`, the existing prefill test's
        tolerance `max(2e-2, |want|·4e-2)`. `chunkedScanShapeGate` —
        `chunkedScanSupports` false for per-channel decay, Dk 32, Dv 64;
        `encodeDeltaStepPrefillChunked(rows: 63)` throws `.tooFewRows`.
        FAIL: undefined.
  - [x] Step 5: kernels + `GDN.swift` + `MetalContext` registration → the
        tests PASS. If the compiler rejects the `(32, 128, 64, true, false)`
        descriptor (the one shape without precedent in the tree), use `(64,
        128, 64, true, false)` over a 64-row view of `u_tile` and keep the
        first 32 destination rows.
  - [x] Step 6: bench `swift run -c release ShrikeBench gdn_scan 20`: serial
        vs chunked at the ornith shape; the bar is ≥ 4× on the M4 Pro (the
        serial scan is ≈ 82 ms per 4,096-row layer call today; target ≤ 20
        ms). If short, the first knob is `chunkedScanValueBlock = 16`
        (`(Dv/16, Hv)` grid, 8 KB state tile) — measured, not assumed.
  - [x] Step 7: scratch + runner + env knob + description + server log line +
        `ServerModelSession` field. Five gates: release build 0 warnings,
        lint, links, suite, TSAN.
  - [x] Step 8: `tools/golden-baseline.sh --check 4` (M4 Pro, server stopped)
        — expected to differ; recapture; commit `gdn: chunked delta-rule scan
        on the matrix path (v12 P4)` with the baseline via `--only`.
  - [x] Step 9: ledger on both boxes (fresh server, 3.7k + 12k,
        `tools/prefill-measure.sh`); mini `tools/mini-deploy.sh --restart`,
        mini golden recapture, scp into `baselines/`. Verdict line here;
        design doc Step 4 + ledger rows + the GDN row of "Where the time
        goes"; task review by a fresh reviewer; fixes folded into the commit.

# v12 implementation plan — Tasks 5–7 (the design's three follow-ons)

Drafted for `docs/v12-implementation-plan.md`, to be appended after Task 4 and
to replace the "Follow-ons (not scheduled)" list at the end of that file. The
plan's **Global constraints** section (macOS 26+/Swift 6.3+, never two model
processes, the mini's 8081 server is production; five gates per commit —
release build with zero warnings, `swiftlint lint --strict --baseline
.swiftlint-baseline.json`, markdown link check, `swift test --no-parallel`, the
same under `env TSAN_OPTIONS=suppressions=tsan-suppressions.txt swift test
--no-parallel --sanitize=thread`; numerics tolerance `maxAbs ≤ 2e-2`, `rel ≤
2e-2` against the fp32 reference, `tools/golden-baseline.sh --check` *expected*
to differ and both boxes recaptured once per step with the before/after greedy
digests in the verdict, never a recapture for an unexplained mismatch; ledger
protocol `tools/prefill-measure.sh` at 3.7k and 12k on both boxes, one send per
prompt per server lifetime, fresh server per prompt set; no comments unless a
genuinely non-obvious why) applies unchanged to all three tasks.

Measured starting point, post-P4 (`docs/v12-prefill-matrix-kernels.md:141-183`),
GPU ms per **prompt** token:

| role | M4 Pro 3.7k | M4 Pro 12k | M4 Pro 25k | M1 3.7k | M1 12k |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prefill_routed_tile` | 1.04 | 0.99 | 1.12 | 3.42 | 3.35 |
| `prefill_gdn_router` | 0.65 | 0.64 | 0.70 | 3.44 | 3.43 |
| `prefill_attn_router` | 0.26 | 0.45 | 0.80 | 1.35 | 2.41 |
| `prefill_shared_expert` | 0.06 | 0.06 | 0.06 | 0.29 | 0.29 |
| gaps (span − busy) | 0.5–0.95 | 0.35 | 0.35 | — | 0.53 |
| **wall** | 3.46 | 2.65 | 3.22 | 10.40 | 10.28 |

Targets from the design's ceiling table (`:61-70`): 1.6 ms/token at 12k on the
M4 Pro, 6.3 on the M1. The three tasks below are modelled to land at ≈ 2.0 and
≈ 8.3; what is left after them is GDN's projection stack and the shared
`matmul2d` efficiency, not these three taxes.

---

### Task 5: P5 — tile command-buffer batching

- [x] **P5: tile command-buffer batching** — target the inter-tile gap
  0.35 → ≤ 0.12 ms/prompt-token (M4 Pro, 12k), 0.95 → ≤ 0.35 at 3.7k, and
  0.53 → ≤ 0.20 (M1, 12k). Modelled wall: M4 Pro 12k 2.65 → ≈ 2.42 (32.6 →
  ≈ 30 s), 3.7k 3.46 → ≈ 2.90; M1 12k 10.28 → ≈ 9.95 (126.3 → ≈ 122 s).
  Nothing about the kernels changes, so the greedy digests must be **identical**
  on both boxes — a golden diff here is a bug, not a numerics change.
  **LANDED a7c8288 + bf469ad (2026-09-02) as a measured null result: the code
  default stays at one tile per command buffer and `SHRIKE_PREFILL_TILE_BATCH`
  remains an A/B knob.** Same-binary A/B on the M4 Pro at 3.7k (two
  interleaved pairs, fresh server per run): GPU busy flat (7.6–8.4 s) and the
  routed role flat (1.03–1.11 ms/token) at every width, but the routed→routed
  gap grew from 0.85 / 0.95 s (1,170 boundaries, ≈ 0.7 ms each) to 3.1 s at
  width 4 (277 boundaries, 11.4 ms each) and 3.0 s at widths 8 and 16 (16
  fitted to 8 by the slot budget); walls 11.4 / 11.7 s → 13.9 / 14.5 (W4) →
  15.2 / 15.0 (W8). At 12k width 4 took the wall from 32.9 to 41.4 s; M1 3.7k
  read 37.2 s at width 1 against 41.4 s at width 4. The gap split added in
  bf469ad (`host_ms` / `driver_ms` / `queue_ms` from `kernelStartTime` /
  `kernelEndTime`) puts the width-4 routed gap at host 3.00 s, driver 0.01,
  queue 0.16; an uncommitted per-step host probe (task-5-report.md, "Probe")
  found no encode step moved (validate + argument buffer 0.012 ms per tile,
  encode 0.019, commit 0.004 at both widths) and the whole penalty in the
  wait, 0.43 → 2.64 ms per tile. The cause is the fetch: `fetchBindingForTile`
  costs 3.9 ms per tile at both widths (its pread, `io_fetch_ms`, is 0.5 ms
  of that) against 3.2 ms of GPU work per tile on the M4 Pro, so at width 1
  the pending-tile overlap already hides all but ≈ 0.7 ms of it — the loop is
  fetch-bound, not commit-bound — and every non-first tile of a batch is
  fetched against an idle GPU, paying the full 3.6 ms. The design's premise
  (≈ 1.3 ms of commit → wait → encode per boundary) was wrong: driver + queue
  per boundary is ≈ 0.3 ms, so even a zero-penalty batch could save ≈ 0.8 s of
  32.9 at 12k. On the M1 the routed→routed gap is already zero at width 1
  (10.6 ms of GPU per tile hides the fetch). Golden identical on both profiles
  on both boxes at the default, and at width 4 on the M4 Pro. Ledger rows at
  the default width match the P4 rows within noise (M4 Pro 12k 32.9 s, M1 12k
  126.0 s). Two follow-on candidates fall out of the split and are recorded
  in the design doc: the fetch's ≈ 3.3 ms of non-I/O latency per tile, and a
  driver cost of ≈ 11 ms (M4 Pro) / ≈ 22 ms (M1) on the first routed buffer
  of every layer (`prefill_shared_expert->prefill_routed_tile` `driver_ms`).

  **The measured cause.** Each routed tile gets its own command buffer
  (`RealForwardRunner.swift:5037-5048`), and the pending-tile machinery that
  would overlap it (`:4917-4944`, `:4946-5056`) is disarmed whenever the tile's
  experts are all resident: `fetchBindingForTile` returns
  `plannedAssignedSlots == []` (`PrefillGroupedRoutedMoE.swift:361-368`), the
  scheduler then hits the empty-slot branch at `RealForwardRunner.swift:4982-4996`
  and `PrefillRoutedTileScheduler.decide` returns
  `.drainBeforeIssue(reason: .pendingTileHasNoAssignedSlots)`
  (`PrefillRoutedTileScheduler.swift:71-73`). So on the M4 Pro at
  `--ram-budget 20G` — and on every full-hit tile on the mini — the loop is
  strictly commit → `waitUntilCompleted` (`:3354-3359`) → encode the next tile.
  ≈ 1.3 ms per boundary × 32 tiles per layer × 40 layers per 4,096-token chunk.

  **Files:**
  - Modify: `sources/Shrike/Kernels/Prefill/MoE/PrefillRoutedTileScheduler.swift:30-55`
    (`tilesPerCommandBuffer`, the slot-budget arithmetic) and `:57-79`
    (a batch-aware decision)
  - Modify: `sources/Shrike/Runtime/Inference/RealForwardRunner.swift:4911-4944`
    (`PendingPrefillTile` → `PendingPrefillBatch`, `drainOldestPendingTile` →
    `drainOldestPendingBatch`), `:4946-5056` (the tile loop: one command buffer
    per batch, `avoidingSlots` widened by the open batch), `:5057-5059` (tail
    drain), `:329` (`prefillRoutedTileSchedulerConfig`), `:377-378` (the env
    knob, beside `SHRIKE_GDN_PREFILL_SCAN`), `:200-206` (a
    `prefillTileBatchDescription` beside `prefillGDNScanPathDescription`)
  - Modify: `sources/ShrikeServer/Core/ServerInference.swift:527-529, 817-818,
    849-851` (`prefill_tile_batch=` on the residency line)
  - Test: `tests/Shrike/Core/Kernels/Prefill/PrefillRoutedTileSchedulerTests.swift`
    (host-only; no Metal, no model)

  **Interfaces:**
  - Consumes: `PrefillStreamedTileFetchResult.plannedAssignedSlots` /
    `.plannedMissSlots` (`PrefillGroupedRoutedMoE.swift:110-133`),
    `PrefillStreamedTileSlotLifetime.begin/complete` (`:153-189`),
    `Model.planRoutedExpertsIfPossible` / `abandonRoutedExpertPlan`,
    `recordKernelGPU(role:_:)` (`RealForwardRunner.swift:1609-1613`).
  - Produces:

    ```swift
    struct PrefillRoutedTileSchedulerConfig: Sendable, Equatable {
        let maxPendingDepth: Int
        let tileExperts: Int
        /// Tiles encoded into one command buffer before it is committed. 1 is
        /// the pre-P5 behaviour: one tile, one commit, one wait.
        let tilesPerCommandBuffer: Int
        init(maxPendingDepth: Int = 1, tileExperts: Int = 8,
             tilesPerCommandBuffer: Int = 1)
        /// Every tile of the open batch and of each pending batch holds its
        /// slots until that batch completes, so the cache must hold them all.
        func fitsSlotBudget(slotCount: Int, reservedHits: Int = 0) -> Bool
            // (maxPendingDepth + 1) * tilesPerCommandBuffer * tileExperts + reservedHits <= slotCount
        func fitting(slotCount: Int, reservedHits: Int = 0) -> Self?
    }
    enum PrefillRoutedTileBatchAction: Sendable, Equatable {
        case appendToOpenBatch
        case commitOpenBatchThenAppend(reason: PrefillRoutedTileBatchFlushReason)
    }
    enum PrefillRoutedTileBatchFlushReason: Sendable, Equatable {
        case batchFull                 // tilesPerCommandBuffer reached
        case slotCollision             // the next tile's plan needs a slot the open batch holds
        case lastTile                  // routes.tiles exhausted
    }
    extension PrefillRoutedTileScheduler {
        func batchAction(openBatchTiles: Int,
                         openBatchSlots: [Int],
                         nextTileAvoidingSlotPlanAvailable: Bool,
                         isLastTile: Bool) -> PrefillRoutedTileBatchAction
    }
    ```

    Runner: `SHRIKE_PREFILL_TILE_BATCH=<n>` (`1` = today's one-tile buffers,
    clamped to `1...16`; default `1` until the ledger says otherwise, then the
    code default moves and the knob keeps the A/B).
    `RealForwardRunner.prefillTileBatchDescription` = `"tiles=\(n) fitted=\(m)"`,
    logged as `prefill_tile_batch=` on the residency line.

  **Constraints this must not break.**
  - *Expert-load discovery.* `fetchBindingForTile` (`:5011-5017`) stays exactly
    where it is: every tile is fetched, validated (`validateCoversPairs`,
    `:5018-5020`) and its argument buffer built (`:5025-5027`) **before** it is
    encoded. Batching only defers `commit()`.
  - *Slot lifetime.* A slot referenced by an encoded-but-uncommitted tile is
    live. So `avoidingSlots` at `:5017` and the pending-slot set at `:4953`
    both gain the open batch's `plannedAssignedSlots`, and
    `tileLifetime.begin` (`:5021-5024`) still runs per tile while
    `tileLifetime.complete` (`:4941-4943`) runs for every tile of the drained
    batch. `PrefillStreamedTileSlotLifetime.begin` already throws
    `.slotReuseBeforeCompletion` on overlap (`:158-170`) — that throw is the
    safety net the tests pin.
  - *Role accounting.* One `recordKernelGPU(role: "prefill_routed_tile", cb)`
    per batch buffer. The role's millisecond sum is still the true GPU span
    (`kernelGPUOccupancy`, `:1637-1655`); only `count` drops by the batch
    factor, and the `prefill_routed_tile->prefill_routed_tile` transition in
    `kernelGPUGaps` (`:1668-1686`) shrinks by the same factor — which is the
    measurement.
  - *`withExtendedLifetime`.* The fetch results and argument buffers of every
    tile in a batch are held until that batch's wait returns (`:4927-4936`
    generalized over an array), so a blob cannot be released under a committed
    encoder.

  Steps (TDD; the host policy first, so the bookkeeping is proven before any
  command buffer moves):

  - [ ] Step 1: failing tests in `PrefillRoutedTileSchedulerTests.swift`:
        `batchOfOneReproducesTheSingleTileDecisions` (for each of the five
        existing `decide` cases, `PrefillRoutedTileSchedulerConfig(
        tilesPerCommandBuffer: 1)` gives the same decision);
        `batchFillsToTheConfiguredWidth` (`tilesPerCommandBuffer: 4`,
        `openBatchTiles: 3` → `.appendToOpenBatch`; `openBatchTiles: 4` →
        `.commitOpenBatchThenAppend(reason: .batchFull)`);
        `batchCommitsWhenTheNextTileNeedsAHeldSlot`
        (`nextTileAvoidingSlotPlanAvailable: false`, `openBatchSlots: [2, 5]`
        → `.commitOpenBatchThenAppend(reason: .slotCollision)`);
        `slotBudgetCountsTheWholeOpenBatch`
        (`PrefillRoutedTileSchedulerConfig(maxPendingDepth: 1, tileExperts: 8,
        tilesPerCommandBuffer: 4).fitsSlotBudget(slotCount: 64)` true,
        `slotCount: 32` false; `.fitting(slotCount: 32)` returns
        `tileExperts: 4`);
        `lastTileAlwaysCommits`. `swift test --no-parallel --filter
        PrefillRoutedTileSchedulerTests` → FAIL: `tilesPerCommandBuffer` and
        `batchAction` undefined.
  - [ ] Step 2: failing test `slotLifetimeRejectsReuseInsideAnOpenBatch` in the
        same file: `var lifetime = PrefillStreamedTileSlotLifetime();
        try lifetime.begin(tileIndex: 0, plannedSlots: [1, 2]);
        #expect(throws: PrefillStreamedTileLifetimeError.self) {
            try lifetime.begin(tileIndex: 1, plannedSlots: [2, 3]) }`, then
        `try lifetime.complete(tileIndex: 0)` and the same `begin` succeeding.
        This is the invariant the batched drain has to keep; it fails today
        only because `PrefillStreamedTileSlotLifetime` is not yet exercised
        from this suite.
  - [ ] Step 3: implement `tilesPerCommandBuffer`, the widened
        `fitsSlotBudget`/`fitting`, and `batchAction`. Both tests PASS.
  - [ ] Step 4: rewrite the tile loop: `PendingPrefillBatch { let tileIndices:
        [Int]; let commandBuffer: MTLCommandBuffer; let fetches:
        [PrefillStreamedTileFetchResult]; let argumentBuffers:
        [PrefillStreamedTileArgumentBuffer] }`; one `ctx.queue.makeCommandBuffer()`
        per batch instead of per tile; `encodeRoutedTileExperts`
        (`:5109-5165`) called once per tile onto the open batch's buffer;
        `commit()` on the batch action; `drainOldestPendingBatch` waits once,
        records the role once, and completes every tile's lifetime entry.
        `avoidingSlots` at `:5017` becomes
        `Set(openBatchSlots + pendingBatches.flatMap { $0.fetches.flatMap(\.plannedAssignedSlots) })`.
        The `while pendingTiles.count > schedulerConfig.maxPendingDepth` drain
        (`:5053-5055`) becomes a batch-count drain, and the tail drain
        (`:5057-5059`) flushes the open batch first.
  - [ ] Step 5: the env knob, `prefillTileBatchDescription`, the residency line
        field, and `PrefillRoutedTileSchedulerConfig(tilesPerCommandBuffer:)`
        from it — clamped, and re-`fitting`ed against
        `model.routedExpertCacheSlotCount()` at `:4867-4878` so a small
        streamed cache shrinks the tile instead of throwing. Five gates:
        release build 0 warnings, lint, links, `swift test --no-parallel`, the
        same under TSAN.
  - [ ] Step 6: same-binary A/B on the M4 Pro at 3.7k, two pairs, fresh server
        per prompt (`--port 8082 --ram-budget 20G --thinking off`,
        `SHRIKE_KERNEL_STATS=1 SHRIKE_RUNNER_STATS=1`):
        `SHRIKE_PREFILL_TILE_BATCH=1` against `=4` and `=8`. Read
        `prefill_routed_tile->prefill_routed_tile` and `busy_ms`/`span_ms` from
        the `Shrike kernel busy_ms` line with `tools/prefill-ledger.py`. Pick
        the width whose gap is lowest and whose `prefill_routed_tile` role ms
        has not moved (batching must not slow the kernels); if two widths tie,
        take the narrower. **Decision point:** if the best width leaves the
        12k gap above 0.25 ms/token, the remaining cost is host encoding, not
        commit+wait — record that in the verdict and stop rather than widening
        further.
  - [ ] Step 7: `tools/golden-baseline.sh --check 4` (M4 Pro, server stopped).
        This step changes no arithmetic, so the expectation is **identical**;
        a diff is a bug in the batched encode order, not a numerics change —
        do not recapture, fix it. Commit `prefill: batch routed tiles per
        command buffer (v12 P5)`; if a baseline did change and the cause is
        understood and signed off, add it with `git commit --only`.
  - [ ] Step 8: ledger on both boxes (fresh server, 3.7k + 12k,
        `tools/prefill-measure.sh <host> <port> <promptdir> <outdir> <tag>
        2k 6k`); mini `tools/mini-deploy.sh --restart`, mini golden
        `--check 4` (expect identical), scp any recapture into `baselines/`.
        Verdict line here with the gap row before → after on both boxes and
        the `prefill_routed_tile` count; design doc: a "Follow-ons" → landed
        entry plus ledger rows; task review by a fresh reviewer; fixes folded
        into the commit (rebase and amend, never a fixup).

---

### Task 6: P6 — routed GEMM grouping

- [x] **P6: routed GEMM grouping** — target `prefill_routed_tile` 0.99 → ≤ 0.70
  ms/prompt-token (M4 Pro, 12k; 1.04 → ≤ 0.73 at 3.7k, 1.12 → ≤ 0.79 at 25k)
  and 3.35 → ≈ 2.35 (M1, 12k). Bar: **≥ 50 % of the 128-row expert-shape
  ceiling** on the M4 Pro — 5.74 TFLOPS for gate/up (128×2048×1024) and 5.88
  for down (128×512×2048), `docs/v12-prefill-matrix-kernels.md:49-50`; today
  the role runs at ≈ 2.0 TFLOPS, 35 % (`:170-171`). The M1's same-shape
  ceilings are 1.84 and 1.78 TFLOPS.
  **LANDED 4c44b8c (2026-09-02): measured `prefill_routed_tile` 1.02 → 0.72
  ms/prompt-token (M4 Pro 3.7k; 0.98 → 0.69 at 12k, 1.12 → 0.82 at 25k) and
  3.40 → 3.22 (M1 3.7k; 3.35 → 3.13 at 12k).** The M4 Pro bars are met at 3.7k
  and 12k and missed by 4 % at 25k; the M1 bar is missed. Bench (`ShrikeBench
  routed_gemm 20`, the ornith tile, 6.44 GFLOP): M4 Pro per-expert 3.81 ms
  (1.69 TFLOPS) → grouped 1.67 ms (3.86 TFLOPS, 2.28×), **67 % of the 5.77
  TFLOPS gate/up ceiling measured in the same run** (29 % before); M1 9.39 ms
  (0.69 TFLOPS) → 8.24 ms (0.78 TFLOPS, 1.14×), 68 % of its same-run ceiling of
  1.15 (60 % before — the M1's eight cores were already filled by the 32-
  threadgroup per-expert grids, so grouping, whose gain is filling idle cores,
  buys 14 % of kernel time there; the M1 bar rested on the M4 Pro's 35 % share,
  which the M1 never had). Neither Step-5 knob was needed: staging 2,048 is a
  no-op at one wave per tile and `tileN` 64 is untried. Same-binary A/B
  (`SHRIKE_PREFILL_ROUTED_GEMM=per-expert`, 1,024-row staging on both arms,
  fresh servers): M4 Pro 3.7k walls 11.11 / 11.12 per-expert against 13.56 /
  10.90 grouped (pair 1's grouped run carried a 22 ms-per-layer driver spike
  on the first routed buffer; pair 2 is the reading), 12k 32.27 → 31.64 s, 25k
  81.2 (P4) → 78.8 s; M1 3.7k 36.8 → 36.5 s, 12k 126.0 → 123.4 s. On the M4
  Pro the GPU saving reappears as routed→routed gap (0.76 → 1.39 ms per
  boundary at 3.7k, `host_ms`) because the loop is fetch-bound at 3.9 ms per
  tile (P5); the M1 hides the fetch under its 7.5 ms tile and keeps the whole
  7 %. Golden: short identical on both boxes (the 51-byte prompt never reaches
  the matrix path); long differs as designed and was recaptured once per box —
  sha256 M4 Pro 7c545301a1d3fe8d → 79aa5873bb2a167f, M1 899a25e60a365e60 →
  39c38734e8f9d22a (the files' `# commit:` header records the parent 811fa7f;
  both captures ran before the commit). Numerics: grouped against the P3 GEMMs
  plus scalar leftovers and against the scalar path at 2e-2 over every row,
  finiteness asserted; the 5- and 3-pair experts of the fixture run inside the
  grouped dispatch. Review (opus): Approved with two Important findings, both
  folded in (the residency field probed with a placeholder shape; the golden
  digests unrecorded) plus seven minors; minors 4, 6, 8, 11, 12 ruled or
  deferred in the ledger. Deviations: the block tables go inline (`setBytes`)
  rather than into shared buffers rewritten per wave — a CPU→GPU hazard with
  the pending-tile overlap; the bench harness is a public façade in the module
  because ShrikeBench sees only public API.

  **What the P3 path costs.** `encodeExpertGEMMs`
  (`PrefillGroupedRoutedMoE.swift:639-688`) loops experts and, per expert row
  block, `encodeExpertRowBlock` (`:690-749`) issues six encoders: gather, gate
  GEMM, up GEMM, `silu_mul_fp16`, down GEMM, scatter — and each
  `MPPPrefillInt4QMM.encode` makes its own compute encoder
  (`MPPPrefillInt4QMM.swift:102-126`). Eight experts per tile is 48 encoders,
  each a full serialization point, and each GEMM's grid is only
  `(n/32, ceil(rows/64))` = 32 threadgroups at the 128-row gate shape
  (`tileM = 64`, `tileN = 32`, `:24-25`) against 20 GPU cores. Experts under
  `matrixPathMinimumRows = 32` (`:447`) fall out as leftovers (`:660-666`) and
  are re-run through the scalar microbatch path (`RealForwardRunner.swift:5119-5134`,
  `PrefillGroupedRoutedMoE.swift:554-631`) at 32 pairs a dispatch.

  **The grouping.** One dispatch per phase over *all* experts' row blocks in a
  tile: a per-64-row-tile lookup selects the expert's packed-weight pointer
  from the tile's argument buffer, so the M dimension of a single dispatch
  becomes the tile's whole padded row count (≈ 1,024 rows at a 4,096-token
  chunk, 8 experts × ~128 pairs) instead of 128. Six encoders per *wave*, not
  per expert; no 32-pair threshold and no leftovers.

  **Files:**
  - Modify: `sources/Shrike/Metal/TensorCore/tensorops.metal:29-121` — add
    `mpp_prefill_affine_grouped_f16` beside `mpp_prefill_affine_threadgroup_f16`
    (identical inner loop; only the row origin and the weight pointer are
    indirected)
  - Modify: `sources/Shrike/Kernels/TensorCore/MPPPrefillInt4QMM.swift:24-26,
    68-128` — `encodeGrouped(...)`, sharing the tile constants and the
    `k.isMultiple(of: tileK)` / offset-alignment guards
  - Modify: `sources/Shrike/Metal/Prefill/prefill.metal:649-685` — grouped
    `prefill_routed_gather_rows_grouped` / `prefill_routed_scatter_rows_grouped`
    (a block table replaces the single `pair_start`/`rows` param)
  - Modify: `sources/Shrike/Kernels/Prefill/MoE/PrefillGroupedRoutedMoE.swift:30-63`
    (`PrefillExpertPairRange` gains the block planner), `:66-99`
    (`PrefillExpertStaging` gains the block table buffers), `:445-468`
    (`matrixPathMinimumRows` retired for the grouped path), `:633-749`
    (`encodeGroupedExpertGEMMs` beside the P3 pair, which stays as the
    fallback)
  - Modify: `sources/Shrike/Runtime/Prefill/PrefillChunkScratch.swift:121-137`
    (staging rows sized for a whole tile, block-table bytes)
  - Modify: `sources/Shrike/Runtime/Inference/RealForwardRunner.swift:5109-5165`
    (`encodeRoutedTileExperts` selects grouped → per-expert → scalar)
  - Create: `sources/ShrikeBench/RoutedGEMMBench.swift` (`routed_gemm` mode),
    registered in `sources/ShrikeBench/ShrikeBench.swift:40-56`
  - Test: `tests/Shrike/Core/Kernels/Prefill/PrefillGroupedRoutedMoETests+Execution.swift`
    (the P3 fixtures at `:215-527` are the oracle)

  **Interfaces:**
  - Consumes: `PrefillStreamedTileBinding.views` / `localSlot(for:)`
    (`PrefillGroupedRoutedMoE.swift:278-311`), the tile argument buffer
    (`:470-485`), `PrefillGroupedRoutedMoEStreamedParams`'s
    `gateWOff/gateSOff/gateBOff/upWOff/…/downBOff` (`:215-226`),
    `PrefillExpertPairRange.ranges(forTile:routes:)` (`:45-63`),
    `sortedPairs` (`:529-552`).
  - Produces:

    ```swift
    /// One expert's contiguous slice of the staging block. `rowTileStart` is
    /// its first 64-row tile in the wave's grid, so a threadgroup recovers its
    /// row origin without a search.
    struct PrefillRoutedExpertBlock: Equatable, Sendable {
        var slot: UInt32          // index into the tile binding's views
        var pairStart: UInt32     // first pair in sortedPairs
        var rows: UInt32          // real pairs; the tail up to a 64 multiple is padding
        var stagingRow: UInt32    // 64-aligned staging row
        var rowTileStart: UInt32  // stagingRow / 64
    }
    /// One command-buffer wave: the blocks whose padded rows fit `stagingRows`.
    struct PrefillRoutedExpertWave: Equatable, Sendable {
        var blocks: [PrefillRoutedExpertBlock]
        var paddedRows: Int       // == blocks.last.stagingRow + roundUp(rows, 64)
    }
    extension PrefillGroupedRoutedMoE {
        static let groupedRowTile = MPPPrefillInt4QMM.tileM   // 64
        /// Splits a tile's experts into waves. An expert longer than
        /// `stagingRows` is split across waves at 64-row boundaries; an expert
        /// with 1 pair takes one 64-row tile with 63 padded rows.
        static func planExpertWaves(ranges: [PrefillExpertPairRange],
                                    binding: PrefillStreamedTileBinding,
                                    stagingRows: Int) throws -> [PrefillRoutedExpertWave]
        /// Six dispatches per wave, whatever the expert count: grouped gather,
        /// gate, up, silu_mul over `paddedRows × F`, down, grouped scatter.
        func encodeGroupedExpertGEMMs(commandBuffer: MTLCommandBuffer,
                                      mpp: MPPPrefillInt4QMM,
                                      hidden: MTLBuffer, hiddenOffset: Int = 0,
                                      sortedPairs: MTLBuffer, sortedPairsOffset: Int = 0,
                                      routePartials: MTLBuffer, routePartialsOffset: Int = 0,
                                      binding: PrefillStreamedTileBinding,
                                      argumentBuffer: PrefillStreamedTileArgumentBuffer,
                                      waves: [PrefillRoutedExpertWave],
                                      staging: PrefillExpertStaging,
                                      params: PrefillGroupedRoutedMoEStreamedParams) throws
    }
    extension MPPPrefillInt4QMM {
        /// `weights` come from `experts.blob[block.slot] + weightsOffset`, so the
        /// caller passes the tile argument buffer and calls `useResource` on every
        /// bound view. `n`/`k` are uniform across the wave; `m` is `paddedRows`.
        @discardableResult
        func encodeGrouped(commandBuffer: MTLCommandBuffer,
                           experts: MTLBuffer, expertViews: [TensorView],
                           blocks: MTLBuffer, rowTileBlock: MTLBuffer,
                           weightsOffset: Int, scalesOffset: Int, biasesOffset: Int,
                           x: MTLBuffer, xOffset: Int = 0,
                           y: MTLBuffer, yOffset: Int = 0,
                           rowTiles: Int, paddedRows: Int, n: Int, k: Int,
                           required: Bool = true) throws -> Path
    }
    ```

    Metal, in `tensorops.metal` (the module is compiled on its own through
    `MetalContext.moduleLibrary(device:module:"tensorops")`, so the blob struct
    and the block struct are declared locally there; the argument encoder comes
    from `mpp_prefill_affine_grouped_f16` itself, and a
    `precondition(encoder.encodedLength == streamedArgEncoder.encodedLength)`
    at init pins the two layouts together):

    ```metal
    struct MPPGroupedExpertBlobsMSL { device const uint8_t* blob[16]; };
    struct MPPGroupedBlockMSL { uint slot; uint pair_start; uint rows;
                                uint staging_row; uint row_tile_start; };
    kernel void mpp_prefill_affine_grouped_f16(
        device const MPPGroupedExpertBlobsMSL& experts [[buffer(0)]],
        device const MPPGroupedBlockMSL*       blocks  [[buffer(1)]],
        device const uint*                     rowTileBlock [[buffer(2)]],
        device half*                           activations  [[buffer(3)]],
        device half*                           output       [[buffer(4)]],
        constant uint& N [[buffer(5)]], constant uint& K [[buffer(6)]],
        constant uint& wOff [[buffer(7)]], constant uint& sOff [[buffer(8)]],
        constant uint& bOff [[buffer(9)]], constant uint& M [[buffer(10)]],
        uint3 tgid, uint3 lid3, uint3 threads3);
    ```

    Body: `const MPPGroupedBlockMSL b = blocks[rowTileBlock[tgid.y]];` then
    `packedWeights = experts.blob[b.slot] + wOff`, `scales/biases` likewise;
    the A-tensor slice origin is `int32_t(b.staging_row + (tgid.y - b.row_tile_start) * 64)`
    over the same `dextents(K, M)` / stride `{1, K}` tensor as
    `tensorops.metal:54-60`; the store guard at `:117` becomes
    `globalM < b.staging_row + b.rows && globalN < N`. Everything else —
    the `kMPPAffineTileK` group loop, the threadgroup weight tile, the fp32
    accumulator — is byte-for-byte the existing kernel, which is why the
    numerics move only by which threadgroup owns a row.

    Grouped gather/scatter in `prefill.metal` take the same `blocks` /
    `rowTileBlock` buffers and a `uint2 gid` of `(D, paddedRows)`; gather
    writes `half(0)` into a padded row (`row - b.staging_row >= b.rows`) and
    scatter skips it, so a 1-pair expert costs a 64-row tile of arithmetic and
    writes exactly one `routePartials` row.

    Scratch: `PrefillChunkScratchLayout.routedExpertStagingRows` rises from
    `min(512, chunkTokens)` (`PrefillChunkScratch.swift:132-135`) to
    `min(1024, chunkTokens)` so a whole 8-expert tile usually fits one wave —
    at ornith that is 1,024×2048 + 2×1,024×512 + 1,024×2048 fp16 = 10 MB, up
    from 5 MB. Two new shared buffers sized
    `(stagingRows/64 + 16) * MemoryLayout<...>.stride`, rewritten in place per
    wave (no per-tile allocation). `usesRoutedExpertMatrixPath` (`:125-129`)
    is unchanged, so the 32-token MTP draft chunk and dense architectures
    allocate none of it.

  Steps (TDD; numerics against the path P3 already proved before any perf work):

  - [ ] Step 1: failing test
        `groupedGEMMsMatchThePerExpertPathAcrossAllExperts` in
        `PrefillGroupedRoutedMoETests+Execution.swift`, reusing
        `runExpertGEMMsMatchTheScalarPath`'s fixture (`:228-253`): four experts
        with 40 / 32 / 5 / 3 pairs, `d = 64`, `f = 64`, `topK = 2`,
        `makeSyntheticExpertPool`, `streamedViewsWithNonzeroOffsets`. Run the
        P3 `encodeExpertGEMMs` + scalar-leftover path into one `routePartials`
        buffer and `encodeGroupedExpertGEMMs` into another; expect
        `RelError.maxAbsDiff ≤ 2e-2` and `RelError.compute ≤ 2e-2` over all
        `rows * topK * d` elements, and — the point of the task —
        `#expect(leftoverRangesFromGrouped.isEmpty)`: the 5- and 3-pair experts
        are now inside the grouped dispatch, so their rows must be non-`-77`
        (the sentinel the existing tests fill with, `:277-280`).
  - [ ] Step 2: failing test `groupedWavePlannerSplitsAndPadsOnRowTiles`
        (host-only, no GPU). With 64-row alignment the fixture's four experts
        (40 / 32 / 5 / 3 pairs) occupy `stagingRow` 0, 64, 128, 192 for
        `paddedRows` 256, so: `stagingRows: 512` gives one wave of four blocks;
        `stagingRows: 128` gives two waves of two, the second restarting at
        `stagingRow` 0; a single 600-pair expert at `stagingRows: 512` splits
        into blocks of 512 and 88 rows with `pairStart` 0 and 512. Expect exact
        `[PrefillRoutedExpertWave]` equality in all three cases.
  - [ ] Step 3: failing test `groupedGEMMsMatchTheScalarPathAcrossWaves` —
        the same fixture with `stagingRows: 128`, forcing three waves; and
        `groupedGEMMsHandleASingleOnePairExpert` — one expert, one pair,
        63 padded rows, compared against `encodeStreamedBatched` on the same
        inputs at 2e-2, with the 63 padded `routePartials` rows still holding
        the `-77` sentinel. `swift test --no-parallel --filter
        PrefillGroupedRoutedMoETests` → FAIL: `encodeGroupedExpertGEMMs`,
        `planExpertWaves`, `encodeGrouped` undefined.
  - [ ] Step 4: implement `mpp_prefill_affine_grouped_f16`, the grouped
        gather/scatter, `MPPPrefillInt4QMM.encodeGrouped` (with
        `useResource(view.buffer, usage: .read)` for every bound view, as
        `encodeStreamedBatched` does at `PrefillGroupedRoutedMoE.swift:593-595`),
        `planExpertWaves`, and `encodeGroupedExpertGEMMs`. All three tests PASS.
  - [ ] Step 5: `swift run -c release ShrikeBench routed_gemm 20` — the ornith
        tile shape (8 experts × 128 rows, D 2048, F 512, int4 group-64): the P3
        per-expert dispatch sequence against the grouped one, ms per tile and
        TFLOPS for gate/up (`128×2048×1024` per expert) and down
        (`128×512×2048`), printed against the `gemm` ceiling the same mode
        prints. **Bar: ≥ 50 % of 5.74 / 5.88 TFLOPS on the M4 Pro.** If short,
        the knobs in order are (a) `stagingRows` 1024 → 2048 so the whole tile
        is one wave, (b) `tileN` 32 → 64 for the `n = 2048` down GEMM only —
        measured, not assumed. If neither clears 50 %, record the achieved
        share and land it anyway if it beats 35 %; say so in the verdict.
  - [ ] Step 6: wire `encodeRoutedTileExperts` (`RealForwardRunner.swift:5109-5165`):
        grouped when `scratch.layout.usesRoutedExpertMatrixPath` and
        `prefillGroupedMoE.matrixPath(for:d:intermediate:)` returns an `mpp`
        (`PrefillGroupedRoutedMoE.swift:456-468`); the P3 per-expert path stays
        behind `SHRIKE_PREFILL_ROUTED_GEMM=per-expert` for the same-binary A/B;
        the whole-tile scalar path stays as the last fallback exactly as at
        `:5139-5146`. Five gates: release build 0 warnings, lint, links,
        `swift test --no-parallel`, the same under TSAN.
  - [ ] Step 7: `tools/golden-baseline.sh --check 4` (M4 Pro, server stopped) —
        expected to differ (the row's reduction is unchanged but its owning
        threadgroup is not, and the 1–31-pair experts move from the scalar
        GEMV to the GEMM); recapture, record the before/after greedy digests,
        and commit `prefill: grouped routed-expert GEMMs (v12 P6)` with the
        baseline via `git commit --only`.
  - [ ] Step 8: ledger on both boxes (fresh server, 3.7k + 12k + 25k on the
        M4 Pro, 3.7k + 12k on the mini, `tools/prefill-measure.sh`); mini
        `tools/mini-deploy.sh --restart`, mini golden recapture, scp into
        `baselines/`. Verdict line with `prefill_routed_tile` before → after on
        both boxes and the achieved share of the 128-row expert-shape ceiling;
        design doc: the "Follow-ons" entry becomes a landed step with its
        ledger rows and the routed row of "Where the time goes"; task review by
        a fresh reviewer; fixes folded into the commit.

---

### Task 7: P7 — attention KV-tile staging per KV-head group

- [x] **P7: attention KV-tile staging per KV-head group** — target
  `prefill_attn_router` 0.45 → ≤ 0.32 ms/prompt-token (M4 Pro, 12k) and
  0.80 → ≤ 0.58 at 25k; 2.41 → ≈ 1.75 (M1, 12k). Bar: **≥ 50 % of the M4 Pro's
  measured attention ceiling**, computed as the design does at
  `docs/v12-prefill-matrix-kernels.md:174-176` — 12.4 TFLOP of scores and
  values at 12k, 52.2 at 25k, against the 7.46 TFLOPS `4096³` figure; the
  shipped kernel sits at ≈ 36 % (2.6–2.7 TFLOPS), the M1 at ≈ 26 %.
  **LANDED 5ba2b89 (2026-09-02): measured `prefill_attn_router` 0.26 → 0.24
  ms/prompt-token (M4 Pro 3.7k; 0.45 → 0.36 at 12k, 0.81 → 0.67 at 25k) and
  1.35 → 1.25 (M1 3.7k; 2.40 → 2.05 at 12k).** The bars are missed on both
  boxes: the M4 Pro sits at 37 % of the 7.46 TFLOPS ceiling at 12k (12.4 TFLOP
  over 4.44 s; 30 % before), the M1 at 2.05 against ≈ 1.75. The spike (14
  geometries under `SHRIKE_ATTN_MATRIX_TILE`, a fresh server per arm, both
  boxes) falsified the task's premise: every form that staged a KV-head
  group's K/V tile through threadgroup memory once for all eight heads was
  slower than P2 — M4 Pro +26–80 %, M1 +5–22 % — because the caches already
  serve the eightfold re-read, so the modelled 2–4× traffic cut bought
  nothing. The device-operand forms, which read the shadow exactly as P2 does
  but from the group-major Q, win by amortising the per-tile fixed work (the
  score round-trip, the R×256 accumulator rescale, the barriers) over more
  keys and fewer rows: 32 rows × 64 keys (P2) 0.460, 32 × 128 0.394, 16 × 256
  0.363, 8 × 512 0.404 ms/tok at 12k on the M4 Pro; eight simdgroups lose to
  four at every shape once the cooperative-tensor loops are unrolled (the
  pragma was worth −35 % on the eight-simdgroup body and nothing on P2's
  four). `g2k256d` — 2 query positions × 8 heads = 16 rows, 256-key tiles,
  4 simdgroups — is the default; `g4k128d` (tied with it on the M1, 2.08 vs
  2.05) and the P2 tiles stay selectable. MPP's register-resident left operand
  (the FlashAttention shape that would drop the score round-trip) is only
  allowed under a single-simdgroup execution scope and is the recorded
  follow-on. Wall: M4 Pro 12k 31.6 → 30.5 s, 25k 78.8 → 77.5 s (3.7k
  10.9 → 11.1 s, inside that prompt's ±2 s swing; routed→routed 1.39 → 1.46 ms
  per boundary at 12k on the fetch-bound loop); M1 12k 123.4 → 119.2 s, 3.7k
  36.5 → 37.2 s. Golden: short identical on both boxes; M4 Pro long differs
  and was recaptured once — sha256 79aa5873bb2a167f → f24c61565618fdab, the
  first divergence being the thinking block's second sentence flipping back to
  the pre-P3 wording (a near-tie logit); M1 long IDENTICAL, 39c38734e8f9d22a
  unchanged. Numerics: all 32 group cases (three fp16-reference and five
  int8/int4-vs-tiled fixtures × the landed and spike tiles) at 2e-2, finiteness
  asserted. Memory: `ensureQGroup` is chunk-bounded (32 MiB at the mini's
  4,096-token chunk); mini at 25k (label 12k, 25,245 tokens): `memory_pressure
  -Q` free percentage 91 % idle → 25 % floor during the request (31 samples,
  10 s apart), server RSS 10.27 GB, attention 3.29 ms/tok, wall 276 s. Five
  gates green (1148 tests, TSAN 0 reports).

  **What P2 left.** `attention_prefill_causal_matrix_r32s4`
  (`attention_matrix.metal:42-210`, instantiated at `:232`) gives one
  threadgroup 32 query rows of **one** query head (`qh = tg.y`, `:72`) and
  reads its K/V tiles straight out of the device fp16 shadow (`:89-96`,
  `:112`, `:177`) — no threadgroup staging at all. With 16 query heads over 2
  KV heads the same 64-key × 256 shadow tile is fetched by eight threadgroups
  (`PrefillAttention.swift:331-336` dispatches
  `(ceil(queryCount/32), numQHeads)`), and the matmul's M is only 32.

  **The change.** A threadgroup owns `Rq` query positions × all **eight**
  query heads of one KV head — `8·Rq` matmul rows — stages the key tile once
  into threadgroup memory, runs one `matmul2d` QKᵀ against it, then reloads
  the same threadgroup buffer with the value tile and runs one `matmul2d` PV.
  Device K/V traffic per (query, head) pair drops by `8·Rq / 32`, and the
  matmul's M rises from 32 to `8·Rq`.

  **The 32 KB budget, stated plainly.** A 64-key × 256 fp16 tile is exactly
  32,768 bytes — the whole threadgroup allocation, with nothing left for the
  score tile. That is why P2 put the shadow in device memory
  (`attention_matrix.metal:5-8`). Three ways to fit, and the spike picks one:

  | variant | rows `8·Rq` | keys S | staged tile (K, then V, one buffer) | score tile fp32 | threadgroup | O accum. bytes/thread |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | `g4k32` (Rq 4, 4 simdgroups, 128 threads) | 32 | 32 | 32×256×2 = 16 KB | 32×32×4 = 4 KB | 20 KB | 256 (64 regs) |
  | `g8k32` (Rq 8, 8 simdgroups, 256 threads) | 64 | 32 | 16 KB | 64×32×4 = 8 KB | 24 KB | 256 (64 regs) |
  | `g8k64` (Rq 8, 8 sg) | 64 | 64 | 64×128×2 = 16 KB, head-dim in two halves, QKᵀ accumulated over two `matmul2d` runs | 64×64×4 = 16 KB | 32 KB | 256 (64 regs) |
  | `g16k32` (Rq 16, 8 sg) | 128 | 32 | 16 KB | 128×32×4 = 16 KB | 32 KB | 512 (128 regs) |

  Staging **K then V through the same 16 KB buffer**, separated by a barrier,
  is what keeps every variant inside 32 KB; `g8k64` additionally splits the
  head dim into two 128-wide halves. The full eightfold traffic cut would need
  `Rq = 32` (256 rows), whose fp32 output accumulators are 256 KB — beyond the
  register file — so the honest modelled cut is 2× (`g8k32`) to 4× (`g16k32`);
  the rest of the gain is the wider matmul M. Say that in the verdict rather
  than claiming 8×.

  **Q must become a uniform-stride operand.** P2's note that "the eight heads'
  Q rows are not a single uniform-stride matrix" (`:179-181`) is true of the
  live layout: a row is `Q + q·qTokenStrideElements + h·headDim`, and
  `qTokenStrideElements` is 4096 while a head is 256. So P7 adds a pack pass,
  the exact twin of `attention_prefill_kv_dequant` (`:26-40`): write
  `qGroup[((kvh · queryCount) + q) · 8 + hLocal][256]`, a matrix of row stride
  256 in which the eight heads of a KV group and the query positions interleave
  on one axis. One read plus one write of `queryCount × numQHeads × headDim`
  fp16 per layer — 33.5 MB at a 4,096-token chunk, ≈ 0.3 ms on the M4 Pro,
  ≈ 0.0001 ms/prompt-token — bought back many times over by the single
  `8·Rq`-row matmul.

  **Files:**
  - Modify: `sources/Shrike/Metal/Prefill/attention_matrix.metal:26-40` (add
    `attention_prefill_q_group_pack`), `:42-210` (a second body template
    `attention_prefill_causal_group_matrix_body<Rq, S, SG>` beside the shipped
    one, which stays untouched), `:212-235` (a second macro and the four
    instantiations `g4k32`, `g8k32`, `g8k64`, `g16k32`)
  - Modify: `sources/Shrike/Kernels/Attention/PrefillAttention.swift:84-95`
    (`matrixTile` becomes a variant enum), `:96-131` (pipeline names, the pack
    pipeline), `:283-295` (`matrixPathAccepts` gains
    `numQHeads / numKVHeads == 8`), `:297-338` (`encodeMatrix`: pack dispatch,
    grouped dispatch geometry), `:340-355` (`ensureQGroup` beside
    `ensureShadow`)
  - Test: `tests/Shrike/Core/Kernels/Attention/PrefillAttentionMatrixTests.swift`
    (`:15-124` are the oracle: `PrefillAttentionRef.apply` on fp16, the
    `KVCacheManager`/`KVCacheQuantizer` int8/int4 cases, the gate cases)
  - `sources/Shrike/Infrastructure/Metal/MetalContext.swift:89-106` needs no
    change: `"attention_matrix"` is already registered after `"prefill"`
    (`:101-102`), which is what gives the new kernels `PrefillAttentionParams`.

  **Interfaces:**
  - Consumes: `PrefillAttentionParams` unchanged
    (`PrefillAttention.swift:4-52`), the device fp16 shadow from
    `attention_prefill_kv_dequant`, `PrefillAttentionRef.apply(_:) -> [Float]`
    and `PrefillAttentionRef.Inputs`, `RelError`, `Fp16Buffer`,
    `KVCacheQuantizer.encode`.
  - Produces:

    ```swift
    extension PrefillAttention {
        enum MatrixTile: String, Sendable {
            case r32s4, r64s8          // P2, one query head per threadgroup
            case g4k32, g8k32, g8k64, g16k32   // P7, one KV-head group per threadgroup
            var kernelName: String     // "attention_prefill_causal_matrix_<rawValue>"
            var groupsEightHeads: Bool { self != .r32s4 && self != .r64s8 }
            var queryRows: Int         // r32s4 32, r64s8 64, g4k32 4, g8k32 8, g8k64 8, g16k32 16
            var threadsPerThreadgroup: Int  // 32 * simdgroups
        }
        /// `SHRIKE_ATTN_MATRIX_TILE` names the variant; anything unrecognised
        /// keeps the measured production choice.
        static let matrixTile: MatrixTile
        /// Group-major Q, `[numKVHeads][queryCount * 8][headDim]`, row stride
        /// `headDim`. Chunk-sized, not context-sized: it never grows past
        /// `prefillChunkTokens * numQHeads * headDim` halves (33.5 MB at 4,096
        /// tokens). Grown in 8 MiB quanta and never released, like the shadow.
        private func ensureQGroup(bytes: Int) throws -> MTLBuffer
    }
    ```

    Metal:

    ```metal
    kernel void attention_prefill_q_group_pack(
        device const half* Q [[buffer(0)]],
        device half* qGroup   [[buffer(1)]],
        constant PrefillAttentionParams& p [[buffer(2)]],
        uint3 gid [[thread_position_in_grid]]);   // (headDim, numQHeads, queryCount)
    // qGroup[((gid.y / 8) * p.queryCount + gid.z) * 8 * p.headDim
    //        + (gid.y % 8) * p.headDim + gid.x]
    //   = Q[gid.z * p.qTokenStrideElements + gid.y * p.headDim + gid.x];
    ```

    and, per key tile inside
    `attention_prefill_causal_group_matrix_<variant>`: cooperative load of
    `S × headDim` (or `S × headDim/2`) halves of `shadowK` into
    `threadgroup half kv_tile[...]`; `matmul2d_descriptor(8*Rq, S, headDim,
    false, true, false)` with the left operand the device `qGroup` slice at row
    `(kvh · queryCount + q0) · 8` and the right operand `kv_tile`; the causal
    bound for row `r` is `min(kvValidCount, startPosition + q0 + r / 8 + 1)`
    (the eight heads of one query share it); the existing lane-parallel online
    softmax (`attention_matrix.metal:127-174`) is reused verbatim with
    `R → 8·Rq`; barrier; reload `kv_tile` from `shadowV`; barrier;
    `matmul2d_descriptor(8*Rq, headDim, S, false, false, false)` for PV; the
    store maps row `r` back to
    `O[(q0 + r/8) * oTokenStrideElements + (kvh * 8 + r % 8) * headDim + d]`.

    Dispatch (`PrefillAttention.swift:331-336`) becomes
    `MTLSize(width: ceil(queryCount / Rq), height: numKVHeads, depth: 1)` with
    `threadsPerThreadgroup = tile.threadsPerThreadgroup`, preceded by the pack
    dispatch `dispatchThreads((headDim, numQHeads, queryCount), (256, 1, 1))`
    on the same encoder as the dequant, so P7 adds no command buffer and no
    encoder — one more dispatch inside the encoder P2 already opens at `:310`.

    Memory: `ensureQGroup` adds 32 MiB at the mini's 4,096-token chunk (64 MiB
    with the MTP draft runner's own instance), independent of context length —
    unlike the KV shadow, which the design bounds at `:276-282`. Record the
    `memory_pressure -Q` free-percentage span on the mini at 25k in the verdict,
    as P2 did.

  Steps (TDD; correctness on the reference before any geometry hunting):

  - [ ] Step 1: failing test `qGroupPackMatchesStridedQuery` in
        `PrefillAttentionMatrixTests.swift`: `makeFixture(start: 512, chunk: 64,
        seed: 0xB130)` (whose `qStride` is `qHeads * headDim + 3`, `:127`),
        a non-zero `qOffset`, run the pack kernel alone, read back and
        `#expect(qGroup[((kvh * chunk + q) * 8 + hLocal) * 256 + d]
        == fixture.q[q * qStride + (kvh * 8 + hLocal) * 256 + d])` as exact
        fp16 bit equality over all 16 heads (a copy must not round). FAIL:
        `attention_prefill_q_group_pack` undefined.
  - [ ] Step 2: failing test
        `groupMatrixMatchesReferenceOnFP16Cache(c:)` — the three existing
        arguments `(single-tile, 0, 64)`, `(ragged-rows-three-key-tiles, 0,
        130)`, `(history-ragged-tail, 600, 40)` (`:15-19`) run through
        `Self.runFP16(..., path: .causalMatrix)` with
        `SHRIKE_ATTN_MATRIX_TILE` forced to the group variant, compared to
        `PrefillAttentionRef.apply(fixture)` at `Self.tolerance` (2e-2) on both
        `RelError.maxAbsDiff` and `RelError.compute`; and
        `groupMatrixMatchesTiledOnQuantizedCache(c:)` — the five existing
        int8/int4 arguments (`:32-37`) against `path: .causalTiled` on the same
        `KVCacheManager` views, same tolerance. Because `matrixTile` is read
        once from the environment (`:87-89`), add
        `PrefillAttention(context:supportsMLA:matrixTile:)` with the static as
        its default so the tests can pin a variant without a process-wide env
        var. FAIL: the group kernels do not exist.
  - [ ] Step 3: failing test `groupTileGateRequiresEightHeadsPerKVHead`:
        `matrixPathAccepts` false for a params with `numQHeads = 16,
        numKVHeads = 4`, still true for 16/2; and
        `rejectedShapeRunsTheTiledKernelEndToEnd` (`:87-96`) re-run with the
        group variant pinned, expecting byte equality with `.causalTiled`.
  - [ ] Step 4: implement `attention_prefill_q_group_pack`,
        `attention_prefill_causal_group_matrix_body<Rq, S, SG>` and the four
        instantiations, `ensureQGroup`, the `MatrixTile` enum, the gate clause
        and the dispatch. `static_assert` the threadgroup arithmetic in the
        body exactly as the shipped one does (`:56-58`) plus
        `static_assert(S * kAttnMatrixHeadDim * 2 + 8 * Rq * S * 4 <= 32768)`.
        `swift test --no-parallel --filter PrefillAttentionMatrixTests` → PASS
        for every variant.
  - [ ] Step 5 (spike, on the ledger — there is no `attn` bench mode in
        `sources/ShrikeBench/`, and P2's own geometry was settled this way):
        one M4 Pro server per variant, fresh each time, `--port 8082
        --ram-budget 20G --thinking off`, `SHRIKE_KERNEL_STATS=1
        SHRIKE_RUNNER_STATS=1`, the 12k prompt sent once, `SHRIKE_ATTN_MATRIX_TILE`
        ∈ {`r32s4` (control), `g4k32`, `g8k32`, `g8k64`, `g16k32`}. Record
        `prefill_attn_router` ms/prompt-token and the derived TFLOPS
        (`12.4e3 / role_ms` at 12k) for each. **Accept the first variant at
        ≥ 50 % of 7.46 TFLOPS** (`prefill_attn_router` ≤ 0.32 ms/token at 12k);
        if none reaches it, take the best above the shipped 36 % and record the
        table plus the achieved share as the verdict. Verdict line: the
        candidate table.
  - [ ] Step 6: make the winner the default `matrixTile`, keep every variant
        selectable by `SHRIKE_ATTN_MATRIX_TILE` (including `r32s4`/`r64s8` and
        `SHRIKE_PREFILL_ATTENTION=tiled` for the scalar kernel), extend
        `prefillAttentionPathDescription` (`RealForwardRunner.swift:189-196`)
        with `tile=<rawValue>` so the residency line records which geometry
        ran. Five gates: release build 0 warnings, lint, links,
        `swift test --no-parallel`, the same under TSAN.
  - [ ] Step 7: `tools/golden-baseline.sh --check 4` (M4 Pro, server stopped) —
        expected to differ (the PV reduction order changes with the tile
        geometry); recapture, record the before/after greedy digests, and
        commit `prefill: KV-head-group attention tiles on the matrix path
        (v12 P7)` with the baseline via `git commit --only`.
  - [ ] Step 8: ledger on both boxes (fresh server, 3.7k + 12k + 25k on the
        M4 Pro, 3.7k + 12k on the mini, `tools/prefill-measure.sh`); the mini
        memory check at 25k (`memory_pressure -Q` free span, server RSS) as P2
        recorded; mini `tools/mini-deploy.sh --restart`, mini golden recapture,
        scp into `baselines/`. Verdict line with `prefill_attn_router` before →
        after at 12k and 25k on both boxes, the achieved share of the ceiling,
        and the modelled-vs-measured traffic cut; design doc: the "Follow-ons"
        entry becomes a landed step with its ledger rows and the attention
        paragraph at `:174-183` updated; task review by a fresh reviewer; fixes
        folded into the commit.

### Task 8: P8 — the MPP GEMM core

- [x] **P8: the MPP GEMM core** — target on the mini at 12k:
  `prefill_routed_tile` 3.13 → ≤ 2.70 ms/prompt-token, `prefill_gdn_router`
  3.43 → ≤ 3.20, `prefill_shared_expert` 0.29 → ≤ 0.25 (3.7k: 3.23 → ≤ 2.80,
  3.44 → ≤ 3.21, 0.29 → ≤ 0.25); GPU busy 9.01 → ≈ 8.3. Bar: **≥ 80 % of the
  mini's same-run 128-row gate/up ceiling** — `ShrikeBench routed_gemm 20` on
  the ornith tile at **≤ 7.0 ms per tile, ≥ 0.92 TFLOPS against the 1.147 the
  same run measures** — up from P6's 8.24 ms / 0.782 / 68 %
  (`docs/v12-implementation-plan.md:947-955`,
  `docs/v12-prefill-matrix-kernels.md:271-282`). Why 80 %: the third P6 left is
  two named terms inside the kernel and this task attacks both, but double
  buffering can only hide a dequant behind a matmul long enough to hide it and
  the M1's per-group matmul (64×32×64) is small — 80 % claims 12 of the 32
  available points, a bit over a third of what is left; the whole third would be
  claiming the dequant is free. **The mini decides.** M4 Pro iteration check:
  bench 1.67 → ≤ 1.40 ms per tile (the same 80 % of its same-run 5.77), 12k
  roles routed 0.69 → ≈ 0.60, gdn 0.64 → ≈ 0.60, shared 0.06 → 0.05 (noise
  floor) — and its **wall may not move**: that loop is fetch-bound at 3.9 ms per
  tile and P6's GPU saving reappeared as routed→routed gap (`:277-283`). The
  mini kept its GPU savings at P6 and P7.
  **LANDED 727a9be (2026-09-02) AS A MEASURED NULL: neither knob beats the P6 kernel on
  the mini — `ShrikeBench routed_gemm 20`, grouped ms per ornith tile, n32b1
  (control) 8.25, n32b2 8.26, n64b1 9.69 (+17 %), n64b2 10.21 (+24 %); M4 Pro
  1.67 / 1.68 / 2.32 / 2.31.** The bars stand unmet; the roles are unchanged —
  3.7k and 12k rows on both boxes within noise of P7's (mini 12k routed 3.14,
  gdn 3.43, shared 0.29, wall 118.9 s; 3.7k 3.24 / 3.46 / 0.29, 36.3 s) — the
  default is bit-identical to what P6 shipped, the tests assert it for every
  variant, and golden is IDENTICAL on both boxes and both profiles.
  What landed is the templated body (`mpp_prefill_affine_body<TILE_N, BUFFERS>`,
  eight instantiations, the bare names = `n32b1`), `TileVariant` with the two
  env knobs as same-binary A/B overrides (P5's precedent for a null), the
  `tile_n=` / `buffers=` residency line and the bit-equality tests. Why the
  knobs fail: double buffering hides latency and the dequant is throughput
  work on the same 128 threads as the matmul; the 64-wide tile's doubled fp32
  accumulators cost more occupancy than the halved A re-reads save. The
  bench-only dequant probe (reverted) gives the mini's cost ledger for the
  8.25 ms tile: unpack arithmetic 0.57 ms (7 %), int4 weight byte loads 1.65
  (20 %), the staged-matmul structure — tile writes, one barrier and one
  64×32×64 `run` per K group — 2.53 (31 %), the plain GEMM 3.50 (42 %); the
  same structure costs 9 % on the M4 Pro (91 % of its ceiling without the
  dequant). The mini's same-run ceiling read 1.84 TFLOPS today against 1.147 in
  P6's run with the grouped tile unchanged, so the control's honest share is
  42 % on the mini (67 % on the M4 Pro), and P6's "68 %" was against a
  depressed ceiling. Two levers follow from the ledger, outside this brief:
  vectorised int4 weight loads (one `uint4` per 32 values) for the 20 %, and a
  128-wide K tile (two quant groups per staged tile, half the barriers and
  runs; not bit-identical) for the 31 %. The GDN role's GEMM share stays
  unmeasured (no role moved). Five gates green; golden digests unchanged.

  **What P6 and P7 left.** After P7 the mini's 12k prefill spends 6.85 of its
  9.01 ms/prompt-token of GPU busy — **76 %** — in three roles that run through
  the two kernels of `tensorops.metal` (`v12-prefill-matrix-kernels.md:286-302`):

  | role (mini, 12k) | ms/tok | kernel, from the code |
  | --- | ---: | --- |
  | `prefill_routed_tile` | 3.13 | `mpp_prefill_affine_grouped_f16` (P6): gate `n=512 k=2048`, up `n=512 k=2048`, down `n=2048 k=512` (`PrefillGroupedRoutedMoE.swift:899-929`), plus grouped gather/scatter and `silu_mul_fp16` |
  | `prefill_gdn_router` | 3.43 | `mpp_prefill_affine_threadgroup_f16` for five projections per layer (`RealForwardRunner.swift:3908-3962` in-proj + z/a/b, `:4000-4010` out-proj, via `encodeAffineProjection`'s MPP branch at `:3599-3614`) **plus** the P4 chunked scan, conv, QK-norm, gated norm |
  | `prefill_shared_expert` | 0.29 | `mpp_prefill_affine_threadgroup_f16`, three GEMMs at the same expert shapes (`PrefillSharedExpert.encodeChunk:102-158`) |

  The GDN row is the table's honest gap: **what share of `prefill_gdn_router` is
  GEMM is unmeasured on the mini.** The only split on record is the M4 Pro's
  after P4 — ≈ 0.07 of 0.64 is scan, ≈ 0.57 "projections, conv and norms"
  (`:172-177`) — which bounds it from above without naming it, and the M1 has
  never been split. So the bench is the gate and the −7 % GDN bar is a floor
  that holds only if the projections are ≳ 45 % of the role; if the role moves
  less, record the implied share rather than calling the task short.
  `prefill_attn_router` (mini 2.05 at 12k) also runs Q/K/V/O through the same
  kernel, so P8's reach is wider than 76 % — but the attention *core* dominates
  that role, so it is not in the bars.

  **The change.** Two knobs on the one inner loop both kernels share
  (`tensorops.metal:75-110`, `:197-230`), today per K group: cooperatively
  dequantize the 32×64 weight tile into threadgroup memory →
  `threadgroup_barrier` → `matmul2d` → fp32 accumulate → a second
  `threadgroup_barrier` whose only job is keeping the *next* group's dequant
  writes off the tile the matmul just read.

  1. **`SHRIKE_MPP_DEQUANT_BUFFERS=1|2`** — two alternating tiles: dequant group
     0 → barrier → per group `g` { dequant `g+1` into `buf[(g+1)&1]` (skipped
     past the last) ; `run` on `buf[g&1]` ; accumulate ; barrier }. The dequant's
     loads issue before the matmul that hides them and the write-after-read
     barrier goes: **two barriers per K group become one.** At `k=2048` that is
     32 groups per gate/up GEMM, 8 per down.
  2. **`SHRIKE_MPP_TILE_N=32|64`** — a 64-wide N tile halves the A-tile
     re-reads: the `n=2048` down GEMM's column tiles drop 64 → 32, gate/up
     16 → 8. At a 1,024-row wave (16 row tiles) the grid goes 1,024 → 512
     threadgroups for down, 256 → 128 for gate/up — still ≫ the mini's 8 cores,
     so P6's "fill the GPU" property survives. Weight traffic per output element
     is unchanged; the cost is registers.

  **Threadgroup memory and registers**, against the 32,768-byte per-threadgroup
  budget at the 4 simdgroups / 128 threads both kernels dispatch today
  (`matmul2d<descriptor, execution_simdgroups<4>>`, `tensorops.metal:44`,
  `:159`; `threadExecutionWidth * 4`, `MPPPrefillInt4QMM.swift:151`, `:253`):

  | variant | weight tile (fp16) | bytes | of 32 KB | dequant/thread | fp32 accum. regs/thread |
  | --- | --- | ---: | ---: | ---: | ---: |
  | `n32b1` (today) | 32×64 | 4,096 | 12.5 % | 16 elems | 2 × (64·32/128) = 32 |
  | `n32b2` | 2 × 32×64 | 8,192 | 25 % | 16 | 32 |
  | `n64b1` | 64×64 | 8,192 | 25 % | 32 | 2 × (64·64/128) = 64 |
  | `n64b2` | 2 × 64×64 | **16,384** | **50 %** | 32 | 64 |

  All four fit. **The 64×64 double-buffered tile is the one to watch** — not
  because it overflows, but because 16 KB is half the budget *and* it doubles
  the cooperative-tensor register footprint (two destination tensors,
  `accumulator` and `groupProduct`, `:61-64`, `:183-186`). How many threadgroups
  stay co-resident under 16 KB and 64 accumulator registers is not something the
  code can tell us; the Step 5 arm measures it. Both destination tensors stay —
  collapsing them would change the reduction order, out of scope.

  **Barriers and hazards.** The surviving barrier carries both edges: it
  publishes group `g+1`'s dequant writes to the matmul reading them next
  iteration, and orders group `g`'s tile reads before that buffer's refill two
  iterations later. Keep the prologue barrier after the group-0 dequant (without
  it the first `run` reads an unwritten tile) and the guard against dequantizing
  past `groupsPerRow` (`:72`, `:194`). And the CPU→GPU edge P6 ruled on: **the
  block tables stay inline via `setBytes`** (`MPPPrefillInt4QMM.swift:222-231`),
  never a shared buffer rewritten per wave, because the next tile is encoded
  while this one runs (`v12-prefill-matrix-kernels.md:459-462`). Double
  buffering is entirely inside the threadgroup and adds no host-side edge;
  `useResource(..., usage: .read)` per view (`:246-248`) is issued on every
  variant path, and no encoder becomes `.concurrent`.

  **Numerics**, per knob — "never recapture for an unexplained mismatch" binds:
  - `DEQUANT_BUFFERS=2` **preserves the per-element K reduction order exactly**:
    the group loop, the in-group `matmul2d` and the `accumulator[element] +=
    groupProduct[element]` sequence are untouched, only *when* a tile is filled
    moves. Output must be **bit-identical**, the tests assert fp16 equality
    against the current kernel, and `tools/golden-baseline.sh --check` is
    expected **IDENTICAL** on both boxes and both profiles. A difference is a
    bug, not a numerics change: do not recapture.
  - `TILE_N=64` leaves the group loop and the accumulate order alone too, but
    changes the descriptor's N, and whether MPP's *in-group* K schedule depends
    on N is not stated in the headers — Step 4 measures it. Bit identical → keep
    the bit assertion, golden expected IDENTICAL. Otherwise the committed bar is
    the chapter's 2e-2 on `maxAbs` and `rel`
    (`PrefillSharedExpertTests.swift:434-436`), golden differs on the long
    profile and is recaptured once per box with the digests and the named reason
    — "the 64-wide `matmul2d` descriptor reorders the in-group K reduction".

  **Files:**
  - Modify: `sources/Shrike/Metal/TensorCore/tensorops.metal:29-121`, `:141-242`
    — one `template <int TILE_N, int BUFFERS>` body plus two macro instantiation
    blocks, in the style of `attention_matrix.metal:212-233` and `:427-449`
    (threadgroup arrays declared in the macro, passed to the body). The two
    bodies differ only in the row origin, the store's row bound (`M` at `:117`
    vs the block's `rowEnd` at `:238`) and where the weight/scale/bias pointers
    come from, so those become body parameters; `:10` (`kMPPAffineTileN`) becomes
    the `n32*` instantiations' argument.
  - Modify: `sources/Shrike/Kernels/TensorCore/MPPPrefillInt4QMM.swift:25-29`
    (`tileN` moves from a static to the variant), `:31-39` + `:45-81` (init takes
    the variant and builds **only its two pipelines**, as `PrefillAttention.init`
    does at `PrefillAttention.swift:106-129`, so startup cost is unchanged),
    `:148`, `:250` (dispatch width from `variant.tileN`). `tileM` (64) and
    `tileK` are untouched, so `PrefillGroupedRoutedMoE.groupedRowTile` (`:92`)
    and the wave planner are unaffected.
  - Modify: `sources/Shrike/Runtime/Inference/RealForwardRunner.swift:185-187` —
    `prefillProjectionPath` gains ` tile_n=<n> buffers=<b>`; the leading token
    stays `affine-threadgroup-f16`, so `ServerInference.swift:819` and
    `tools/mini-deploy.sh:79`'s `prefill_projection_path=[a-z0-9-]*` grep keep
    matching.
  - Modify: `sources/ShrikeBench/RoutedGEMMBench.swift:15-18` and
    `sources/Shrike/Kernels/Prefill/MoE/PrefillRoutedGEMMBenchmark.swift:6-20` —
    print `variant=` so the arms are self-labelling. Nothing else in the bench
    changes: `:30` builds `MPPPrefillInt4QMM(context:weightBits:)`, which takes
    the variant from the environment through the static default.
  - Test: `tests/.../TensorCore/MPPPrefillInt4QMMTests.swift` — `makeInputs` /
    `makeBuffer` / `cpuReference` / `runShape` (`:28-205`) are the oracle,
    `:272-322` the fp16 byte-equality idiom;
    `tests/.../Prefill/PrefillSharedExpertTests.swift:365-438` (`chunkD = 2048`
    / `chunkF = 512` at `:14-15`, rows 64 and 33) is the real-shape,
    32-K-group, ragged-M oracle;
    `tests/.../Prefill/PrefillGroupedRoutedMoETests+Execution.swift:228-373`
    (`FourExpertTile`; `d = f = 64` is **one** K group, so it proves the block
    table and the store guard under a wider N tile, not the double buffering),
    `:437-502`, `:556-594`, `:596-704`.
  - Unchanged deliberately: the P6 grouped kernel's name and default shape (bare
    `mpp_prefill_affine_grouped_f16` is the `n32b1` instantiation and must emit
    byte-for-byte what it does today), the P3 per-expert path (`:735-846`) with
    its `SHRIKE_PREFILL_ROUTED_GEMM=per-expert` A/B, `matrixPathMinimumRows`, the
    scalar fallback, the wave planner, staging sizes.

  **Interfaces:**
  - Consumes: `MPPPrefillInt4QMM.encode` / `.encodeGrouped` signatures unchanged
    (`:96-105`, `:161-174`); `MetalContext.moduleLibrary(device:module:"tensorops")`
    (`:260-262`). The `globalN < N` store guard (`:85`, `:207`) already covers a
    ragged N, so `TILE_N = 64` adds no shape gate and `k.isMultiple(of: tileK)`
    is unchanged.
  - Produces:

    ```swift
    extension MPPPrefillInt4QMM {
        enum TileVariant: String, CaseIterable, Sendable {
            case n32b1, n32b2, n64b1, n64b2
            var tileN: Int              // n32* 32, n64* 64
            var dequantBuffers: Int     // *b1 1, *b2 2
            /// `n32b1` keeps the two shipped names; the rest append "_<rawValue>".
            var kernelName: String
            var groupedKernelName: String
            init?(tileN: String?, buffers: String?)
        }
        /// `SHRIKE_MPP_TILE_N` (32|64) and `SHRIKE_MPP_DEQUANT_BUFFERS` (1|2)
        /// name the variant; anything unrecognised keeps the measured choice.
        static let tileVariant: TileVariant
        var tileN: Int { variant.tileN }
        // init(context:weightBits:variant:) with the static as its default, so
        // a test pins a variant without a process-wide env var — the shape
        // PrefillAttention(context:supportsMLA:matrixTile:) already uses.
    }
    ```

    Metal. Function constants cannot serve here (`kMPPAffineTileN` is a
    `constexpr` argument of `matmul2d_descriptor` and a `threadgroup` array
    bound), so the variants are separate kernels behind separate pipelines:

    ```metal
    template <int TILE_N, int BUFFERS>
    static void mpp_prefill_affine_body(
        device const uint8_t* packedWeights, device const bfloat* scales,
        device const bfloat* biases, device half* activations, device half* output,
        uint N, uint K, int32_t rowOrigin, uint rowEnd,
        uint3 tgid, uint lid, uint threads,
        threadgroup half* weightTile);  // TILE_N * kMPPAffineTileK * BUFFERS halves
    // MPP_AFFINE_KERNEL(NAME, TILE_N, BUFFERS) — the 8 shipped args;
    // MPP_GROUPED_KERNEL(...) adds blobs, blocks, rowTileBlock. Instantiated as
    // mpp_prefill_affine_{threadgroup,grouped}_f16{,_n32b2,_n64b1,_n64b2},
    // the bare names being (32, 1).
    ```

    with `static_assert(TILE_N * kMPPAffineTileK * BUFFERS * 2 <= 32768)` in the
    body, as `attention_matrix.metal:278` does for its own budget. Lint:
    `MPPPrefillInt4QMM.swift` has no `function_body_length` baseline entry and
    `encodeGrouped` is already 98 lines (`:161-258`); if variant plumbing pushes
    it past 120, extract the guard block (`:177-205`) into a private validator
    rather than growing the body.

  Steps (TDD; the bit-equality claim is proved before any perf work):

  - [ ] Step 1: failing test `doubleBufferedDequantIsBitIdenticalToTheSingleBuffered`
        in `MPPPrefillInt4QMMTests.swift`, reusing `makeInputs`/`makeBuffer`
        (`:28-87`): `.n32b1` against `.n32b2` over shapes with more than one K
        group — `(m 64, n 32, k 128)`, `(m 33, n 512, k 2048)` (gate/up with a
        ragged M), `(m 128, n 2048, k 512)` (down) — asserting fp16 equality
        element-for-element with the `:272-322` idiom, plus finiteness. FAIL:
        `TileVariant` and `init(context:weightBits:variant:)` undefined.
  - [ ] Step 2: failing test `wideNTileMatchesTheNarrowTile` — the same three
        shapes, `.n64b1` and `.n64b2` against `.n32b1` at `RelError.maxAbsDiff
        ≤ 2e-2` and `RelError.compute ≤ 2e-2`, plus `cpuReference` (`:89-121`) at
        `runShape`'s own bars (`maxAbs ≤ 0.03`, `rel ≤ 1e-3`) so a wider tile
        cannot pass by matching a broken neighbour. FAIL: no `n64*` kernels.
  - [ ] Step 3: failing tests at the call sites —
        `runChunkSharedExpertMatchesRowLoop`
        (`PrefillSharedExpertTests.swift:365-438`) over all four variants at rows
        64 and 33; `groupedGEMMsMatchThePerExpertPathAcrossAllExperts`
        (`:437-502`), `groupedGEMMsMatchTheScalarPathAcrossWaves` (`:556-594`)
        and `groupedGEMMsHandleASingleOnePairExpert` (`:596-704`) over all four
        by giving `FourExpertTile` (`:228-373`) a `variant` parameter — the block
        table, the padded-row `-77` sentinel and the store guard under a 64-wide
        N tile are what those cover. `swift test --no-parallel --filter
        "MPPPrefillInt4QMM|PrefillSharedExpert|PrefillGroupedRoutedMoE"` → FAIL.
  - [ ] Step 4: implement the templated body, the eight instantiations, the
        `TileVariant` enum, the env parsing and the per-variant pipeline build;
        Steps 1–3 PASS. Then the probe: tighten `wideNTileMatchesTheNarrowTile`
        to fp16 bit equality and run it. Passes → land the assertion. Fails →
        revert to the 2e-2 bar, record the first differing shape and element, and
        carry "golden differs, one recapture per box" into Step 7.
  - [ ] Step 5 (spike, the gate): `swift run -c release ShrikeBench routed_gemm 20`
        per arm under `env SHRIKE_MPP_TILE_N=… SHRIKE_MPP_DEQUANT_BUFFERS=…` —
        four arms, `n32b1` the P6 control — on the M4 Pro first, then on the
        **mini, which decides**. `tools/mini-deploy.sh` copies only
        ShrikeServer/ShrikeCLI/ShrikeRepack (`:24`, `:36`, `:47`), so
        `scp .build/release/ShrikeBench` and the `*.bundle` directories into
        `~/shrike-runtime/bin/`, stop production (`pgrep` check first), run the
        four arms, relaunch with `tools/mini-deploy.sh --restart`. Record
        per-tile ms, TFLOPS and the same-run `gemm_expert_gateup_128rows` /
        `gemm_expert_down_128rows` ceilings per arm on both boxes. **Accept rule:
        the arm with the lowest mini per-tile ms among those at ≥ 80 % of the
        mini's same-run gate/up ceiling (≤ 7.0 ms). Arms within 3 % are a tie
        (P6 saw 4 % between nominally identical staging arms) and a tie breaks
        toward the lower `TILE_N`, because `n32b2` is bit-identical and keeps
        golden identical. If no arm reaches 80 %, take the mini's best above P6's
        68 % and record the achieved share; if none beats 68 %, land the tests
        only and say so.** Verdict line: the eight-row table.
  - [ ] Step 6: make the winner the default `tileVariant`, keep all four
        selectable on the same binary, and extend `prefillProjectionPath`
        (`RealForwardRunner.swift:185-187`) with `tile_n=` / `buffers=` so the
        residency line records which variant ran. Five gates: release build 0
        warnings, `swiftlint lint --strict --baseline .swiftlint-baseline.json`,
        markdown link check, `swift test --no-parallel`, the same under
        `env TSAN_OPTIONS=suppressions=tsan-suppressions.txt … --sanitize=thread`.
  - [ ] Step 7: `tools/golden-baseline.sh --check` (M4 Pro, server stopped).
        **Expected IDENTICAL, short and long, if the winner is `n32b2`** — the K
        reduction order is provably unchanged and no recapture is allowed;
        expected to differ on the long profile only if the winner carries
        `TILE_N=64` *and* Step 4's probe showed the descriptor reorders the
        in-group reduction, in which case recapture once per box and record the
        before/after greedy digests (P7's are M4 Pro long `f24c61565618fdab`, M1
        long `39c38734e8f9d22a`). Commit `prefill: double-buffered dequant and
        wider N tile in the MPP GEMM (v12 P8)`, baseline via `git commit --only`
        if there is one.
  - [ ] Step 8: ledger on both boxes (fresh server, `tools/prefill-measure.sh`;
        **mini 3.7k + 12k is the verdict**, M4 Pro 3.7k + 12k + 25k the check);
        `tools/mini-deploy.sh --restart`, mini golden check (recapture only if
        Step 7 established a deliberate change), scp into `baselines/`. Verdict
        line: the three roles before → after at 3.7k and 12k on the mini with the
        M4 Pro's rows as the check; the achieved bench share on both boxes and
        the winning arm; the implied GEMM share of `prefill_gdn_router` on the
        mini, which this task is the first to bound; the golden outcome per box;
        wall on both boxes, noting that the M4 Pro's routed saving may sit in the
        gap. Design doc: the "**The routed GEMM's last third.**" follow-on
        (`docs/v12-prefill-matrix-kernels.md:525-532`) becomes a landed
        `### Step 7 — the MPP GEMM core (…)` section after `:468`, an "**After
        P8**" ledger block after `:286-323`, and the routed row of "Where the
        time goes" (`:334`) gains the new share. Task review by a fresh reviewer;
        fixes folded into the commit (rebase and amend, never a fixup commit).

### Task 9: P9 — vectorised int4 weight loads and a 128-wide K tile in the MPP GEMM

- [x] **P9: the two levers P8's probe exposed** — target on the mini at 12k:
  `prefill_routed_tile` 3.13 → ≤ 2.60 ms/prompt-token, `prefill_shared_expert`
  0.29 → ≤ 0.25, `prefill_gdn_router` 3.43 → ≤ 3.30 (a floor, see below); at
  3.7k 3.24 → ≤ 2.70, 0.29 → ≤ 0.25, 3.46 → ≤ 3.33; GPU busy 9.01 → ≈ 8.3 at
  12k and 8.44 → ≈ 7.7 at 3.7k. Bar: **`ShrikeBench routed_gemm 20` on the
  ornith tile (8 experts × 128 rows, d 2048, f 512, 6.44 GFLOP, staging 1024),
  grouped ms per tile on the mini — ≤ 6.6 ms (≥ 0.98 TFLOPS, 53 % of the
  mini's 1.84 same-run gate/up ceiling) for an arm carrying both levers,
  ≤ 7.3 ms (≥ 0.88 TFLOPS, 48 %) for a single lever** — against P8's control
  of 8.25 ms / 0.78 TFLOPS / 42 % (`docs/v12-implementation-plan.md:1487-1490`,
  `docs/v12-prefill-matrix-kernels.md:505-531`). Why those two numbers: the P8
  probe splits the mini's 8.25 ms tile into unpack arithmetic 0.57 (7 %), int4
  weight byte loads 1.65 (20 %), the staged-matmul structure 2.53 (31 %) and
  the plain GEMM 3.50 (42 %). Lever A takes the load term plus the part of the
  unpack term that is per-element address arithmetic; lever B halves the
  barriers and the `matmul2d` runs but **not** the threadgroup tile writes, so
  at most half of 2.53. Model: A ≈ 6.6–6.9, B ≈ 7.0, A+B ≈ 5.7. The 6.6 bar
  claims 1.65 ms of a modelled 2.6 — the load lever in full and nothing from
  the structure lever; the 7.3 bar claims 0.95 of a modelled 1.3.
  **The mini decides.** M4 Pro iteration check only: bench 1.66 → ≈ 1.45 (A+B),
  ≈ 1.55 (A), ≈ 1.60 (B); 12k roles routed 0.69 → ≈ 0.62, gdn 0.64 → ≈ 0.62,
  shared 0.06 → 0.05 (noise floor), busy 1.78 → ≈ 1.70 — that box's *entire*
  probe budget above a plain GEMM is 0.51 ms of a 1.66 ms tile (load 0.10,
  unpack 0.30, structure 0.11), so a null there decides nothing, and its **wall
  may not move**: the loop is fetch-bound at 3.9 ms per tile and P6's GPU saving
  reappeared as routed→routed gap (`v12-prefill-matrix-kernels.md:277-283`).
  **LANDED 560c888 (2026-09-03): measured on the mini `prefill_routed_tile` 3.24 →
  1.92 ms/prompt-token (3.7k; 3.14 → 1.87 at 12k), `prefill_shared_expert`
  0.29 → 0.17 (both), `prefill_gdn_router` 3.46 → 2.38 (3.7k; 3.43 → 2.37 at
  12k), GPU busy 8.44 → 5.64 (3.7k) and 9.00 → 6.26 (12k), wall 36.3 → 25.7 s
  and 118.9 → 86.0 s.** Every bar is cleared by a wide margin (routed ≤ 2.60,
  shared ≤ 0.25, gdn ≤ 3.30; the 6.6 ms tile bar against a measured 4.92).
  Bench, grouped ms per ornith tile on the mini: byte loads 9.37 (P8's run of
  the same arm read 8.25 — the two-body kernel's byte fallback is ≈ 13 %
  slower there and is now only the unaligned-base path) → vector 6.19 → vector
  + 128-wide K 4.92, 1.31 TFLOPS = 71 % of the same-run 1.84 ceiling (42 %
  before); the 128-wide tile alone 8.83. M4 Pro check: 1.70 → 1.22 → 1.13 ms
  (99 % of its ceiling), 12k roles routed 0.69 → 0.53, gdn 0.64 →
  0.60, shared 0.06 → 0.05, busy 1.78 → 1.60, wall 30.2 → 29.6 s
  (3.7k busy 1.69 → 1.59, wall 10.8 → 10.9 s; 25k busy 2.32 → 1.65, wall 77.5 →
  64.7 s — the loop is fetch-bound there, so the GPU saving reappears partly
  as routed→routed gap). Winner `n32k128b1` with vector loads, the
  default; every arm selectable (`SHRIKE_MPP_TILE_K`, `SHRIKE_MPP_WEIGHT_LOADS`).
  The GDN role moved 31 %, the first bound this chapter has for its GEMM
  share on the mini. Numerics: lever A is bit-identical (asserted on regular
  and full-mantissa inputs, at an unaligned offset and at 8 bits); lever B is
  not — MPP's 128-wide run reorders the K reduction, last-ulp differences on
  ≈ 0.15 % of elements (`ShrikeBench mpp_compare`: 372 of 262,144 at 4 bits,
  max abs 0.000488; 401 at 8 bits, max abs 0.0078; identical counts on both
  boxes) — within the 2e-2 bar against the CPU reference. The Step 4
  bit-equality probe passed and was wrong: the tests' inputs (integers over 64,
  a few bf16 scales) have fp32-exact partial sums that no reduction order can
  disturb; the probe now uses full-mantissa data and the wide-K tests assert
  the 2e-2 bar. Golden: short identical on both boxes; long differs on both
  and was recaptured once per box with that reason — sha256 M4 Pro
  f24c61565618fdab → 6fc391949c998e66, M1 39c38734e8f9d22a → 899a25e60a365e60
  (the mini's output returned to its pre-P6 text: the same near-tie sentence
  P6 and P7 moved). One mini recapture had run before the cause was measured
  (the leg-3 wrapper recaptures on mismatch) and was reverted; it was redone
  after `mpp_compare` bounded the difference. Five gates green (1154 tests,
  TSAN 0 reports); build + lint re-run on the final tree.

  **What P8 left.** The probe ledger for the mini's 8.25 ms tile, and the same
  split on the M4 Pro (`.superpowers/sdd/v12-implementation-plan/task-8-report.md`,
  "The dequant probe"):

  | term | mini ms | share | M4 Pro ms | share |
  | --- | ---: | ---: | ---: | ---: |
  | unpack arithmetic | 0.57 | 7 % | 0.30 | 18 % |
  | int4 weight byte loads | 1.65 | 20 % | 0.10 | 6 % |
  | staged-matmul structure | 2.53 | 31 % | 0.11 | 7 % |
  | the GEMM itself (same-run ceiling) | 3.50 | 42 % | 1.15 | 69 % |
  | **full tile** | **8.25** | | **1.66** | |

  The weight traffic is not the constraint: one grouped tile reads each expert's
  1.57 MB of packed int4 twice (128 rows = 2 row tiles of 64), 25 MB per tile,
  3 GB/s at 8.25 ms — an order under the M1's bandwidth. The 1.65 ms is issue
  slots and address arithmetic, not DRAM, which is why a wider load can take
  most of it.

  **Lever A — one vector load per thread per staged tile.** `mpp_affine_value`
  (`tensorops.metal:13-26`) computes `bit_offset = element * bits` and loads one
  byte (two only when a field straddles, which 4-bit at `bit_offset % 8 ∈ {0,4}`
  and 8-bit never do), so the 4-bit path issues exactly one byte load per
  dequantized element: 2,048 loads per 32×64 tile, 16 per thread at the 128
  threads both kernels dispatch (`threadExecutionWidth * 4`,
  `MPPPrefillInt4QMM.swift:162`, `:264`). Replace `mpp_affine_dequant_tile`'s
  `linear` → (localN, localK) walk (`tensorops.metal:43-58`, K fastest) with a
  **chunk** walk: thread `lid` takes chunk `c`, covering `E` consecutive
  elements of one row's K range, `E = bytesPerLoad * 8 / bits`.
  - `localN = c * E / TILE_K`, `localK0 = (c * E) % TILE_K`, and because `E`
    divides `TILE_K` a chunk never straddles a row **or a quant group** — the
    element's group is `(tileIndex * TILE_K + localK0) / kW4A8GroupSize`,
    computed once per chunk instead of once per element, and the
    `globalN < N` zero-fill guard (`:48`, `:56`) becomes one test per chunk.
  - Picking the load width. At `TILE_N 32, TILE_K 64` the tile is 2,048
    elements = 1,024 packed bytes at 4 bits, so a `uint4` (16 B = 32 int4
    values) gives **64 loads over 128 threads — half the threads idle in the
    dequant phase**, giving back exactly what the lever buys (P8: the dequant is
    throughput work on the matmul's own threads). Two rows per thread does not
    fix it — rows are `rowBytes` apart, never contiguous. The two mappings that
    keep all 128 threads loading are a `uint2` (8 B = 16 values) at `TILE_K 64`,
    or a `uint4` once the tile's K width doubles, which is lever B. **Take
    both:** `bytesPerLoad = 8` for 4-bit at `TILE_K == 64`, `16` otherwise —
    `E = 16` at `TILE_K 64` (both bit widths), `E = 32` at `TILE_K 128` (4-bit
    one `uint4`, 8-bit two), and 2,048 loads per tile become 128 in every case.
    `bits` is a function constant (`:11`), so the choice folds at pipeline
    build; at 8 bits the `>> shift & mask` chain collapses to a byte extract
    from the loaded word. Keep the loop strided
    (`for c = lid; c < chunksPerTile; c += threads`) so a device whose
    `threadExecutionWidth` is not 32 still covers the tile, and store each chunk
    with vector `half` writes so the store count falls with the load count —
    whether the compiler already vectorises today's scalar store is not
    something the source can tell us, and Step 5 measures the pair, not halves.
  - **Bit-identical.** Same integer `q`, same `scale`/`bias` from the same
    indices, same `half(fma(...))`, same tile slot, same matmul, same
    accumulate order. Only the addressing changes.

  **Lever B — `TILE_K = 128`, two quant groups per staged tile.** The
  descriptor's K doubles (`matmul2d_descriptor(kMPPAffineTileM, TILE_N, TILE_K,
  false, true, false)`, `tensorops.metal:87-89`); `tilesPerRow` replaces
  `groupsPerRow = K / kW4A8GroupSize` (`:120`) and halves — 32 → 16 iterations
  per gate/up GEMM at k 2048, 8 → 4 for down — halving the barrier count
  (`:156`) and the `run` count (`:148-152`) with it. The dequant already applies
  each element's own scale and bias, and the chunk mapping above keeps one
  (scale, bias) pair per chunk, so the group boundary inside the tile costs
  nothing. The weight tile is 32×128 halves = 8 KB.
  - **Not bit-identical.** MPP reduces K 128 in one run where today two 64-runs
    are summed in fp32 through `accumulator[element] += groupProduct[element]`
    (`:153-155`). P8's Step 4 probe established that the in-run K order does not
    depend on the descriptor's **N**; it says nothing about K, and there is no
    reason it should carry over. Expect a difference; Step 4 measures it.

  **Threadgroup memory and registers**, against the 32,768-byte budget at 4
  simdgroups / 128 threads (`tensorops.metal:90`, `:183`, `:239`):

  | variant | weight tile (fp16) | bytes | of 32 KB | elems/thread | loads/thread (4-bit / 8-bit) | fp32 accum. regs/thread |
  | --- | --- | ---: | ---: | ---: | --- | ---: |
  | `n32b1` (default) | 32×64 | 4,096 | 12.5 % | 16 | 1×`uint2` / 1×`uint4` | 2 × (64·32/128) = 32 |
  | `n32k128b1` (new) | 32×128 | 8,192 | 25 % | 32 | 1×`uint4` / 2×`uint4` | 32 |
  | `n32b2` (P8) | 2 × 32×64 | 8,192 | 25 % | 16 | 1×`uint2` / 1×`uint4` | 32 |
  | `n64b1` / `n64b2` (P8) | 64×64 / 2 × 64×64 | 8,192 / 16,384 | 25 % / 50 % | 32 | 1×`uint4` / 2×`uint4` | 64 |

  The row that matters: **`TILE_K` does not touch the accumulator footprint.**
  The destination cooperative tensors are M×N (`:109-113`), so doubling K leaves
  32 fp32 registers per thread where `TILE_N = 64` doubled them to 64 — and P8's
  reading of why `n64*` lost was exactly that occupancy cost. What B does add is
  4 KB of threadgroup memory and whatever MPP stages internally for a 64×128 A
  fragment, which the source cannot tell us. `static_assert` becomes
  `TILE_N * TILE_K * BUFFERS * 2 <= 32768`.

  **Alignment — what the code guarantees, and the fallback.** A vector load of
  W bytes needs the thread's address `base + globalN * rowBytes + chunkBytes`
  aligned to W.
  - `rowBytes = K * bits / 8` (`tensorops.metal:119`) and `encode` already
    requires `k.isMultiple(of: Self.tileK)` with `tileK = Quantization.groupSize
    = 64` (`MPPPrefillInt4QMM.swift:26`, `:120`, `:204`;
    `Quantization.swift:5`), so `rowBytes` is a multiple of 32 at 4 bits and of
    64 at 8 bits — **always 16-byte aligned.**
  - `chunkBytes` is a multiple of 8 (4-bit, `TILE_K 64`) or 16 (every other
    case), from `E` dividing `TILE_K`.
  - The **base** is the open question, and **it is not guaranteed**.
    `weightsOffset >= 0` is the only weight guard in either encoder (`:121`,
    `:205`); odd offsets are legal today and
    `affineThreadgroupCandidateMatchesFP32AffineReference` exercises
    `weightOffset: 13` (`MPPPrefillInt4QMMTests.swift:324`). Upstream the buffer
    *bases* are page aligned — the expert pool is `posix_memalign`ed to 2 MiB
    with a page-rounded slot stride (`PreadExpertStreamer.swift:220`, `:341-345`,
    `:382`, `:403`) — but both offset spaces are running cursors with **no
    padding**: `wOff` over per-expert sub-tensor sizes
    (`RepackPlanner.swift:772-791`), resident attention/GDN tensors from a
    16 KB-aligned `indexSize` (`RepackPlanner.swift:373-401`, read back at
    `Model.swift:404-421`). For every shape the six models issue those running
    sizes happen to be multiples of 16 (a packed int4 projection is
    `rows * cols / 2` with `cols % 64 == 0`; bf16 scales/biases are
    `rows * cols / 64 * 2`), but one odd-sized tensor would misalign everything
    after it.
  - So the vector path needs a **runtime-uniform fallback, not a function
    constant**: alignment varies per *dispatch* (a different tensor, a different
    `wOff`), while a function constant is fixed at pipeline build and would
    double the pipeline count for something the host already knows. Add one
    `constant uint&` (plain kernel index 8, grouped index 11) set from
    `weightsOffset % 16 == 0`, branched on once outside the chunk loop —
    threadgroup-uniform, no divergence, at the cost of both dequant bodies
    staying in the binary. Keep `mpp_affine_value` as the fallback body; the
    `weightOffset: 13` test becomes its coverage.

  **K divisibility — which production GEMMs are `K % 128 == 0`.** Every GEMM on
  the MPP path, from the arch configs (`ModelTypes.swift:197-235`, `:242-271`,
  `:286-314`, `:319-357`) and the call sites:

  | GEMM | K | `% 128` |
  | --- | ---: | --- |
  | routed gate/up then down, ornith & qwen36 (`PrefillGroupedRoutedMoE.swift:896-929`: `n: f, k: d` then `n: d, k: f`) | 2048, 512 | ✓ |
  | shared expert gate/up, down (`PrefillSharedExpert.swift:135-140`, `:155-156`) | 2048, 512 | ✓ |
  | attention q/kv then o, ornith (`RealForwardRunner.swift:3589-3619`, the MPP branch of `encodeAffineProjection`) | 2048, 4096 | ✓ |
  | GDN in-proj / z / a / b (`:3908-3962`, `columns: D`), out-proj (`:4001-4011`, `columns: la.valueDim`) | 2048, 4096 | ✓ |
  | kimi MLA q / kv-a and KDA in-proj (`:4033-4055`, `columns: D`); routed, shared and the layer-0 dense MLP | 2304, 1024, 9216 | ✓ |
  | **gpt-oss-20b attention q/kv (`hiddenSize 2880`), routed gate/up and down (`moeIntermediateSize 2880`)** | **2880** | **✗** (45 K groups, odd) |

  The test file's `selectedProductionAttentionShapesMatchCurrentPolicy`
  (`MPPPrefillInt4QMMTests.swift:347-370`) pins K 2816 / 4096 / 8192, all
  multiples of 128; those are a policy fixture, not this model's shapes.
  **Rule: `MPPPrefillInt4QMM.tileK` stays the static 64 and the 128-wide tile is
  picked per dispatch.** It has to stay 64 because it is also the *admission*
  test for the whole matrix path — `PrefillSharedExpert.matrixPath` (`:34-35`)
  and `PrefillGroupedRoutedMoE.matrixPath` (`:552-553`) refuse the path when
  `d`/`intermediate` is not a multiple of it, so raising the static would drop
  gpt-oss off the matrix path entirely and onto the scalar kernels. Instead an
  instance whose variant carries `TILE_K 128` builds **both** pipeline pairs —
  its own and the `n32b1` sibling's — and `encode`/`encodeGrouped` select
  `k.isMultiple(of: 128) ? wide : narrow`. No masked K tail: the second half of a
  ragged last tile would zero the B side, but the A tensor is
  `activations + tile * TILE_K` with a K extent of `TILE_K` and a row stride of
  `K` (`tensorops.metal:143-146`), so the run would read a following row — NaN or
  Inf times zero is not zero, and the last row reads past the buffer. Falling
  back to the 64-wide kernel costs nothing and is bit-identical.

  **Variants and knobs.** Extend `TileVariant` rather than adding a parallel
  enum: one new case `n32k128b1`, `tileK` a computed property (64 for the four
  P8 cases, 128 for the new one), a third env var `SHRIKE_MPP_TILE_K=64|128`
  folded into the existing `init?`, and `TILE_K = 128` accepted only with
  `TILE_N 32` / one buffer (anything else rejects the whole selection, as an
  unrecognised value already does). The four P8 raw values keep their spelling
  so the P8 verdict's arm names stay quotable and `n32b1` keeps the bare kernel
  names. Lever A is **not** a variant: it is the per-dispatch uniform above,
  with `SHRIKE_MPP_WEIGHT_LOADS=byte|vector` forcing the flag off for the A/B —
  P5's and P8's precedent of a same-binary override, and it costs **zero**
  instantiations. Total: the template becomes
  `<TILE_N, TILE_K, BUFFERS>` over `(32,64,1) (32,64,2) (64,64,1) (64,64,2)
  (32,128,1)` — **5 combinations × 2 macros = 10 kernels**, up from eight. Two
  more `matmul2d` instantiations is the whole cost of the count P8's review
  flagged, and each test still builds one `MetalContext`; the spike needs no
  more than this, because its four arms are the P8 default, A alone, B alone and
  A+B, all at N 32 / one buffer.

  **Numerics**, per lever — "never recapture for an unexplained mismatch" binds:
  - **Lever A is bit-identical.** The tests assert fp16 element-for-element
    equality against `n32b1` with byte loads, over the three P8 shapes plus the
    odd-offset and the 8-bit cases, and `tools/golden-baseline.sh --check` is
    expected **IDENTICAL** on both boxes and both profiles. A difference is a
    bug: do not recapture.
  - **Lever B is not.** The committed bar is the chapter's 2e-2 on `maxAbs` and
    `rel` against `n32b1` *and* against `cpuReference` at `runShape`'s own bars
    (`MPPPrefillInt4QMMTests.swift:184-203`), so a 128-wide tile cannot pass by
    matching a broken neighbour. Golden is expected to differ on the **long**
    profile only (the short prompt never reaches the matrix path,
    `v12-prefill-matrix-kernels.md:320-322`) and is recaptured once per box with
    the before/after greedy digests and the named reason — "MPP reduces K 128 in
    one run where the 64-wide tile summed two runs in fp32". Digests to carry
    forward: M4 Pro long `f24c61565618fdab`, M1 long `39c38734e8f9d22a`
    (P7's, unchanged through P8).

  **Files:**
  - Modify: `sources/Shrike/Metal/TensorCore/tensorops.metal:13-26` (keep
    `mpp_affine_value` as the unaligned fallback), `:28-59`
    (`mpp_affine_dequant_tile<TILE_N>` becomes `<TILE_N, TILE_K>` with the chunk
    walk and the two load bodies), `:60-168` (`mpp_prefill_affine_body` gains
    `TILE_K` and the `vectorLoads` uniform; `groupsPerRow` at `:120` becomes
    `tilesPerRow = K / TILE_K`; `:88`, `:98-103`, `:144-146` take `TILE_K`;
    `:86`'s `static_assert` takes `TILE_K`), `:170-194` and `:215-250` (the two
    macros gain the argument and the fifth instantiation each).
  - Modify: `sources/Shrike/Kernels/TensorCore/MPPPrefillInt4QMM.swift:26`
    (`tileK` stays 64 and gains a one-sentence why — it is the matrix path's
    admission test at two call sites, not the kernel's tile width), `:29-34`
    (the third env var), `:35-36`, `:54-90` (init builds the variant's pair plus
    the `n32b1` pair when `variant.tileK == 128`), `:107-167` and `:172-269`
    (pick the pipeline from `k.isMultiple(of: variant.tileK)`, set the alignment
    uniform), `:276-327` (`TileVariant` gains `n32k128b1`, `tileK`, and a
    `WeightLoads` enum with its static).
  - Modify: `sources/Shrike/Runtime/Inference/RealForwardRunner.swift:185-188` —
    `prefillProjectionPath` becomes
    `affine-threadgroup-f16 tile_n=32 tile_k=128 buffers=1 loads=vector`. The
    leading token is unchanged, so `ServerInference.swift:819` and
    `tools/mini-deploy.sh:79`'s `prefill_projection_path=[a-z0-9-]*` grep keep
    matching.
  - Modify: `sources/Shrike/Kernels/Prefill/MoE/PrefillRoutedGEMMBenchmark.swift:15`,
    `:147` (a `weightLoads` field beside `variant`) and
    `sources/ShrikeBench/RoutedGEMMBench.swift:15-18` (print `loads=` beside
    `variant=`) so every arm is self-labelling. `:180`'s
    `MPPPrefillInt4QMM(context:weightBits:)` keeps taking both from the
    environment through the statics.
  - Test: `tests/.../TensorCore/MPPPrefillInt4QMMTests.swift` —
    `makeInputs`/`makeBuffer`/`cpuReference`/`runShape` (`:28-205`) are the
    oracle, `runPair` (`:213-244`) the pair harness, `:255-262` the fp16
    byte-equality idiom; `tests/.../Prefill/PrefillSharedExpertTests.swift:184-187`,
    `:362-461` (the real 2048/512 shapes, 32 K groups, ragged M);
    `tests/.../Prefill/PrefillGroupedRoutedMoETests+Execution.swift:228-373`
    (`FourExpertTile`), `:438-502`, `:558-594`, `:598-704`.
  - Unchanged deliberately: `MPPPrefillInt4QMM.tileM` (64) and the static
    `tileK` (64), so `PrefillGroupedRoutedMoE.groupedRowTile` (`:92`), the wave
    planner and both `matrixPath` gates are untouched; the four P8 variants and
    their two env knobs; the P3 per-expert path (`PrefillGroupedRoutedMoE.swift:816-834`)
    with its `SHRIKE_PREFILL_ROUTED_GEMM=per-expert` A/B; `matrixPathMinimumRows`;
    the scalar fallback; staging sizes; the inline `setBytes` block tables
    (`MPPPrefillInt4QMM.swift:233-243`), which P6 ruled must not become a shared
    buffer.

  **Interfaces:**
  - Consumes: `encode` / `encodeGrouped` signatures unchanged (`:107-118`,
    `:172-185`); `MetalContext.moduleLibrary(device:module:"tensorops")`
    (`:271-273`). The `globalM < rowEnd && globalN < N` store guard
    (`tensorops.metal:159-167`) already covers a ragged M and N.
  - Produces:

    ```swift
    extension MPPPrefillInt4QMM {
        enum TileVariant: String, CaseIterable, Sendable {
            case n32b1, n32b2, n64b1, n64b2, n32k128b1
            var tileN: Int          // n32* 32, n64* 64
            var tileK: Int          // n32k128b1 128, the rest 64
            var dequantBuffers: Int
            var kernelName: String  // n32b1 keeps the bare names
            var groupedKernelName: String
            /// `tileK == 128` is instantiated only at N 32 / one buffer; any
            /// other combination rejects the selection, as an unrecognised
            /// value already does.
            init?(tileN: String?, tileK: String?, buffers: String?, fallback: TileVariant)
        }
        enum WeightLoads: String, CaseIterable, Sendable { case byte, vector }
        /// `SHRIKE_MPP_TILE_N` (32|64), `SHRIKE_MPP_TILE_K` (64|128) and
        /// `SHRIKE_MPP_DEQUANT_BUFFERS` (1|2) name the variant;
        /// `SHRIKE_MPP_WEIGHT_LOADS` (byte|vector) is a per-dispatch override.
        static let tileVariant: TileVariant
        static let weightLoads: WeightLoads
        // init(context:weightBits:variant:weightLoads:) with the statics as
        // defaults, so a test pins both without a process-wide env var.
    }
    ```

    Metal:

    ```metal
    template <int TILE_N, int TILE_K>
    static inline void mpp_affine_dequant_tile(
        threadgroup half* tile, device const uint8_t* packedWeights,
        device const bfloat* scales, device const bfloat* biases,
        uint tileIndex, uint columnTile, uint N, uint rowBytes,
        uint groupsPerRow, uint bits, bool vectorLoads, uint lid, uint threads);
    template <int TILE_N, int TILE_K, int BUFFERS>
    static inline void mpp_prefill_affine_body(/* … , bool vectorLoads, … */);
    // MPP_AFFINE_KERNEL(NAME, TILE_N, TILE_K, BUFFERS) and
    // MPP_GROUPED_KERNEL(...) instantiate
    // mpp_prefill_affine_{threadgroup,grouped}_f16{,_n32b2,_n64b1,_n64b2,_n32k128b1},
    // the bare names being (32, 64, 1).
    ```

    Lint: `MPPPrefillInt4QMM.swift` has no `function_body_length` baseline entry,
    so a new violation fails the strict gate. `encode` is 61 lines (`:107-167`)
    and has room; `encodeGrouped` is 98 (`:172-269`) and the pipeline pick plus
    the uniform add ~6 — if it crosses 120, extract the guard block (`:186-216`)
    into a private validator rather than growing the body. Comments: the repo
    rule holds. Two earn their place — why the static `tileK` stays 64 (it gates
    two `matrixPath` call sites, not the kernel's tile), and why `% 16` is a
    sufficient alignment test (the row stride is already a multiple of 16 from
    the `k % 64` guard). Nothing else.

  Steps (TDD; lever A's bit-identity is proved before any perf work):

  - [ ] Step 1: failing test `vectorWeightLoadsAreBitIdenticalToByteLoads` in
        `MPPPrefillInt4QMMTests.swift`, `runPair` (`:213-244`) over
        `variantShapes` (`:207-211`: `(64, 32, 128)`, `(33, 512, 2048)`,
        `(128, 2048, 512)`) with `.byte` against `.vector` on `.n32b1`,
        asserting fp16 equality element-for-element and finiteness; plus the
        8-bit instance at `(33, 35, 128)` and the unaligned case — a
        `runPair` variant taking `weightOffset: 13` so the vector instance
        provably falls back to the byte body and still matches. FAIL:
        `WeightLoads` and `init(context:weightBits:variant:weightLoads:)`
        undefined.
  - [ ] Step 2: failing test `wideKTileMatchesTheNarrowTile` — `.n32k128b1`
        against `.n32b1` over the same three shapes (all `K % 128 == 0`) at
        `RelError.maxAbsDiff ≤ 2e-2` and `RelError.compute ≤ 2e-2`, plus
        `runShape(compareCPUReference: true)` at its own bars (`maxAbs ≤ 0.03`,
        `rel ≤ 1e-3`); and `wideKTileFallsBackOnARaggedK` — `(64, 32, 192)` and
        the gpt-oss shape `(33, 128, 2880)`, both `% 64` but not `% 128`,
        asserting the `.n32k128b1` instance still returns
        `.affineThreadgroupF16` and is **bit-identical** to `.n32b1` because it
        took the narrow pipeline. FAIL: no `n32k128b1` kernel.
  - [ ] Step 3: failing tests at the call sites.
        `PrefillSharedExpertTests.chunkSharedExpertMatchesRowLoop(rows:variant:)`
        (`:184-187`) picks up the fifth variant through `allCases` (2 rows × 5 =
        10 cases); add one non-crossed case pinning `.vector` on `.n32b1` rather
        than crossing the two axes — the cross product belongs in
        `MPPPrefillInt4QMMTests`, where the kernels are compared directly.
        `PrefillGroupedRoutedMoETests+Execution`: **the fixture cannot run this
        task as written** — `FourExpertTile`'s `d = f = 64` (`:229-230`) is one K
        group at `TILE_K 64` and not a multiple of 128 at all, so a
        `.n32k128b1` arm would silently take the narrow fallback and prove
        nothing. Raise it to `d = f = 256` (4 K groups narrow, 2 wide, both loops
        exercised) — `makeSyntheticExpertPool(numExperts:d:f:)`
        (`PrefillGroupedRoutedMoETests.swift:115-119`) and every downstream size
        are already parameterised on `d`/`f`, and this also closes P8 review item
        #3, the grouped kernel's multi-group `BUFFERS == 2` loop having no test.
        Do the same for `groupedGEMMsHandleASingleOnePairExpert`'s local
        `d = f = 64` (`:601-602`). The three parameterised tests (`:438`, `:558`,
        `:598`) then cover all five variants. Leave the file's other
        `d = f = 64` fixtures (`:10-11`, `:112-113`, `:710`) alone — they are
        scalar/streamed-batched paths below `matrixPathMinimumRows` and never
        reach `encodeGrouped`. `swift test --no-parallel --filter
        "MPPPrefillInt4QMM|PrefillSharedExpert|PrefillGroupedRoutedMoE"` → FAIL.
  - [ ] Step 4: implement the chunk dequant with both load bodies, the
        `TILE_K` template axis and its two instantiations, the alignment
        uniform, the pipeline pick, `TileVariant.n32k128b1`, `WeightLoads` and
        the env parsing; Steps 1–3 PASS. Then the probe: tighten
        `wideKTileMatchesTheNarrowTile` to fp16 bit equality and run it. Passes →
        land the assertion and golden stays IDENTICAL for every arm. Fails →
        revert to the 2e-2 bar, record the first differing shape and element,
        and carry "golden differs on the long profile, one recapture per box"
        into Step 7.
  - [ ] Step 5 (spike, the gate): `swift run -c release ShrikeBench routed_gemm 20`
        per arm under `env SHRIKE_MPP_TILE_K=… SHRIKE_MPP_WEIGHT_LOADS=…` — four
        arms (`n32b1`+byte = the P8 control, `n32b1`+vector = A, `n32k128b1`+byte
        = B, `n32k128b1`+vector = A+B), all at N 32 / one buffer — on the M4 Pro
        first, then on the **mini, which decides**. On the mini run
        `tools/mini-deploy.sh` (copy only) **first**: it copies the `*.bundle`
        directories that carry the shader sources, and P8's bench first failed
        because only the `ShrikeBench` binary had been copied and the bundles
        were stale. Then `scp .build/release/ShrikeBench macmini:shrike-runtime/bin/`
        — `mini-deploy.sh:24`, `:36`, `:47` copy only ShrikeServer / ShrikeCLI /
        ShrikeRepack — `pgrep` for a running server, stop production, run the
        four arms at both staging sizes, and relaunch with
        `tools/mini-deploy.sh --restart`. Record per-tile ms, TFLOPS and the
        same-run `gemm_expert_gateup_128rows` / `gemm_expert_down_128rows`
        ceilings per arm on both boxes. **Accept rule: the lowest mini per-tile
        ms at staging 1024 among arms at or under their bar — ≤ 6.6 ms for an arm
        carrying both levers, ≤ 7.3 ms for a single lever. Arms within 3 % are a
        tie (P6 saw 4 % between nominally identical staging arms) and a tie
        breaks toward the bit-identical arm — vector loads without the wide K
        tile — because that keeps golden identical. If no arm reaches its bar but
        one beats the 8.25 ms control by ≥ 3 %, land it as the default and record
        the achieved share; if none beats the control, land the tests and the
        template only and say so, as P8 did.** Verdict line: the five-row table
        (four arms + ceiling) per box per staging size.
  - [ ] Step 6: make the winner the default (`tileVariant` and/or
        `weightLoads`), keep every arm selectable on the same binary, and extend
        `prefillProjectionPath` (`RealForwardRunner.swift:185-188`) with
        `tile_k=` / `loads=` so the residency line records which arm ran. Five
        gates: release build 0 warnings,
        `swiftlint lint --strict --baseline .swiftlint-baseline.json`, markdown
        link check, `swift test --no-parallel`, the same under
        `env TSAN_OPTIONS=suppressions=tsan-suppressions.txt swift test
        --no-parallel --sanitize=thread`.
  - [ ] Step 7: `tools/golden-baseline.sh --check` (M4 Pro, server stopped).
        **Expected IDENTICAL, short and long, if the winner is lever A alone** —
        the values and the reduction order are provably unchanged and no
        recapture is allowed. Expected to differ on the long profile only if the
        winner carries `TILE_K 128` *and* Step 4's probe showed the 128-wide run
        reorders the K reduction, in which case recapture once per box and record
        the before/after greedy digests (P7's, unchanged through P8: M4 Pro long
        `f24c61565618fdab`, M1 long `39c38734e8f9d22a`). Commit
        `prefill: vectorised int4 weight loads and a 128-wide K tile in the MPP
        GEMM (v12 P9)`, baseline via `git commit --only` if there is one.
  - [ ] Step 8: ledger on both boxes (fresh server, `tools/prefill-measure.sh`,
        one send per prompt per server lifetime; **mini 3.7k + 12k is the
        verdict**, M4 Pro 3.7k + 12k + 25k the check); `tools/mini-deploy.sh
        --restart`, mini golden check (recapture only if Step 7 established a
        deliberate change), scp into `baselines/`. Verdict line: the three roles
        before → after at 3.7k and 12k on the mini with the M4 Pro's rows as the
        check; the winning arm and the achieved bench share on both boxes; **the
        implied GEMM share of `prefill_gdn_router` on the mini** — that role's
        GEMM fraction is still unmeasured (the only split on record is the M4
        Pro's after P4, ≈ 0.07 of 0.64 scan and ≈ 0.57 "projections, conv and
        norms", `v12-prefill-matrix-kernels.md:172-177`), so the −4 % gdn bar is
        a floor that holds only if the projections are ≳ 45 % of the role; if it
        moves less, record the implied share rather than calling the task short.
        Also: the golden outcome per box; wall on both boxes, noting that the M4
        Pro's saving may sit in the routed→routed gap rather than the wall.
        Design doc: the two follow-on bullets (`:558-563`, `:564-571`) are
        removed and become a landed `### Step 8 — vectorised int4 weight loads
        and a 128-wide K tile (…)` section after `:531`; an "**After P9**" ledger
        block after the After P7 table (`:286-322`) if any role moves; the routed
        row of "Where the time goes" (`:334`) gains the new share. Plan: Task 9
        `[x]` with the landed paragraph. Task review by a fresh reviewer; fixes
        folded into the commit (rebase and amend, never a fixup commit).

### Task 10: P10 — a 256-wide K tile in the MPP GEMM

- [x] **P10: the structure half of what P9 left** — target on the mini at 12k:
  `prefill_routed_tile` 1.87 → ≤ 1.82 ms/prompt-token, `prefill_gdn_router`
  2.37 → ≤ 2.32; at 3.7k 1.92 → ≤ 1.87 and 2.38 → ≤ 2.33; GPU busy 6.26 →
  ≈ 6.05 at 12k and 5.64 → ≈ 5.44 at 3.7k, wall 86.0 → ≈ 83.5 s and 25.7 →
  ≈ 24.9 s. `prefill_attn_router` (1.75 → ≈ 1.72) and `prefill_shared_expert`
  (0.17 → ≈ 0.16) are **recorded, not barred**: their modelled movement is at
  the ledger's printing precision. Bar: **`ShrikeBench routed_gemm 20` on the
  ornith tile (8 experts × 128 rows, d 2048, f 512, 6.44 GFLOP), grouped ms
  per tile on the mini — ≤ 4.70 ms (≥ 1.37 TFLOPS, ≥ 74 % of the mini's
  same-run gate/up ceiling) at *both* staging sizes**, against P9's default
  arm re-run in the same process (4.92 ms / 1.31 TFLOPS / 71 % of a 1.835
  ceiling, `docs/v12-implementation-plan.md:1810-1818`,
  `docs/v12-prefill-matrix-kernels.md:559-596`). Why that number: the tile's
  floor is a plain GEMM at the same-run ceiling, 6.44 / 1.835 = 3.51 ms (P8's
  probe measured 3.50), so 4.92 − 3.51 = 1.41 ms — 29 % of the tile — is
  everything above a plain GEMM. A 256-wide tile takes only the part of that
  which scales with the tile *count*: barriers, `run` set-up, the
  `groupProduct` zero and the fp32 accumulate, and the per-chunk (scale, bias)
  pair. That term is **measured, not modelled**: P9's byte-load pair isolates
  it, because the byte dequant body's per-element work does not depend on
  `TILE_K` at all (`tensorops.metal:110-128`, a linear walk over
  `TILE_N * TILE_K` elements with `tilesPerRow` tiles — `K * TILE_N` elements
  per row either way), so the mini's 9.37 → 8.83 step at K64 → K128 is the
  halving of the structure term alone: ≈ 1.08 ms at K64, ≈ 0.54 at K128,
  ≈ 0.27 after one more halving. Model: 4.92 − 0.27 ≈ **4.65 ms** = 1.385
  TFLOPS = 75 % of the ceiling; the ≤ 4.70 bar claims 0.22 of a modelled 0.27.
  **The mini decides — and this time the M4 Pro cannot even confirm the
  lever.** There the same byte-load pair read 1.700 → 1.701 (a null): the
  structure term is ≈ 0.11 ms of a 1.66 ms tile by P8's probe and its halving
  did not register, and vector/K128 at 1.13 ms is already 99 % of that box's
  5.4–5.8 TFLOPS ceiling. Expect 1.10–1.15 ms and no role movement there; a
  null on the M4 Pro decides nothing.
  **LANDED 3710611 (2026-09-03): measured on the mini `prefill_routed_tile`
  1.92 → 1.79 ms/prompt-token (3.7k; 1.87 → 1.74 at 12k), `prefill_gdn_router`
  2.38 → 2.36 (3.7k; 2.37 → 2.34 at 12k), `prefill_attn_router` 0.96 → 0.95 /
  1.75 → 1.75, `prefill_shared_expert` 0.17 → 0.17, GPU busy 5.64 → 5.47 and
  6.26 → 6.10, wall 25.7 → 25.4 s and 86.0 → 84.2 s.** Bars: routed cleared
  (≤ 1.87 / ≤ 1.82); gdn missed (≤ 2.33 / ≤ 2.32 against 2.36 / 2.34); the
  4.70 ms tile bar cleared against a measured 4.48. Bench, grouped ms per
  ornith tile on the mini at staging 1024 / 2048: K64 6.23 / 5.90, K128
  4.93 / 4.83 (the control), K256 4.48 / 4.49 = 1.44 TFLOPS = 78 % of the
  same-run 1.83 ceiling (71 % before). M4 Pro check: 1.23 / 1.13 / 1.09 ms;
  its 12k roles and wall moved far more than the tile — a paired A/B on one
  binary (K128 → K256, back to back) read routed 0.55 → 0.44, gdn 0.62 →
  0.52, attention 0.38 → 0.33, busy 1.66 → 1.35 (−19 %), wall 31.7 → 28.0 s
  (−12 %); 3.7k wall 10.9 → 10.7, 25k 64.7 → 64.0 s (fetch-bound). Winner
  `n32k256b1`, the default; every arm selectable (`SHRIKE_MPP_TILE_K`,
  `SHRIKE_MPP_WEIGHT_LOADS`). The implied GDN GEMM share on the mini: ≈ 11 %
  at the tile's gain rate — the 4,096-row dense projections did not follow the
  tile on the M1 (−1 %) where they did on the M4 Pro (−17 %), so the draft's
  65 % model (from P9's −31 %, mostly the load lever) does not carry to the
  structure lever; the dense shape is box-dependent and has never been benched
  in isolation — a follow-on (design doc). Numerics: K256 reorders the K
  reduction against K128 (`ShrikeBench mpp_compare`: 456 of 262,144 at 4 bits,
  max abs 0.000488; 489 at 8 bits, max abs 0.0078; identical counts on both
  boxes) — within the 2e-2 bar against the CPU reference; no bit-equality is
  claimed between K widths. Golden: short identical on both boxes; long
  recaptured once on the M4 Pro, sha256 `6fc391949c998e66` →
  `e04d4e8ee7f1590d`; the M1's long held (`899a25e60a365e60` — the last-ulp
  differences crossed no greedy tie, P4's precedent). Five gates green (1161
  tests, TSAN 0 reports).

  **What P9 left.** The mini's 4.92 ms tile, split from measurements already
  on record (`.superpowers/sdd/v12-implementation-plan/task-9-report.md`,
  "Step 5 spike"; `docs/v12-prefill-matrix-kernels.md:526-558` for the P8
  probe):

  | term | mini ms | share | where it comes from | K256 takes |
  | --- | ---: | ---: | --- | --- |
  | plain GEMM at the same-run ceiling | 3.51 | 71 % | 6.44 GFLOP / 1.835 TFLOPS (`gemm_expert_gateup_128rows`, same run) | nothing |
  | staged structure per K tile | ≈ 0.54 | 11 % | 2 × the byte-load step 9.37 → 8.83 at K64 → K128 | **half** |
  | dequant: loads, unpack, tile writes | ≈ 0.87 | 18 % | the remainder | nothing |
  | **full tile** | **4.92** | | measured, P9 Step 5 | |

  The M4 Pro's tile has no measurable budget left: 1.13 ms against a
  1.12–1.19 ms floor at its 5.43–5.76 TFLOPS same-run ceiling.

  **The lever — `TILE_K = 256`, four quant groups per staged tile.** The
  template axis already exists; P9 built it. `mpp_prefill_affine_body<TILE_N,
  TILE_K, BUFFERS>` (`tensorops.metal:137-235`) puts `TILE_K` in the
  descriptor (`:157-159`), derives `tilesPerRow = K / TILE_K` (`:191`), and
  runs one `threadgroup_barrier` (`:227`), one `operation.run` (`:219-223`),
  one `groupProduct` zero over `get_capacity()` (`:197-199`) and one fp32
  accumulate (`:224-226`) per tile. Doubling `TILE_K` halves every one of
  those per row: 16 → 8 iterations for a gate/up GEMM at k 2048, 4 → 2 for
  down at k 512, 32 → 16 for an attention o-proj at k 4096.
  - The dequant needs nothing new at 4 bits. `mpp_affine_dequant_tile<TILE_N,
    TILE_K>` sets `E = TILE_K / 4` (`:69`), so a chunk is 64 elements =
    32 bytes = the existing two-`uint4` branch of `mpp_affine_load_words`
    (`:27-45`, the `else` at `:39-44`), and `chunksPerTile = TILE_N * 4 = 128`
    still equals the 128 threads both kernels dispatch
    (`MPPPrefillInt4QMM.swift:197-199`, `:303-305`).
  - `static_assert(kW4A8GroupSize % E == 0)` (`:72`) holds at E = 64 **exactly**
    — one chunk is one whole quant group. 256 is therefore the last K width
    this chunk mapping supports; at 512 the assert fires, and the follow-on
    after this one is the register-resident weight tile, not a 512-wide one.
  - `static_assert(TILE_K % kW4A8GroupSize == 0)` (`:155`) and
    `static_assert(TILE_N * TILE_K * BUFFERS * 2 <= 32768)` (`:156`) both hold:
    256 % 64 = 0, and 32 × 256 × 1 × 2 = 16,384 bytes, 50 % of the budget.

  **What it does not take**, and why the prize is one term and not two:
  - **The threadgroup tile writes.** 32 × K halves per row are staged either
    way; only the tile boundary moves.
  - **The vector loads.** `E` grows with `TILE_K`, so a 256-wide tile issues
    two `uint4`s per chunk where a 128-wide one issues one: 2,048 `uint4`
    loads per 2048-long row in both cases. This is the difference from P9,
    where K64 → K128 *also* halved the load instruction count (4,096 `uint2`
    → 2,048 `uint4`) — which is why lever B was worth 1.27 ms with vector
    loads but only 0.54 with byte loads. P10 gets the 0.54-shaped half only.
  - **The unpack arithmetic.** The same shift/mask/`fma` per element.

  **The 8-bit chunk overflows the word buffer — the one new piece of Metal.**
  At 8 bits and `TILE_K 256`, `chunkBytes = E * bits / 8` (`:73`) is **64**,
  but `mpp_affine_load_words` (`:27-45`) branches on 8, 16 and 32 bytes only
  and its caller's buffer is `uint words[8]` (`:87`) — 32 bytes. A 64-byte
  chunk needs four `uint4` loads into `uint words[16]`. This is not
  hypothetical: `weightBits` is `model.attentionWeightBits`
  (`RealForwardRunner.swift:724-727`, `Model.swift:57`), which the manifest may
  set to 8, and the suite already runs 8-bit wide-K shapes
  (`MPPPrefillInt4QMMTests.swift:359-363`). Without the fix the vector body
  reads the first half of each chunk and dequantises garbage into the second.
  Alignment is unaffected — four loads at +0/+16/+32/+48 still need only the
  16-byte base the host already tests (`MPPPrefillInt4QMM.swift:310-314`).

  **K divisibility — a three-rung ladder, and every rung has production
  traffic.** From the arch configs (`ModelTypes.swift:201-245`, `:242-271`,
  `:286-323`, `:324-357`, `LinearAttentionConfig:57-62`, `MLAConfig` at
  `:440-441`) and the call sites:

  | model (`.gturbo`) | GEMM | K | rung |
  | --- | --- | ---: | --- |
  | ornith15, qwen36 | attention q/k/v, GDN in-proj / z / a / b (`RealForwardRunner.swift:3910`, `:3931`, `:3942`, `:3954`, `:4184-4210`, `columns: D`) | 2048 | K256 |
  | ornith15, qwen36 | attention o (`:4373`, `columns: qDim`), GDN out-proj (`:4002`, `columns: la.valueDim`) | 4096 | K256 |
  | ornith15, qwen36 | routed gate/up then down (`PrefillGroupedRoutedMoE.swift:896-929`: `n: f, k: d` then `n: d, k: f`), shared expert (`PrefillSharedExpert.swift:135-138`, `:155-156`) | 2048, 512 | K256 |
  | ornith15-mtp, qwen36-mtp | the same shapes, one layer, full attention only | 2048, 4096, 512 | K256 |
  | kimi-linear-48b | KDA in-proj, MLA q / kv-a, KDA low-rank *first* legs (`:4126`, `:4148`, `columns: D`) | 2304 | K256 (9 × 256) |
  | kimi-linear-48b | KDA out-proj, MLA o-proj (`:4095-4105`, `columns: H * valueHeadDim`) | 4096 | K256 |
  | kimi-linear-48b | routed gate/up, down; the layer-0 dense MLP | 1024, 2304, 9216 | K256 |
  | **kimi-linear-48b** | **KDA low-rank second legs `linFB` / `linGB` (`:4137`, `:4159`, `columns: low` = `keyHeadDim`)** | **128** | **K128** |
  | **gpt-oss-20b** | **attention q/k/v (`hiddenSize` 2880), routed gate/up and down (`moeIntermediateSize` 2880)** | **2880** | **K64** (45 × 64, not a multiple of 128) |
  | gpt-oss-20b | attention o (`columns: qDim` = 64 × 64) | 4096 | K256 |

  **Rule: `MPPPrefillInt4QMM.tileK` stays the static 64** — it is the matrix
  path's *admission* test, not the kernel's tile width
  (`MPPPrefillInt4QMM.swift:25-30`, gating `PrefillSharedExpert.matrixPath`
  `:25-38` and `PrefillGroupedRoutedMoE.matrixPath` `:543-555`); raising it
  would drop gpt-oss off the matrix path entirely. P9's instance already
  builds a second pair when `variant.tileK != Self.tileK` (`:105-113`) and
  selects on `k.isMultiple(of: variant.tileK)` (`:165`, `:252-253`). P10 turns
  that binary choice into an ordered ladder — **the widest instantiated tile
  that divides `k`**. Still no masked K tail: the A tensor is
  `activations + tile * TILE_K` with a K extent of `TILE_K` and a row stride of
  `K` (`tensorops.metal:174-176`, `:214-218`), so a ragged last tile would read
  the following row (NaN or Inf × 0 is not 0) and the last row would read past
  the buffer. Falling to a narrower rung costs nothing.

  **Variants and knobs.** One new case `n32k256b1`, `tileK` returning 256
  (`MPPPrefillInt4QMM.swift:326-346`); `SHRIKE_MPP_TILE_K` accepts `256`
  (`:361-389`), and 256 — like 128 — is accepted only with `TILE_N 32` / one
  buffer, anything else rejecting the whole selection. `n32k128b1` stays
  selectable on the same binary as the A/B control, which is what the ledger
  legs compare against. `SHRIKE_MPP_WEIGHT_LOADS=byte|vector` is unchanged.
  The template becomes `(32,64,1) (32,64,2) (64,64,1) (64,64,2) (32,128,1)
  (32,256,1)` — **6 combinations × 2 macros = 12 kernels**, up from ten.
  `prefillProjectionPath` (`RealForwardRunner.swift:185-189`) prints
  `tile_k=256`; the leading token is unchanged, so `ServerInference.swift:819`
  and `tools/mini-deploy.sh:79`'s `prefill_projection_path=[a-z0-9-]*` grep
  keep matching.

  **Numerics — not bit-identical, and no bit-equality probe.** MPP reduces
  K 256 in one `run` where the 128-wide tile sums two runs in fp32
  (`tensorops.metal:224-226`), the same reordering P9 measured going 64 → 128,
  so **the task must not assert bit identity between K widths.** What it
  asserts is the chapter's 2e-2 on `maxAbs` and `rel` against `n32b1` *and*
  against `cpuReference` at `runShape`'s own bars
  (`MPPPrefillInt4QMMTests.swift:203-219`), so a 256-wide tile cannot pass by
  matching a broken neighbour, and it characterises the difference with
  `ShrikeBench mpp_compare` on both boxes before any golden decision. Every
  numerics test uses `makeInputs(irregular: true)`
  (`MPPPrefillInt4QMMTests.swift:28-37`) — P9's leg-4 lesson: the regular
  synthetic inputs (integers over 64, a few bf16 scales) have fp32-exact
  partial sums that no reduction order can disturb, so a bit-equality probe on
  them passes and proves nothing. Golden is expected to differ on the **long**
  profile only (the short prompt never reaches the matrix path,
  `docs/v12-prefill-matrix-kernels.md:320-322`) and is recaptured once per box
  with the before/after greedy digests and the named reason. Digests to carry
  forward (P9's): M4 Pro short `81b62049…806db` / long `6fc391949c998e66`,
  M1 short `5487823b…80ec9` / long `899a25e60a365e60`. Never recapture for an
  unexplained mismatch.

  **The prize, per role** (mini, 12k, after P9). The measured column is each
  role's own P9 movement — a *lower bound* on its GEMM share, since the GEMM
  cannot have gone to zero. The modelled column divides that by the bench
  tile's own −47.5 % (9.37 → 4.92) on the same commit; it assumes the dense
  projections sped up like the grouped tile, which is a model, not a
  measurement:

  | role | after P9 | measured Δ at P9 | modelled GEMM share | modelled ms in the body | modelled P10 saving |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | `prefill_gdn_router` | 2.368 | −31.0 % | 65 % | 1.54 | 0.085 |
  | `prefill_routed_tile` | 1.869 | −40.5 % | 85 % | 1.59 | 0.087 |
  | `prefill_attn_router` | 1.751 | −14.0 % | 29 % | 0.51 | 0.028 |
  | `prefill_shared_expert` | 0.170 | −41.4 % | 87 % | 0.15 | 0.008 |
  | **GPU busy** | **6.26** | −30.4 % | | **3.78 (60 %)** | **0.21** |

  So ≈ 60 % of the mini's prefill GPU time now runs through this one
  templated body, and P10's −5.5 % of it is ≈ 0.21 ms/token: busy 6.26 →
  ≈ 6.05, wall 7.00 → ≈ 6.79 ms/token (86.0 → ≈ 83.5 s), closing about a
  third of the remaining gap to the chapter's 6.3 target. The GDN role's
  share is the one to watch: the ledger's Task 4 note has only the M4 Pro's
  split (≈ 0.07 of 0.64 was the chunked scan,
  `docs/v12-prefill-matrix-kernels.md:172-177`), and 65 % here is the first
  modelled figure for the mini — Step 9 records the implied share again.

  **Files:**
  - Modify: `Sources/Shrike/Metal/TensorCore/tensorops.metal:27-45`
    (`mpp_affine_load_words` gains a 64-byte branch — four `uint4` loads at
    +0/+16/+32/+48), `:87` (`uint words[8]` → `words[16]`), `:266` and `:324`
    (`MPP_AFFINE_KERNEL(mpp_prefill_affine_threadgroup_f16_n32k256b1, 32, 256,
    1)` and `MPP_GROUPED_KERNEL(mpp_prefill_affine_grouped_f16_n32k256b1, 32,
    256, 1)`). The body (`:137-235`) and the dequant walk (`:53-128`) are
    unchanged — `TILE_K` is already the template axis.
  - Modify: `Sources/Shrike/Kernels/TensorCore/MPPPrefillInt4QMM.swift:54-57`
    (the two `narrow*Pipeline` fields become one ordered ladder, widest first),
    `:105-113` (init builds every instantiated rung at or below
    `variant.tileK` — 256, 128, 64 — reusing `makePipeline`), `:165` and
    `:252-253` (one private `pipeline(forK:)` / `groupedPipeline(forK:)`
    selecting the widest rung dividing `k`), `:326-346` (`n32k256b1` and its
    `tileK`), `:361-389` (the env init accepts `"256"` under the same N 32 /
    one-buffer rule). Extracting the selector matters for lint: `encodeGrouped`
    is 102 lines (`:207-308`) and `MPPPrefillInt4QMM.swift` has **no**
    `function_body_length` baseline entry, so a new violation fails the strict
    gate.
  - Modify: `Sources/Shrike/Kernels/Prefill/MoE/PrefillRoutedGEMMBenchmark.swift:238-249`
    — `compareTileK` takes the two variants instead of hard-coding `.n32b1` and
    `.n32k128b1`, so the K128-vs-K256 difference can be bounded on the mini,
    which has no test suite; `Sources/ShrikeBench/ShrikeBench.swift:66-75`
    passes them and names the pair in the printed line.
    `Sources/ShrikeBench/RoutedGEMMBench.swift:15` already prints `variant=`
    and `loads=` — every arm is self-labelling, nothing to change.
  - Modify: `Sources/Shrike/Runtime/Inference/RealForwardRunner.swift:185-189`
    — no code change; `tile_k=` picks up 256 from `variant.tileK`.
  - Test: `Tests/Shrike/Core/Kernels/TensorCore/MPPPrefillInt4QMMTests.swift`
    (`runShape` `:139-221`, `runPair` `:229-266`, `expectBitIdentical`
    `:316-323`, `variantShapes` `:223-227`, the wide-K tests `:348-402`, the
    env test `:404-425`);
    `Tests/Shrike/Core/Kernels/Prefill/PrefillSharedExpertTests.swift`
    (the `allCases` parameterisation picks up the sixth variant);
    `Tests/Shrike/Core/Kernels/Prefill/PrefillGroupedRoutedMoETests+Execution.swift:250-251`
    and `:604-605` (`d = f = 256` is exactly **one** 256-wide tile, so the
    multi-tile loop would go untested — raise to 512: 2 wide tiles, 4 at K128,
    8 at K64).
  - Unchanged deliberately: `MPPPrefillInt4QMM.tileM` (64) and the static
    `tileK` (64), so `PrefillGroupedRoutedMoE.groupedRowTile`, the wave planner
    and both `matrixPath` gates are untouched; the five existing variants and
    their env knobs; `SHRIKE_MPP_WEIGHT_LOADS` and the byte fallback body;
    `vectorLoadsFlag` (`:310-314`); `matrixPathMinimumRows`; the P3 per-expert
    path with its `SHRIKE_PREFILL_ROUTED_GEMM=per-expert` A/B; staging sizes;
    the inline `setBytes` block tables (`:269-278`), which P6 ruled must not
    become a shared buffer.

  **Interfaces:**
  - Consumes: `encode` / `encodeGrouped` signatures unchanged (`:140-149`,
    `:207-220`); `MetalContext.moduleLibrary(device:module:"tensorops")`
    (`:316-318`). The `globalM < rowEnd && globalN < N` store guard
    (`tensorops.metal:231-235`) already covers a ragged M and N.
  - Produces:

    ```swift
    extension MPPPrefillInt4QMM {
        enum TileVariant: String, CaseIterable, Sendable {
            case n32b1, n32b2, n64b1, n64b2, n32k128b1, n32k256b1
            var tileK: Int          // n32k256b1 256, n32k128b1 128, the rest 64
            /// `tileK` 128 and 256 are instantiated only at N 32 / one buffer;
            /// any other combination rejects the selection.
            init?(tileN: String?, tileK: String?, buffers: String?, fallback: TileVariant)
        }
        /// Widest instantiated K tile that divides `k`: 256 → 128 → 64. The
        /// static `tileK` (64) stays the matrix path's admission unit.
        private func pipeline(forK k: Int) -> MTLComputePipelineState?
        private func groupedPipeline(forK k: Int) -> MTLComputePipelineState?
    }
    public enum PrefillRoutedGEMMBenchmark {
        // compareTileK(context:m:n:k:bits:narrow:wide:) — the pair is a
        // parameter, so `mpp_compare` can bound K128 vs K256 as well.
    }
    ```

    Metal:

    ```metal
    // words[16]; a 64-byte chunk (8 bits × TILE_K 256) loads four uint4.
    static inline void mpp_affine_load_words(thread uint* words,
                                            device const uint8_t* src,
                                            uint chunkBytes);
    MPP_AFFINE_KERNEL(mpp_prefill_affine_threadgroup_f16_n32k256b1, 32, 256, 1)
    MPP_GROUPED_KERNEL(mpp_prefill_affine_grouped_f16_n32k256b1, 32, 256, 1)
    ```

    Comments: the repo rule holds. At most one earns its place — why E = 64 is
    the widest chunk the quant group allows (the `static_assert` says *that* it
    holds, not that 256 is therefore the last supported width). Nothing else.

  Steps (TDD; the ladder and the 8-bit loader are proved before any perf work):

  - [ ] Step 1: failing test `wideK256TileMatchesTheNarrowTile` in
        `MPPPrefillInt4QMMTests.swift` — `.n32k256b1` against `.n32b1` over
        `(64, 32, 256)`, `(33, 512, 2048)`, `(128, 2048, 512)` (all
        `K % 256 == 0`; note `variantShapes`' first entry K 128 is **not**, so
        the task needs its own list) at `RelError.maxAbsDiff ≤ 2e-2` and
        `RelError.compute ≤ 2e-2` on `makeInputs(irregular: true)`, plus
        `runShape(compareCPUReference: true, irregular: true)` at its own bars
        (`maxAbs ≤ 0.03`, `rel ≤ 1e-3`). No bit-equality assertion, ever.
        FAIL: `TileVariant.n32k256b1` undefined.
  - [ ] Step 2: failing test `wideK256TileTakesTheWidestRungThatDividesK` —
        `(64, 32, 384)` (`% 128`, not `% 256`) asserts the `.n32k256b1`
        instance is **bit-identical to `.n32k128b1`** (same kernel, same
        reduction order — a real property), and `(33, 128, 2880)` (`% 64`
        only, the gpt-oss shape) asserts it is bit-identical to `.n32b1`; both
        on `irregular: true` inputs, without which the rungs are
        indistinguishable. Extend `tileVariantsParseTheirEnvironmentNames`
        (`:404-425`) for `"256"`, the rejected `(64, 256)` and `(256, 2
        buffers)` combinations, and `n32k256b1.kernelName`. FAIL: today's
        binary selection sends K 384 to the K64 pair, so the first assertion
        fails on a real mismatch.
  - [ ] Step 3: failing test `wideK256TileMatchesTheNarrowTileAtEightBits` —
        an 8-bit `.n32k256b1` instance at `(33, 35, 256)` and `(33, 512,
        2048)`, byte loads against vector loads asserted **bit-identical**
        (lever A's property, which does hold within one K width), and against
        `.n32b1` at 2e-2. FAIL: `chunkBytes` is 64 and
        `mpp_affine_load_words` has no 64-byte branch, so the vector body
        dequantises only the first half of each chunk. This is the test that
        pins the loader bug; it must exist before the instantiation lands.
  - [ ] Step 4: failing tests at the call sites.
        `PrefillSharedExpertTests.chunkSharedExpertMatchesRowLoop(rows:variant:)`
        picks up the sixth variant through `allCases` (2 rows × 6 = 12 cases).
        `PrefillGroupedRoutedMoETests+Execution`: raise `FourExpertTile`'s
        `d = f = 256` (`:250-251`) and `groupedGEMMsHandleASingleOnePairExpert`'s
        (`:604-605`) to 512 — at 256 a K256 arm runs exactly one tile and the
        multi-tile loop is never exercised; `makeSyntheticExpertPool(numExperts:d:f:)`
        and every downstream size are already parameterised. Add
        `groupedWideK256InstanceTakesTheK128Rung` at `d = f = 384`, asserting
        the grouped `.n32k256b1` instance is bit-identical to the grouped
        `.n32k128b1` across waves. `swift test --no-parallel --filter
        "MPPPrefillInt4QMM|PrefillSharedExpert|PrefillGroupedRoutedMoE"` → FAIL.
  - [ ] Step 5: implement — the 64-byte load branch and `words[16]`, the two
        instantiations, `TileVariant.n32k256b1`, the pipeline ladder and its
        selector, the env parsing. Steps 1–4 PASS. Then, before any golden
        work, extend `compareTileK` to take the pair and run
        `swift run -c release ShrikeBench mpp_compare` locally at K128 vs K256
        (4 and 8 bits), recording mismatch count, max abs and max rel. P9's
        lesson in order: measure the difference first, decide about golden
        second.
  - [ ] Step 6 (spike, the gate): `swift run -c release ShrikeBench routed_gemm 20`
        per arm under `env SHRIKE_MPP_TILE_K=64|128|256` at the default vector
        loads — three arms plus the two ceiling GEMMs the mode already runs
        (`RoutedGEMMBench.swift:20-21`) — on the M4 Pro first, then on the
        **mini, which decides**. On the mini run `tools/mini-deploy.sh` (copy
        only) **first**: it copies the `*.bundle` directories that carry the
        shader sources (`:39-45`), and P8's bench first failed because only the
        binary had been copied. Then
        `scp .build/release/ShrikeBench macmini:shrike-runtime/bin/` —
        `mini-deploy.sh:24`, `:36`, `:47` copy only ShrikeServer / ShrikeCLI /
        ShrikeRepack — `pgrep` for a running server, stop production, run the
        three arms at both staging sizes, relaunch with
        `tools/mini-deploy.sh --restart`. **Accept rule: `n32k256b1` must clear
        ≤ 4.70 ms at staging 1024 *and* 2048 against the `n32k128b1` control
        re-run in the same process.** The modelled prize (−5.5 %) is barely
        outside the bench's own tie band (P6 saw 4 % between nominally
        identical staging arms), so one staging size is not a verdict and
        arms within 3 % are a tie — a tie goes to `n32k128b1`, which needs no
        recapture and two fewer pipelines. If K256 beats the control by ≥ 3 %
        at both staging sizes but misses 4.70, land it as the default and
        record the achieved share. If it does not, land the tests, the 8-bit
        loader fix and the instantiation, keep `n32k128b1` the default, and
        say so — P8's precedent. Verdict line: the four-row table (three arms
        + ceiling) per box per staging size.
  - [ ] Step 7: if it won, make `n32k256b1` the default
        (`MPPPrefillInt4QMM.swift:36-42`), every arm still selectable on the
        same binary; the residency line reads
        `affine-threadgroup-f16 tile_n=32 tile_k=256 buffers=1 loads=vector` on
        both boxes. Five gates: release build 0 warnings,
        `swiftlint lint --strict --baseline .swiftlint-baseline.json`, markdown
        link check, `swift test --no-parallel`, the same under
        `env TSAN_OPTIONS=suppressions=tsan-suppressions.txt swift test
        --no-parallel --sanitize=thread`.
  - [ ] Step 8: `tools/golden-baseline.sh --check` (M4 Pro, server stopped).
        Expected: **short IDENTICAL, long differs**, recaptured once per box
        with Step 5's `mpp_compare` numbers as the named reason. A short-profile
        mismatch is a bug — do not recapture. Commit
        `prefill: a 256-wide K tile in the MPP GEMM (v12 P10)`, baseline via
        `git commit --only`.
  - [ ] Step 9: ledger on both boxes (fresh server per prompt,
        `tools/prefill-measure.sh`, one send per prompt per server lifetime;
        **mini 3.7k + 12k is the verdict**, M4 Pro 3.7k + 12k + 25k the check);
        `tools/mini-deploy.sh --restart`, mini golden recapture, scp into
        `baselines/`. Design doc: a landed `### Step 9 — a 256-wide K tile
        (…)` section after `docs/v12-prefill-matrix-kernels.md:596`; an
        "**After P10**" ledger block after the After P9 table (`:324-350`); the
        routed row of "Where the time goes" (`:361`) gains the new share; the
        follow-on bullet `:630-635` is rewritten to drop the 256-wide step and
        keep the register-resident question (and to record that 256 is the
        last width the chunk mapping supports). Plan: Task 10 `[x]` with the
        landed paragraph. Task review by a fresh reviewer; fixes folded into
        the commit (rebase and amend, never a fixup commit).

  **Verdict line template** (the controller fills in the measured values):

  > **LANDED `<sha>` (`<date>`): measured on the mini `prefill_routed_tile`
  > 1.92 → `<x>` ms/prompt-token (3.7k; 1.87 → `<x>` at 12k),
  > `prefill_gdn_router` 2.38 → `<x>` (3.7k; 2.37 → `<x>` at 12k),
  > `prefill_attn_router` 0.96 → `<x>` / 1.75 → `<x>`,
  > `prefill_shared_expert` 0.17 → `<x>`, GPU busy 5.64 → `<x>` and
  > 6.26 → `<x>`, wall 25.7 → `<x>` s and 86.0 → `<x>` s.** Bars `<cleared |
  > missed>` (routed ≤ 1.87 / ≤ 1.82, gdn ≤ 2.33 / ≤ 2.32; the 4.70 ms tile
  > bar against a measured `<x>`). Bench, grouped ms per ornith tile on the
  > mini at staging 1024 / 2048: K64 `<x>`, K128 `<x>` (the control), K256
  > `<x>` = `<x>` TFLOPS = `<x>` % of the same-run `<x>` ceiling (71 %
  > before). M4 Pro check: `<x>` / `<x>` / `<x>` ms, 12k roles and wall
  > `<moved | unchanged>`. Winner `<n32k256b1 | n32k128b1>`, the default;
  > every arm selectable (`SHRIKE_MPP_TILE_K`, `SHRIKE_MPP_WEIGHT_LOADS`). The
  > implied GDN GEMM share on the mini: `<x>` %. Numerics: K256 reorders the
  > K reduction against K128 (`ShrikeBench mpp_compare`: `<x>` of 262,144 at
  > 4 bits, max abs `<x>`; `<x>` at 8 bits, max abs `<x>`; both boxes) —
  > within the 2e-2 bar against the CPU reference; no bit-equality is claimed
  > between K widths. Golden: short identical on both boxes; long recaptured
  > once per box, sha256 M4 Pro `6fc391949c998e66` → `<x>`, M1
  > `899a25e60a365e60` → `<x>`. Five gates green (`<n>` tests, TSAN 0
  > reports).

  **Risks:**
  - **Occupancy, the most likely way this comes back a null.** 16 KB is 50 %
    of the 32 KB threadgroup budget; on the M1 that may leave one threadgroup
    resident per core where 8 KB left two, and the dequant latency currently
    hidden across threadgroups stops being hidden. This is the footprint at
    which `n64b2` was P8's worst arm on the mini (10.21 ms) — though there the
    stated cause was the doubled fp32 accumulator, and `TILE_K` does **not**
    touch it: the destination cooperative tensors are M × N
    (`tensorops.metal:179-183`), so 32 fp32 registers per thread either way.
    Only the bench separates the two, which is why Step 6 is the gate.
  - **`maxTotalThreadsPerThreadgroup`.** Both encoders dispatch
    `threadExecutionWidth * 4` = 128 unconditionally
    (`MPPPrefillInt4QMM.swift:197-199`, `:303-305`); a pipeline whose max fell
    below that would produce an invalid dispatch. The existing
    `#expect(commandBuffer.error == nil)` in `runPair` / `runShape` catches it
    loudly on Step 1's first run.
  - **The K ladder across the six models.** A wrong rung is either slow
    (narrower than necessary — silent) or wrong (wider than `k` divides — the
    A tensor reads the following row and the last row reads past the buffer).
    Steps 2 and 4 are the guard, and they only work on `irregular` inputs.
  - **Not bit-identical.** No probe, no assertion, no exception: golden moves
    on the long profile on both boxes and is recaptured once per box with the
    measured reason. A short-profile move is a bug.
  - **Same-run ceilings only.** The mini's ceiling bench drifts between runs
    (1.15 TFLOPS at P6, 1.84 at P8 and P9 with the tile unchanged), so every
    bar is a share of the ceiling measured in the *same* `routed_gemm` run,
    never an absolute ms alone.
  - **8 bits.** The loader fix is load-bearing for correctness, not
    performance; without Step 3 the 8-bit K256 instance is silently wrong and
    no 4-bit test would notice.
  - **256 is the end of this mapping.** `kW4A8GroupSize % E == 0` at E = 64 is
    exact; a 512-wide tile fails the assert. If P10 pays, the next lever is
    the register-resident weight tile, not a wider one.

### Task 11: P11 — the FlashAttention-shape attention body

- [ ] **P11: Q and the probabilities in registers under a single-simdgroup
  scope** — target on the mini at 12k: `prefill_attn_router` 1.75 → **≤ 1.58**
  ms/prompt-token (−10 %); at 3.7k 0.96 → **≤ 0.91**; GPU busy 6.26 → ≈ 6.09 at
  12k, wall 86.0 → ≈ 84 s. Bar: **the attention core reaches ≥ 55 % of the
  mini's 1.84 TFLOPS same-run gate/up ceiling, from a modelled 48 % today**,
  read as role ms from `tools/prefill-measure.sh` — there is no `attn` mode in
  `Sources/ShrikeBench/` (the modes are `gdn_scan`, `routed_gemm`,
  `mpp_compare`, `ShrikeBench.swift:23-25`), so unlike P8/P9 there is **no
  isolated tile-ms bar** and the role is the only instrument. **The mini
  decides.** M4 Pro iteration check only: role 0.39 → ≤ 0.36 at 12k, 0.54 →
  ≤ 0.51 at 25k; an arm that regresses there never reaches the mini.

  **Why those numbers.** The role is not the core, and the mini's split has
  never been measured directly. Take the design's own model — attention time
  proportional to query–key pairs, the rest of the role flat per token
  (`docs/v12-prefill-matrix-kernels.md:187-193`) — and fit it to the two
  measured After-P9 mini rows (`:330`): 0.96 at 3,757 tokens, 1.75 at 12,285, a
  per-token pair ratio of 3.27. That gives **flat ≈ 0.61 and core ≈ 1.14
  ms/prompt-token at 12k** (flat = Q/K/V/O through the MPP GEMM, the q-group
  pack, the KV dequant). Modelled, two points, no residual to check — there is
  no mini 25k row after P9. The same fit on the M4 Pro gives flat ≈ 0.19 / core
  ≈ 0.20 and over-predicts its measured 25k row by 12 %, inside that box's
  ±9 % run-to-run swing plus model error.

  Inside the core, P7's own sweep bounds the per-tile fixed term. On the mini,
  at four simdgroups with device operands, `r32s4` (2,048 (row, key) pairs per
  tile) read 2.397 while `g4k128d` and `g2k256d` (4,096 pairs per tile) read
  2.076 and 2.050: halving the tiles per unit work bought 0.347, and the two
  4,096-pair shapes agree to 1.3 % (`task-7-report.md:96-102`). So **the
  per-tile fixed term at the shipped geometry is ≈ 0.35 ms/token** and the
  matmul math is ≈ 0.79 — which is 12.4 TFLOP over 0.79 ms × 12,285 tokens =
  1.28 TFLOPS, **68 % of the mini's 1.84 TFLOPS same-run ceiling**
  (`docs/v12-implementation-plan.md:1819`), the same two thirds the routed tile
  reaches after P9. The ledger closes, so the fixed term is the only thing left.

  P7 named four items in it: the score round-trip through threadgroup memory,
  the rescale of the R×256 fp32 accumulator, the barriers, the `run` set-up
  (`docs/v12-prefill-matrix-kernels.md:313-315`). This shape removes **the
  round-trip and every barrier**, keeps **the accumulator rescale**, and
  **doubles the `run` count per row** (one simdgroup covers 8 rows where four
  cooperating simdgroups covered 16). The bar claims half of the modelled
  0.35 — 0.17 ms/token — and nothing from the other two: 1.75 → 1.58. The
  unreachable floor if the whole fixed term went is 0.61 + 0.79 = **1.41**;
  ≤ 1.48 is the stretch. At 3.7k the core is only 0.35 of the 0.96 role, so the
  same saving is 0.05.

  **The chapter's 50 % bar is retired as the verdict; here is the arithmetic.**
  "≥ 50 % of the M4 Pro's attention ceiling" was computed against the *role* —
  12.4 TFLOP at 12k over 7.46 TFLOPS, i.e. 0.32 ms/token
  (`docs/v12-implementation-plan.md:1214-1218`). With a modelled 0.19 flat term
  that leaves 0.13 for a core running at 7.65 TFLOPS, above the 7.46 ceiling:
  **no change to the attention core can reach it.** On a core basis that box is
  already at ≈ 5.0 TFLOPS, 67 %. The mini bar restates the same 50 % idea where
  there is real headroom (48 % → 68 % if the fixed term went).

  **What P7 left.** `attention_prefill_causal_group_matrix_body<Rq, KEYS, SG>`
  (`Sources/Shrike/Metal/Prefill/attention_matrix.metal:263-425`, default
  `g2k256d` at `:447`) gives one threadgroup of four simdgroups 16 matmul rows
  — 2 query positions × the 8 query heads of one KV head — against 256-key
  tiles read from the device fp16 shadow. Per tile it writes 16×256 fp32 scores
  to threadgroup memory (`:353-354`), barriers (`:356`), runs a lane-parallel
  online softmax that reads them back and writes 16×256 fp16 weights plus a
  `row_scale` entry (`:358-395`), barriers (`:396`), rescales the fp32 output
  accumulator in registers (`:398-403`), runs PV against the threadgroup weight
  tile (`:405`) and barriers again (`:406`). 24.1 KB of threadgroup memory,
  three barriers and a 24 KB round-trip per tile — all of which this task
  deletes.

  **The change.** One **simdgroup** owns one query position's eight query heads
  — M = 8 rows, head dim 256 (`kAttnMatrixHeadDim`, `:24`; the gate pins 16
  query heads over 2 KV heads, `PrefillAttention.swift:300-303`). Q lives in a
  left-input cooperative tensor loaded once before the key loop. Per key tile:
  QKᵀ into a destination cooperative tensor; mask, running max, `exp`, row sum
  and the accumulator rescale element-wise on register-resident data; the
  probabilities converted in place into PV's left-input cooperative tensor; PV
  accumulating into the register-resident O. **No threadgroup memory, no
  barrier, nothing round-trips.** The threadgroup's `SG` simdgroups take one
  query position each and share the K/V tile only through the caches — P7
  measured that sharing serving P2's eightfold re-read and every staged arm
  losing (`docs/v12-prefill-matrix-kernels.md:302-308`), so staging is not
  revisited.

  **What the MPP headers allow, cited** (Xcode 26.6.0 SDK,
  `…/MetalPerformancePrimitives.framework/Headers`):
  - **Input cooperative tensors need a single simdgroup.**
    `__impl/MPPTensorOpsMatMul2dImpl.h:3295-3296` —
    `static_assert(__is_same_v<scope, metal::execution_simdgroup>, "Input
    cooperative tensors require a single SIMD group")`; same on the right-input
    factory (`:3394`) and on `reduce_rows` (`:8793-8798`). This is the whole
    reason the shape needs per-simdgroup matmuls.
  - **A destination cooperative tensor becomes the next matmul's left input,
    with no round-trip.** `MPPTensorOpsMatMul2d.h:450-456` declares
    `get_left_input_cooperative_tensor(const thread cooperative_tensor& src)`;
    the impl (`Impl.h:3311-3377`) asserts the source is a matmul2d
    *destination* layout (`:3316-3319`), the same simdgroup scope
    (`:3320-3323`), `srcDesc.m == dstDesc.m` and `srcDesc.n == dstDesc.k`
    (`:3337-3338`), an untransposed left operand (`:3340`), a static K
    (`:3334-3335`) and `src_elem_type == left_element_type` (`:3328-3329`),
    then performs one in-register relayout
    (`__tensorops_impl_matmul2d_op_cooperative_tensor_copy`, `:3362-3374`).
    QKᵀ's (m 8, n KEYS) destination feeds PV's (m 8, k KEYS) left input exactly.
  - **The probabilities stay fp32.** For `operand_index == left` the tensor's
    storage element type *is* `left_element_type` (`Impl.h:2540-2542`), and
    `:3328` forces it to equal the QKᵀ destination's `float`. PV runs
    float × half → float, a supported combination
    (`MPPTensorOpsMatMul2d.h:24`). P7 rounds its weights to fp16
    (`attention_matrix.metal:383-387`); this shape does not.
  - **Element-wise access is available on every cooperative tensor and its
    layout is implementation-defined.** `MPPTensorOpsMatMul2d.h:216-225`,
    `:246-250` ("not all threads and even all elements within a thread need be
    valid"), with `get_capacity` / `get_mask` / `get_multidimensional_index` /
    `operator[]` shown at `:259-290`; the same `__operand_layout` supplies them
    for the left index (`Impl.h:3155-3157`, `:3067`). So the mask, the `exp`
    and the rescale are legal in registers. `#pragma unroll full` on these loops
    is "imperative for performance" (`:256-258`); P7 measured −35 %
    (`task-7-report.md:113`).
  - **Legal shapes with one cooperative input.** `Impl.h:4257-4259`: M % 8 == 0,
    N % 8 == 0, at least one of M/N a multiple of 16; `:4272`: static K a
    multiple of 16. **(m 8, n 256, k 256)** for QKᵀ and **(m 8, n 256,
    k KEYS)** for PV both pass. The much tighter rule at `:4249-4252` — M, N and
    K each 16 or 32 — applies only when **both** inputs are cooperative
    tensors, which never happens here: K and V stay device tensors.
  - **`load()` fills an operand cooperative tensor from a device or threadgroup
    tensor** of rank 1 or 2 with a matching element type (`Impl.h:2634-2643`,
    `:2670-2684`). That is how Q enters registers.
  - **A row reduction exists but is a Step-6 arm, not the baseline.**
    `reduce_rows` (`MPPTensorOpsMatMul2d.h:587-597`) reduces a destination
    tensor into a rank-1 one with `reduction_operation::max` or `sum`
    (`:342-347`), single-simdgroup only, destination from
    `get_row_reduction_destination_cooperative_tensor` (`:559-565`) on the same
    descriptor (`Impl.h:8812`). Two traps: the public wrapper defaults
    `identity` to `sum_identity` even for a max (`:591-593`), so a max must pass
    `reduction_operation_identity<float>::max_identity`; and pairing the rank-1
    result back to the scores needs `is_iterator_compatible`/`map_iterator`,
    whose documented fallback is "storing sourceCT to threadgroup memory"
    (`:611-625`) — the round-trip this task removes. Nothing in the repo uses
    `execution_simdgroup`, `load()`, an input cooperative tensor or
    `reduce_rows` today (grepped); this is new ground on both boxes.

  **The body, per key tile** (`SG` simdgroups per threadgroup, `KEYS` keys per
  tile, both template arguments). Before the loop, `qCT.load(query_slice)` on
  the P7 `qGroup` tensor sliced at row `(kvh · queryCount + q) · 8`, and the O
  destination zeroed. Then: (1) `qk_op.run(qCT, key_slice, scoreCT)` under
  `mode::multiply`, so the tile's scores are fresh and P7's zeroing loop
  (`attention_matrix.metal:343-346`) disappears. (2) Over `scoreCT`'s elements
  — `position[1]` the row 0–7, `position[0]` the key, the convention the
  shipped body uses at `:352` — all eight rows are the *same* query position, so
  the causal bound is one scalar per simdgroup where P7 carries one per row
  (`:361`); `s = valid ? score · scale : -INFINITY`, tile max per row over each
  lane's elements, then a `simd_max` butterfly over an 8-element register array
  to make it simdgroup-wide. (3) `next_max`, `old_scale = exp(run_max −
  next_max)`, `p = valid ? exp(s − next_max) : 0` written back into `scoreCT` in
  place, `tile_sum` by the same butterfly with `simd_sum`, `run_sum = run_sum ·
  old_scale + tile_sum`. (4) `outCT[e] *= old_scale[row]` — the one fixed-cost
  item this shape keeps, 64 fp32 multiplies per thread per tile at KEYS 256.
  (5) `pCT = pv_op.get_left_input_cooperative_tensor<float, half,
  float>(scoreCT)`, then `pv_op.run(pCT, value_slice, outCT)` under
  `mode::multiply_accumulate` — the mode P7's GREEN attempt 1 proved is
  load-bearing (`task-7-report.md:17-19`). After the loop, divide by `run_sum`
  and store to device O; the row → (query, head) map is P7's with `r / 8`
  collapsing to the simdgroup's own query (`attention_matrix.metal:419-421`).

  There is **no `threadgroup_barrier` anywhere in the body**, so a simdgroup
  past `queryCount` may return early where P7 had to keep its early return
  threadgroup-uniform (`:297`, `task-7-review.md:30`). Each simdgroup runs its
  own causal trip count; the spread across a threadgroup is at most one tile.

  **Registers, and why `KEYS` is the knob.** Per thread, 32-bit registers, 32
  lanes per simdgroup:

  | tensor | shape | type | regs/thread |
  | --- | --- | --- | ---: |
  | Q left input | 8 × 256 | half | 32 |
  | QKᵀ destination (scores → probabilities) | 8 × KEYS | float | KEYS / 4 |
  | PV left input | 8 × KEYS | float | KEYS / 4 |
  | O destination | 8 × 256 | float | 64 |
  | **total** at KEYS 256 / 128 / 64 | | | **224 / 160 / 128** |

  Plus the max and sum arrays, addresses and loop state. The source cannot say
  what fits: **the pipeline is the measurement.** The kernels carry
  `[[max_total_threads_per_threadgroup(32 · SG)]]` as P7's macro does
  (`attention_matrix.metal:428`), so if registers will not allow that many
  threads, `newComputePipelineState` **fails at init** and `matrixPathAvailable`
  goes false with the reason recorded — on both boxes, from a unit test, before
  any bench. Smaller `KEYS` buys registers at the cost of more tiles, hence more
  `run` set-ups and more O rescales — the item this shape does *not* remove.
  That trade is the whole spike.

  **Files:**
  - Modify: `Sources/Shrike/Metal/Prefill/attention_matrix.metal` — a third body
    template `attention_prefill_causal_flash_body<SG, KEYS>` after the P7 body
    (`:263-425`), a third macro beside `ATTN_GROUP_MATRIX_KERNEL` (`:427-444`),
    and the instantiations. The P2 and P7 bodies stay untouched, as P7 left P2.
  - Modify: `Sources/Shrike/Kernels/Attention/PrefillAttention.swift:468-493` —
    `MatrixTile` gains the `f*` cases with `queryRows` = `SG` and
    `threadsPerThreadgroup` = `32 · SG`. `groupsEightHeads` is an exclusion list
    (`:479`), so the new cases inherit `true` and reuse the q-group pack and the
    grouped dispatch geometry (`:341-352`) **with no change to `encodeMatrix`,
    `encodeQGroupPack`, `ensureQGroup` or `matrixPathAccepts`**. Only the
    default at `:90-92` moves, and only in Step 7.
  - Test: `Tests/Shrike/Core/Kernels/Attention/PrefillAttentionMatrixTests.swift`
    — `groupTiles` (`:14`) is the list the two numeric tests sweep (`:34-40`,
    `:63-69`); a `flashTiles` list joins it, and
    `matrixTileVariantsDescribeTheirGeometry` (`:187`) plus
    `rejectedShapeRunsTheTiledKernelEndToEnd` (`:167-177`, over `allCases`) pick
    the new cases up for free.
  - Unchanged deliberately: `attention_prefill_kv_dequant` and the fp16 shadow;
    `attention_prefill_q_group_pack` and its sizing; the P2 and P7 bodies and
    all four shipped tiles; `matrixPathAccepts`, `matrixPathMinimumQueries`,
    `matrixPathMaxContext`; the `.causalTiled` fallback and
    `SHRIKE_PREFILL_ATTENTION=tiled`; `prefillAttentionPathDescription`, which
    already prints `tile=<rawValue>` (`RealForwardRunner.swift:191`).

  **Interfaces:**
  - Consumes: `PrefillAttentionParams` unchanged; the device fp16 shadow; the
    group-major `qGroup`; `PrefillAttentionRef.apply(_:) -> [Float]`,
    `RelError`, `Fp16Buffer`, `KVCacheQuantizer.encode`;
    `PrefillAttention(context:supportsMLA:matrixTile:)` (`:105-106`), which
    already lets a test pin a variant without a process-wide env var.
  - Produces:

    ```swift
    extension PrefillAttention {
        enum MatrixTile: String, CaseIterable, Sendable {
            case r32s4, r64s8                    // P2, one query head
            case g4k128d, g2k256d                // P7, one KV-head group
            case f4k256, f4k128, f4k64, f8k128   // P11, one query per simdgroup
            var queryRows: Int  // f<SG>k<KEYS> -> SG: one query position each
            var threadsPerThreadgroup: Int  // 32 * SG
        }
    }
    ```

    ```metal
    template <int SG, int KEYS>
    static inline void attention_prefill_causal_flash_body(
        device half* qGroup, device half* shadowK, device half* shadowV,
        device half* O, constant PrefillAttentionParams& p,
        uint3 tg, uint sgid /* simdgroup_index_in_threadgroup */,
        uint lane /* thread_index_in_simdgroup */);
    // matmul2d<matmul2d_descriptor(8, KEYS, 256, false, true, false,
    //          mode::multiply),            execution_simdgroup> qk_op;
    // matmul2d<matmul2d_descriptor(8, 256, KEYS, false, false, false,
    //          mode::multiply_accumulate), execution_simdgroup> pv_op;
    // ATTN_FLASH_KERNEL(attention_prefill_causal_matrix_f4k256, 4, 256), …
    ```

    No new buffers, no new dispatch, no threadgroup allocation — the macro
    declares none where P7's declares three arrays (`:438-440`). Lint: keep the
    softmax step in its own `static inline` helper so nothing on the Swift side
    approaches the 120-line `function_body_length` warn. Comments: the repo rule
    holds; two earn their place — why the scope is `execution_simdgroup` (the
    header assert at `Impl.h:3295`, not a choice) and why there is no barrier.

  Steps (TDD; correctness on the reference before any geometry hunting):

  - [ ] Step 1: failing tests `flashMatrixMatchesReferenceOnFP16Cache(c:tile:)`
        — the three `fp16Cases` (`PrefillAttentionMatrixTests.swift:15-19`) ×
        `flashTiles` through `checkFP16Reference` against
        `PrefillAttentionRef.apply` at the suite's 2e-2 on both
        `RelError.maxAbsDiff` and `RelError.compute`, finiteness asserted — and
        `flashMatrixMatchesTiledOnQuantizedCache(c:tile:)`, the five
        `quantizedCases` (`:20-26`) against `.causalTiled`. FAIL: no `f*`
        kernels.
  - [ ] Step 2: failing test `flashTilesBuildTheirPipelines` — for every `f*`
        case, construct `PrefillAttention(context:supportsMLA:matrixTile:)` and
        `#expect(attention.matrixPathAvailable)` with `matrixUnavailableReason`
        in the message. This is the register gate: the kernel's
        `max_total_threads_per_threadgroup(32·SG)` makes pipeline creation fail
        rather than silently derate, and the test says so on whichever box runs.
  - [ ] Step 3: extend `matrixTileVariantsDescribeTheirGeometry` (`:187`) to pin
        `queryRows`/`threadsPerThreadgroup` for the `f*` cases, and re-run
        `rejectedShapeRunsTheTiledKernelEndToEnd` (`:167-177`) over `allCases`
        for byte equality with `.causalTiled` on a rejected shape.
        `swift test --no-parallel --filter PrefillAttentionMatrixTests` → FAIL.
  - [ ] Step 4: implement the body, the macro and the instantiations `f4k256`,
        `f4k128`, `f4k64`, `f8k128`, plus the `MatrixTile` cases. Every
        cooperative-tensor loop carries `#pragma clang loop unroll(full)`
        (`attention_matrix.metal:336`, `:343`;
        `MPPTensorOpsMatMul2d.h:256-258`). Steps 1–3 PASS. **Record which `KEYS`
        widths built a pipeline on each box** — that is the register answer the
        source could not give, and it belongs in the verdict whether or not the
        spike proceeds.
  - [ ] Step 5 (optional, recommended): add an `attn` mode to
        `Sources/ShrikeBench/ShrikeBench.swift` (`:23-25` is the mode list) that
        times the causal-matrix kernel alone at the ornith shape over synthetic
        buffers, `RoutedGEMMBench`-style, per `MatrixTile`. It turns each spike
        arm from a fresh server plus an 86-second 12k prefill into a bench run,
        and gives every future attention task the isolated instrument P7 and
        this task both lacked. If skipped, say so in the verdict; the server A/B
        is the gate of record either way.
  - [ ] Step 6 (spike, the gate): one **M4 Pro** server per arm, fresh each
        time, `--port 8082 --ram-budget 20G --thinking off`,
        `SHRIKE_KERNEL_STATS=1 SHRIKE_RUNNER_STATS=1`, the 12k prompt sent once
        per server lifetime (a resend hits the prompt cache — Global
        constraints, `docs/v12-implementation-plan.md:45-50`);
        `SHRIKE_ATTN_MATRIX_TILE` ∈ {`g2k256d` (control), every `f*` that built
        a pipeline}, plus a `reduce_rows` arm on the winning geometry if the
        butterfly shows up as a cost. Drop any arm that regresses against the
        0.39 control; carry the survivors to the **mini, which decides**:
        `tools/mini-deploy.sh` (copy only) **first** — it copies the `*.bundle`
        directories carrying the shader sources, and P8's bench first failed
        because they were stale — then `pgrep`, stop production, run the arms at
        3.7k and 12k, relaunch with `tools/mini-deploy.sh --restart`. **Accept
        rule: the lowest mini `prefill_attn_router` at 12k among arms at or
        under 1.58 ms/token with 3.7k not regressed. Arms within 3 % are a tie
        (P6 saw 4 % between nominally identical arms) and a tie breaks toward
        the larger `KEYS`, which costs fewer `run` set-ups. If no arm reaches
        1.58 but one beats the 1.75 control by ≥ 3 %, land it as the default and
        record the achieved core share; if none beats the control by 3 %, land
        the tests and the variants only, keep `g2k256d` the default, and say
        so — P5's and P8's precedent, with the measured-loser knob staying
        selectable on the same binary.** Verdict line: the arm table per box.
  - [ ] Step 7: make the winner the default `matrixTile`
        (`PrefillAttention.swift:90-92`) and keep every variant selectable by
        `SHRIKE_ATTN_MATRIX_TILE`. Five gates: release build 0 warnings;
        `swiftlint lint --strict --baseline .swiftlint-baseline.json`; markdown
        link check; `swift test --no-parallel`; the same under
        `env TSAN_OPTIONS=suppressions=tsan-suppressions.txt swift test
        --no-parallel --sanitize=thread`.
  - [ ] Step 8: `tools/golden-baseline.sh --check` (M4 Pro, server stopped).
        **Expected IDENTICAL on the short profile** (it never reaches the matrix
        path, `docs/v12-prefill-matrix-kernels.md:320-322`) and **expected to
        differ on the long profile** for two named reasons: the probabilities
        are fp32 where P7 rounds them to fp16
        (`attention_matrix.metal:383-387` vs `Impl.h:2540-2542`), and MPP
        reduces QKᵀ's K = 256 from a register-resident left operand rather than
        a device one, an order the source does not fix. Recapture once per box
        with those reasons and the before/after greedy digests; digests to carry
        forward, P9's: M4 Pro long `6fc391949c998e66`, M1 long
        `899a25e60a365e60` (`docs/v12-implementation-plan.md:1838-1840`).
        **Never recapture for an unexplained mismatch.** Any probe that tries to
        bound the difference must use full-mantissa inputs — P9's Step 4
        bit-equality probe passed and was wrong because integer-valued fixtures
        have fp32-exact partial sums no reduction order can disturb
        (`:1833-1837`). Commit `prefill: the FlashAttention-shape attention body
        on the matrix path (v12 P11)`, baselines via `git commit --only`.
  - [ ] Step 9: ledger on both boxes (fresh server, `tools/prefill-measure.sh`,
        one send per prompt per server lifetime; **mini 3.7k + 12k is the
        verdict**, M4 Pro 3.7k + 12k + 25k the check); `tools/mini-deploy.sh
        --restart`, mini golden recapture, scp into `baselines/`. Verdict line:
        `prefill_attn_router` before → after at 3.7k and 12k on the mini with
        the M4 Pro's three rows as the check; the winning geometry and which
        `KEYS` widths built on each box; **the achieved core share against the
        flat/core split above, re-fitted from the new rows** — if the role moves
        less than the core model predicts, record the implied flat term rather
        than calling the task short; the golden outcome per box; wall on both
        boxes, noting the M4 Pro's saving may sit in the routed→routed gap
        rather than the wall (`docs/v12-prefill-matrix-kernels.md:343-345`).
        Design doc: the "Attention: Q and the probabilities in registers"
        follow-on bullet (`:640-648`) is removed and becomes a landed
        `### Step 10 — the FlashAttention-shape attention body (…)` section; an
        "**After P11**" ledger block after the After P9 table (`:324-349`); the
        attention paragraph at `:174-185` updated. Plan: Task 11 `[x]` with the
        landed paragraph. Task review by a fresh reviewer; fixes folded into the
        commit (rebase and amend, never a fixup commit).

  **Verdict template** (fill from Step 9; every number measured or labelled):
  **LANDED `<sha>` (`<date>`): measured on the mini `prefill_attn_router`
  `0.96 → <x>` ms/prompt-token (3.7k; `1.75 → <y>` at 12k), GPU busy
  `6.26 → <z>` at 12k, wall `86.0 → <w>` s.** Bar `<cleared|missed>` (≤ 1.58 at
  12k, ≥ 55 % core share). Arm table per box. Pipelines built: `<KEYS widths>`
  on the M4 Pro, `<…>` on the mini. Core share `<a> % → <b> %` of the mini's
  1.84 TFLOPS same-run ceiling, re-fitted flat term `<c>`. M4 Pro check:
  `<3.7k/12k/25k>`, wall `<…>`. Numerics: all `<n>` cases at 2e-2, finiteness
  asserted. Golden: short identical on both boxes; long `<identical|recaptured>`
  per box with digests. Five gates green (`<n>` tests, TSAN 0 reports).

  **Risks:**
  - **Register pressure and occupancy on the M1.** 224 registers per thread at
    `KEYS` 256 is the largest register footprint in the repo by a wide margin;
    Step 4 finds out whether the pipeline builds, and `f4k64` (128 registers)
    exists so the task has somewhere to land if it does not. The hedge has a
    price: `KEYS` 64 quadruples the tiles, hence the `run` set-ups and the O
    rescales — the fixed term this shape *keeps*. Every width that fits losing
    to `g2k256d` is a possible outcome; it is the accept rule's last clause, not
    a failure to report.
  - **The layout copy is unpriced.** `get_left_input_cooperative_tensor(src)`
    performs a real relayout (`Impl.h:3362-3374`) whose cost is
    implementation-defined and could be shuffles proportional to the tile. It
    replaces a 24 KB threadgroup round-trip, so it should win — modelled, not
    measured. Passing the destination tensor straight to `run` as the left
    operand is **not** a documented path and must not be tried as a shortcut.
  - **What the cooperative tensor forbids.** The per-lane layout is
    implementation-defined (`MPPTensorOpsMatMul2d.h:216-225`), so nothing may
    assume which lane holds which element; every access goes through
    `get_multidimensional_index` and the validity mask (`:246-250`), and the row
    max/sum must be a simdgroup-wide reduction, never lane-local. The
    `reduce_rows` alternative carries the `max_identity` default trap
    (`:591-593`) and may fall back to threadgroup memory by the header's own
    advice (`:611-625`).
  - **The causal mask on the diagonal tile.** All eight rows share one query
    position, so the bound is one scalar — but a fully masked tile would make
    `run_max` `-INFINITY` and `exp(-inf − -inf)` NaN. The loop bound
    `key_start < min(kvValidCount, startPosition + q + 1)` guarantees the first
    tile has a visible key, and masked elements take an exact `0` probability
    rather than relying on `exp`, as P7 does (`attention_matrix.metal:383-386`).
    `deep-history-ragged` and `deep-history-short-block` (`:22-24`) are the
    coverage.
  - **fp16 probabilities vs fp32 sums.** P7's review noted that taking the row
    sum from the *same* rounded fp16 weights that feed PV bounds the relative
    error at ≈ 2⁻¹¹ independent of context length (`task-7-review.md:29`). This
    shape keeps everything fp32, so the bound strictly improves — but numerator
    and denominator now come from different reductions (the matmul's internal K
    order vs the simdgroup butterfly), which is why golden is expected to move
    and why the 2e-2 bar against the fp32 reference, not bit equality, is the
    gate.
  - **The M1's `matmul2d` support at m = 8.** The static asserts
    (`Impl.h:4257-4259`, `:4272`) are compile-time and box-independent, but
    whether an 8-row simdgroup matmul is *efficient* on an M1 is not something
    the headers say; P7 already measured the trend reversing past 16 rows × 256
    keys (`task-7-report.md:118-119`) and m = 8 goes further that way per
    simdgroup. `f8k128` gives the sweep a threadgroup-shape degree of freedom
    when m cannot move.
  - **The baseline moves if P10 lands first.** Task 10 (the 256-wide K tile)
    models `prefill_attn_router` 1.75 → ≈ 1.72 on the mini at 12k through the
    Q/K/V/O projections — the *flat* term of the fit above, not the core. If it
    lands, restate every bar here against the measured post-P10 row and shift
    the flat term by the same amount; the core term, the 0.35 fixed term and
    the 0.17 claim are untouched by it. If P10 does not land, renumber this to
    Task 10 and change the design-doc heading in Step 9 to `### Step 9`.

### Task 12: P12 — the prefill gap levers: first-routed-buffer driver cost and per-layer host routing

- [x] **P12: the two gaps P5 exposed and P9 left** — the mini's prefill GPU is
  idle 0.56 ms per prompt token at 12k (8.9 % of the 6.26 ms it is busy), and
  91 % of that idle sits in four role transitions. This task takes the two that
  are host and driver work, not kernels. **Targets on the mini at 12k
  (12,285 prompt tokens, `tools/prefill-prompts.py:10`): `prefill_shared_expert
  ->prefill_routed_tile` `driver_ms` 2.65 s → ≤ 0.40 s; `prefill_gdn_router->
  prefill_shared_expert` + `prefill_attn_router->prefill_shared_expert`
  `host_ms` 1.86 s → ≤ 0.50 s; gaps (span − busy) 0.56 → ≤ 0.20 ms/prompt-token;
  wall 86.0 → ≤ 82.5 s (7.00 → ≤ 6.72 ms/prompt-token). At 3.7k (3,756 tokens):
  driver 1.02 → ≤ 0.20 s, host 0.64 → ≤ 0.20 s, gaps 0.72 → ≤ 0.30, wall
  25.7 → ≤ 24.4 s. Single lever: A alone wall ≤ 84.0 s, B alone ≤ 84.5 s at
  12k. GPU busy must not move (6.26 ± 2 %) and golden must be IDENTICAL —
  neither lever touches a kernel or an operand.** The wall bars are the two
  measured prizes (2.65 s driver + 1.94 s host = 4.6 s of 86.0, modelled
  81.4 s) plus 1.1 s of noise margin. **The mini decides**; the M4 Pro is the
  iteration bench (≈ 11 ms driver and ≈ 2.8 ms host per layer,
  `v12-prefill-matrix-kernels.md:614-622`, `:636-639`, so a null there decides
  nothing — and it is fetch-bound, so a saving can reappear as routed→routed
  gap rather than wall, `:277-283`).
  **LANDED 188d2b7 (2026-09-03): measured on the mini at 12k
  `prefill_shared_expert->prefill_routed_tile` total 3,525 → 916 ms (host
  820 → 827, driver 2,664 → 58, count 120), `prefill_gdn_router->prefill_shared_expert`
  1,432 ms (host 1,388) and `prefill_attn_router->prefill_shared_expert` 486 ms
  (host 465) no longer transitions at all, gaps 0.581 → 0.273 ms/prompt-token,
  GPU busy 6.107 → 6.096, wall 84.41 → 80.38 s (6.87 → 6.54 ms/token, 1.04× the
  6.3 target); at 3.7k gaps 0.759 → 0.488, busy 5.47 → 5.42, wall 25.42 →
  24.32 s.** Bars: wall ≤ 82.5 cleared; driver 2.65 → ≤ 0.40 s cleared (0.058);
  the `->shared` host 1.86 → ≤ 0.50 s cleared (the transitions are gone); gaps
  ≤ 0.20 missed at 0.273 (`routed->routed` 1,182 ms of per-tile host work,
  `shared->routed` 827 ms of metadata build plus the first tile's unhidden
  fetch — the follow-on). Step 1's arms on the P10 binary: A1 (`--ram-budget
  4G`) driver −51 %, A2 (per-slot) −91 % — residency of the referenced buffer,
  proportional to its length (each layer's ≈ 226 MB pool, forty of them); the
  queue residency set landed as lever A and no per-slot wrappers were needed.
  Single-lever arms on one binary at 12k: overlap alone 82.15 s, residency
  alone 82.46 s — the levers add. At 3.7k: driver ≤ 0.20 s cleared (0.022),
  the `->shared` host cleared (gone), wall ≤ 24.4 s cleared (24.32), gaps
  ≤ 0.30 missed at 0.488 — the same residue as at 12k. Steps 3, 4 and 6 (the `p12probe` split, the
  counting sort and its tests) were not run: under the overlap the `->shared`
  host transitions vanish entirely (17.4 ms of shared-expert GPU covers the
  16 ms of routing), so nothing of the routing is left to shrink; the counting
  sort stays a follow-on if a future change exposes it. The four role rows are
  unchanged within noise. M4 Pro check: driver 570 → 10 ms, `gdn->shared` host
  219 → gone, wall 28.01 → 27.52 s at 12k and 10.16 → 9.97 s at 3.7k, the rest
  fetch-bound. Golden IDENTICAL on both boxes, short and long. Memory:
  `memory_pressure -Q` on the mini 24–26 % free right after a 12k prefill with
  the set vs 73–75 % without, 83 % a minute later; `vm_stat` at idle with the
  set 2.2 GB wired and the pools as 11 GB of active pages — not pinned;
  `SHRIKE_PREFILL_POOL_RESIDENCY=none` is the one-env rollback. Lint baseline
  regenerated for `encodeRoutedMoEPrefill` (261 → 230 lines, the routing
  extracted to `buildPrefillRoutes`) and `ServerInference.load` (180 → 181).
  Five gates green (1163 tests, TSAN 0 reports in 1,866 s).

  **What P5 and P9 left.** P5 added the three-way gap split (`host_ms` =
  previous buffer's GPU end → this buffer's `kernelStartTime`, `driver_ms` =
  `kernelStartTime` → `kernelEndTime`, `queue_ms` = `kernelEndTime` → GPU
  start; `RealForwardRunner.swift:1733-1737`, `:1752-1754`) and found the
  routed loop fetch-bound, not commit-bound — a measured null
  (`docs/v12-implementation-plan.md:717-755`, task-5-report.md:505-546). P9 cut
  the kernels 28 % but moved none of the gaps: they are host and driver time and
  are the same absolute milliseconds they were at P5, so their *share* of busy
  rose. Measured at the P9 build on the mini at 12k (`Shrike gap` lines; the
  parser's `per_token_ms` divides by the generated tokens, 8 by
  `tools/prefill-prompts.py:44`, so `total_ms` = 8 × the value — read
  `total_ms` straight off the log rather than reconstructing it, and confirm
  the `count=` and `generated=` fields, which the counts below are *derived*
  from the model's shape rather than read):

  | transition | parser ms | total ms | share of busy | count | ms per boundary |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | `prefill_shared_expert->prefill_routed_tile` | 433.8 | 3,470 | 4.5 % | 120 | 28.9 |
  | `prefill_gdn_router->prefill_shared_expert` | 182.1 | 1,457 | 1.9 % | 90 | 16.2 |
  | `prefill_routed_tile->prefill_routed_tile` | 107.2 | 858 | 1.1 % | ≈ 3,510 | 0.24 |
  | `prefill_attn_router->prefill_shared_expert` | 60.7 | 486 | 0.6 % | 30 | 16.2 |
  | **the four** | | **6,271** | **8.1 %** | | |
  | gaps, all eight (span − busy) | | 6,880 | 8.9 % | | |

  The counts are the model's shape: ornith is 40 MoE layers, 30 GDN and 10
  attention (`v12-prefill-matrix-kernels.md:34-36`), and the 12k prompt is three
  4,096-token chunks — 120 layer-chunks, of which 90 are GDN and 30 attention.
  The two `->prefill_shared_expert` rows cost **16.2 ms each per layer-chunk**,
  identical to three digits across two different preceding roles: that is the
  signature of work that happens *between* the buffers on the host, not of
  anything either kernel does. Cross-check on the totals: 4.5 % × (6.26 ×
  12,285) = 3,461 ms against the parser's 3,470 — the two routes agree to 0.3 %.

  **Lever A — the ≈ 22 ms of driver time on the first routed buffer of each
  layer.** What that buffer references and nothing earlier in the layer does is
  the **expert slab**. Under the pool layout (the default, and the mini's
  production launch) every cache slot is *one* `MTLBuffer`:
  `PreadExpertStreamer.swift:379-404` `posix_memalign`s
  `poolSlotStride * slotCount` bytes at 2 MiB alignment, wraps the whole
  allocation once with `makeBuffer(bytesNoCopy:length:options:.storageModeShared)`
  and hands every slot that same buffer at a different offset (`:398-404`);
  `ExpertResidencyTable.expertPool` (`:26`, `PreadExpertStreamer.swift:1036`)
  is it. On the mini at `--ram-budget 8G` that is 128 slots ≈ **9.06 GB**
  (CLAUDE.md). A routed tile binds it through
  `PrefillStreamedTileBinding.views` and calls `useResource(view.buffer,
  usage: .read)` once per view — eight calls naming the same 9.06 GB buffer,
  per encoder, per tile (`PrefillGroupedRoutedMoE.swift:690`, `:715`;
  `MPPPrefillInt4QMM.swift:296-298`) — reaching the sub-tensors through a
  per-tile argument buffer (`PrefillGroupedRoutedMoE.swift:564-579`). Nothing
  else in a layer touches it: the GDN/attention buffer and the shared expert
  read resident weights and chunk scratch only (`RealForwardRunner.swift:4947-4966`,
  `:5034-5038`, `PrefillChunkScratch.swift:200-201`). So the first routed buffer
  of a layer is the first in a while to name a 9.06 GB allocation, and it pays
  22.1 ms in the driver where the layer's later tiles pay ≈ 0.01 ms
  (`v12-prefill-matrix-kernels.md:614-622`). One fact already kills the obvious
  rival: the per-tile argument buffer and the per-layer `sortedPairs` buffer
  are *freshly allocated* every tile (`PrefillGroupedRoutedMoE.swift:566`,
  `:640-643`), so a first-touch cost on a new allocation would be paid by every
  tile, not the first. What survives beside residency is the routed scratch
  (`routedGateUpActScratch`, `routedDownScratch`, `routePartials`,
  `routedExpertStaging`), also referenced only by routed buffers — the one
  size-independent rival, and the one Step 1's arms separate.
  **There is no `MTLResidencySet`, no `MTLHeap` and no `addResidencySet`
  anywhere in `sources/`** (`rg -n "MTLResidencySet|ResidencySet|useHeap"
  sources/` finds nothing; `MetalContext.swift:70-76` makes one plain queue),
  so the model is untested and Step 1 tests it before any code is written.

  **Lever B — the 16.2 ms of host routing between the router buffer and the
  shared expert.** Every millisecond of it is in
  `RealForwardRunner.encodeRoutedMoEPrefill` (`:4926`) between
  `waitForCompletion(cb)` (`:4970`) and `sharedCB.commit()` (`:5039`):
  - `:4982-4996` — the "readback". `routeIDs` / `routeWeights` are
    `.storageModeShared` (`PrefillChunkScratch.swift:268-271`), so this is a
    `.contents()` pointer, **not** a blit and not a transfer: 32,768 `UInt32` +
    32,768 `Float16` = 192 KB memcpy'd into two host arrays. **This kills
    option (iii) up front** — uint16 indices and fp16 weights would shrink a
    copy already in the tens of microseconds on unified memory.
  - `:4997-5000` — `makeTokenExpertPairs` (`PrefillRouter.swift:111-131`)
    materialises 32,768 × 16-byte pairs = 512 KB, one `append` at a time.
  - `:5015-5021` — `PrefillMoEGrouping.groupTokenExpertPairs`
    (`PrefillMoEGrouping.swift:88-192`), the bulk: a `Set<UInt64>` insert **per
    pair** (`:120-137`) — a hash table built and thrown away 120 times per
    request — then `pairs.sorted { … }` (`:139-148`), a comparison sort of
    32,768 elements whose closure does two `expertSortKeys` subscripts per
    comparison, ≈ 480,000 of them. The group and tile scans (`:150-183`) are
    O(n) and cheap.
  The shared expert does **not** depend on any of that. `encodeSharedExpertBlock`
  (`:4757-4800`) reads `scratch.routedX` (written by the buffer we just waited
  on) and writes `scratch.h1` plus its own `sharedGateScratch`/`sharedUpScratch`;
  the routed tiles read `routedX` and write `routePartials`,
  `routedGateUpActScratch`, `routedDownScratch` — disjoint (`:5245-5250`). The
  only reason the wait at `:4970` exists is the router readback, which the
  runner already says in as many words on the decode path: "Same queue either
  way, one wait on the last CB; only the router readback forces the barrier"
  (`:2577-2582`); that path already moved the shared-expert chain into another
  buffer for the same reason (`:5800-5806`). **So option (i) — the reorder — is
  lever B:** commit the shared expert first and group on the host while it
  runs. The prize is bounded by which is longer, and the shared role is
  0.17 ms/prompt-token = 2,088 ms over 120 layer-chunks = **17.4 ms of GPU per
  layer-chunk** against 16.2 ms of host: it covers it, with 7 % to spare.
  Option (ii), a GPU counting sort handing the host a finished block table,
  buys nothing the reorder does not and costs a kernel and a numerics review —
  **a follow-on, not scheduled.** The real second lever is the host loop
  itself: a stable counting sort by expert (256 buckets, ranked once by
  `(expertSortKeys[e], e)`) reproduces `sorted`'s order *exactly*, because the
  pairs arrive token-major/rank-minor and that is precisely the comparison
  sort's tie-break, plus a bitset in place of the `Set` — O(n) against
  O(n log n), bit-identical by construction. It only *matters* if the routing
  still outruns 17.4 ms after the reorder, so Step 6 is gated on that
  measurement rather than assumed.

  **What A and B do not cover.** `prefill_attn_router->prefill_shared_expert`
  is the same host routing as the GDN row — same 16.2 ms, different preceding
  role — so lever B covers it. `prefill_routed_tile->prefill_routed_tile`,
  858 ms (1.1 % of busy, 0.9 % of wall), is 0.24 ms per boundary over ≈ 3,510
  boundaries — the routed fetch's excess over the GPU tile, the design's own
  follow-on (`v12-prefill-matrix-kernels.md:604-613`): `fetchBindingForTile`
  latency (3.7 ms per tile on the M4 Pro, 0.5 ms of it the pread,
  task-5-report.md:527) against a tile now ≈ 6.3 ms of mini GPU, so the
  one-tile overlap still hides most of it. Neither lever touches it, though B
  incidentally gives the *first* tile's fetch the shared expert to hide under.

  **Numerics.** Nothing here changes an operand, a kernel, a dispatch shape or
  a reduction order: A changes when the driver makes a buffer resident, B the
  order two independent command buffers are submitted in and (optionally) the
  algorithm producing an array that must compare equal.
  `tools/golden-baseline.sh --check` is expected **IDENTICAL**, short and long,
  on both boxes; a mismatch is a defect — do not recapture
  (`docs/v12-implementation-plan.md:38-42`). The counting sort's equality is
  asserted against the comparison sort in a unit test, not inferred from
  golden.

  **Files:**
  - Modify: `Sources/Shrike/Runtime/Inference/RealForwardRunner.swift:5029-5045`
    (commit the shared expert without waiting; move `:4982-5021`'s routing work
    after that commit; `makeStreamedMetadataBuffers` follows the routing as it
    does today), `:5178-5186` (wait for the shared buffer and
    `recordKernelGPU(role: "prefill_shared_expert", …)` there, after the tile
    drain and before the tail encode — it is long complete by then, and the
    wait keeps the error path), `:348-374` (the two env-knob statics beside
    `environmentPrefillTileBatch`), `:185-188` (`prefillProjectionPath` is the
    line `mini-deploy.sh:79` greps; leave its leading token alone and append
    `route_overlap=` / `pool_residency=` so the residency line records the arm).
  - Add: `Sources/Shrike/Infrastructure/Metal/ExpertPoolResidency.swift` — the
    `MTLResidencySet` holder (lever A's productisation), ≈ 40 lines.
  - Modify: `Sources/Shrike/Kernels/Prefill/MoE/PrefillMoEGrouping.swift:118-148`
    (bitset validation, counting sort) — only if Step 3 and Step 5 say it is
    still worth it.
  - Modify: `Sources/Shrike/Runtime/Inference/Model.swift` (wherever the
    streamer's `ExpertResidencyTable` reaches the runner) to hand the pool
    buffer to the residency holder once at attach
    (`ExpertResidencyTable.swift:26`, `PreadExpertStreamer.swift:1036`).
  - Test: `Tests/Shrike/Core/Kernels/Prefill/PrefillMoEGroupingTests.swift`
    (214 lines; `:5`, `:55`, `:84`, `:117`, `:175` the existing cases, `:205`
    the pair helper) and a new
    `Tests/Shrike/Core/Infrastructure/Streaming/ExpertPoolResidencyTests.swift`
    (that directory exists; there is no `Infrastructure/Metal` one).
  - Unchanged deliberately: every kernel and every `.metal` file; the tile
    scheduler and `PrefillRoutedTileSchedulerTests.swift`; `fetchBindingForTile`
    and the pending-tile overlap; `SHRIKE_PREFILL_TILE_BATCH`'s default of 1;
    pool as the expert cache's default layout.
  - **Lint:** the baseline records `encodeRoutedMoEPrefill` at 261 lines. Any
    change to its length changes the reason string and the strict gate fails on
    the stale entry — regenerate with `swiftlint lint --write-baseline
    .swiftlint-baseline.json` in the same commit, as P5 did
    (task-5-report.md:131-143). Prefer extracting the routing block into a
    private `buildRoutes(...) throws -> PrefillMoEGroupedRoutes` so the
    function shrinks rather than grows.

  **Interfaces:**
  - Consumes: `MTLDevice.makeResidencySet(descriptor:)`
    (`MTLDevice.h:1324-1327`, `newResidencySetWithDescriptor:`, **nullable** —
    keep the holder failable), `MTLResidencySet.addAllocation(_:)` / `.commit()`
    / `.requestResidency()` / `.allocatedSize` (`MTLResidencySet.h:61-136`), and
    `MTLCommandQueue.addResidencySet(_:)` (`MTLCommandQueue.h:58-62`: "Marks
    the residency set as part of the command queue execution. This ensures that
    the residency set is resident during execution of all the command buffers
    within the queue"). All `API_AVAILABLE(macos(15.0))`; the repo requires
    macOS 26, so no availability guard.
  - Produces:

    ```swift
    /// Keeps the routed-expert pool resident for the queue's lifetime, so the
    /// first buffer of a layer to name the 9 GB slab does not pay in the driver.
    final class ExpertPoolResidency {
        init?(device: MTLDevice, pool: MTLBuffer)
        func attach(to queue: MTLCommandQueue)
        var allocatedBytes: UInt64 { get }   // MTLResidencySet.allocatedSize
    }

    extension RealForwardRunner {
        /// `SHRIKE_PREFILL_ROUTE_OVERLAP=on|off` — on (the new default) commits
        /// the shared expert before the host builds the routing.
        static let prefillRouteOverlap: Bool
        /// `SHRIKE_PREFILL_POOL_RESIDENCY=set|none` — set (the new default)
        /// attaches the pool's residency set to the queue.
        static let prefillPoolResidency: Bool
    }

    // groupTokenExpertPairs gains `sort: SortKind = .counting`
    // (`enum SortKind: String, Sendable { case counting, comparison }`);
    // every other parameter and the return type are unchanged.
    ```

    Both env knobs default to the *new* behaviour once landed and keep the P9
    behaviour reachable on the same binary (`SHRIKE_MPP_WEIGHT_LOADS`'s
    precedent), so every ledger row is an A/B of one binary.

  Steps (TDD; the go/no-go is Step 1, and it needs no code at all):

  - [ ] Step 1 (spike A, the gate, **zero code**): three arms on the mini, one
        fresh server each, the 12k prompt once per server lifetime
        (`tools/prefill-measure.sh`; **the ledger's column names are not the
        prompt labels** — `tools/prefill-prompts.py:10-11` writes the
        12,285-token prompt as `prompt-6k.json`, the 3,756-token one as
        `prompt-2k.json` and the 25,245-token one as `prompt-12k.json`, so the
        ledger's 12k column is the script's `6k` label), reading `driver_ms`
        and `count` on `prefill_shared_expert->prefill_routed_tile` via
        `tools/prefill-ledger.py`. Record the startup log's slot count and
        allocated bytes for each arm.
        - **A0 control:** the production launch (`--ram-budget 8G`, pool,
          `tools/mini-deploy.sh:69`). Expect ≈ 22 ms per boundary.
        - **A1 half slab:** the same launch with `--ram-budget 4G`.
        - **A2 per-slot:** `--ram-budget 8G` with
          `env SHRIKE_EXPERT_CACHE_LAYOUT=per-slot
          SHRIKE_DECODE_EXPERT_EXECUTION=hit-fixup` — the second var is
          required, speculative decode refuses any layout but pool
          (`RealForwardRunner.swift:5793-5796`). Total allocation is unchanged;
          the *referenced* bytes per routed buffer fall from 9.06 GB to
          8 × ≈ 70.8 MB.

        | hypothesis | A1 predicts | A2 predicts |
        | --- | --- | --- |
        | residency of the referenced buffer, ∝ its length | ≈ halves | collapses, ≈ 1/16 |
        | residency ∝ the total allocation | ≈ halves | unchanged |
        | per-buffer work independent of size (scratch, argument buffers, first touch) | unchanged | unchanged |

        **Accept rule: A1 must fall by ≥ 35 % and A2 by ≥ 70 % for the
        residency model to survive.** Both flat → the model is falsified, lever
        A is a measured null, the task lands the ledger row that says so and
        proceeds with B alone. A1 down and A2 flat → the cost is the total
        allocation; the residency set is still the right tool, continue to
        Step 2. Note that A1 and A2 change cache pressure and therefore wall
        and the routed rows — `driver_ms` on this one boundary is the reading,
        not the wall.
  - [ ] Step 2 (only if Step 1 keeps the model alive): failing test
        `expertPoolResidencySetHoldsThePoolBuffer` in
        `ExpertPoolResidencyTests.swift` — build an `ExpertPoolResidency` over a
        small `makeBuffer(bytesNoCopy:)` allocation, assert non-nil,
        `allocationCount == 1`, `allocatedBytes >= length` after `commit()`, and
        that `attach(to:)` on a fresh queue does not throw. FAIL: the type does
        not exist. Then implement it — descriptor, `addAllocation`, `commit()`,
        `requestResidency()`, `queue.addResidencySet` — and wire it at model
        attach behind `SHRIKE_PREFILL_POOL_RESIDENCY`. **Keep the existing
        `useResource` calls**: they also carry usage and hazard information for
        the argument-buffer indirection, and this task is not the place to find
        out which of the two the driver was using. Re-run Step 1's A0 arm with
        `SHRIKE_PREFILL_POOL_RESIDENCY=set` against `=none` on one binary.
        **Bar: `driver_ms` 2.65 → ≤ 0.40 s at 12k, ≤ 0.20 s at 3.7k.** Read
        `memory_pressure -Q` before and after — `requestResidency()` on 9.06 GB
        of a 16 GB box is this arm's risk, and a pressure change is a finding.
        If the set attaches and `driver_ms` does not move, the *mechanism* is
        wrong even though the size scaling held; fall back to what A2 already
        priced — keep the pool's single `posix_memalign` (the I/O path needs its
        contiguity) and wrap each slot in its own `makeBuffer(bytesNoCopy:)`
        over the same pages, legal because `poolSlotStride` is page-rounded
        (`PreadExpertStreamer.swift:341-345`), so a routed buffer names 0.57 GB
        instead of 9.06. Record whichever landed.
  - [ ] Step 3 (probe for lever B, uncommitted): a `Shrike p12probe` line
        accumulated per prefill chunk under `SHRIKE_KERNEL_STATS`, splitting
        `:4982-5021` into `copy_ms` / `pairs_ms` / `validate_ms` / `sort_ms` /
        `group_ms` / `meta_ms` with a layer count — P5's precedent exactly
        (`Shrike p5probe`, task-5-report.md:505-521), built into an isolated
        scratch path (`swift build -c release --scratch-path
        /Volumes/BuildSSD/SwiftPM/Shrike-probe`) so `.build` is untouched, run
        on the M4 Pro at 3.7k, then reverted. It answers one question: is the
        16.2 ms mostly `sort_ms` + `validate_ms` (then Step 6 is worth landing)
        or mostly the copy and the pair build (then the reorder is the whole
        lever). Verdict line: the six terms per layer on the M4 Pro, and the
        M4 Pro's 2.8 ms total as the sanity check.
  - [ ] Step 4: failing test in `PrefillMoEGroupingTests.swift` —
        `countingSortMatchesTheComparisonSortExactly`: over randomized inputs
        (4,096 tokens × top-8 against 256 experts, with and without
        `expertSortKeys`, including duplicate sort keys and experts with zero
        pairs), assert `groupTokenExpertPairs(…, sort: .counting)` equals
        `(…, sort: .comparison)` element-for-element in `sortedPairs`,
        `perExpertOffsets`, `perExpertCounts`, `groups` and `tiles`. Add
        `countingSortRejectsTheSameInvalidMetadata` so the bitset validation
        throws the *same* error as the `Set` on each case
        `groupingRejectsInvalidMetadataBeforeKernelUse` (`:117-174`) already
        covers, in the same input order. FAIL: `SortKind` undefined.
  - [ ] Step 5 (lever B, the reorder): commit `sharedCB` at `:5039` without
        waiting; run `:4982-5021`'s routing after it; wait for it and call
        `recordKernelGPU` after the tile drain (`:5178-5181`) so its GPU
        interval is still recorded and a failed buffer still throws. Behind
        `SHRIKE_PREFILL_ROUTE_OVERLAP`. Ordering is by commit order on one
        queue — the same guarantee `:2577-2582` already names and the decode
        path already uses (`:5800-5806`) — and the buffers are disjoint
        (`h1`/`sharedGateScratch` against `routePartials`/`routedGateUpActScratch`,
        both reading `routedX`). `swift test --no-parallel` then the same under
        TSAN: this step changes host/GPU concurrency, so the sanitizer run is
        the point, not a formality.
  - [ ] Step 6 (conditional): land the counting sort and the bitset as the
        default **only if** Step 3 puts `sort_ms + validate_ms` above half the
        per-layer host time *and* Step 5's ledger leaves a residual
        `shared->routed` `host_ms` above 5 ms per layer-chunk — i.e. only if
        the routing still outruns the shared expert's 17.4 ms of GPU. Otherwise
        keep Step 4's tests (they cost nothing and pin the equality) and record
        the counting sort as a follow-on with the measured reason.
  - [ ] Step 7: five gates on the final tree — release build with zero
        warnings; `swiftlint lint --strict --baseline .swiftlint-baseline.json`
        (regenerate the baseline if `encodeRoutedMoEPrefill`'s length moved);
        markdown link check; `swift test --no-parallel`; the same under
        `env TSAN_OPTIONS=suppressions=tsan-suppressions.txt swift test
        --no-parallel --sanitize=thread`. Then `tools/golden-baseline.sh
        --check` on the M4 Pro with the server stopped: **expected IDENTICAL,
        short and long. A difference is a bug — do not recapture.** Commit
        `prefill: queue-resident expert pool and overlapped host routing (v12
        P12)`.
  - [ ] Step 8: ledger on both boxes (`tools/mini-deploy.sh --restart`,
        `tools/prefill-measure.sh`, fresh server per prompt, one send per prompt
        per server lifetime; **mini 3.7k + 12k is the verdict**, M4 Pro 3.7k +
        12k + 25k the check), then the mini golden check. Same-binary A/B per
        lever with the two env knobs so each arm is one binary. Design doc: the
        two follow-on bullets (`:614-622`, `:636-639`) are removed and become a
        landed `### Step 12 — …` section, plus an "**After P12**" ledger block
        after the After P9 table (`:324-337`). Plan: Task 12 `[x]` with the
        landed paragraph. Task review by a fresh reviewer; fixes folded into
        the commit (rebase and amend, never a fixup commit).

  **Verdict template.** Mini 3.7k and 12k, before → after:
  `prefill_shared_expert->prefill_routed_tile` total / `host_ms` / `driver_ms` /
  `queue_ms` / `count`; the two `->prefill_shared_expert` rows the same; gaps
  (span − busy) and GPU busy in ms/prompt-token; wall in ms/prompt-token and
  seconds. Then: which of Step 1's three arms fired and what that says the
  driver cost *is*; whether the residency set or the per-slot wrappers landed;
  the Step 3 probe's six terms; whether Step 6 landed and why; the four role
  rows unchanged within noise (they must be — no kernel moved); the golden
  outcome per box (**identical**, else the task failed); `memory_pressure -Q`
  on the mini before and after with the residency set attached; and the M4
  Pro's rows as the check, noting that its saving may sit in routed→routed
  rather than wall.

  **Risks.**
  - *The residency model is wrong.* The likeliest single outcome, and Step 1
    costs three server restarts to find out. A measured null lands as a ledger
    row (P8's precedent) and lever B carries the task.
  - *`requestResidency()` on 9.06 GB.* Wiring the slab permanently on a 16 GB
    box is the one change here that can move memory pressure. Read
    `memory_pressure -Q` before and after; if it degrades, fall back to the
    per-slot wrappers, which name 0.57 GB per buffer and wire nothing extra.
  - *The overlap is marginal by construction.* 17.4 ms of shared-expert GPU
    against 16.2 ms of host routing is a 7 % margin, and a ragged last chunk
    has fewer rows and less shared-expert GPU. If the residue reappears in
    `shared->routed` `host_ms`, Step 6 closes it — that is why it is
    conditional rather than dropped.
  - *Command-buffer ordering.* The reorder leaves two buffers in flight with no
    host wait between them for the first time in the *prefill* path. TSAN
    covers the host side, golden identity the GPU side; a golden difference
    here is an ordering bug, not numerics, and is debugged, never recaptured.
  - *AGX trap.* The known driver crash is a `.concurrent` encoder plus a later
    indirect dispatch on one command buffer (MEMORY.md, T2). Neither lever adds
    an encoder or changes a dispatch type — every encoder in this path is still
    created and ended inside its own encode call (task-5-report.md:180-183) —
    but the check belongs in the review.
  - *Two model processes.* Every arm is a server run: `pgrep -fl
    'ShrikeServer|ShrikeMac|ShrikeDecodeService|ShrikeCLI|…'` first, every
    time; never terminate a process this session did not start.

### Task 13: P13 — the GDN pre-scan chain and the dense GEMM shape: measure, then take the larger

- [x] **P13: two candidate levers, neither of them measured** — the audit's L1
  (the GDN pre-scan chain, modelled 0.32–0.63 ms/prompt-token) and L8
  (`kMPPAffineTileM` fixed at 64, the dense MPP GEMM never benched, modelled
  0–0.25) are the largest unclaimed items on the mini, and **both bands are
  arithmetic, not measurement**. Every earlier task in this chapter had a probe
  before it had a lever (P8's tile split fed P9; P9's byte-load pair fed P10);
  these two have nothing. So Step 1 is two bench-only additions at no production
  risk, and the kernel work is chosen by what they measure. Bars are stated **as
  formulas over Step 1's measurements**; the numbers below are those formulas at
  the middle of each modelled band, re-derived from the measured values before a
  line of kernel code is written. **The mini decides.** M4 Pro iteration check
  only — and for the dense arm it cannot even confirm the lever, because P10
  showed the boxes disagree about this exact shape (M1 dense −1 %, M4 Pro −17 %
  from the same 256-wide tile, `docs/v12-implementation-plan.md:2320-2325`).

  **LANDED 8692c3a (2026-09-03) — benches only; neither drafted arm proceeds.**
  Step 1 measured the chain at **C = 9.21 ms** per GDN layer-chunk on the mini
  (2.9 % of the role's 319.4; the modelled band was 50–91): conv 4.40, tail
  0.005, qk-norm 1.19, gated norm 1.65, 2 × rmsnorm 1.12, residual add 0.84,
  419.5 MB at 45.6 GB/s against a 59.9 GB/s same-run floor (7.0 ms). Step 2
  measured the dense headroom at **H ≈ 12.2 ms** per layer-chunk: (4096, 2048,
  8192) 83.2 ms = 1.65 TFLOPS = 99 % of the same-run MPS ceiling, (4096, 2048,
  4096) 85 %, (4096, 4096, 2048) 88 %, (4096, 2048, 32) 0.44 ms with no MPS
  twin; the five projections 166.3 ms. The rule would take the dense arm
  (0.40 × C = 3.7 vs 0.50 × H = 6.1), but neither clears the 0.15 ms/token
  threshold (chain 0.027 by the rule, 0.017 by the floor gap; dense 0.045), so
  Steps 4–9 were not run (P8's precedent). K256 vs K128 on the dense shape: −1…−3 % on the mini and
  20.11 → 20.22 ms on the M4 Pro — P10's "the dense projections follow the
  tile on the M4 Pro" was that box's state between two fresh servers (its
  routed row stands), corrected in the design doc. Step 1b, added when the GDN ledger would not close
  (56 + 166 + 9 = 231 of 319): **`prefill_router_block` 83.40 ms per
  layer-chunk on the mini = 0.051 TFLOPS = 2.9 % of the same-run MPS ceiling
  (2.45 ms); M4 Pro 18.10 ms = 5.7 %** — one threadgroup per token, 256 threads
  each walking the 2,048-long row with three loads and a byte extraction, the
  512 KB weight re-read per token, thread 0 doing the top-8 alone; in all 120
  layer-chunks: 10.0 s of the 80.38 s 12k wall (12.4 %). The GDN role's table
  closes at 315 of 319 ms. The router is Task 14. Gates on 8692c3a: release
  build 0 warnings, lint 0, links 0, `swift test --no-parallel` (see the
  ledger); TSAN folded into Task 14's run (bench-only code, no concurrency).
  Golden untouched (no production change).

  **Where the 2.34 goes.** After P12 (HEAD 188d2b7) the mini's 12k prompt
  (12,285 tokens, `tools/prefill-prompts.py:10`; 3 chunks of 4,096 → 120
  layer-chunks, 90 GDN + 30 attention) is 80.38 s wall = 6.54 ms/prompt token,
  busy 6.10, gaps 0.273; roles (measured, After-P10, unchanged by P12) gdn 2.34 ·
  routed tile 1.74 · attn 1.75 · shared expert 0.17. One `prefill_gdn_router`
  layer-chunk is 2.34 × 12,285 / 90 = **319.4 ms**; the role is one command
  buffer (`RealForwardRunner.swift:5058-5059`) holding, per the encode order:

  | term | ms per GDN layer-chunk | source |
  | --- | ---: | --- |
  | `gdn_chunk_factors` + `gdn_chunk_scan` | 56.07 | **measured** — `ShrikeBench gdn_scan`, mini, 0.383 TFLOPS (serial 183.6) |
  | 5 dense MPP projections, 275.95 GFLOP | 169–211 | **modelled** at 1.63–1.31 TFLOPS |
  | `prefill_router` block (4.29 GFLOP + top-8) | ≈ 3 | **modelled**, never benched |
  | conv mix + conv tail + qk norm + gated norm + 2 × `prefill_rmsnorm_bf16w_block` + 1 × `residual_add_fp16` | **50–91** | **modelled by subtraction** — this task's first arm |
  | total | 319.4 | measured |

  275.95 GFLOP = 2·4096·2048·(8192 + 4096 + 32 + 32) + 2·4096·4096·2048 — the five
  `encodeAffineProjection` calls at `RealForwardRunner.swift:3930` (qkv, n 8192),
  `:3951` (z, n 4096), `:3962` (a, n 32), `:3974` (b, n 32), `:4022` (out, k 4096
  → n 2048), each landing in `MPPPrefillInt4QMM.encode` at m = 4,096
  (`:3608-3637`); 1.31 TFLOPS is P9's measured grouped rate, 1.63 what
  `prefill_shared_expert` implies (audit L1). **The 50–91 ms residual has never
  been measured.** (The audit states 44–86 from the P9 role of 2.368; at 2.34
  minus the router it is 50–91 — they do not reconcile, one more reason to
  measure.)

  **What is in the chain, and its floor.** Per GDN layer-chunk at T = 4,096,
  C = `qkvDim` 8192, D 2048, `valueDim` 4096:

  | kernel | where | shape | bytes |
  | --- | --- | ---: | ---: |
  | `gdn_conv_mix_prefill` | `gdn.metal:280`, `GDN.swift:224` | 33.6 M threads, one per (channel, row), 4 taps | 134.2 MB |
  | `gdn_conv_tail_update` | `gdn.metal:314`, `GDN.swift:250` | 8192 × 3 | 0.1 MB |
  | `gdn_qk_norm` | `gdn.metal:390`, `GDN.swift:295` | 2·16 × 4,096 = 131,072 groups × 128 threads, **one element per thread**, 2 barriers | 67.1 MB |
  | `gdn_gated_norm` | `gdn.metal:744`, `GDN.swift:543` | 32 × 4,096 = 131,072 groups × 128 threads, same shape | 100.7 MB |
  | 2 × `prefill_rmsnorm_bf16w_block` | `prefill.metal:158`; called `RealForwardRunner.swift:2220`, `:2261` | 4,096 groups each | 67.1 MB |
  | 1 × `residual_add_fp16` | `utility.metal:95`; called `:2257` | 8.4 M threads | 50.3 MB |
  | | | | **419.4 MB** |

  419.4 MB is a **7.0 ms floor at 60 GB/s** (the audit's achieved M1 figure; the
  bench measures it same-run rather than assuming it) against a modelled 50–91 ms
  — **7–13× off bandwidth**, every kernel moving 2 bytes per thread.
  `gdn_qk_norm` normalises `gdnConvOut` in place and both scan kernels read
  `conv_out` again after it (`gdn_chunked.metal:45`, `:164`), so conv → qk-norm
  is a 67 MB round trip inside one buffer. The layer's other two residual adds
  (`:5230`, `:5234`) are in `tailCB` under `prefill_moe_reduce`, not here — the
  audit's L1 and L11 counted two adds in this role; the encode order has one.

  **What the dense shape dispatches.** Four of the five GDN projections and every
  attention projection go through `MPPPrefillInt4QMM.encode`, whose grid is
  `ceil(n/32)` × **64** at m = 4,096 (`MPPPrefillInt4QMM.swift:208-215`,
  `tensorops.metal:9`). A threadgroup dequants one `TILE_N × TILE_K` weight tile
  and multiplies it by `TILE_M` rows (`tensorops.metal:145-248`), so **every
  column tile's dequant is repeated 64× at this shape**. `TILE_M` has never moved
  — P8 swept `TILE_N` and the buffer count, P9 the loads and `TILE_K`, P10
  `TILE_K` again — and nothing benches the dense path: `ShrikeBench gemm`
  measures MPS fp16 (`GEMMBench.swift:22-30`), `routed_gemm` the *grouped* kernel
  at m = 128. P10 is the evidence that this matters and that the model does not
  carry: the 256-wide tile took the mini's grouped tile 4.93 → 4.48 ms (−9 %) and
  the routed role −7 %, but the GDN role only −1 % and attention 0.3 %, where the
  same binary moved GDN −17 % on the M4 Pro. **The 4,096-row dense kernel behaves
  differently from the 128-row grouped tile on the M1 and nothing says why.**

  **Out of scope, already claimed:** L2, L3, L4 (landed in P12), L5, L6, L7, L9,
  L10, and the audit's own "Not levers" list. One thing Step 1 exposes but does
  **not** take: production issues the GDN in-projection as **four** dispatches
  (n 8192, 4096, 32, 32) where `GEMMBench`'s `gdn_inproj_chunk4096` models one
  fused n = 12,288; fusing needs the four weight blocks contiguous in the
  `.gturbo` — a follow-on, like L7.

  **Step 1 — the two benches.** Bench-only code, no production path change.

  - **`ShrikeBench gdn_pre <iters>`** — the chain at the ornith 4,096-row shape
    (Hk 16, Hv 32, Dk = Dv 128, C 8192, D 2048, 4 taps), each kernel timed
    **alone** over synthetic buffers, with its bytes moved and achieved GB/s,
    plus the chain total. Mirrors `GDNScanBench.swift`: private fixture struct,
    `XorShift` fill, two warm-ups then `iterations` runs, `cb.gpuEndTime -
    cb.gpuStartTime`, one `print` per kernel. PSOs by name via
    `context.pipeline(_:)` (public, `MetalContext.swift:169`) — `gdn_qk_norm` and
    `gdn_gated_norm` **must** be built with `MetalFunctionConstant(index: 95,
    value: .uint32(128))`, the pair `GDN.swift:68-71`, `:76-79` uses, or the
    bench measures a different kernel. `residual_add_fp16` over the same 8.4 M
    elements is pure streaming, so its GB/s **is** the box's same-run floor.
  - **`ShrikeBench dense_gemm <iters>`** — `MPPPrefillInt4QMM.encode` at the four
    production dense shapes, all m = 4,096: **(k 2048, n 8192)** = the GDN qkv
    in-projection *and* the attention q-projection, one shape serving both,
    137.44 GFLOP; **(k 2048, n 4096)** z, 68.72; **(k 4096, n 2048)** out, 68.72;
    **(k 2048, n 32)** a/b, 0.537 each on a 64-threadgroup grid. Swept over
    `.n32b1` / `.n32k128b1` / `.n32k256b1` × `.byte` / `.vector` **in one
    process**, with the matching MPS ceilings (`qproj_chunk4096`,
    `oproj_chunk4096`, a new `gdn_zproj_chunk4096`) from the same run. Per shape
    and arm: ms, TFLOPS, and **share of the same-run ceiling** — the ratio P10's
    risks require, because the mini's ceiling bench drifts between runs (1.15
    TFLOPS at P6, 1.84 at P8/P9).

  **The decision rule.** On the mini let `C` = the measured chain total in ms per
  GDN layer-chunk and `H` = the measured dense headroom (over the five
  projections, measured ms − ms at that shape's same-run MPS ceiling), also per
  layer-chunk. **Take the chain arm if 0.40 × C > 0.50 × H, the dense arm
  otherwise** — 0.40 is the chain fraction the three fusions below address, 0.50
  the headroom fraction `TILE_M` reaches (P8's mini probe: unpack 7 % + weight
  loads 20 % of a tile whose whole remainder above a plain GEMM is 29 %; `TILE_M`
  halves both). **Proceed only if the winner's saving clears 0.15
  ms/prompt-token** — 90 × saving / 12,285 ≥ 0.15, i.e. ≥ 20.5 ms per
  layer-chunk, ≈ 1.8 s of the 80.38 s wall (2.3 %). If neither clears, land the
  benches and the measured verdict and stop, as P8 did.

  **Step 2a — the chain (if it wins).** Largest first:

  1. **Fuse conv → qk-norm.** One kernel, threadgroup per (head-slot, row) over a
     row's 64 slots (16 q + 16 k + 32 v, 128 channels each); 128 threads each
     compute one channel's 4-tap conv + SiLU into `threadgroup half tile[128]`,
     barrier, then q/k slots reduce and scale before the single device write.
     Deletes `gdn_qk_norm`'s 131,072 launches and 67.1 MB of round trip. **The
     fp16 rounding of the conv output before the sum of squares is
     load-bearing** — the fp32 reference does it (`GDNReference.swift:74`,
     `conv[ch] = Float(Float16(silu(acc)))`) and today the kernel gets it free
     from the device store. Fast-math elides a rounding on a register-resident
     value (T2, `d89d172`); a `threadgroup half` stage is a real memory round
     trip and cannot be elided. That is why the tile is `half`.
  2. **Vectorise `gdn_gated_norm`** (and `gdn_qk_norm`'s surviving decode form)
     to `half4`: 128 threads covering **four** (head, row) pairs, one simdgroup
     per pair, each thread a `half4`. 4× fewer threadgroups, `simd_sum` only —
     both `threadgroup_barrier`s and the `partial[]` array disappear.
  3. **A T-row `prefill_residual_add_rmsnorm_bf16w_block`** folding the
     `residual_add_fp16` at `RealForwardRunner.swift:2257` into the
     `prefill_rmsnorm_bf16w_block` at `:2261` — the fusion
     `residual_add_rmsnorm_bf16w` (`rmsnorm.metal:103`) already does in decode,
     one threadgroup per row. 16.8 MB × 120 fusions = 2.0 GB per prompt, and it
     runs in the attention role too, so `prefill_attn_router` moves with it.
  4. **Not doable, and why:** folding the gated norm into the scan epilogue. The
     scan's threadgroup owns a 32-column block of Dv (`kGDNChunkValueBlock`,
     `gdn_chunked.metal:23`; grid `valueHeadDim / 32` × Hv, `GDN.swift:519-521`)
     and the norm reduces over all 128 — the fold needs a cross-threadgroup
     reduction or a re-blocked scan. Dropped; item 2 takes that kernel instead.

  Numerics: item 1 is **bit-identical if the staged rounding holds** (same taps,
  same order, same fp16 rounding, same per-head reduction) and Step 5's probe
  either proves it or names the first differing element. Item 2 **reorders the
  sum of squares** (4 elements folded per thread before a 32-lane `simd_sum`
  instead of 128 across 4 simdgroups) — not bit-identical. Item 3 **is**
  bit-identical by the decode kernel's own argument (`rmsnorm.metal:95-101`): the
  add rounds through fp16 storage before the sum of squares and the reduction is
  the same `prefill_rms_block_inv` at the same `lsize`.

  **Step 2b — the dense GEMM shape (if it wins).** A `TILE_M` template axis on
  `mpp_prefill_affine_body` and one 128-row instantiation of the **plain** kernel,
  selected per dispatch. **The grouped kernel keeps 64 and must**:
  `MPPGroupedBlockMSL.row_tile_start` is `staging_row / 64`
  (`tensorops.metal:285-293`) and `PrefillGroupedRoutedMoE.groupedRowTile =
  MPPPrefillInt4QMM.tileM` (`:92`) — its row tiles *are* the expert blocks. So
  the axis is **not** a new `TileVariant` case (that would name a grouped kernel
  which must not exist) but a separate `PlainRowTile` enum with its own env
  override, consulted only in `encode`. Selection: 128 rows when `m ≥ 128` **and**
  `ceil(n/TILE_N) × ceil(m/128) ≥ 256` — admits n 8192 / 4096 / 2048, excludes
  the n = 32 a/b pair whose grid would fall 64 → 32 threadgroups on an 8-core M1.
  **Bit-identity is expected but not provable from source.** `TILE_M` is the
  `matmul2d_descriptor`'s first argument (`tensorops.metal:166-168`); the K
  reduction per output element is over the same `TILE_K` elements in the same
  tile order with the same `accumulator[e] += groupProduct[e]` fold (`:234`), so
  *which* values are summed *in what order* for a given (m, n) does not change —
  what changes is which simdgroup owns which output element inside MPP's opaque
  `run`. Prove it as P9 proved lever A: a `runPair` bit-equality assertion on
  `irregular` inputs across `variantShapes`
  (`MPPPrefillInt4QMMTests.swift:223-227`) plus m = 4,096 and a ragged m = 4,097,
  where the `globalM < rowEnd` store guard (`tensorops.metal:244`) is the only
  thing between the arms. If the probe fails, fall back to 2e-2 and carry a
  golden recapture into Step 7.

  **Bars.** Mini-first, 12k unless stated. **They are formulas**; the numbers are
  those formulas at the middle of each modelled band (chain `C` = 70 ms, dense
  `H` = 57 ms, i.e. 211 − 154 at the ceiling), re-derived from Step 1 measurements
  before Step 4.

  - Chain arm: `prefill_gdn_router` = 2.34 − 0.40 × C × 90 / 12,285 →
    **2.34 → ≤ 2.15** (claims 0.19 of a modelled 0.205); at 3.7k 2.38 → ≤ 2.19.
    `prefill_attn_router` **1.75 → ≤ 1.73** (item 3 only). Busy 6.10 → ≤ 5.93,
    wall **80.38 → ≤ 78.5 s**; at 3.7k busy 5.42 → ≤ 5.27, wall 24.32 → ≤ 23.8 s.
    Bench bars on the mini: fused conv+qk-norm ≤ **0.65 ×**
    (`gdn_conv_mix_prefill` + `gdn_qk_norm`) measured separately in Step 1;
    vectorised `gdn_gated_norm` ≤ **0.60 ×** its Step-1 ms; chain total
    ≤ **0.65 ×** its Step-1 total.
  - Dense arm: `prefill_gdn_router` = 2.34 − 0.50 × H × 90 / 12,285 →
    **2.34 → ≤ 2.24** (claims 0.10 of a modelled 0.21 — deliberately half the
    model, because the floor here is genuinely zero); `prefill_attn_router`
    **1.75 → ≤ 1.69**; busy 6.10 → ≤ 5.98, wall **80.38 → ≤ 79.2 s**. Bench bar:
    at (m 4096, k 2048, n 8192) on the mini, `TILE_M 128` ≤ **0.92 ×** `TILE_M 64`
    at the same `TILE_K` and load body, same run, no shape regressing over 3 %.
  - Both arms: golden **short identical on both boxes**; long may move only where
    a listed change reorders a reduction, recaptured once per box with the
    reason. Five gates green. The M4 Pro decides nothing — the chain arm should
    move there too, and the dense arm will probably move *more* there than on the
    mini, which P10's split says is not evidence.

  **Files:**
  - Add `sources/ShrikeBench/GDNPreScanBench.swift`
    (`static func runGDNPreScan(iterations:context:)` in an `extension
    ShrikeBench`, modelled on `GDNScanBench.swift`) and
    `sources/ShrikeBench/DenseGEMMBench.swift` (`runDenseGEMM`, calling the
    façade plus `runGEMMShape` for the same-run ceilings).
  - Add `sources/Shrike/Kernels/TensorCore/MPPPrefillDenseBenchmark.swift` — the
    public façade (`MPPPrefillInt4QMM` is internal), shaped like
    `PrefillRoutedGEMMBenchmark`, taking the variant and load body as
    **parameters** rather than through the process-wide statics so one process
    sweeps every arm against one ceiling.
  - Modify `sources/ShrikeBench/ShrikeBench.swift:79-93` (two `if kernelName ==`
    blocks — **`gdn_pre` must come before the `hasPrefix("gdn")` block at `:95`**
    or `runGDN` swallows it; `main` is ~100 lines against the 120 error bar and
    has no baseline entry) and `sources/ShrikeBench/GEMMBench.swift:22-31` (add
    `GEMMShape(label: "gdn_zproj_chunk4096", m: 4096, k: 2048, n: 4096)`).
  - Step 2a modifies `sources/Shrike/Metal/GDN/gdn.metal` (a fused
    `gdn_conv_mix_qknorm_prefill`; `gdn_gated_norm_body` `:696-742` gains the
    `half4` walk), `prefill.metal:158` (a
    `prefill_residual_add_rmsnorm_bf16w_block` beside it), `GDN.swift:224-247`,
    `:295-322`, `:527-566`, `PrefillPrimitives.swift:50-83`, and
    `RealForwardRunner.swift:3985-4011` (conv + tail + qk-norm becomes
    conv-fused + tail) and `:2255-2267` (the pair becomes one call).
    `encodeLinearAttentionPrefill` already carries `lint:allow-long`
    (`:3894-3899`); `executePrefillChunk`'s `function_body_length` baseline entry
    **shrinks** — regenerate the baseline if it goes stale.
  - Step 2b modifies `tensorops.metal:145-248` (`mpp_prefill_affine_body` gains a
    `TILE_M` template parameter replacing `kMPPAffineTileM` at `:166`),
    `:250-269` (`MPP_AFFINE_KERNEL` gains the argument; `rowOrigin` takes
    `TILE_M`), `:271-276` (one instantiation, `…_n32k256m128b1`) —
    `MPP_GROUPED_KERNEL` (`:298-335`) passes 64, otherwise untouched; and
    `MPPPrefillInt4QMM.swift:25` (`tileM` stays the grouped path's 64 with a
    one-sentence why), `:86-116` (init builds the 128-row pipeline too),
    `:208-215` (grid height and pipeline pick), a `PlainRowTile` enum + static
    beside `weightLoads` at `:45-47`. Step 6 adds `row_tile=` to
    `prefillProjectionPath` (`RealForwardRunner.swift:185-188`); the leading
    token is unchanged, so `ServerInference.swift:819` and
    `tools/mini-deploy.sh:79`'s `prefill_projection_path=[a-z0-9-]*` grep keep
    matching.
  - Tests: `tests/Shrike/Core/Kernels/GDN/GDNKernelTests.swift` — the oracle is
    `GDNReference.normalize(qkvRaw:)` (conv + SiLU + qk-norm in fp32 with the
    fp16 roundings, `GDNReference.swift:61-93`) and `.gatedNorm(y:z:)` (`:136`);
    `prefillChunkMatchesSequentialDecode` (`:207`), its per-channel twin (`:466`)
    and `shortChunkTailCarry` (`:713`) are the end-to-end guards.
    `.../TensorCore/MPPPrefillInt4QMMTests.swift` — `runPair` (`:229`),
    `expectBitIdentical` (`:316`), `variantShapes` (`:223`),
    `runShape(compareCPUReference:)` (`:139`).
  - Unchanged deliberately: `MPPPrefillInt4QMM.tileK` (the matrix path's
    admission unit), the six `TileVariant` cases and their env knobs, the grouped
    kernel and `groupedRowTile`, the scan and factors kernels,
    `kGDNChunkValueBlock`, the decode `gdn_conv_mix_decode` and
    `residual_add_rmsnorm_bf16w` paths.

  **Interfaces:** consumes `MetalContext.pipeline(_:constants:)` (`:173`) and
  `cb.gpuStartTime`/`gpuEndTime`; `GDN.encodeConvPrefill`/`encodeQKNorm`/
  `encodeGatedNorm` keep their shapes, the fused encoder replacing the first two.
  Produces `ShrikeBench.runGDNPreScan(iterations:context:)` and
  `runDenseGEMM(iterations:context:)`; a `public enum MPPPrefillDenseBenchmark`
  with `Result { m, k, n, variant, weightLoads, millisPerLaunch, gflop, tflops }`
  and `run(context:iterations:m:k:n:variant:weightLoads:) throws -> Result`; and,
  for Step 2b, `enum PlainRowTile: String { case m64, m128 }` with
  `static let plainRowTile` from `SHRIKE_MPP_ROW_TILE` (64|128) plus
  `init(context:weightBits:variant:weightLoads:rowTile:)` so a test pins it
  without a process-wide env var — **a second axis, not a `TileVariant` case**.
  In Metal, `template <int TILE_M, int TILE_N, int TILE_K, int BUFFERS>
  mpp_prefill_affine_body(...)` and `MPP_AFFINE_KERNEL(NAME, TILE_M, TILE_N,
  TILE_K, BUFFERS)`; `MPP_GROUPED_KERNEL` passes 64 and keeps its six names.
  Lint: neither bench file has a baseline entry, so a `function_body_length`
  violation fails the strict gate — keep each `run*` under 120 lines by
  extracting the fixture into a private struct, as `GDNScanBench` does
  (`runMoE`/`runGDN` carry `lint:allow-long` and are **not** the pattern for new
  code). Comments: repo rule. Two earn their place — why the conv result stages
  through `threadgroup half`, and why the grouped kernel cannot take a 128-row
  tile. Nothing else.

  Steps (TDD; the measurement precedes any kernel decision):

  - [ ] Step 1: `ShrikeBench gdn_pre`. New file, no production change, no test —
        a bench is not under test; its bar is that its PSOs and function
        constants match `GDN.swift`, which the review checks. Five gates, then
        `swift run -c release ShrikeBench gdn_pre 20` on the M4 Pro, then the
        mini: `tools/mini-deploy.sh` (copy only — it also copies the `*.bundle`
        directories carrying the shader sources; P8's bench first failed because
        only the binary was copied), `scp .build/release/ShrikeBench
        macmini:shrike-runtime/bin/` (`mini-deploy.sh:24`, `:36`, `:47` copy only
        ShrikeServer / ShrikeCLI / ShrikeRepack), `ssh macmini 'pgrep -fl
        ShrikeServer'`, stop production, run, relaunch with
        `tools/mini-deploy.sh --restart`.
  - [ ] Step 2: `ShrikeBench dense_gemm` and the façade. Five gates, same deploy
        and run on both boxes: `swift run -c release ShrikeBench dense_gemm 20`.
  - [ ] Step 3 (the gate): fill the ledger — per kernel ms, bytes, GB/s and the
        chain total `C`; per dense shape ms, TFLOPS, ceiling share and the
        headroom `H`. Apply the decision rule and re-derive the chosen arm's bars
        from the measured value. **Record the Step 1/2 verdict in the design doc
        whichever way it goes** — the benches are the durable deliverable even if
        neither arm proceeds. Commit `prefill: bench the GDN pre-scan chain and
        the dense MPP GEMM (v12 P13)`.
  - [ ] Step 4: failing tests for the chosen arm.
        **Chain:** `fusedConvQKNormMatchesTheReference` in `GDNKernelTests` —
        against `GDNReference.normalize` at 2e-2 `maxAbs`/`rel`, **plus bit
        equality against the unfused `gdn_conv_mix_prefill` + `gdn_qk_norm`
        pair** (the elision probe: a 2e-2 assertion alone would pass while the
        arithmetic silently changed); `vectorisedGatedNormMatchesTheReference` at
        2e-2; `prefillResidualAddNormIsBitIdenticalToThePair` at T = 3 and 4,096.
        **Dense:** `oneTwentyEightRowTileIsBitIdenticalToTheSixtyFourRowTile` —
        `runPair` `.m128` against `.m64` over `variantShapes` plus
        (4096, 8192, 2048) and a ragged (4097, 2048, 512), `irregular: true`,
        `expectBitIdentical`; and `oneTwentyEightRowTileFallsBackOnANarrowGrid`,
        asserting an n = 32 dispatch still returns `.affineThreadgroupF16` and is
        bit-identical because it took the 64-row pipeline. `swift test
        --no-parallel --filter
        "GDNKernel|GDNChunkedScan|MPPPrefillInt4QMM|PrefillSharedExpert"` → FAIL.
  - [ ] Step 5: implement; Step 4 PASS. Then the probe: for the chain, tighten
        the fused conv+qk-norm to bit equality and run it — passes → land the
        assertion, golden stays identical for that item; fails → record the first
        differing element, keep 2e-2, carry "golden differs on the long profile"
        into Step 7. For the dense arm the bit-equality assertion **is** Step 4,
        and its failure means 2e-2 with the reason recorded.
  - [ ] Step 6 (the spike, the second gate): re-run the Step 1/2 bench for the
        chosen arm on both boxes; the mini decides. **Accept rule: the mini
        clears every bench bar above, or beats its Step 1/2 control by ≥ 5 % with
        no shape regressing more than 3 %.** Arms within 3 % are a tie (P6 saw
        4 % between nominally identical staging arms), broken toward the
        bit-identical arm. If the mini shows nothing, land the tests and the
        kernel behind its env override with the default unchanged, and say so —
        the P8 precedent.
  - [ ] Step 7: five gates —
        `swift build -c release 2>&1 | grep -E "warning:|error:"` (empty),
        `swiftlint lint --strict --baseline .swiftlint-baseline.json`,
        `python3 tools/check-md-links.py`, `swift test --no-parallel`, and
        `env TSAN_OPTIONS=suppressions=tsan-suppressions.txt swift test
        --no-parallel --sanitize=thread`. Then `tools/golden-baseline.sh --check`
        (M4 Pro, server stopped): **short expected IDENTICAL on both boxes**;
        long identical unless a listed change reordered a reduction, in which
        case recapture once per box and record the before/after greedy digests
        (P10's, unchanged through P12: M4 Pro long `e04d4e8ee7f1590d`, M1 long
        `899a25e60a365e60`). Commit `prefill: <the chosen lever> (v12 P13)`,
        baseline via `git commit --only` if there is one.
  - [ ] Step 8: ledger on both boxes. `tools/prefill-measure.sh <host> <port>
        <promptdir> <outdir> <tag> 3p7k 12k` against a **fresh server, one send
        per prompt per server lifetime** (a repeat hits the multi-prefix prompt
        cache and prefills only a suffix), roles via `tools/prefill-ledger.py` on
        the server log with a distinct tag per box; **mini 3.7k + 12k is the
        verdict**, M4 Pro 3.7k + 12k + 25k the check. `tools/mini-deploy.sh
        --restart`, mini golden check (recapture only if Step 7 established a
        deliberate change), scp into `baselines/`.
  - [ ] Step 9: design doc — a landed section carrying the Step 1/2 bench tables
        (they close L1's and L8's bands whichever arm won), an "**After P13**"
        ledger block, and the GDN row of "Where the time goes" gaining the
        measured split. Follow-ons recorded: the fused n = 12,352 in-projection;
        the gated-norm-in-scan fold and what would make it possible; the arm this
        task did **not** take, with its measured size. Plan: Task 13 `[x]` with
        the landed paragraph. Task review by a fresh reviewer; fixes folded into
        the commit (rebase and amend, never a fixup commit).

  **Verdict line template** (the controller fills in the measured values):

  > **LANDED `<sha>` (`<date>`): measured on the mini `prefill_gdn_router`
  > 2.38 → `<x>` ms/prompt-token (3.7k; 2.34 → `<x>` at 12k),
  > `prefill_attn_router` 0.95 → `<x>` / 1.75 → `<x>`, `prefill_routed_tile`
  > 1.79 → `<x>` / 1.74 → `<x>`, `prefill_shared_expert` 0.17 → `<x>`, GPU busy
  > 5.42 → `<x>` and 6.10 → `<x>`, wall 24.32 → `<x>` s and 80.38 → `<x>` s.**
  > Step 1 measured the chain at `<C>` ms per GDN layer-chunk (`<x>` % of the
  > role's 319.4; modelled 50–91) against a `<x>` GB/s same-run floor: conv
  > `<x>`, tail `<x>`, qk-norm `<x>`, gated norm `<x>`, 2 × rmsnorm `<x>`,
  > residual add `<x>`. Step 2 measured the dense headroom at `<H>` ms per
  > layer-chunk: (4096, 2048, 8192) `<x>` ms = `<x>` TFLOPS = `<x>` % of the
  > same-run MPS ceiling, (4096, 2048, 4096) `<x>` %, (4096, 4096, 2048) `<x>` %,
  > (4096, 2048, 32) `<x>` %. The rule took the **`<chain | dense>`** arm
  > (0.40 × C = `<x>` vs 0.50 × H = `<x>`). Bars `<cleared | missed>` (gdn
  > ≤ `<x>`, attn ≤ `<x>`, wall ≤ `<x>` s; the bench bar against a measured
  > `<x>`). M4 Pro check: `<x>`. Numerics: `<the bit-identity outcome per item>`.
  > Golden: short identical on both boxes; long `<identical | recaptured once
  > per box, sha256 M4 Pro e04d4e8ee7f1590d → <x>, M1 899a25e60a365e60 → <x>>`.
  > Five gates green (`<n>` tests, TSAN 0 reports).

  **Risks:**
  - **The half round-trip elision — how the chain arm comes back wrong.**
    Fast-math elides half roundings on register-resident values (T2, `d89d172`);
    the fp32 reference and today's kernel both round the conv output to fp16
    *before* the sum of squares. The `threadgroup half` staging tile is the fix,
    and Step 4's bit-equality assertion against the unfused pair proves it took.
  - **Occupancy on `TILE_M 128` — how the dense arm comes back a null, and there
    is a measured prior that it will.** Both cooperative tensors are M × N
    (`tensorops.metal:188-191`), so 128 rows takes them 32 → 64 fp32/thread.
    P8's mini arms: `n32b1` 8.25 ms, `n64b1` 9.69, `n64b2` 10.21
    (`docs/v12-prefill-matrix-kernels.md:599-601`) — doubling the accumulator
    cost 17 % on the M1 and P8's verdict named it as the cause. `TILE_N 64` also
    doubled the threadgroup weight tile, so it is not a clean isolate, but it is
    the strongest evidence on record and it points at a regression. Only the
    bench separates the two; Step 6 is the gate.
  - **Grid fill.** `TILE_M 128` halves the grid height; the n = 32 a/b
    projections fall 64 → 32 threadgroups on an 8-core M1. The guard excludes
    them, but its threshold is unmeasured — Step 2's per-shape numbers set it.
  - **`maxTotalThreadsPerThreadgroup`.** Both encoders dispatch
    `threadExecutionWidth * 4` = 128 unconditionally
    (`MPPPrefillInt4QMM.swift:211-215`, `:316-318`); a 128-row pipeline whose
    register pressure drops its max below 128 produces an invalid dispatch. The
    `#expect(commandBuffer.error == nil)` in `runPair` catches it on Step 4's
    first run.
  - **The bench sums kernels that production overlaps.** Each `gdn_pre` kernel
    runs alone in its own command buffer; in production the seven run back to
    back in one CB where the driver can overlap one's tail with the next's head.
    `C` is therefore an **upper bound**, and the Step 8 ledger is the verdict —
    the bench sizes the lever, it does not score it.
  - **Same-run ceilings only.** The mini's ceiling bench drifts between runs
    (1.15 TFLOPS at P6, 1.84 at P8/P9 with the tile unchanged), so `H` and every
    dense bar is a *share* of the ceiling measured in the same process, never an
    absolute ms.
  - **Golden moves on the long profile.** The vectorised norms reorder a sum of
    squares; that is expected and recaptured once per box with the measured
    reason. **A short-profile change is a bug**, not a recapture.
  - **The mini is production.** Both benches stop the server on 8081 and relaunch
    it; Turbo on 8080 is a different project and is never touched. One model
    process at a time — `pgrep` first, every time.

### Task 14: P14 — the prefill router block on an operand-reusing kernel

- [x] **P14: the largest unclaimed cost in the chapter** — `prefill_router_block`
  runs in every one of the 120 layer-chunks of a 12k prefill and costs
  **83.40 ms each on the mini: 10.0 s of the 80.38 s wall, 12.4 %**, at
  **2.9 % of the same-run MPS ceiling** at its own shape. It is one threadgroup
  per token, 256 threads each walking a 2,048-long row with a byte extraction
  and three loads per element, the whole 512 KB weight re-read by every one of
  the 4,096 threadgroups, and the top-8 done by thread 0 while 255 lanes wait.
  This task keeps the arithmetic exactly as it is and changes only where the
  operands come from and which thread owns what. **Targets on the mini
  (M1, 16 GB, the box that decides): `ShrikeBench router_block` per launch at
  T 4,096 / D 2,048 / 256 experts / top-8 / int8 83.40 → ≤ 15.0 ms (5.6×,
  0.286 TFLOPS, 16.3 % of the 1.751 TFLOPS ceiling); 12k wall 80.38 → ≤ 72.2 s
  (6.54 → ≤ 5.88 ms/prompt-token — under the chapter's 6.3 target for the first
  time); GPU busy 6.10 → ≤ 5.44; `prefill_gdn_router` 2.34 → ≤ 1.84 and
  `prefill_attn_router` 1.75 → ≤ 1.59 ms/prompt-token; 3.7k wall 24.32 →
  ≤ 21.9 s. Golden `--check` **IDENTICAL on both boxes and both profiles** is a
  hard bar, not an expectation — the default arm is bit-identical by
  construction and a difference is a defect, never a recapture.**

  Derivation of the wall bars from the one measured number: 83.40 − 15.0 =
  68.4 ms saved per layer-chunk × 120 layer-chunks = 8.21 s; 80.38 − 8.21 =
  72.17 s ÷ 12,285 prompt tokens = 5.875 ms/token; busy 6.10 − 8210/12285 =
  5.43. The two router roles carry the whole cut: the GDN role is 319.4 ms per
  layer-chunk over 90 chunks (2.34 × 12,285 ÷ 90), 319.4 − 68.4 = 251.0 →
  1.839 ms/token; the attention role is 716.6 ms over 30 chunks, 716.6 − 68.4 =
  648.2 → 1.583 ms/token. At 3.7k the chunk is 3,756 rows, so the router scales
  to 83.4 × 3756/4096 = 76.5 ms over 40 layer-chunks = 3.06 s of the 24.32 s
  wall; at the bar it is 13.8 × 40 = 0.55 s, so −2.51 s → 21.81 s (5.81
  ms/token). **Stretch: ≤ 10.0 ms** (8.3×, 24.5 % of the ceiling) → −8.81 s →
  71.6 s / 5.83 ms/token. **Conservative floor** if only the most defensive
  staging survives the bit-equality test (V3 below): ≤ 25 ms (3.3×) → −7.01 s →
  73.4 s / 5.97 ms/token, still under the 6.3 target. **Abandon rule: if no arm
  beats today's kernel by ≥ 3× on the mini (≤ 27.8 ms), land the tests and the
  new kernel behind its knob, keep `prefill_router_block` as the default, and
  say so in the verdict** — P8's precedent, a measured null is a result.

  **LANDED dfaa69e (2026-09-03): measured on the mini `prefill_gdn_router` 2.36
  → 1.79 ms/prompt-token (3.7k; 2.34 → 1.78 at 12k), `prefill_attn_router`
  0.95 → 0.76 / 1.75 → 1.57, `prefill_routed_tile` 1.79 → 1.79 / 1.74 → 1.74,
  `prefill_shared_expert` 0.17 → 0.17, GPU busy 5.42 → 4.67 and 6.09 → 5.34
  (the same-binary block arm at 12k: 6.09, wall 80.41 s), wall 24.32 → 21.52 s
  and 80.41 → 70.76 s — 5.76 ms/prompt-token, under the chapter's 6.3 target
  for the first time.** Every bar cleared: bench ≤ 15.0 → **10.0 ms** (8.3×,
  0.429 TFLOPS, 24 % of the 1.75 TFLOPS same-run ceiling; the ≤ 10.0 stretch
  met), wall ≤ 72.2, busy ≤ 5.44, gdn ≤ 1.84, attention ≤ 1.59, 3.7k ≤ 21.9.
  Bench per box, block → tiled: mini 83.4 → 10.0 ms; M4 Pro 18.1 → 2.15 ms
  (8.4×); token block on the mini 4 / 8 / 12 / 16 / 24 = 11.5 / 10.4 / 10.0 /
  14.9 / 12.6 ms, 12 the default on both boxes (threadgroup memory 12 KB + 3 KB
  at 12; 24 is the cap, 32 would need 40 KB). The ladder: V1 (every thread
  stages the products, the token threads sum the staged products in k order)
  proved bit-identical — **Metal does not contract `sum_x += xv` in the block
  kernel** — and so did token-minor `float4` loads; neither moved the mini
  (17 ms at 24 tokens), nor did an 8 KB chunked score transpose (worse, 21 ms,
  reverted). The M1's limit was the per-thread byte walk of its weight row (32
  rows 2 KB apart per SIMD group, a cache line per element); loading a group's
  weight bytes as `uint4`s and unpacking from registers took it 16.8 → 10.0 ms,
  with the byte path kept for an unaligned base (tested at a 13-byte offset,
  bit-identical). The top-8 scan is one thread per token in the original
  expert order (`prefill_router_select`, extracted verbatim and shared by both
  kernels). The `scores_only` arm was not built — the rungs priced the phases
  by elimination. M4 Pro check: 12k block → tiled gdn 0.517 → 0.403, attention
  0.327 → 0.292, busy 1.353 → 1.204, wall 27.61 → 25.68 s; 3.7k wall 9.97 →
  9.33 s; 25k wall 64.03 → 58.97 s. The ledger's cut per layer-chunk is 77.4 ms
  from both roles (gdn −0.567 × 12,285 / 90, attention −0.189 × 12,285 / 30)
  against the bench's 73.4 — the bench's fixture is on shared storage, the
  production buffers private. Numerics: bit-identical by construction and
  by test (eight cases incl. 4-bit, sigmoid, partial blocks, unaligned bases).
  Golden IDENTICAL on both boxes, short and long (digests unchanged: M4 Pro
  long `e04d4e8ee7f1590d`, M1 long `899a25e60a365e60`). Lint baseline
  regenerated for `ServerInference.load` (181 → 182). Five gates green (1165
  tests, TSAN 0 reports in 1,820 s).

  **What Task 13 measured.** `ShrikeBench router_block 20`, the production
  encoder driving the production pipeline (`PrefillRouterBenchmark.run`, so the
  function constant and threadgroup width cannot drift from the runner's):

  | box | per launch | TFLOPS | MPS ceiling at (4096, 2048, 256) | share |
  | --- | ---: | ---: | ---: | ---: |
  | mini (M1) | **83.40 ms** | 0.051 | 2.45 ms / 1.751 TFLOPS | **0.029** |
  | M4 Pro | 18.10 ms | 0.237 | 1.04 ms / 4.140 TFLOPS | 0.057 |

  (mini: `p13-mini-router-block.log`, quoted in `progress.md:265`; M4 Pro:
  `task-13-report.md:245-246`; the ceiling is `router_chunk4096` run in the same
  process, `ShrikeBench.gemmShapes`.) The GDN role's cost table closes on the
  mini with it — scan 56.1 + five projections 166.3 + pre-scan chain 9.2 +
  router 83.4 = 315.0 of the measured 319.4 ms per layer-chunk, 98.6 %
  (`progress.md:265`) — so there is no third term hiding behind this one. The
  120 layer-chunks are the model's shape: 40 MoE layers (30 GDN + 10 attention)
  × three 4,096-row chunks at 12,285 prompt tokens.

  **Where the 83.4 ms goes** (modelled from `prefill.metal:372-486`; no
  per-phase measurement exists — Step 4's `scores_only` arm gets one). Per
  (token, expert, element) triple: `prefill_affine_value` (shift, `>>3`, `&7`,
  byte load, conditional second byte, shift, mask ≈ 6 ALU + 1 load), `float(q)`,
  `float(xg[k])` and `float(eg[k])` (2 loads + 2 widens), one multiply, one
  `fma`, one add — **≈ 13 scalar ops and 3 loads per triple**; 2.147e9 triples
  per layer-chunk = 2.8e10 ops in 83.4 ms = 335 Gops/s. The top-8 on thread 0
  (`:424-486`) is 256 compares plus ≈ 40 insertions of up to 16 moves plus 8
  `exp`s ≈ **2–3 k scalar ops per token on one lane while 255 idle** —
  modelled at ≈ 8 % of the kernel, which Step 4 checks rather than assumes.

  **The design constraint (a ruling, not a preference).** The default arm keeps
  the router's exact per-(token, expert) fp32 arithmetic: the same byte
  extraction producing the same integer `q`, the same
  `xv = float(x[k]) * float(e[k])`, the same two accumulations per element in
  the same ascending `k` order, the same per-group
  `acc = fma(s, dot_qx, acc); acc = fma(b, sum_x, acc)`, the same tie rule
  `s > top || (s == top && e < idx)` scanned over experts ascending, the same
  softmax. Only **where the operands come from and which thread owns which
  (token, expert)** changes, so logits, indices and route weights are
  bit-identical by construction, routing cannot flip, and golden must not move.
  The old kernel stays as the A/B reference under
  `SHRIKE_PREFILL_ROUTER=block` and as the test's comparand.

  **The kernel.** Thread mapping, in two lines: **one threadgroup owns a block
  of `TOK` consecutive tokens and all 256 experts; thread `e` owns expert `e`
  (the existing `for e = tid; e < NE; e += tg_size` stride is kept) and carries
  `TOK` accumulator pairs, so one weight read serves `TOK` tokens from
  registers.** The hidden row's contribution is staged in threadgroup memory
  once per (token, group) and broadcast to all 256 experts. Grid =
  `ceil(T / TOK)` threadgroups × the same `tgWidth` the encoder already
  computes.

  - **W is not staged.** Each thread reads its own expert's row, so there is no
    inter-thread reuse for threadgroup memory to buy; the reuse is across tokens
    and it lives in registers — a group's 64 bytes (int8) or 32 bytes (4-bit)
    load once and serve all `TOK` tokens. That also settles "stage dequantised
    values?": no. `q` is an integer either way so `float(q)` is exact and
    staging it would be bit-identical, but it costs 4× the memory for a value
    only one thread reads. **The byte extraction stays**,
    `prefill_affine_value` unchanged, over a register-held chunk instead of
    device memory. 4-bit keeps working untouched: a group is 32 bytes at byte
    offset `g * 32` in a row of `D * bits / 8` = 1,024 bytes, always aligned.
  - **`x ⊙ e` is staged**, fp32, `TOK × 64` per group = 4 KB at TOK 16. All 256
    threads read the *same* address at a given `(t, k)` — a threadgroup
    broadcast, no bank conflict — so the two device loads and two widenings per
    element are paid once per (token, k) instead of 256 times.
  - **`sum_x` is expert-independent.** `sum_x = Σ_k float(x[k]) * float(e[k])`
    over a group does not depend on the expert, yet today it is recomputed 256
    times per (token, group). Hoisting it — by one thread per token in the same
    ascending `k` order, never a tree reduction — removes one add per triple.
    **Whether that is bit-identical is a compiler question, not a reading
    question**: Metal's fast math may already contract `sum_x += xv` into an
    `fma` in today's kernel, in which case a hoist from a materialised `xv`
    differs in the last ulp (MEMORY.md's "Metal half round-trip elision"
    family). Step 3 therefore implements a ladder and the test picks the rung:
    **V1** stages the product `xe[t][k]` and hoists `sum_x[t][g]` — inner loop
    1 threadgroup load + 1 `fma` ≈ 2.4 ops/triple with the extract amortised
    over `TOK`, modelled 5.4×; **V2** stages the product and keeps
    `sum_x += xe[t][k]` inline, ≈ 3.4 ops, modelled 3.8×; **V3** stages `x` and
    `e` separately as fp32 and reproduces today's three source lines verbatim
    (`xv = xs[k] * es[k]; dot_qx = fma(q, xv, dot_qx); sum_x += xv;`) so
    whatever the compiler does it does identically in both kernels, ≈ 5.4 ops,
    modelled 2.4× on ALU count. Instruction-issue floors on the mini's 8-core
    GPU (≈ 1.3e12 lane-ops/s, modelled): V1 3.3 ms, V2 5.0, V3 8.3, against
    2.45 ms of matrix ceiling — so ≤ 15 ms is 3–4.5× off the floor of whichever
    rung survives, which is where a scalar kernel with staged loads has landed
    before (P1's shared expert, P9's vectorised loads).
  - **The top-k is parallelised by token, not by expert.** After the score
    barrier, thread `t` (for `t < TOK`) runs the *existing* scan over experts
    `0…NE-1` ascending for its own token — identical insertion, tie rule and
    softmax, writing `out_indices[row * top_k + i]` and `out_weights[…]` as
    today. `TOK` scans run concurrently, so the serial cost divides by `TOK`;
    the `256 − TOK` idle lanes are a bounded waste (at TOK 16, ≈ 6 % of the
    lanes for ≈ 1/32 of the kernel's time). **A simdgroup-parallel top-k is
    forbidden here** — it reorders the scan and breaks the `s == top` tie rule,
    the one place a bit difference becomes a routing flip.
  - **The sigmoid variant** (`cfg.routerUsesSigmoidScores`, true only for
    `family == .kimiLinear48b`, `ModelTypes.swift:459-461`; ornith and qwen36
    take the softmax path the bench measured) shares the same body through the
    same `sigmoid_scores` literal and `scaling` argument, and gets the same
    treatment and the same bit-equality test.

  **Traffic and budget, before → after** (per layer-chunk, T 4,096, D 2,048,
  256 experts, int8; modelled from the code):

  | | today | TOK 8 | TOK 16 |
  | --- | ---: | ---: | ---: |
  | weight-read amplification | 4,096× | 512× | 256× |
  | weight bytes read | 2.147 GB | 268 MB | 134 MB |
  | hidden + `effective_scale` load issue | 8.59 GB | 33.6 MB | 17.8 MB |
  | threadgroup memory / threadgroup | 1.0 KB | 10.1 KB | 20.1 KB |
  | fp32 accumulators + W bytes per thread | ≈ 20 regs | ≈ 38 | ≈ 70 |

  Threadgroup memory is `TOK × 256` fp32 scores + `TOK × 64` fp32 staged
  products + `TOK` fp32 sums, against the 32 KB limit the repo already asserts
  against (`attention_matrix.metal:278`, `tensorops.metal:165`). The hidden and
  `effective_scale` rows are 4 KB each and were L1-resident, so their 480× cut
  is instruction issue, not DRAM — the weight amplification is the DRAM/L2 one.
  **The occupancy trade is the open question**: 20.1 KB at TOK 16 may leave one
  threadgroup resident per core on the M1, and ≈ 70 registers may cost
  occupancy again. That is why `TOK` is a function constant Step 4 sweeps
  rather than a number this draft picks. The 8 tokens × 32 experts tile the
  brief floats is rejected: it buys the same amplification (4,096 threadgroups
  × 64 KB = 268 MB, identical to TOK 8) but no threadgroup then sees all 256 of
  a token's scores, so it needs a `T × 256` fp32 logits buffer (4 MB of new
  scratch) and a second top-k kernel. Same traffic, more surface.

  **The second arm — the logits as an MPP GEMM — only if the exact-order kernel
  misses the abandon bar.** `Σ_k (s_g q_k + b_g)(x_k e_k)` is exactly the int8
  affine-dequant GEMM of `W` against `x ⊙ e`, and `MPPPrefillInt4QMM.encode`
  already supports 8-bit weights (function constant 78,
  `MPPPrefillInt4QMM.swift:78`) at m 4,096 × n 256 × k 2,048 through
  `n32k256b1` (tileK 256, of which 2,048 is a multiple), producing fp16 `y`; a
  top-k kernel then runs over the logits. **Its cost, plainly: it rounds
  `x ⊙ e` to fp16 into a 16 MB staging buffer, reorders the K reduction, and
  returns fp16 logits the top-k then compares — three routing-level numerics
  changes. Near-tie experts can flip, the greedy text can change, and golden
  moves on *every* profile including short, on both boxes.** A chapter-scale
  numerics event for a kernel whose exact-order form is modelled to reach
  16–25 % of the same ceiling — which is why it is second and conditional.

  **Files:**
  - Modify: `sources/Shrike/Metal/Prefill/prefill.metal` — add
    `prefill_router_block_tiled_body` plus `prefill_router_block_tiled` and
    `prefill_router_block_tiled_sigmoid` beside the existing pair, and
    `constant uint FC_PREFILL_ROUTER_TOKENS [[function_constant(123)]]` (123 is
    free; the highest index in use across `sources/Shrike/Metal/` is 122,
    `prefill.metal:29`). **Leave `prefill_router_block_body` (`:372-486`),
    `prefill_router_block` (`:488-513`) and `prefill_router_block_sigmoid`
    (`:515-541`) byte-for-byte alone** — they are the test's reference and the
    `=block` A/B arm. `prefill_affine_value` (`:43-57`), `kPrefillGroupSize`
    (`:9`), `kPrefillRouterMaxExperts` (`:11`) and `kPrefillRouterMaxTopK`
    (`:12`) are reused unchanged.
  - Modify: `sources/Shrike/Kernels/Prefill/MoE/PrefillRouter.swift` — a `Kind`,
    the pipeline choice, the new dispatch geometry, `description`. The
    `encodeBlock(...)` signature and every buffer it binds stay as they are, so
    neither runner call site changes.
  - Modify: `sources/Shrike/Kernels/Prefill/MoE/PrefillRouterBenchmark.swift`
    (`kind` parameter, carried on `Result`) and `sources/ShrikeBench/RouterBlockBench.swift`
    (print `block`, `tiled` and the ceiling from one run, plus a `scores_only`
    arm for the top-k split).
  - Modify: `sources/Shrike/Runtime/Inference/RealForwardRunner.swift` — the env
    knob beside `prefillRoutedGEMMGrouped` / `prefillRouteOverlap` (`:459-462`),
    its stored property beside `:378-384`, `prefillRouterDescription` beside
    `prefillGapLeversDescription` (`:244-253`), and the two `PrefillRouter`
    constructions (`:709-718`, `:772-774`). Both `encodeBlock` call sites
    (`:4572-4591`, `:5037-5056`) are untouched.
  - Modify: `sources/ShrikeServer/Core/ServerInference.swift:820-822` (append
    `prefill_router=` to the residency line — **do not touch the leading
    `prefill_projection_path=` token**, which `tools/mini-deploy.sh:79` greps)
    and `:853-858`. `ServerInference.load` sits in the lint baseline at 181
    lines; if the added lines move it, regenerate the baseline in the same
    commit or the strict gate fails on a stale entry.
  - Test: `tests/Shrike/Core/Kernels/Prefill/PrefillRouterTests.swift` (346
    lines; `:33` and `:100` the existing block-vs-scalar cases, `:150`
    `makeBuffers`, `:245` `makeStableWeights`, `:299` `packWeights` — which
    needs a 4-bit sibling using `Quantization.quantizeInt4Affine`,
    `Quantization.swift:50`).
  - Docs: `docs/v12-prefill-matrix-kernels.md` — a landed
    `### Step 12 — the router block on an operand-reusing kernel` after
    `### Step 11` (`:725`), the ledger's "**After P14**" block, and Step 11's
    closing "the router is the next task (Task 14)" updated;
    `docs/v12-implementation-plan.md` — Task 14 `[x]`.
  - Unchanged deliberately: the decode router (`moe.metal` `router_gemv_body`
    `:291`, `router_topk_select_body` `:359`) and `RouterTopKTests.swift`; the
    routed tiles, the shared expert, the scratch layout (the new kernel needs
    no new buffer); `effectiveScaleBuffers`, `onesPerExpertScale`,
    `routerLogitBias`.

  **Interfaces:**
  - Consumes: `prefill_affine_value(packed:element:bits:)`,
    `FC_PREFILL_ROUTER_BITS` (index 79, set from `model.routerWeightBits`,
    `Model.swift:58`), `threadgroup_barrier(mem_flags::mem_threadgroup)`,
    `MTLComputeCommandEncoder.dispatchThreadgroups`.
  - Produces:

    ```swift
    extension PrefillRouter {
        /// `block` is the P0-era kernel, kept as the reference and the A/B arm;
        /// `tiled` is P14's TOK-token × all-experts kernel and the default.
        enum Kind: String, Sendable { case block, tiled }
        /// SHRIKE_PREFILL_ROUTER=block|tiled — anything else takes `tiled`.
        static func environmentKind() -> Kind
        init(context: MetalContext, weightBits: Int, sigmoidRouterScores: Bool,
             routedScalingFactor: Float, kind: Kind, tokenBlock: Int)
        /// "tiled tokens=16 bits=8" / "block bits=8"
        var description: String { get }
        static let defaultTokenBlock = 16   // SHRIKE_PREFILL_ROUTER_TOKENS overrides, 1...32
    }
    ```

    Metal: the two tiled kernels take the identical buffer bindings 0…14 as
    their `_block` counterparts, so only the pipeline and the threadgroup count
    change in the encoder. `FC_PREFILL_ROUTER_TOKENS` (123) carries `TOK`; a
    second internal constant selects the V1/V2/V3 rung so Step 3's ladder is one
    binary. The knob defaults to the *new* behaviour and keeps the old kernel
    reachable on that binary (`SHRIKE_MPP_WEIGHT_LOADS`'s precedent), so every
    ledger row is a one-binary A/B.

  Steps (TDD; the gate is Step 3's bit-equality result, which decides the rung
  and therefore the bar):

  - [ ] Step 1: the failing test, in `PrefillRouterTests.swift` —
        `tiledRouterIsBitIdenticalToTheBlockRouter`. Run both `PrefillRouter`
        kinds over one set of buffers; compare `outIndices` element-for-element
        and `outWeights` **by bit pattern**, borrowing
        `MPPPrefillInt4QMMTests.expectBitIdentical` (`:316-323`: assert finite,
        report the first mismatching index) and its `makeInputs(irregular:)`
        rationale (`:28-31`) — full-mantissa pseudo-random `x`, scales and
        biases, because the suite's current fixtures are integers over 64 whose
        partial sums are exact in fp32 and would compare equal under *any*
        reduction order. Cases: T ∈ {1, 7, 17, 4096} (1 and 7 under-fill a
        block, 17 leaves a one-token tail at TOK 16, 4096 is production),
        `weightBits` ∈ {4, 8}, both score paths (the sigmoid arm at
        `routedScalingFactor` 2.446, kimi's value), and one at
        `hiddenStrideElements = d + 13` that re-asserts
        `assertPaddingUnchanged` (`:329`).
  - [ ] Step 2: `swift test --no-parallel --filter PrefillRouterTests` — expect
        FAIL: `PrefillRouter.Kind` undefined.
  - [ ] Step 3: implement the tiled kernel and walk the ladder. Start at **V1**
        (staged product + hoisted `sum_x`); if any case is not bit-identical,
        drop to **V2**, then to **V3**, and record in the verdict which rung
        survived and what that says about Metal's contraction of
        `sum_x += xv` — that is a reusable finding, not a footnote. Do not
        "fix" a mismatch by loosening the test to a tolerance: the whole
        argument for golden identity is bit equality against the current
        kernel.
  - [ ] Step 4 (the bench spike, both boxes, **the mini decides**):
        `swift run -c release ShrikeBench router_block 20` prints, from one
        process, the `router_chunk4096` MPS ceiling, `block`, `tiled` and
        `scores_only` (the tiled kernel with the top-k compiled out — it prices
        the top-k, only modelled today). Sweep
        `SHRIKE_PREFILL_ROUTER_TOKENS` ∈ {4, 8, 16, 32} on both boxes, recording
        threadgroup memory and TFLOPS per arm; the boxes have disagreed about
        exactly this kind of tile before (P10), so the landed default is
        whatever the **mini** picks. **Bar ≤ 15.0 ms on the mini; stretch
        ≤ 10.0; abandon below 3× (> 27.8 ms) → land the tests and the kernel
        behind `SHRIKE_PREFILL_ROUTER=tiled`, keep `block` as the default,
        write the null into the verdict and the design doc, and stop — do not
        open the MPP arm without the owner's call, because it moves golden on
        every profile.**
  - [ ] Step 5: the five gates — release build with zero warnings;
        `swiftlint lint --strict --baseline .swiftlint-baseline.json`
        (regenerate if `ServerInference.load`'s length moved); markdown link
        check; `swift test --no-parallel`; the same under
        `env TSAN_OPTIONS=suppressions=tsan-suppressions.txt swift test
        --no-parallel --sanitize=thread`.
  - [ ] Step 6: `tools/golden-baseline.sh --check` on this box with the server
        stopped — **expected IDENTICAL, short and long; a difference is a kernel
        bug, not a numerics change: debug it, never recapture**
        (`docs/v12-implementation-plan.md:38-42`). Commit
        `prefill: the router block on an operand-reusing kernel (v12 P14)`.
  - [ ] Step 7: ledger on both boxes. `tools/mini-deploy.sh --restart`, then
        `tools/prefill-measure.sh` with a fresh server per prompt and one send
        per prompt per server lifetime (a repeat hits the multi-prefix prompt
        cache), reading roles with `tools/prefill-ledger.py`. **The ledger's
        column names are not the script's prompt labels**: the 12,285-token
        prompt is written as `prompt-6k.json` and the 3,756-token one as
        `prompt-2k.json` (`tools/prefill-prompts.py:10-11`), so the ledger's
        "12k" column is the script's `6k` label. Mini 3.7k + 12k is the verdict,
        M4 Pro 3.7k + 12k + 25k the check; same-binary A/B with
        `SHRIKE_PREFILL_ROUTER=block`. Then the mini golden check — IDENTICAL.
        Nothing here changes allocation, so a `memory_pressure -Q` move would
        be a finding.
  - [ ] Step 8: docs and review. Design doc gets the landed Step 12 section and
        the "After P14" ledger block; the plan gets Task 14 `[x]`. Fresh
        reviewer; fixes folded into the commit (rebase and amend, never a
        fixup commit).

  **Verdict template.** The bench first: mini and M4 Pro, `block` → `tiled` ms
  per launch, TFLOPS, share of the same-run ceiling, the winning `TOK` with its
  threadgroup memory, and the `scores_only` split that prices the top-k. Then
  which rung (V1/V2/V3) proved bit-identical and what that says about
  fast-math contraction of `sum_x += xv`. Then the ledger, mini 3.7k and 12k
  before → after: `prefill_gdn_router` and `prefill_attn_router` in
  ms/prompt-token, GPU busy, gaps (span − busy), wall in seconds and
  ms/prompt-token; the M4 Pro's rows as the check. Then the golden outcome per
  box and profile — **identical, else the task failed** — with the digests
  recorded unchanged, and the other role rows unchanged within noise.

  **Risks.**
  - *The `sum_x` hoist is not bit-identical.* The likeliest single surprise, and
    it is a compiler question no amount of reading settles — Metal's fast math
    may already be contracting `sum_x += float(x[k]) * float(e[k])` into an
    `fma` in today's kernel (MEMORY.md's "Metal half round-trip elision" family,
    found the hard way in T2 `d89d172`). Step 3's ladder is the mitigation and
    Step 1's test is the detector; the cost of landing on V3 is a lower bar
    (≤ 25 ms, 3.3×, wall 73.4 s) not a failed task.
  - *The `s == top_score[i]` tie rule.* The selection is identical **only
    because** the scan still walks experts in ascending order on a single
    thread per token. Any attempt to widen the top-k across a simdgroup breaks
    it silently — the logits stay identical and the *chosen expert* changes on a
    tie. The review must check this specifically; the near-tie case
    (`blockRouterNearTieMatchesScalarPath`, `PrefillRouterTests.swift:100`,
    whose fixture puts experts 7 and 8 within 1e-4) is the existing guard and
    the new test must cover the tiled kernel with it.
  - *Threadgroup memory and occupancy on the M1.* 20.1 KB at TOK 16 is inside
    the 32 KB per-threadgroup limit but may leave one threadgroup resident per
    core on an 8-core M1, and ≈ 70 registers may cost occupancy again — the
    exact failure mode where a "more reuse" kernel gets slower. This is
    modelled, not measured; Step 4's sweep is the answer, and TOK 8 (10.1 KB,
    ≈ 38 regs, 512× amplification) is the fallback that still models at 4.5×.
  - *`T` not a multiple of `TOK`.* The last block is partial in every real
    request (the 3.7k chunk is 3,756 rows). Every staging write, every score
    write, the accumulator loop and the top-k mask on `row0 + t < T`, and
    **nothing may be written to `out_indices` / `out_weights` past `T`** — the
    runner reads `t * topK` entries and a stray write lands in another token's
    route. T ∈ {1, 7, 17} in Step 1 is the guard.
  - *`hidden_stride` and 4-bit alignment.* Staging must read
    `hidden + (row0 + t) * hidden_stride`, not a packed row (the suite already
    asserts inter-row padding is untouched, `:329`; the new test reuses it at
    `d + 13`). Groups are byte-aligned by construction at 4 bits, but a
    vectorised uint4 register load of the row also needs
    `weightsOffset % 16 == 0` and `router.offset` comes from the model view —
    mirror P9: an alignment check with a scalar byte-load fallback, both
    bit-identical, tested at a deliberately unaligned offset
    (`MPPPrefillInt4QMMTests:340` does exactly this at `weightOffset 13`).
  - *The sigmoid path is not the measured one.* Everything above is measured on
    the softmax router; the kimi arm is covered by construction and the
    bit-equality test and by nothing else. Say so in the verdict rather than
    implying it was benched.
  - *AGX trap, and two model processes.* The known driver crash is a
    `.concurrent` encoder plus a later indirect dispatch on one command buffer
    (MEMORY.md, T2); this task adds neither — one ordinary compute encoder, one
    `dispatchThreadgroups` — but the check belongs in the review. Every bench
    and ledger arm is a GPU run: `pgrep -fl 'ShrikeServer|ShrikeMac|ShrikeDecodeService|ShrikeCLI|ShrikePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'`
    first, every time; never terminate a process this session did not start.

### Task 15: P15 — the routed GEMM's padding tax and the prefill expert-cache sweep order: probe both, then take the larger

- [x] **P15: two levers, neither of them measured** — the audit's L2 (the grouped
  routed GEMM rounds every expert block up to a whole 64-row tile, modelled
  0.15–0.40 ms/prompt-token) and L5 (every chunk sweeps its ~237 experts through
  a 128-slot cache in the same direction, so `expert_hits_prefill` is **0**,
  modelled 0.05–0.11 direct) are the two largest unclaimed items left on the
  mini after P14, and **both bands are arithmetic, not measurement**. L2's band
  is worse than unmeasured: two independent estimates of the same quantity
  disagree by 2× (below), and nothing in the tree prints the padded-row count.
  So Steps 1 and 2 are two cheap probes — one instrumented build that is
  reverted, one comparator flip behind an env knob that lands either way — and
  the kernel work in Step 3 is chosen by what they measure. **The mini decides.**
  The M4 Pro is the iteration check: its prefill hit rate is also 0 %
  (`docs/v12-prefill-matrix-kernels.md:833-836`), so L5 should move there too,
  but its SSD, its slot pressure and its tile/fetch balance differ and it
  confirms nothing about the mini's I/O.

  **LANDED dd78c27 (2026-09-03; amended in review fix round 1), Steps 1–4 and
  7–9 — the tail tile (Steps 5–6) is Task 15b below.** Step 1 measured the padding on the mini at 12k:
  `P64` = 4,946,880 rows against 3,931,200 real per prompt (**+25.8 %**; the
  uniform-remainder model's +26 % was right, the subtraction estimate's
  +11–16 % wrong), `P32` = 4,410,976, `P16` = 4,156,704, over 30,506 blocks in
  6,559 waves with 2,102 wave splits (per layer-chunk: 32,760 real, 41,224 /
  36,758 / 34,639 padded, 254 blocks, 55 waves, 17.5 splits); 3.7k +29.6 %.
  Step 1b: `per_tile_ms` at 128 / 97 / 65 rows per expert = 4.461 / 4.419 /
  4.331 on the mini, 1.086 / 1.074 / 1.066 on the M4 Pro — a padded row
  **does** cost a real row (within 3 %). Step 2 measured on the mini
  `expert_hits_prefill` 0 → 9,549, `expert_misses_prefill` 28,404 → 18,855,
  12k wall 70.33 s against 71.34 s on the same binary with
  `SHRIKE_PREFILL_SWEEP=fixed` (−1.41 %); `routed→routed` 0.096 → 0.072
  ms/token, `shared→routed` 0.073 → 0.039, gaps 0.273 → 0.217; the 3.7k
  control moved 0.0 % (21.49 → 21.49 s, hits 0 in both arms), decode counters
  identical. The rule took **both**: L5 cleared its bar (−1.41 % ≥ 1.0 % on the
  same-binary A/B, every counter and gap bar met, control unmoved; the ≤ 70.0 s
  absolute wall bar is met at the default, 69.97 s — the A/B arm read 70.33 s
  against a 71.34 s same-binary control) and is the default (Step 4;
  `=fixed` the A/B); L2's modelled Δ from the measured counts is 0.152–0.158
  ms/token at α = 0.55 and 0.102–0.105 at 0.70 — over the 0.10 threshold at
  any α ≤ 0.70 and over L5's measured 0.081 — so the tail tile proceeds as
  Task 15b once α is measured (the descriptor at TILE_M = 32 and 16 compiles
  and links offline with `xcrun metal -std=metal4.0`, which retires the
  draft's first risk to a runtime check). M4 Pro check: 12k 26.61 → 22.71 s
  (−14.6 %, hits 0 → 9,558), 25k 59.00 → 49.10 s (−16.8 %, hits 0 → 28,909 of
  65,776) — the fetch was that box's whole gap. At the default the mini reads
  69.97 s at 12k = **5.70 ms/prompt-token** (3.7k 21.58 s). Numerics:
  completions byte-identical between arms at both sizes; golden identical on
  both boxes and both profiles (digests unchanged: M4 Pro long
  `e04d4e8ee7f1590d`, M1 long `899a25e60a365e60`). Five gates green on the
  P15 code commit (counts on the SDD ledger). Review fix round 1: the parity
  divisor is the configured chunk width (`config.chunkTokens`), not the 4,096
  ceiling — the lever was inert on any model with a smaller prefill chunk. Found on the way, not a lever
  of this chapter: the cache settle re-prefills the whole prompt after a
  degenerate (`finish=length`) turn — 66.8 s of mini GPU after the 12k
  response — the v10 plan's open boundary-snapshot item; recorded in the
  design doc's follow-ons.

  **The decision rule, in two lines.** Step 2's knob lands on its own measured
  merit — it is a comparator flip, so if the mini's 12k wall improves by ≥ 1.0 %
  (≥ 0.71 s) with golden identical and the 3.7k control unmoved, `alternate`
  becomes the default and `=fixed` the A/B; below that it lands defaulted to
  `fixed` and the verdict says so (P8's precedent, a measured null is a result).
  Step 3 — the kernel work — proceeds **only if** L2's saving modelled from Step
  1's counts clears **0.10 ms/prompt-token** (1.23 s of the 70.76 s wall, 1.7 %)
  *and* exceeds Step 2's measured delta; if L5 measured larger, this task's body
  is L5 and L2's probe numbers are recorded as the follow-on's size.

  **Where the 5.76 goes.** After P14 (HEAD `e3dea75`) the mini's 12k prompt
  (12,285 tokens, `tools/prefill-prompts.py:9-11`; 3 chunks of 4,096 → **120
  layer-chunks**, all 40 layers routed) is 70.76 s wall = **5.76 ms/prompt
  token**, GPU busy 5.337, gaps 0.273; roles `prefill_gdn_router` 1.775 ·
  `prefill_routed_tile` 1.737 · `prefill_attn_router` 1.565 ·
  `prefill_shared_expert` 0.169 (`docs/v12-prefill-matrix-kernels.md:424-433`).
  At 3.7k: wall 21.52 s, busy 4.665, gaps 0.493, routed 1.79. The gaps split
  measured at P12 and unchanged at P14 (`:742-744`): `routed→routed` ≈ 0.10
  (the per-tile host work over ≈ 3,485 tile boundaries — audit L10) and
  `shared→routed` ≈ 0.07 (the tile metadata build plus the first tile of each
  layer-chunk fetched with nothing in flight). Derived from those rows:

  | quantity | value | source |
  | --- | ---: | --- |
  | routed role, per layer-chunk | 177.8 ms | 1.737 × 12,285 / 120 |
  | routed role, per tile | 5.918 ms | 1.737 × 12,285 / 3,605 tiles |
  | pairs per layer-chunk / per tile | 32,768 / 1,091 | 4,096 × top-8; 30.04 tiles |
  | useful GFLOP per tile | 6.864 | 1,091 × 6 × 2048 × 512 |
  | achieved on the useful rows | **1.160 TFLOPS** | 6.864 / 5.918 ms |
  | the same kernel on a padding-free bench tile | **1.438 TFLOPS** | 6.442 GFLOP / 4.48 ms (P10, 78 % of the mini's same-run ceiling, `:453`) |

  **L2 — the padding tax, and why two estimates of it disagree.**
  `planExpertWaves` rounds each expert block up to a whole 64-row tile —
  `cursor += (rows + tile - 1) / tile * tile`
  (`sources/Shrike/Kernels/Prefill/MoE/PrefillGroupedRoutedMoE.swift:129`) — and
  the wave close at `:121` truncates an expert at the 1,024-row staging boundary
  (`sources/Shrike/Runtime/Prefill/PrefillChunkScratch.swift:130-139`), adding
  another partial tile. The padded rows are **zero-filled** by the gather
  (`sources/Shrike/Metal/Prefill/prefill.metal:924`), computed in full by
  `matmul2d` — `kMPPAffineTileM` is 64 and the store guard is the only mask
  (`globalM < rowEnd`, `sources/Shrike/Metal/TensorCore/tensorops.metal:244`,
  with `rowEnd = b.staging_row + b.rows` at `:326`) — and also carried by the
  activation, which runs over `wave.paddedRows * f`
  (`PrefillGroupedRoutedMoE.swift:919`). So a padded row costs a real row in
  three of the six dispatches per wave.

  Two estimates of how much:

  - **By the uniform-remainder model** (the audit's, `missed-levers-audit.md:71-77`):
    ~237 expert blocks per layer-chunk plus ~30 wave-split blocks ≈ 267 blocks
    over 32,768 pairs, ~123 rows each; a uniform remainder pads +31.5 rows per
    block = 8,410 rows = **+26 %**.
  - **By subtraction from the ledger**: the bench-equivalent GEMM time for the
    tile's 1,091 real rows is 6.864 / 1.438 = 4.774 ms; the audit's bandwidth
    model for the gather/activation/scatter passes is 12–18 ms per layer-chunk =
    0.40–0.60 ms per tile; the residual is 0.544–0.744 ms per tile, which at the
    bench rate buys 124–170 extra rows = **+11 to +16 %**.

  The second is 1.96–2.68 s of the 12k wall = **0.16–0.22 ms/prompt-token**; the
  first is nearly double that. They cannot both be right, and no line in the tree
  prints the padded-row count. **Step 1 is one print.**

  **L5 — why the prefill cache never hits.**
  `PrefillMoEGrouping.groupTokenExpertPairs` sorts every chunk's pairs by
  `expertSortKeys` = `model.routedExpertPhysicalOffsets(layer:)`
  (`PrefillMoEGrouping.swift:140-149`; `RealForwardRunner.swift:4844`,
  `ModelExpertIO.swift:94-96`), i.e. **ascending physical offset, the same
  direction in every chunk**. Each layer has its own 128-slot streamer
  (`ModelExpertIO.swift:100-103`; `--ram-budget 8G` snaps to 128,
  `RuntimeConfiguration.swift:142`, `:204-214`), of which the tile scheduler
  holds 16 for in-flight tiles (`maxInFlightTiles × tileExperts = 2 × 1 × 8`,
  `PrefillRoutedTileScheduler.swift:55-58`, `:81-82`), leaving ~112 evictable
  residents against the 236.6 experts a layer-chunk touches (28,387 / 120,
  measured). The eviction policy is aging-LFU with an LRU tiebreak
  (`PreadExpertStreamer.swift:293`, `:1137-1150`); under a uniform sweep every
  expert's use count is equal, so the tiebreak decides and it is exactly LRU —
  **the sequential-scan pathology**: the resident set is always the ~112 experts
  the sweep will reach last, and the one it needs next was evicted a pass ago.
  Measured: `runner.expert_hits_prefill=0 expert_misses_prefill=28387`
  (`ServerInference.swift:1997-1999`). At an expert stride of 1,769,472 B
  (3 × 512 × 2048 / 2 int4 bytes + 3 × 2 × 16,384 bf16 scales/biases) that is
  **50.2 GB of `F_NOCACHE` reads per prompt**, ≈ 17.9 s at the doc's measured
  2.8 GB/s, behind a 21.3 s routed GPU span at pipeline depth 2 — mostly hidden,
  which is why it shows today only as the two gap rows and as the first tile of
  each layer-chunk, planned and fetched with nothing pending
  (`RealForwardRunner.swift:5153-5169`; `heldSlots` is empty, so `plan` is nil
  and `fetchBindingForTile` plans and fetches inline).

  Alternating the sort direction on odd chunks turns the pathology into its
  best case: LRU with an alternating sweep hits on the previous pass's tail,
  ≈ 112 of 237. At 12k there are three chunks — chunk 0 cold on a fresh server,
  chunks 1 and 2 hitting ≈ 112 each — so the modelled counters are hits ≈ 8,960
  and misses ≈ 19,400 (−31 %), bytes 50.2 → 34.4 GB, ≈ 5.6 s of reads removed of
  which only the unhidden part shows in the wall. The realistic ceiling is the
  two gap rows it feeds, 0.17 ms/token if they vanished entirely; the audit's
  0.05–0.11 is the direct band. **The measurement is the result.**

  **What does not change, from the code.** `routePartials` is written per
  `(token, rank)` slot by the pair's own expert block
  (`prefill.metal:946-947`) and the reduce folds the 8 ranks per token in rank
  order (`prefill_moe_reduce_token_major`), so **the order tiles are issued in
  cannot change any value**. The comparator flip touches only the expert key;
  `$0.token`/`$0.rank` stay ascending within a group
  (`PrefillMoEGrouping.swift:146-148`), so each expert block presents the same
  rows in the same order to the same GEMM. Golden **IDENTICAL on both boxes and
  both profiles is a hard bar for Step 2**, not an expectation.

  **Step 1 — the L2 probe (instrumented build, reverted).** In
  `encodeGroupedExpertGEMMs` (`PrefillGroupedRoutedMoE.swift:847-941`, the wave
  loop at `:873-879`) accumulate four `inout UInt64` counters threaded out
  through `encodeRoutedTileExperts` → `encodeRoutedMoEPrefill` alongside the
  existing `prefillActiveExperts` (`RealForwardRunner.swift:5030`, `:5100`), and
  print them in the existing per-chunk `SHRIKE_PHASES` block (`:2334-2343`):
  `Σ pairCount` (the real rows), **`Σ paddedRows` today**, **`Σ ceil(rows/32)·32`**
  and **`Σ ceil(rows/16)·16`** recomputed over the same blocks, plus the wave
  count, the block count, and the number of blocks whose `pairStart` continues a
  previous block (the wave splits). The three padded sums price the three tail
  sizes directly and need no histogram. Build isolated —
  `swift build -c release --scratch-path /Volumes/BuildSSD/SwiftPM/Shrike-probe`
  (the path exists; the P5/P8 precedent) — deploy with `tools/mini-deploy.sh`,
  launch manually with `SHRIKE_PHASES=1` added to the production env, one send
  per prompt on a fresh server at 3.7k and 12k, then **revert the probe** and
  redeploy the clean build before anything else runs.

  **Step 1b — the bench control (lands).** `PrefillRoutedGEMMBenchmark.run`
  already takes `experts`, `rowsPerExpert` and `stagingRows`
  (`PrefillRoutedGEMMBenchmark.swift:25-31`); `runRoutedGEMM` calls it only at
  the defaults (`sources/ShrikeBench/RoutedGEMMBench.swift:6-22`). Add a padding
  sweep at `experts: 8, stagingRows: 1024` over `rowsPerExpert ∈ [128, 97, 65]`
  — all three plan the **same 16 row tiles and the same 1,024 padded rows**,
  with 1,024 / 776 / 520 real rows. If `per_tile_ms` is flat across the three,
  a padded row costs exactly a real row and the tax is exactly the padded
  fraction; if it falls, the model is wrong and Step 3 does not proceed. ~12
  lines, no production path.

  **Step 2 — the L5 knob (lands either way).** `SHRIKE_PREFILL_SWEEP=alternate`
  (default `fixed` in this step; the repo's knobs read
  `environment[...] != "<opt-out>"`, so the default flips to `alternate` in
  Step 4 only if it wins — `RealForwardRunner.swift:465-468` is the pattern).
  `buildPrefillRoutes` does not currently know the chunk index
  (`:4806-4846`); `executePrefillChunk` has `startPosition`
  (`:2101-2107`) and `PrefillChunkPlanner.spans` lays chunks out contiguously
  from it, so the parity is `(startPosition / config.maxChunkTokens) % 2` —
  pass it into `buildPrefillRoutes` and on to `groupTokenExpertPairs` as a
  `descending: Bool`, flipping only the `expertSortKeys` comparison at
  `PrefillMoEGrouping.swift:144`. Log it on the residency line beside the other
  levers (`prefillGapLeversDescription`, `RealForwardRunner.swift:243-254`;
  emitted at `ServerInference.swift:821-824`).

  **Step 3 — the tail tile (only if L2 wins).** Three options, priced:

  1. **A 32-row tail tile (the proposal).** Pack each wave body-first: full
     64-row tiles in `[0, bodyRows)`, every block's remainder in a 32-aligned
     tail region `[bodyRows, paddedRows)`, and issue **two** grouped dispatches
     per GEMM — the existing 64-row kernel over `bodyRows / 64` row tiles and a
     32-row instantiation over `tailRows / 32`, each skipped when its region is
     empty. The tail is taken **only when the block's remainder is ≤ 32**: two
     32-row tiles cost `2α` against one 64-row tile, so a remainder in
     `[33, 63]` must stay on the 64-row path or the change is a regression at
     any `α > 0.5`.
  2. **Two experts' tails packed into one 64-row tile.** No arithmetic saving
     over option 1 — a threadgroup dequants one weight tile and all 64 rows
     multiply it, so two experts in one tile is two dequants and two half-height
     matmuls, with `MPPGroupedBlockMSL` (`tensorops.metal:286-293`) growing a
     second slot and the body a second accumulator. Its one advantage is that
     the grid stays 64-row-granular. The fallback if option 1's descriptor does
     not compile, not the first arm.
  3. **Larger chunks (audit L9).** Out of scope — memory, and the design says so.

  **How much option 1 is worth, as a formula over Step 1 and a measured `α`.**
  Let `P64` = `Σ paddedRows` and `P32` = `Σ ceil(rows/32)·32` per layer-chunk
  (Step 1), `G` = the GEMM ms per layer-chunk (177.8 minus the 12–18 ms of
  gather/activation/scatter = 160–166), and `α` = the measured cost of a 32-row
  tile as a fraction of a 64-row tile. Conversions are `(P64 − P32)/32`, tiles
  today `P64/64`, so

      Δ ms/prompt-token = G × (1 − α) × 2 × (P64 − P32) / P64 × 120 / 12,285

  `α` is bounded below by 0.5 (nothing but the matmul scales with M) and above
  by 1.0 (a measured null): the dequant of a `TILE_N × TILE_K` weight tile is
  per threadgroup and does **not** shrink with M, and P8's mini split of the
  tile's remainder above a plain GEMM was 7 % unpack + 20 % weight loads + 31 %
  staged structure (`docs/v12-prefill-matrix-kernels.md:453`) — the first two
  are M-independent. **Modelled `α` ≈ 0.55, but it is measured in Step 5's bench
  before a line of production code is wired.** At `α = 0.55` the two anchors for
  `(P64 − P32)/P64` give **0.083 ms/token** (the subtraction estimate, 0.0595)
  and **0.149** (the uniform-remainder estimate, 0.104) — straddling the 0.10
  threshold, which is the whole reason Step 1 comes first. At `α = 0.70` neither
  anchor clears it. The 16-row rung's ceiling is the entire padding residual,
  0.16–0.22 ms/token, reached only at `α₁₆ ≈ 0.25`; Step 1's `P16` prices it and
  the bench decides, but the fixed dequant makes it unlikely.

  **Numerics for option 1.** The K reduction per output element is unchanged: it
  is the same `TILE_K`-wide loop over the same tiles with the same
  `accumulator[e] += groupProduct[e]` fold (`tensorops.metal:210-236`), and
  `TILE_M` is only the descriptor's first argument (`:166-168`) — it changes
  which rows share a threadgroup, not what is summed or in what order for a
  given (row, column). **Bit-identity is therefore expected but not provable
  from source** (MPP's `run` is opaque), exactly as P13's draft argued for
  `TILE_M 128`. Prove it as P9 proved lever A: `runPair`-style bit equality on
  `irregular` inputs, here through
  `PrefillGroupedRoutedMoETests+Execution.swift:603-634`
  (`groupedPartialsAcrossWaves`) with the tail path on and off. If it fails,
  fall back to 2e-2 against the fp32 reference and carry a golden recapture.

  **Bars.** Mini-first, 12k unless stated, as ms per prompt token. They are
  formulas; the numbers are those formulas at the anchors above.

  - **L5 (Step 2).** `runner.expert_hits_prefill` **0 → ≥ 6,000** at 12k
    (modelled 8,960; two of three chunks hitting ≈ 112 of 237 per layer);
    `expert_misses_prefill` 28,387 → ≤ 22,500. `routed→routed` ≤ 0.09 and
    `shared→routed` ≤ 0.065 ms/token; wall **70.76 → ≤ 70.0 s** (−1.0 %, the
    proceed threshold) with the stretch at −2 % (69.3 s). **Control: at 3.7k
    (3,756 tokens, one chunk) the arm must read within ±1 % of `fixed`** —
    there is nothing to alternate, and a move there means the knob is doing
    something it does not claim. Golden **identical, hard bar**.
  - **L2 (Step 3, if it proceeds).** `prefill_routed_tile` = 1.737 − Δ with Δ
    from the formula; at the mid anchor **1.737 → ≤ 1.62**, busy 5.337 → ≤ 5.22,
    wall **70.76 → ≤ 69.3 s**; at 3.7k routed 1.79 → ≤ 1.68, wall 21.52 →
    ≤ 21.1 s. Bench bar in `routed_gemm` on the mini: at `rowsPerExpert: 65`
    (a 1-row remainder, so every block takes the tail) the arm is ≤ **0.85 ×**
    the 64-row path — 2 tiles → 1 + α, i.e. −22 % at α = 0.55; and at
    `rowsPerExpert: 128` (no remainder) and `97` (a 33-row remainder, which the
    ≤ 32 rule leaves on the 64-row path) it is **within 1 %** — the tail path
    must not touch a wave it cannot help.
  - **Both.** Five gates green. M4 Pro 3.7k + 12k + 25k as the check; it decides
    nothing.

  **Files:**
  - Step 1 (probe, **reverted before Step 2 lands**):
    `sources/Shrike/Kernels/Prefill/MoE/PrefillGroupedRoutedMoE.swift:873-879`
    (four counters in the wave loop),
    `sources/Shrike/Runtime/Inference/RealForwardRunner.swift:5019-5030`,
    `:5300-5350` (thread them through as `inout`), `:2334-2343` (the
    `SHRIKE_PHASES` print).
  - Step 1b (lands): `sources/ShrikeBench/RoutedGEMMBench.swift:6-22` — a
    `rowsPerExpert` sweep; that file has no baseline entry, so keep
    `runRoutedGEMM` under 120 lines.
  - Step 2 (lands): `PrefillMoEGrouping.swift:91-149` (a
    `descending: Bool = false` parameter, one comparison flipped at `:144`),
    `RealForwardRunner.swift:465-468` (the knob), `:4806-4846`
    (`buildPrefillRoutes` takes the chunk parity), `:5090` (the call site),
    `:2101-2110` + the chunk loop at `:1949-1968` (the parity from
    `startPosition`), `:243-254` (`sweep=` on the levers line).
  - Step 3 (only if L2 wins):
    `sources/Shrike/Metal/TensorCore/tensorops.metal:145-248` (`TILE_M` becomes a
    template parameter of `mpp_prefill_affine_body`, replacing `kMPPAffineTileM`
    at `:166`; `MPP_AFFINE_KERNEL` at `:250-269` passes 64, the plain path
    unchanged), `:296-334` (`MPP_GROUPED_KERNEL` gains `TILE_M`; `rowOrigin` at
    `:322-326` takes it), `:336-341` (six `…_m32` instantiations);
    `prefill.metal:891` (`kPrefillRoutedGroupedRowTile` becomes a field of
    `PrefillRoutedGroupedParamsMSL` at `:900-905`, read at `:920` and `:944`);
    `PrefillGroupedRoutedMoE.swift:88-148` (body-then-tail packing;
    `PrefillRoutedExpertWave` gains `bodyPaddedRows`; two tables), `:847-941`
    (two dispatches per GEMM, each skipped when empty);
    `MPPPrefillInt4QMM.swift:52` (`groupedMaxRowTiles` 32 → 64 — a 32-granular
    table over 2,048 staging rows is 64 entries), `:222-317` (`encodeGrouped`
    takes the row tile, picks the pipeline and the grid height).
  - Tests: `tests/Shrike/Core/Kernels/Prefill/PrefillMoEGroupingTests.swift:84-116`
    (the descending twin of
    `groupingCanOrderTilesByExpertSortKeysWhileKeepingPairRangesContiguous`);
    `.../PrefillGroupedRoutedMoETests+Execution.swift:510-560`
    (`groupedWavePlannerSplitsAndPadsOnRowTiles` — the 40/32/5/3-pair fixture is
    already the tail case: 256 padded rows today, 160 with a 32-row tail),
    `:562-601`, `:603-634`, `:667-716` (the one-pair expert).
  - Lint: `executePrefillChunk` (`:2101`) and `encodeRoutedMoEPrefill` (`:5019`)
    both carry `function_body_length` baseline entries keyed by line **and**
    span; every step here moves one or both, so regenerate with
    `swiftlint lint --write-baseline .swiftlint-baseline.json` and commit it with
    the change, or the strict gate fails on a stale entry.
  - Unchanged deliberately: `MPPPrefillInt4QMM.tileK` and the six `TileVariant`
    cases; the plain kernel's `kMPPAffineTileM = 64`; the eviction policy and
    slot count; `PrefillRoutedTileSchedulerConfig`; the decode paths.

  **Interfaces:** Step 2 produces
  `PrefillMoEGrouping.groupTokenExpertPairs(..., descending: Bool = false)` and
  a `sweep=alternate|fixed` token on the gap-levers line. Step 3 produces
  `enum GroupedRowTile: Int { case m64 = 64, m32 = 32 }` on `MPPPrefillInt4QMM`
  with `SHRIKE_PREFILL_TAIL_TILE=off|32` (default `off` until Step 6 clears the
  bar), `encodeGrouped(..., rowTile: GroupedRowTile)`, and in Metal
  `template <int TILE_M, int TILE_N, int TILE_K, int BUFFERS>
  mpp_prefill_affine_body(...)` with `MPP_GROUPED_KERNEL(NAME, TILE_M, TILE_N,
  TILE_K, BUFFERS)`. `PrefillRoutedExpertWave` gains `bodyPaddedRows: Int`;
  `rowTileTable(for:)` becomes `rowTileTables(for:) -> (body: [UInt32], tail:
  [UInt32])`. Comments: repo rule. One earns its place — why the tail is a
  second dispatch rather than a second row height inside one grid (the pipeline
  is fixed per dispatch). Nothing else.

  Steps (TDD; the measurement precedes any kernel decision):

  - [x] Step 1: the L2 probe. Isolated build
        (`swift build -c release --scratch-path /Volumes/BuildSSD/SwiftPM/Shrike-probe`),
        `pgrep -fl 'ShrikeServer|ShrikeMac|ShrikeDecodeService|ShrikeCLI'` first,
        `tools/mini-deploy.sh` (copy only), stop the server, relaunch manually
        with `SHRIKE_PHASES=1` beside the production env, one send per prompt on
        a fresh server at 3.7k and 12k
        (`tools/prefill-measure.sh macmini 8081 <promptdir> <outdir> p15-probe-mini 2k 6k`
        — the labels `2k`/`6k` are the 3,756- and 12,285-token prompts,
        `tools/prefill-prompts.py:9-11`). **Revert the probe**, rebuild clean,
        redeploy.
  - [x] Step 2: the Step 1b bench sweep and the Step 2 knob, both landing. Five
        gates. `swift run -c release ShrikeBench routed_gemm 20` on both boxes.
        Then the mini A/B on one binary: `tools/mini-deploy.sh --restart` for the
        `fixed` arm, a manual relaunch with `SHRIKE_PREFILL_SWEEP=alternate` for
        the other (the `--restart` launch command is fixed, so the knob arm is
        launched by hand), 3.7k + 12k on a fresh server per arm, roles via
        `tools/prefill-ledger.py` with a **distinct log tag per box** (P10's
        lesson: `resp-<tag>-<label>.json` collides otherwise). Read
        `expert_hits_prefill` / `expert_misses_prefill`, both gap rows, busy and
        the wall.
  - [x] Step 3 (the gate): fill the ledger — `Σ pairCount`, `P64`, `P32`, `P16`,
        waves, blocks and splits per layer-chunk at both prompt sizes; the bench
        sweep's flatness; L5's measured deltas. Apply the decision rule, re-derive
        L2's bar from `P64`/`P32`, and **record the verdict in the design doc
        whichever way it goes**. Commit
        `prefill: alternate the prefill expert sweep and bench the padding tax (v12 P15)`.
  - [x] Step 4: if L5 cleared its bar, flip `SHRIKE_PREFILL_SWEEP`'s default to
        `alternate` with `=fixed` as the A/B, in the same commit as Step 3's
        verdict. If it did not, leave the default at `fixed` and say so.
  - [x] Step 5 — moved to Task 15b (its Steps 1–3): failing tests first —
        `tailTilePlannerPacksRemaindersIntoThirtyTwoRowTiles` on the
        40/32/5/3-pair ranges (256 → 160 padded rows, two tables);
        `thirtyTwoRowTailIsBitIdenticalToTheSixtyFourRowTile` over
        `MPPPrefillInt4QMM.TileVariant.allCases` through
        `groupedPartialsAcrossWaves`, `irregular` inputs, plus a one-pair expert
        and a ragged `d`/`f`; `tailTileIsSkippedWhenNoBlockHasARemainder`.
        `swift test --no-parallel --filter "PrefillGroupedRoutedMoE|PrefillMoEGrouping"`
        → FAIL. Then implement, and **before wiring production** measure `α` in
        `routed_gemm` on the mini; if `α` puts Δ under 0.10 ms/token, land the
        kernel behind `SHRIKE_PREFILL_TAIL_TILE=32` with the default `off` and
        stop (P8's precedent).
  - [x] Step 6 — moved to Task 15b (its Step 4): re-run `routed_gemm` on both boxes.
        **Accept rule: the mini clears the bench bar, or beats its Step 2 control
        by ≥ 5 % with no shape regressing more than 3 %.** Arms within 3 % are a
        tie, broken toward the 64-row path.
  - [x] Step 7: five gates —
        `swift build -c release 2>&1 | grep -E "warning:|error:"` (empty),
        `swiftlint lint --strict --baseline .swiftlint-baseline.json`,
        `python3 tools/check-md-links.py`, `swift test --no-parallel`, and
        `env TSAN_OPTIONS=suppressions=tsan-suppressions.txt swift test
        --no-parallel --sanitize=thread`. Then `tools/golden-baseline.sh --check`
        on both boxes: **short and long IDENTICAL** (M4 Pro long
        `e04d4e8ee7f1590d`, M1 long `899a25e60a365e60`, unchanged since P10) —
        a difference is a defect, not a recapture, for both arms.
  - [x] Step 8: ledger on both boxes. `tools/prefill-measure.sh <host> <port>
        <promptdir> <outdir> <tag> 2k 6k` against a **fresh server, one send per
        prompt per server lifetime**; **mini 3.7k + 12k is the verdict**, M4 Pro
        `2k 6k 12k` (3.7k + 12k + 25k tokens) the check.
        `tools/mini-deploy.sh --restart`, mini golden check, scp into
        `baselines/` only if Step 7 established a deliberate change (it should
        not).
  - [x] Step 9: design doc — a landed section carrying Step 1's padded-row table
        (it closes L2's band whichever way the task goes) and Step 2's counter
        table, an "**After P15**" ledger block, and the "Where the time goes"
        routed row gaining the measured padding split. Follow-ons recorded: the
        arm not taken with its measured size; the 16-row rung; audit L10's
        per-tile host work, which the `routed→routed` row will still hold. Plan:
        Task 15 `[x]` with the landed paragraph. Task review by a fresh reviewer;
        fixes folded into the commit (rebase and amend, never a fixup commit).

  **Verdict line template** (the controller fills in the measured values):

  > **LANDED `<sha>` (`<date>`): measured on the mini `prefill_routed_tile`
  > 1.79 → `<x>` ms/prompt-token (3.7k; 1.737 → `<x>` at 12k),
  > `prefill_gdn_router` 1.786 → `<x>` / 1.775 → `<x>`, `prefill_attn_router`
  > 0.757 → `<x>` / 1.565 → `<x>`, `prefill_shared_expert` 0.169 → `<x>`, GPU
  > busy 4.665 → `<x>` and 5.337 → `<x>`, gaps 0.493 → `<x>` and 0.273 → `<x>`
  > (`routed→routed` `<x>`, `shared→routed` `<x>`), wall 21.52 → `<x>` s and
  > 70.76 → `<x>` s = `<x>` ms/prompt-token.** Step 1 measured the padding at
  > `P64` = `<x>` rows against `<x>` real per layer-chunk (`<x>` %; the two
  > models said +11–16 % and +26 %), `P32` = `<x>`, `P16` = `<x>`, over `<x>`
  > blocks in `<x>` waves with `<x>` wave splits. Step 1b: `per_tile_ms` at
  > 128 / 97 / 65 rows per expert = `<x>` / `<x>` / `<x>` — a padded row
  > `<does | does not>` cost a real row. Step 2 measured
  > `expert_hits_prefill` 0 → `<x>`, `expert_misses_prefill` 28,387 → `<x>`,
  > 12k wall `<x>` s against `<x>` s on the same binary with
  > `SHRIKE_PREFILL_SWEEP=fixed`; the 3.7k control moved `<x>` %. The rule took
  > **`<L2 | L5 | both | neither>`** (L2 modelled `<x>` ms/token at a measured
  > α = `<x>`, L5 measured `<x>`). Bars `<cleared | missed>`. M4 Pro check:
  > `<x>`. Numerics: `<the bit-identity outcome>`. Golden identical on both
  > boxes and both profiles (digests unchanged: M4 Pro long `e04d4e8ee7f1590d`,
  > M1 long `899a25e60a365e60`). Five gates green (`<n>` tests, TSAN 0 reports).

  **Risks:**
  - **The 32-row `matmul2d` descriptor may not compile, or may not be
    supported.** `matmul2d_descriptor(kMPPAffineTileM, TILE_N, TILE_K, ...)`
    (`tensorops.metal:165-168`) has only ever been instantiated at M = 64, and
    MPP's legal shapes are not visible in this tree. **Step 5's first action is
    to compile a 32-row instantiation and run one `runPair` through it** —
    before the planner, the tables or the encoder change. If it does not
    compile, option 2 (two tails inside one 64-row tile) is the fallback and the
    task's size estimate does not change.
  - **`α` is the whole lever and it is unmeasured.** The dequant of a weight
    tile does not shrink with M, so a 32-row tile can cost anywhere from half a
    64-row tile to all of one. At `α ≥ 0.70` neither anchor clears the
    threshold. This is why the bench precedes the production wiring, and why the
    abandon rule is written before the work.
  - **Two dispatches per GEMM is more host work, and `routed→routed` is already
    a gap.** Three extra encoders per wave × ~60 waves per layer-chunk × 120 is
    ~21,600 extra encoders per prompt, landing in the same `routed→routed` gap
    audit L10 owns. Mitigation: skip each region's dispatch when it is empty
    (a padding-free wave issues exactly today's six). If the ledger shows the
    gap growing by more than the GEMM saving, that is the null.
  - **The row tile is a constant in three places.** `tensorops.metal:9`,
    `prefill.metal:891` (separately compiled modules; the comment at `:887-890`
    says so) and the Swift mirror `PrefillGroupedRoutedMoE.swift:92`. A
    32-granular table over the bench's 2,048 staging rows is 64 entries against
    a cap of 32 (`MPPPrefillInt4QMM.swift:52`) — raise the cap in the same
    change or the bench throws `invalidArguments`.
  - **The tile scheduler's slot reservations may not behave as L5's model
    predicts.** `makeExpertCachePlan` reserves loading slots, the caller's
    `avoidingSlots` and every hit before selecting victims
    (`PreadExpertStreamer.swift:620-655`), and the policy is aging-LFU with a
    halving every 1,024 plans (`:622-629`), not plain LRU. Under a uniform sweep
    the counts are equal and the tiebreak is LRU, but a real routing trace is
    not uniform. The counters are the arbiter, not the model: if
    `expert_hits_prefill` does not move, the sweep order was not the binding
    constraint and the knob lands defaulted `fixed`.
  - **The alternation changes which experts are resident when decode starts.**
    The measure prompts decode 8 tokens (`tools/prefill-prompts.py:44`), so the
    effect on the wall is negligible, but read `expert_hits_decode` /
    `expert_misses_decode` in the same runs and record them; a long generation
    after a long prompt is a different regime and belongs in the follow-ons if
    the numbers move.
  - **Descending physical offsets may read worse than ascending.** Expert reads
    are `F_NOCACHE` pread (`PreadExpertStreamer.swift:234`) and nothing in the
    reader sorts or coalesces, but the device's own readahead is not modelled
    here. If the alternate arm's hit rate rises while the wall does not, this is
    the first thing to check — `io_ms` is decode-only
    (`RealForwardRunner.swift:5595`, `:6130`), so the evidence is the gap rows,
    not the I/O counters.
  - **The probe must not be left in the tree.** It threads `inout` counters
    through two hot encoders; Step 1 ends with a revert and a clean redeploy,
    verified by `prefill_projection_path=` on the relaunched server's log
    (`tools/mini-deploy.sh:79`).
  - **The mini is production.** Every step here stops the server on 8081 and
    relaunches it; Turbo on 8080 is a different project and is never touched.
    One model process at a time — `pgrep` first, every time.

### Task 15b: P15b — a 32-row tail tile for the grouped routed GEMM

- [ ] **P15b: the padding tax's recoverable half.** Task 15 measured the
  grouped routed GEMM's padding on the mini at **+25.8 %** of the real rows
  per 12k prompt (`P64` 4,946,880 against 3,931,200; a 32-row tail would hold
  them in 4,410,976, (P64 − P32) / P64 = 10.8 %), and its bench control showed
  a padded row costs a real row within 3 %. This task is Task 15's option 1:
  pack each wave body-first — full 64-row tiles in `[0, bodyRows)`, every
  block's remainder of ≤ 32 rows in a 32-aligned tail region — and issue two
  grouped dispatches per GEMM, the 64-row kernel over the body and a 32-row
  instantiation over the tail, each skipped when its region is empty; a
  remainder in `[33, 63]` stays on the 64-row path (two 32-row tiles cost 2α
  against one 64-row tile). The descriptor compiles and links at TILE_M = 32
  and 16 (offline, `xcrun metal -std=metal4.0`); pipeline creation on the M1
  is the runtime check.

  **The α gate comes first.** α is a 32-row tile's cost relative to a 64-row
  tile's — bounded below by 0.5 (only the matmul scales with M) and above by
  1.0 (the `TILE_N × TILE_K` dequant is per threadgroup and does not shrink).
  `ShrikeBench routed_gemm` gains a `routed_gemm_row_tile row_tile=32` line:
  the same 1,024 rows as 32 tiles of 32 against 16 tiles of 64, α = t₃₂ / 2 t₆₄
  (an upper bound — the wave's gather/activation/scatter are inside both).
  With Δ = G × (1 − α) × 2 × 0.1083 × 120 / 12,285 and G = 160–166 ms per
  layer-chunk of GEMM: α = 0.55 → 0.152–0.158 ms/token, 0.70 → 0.102–0.105.
  **Proceed to the production packing only if the mini's α ≤ 0.70 (Δ ≥ 0.10
  ms/token)**; otherwise land the instantiation and the `rowTile:` plumbing
  behind `SHRIKE_PREFILL_TAIL_TILE=32` defaulted `off`, record α, and stop
  (P8's precedent — a measured null is a result). The M4 Pro's α is a check.

  **Bars (mini, 12k, ms per prompt token unless stated).** `prefill_routed_tile`
  1.734 → ≤ 1.734 − Δ (≤ 1.58 at α = 0.55, ≤ 1.63 at 0.70); GPU busy 5.33 →
  ≤ 5.18 / ≤ 5.23; wall 69.97 → ≤ 68.1 s / ≤ 68.7 s; 3.7k routed 1.79 → ≤ 1.66.
  `routed→routed` (0.072 after P15) may not grow by more than the GEMM saving:
  three extra encoders per wave × ≈ 55 waves × 120 layer-chunks ≈ 20k per
  prompt is the risk, and an empty region issues no dispatch. Bench, both
  boxes: at `rowsPerExpert: 65` (every block a 1-row remainder) the tail arm
  ≤ 0.85 × the 64-row path; at 128 (no remainder) and 97 (a 33-row remainder,
  which stays on the 64-row path) within 1 %. Golden **IDENTICAL** on both
  boxes and both profiles — bit-identity is expected (TILE_M changes which
  rows share a threadgroup, not the K loop or the `accumulator += groupProduct`
  fold) and is asserted by `thirtyTwoRowGroupedTileIsBitIdenticalToTheSixtyFourRowTile`
  on full-mantissa inputs; if that test fails, the bar becomes 2e-2 against the
  fp32 reference and a golden recapture with before/after digests, stated in
  the verdict. Five gates. M4 Pro 12k + 25k as the check.

  **Files.** `sources/Shrike/Metal/TensorCore/tensorops.metal` (`TILE_M` as the
  body's first template parameter; `MPP_GROUPED_KERNEL(NAME, TILE_M, TILE_N,
  TILE_K, BUFFERS)`; `mpp_prefill_affine_grouped_f16_n32k256b1_m32`; the plain
  kernel keeps `kMPPAffineTileM`), `sources/Shrike/Metal/Prefill/prefill.metal`
  (`row_tile` in `PrefillRoutedGroupedParamsMSL`; `kPrefillRoutedGroupedRowTile`
  retired), `sources/Shrike/Kernels/TensorCore/MPPPrefillInt4QMM.swift`
  (`enum GroupedRowTile { m64, m32 }`, `groupedMaxRowTiles` 32 → 64,
  `encodeGrouped(..., rowTile:)`, the `_m32` pipeline compiled for the default
  variant only), `sources/Shrike/Kernels/Prefill/MoE/PrefillGroupedRoutedMoE.swift`
  (`planExpertWaves(..., rowTile:)`, `rowTileTable(for:rowTile:)`,
  `encodeGroupedExpertGEMMs(..., rowTile:)`; then the packing:
  `PrefillRoutedExpertWave.bodyPaddedRows`, `rowTileTables(for:) -> (body,
  tail)`, two dispatches per GEMM), the bench façade and
  `sources/ShrikeBench/RoutedGEMMBench.swift` (`rowTile:`, the α line, the tail
  arm), `tests/Shrike/Core/Kernels/Prefill/PrefillGroupedRoutedMoETests+Execution.swift`
  (the bit-identity test on irregular inputs; the 40/32/5/3 planner fixture at
  a 32-row tile: 256 → 160 padded rows; the tail skipped when no block has a
  remainder ≤ 32). Lint: baselined functions whose spans move → regenerate.
  Unchanged: `tileK` and the six variants, the plain kernel's 64, the tile
  scheduler, decode.

  Steps:
  - [ ] Step 1 (the spike, lands): the `TILE_M` plumbing, the `_m32`
        instantiation, the `rowTile:` parameters (default 64 everywhere), the
        α bench line, the bit-identity and planner tests (RED first);
        `swift test --no-parallel --filter PrefillGroupedRoutedMoE`;
        `ShrikeBench routed_gemm 20` on the mini (GPU idle) then the M4 Pro;
        record α on both boxes.
  - [ ] Step 2 (the gate): Δ from the mini's α; proceed only at α ≤ 0.70.
  - [ ] Step 3: failing tests for the body/tail packing and the empty-region
        skip; implement; `SHRIKE_PREFILL_TAIL_TILE=off|32` (default `32` if
        Step 2 passed, else `off`); `tail_tile=` on the projection-path line.
  - [ ] Step 4: `routed_gemm` tail arm at 128 / 97 / 65 on both boxes (bars
        above); on a miss, default `off` and say so.
  - [ ] Step 5: five gates; golden both boxes (IDENTICAL); deploy; ledger rows
        mini 3.7k + 12k (the verdict), M4 Pro 12k + 25k (the check).
  - [ ] Step 6: design doc Step 14 + "After P15b"; plan `[x]` with the landed
        paragraph; task review; fixes folded in.

## Follow-ons (not scheduled)

- The mini's SSD term (v10 P3 follow-on: batched miss loads, deeper queue
  depth) — also the second lever for speculative decode's verify pass
  (measured at P4 on the mini: 6.6 tok/s against 25.5 plain, and again at P9:
  6.3–6.5 tok/s with the verify backbone unchanged at 155–159 ms; see the design's
  out-of-scope note).
- Speculative decode's acceptance rate: 25.9 % (22.3 % at P9) on the counting rig prompt at
  P4 — audit the draft/verify path before any kernel work on that track.
- The `expert_hit_rate_prefill` counter that reads 0–14 % with every expert
  resident.
