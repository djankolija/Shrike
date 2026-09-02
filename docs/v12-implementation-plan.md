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

## Follow-ons (not scheduled)

- The mini's SSD term (v10 P3 follow-on: batched miss loads, deeper queue
  depth) — also the second lever for speculative decode's verify pass
  (measured at P4 on the mini: 6.6 tok/s against 25.5 plain; see the design's
  out-of-scope note).
- Speculative decode's acceptance rate: 25.9 % on the counting rig prompt at
  P4 — audit the draft/verify path before any kernel work on that track.
- The `expert_hit_rate_prefill` counter that reads 0–14 % with every expert
  resident.
