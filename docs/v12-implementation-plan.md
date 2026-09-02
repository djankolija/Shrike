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

- [ ] **P4: GDN chunked scan** — scheduled by the P3 decision rule (M4 Pro 3.7k
  GPU busy 2.42 ms/tok > 2.0). Target `prefill_gdn_router` 1.00 → ≈ 0.45
  ms/prompt-token (M4 Pro, 12k): the scan is ≈ 0.6 of the role's 1.0 ms; the
  projections, conv and norms stay. Scope: the scalar per-head decay shape
  only (ornith: `in_proj_a` has `Hv = 32` rows). The per-channel variant
  (`gdn_delta_step_prefill_vec`, Kimi KDA), any shape other than
  `Dk = Dv = 128`, chunks under 64 rows and the 32-token MTP draft chunk keep
  the serial kernel.

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
        static func chunkFactorsBytes(config: LinearAttentionConfig, chunkTokens: Int) -> Int  // Hv · ⌈chunkTokens/64⌉ · 17_408
        var chunkedScanAvailable: Bool             // shape supported and both pipelines compiled
        var chunkedScanUnavailableReason: String?  // nil when available
        /// Same contract as `encodeDeltaStepPrefill` plus the factors scratch. Throws
        /// `GDNChunkedScanError.tooFewRows` below 64 rows, `.unavailable(reason)`,
        /// `.factorsTooSmall(needed:have:)`.
        func encodeDeltaStepPrefillChunked(commandBuffer:, convOut:, convOutOffset:,
                                           aProj:, aProjOffset:, bProj:, bProjOffset:,
                                           aLog:, aLogOffset:, dtBias:, dtBiasOffset:,
                                           state:, checkpointState:, y:, yOffset:,
                                           rows: Int, factors: MTLBuffer) throws
    }
    enum GDNChunkedScanError: Error {
        case tooFewRows(Int), unavailable(String), factorsTooSmall(needed: Int, have: Int)
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
    `PrefillChunkScratchLayout.usesGDNChunkedScan` (`gdnQKVDim > 0 &&
    !perChannel && Dk == Dv == 128 && chunkTokens >= 64` — the 32-token draft
    chunk allocates nothing) and `gdnChunkFactorBytes`, counted in
    `devicePrivateBytes` (34 MB at 4,096-token chunks for ornith);
    `PrefillChunkScratchBuffers.gdnChunkFactors: MTLBuffer?`. Bench:
    `ShrikeBench gdn_scan [iterations]` — ornith shape, `T = 4096`, serial vs
    chunked ms per call and µs per token, chunked TFLOPS against the `gemm`
    ceiling.

  Steps (TDD; the CPU model first so the math is verified before any Metal):

  - [ ] Step 1: `GDNReference` split — `normalize`, `deltaRule`, `gatedNorm`;
        `step` = the three in sequence. `swift test --no-parallel --filter
        GDNKernelTests` → still green.
  - [ ] Step 2: failing test `GDNChunkedScanTests.chunkedReferenceMatchesSerialReference`:
        cfg `(Hk 1, Hv 2, Dk 128, Dv 128, conv 4)`; seeded normed rows (q/k unit
        vectors scaled like the kernel's norm, then fp16-rounded; v fp16 in
        [−1, 1]), `a, b` in [−1, 1], `A_log` in [−1, 1.5], `dt_bias` in
        [−0.5, 0.5], a random non-zero incoming state in [−0.5, 0.5]; T ∈ {64,
        200}; serial = `deltaRule` per row on a copy of the state; expect
        `RelError.maxAbsDiff(y) ≤ 1e-3`, state `≤ 1e-3`, checkpoint == serial
        state after row 0 within 1e-5. FAIL: `GDNChunkedReference` undefined.
  - [ ] Step 3: implement `GDNChunkedReference.run` exactly as the math block
        (fp32, `T⁻¹` by forward substitution). PASS.
  - [ ] Step 4: failing tests: `chunkedKernelMatchesSerialKernel` — same
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
  - [ ] Step 5: kernels + `GDN.swift` + `MetalContext` registration → the
        tests PASS. If the compiler rejects the `(32, 128, 64, true, false)`
        descriptor (the one shape without precedent in the tree), use `(64,
        128, 64, true, false)` over a 64-row view of `u_tile` and keep the
        first 32 destination rows.
  - [ ] Step 6: bench `swift run -c release ShrikeBench gdn_scan 20`: serial
        vs chunked at the ornith shape; the bar is ≥ 4× on the M4 Pro (the
        serial scan is ≈ 82 ms per 4,096-row layer call today; target ≤ 20
        ms). If short, the first knob is `chunkedScanValueBlock = 16`
        (`(Dv/16, Hv)` grid, 8 KB state tile) — measured, not assumed.
  - [ ] Step 7: scratch + runner + env knob + description + server log line +
        `ServerModelSession` field. Five gates: release build 0 warnings,
        lint, links, suite, TSAN.
  - [ ] Step 8: `tools/golden-baseline.sh --check 4` (M4 Pro, server stopped)
        — expected to differ; recapture; commit `gdn: chunked delta-rule scan
        on the matrix path (v12 P4)` with the baseline via `--only`.
  - [ ] Step 9: ledger on both boxes (fresh server, 3.7k + 12k,
        `tools/prefill-measure.sh`); mini `tools/mini-deploy.sh --restart`,
        mini golden recapture, scp into `baselines/`. Verdict line here;
        design doc Step 4 + ledger rows + the GDN row of "Where the time
        goes"; task review by a fresh reviewer; fixes folded into the commit.

## Follow-ons (not scheduled)

- Tile command-buffer batching (the ~1.3 ms per tile boundary), once P3 lands.
- The mini's SSD term (v10 P3 follow-on: batched miss loads, deeper queue depth),
  once P2 exposes it.
- The `expert_hit_rate_prefill` counter that reads 8–14 % with every expert
  resident.
