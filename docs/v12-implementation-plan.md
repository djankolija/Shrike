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

### Task 4: P4 — GDN chunked scan (conditional)

- [ ] **P4: GDN chunked scan** — conditional on the P3 decision point; target
  1.0 → ~0.45 ms/token. Chunked gated delta rule at 64-token chunks: per chunk,
  build the WY factors (`W = (I + tril(β K Kᵀ))⁻¹ β K`, forward-substitution on
  a 64×64 lower-triangular block), intra-chunk output `Q·(W-corrected KV)` via
  `matmul2d`, inter-chunk state `S ← decay · S + Kᵀ·U` carried in the same
  `state` buffer `gdn_delta_step_prefill` writes; the per-channel decay variant
  (`perChannelG`) needs the decay folded into the chunk's `K` rows before the
  factorization. Oracle: `GDNReference` (`Sources/ShrikeValidation/Support/
  Reference/GDN/GDNReference.swift`), tolerance `2e-2`, at `T ∈ {64, 200,
  4096}` with a non-zero incoming state. Bench: `ShrikeBench gdn_scan` at
  `T = 4096`. Same gates, recapture, ledger. Written out in full only when
  scheduled; the P3 verdict says whether it is.

## Follow-ons (not scheduled)

- Tile command-buffer batching (the ~1.3 ms per tile boundary), once P3 lands.
- The mini's SSD term (v10 P3 follow-on: batched miss loads, deeper queue depth),
  once P2 exposes it.
- The `expert_hit_rate_prefill` counter that reads 8–14 % with every expert
  resident.
