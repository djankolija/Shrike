# v12 — Prefill on the matrix path

Status of record: [v12-implementation-plan.md](v12-implementation-plan.md).

## The problem

Decode closed at 38.1 ms/token on the mini ([v10](v10-implementation-plan.md),
[v11](v11-implementation-plan.md)). Prefill was never touched, and it is now the
slower phase per token at any prompt that fills more than one chunk:

| box | prompt tokens | prefill wall | per prompt token | vs its own decode |
| --- | ---: | ---: | ---: | ---: |
| M1 mini, 8-core GPU, 16 GB | 3,756 | 110.7 s | 29.5 ms | 0.8× |
| M1 mini | 12,285 | 725.2 s | 59.0 ms | **1.5× slower** |
| M4 Pro, 20-core GPU, 48 GB | 4,305 | 37.6 s | 8.7 ms | |
| M4 Pro | 25,245 | 666.8 s | 26.4 ms | |

Prefill is the compute-bound phase: 4,096 tokens share every weight read, so the
cost is arithmetic, not bandwidth. Against each box's *measured* matmul ceiling
(below), both sit at the same tenfold distance. The M1/M4 Pro wall ratio is
3.9×; the ceiling ratio is 3.95×. It is the code, not the machine.

## Method: the target from the measured ceiling

Same derivation that settled decode's target: hardware ceiling, discounted for
what a real kernel achieves, then per-term work from the model's actual shape.

**Work per token** (from `manifest.json`: hidden 2048, 10 full-attention + 30
GDN layers, 256 experts top-8 with intermediate 512, gated shared expert 512,
16 query heads × 256 over 2 KV heads):

| term | GFLOP per token |
| --- | ---: |
| attention projections, 10 layers (gated Q 8192, K/V 512, O) | 0.55 |
| GDN projections + recurrence, 30 layers | 2.15 |
| routed experts, 40 layers × 8 × 3 × (2048×512) | 2.01 |
| shared expert + router, 40 layers | 0.29 |
| **length-independent total** | **5.0** |
| attention scores + values, extra per token at prompt length N | 0.082 × N/1000 |

**Ceiling, measured** with `ShrikeBench gemm` (Apple's MPS fp16 GEMM, the
best-known kernel, at the prefill shapes; `gpuEndTime − gpuStartTime`):

| shape | M4 Pro TFLOPS | M1 TFLOPS |
| --- | ---: | ---: |
| 4096³ | 7.46 | 1.86 |
| gated Q projection, 4096×2048×8192 | 7.58 | 1.67 |
| GDN in-projection, 4096×2048×12288 | 7.57 | 1.62 |
| one expert's gate/up at 128 routed rows, 128×2048×1024 | 5.74 | 1.84 |
| one expert's down at 128 rows, 128×512×2048 | 5.88 | 1.78 |
| same at 32 rows | 3.62 | 0.97 |

Sticker (core count × ALUs × clock) is ~9.2 and 2.6 TFLOPS, so the real
ceiling is 80 % and 72 % of sticker. 128 routed rows per expert, which is what
a 4,096-token chunk yields on average, still gets three quarters of the big-shape
rate; the matrix path pays off at the expert shape too.

**Target**: 50 % of the measured ceiling for our kernels (int4 dequant inside the
loop, per-expert row gathers, attention tiles), the analogue of decode's 65 %.

| | M4 Pro | M1 mini |
| --- | ---: | ---: |
| effective rate at 50 % | 3.7 TFLOPS | 0.95 TFLOPS |
| per token at 4k | 1.4 ms | 5.6 ms |
| per token at 12k | 1.6 ms | 6.3 ms |
| per token at 25k | 1.9 ms | 7.4 ms |
| SSD term per 4,096-token chunk (mini: 128 uncached experts × 40 layers × 1.77 MB at 2.8 GB/s) | none, experts resident | 3.2 s = 0.8 ms/token, hidden only if pipelined |
| **tok/s at 4k** | **~700** | **~180** |

Today is 5× above this target on both boxes at 4k and 10× at 12k.

## The ledger

Measured 2026-09-01 at commit 3c326d9, `SHRIKE_KERNEL_STATS=1`, 4,096-token
chunks, 8-bit KV cache, greedy, 8 new tokens, one request per prompt, prompts
from `tools/prefill-prompts.py` (the golden-baseline ledger text at 60/180/360
entries; it tokenizes at 2.6 bytes/token). The mini's rows came through its
production server; the M4 Pro's through a fresh server with `--ram-budget 20G`
(all experts resident). GPU ms per **prompt** token, from the role sums:

| role | M4 Pro 3.7k | M4 Pro 4.3k | M4 Pro 12k | M4 Pro 25k | M1 3.7k | M1 12k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `prefill_attn_router` (10 layers) | 3.39 | 3.87 | 11.85 | 22.19 | 13.80 | 43.43 |
| `prefill_routed_tile` | 2.03 | 2.03 | 2.04 | 2.03 | 5.82 | 5.82 |
| `prefill_gdn_router` (30 layers) | 0.99 | 1.00 | 1.01 | 1.01 | 4.37 | 4.39 |
| `prefill_shared_expert` (40 layers) | 0.83 | 0.83 | 0.85 | 0.84 | 3.92 | 3.97 |
| `prefill_moe_reduce` | 0.01 | 0.01 | 0.01 | 0.01 | 0.04 | 0.04 |
| **GPU busy** (prefill roles + the 8 decode tokens) | 7.34 | 7.77 | 15.77 | 26.09 | 28.13 | 57.82 |
| gaps (span − busy) | 0.87 | 0.95 | 0.22 | 0.32 | 1.30 | 1.19 |
| outside the span | 0.43 | 0.01 | 0.01 | 0.01 | 0.03 | 0.01 |
| **wall** | 8.64 | 8.74 | 16.00 | 26.41 | 29.46 | 59.03 |

The ledger closes: wall = busy + gaps + a per-request remainder (the first
column's remainder is the fresh server's first request).

**After P1** (commit 1450c1c, 2026-09-02, same protocol, fresh servers):

| role | M4 Pro 3.7k | M4 Pro 12k | M1 3.7k | M1 12k |
| --- | ---: | ---: | ---: | ---: |
| `prefill_attn_router` | 3.40 | 13.53 † | 13.82 | 43.58 |
| `prefill_routed_tile` | 2.02 | 2.10 | 5.82 | 5.81 |
| `prefill_gdn_router` | 1.00 | 1.03 | 4.38 | 4.38 |
| `prefill_shared_expert` | **0.058** | **0.061** | **0.29** | **0.29** |
| **GPU busy** | 6.52 | 16.74 | 24.52 | 54.16 |
| **wall** | 7.32 | 16.92 | 25.66 | 55.02 |

† The M4 Pro's 12k attention row moved +14 % on an unchanged kernel while the
M1's stayed within 0.4 %; that laptop had run hours of test suites before
the measurement. Treat the M1 as the reference for cross-step comparisons of
untouched roles.

**After P2** (commit 066fe67, 2026-09-02; matrix-path attention, 32 rows ×
4 simdgroups, lane-parallel softmax):

| role | M4 Pro 3.7k | M4 Pro 12k | M4 Pro 25k | M1 3.7k | M1 12k |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prefill_attn_router` | **0.27** | **0.47** | **0.85** | **1.36** | **2.40** |
| `prefill_routed_tile` | 2.03 | 2.03 | 2.24 † | 5.83 | 5.82 |
| `prefill_gdn_router` | 1.01 | 1.00 | 1.09 † | 4.38 | 4.36 |
| `prefill_shared_expert` | 0.06 | 0.06 | 0.07 | 0.29 | 0.29 |
| **GPU busy** | 3.42 | 3.58 | 4.26 | 12.07 | 12.96 |
| **wall** | 4.31 | 3.76 | 4.52 | 13.19 | 13.51 |
| wall, seconds | 16.2 | 46.1 | 114.0 | 49.5 | 166.0 |

**After P3** (commit 9323fa8, 2026-09-02; per-expert GEMMs for routed
experts with ≥ 32 pairs in a tile, scalar path for the rest):

| role | M4 Pro 3.7k | M4 Pro 12k | M4 Pro 25k | M1 3.7k | M1 12k |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prefill_attn_router` | 0.27 | 0.46 | 0.89 | 1.36 | 2.40 |
| `prefill_routed_tile` | **1.04** | **1.00** | **1.21** | **3.42** | **3.35** |
| `prefill_gdn_router` | 1.01 | 0.99 | 1.16 † | 4.39 | 4.37 |
| `prefill_shared_expert` | 0.06 | 0.06 | 0.07 | 0.29 | 0.29 |
| **GPU busy** | 2.42 | 2.53 | 3.35 | 9.68 | 10.51 |
| **wall** | 3.42 | 2.79 | 3.65 | 10.80 | 11.04 |
| wall, seconds | 12.9 | 34.3 | 92.1 | 40.6 | 135.6 |

After P3 the 12k prompt was 6.1× faster than the chapter's start on the M4
Pro (207.8 → 34.3 s) and 5.3× on the M1 (725.2 → 135.6 s).

**After P4** (commit 9b9374d, 2026-09-02; chunked delta-rule scan for the
GDN layers, 64-row chunks):

| role | M4 Pro 3.7k | M4 Pro 12k | M4 Pro 25k | M1 3.7k | M1 12k |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prefill_attn_router` | 0.26 | 0.45 | 0.80 | 1.35 | 2.41 |
| `prefill_routed_tile` | 1.04 | 0.99 | 1.12 | 3.42 | 3.35 |
| `prefill_gdn_router` | **0.65** | **0.64** | **0.70** | **3.44** | **3.43** |
| `prefill_shared_expert` | 0.06 | 0.06 | 0.06 | 0.29 | 0.29 |
| **GPU busy** | 2.11 | 2.17 | 2.71 | 8.72 | 9.58 |
| **wall** | 3.46 | 2.65 | 3.22 | 10.40 | 10.28 |
| wall, seconds | 13.0 | 32.6 | 81.2 | 39.1 | 126.3 |

The scan took the GDN role from 1.0 to 0.65 ms/token on the M4 Pro (0.70 at
25k) and from 4.4 to 3.4 on the M1; the other roles moved within their
run-to-run noise. Same-binary A/B at 3.7k on the M4 Pro
(`SHRIKE_GDN_PREFILL_SCAN=serial`), two pairs: GDN 0.65 / 0.69 chunked
against 0.96 / 0.99 serial, GPU busy 2.11 / 2.18 against 2.37 / 2.41. The
first pair's walls inverted (12.98 chunked, 12.49 serial) on a 1.5 s
difference in tile-boundary gaps — host-side, ±0.4 ms/token run to run at
this length — and the second pair read 11.5 against 12.5 s. What is left of
the role, ≈ 0.64 ms/token on the M4 Pro, is ≈ 0.07 of scan (9.5 ms × 30
layers per 4,096 rows) and ≈ 0.57 of projections, conv and norms; the in
and out projections already run on the matrix path, so the role is now
GEMM-bound like the routed experts.

Against the chapter's start, the 12k prompt is 6.4× faster on the M4 Pro
(207.8 → 32.6 s) and 5.7× on the M1 (725.2 → 126.3 s); the M1 now prefills at
10.3 ms per prompt token against its 38.1 ms decode token. The routed GEMMs run
at ≈ 2.0 TFLOPS on the M4 Pro, 35 % of the 128-row expert-shape ceiling; the
tile-boundary gaps grew with the extra host encoding per tile (0.86 s of 12.9
at 3.7k), which promotes the command-buffer batching follow-on.

Attention-core efficiency on the shipped kernel: ≈ 36 % of the M4 Pro's
measured ceiling at 12k and 25k (2.6–2.7 TFLOPS on 12.4 / 52.2 TFLOP of
scores and values), ≈ 26 % on the M1. The remaining cost is the eightfold
re-read of each K/V tile by the eight query heads of a KV group (each
threadgroup owns one head, because the eight heads' Q rows are not a single
uniform-stride matrix) and the 32-row `matmul2d` tile; a kernel that stages a
KV-head group's K/V tile once for all eight heads is the follow-on. The 64-row
× 8-simdgroup variant measured 10 % slower and stays selectable
(`SHRIKE_ATTN_MATRIX_TILE=r64s8`); `SHRIKE_PREFILL_ATTENTION=tiled` A/Bs the
scalar kernel on the same binary.

- **Attention time is proportional to query–key pairs, not tokens.** Across
  chunks the pairs grow as 4096 × (2048 + 6144 + 10240 + …); 12k has 10.7× the
  pairs of 3.7k and took 11.4× the time. The projections inside the role are
  invisible at this scale (intercept ≈ 0 on the M4 Pro, ~0.8 ms/token on the
  M1). At 25k tokens the attention core is 85 % of prefill.
- **The other three roles are flat per token** on both boxes, at 3.9–4.7× the
  M4 Pro's cost on the M1.
- **Gaps are a fixed tax per tile.** `prefill_routed_tile→prefill_routed_tile`
  is ~1.3 ms per tile boundary (its own command buffer, commit and wait): 2.8 s
  of the 37.6 s at 4.3k. Small today, but it does not shrink when kernels do.
- **I/O is hidden under the kernels** on both boxes (occupancy 89–99 %,
  runner `io_ms` under 50 ms per request). That corrects the v10 P3 entry's
  "a large slice of prefill's 33 ms/token": the 28–33 ms/token is kernel time.
  The SSD term surfaces only once the kernels are ~5× faster.
- The runner's `expert_hit_rate_prefill` reads 8–14 % even with every expert
  resident on the M4 Pro; warm and cold runs cost the same, so the counter,
  not the cache, is what is off. Parked.

**After P5** (commits a7c8288 + bf469ad, 2026-09-02; routed tiles batched per
command buffer behind `SHRIKE_PREFILL_TILE_BATCH`, default 1 — a measured
null result, rows at the default width; the gap split `host_ms` / `driver_ms`
/ `queue_ms` lands with it):

| role | M4 Pro 3.7k | M4 Pro 12k | M1 3.7k | M1 12k |
| --- | ---: | ---: | ---: | ---: |
| `prefill_attn_router` | 0.27 | 0.47 | 1.35 | 2.40 |
| `prefill_routed_tile` | 1.03 | 1.00 | 3.41 | 3.35 |
| `prefill_gdn_router` | 0.65 | 0.66 | 3.43 | 3.42 |
| `prefill_shared_expert` | 0.06 | 0.06 | 0.29 | 0.29 |
| **GPU busy** | 2.05 | 2.21 | 8.70 | 9.57 |
| gaps (span − busy) | 0.53 | 0.36 | 0.66 | 0.52 |
| **wall** | 2.91 | 2.68 | 9.89 | 10.26 |
| wall, seconds | 10.9 | 32.9 | 37.2 | 126.0 |

Batching lost at every width. Same binary, M4 Pro, 3.7k, two interleaved
pairs: GPU busy and the routed role did not move, the routed→routed gap went
from 0.85 / 0.95 s at width 1 (1,170 boundaries, ≈ 0.7 ms each) to 3.1 s at
width 4 (277 boundaries, 11.4 ms each) and 3.0 s at width 8; the 12k wall
went 32.9 → 41.4 s, the M1's 3.7k wall 37.2 → 41.4 s. The split says where:
at width 4 the routed gap is host 3.00 s, driver 0.01, queue 0.16. A per-step
host probe (uncommitted; the SDD report for Task 5) found every encode-side
step flat across widths — validate + argument buffer 0.012 ms per tile,
encode 0.019, commit 0.004 — and the whole penalty in the wait. The reason is
the fetch. `fetchBindingForTile` costs 3.9 ms per tile on the M4 Pro at
0 % prefill hit rate, of which the pread (`io_fetch_ms`) is 0.5 ms; the
GPU tile is 3.2 ms. At width 1 the pending-tile overlap fetches tile i+1
while tile i runs, so the GPU only waits for the fetch's excess (≈ 0.7 ms per
boundary, 2.3 s of the 32.9 at 12k) — the loop is fetch-bound, not
commit-bound. Batching serializes the fetches of tiles 2..W against an idle
GPU, which is the measured 3.6 ms per non-first tile. The premise that a
boundary was ≈ 1.3 ms of commit → wait → encode was wrong: driver + queue per
boundary is ≈ 0.3 ms, so a zero-penalty batch could have saved ≈ 0.8 s of
32.9. On the M1 the routed tiles abut at width 1 (10.6 ms of GPU per tile
hides the fetch entirely); its gaps live elsewhere.

Where the gaps are now, at 12k (ms per prompt token): M4 Pro 0.36 = routed→
routed 0.19 (the fetch excess) + shared→routed 0.10 (≈ 10 ms per layer, of
which ≈ 11 ms at 3.7k is `driver_ms`, see the follow-ons) + GDN→shared 0.02
(host routing) + the decode tokens' fixup transitions 0.02 + ≈ 0.03 spread.
M1 0.52 = shared→routed 0.29 (29 ms per layer, 22 ms of it `driver_ms`) +
GDN→shared 0.12 (15 ms per layer of host routing) + attention→shared 0.04
(16 ms per layer, host) + fixup 0.03 + ≈ 0.04 spread.

**After P6** (commit 4c44b8c, 2026-09-02; one grouped dispatch per phase over
every expert's rows in a tile, the 1–31-pair experts inside it, 1,024 staging
rows; `SHRIKE_PREFILL_ROUTED_GEMM=per-expert` keeps the P3 path):

| role | M4 Pro 3.7k | M4 Pro 12k | M4 Pro 25k | M1 3.7k | M1 12k |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prefill_attn_router` | 0.26 | 0.45 | 0.81 | 1.35 | 2.40 |
| `prefill_routed_tile` | **0.72** | **0.69** | **0.82** | **3.22** | **3.13** |
| `prefill_gdn_router` | 0.65 | 0.64 | 0.74 | 3.43 | 3.42 |
| `prefill_shared_expert` | 0.06 | 0.06 | 0.07 | 0.29 | 0.29 |
| **GPU busy** | 1.73 | 1.88 | 2.47 | 8.50 | 9.35 |
| gaps (span − busy) | 0.77 | 0.57 | 0.59 | 0.65 | 0.52 |
| **wall** | 2.90 | 2.58 | 3.12 | 9.70 | 10.04 |
| wall, seconds | 10.9 | 31.6 | 78.8 | 36.5 | 123.4 |

The routed role fell 29 % on the M4 Pro and 7 % on the M1, and the bench says
why the two differ: on the ornith tile (8 experts × 128 rows) the per-expert
sequence ran at 1.69 TFLOPS on the M4 Pro, 29 % of the 5.77 TFLOPS the same
run measures for the 128-row gate/up GEMM, and the grouped dispatch runs at
3.86, 67 % — 2.28× per tile. On the M1 the per-expert sequence was already at
0.69 of a 1.15 TFLOPS ceiling, 60 %, and grouping lifts it to 0.78, 68 %,
1.14×. Grouping's win is a grid large enough to fill the GPU — 256
threadgroups instead of 32 per GEMM — and the M1's eight cores were full
already. Both boxes now sit at two thirds of the MPS ceiling for this shape;
what is left is inside the kernel (the per-K-group dequant → barrier → matmul
serialization and the 32-wide N tile), not in how many experts a dispatch
covers. Same-binary A/B against the P3 path on the M4 Pro: 3.7k wall 11.1 →
10.9 s (pair 2; pair 1's grouped run read 13.6 s on a 22 ms-per-layer driver
spike on the first routed buffer), 12k 32.3 → 31.6 s, 25k 81.2 → 78.8 s — the
GPU saving turns into routed→routed gap on this box (0.76 → 1.39 ms per
boundary, host-side) because the loop is fetch-bound at 3.9 ms per tile; the
M1 hides the fetch under its 7.5 ms tile and keeps the saving: 3.7k 36.8 →
36.5 s, 12k 126.0 → 123.4 s. Golden differs on the long profile on both boxes
(the short prompt never reaches the matrix path) and was recaptured once per
box; the digests are in the plan's verdict.

## Where the time goes

Every dense projection in prefill (attention Q/K/V/O, GDN in/out) already runs
through `mpp_prefill_affine_threadgroup_f16`, the one SIMD-group matrix kernel in
the codebase, which is why they cost nothing measurable. The three slow roles are
all scalar kernels:

| role | kernel | structure | roofline gap (M4 Pro) |
| --- | --- | --- | ---: |
| attention core | `attention_prefill_causal_tiled` | one threadgroup per (query, head), 256 threads over head-dim, a serial walk over every key with a two-barrier threadgroup reduction per key. The decode kernel's shape, run once per prompt token: no K/V reuse across queries, none across the 8 query heads that share a KV head. | 30× at 3.7k, 60× at 25k |
| routed experts | `prefill_grouped_routed_moe_batched_phase1` / `_down` | each thread computes one or two 2048-long scalar dot products; 32-pair microbatches; a command buffer per 8-expert tile | 7×; 2.6× after P6 (`mpp_prefill_affine_grouped_f16`, one dispatch per phase over the tile's experts: 0.69 ms/token at 12k against 0.27 at the 7.46 TFLOPS ceiling) |
| shared expert | `PrefillSharedExpert.encodeBlock` | a `for row in 0..<queryCount` loop over the decode runtime: 4–6 M=1 GEMV dispatches per token | 25× |
| GDN | `gdn_delta_step_prefill` | the delta-rule scan is a serial loop over the chunk inside one dispatch of 32×32 threadgroups; projections, conv and norms are fine | 3.6× (scan ≈ 0.6 of the 1.0 ms) |

A matrix-path attention kernel exists in `prefill.metal`
(`attention_prefill_full_tensorops_2d_validity_v2`, `matmul2d` over 64-key
tiles with online softmax) but is gated to head-dim 512, fp16 KV, scale 1.0,
so this model never selects it and silently gets the scalar kernel. Its tile is
also thin: one query × 8 heads per threadgroup.

## Design

Kernel replacement in ledger order, smallest risk first. Each step is gated by
the same four checks before the next starts:

1. `ShrikeBench` microbench of the new kernel against the `gemm` ceiling.
2. fp32 reference test in `ShrikeValidation` (references exist for attention,
   GDN, MoE and the dequant GEMVs), at single-chunk and multi-chunk shapes.
3. The five repo gates, then `tools/golden-baseline.sh --check` — expected to
   differ (see numerics policy), recaptured on sign-off.
4. The ledger re-measured on both boxes at 3.7k and 12k; the table above gains
   a row.

### Step 0 — harness (landed)

`ShrikeBench gemm`, `tools/prefill-prompts.py`, `tools/prefill-ledger.py`,
`tools/prefill-measure.sh` (the `decode-measure.sh` twin: sends the prompt set
to a running server and prints the ledger). Plus a one-line log at runner init
naming the projection path (`affine-threadgroup-f16` or the fallback), because
nothing today records whether the matrix kernel compiled on a given box.

### Step 1 — shared expert on the matrix path (−11 %)

Replace the per-row loop with, per layer per chunk: gate GEMM (T×2048 →
T×512), up GEMM, one `silu_mul` over T×512, down GEMM (T×512 → T×2048), one
T-row scalar-gate kernel, one `sigmoid_scalar_mul` over T rows — through the
existing `MPPPrefillInt4QMM.encode` (the shared expert is affine int4 group-64
like the projections). Below 32 tokens the row loop stays, mirroring the
projection helper's threshold. Decode is untouched (M=1 there is right).
Expected: 0.83 → ~0.06 ms/token on the M4 Pro.

### Step 2 — attention core on the matrix path (−45 % at 4k, −85 % at 25k)

A blocked causal kernel: a threadgroup owns a block of queries for one KV head
group (the 8 query heads sharing that head, so each K/V tile serves 8× the
rows), streams 32–64-key tiles of K and V through threadgroup memory, computes
QKᵀ and PV with `matmul2d` (the same Metal 4 primitive the projection kernel
uses), keeps the running max/sum per row (online softmax) in registers, and
applies the causal mask only on the diagonal tile. Keys and values load from the
8-bit cache with the same dequant the scalar kernel uses (`prefill_load_kv`), so
the numerical basis is unchanged: same quantized keys, different reduction order.
Tile geometry (rows per threadgroup, key tile, head-dim split against the 32 KB
threadgroup budget at head-dim 256) is decided by a measured spike in
`ShrikeBench`, acceptance ≥ 40 % of the `gemm` ceiling at the 3.7k and 12k
shapes. Gate: this model's shape (head-dim 256, 16/2 heads, no sinks, full
visibility — no sliding window, or a window that already covers the whole
context (the runner passes `kvValidCount` as the window for full layers) —
and `kvValidCount <= 65,536`); any KV bit width reaches it (4, 8, or 16:
`prefill_load_kv` handles 4-bit and the int4 case is tested). Every other
shape, the MLA twin, and the MTP verify chunk keep the scalar kernel, exactly
as today's tensor-ops gate does.
Expected: 3.39 → ~0.2 ms/token at 3.7k; 22.2 → ~0.8 at 25k.

Memory: the KV shadow (`PrefillAttention.ensureShadow`) costs 2 KB per
context token per runner — K and V, fp16, 512 elements each
(`numKVHeads * headDim = 2 * 256`) — grown in 8 MiB quanta and never
released. That is 64 MiB at the mini's 32k max context, and up to twice
that with the MTP draft runner allocating its own shadow; the gate's
`kvValidCount <= 65,536` ceiling bounds it further. It lives outside the
chunk-scratch worksheet by design — sized by context length, not chunk size.

### Step 3 — routed experts as per-expert GEMMs (−20 %)

The tile scheduler already groups token–expert pairs by expert. Per expert in a
tile: gather that expert's rows (~128 at a 4k chunk) into a staging block, gate
and up GEMMs through the MPP kernel against the expert's packed int4 weights,
`silu_mul`, down GEMM, written per pair so `prefill_moe_reduce_token_major` and
the router-weight reduce stay as they are. Below a row threshold per expert
(short prompts, the 32-token verify chunk) the existing scalar kernels stay.
Expected: 2.03 → ~0.6 ms/token (128-row GEMMs at 60 % of their ceiling).

After steps 1–3 the M4 Pro ledger models to ~1.9 ms/token at 4k (3.9×) and
~2.5 at 25k (10×); step 4 is what closes the last 0.5 to the 1.4 target.

### Step 4 — GDN chunked scan (landed)

The only step with new math, scheduled by P3's decision rule. The serial
kernel walks a chunk's rows inside one dispatch: each threadgroup (one value
head, four `dv` rows) keeps its state rows in registers and does one delta
step per row, so a 4,096-row chunk is 4,096 dependent steps of a few dozen
FMAs — 51.7 ms per layer call at the ornith shape on the M4 Pro, 0.25 TFLOPS.
The chunked gated delta rule (Yang et al.; flash-linear-attention's
`chunk_gated_delta_rule`) rewrites each 64-row chunk as matrix products with
the intra-chunk dependency solved through the unit-lower-triangular
`I + A`; the derivation, with the symbols the code uses, is in the plan's
Task 4. Two kernels in `gdn_chunked.metal`: `gdn_chunk_factors` (parallel
over chunks and heads) builds `T⁻¹ = (I + A)⁻¹`, `M` and the per-row decay
scalars into a 34 MB factors scratch (4,096-token chunks, 32 heads, 17,408
bytes per head-chunk); `gdn_chunk_scan` (one threadgroup per head and
32-column state block) walks the chunks in order with the state block in
threadgroup memory, five `matmul2d` products per chunk. fp16 factors, fp32
accumulation and state — the precision flash-linear-attention ships.

`ShrikeBench gdn_scan` (T = 4,096, Hv 32, Dk = Dv = 128, M4 Pro): serial 51.7
ms per layer call, chunked 9.5 ms, 5.4× at 2.27 TFLOPS (30 % of the measured
ceiling); on the same inputs and state the chunked output differs from the
serial kernel by 3.1e-5 maxAbs (rel 5.8e-4) and the state by rel 2.2e-5. The
kernels are compiled for the scalar per-head decay shape with Dk = Dv = 128
and take chunks of 64+ rows; the 32-token MTP draft chunk, per-channel decay
(Kimi KDA) and other head dims keep the serial kernel, and no scratch is
allocated for them. `SHRIKE_GDN_PREFILL_SCAN=serial` re-selects the serial
kernel on the same binary; the server logs `prefill_gdn_scan=` on its
residency line. Measured on the ledger: see "After P4" above.

### Step 5 — grouped routed GEMMs (−29 % on the M4 Pro, −7 % on the M1)

One dispatch per phase over every expert's rows in a tile. A wave planner
packs the tile's experts into 64-row-aligned slots of the staging block
(1,024 rows, a whole eight-expert tile at a 4,096-token chunk) and splits a
long expert at row-tile boundaries; `mpp_prefill_affine_grouped_f16` is the
per-expert kernel with the row origin and the weight, scale and bias pointers
indirected through a per-64-row-tile block table and the tile's expert
argument buffer, and a store guard at each block's real rows. The grouped
gather zeroes the padded rows and the grouped scatter skips them, so the 1–31-
pair experts that P3 sent back to the scalar microbatch path run inside the
GEMM. The block tables go inline with each encoder: a shared table rewritten
per wave would be read by the GPU after the CPU had overwritten it, because
the next tile is encoded while this one runs. `SHRIKE_PREFILL_ROUTED_GEMM=per-expert`
keeps the P3 path; `ShrikeBench routed_gemm` times the ornith tile through
both against the expert-shape ceilings. Measured on the ledger: see "After
P6" above — 67 % of the 128-row ceiling on both boxes, which is why the M1,
whose per-expert path was already at 60 %, gains a seventh of what the M4
Pro does.

### Follow-ons, not scheduled

- **Tile command-buffer batching — landed as a null result (P5, a7c8288 +
  bf469ad).** The boundary was never ≈ 1.3 ms of commit → wait → encode; it is
  the routed fetch's excess over the GPU tile, and batching removes the
  overlap that hides the rest. See "After P5" in the ledger. The knob stays
  as an A/B override; the default is one tile per buffer.
- **The routed fetch's latency (M4 Pro).** `fetchBindingForTile` costs
  ≈ 3.9 ms per tile at 0 % prefill hit rate against 0.5 ms of pread inside
  it. With one tile of overlap it is the critical path at 3.2 ms of GPU per
  tile (≈ 0.66 ms per boundary, 2.3 s of 32.9 at 12k) and the first tile of
  every layer pays it unhidden. Either find the ≈ 3.3 ms in the load
  operation's completion path — it is not the I/O — or run two fetches in
  flight. On the M1 the tile is 7.5 ms of GPU after P6 and the fetch is still
  hidden, so this is an M4 Pro lever; there P6 shrank the tile to 2.25 ms and
  the fetch is now the whole boundary (routed→routed 1.39 ms per boundary,
  4.5 s of 31.6 at 12k).
- **The first routed buffer of each layer costs ≈ 11 ms (M4 Pro) / ≈ 22 ms
  (M1) in the driver** — `prefill_shared_expert->prefill_routed_tile`
  `driver_ms` 0.45 s / 1.02 s at 3.7k over 40 layers, 2.65 s at M1 12k over
  120; the later tiles of a layer pay ≈ 0.01 ms. Worth ≈ 4 % of the M4
  Pro's 12k wall and ≈ 2 % of the M1's. Unexplained. Its shape — once per
  layer, on the first buffer that touches the expert slab after buffers
  that do not — suggests residency work proportional to the slab, which an
  `MTLResidencySet` on the queue would remove; that is a model, not a
  measurement.
- **The routed GEMM's last third.** After P6 both boxes run the routed tile
  at two thirds of the MPS ceiling for the 128-row shape. Inside
  `mpp_prefill_affine_grouped_f16` every K-group serializes dequantizing the
  32×64 weight tile into threadgroup memory, a barrier, then the matmul;
  double-buffering the weight tile would overlap the dequant with the
  previous group's matmul, and a 64-wide N tile would halve the A-tile
  re-reads on the `n = 2048` down GEMM. Both are unmeasured; the bench is the
  gate.
- **Per-layer host routing on the M1.** ≈ 15 ms per layer between the GDN or
  attention buffer and the shared expert (`host_ms` 1.38 + 0.48 s at 12k,
  ≈ 1.5 % of wall); 2.8 ms per layer on the M4 Pro. The router readback,
  pair building and grouping over 4,096 rows on the CPU.
- **Attention: stage a KV-head group's tile once.** P2's kernel reads each
  K/V tile eight times (once per query head); staging a dequantized 64-key
  tile in threadgroup memory for all eight heads, with Q streamed per head,
  would remove that re-read. Worth roughly the gap between 36 % and the
  ceiling's practical 50–60 %; measured on the ledger like P2.
- **The mini's SSD term** (0.8 ms/token per chunk) surfaces after step 2; the
  v10 P3 follow-on (batched miss loads, deeper queue depth) is the lever then.

## Numerics policy

Every step changes the order of floating-point additions, so outputs differ in
the low bits from the scalar kernels. Each step is qualified against the fp32
reference first; then `tools/golden-baseline.sh --check` is expected to
differ, and the baselines are recaptured on both boxes, one recapture per
step, as the deliberate numerics change the gate allows. Never recapture to
make an unexplained mismatch go away. The greedy digests before and after each
step are recorded in the plan.

## Out of scope

- ANE prefill ([ane-prefill.md](ane-prefill.md)) stays opt-in and untouched;
  its layer breakdown was the prior record and is superseded by the ledger here.
- Decode kernels, the KV cache layout, chunk size (4,096 stays), the `.gturbo`
  format, and the expert streamer.
- Other model shapes (MLA/Kimi, gpt-oss sinks, sliding windows) keep the scalar
  kernels behind the same gates that select them today.
- Speculative decoding's verify step. It is a width-2 forward on the decode
  kernels (the `pair` schedule), bandwidth-bound like decode; on the mini its
  cost is expert-miss I/O, not these kernels. Measured at the P4 build on the
  mini (rig prompt, `--mtp-model ornith15-mtp`, 5 runs): 6.6 tok/s against
  25.5 plain; per pass 17 ms proposal + 168 ms verify (156 of it the width-2
  backbone, 4× a 39 ms decode step) at 25.9 % acceptance, 1.26 tokens per
  pass. Break-even would need a pass under 49 ms at that acceptance, below
  the two-row union's bandwidth floor, so viability hinges first on the
  acceptance rate (26 % on a counting prompt is the thing to audit), then on
  the verify pass's miss I/O — a decode-chapter follow-on, not a prefill one.

## Risks

- Threadgroup memory at head-dim 256 caps the attention tile; if 40 % of the
  ceiling is not reachable with `matmul2d`, the fallback is hand-written
  `simdgroup_matrix` fragments (more code, same math). The spike decides.
- Settled: the step-0 log line observed the matrix kernel compiling on both
  boxes — M4 Pro and M1 both log `prefill_projection_path=affine-threadgroup-f16`.
- Per-expert GEMMs at short prompts fall below the efficient row count; the
  threshold keeps the scalar path there, so short prompts do not regress.
