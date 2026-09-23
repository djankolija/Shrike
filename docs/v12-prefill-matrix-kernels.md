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
column's remainder is the fresh server's first request). From P4 on the mini's
rows also come through a fresh server per prompt, and that remainder was
measured at the P10 build as `completed in` − (`prefill_s` + `decode_s`) from
the server log: a server's **first** request pays 1.93 s at 3.7k and 2.05 s at
12k (2.09 s with the prompt cache off) — the same at both lengths, so not the
tokeniser: first-encode pipeline specialisation and first-touch work — while
the **second** request on the same server pays 0.13 s at 12k with the cache off
and 0.53 s at 4.3k with it on. Every fresh-server wall row from P4 on therefore
carries ≈ 2 s the production server pays once per lifetime; the rows stay
comparable with each other, and against production the 2 s comes off. The M4
Pro's fresh-server rows carry the same shape of remainder, 1.62 s at 3.7k and
1.60 s at 12k. (Task-10 ledger files: `server-p10-mini-{2k,6k}.log`,
`server-l3-nocache-6k.log`, `server-outside-2k-2kb.log`,
`server-outside-nocache-2k-6k.log`, `server-p10-local-{2k,6k}.log`.)

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

Attention-core efficiency after P7: ≈ 37 % of the M4 Pro's measured ceiling
at 12k (2.8 TFLOPS on 12.4 TFLOP of scores and values), ≈ 30 % before; the
M1 ≈ 29 % of its 3.0 TFLOPS peak — measured at Step 15 as 0.87 TFLOPS, which
is 47 % of the 1.84 TFLOPS same-run gate/up ceiling the later bars use. The
remaining cost is per key tile, not per byte: the scores go
through threadgroup memory and back as fp16 weights, the R×256 fp32 output
accumulator is rescaled once per tile, and the tile loop is barriered — which
is why 256-key tiles on 16 rows beat P2's 64-key tiles on 32 rows and why
staging K/V once per KV-head group (the follow-on P7 set out to build) lost
on both boxes. The FlashAttention shape — Q and the probabilities resident in
registers, no score round-trip — needs MPP's single-simdgroup execution scope;
it was built and measured at Step 15 and is a null, 3× slower on the M1: the
K/V reuse it gives up costs more than the round-trip it removes.
`SHRIKE_ATTN_MATRIX_TILE` selects `g4k128d`, `r32s4`, `r64s8`, `f4k128`,
`f4k64` or `f8k128` on the same binary; `SHRIKE_PREFILL_ATTENTION=tiled` A/Bs
the scalar kernel.

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
  not the cache, is what is off. Parked. **Superseded: its premise was false (P15, :926-946) (noted 2026-09-23).**

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

**After P7** (commit 5ba2b89, 2026-09-02; attention on the matrix path as
2-query × 8-head groups against 256-key tiles read from the shadow, Q packed
group-major once per layer; `SHRIKE_ATTN_MATRIX_TILE=r32s4` keeps the P2
geometry):

| role | M4 Pro 3.7k | M4 Pro 12k | M4 Pro 25k | M1 3.7k | M1 12k |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prefill_attn_router` | **0.24** | **0.36** | **0.67** | **1.25** | **2.05** |
| `prefill_routed_tile` | 0.72 | 0.69 | 0.82 | 3.23 | 3.13 |
| `prefill_gdn_router` | 0.65 | 0.64 | 0.74 | 3.44 | 3.43 |
| `prefill_shared_expert` | 0.06 | 0.06 | 0.07 | 0.29 | 0.29 |
| **GPU busy** | 1.71 | 1.78 | 2.32 | 8.42 | 9.01 |
| gaps (span − busy) | 0.79 | 0.57 | 0.69 | 0.66 | 0.53 |
| **wall** | 2.95 | 2.48 | 3.07 | 9.90 | 9.70 |
| wall, seconds | 11.1 | 30.5 | 77.5 | 37.2 | 119.2 |

The attention role fell 21 % on the M4 Pro at 12k and 14 % on the M1, and not
for the reason the follow-on gave. The spike measured fourteen geometries on
both boxes; every one that staged a KV-head group's K/V tile through
threadgroup memory once for all eight heads — the modelled 2–4× traffic cut —
was slower than P2 (M4 Pro +26–80 %, M1 +5–22 %): the caches serve P2's
eightfold re-read, and the staging only adds a copy and barriers. The forms
that read the shadow as P2 does, with only the group-major Q and the tile
shape changed, order monotonically by keys per tile and inversely by rows:
32 × 64 (P2) 0.460, 32 × 128 0.394, 16 × 256 0.363, then 8 × 512 back up to
0.404 ms/token; four simdgroups beat eight at every shape once the
cooperative-tensor loops carry an unroll pragma (−35 % on the eight-simdgroup
body, nothing on P2's). The kernel is bound by its per-tile fixed work — the
score round-trip through threadgroup memory, the rescale of the R×256 fp32
accumulator, the barriers — not by K/V traffic. Ceiling share at 12k on the
M4 Pro: 30 % → 37 %; the step's 50 % bar is not reached. Wall on the M4 Pro
follows the GPU at 12k and 25k (31.6 → 30.5 s, 78.8 → 77.5 s) and hides in the
3.7k prompt's ±2 s swing (10.9 → 11.1 s); the M1 keeps it at 12k (123.4 →
119.2 s). Golden: short identical on both boxes; the M4 Pro's long profile
differs (a near-tie logit at the thinking block's second sentence flips back
to its pre-P3 wording) and was recaptured once; the M1's long profile is
byte-identical.

**After P9** (commit 560c888, 2026-09-03; the MPP GEMM's dequant issues one
vector load per thread and stages two quant groups per K tile;
`SHRIKE_MPP_WEIGHT_LOADS=byte SHRIKE_MPP_TILE_K=64` keeps the P8 kernel):

| role | M4 Pro 3.7k | M4 Pro 12k | M4 Pro 25k | M1 3.7k | M1 12k |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prefill_attn_router` | 0.25 | 0.39 | 0.54 | **0.96** | **1.75** |
| `prefill_routed_tile` | **0.59** | **0.53** | **0.50** | **1.92** | **1.87** |
| `prefill_gdn_router` | 0.65 | **0.60** | **0.54** | **2.38** | **2.37** |
| `prefill_shared_expert` | **0.05** | **0.05** | **0.04** | **0.17** | **0.17** |
| **GPU busy** | 1.59 | 1.60 | 1.65 | 5.64 | 6.26 |
| gaps (span − busy) | 0.90 | 0.71 | 0.85 | 0.72 | 0.56 |
| **wall** | 2.90 | 2.41 | 2.56 | 6.84 | 7.00 |
| wall, seconds | 10.9 | 29.6 | 64.7 | 25.7 | 86.0 |

Every role with an MPP projection moved on the M1: routed −40 %, GDN −31 %
(its first measured GEMM share), shared −41 %, attention −14 % (its
Q/K/V/O). The mini's 12k wall fell 118.9 → 86.0 s (−28 %) and its 3.7k wall
36.3 → 25.7 s; the mini now prefills at 7.0 ms/token, 1.1× the chapter's
6.3 target. On the M4 Pro the GEMM was already near its ceiling, so the roles
moved less and the fetch-bound loop turned part of the saving into
routed→routed gap (12k gaps 0.57 → 0.71); its 25k wall fell 77.5 → 64.7 s.
Golden differs on the long profile on both boxes — the 128-wide K tile
reorders the reduction (last-ulp differences on ≈ 0.15 % of elements, measured
by `ShrikeBench mpp_compare` on both boxes) — and was recaptured once per box
with that reason; the digests are in the plan's verdict.

**After P10** (commit 3710611, 2026-09-03; the MPP GEMM stages four quant
groups per K tile and picks the widest instantiated tile that divides `k`;
`SHRIKE_MPP_TILE_K=128` keeps the P9 kernel):

| role | M4 Pro 3.7k | M4 Pro 12k | M4 Pro 25k | M1 3.7k | M1 12k |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prefill_attn_router` | **0.21** | **0.33** | **0.52** | 0.95 | 1.75 |
| `prefill_routed_tile` | **0.47** | **0.44** | **0.46** | **1.79** | **1.74** |
| `prefill_gdn_router` | **0.54** | **0.52** | **0.52** | 2.36 | 2.34 |
| `prefill_shared_expert` | 0.04 | 0.04 | 0.04 | 0.17 | 0.17 |
| **GPU busy** | 1.32 | 1.35 | 1.56 | 5.47 | 6.10 |
| gaps (span − busy) | 1.10 | 0.80 | 0.91 | 0.76 | 0.58 |
| **wall** | 2.85 | 2.28 | 2.54 | 6.77 | 6.85 |
| wall, seconds | 10.7 | 28.0 | 64.0 | 25.4 | 84.2 |

On the M1 the routed role fell 7 % (its tile 4.93 → 4.48 ms on the bench,
78 % of the same-run ceiling) and nothing else moved: the GDN role −1 %,
attention flat — the 4,096-row dense projections do not respond to the wider
tile there. On the M4 Pro they do: a paired A/B on one binary at 12k (K128 →
K256, back to back) read routed 0.55 → 0.44, GDN 0.62 → 0.52, attention
0.38 → 0.33, busy 1.66 → 1.35 (−19 %), wall 31.7 → 28.0 s (−12 %); at 25k the
wall is fetch-bound and moved 64.7 → 64.0 s. (Corrected by Step 11 for the
dense rows — GDN, attention: the M4 Pro's dense shape reads 20.11 ms at K128
and 20.22 at K256 with the arms interleaved in one process, so those rows of
the paired A/B measured the box's state between two fresh servers, not the
tile; the routed row is the grouped kernel, which the tile bench measured on
its own.) The mini's 12k wall 86.0 →
84.2 s (6.85 ms/token, 1.09× the target); 3.7k 25.7 → 25.4 s. Golden: the
M4 Pro's long profile moved (recaptured once, the K256-vs-K128 reduction
order); the M1's held on both profiles.

**After P12** (commit 188d2b7, 2026-09-03; every layer's expert pool held in a
queue residency set and the shared expert committed before the router wait;
`SHRIKE_PREFILL_POOL_RESIDENCY=none SHRIKE_PREFILL_ROUTE_OVERLAP=off` keep the
P10 behaviour; no kernel moved, so the roles are the After P10 rows within
noise and only the gaps and the wall are listed; the M4 Pro's 25k row was not
taken):

| | M4 Pro 3.7k | M4 Pro 12k | M1 3.7k | M1 12k |
| --- | ---: | ---: | ---: | ---: |
| **GPU busy** | 1.26 | 1.35 | 5.42 | 6.10 |
| gaps (span − busy) | **1.05** | **0.76** | **0.49** | **0.27** |
| **wall** | 2.65 | 2.24 | 6.47 | 6.54 |
| wall, seconds | 10.0 | 27.5 | 24.3 | 80.4 |

On the M1 at 12k the `shared→routed` transition's `driver_ms` fell 2,664 →
58 ms and the two `->shared` transitions (1,388 + 465 ms of host routing)
left the gap list; each lever alone is worth ≈ 2 s and together 4.0 s
(84.4 → 80.4 s, 6.54 ms/token, 1.04× the chapter's 6.3 target). On the M4
Pro the driver cost was ≈ 4.7 ms per layer-chunk (570 → 10 ms at 12k) and the
routing 1.8 ms; its wall moved 28.0 → 27.5 s, the rest being the fetch-bound
`routed→routed` gap. Golden identical on both boxes, short and long.

**After P13** (commit 8692c3a, 2026-09-03): benches only, no production change;
the rows are the After P12 rows. See Step 11 for what they measured.

**After P14** (commit dfaa69e, 2026-09-03; the router block on the tiled
kernel, 12 tokens per threadgroup; `SHRIKE_PREFILL_ROUTER=block` keeps the P13
kernel; the routed and shared rows are unchanged within noise):

| role | M4 Pro 3.7k | M4 Pro 12k | M4 Pro 25k | M1 3.7k | M1 12k |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prefill_attn_router` | **0.17** | **0.29** | **0.49** | **0.76** | **1.57** |
| `prefill_routed_tile` | 0.46 | 0.44 | 0.46 | 1.79 | 1.74 |
| `prefill_gdn_router` | **0.41** | **0.40** | **0.40** | **1.79** | **1.78** |
| `prefill_shared_expert` | 0.04 | 0.04 | 0.04 | 0.17 | 0.17 |
| **GPU busy** | 1.11 | 1.20 | 1.41 | 4.67 | 5.34 |
| gaps (span − busy) | 1.05 | 0.76 | 0.86 | 0.49 | 0.27 |
| **wall** | 2.48 | 2.09 | 2.34 | 5.73 | **5.76** |
| wall, seconds | 9.3 | 25.7 | 59.0 | 21.5 | **70.8** |

On the M1 the same binary with the block kernel reads 80.41 s at 12k (gdn 2.34,
attention 1.75, busy 6.09); the tiled kernel takes 9.65 s off — the GDN role
−0.57 ms/token and attention −0.19, 77 ms per layer-chunk over 90 and 30
chunks against the bench's 73 (its fixture is on shared storage) — and the
mini prefills at **5.76 ms/token, under the
chapter's 6.3 target for the first time** (3.7k 24.3 → 21.5 s). The M4 Pro's
12k wall 27.6 → 25.7 s on the same A/B, 25k 64.0 → 59.0 s. Golden identical on
both boxes and both profiles, as the exact-order design requires.

**After P15** (commit dd78c27, 2026-09-03; the prefill expert sweep alternated
on odd chunks — `SHRIKE_PREFILL_SWEEP=fixed` keeps every chunk ascending; the
roles are unchanged, the gap rows fall; the M4 Pro 3.7k column is P14's — one
chunk, nothing to alternate):

| role | M4 Pro 3.7k | M4 Pro 12k | M4 Pro 25k | M1 3.7k | M1 12k |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prefill_attn_router` | 0.17 | 0.29 | 0.49 | 0.76 | 1.56 |
| `prefill_routed_tile` | 0.46 | 0.44 | 0.46 | 1.79 | 1.73 |
| `prefill_gdn_router` | 0.41 | 0.40 | 0.40 | 1.79 | 1.78 |
| `prefill_shared_expert` | 0.04 | 0.04 | 0.04 | 0.17 | 0.17 |
| **GPU busy** | 1.11 | 1.21 | 1.41 | 4.68 | 5.33 |
| gaps (span − busy) | 1.05 | **0.51** | **0.47** | 0.49 | **0.22** |
| **wall** | 2.48 | **1.85** | **1.94** | 5.74 | **5.70** |
| wall, seconds | 9.3 | **22.7** | **49.1** | 21.6 | **70.0** |

`expert_hits_prefill` 0 → 9,549 of 28,404 on the M1 at 12k; 0 → 28,909 of
65,776 on the M4 Pro at 25k. Golden identical on both boxes and both profiles.

**After P15b** (commit cbff561, 2026-09-03; the grouped routed GEMM's 32-row
tail tile, default on — `SHRIKE_PREFILL_TAIL_TILE=off` keeps every block on
64-row tiles; the M4 Pro 3.7k column is P14's, one chunk on a box that decides
nothing here):

| role | M4 Pro 3.7k | M4 Pro 12k | M4 Pro 25k | M1 3.7k | M1 12k |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prefill_attn_router` | 0.17 | 0.29 | 0.48 | 0.77 | 1.58 |
| `prefill_routed_tile` | 0.46 | 0.44 | 0.45 | **1.65** | **1.61** |
| `prefill_gdn_router` | 0.41 | 0.40 | 0.40 | 1.80 | 1.78 |
| `prefill_shared_expert` | 0.04 | 0.04 | 0.04 | 0.17 | 0.17 |
| **GPU busy** | 1.11 | 1.20 | 1.40 | **4.55** | **5.23** |
| gaps (span − busy) | 1.05 | 0.56 | 0.48 | 0.54 | 0.24 |
| **wall** | 2.48 | 1.86 | 1.94 | 5.67 | **5.62** |
| wall, seconds | 9.3 | 22.9 | 49.0 | 21.3 | **69.1** |

Same binary, tail off: M1 12k 70.81 s (routed 1.75, gaps 0.22), 3.7k 21.57 s.
Golden identical on both boxes and both profiles.

**After P11** (commit 1a3175e, 2026-09-03): a measured null (Step 15) — the
`g2k256d` default is untouched, the rows are the After P15b rows; the `attn`
bench puts the M1's attention core at 1.155 ms/token of the role's 1.58 (the
table above).

**After P16** (commit 1b13aed, 2026-09-03; the routed tile pipeline two tiles
deep — `SHRIKE_PREFILL_TILE_DEPTH=1` restores the one-tile bank. The rows were
measured on ccac897's binary with `SHRIKE_PREFILL_TILE_DEPTH=2`, the value the
flip in 1b13aed makes the default; golden on 1b13aed is identical on both
boxes. Every role is read from the same arm's stats line; the M4 Pro 3.7k
column is P14's):

| role | M4 Pro 3.7k | M4 Pro 12k | M4 Pro 25k | M1 3.7k | M1 12k |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prefill_attn_router` | 0.17 | 0.29 | 0.49 | 0.76 | 1.58 |
| `prefill_routed_tile` | 0.46 | 0.44 | 0.45 | 1.65 | 1.61 |
| `prefill_gdn_router` | 0.41 | 0.41 | 0.41 | 1.79 | 1.78 |
| `prefill_shared_expert` | 0.04 | 0.04 | 0.04 | 0.17 | 0.17 |
| **GPU busy** | 1.11 | 1.20 | 1.41 | 4.54 | 5.23 |
| gaps (span − busy) | 1.05 | 0.51 | 0.47 | **0.39** | **0.16** |
| **wall** | 2.48 | 1.84 | 1.94 | **5.50** | **5.57** |
| wall, seconds | 9.3 | 22.7 | 48.9 | **20.6** | **68.4** |

Same binary, depth 1: M1 12k 69.12 s (gaps 0.24), 3.7k 21.18 s (gaps 0.54);
M4 Pro 25k 49.44 s on a quiet box, 12k 23.82 s under a decaying load (that
pair's wall decides nothing; its host term moved −3 %). Golden identical on
both boxes and both profiles.

## Where the time goes

Every dense projection in prefill (attention Q/K/V/O, GDN in/out) already runs
through `mpp_prefill_affine_threadgroup_f16`, the one SIMD-group matrix kernel in
the codebase, which is why they cost nothing measurable. The three slow roles are
all scalar kernels:

| role | kernel | structure | roofline gap (M4 Pro) |
| --- | --- | --- | ---: |
| attention core | `attention_prefill_causal_tiled` | one threadgroup per (query, head), 256 threads over head-dim, a serial walk over every key with a two-barrier threadgroup reduction per key. The decode kernel's shape, run once per prompt token: no K/V reuse across queries, none across the 8 query heads that share a KV head. | 30× at 3.7k, 60× at 25k |
| routed experts | `prefill_grouped_routed_moe_batched_phase1` / `_down` | each thread computes one or two 2048-long scalar dot products; 32-pair microbatches; a command buffer per 8-expert tile | 7×; 2.6× after P6 (`mpp_prefill_affine_grouped_f16`, one dispatch per phase over the tile's experts: 0.69 ms/token at 12k against 0.27 at the 7.46 TFLOPS ceiling); P8's bench puts the tile at 67 % of the M4 Pro's same-run ceiling and 42 % of the M1's, the M1's remainder split 7 % unpack / 20 % weight loads / 31 % staged structure (Step 7); after P9 (vector loads + 128-wide K) 71 % of the M1's ceiling and 99 % of the M4 Pro's (Step 8); after P10 (256-wide K) 78 % of the M1's (Step 9); after P15 the M1's tile carries +25.8 % padded rows (Step 13; the 32-row tail tile of Step 14 recovers 0.14 ms/token of it, to 1.61) and the expert sweep alternates per chunk |
| shared expert | `PrefillSharedExpert.encodeBlock` | a `for row in 0..<queryCount` loop over the decode runtime: 4–6 M=1 GEMV dispatches per token | 25× |
| GDN | `gdn_delta_step_prefill` | the delta-rule scan is a serial loop over the chunk inside one dispatch of 32×32 threadgroups; projections, conv and norms are fine | 3.6× (scan ≈ 0.6 of the 1.0 ms); after P12 the mini's 319 ms per layer-chunk is scan 56 + projections 166 (85–99 % of the MPS ceiling) + pre-scan chain 9 + router 83 → 10 after P14 (Steps 11–12) |

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

### Step 6 — attention tile shape on the matrix path (−21 % on the M4 Pro, −14 % on the M1)

Planned as "stage a KV-head group's K/V tile once for all eight query heads";
measured as something else. The spike built the staged kernel four ways
(K then V through one 16 KB threadgroup slab, 32 to 128 rows of a KV-head
group, 32- or 64-key tiles) and every one of them was slower than P2 on both
boxes — the M4 Pro by 26–80 %, the M1 by 5–22 %. The eightfold re-read the
follow-on targeted is served by the caches; copying the tile through
threadgroup memory only adds barriers and a copy. What the same spike found
instead, once the K/V operands went back to the device shadow as P2 reads
them, is that the kernel is bound by its per-tile fixed work — the score
round-trip through threadgroup memory, the rescale of the R×256 fp32 output
accumulator, the barriers and the `run` set-up — and that fewer query rows
with more keys per tile amortise it monotonically: 32 × 64 keys (P2) 0.46
ms/token at 12k on the M4 Pro, 32 × 128 0.39, 16 × 256 0.36; 8 × 512 turns
back up (0.40), and eight simdgroups lose to four at every shape once the
cooperative-tensor loops carry an unroll pragma (worth −35 % on the
eight-simdgroup body, nothing on P2's four). `attention_prefill_causal_matrix_g2k256d`
landed as the default: one threadgroup owns 2 query positions × the 8 query
heads of one KV head (16 matmul rows) against 256-key tiles, the Q block
packed group-major once per layer by `attention_prefill_q_group_pack`
(33.5 MB of traffic at a 4,096-token chunk, a chunk-sized buffer that never
grows with context), scores fp32 in threadgroup memory, the softmax weights
fp16 in a second region with the row sums taken from the rounded weights.
`SHRIKE_ATTN_MATRIX_TILE` still selects `r32s4`, `r64s8` or the runner-up
`g4k128d` (tied with the winner on the M1). MPP's register-resident left
operand — the FlashAttention shape that would drop the score round-trip
entirely — is only allowed under a single-simdgroup execution scope, so it
needs per-simdgroup matmuls; that is the follow-on below. Ceiling share at
12k on the M4 Pro: 30 % → 37 %; the task's 50 % bar was not reached.

### Step 7 — the MPP GEMM core: a measured null, with the mini's cost ledger

Two knobs on the K-group loop both kernels share, each an env-var A/B on the
same binary (`SHRIKE_MPP_DEQUANT_BUFFERS=1|2`, `SHRIKE_MPP_TILE_N=32|64`;
the bodies are now one `mpp_prefill_affine_body<TILE_N, BUFFERS>` template
with `n32b1` under the bare kernel names, bit-identical to what P6 shipped —
every variant is, the tests assert it). `ShrikeBench routed_gemm 20`, grouped
ms per ornith tile:

| arm | M4 Pro | M1 |
| --- | ---: | ---: |
| n32b1 (P6) | 1.67 | 8.25 |
| n32b2, double-buffered dequant | 1.68 | 8.26 |
| n64b1, 64-wide N tile | 2.32 | 9.69 |
| n64b2 | 2.31 | 10.21 |

Neither pays. Double buffering only hides latency, and the dequant is not a
latency problem: the same 128 threads do the dequant and the matmul, so it is
throughput work. The 64-wide tile halves the A re-reads but doubles the fp32
accumulator footprint, and the occupancy that costs outweighs the traffic it
saves on both GPUs. A bench-only probe that replaced the dequant with a single
byte load per element, then with a constant, gives the ledger for the mini's
8.25 ms tile: unpack arithmetic 0.57 ms (7 %), the int4 weight byte loads
1.65 ms (20 %), the staged structure itself — threadgroup tile writes, one
barrier and one 64×32×64 `matmul2d` per K group — 2.53 ms (31 %) over the
3.50 ms a plain GEMM at the same-run ceiling would take (42 %). The same
structure costs 9 % on the M4 Pro, which is why the two boxes' shares differ
(the M4 Pro's kernel runs at 67 % of its ceiling, the M1's at 42 %; P6's 68 %
for the M1 was measured against a depressed ceiling — the mini's ceiling bench
read 1.15 TFLOPS that run and 1.84 today with the grouped tile unchanged). The
two levers this leaves are in the follow-ons: vectorised weight loads and a
128-wide K tile.

### Step 8 — vectorised int4 weight loads and a 128-wide K tile (−40 % on the M1's routed role)

The two levers P8's probe named: the first bit-identical to the kernel P6
shipped, the second a reduction-order change inside the 2e-2 bar.
The dequant of the staged weight tile issued one byte load, a scale and a bias
read and a fused multiply-add per element; it now issues one vector load per
thread — a `uint2` or `uint4` carrying 16 or 32 consecutive values of one row,
never straddling a row or a quant group — reads the (scale, bias) pair once per
chunk and stores `half4`s. The byte body stays as the fallback for a weight
base that is not 16-byte aligned: the `.gturbo` offset spaces are unpadded
running cursors, so alignment is a per-dispatch fact the host passes as a
uniform, not a pipeline property. The K tile gains a template axis:
`n32k128b1` stages two quant groups per tile (32 × 128 halves, 8 KB), halving
the barriers and the `matmul2d` runs per row; an instance carrying it also
builds the 64-wide pair and picks per dispatch on `k % 128`, which keeps
gpt-oss's K = 2880 on the matrix path. The static admission unit stays 64.
`ShrikeBench routed_gemm 20`, grouped ms per ornith tile:

| arm | M4 Pro | M1 |
| --- | ---: | ---: |
| byte loads, K 64 (P8's default) | 1.70 | 9.37 (8.25 in P8's run) |
| vector loads, K 64 | 1.22 | 6.19 |
| byte loads, K 128 | 1.70 | 8.83 |
| **vector loads, K 128 — the default** | **1.13** | **4.92** |
| same-run gate/up ceiling, TFLOPS | 5.4–5.8 | 1.84 |

On the M1 the vector path takes a third off the tile — far more than the
probe's 1.65 ms load term, because the per-element loop, its address
arithmetic and the scale/bias re-reads go with the loads — and the wider K
tile, a null on its own, takes a further fifth once the dequant no longer
dominates: 4.92 ms is 1.31 TFLOPS, 71 % of the mini's same-run ceiling, from
42 %. The M4 Pro lands at 99 % of its ceiling. Measured on the ledger: see
"After P9" above. The vector loads are bit-identical (the tests assert it on
regular and full-mantissa inputs); the 128-wide run is not — MPP reduces K in
a different order than two 64-wide runs summed in fp32, last-ulp differences
on ≈ 0.15 % of elements (`ShrikeBench mpp_compare`, both boxes) — so golden
moved on the long profile on both boxes and was recaptured once per box.

### Step 9 — a 256-wide K tile (−9 % on the M1's routed tile; the dense projections did not follow)

The structure half of what P9 left: `n32k256b1` stages four quant groups per
tile (32 × 256 halves, 16 KB, half the threadgroup budget), halving the
barriers, the `run` set-ups and the fp32 accumulates per row once more; the
vector loader gains the 64-byte chunk an 8-bit 256-wide tile needs (four
`uint4`), and the per-dispatch choice becomes a ladder — the widest
instantiated tile that divides `k`: 256, else 128 (Kimi's 128-wide low-rank
legs), else 64 (gpt-oss's 2880). 256 is the last width this chunk mapping
supports: at E = 64 a chunk is one whole quant group and the
`kW4A8GroupSize % E` assert is exact. `ShrikeBench routed_gemm 20`, grouped ms
per ornith tile at staging 1024 / 2048, vector loads:

| arm | M4 Pro | M1 |
| --- | ---: | ---: |
| K64 | 1.23 / 1.22 | 6.23 / 5.90 |
| K128 (P9's default) | 1.13 / 1.13 | 4.93 / 4.83 |
| **K256 — the default** | **1.09 / 1.10** | **4.48 / 4.49** |
| same-run gate/up ceiling, TFLOPS | 5.4–5.8 | 1.83 |

On the M1 the tile is 1.44 TFLOPS, 78 % of its same-run ceiling (71 % after
P9), and the routed role followed (−7 % at 12k). The dense projections did
not: the GDN role moved 1 % and attention 0.3 %, against a modelled 65 % GDN
GEMM share inferred from P9's −31 % — P9's gain there was the load lever, and
the structure lever does not carry to the 4,096-row dense shape. Step 11
benched it: the dense projections already run at 85–99 % of the MPS ceiling
on the M1, so there was nothing for the wider tile to take (on the M4 Pro the
same bench reads K128 and K256 within 0.5 %). Not bit-identical to the 128-wide tile: `ShrikeBench
mpp_compare` puts the K256-vs-K128 difference at 456 / 489 of 262,144
elements at 4 / 8 bits (max abs 0.000488 / 0.0078, identical counts on both
boxes); golden moved on the M4 Pro's long profile and was recaptured once;
the mini's held on both profiles. Measured on the ledger: see "After P10"
above.

### Step 10 — the prefill gap levers (−4.0 s of the M1's 12k wall, no kernel touched)

Two of the four gaps P5's instrumentation named were host and driver work,
not kernels. The first routed command buffer of every layer paid ≈ 22 ms in
the driver on the M1 (≈ 4.7 ms on the M4 Pro): a zero-code spike showed the
cost is residency of the buffer it references — halving the expert slab
halved it, a per-slot layout removed 91 % of it — so an `MTLResidencySet`
attached to the queue now holds every layer's pool slab from the moment its
streamer opens (`ExpertPoolResidency`), and the cost is 58 ms per 12k prompt
instead of 2,664. The ≈ 16 ms of host routing per layer-chunk (the router
readback, the pair build and the grouping) sat between the router buffer and
the shared expert, which depends on none of it; the shared expert's command
buffer is now committed before the router wait, so its ≈ 17 ms of GPU covers
the routing and the two `->prefill_shared_expert` transitions disappear from
the gap list. Each lever is worth ≈ 2 s at 12k on the mini and they add:
84.4 → 82.2 s (overlap alone) / 82.5 s (residency alone) / 80.4 s (both), GPU
busy unchanged, golden identical on both boxes. The counting sort the task
held in reserve for the routing was not needed: nothing of the routing is
left exposed. What remains of the gaps at 12k on the M1 (0.27 ms/token): the
per-tile host work in `routed→routed` (≈ 0.10) and ≈ 6.9 ms per layer-chunk
of metadata build plus the first tile's unhidden fetch in `shared→routed`
(≈ 0.07). The residency set wires the pool pages only while the queue
executes (`vm_stat` at idle: 2.2 GB wired, the pools as active pages);
`memory_pressure -Q` reads ≈ 25 % free right after a 12k prefill against
≈ 74 % without the set, and 83 % a minute later; `SHRIKE_PREFILL_POOL_RESIDENCY=none`
is the one-env rollback. Measured on the ledger: "After P12" above.

### Step 11 — measuring the GDN pre-scan chain, the dense GEMM and the router (two nulls, and the chapter's largest unclaimed cost)

Three benches, no production change. `ShrikeBench gdn_pre` times the GDN
pre-scan chain kernel by kernel at the 4,096-row shape; `dense_gemm` runs
`MPPPrefillInt4QMM.encode` at the four dense projection shapes against the
same-run MPS ceilings; `router_block` times `prefill_router_block` against an
MPS GEMM of its shape. On the mini (M4 Pro in parentheses), per layer-chunk:

| kernel | mini ms | GB/s or share | M4 Pro ms |
| --- | ---: | ---: | ---: |
| `gdn_conv_mix_prefill` | 4.40 | 30.5 GB/s | 1.13 |
| `gdn_conv_tail_update` | 0.005 | 20.3 | 0.003 |
| `gdn_qk_norm` | 1.19 | 56.6 | 0.39 |
| `gdn_gated_norm` | 1.65 | 60.9 | 0.43 |
| 2 × `prefill_rmsnorm_bf16w_block` | 1.12 | 59.7 | 0.25 |
| `residual_add_fp16` | 0.84 | 59.9 | 0.41 |
| **pre-scan chain** | **9.21** | 45.6 GB/s; floor 7.0 ms at 60 GB/s | 2.61 |
| dense (4096, 2048, 8192), `n32k256b1`, vector loads | 83.2 | **0.99** of the 82.1 ms MPS ceiling | 20.2 (0.88) |
| dense (4096, 2048, 4096), same arm | 41.5 | 0.85 of 35.5 | 10.1 |
| dense (4096, 4096, 2048), same arm | 40.7 | 0.88 of 35.6 | 10.3 |
| **`prefill_router_block`** | **83.4** | **0.029** of the 2.45 ms ceiling | 18.1 (0.057) |

The audit's model for the chain (50–91 ms by subtraction) was wrong by an
order of magnitude: the chain moves 419 MB at 46 GB/s, within 2.3 ms of the
box's streaming floor, so fusing it is worth ≤ 0.02 ms per prompt token. The
dense projections already run at 85–99 % of the MPS ceiling on the M1 — which
is why P10's wider K tile moved the GDN role 1 % (the dense rows of the M4
Pro's paired A/B in "After P10" were that box's state between two fresh
servers: the bench's interleaved arms read K128 20.11 ms and K256 20.22 ms;
the routed row stands on the tile bench) —
and a taller row tile has at most half of ≈ 12 ms per layer-chunk to take,
≈ 0.05 ms per token. Neither arm clears the task's 0.15 ms/token threshold,
so neither proceeds. What the benches did find is the term nobody had
measured: the router block, one threadgroup per token with 256 threads each
walking the 2,048-long row serially and re-reading the 512 KB weight, then
one thread doing the top-8 while the others wait — 83.4 ms per layer-chunk at
one thirty-fifth of its ceiling, in every one of the 120 layer-chunks: 10.0 s
of the mini's 80.4 s 12k wall. The GDN role's cost table closes with it
(scan 56 + projections 166 + chain 9 + router 83 = 315 of 319 ms), and the
router became Task 14 (Step 12).

### Step 12 — the router block on an operand-reusing kernel (−12 % of the M1's 12k wall, bit-identical)

`prefill_router_block` was one threadgroup per token: 256 threads each walking a
2,048-long row with a byte extraction and three loads per element, the 512 KB
weight re-read by every one of the 4,096 threadgroups, the top-8 on one thread.
`prefill_router_block_tiled` gives a threadgroup a block of 12 tokens against
every expert: thread `e` keeps expert `e`'s row and 12 accumulators, so one
weight read serves 12 tokens; a group's 64 weight bytes arrive as `uint4`s and
are unpacked from registers (the byte path stays for an unaligned base); the
pre-scaled activations `x ⊙ e` are staged token-minor in threadgroup memory,
their per-token sums taken in k order by the token threads, and loaded as
`float4`; each token's top-8 runs on its own thread over the experts in
ascending order. The arithmetic per (token, expert, element) and its order are
the block kernel's — the same `q`, the same `float(x) · float(e)`, the same two
fmas, the same tie rule — so logits, indices and route weights are
**bit-identical** (asserted on full-mantissa inputs at 8 and 4 bits, softmax and
sigmoid, partial blocks, unaligned bases, the production shape) and golden is
identical on both boxes and both profiles. Two things the ladder settled on the
way: Metal does not contract the block kernel's `sum_x += xv` (the sum of the
staged, rounded products is bit-identical), and the M1's limit was the
per-thread byte walk of the weight row — 32 rows 2 KB apart per SIMD group,
one cache line per element — which vector activations, parallel staging and a
smaller threadgroup-memory footprint could not touch and the vector weight
loads removed. `ShrikeBench router_block 20`, both kinds in one process:

| box | block | tiled (12 tokens) | speedup | share of the same-run ceiling |
| --- | ---: | ---: | ---: | ---: |
| M1 | 83.4 ms | **10.0 ms** | 8.3× | 2.9 % → 24 % |
| M4 Pro | 18.1 ms | 2.15 ms | 8.4× | 5.7 % → ≈ 50 % |

Token block on the mini: 4 → 11.5, 8 → 10.4, **12 → 10.0**, 16 → 14.9,
24 → 12.6 ms; `SHRIKE_PREFILL_ROUTER_TOKENS` sweeps it and
`SHRIKE_PREFILL_ROUTER=block` keeps the old kernel on the same binary. Measured
on the ledger: "After P14" above — the cut lands in full in both router roles.

### Step 13 — the padding tax measured, and the prefill expert sweep alternated (−1.4 % of the M1's 12k wall, −15 % of the M4 Pro's; completions byte-identical on both arms, golden identical)

Two levers the audit modelled and nothing had measured.

**The padding.** The grouped routed GEMM rounds every expert block up to a
whole 64-row tile, and the wave close at the 1,024-row staging boundary adds a
partial tile more. One probe print in an instrumented build (reverted) counted
the rows on the M1: per 12k prompt **3,931,200 real rows against 4,946,880
padded (+25.8 %)**, 254 blocks and 55 waves per layer-chunk with 17.5 wave
splits (3.7k: +29.6 %). A 32-row tail tile would hold the same rows in
4,410,976 ((P64 − P32) / P64 = 10.8 %), a 16-row tail in 4,156,704 (16.0 %).
The bench control — `ShrikeBench routed_gemm`, eight experts at 128 / 97 / 65
rows each, the same 16 tiles and 1,024 padded rows with 1,024 / 776 / 520 real
— reads 4.46 / 4.42 / 4.33 ms on the M1 and 1.086 / 1.074 / 1.066 on the M4
Pro: **a padded row costs a real row within 3 %**, so the tax is the padded
fraction, ≈ 0.36 ms/token of the M1's 1.73 routed row at 12k. A 32-row tail
tile recovers 2 × 10.8 % × (1 − α) of the GEMM's share, α the cost of a 32-row
tile relative to a 64-row one — 0.10–0.16 ms/token at α = 0.70–0.55, the
kernel work of Task 15b. The two earlier estimates of this quantity (+11–16 %
by subtraction from the ledger, +26 % by the audit's uniform-remainder model)
are settled: the audit was right.

| per 12k prompt on the M1 | rows | over the real rows |
| --- | ---: | ---: |
| real (`Σ pairCount`) | 3,931,200 | — |
| padded to 64-row tiles (`P64`, today) | 4,946,880 | +25.8 % |
| with a 32-row tail (`P32`) | 4,410,976 | +12.2 % |
| with a 16-row tail (`P16`) | 4,156,704 | +5.7 % |

**The sweep.** Every chunk sorted its (token, expert) pairs by ascending
physical offset, so each layer's 128-slot expert cache — 112 evictable under
the tile scheduler's 16 held — swept its ≈ 237 experts in the same direction
every chunk: the sequential-scan pathology under the aging-LFU's LRU tiebreak,
and `expert_hits_prefill` was **0** at every size on both boxes.
`SHRIKE_PREFILL_SWEEP=alternate` (the default until v13 T0 made `carry` the
default; `=fixed` is the A/B) reverses
the expert key on odd chunks, parity from `startPosition / 4096`, and only the
key: tokens and ranks stay ascending within a group, each expert block presents
the same rows in the same order to the same GEMM, and `routePartials` is
written per (token, rank) slot, so tile order cannot change a value —
completions byte-identical between arms at 3.7k and 12k, golden identical on
both boxes and both profiles. Measured on one binary, a fresh server per prompt
(the 3.7k rows are one chunk: nothing to alternate, the control):

| box, prompt | hits / misses, fixed → alternate | `routed→routed` | `shared→routed` | gaps | wall |
| --- | --- | ---: | ---: | ---: | ---: |
| M1 12k | 0 / 28,404 → **9,549** / 18,855 | 1,179 → 887 ms | 891 → 476 ms | 0.273 → 0.217 | 71.34 → **70.33 s** (−1.4 %) |
| M1 3.7k | 0 / 9,568 → 0 / 9,568 | 490 → 486 | 324 → 329 | 0.492 → 0.490 | 21.49 → 21.49 s |
| M4 Pro 12k | 0 / 28,401 → 9,558 / 18,843 | 8,501 → 5,306 | 580 → 242 | 0.814 → 0.511 | 26.61 → **22.71 s** (−14.6 %) |
| M4 Pro 25k | 0 / 65,776 → 28,909 / 36,867 | 19,488 → 10,731 | 1,311 → 300 | 0.858 → 0.472 | 59.00 → **49.10 s** (−16.8 %) |

The M1 hides most of each fetch behind its 5.9 ms tile, so the box that decides
moved 1.4 %; the M4 Pro's 1.5 ms tile hid nothing and the check box moved 15 %.
The decode counters are unchanged (the last chunk is even, ascending in both
arms). At the default the M1 prefills 12k in 69.97 s = **5.70 ms/token**.

### Step 14 — a 32-row tail tile for the grouped routed GEMM (−2.5 % of the M1's 12k wall, bit-identical)

Step 13 measured the padding at +25.8 % of the real rows and showed a padded
row costs a real row; this is the padding's recoverable half. `TILE_M` became
the affine body's first template parameter and the grouped kernel gained a
32-row instantiation for the default variant. The wave planner packs
body-first: every block's full 64-row tiles, then each block's remainder of
≤ 32 rows in a 32-row tail region after the body — a larger remainder stays on
a padded 64-row tile, since two 32-row tiles cost 2α against one. Each GEMM is
two dispatches on one encoder, the 64-row kernel over the body and the 32-row
one over the tail, either skipped when its region is empty; the gather and
scatter walk the wave at 32-row granularity. α — a 32-row tile's cost relative
to a 64-row one — measured **0.545** on the M1 (0.53 on the M4 Pro): the
dequant of the weight tile is per threadgroup and does not shrink with M, so a
half-height tile costs a little over half. `TILE_M` changes which rows share a
threadgroup, not the K loop or the `accumulator += groupProduct` fold; the
intra-tile reduction inside `matmul2d` is asserted, not derived — the tail path
is **bit-identical** on full-mantissa inputs across split and mixed waves, and
golden identical on both boxes and both profiles.

| `ShrikeBench routed_gemm`, tail / off | 128 rows per expert | 97 (a 33-row remainder) | 65 (a 1-row tail per block) |
| --- | ---: | ---: | ---: |
| M1 | 1.004 | 1.001 | **0.816** (modelled (8 + 8α) / 16 = 0.77) |
| M4 Pro | 1.010 | 0.999 | 0.871 |

On the M1 at 12k, one binary, a fresh server per prompt: routed 1.751 →
**1.613** ms/token (−0.139; the model at α = 0.545 said −0.155), GPU busy
5.373 → 5.228, gaps 0.215 → 0.244, wall 70.81 → **69.07 s (−2.5 %) = 5.62
ms/token**; 3.7k 21.57 → 21.29 s. The gap growth is not host encoding — three
encode variants (separate encoders, one encoder, reused bindings) read the same
`routed→routed` host term, 937–960 ms — it is the expert fetch surfacing: the
routed tile shrank from 5.96 to 5.50 ms of GPU while a tile's eight experts are
≈ 14.2 MB, ≈ 5.1 ms at the M1's 2.8 GB/s (the v10 measurement; less on
average with P15's 34 % hits, but per tile), so the depth-2 fetch pipeline no
longer hides all of it. A quarter of the GEMM cut resurfaces as fetch wait; the
M1's routed tile now sits on its SSD floor, and the next routed lever is fetch
overlap, not arithmetic. The M4 Pro, fetch-bound since P15, reads flat on the same binary: 12k 22.58 → 22.89 s, 25k 49.31 → 48.95 s (routed −3 to −5 %, the `routed→routed` gap +0.04 and +0.01 ms/token) — the GEMM cut lands in its fetch wait, as the mechanism predicts.

### Step 15 — the FlashAttention-shape attention body: a measured null (commit 1a3175e)

The follow-on that P7 left — keep Q and the probabilities in registers, one
simdgroup per query position — was built and measured. The body is exactly the
shape the headers allow: input cooperative tensors need `execution_simdgroup`,
so one simdgroup owns one query's eight heads (M = 8); Q loads once into a
left-input cooperative tensor; QKᵀ lands in a destination cooperative tensor;
the mask, the running max, the `exp`, the row sums and the O rescale run
element-wise in registers; the probabilities relay into PV's left input; no
threadgroup memory, no barrier. It is numerically right — nine fp16 reference
cases and fifteen quantized-cache cases at 2e-2 across `f4k128`, `f4k64` and
`f8k128` — and every width built a pipeline on both boxes, so registers were
not the limit. It is also slower on both boxes by a wide margin, and the
`ShrikeBench attn` mode this step adds (the isolated attention instrument P7
and this task both lacked) says where. Per 12k prompt, dequant and q-group
pack included:

| tile | M1 ms | M1 TFLOPS | M4 Pro ms | M4 Pro TFLOPS |
| --- | ---: | ---: | ---: | ---: |
| `g2k256d` (shipped) | 1,419–1,423 | 0.87 | 222 | 5.56 |
| `g4k128d` | 1,472 | 0.84 | 263 | 4.70 |
| `f4k64` | 3,911 | 0.32 | 293–296 | 4.2 |
| `f4k128` | 4,839 (4,783 on the pre-ablation build) | 0.26 | 420–425 | 2.9 |
| `f8k128` | 5,072 | 0.24 | 419–424 | 2.9 |

Ablations of `f4k128` (temporary kernels) price the pieces: on the M1, QKᵀ
alone under this shape costs 3,236 ms — **2.3× the shipped kernel's whole QK +
softmax + PV** — and along the M1's QK → softmax → PV path the softmax adds
≈ 0.7 s and PV with its relayout ≈ 0.9 s (the other path splits them the other
way, and the M4 Pro's arms invert — QK-only 292, QK + softmax 270, QK + PV 446,
full 425 against 222 — the ablation kernels dead-code-eliminate differently,
so only the QK-only arm compares across boxes). Replacing the runtime-indexed
per-row arrays with select chains changed nothing. The mechanism is the
operand reuse the shape gives up: a single-simdgroup matmul with a cooperative
left input at M = 8 reads each K/V tile for eight rows where the group kernel's
four cooperating simdgroups share a 16-row block, and the M = 8 op runs at a
fraction of the M = 16 op's rate on both GPUs. The shipped kernel with its
dequant and q-group pack — 1.155 ms/token on the M1 — is consistent with the
fit's core of 1.14–1.16 (the two included flat items are under 1 % of it,
modelled), so the flat/core split of the role stands; its 0.87 TFLOPS is 47 %
of the 1.84 TFLOPS same-run gate/up ceiling — the brief's modelled 48 %
confirmed, and the ≥ 55 % bar was never in reach without the fixed term going.

Two SDK facts found on the way (MacOSX26.5.sdk): a cooperative tensor cannot be
copy-assigned (its `operator=` hands a const source to MPP's non-const
`copy_assign`), so PV's left operand is constructed per tile; and **a
single-simdgroup `matmul2d` with a cooperative left input writes only the
first 128 columns of a 256-wide destination** (a one-hot probability row
against V[key][d] = d/256 came back as d/256 below column 128 and 0 above), so
PV runs as two 128-column halves and a 256-key QKᵀ tile is unusable — the
planned `f4k256` is not in the ladder.

Landed: the tests, the three variants (selectable by
`SHRIKE_ATTN_MATRIX_TILE=f4k128|f4k64|f8k128` on the same binary), the bench
mode. `g2k256d` stays the default; golden identical on both boxes and both
profiles, no ledger row moves.

### Step 16 — the routed tile pipeline's depth: a knob and a sweep (−1.1 % of the M1's 12k wall, −2.6 % at 3.7k; scheduling only, golden identical; commit 1b13aed)

The task was drafted as "fetch-overlap depth" and the draft's first finding
inverted it: the runner awaits each tile's fetch in the expression that issues
it (`PrefillStreamedTileBinding.fetchBindingForTile` →
`model.fetchRoutedExperts(plan:)` = `beginFetchRoutedExperts(plan:).completion()`),
so at most one tile's fetch is ever in flight. The drive sees queue depth ≈ 4
inside a tile (`ParallelExpertReader(threads: 4)`) and 1 across tiles, and
`PrefillRoutedTileSchedulerConfig.maxPendingDepth` changes neither: it sets how
many committed tiles of GPU work are banked against the host's next
plan → fetch → encode → commit. The knob is therefore `SHRIKE_PREFILL_TILE_DEPTH`
(beside P5's `TILE_BATCH`, the width), clamped 1…8, printed as `depth=` in the
projection-path line's tile field; the type and its slot budget
((D + 1) × width × experts + hits ≤ slots) are unchanged, and `fitting` never
narrows the depth.

What the boundary costs was measured from counters already on the stats line
(`io_fetch_ms` over the routed role's count; the P15b logs): fetch wall per tile
F̄ against GPU per tile G — M1 12k **3.20 / 5.50 ms (0.58)**, M1 3.7k **4.86 /
5.10 (0.95)**, M4 Pro 12k **2.67 / 1.49 (1.80)**; the marginal read 0.630 ms per
expert (2.81 GB/s at four threads, intercept ≈ 0) and the non-read host cost per
tile ≈ 0.72 ms (an upper bound). At 12k on the M1 the mean is hidden and the
exposed 0.27 ms per boundary (host-late 942 ms over 3,485) is per-tile
variance — the one regime a deeper bank can smooth. The sweep, one binary,
fresh server and one send per arm:

| M1, depth | 12k wall | host-late | banked / boundary | hits | 3.7k wall | host-late |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (control) | 69.12 s | 942 ms | 2.43 ms | 9,549 | 21.18 s | 585 ms |
| **2** | **68.38 s** | **94 ms** | 6.87 ms | 9,546 | **20.64 s** | **116 ms** |
| 3 | 68.27 s | 63 ms | 11.5 ms | 9,542 | 20.59 s | 107 ms |
| 4 | 68.46 s | 63 ms | 15.8 ms | 9,537 | — | — |

The 12k row is the model's prediction: one banked tile takes ≈ 90 % of the
host-late term, the second ≈ 3 % more, the third nothing, while the banked
queue grows linearly — the residual (≈ 63 ms host + ≈ 190 ms driver) is
turnaround, not fetch. The 3.7k row falsifies the model's null: with every tile
a miss (hits 0) the host's per-tile time F̄ + c ≈ 5.58 ms exceeds the GPU's
5.10 on the mean, which a steady-state pipeline cannot hide — but a layer-chunk
is ≈ 29 tiles and its boundary resets the pipeline, so the accumulated deficit
per layer-chunk (≈ 14 ms) is of the bank's order and a bank absorbs it up to
its size; the ≈ 107 ms left at depth ≥ 2 is per-layer-chunk pipeline fill.
The M4 Pro, mean-bound at 1.80, gains one bank's worth per layer-chunk: 12k
host −3 %, 25k host −6 % (49.44 → 48.91 s on a quiet box, routed GPU identical).
Depth 2 is the default by the rule (≥ 0.5 % on the M1's 12k; depth 3 a further
−0.16 % there, within the 0.2 % tie band the rule is scoped to; at 3.7k a
further −0.25 %, outside the band and judged noise-level against eight more
held slots) — 24 of the M1's 128 slots held instead of 16,
hits within 0.13 % across the sweep, memory unmoved; `SHRIKE_PREFILL_TILE_DEPTH=1`
is the A/B. `SHRIKE_PREFILL_TILE_BATCH` stays at 1 and its widths > 1 were last
validated end to end at P5 (M4 Pro, before the alternating sweep, the tail tile and
depth 2); at depth 2 a width of 4 would hold 96 of the mini's 128 slots — an A/B
knob only, unvalidated in the shipped configuration. Numerics cannot move (commit order and buffer contents are
unchanged; the shared staging scratch relies on the queue's serial execution
exactly as depth 1 did) and golden is identical on both boxes and both profiles.

### Step 17 — the speculative-decode economics audit: a measured verdict, no code (Task 17)

Measured on the mini on the P16 build, one server lifetime per arm, the MTP arm's
own runner / kernel / gap counters kept this time. Three shapes, plain against
`--mtp-model ornith15-mtp`. Conventions: "tok/s" is the server's end-to-end
`decode_tok_s` on both arms; "plain body" is the decode step's `body_ms` (the
per-token step, ≈ 6 ms under the end-to-end per-token time); "MTP pass" is
`decode_s / passes`, the whole pass; the digest is sha256 of `message.content`
only (it does not cover tool calls — see below):

| shape | prompt / completion tokens (plain / MTP) | acceptance | plain body ms (hit rate) | plain tok/s | MTP tok/s | MTP pass ms | tokens per pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| counting rig (`max_tokens` 200) | 26 / 171 / 171 | 20.6 % | 39.6 (0.990) | 22.0 | 5.7 | 212 | 1.21 |
| prose, tools-off, the first user turn | 48 / 520 / 529 | 31.7 % | 48.2 (0.956) | 18.4 | 5.7 | 233 | 1.32 |
| tool-call continuation, tools on, 1.4k context | 1,425 / 139 / 137 | **75.6 %** | 72.5 (0.880) | 12.5 | 5.6 | 312 | 1.76 |

The counting ladder (`max_tokens` 20 / 50 / 100 / 200 on one lifetime) read
11.8 / 25.0 / 17.9 / 20.6 % — not monotone in pass count, 2 of the first 17
passes accepted — so the candidate the contract read left open (the reject
rewind compressing the sidecar's context, `StreamingMTP.swift:417-423`) is not
the cause of the low rate; the head is wrong from the first passes and the
three-line A/B was not built. **Acceptance is prompt-dependent, and the premise
that a counting prompt must approach 100 % was wrong:** a one-layer head
copying URLs and JSON out of context does well; incrementing a number across a
newline is a reasoning step it does badly.

**Greedy speculation is not lossless on this runtime, on two of the three
shapes.** Counting matched the plain arm exactly (`494bab3edb62`, 171 tokens
both). Prose diverged (`1a22a1f2773a` against `682f3da2f610`, 529 against 520
tokens). Tools-on matched on the 61-character `content` (`fe2d64126a6f` both)
but not on the tool calls: the second call's arguments differ, 137 against 139
completion tokens. So the tools-on acceptance of 75.6 % is a valid measurement
of the MTP path on that prompt, not a like-for-like comparison against the plain
trajectory. The mechanism is not measured here; it is consistent with the pair
path and the decode path differing in the last ulp and a near-tie argmax
flipping. Recorded as a correctness finding (plan follow-ons), not chased.

**Where the pass goes.** The verify runs the prefill path's kernels at width 2:
the MTP arm's kernel roles are `prefill_gdn_router`, `prefill_shared_expert`,
`prefill_attn_router` and one pair kernel, `verify_routed_pair`, for the routed
experts (`RealForwardRunner.swift:4641-4652`, `:4707`). Per pass on the counting
lifetime (141 passes; the request's own 26-token prefill excluded): **GDN 57.9
ms** (the prefill GDN kernels at two rows — 47 % of the pass's GPU), the routed
pair kernel 30.5, attention 18.7, the verify head 9.2, the shared expert 7.8 —
**GPU ≈ 124 ms against a plain step's 43 ms of GPU**; intra-pass host gaps
≈ 46 ms (≈ 1.16 ms per layer boundary across the router → shared → pair →
next-layer round trips, ≈ 80 command-buffer completions per pass), driver
≈ 17 ms, and the pass turnaround ≈ 26 ms (the proposal's 18 ms plus checkpoint
and commit) = 213 against the measured 212; occupancy 57 %. The control arm
(`SHRIKE_MTP_VERIFY=tile`, the older grouped-tile schedule, same prompt): 24.1 %
acceptance on a 137-pass trajectory with the same text, pass 257 ms, GPU ≈ 142
(routed on the grouped tile 42.3 against the pair kernel's 30.5; GDN 60.4,
attention 19.6, shared 10.0), and worse gaps (`shared→routed` host 34 ms per
pass) — the excess is outside the routed kernel on both schedules. The verify
backbone is 3.7–4.5× a plain step's body (the full verify phase 3.8–4.8×). The
MTP arm's decode hit rate is 0.979 and `io_fetch_ms` 12 ms, so the earlier
attribution of this pass to expert-miss I/O does not hold: the excess is
structure, and the GDN prefill path at width 2 is the largest single term.

**A side finding: plain decode is hit-rate-bound and prompt-dependent** — body
39.6 / 48.2 / 72.5 ms (22.0 / 18.4 / 12.5 tok/s end to end) across the three
shapes as the expert hit rate falls from 0.99 to 0.88; the 25.5 tok/s of record
is `1000 / body_ms` on the counting rig.

**The economics.** With one drafted token per pass, tokens per pass is 1 + p and
the whole pass must cost under (1 + p) × the plain per-token time to tie: 55 ms
on counting, 72 on prose, **141 on the tools-on shape** — and the pass's GPU
work alone is 124 today. A path to ≈ 1.3× plain exists on the tools-on shape
only, and needs both a verify on decode-width kernels without the per-layer
round trips (a k = 1 pass near 1.3–1.5× a plain step, ≈ 105–120 ms there) and
two-token drafting at an acceptance that holds along the chain — and the two
conditions compose tighter than they read: a k = 2 pass carries a second
proposal (21.6 ms measured on that shape) and a width-3 verify, against a
budget of ≈ 140 ms for 1.3× at p = 0.76 (T ≈ 2.3). The head is one-step by
design and the second condition is unmeasured. The brief's k × p table was not
re-run: k = 1 is a hard gate, so the three thresholds above are the useful
object. Verdict: exit (c) — speculation is retired as a lever for now, the
priced path is recorded below, and the decode fetch work goes to plain decode,
where the hit-rate numbers say the prize is.

### The chapter's close (2026-09-04)

Whole-branch review (opus) over 06278ed..dc8c5db — P4 onward as a diff; P0–P3
(3c326d9..06278ed, 2,517 source and test lines: the first grouped routed MoE, the
matrix attention kernel, the shared expert's matrix path, the runner's prefill
restructure) were reviewed per task when they landed and re-read in their current
form at the close, not re-reviewed as a diff. Verdict: ready to merge; 0 Critical,
2 Important (the `residency=` print under the per-slot layout and the cache layout
missing from the projection-path line — fixed in the close commit; the review
range — recorded here), 10 Minor (fixed or ruled on the SDD ledger); the 18 deferred
minors triaged 17 fine, 1 moot. Merge target and push are the owner's.

### Follow-ons, not scheduled

- **The cache settle's re-prefill after a degenerate turn (not a
  prefill-matrix lever; measured here).** A request that ends by `max_tokens`,
  a stop string or an unclosed thought is normalised by dropping the emission,
  and with no snapshot at the prompt boundary the drop is a reset and a full
  re-prefill of the prompt: **66.8 s of M1 GPU after a 12k `finish=length`
  response** (17.7 s after 3.7k), deferred on the session actor, and the next
  request's first await joins or aborts it — abort is checked between layers
  and between chunks. The measure prompts have exactly that shape; the fix (capture the
  boundary snapshot at decode start; decouple the join) is the v10 plan's open
  item ([v10-implementation-plan.md](v10-implementation-plan.md)), not this
  chapter's. Measurement hygiene: one send per server lifetime. **The fix is superseded by the owner's ruling that resolved v10's Q1 (v10-implementation-plan.md); the leftover re-prefill cost is filed in tt as SHRIKE-28 (2026-09-23).**
- **The 16-row tail rung.** `P16` prices it (the table in Step 13): the
  whole padding residual is 0.225 ms/token at α₁₆ = 0.55, but a 16-row tile
  carries the same fixed dequant as a 64-row one; Task 15b's measured α says
  whether a second rung is worth a bench. **Filed in tt as SHRIKE-48 (2026-09-23).**
- **The per-tile host work on the M4 Pro (the audit's L10).** After P15
  `routed→routed` is 5.3 s of the M4 Pro's 22.7 s at 12k — `host_ms` 4.5 s
  over 3,482 boundaries, ≈ 1.3 ms each — the largest item left on that box; on
  the M1 it is 0.07 ms/token (887 ms over 3,485). An M4 Pro lever. **Filed in tt as SHRIKE-50 (2026-09-23).**
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
  4.5 s of 31.6 at 12k). **The ≈ 3.3 ms is filed in tt as SHRIKE-50 (2026-09-23); two fetches in flight is done (v13 T1).**
- **The first tile of each layer-chunk on the M1** still pays ≈ 6.9 ms of
  host time between the shared expert and the routed tiles after P12
  (`shared→routed` `host_ms` 827 ms at 12k over 120): the tile metadata
  build plus the first tile's fetch, which nothing hides. The routed fetch
  follow-on above is the same cost seen from the other box. After P15 the
  alternating sweep leaves the modelled ≈ 112 (measured 119) of the layer's
  experts resident from the previous chunk and the row reads 400 ms (3.3 ms
  per layer-chunk). **The shared-to-routed gap is filed in tt as SHRIKE-31, tile 0's fetch within it as SHRIKE-33 (2026-09-23).**
- **The MPP GEMM's byte-load fallback on the M1.** After P9 the dequant's
  byte-load body serves only weight bases that are not 16-byte aligned (and
  the `SHRIKE_MPP_WEIGHT_LOADS=byte` A/B), and inside the two-body kernel it
  runs ≈ 13 % slower on the M1 than P8's single-body kernel did (9.35 vs
  8.25 ms per ornith tile, reproducible; the M4 Pro is unaffected). No
  production tensor of the six models takes it today; a byte-only
  instantiation would restore it if one ever does.
- **The MPP GEMM's remaining fifth on the M1's routed tile.** After P10 the
  routed tile runs at 78 % of the mini's same-run ceiling, and 256 is the last
  K width the chunk mapping supports, so what is left of the staged structure
  needs the register-resident weight tile, not a wider one. The dense
  projections are at 85–99 % of the ceiling (Step 11) — a `kMPPAffineTileM`
  sweep has ≈ 0.05 ms per token to take and is not scheduled. **The register-resident weight tile is filed in tt as SHRIKE-22 and the `kMPPAffineTileM` sweep as SHRIKE-48 (2026-09-23).**
- **The GDN pre-scan chain** is within 2.3 ms per layer-chunk of its
  streaming floor (Step 11); fusing conv → qk-norm and vectorising the norms
  would take ≤ 0.02 ms per token and is not scheduled. **Filed in tt as SHRIKE-48 (2026-09-23).**
- **Attention after the FlashAttention null (Step 15).** The matrix-path
  attention is bound by its per-tile fixed work at 16 rows per K/V read, and
  the register-resident shape that removes the round-trip loses that reuse and
  runs 3× slower on the M1. The 32-row block at 128 keys is `g4k128d`,
  measured at Step 15 at 1,472 ms against the control's 1,419 (3.7 % behind);
  32 rows at 256 keys exceeds the group body's 32 KB threadgroup budget. So
  more rows per K/V read needs either the score round-trip gone (the register
  shape did that and lost the reuse) or a narrower staged tile; no cheap shape
  is left, and `ShrikeBench attn` is the instrument for whatever is proposed.
- **The mini's fetch term after Step 16.** The variance half went with the
  two-tile bank (host-late 942 → 94 ms at 12k); what remains is the mean
  regime — the host's per-tile time exceeds the GPU's wherever the hit rate
  is low (M1 3.7k: 5.58 vs 5.10 ms; the M4 Pro at 12k, its one measured point:
  3.4 vs 1.5) —
  and the bank only absorbs one layer-chunk's deficit at a time. Three
  unscheduled levers, priced from Step 16's counters: **two fetches in
  flight** (issue tile N+1's `beginFetchRoutedExperts` before tile N's encode,
  await it at the next head — cross-tile drive depth 1 → 2, and the ≈ 0.72 ms
  of non-read host cost per tile off the critical path; ≈ 0.1 s at the M1's
  12k, ≈ 0.1 s at 3.7k, up to ≈ 9 s of the M4 Pro's 49); **the first tile of
  each layer-chunk issued before `waitForCompletion(sharedCB)`** — routes are
  built before that wait, so tile 0's fetch could ride the shared expert's
  ≈ 17 ms of GPU (`shared→routed` host ≈ 410 ms at 12k, 0.03 ms per token);
  and **the reader's thread count** (a literal 4: production's marginal read is
  2.81 GB/s against the v10 probe's 3.25 GB/s at queue depth 4 —
  `docs/v10-implementation-plan.md`, P3 — and depth 8 is unmeasured). None
  reaches the GPU roles; every further routed-GEMM cut on the M1 still lands in
  the fetch term first. **Tile 0 is filed in tt as SHRIKE-33 (2026-09-23); two fetches in flight is done (v13 T1), and the reader's thread count is done as a measured null (v13 T2).**

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
- Speculative decoding's verify step — audited at Step 17, retired as a lever
  for now. It is a width-2 forward that runs the *prefill* kernels at two rows
  (only the routed experts have a pair kernel); its verify backbone is 3.7–4.5×
  a plain step's body on the mini, the GDN prefill kernels at width 2 the
  largest term, and its cost is structure, not expert-miss I/O (verify hit rate
  0.98). Acceptance is prompt-dependent: 21 % on the counting rig, 32 % on
  prose, 76 % on a tool-call continuation. The P4 → P9 shift on the counting
  prompt (25.9 → 22.3 %; 20.6 % at P16) stays open: the audit measured neither
  build, and the ladder was not monotone; it is consistent with a trajectory
  change (at P9 the MTP arm emitted 170 tokens against the plain arm's 171, at
  P16 it emits 171 with the plain digest), which is not a measured mechanism.
  With one drafted token the path cannot beat plain at any acceptance while
  the pass exceeds (1 + p) × the plain per-token time. The priced path, not
  scheduled: a verify on decode-width kernels without the per-layer round
  trips **and** two-token drafting — ≈ 1.3× plain on tool-heavy shapes only,
  the two conditions composing tightly and the second unmeasured. The decode
  fetch work goes to plain decode instead, where decode is hit-rate-bound
  (22 → 12.5 tok/s end to end from a counting prompt to a 1.4k tools context). **The P4 → P9 shift is superseded, MTP deleted (efdc628); the priced path is filed in tt as SHRIKE-16 (2026-09-23).**
## Risks

- Threadgroup memory at head-dim 256 caps the attention tile; if 40 % of the
  ceiling is not reachable with `matmul2d`, the fallback is hand-written
  `simdgroup_matrix` fragments (more code, same math). The spike decides.
- Settled: the step-0 log line observed the matrix kernel compiling on both
  boxes — M4 Pro and M1 both log `prefill_projection_path=affine-threadgroup-f16`.
- Per-expert GEMMs at short prompts fall below the efficient row count; the
  threshold keeps the scalar path there, so short prompts do not regress.
