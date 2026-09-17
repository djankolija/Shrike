# v19: the scan rewrite

The chapter that takes the decode attention scan from twelve times the byte roof to
near it, on the contexts a coding session actually runs at. Companion plan:
[v19-implementation-plan.md](v19-implementation-plan.md), whose checkboxes are the
status of record. The board this chapter was chosen from is
[v18-avenues.md](v18-avenues.md) (section B for the surface, section 5 for Davor's
ruling of 2026-09-09 that the scan rewrite runs first and alone among the class-2
avenues); the tree it starts from is `main` at `2d357c6`, v18's close
([v18-quiet-host.md](v18-quiet-host.md)).

Every number in this document is labelled **measured** (a counter or a clock on the
mini), **modelled** (arithmetic on measured inputs) or **remembered** (an earlier
chapter's record, cited, not re-run), and carries the grade of the Method (M, T, C,
R) where it is load-bearing.

## The problem

Only the ten full-attention layers of the served model scale with context; the thirty
gated-DeltaNet layers keep a fixed state (`ModelTypes.swift:273-279`, every fourth
layer of forty is attention). The attention row, `attn_layer_kv` per decoded token
against context (measured, the rig's four points and the mini's live log of
2026-09-08, [v18-avenues.md](v18-avenues.md) section 3):

| context | `attn_layer_kv` ms per token | the token ms | source |
| ---: | ---: | ---: | --- |
| 41 to 54 | 4.7 to 5.1 | 51 to 57 | live log |
| 289 | 5.40 | 60.9 | the rig, the 300 |
| 1,069 | 7.21 | | the rig, the 1k |
| 2,125 | 9.30 | | the rig, the card |
| 4,579 | 14.6 | 69.3 | live log |
| 7,062 | 21.3 | 74.2 | live log |

**2.0 to 2.4 ms per 1,000 context tokens per decoded token (M)**, linear across the
rig and the day's sessions. The 4.7 to 5.1 ms at no context is the row's
context-independent part: the projections, the KV quantize, the combine and the
dispatch walls. Everything above it is the scan.

The KV cache is stored at 8 bits in production: one K row and one V row of 544 bytes
per position per layer, 512 packed values for both KV heads plus eight fp16 scales
and eight fp16 biases inside the row (`KVCacheManager.swift:477-485`,
`kv_cache_quantize.metal:46-51`). Ten layers at 1,088 bytes a position over
62.5 GB/s is **0.17 ms per 1,000 per token (C)**: the scan runs at twelve times its
byte roof.

Step zero of v18 measured what the same silicon allows (S0.8, [v18-avenues.md](v18-avenues.md) B5,
measured on the mini with Shrike stopped): MLX's decode attention through the box's
own oMLX, on Qwen3-14B with fp16 KV at 6,592 context, scans at **20 ns per KB against
the 16 ns roof**, 80 % of the byte roof. Ours scans at 200 ns per KB. The gap is our
kernel, not the M1. At the reference's rate the slope falls from 2.0 to 2.4 toward
0.2 to 0.3 per 1,000, worth about 14 ms per token at 7k context and about 30 at 16k
(modelled from the measured slopes; graded below).

At 7k the attention row is 21.3 ms of a 74.2 ms token (29 %, measured) against the
miss window's 13 (18 %). At the working contexts of a coding session attention
already outweighs the drive, and it is the largest single number on the board.

## What the kernel does today (read at `2d357c6`)

The production path for the served model is `attention_decode_partial_shared`
(`attention.metal:509`), v11's KV-head-shared partial, followed by
`attention_decode_combine` (`attention.metal:853`). Two dispatches per attention layer
per token, both on the layer's one encoder (`Attention.swift:512`, `:551`), after
two KV-quantize dispatches that write the token's K and V rows.

**The grid.** One threadgroup per (KV head, chunk): two KV heads by 64 chunks, 128
threadgroups of 256 threads, eight simdgroups each (`Attention.swift:493-498`,
`:538-542`; the 64-chunk budget exists because the shared path has NKV-fold fewer
threadgroups than the per-head path and "takes the full chunk budget to keep the
machine occupied"). At 7k context each threadgroup owns about 110 positions. The
shape gate is at `Attention.swift:50-60`: head dim at most 256, at most eight query
heads per KV head, no ring, not the SWA-GQA path.

**The loop.** Simdgroup *g* owns query head `kv_head * 8 + g`. Q for the eight heads
is copied into 8 KB of threadgroup memory once (`attention.metal:553-557`). Then, per
block of four positions (`kAttnSharedPosBlock`, `attention.metal:506`, "4 keeps k+v
staging at 8 KB"): all 256 threads stage the four K rows and the four V rows into
threadgroup memory as fp32, each thread loading four packed bytes of K and four of V
per row and dequantizing them with the row's group scale and bias
(`attn_stage_kv4`, `attention.metal:152`); a threadgroup barrier; each live simdgroup
dots its head against the four staged rows, one lane-strided chain and one `simd_sum`
per position, and updates its online softmax with a rescale on every position; a
second barrier. Two barriers per four positions, and nothing is in flight from the
device while a threadgroup computes. Each threadgroup holds 16 KB of threadgroup
memory for its whole life, 8 KB of it the Q copy that is read once.

**The combine.** One threadgroup per query head reads the 64 partials, recomputes the
global max and the denominator serially over chunks and writes the fp16 output
(`attention.metal:874-904`). Its cost is context-independent: 1,024 partials of 256
floats written and read per layer, about 2 MB, about 30 µs a layer at the roof (C).

**The history (remembered, [v11-kv-attention-inner-loop.md](v11-kv-attention-inner-loop.md)
and its plan).** v11 found the slope at 8.0 per 1,000, pinned the mechanism by
microbench as traffic-bound at about 40 % line utilisation and amplified eightfold by
GQA (each of the eight query heads re-reading its shared KV head), and landed the
shared partial for 2.81 on the mini; the V4.1 function-constant specialisation took
it to 2.19 and V5's vec4 staging to 2.07. v11's thesis that the per-position
barriers were the depth tax was falsified by its own twin (V3, a null; the kernel
survives as `attention_decode_partial_sg`). Its ledger exonerated the chunk count,
the barriers as such, the per-element dequant, the storage mode and a planar
re-layout; its residual list names line utilisation on half-row slices, the 64-chunk
wall and the four-position block. It never tested occupancy (the 16 KB footprint) or
latency hiding (a second block in flight while one computes), because the kernel's
shape made neither askable.

**The tests.** `AttentionTests` (`AttentionTests.swift:261`, `kvSharedTracksReference`
and its siblings) compare the kernel against the CPU reference
(`ShrikeValidation/Support/Reference/Attention/Attention.swift`, fp32, vDSP) at a
tolerance of 1e-2; `KVCacheQuantizedAttentionTests` runs the quantized cache end to end
at 0.02 for int8. Bitwise arms exist only for the V4.1 specialisation against the
unspecialised kernel. The only microbench is a test suite gated by an environment
variable (`AttentionDepthBenchTests.swift:12`), local only: the mini has no toolchain.

## What the reference does (read 2026-09-17)

MLX `main` at `c948334a`, `mlx/backend/metal/kernels/sdpa_vector.h` and
`mlx/backend/metal/scaled_dot_product_attention.cpp`. The probe ran the MLX that
oMLX 0.5.3 pins, not this commit; the kernels below predate the probe, and the
read-once GQA variant that does not (merged 2026-08-18) did not apply to the probe's
shape. **R** for the version, read for the structure.

- **The mapping.** One threadgroup per query head (the one-pass kernel: 1,024
  threads, 32 simdgroups) or per KV head and block (the two-pass kernel: 32 lanes by
  the GQA factor, one simdgroup per query head, all of them streaming the same
  block). Each simdgroup takes one key position per iteration; the 32 lanes split
  the head dimension, `D / 32` elements each.
- **The loop.** A lane loads its slice of the K row from device into registers, dots
  it with its slice of Q, and the simdgroup reduces with one `simd_sum`; the online
  softmax's running max and sum are per-lane registers, rescaled on every accepted
  position with `fast::exp`, the scale folded into Q at load; the lane loads its
  slice of the V row and accumulates in fp32. No threadgroup memory and no barrier
  anywhere in the loop; the simdgroups of a threadgroup merge once at the end
  through threadgroup memory.
- **No matrix unit.** `simdgroup_matrix` appears nowhere in the decode kernels.
- **The two-pass split.** Taken when the context is at least 4,096 and the model is
  GQA: 64 blocks on the base chips, positions interleaved across blocks, each block
  writing an unnormalised partial with its max and sum; the second pass reduces the
  blocks per query head.
- **At the probe's shape** (Qwen3-14B: 40 query heads over 8 KV heads, head dim 128,
  6,592 context): the two-pass kernel, one threadgroup per (KV head, block) of five
  simdgroups streaming the same rows, 512 threadgroups, **about 2,560 independent
  simdgroups in flight** against our 128 threadgroups of eight lockstepped
  simdgroups. Each K and V row was read five times, once per query head, and the
  kernel still ran at 80 % of the roof counted in unique bytes: the re-reads were
  absorbed by the cache.
- **Quantized KV** is not in MLX `main`. An open pull request (#3026) carries a
  design for it: four lanes per key, eight keys per simdgroup per iteration, the
  dequant fused into the dot with one scale per group, scales in separate arrays.
  A second design point for our packed rows, unmerged code, **R**.

## The diagnosis (a hypothesis, priced by step zero)

The two kernels do the same arithmetic per byte to within a small factor: one
lane-strided or contiguous chain, one reduction and one rescale per head per
position, fp32 throughout. They differ in structure:

1. **Latency exposure.** Ours loads, waits at a barrier, computes, waits again, and
   loads the next block; the reference's loads land in registers and the next
   position's loads are issued while this one computes, across thousands of
   independent simdgroups. Ours also loads four bytes per thread per row where the
   memory system wants sixteen.
2. **Occupancy.** 16 KB of threadgroup memory per threadgroup bounds how many
   threadgroups a core keeps resident, and the eight simdgroups of each one march in
   lockstep between barriers. The reference's loop holds no threadgroup memory.
3. **Reduction granularity** is the same in both (one `simd_sum` per head per
   position) and is not the suspect.

Which of the first two binds, and by how much, is a measurement, not an argument.
v11's M4 numbers did not transfer to the M1 (its plan says so), so the ablation runs
on the mini.

## Step zero: the constraint named and the prototype priced

Measurement and tooling; no runtime code changes. The bench runs on the mini with
Shrike stopped (a model-using run under the process rules; deploy leave per
session). Every arm is a rate in ns per KB scanned and µs per position per layer,
against the 16 ns per KB roof and the reference's 20.

- **S0.1 The 7k shape on the rig, and the four-shape baseline.** The three rig
  prompts are generated by `tools/turn-prompts.py:45-50` (a ledger of entries at a
  per-shape offset). A `t7k` / `t7kb` pair of about 112 entries at a fresh offset,
  about 7,000 prompt tokens, joins them; the decode rig
  (`tools/decode-rig.sh:3`, `:129`) gains `d512-7k`. Two production lifetimes per
  shape on the current tree, four shapes, recorded as the chapter's ledger. The
  6k pair that already sits unreferenced in the v13 prompt archive is not used:
  7k is the context the live log measured.
- **S0.2 The bench executable.** A small executable target `ShrikeAttnBench` under
  `sources/`, depending on `Shrike` for the Metal context, the kernel library and
  the KV row format. It builds synthetic K, V and Q at the served shape (two KV
  heads, sixteen query heads, head dim 256, int8 rows of 544 bytes with random
  scales and biases), carries its kernel variants as Metal source under its own
  `Metal/` resource directory compiled at run time (the project's own pattern,
  `Package.swift:48-50`, and what `v5CostLedger` does locally,
  `AttentionDepthBenchTests.swift:478`) so the ablation never touches the production
  kernels, and times each arm by the command buffer's GPU start and end over a set
  of position counts and repeats, with a checksum of each arm's output so a broken
  arm cannot post a fast number. Deployed to
  the mini like the CLI (the binary and its own resource bundle beside the bundles already there). Kept as a
  tool: v20 and v21 have kernels of their own to measure on the target.
- **S0.3 The ablation ladder (B7).** The production kernel's source with one thing
  removed or changed per arm, at 1k, 4k and 8k positions:
  1. as shipped (the baseline, and the bench's calibration against the runner's
     `attn_layer_kv` slope);
  2. Q in registers (the 8 KB copy gone; occupancy);
  3. an eight-position block (the staging doubled to 16 KB; occupancy the other
     way);
  4. double-buffered staging (block *n+1* loading while *n* computes; latency);
  5. the softmax removed (the dot and the V accumulate only; ALU);
  6. V removed (the K scan and the softmax only);
  7. a pure load at the same layout (the half-row slices, nothing computed: the
     floor of this access pattern);
  8. a pure load over the full row (both KV heads' 512 bytes plus the tail; the
     floor of the layout itself).
  The arm that collapses the slope names the constraint. The distance between 7
  and 8 prices the half-row slicing; between 1 and 7, everything else.
- **S0.4 The streaming prototype.** Approach A below, as a bench kernel over the
  same synthetic rows, swept over heads per simdgroup (1, 2, 4, 8), positions per
  simdgroup per iteration (1, 2) and one or both KV heads per threadgroup, at 1k,
  4k and 8k positions; each configuration against the shipped kernel and the
  pure-load floor. The prototype is throwaway if its structure loses and the seed
  of Task 3 if it wins. Its no-load twin (the same kernel with the loads replaced
  by constants) says whether it is ALU-bound, which decides whether B6 is needed.
- **S0.5 The record and the ruling.** The ladder, the sweep and the prices in this
  document; then Davor's ruling among three outcomes: the class-1 repair first and
  the rewrite after (if the ladder prices the repair at a large share of the gap);
  the rewrite alone (if it does not); or neither (if the pure-load floor of our
  layout is itself far from the roof, which would make the row's problem the
  cache's layout and a different chapter).

## Approaches

**A. The streaming scan (the recommendation for the rewrite).** The reference's
structure on our rows. One threadgroup per (KV head, chunk) as today, or per chunk
over both KV heads if S0.4 says the full row reads better. Each simdgroup owns a run
of the chunk's positions and, per position, its lanes load their contiguous
eight-byte slice of the 256-byte half-row, dequantize with the group's scale and bias
in fp32 exactly as `attn_load_kv` does, dot against the heads it carries (the count
swept in S0.4; the reference found four heads per simdgroup at head dim 128 the
register limit, so two is the expectation at 256), reduce once per head with
`simd_sum`, update the online softmax per head with the same `attn_softmax_exp` and
the same scale applied to the score as today, then load and dequantize the V slice
and accumulate in fp32. No threadgroup memory and no barrier in the loop. At the
chunk's end the threadgroup's simdgroups merge their partials once through
threadgroup memory (max, rescaled sum, rescaled output) so that the combine's
contract is unchanged: one partial per (query head, chunk), `attention_decode_combine`
untouched. Class 2 at its mildest: the same dequant, the same exp, the same scale,
fp32 accumulation; what changes is the order of the summation within a lane's chain
and the order in which positions enter the online softmax.

**B. The class-1 repair.** The shipped kernel with Q in registers, double-buffered
staging and sixteen-byte loads. Bitwise by construction (the chain, the reduction and
the softmax untouched), so it lands under the golden with no gate cost. Its ceiling
is the shipped structure's; its worth is S0.3's arms 2 and 4. If they close most of
the gap, it lands first as Task 2 and the rewrite's remaining prize is re-priced.

**C. The matrix-unit tile (B6, held).** The scores for eight heads by eight
positions as `simdgroup_matrix` multiplies over a dequantized half-precision K tile,
then P by V the same way. A higher ALU ceiling, but the tile needs staging into
threadgroup memory again and half-precision inputs widen the numerics change beyond
summation order. The reference reaches 80 % of the roof without it, and the
arithmetic says A is not ALU-bound (about nine fp32 operations per byte at the roof,
under a third of the M1's budget, C). Built only if S0.4's no-load twin says
otherwise.

## Tasks

### Task 1: the instrument (class 1; runs beside step zero)

The board's class-2 gate ([v18-avenues.md](v18-avenues.md) section 5) names three
instruments; the second does not exist. This task builds it once, for this chapter
and v21, and closes the golden's coverage gap the v18 review found on the way.

- **A forced-token decode mode in the CLI** (`--force-tokens <file>`, one token id
  per line, carried as `forcedTokens` on `GenerationConfig`): the loop at
  `RawCompletion.swift:275-296` takes the file's next id in place of the sampler's,
  from the first generated position, through the synchronous logits path (never
  the boundary producer), and stops when the list ends. Forcing removes the butterfly effect: two builds decode the same positions
  with the same inputs, so the comparison measures the kernels, not the trajectory.
- **A logits dump** (`--dump-logits <file>`, a `LogitsSink` on
  `GenerationConfig` that the loop calls once per position): every position's full
  fp32 logits (248,320 floats, about 1 MB) read back after the head's command
  completes and appended as raw rows, with a JSON sidecar carrying the vocabulary size, the
  position count, the tokens fed and the build's commit. Both golden prompts, the
  96 and the 128 positions, come to a few hundred megabytes per build.
- **The comparison** (`tools/logit-compare.py old new`): per position the KL
  divergence old to new in fp64, the maximum |Δ logit|, the argmax of each and the
  old build's top-2 margin; the band is three times the run's maximum |Δ|; a flip
  whose margin sits inside the band is variance, a flip at a wider margin is a
  defect, and the script says which. Old against old must report zero everywhere:
  that is the instrument's own test, and it is run first.
- **`--logits-head`** in the CLI: the runtime already takes the flag
  (`RuntimeConfiguration.swift:178`, `forceLogitsHead`); the CLI passes it only when
  the sampler is not pure greedy (`Run.swift:218`), the server always
  (`ServerInference.swift:723`). The flag forces the server's head path at
  temperature zero. `tools/golden-baseline.sh` gains that mode and two profiles
  through it, captured on both boxes on the current tree at this task, so the gate
  covers the boundary path for every class-1 commit after.
- **What the task is not.** No kernel changes; the four gates and the golden
  identical on both boxes. The first use of the instrument is the fused greedy head
  against the logits head on the same build at temperature zero: they should agree
  at every position except exact ties, a calibration of the band on a known
  class-1 pair.

### Task 2: the class-1 repair (only if S0.5 rules it)

Approach B on the shipped kernel: Q in registers, the staging double-buffered, the
loads widened, in whichever order S0.3 priced them. Bitwise: the existing bitwise
arms extended to compare the repaired kernel against the shipped one at every shape
class, the golden identical on both boxes, the arms on four shapes. Pre-registered
row: `attn_layer_kv` at 7k, from 21.3 by the share S0.3 measured. A Task 3 that
follows replaces this kernel; the work is not wasted only if the ruling was that
the repair's floor stands on its own.

### Task 3: the streaming scan (class 2)

Approach A in the runner, its configuration the one S0.4 chose.

- **The kernel**, `attention_decode_partial_stream`, beside the shipped one in
  `attention.metal`, on the V4.1 function constants (bits, stride, value bytes,
  group size folded), the same buffers and the same partial contract. The shape gate as today's (`Attention.swift:50-60`);
  the chunk budget kept at 64 unless S0.4 found a better count, so that the combine
  and its tests are untouched.
- **The kernel arm** (the gate's first instrument): the tolerance arms of
  `AttentionTests` (the CPU reference at 1e-2 and the quantized cache at 0.02)
  extended to the new kernel at the small, the straddling and the served shapes
  and at int8, int4 and fp16 rows; a new-against-old arm at the served shape with
  the maximum |Δ| recorded (the fp16-accumulation band the gate names); the PSO
  engagement test.
- **The instrument** (the second): both golden prompts forced through the old and
  the new build on both boxes, the comparison's table in the task record, every
  flip inside the band or the task does not land.
- **The read** (the third): the golden prompts free-run on the new build on both
  boxes, the answers in the record, read by Davor for route and language.
- **The golden re-captured** on both boxes after acceptance, once; the bitwise gate
  again for everything after.
- **The arms** on four shapes against S0.1's ledger. Pre-registered rows, graded:

  | row | today (M) | expected | grade |
  | --- | ---: | ---: | --- |
  | `attn_layer_kv`, the 300 | 5.40 | 5.0 to 5.2 | T |
  | `attn_layer_kv`, the 1k | 7.21 | 5.1 to 5.8 | T |
  | `attn_layer_kv`, the card | 9.30 | 5.3 to 6.6 | T |
  | `attn_layer_kv`, the 7k | about 21 | 6 to 11 | T |
  | the slope, ms per 1,000 | 2.0 to 2.4 | 0.2 to 0.8 | T |
  | the 7k token, ms | about 74 | 59 to 64 | T |

  The expectation is the reference's 0.2 to 0.3 per 1,000 carrying the 2 to 4×
  range that transfers have missed by, on the measured slope, over the row's
  measured context-independent 4.7 to 5.1. The 300 sits under the rig's noise
  (1.7 % of tok/s across clean lifetimes); the 1k and the card should show 2 to
  6 %; the verdict lives on the 7k. The misses and the miss window are expected
  flat. The answers are not expected identical (class 2): they are read.

### Task 4, held: the matrix-unit tile (B6)

Built only if S0.4's no-load twin puts Approach A at the ALU wall short of the
roof. Its numerics gate is Task 3's, with the wider band the half-precision inputs
imply, recorded before the arm runs.

## Method

v18's Method holds ([v18-quiet-host.md](v18-quiet-host.md)): the four gates per
commit, ThreadSanitizer once at the close, two production lifetimes per shape on the
mini, the first lifetimes after a deploy discounted, `prefetch_late` lifetimes read
as noise, subagents for the gates and the rigs with verbatim diagnostics, and every
cost graded M, T, C or R with a range, tasks ranked by the floor. Added this
chapter:

- A bench number on the mini with Shrike stopped is **M for the kernel in
  isolation** and **T for the runner**; the runner's `attn_layer_kv` on the rig is
  the verdict, the bench the diagnosis. S0.3's first arm calibrates the two.
- The 7k shape is the chapter's verdict shape; the three fixed shapes guard against
  a regression at short context, where the row is walls and projections.
- A class-2 task's record carries the instrument's table and the read's answers,
  not a digest.

## Numerics policy

Class 1 for step zero, Task 1 and Task 2: every commit byte-identical on both golden
profiles on both boxes, and from Task 1 on the two logits-head profiles as well.
Class 2 for Task 3 (and Task 4 if built) under the gate as ruled: the kernel arm,
the forced-token comparison inside the band, Davor's read, then one golden
re-capture on both boxes. Class 3 stays closed: the KV cache stays at 8 bits, no
approximation enters the scan.

## Out of scope, and where it went

- **The 4-bit KV cache** (B2): class 3, closed for this chapter.
- **The context-independent part of the attention row** (B3, B4: the walls around
  the projections, the combine's fixed 2 MB per layer, the two-pass structure at
  short context): priced after this chapter's arms show the new slope, a class-1
  chapter of its own or a v20 task.
- **Prefill**: untouched; the scan is decode-only and prefill attention runs the
  matrix kernels of v12.
- **The SSD mechanism and the fold**: v20. **Compression**: v21. **The other
  class-2 avenues** (J1, G1, the GDN chain's math): after v21, on this chapter's
  instrument.

## Risks

- **The transfer.** The reference's rate was measured on fp16 rows of a different
  model through a different runtime; our packed rows cost a dequant per element and
  a group read per lane. S0.3's pure-load arms bound what the layout allows before
  any kernel is designed on it.
- **Register pressure.** Eight heads by eight elements of Q and of output per lane
  is 128 registers before the loop's own; the reference spilled at four heads by
  128. S0.4's sweep finds the count that holds; the fallback is fewer heads per
  simdgroup and more simdgroups reading the same row, the reference's own shape.
- **The band.** The instrument's rule is mechanical, but its band is set by the
  run's own maximum |Δ|; a kernel defect that is small everywhere would widen the
  band and hide a flip. The kernel arm's absolute tolerance is the check against
  that, and the read is the last.
- **The mini's deploy leave.** Step zero's bench and every task's arms stop
  production; each run is asked per session, and production is restored and
  verified golden-identical (or, after Task 3, against the re-captured golden)
  before the session ends.
