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

Since v18's one command per layer the kernel counters no longer carry
`attn_layer_kv`: the attention layer's whole held command reports as `layer_kv`
(the norm, the attention, the projection and the layer's speculative routed work),
so the row on the current tree is larger by the layer's MoE share and its slope is
the scan's alone. S0.1 measured it on the four shapes (the step-zero record below):
8.4 ms at the 300, 24.4 at 7,463 context, **2.23 ms per 1,000 (M)**, the same line.

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

### Step-zero record

**S0.1 (2026-09-17, the mini at the v18 close's build, two production lifetimes per
shape, the 7k shape new).** The `t7k` pair is 112 ledger entries at offsets 900 and
1,000, 7,463 prompt tokens measured (the entries run 66.6 tokens each, not the
62.6 the earlier shapes averaged); `d512-7k` in the decode rig. The rows, the cold
answer of each lifetime (`s01-ledger.py` over the arms in
`~/.claude/handoffs/archive/shrike-v19-step0/`):

| shape | run | prompt | answer | tok/s | ms per token | `layer_kv` | `layer_linear` | fixup | head | misses | io ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| the card | 1 | 2,125 | 219 | 16.20 | 61.73 | 12.338 | 24.722 | 1.773 | 4.647 | 4,392 | 14.73 |
| the card | 2 | 2,125 | 219 | 15.98 | 62.58 | 12.340 | 24.704 | 1.777 | 4.666 | 4,407 | 14.75 |
| the 300 | 1 | 289 | 314 | 17.00 | 58.82 | 8.408 | 24.994 | 1.897 | 4.758 | 6,291 | 15.09 |
| the 300 | 2 | 289 | 314 | 17.15 | 58.31 | 8.362 | 24.879 | 1.886 | 4.717 | 6,291 | 14.90 |
| the 1k | 1 | 1,069 | 405 | 16.95 | 59.00 | 10.259 | 24.864 | 1.734 | 4.711 | 7,643 | 14.02 |
| the 1k | 2 | 1,069 | 405 | 16.98 | 58.89 | 10.245 | 24.833 | 1.731 | 4.693 | 7,653 | 14.04 |
| the 7k | 1 | 7,463 | 368 | 13.66 | 73.21 | 24.400 | 24.783 | 1.659 | 4.675 | 6,794 | 13.69 |
| the 7k | 2 | 7,463 | 368 | 13.76 | 72.67 | 24.397 | 24.791 | 1.657 | 4.668 | 6,798 | 13.69 |

What the ledger says: `layer_kv` is 8.4 ms at 289 context and 24.4 at 7,463, a
slope of **2.23 ms per 1,000 (M)** on the current tree, the board's line; the other
rows are flat with context (`layer_linear` 24.7 to 25.0, the fixup 1.7 to 1.9, the
head 4.7, the miss window's io 13.7 to 15.1), so the 7k token's extra 14 ms over
the 300's is the scan and nothing else. The lifetimes agree to 0.1 % on the roles
and 0.7 % on the wall (the 300's two lifetimes 0.9 % apart on tok/s). The 7k
lifetime runs 2.6 minutes: 37 s of prefill (5.0 ms per prompt token) and a
368-token answer at 13.7 tok/s. An oMLX server (a standing service on the box,
not ours, listening on localhost only) was up throughout; whether it held a model
could not be read without its key, and the box reported 34 % of memory free
beside production.

**S0.2 (2026-09-17).** `ShrikeAttnBench` (`sources/ShrikeAttnBench/`, the kernel copy
in `Metal/ladder.metal`): synthetic rows at the served shape through the production
quantizer; the production pipeline through the library's own wrapper
(`Attention.encodeFull`, made public with the wrapper's init and `KVCacheQuantizer`
for the purpose); the ladder kernel compiled from source at run time with one
function constant per switch, dispatched on the production geometry with the
threadgroup memory carved per arm so each carries its own footprint; the median of
nine command buffers after three untimed and a fifteen-run spin-up (the first timed
arm of a process ran 3× slow on the M4 Pro before the GPU ramped). Every arm prints
a hash of its partials, so arms with the same arithmetic must hash equal, and the
maximum difference of its CPU-combined output against the production pipeline's
fp16 output. The checks: every arithmetic-preserving arm hashes equal to the copy
and combines to within 3.8e-6 of production at 4k and 8k; the copy times 7 % under
the production pipeline on the mini (the combine), inside the 10 % rule; and the
production pipeline's 0.217 µs per position at 8k is the runner's 2.23 ms per 1,000
over ten layers to 3 %, so the bench and the rig agree. The unspecialized production
pipeline is 24 % slower on the mini (v11's 19 % holds).

**S0.3 (2026-09-17, the mini, Shrike stopped, 1k, 4k and 8k positions, nine repeats,
the ladder run in both arm orders).** At 8k, µs per dispatch, forward / reversed:

| arm | µs | ns per KB | reading |
| --- | ---: | ---: | --- |
| prod (the pipeline, partial and combine) | 1,778 / 1,786 | 204 | the row's 200 ns per KB, as the board had it |
| copy (the shipped kernel) | 1,654 / 1,658 | 190 | the ladder's baseline |
| qregs (Q in registers) | 1,868 / 1,869 | 215 | 13 % slower |
| block8 (eight-position blocks) | 2,097 / 2,103 | 241 | 27 % slower |
| dbuf (double-buffered staging) | 2,143 / 2,321 | 246 to 267 | 30 to 40 % slower |
| load8 / load16 | 1,650 / 1,647 and 1,862 / 1,871 | 190 and 214 | null and 13 % slower |
| nosoftmax | 1,605 / 1,818 | 184 to 209 | the exp chain is not the cost |
| nov (V never loaded or accumulated) | 594 / 618 | 68 to 71 | half the bytes, 64 % of the time |
| loadonly (the staging only, the same layout) | 177 / 136 | 16 to 20 | at the roof; 50 to 65 GB/s |
| loadonly+fullrow (the whole row per group) | 258 / 194 | 22 to 30 | the half-row slicing costs nothing |

The reading. The layout streams at the roof when nothing computes, so v11's
line-utilisation residual is closed; every occupancy and latency remedy the design
guessed at (Q in registers, larger blocks, double buffering, wider loads) is null or
slower; the softmax is innocent; and the V half of the work costs 64 % of the time
for half the bytes. What all of that fits is the loop form, not the memory system:
the per-lane loops walk `i = lane, lane + 32, ...` with a slot counter the compiler
cannot bound, so the accumulator array is indexed dynamically and the loop is not
unrolled. Q in registers getting slower is the same mechanism (a register array
indexed the same way).

**S0.3b (2026-09-17, the same rig): the loop form as a switch.** The copy's loops
with a static trip count (`s = 0 ..< 8`, `i = lane + 32 s`, the same elements in
the same order), everything else untouched:

| arm | µs at 8k | ns per KB | against the copy |
| --- | ---: | ---: | ---: |
| sloops | 539 | 62 | 3.07× faster |
| sloops+nov | 334 | 38 | the V half now costs half |
| nov (the shipped loop form) | 594 | 68 | the K side alone was 1.8× the static K side |
| sloops+qregs, sloops+load8, sloops+qregs+load8 | 560 to 690 | 64 to 79 | null to slower |
| sloops+dbuf | 678 to 1,168 | 78 to 134 | still slower |
| loadonly | 139 to 196 | 16 to 22 | the floor |

The static form's partials do not hash identical to the copy's, while its combined
output meets production to the same 3.8e-6: the difference is which multiply the
compiler fuses in `o * alpha + p * v`, not the algorithm.

**S0.3c (2026-09-17, the same rig): the form search.** The static loops with the V
accumulate written explicitly, against the copy's hash at 1k and 8k:

| arm | µs at 8k | partials | against the copy |
| --- | ---: | --- | ---: |
| copy | 1,653 | the shipped bits | |
| sloops (as written) | 548 | differ | 3.02× |
| sloops+o1, `fma(o, alpha, p * v)` | 613 | **the shipped bits** | **2.70×** |
| sloops+o2, `fma(p, v, o * alpha)` | 619 | as sloops | |
| sloops+d1, the denominator as an explicit `fma` | 963 | as sloops | the same bits, slow |
| sloops+safemath, contraction and reassociation off | 961 | the shipped bits | |
| copy+safemath | 1,780 | the shipped bits | |

So the shipped kernel fuses `o * alpha` into the add and the static loop as
written fuses `p * v`; the explicit `fma(o, alpha, p * v)` reproduces the shipped
kernel's partials bit for bit at 2.70× its speed. The denominator's form is
hash-neutral either way (one multiply, one fusion) and the explicit call is slow for
no visible reason, so it stays as written. The repair is class 1 on the bench; the
golden on both boxes is its gate in the runner. A caveat on the S0.3b pair of runs: the later arms of each order came out up to 1.8×
slower than the same arm early in the other order (the first ladder's pair did not
show this), so each arm's clean number above is its earlier position, and the
repair's rig arms are the verdict, not the bench.

**S0.4 (2026-09-17, after Task 2; the mini, Shrike stopped): the streaming prototype.**
`sources/ShrikeAttnBench/Metal/stream.metal`, Approach A as designed: a threadgroup
per KV head and chunk, its eight simdgroups as `S_HPT` position streams by
`8 / S_HPT` head sets, each stream a contiguous run of the chunk, each lane loading
its contiguous eight bytes of the half-row into registers and dequantizing in fp32
as production does, the dots and one `simd_sum` per head, the softmax per head,
V the same way into register accumulators; no threadgroup memory and no barrier;
`64 / S_HPT` chunks dispatched so the partials stay 64 per query head and the
combine is untouched. Correct: every variant meets production's fp16 output to one
ulp, and the three head counts hash identical (the same 64 position runs in the same
order). The sweep at 8k, µs, the two orders:

| arm | µs | ns per KB |
| --- | ---: | ---: |
| prod (the fix's pipeline) | 618 / 702 | 71 / 81 |
| sloops+o1 (the fix's form) | 541 / 613 | 62 / 70 |
| stream2 | 433 / 383 | 50 / 44 |
| stream4 | 438 / 317 | 50 / 36 |
| stream8 | 2,337 / 2,340 | 268 | 
| stream2+noload, stream4+noload | 260 / 260, 268 / 238 | 30, 27 to 31 |
| loadonly | 151 / 139 | 17 / 16 |

Eight heads per simdgroup spills, as the register count said (128 floats of Q and
output per lane before the loop's own). Two and four are close; four reads each row
half as often.

**S0.4b (the same day): the levers, interleaved.** The fix's own arm drifted from
541 to 958 µs inside this run (1.77×, the box's second such episode today), so each
variant is read only against the fix's form run immediately before it:

| variant / the fix's form | 8k | 4k |
| --- | ---: | ---: |
| stream4 | 0.59 | 0.57 |
| stream4+lazy (the rescale only when the max moves) | 0.88 | 0.74 |
| stream4+u2 (two positions per iteration) | 0.54 | 0.68 |
| stream4+u2+lazy | 0.67 | 0.67 |
| stream2+lazy, stream2+u2+lazy | 0.78, 1.01 | 0.78, 0.79 |
| stream4+u2+lazy+noload | 0.69 | 0.55 |
| loadonly | 0.21 | 0.18 |

The reading. The structure is worth **1.7× on the kernel over the fix's form (M,
interleaved)**. The lazy rescale is a loss on both boxes: the uniform branch and
the eight-wide multiply under it cost more than the exp they skip. The unroll is a
null inside the drift. The no-load twins sit at 240 to 270 µs against a load floor
of 140 to 150, so what binds now is the per-position chain (the dequant, the dots,
the reductions, the exps, the rescale) under a register-limited occupancy, about a
hundred registers a lane at four heads; the reference runs the same chain behind
ten times the simdgroups, which our register budget cannot buy. The arithmetic
levers left (the affine dot that folds the dequant into one scale per head, a
shared butterfly reduction across the four heads) are class-2 and worth 5 to 10 %
each (C), none structural.

Transferred to the runner (T, the bench's 1.7× on the fix's measured slope): 0.72
to about 0.42 ms per 1,000; at 7k the scan 5.2 to 3.0 ms, **about 2 ms per token,
3.4 %**, above the rig's 1.7 % noise; the card about 0.6 ms, the 1k 0.3, the 300
nothing. The reference's rate stays 1.5 to 2× away.

**The drift, for the record.** Twice today (S0.3b and S0.4b) the mini slowed 1.77×
in the middle of a bench run and stayed slow; four other runs held steady. The cause
was not found in the box's process list; the interleaved pairing is the method that
survives it, and the rig's two-lifetime arms read against each other the same way.

**What step zero changes.** The constraint is named and it is neither occupancy nor
latency nor the layout: it is the loop form. Approach B is no longer a guess about
staging; it is the static trip count with the explicit fused form, worth 2.70× on
the kernel in isolation, bit-identical (M on the bench). Transferred to the runner
(T): the slope 2.23 to about 0.83 ms per 1,000, the 7k row 24.4 to about 14.4, the
7k token 73 to about 63 ms, most of the chapter's prize, as class 1 under the
golden. The rewrite's remaining prize is the distance from 0.83 to the reference's
0.2 to 0.3: about 4 to 4.5 ms at 7k, still above the noise. The bench's
load-only floor at 4k sits partly in the M1's system cache (the rows are 4.4 MB),
so the roof stays the nominal 62.5 GB/s, not the bench's best number.

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
  logits as the head writes them and the sampler reads them, fp16 (248,320 values,
  about 500 KB), appended as raw rows once they are on the host, with a JSON
  sidecar carrying the vocabulary size, the position count, the tokens chosen, the
  tokens forced and the binary's hash. Both golden prompts, the 96 and the 128
  positions, come to about 110 MB per build.
- **The comparison** (`tools/logit-compare.py old new`, pure Python since neither
  box has numpy, about a minute per pair): per position the KL divergence old to
  new in fp64, the maximum |Δ logit|, the argmax of each and the old build's top-2
  margin; the band is three times the **median** over positions of the maximum
  |Δ| (the board wrote the run's maximum; the script's self-test showed that one
  bad position then widens the band and hides every flip, so the median sets the
  band and a position whose maximum |Δ| is ten times the median is a defect on
  its own); a flip whose margin sits inside the band is variance, a flip at a
  wider margin is a defect, and the script says which and exits non-zero on a
  defect. Old against old must report zero everywhere: that is the instrument's
  own test, and it is run first.
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

**The record (2026-09-17).** Built as designed: `LogitsSink` and the two fields on
`GenerationConfig`; the loop takes a forced id in place of the sampler's, records
each position's logits before the choice and the choice after it, bypasses the
boundary producer and refuses a fused greedy head when instrumented; the CLI's
three flags, `FileLogitsSink` with its sidecar, the forced-id reader;
`tools/logit-compare.py` with the median band and the outlier rule; the golden
script's `HEAD=logits` mode. Tests: the forced list replaces the sampler and stops
when it runs out, the sink's rows carry the logits each token was chosen from, an
empty list is refused, the flags parse and the usage inventory holds. On the dev
box with the real model, the short prompt at twelve positions: the free run's dump
against the forced replay of its own ids, KL and |Δ| zero at every position, the
texts identical (the instrument's old-against-old test); a shifted list stops
where it ends with a different text. The logits-head golden captured on both
profiles locally and compared with the fused-head baselines: character-identical
on both, so the two heads agree at every position of these prompts and the
gate now covers the server's path. A lesson on the way: the first full gate run
segfaulted in the server suites (a concurrency job with no Shrike frames in the
crash report), reproduced four times, passed on the last commit with the tree
stashed, and vanished on a clean debug build with the change fully in place. A
public struct gaining stored properties (`GenerationConfig`) left stale objects in
a dependent module's incremental build; the same staleness could sit in the
release products, so the release was rebuilt from scratch before the mini saw
it. Added to the Method: after a layout change to a public struct, clean the
build before trusting a crash. Then the four gates on the clean builds (1,254
tests in 173 suites in 206 s, four new; zero warnings; lint and links clean), the
local golden identical on all four profiles. On the mini (`beeba44a30bbeb12`,
the server down): the fused golden identical on both profiles, the two
logits-head profiles captured (`baselines/ornith15-int4-{short,long}-lh.mini.txt`)
and checked, their text identical to the fused profiles' there too; production
relaunched on this build. The gate now has four profiles on each box.

### Task 2: the loop form (class 1; Davor's ruling 2026-09-17: first)

What step zero found, in the production kernel: the per-lane loops of
`attention_decode_partial_shared` walk `slot = 0 ..< 8` with `i = lane + 32 slot`
and a guard on the head dimension (folded when the shape is specialized), and the
V accumulate is written `fma(o, alpha, p * v)`, the multiply the shipped compiler
fused, so the partials are the shipped kernel's bit for bit (S0.3c). Nothing else
moves: the staging, the barriers, the block, the Q copy, the combine.

**Pre-registered (2026-09-17, before the kernel changed).** The bench's 2.70× on the
kernel transferred to the runner's scan (T): the slope 2.23 to about 0.83 ms per
1,000, the row's context-independent part unchanged.

| row (ms per token) | S0.1 (M) | expected | grade |
| --- | ---: | ---: | --- |
| `layer_kv`, the 300 | 8.36 to 8.41 | 8.0 | T, under the noise |
| `layer_kv`, the 1k | 10.25 | 8.8 | T |
| `layer_kv`, the card | 12.34 | 9.3 | T |
| `layer_kv`, the 7k | 24.40 | 14.4 | T |
| the slope, ms per 1,000 | 2.23 | 0.7 to 1.0 | T |
| the 7k token | 72.7 to 73.2 | 62 to 64 | T |
| the card's token | 61.7 to 62.6 | 58.5 to 60 | T |
| the 1k's token | 58.9 to 59.0 | 57 to 58 | T |

The misses, the miss window, `layer_linear`, the fixup and the head are expected
flat. The answers are expected identical to S0.1's (class 1): the golden on both
boxes is the gate, the bench's hash the kernel-level arm, the test suite's
specialized-against-unspecialized bitwise arms the check that the guard folds the
same way on both paths.

**The record (2026-09-17).** The kernel as described; the four gates (1,250 tests in
173 suites in 203 s, zero warnings, lint and links clean); the golden identical on
both profiles on both boxes, the mini's with the server down before the arms; the
bench's production arm on the M4 Pro 246 µs at 8k against about 650 before, within
10 % of the bench's bit-identical form plus the combine. Deployed to the mini
(`ca2d3bab87469ef2`), two production lifetimes per shape against S0.1's, the
scripts and arms at `~/.claude/handoffs/archive/shrike-v19-t2/`:

| shape | `layer_kv` before | after | the token before | after | tok/s before | after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| the 300 | 8.36 to 8.41 | 7.67 to 7.68 | 58.3 to 58.8 | 57.7 to 57.9 | 17.00 to 17.15 | 17.27 to 17.32 |
| the 1k | 10.25 | 8.26 | 58.9 to 59.0 | 57.2 to 58.2 | 16.95 to 16.98 | 17.17 to 17.47 |
| the card | 12.34 | 8.96 to 9.02 | 61.7 to 62.6 | 59.0 to 59.4 | 15.98 to 16.20 | 16.85 to 16.95 |
| the 7k | 24.40 | 12.82 to 12.84 | 72.7 to 73.2 | 61.6 to 61.8 | 13.66 to 13.76 | 16.19 to 16.24 |

The slope **2.23 to 0.72 ms per 1,000 (M)**, 3.1× against the 2.7× transferred;
every pre-registered row moved past its expectation (the 7k row 12.8 against 14.4,
the 7k token 61.7 against 62 to 64, the card 9.0 against 9.3, the 1k 8.3 against
8.8); `layer_linear`, the fixup, the head, the misses and the miss window flat
within their spreads; the eight answers character-identical to S0.1's. The 300
moved 0.7 ms where its scan was 0.6: the loop form pays on the kernel's fixed cost
too, not only on its slope. On the 7k, the token is 18 % faster and the attention
layers' command has gone from a third of the token to a fifth. Kept: class 1, bit
for bit, and the largest single move on the 7k of any task since the rig was
built.

**What remains of the row after Task 2.** At 7k the scan is about 5.2 ms of the
12.8 (the slope times the context), against the reference's 1.4 to 2.1 at its rate:
the rewrite's remaining prize is about 3 to 4 ms per token at 7k (T), about 5 % of
the token, above the noise. The context-independent 7.7 (the 300's row) is the
projections, the quantize, the combine, the walls and the layer's speculative
routed work, none of it the scan's.

### Task 3: the streaming scan (class 2)

Approach A in the runner, its configuration the one S0.4 chose: four query heads
per simdgroup, two head sets by four position streams per threadgroup, contiguous
runs, sixteen threadgroups per KV head writing sixty-four partials per query
head; no unroll, no lazy rescale.

**Pre-registered (2026-09-17, before the kernel landed).** From the bench's 1.7×
over the fix's form, interleaved on the mini (T for the runner):

| row (ms per token) | Task 2 (M) | expected | grade |
| --- | ---: | ---: | --- |
| `layer_kv`, the 300 | 7.67 | 7.6 | T, under the noise |
| `layer_kv`, the 1k | 8.26 | 7.9 | T |
| `layer_kv`, the card | 8.96 to 9.02 | 8.2 | T |
| `layer_kv`, the 7k | 12.82 to 12.84 | 10.7 | T |
| the slope, ms per 1,000 | 0.72 | 0.4 to 0.5 | T |
| the 7k token | 61.6 to 61.8 | 59 to 60 | T |
| the card's token | 59.0 to 59.4 | 58.3 to 58.8 | T |

The kernel arm's band: relative error against the CPU reference at most 0.02 on
int8 rows (the shared kernel's own bound), and the stream kernel against the
shared kernel on the same rows a maximum |Δ| under 1e-2 on outputs of order one
half. The instrument: every flip inside three times the median |Δ|, no position
above ten times it. The misses and the other rows flat. The answers not expected
identical.

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

  | row (ms per token) | S0.1 (M, two lifetimes) | expected | grade |
  | --- | ---: | ---: | --- |
  | `layer_kv`, the 300 | 8.36 to 8.41 | 7.8 to 8.0 | T |
  | `layer_kv`, the 1k | 10.25 | 8.1 to 8.8 | T |
  | `layer_kv`, the card | 12.34 | 8.0 to 9.3 | T |
  | `layer_kv`, the 7k | 24.40 | 9.8 to 14.1 | T |
  | the slope, ms per 1,000 | 2.23 | 0.2 to 0.8 | T |
  | the 7k token | 72.7 to 73.2 | 59 to 63 | T |

  The expectation takes the measured slope off each row and puts back the
  reference's 0.2 to 0.3 per 1,000 carrying the 2 to 4× range that transfers have
  missed by; the row's context-independent part (the 300's 8.4 less its 0.6 of
  scan) stays. The 300 sits under the rig's noise (1.7 % of tok/s across clean
  lifetimes); the 1k and the card should show 2 to 6 %; the verdict lives on the
  7k. The misses and the miss window are expected flat. The answers are not
  expected identical (class 2): they are read.

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
- After a public struct changes layout (a stored property added or removed),
  clean the debug and release build directories before trusting a crash or a
  deploy: SwiftPM's incremental build left a dependent module's objects stale
  in Task 1 and the test helper segfaulted with no Shrike frame on the stack.

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
