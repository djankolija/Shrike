# v21: lossless compression

The chapter that makes every byte of expert weight carry more information: the
same 4-bit indices, scales and biases the kernels compute with today, stored in
fewer bits and decoded on the way in, so the SSD read, the pool's slot and the
GEMV's stream all shrink without a value changing anywhere. Companion plan:
[v21-implementation-plan.md](v21-implementation-plan.md), whose checkboxes are
the status of record. The avenue is H2 on the board in
[v18-avenues.md](v18-avenues.md) (section 4 for the avenue's pricing at the v18
step zero, section 5 for the rulings of 2026-09-09 that placed compression as v21,
scoped to the expert kernels and the head); the tree it starts from is `main` at
`17babb9`, the Q3 close ([v20-ssd-mechanism.md](v20-ssd-mechanism.md) is the
previous chapter).

Every number in this document is labelled **measured** (a counter or a clock on
the mini, or a count over the model file), **modelled** (arithmetic on measured
inputs) or **remembered** (an earlier chapter's record, cited, not re-run), and
carries the grade of the Method (M, T, C, R) where it is load-bearing.

**The ruling this chapter opens under (Davor, 2026-09-18).** v21 ships only if the
decoder inside the GEMV lanes pays: the chapter's gate is the in-lane decoder's
microbench at step zero (S0.4), and if the decoder is not free the chapter closes
at step zero for the price of the bench. The fallback, decoding once when a read
lands in the pool, wins the read row only (S0.2 found the pool's prize belongs to
the coded bytes sitting in the pool, which only the in-lane design has), about
1 ms per token modelled, and that is below the bar for a format change that
touches the repack, the receipts and a redeploy. The dense weights (GDN,
attention, the shared expert) are not ruled in or out: S0.4's number decides
whether widening the scope is worth their decoder sites.

## The problem

The mini's token at the v20 close is 54 ms on every shape (measured, the close's
tally), of which 10.6 to 12.3 ms is exposed expert io on 14 to 17 misses per token
at 0.73 to 0.80 ms per read (measured), and the rest is the GPU's own work, most of
it streaming weights. The intrinsic decode traffic is 1,800 MB per token and the
streaming roof 62.5 GB/s (remembered, v10's two laws; the ledger in
[v18-avenues.md](v18-avenues.md) section 2 closes to 0.4 ms against it). Three
rows of that ledger are the expert bytes this chapter can shrink, all at the v17
close's pre-fold build, per token, the card:

| row | measured ms | bytes at the roof | what it streams |
| --- | ---: | ---: | --- |
| the routed experts on all-hit layers (`moe_spec_routed`) | 11.4 | 6.6 | 8 experts × 1.77 MB on ~24 layers |
| the hit command and the fixups on miss layers | 4.7 | 3.6 | 8 experts × 1.77 MB on ~16 layers |
| the LM head (`head_logits`) | 4.5 | 4.6 | 248,320 rows × 2048 at 4 bits |

The head runs at the roof; the expert kernels run at about 1.6 times it
(measured against modelled), so a byte saved in the head converts fully and a byte
saved in the experts converts only as far as the kernel is bandwidth-bound. The
routed experts are 566 MB of the 1,800 per token, the head 254 MB, together 820 MB
(modelled from the manifest); the dense projections and the shared experts are
most of the rest.

**The bytes are not full.** The v18 step zero measured ornith15's 4-bit affine
format against its own information (measured, `s06-expert-entropy.py` at
`~/.claude/handoffs/archive/shrike-v18-step0/`, twenty-four experts each from
layers 0 and 20 and 64 rows of the head): the indices carry 3.43 bits per weight
at layer 0, 3.71 at layer 20, 3.69 on the head, against the 4 stored; the bf16
scales and biases carry 6.8 to 8.3 bits of their 16, using 700 to 960 of 65,536
patterns. All in, against the format's 4.5 bits per weight, a nibble code lands at
3.64 / 3.97 / 3.95 bits, **11.8 to 12.3 % fewer bytes** on the typical layer and
the head, the shallow layers more. The scales and biases are 11 % of the stride
today and would be 21 % of a coded expert.

**The three places the bytes pay.**

- **The read per miss.** A read of 1.77 MB costs 0.77 ms at p50 on the mini's
  drive, a read of 64 KB 0.12 ms (measured, the P3 follow-on of 2026-09-04,
  `docs/v10-implementation-plan.md` P3): a fixed part near 0.1 ms and a size part
  near 0.65. Twelve percent off the size part is about 0.08 ms per read, 1.2 to
  1.5 ms per token at 14 to 17 misses (modelled), if the drive scales a single
  read with its size; the P3 probes found the drive's rate set by bytes in flight,
  so this row needs its own probe (S0.3).
- **The pool's capacity.** At `--ram-budget 8G` the pool holds 128 slots per
  layer, 9.06 GB (measured); the same bytes hold about 145 slots of a coded expert.
  The replay prices the misses that buys, offline (S0.2): 22 to 29 % fewer, about
  2.5 to 3.5 ms per token, the largest row.
- **The stream inside the GEMV.** Twelve percent of the expert and head bytes at
  the roof is about 1.8 ms per token (modelled: 12 % of 820 MB at 62.5 GB/s), of
  which the head's 0.55 converts fully and the experts' 1.2 as far as their
  kernels are bandwidth-bound. Widened to every weight-reading kernel it is about
  3.5 ms (12 % of 1,800 MB), the v18 figure. This row exists only if the decoder
  runs inside the GEMV at no visible cost (S0.4).

The ideal in the ruled scope, after step zero's two offline items, is about 5 to
6 ms per token, 10 % of the token, before the decoder costs anything; the v20
chapter was 3 to 4 ms. The fallback is about 1 ms. (S0.1 prices the saving at
11 % after the code's header rather than 12; S0.2 prices the pool row.)

## What the format and its readers are today (read at `17babb9`)

**The format.** A `.gturbo`'s `packed_experts/layer_NN.bin` holds the layer's 256
experts at a fixed stride of 1,769,472 bytes, described by `packed_experts/
layout.json`: per expert the roles gate, up, down in that order, each role its
4-bit indices (two per byte, the low nibble the even column), then its bf16 scales,
then its bf16 biases, one scale and one bias per group of 64 along the row; gate
and up are 512 × 2048 (32 groups per row), down is 2048 × 512 (8 groups per row).
An expert is 3 × 524,288 bytes of indices and 6 × 32,768 bytes of scales and biases:
3,145,728 weights at 4.5 bits. The manifest's `quant.routedExpert` slot names the
scheme (`affine`, 4 bits, bf16 scales and biases, group 64) and the loader refuses
anything else (`ManifestReader.swift:232`), so a new scheme string is refused
loudly by every binary older than this chapter, which is the property a format
change wants. The head (`language_model.lm_head.weight`) and the embedding are the
same affine format inside `model_weights.bin`, addressed by the resident index.

**The read path.** A miss reads one whole expert, `expertStride` bytes at the
expert's offset, with `pread` into a slot of the pool (`PreadExpertStreamer.swift`,
`ModelExpertIO.swift`); the pool is one slab per layer of `slots × stride` bytes
plus the ring's cells at the same stride, allocated at load from the budget and,
since v20 Task 1, the per-layer slot table (`SHRIKE_EXPERT_SLOT_TABLE`). Every
consumer addresses an expert as `slab + slot × stride` and a role inside it by the
fixed `ExpertOffsets`. Prefill's routed tiles read whole experts the same way into
the staging tile.

**The readers.** The decode phase-1 kernel (`moe_phase1_gate_up_act_u16load` and
its `_subset` and `_spec` variants, `moe.metal:766`) walks a row in blocks of 128
bytes: each lane loads 4 bytes of gate and 4 of up as two 16-bit words, eight
weights, and does eight multiply-adds per role against eight activations, with the
group's scale and bias applied once per lane per block (eight lanes share a
group); phase 2 (`moe_phase2_down_reduce_k8` and `_spec`) reads the down role the
same way. The head's fused greedy path (`lm_head_greedy_int4_rows_chunk_raw`,
`LMHeadChainInt4.swift`) and the logits head (`dequant_int4_gemv_simd`) read the
head's rows as nibbles. Prefill's grouped routed kernels
(`prefill_grouped_routed_moe_batched_phase1` and `_down`) and the tensor-core int4
QMM (`MPPPrefillInt4QMM.swift`, `tensorops.metal`) read the same nibbles at matrix
width. The shared expert, the GDN and the attention projections have their own
kernels over the same affine format and are outside the ruled scope.

## The design (a hypothesis, priced by step zero)

**One format on disk, chosen by the manifest.** The repack tool writes a coded
expert file beside a coded head under a new scheme string; the loader picks the
decoder per model from the manifest, so the plain and the coded builds of ornith15
load on the same binary and the golden compares them byte for byte on the same
box. That comparison is the chapter's class-1 gate: the coded model must produce
the plain model's bytes on every profile.

**The stride per layer.** A coded expert's size varies with its entropy, so the
fixed stride becomes a per-layer stride, the largest coded expert of the layer,
recorded in `layout.json`; the pool's slab, its slots and the ring's cells are per
layer already, and `expertStride` becomes a per-layer field where the read path
and the kernels take it. The worst expert of a layer bounds the whole layer's
saving, which S0.1 measures.

**The candidates for the code**, priced at S0.1 and benched at S0.4:

- **A. The entropy code with per-lane streams.** Each row's 2048 indices coded as
  32 streams of 64 symbols, one per lane, a static prefix code per layer over the
  16 levels (the order-0 entropy is 3.4 to 3.7 bits, the code within 0.1 of it),
  the 32 stream offsets in a row header. The full 12 %, less the header (about
  4 % of a row at 12-bit offsets), and a decoder that is a table lookup and a bit
  cursor per symbol instead of a shift and a mask.
- **B. The palette per group.** Each group of 64 indices stores the set of levels
  it uses as a 16-bit mask and its indices at the width that set needs, 1 to 4
  bits, the width known from the mask's population count. Random access stays
  block-aligned within a group, the decoder is a shift and a mask at a width the
  lane reads once per group, and the saving is whatever the groups' palettes
  allow. **Closed at S0.1:** the groups use all sixteen levels.
- **C. The aux table.** The scales and biases of a (layer, role) as indices into a
  table of the distinct bf16 patterns they use, 700 to 960 measured on two layers,
  so 10 bits each if every layer fits 1,024. Fixed width, one table lookup per group
  per lane, no bit cursor: about 4 % of the stride at a decoder cost of nothing,
  and composable with A or B.

**The decoder's home.** In the lane, decoding as it streams, is the design the
ruling requires: the SSD read, the pool's slot and the GEMV's stream all see the
coded bytes. Prefill decodes into its staging tile when the tile lands, a pass
over the bytes before the QMM that is small beside the QMM's compute, unless S0.4
finds a QMM variant worth its own site. Decoding at landing for decode is the
fallback the ruling closes the chapter on.

## Step zero: the board priced on the current tree

- **S0.1 The format candidates priced offline** (no run). Over the model file:
  the palette widths per group and the distinct scale and bias patterns per
  (layer, role) on layers 0, 5, 10, 20, 30 and 39 and the head; the bytes per
  expert under A, B, C and their combinations against the stride; the per-layer
  worst expert that sets the stride. Decides which code S0.4 benches.
- **S0.2 The pool's prize** (no run). The replay over the v19 and the Q3 traces at
  the slot counts the coded stride allows, uniform 128 → about 145 and the
  production table scaled by the same ratio: misses per token saved.
- **S0.3 The read-size probe** (the mini's drive, no Shrike code). `pread` of
  1.77, 1.60 and 1.50 MB from the model file with `F_NOCACHE`, p50 over a few
  hundred reads, serial and at the miss window's concurrency: the read row's
  prize.
- **S0.4 The in-lane decoder microbench** (a model-free kernel run on the mini,
  the gate). The production phase-1 kernel's shape at M = 1 over a layer's worth
  of experts in three variants, today's nibbles, the palette (B), the entropy
  streams (A), each with and without the aux table (C), on synthetic experts at
  the measured entropy; bytes streamed and microseconds per expert against the
  roof. The chapter is built only if a coded variant is faster than today's by at
  least its byte saving's share, on the mini.
- **S0.5 The record and the ruling** (for Davor).

### Step-zero record

**S0.1 The format candidates priced offline (measured over the model file,
2026-09-18; `v21-s01-format-pricing.py` and its JSON at
`~/.claude/handoffs/archive/shrike-v21-step0/`).** Layers 0, 5, 10, 20, 30 and
39, every expert's scales and biases and thirty-two experts' indices per layer.

- **B, the palette, is closed.** Every group of 64 indices uses all sixteen
  levels on every layer but layer 0: the width histogram is 4 bits for 100 % of
  groups on layers 5 to 39, so the mask costs more than it saves and the palette
  lands at 4.25 bits per weight, above the 4 stored. Layer 0 is the exception the
  v18 record's zero spike explains: 30 % of its groups are a single level, which a
  one-bit "constant group" flag would take to 3.33 bits per weight there, worth
  0.7 % of the model's expert bytes on one layer of forty. The levels' skew is in
  their frequencies, not in a sparse alphabet, which is the entropy code's case
  and no one else's.
- **C, the aux table, holds at 11 bits.** The distinct bf16 patterns per (layer,
  role) run from 902 to 1,454 over the six layers, so the scales and biases code
  in 10 or 11 bits each, 11 for the model: 3.5 % of the stride at a decoder cost
  of one fixed-width lookup per group per lane (`table only` 0.958 to 0.965 of
  the stride by layer).
- **A, the entropy code, is the only path to the 12 %**, and it pays a header.
  Per-lane streams of 64 symbols cost their lengths in a row header; at 8 bits per
  length (a prefix sum across the simdgroup turns lengths into offsets) that is 32
  bytes on a row of about 890 coded bytes, 3.6 %, so the indices save about 8.5 %
  of the stride's index bytes rather than 12. With C beside it the expert lands at
  about 0.89 of the stride, an 11 % saving (modelled from the v18 entropies; the
  exact figure is the coder's, at Task 1).

Reading: the code S0.4 benches is A with C, and the byte saving the gate is
judged against is 11 %, not 12. The per-layer worst expert is not yet counted;
the coder's first run at Task 1 records it.

**S0.2 The pool's prize (modelled by replay, 2026-09-18; grade T, the replay is
the model that priced v20's slot table within its measured move;
`v21-s02-pool-prize.py` and its JSON at the same archive).** Decode misses per
position on the eight v19 traces (the mini's own captures, two lifetimes per
shape) and the four Q3 traces (this box, one per shape), at today's slot counts
and at the counts the same bytes hold when an expert shrinks:

| pool | stride ratio | slots (uniform / table total) | misses per position | the cut |
| --- | ---: | ---: | ---: | ---: |
| bare (uniform, aging-LFU) | today | 128 / 5,120 | 27.6 to 32.2 | |
| bare | C only, 0.96 | 133 / 5,334 | 25.7 to 30.1 | 7 to 9 % |
| bare | A with C, 0.88 | 145 / 5,820 | 21.3 to 25.5 | 20 to 26 % |
| production (the table, SLRU 0.5) | today | 128 / 5,120 | 24.2 to 30.1 | |
| production | C only, 0.96 | 133 / 5,334 | 22.3 to 27.7 | 8 to 9 % |
| production | A with C, 0.88 | 145 / 5,820 | 18.5 to 21.8 | 22 to 29 % |

The replay's absolute count runs above production's 14 to 17 misses per token
(it holds no ring and no adopted landings), so the cut is the number to carry: a
22 to 29 % cut of 14 to 17 misses is 3 to 5 misses per token, at 0.73 to 0.80 ms
each about **2.5 to 3.5 ms per token** (modelled), the largest row on the board,
above the streaming row and the read row together. The pool's response to
capacity is steep because the working set sits just above the pool at 128: the
same shape v20's table lever found by moving slots rather than adding them.

**S0.3 The read-size probe (measured on the mini's drive, 2026-09-18, three
seeds of 300 serial `pread`s each, `F_NOCACHE`, random experts across the forty
layer files, production up but idle; `ssd-size-probe.c` and the runs at the same
archive).** p50 per read: the plain stride 0.764 to 0.771 ms; a read of 0.96 of
it 0.733 to 0.742; of 0.89, 0.692 to 0.708; of 0.875, 0.690 to 0.699; a half
0.474 to 0.477; a quarter 0.307 to 0.341. The drive scales a single read with its
size: about 0.16 ms fixed and 0.35 ms per MB (modelled from the four sizes), so
the coded read at 0.89 saves 0.065 to 0.07 ms per miss, **0.9 to 1.2 ms per
token** at 14 to 17 misses (modelled), before the pool row lowers the count.

Reading: the pool's prize exists only when the coded bytes sit in the pool, which
is the in-lane design; the fallback that decodes at landing holds plain experts
and wins none of it. The in-lane design's board is now the read row (1.2 to
1.5 ms), the pool row (2.5 to 3.5 ms) and the streaming row (about 1.5 ms in the
ruled scope), about 5 to 6 ms per token, 10 % of the token, before the decoder's
cost; the fallback is the read row alone, about 1.2 ms. C alone, with its
fixed-width decoder, would be about 1.5 ms across the three rows. (S0.4 below
found the bytes at 4 %, not 11, which scales every row down by about 2.5.)

**S0.4 The in-lane decoder microbench (measured on the mini and on the M4 Pro,
2026-09-18; `ShrikeExpertBench`, a new executable target that stays in the tree;
the runs at the same archive).** The bench loads eight real experts of a layer
from the `.gturbo`, runs the production phase-1 kernel
(`moe_phase1_gate_up_act_u16load` at the production function constants, through
the context's own library) as the plain arm, and a coded kernel compiled behind
`moe.metal`'s source with the production kernel's arithmetic in the production
kernel's order: lane l of a row holds, for every block of 256 weights, the eight
weights the plain lane reads, so the fma chains see the same operands in the
same order and the arms compare bit for bit. The code is A with per-lane byte
streams (a canonical prefix code limited to eight bits, a 256-entry table, 32
stream lengths per row and a row directory) and C as 12-bit indices into
per-role tables; every expert of the layer is laid out at the section offsets
the layer's largest expert needs, the per-layer stride of the design. Twenty
dispatches per command buffer so the GPU holds its clock (a single dispatch per
buffer read bimodally, 3 to 4× apart, on the M4 Pro), the median of fifteen.

| box | layer | plain µs (GB/s) | coded µs (GB/s), × plain | coded with aux µs, × plain | bytes, coded / with aux | mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| the mini | 20 | 149.8 (63.0) | 1,034.9 (9.0), 6.9× | 1,154.8, 7.7× | 0.986 / 0.958 | 0 / 0 |
| the mini | 39 | 152.9 (61.7) | 1,029.2 (9.0), 6.7× | 1,139.4, 7.5× | 0.985 / 0.958 | 0 / 0 |
| the mini | 0 | 155.9 (60.5) | 821.6 (9.9), 5.3× | 955.4, 6.1× | 0.862 / 0.834 | 0 / 0 |
| the M4 Pro | 20 | 35.8 (263.2) | 145.6 (63.9), 4.1× | 172.5, 4.8× | 0.986 / 0.958 | 0 / 0 |
| the M4 Pro | 39 | 42.3 (223.3) | 134.7 (69.0), 3.2× | 173.0, 4.1× | 0.985 / 0.958 | 0 / 0 |
| the M4 Pro | 0 | 51.9 (182.0) | 122.7 (66.3), 2.4× | 158.3, 3.1× | 0.862 / 0.834 | 0 / 0 |

Three findings, each enough on its own.

- **The decoder is not free by an order of magnitude.** On the mini the plain
  kernel runs at the streaming roof (60.5 to 63.0 GB/s against v10's 62.5), so
  it is bandwidth-bound as the design assumed, and the coded kernel runs at 9 to
  10 GB/s: its lanes decode at about 19 billion symbols per second on that GPU
  where the roof asks for about 110. The in-lane decoder would turn the ten
  milliseconds of expert streaming per token into sixty. The aux table's
  fixed-width lookups cost another 0.8 of a plain pass on their own (the
  difference between the two coded arms), so C alone is not free either.
- **The bytes are 4 %, not 11.** The prefix code lands at 3.795 bits per index
  on layers 20 and 39 (their order-0 entropy is 3.71), a 5 % saving on the
  indices before framing; the 32 stream lengths and the byte padding cost 4.7 %
  of a row, so the coded indices come out at 0.986 of plain and the whole
  phase-1 expert at 0.958 with the 12-bit aux table. Only layer 0, with its zero
  spike, codes to 0.83 to 0.86, and there the layer's largest expert exceeds the
  plain stride (1.04 to 1.07), because one code table serves experts whose
  histograms differ. The design's 11 % double-counted: the v18 entropy figure's
  12 % was 7 % from the indices at their entropy, without framing, plus 5 % from
  the scales and biases at theirs, which a fixed-width table does not reach.
- **The arithmetic is class 1.** Zero mismatches on every arm, layer and box:
  the coded kernel reproduces the production kernel's half outputs bit for bit,
  which is the property a production decoder would have needed and the one
  thing this bench did not have to invent twice.

**S0.5 The record and the ruling (2026-09-18, for Davor).** The board as
measured: the bytes an in-lane code saves on the typical layer are about 4 %
(the aux table 3.5 to 4, the indices 1 after framing), so the pool row is about
8 % fewer misses (S0.2's C-only arm), the read row about 0.4 ms, the streaming
row about 0.4 ms, roughly 1.5 ms per token in all if the decoder were free; and
the decoder costs six to seven plain passes on the mini. Two independent
reasons, either sufficient under the ruling this chapter opened under.
**Recommendation: close v21 at step zero.** Nothing on the tree changes: the
bench stays as the instrument that runs the production phase-1 kernel on real
experts at the served shape (the someday item that ShrikeBench's MoE mode could
not measure production is answered by it), the documents record the numbers,
and the H2 avenue is closed on the board: lossless coding of this format cannot
pay on this GPU, because the indices are within 8 % of their entropy and any
decoder that recovers it runs an order of magnitude below the roof.

**Davor's ruling (2026-09-18): closed at step zero.** The three venues that
reach the same prizes are all to be tried, in the order of Claude's choosing:
the pool's capacity by freeing RAM (S0.2's slope, a RAM ledger on the mini
first), speculation revisited with the drafter's measured acceptance (H1, Q3's
82 to 86 %), and the attention row's fixed part (B3, B4) with its instrument. **The pool's capacity is done (v22, v22-pool-capacity.md, noted 2026-09-23); speculation revisited is filed in tt as SHRIKE-16 and the attention row's fixed part as SHRIKE-19 (2026-09-23).**

## The chapter's close (2026-09-18)

Closed at step zero, the same day it opened. No runtime code changed. What the
chapter leaves: `ShrikeExpertBench`, the bench that runs the production phase-1
kernel on real experts at the served shape beside any variant (the plain arm is
the production pipeline itself; a variant is held to bit-identity); the
step-zero records S0.1 to S0.4 with their scripts, probes and runs archived at
`~/.claude/handoffs/archive/shrike-v21-step0/`; the H2 entry on the board closed
with the numbers; and two facts for later levers: the mini's drive scales a
single read with its size (0.16 ms fixed, 0.35 ms per MB), and the pool's
misses fall about twice as fast as its capacity grows.

What it cost: one day. What it taught: price a code's framing and its
inefficiency against the entropy before the entropy is carried as a prize; a
GEMV at the roof has no room for a per-lane dependency chain, however short;
and a bench that holds a variant to bit-identity against the production
pipeline is cheap to build and settles the numerics question before the
performance one.

## Approaches for the decoder

Held until S0.1 names the code; the candidates' decoders are sketched above. **Superseded: the chapter closed at step zero by the ruling of 2026-09-18 (noted 2026-09-23).**

## Tasks

Held until S0.5. The expected shape, if the gate opens: Task 1 the repack and the
loader (the coded expert file and head, the per-layer stride, the scheme gate, the
golden on both builds); Task 2 the decode kernels (phase 1, phase 2, their
variants); Task 3 the head; Task 4 prefill's landing decode; Task 5 the dense
weights only by S0.4's number; the close. **Superseded: the gate did not open and the chapter closed at step zero by the ruling of 2026-09-18 (noted 2026-09-23).**

## Method

As v20's: every task's arms on the mini, two production lifetimes per shape on the
four shapes, read against the previous task's arms; the rows pre-registered; the
four gates per commit; the golden byte-identical on the plain build at every
commit and on the coded build once it exists; ThreadSanitizer once at the close.

## Numerics policy

Class 1 throughout, by construction: the decoder returns the index the kernel
reads today, and the scales and biases come back as the same bf16 patterns. A
coded model that differs from the plain one in any byte of any profile is a bug in
the coder or the decoder, never a numerics decision.

## Out of scope, and where it goes

- **The dense weights** (GDN, attention, the shared expert): by S0.4's number. **Superseded: S0.4 failed the gate and H2 closed at step zero (noted 2026-09-23).**
- **KV compression**: the KV row is already 8-bit affine with `--kv-bits 4|16`
  on the shared kernel; a lossless code over it is another chapter's question. **Closed 2026-09-23 with no tt entry, by the owner's ruling: even with the scan at its byte roof (0.17 ms per 1,000 context per token; measured 0.39 after v19) the whole KV read is about 1.2 ms per token at 7k and 5.4 ms at 32k, int8 with a per-64 affine range leaves only a few percent of entropy slack (an estimate; v21 measured about 4 % on the similar expert format), so a code saves at most about 0.3 ms at 32k, under the rig's 1.7 % noise, before a decoder v21 measured at 5 to 7× slower, and a few percent of the KV's 356 MB at 32k is a quarter of a slot. Only fewer bits move KV bytes, and that is class 3 (closed; `--kv-bits 4` exists).**
- **Lossy requantization** of any kind: not this chapter, not this model.
- **The ANE**: unchanged.

## Risks

- **The decoder is not free.** The gate. A GEMV that today does eight
  multiply-adds per four bytes may hide a table lookup per symbol under its memory
  stall or may not; the expert kernels already run at 1.6 times the roof, so they
  are not purely bandwidth-bound and a byte saved there converts less than fully.
- **The worst expert sets the stride.** A layer's saving is its largest coded
  expert's, not its mean's; S0.1 measures the spread.
- **The header's overhead.** Per-lane stream offsets cost bits; a code that saves
  12 % at the symbol level may save 8 after the header.
- **The blast radius of a format.** The repack, `layout.json`, the receipts, the
  loader's scheme gate, every reader; the mini's other five `.gturbo`s stay in the
  plain format and must keep loading.
- **Prefill's QMM** reads nibbles at matrix width; a landing decode is the plan,
  and a QMM variant would be a site of its own.
