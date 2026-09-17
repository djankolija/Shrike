# v20: the SSD mechanism

The chapter that takes the expert reads off the token's critical path where the
board says they can go, and rebuilds the host's side of the miss path as the
structure the fold needs. Companion plan:
[v20-implementation-plan.md](v20-implementation-plan.md), whose checkboxes are the
status of record. The board this chapter was chosen from is
[v18-avenues.md](v18-avenues.md) (section A for the surface, section 5 for the
rulings of 2026-09-09 that the SSD mechanism follows the scan rewrite and carries
the fold as structure); the tree it starts from is `main` at `c4a96d6`, v19's close
([v19-scan-rewrite.md](v19-scan-rewrite.md)).

Every number in this document is labelled **measured** (a counter or a clock on the
mini), **modelled** (arithmetic on measured inputs) or **remembered** (an earlier
chapter's record, cited, not re-run), and carries the grade of the Method (M, T, C,
R) where it is load-bearing.

## The problem

Forty of the served model's layers are routed: each picks eight experts of the
layer's set for the token, and an expert not resident in the layer's pool is read
from the SSD while the GPU waits. The wait is the miss window. On the mini after
v19 the token is about 58 ms on every shape and the window's io is 13.7 to 15.1 ms
of it (measured, the v19 ledger's `io ms` column at the v18 close's build, expected
and recorded flat through v19; [v19-scan-rewrite.md](v19-scan-rewrite.md), step
zero and Task 3): the largest row left, about a quarter of the token, and flat with
context, so it is the same quarter on the 7k as on the 300.

The current tree on the mini (remembered, v19 Task 3's arms at `3042665f2fb11370`
and the close's deploy, two production lifetimes per shape):

| shape | the token ms | tok/s | misses per token | io ms per token |
| --- | ---: | ---: | ---: | ---: |
| the 300 | 57.4 to 58.5 | 17.1 to 17.4 | 20.0 | 14.9 to 15.1 |
| the 1k | 57.4 to 57.5 | 17.4 | 18.9 | 14.0 |
| the card (2k) | 57.8 to 58.7 | 17.0 to 17.3 | 20.1 | 14.7 |
| the 7k | 58.5 to 58.8 | 17.0 to 17.1 | 18.5 | 13.7 |

The misses and io columns are the v19 step-zero ledger's (the v18 close's build);
v19 changed nothing on the miss path and its arms read those rows within their
spreads.

**What the window is made of** (modelled from measured inputs, 2026-09-09,
[v18-avenues.md](v18-avenues.md) section 3). One expert is 1.77 MB. The drive's
decode ceiling is 3.5 GB/s (v14, measured), so one transfer is 0.51 ms. Twenty
demand reads per token are 10.1 ms of transfer against a window of 13.1: the rest is
2.1 ms of dead time after the bytes land and about 1 ms of per-layer latency and
issue. The window is bandwidth-bound while it is open and the drive is idle for the
other two thirds of the token, during the GPU-busy stretches. So no change to the
wake, the threads, the placement or the join takes the window much below 10 ms while
the twenty demand reads stay inside it; what moves the floor is **lead** (reads
issued during the idle two thirds), **fewer misses** (a better policy, a better
split), and, in v21, fewer bytes per read.

**What the ring does today, per token on the 300** (measured, the runner's
counters, v18 step zero):

| | count |
| --- | ---: |
| non-resident experts in the actual top-8 (misses plus landed hits) | 26.2 |
| predictions issued | 21.0 |
| of which right (adopted) | 10.3 |
| of which right and landed before the classifier (a phase-1 hit) | 6.2 |
| of which right but late (joined by the fixup) | 4.1 |
| of which wrong (reclaimed) | 10.7 |
| predictions refused for lack of a cell | 16.9 |
| misses still waited on | 20.0 |

Two readings. The scout is right about half the time and lands about a quarter of
what is needed, so the window is three quarters unserved. And the ring is already
cell-starved: 16.9 predictions per token are refused because no cell is free. Cells
are the second currency of this chapter beside the drive's reads.

**The drive's budget, the chapter's currency.** Reads today are 20 demand plus 21
speculative per token, 72.6 MB per 60.9 ms token, 1.19 GB/s against the 3.5 GB/s
ceiling: reads can rise about 2.9× before the drive is the wall (modelled). At 0.51
ms a read the ceiling is about 115 reads per token at the 58 ms token. Every lever on
this surface spends reads, and the design allocates them: a wider scout, a token of
lead, and the demand reads themselves all draw on the same 115.

**The clairvoyant bound** (measured by replay, v18 step zero S0.4, pool mode without
the ring): aging-LFU, the shipped policy, misses 30.2 / 28.2 / 30.7 per token on the
300 / the 1k / the card; the best online alternatives (SLRU, LRU) 25.5 to 28.4 on
the longer shapes and nothing better on the 300; **Belady 11.9 / 10.2 / 10.8**. No
heuristic is worth more than about a miss per token; the clairvoyant bound is 2.5 to
2.8× below every online policy. The prize in this surface is knowledge of the next
tokens' routes, not a better guess about the past.

**The miss profile by depth** (measured by replay, same source): U-shaped. Layers
0-3 carry 22 % of the demand over 2.7 reading layers; 4-9 18 %; 10-19 19 %; 20-29
17 %; 30-39 25 % over 4.8 reading layers. Layer 0 alone misses on 66 % of positions,
the highest of any layer, and no probe serves it (the probe predicts L+1 from L).

**Routing locality** (measured, v18 step zero S0.2 over six T4 traces, every decode
position matched to its streamed token; the script at
`~/.claude/handoffs/archive/shrike-v18-step0/s02-routing-locality.py`). Top-8
overlap per layer group 0-9 / 10-19 / 20-29 / 30-39 / all: consecutive positions
0.23 / 0.32 / 0.35 / 0.35 / 0.31; the same token text at two positions 0.35 / 0.31
/ 0.29 / 0.31 / 0.31, and 0.61 / 0.47 / 0.42 / 0.41 / 0.48 for texts of three or
more non-space characters; random pairs 0.07 / 0.13 / 0.15 / 0.11 / 0.11. At layers
0-3 identity beats the previous position three to one (0.49 against 0.16; at layer
0, 0.63 against 0.07). The last-occurrence predictor recalls 0.44 / 0.37 / 0.34 /
0.38 / 0.38 of the actual top-8 and covers 56 / 64 / 70 % of positions (the card /
the 300 / the 1k); the union of the last two occurrences 0.48 at 12 predicted, the
last three 0.54 at 15; a frequency table's top-12 0.47, top-16 0.52. Identity
locality decays with distance (0.38 within ten positions to 0.23 past two hundred).
The reading: a token-id table is a real predictor, 3.5× the random baseline, weaker
per expert than the hidden-state probe at distance one but with a lead the probe
cannot have; and from layer 10 on the identity's share and the neighbour's share
are each about a third and largely complementary.

## What the mechanism does today (read at `c4a96d6`)

**The probe and the readback.** At layer L the fused router kernel runs layer L's
router and, from the same hidden state, layer L+1's (the second operand set,
`RealForwardRunner.swift:2679-2686`); the classifier publishes the top-k ids,
weights, hit and miss positions and the probe's predicted ids as tagged words the
host polls (`moe.metal:152-175`, `RouterHostReadback.swift:19-62`). The arrays are
eight wide: `top_k` may be 1 to 8 and "k8" is the capacity (`moe.metal:469`), so
the probe's ranking past its eighth candidate exists in the kernel and is never
written anywhere.

**The ring.** `ExpertPrefetchRing` owns the top-k plus one cells, nine on this
model, and at most one read in flight across all layers
(`ExpertPrefetchRing.swift:39-46`, `RealForwardRunner.swift:421`). At layer L's wake
the probe's top-8 for L+1 is issued after the demand submission, one layer ahead; a
landing is published `resident` from the storage thread and layer L+1's classifier
can hit it (the race v16 measured and won); the plan swaps a wanted landing into the
pool by index, the reclaim drops an unwanted one when the ring needs the cell. The
arena is one address space for every cell the classifier can name, the layers'
slots plus the ring's nine, so a cell changes owner without a byte moving
([architecture.md](architecture.md), "The arena and the ring"). The distance above
one and the in-flight budget above one are closed levers (v15, measured null).

**The miss path.** On the word the host builds the fixup for the layer's misses,
issues their reads, and the fixup command waits on the reads' event; the plan
(victims, the swap) runs on the path. v18's Task 2 priced moving the plan off the
path at most 0.34 ms per token and skipped it as a performance task; its step list
survives as the agreed-cell mechanism for the fold
([v18-quiet-host.md](v18-quiet-host.md), Task 2 and Task 5).

**The instruments.** `SHRIKE_ROUTE_TRACE` writes one line per plan the pool
received (decode: `position layer e0 .. e7`; prefill: a `p` line per tile; a
request's start as `r cachedTokens promptTokens`), `SHRIKE_PREFETCH_TRACE` one JSON
line per decode plan with the probe's top-8 (`RealForwardRunner.swift:1456-1463`);
the decode rig sets both (`tools/decode-rig.sh:101-103`).
`tools/expert-pool-replay.py` reproduces production's misses from a route trace
within 0.2 per token (`--fill-mode ring-retain`) and takes fills, a
`(layer, position) -> predicted experts` map, placed before the plan at that
position (`:428`, `:1066-1076`). `tools/prefetch-coverage.py` joins a prefetch
capture to itself and reports coverage, precision and wasted reads per width. The
archived traces on the current tree's ancestor: eight lifetimes at the v18 close's
build, two per shape including the 7k, with their token streams
(`~/.claude/handoffs/archive/shrike-v19-step0/arms/route-v19s01-base-*.trace`,
`tokens-v19s01-base-*.json`); six at v17's Task 4
(`~/.claude/handoffs/archive/shrike-v17-t4/v17t4-arms/`).

## The design (a hypothesis, priced by step zero)

Three levers on the same budget, and one structure.

**Lead: the token-id table.** Per layer, keyed by token id, the eight that id routed
to at its most recent occurrence (or the union of its last two or three). It fills
from the session's own routes, first from the prompt's prefill if that seeding is
priced worth it, then from every decoded token; no training, no data shipped, a few
kilobytes per layer. The next token's id is known at the end of the pass when the
sampler picks it, so a prediction from it has a lead of about L milliseconds at
layer L (the pass runs its forty layers in order over about 58 ms): nothing at layer
0, thirty at layer 30. With that lead the drive's latency is hidden and only its
bandwidth binds, and the reads for the deep layers go out as one batch at the
token boundary, landing in cells while the GPU works through the shallow layers.
Selective by construction: at full width on forty layers the table would issue
about 300 reads per token against the 115, so it serves the layers where the lead
is long and the misses are many (30-39, a quarter of the demand) and layer 0, which
misses on two thirds of positions and which the probe cannot serve. The union with
the current token's route at the same layer (known a pass ahead for the same reason)
is priced beside it.

**Lead for layer 0: the draft.** Layer 0 runs the moment the id is known, so the
table has no lead there. A draft for token t+2 during pass t+1 gives it one. The
draft is prompt lookup, an n-gram match over the context on the CPU (the last few
ids matched against the prompt and the answer so far, the token that followed the
match proposed), no model, no GPU, strong where the answer copies spans of the
context, which is what code and tool turns do; llama.cpp ships the same idea as
prompt lookup decoding (remembered; cite the source in Task 1's record). When the
draft is right the table's entry for it is layer 0's route a pass early; when it is
wrong the entry is another token's and the overlap falls toward the random line. The
draft is priced in the table form, never through a forward pass: a draft's true
route costs a pass, and demanding the draft's experts during it is v12's speculation,
whose reads moved earlier onto the same critical path rather than off it (P17,
remembered). The MTP head stays out of the runtime.

**Width: the probe past eight.** Full coverage of a layer is what stops its serial
read, and at width eight the probe covers all of a miss layer's misses 43 to 46 % of
the time (measured). The router ranks every expert; keeping its ninth to
twenty-fourth candidates is a wider net that should hold the real eight more often,
since the probe's state drifts 20 to 27 % per layer and the displaced experts sit
near the top of the ranking rather than anywhere in it (a hypothesis). Width is paid
in wasted reads on the shared drive: full coverage at width sixteen would need about
320 reads per token, 2.4× the ceiling, so the question is the marginal precision of
ranks nine to twenty-four, unknown until they are logged.

**Knowledge in the policy.** The replay's Belady path fed a predicted future (the
table's entries for the tokens prompt lookup names next) instead of the real one
prices what a policy that evicts against a predicted future is worth over aging-LFU,
at zero reads. The slot split (A1) moves slots from the middle layers to both ends
of the U.

**The structure: the agreed cells and the fold.** The fixup encoded before the route
is known, its misses read into cells named in advance (miss i into cell i), the
host's on-word job reduced to the preads and the event, the plan moved to the next
wake; then one command per token with two in flight, the host feeding reads and
signalling events. The architecture is the win and any speedup a bonus (0.6 to 0.85
ms per token modelled, under the arms' resolution). Built last, on the batch's
measured shape, and designed on its four edges before it is built: the stop path
against the GDN state a committed pass mutates in place, the error surfacing per
layer when a token is one command, the agreed-cell contract encoded before the
router has run, the cancel with two in flight (Davor's ruling, 2026-09-09).

**Why the order is predictor, policy, fold.** The predictor is the prize and does
not need the fold: the ring already lands predictions in cells and publishes them
resident before the classifier runs. The fold's hardest edge, the cancel with two
commands in flight, has to drop the batch's reads in flight, so it is designed
knowing the batch's shape. Everything is class 1, so the bitwise golden gates every
commit.

## Step zero: the board priced on the current tree

Measurement and tooling. Two model runs, both marked; the rest reads archived data
or runs the replay. Every arm reports **reads per token** and **cells held** beside
misses saved, because those are what the design allocates.

- **S0.1 The replay's table mode.** `tools/expert-pool-replay.py` gains a fills
  source computed from the trace and its token stream: `--table-fills TOKENS.json`
  with `--table-layers` (a layer list or ranges), `--table-width` (8, 12, 15, 16),
  `--table-source` (`last`, `last2`, `last3`, `freq`), `--union-previous` (the
  current token's route at the same layer added to the prediction), and
  `--draft prompt-lookup:N` (the prediction keyed by the draft's id where the draft
  exists, by the real id where it does not, with the draft's hit rate reported). The
  prompt's ids for the draft come from the tokenizer (`GFTokenizer.encode`,
  `Tokenizer.swift:568`) through a small `--tokenize` path in the CLI. Fills go
  through the existing `ring` mode with a cell budget per layer, and the tool reports
  fills, useful, wasted, reads per token and the peak cells held per layer. Self-tests
  extended for every switch.
- **S0.2 The table as fills** (A9's design question). On the eight v19 traces, four
  shapes: misses saved per token and reads per token, by layer group, width and
  source; the selective set (layer 0 and 30-39) against every layer; the union
  predictor. Decides whether the table is built, at which layers and width, and what
  it costs in reads and cells.
- **S0.3 The draft** (Q3, the table form). Prompt lookup's proposal rate and hit
  rate per shape at N of 2, 3 and 4; the table keyed on the draft against keyed on
  the real token at layer 0 and at every layer; misses saved at layer 0 per token.
  Decides whether the draft is built for layer 0.
- **S0.4 The policy and the split** (A2's variant, A1). The predicted-future
  Belady: the replay's Belady path fed the table's entries for prompt lookup's next
  tokens, against aging-LFU and the real-future bound. The slot split: a per-layer
  slot count drawn from the U-shaped profile (the middle layers' spare slots to
  both ends), the total held. Decides Task 2.
- **S0.5 The wide capture** (**a model run on the mini**, one production lifetime
  per shape, four shapes, Shrike stopped for the duration, Davor's leave given
  2026-09-17). A diagnostic, off unless `SHRIKE_PREFETCH_TRACE` is set: the probe's
  full router scores per position and layer written beside the top-8 (the fused
  router's second operand set given an optional scores output, or the ranking
  widened on a diagnostic path; the golden proves production inert), and the
  prefill's per-token top-8 per layer written as a new trace line kind the replay
  and the locality script read. Then `tools/prefetch-coverage.py --top-m 8 12 16
  24`: coverage, precision and wasted reads per token at each width against the
  115; and the table's coverage with the prompt seeded, from the prefill lines.
  Decides whether the wider probe is built and at which width, and whether the
  table seeds from the prompt. The same lifetimes are the chapter's opening ledger
  on the current tree.
- **S0.6 The attention row's fixed part** (B3, B4; **a model run on the mini**, the
  rig with `SHRIKE_KERNEL_STATS=1`, one lifetime on the 300). The attention layers'
  held command split by kernel: the projections, the KV quantize, the scan, the
  combine, the layer's speculative routed work, and the remainder as walls. Prices
  B3 (the folds into neighbours) and B4 (one pass at short context) on the current
  tree; the v18 ledger's 2.4 ms of walls predates v18's one command per layer.
  Decides whether B3 is a v20 task, a later chapter, or nothing.
- **S0.7 The record and the ruling.** The pricing in this document; the read budget
  allocated among the wider probe, the table and the demand reads; Task 1's shape
  (the table, the probe, both, or neither); Task 2's go or no-go; the fold's design
  note scheduled; B3's home. Davor's ruling on each.

### Step-zero record

**S0.1 (2026-09-17).** `tools/expert-pool-replay.py` gained the table mode:
`--table-fills`, `--table-layers`, `--table-width`, `--table-source
last|last2|last3|freq|none`, `--table-cells`, `--union-previous`, `--draft
prompt-lookup:N` with `--draft-layers` and `--prompt-pieces`, `--table-seed
prefill`, `--table-future` (the clairvoyant path on the table's predictions),
`--table-protect` (the online horizon-one form), `--slots-json` (per-layer slot
counts); fills as `(candidates, budget)` pairs so two sources keep their own
budgets; the `q` and `t` line kinds parsed and dropped from the plans; the report
by layer group with fills, useful, wasted, misses, reads per position and the
cells at the pass start; misses and miss layers per (request, layer) in the fill
stats; the self-tests extended. The table's fills follow production's
ring-retain profile (a wanted landing is swapped into the pool). `ShrikeCLI
--tokenize <path>` renders the prompt as a run would and writes its ids and
pieces without loading the model; the four rig prompts tokenize to exactly the
counts the traces' `r` lines carry (289, 1,069, 2,125, 7,463). The four gates
passed on the CLI change (1,259 tests in 174 suites). Scripts, outputs and the
prompt pieces at `~/.claude/handoffs/archive/shrike-v20-step0/`.

The table is keyed by the streamed piece, the s02 alignment (the pieces and the
first request's decode positions are equal in count and matched by index; the
piece is the pass's input token). Costs are modelled from the v18 ledger: a saved
miss layer 1.15 ms (the first miss's latency, transfer and wake), a saved further
miss 0.65 ms (transfer only). Two bases: the eight v19 traces (four shapes, two
lifetimes each, the v18 close's build) against the replay's pool without the
ring, and the three v14 Task 1 captures (the card, the 300, the 1k, one
lifetime each, the probe's jsonl beside the route trace) against the probe at its
in-flight budget of one, which reproduces production's misses (29.7 to 19.5 per
position on those captures, production's 20).

**S0.2 The table as fills (measured by replay, 2026-09-17).** Over the pool, per
position, the eight lifetimes' mean:

| served layers, source | saved | fills | useful | precision | reads | modelled ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| layer 0, last | 0.33 | 0.51 | 0.33 | 0.64 | 29.3 | 0.26 |
| layers 0-3, last | 0.74 | 1.83 | 0.73 | 0.40 | 30.2 | 0.57 |
| layers 30-39, last | 0.21 | 1.19 | 0.21 | 0.18 | 30.1 | 0.19 |
| 0 and 30-39, last | 0.55 | 1.71 | 0.54 | 0.32 | 30.3 | 0.45 |
| 0 and 30-39, last3 at 16 | 0.66 | 3.08 | 0.66 | 0.21 | 31.5 | 0.55 |
| all forty, last | 1.34 | 5.14 | 1.34 | 0.26 | 32.9 | 1.11 |
| 0 and 30-39, previous position only | 0.00 | 0.15 | 0.00 | 0.02 | 29.2 | 0.00 |

The baseline is 29.1 misses per position. Over the probe on the v14 captures the
probe itself saves 10.2 misses per position (5.2 miss layers) for 20.8 fills at
precision 0.49, 9.3 ms modelled; the table on top of it: layer 0 0.31 saved (0.24
ms, precision 0.70 on its own fills), layers 30-39 0.10, both 0.41 (0.33 ms), all
forty 0.85 (0.71 ms) for 4.2 more fills per position and 25 cells at the pass
start (74 at the peak). The union with the previous position is a null by
construction: the previous pass's experts are still resident, so the union adds
0.15 fills and saves nothing; the pool already holds the neighbour's route. The
wider sources buy a fifth more at twice the fills. **Reading:** the table is real
and small. Its best form, every layer at width eight, is worth 0.7 ms per token
over production (1.2 %, modelled), under the arms' resolution; the selective set
0.33 ms. Layer 0 is the one place it is precise.

**S0.3 The draft (measured by replay, 2026-09-17).** Prompt lookup over the
prompt's pieces plus the answer so far proposes on 0.44 of positions at n = 2 with
a hit rate of 0.39, 0.25 at 0.46 for n = 3, 0.15 at 0.42 for n = 4; the 7k, the
shape with the most context to copy from, 0.39 at 0.38. The table keyed on the
draft at layer 0 saves 0.05 misses per position (0.05 ms) against 0.33 keyed on
the real token, precision 0.23 against 0.65; at layers 0-3, 0.19 against 0.74.
Over the probe the same. **Reading:** the draft path is closed for this chapter.
Its ceiling was the real-key number, 0.26 ms, and the draft reaches a fifth of it.
The rig's prompts are synthetic ledgers; a tool turn that copies more might
propose more, but the hit rate, not the proposal rate, is what limits it.

**S0.4 The policy and the split (measured by replay, 2026-09-17).** Decode misses
per position on the v19 traces, the eight lifetimes' mean, the pool without the
ring: aging-LFU 29.09 (shipped); LRU 28.17; SLRU at 0.5 27.77; Belady 10.69. Belady
fed the table's predictions as its future reaches 17.76, but that form leaks the
future tokens' identities to the policy and is a bound, not a policy. The online
form at a horizon of one, the next token's table entry protecting its experts
from eviction at the plan, saves 0.06 per position keyed on the real next token
(the ceiling, every plan deferred past the sampler) and 0.02 keyed on the draft:
null. Knowledge in the policy is closed.

The split is the surface's lever. One allocation derived from the 300's first
lifetime (its aging-LFU miss profile per layer, all requests, blended half-way
between uniform and proportional, 94 to 210 slots per layer at the same total of
5,120) applied to every trace:

| basis | trace | misses per position | saved | modelled ms |
| --- | --- | ---: | ---: | ---: |
| the pool (v19) | the card | 30.68 | 3.34 | 2.20 |
| | the 300 (in sample) | 30.25 | 2.06 | 1.30 |
| | the 1k | 28.21 | 1.77 | 1.03 |
| | the 7k | 27.22 | 4.16 | 3.13 |
| the probe (v14) | the card | 19.97 | 3.65 | 2.62 |
| | the 300 | 19.84 | 2.76 | 2.06 |
| | the 1k | 18.72 | 2.44 | 1.76 |

Over the probe the mean is 2.95 of 19.5 misses per position, 15 %, 2.15 ms
modelled; the blends at 0.35 and 0.65 are within a tenth of it, proportional
alone is worse than uniform (the middle layers starve). The allocation per layer
0 to 39: 210 201 207 175 167 148 145 131 147 118 132 143 117 109 109 104 96 103
94 96 103 99 110 121 106 111 99 115 116 114 103 107 137 125 133 125 125 126 143
150. The split combined with the horizon-one protect adds nothing over the split.
**Reading:** the U-shaped profile is stable enough across shapes that a fixed
per-layer allocation transfers, and the same memory serves 15 % fewer misses at
zero reads and zero cells. SLRU is worth about a miss per position on the longer
shapes and nothing on the 300, as v18 found.

**S0.5 The wide capture (measured on the mini, 2026-09-17, two runs).** The
diagnostic, off unless `SHRIKE_PREFETCH_TRACE` names a file: the probe router's
scores kept per layer in two banks by the position's parity (`MoE.probeLogitsPair`,
128 slots of 256 floats, the pair GEMV and the select reading and writing the
layer's slot, so production runs the same kernels at a different offset), the
previous position's rankings to width 32 written one JSON line per layer once its
commands have completed, in the select kernel's order (the logit plus the bias,
descending, the lower index on a tie); `SHRIKE_ROUTE_TRACE` gaining `t position id`
per decode position from the loop and `q position layer e0..` per prefill row from
the readback the tile planner already makes. The golden identical on all four
profiles on both boxes with the variables unset. The first run wrote no rankings:
the pending position was set only on the plain head path and the server takes the
boundary path, whose held layer zero would also have overwritten slot 0 before the
read; the second bank and the pending on both paths fixed it, the golden identical
again, the second run carrying 9,282 / 14,586 / 11,895 / 13,962 ranking rows (the
card, the 300, the 1k, the 7k) of which none has a top-8 differing from its plan
row's, the coverage tool's stale-slot check. The two runs' answers are identical
token for token; the lifetimes decode at 16.5 to 16.8 tok/s with all three traces
on (16.6 and 16.8 on the 7k's two runs), about twenty misses per token, io 14.7 ms
on the card: the diagnostic's cost in a lifetime is inside the drift. The captures at
`~/.claude/handoffs/archive/shrike-v20-step0/capture/`, the pricing at
`s05-price.md` beside them.

The probe's ranking past eight, joined to the target layer's own misses (the
misses left after the ring has landed what it lands), per width, the four shapes
within 0.02 of each other:

| width | per-miss recall | precision | full-layer coverage | reads per token |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.30 to 0.33 | 0.15 to 0.16 | 0.20 to 0.21 | 28 to 30 |
| 12 | 0.54 to 0.57 | 0.13 to 0.14 | 0.47 to 0.48 | 59 to 62 |
| 16 | 0.67 to 0.69 | 0.10 | 0.61 to 0.62 | 102 to 106 |
| 24 | 0.80 to 0.82 | 0.06 | 0.76 to 0.77 | 207 to 217 |
| 32 | 0.86 to 0.87 | 0.04 | 0.83 to 0.84 | 327 to 349 |

(The reads column is the offline scheme's non-resident predictions per token; the
ring's actual reads at its in-flight budget of one are 21.) **Reading:** the
ranking carries real information past eight, the displaced experts do sit near the
top as the hypothesis said, coverage of the remaining miss layers more than
doubling at width twelve and reaching five sixths at thirty-two. And it cannot be
spent. Ranks nine to twelve buy about five more useful reads per token for
thirty-two more wasted ones, a marginal precision of 0.13, so each saved miss
(0.65 to 1.15 ms) costs 6.5 wasted reads, 3.3 ms of the drive's time, inside a
window that is bandwidth-bound while it is open. At distance one a read has about
one layer, 1.05 ms, to land, and the drive moves two experts in that time; the
in-flight budget above one was measured null in v15 because a speculative read
shares the drive with the demand read. So the constraint is not the width but the
reads per layer that can land, and it is already at its ceiling. **The width lever
is closed at distance one on this drive.** It reopens only with lead, reads issued
during the GPU-busy two thirds of the token, and the one lead source priced, the
table, is small (S0.2). The diagnostic stays as an instrument.

The table seeded from the prompt's routes (the `q` lines, every layer at width
eight), useful fills per position unseeded against seeded: the card 0.86 against
2.17 at 3.6 against 17.5 fills, the 300 1.80 against 2.00 at 7.0 against 16.2, the
1k 1.05 against 1.21 at 4.3 against 17.9, the 7k 1.62 against 1.74 at 6.4 against
18.1; the added fills at a precision of 0.03 to 0.10 and the cells at the pass
start to 190 or more. **Seeding is closed**: the prompt's route for a token is a
poor predictor of the answer's, a different context.

**S0.5b The ranking at distance two and three (pre-registered 2026-09-17, on
Davor's ruling that the width is not closed while lead could make it free).** The
argument: width twelve asks about thirty more non-resident reads per token than
eight, inside the drive's idle capacity of about seventy-five during the GPU-busy
stretches; what makes them costly today is the timing, one millisecond of lead
issued from inside the windows. Two layers of lead doubles the time and lets the
reads be issued from the GPU-busy layers, where the drive is idle, given a
demand-first discipline in the reader (the piece v15's null lacked). The unknown
is what the ranking loses at distance two: v14 measured a tenth of coverage lost
per layer of lookahead at width eight, and nothing past eight. The instrument: the
routers two and three layers ahead evaluated on layer L's state on the diagnostic
path (a scores-only GEMV each, `MoE.encodeRouterScores`, into the probe buffer's
further banks), ranked and dumped beside the pair's as `probe_ranking_d2` and
`probe_ranking_d3`; the coverage tool joins at `--distance`. **The bar:** width
twelve at distance two must recall more of the remaining misses than width eight
does at distance one today (0.32) at a precision no worse than ranks nine to
twelve show at distance one (0.13). Cleared, the prize is the recall times the
remaining twenty misses per token at 0.65 to 1.15 ms each, potentially larger than
the split; not cleared, the width closes with a number.

**Result (measured on the mini, 2026-09-17, one lifetime per shape at `3060034`,
9,282 / 14,586 / 11,895 / 13,962 ranking rows, none stale).** Per-miss recall,
precision and full-layer coverage of the remaining misses, the four shapes' range,
with the offline scheme's non-resident reads per token:

| distance, width | recall | precision | coverage | reads |
| --- | ---: | ---: | ---: | ---: |
| one, 8 (today's ring) | 0.29 to 0.33 | 0.14 to 0.16 | 0.20 to 0.21 | 28 to 31 |
| one, 12 | 0.54 to 0.57 | 0.13 to 0.14 | 0.47 | 59 to 62 |
| two, 8 | 0.26 to 0.31 | 0.11 to 0.12 | 0.20 to 0.22 | 30 to 35 |
| two, 12 | 0.40 to 0.45 | 0.08 to 0.09 | 0.35 to 0.37 | 63 to 70 |
| two, 16 | 0.51 to 0.55 | 0.06 to 0.07 | 0.45 to 0.47 | 107 to 113 |
| two, 32 | 0.73 to 0.75 | 0.03 | 0.69 to 0.70 | 336 to 348 |
| three, 8 | 0.22 to 0.28 | 0.08 to 0.10 | 0.18 to 0.20 | 32 to 37 |
| three, 12 | 0.35 to 0.39 | 0.06 to 0.07 | 0.30 to 0.31 | 66 to 73 |
| three, 32 | 0.65 to 0.66 | 0.03 | 0.59 to 0.62 | 334 to 343 |

**The bar is half cleared.** Recall clears it: width twelve at distance two names
0.40 to 0.45 of the remaining misses against 0.32 for today's ring, 8 to 9
useful reads per token against 6. Precision fails it: 0.08 to 0.09 against 0.13,
and ranks nine to twelve alone at 0.076, worse than at distance one, since the
state drifts and the tail of the ranking drifts more. What the numbers say in the
drive's terms: distance two at width twelve buys about two more useful reads per
token (1.4 to 2.5 ms) for about 37 more reads per token, 19 ms of drive time,
which the GPU-busy stretches could hold in volume only if the reader kept every
one of them out of the windows and off the demand reads' path, and the mini's
drift days would eat the margin. **The width closes for this chapter with a
number.** One shape the table leaves open for a later chapter: distance two at
width eight recalls nearly what distance one does (0.26 to 0.31 against 0.29 to
0.33) with twice the lead, which would move some of the ring's 4.1 late landings
per token to hits; its precision is lower (0.11 against 0.15), its form is v15's
two-distance queue, measured null then, and its ceiling is about a millisecond.
The instrument stays.

**S0.6 The attention row's fixed part (read, 2026-09-17; no run).** The arm as
written cannot be run on the current tree: since v18's one command per layer the
kernel stats report an attention layer's whole held command as one role
(`layer_kv`, `layerKernelRecords` in the runner), and Apple's GPU counters sample
at encoder boundaries, not dispatch boundaries, so a per-kernel split of the 7.4 ms
needs one of two instruments neither of which exists: the layer re-encoded as
per-kernel commands on a diagnostic switch (the three-role shape v18 retired for
routed layers survives only in `produceDenseLayer`, the dense-layer path the
served model never takes), or the bench extended from the scan to the whole layer.
Either is a day's work with a golden gate, for a number whose use is to slot B3 and
B4, both of which the ledger already grades under the arms' resolution once v18's
walls are taken out. **Recommendation:** no diagnostic in this chapter; B3 and B4
go to a chapter of their own on the attention row's fixed part, whose step zero
builds the per-kernel instrument once. Davor's ruling at S0.7.

**S0.7 The record and the ruling (2026-09-17, for Davor).** The board priced,
ranked by the modelled floor over production's probe:

| lever | per token, modelled | reads | cells | verdict |
| --- | ---: | ---: | ---: | --- |
| the split (A1) | 1.8 to 2.6 ms, 3 to 4.5 % | 0 | 0 | the lever |
| SLRU (A2) | up to 0.65 ms on the longer shapes, 0 on the 300 | 0 | 0 | cheap, small |
| the table (A9) | 0.24 ms at layer 0, 0.7 at every layer | +0.4 to +4.2 | 1 to 25 | small, under the arms' resolution |
| the width (A0) | none at distance one; at distance two 1.4 to 2.5 ms for 19 ms of drive time | +37 at width twelve | | closed by S0.5b's number |
| the draft (Q3), knowledge in the policy (A2's variant), seeding | null | | | closed |

The lead lever is priced small because the pool already holds the neighbour's
route and a deep route is only a third identity; the clairvoyant gap is context,
which no table sees. The width is real information that the window's bandwidth
cannot spend at distance one.

**Recommendation.** Task 1 becomes *the pool's allocation*: the per-layer slot
count from a production miss profile (the capture's plan rows carry it, the
misses per layer with the ring in place) at the same total, re-priced by replay
with the probe's fills before it is built, then built as a table the arena and
the residency index take per layer; SLRU as its policy in the same task, a
second commit and arm. Modelled 2 to 2.7 ms per token together, 17.0 to about
17.7 tok/s on the mini's three answers; class 1; the rig's two lifetimes per
shape the verdict. Task 2 folds into Task 1. The table is not built: 0.24 ms at
its precise layer does not pay for a second prediction source, its cells and its
reads, and it is under the arms' resolution; the design and the replay mode stay
on record for a chapter with lead to spend. Task 3, the agreed cells and the fold,
as planned, its design note first. Task 4 (B3, B4) to a chapter of its own; no
splitting diagnostic here. The wide capture stays as an instrument. The read
budget is untouched: nothing recommended spends a read.

**Davor's ruling (2026-09-17): proceed with what the data says.** The
recommendation as written: Task 1 the pool's allocation, the split with SLRU as
its policy; Task 2 folded in; the table not built; Task 3 the agreed cells and the
fold as planned, its design note first; Task 4 to a chapter of its own; the width
closed by S0.5b's number, the distance instrument kept. Step zero closed.

## Approaches for the predictor's plumbing

**A. A second source feeding the ring (recommended, subject to S0.2 and S0.5).** The
table's batch is issued at the token boundary, after the sampler's id is read back,
into ring cells through the ring's own landing path; the in-flight budget is raised
for the batch only (the demand reads keep their lane); the batch's cells are held
until their layer's classifier has run or the reclaim needs them. The wider probe,
if built, is the same path with a longer list. One landing path, one publish, one
race already won; the cost is the cell budget, which S0.2 sizes. Class 1.

**B. A separate pass-ahead lane.** The batch gets cells of its own beside the ring's
nine (the arena grows by the batch's width per served layer), its own reader batch
and its own reclaim, so the deep layers' long holds never starve the distance-one
ring. Cleaner accounting and more memory: on the mini the arena sits 0.43 GiB under
`maxBufferLength` with the ring, so the lane's cells come out of the slots, which
S0.4's split can price. Taken only if A's cell starvation shows in S0.2.

**C. Knowledge in the policy only.** No read ahead; the table's predictions protect
their experts from eviction and steer the victim choice (S0.4's predicted-future
policy). Zero extra reads, zero cells, the smallest prize; the fallback if the read
budget in S0.2 and S0.5 does not allow a batch.

## Tasks

### Task 1: the pool's allocation (class 1; the shape ruled at S0.7)

The split with SLRU as its policy. The predictor this task was to carry is not
built (S0.2, S0.3, S0.5, S0.5b); its design and the replay's table mode stay on
record above.

**T1.1 The pre-registration (2026-09-17; measured by replay, the arms graded T).**
The production miss profile, the misses each layer paid with the ring in place,
summed over every plan row of the four S0.5 captures (51,160 rows), per layer 0
to 39: 2373 1795 1651 1206 978 730 715 625 723 447 639 690 522 377 308 280 254
229 236 214 393 265 349 370 301 298 215 307 312 369 284 277 554 551 623 588 673
603 807 1009. Eleven to one between layer 0 and the quietest layer, sharper than
the pool-basis profile S0.4 used (six to one), because the ring serves the middle
layers better than the ends. The allocation is a blend between uniform and
proportional to that profile at the same total of 5,120, each layer at least 32;
the blend is relative to the profile's peakedness, so the production profile
wants a milder one than S0.4's pool profile did. Candidates re-priced by replay
over the probe at its in-flight budget of one on the four captures (the current
tree) and on the three v14 captures (an older build), misses saved per position
and the modelled ms:

| blend | min / max slots | card | the 300 | the 1k | the 7k | mean (current tree) | mean (v14) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.2 | 111 / 204 | 2.91 | 2.51 | 1.69 | 3.20 | 2.58 (1.84 ms) | 2.42 (1.74) |
| 0.25 | 107 / 219 | 3.34 | 2.80 | 1.96 | 3.59 | 2.92 (2.11) | 2.72 (1.97) |
| **0.3** | 103 / 240 | 3.56 | 2.84 | 2.03 | 3.86 | **3.08 (2.24)** | 2.91 (2.12) |
| 0.35 | 99 / 261 | 3.62 | 2.77 | 1.93 | 3.98 | 3.07 (2.24) | 2.91 (2.12) |
| 0.5 | 87 / 316 | 3.10 | 1.79 | 0.92 | 3.56 | 2.34 (1.54) | 2.19 (1.46) |
| S0.4b's table | 94 / 210 | 3.53 | 2.95 | 1.87 | 3.89 | 3.06 (2.19) | 2.95 (2.15) |

**The reference allocation is blend 0.3**, per layer 0 to 39: 240 204 195 166
152 136 135 129 136 118 130 134 123 114 109 107 106 104 105 103 115 106 112 113
109 109 103 109 109 113 108 107 125 125 129 127 132 128 141 154. The choice is
not delicate: 0.25 to 0.35 and S0.4b's table sit within a tenth of a miss of it;
0.5 over-corrects the middle layers on the 1k.

SLRU on the pool basis (the SLRU pool models no fills), decode misses per
position, aging-LFU against SLRU at 0.5: the card 29.9 to 28.2, the 300 29.4 to
29.0, the 1k 28.0 to 24.9, the 7k 27.0 to 27.0; with the split beside it the card
26.2, the 300 28.4, the 1k 24.7, the 7k 24.0. SLRU's gain is shape-dependent, up
to three misses on the 1k and nothing on the 7k, and it adds to the split's on
three of four shapes.

**The rows pre-registered for T1.6's arms** (two production lifetimes per shape
against the S0.5 lifetimes at the same build family; the misses per token and the
io the rows, the token and tok/s the verdict; T, the range from half to all of
the modelled saving since a saved further miss overlaps in latency):

| shape | misses per token, before | after the split (expected) | modelled ms | the token ms, before | after (expected) | tok/s after (expected) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| the card | 19.8 | 16.2 to 16.6 | 2.6 | 58.7 | 56.1 to 57.4 | 17.4 to 17.8 |
| the 300 | 19.5 | 16.6 to 17.1 | 2.1 | 57.9 | 55.8 to 56.9 | 17.6 to 17.9 |
| the 1k | 19.0 | 17.0 to 17.5 | 1.3 | 57.5 | 56.2 to 56.9 | 17.6 to 17.8 |
| the 7k | 18.1 | 14.3 to 15.0 | 3.1 | 58.6 | 55.5 to 57.1 | 17.5 to 18.0 |

SLRU on top (T1.3): zero to three misses per token by shape, the 1k the most,
the 7k nothing, graded T. The answers expected identical (class 1): which experts
compute never changes. The misses per token may move beyond the expectation on
the mini's drift days; interleaved lifetimes are the reading.

**T1.2 and T1.3, what was built (2026-09-17, `bc3e25c` and `f0e056c`).** The
streaming mode carries an optional per-layer slot table and the eviction policy
beside the uniform count; the model's lazy streamer construction takes each
routed layer's count with prefix-sum cell ranges and sizes the arena from their
sum, the ring's cells after it as before; the residency table is expert-indexed
and the kernels address arena cells, so nothing on the GPU side changed. The
runner's two prefill sites ask per layer. `SHRIKE_EXPERT_SLOT_TABLE` (a comma
list or a JSON path) is refused unless the count matches the model's layers,
every routed layer has at least 8, every leading dense layer has 0 and the total
equals the budget's, so the memory is unchanged and a malformed table never
silently falls back; `SHRIKE_EXPERT_POLICY` is `aging-lfu`, `slru` or
`slru:<share>`. The known names are fifteen. The streamer's SLRU is the replay's
rule (a probation hit promotes; past the capacity the protected slot used longest
ago drops to probation as its most recent; every placement lands probation; the
victim order empty, probation, protected, oldest). The server and the CLI honour
both variables, so the golden covers the configured pool: identical on all four
profiles bare and configured on the dev box, and on the mini. The load
description names the configuration (`expert_slots=103..240 policy=slru:0.5`),
and every arm's server log carries it.

**T1.6 The arms (measured on the mini at `f0e056c`'s server, 2026-09-17, three
arms per shape interleaved, two production lifetimes each, the first request of
each; the answers identical in length across the arms on every shape).**

| shape | arm | misses per token | io ms | the token ms | tok/s | tok/s against base |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| the card | base | 20.0 | 14.7 | 57.7 to 57.8 | 17.3 | |
| | the split | 16.5 | 12.6 | 55.1 to 55.4 | 18.0 to 18.1 | +4.4 % |
| | the split with SLRU | 15.8 | 12.1 | 54.8 to 55.4 | 18.1 to 18.2 | +4.8 % |
| the 300 | base | 19.5 | 14.5 to 15.1 | 57.0 to 58.8 | 17.0 to 17.5 | |
| | the split | 16.7 | 12.8 | 55.1 to 55.6 | 18.0 to 18.2 | +4.6 % |
| | the split with SLRU | 16.7 | 12.8 | 55.3 to 55.6 | 18.0 to 18.1 | +4.4 % |
| the 1k | base | 19.0 | 14.2 to 14.9 | 57.4 to 59.2 | 16.9 to 17.4 | |
| | the split | 17.0 | 12.9 | 55.6 to 56.1 | 17.8 to 18.0 | +4.4 % |
| | the split with SLRU | 15.2 | 11.6 | 54.1 | 18.5 | +7.8 % |
| the 7k | base | 18.1 | 13.5 | 58.4 to 59.0 | 16.9 to 17.1 | |
| | the split | 14.1 to 14.2 | 10.9 to 11.0 | 55.6 to 55.9 | 17.9 to 18.0 | +5.3 % |
| | the split with SLRU | 14.3 | 10.9 to 11.0 | 56.0 to 56.3 | 17.8 to 17.9 | +4.5 % |

Against the pre-registration: the misses per token after the split landed inside
the expected range on every shape (the card 16.5 for 16.2 to 16.6, the 300 16.7
for 16.6 to 17.1, the 1k 17.0 for 17.0 to 17.5, the 7k 14.1 for 14.3 to 15.0, a
touch better); the token beat the expected range on three shapes and met it on
the 7k, the split alone saving 2.4 to 3.0 ms per token where 1.3 to 3.1 was
modelled, the 1k's saved misses worth more than the 0.65 each the model gave
them. SLRU on top is the pre-registered shape: 1.8 ms more on the 1k, 0.2 on the
card, nothing on the 300, 0.4 less on the 7k, the last within the rig's noise;
across the four shapes it adds 0.5 ms on average and 1.8 where it matters.
**Reading:** the split is real and free, the first lever on this surface since
v16 that moved the token on every shape, +4.4 to +5.3 % alone and +4.4 to
+7.8 % with SLRU; the misses per token fell from 19 to 20 to 14 to 17, the io
from 13.5 to 15.1 ms to 10.9 to 12.9. The class-1 gate held throughout.

**The decision on SLRU and the launch.** Both ship as the mini's production
configuration: the reference table and `slru`, the built-in defaults staying
uniform and aging-LFU so a bare launch is unchanged and another model is not
handed this model's table. The mini's launch line carries the two variables (the
project instructions record it); the table lives in this document and in the
archive as `t11-slots-blend0.3.json`. A follow-up for a later chapter, not this
one: the table shipped beside the model rather than in the launch, and an
allocation derived at load from a profile the model carries.

### Task 2: the policy and the split (class 1; only on S0.4's number)

The predicted-future eviction and the per-layer slot count, each its own commit and
arm; misses per token the row; the answer identical.

### Task 3: the agreed cells and the fold (class 1; structure)

The design note first (the four edges, for Davor's ruling before a line is built),
then the agreed cells (v18's T2.1 to T2.5 as written), then one command per token
with two in flight, the cancel and the stop path, the error surfacing per layer.
Golden identical; the wall expected flat within the arms' resolution; misses per
token may drift since the plan runs later, recorded.

**T3.0 The design note (2026-09-17, for Davor's ruling; the tree read at
`7596f86`).** The runner's anchors in [architecture.md](architecture.md) are stale by
about 145 lines since v20's diagnostics landed; the anchors below are the tree's, and
the close re-anchors the document. Every number here is remembered from the chapters
cited or modelled from them; nothing was run.

**The fold in three commits.** T3.1 *the agreed cells*: a layer's fixup encoded with
the layer, before its router has run, behind an event wait, its misses read into
cells the host names on the word; the plan off the path, at the next wake. T3.2 *one
command per token*: the forty layers' commands and the boundary's as encoders of one
command, encoded a layer per word during the previous token and committed on the
boundary word after the stop check, carrying the drain invariant and the per-encoder
error naming without which it cannot ship. T3.3 *the stop path's ruled shape* and the
cancel's tests. Each commit golden byte-identical on all four profiles on both boxes;
the arms after T3.1 and after T3.3, expected flat within the drift. The modelled
prize is T6.7's 0.6 to 0.85 ms per token less what the ruling on the stop path
leaves (the decision below); under the arms' resolution either way. The architecture
is the win.

**The tree on the edges (read, 2026-09-17).**

*The token today.* Forty-one command buffers plus one per miss layer: every served
layer folds its attention, tail and speculative routed work into one command
(`tailCB` and `specCB` never exist on this model, `RealForwardRunner.swift:2138`,
`:2188`), the fixup is a second command on a miss layer (`:3272`), the boundary a
last one, committed and not waited on (`:2369`); a non-continued pass adds the
synchronous embed. Layer L+1 is encoded while L runs (`:2470`). The next pass's held
layer 0 is committed only after the stop check on the boundary word
(`RawCompletion.swift:313-357`, the produce at `:362-372`); nothing runs a pass late
(v18 Task 4). No cancel of a committed command exists anywhere; a client's disconnect
cancels the request's Task, which the loop sees at its top (`:290`), between passes,
never inside one.

*What a pass mutates in place.* The thirty GDN layers' recurrent state, one FP32
`[32, 128, 128]` buffer of 2 MiB per layer, and the conv tail, `[3, 8192]` FP16 of
48 KiB, single-buffered and updated in place by `gdn_delta_step_decode`
(`gdn.metal:477`, `:489`: `s[i] = srow[idx] * g`, then `srow[idx] = s[i]`) and
`gdn_conv_mix_decode` (`:268-273`, the shift in place): 61.4 MiB across the model
(modelled from `ArchConfig.qwen36_35B_A3B`). The KV row at the cursor on the ten
attention layers: the cursor is `KVCacheManager.position`, advanced only by
`produceToken` (`:2290`), and a row written past it is dead bytes to every read
(the views are sized by the cursor, `KVCacheManager.swift:487`). Partial rewind is
refused whenever GDN state exists (`RealForwardRunner.swift:1075-1078`, "recurrent
GDN has absorbed every token it was advanced over"); the only restore is the whole
snapshot at a prompt boundary (`captureInferenceState`, `:1090`). The decode scratch
(`hidden`, `moeActs`, the readback and dispatch-argument buffers) is single and
rewritten per layer. The sampler's seed is pure in the position when a seed is set
(`Sampler.swift:298`); the trace files are append-only. After a generation that did
not complete, the server invalidates the single-prefix prompt cache and resets the
runner, KV and GDN state included (`ServerInference.swift:1169-1177`,
`RealForwardRunner.swift:1046-1051`).

*The miss path today.* The fixup is host-built after the word: the batch's event
wait (`:3282`), then one encoder with phase 1 over the misses
(`moe_phase1_gate_up_act_subset_u16load`, a direct dispatch over `active_count` rows,
the miss positions in `moeMissActiveSlots`, the misses' bytes reached through an
argument buffer of device pointers filled from the plan's slots, `MoE.swift:606-613`)
and phase 2 over all eight (`moe_phase2_down_reduce_k8`). The speculative kernels
are the other shape: pool-addressed, `expert_pool` plus `resolved_slots[position]`
times `pool_slot_stride` (`moe.metal:923-935`, `:975-989`), their grids the
classifier's `MoESpecDispatchArgs` (`:109-112`, filled at `:211-216`: phase 1 full
always, phase 2 full only when no expert missed), the classifier writing
`0xffffffff` into a miss's resolved slot (`:141`). The gate is `moe_io_ready`
(`:36-39`) on the batch's status word, 0 loading, 1 complete, 2 failed; phase 1
returns on not-ready (`:823`), phase 2 writes the residual through and returns
(`:894-897`). The timeline is one `MTLSharedEvent`, one value and one status word per
batch reserved at submit (`PreadExpertStreamer.swift:568`), published in order with
the out-of-order hold (`ExpertIOEventCoordinator.swift:83-88`); a failure publishes
status 2 and advances the timeline "so a pre-submitted GPU command cannot deadlock"
(`ExpertLoadOperation.swift:147-150`); an empty plan publishes its value at once
(`PreadExpertStreamer.swift:572-578`). The host learns of a failed read at the next
layer's wake, from `finishPendingRoutedCommand`'s `storage.wait()` (`:5276`), as
`readFailed(errno:)` with no layer in the error. The prefetch's reads never touch the
timeline (`beginPrefetch` is its own entry; the demand path alone reserves,
`:5657`). The ring: nine cells, one read in flight, a free cell is `expert < 0`
(`ExpertPrefetchRing.swift:114`), the refusal counted (`:117`), leases at
`readyCells` with the 400 µs join (`:161-202`), consumed at the plan (`:222-238`),
the lock order ring then cache. The reader has no per-batch cancel
(`shrike_expert_reader_cancel_slot` runs at shutdown only); a third batch parks; a
failure is thrown on the storage thread. `abandonExpertCachePlan` exists with no
production caller (`ModelExpertIO.swift:159-163`). The coordinator's status words
are capped at 4,096 chunks of 4,096 (`ExpertIOEventCoordinator.swift:31-32`).

*Errors today.* `ModelError.commandBufferFailed(detail: String)` carries no layer;
the deferred drain names the command's role (`layer_linear`, `layer_kv`, "routed
layer command buffer", `:3431-3449`); the immediate checks name nothing. The
command buffers are made without a descriptor, so Metal's per-encoder error status
is not requested; there is one `encodeWaitForEvent` (the fixup's) and no
`encodeSignalEvent` (the host signals by `signaledValue`).

**Edge (c): the agreed-cell contract (T3.1).**

*The contract.* For every routed layer L of token t the host owns two things before
L's router runs: a timeline value v(L, t) with its status word, reserved from the
coordinator at L's encode in layer order, and an `agreed_cells[L]` array of top-k
words, host-written only, one per top-k position, the sentinel for a hit. The layer's
fixup is encoded with the layer, after its speculative routed work, as: the event
wait on v(L, t); phase 1 over the misses, the pool-addressed shape of the speculative
phase 1 with its rows over `miss_count × F` and each row group's expert
`topk_indices[miss_positions[j]]` at cell `agreed_cells[L][miss_positions[j]]`;
phase 2 over all eight, the speculative phase 2 kernel with one resolve added, a
position's cell `resolved_slots[p]` unless it is the sentinel, then
`agreed_cells[L][p]`; both behind `moe_io_ready` on v(L, t)'s status word. The grids
are two new indirect triples the classifier writes beside its two
(`MoESpecDispatchArgs` grows to four): the fixup's phase 1 at
`ceil(miss_count × F / 16)` threadgroups, its phase 2 at `D` when any expert missed
and zero otherwise, so an all-hit layer dispatches nothing and a miss layer's
speculative phase 2 stays zero as today. "Miss i into cell i" is therefore the
per-position word: the miss at position p is read into `agreed_cells[L][p]`, and the
kernel finds it there. No argument buffer, no host-built command, no word written by
both sides: the classifier writes `resolved_slots`, the host writes `agreed_cells`,
and the event's happens-before (the host's stores before `signaledValue`, the GPU's
wait before the reads, the same edge today's fixup crosses for the expert bytes
themselves) orders the host's cells before the kernel's use.

*The host on the word, in order.* Read the readback (unchanged). For each miss
position p in the classifier's list: if the ring holds the expert landed, lease its
cell (`readyCells`' leasing as today); if in flight, join up to 400 µs and lease
(v15's join, its order before the issue kept, C7's note stands); otherwise claim a
free ring cell as a landing is claimed (`loading` published at the cell, the
generation bumped, the ring lock then the cache lock) and add the read to the batch;
**the fallback when no ring cell is free**: choose a victim slot of the pool by the
policy, on the path, for this miss alone (the victim published `empty` after its
cell's bump, the slot `loading`), and the cell is the victim's; write
`agreed_cells[L][p]`. Submit the batch into the demand lane with v(L, t) as its
token; a batch with no reads publishes v(L, t) at once (the empty-plan path today).
Issue the next layer's prediction (unchanged). Then run the *previous* layer's plan,
off the path: under L−1's cache lock, the hits' use counts and SLRU promotions (the
policy sees the same access sequence one layer late), every leased cell swapped into
the pool by index against a victim chosen now (a cell whose read has not landed yet
swaps the same way; its generation travels with the cell and the completion
publishes `resident` at it under the guard), the freed pool cells back to the ring,
the overflow misses already in the pool needing nothing; the route and prefetch
trace rows written here, unchanged in content. The last layer's plan runs at the
token's end on every exit. The classifier's miss list is the authority; the fail-
closed cross-check becomes the deferred plan's: every leased cell's expert must be in
the layer's route, an `internalInconsistency` otherwise.

*Why the fallback is a victim and not v18's host-built fixup.* v18's T2.2 fell back
to today's fixup command when a layer's misses exceed its free cells. Under the fold
a separate command cannot be inserted between two layers of one command, so a second
GPU path would have to exist for T3.1 and be removed at T3.2. The victim on the path
is one GPU path throughout: the kernel reads a cell and does not care whether the
ring or the pool owns it, the read lands where the plan would have put it anyway,
and the cost is today's plan for that miss alone, counted as `agreed_overflow` per
token. The ring's free cells at a word are nine less the predictions landed or in
flight (one issued per layer, consumed a layer later, so one or two held) less the
cells leased to the previous layer's misses until its plan runs; a layer with more
than about six misses overflows, which the miss profile makes rare (1.5 per miss
layer on average). If the arms read `agreed_overflow` above 0.1 per token the remedy
is the ring's size, top-k more cells (14 MB), as an arm.

*What T3.1 removes from the path.* T2.0's chain: the plan, the pin and the submit
(`cache_plan_ms`, `path_pin_ms`, `path_submit_ms`, at most 25 µs per miss layer), the
fixup's build and its commit-to-kernel (`path_fixup_build_ms`, the 27 µs), one
command per miss layer. Modelled at most 0.34 ms per token, about 0.2 (T2.0, C);
under the drift. The decode plan stops pinning: it is the decode path's only evictor
and runs after the layer's kernels are done (encoder order, then the next word), and
the pin field stays for the other planners. The residency publish stays one release
store per transition: the claim's `loading`, the completion's `resident`, the
victim's `empty`; the swap writes nothing, as today.

**Edge (b): the error surfacing per layer when a token is one command (T3.2).**

Three failures reach a token, and each must name its layer and leave no GPU wait
unsatisfied.

*A failed read.* Unchanged on the GPU: the batch publishes status 2, the timeline
advances, phase 1 returns and phase 2 writes the residual through, the command
completes. On the host, the per-layer record inside the token (the tag, the value,
the status word, the operation) is checked at the next word as
`finishPendingRoutedCommand` checks it today, and the failure is wrapped with its
layer: a new `ModelError.expertReadFailed(layer:errno:)`, since the type has no
layer field today. The host then drains (below) and throws; the server's defer
resets the runner and invalidates the cache as it does for any incomplete
generation, so the pass's half-updated state is discarded whole, as today.

*A GPU fault.* With one command per token Metal's error names the token. The
command is made from a `MTLCommandBufferDescriptor` with
`errorOptions = .encoderExecutionStatus` (macOS 11 and later): on a failure the
error's `userInfo` carries one `MTLCommandBufferEncoderInfo` per encoder with its
label and its state (completed, affected, or the faulting one), and every encoder
is labelled with its layer and stage (`layer 12 attention`, `layer 12 fixup`,
`boundary head`), so the drain reports the first encoder that did not complete and
the affected ones after it. The header's caveat, verbatim: "enabling this error
reporting option may increase CPU, GPU, and/or memory overhead on some platforms;
testing for impact is suggested". The option is on from the first T3.2 build and its
cost is read on the same-box A/B (the token's GPU time and the wall); if it costs
above the noise it moves behind `SHRIKE_RUNNER_STATS`, not a knob of its own; the
labels stay in any case, they are free and name the encoders in a GPU capture too.

*A host throw mid-token.* A stale tag, the deferred plan's cross-check, a cursor
mismatch: today they unwind the pass with the committed layer commands completing
on their own. Under the fold the whole token's command is committed and its later
layers wait on values only the host publishes, so an unwind without a drain hangs
the GPU forever, the one outcome worse than an error. Hence **the fold's invariant:
a committed token's command is always drained: every value it waits on is
published, by the reads' completion, by the host at the word, or by the drain.**
The drain is one routine on every abnormal exit of the token's loop: publish every
remaining value of the token as failed (status 2, so the fixups skip; the other
kernels run on what is resident, into scratch that is dead), issue no reads, claim
no cells, run the pending plan as a drop (the leased cells back to the ring, the
`loading` entries to `empty` through `abandonExpertCachePlan`'s path, which exists
with no caller today), wait for the command, then throw with the layer. A read still
in flight lands later and its landing is dropped by the reclaim under the generation
guard; the reader needs no cancel (at most two batches of eight, a few milliseconds).
The one-second fallbacks of the word wakes (`waitForRouterReadback`,
`awaitBoundaryToken`) then wait on the token's command, up to a token; they gain a
deadline (ten seconds) after which they throw `commandBufferFailed` naming the layer
whose word never landed, so a wait the drain missed is a loud, fatal error in the
server's log rather than a silent hang.

*The status words.* Every routed layer now consumes a value, forty per token where
about fourteen did (the all-hit layers reserved none), so the coordinator's cap of
16.7 million words is 420 thousand tokens per process. T3.2 recycles the status
words by token: a word is free once its token's command completed, so a ring of two
tokens' worth (eighty) suffices, indexed by value.

**Edge (a): the stop path (T3.3, the decision for the ruling).**

The fold's original sentence is "the next token's command committed before this one
completes". Two shapes satisfy the structure; they differ in whether a pass can run
past the stop.

*Shape A, encoded ahead, committed on the word (recommended).* Token t+1's command
is encoded in full during token t, a layer at each of t's words, as layer L+1 is
encoded at L's word today, and committed at t's boundary word after the stop check,
which is today's `holdLayerZero` at the grain of the token. In flight: one committed
command and one encoded. The stop check stays on time; nothing runs past the stop;
no state is undone; the GDN buffers, the cursor, the traces and the counters are
untouched. The cancel with two in flight is the discard of an uncommitted command:
its forty reserved values published as succeeded (nothing waits on them), its tags
retired, `discardBoundaryState` at the token's grain; the running command has
already published all of its values by the time the loop can see a cancellation
(the produce returns after the last word), and only its boundary encoders remain,
which `finishPreviousBoundary` waits for as today. Max tokens, the stop token, the
stop strings and the external stop all end at the boundary word as today; a
disconnect ends at the next boundary, the same latency as today's. What Shape A
leaves on the table: the boundary's one gap, 0.25 ms per token (M, v18 Task 4's
rows: the word's 63 µs, the host's checks, the commit and the driver's start), 0.4 %
of the token, unreadable by the arms (the drift is 1.7 %). The GPU does see the
token boundary; the commit does precede the driver's completion mark of the running
command (about 160 µs after its GPU end) but not its GPU end.

*Shape B, committed ahead.* Token t+1's command is committed as soon as it is
encoded, after t's last word and before t's sampler runs, so the GPU flows from t's
embed into t+1's layer 0 with no gap. Then every stop token, stop string and
external stop is seen one pass late (max tokens is not: the host knows at encode
time that the pass after the last token is never wanted, and does not encode it),
and the extra pass has by then embedded the stop token and started updating state.
The snapshot the ruling names is a parity: the GDN state and conv tail of every
linear layer double-buffered by token, the kernels taking `state_in` and `state_out`
(the arithmetic unchanged, one pointer more; class 1, the golden proves it), prefill
writing the parity the decode continues from, the snapshot and restore reading and
writing the current parity; 61.4 MiB more (modelled), affordable beside the 8.45 GiB
arena on the mini. The extra pass writes the other parity and the current one
survives; the rewind is the parity pointer left where it was, the cursor back by
one (`rewind(to:)` unlocked for this one case, whose reason the parity removes),
`hidden` and the scratch dead after a stop, the sampler's nondeterministic counter
one ahead (harmless), the extra pass's trace rows and counters suppressed by the
drain. The cancel of the extra pass is edge (b)'s drain: all forty values published
as failed at the boundary word, no reads, the wait. Unguarded, the drain costs the
pass's GPU time without its io, about 40 to 45 ms once per answer (modelled: the
token is 55 to 58 ms with 11 to 13 of io); guarded, a cancel word every heavy kernel
reads at entry (the attention scan, the GDN delta step, the routed phases, the head
GEMV: five families, the host's store visible within layer 0's traffic by v16's
probe), about 3 ms. Against the 0.25 ms per token saved: the 300-token answer gains
about 75 ms per answer less the drain (0.4 % of the answer unguarded, 0.5 % guarded);
the turn chapter's 21-token follow-up loses 40 ms unguarded (2.9 % of 1.40 s) and
gains 2 guarded. Neither is readable by the arms. Shape B's price is the parity
buffers, the rewind, the guard in five kernel families, and a stop-path surface the
golden cannot see (the golden checks the answer, not the state after the stop),
which a new gate would have to cover: a two-turn continuation across a stop token
byte-identical against the same turns without the early commit.

*The recommendation: Shape A for T3.3.* The ruling asked for a snapshot or a cancel;
Shape A needs neither, because it keeps v18 Task 4's finding at the grain of the
token: the stop check on the word, the commit after it. The structural goal is met
by it in full: one command per token, the host feeding reads and publishing values,
the plan off the path, two commands in flight in the sense that the next is encoded
before the current completes. The 0.25 ms Shape B buys is under the arms'
resolution and is eaten on short turns by its own drain. Shape B is designed above
and stays on record; if the ruling is for it, the parity and the guard are T3.3's
build, and the continuation gate is added to the golden's profiles. An event-gated
variant (t+1 committed early but its first encoder waiting on a value the host
publishes after the stop check) was considered and is not proposed: the GPU idles at
the wait instead of at the commit, and the signal's latency is not better than the
commit's (v10's `io_fixup_wake_ms` measured about 157 µs from a host publish to the
kernel's start), so it buys nothing and adds a hang path.

**Edge (d): the cancel with two in flight, by trigger (T3.3).** Under Shape A: a stop
of any kind at the boundary word, the held command discarded (above), the last
layer's plan run, the ring and the reader untouched (nothing was claimed or issued
for the held token); a disconnect, the same at the next boundary; a failed read or a
host throw mid-token, edge (b)'s drain of the running command, then the discard of
the held one. The ring's reads in flight at any of these: the current layer's demand
batch completes into leased cells and the drain's drop returns them; the speculative
read in flight lands and is reclaimed; the generation guard keeps a late completion
from publishing over a reused cell. Under Shape B, add the extra pass's drain and
the rewind. No trigger needs a cancel inside the reader, and none needs a cancel of
a running command beyond the drain, which Metal does not offer in any case.

**Class 1.** Which experts compute, in what order and with what arithmetic is
unchanged: the fixup's phase 1 computes the misses' rows into `moeActs` at their
positions as the subset kernel does, phase 2 reduces the eight in the same order
with the router's weights and the same epilogue; the pool-addressed variant reaches
the same bytes (`expert_pool + cell × stride + offsets`) that the argument buffer's
pointers reach. The parity buffers of Shape B change pointers, not arithmetic. The
golden gates every commit on both boxes; misses per token may drift because the
plan runs a layer later and the leased cells are held a layer longer, and the arms
read the drift.

**The instruments the fold retires and what replaces them.** The kernel stats price
per command (`kernelGPUTimings` from each command's GPU start and end), so with one
command per token the per-layer rows (`layer_linear`, `layer_kv`, the transitions,
the fixup roles) collapse to one row per token; S0.6 already found the held command
unsplittable. The per-layer clock becomes the host's: every word's arrival is
recorded into a per-token array and a layer's wall is the gap between consecutive
words (M at the word's resolution, 42 to 45 µs after the classifier by
`MidCommandVisibilityTests`), which `tools/decode-rows.py` reads as the layer rows;
the io rows and the `path_*` rows are host-side and survive. The per-encoder GPU
clock (counter sampling at stage boundaries) is B3 and B4's chapter's instrument,
not this one's. Two counters the Method asks for and the runner line lacks are
added at T3.1: `agreed_overflow` per token and the ring's `cells_leased_peak`; reads
per token the rows tool derives from `expert_read_mib`.

**The build order, v18's T2.1 to T2.5 as written with two amendments.** T2.1 the
read is this note. T2.2 the build as above, amended in the fixup's addressing (the
speculative kernels' pool-addressed shape with the indirect grids, not the argument
buffer) and in the fallback (the victim on the path, not the host-built fixup).
T2.3 the tests: the contract on the toy model with a forced miss set, the encoded
fixup against the host-built one bit for bit at zero, one and k misses; the overflow
taking the victim path with the same output and the counter; the deferred plan's
ordering under the cache lock and the one-store publish (the landing tests' shape);
the lock order; the timeline's pre-reserved values published in order, an all-hit
layer's at the word, a discarded token's at the discard. T2.4 the gates and the
golden. T2.5 the deploy and the arms, the host's path fields off the path, misses
per token recorded for the drift. Then T3.2 with the drain invariant's tests (a
throw injected at layer k of a committed token completes the command, names layer
k, hangs nothing, and the next request runs; a read failure injected names its
layer) and the encoders' labels; T3.3 the ruled shape and the cancel's tests (the
stop token, a stop string, max tokens, a disconnect, each leaving the timeline
published, the ring empty of leases and the runner reusable).

**The pre-registration for the arms (T, two production lifetimes per shape on four
shapes against Task 1's arms at `~/.claude/handoffs/archive/shrike-v20-t1/arms/`,
interleaved).** After T3.1: the token flat within the drift (at most 0.34 ms
modelled, C), misses per token within 0.3 of Task 1's, `agreed_overflow` under 0.1
per token, `cache_plan_ms` and `path_fixup_build_ms` off the path and their time
reappearing under the deferred plan's row. After T3.3 under Shape A: the token flat
within the drift (the forty command boundaries, 0.2 to 0.4 ms, C), the boundary gap
unchanged at 0.25, the answers identical in length. Under Shape B: the boundary gap
gone from the per-token rows and the drain once per answer in the answer's total.

**Risks.** A pre-reserved value never published hangs every later wait: the drain
invariant and the deadline. The cells: the leases held a layer longer shrink the
ring's free cells; `agreed_overflow` and the ring's size as the remedy. The error
option's cost: measured on the first build, behind the stats flag if it shows. The
per-layer GPU rows retire: the ledger's reading changes to the word clock. The
encode-ahead shares single scratch buffers across layers, safe by encoder order
within the command; the host-written `agreed_cells` are per layer so the host never
writes a word the GPU also writes.

**What the note asks Davor to rule on.** (1) The stop path: Shape A recommended,
Shape B designed. (2) The overflow fallback as the victim on the path, one GPU path,
in place of v18's host-built fixup fallback. (3) The per-layer GPU rows retiring
with the fold and the word clock as the per-layer instrument. (4) The per-encoder
error option on by default and measured on the first T3.2 build. (5) The order T3.1,
T3.2, T3.3, the golden on every commit, the arms after T3.1 and T3.3.

**Davor's ruling (2026-09-17): Shape B for the stop path; the overflow as the
on-the-spot eviction; the word clock in place of the per-layer GPU rows; the
per-encoder error option on and measured on the first T3.2 build; the order T3.1,
T3.2, T3.3 as proposed.** His reasoning on the stop path: a human cannot write the
next prompt within 40 ms, and an agentic loop's tool result takes longer than that
before its prefill begins, so the drain's cost once per answer is never seen. The
refinement recorded with it: what hides the drain is the client's turnaround, not
the prefill's length, and the answer's finish frames go out before the drain, the
prompt cache's settle after it, so the answer's end is unchanged even for a tight
loop. Consequences: the unguarded drain is taken and no cancel word is built, so
the only kernel change of Shape B is the GDN pair's second state pointer; the
parity buffers, the rewind by one, the suppressed rows and the two-turn
continuation gate are T3.3's build; T3.2 lands one command per token committed on
the word, the stop path unchanged, and T3.3 moves the commit ahead.

**T3.1 The agreed cells (2026-09-17, `2a0f7d9`; the arms measured on the mini at
that tree, graded M, the verdict against Task 1's arms at `f0e056c`).**

*What was built.* No new kernel. `MoESpecDispatchArgs` carries four grids: the
classifier writes the agreed fixup's phase 1 and phase 2 grids full only when an
expert missed, beside the speculative pair. The speculative phase 1 gained the
batch's status word and the speculative phase 2 the status word and a fallback
cell array, so the same two kernels serve both dispatches: the speculative one
binds an always-ready word and the classifier's array twice, the fixup binds the
layer's value's word and the host's row (the generic pipelines now carry the
event-gate constant the specialized ones already had). Each routed layer's held
command carries a timeline value reserved at its encode; the fixup is encoded
behind `encodeWaitForEvent` after the speculative work, phase 1 over the layer's
row of `agreedCells` with the sentinel at the hits, phase 2 resolving a sentinel
through that row. At the word the host reads the route, leases the landed
predictions (the 400 µs join kept, a prediction issued between the join and the
claim joined once more), gives every miss its cell (a landing's, a free ring cell
claimed as a landing through the ring's `claimDemand` and the streamer's
`claimLanding`, or, when the ring has none, a pool victim from
`reserveOverflowSlot`, counted as `agreed_overflow`), writes the row, submits the
batch through `beginAgreedReads` on the demand lane with the layer's value (an
empty batch publishing at once), attaches the ring's demand cells and issues the
next layer's prediction; the pins are gone from the decode path. At the next wake
the previous layer's command and batch are checked (a failed read is
`ModelError.expertReadFailed(layer:detail:)`), the io rows taken, then the plan
runs off the path: `planRoutedExperts` with the predictions and the demand cells
as its leased landings and the misses counted as the reads issued, the swap by
index, the freed cells back to the ring, the trace rows; the last layer's plan
runs at the token's end on every exit. The fold's invariant lands in its T3.1
form: the values reserved for encoded layers stay armed until a batch or the word
takes them, and a pass's throw path and the boundary state's discard publish
every armed one as failed, returning the ring's leases and the overflow slots on
the way out. The coordinator's status words are one ring of 4,096 recycled by
value (T3.2's item, taken here since every routed layer now reserves one).
Retired with the host-built fixup: its argument-buffer path on decode,
`DecodeExpertPartition`, the pending routed command and its deferred record, the
completion clock, and the rows `cb2_ms`, `io_hidden_pct`, `io_fixup_wake_ms`,
`path_pin_ms`, `path_fixup_build_ms`, `path_fixup_commit_to_kernel_ms` and
`io_host_waits_avoided` (the app's `cb2 / token` row with them); added
`agreed_overflow` and `cells_leased_peak`; `cache_plan_ms` is now the deferred
plan's time. The kernel stats lose the `moe_phase1_miss_fixup_phase2` roles and
the window's gap rows: the layer rows `layer_linear` and `layer_kv` now hold the
fixup's wait inside the command, and `tools/decode-rows.py` reads the gaps n/a.
Tests: the agreed fixup against the host-built fixup bit for bit at four misses
and the skip on a failed status word, the classifier's four grids, the ring's
demand claims outside the prediction budget, the failed slot's reuse and a
prediction in flight never doubled, the streamer's overflow victim, the agreed
reads into a ring cell and a pool cell publishing the token, the empty batch, the
failed batch dropping and emptying and still publishing, the counted misses, the
coordinator's recycled words; 1,275 tests in 174 suites. The four gates; the
golden identical on all four profiles bare and configured on both boxes.

*The arms* (two production lifetimes per shape, bare and the production
configuration interleaved, the first request of each; the answers identical in
length across the arms and to Task 1's on every shape, 226 / 369 / 300 / 353):

| shape | arm | misses per token | io ms | overflow per token | cells leased peak | the token ms | tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| the card | bare | 19.9 | 14.3 | 0.00 | 8 | 57.7 to 58.7 | 17.0 to 17.3 |
| | configured | 15.7 | 11.7 to 12.1 | 0.00 | 6 | 54.9 to 55.6 | 18.0 to 18.2 |
| the 300 | bare | 19.3 | 14.0 | 0.00 | 8 | 56.8 to 57.0 | 17.5 to 17.6 |
| | configured | 16.6 | 12.3 to 12.6 | 0.00 | 7 | 55.6 to 55.8 | 17.9 to 18.0 |
| the 1k | bare | 18.9 | 13.6 to 13.7 | 0.00 | 8 | 57.0 to 57.8 | 17.3 to 17.5 |
| | configured | 15.0 | 11.1 | 0.00 | 7 | 54.2 to 54.8 | 18.2 to 18.4 |
| the 7k | bare | 18.0 | 13.1 | 0.00 | 8 | 58.7 to 59.3 | 16.9 to 17.1 |
| | configured | 14.2 | 10.6 | 0.00 | 8 | 55.9 | 17.9 |

Against Task 1's arms at the same configuration (Task 1 / T3.1): the card
15.8 / 15.7 misses per token, 12.1 / 11.7 to 12.1 ms of io, the token 54.8 to
55.4 / 54.9 to 55.6; the 300 16.7 / 16.6, 12.8 / 12.3 to 12.6, 55.3 to 55.6 /
55.6 to 55.8; the 1k 15.2 / 15.0, 11.6 / 11.1, 54.1 / 54.2 to 54.8; the 7k 14.3 /
14.2, 10.9 to 11.0 / 10.6, 56.0 to 56.3 / 55.9. The bare arm the same way: the
card 20.0 / 19.9 and 57.7 to 57.8 / 57.7 to 58.7, the 300 19.5 / 19.3 and 57.0 to
58.8 / 56.8 to 57.0, the 1k 19.0 / 18.9 and 57.4 to 59.2 / 57.0 to 57.8, the 7k
18.1 / 18.0 and 58.4 to 59.0 / 58.7 to 59.3. The ring's counters per token
(Task 1 / T3.1, the configured arm): issued 19.3 / 19.6 on the card, 20.1 / 20.6,
19.1 / 19.8 and 18.4 / 18.6 on the 300, the 1k and the 7k; adopted within 0.2;
refused 12.6 / 12.3, 14.0 / 13.5, 13.2 / 12.5 and 11.4 / 11.2; landed before the
classifier 3.4 / 5.2, 3.4 / 4.9, 2.8 / 4.5 and 5.2 / 6.2, with the joins at the
plan down by as much (2.9 / 1.5, 3.6 / 2.2, 2.7 / 1.4, 1.5 / 0.9).

**Reading.** Flat, as pre-registered: the token moved 0.3 to 0.4 ms at most on
any shape, in both directions, under the rig's drift; the misses per token
within 0.2 of Task 1's on every shape (the bar was 0.3); no lifetime overflowed
(the bar was 0.1 per token), the ring's leased peak 6 to 8 of its nine cells;
the answers identical. The io is 0.3 to 0.5 ms per token lower on every arm and
the predictions land before the classifier 1.2 to 1.7 more often per token with
as many fewer joined at the plan: with the plan out of the word's way the demand
batch reaches the drive a few tens of microseconds earlier on a miss layer, and
the ring's reclaim at the claim frees a cell a layer earlier, so the same
predictions arrive a little earlier relative to the next classifier (a reading;
neither term was measured on its own). The host's on-path rows: `path_submit_ms`
0.14 per token, `cache_plan_ms` 0.14 to 0.21 per token now spent at the next
wake. The structure is in place for T3.2: every routed layer is one command
holding its fixup, the host feeds reads and the batch publishes the value.

**T3.2 One command per token (2026-09-17, `51fc54f`; the arms measured on the
mini at that tree, graded M, the verdict against T3.1's arms at `2a0f7d9`).**

*What was built.* The token is one command: `TokenCommand` holds the command
buffer, made from a `MTLCommandBufferDescriptor` with `encoderExecutionStatus`
(always on; the arms below priced it through a variable that existed for their
lifetimes only and is not shipped, so the known names stay fifteen), and the
routed layers' `TokenLayer` records (the readback tag, the agreed value). `encodeLayers` encodes a token's layers in order as
encoders of that command, a dense layer's chain or a routed layer's input norm,
attention and tail with the classifier (one serial encoder on GDN and gated
layers, labelled `layer L attention`), the speculative work (`layer L routed`),
the wait on the layer's value and the agreed fixup (`layer L fixup`); the gpt-oss
and plain attention paths, which kept a separate softmax command, encode their
softmax on the same command in order. `produceToken` takes the held command for a
continued pass (or makes one and encodes the embed and every layer for a fresh
token), encodes the boundary as its last encoders (the final norm, the head, the
caller's sampler writing the token word, the next embed from it; `encodeHead` for
the synchronous head), commits it, and only then waits for the previous token's
command (complete once its boundary word landed) and records it under the `token`
role. The host's loop over the routed layers: the next token's layers encoded a
layer per word into the next command (`kv.reserve` a position ahead), the word
awaited (`waitForWord`, the classifier's tag), the previous layer finished, the
layer serviced as T3.1 left it; after the last word the next token's remaining
layers are encoded, the cursor advances and the next command becomes the held
one, to be committed by the next produce after the loop's stop check. The stop
path is unchanged: nothing runs past the stop. The drain invariant in full: every
abnormal exit of the token's loop abandons the pending plan, publishes every
remaining armed value as failed, waits for the running command with
`awaitCompletion`'s ten-second deadline, and unwinds; `awaitCompletion` is also
the fallback of the boundary wake after its first second, and the word wake keeps
polling its word to the same deadline (the close's review fold: the token's
command cannot complete before the host has serviced every later layer, so a
completion wait there would have turned any read slower than a second into a
certain failure), so a wait nothing will publish ends as `commandBufferFailed`
naming the layer or the boundary rather than a hang, and the drain on the throw
path then releases the GPU. A failed command is described by `describeCommandBufferError`: the encoders
that faulted by label and the affected count from `MTLCommandBufferEncoderInfo`
when the option is on, the plain error otherwise. Retired: the per-layer command
buffers and their records (`HeldLayerCommands`, the deferred GPU records, the
router-wake record and `path_router_wake_ms`, `runSync` on decode, the separate
embed command), and the prefetch race split with its seven rows (its instrument
was the tail command's GPU window, which is the token's now); the kernel stats'
per-layer roles collapse to one `token` row per token and the transitions to the
token boundary's gap. In their place `DecodeWordClock`: the first routed layer's
wall from the commit, each later layer's from the previous word, the boundary's
from the last word to the token word, printed under the kernel stats as `Shrike
word_clock tokens=N first_ms=… layer_ms=… boundary_ms=…` and read by
`tools/decode-rows.py` as the layer rows' sum, the slowest layer and the boundary.
Tests: the faulted encoder named and the affected counted from a synthetic error,
a wait on a command that never completes ending at its deadline naming the layer,
the word clock's accounting, and on the toy runner a failed expert read at layer
1 naming its layer, hanging nothing and leaving the runner reusable after a
reset; the gpt-oss and Qwen toy runners' existing decode and prefill tests on the
one-command path; the prefetch race split's tests gone with it; 1,280 tests in
175 suites. The four gates; the golden identical on all four profiles bare and
configured on both boxes.

*The arms* (measured on the mini at the T3.2 tree, three arms per shape
interleaved, two production lifetimes each, the first request of each: the bare
launch, the production configuration, and the production configuration with the
per-encoder error status off; the answers identical in length across the arms
and to T3.1's on every shape, 226 / 369 / 300 / 353):

| shape | arm | misses per token | io ms | overflow per token | cells leased peak | the token ms | tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| the card | bare | 19.9 | 14.3 | 0.00 | 8 | 57.3 to 58.3 | 17.2 to 17.4 |
| | configured | 15.7 | 11.6 to 12.3 | 0.00 | 6 | 54.4 to 55.8 | 17.9 to 18.4 |
| | configured, no error status | 15.7 | 11.6 | 0.00 | 6 | 54.3 to 55.2 | 18.1 to 18.4 |
| the 300 | bare | 19.3 | 14.0 | 0.00 | 8 | 56.4 to 57.2 | 17.5 to 17.7 |
| | configured | 16.5 | 12.2 to 12.8 | 0.00 | 7 | 54.7 to 55.6 | 18.0 to 18.3 |
| | configured, no error status | 16.5 | 12.2 to 12.8 | 0.00 | 7 | 55.4 to 56.0 | 17.9 to 18.1 |
| the 1k | bare | 18.9 | 13.6 | 0.00 | 8 | 56.6 to 56.7 | 17.6 |
| | configured | 15.0 | 11.1 | 0.00 | 7 | 53.6 to 54.5 | 18.4 to 18.6 |
| | configured, no error status | 15.0 | 11.1 | 0.00 | 7 | 53.6 to 54.3 | 18.4 to 18.6 |
| the 7k | bare | 18.0 | 13.0 | 0.00 | 8 | 58.3 to 58.6 | 17.1 to 17.2 |
| | configured | 14.2 | 10.5 to 10.8 | 0.00 | 8 | 55.9 to 56.1 | 17.8 to 17.9 |
| | configured, no error status | 14.2 | 10.5 | 0.00 | 8 | 55.5 | 18.0 |

Against T3.1's arms at the same configuration (T3.1 / T3.2, the token ms): the
card 54.9 to 55.6 / 54.4 to 55.8, the 300 55.6 to 55.8 / 54.7 to 55.6, the 1k
54.2 to 54.8 / 53.6 to 54.5, the 7k 55.9 / 55.9 to 56.1; the bare arm 57.7 to
58.7 / 57.3 to 58.3, 56.8 to 57.0 / 56.4 to 57.2, 57.0 to 57.8 / 56.6 to 56.7,
58.7 to 59.3 / 58.3 to 58.6. The misses per token, the io and the leased peak
identical to T3.1's on every row.

*The instruments, the same lifetimes.* The kernel stats' one `token` row spans
53.2 to 58.0 ms of GPU time per token, within half a millisecond of the token
itself on every arm: the GPU is busy through the token and idle only at its
boundary. The boundary is now measured directly, as the gap between consecutive
`token` commands: 0.26 to 0.34 ms per token on every arm and shape (M), the
number v18 Task 4 read as 0.25 and the whole of Shape B's prize in T3.3. The
word clock: the first routed layer's word 0.70 to 0.77 ms after the commit, the
thirty-nine later layers 47.5 to 52.1 ms between them (layer 1 the slowest at
2.4 ms, the middle layers 1.1), and the boundary 5.5 to 5.8 ms from the last word
to the token word, which is layer 39's fixup, the head's vocabulary GEMV, the
sampler and the embed; the three sum to the token within 0.3 ms.

**Reading.** Flat within the drift, leaning faster by what the forty command
boundaries were priced at: the token 0.2 to 0.6 ms below T3.1's on most rows in
both arms (T6.7's 0.2 to 0.4 ms, C, from about 10 µs a boundary) and level on
the rest, the misses, the io and the answers unchanged, no overflow. The
per-encoder error status costs nothing the arms can see: the two configured
arms sit inside each other's spread on every shape, with the sign changing from
shape to shape, so the option stays on and its knob is not shipped (the known
names stay fifteen; the arm's variable existed for these lifetimes only). The
structure the fold needs is complete: the token is one command, the host feeds
reads and publishes values, every value has a drain, a failed command names its
encoder, a failed read names its layer, and the boundary gap is a row the
kernel stats print. T3.3 moves the commit ahead of the stop check and takes
that row.

**T3.3 Committed ahead, Shape B (2026-09-17, `d356a8e`; the gates, the golden
on both boxes and the continuation gate at that tree; the arms at T3.4).**

*What was built.* The GDN state and the conv tail of every linear layer in
two parities (`GDNStateManager`, `[parity][layer]`, 61.4 MiB more on
ornith15): `gdn_conv_mix_decode` and the `gdn_delta_step_decode` pair take
the tail and the state entering the step and a `tail_out` / `state_out`
leaving it, the arithmetic unchanged (every element read before its row is
written, so the same buffer twice is the old in-place step); a decode pass
reads the parity holding the state at the cursor and writes the other, and
the cursor's advance flips the runner's parity; prefill, the snapshot and the
restore work in place on the cursor's parity; `reset` zeroes both. The next
token's command is committed after the current token's last word: the
boundary encoders need the caller's sampler, so `BoundaryLogitProducer`'s
sampler closure now takes the position of the pass it ends and the word its
token goes into, and the runner encodes it for this pass when the pass is
fresh and for the next pass at the end of every pass; the loop passes `last`
on the pass whose boundary sample would reach max tokens, and that pass
commits nothing ahead. Two boundary words by token parity, the runner's,
since the pass committed ahead would otherwise overwrite the one word before
the host read it; the sentinel goes into a token's word at its commit. At the
loop's exit on every path (the stop token, a stop string, the external stop,
a disconnect's cancellation) `releasePassAhead` publishes the forty values
the pass committed ahead waits on as failed, so it runs through with its
fixups skipped during the finish frames and the client's turnaround; the
wait is the next entry point's (`reset`, `prepareForContinuation`, `rewind`,
`restoreInferenceState`, `prefillChunked`, a fresh `produce`, the CLI's
`settle` before exit), counted as `drained_passes` and `drain_ms` on the
runner line after the submission that waited it out. The extra
pass's trace rows and counters never exist: the host's word loop never runs
for it, and no `token` row is recorded for a drained command. The word clock's
first row for a token committed ahead runs from the previous token's word,
not the commit. The kernel stats' `gap token->token` row is the boundary gap
the commit ahead removes.

*Two deviations from the box, on the tree's evidence.* The cursor is not
rewound: a pass's cursor advance sits at the end of its own word loop, and
the extra pass's loop never runs, so after the stop the cursor is where T3.2
left it (the parity at the cursor is the one the stop saw) and
`rewind(to:)` stays refused under GDN state, gaining only the drain. The
prompt cache's settle does not wait for the drain: the snapshot reads the
cursor's parity and the rows below the cursor, which the extra pass never
writes, so the finish frames and the settle go out as at T3.2 and the pass
runs through behind them; the wait lands at the next request's entry, by then
usually complete.

*Tests.* The GDN decode step into the other parity matching the in-place
step bit for bit with the input untouched; on the Qwen toy, the state the
stop saw surviving the extra pass and its drain (the continuation's logits
identical to the synchronous path's, the timeline empty, the ring without
leases), the last pass committing nothing ahead, a reset during the extra
pass leaving the runner reusable, a snapshot taken during the extra pass
restoring to the same logits, the release letting the pass run through before
the drain waits, `settle` idempotent; the loop naming only the pass before
max tokens as the last and releasing the pass ahead on a stop token and a
stop string; the two-turn continuation gate as the golden's `turns-lh`
profile (the CLI's `--follow-up`), its reference captured at `4df0be3` on
both boxes. 1,289 tests in 176 suites; the four gates; the golden identical on
all five profiles bare and configured on both boxes.

*The arms* (measured on the mini at the T3.3 tree, two arms per shape
interleaved, two production lifetimes each, the first request of each: the
bare launch and the production configuration; the answers byte-identical to
T3.2's on every arm and shape, all twenty responses, and so identical in
length, 226 / 369 / 300 / 353):

| shape | arm | misses per token | io ms | overflow per token | cells leased peak | the token ms | tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| the card | bare | 19.9 | 14.3 | 0.00 | 8 | 56.6 to 57.5 | 17.4 to 17.7 |
| | configured | 15.7 | 11.6 to 12.1 | 0.00 | 6 | 54.0 to 54.4 | 18.4 to 18.5 |
| the 300 | bare | 19.4 | 14.0 to 14.3 | 0.00 | 8 | 56.6 to 56.7 | 17.6 to 17.7 |
| | configured | 16.6 | 12.3 | 0.00 | 7 | 54.3 to 54.8 | 18.2 to 18.4 |
| the 1k | bare | 18.9 to 19.0 | 13.9 to 14.2 | 0.00 | 8 | 56.8 to 56.9 | 17.6 |
| | configured | 15.0 to 15.1 | 11.1 | 0.00 | 7 | 53.0 to 53.3 | 18.8 to 18.9 |
| the 7k | bare | 18.0 | 13.1 | 0.00 | 8 | 57.4 to 57.6 | 17.4 |
| | configured | 14.3 | 10.6 | 0.00 | 8 | 54.9 | 18.2 |

Against T3.2's arms at the same configuration (T3.2 / T3.3, the token ms):
the card 54.4 to 55.8 / 54.0 to 54.4, the 300 54.7 to 55.6 / 54.3 to 54.8,
the 1k 53.6 to 54.5 / 53.0 to 53.3, the 7k 55.9 to 56.1 / 54.9; the bare arm
57.3 to 58.3 / 56.6 to 57.5, 56.4 to 57.2 / 56.6 to 56.7, 56.6 to 56.7 /
56.8 to 56.9, 58.3 to 58.6 / 57.4 to 57.6. The misses per token, the io, the
overflow and the leased peak identical to T3.2's on every row.

*The instruments, the same lifetimes.* The boundary gap, the gap between
consecutive `token` commands in the kernel stats: 0.033 to 0.038 ms per token
on every arm and shape (M), against T3.2's 0.26 to 0.34; what remains is the
driver's turnaround between two commands already queued. The `token` row's
GPU span now equals the token within 0.2 ms on every arm (52.8 to 57.3 ms):
the GPU is busy through the token and its boundary both. The word clock: the
first routed layer's word 0.57 to 0.59 ms after the previous token's word
(T3.2's 0.70 to 0.77 after the commit; the commit, the driver's start and
the host's round trip are gone from it, the embed and layer 0 remain), the
thirty-nine later layers 47.1 to 51.5 ms, the boundary 5.44 to 5.79 ms from
the last word to the token word, unchanged; the three sum to the token
within 0.3 ms. The drain: the rig's first request of a lifetime has no
predecessor, so its line reads zero; the answer's stop is followed by the
prompt cache's settle, which re-prefills the closed turn's two tokens
(`settle_rewind`), and that submission is where the pass committed ahead is
waited out, after the completion's finish frames: on a card lifetime run for
it, the second request's line reads one drained pass and a wait of 0.000 ms,
the pass having run through during the finish frames and the cache's capture
(the release at the loop's exit is what makes that so; a drain that only
published at the next submission would have charged the whole pass there).

**Reading.** The prize taken as pre-registered: the boundary gap 0.26 to
0.34 ms per token at T3.2 is 0.033 to 0.038 at T3.3 on every arm and shape,
and the token is faster by about that or more on every configured row (0.4
to 1.2 ms, the larger differences inside the mini's drift) and on two of the
four bare rows, level on the other two; the misses, the io and the answers'
bytes unchanged, no overflow. The GPU now runs from one token's embed into
the next token's first layer with nothing of the host between them: the fold
is complete in its ruled shape.

### Task 4, held: the attention row's fixed part (B3, B4)

Only on S0.6's number and Davor's ruling.

## The chapter's close (2026-09-18)

**The tally, the mini, two production lifetimes per shape, from the opening ledger at
the v19 close's build to T3.4's arms under the production configuration:**

| shape | misses per token | io ms per token | the token ms | tok/s | the move |
| --- | ---: | ---: | ---: | ---: | ---: |
| the card (2k) | 20.1 to 15.7 | 14.7 to 11.6 to 12.1 | 57.8 to 58.7 → 54.0 to 54.4 | 17.0 to 17.3 → 18.4 to 18.5 | +6.4 to +8.8 % |
| the 300 | 20.0 to 16.6 | 14.9 to 15.1 → 12.3 | 57.4 to 58.5 → 54.3 to 54.8 | 17.1 to 17.4 → 18.2 to 18.4 | +4.6 to +7.6 % |
| the 1k | 18.9 to 15.0 to 15.1 | 14.0 → 11.1 | 57.4 to 57.5 → 53.0 to 53.3 | 17.4 → 18.8 to 18.9 | +8.0 to +8.6 % |
| the 7k | 18.5 to 14.3 | 13.7 → 10.6 | 58.5 to 58.8 → 54.9 | 17.0 to 17.1 → 18.2 | +6.4 to +7.1 % |

Two levers, both real and both free. The pool's allocation (Task 1: the per-layer
slot table from the production miss profile with segmented LRU) took the misses per
token from 19 to 20 down to 14 to 17 and the io by 2.6 to 3.4 ms per token, +4.4 to
+7.8 % tok/s at its own arms; the fold (Task 3: the agreed cells, one command per
token, the commit ahead) took the host out of the token's critical path, the forty
command boundaries and the token boundary with it, about 0.5 to 1.5 ms per token on
the bare arm (56.6 to 57.6 ms against the opening's 57.4 to 58.8) and 0.4 to 1.2
against T3.2's on the configured one, the boundary gap 0.26 to 0.34 ms per token to
0.033 to 0.038. Every answer identical in bytes across every arm since Task 1's;
the golden byte-identical on both boxes at every commit; the class-1 gate never
opened.

**The count:** twenty-one commits (`043beed` to `a4c6431`) plus the close's; 50
source, test and tool files changed, 3,358 insertions and 1,587 deletions outside
the docs and baselines; no Metal kernel added or retired, 66 in the tree (the
classifier gained two grids, the speculative pair a status word and a fallback
array, the GDN decode pair an out pointer); tests 1,258 to 1,289, suites 174 to 176;
two environment names added under the tripwire, 13 to 15 (`SHRIKE_EXPERT_SLOT_TABLE`,
`SHRIKE_EXPERT_POLICY`, both product settings, both the mini's launch
configuration); two CLI flags (`--tokenize`, `--follow-up`); one golden profile per
box added (`turns-lh`, the two-turn continuation); no rig shape added; the runner
line lost eight rows that measured paths that no longer exist and gained four;
the replay's table mode and the coverage tool's distance for the pricing.

**What the chapter settled.**

- The split is the lever on the pool, not the predictor. Step zero's replay priced
  the token-id table at 0.24 ms at its precise layer for a second prediction source
  and its reads; the split of the same 5,120 cells by the production miss profile
  saved 2.4 to 3.0 ms per token where 1.3 to 3.1 was modelled, and SLRU on top 1.8
  ms where it mattered. The table is designed and on record, not built.
- The width is closed at distance one by the window's bandwidth and at distance
  two by a number: recall 0.40 to 0.45 of the remaining misses at precision 0.08
  to 0.09 against the bar's 0.13.
- The fold's structure holds in its ruled shape. The token is one command; the
  host feeds reads and publishes values; every value reserved at an encode is
  published by the batch, by the host at the word or by the drain, and every wait
  has a deadline that names what it waited on; a failed command names its encoder,
  a failed read its layer; the GPU runs from one token's embed into the next
  token's first layer with nothing of the host between them.
- Shape B's price was the wait, not the pass. The design note priced the
  unguarded drain at 40 to 45 ms once per answer and weighed a cancel word in five
  kernel families against it. Released at the loop's exit and waited out at the
  next GPU submission, the pass committed ahead of a stop runs through during the
  finish frames and the cache's capture, and the wait measures 0.000 ms; the
  guard was never needed.
- Two of the box's steps were not needed, on the tree's evidence: the cursor
  needs no rewind, because a pass's advance sits at the end of its own word loop
  and the extra pass's loop never runs; the settle needs no drain, because the
  snapshot reads the cursor's parity and the rows below the cursor, neither of
  which the extra pass writes. Both are held by construction and by the toy tests,
  and the continuation gate checks the whole path end to end.
- A counter charged between two requests' snapshots reads zero on both lines. The
  drain's first rows were all zero because the settle's re-prefill did the waiting
  after one runner line and before the next snapshot; the rows now read against
  what the last line reported.
- The mini drifts, still; interleaved arms, two lifetimes per shape, remain the
  only readings that survive it.

**The whole-branch review (a fresh reader over `c4a96d6..a4c6431` with the records
as the specification; the report at `~/.claude/handoffs/archive/shrike-v20-t33/close-review.md`).**
Five findings, each verified against the code before its fold, folded into the commit
that owned the defect:

1. *KV growth under a running command* (into T3.3's commit). `KVCacheManager.reserve`
   replaces a full layer's buffers and copies only the rows below the cursor; since
   T3.2 the reserve for the next token ran after the current token's commit, so at a
   growth boundary (position 8,191, 16,383 and 32,767 on the mini's context) the row
   the running command was writing would have been lost from the new buffers and every
   later layer would have attended over it for the rest of the conversation. Silent:
   the golden's profiles and the rig's shapes never decode across 8,191. Now a pass
   whose next position would grow the KV commits nothing ahead, growth happens at the
   next produce with nothing in flight (the reserve moved behind the drain), and a
   continued pass that finds no command ahead runs fresh from the token the previous
   boundary's sampler wrote (`continuedTokenFromTheWord`); `needsGrowth` is the guard.
2. *The word wake's fallback* (into T3.2's commit). Past its first second the wake
   waited for the token's command to complete, which cannot happen before the host has
   serviced every later layer, so any read slower than a second became a certain
   failure ten seconds later. The wake now keeps polling its word, gently, to the
   deadline, checking the command's status as it goes.
3. *The overflow victim among the route's own hits* (into T3.1's commit).
   `reserveOverflowSlot` chose its victim with only the loading slots reserved, so
   with the ring out of cells it could evict an expert the classifier had just
   resolved as a hit, tearing the bytes under the layer's speculative work and
   throwing at the next plan. The route's experts are now protected before the victim
   is chosen; never fired on the arms.
4. *The drain's swallowed fault* (into T3.3's commit). A drained command's fault or
   deadline went unrecorded; it is now counted (`drain_failures` on the runner line)
   with the last error's description kept.
5. *SLRU's overflow placement* (into T3.1's commit). An overflow read was resident by
   the time its own route's deferred plan ran, so the plan promoted it on the placing
   route instead of landing it in probation as the record says; the streamer now
   keeps the placing route's overflow slots out of that one promotion. A fidelity
   fix, bounded by the overflow rate.

The tests: `needsGrowth`, the protected overflow victim, the continued pass with no
command ahead running fresh from the word and refused without one; 1,292 tests in 176
suites. Nothing real was found in the drain invariant's paths, the parities, the
boundary words, the `last` flag, the seed mapping, the status-word ring, the ring's
leases, the CLI's resume or the server's rows. ThreadSanitizer over the whole suite:
clean at `a4c6431`'s tree before the folds and at the final tree after them.

**What remains, and where it went.** The read itself: 0.73 to 0.80 ms at p50 per
miss on the mini's drive is untouched by every chapter since v13, and the token's
largest row is now the misses' count times that cost; the count moved here, the cost
is the drive's. The attention row's fixed part (B3, B4) went to a chapter of its own
with the instrument it needs (S0.6). The token-id table (A9) stays designed on record,
and the table shipped beside the model or derived at load from a profile the model
carries is Task 1's follow-up. Distance two at width eight is noted in S0.5b. Lossless
compression is v21. The KV row at the cursor after a stop is written by the extra
pass and read by nothing, overwritten by the next prefill; recorded, not a defect.
The cancel word stays unbuilt.

## Method

v19's Method holds ([v19-scan-rewrite.md](v19-scan-rewrite.md)): the four gates per
commit, ThreadSanitizer once at the close, two production lifetimes per shape on the
mini, the first lifetimes after a deploy discounted, `prefetch_late` lifetimes read
as noise, subagents for the gates and the rigs with verbatim diagnostics, every cost
graded with a range and tasks ranked by the floor, the mini's drift met by
interleaved arms. Added this chapter:

- Every arm on this surface carries **reads per token** and **cells held** beside
  its misses and its io, since the drive's budget and the cells are what the design
  allocates; a lever that saves misses by spending reads is read on the io row, not
  the misses row.
- The replay is the pricing instrument and production is the verdict: a replay's
  misses saved is **M for the policy** and **T for the runner**; the runner's
  misses per token and io on the rig are the verdict.
- A diagnostic that touches a kernel (S0.5) ships behind the trace env var and the
  golden proves it inert before its capture is read.

## Numerics policy

Class 1 throughout: every commit byte-identical on all four golden profiles on both
boxes. The predictor and the policy change which experts are resident and when,
never which experts compute or in what order; misses per token and the io may move,
the answer does not. The fold changes who dispatches and in which command; nothing
about the arithmetic. Class 2 and 3 stay closed.

## Out of scope, and where it went

- **Lossless compression** (H2): v21, scoped to the expert kernels and the head.
- **The draft through a forward pass** (Q3's other form), **speculation revisited**
  (H1) and **the draft in the spin-wait** (H3): after v21; this chapter prices the
  draft in the table form only.
- **The other class-2 avenues** (J1, G1, the scan's arithmetic levers): after v21,
  on v19's instrument.
- **The ANE**: unchanged.
- **The attention row's fixed part** (B3, B4): priced here in S0.6, built only by
  ruling.

## Risks

- **The read budget.** A batch issued at the token boundary lands during layers 0-3,
  which carry 22 % of the demand; if its reads share the drive with those layers'
  demand reads the window grows where it was meant to shrink. The batch's issue
  point and the demand lane's priority are design questions S0.2's reads-per-token
  row informs and Task 1's arms decide; the reader's batches and the in-flight
  budget are the levers.
- **The cells.** The ring refuses 16.9 predictions per token today for lack of a
  cell; a deep-layer batch holds cells for tens of milliseconds. S0.2 reports the
  peak cells held; approach B exists for the case where A starves.
- **The table's cold start.** A fresh session's table is empty until the answer
  has history; the last-occurrence predictor covered 56 to 70 % of positions on
  the archived answers. Seeding from the prefill is priced in S0.5; without it the
  first answer of a session pays today's window.
- **The draft's false lead.** A wrong draft names another token's experts; at
  layer 0 the random line is 0.07. S0.3 prices the trade and the draft is built
  only if the net is positive at layer 0.
- **The fold's edges.** The stop path against in-place GDN state, the error
  surfacing per layer, the cancel with two in flight: the design note precedes the
  build and Davor rules on it.
- **The mini's drift.** Twice in one day in v19 an unchanged kernel slowed 1.77×
  mid-run; interleaved arms are the only readings that survive it.
