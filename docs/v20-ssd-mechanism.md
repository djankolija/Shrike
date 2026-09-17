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

Davor's ruling: (pending).

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

### Task 1: the predictor (class 1)

What step zero supports, in this order of preference: the token-id table with its
batch on the ring's landing path at the layers and width S0.2 names; the wider
probe at the width S0.5 names; the draft for layer 0 if S0.3 earns it; the table
seeded from the prefill if S0.5 earns it. The pre-registration names the rows: the
misses per token by layer group, the reads per token, the cells held, the io ms and
the token on four shapes, graded T with a range from the replay's numbers; the
answers expected identical (class 1).

### Task 2: the policy and the split (class 1; only on S0.4's number)

The predicted-future eviction and the per-layer slot count, each its own commit and
arm; misses per token the row; the answer identical.

### Task 3: the agreed cells and the fold (class 1; structure)

The design note first (the four edges, for Davor's ruling before a line is built),
then the agreed cells (v18's T2.1 to T2.5 as written), then one command per token
with two in flight, the cancel and the stop path, the error surfacing per layer.
Golden identical; the wall expected flat within the arms' resolution; misses per
token may drift since the plan runs later, recorded.

### Task 4, held: the attention row's fixed part (B3, B4)

Only on S0.6's number and Davor's ruling.

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
