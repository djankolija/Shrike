# v18: the avenues (a running record of ideas and findings)

A working document, opened 2026-09-08 after v17's close, before any chapter is chosen.
The rule for this file: open every avenue first, price what can be priced from data
already on disk, and only then decide on spikes and sequencing. Nothing here is a plan.
When a chapter is chosen it gets the usual pair (`v18-<name>.md` and
`v18-implementation-plan.md`) and this file becomes its history.

Every number carries one of three labels. **Measured** is a counter or a clock on the
mini. **Modelled** is arithmetic on measured inputs. **Remembered** is a claim from an
earlier chapter's record, cited but not re-run this session.

## 1. Orientation

Where we started and where we are, on the mini (Apple M1, 16 GB, `--ram-budget 8G`,
128 of 256 expert slots per layer resident):

| point | tok/s | source |
| --- | ---: | --- |
| before the optimisation chapters | ~6 | remembered (Davor) |
| v17 close, the 300 / the 1k / the card | 16.3 / 16.2 / 15.6 | measured, [v17-consolidation.md](v17-consolidation.md) close |
| today, 52-token prompt, warm cache, 565-token answer | 19.5 | measured, the mini's live log |
| the byte ceiling at 100 % hits and every kernel at the roof | 34.7 | modelled, below |

**The ceiling's derivation.** v10's two measured laws
([v10-decode-kernel-mergers.md](v10-decode-kernel-mergers.md), 2026-08-31): the
intrinsic decode traffic is 1,800.5 MB per token and the empirical streaming roof on
the M1 is 62.5 GB/s. 1,800.5 / 62.5 = 28.8 ms = 34.7 tok/s. A per-row byte count from
the manifest (`ornith-1.5-35b-a3b-4bit`: 4-bit affine, group 64; 40 layers, 30 GDN
and 10 full attention; 256 experts, top-8, 1,769,472 bytes per expert; hidden 2048;
vocab 248,320; 16 query heads over 2 KV heads at head dim 256) lands within 2 % of
1,800 MB. The ceiling is not a target. It says the gap is real: 60.9 ms per token
today against 28.8 at the physics.

## 2. The ledger at the v17 close

Source: the mini's own kernel-role and gap counters (`Shrike kernel role=` and
`Shrike gap` lines, `SHRIKE_KERNEL_STATS=1`; the `Shrike runner` line,
`SHRIKE_RUNNER_STATS=1`) in the v17 Task 4 arms logs, captured 2026-09-08 on the
final tree's pre-fold build (`~/.claude/handoffs/archive/shrike-v17-t4/v17t4-arms/`,
`server-mini-v17t4-prod-{1,2}-{card,d512-300,d512-1k}.log`). The close's fold measured
flat within drift with identical answers, so this is production as it runs. Two
lifetimes per shape agree to the tenth; the first is shown. Per decoded token, the
cold 512-token answer of each shape:

| row (ms per token) | the 300 | the 1k | the card (2k) | bytes at the roof |
| --- | ---: | ---: | ---: | ---: |
| GPU idle: the miss window (`moe_phase1_hit` to `moe_phase1_miss_fixup_phase2`) | 13.13 | 12.35 | 13.00 | 0 |
| GDN block + the layer's tail, 30 layers (`attn_layer_linear`) | 15.73 | 15.54 | 15.44 | ~11.6 |
| the speculative command, 40 layers: the shared expert, and the 8 experts on all-hit layers (`moe_spec_routed`) | 10.88 | 11.57 | 11.42 | ~6.6 |
| the hit command + the fixup + the adopted fixup, on the 15.7 layers with a miss | 5.08 | 4.62 | 4.73 | ~3.6 |
| full attention block + tail, 10 layers (`attn_layer_kv`) | 5.40 | 7.21 | 9.30 | 2.6 + 0.17 per 1k ctx |
| LM head (`head_logits`) | 4.57 | 4.56 | 4.54 | 4.6 |
| GPU idle: the spec pass to the phase-1 submit | 1.84 | 1.72 | 1.70 | 0 |
| GPU idle: attention chain to MoE chain and back, 80 per token | 2.64 | 2.89 | 2.55 | 0 |
| GPU idle: the token boundary (head to sample to embed to layer 0) | 0.86 | 0.85 | 0.85 | 0 |
| GPU idle: the spec pass to the head | 0.14 | 0.15 | 0.15 | 0 |
| sample + embed | 0.16 | 0.16 | 0.16 | ~0 |
| **sum** | **60.43** | **61.62** | **63.84** | **~28.8** |
| wall (`decode_tok_s`) | 60.87 | 61.97 | 64.21 | |

The ledger closes to 0.4 ms on every shape. The roof column is modelled: bytes per row
at 62.5 GB/s (re-derived 2026-09-09 from the architecture document's account of what
each command holds; the column's sum is v10's 1,800 MB to the millisecond).

**What the rows are.** Every one of the 40 layers is a token-mixing block followed by
an MoE block. The token-mixing block is GDN on 30 layers (Gated DeltaNet: a recurrent
block with a fixed-size state, no context scan) and full attention on 10 (the KV
cache, the context scan). The MoE block on every layer is a router choosing 8 of 256
experts, plus one shared expert that always runs. The runner encodes each layer as
two commands. The attention command holds the token-mixing block and, folded into it,
the layer's tail: the residual, the post-attention norm, the router, the next layer's
router on the same hidden state (the prefetch's probe) and the residency classifier;
its GPU time is the `attn_layer_linear` or `attn_layer_kv` row. The speculative
command holds the shared expert and the routed phase 1 and phase 2 for the layer's
eight experts, encoded before the host knows the route; the classifier, the tail's
last kernel, checks the router's top-8 against the residency table on the GPU and
sizes the routed grids to the full eight when all are resident and to zero when any
is missing. So on the 24.3 all-hit layers per token the speculative command is the
whole MoE block and no host round trip occurs; on the 15.7 layers with a miss it
runs only the shared expert, and the host builds a hit command (phase 1 for the
resident experts) and a fixup (the event wait on the read, phase 1 for the missing
experts, phase 2 over all eight, the residual). Phase 1 is the gate and up
projections (1.18 MB per expert), phase 2 the down projection (0.59 MB).

Counts per token on the 300 (measured): 320 expert lookups, 20.0 misses (hit rate
93.76 %), the fixup on 13.6 layers, the phase-1 hit pass on 15.7 layers, the adopted
fixup on 2.1 layers, the speculative pass on all 40. `io_fixup_wake_ms` 2.10 ms per
token: the time from the bytes landing to the fixup kernel's GPU start, inside the
miss window.

GPU busy on decode roles is 41.8 ms (69 %), GPU idle 18.6 ms (31 %) on the 300.

### The shadow ledger (Davor's rule, 2026-09-09)

"X is hidden by Y" describes today's dependency graph, not X: it holds only while
nothing shortens Y by more than X's slack. Every hidden item stays here with its cost,
what hides it, its slack and its exposer, so that the task which shortens a Y knows
what it will find on the path and pre-registers it (the design document's method).
Costs measured on the mini in Task 1's arms (the 300); the slack modelled from the
ledger's rows.

| item | ms per token | hidden by | slack | exposer |
| --- | ---: | --- | ---: | --- |
| the fixup's build and commit on miss layers (`path_fixup_build_ms` 0.17, `path_fixup_commit_to_kernel_ms` 0.40) | 0.57 | the read's flight | about 1.0 ms a layer | a read under about 70 µs; not this drive |
| the word's visibility on the 24 all-hit layers (61 µs each; A8) | 1.5 | the speculative command's own work | about 0.3 ms a layer | a speculative command under 100 µs; bandwidth-bound, not plausible |
| the all-hit layers' plan, pin, submit and the next layer's commit | about 0.2 | the same | about 0.3 ms a layer | the same |
| the hit command (deleted in Task 1) | was 2.1 | the read's flight | about 1.0 ms a layer | none: deleted, a simplification paid ahead |

The rule cuts the other way too. An item on the path today can have shrinkers and no
exposer: C6's slice (the host's plan, pin and submit on miss layers, at most 0.34 ms
per token by T2.0) is serial now, but every lever ahead of it (A9 hiding the word,
A1 and A2 cutting misses) turns miss layers into hit layers and takes the slice into
the shadow with them, so its price is a ceiling, not a floor.

## 3. The row the fixed shapes hide: attention grows with context

Only the ten full-attention layers scale with context; the thirty GDN layers keep a
fixed state. The slope, `attn_layer_kv` per decoded token against context:

| context (prompt rows) | ms per token | source |
| ---: | ---: | --- |
| 289 | 5.40 | measured, this tree |
| 1,069 | 7.21 | measured, this tree |
| 2,125 | 9.30 | measured, this tree |
| ~12,000 | 28.6 | measured, v13's tree (`shrike-v13-t0/t0-out/server-mini-t0-carry-mini-12k-after300.log`) |

**2.0 to 2.3 ms per 1,000 context tokens per decoded token**, linear across the four
points and two trees. The KV cache is stored at 8 bits in production (the server's
`--kv-bits` default, `ServerArguments.swift:92`), 1 KB of K and V per position per
layer, so the byte roof is 0.17 per 1,000 for ten layers at 62.5 GB/s: the kernel
runs at about twelve times the roof. (Corrected 2026-09-09; an earlier draft assumed
16-bit rows and said six.)

Today's live log on the mini (`/tmp/ornith.log`, 2026-09-08, Davor's Pi sessions,
measured):

| prompt tokens | cached | answer tokens | decode tok/s | ms per token |
| ---: | ---: | ---: | ---: | ---: |
| 52 | 0 | 565 | 19.5 | 51.4 |
| 54 | 0 | 483 | 18.6 | 53.7 |
| 41 | 0 | 159 | 17.6 | 57.0 |
| 503 | 0 | 98 | 15.6 | 64.2 |
| 1,733 | 600 | 140 | 14.5 | 69.1 |
| 3,114 | 1,872 | 133 | 14.9 | 67.3 |
| 4,579 | 3,246 | 133 | 14.4 | 69.3 |
| 6,073 | 4,711 | 98 | 14.2 | 70.3 |
| 7,062 | 6,170 | 944 | 13.5 | 74.2 |

The mini's production launch carries the kernel counters, so the same log has the
attention row itself at each context, on the current tree (measured, 2026-09-08):

| prompt tokens | `attn_layer_kv` ms per token |
| ---: | ---: |
| 41 to 54 | 4.7 to 5.1 |
| 503 | 5.6 |
| 1,733 | 8.6 |
| 3,114 | 11.3 |
| 4,579 | 14.6 |
| 6,073 | 17.7 |
| 7,062 | 21.3 |

2.3 to 2.4 ms per 1,000 across the day's sessions, the same line as the rig's four
points. At 7k the attention row is 21 ms of a 74 ms token, 29 %, against the miss
window's 13 ms at 18 %.

The 7k answer at 74.2 ms against the ledger's 60.9 at 300: the difference is 13.3 ms
and the slope predicts 13.6. At 7k the attention row is about 19 ms (modelled from the
measured slope) and the miss window about 13 (measured, flat with context), so at the
working contexts of a coding session attention already outweighs the drive. The short
answers (98 to 140 tokens) sit a few ms above the line: the first tokens after a
prefill carry more misses (remembered from v13, not re-measured here).

History of the row: [v11-kv-attention-inner-loop.md](v11-kv-attention-inner-loop.md)
(2026-09-01) found the slope at 8.0 per 1,000, pinned the mechanism by microbench as
traffic-bound at about 40 % line utilisation and amplified 8× by GQA (each of 16 query
heads re-reading its shared KV head's bytes), and landed the KV-head-shared partial
kernel (`attention_decode_partial_shared`) for 2.8 on the mini; later folds took it to
today's 2.0. v11's own floor estimate with honest per-dispatch walls was about 1.3 per
1,000; its named residual: line utilisation on half-row slices, the 64-chunk
threadgroup wall, the barrier cadence of the 4-position blocks. The chapter stopped
there because the miss window was the larger number at the time.

## 4. The surfaces

Five surfaces from the ledger plus two outside it. Each carries what it is, its
measured size, its floor, what earlier chapters found, and the avenues opened so far.
An avenue's status is one of: **idea** (not yet priced), **priceable offline** (a
replay or a read can price it with no model run), **measured null** (an earlier
chapter tried it and it did nothing, cited), **numerics** (changes the output; Davor's
call, needs a fresh golden).

### A. The miss window (13 ms, 21 % at 300, ~17 % at 7k)

Twenty misses per token over 13.6 layers. The first miss in a layer is a serial read
of about 0.96 ms (the layer cannot run its fixup until it lands); further misses in
the same layer overlap on the reader's four threads. Inside the window, 2.1 ms per
token is dead time after the bytes have landed. Flat with context in absolute terms.

**What hiding the reads entirely would take (worked 2026-09-08).** Every dispatch in a
token depends on the previous layer's output, so the only GPU work a read can hide
behind is the following layers'. One layer averages 1.05 ms of GPU (41.8 ms busy over
40, measured); a first-miss read costs 0.96 ms (measured). At distance one the read
and the layer are the same length, which is the race v16 measured; at distance two
the landing is safe. But distance is the smaller half of the problem:

| what the ring does today, per token on the 300 (measured, the runner's counters) | count |
| --- | ---: |
| non-resident experts in the actual top-8 (misses + landed hits) | 26.2 |
| predictions issued | 21.0 |
| of which right (adopted) | 10.3 |
| of which right and landed before the classifier (a phase-1 hit) | 6.2 |
| of which right but late (joined by the fixup) | 4.1 |
| of which wrong (reclaimed) | 10.7 |
| predictions refused for lack of a cell | 16.9 |
| misses still waited on | 20.0 |

The probe's quality is measured (v14 Task 1, v15 Task 1, the coverage tool on the
traces): at distance one and width eight it covers all of a miss layer's misses 43 to
46 % of the time at precision 0.41 to 0.47; at distance two, 34 to 37 % at precision
0.29 to 0.34 (two layers ahead costs 0.10 of coverage). Full coverage is what stops a
layer's read, and no probe reaches it: the router's decision at layer L depends on
L's input, which does not exist until L-1 is done. Coverage rises with width, and
width is paid in wasted reads. The drive binds there: decode reads today are 20
demand plus 21 speculative, 72.6 MB per 60.9 ms token, 1.19 GB/s against the 3.5 GB/s
v14 measured as the drive's decode ceiling (modelled from the counts), so reads can
rise about 2.9× before the drive is the wall. Full coverage at width sixteen would
need roughly 320 reads per token, 2.4× the ceiling. So the surface's floor is set by
the drive's bandwidth times the probe's precision, not by its latency: the reads
cannot all be hidden on this box, and the question is how much of the 13 ms a wider
or better-placed net buys under the bandwidth line. The two-distance queue, which
would have used the idle half of each window for the layer after next, priced at 0.3
to 1.2 useful adoptions per token on the replay and was not built (v15, remembered).

**The window is bandwidth-bound while it is open (Claude, 2026-09-09; modelled from
measured inputs).** One expert is 1.77 MB, and at the drive's 3.5 GB/s decode
ceiling its transfer alone is 0.51 ms. Twenty demand reads per token are 10.1 ms of
transfer, and the window measures 13.1: the remainder is the 2.1 ms of wake dead time
and about 1 ms of per-layer latency and issue. v14's law reads the same way: the
first miss in a layer costs 1.1 to 1.2 ms (latency plus transfer plus wake) and each
further miss 0.61 to 0.72 (transfer, overlapped only in latency). The drive is idle
for two thirds of the token, during the GPU-busy stretches, and saturated inside the
windows. Three consequences: no change to the wake, the thread count, the placement
or the join can take the window below about 10 ms while the twenty demand reads stay
inside it (A5 and A8 are worth at most their 2 and 0.8 ms); the levers that move the floor are lead
(reads issued during the idle two thirds: A0, A9), fewer misses (A1, A2) and fewer
bytes per read (H2); and the "further miss" cost is the drive's transfer rate, so
splitting the fixup per expert (running the first landed expert while the second
lands) hides 30 µs of compute per expert and nothing else.

- **A0. Width at distance one** (priceable offline, zero runs). The archived probe
  captures (`shrike-v14-t1/step1/t1-capture/prefetch-t1-d1-*.jsonl`, all three
  shapes) hold every position's top-8 and the probe's ranking; `tools/prefetch-coverage.py
  --top-m` reports coverage, precision and wasted reads at widths 12, 16 and 24. The
  prize is coverage times the 13 ms; the cost is reads against the 2.9× headroom and
  the ring's cells (nine per layer today). This is the first thing to run.
  **Priced 2026-09-09 (S0.3, measured on the v14 captures):** blocked beyond width
  eight. The captures hold only the probe's top-8, so widths 12, 16 and 24 reproduce
  width eight's numbers exactly. At width eight, full-layer coverage is 0.442 / 0.428
  / 0.462 (the 300 / the 1k / the card) at distance one and 0.342 / 0.337 / 0.367 at
  distance two. Pricing a wider net needs one capture with the probe's top-24 logged,
  a diagnostic change and a model run; it goes to v20's step zero.
- **A1. Fewer misses by a better slot split** (priceable offline). 128 slots per layer
  is uniform. If some layers route more concentrated than others, their spare slots
  belong to the flat layers. `tools/expert-pool-replay.py` reproduces the box's miss
  totals exactly from a route trace (v14 step zero), so a per-layer allocation can be
  priced with zero model runs on the archived traces
  (`route-v17t4-prod-*-*.trace`).
- **A2. Fewer misses by a better policy** (priceable offline). Aging-LFU is the one
  policy since v17. The same replay prices alternatives: LRU, frequency over a longer
  horizon, or a policy that knows the layer's router probe. v14 step zero compared
  aging-LFU with resident-first and found the miss count the rate at every scale
  (remembered).
  **Priced 2026-09-09 (S0.1 and S0.4, measured by replay on the six T4 traces,
  pool mode without the ring, so 28 to 31 misses per token against production's 20;
  the ring's landed hits are not modelled here):** the miss profile by depth is
  U-shaped. Per token, pooled: layers 0-3 carry 6.5 misses (22 %) over 2.7 reading
  layers; 4-9 5.3 (18 %); 10-19 5.5 (19 %); 20-29 5.0 (17 %); 30-39 7.4 (25 %) over
  4.8 reading layers. Layer 0 alone misses on 66 % of positions, the highest of any
  layer, and no probe serves it (the probe predicts L+1 from L). The eviction policies
  at 128 slots, decode misses per token on the 300 / the 1k / the card: aging-LFU (the
  shipped one) 30.2 / 28.2 / 30.7; LFU identical; LRU 31.2 / 26.0 / 28.4; ARC 31.3 /
  26.1 / 28.6; SLRU 30.3 / 25.5 / 28.3; LRU-2 30.6 / 26.8 / 29.9; **Belady, the
  clairvoyant bound, 11.9 / 10.2 / 10.8.** Two readings: no online policy beats
  aging-LFU by more than 8 to 10 % on the longer shapes (SLRU, LRU) and none on the
  300, so A2 is worth about a miss per token, cheap but small; and the clairvoyant
  bound is 2.5 to 2.8× below every online policy, so the prize in this surface is
  knowledge of the next tokens' routes, not a better heuristic. That is A9's
  direction, and it names a variant for v20's step zero: a policy that evicts against
  a predicted future (prompt lookup's next tokens through the token-id table) rather
  than a past frequency, priced by feeding the replay's Belady path a predicted
  future instead of the real one. The slot split (A1) was not run: the replay takes
  one slot count for all layers, and the U-shaped profile says the split would move
  slots from the middle layers to both ends; a small patch of the replay prices it,
  deferred to v20's step zero with the rest of the SSD pricing.
- **A3. Earlier prediction** (idea; one arm measured null). The ring predicts the next
  layer's top-8 from the router probe at distance one and lands 6 to 7 hits per token.
  The two-distance queue measured null in v15 (remembered). Unexplored: predicting
  from the previous token's routing at the same layer (temporal locality), priceable
  offline from the traces by counting how often layer L's top-8 at token t+1 overlaps
  token t's.
- **A4. Faster reads** (one arm measured null). Splitting one expert read across N
  preads is null on the mini's drive (2026-09-04, remembered). Unexplored: splitting
  by matrix so the fixup's gate and up projections start while the down projection is
  still landing; that overlaps compute with the tail of the read rather than adding
  drive parallelism, a different mechanism from the null. Worth at most the fixup
  kernel's own time on miss layers; small.
- **A5. The wake dead time, 2.1 ms** (partially explored). v15 and v16 took the word
  wake, the spin host wait and the landing race; 0.155 ms per miss layer remains
  between the pread's completion and the fixup's GPU start. Unexplored: what the
  0.155 is made of now (the host's publish, the event signal, the command buffer's
  scheduling), from a trace of one miss layer. T2.0 (2026-09-09) adds the hand-off
  on the way in to the same family: `io_queue_ms`, 26 µs per miss layer, 0.35 ms per
  token, from the submit to the reader thread's `markInFlight`, a parked thread
  woken through a condition variable; a spinning demand reader is the probe, and
  the 155's own anatomy still wants the trace.
- **A6. Hide the wait** (numerics; rejected by policy). Running the next layer on the
  resident experts only and correcting later changes the output. Recorded so nobody
  re-proposes it.
- **A7. Calibration only, not an avenue.** All 40 layers' experts are 18.1 GB. A box
  that holds them makes this surface zero. The mini is the target (Davor's ruling);
  this line exists to state what the surface is worth: about 13 ms of the token.
- **A8. The word's visibility, 0.8 ms on the path** (idea; T2.0's finding, 2026-09-09).
  The other half of A5's round trip. The host sees the classifier's word 61 µs after
  the tail command's GPU end (`path_router_wake_ms`, 2.4 ms per token over the 40
  layers in Task 1's arms; v14 measured 63). On an all-hit layer it sits under the
  speculative command's own work; on a miss layer the read cannot be issued until the
  word is seen, so the landing and the fixup behind it arrive 61 µs late, 13.5 layers a
  token, about 0.8 ms. It is the largest item in the 112 µs between the tail's end and
  the pread (T2.0: 61 of visibility, at most 25 of the host's plan, pin and submit, 26
  of the reader thread's hand-off). C6 concedes it, since the miss list arrives with
  the word; A3, A9 and H3 hide it by reading before the word exists; no avenue asks
  what the 61 is made of. Unexplored, from a trace of one layer: how much is the GPU's
  drain after the last kernel, how much the driver's completion path, and whether the
  host can observe the write earlier than the command's end (whether anything runs
  after the classifier inside the tail command, a shared event signalled at the
  encoder boundary instead of a polled word, the readback buffer's storage and cache
  mode). Class 1 by construction: nothing changes which bytes move or in what order.
- **A9. A token of lead instead of a layer (Davor, 2026-09-08).** Draft-token
  speculation washed on the mini because the draft's tokens were not accepted often
  enough (v12: 21 / 32 / 76 % by shape). Token acceptance is the wrong metric for a
  different use: a draft token that is wrong can still route to most of the same
  experts as the right one, and a prediction made a whole token ahead has a lead of
  the entire pass for the deep layers, tens of ms instead of the ring's one layer.
  With that lead the drive's bandwidth, not its latency, is the only limit, and the
  reads for the next token go out as one batch at the start of the pass. Three
  questions, in order of cost:
  - **Q1, priceable offline, zero runs: how much of the routing is the token's
    identity.** The v17 arms hold the route trace (top-8 per position and layer) and
    the streamed token text per position (`tokens-*.json`). Per layer: the overlap of
    the top-8 between two occurrences of the same token text, and between consecutive
    positions. If identity predicts routing at layer L, then the sampled token t+1,
    known at the end of pass t with no draft at all, predicts pass t+1's experts at
    layer L with a lead of about L ms from a table keyed by token id, warm from the
    session's own history. The shallow layers keep today's one-layer lead.
  - **Q2, priceable offline: the miss profile by layer.** How much of the 13 ms sits
    in layers deep enough to have a lead. `tools/expert-pool-replay.py` reproduces
    the box's misses per (position, layer) from the trace.
  - **Q1 and Q2 priced 2026-09-09 (measured; the script and per-layer JSON at the
    session's scratchpad, `s02-routing-locality.*`; six T4 traces, the cold answer
    of each, every decode position matched to its streamed token).** Top-8 overlap
    per layer group 0-9 / 10-19 / 20-29 / 30-39 / all: consecutive positions 0.23 /
    0.32 / 0.35 / 0.35 / 0.31; the same token text at two positions 0.35 / 0.31 /
    0.29 / 0.31 / 0.31, and 0.61 / 0.47 / 0.42 / 0.41 / 0.48 for texts of three or
    more non-space characters; random pairs 0.07 / 0.13 / 0.15 / 0.11 / 0.11. The
    last-occurrence predictor (the top-8 at the text's most recent earlier
    occurrence) recalls 0.44 / 0.37 / 0.34 / 0.38 / 0.38 of the actual top-8 and
    covers 56 / 64 / 70 % of positions (the card / the 300 / the 1k). Wider tables:
    the union of the last two occurrences 0.48 at 12 predicted, the last three 0.54
    at 15; a frequency table's top-12 0.47, top-16 0.52. Identity locality decays
    with distance (0.38 within ten positions to 0.23 past two hundred). At layers
    0-3 identity beats the previous position three to one (0.49 against 0.16; at
    layer 0, 0.63 against 0.07), and no probe serves those layers today. Q2's
    profile is under A2 above: 22 % of the demand sits in layers 0-3 with no lead,
    78 % in layers with a lead of 4 to 39 ms. **Reading:** a token-id table is a real
    predictor, 3.5× the random baseline, weaker per expert than the hidden-state
    probe at distance one (whose per-miss recall was 0.64, remembered from v14) but
    with a lead the probe cannot have; at full width over all layers it is
    drive-bound (15 predicted per layer, about half already resident, is roughly 300
    reads per token against a ceiling near 120), so its use is selective: the deep
    layers where the lead is long and the miss share is high (30-39, a quarter of
    the demand), and layer 0, where it is the only predictor and the miss rate is
    the highest. Belady's bound (A2) says what perfect knowledge would be worth.
    The design, the width and the cell accounting are v20's.
  - **Q3, needs a run: a real draft's expert overlap when its token is wrong.** The
    MTP head is out of the runtime since v17 (git history has it; the sidecar bundle
    is on both boxes). Cheaper drafts: prompt lookup (an n-gram match in the context,
    zero GPU cost, strong on code, which is what the box serves), or a small dense
    model on the ANE or CPU. A draft on the GPU costs the token what it saves; the GPU
    is 69 % busy and is the bottleneck when it is.
  - **The economics** (modelled): 26 non-resident experts per token at precision p
    means 26 / p reads; the drive's headroom is about 120 reads per token, so p above
    roughly 0.3 fits. The ring's cells are 9 per layer, 360 in all, enough for a
    pass-ahead batch. The related family in the offloading literature (predicting the
    next token's experts from the token or the hidden state) is remembered, not
    verified; cite it properly when a design record is written.

### B. Attention at context (5.4 ms at 300, ~19 at 7k, ~37 at 16k modelled)

Section 3 has the evidence. The floor is 0.17 per 1,000 at the 8-bit KV production
runs; v11's honest floor 1.3; measured 2.0.

- **B1. The kernel's residual** (needs a read to price). v11's list: line utilisation
  on half-row slices, the 64-chunk wall, the 4-position barrier cadence. A read of
  `attention_decode_partial_shared` and its dispatch geometry against the M1's 8
  cores, then a microbench if the read is inconclusive. The prize: the slope from 2.0
  toward 1.0 per 1,000, about 7 ms per token at 7k and 16 at 16k.
- **B2. Fewer bytes per position** (numerics). Production already stores the KV cache
  at 8 bits; the engine's 4-bit mode is the remaining step and halves the roof under
  B1. Davor's call; a fresh golden. Note that the 8-bit rows mean the scan already
  dequantises per element, which v11's microbench exonerated as a cost.
- **B3. The context-independent part of the attention layers** (idea, bitwise-safe if
  done the v10 way). 5.4 ms at 300 context is 540 µs per layer: about 60 µs of scan,
  240 µs of projections at the roof, and about 240 µs of dispatch walls, so roughly
  2.4 ms per token of walls across the ten layers, the same class as the GDN chain
  (D). Candidates: fold the combine pass into the o-projection's prologue, the RoPE
  and the KV append into the projection's epilogue. v10's half round-trip lesson
  applies (fused kernels need the volatile slot or the bitwise arm catches it).
- **B4. The two-pass structure** (idea). Partial plus combine is 20 dispatches per
  token. At short context a single pass may pay fewer walls; at long context the split
  is what gives parallelism. Price after B1.
- **B5. The calibration probe: what the hardware allows** (one model run, no Shrike
  code; Claude, 2026-09-08). Before anyone rewrites the scan, measure the best-known
  Metal decode-attention kernels (MLX's vector SDPA, llama.cpp's flash-attention
  decode) on the mini with a GQA model already on one of the boxes, and read their
  slope in the same unit: µs per context position per layer, scaled to a 1 KB KV row
  (8-bit; the reference kernels' 16-bit rows count double). Ours is 0.20. If theirs
  is near the 0.017 roof, the 12× is our kernel and the
  rewrite has a demonstrated target on this exact silicon; if theirs is near ours,
  the hardware is the wall and B1 and B6 close. Observation before explanation.
  **Measured 2026-09-09 (S0.8, the mini, Shrike stopped, Qwen3-14B at 4 bits through
  the box's own oMLX 0.5.3, MLX's decode attention; fp16 KV, TurboQuant off, so 8 KV
  heads × 128 × K and V = 4 KB per position per layer, 40 layers; decode ms per
  token = wall at 136 tokens minus wall at 8 on the same cached prompt, over 128; two
  repeats agree to 0.03 s):**

  | context tokens | decode ms per token |
  | ---: | ---: |
  | 486 | 127.1 |
  | 6,592 | 147.0 |

  (8k and 12k prompts died in oMLX's memory guard; two points suffice.) The slope is
  3.26 ms per 1,000 context tokens per token for 40 layers at 4 KB a row: **81 ns per
  position per layer, 20 ns per KB, against the 16 ns per KB roof at 62.5 GB/s.** The
  reference kernel runs the scan at about 80 % of the byte roof on this chip. Ours
  runs at 200 ns per position per layer on 1 KB rows, 200 ns per KB: ten times the
  reference and twelve times the roof. **Verdict: the gap is our kernel, not the M1.**
  The prize at the reference's rate, modelled from the measured slopes: our 2.0 to
  2.3 ms per 1,000 would fall toward 0.2 to 0.3, worth about 14 ms per token at 7k
  context and about 30 at 16k, the largest single number on the board for the
  contexts a coding session runs at. It is class 2 (B6, J2) and the ruling puts class
  2 last; this measurement is the case for reconsidering that order for the scan
  alone, and the decision is Davor's (section 5).
- **B6. The scan on the matrix unit** (idea; rounding-level numerics, v11's class).
  Eight query heads share each KV head. Today each head's dot with a K row is a
  32-lane reduction and the softmax bookkeeping runs per head per position: a huddle
  after every element. Cast the eight heads as the rows of an 8 × 256 matrix and the
  K rows of an 8-position tile as its partner, and the scores for 8 heads × 8
  positions are one `simdgroup_matrix` multiply, K and V read once per tile, the
  reduction once per tile instead of once per position. This is how the reference
  kernels in B5 treat GQA. The output changes at the level of fp32 summation order,
  the class v11 accepted with a fresh golden (`a264b22`); not an approximation, but
  not byte-identical either. Davor's call whether that class is open in v18.
- **B7. The ablation microbench** (a day, throwaway). If B5 says the hardware allows
  it, find what binds the current kernel before touching it: the same kernel with the
  softmax removed, with V removed, and as a pure load at the same layout. The one
  that collapses the slope names the constraint; v11's residual list is the
  hypothesis set, and its M4 numbers do not transfer to the M1.
- **A note that closes a class of ideas.** Unified memory is one bus. The CPU, the
  ANE and the GPU draw from the same 68 GB/s, and the GPU already streams at 62.5. A
  second engine adds compute, never bytes; the ANE prefill wins on compute, and
  nothing bandwidth-bound moves to the CPU for speed.

### C. The routed-expert fixup machinery (6.9 ms, 11 %)

Re-derived 2026-09-09 from the command structure above. The speculative command's
bytes are the shared expert on 40 layers (71 MB) plus the eight experts on the 24.3
all-hit layers (344 MB): 6.6 ms at the roof against 10.9 measured, so about 4 ms of
walls and under-roof kernels across its 120 or so dispatches. The hit command and the
two fixups on the 15.7 miss layers move about 220 MB (phase 1 for 105 resident
experts, phase 1 for 20 missing ones, phase 2 for all eight on each such layer): 3.6
ms at the roof against 5.1 measured. And the GPU idles 1.8 ms per token between the
speculative command's end and the hit command's start, the host's round trip on a
miss layer. Together about 7 ms of the token above the roof. This surface has never
had a chapter of its own.

- **C1. Experts per small dispatch** (priceable offline). If the hit pass runs one or
  two experts per layer, each dispatch is wall-bound: 30 µs of bytes for 150 µs of
  wall. The route traces plus the ring's residency give the count.
  **Priced 2026-09-09 (C1, arithmetic on the T4 counters, measured inputs):** the hit
  command runs on 15.7 layers per token, 13.6 of them with 1.47 misses on average
  (6.5 hits) and 2.1 adopted-only layers with all eight, so 6.7 experts per hit
  dispatch, 7.9 MB of phase-1 bytes, 127 µs at the roof against 145 measured: the
  hit dispatch is at the roof, not wall-bound. The fixup dispatch moves 1.47
  experts' phase 1 (1.7 MB) plus the eight-expert reduce (4.7 MB), about 103 µs at
  the roof against 180 measured, 1.75×: the small, guarded kernels pay. C3's premise
  (a wall-bound hit dispatch) was wrong; C5 and C6 stand on the round trip, not on
  the dispatch.
- **C2. The submit gap, 1.8 ms** (needs a read). 117 µs of GPU idle per hit layer
  between the spec kernel's end and the hit kernel's start: a host round trip to
  classify and encode. Unexplored: encoding the hit dispatch unconditionally as an
  indirect dispatch off the GPU-side classifier (v16 put the classifier on the GPU),
  so an empty hit layer costs a 10 µs wall and a full one no round trip.
- **C3. Fold the hit pass into the fixup pass on miss layers** (idea, small). On the
  13.6 layers that read, the hit kernel runs before the window opens; folding it into
  the fixup dispatch saves a wall only on the 2.1 layers with hits and no miss.
- **C4. Closed (2026-09-09).** An earlier draft asked whether the speculative pass
  computes a ninth expert. It computes the classifier's hits among the router's
  top-8, never more; the ring's nine cells per layer are the prefetch's, not the
  pass's.
- **C5. The miss layer's two host-built commands** (idea). On a miss layer the hits
  wait for the host's plan and a separate command; the speculative command already
  knows the hits (the classifier wrote their positions and cells) and only lacks
  permission to run them. If the classifier sized phase 1 to the hits on every layer
  and zeroed only phase 2, the hit command and its 1.8 ms submit gap would go, leaving
  the fixup alone to wait on the read. The risk is the plan's swap: a landed
  prediction the classifier did not see becomes a hit after the plan, and the fixup
  computes it today (the adopted path); that stays as is.

  **C5 built and measured as v18 Task 1 (2026-09-09): a null on the wall.** The hit
  command and its 1.8 ms submit gap went to zero, the speculative command grew by the
  hits' phase 1, the window grew by the same amount, the wall flat within the drift on
  all three shapes, golden identical on both boxes. The mechanism: the reads are
  submitted to the storage threads before the hit command was built, so the hit
  command and the GPU's wait for it were inside the read's shadow, never on the path.
  Kept as a simplification if Davor rules so; the record is in the design doc.
- **C6. Hide the conversation instead of shortening it (Davor's question, 2026-09-09).**
  The CPU cannot learn the route on its own: the route is a function of this layer's
  post-attention hidden state, which only the GPU has, so any CPU decision built on
  it waits the same 63 µs as the word. The CPU already acts before the word on what
  it does have, the probe's prediction from the previous layer (the ring). What can
  change is whether the GPU idles while the CPU thinks. Today on a miss layer the
  speculative command holds only the shared expert, about 70 µs of work, and the
  round trip is about 130 µs from the tail's end (63 of visibility, 40 of host work,
  27 of commit), so the GPU idles for the difference: the 117 µs submit gap. With C5
  the speculative command also runs the hits' phase 1, about 120 µs more, and the
  round trip fits inside the GPU's own work: the GPU idles on the drive alone. The
  second half of the same idea, priceable after C5: encode the fixup as a speculative
  command too, an event wait plus an indirect phase 1 over the missing experts plus
  phase 2 over all eight, with the misses read into ring cells agreed in advance (miss
  i into cell i; the classifier already writes the miss positions and the resolved
  cells, and v16's index swap moves no bytes when a cell becomes a slot). Then the
  CPU's whole critical-path job on a miss layer is to see the word and call `pread`
  for each miss, about 5 µs; the plan (victims, the swap into the pool) runs after
  the reads are issued, off the path, at the next wake. Numerics-free by
  construction. **Re-priced after Task 1 (2026-09-09):** C5's 1.8 ms was in the read's
  shadow and is gone from the sum; what C6 can buy is the latency that precedes the
  read's issue, the word's 63 µs plus the plan, pin and submit's 25 µs, per miss
  layer, about 1.2 ms per token, and only if the encoded fixup lets the host issue
  the reads the moment it sees the word. **T2.0 (2026-09-09), priced from Task 1's
  arms:** the 63 is not C6's, since the miss list arrives with the word and nothing
  the host does issues a read before it; C6's slice is the plan, pin and submit alone,
  at most 0.34 ms per token (all 40 layers' sum charged to the 13.5 miss layers) and
  about 0.2 with the all-hit layers' share removed, under the mini's drift. The 63
  went to A8, the reader's hand-off to A5, the join's order to C7; the shadow ledger
  (section 2) records the slice as having shrinkers and no exposer. **Davor's ruling
  (2026-09-09): skipped as a performance task**; the agreed-cell mechanism is the
  endpoint's structure and moves to the fold's design note (the design document's
  Task 2 record). What remains on a miss layer is the drive's transfer (A's floor),
  the word (A8) and the 155 µs wake (A5).
- **C7. Issue before the join (T2.0, 2026-09-09).** The ring's join runs before the
  plan: when a prediction for this layer is still in flight, the host waits up to
  400 µs for it to land (`readyCells`, `prefetchJoinNanos`) before it plans, pins and
  submits the layer's other misses. On a layer with both a joined prediction and an
  unpredicted miss, the unpredicted read's issue is delayed by the join's wait and
  the fixup waits for that read. The join itself is a measured win (v15: +3.9 % on
  the 300, late to zero), so the reorder keeps it and moves it after the issue: plan
  and submit the misses the ring does not hold, then join, then plan the adoption.
  Unpriced: `prefetch_joined` is 1.7 per token on the 300 and the wait is uncounted,
  so the bound is 1.7 × 400 µs, 0.7 ms per token, and the truth is the share of
  joined layers that also carry an unpredicted miss times the mean wait; a
  diagnostic counter (the join's wait on layers with another miss) prices it in one
  lifetime. Class 1: the same experts compute in the same order, only when the reads
  go out changes; no agreed-cell contract needed.

### D. The GDN chain (15.7 ms, 26 %)

Thirty layers at 524 µs each. The role is the GDN block (the in-projection at 14.2 MB,
the conv, the delta-rule state at 4.2 MB read and written, the out-projection at 4.7
MB) plus the layer's tail (the residual, the norm, the router and the probe at about
0.5 MB each, the top-k select, the classifier). Bytes: 693 MB for the blocks and about
30 MB for the tails, 11.6 ms at the roof against 15.7 measured. v10's honest floor for
the block alone with per-dispatch walls was 409 µs per layer (remembered,
[v10-decode-kernel-mergers.md](v10-decode-kernel-mergers.md)). The 4 ms above the
roof is consistent with about 13 dispatches per layer at the 10 µs wall v10 measured;
the tail alone is five or six of them. v10 landed mergers where the bits allowed.

- **D1. Dispatches per GDN layer now** (needs a read). Count them on the current tree
  and price each at the 10 to 15 µs wall v10 measured. The recoverable pool is
  whatever exceeds one wall per unavoidable dependency, on the order of 2 to 3 ms per
  token.
  **Priced 2026-09-09 (D1, a read of the encoders at `c503f7c` by subagent, every
  count cited to its dispatch site; measured in the sense of counted).** Dispatches
  per decode layer, one encoder unless noted: a GDN layer's attention command 11
  (input norm; the fused in-projection; conv; qk norm; the delta step; the gated
  norm; o_proj; then the tail's residual-plus-norm, the paired router GEMV, the
  paired top-k select, the classifier); a full-attention layer's 14 (norm; the fused
  qkv GEMV; the q/gate split; the qk-norm-plus-RoPE epilogue; two KV quantise
  kernels at 8-bit; the shared partial; the combine; the sigmoid gate; o_proj; the
  same four-kernel tail); the speculative command 7 on four encoders (the shared
  expert's gate GEMV, up GEMV, scalar gate, and the fused silu-mul-down-gate; then
  phase 1, phase 2 and the residual, each indirect on its own encoder, all three
  launching empty on a miss layer); the hit command 1; the fixup 3 on three encoders
  behind the event wait; the head 2 (norm, the head GEMV) plus the sampler's 6 on
  the production path (three softmax stages, three top-k stages), or 4 in all on the
  fused greedy path the server never takes. **Per token: 30 × 11 + 10 × 14 + 40 × 7
  + 15.7 × 1 + 15.7 × 3 + embed + head + sampler, about 820 dispatches**, no
  blits. The GDN row's 4 ms above its roof over 330 dispatches is 12 µs each and the
  speculative row's 4 ms over 280 is 14 µs each, both inside v10's 10 to 15 µs wall,
  so the launch-wall reading holds for both rows. One caveat the count cannot
  settle: the speculative command spends 4 encoders on 7 dispatches and the fixup 3
  on 3, against 1 encoder for the attention command's 11 or 14; if the wall is
  partly per encoder, those two commands are the expensive ones per dispatch.

  Six merges the read judged to keep every arithmetic operation and its order
  (class 1 if that holds; each is checked against the bitwise golden when built):
  conv plus qk norm (the conv's 256-wide threadgroup already owns whole head
  slices; saves 30 per token); the top-k select plus the classifier (one-lane work
  on the buffers the other just wrote; saves 40); the input norm into the
  in-projection (the in-projection already stages `x`; redundant work per
  threadgroup, not different arithmetic; the weakest; saves 30); the shared gate
  and up GEMVs as one grid over both row sets (the trick the fused qkv and
  in-projection GEMVs already use; the cleanest; saves 40); the shared scalar gate
  into that same dispatch (saves 40); spec phase 2 plus its residual (one
  threadgroup per output index in both, the same indirect grid, the zero-grid miss
  behaviour preserved; saves 40). 220 of 820, about 27 %, worth roughly 2.5 to 3 ms
  per token at the measured wall if every merge lands. Not free, for the record: the
  router GEMV into the select (cross-threadgroup), the delta step into the gated norm
  (32 threadgroups per head against a whole-head reduce), spec phase 1 into phase 2
  (phase 2 reads across all of the intermediate). These are v18's Task 3 neighbours
  or a small chapter of their own; the doc records them here. **Scheduled as v18
  Task 6 (Davor's ruling, 2026-09-09), after Task 3 and before the fold's ruling:**
  Task 3 measured the boundary between two dependent kernels at about 25 µs inside
  one command, twice the per-dispatch wall counted here, which makes the caveat
  above the live question; T6.0 prices the encoder boundary against the dispatch
  boundary by putting the speculative command's seven dispatches and the fixup's
  three on one encoder each, then the six merges follow in the order of prize and
  risk, each with its bitwise arm. With the slack rule applied (a merge inside the
  speculative command counts on the 26.5 all-hit layers only), about 2.2 ms per
  token modelled at 12 µs a wall. **T6.0 landed (2026-09-09):** the speculative
  command's four encoders to one and the fixup's three to one, no kernel change,
  took 3.3 ms per token off the GPU's role time, about 22 µs an encoder boundary,
  and +1.6 to +2.8 % off the wall on the 300 and the 1k; so the caveat was the
  larger term, an encoder boundary costs about twice a dispatch boundary, and the
  merges' pricing at 12 stands. The boundary command's nine encoders are the next
  encoder-only step (T6.0b, about 0.18 ms).
- **D2. The in-projection at the roof?** (needs a finer instrument). The role covers
  the chain; a per-kernel split needs a Metal capture or a kernel-level stats mode.
  14.2 MB per layer is 227 µs at the roof; if the GEMV runs slower than that, the
  chain has a second pool.
- **D3. The delta state** (numerics). 4.2 MB read and written per layer, 2 ms per
  token at the roof, inevitable for GDN at 16-bit. A narrower state changes the
  output. Recorded, not proposed.

**The round trip's anatomy (2026-09-09; every number measured).** A flag lookup on the
GPU is free; a conversation with the CPU is about 100 µs regardless of payload, and it
is the mechanism under surfaces C and E. The halves: a GPU write becomes visible to a
spinning host 63 µs after the command's GPU end (v14; the driver's completion mark
takes 160); the host's own work on a miss layer is about 40 µs (the plan under the
cache lock, the pin and submit, the argument buffer, the hit encode; the `path_*`
fields); a committed command starts on the GPU 25 to 27 µs later (the driver's
commit-to-kernel latency, hit and fixup alike). That is the 117 µs submit gap per miss
layer, 15.7 times a token. The same physics is the 155 µs wake dead time after a read
lands (the reader thread signals the shared event, the GPU's blocked command resumes)
and the 0.9 ms token boundary (the sampled token crosses to the CPU, which encodes
the next pass). The miss layers converse with the CPU because only the CPU can issue
a read; today each such layer converses twice (the hit command, then the fixup), and
C5 is the observation that the first conversation carries nothing the GPU did not
already know. The small-dispatch tax inside the speculative and GDN rows is a
different animal: GPU-internal launch gaps of about 10 µs with no CPU involved, cured
by fewer and larger kernels, not by fewer conversations. **Correction after Task 1
(2026-09-09):** the second conversation on a miss layer, the hit command, ran while
the reads were already in flight, so removing it moved nothing; only the part of the
round trip that precedes the reads' issue is on the path. A ledger row of GPU idle is
serial only if nothing else the token waits for is in flight beneath it.

### E. The boundary taxes (3.6 ms, 6 %)

- **E1. Chain transitions, 2.6 ms** (one arm measured null). Eighty per token at about
  33 µs: attention command buffer to MoE command buffer and back each layer. v10
  measured command-buffer-count and single-seam fusions as zero wall (remembered), so
  these may be the GPU's own drain-and-launch cost. Unexplored: whether one command
  buffer per layer with the MoE dispatches encoded indirectly (as in C2) removes the
  seam rather than merging across it. **Landed as v18 Task 3 (2026-09-09), one
  command per layer:** the forty tail-to-speculative command boundaries are gone and
  the merged commands grew by nearly the same amount, the drain now an encoder
  boundary inside the command; the net is about 10 µs a layer, the wall +0.4 to
  +0.8 % on the clean lifetimes, about 0.3 to 0.5 ms per token, a third of the model.
  v10 was mostly right: the seam is the GPU's own drain, and a command boundary
  costs only about 10 µs more than an encoder boundary. The forty layer-to-layer
  boundaries that remain (1.1 ms) are the fold's, worth about 0.4 by the same
  measurement. The record in the design document's Task 3 section.
- **E2. The token boundary, 1.0 ms** (idea, numerics unchanged). Three gaps of 0.3 ms:
  head to sample, sample to embed, embed to layer 0. The sampler already runs on the
  GPU; the host reads the token back and encodes the next pass. Unexplored: the
  sampler writes the token id to a buffer the next pass's embed reads, the host
  encodes the next pass before the sample finishes and reads the token back
  asynchronously for streaming. The stop check runs one pass late (one wasted pass
  per answer). Worth about 1.5 % per token. **Landed as v18 Task 4 (2026-09-09),
  with one change to the idea:** the stop check runs on the token's word (about 63
  µs after the sample kernel) and layer 0 is committed only when there is none, so
  nothing runs a pass late and no state needs undoing (a pass mutates the GDN state
  in place). The three gaps became one of 0.25 to 0.27 ms; +1.4 % on the same-box
  A/B, about 0.8 ms per token; golden identical on both boxes; the record in the
  design document's Task 4 section. What remains of the boundary, the one gap and
  the sampler's encode at the head's front, is the fold's.

### F. The LM head (4.6 ms, 7.5 %)

At the roof: 286 MB of 4-bit weights over 248k vocabulary rows. No kernel lever.
Fewer bytes means a narrower head, which changes the output. Nothing to overlap it
with: it needs layer 40's output and the next token needs its result. Closed unless a
numerics lever is ever on the table.

### G. Outside the decode ledger: prefill and time to first token

The warm 300-token turn is 3.17 s with 2.82 s of prefill. The prompt cache keeps the
previous turns resident, so a session's prefill is the delta only, and on a coding
session the delta is the tool's output, about a thousand tokens per turn.

**The turn, not the token (Claude, 2026-09-09; measured from the mini's live log,
2026-09-08, Davor's Pi session).** Five consecutive tool turns, each prefilling the
previous tool's output and answering with the next call:

| new prompt tokens | answer tokens | prefill s | decode s | prefill's share of the turn | GPU busy inside the prefill |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,133 | 140 | 15.7 | 9.7 | 62 % | 6.0 s |
| 1,242 | 133 | 6.7 | 8.9 | 43 % | 6.6 s |
| 1,333 | 133 | 9.7 | 9.3 | 51 % | 7.5 s |
| 1,362 | 98 | 10.2 | 6.9 | 60 % | 7.9 s |
| 892 | 944 | 6.2 | 70.1 | 8 % | 5.8 s |

On the tool turns the model spends as long reading the tool's output as writing its
answer, or longer, and that time is GPU compute, not the drive: four of the five
prefills are within 0.1 to 2.3 s of their own GPU busy time (the first is an outlier
at 9.7 s exposed, unexplained, one of five). Each prefill reads about 4,500 missing
experts, 8 GB, at a prefill hit rate of 52 %, hidden under the compute. Per new token
the GPU spends 5.3 to 5.8 ms: the routed expert tiles 2.4 to 3.1 s per prefill, the
GDN chunk kernels 1.7 to 2.5 s, the attention block 0.8 to 2.0 s growing with the
context, the shared expert and the reduce 0.2 to 0.3 s. The compute floor at the M1's
2.6 TFLOPS is about 2.8 s for 1,200 tokens (3.3 B active parameters, two FLOPs each);
the measured 6 to 8 s is 2.5× off it, and the routed tile alone runs at about 31 % of
peak (modelled from the parameter count).

Levers, sized at these shapes:

- **G1. The ANE prefill switch** ([ane-prefill.md](ane-prefill.md)) moves the
  full-attention blocks to the Neural Engine, 26.7× faster on the block. Its 2.31× was
  a 6k prompt with the cache off, where attention was 63 % of the prefill and
  quadratic. With the cache on, the new chunk attends over the cached context and the
  attention block is 0.8 to 2.0 s of 6.7 to 10.2 here, 12 to 25 %, so the switch is
  worth 1 to 2 s per tool turn at these contexts, more as the session grows. The
  record notes the two arms' outputs differ (the ANE runs fp16), a numerics question
  for Davor before it is on.
- **G2. The routed tile and the GDN chunk kernels' efficiency** (needs a read). v12
  built the matrix kernels and took 12k from 725 to 68 s; the tile is at about a third
  of peak, the GDN chunk kernel unmeasured against a floor. The M1 has simdgroup
  matrix units and no more; the headroom is real but its size is a microbench away.
- **The outlier, read 2026-09-09 (S0.7, measured from the log):** the 20:07:21
  request (1,733 prompt, 600 cached, 140 answer tokens) took 25.5 s, and the GPU's
  kernel span for it was 9.9 s against 15.8 and 15.6 s for its two neighbours, whose
  spans match their walls. So about 15 s of that request ran with no kernel on the
  GPU at all, most of it inside the 15.7 s the server booked as prefill; the
  runner's I/O counters for it (fetch time, reads, evictions) sit within 10 to 20 %
  of the neighbours', so the drive was not slow. What differs upstream: the
  previous turn ended in a tool call and, unlike the three turns before it, wrote no
  settle lines; the outlier is the first request after it and the first with a
  live-tier cache hit. The cause is not in the log. Not chased: one of five, and the
  reproduction (a short-context tool turn followed by a 1.1k-token delta) belongs to
  the turn rig if it recurs.
- **G3. Nothing to cut in what is prefilled.** The tool output is what the model must
  read; the cache already spares the history. The prefill's 8 GB of reads are hidden
  today, so H2's compression buys nothing here until the compute falls under the
  drive's 2.3 s.

Reframing: on a tool turn the decode ledger's whole remaining prize on a 133-token
answer is about 2 s (61 to 47 ms per token at the byte roofs), while the prefill of
the same turn has 3 to 5 s between its measured time and its compute floor. The turn's
wall is the metric that matches the box's use, and for it prefill is the larger half.

### H. Converting compute into bandwidth (Davor's direction, 2026-09-09)

The GPU's arithmetic is nearly idle: the token's useful work is a few GFLOP against a
2.6 TFLOPS part, and every large row is bytes. On unified memory compute cannot make
bytes (one bus, section B's note). It can do exactly two things: make each byte serve
more tokens, or make each byte carry more information. Both exist as engineering.

- **H1. Each byte serves more tokens: speculative verification, revisited.** A
  width-k pass reads the dense rows once for k+1 candidate tokens: the GDN chain, the
  attention weights and the head, 26 ms of the 61 today, are per pass. So is the miss
  window: all k tokens' misses at a layer are known together and read in one
  overlapped window, so the 13 ms is paid per pass, not per token. What scales with k
  is the routed experts' bytes, the union of the k tokens' top-8 per layer (A9's Q1
  overlap statistic prices that union too). MTP lost by 4× on the mini (v12 P17,
  2026-09-03, measured) for reasons the record separates: the verify ran the prefill
  path at width 2 (GPU 124 ms per pass against a plain step's 43, the GDN prefill
  kernels 57.9 of it) with 46 ms of host gaps and 26 of turnaround per pass, and the
  MTP head accepted 20.6 / 31.7 / 75.6 % on counting / prose / tool-call
  continuation. P17's own exit priced a decode-width verify without round trips at
  about 1.3× plain on tool-heavy shapes at k = 2, unmeasured. Two things changed
  since: the unmeasured draft is prompt lookup (an n-gram match in the context, zero
  cost, high acceptance on the copied spans that code editing is made of), and the
  ledger now prices the pass. Modelled with a decode-class width-k path: at k = 2 and
  76 % acceptance about 1.4×; at 30 % about 1.05×; at k = 4 on a copied span up to
  2 to 3×. Two costs the record names that are not economics: P17 found greedy
  speculation not lossless on two of three shapes (the width-2 kernels rounded
  differently from the width-1 ones, so row 0 diverged), so a width-k kernel must
  produce row 0 bit-identical to the plain GEMV or the output changes; and the
  width-k decode path is a build (the k-row GEMVs, the expert-union dispatch, the KV
  append and the GDN checkpoint with rollback, which the fold took out of the kernels
  and git history keeps). Workload-specific: the prize lives on code and tool turns.
- **H2. Each byte carries more information: lossless compression with in-kernel
  decode.** The 4-bit affine format spends 4.5 bits per weight: 256 bits of indices
  and a bf16 scale and bias per group of 64. The indices are not uniform over 16
  levels; their entropy is on the order of 3.5 to 3.7 bits (remembered from the
  quantisation literature, not measured on ornith15), and the scale and bias pairs
  are smooth. An entropy code over both, decoded inside every GEMV by lanes that are
  otherwise idle, cuts every row's bytes and every SSD read by a modelled 10 to 15 %,
  with no change to any value the kernel computes. Bounded prize about 5 ms per token
  if the decoder is free, half that if it is not. Engineering: a repack, a decoder in
  every weight-reading kernel, per-group offsets so lanes can start in parallel. The
  entropy on this model is measurable offline from the `.gturbo` in an afternoon.
  **Priced 2026-09-09 (S0.6, measured on ornith15's gturbo by subagent; script
  `s06-expert-entropy.py` in the session's scratchpad; layout confirmed at
  `RepackPlanner.swift:735-795` and `packed_experts/layout.json`: per expert gate,
  up, down, each as 4-bit indices low nibble first then bf16 scales then bf16
  biases, groups of 64 along the row).** Twenty-four experts each from layers 0 and
  20, and 64 rows of the head. Order-0 entropy of the indices: 3.43 bits per weight
  at layer 0, 3.71 at layer 20, 3.69 on the head; the scales and biases 6.8 to 8.3
  bits each of their 16 (they use 700 to 960 of 65,536 patterns and still cost
  that); delta-coding the scales loses everywhere. All in, against the format's
  4.5 bits per weight: a nibble code 3.64 / 3.97 / 3.95 bits (layer 0 / layer 20 /
  the head), 12 to 19 % off; a byte-symbol code 3.30 at layer 0 (its zeros pair
  within a byte) and nothing more at layer 20. The typical layer and the head sit at
  **11.8 to 12.3 % fewer bytes**, the shallow layers more. The scales and biases are
  11 % of the stride today and would be 21 % of a coded expert, so they, not the
  indices, are what a second-order code would go after. Modelled prize at the
  roof: 12 % of 1,800 MB is 3.5 ms per token of streaming and 1.2 ms off the miss
  window's transfer floor, before the decoder's own cost. v20's basis.
- **H3. The draft lives in the host's spin-wait (Claude, 2026-09-09).** A9 and H1
  both want a draft token and both are priced against the draft's cost. MTP's head
  cost 17.5 ms per pass on the GPU and on the critical path. There is a place where a
  draft is free: the host thread spins for the 13 ms of miss windows per token, in
  13.6 fragments of about 1 ms, on P-cores that are otherwise idle, and the memory bus
  is idle then too since the GPU is waiting and the drive's DMA is 35 MB per token.
  A small dense draft (a 0.3B to 0.5B model at 4 bits is 200 to 300 MB of weights,
  about 6 to 9 ms of CPU time per token at the rates llama.cpp reaches on M1 cores,
  remembered not measured) runs layer by layer between the waits, its traffic landing
  in the GPU's idle stretches, and produces token t+2 during pass t+1 at zero wall
  cost. Prompt lookup (H1) is the cheaper draft on copied spans and needs no model at
  all; the CPU draft is the fallback where the context has no match. What the free
  draft buys: for A9, pass t+2's experts predicted a whole pass ahead for all forty
  layers, including the shallow ones the token-id table cannot lead; for H1, the
  proposal cost gone. Risks named: the spin wait is the host's fastest path to the
  wake, so the draft's slices must yield at the read's completion (a stall there adds
  to A5's dead time); the draft's bus traffic during a GPU-busy stretch would slow the
  GPU's GEMVs, so it must run only inside windows; and the second GPU queue variant
  of this idea is not recommended (v15's dedicated queue paid 162 µs per miss layer,
  and the AGX driver's concurrent-encoder trap).
- **What does not convert.** Recompute instead of read: the KV rows are already
  smaller than the hidden states they come from, and the GDN state is a sequence
  accumulator. Skipping small activations in the expert down-projection: lossy.
  Batching a second request: one user, one stream, unless the client runs parallel
  subagents. The reader's own traffic: 35 MB per token over the shared bus, 0.6 ms;
  negligible.

### I. Outside the transformer: the sampler, the host loop, the driver (Davor's
question, 2026-09-09)

**The sampler.** After the head writes 248,320 fp16 logits, the server runs
`logit_softcap_softmax` (tiled) over the vocabulary into a probability buffer, then
one of two kernels in `logit.metal`: for temperature above zero with top-k in 1 to
64, `sample_topk64` (a top-64-per-tile stage, then top-p over the survivors, then
the temperature draw); otherwise the generic `sample` kernel, which at temperature
zero is an argmax. The random number is drawn on the GPU by one thread: xorshift64*
(a 64-bit state, the top 24 bits as a [0,1) float), seeded per token by the host
(`Sampler.seedFor`): splitmix64 of the client's seed plus the position when a seed is
given, otherwise the monotonic clock's nanoseconds mixed with a counter. The draw
itself costs nanoseconds; the softmax and the selection over 248k entries are the
work. Measured on the arms (greedy): the `sample` role 0.16 ms of GPU per token, and
the host's `loop_sample_ms` 0.44 ms of wall, which is the whole step: encode, commit,
the GPU's softmax and selection, and the wait for the token to come back. The
sampled path (the server's defaults, temperature 0.6, top-k 20, top-p 0.95, when a
client sends nothing) is unmeasured on the arms; same order of cost. What Pi sends is
not logged (the request's sampling fields do not reach the log).

**The host loop, per token on the 300 (measured, the runner's loop phases):**

| phase | ms | what it is |
| --- | ---: | --- |
| produce | 60.38 | the forward pass, everything in sections 2 and 4 |
| sample | 0.44 | the sampler step above, including its GPU time and the readback |
| progress | 0.04 | the streaming callback hand-off (the socket write is asynchronous on the event loop) |
| detokenise | 0.005 | the detokeniser and the stop-string matcher |
| **sum** | **60.87** | equals the server's wall per token to the hundredth |

So the loop outside the pass is 0.5 ms per token, 0.8 %, and 0.44 of it is the
sampler's round trip. The HTTP layer adds nothing measurable: the client's streamed
inter-arrival equals the server's rate.

**The runtime's own host work inside the pass** (the `path_*` fields, the 300, per
token): argument buffers 0.14, the hit encode 0.13, the fixup build 0.16, the pin
0.05, the submit 0.08, the cache plan 0.21, the I/O queue 0.35, the first command
buffer's encode 1.50; the driver's commit-to-kernel latency 0.39 (hit) plus 0.37
(fixup). All of it already appears in section 2 as the submit gap and the transition
gaps; it is not a hidden row. The host waits (`path_router_wake` 2.42,
`path_hit_kernel_to_gpu` 1.84) are the GPU's time seen from the host, not extra.

- **I1. The fused greedy head is never used by the server** (bitwise-safe, small).
  The runtime can fold the argmax into the head kernel and skip the softmax, the
  sample kernel and the logits round trip (`headPath == .fusedRows`, the CLI's path
  for pure greedy). The server passes `forceLogitsHead: true` unconditionally
  (`ServerInference.swift:729`, `:837`), so every server token, greedy or not, takes
  the logits path. For a temperature-zero request the fused path saves the 0.16 ms
  sample role and part of the 0.9 ms token boundary. Worth it only if Pi sends
  temperature zero; find out first.
- **I2. The decode loop's core and QoS** (verify, one run). The reader queues are
  `.userInitiated`; the decode loop runs on Swift concurrency's cooperative pool with
  no priority I could find. On an idle box it lands on P-cores anyway, but QoS also
  steers the CPU's clock ramp, and the token's host round trips (0.5 ms of sampler
  plus the spin waits) are latency-bound. A spindump or `taskpolicy` read during a
  decode says which cores and at what class; a `.userInteractive` pin is a one-line
  arm if it is not already there.
- **I3. The GPU's clock after an idle window** (hypothesis, priceable from the
  timestamps). The GPU idles 31 % of the token in slices of about a millisecond. If
  the power manager lowers the GPU clock in a gap and ramps it back on the next
  dispatch, every kernel that follows a window runs slow for its first microseconds.
  One hint, not evidence: the fixup dispatch, which always follows a window, averages
  180 µs against the hit dispatch's 145 for similar work. The per-command-buffer GPU
  timestamps the runner already records can plot a kernel's duration against the
  idle length before it; a flat line closes this, a slope opens a lever (keeping the
  GPU warm through a window costs power, not correctness).
  **Measured 2026-09-09 (S0.9, the mini, a throwaway build of `c503f7c` plus a
  per-command-buffer timestamp dump, never committed; the 300 prompt at 512 answer
  tokens, two generations, 71,500 command buffers; the dump and the analyser at
  `~/.claude/handoffs/archive/shrike-v18-step0/s09-*`). Closed.** Kernels with fixed
  work show no clock effect after an idle window: the GDN attention command averages
  0.52 ms after under 20 µs of idle (n≈8,800) and 0.51 to 0.53 after 0.4 to 3 ms
  (n≈600), the full-attention command 0.54 against 0.53 to 0.56, the head 4.53 to
  4.60 in every bin, the sampler 0.16 in every bin. The fixup command alone grows
  with the idle before it, 0.168 ms under 0.8 ms of idle to 0.208 at 1.5 to 3 ms and
  0.28 past 3 ms, and that is its own work, not the clock: a layer with more misses
  waits longer for its reads and then computes more experts. The effect on a fixed
  kernel after a long idle is within 1 to 3 %, at most 0.2 ms per token if it exists
  at all. No lever.

### J. Touching the math (Davor's question, 2026-09-09)

**What is built in and what is ours.** Sixty-five hand-written Metal kernels; no Metal
Performance Shaders, no BLAS. One family uses a built-in primitive: the GDN chunked
prefill (`gdn_chunked.metal`) computes its chunk products with Metal 4's
`mpp::tensor_ops::matmul2d`. Everything else, every decode GEMV, the expert phases,
the attention scan, the routed prefill tile (`prefill_dequant_affine_qmm_f16_block`, a
threadgroup fp16 tile of scalar FMAs) and the sampler, is our arithmetic, instruction
by instruction. No kernel uses the `simdgroup_matrix` intrinsics.

**Three classes of "same result".** The bar decides which are open.

1. **Bit-exact.** The rewrite changes no rounding anywhere: `x / 2` to `x * 0.5` is
   one (both exact in IEEE), as is any change to control flow, layout, dispatch shape
   or memory traffic that leaves each arithmetic operation and its order intact. Every
   chapter since v13 held this bar; the golden is its instrument.
2. **Exact in real arithmetic, different rounding.** Reassociation, a factored scale,
   a fused multiply-add where there were two operations, a different summation tree,
   a chunked form of a recurrence. The output text is the same in practice and the
   golden digest is not. v11 accepted this class once (`a264b22`) with a numeric arm
   and a fresh golden.
3. **Approximation.** Fewer bits, dropped terms, thresholds. Closed for v18 (Davor,
   2026-09-09).

**Where the math has already been touched.** The expert GEMVs factor the affine
dequant per group of 64 as `s · Σ(q·x) + b · Σx` (`moe.metal`, two FMAs per group
instead of a multiply per weight), the class-2 form of Davor's example, in place
since the format landed. The GDN prefill's serial recurrence was rewritten over
64-row chunks as matrix products (v12 P4), the same math in a different order. The
attention scans use the online softmax and merge chunk partials, again class 2 and
already accepted.

**Where touching the math can still pay, and where it cannot.** In a bandwidth-bound
kernel the arithmetic is free: a 1.77 MB expert costs its bytes however the multiply
is written, so no rewrite of a decode GEMV moves the ledger. The math levers live
where compute or structure binds:

- **J1. The routed prefill tile on the matrix units** (class 2). The tile runs
  scalar FMAs at about a third of the M1's peak (G2). Hand-written
  `simdgroup_matrix` fragments were v12's named fallback ("more code, same math")
  and were never built. The dequantised 4-bit weights become 8×8 half tiles; the
  reduction order changes.
- **J2. The attention scan's arithmetic** (class 2; B6 is the structural half).
  Lazy rescaling in the online softmax (rescale the accumulator only when the
  running max moves by more than a threshold, exact in real arithmetic), `exp2` with
  the scale folded into the query, and the 8-heads-as-a-matrix tile. All change
  rounding, none change the math.
- **J3. An exact argmax over fewer rows** (class 1 for greedy only). The head reads
  all 248k rows because the maximum could be anywhere. With a stored norm per row
  (1 MB), rows whose `‖w‖·‖h‖` cannot beat the best logit found so far are provably
  below it and need not be read; compute a candidate set first (the context's own
  tokens, the frequent ones), then the survivors. Bit-identical for the rows that are
  computed, so the greedy token is the same. Useless for the sampled path, whose
  top-p needs the full distribution. The pruning ratio is unknown and depends on how
  tight the norm bound is on real hidden states; measurable in an hour with one dump
  of hidden states. Worth up to the head's 4.6 ms on greedy requests only.
- **What does not pay.** Any rewrite inside a decode GEMV (bytes are bytes); integer
  dot products (the M1 has no int8 matrix path, and quantising activations is class
  3); folding the router weights into the expert's down projection (no bytes saved).

### K. The endpoint: one command per token (Claude, 2026-09-09; class 1 throughout)

v9's speculative command removed the CPU from the all-hit layers. C5 and C6 remove it
from the miss layers' compute, E2 from the token boundary, and E1 asks whether the
chain transitions are a command-buffer cost. Taken together they have one endpoint:
**the whole token as one command buffer**, encoded and committed once, forty layers
of attention, tail, classifier, shared expert and experts with every routed grid
sized by the classifier's indirect arguments, every miss layer's fixup already
encoded behind an event wait on reads into pre-agreed cells, the head and the sampler
at the end writing the token id into the buffer the next token's embed reads, and the
next token's command committed before this one finishes so two are always in flight.
The CPU's job on the path shrinks to one thing: see a miss layer's word and issue
its reads. Everything else it does today (the plan, the swap, the encodes, the
commits, the token readback for streaming and the stop check) moves off the path or
disappears. The GPU waits on the drive and on nothing else.

What it is worth, per token on the 300, modelled from the ledger's gap rows and
re-priced after Task 1 (the submit gap was in the read's shadow, 0 not 1.8): the
pre-issue latency on miss layers about 1.2 (C6), the token boundary 1.0 to about 0.2
(E2), and the chain transitions 2.6 to perhaps 0.8 if an encoder boundary inside one
command costs the 10 µs a dispatch does rather than the 33 a command boundary does.
About 3.5 ms, 6 %, with the last term the uncertain one: v10 measured command-buffer merges as zero wall when the host round
trip still dominated the token ("GPU-side savings are worthless at the margin; only
removing the round trip cashes them"), and the round trips are mostly gone now, so
that null is due for a re-measure rather than a citation. Nothing in it changes a
number the kernels compute.

**Re-examined after T2.0 (2026-09-09).** The pre-issue term is at most 0.34 and
about 0.2, not 1.2: the word's 63 is not the host's to remove (A8). K is then E2's
0.7, the transitions' 0 to 1.8 and the slice's 0.2: 0.9 to 2.7 ms, 1.5 to 4.5 %,
and the transitions decide it. E2 and one command per layer each stand alone; the
fold buys only the per-layer boundaries that remain after the merge and C6's slice,
so it is decided on Task 3's arms, and C6's mechanism enters the fold's design note
as structure rather than as a task of its own (Davor's ruling, 2026-09-09).

**Re-examined after Tasks 4 and 3 (2026-09-09).** E2 landed at about 0.8 ms per
token and one command per layer at about 0.3 to 0.5, with the boundary's cost
measured at about 10 µs more than an encoder boundary's. What the fold would still
buy: the forty layer-to-layer boundaries at that rate (about 0.4), Task 4's one
remaining boundary gap (0.25) and C6's slice (0.2), about 0.85 ms per token, 1.4 %,
against the agreed-cell mechanism, two commands in flight and the cancel path.
Davor's ruling on Task 5, taken after Task 6 (the walls, D1) has re-measured the
boundary costs the fold is priced on.

The steps are the avenues in order, each measurable on its own: C5 (the hits in the
speculative command), C6 (the fixup as a speculative command, reads into agreed
cells, the plan after), one command per layer (attention and speculative merged,
prices E1), E2 (the sampler feeds the next embed, the stop check one pass late), and
then the fold into one command per token with two in flight. The constraints that
hold throughout: the classifier must see the host's residency writes, which v16's
probe showed a running command does once it has streamed 1 MB (the attention command
does); an event wait sits between encoders, never inside one; the residency publish
stays one release store per cell; and the stop check running one pass late wastes one
token's work per answer.

## 5. Decisions

**2026-09-09, Davor: class 3 closed, class 2 open under the variance rule.** No
approximations this chapter (the KV cache at 4 bits, a narrower GDN state, quantised
activations stay options for later). A rounding-level change (section J's class 2) is
not "output changing" when its effect is variance: "correct all along" for "right all
along". It is output changing when the response takes a different route entirely or
produces a token that does not belong, a Chinese character where none should be. In
Davor's words, that is as close as the bar can be put without days of thought.

**The class-2 gate (how the ruling is applied; Claude's operationalisation, to be
refined when the first class-2 change arrives).** A digest cannot judge variance, so
a class-2 commit passes three instruments and then re-captures the golden:

1. **The kernel arm.** The new kernel against the old on real weights and every
   shape class the runner dispatches, max |Δ| within the fp16-accumulation band.
   v11's V0 is the template (`AttentionTests`' kvShared reference arms, tolerance
   not bitwise). This catches defects: a wrong tile index or an overflow shows as a
   delta orders of magnitude above rounding, which is where the foreign character
   comes from.
2. **The forced-token logit comparison.** Decode the golden prompts with the golden's
   own tokens fed in place of the sampler's, old build and new, and compare the
   logits position by position: the KL divergence per position, and every argmax
   flip with the old build's top-2 margin at that position. Variance flips happen
   only at near-ties, so the rule is mechanical: every flip's margin must sit inside
   the measured band of |Δ logit| (three times its maximum over the run); a flip at a
   wide margin is a defect. Forcing the tokens removes the butterfly effect, so the
   comparison measures the kernel, not the trajectory. This instrument does not exist
   yet: a forced-token decode mode plus a logits dump in the runner, a diagnostic and
   itself class 1. Build it once, with the first class-2 change.
3. **The read.** The golden prompts free-run on the new build, the answers read by
   Davor for route and language. This is the one judgment only a reader makes; the
   instruments above make it rare that the read finds anything. It is not a
   formality: a real example the same day, from another model entirely, was
   `служnosti` for the Croatian *služnosti*, four Cyrillic letters inside a Latin
   word. Cyrillic and Latin spell the same sounds in that language, so the two
   tokens sit at a near-tie and the margin rule would call the flip variance, while
   the bar calls the result output changing. Both are right. A near-tie between two
   spellings is variance; a near-tie between two scripts, or between a word and
   garbage, is a visible defect born of legitimate variance, and only the read sees
   it. Should it ever appear on Shrike, the remedy is a decoding policy in the
   sampler (a script-consistency guard on the candidate set), output changing by
   design and its own decision, never a kernel change.

After acceptance the golden is re-captured on the machine that checks it, and it is
the bitwise gate again for every class-1 commit that follows. The ANE prefill's fp16
blocks fall under the same gate with a wider expected band; the record already notes
its arms differ.

**What this opens:** B6 (the scan on the matrix units), J1 (the prefill tile on the
matrix units), J2 (the scan's arithmetic), G1 (the ANE switch, behind the gate), and
any merger whose fused form rounds differently. **What stays closed:** B2 at 4 bits,
D3, activation quantisation, thresholded sparsity.

**2026-09-09, Davor: the sequencing and the chapter split delegated to Claude ("you
and the other peers will be doing the work"). The decision:** v18 is **the quiet
host**, the fixup chain to its endpoint (C5, C6, one command per layer, E2, K) with
the whole board priced once at its step zero; its pair is
[v18-quiet-host.md](v18-quiet-host.md) and
[v18-implementation-plan.md](v18-implementation-plan.md). v19 is the SSD mechanism
step zero names (A0, A9 or A1/A2); v20 is lossless compression scoped to the experts
and the head (H2); v21 is the class-2 block (B6, J1, J2, G1) with the forced-token
instrument and one golden re-capture; H1 and H3 after that. Each chapter sized like
v13 through v17 so each gets a clean measured close.

**2026-09-09, step zero's close (S0.10): what the pricing says about the order, for
Davor's ruling.** The SSD surface's mechanism, as priced: no online policy is worth
more than about a miss per token (SLRU or LRU on the longer shapes; A2), the
token-id predictor is real but drive-bound at full width and best used selectively
(layer 0, where nothing else predicts, and layers 30-39, a quarter of the demand;
A9), and the wider probe needs a capture before it can be priced (A0). A realistic
v19 on that surface is a few milliseconds of the 13. Against that, the calibration
probe (B5) turned the attention scan from a hypothesis into a demonstrated target:
the reference kernel runs the scan at 20 ns per KB on this M1 against our 200, and
at the reference's rate the row is worth about 14 ms per token at 7k context and 30
at 16k, on the contexts the box actually serves. That is larger than the whole
quiet-host chain and larger than any SSD mechanism the pricing supports, and it is
class 2. **Recommendation (Claude):** keep v18 as decided; make v19 the scan rewrite
(B6, J2) behind the variance gate with the forced-token instrument built first, so
the class-2 block's one golden re-capture happens once and early rather than once
and late; move the SSD mechanism to v20 with its step zero (the wide capture, the
predicted-future Belady variant, the selective token-id table) and compression to
v21. If the class-2-last ruling stands, v19 is the selective token-id predictor
plus the SLRU switch, and the scan waits. Davor's call.

**2026-09-09, Davor: the recommendation stands ("the answer is pretty clear").** v19
is the attention scan rewrite behind the variance gate, the forced-token instrument
first; the SSD mechanism moves to v20, compression to v21. The class-2-last rule
holds within v18, which stays class 1 throughout. Go given for v18 Task 1.

**2026-09-09, Davor: the sequence.** Tolerance is owed (the `служnosti` example came
from a frontier model; absolutes are not the bar), but the class-2 avenues go last in
the chapter's order. Class 1 first, so every commit until then keeps the bitwise
golden as its gate and the chapter runs on the instrument it already has; the class-2
work at the end builds the forced-token comparison once and re-captures the golden
once, at the close rather than after every change. Class 3, the real output-changing
levers, stays off the table regardless of order.

## 6. Sequencing (a draft, 2026-09-09; Davor's ruling pending)

**The interactions that decide the order.** Five, none of them visible from the
avenues' sizes alone:

1. **The SSD avenues compete for one budget** (the ring's cells, the reader's four
   threads, the drive's 2.9× headroom) and each changes the miss profile the others
   are priced on. Fewer misses (A1, A2) shrink the prize of any lead; a longer lead
   (A9) makes width (A0) unnecessary. Price all three offline first, then build one
   mechanism, priced on the post-A1 profile. Never width and lead both.
2. **The fixup chain is ordered by dependency, not by ease**: C5, then C6, then one
   command per layer (which prices E1), then E2, then the fold (K). A small win that
   touches the hit command's encode before C5 deletes that command is throwaway work.
3. **H2's decoder must be written once.** A class-2 kernel rewrite (J1, B6) would
   redo it. Scope H2 to the kernels no rewrite touches, the expert GEMVs and the head
   (all of the SSD bytes and 40 % of the streamed ones), and leave the dense weights'
   decoder to the rewrites.
4. **The class-1 measurements that gate the class-2 block run early, not last.** B5
   (the calibration probe) and B7 (the ablation) are measurements; if the hardware
   says the scan is at its wall, B6 is dropped from the plan before anyone budgets
   for it.
5. **Every landed step re-baselines the ledger.** The miss path's counters change
   meaning as C5 and C6 land; each step's prize is measured against the previous
   step on the arms rig, never against the v17 close.

**The rule:** price first, then build in dependency order within a surface, and
interleave surfaces so each step is measurable alone.

**Phase 0, pricing (days; no runtime code; the two probes are the only model runs).**
A9's Q2 (the miss profile by layer) and Q1 (token-identity and consecutive-position
overlap) from the T4 traces and token files; A0 (the width sweep at both distances)
from the archived captures; A1 and A2 through the replay; C1 and D1 as reads; H2's
entropy on ornith15 from the `.gturbo`; the prefill outlier's log; B5 on the mini
(one lifetime, Shrike stopped) and I3 with a per-dispatch timestamp dump. Output: a
priced board, the one SSD mechanism chosen, B6 kept or dropped.

**Phase 1, the fixup chain (class 1, ordered).** C5, C6, one command per layer, E2.
Each landed with the four gates, the golden identical, the arms on the mini. About 5
ms modelled on the 300, the transitions' share uncertain.

**2026-09-09, after T2.0, Davor:** C6 skipped as a performance task (at most 0.34
ms, under the drift). Phase 1 is C5 (landed, a measured null kept as a
simplification), then E2, then one command per layer, then the fold decided on the
transitions' arms; 0.9 to 2.7 ms modelled, the transitions the term that decides.

**2026-09-09, after Task 3, Davor:** the kernel merges (D1's six, about 2.2 ms
modelled with the slack rule) had no phase; they are v18's Task 6, after Task 3
and before the fold's ruling, so nothing on the board is left unowned.

**Phase 2, the SSD mechanism chosen in phase 0.** A1 or A2's product first if the
replay pays (a slot allocation is small code), then A9's predictor or A0's width,
one of them, measured against the post-phase-1 ledger.

**Phase 3, the fold (K).** One command per token, two in flight.

**Phase 4, H2 scoped to the experts and the head (class 1).** The format, the
repack, the decoders in the kernels that stay. Independent of phases 1 to 3 and
large; it can run beside them if the chapter has two hands.

**Phase 5, the class-2 block, last.** The forced-token instrument first, then B6 and
J2 if B5 allowed them, J1, and G1 if wanted; one golden re-capture at the end.

**Next chapter:** H1 (speculation with prompt lookup and a decode-width pass), H3
(the draft in the spin-wait), G2's remainder.

**What "fastest and easiest first" gets right:** phase 0 is exactly that, and it is
the correct first week because its outputs decide which mechanisms exist. **What it
gets wrong:** applied to the builds it would spend work on paths the next step
deletes, build two competing SSD mechanisms, and leave the class-2 block's go/no-go
measurement for the end.

## 7. Open questions

- Which surface opens first: the miss window (the larger number on the fixed shapes),
  attention at context (the larger number in the sessions the box actually serves),
  or the fixup machinery (never had a chapter)?
- Resolved 2026-09-09 (section 5): class 3 closed, class 2 open under the variance
  rule with the three-instrument gate; C4 resolved (eight, never nine).
- What temperature Pi sends: the model's default per Davor, likely 0.6 for ornith
  (its model card carries suggested values); not logged by the server. Decides I1.
