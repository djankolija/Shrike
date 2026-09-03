# v13 — the turn: prefill and decode at the shapes a conversation actually runs

## The problem

v12 ([v12-prefill-matrix-kernels.md](v12-prefill-matrix-kernels.md)) closed the
long-prompt prefill: 12k tokens on the mini went 725 → 68.4 s. The shapes this box
is used for are not that. LLMBench's cards are two-turn conversations with a
56-character system prompt and 50-character questions (tools off: the model
prefills 48 tokens and decodes 600–800) or a chain of tool rounds (tools on: a
495-token opening, then 600–2,000 tokens of tool results per round, 2–7
rounds, then the answer). A conversation reaches 12k only late in a tool-heavy
card. So the chapter's ledger is measured at 300 / 1k / 2k new tokens, cold and
warm, and at the follow-up turn, and its target is the wall of a turn, not a
per-token rate.

## Step zero — the ledger (mini, v12's close build 3774ef1, production launch)

One fresh server per pair; the warm arm is a different prompt of the same
length after the prompt cache's `settle_done`; eight decode tokens per request.
Prompts: the `tools/prefill-prompts.py` ledger construction at 4 / 16 / 32
entries (≈ 62.6 tokens each); the turns append the answer and a new question.

| shape (new tokens) | cold wall | warm wall | warm prefill_s | prefill GPU | `routed→routed` host | experts fetched (warm) | prefill hit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 289 / 305 | 8.91 s | 5.46 s | 5.04 | 2.5 s (8.2 ms/tok) | 2,001 ms / 942 | 6,875 (11.8 GB) | 11 % |
| 1,069 / 1,085 | 11.44 | 8.05 | 7.62 | 5.3 s (4.9) | 1,820 / 1,107 | 8,224 (14.1 GB) | 9 % |
| 2,093 / 2,125 | 14.45 | 11.21 | 10.73 | 9.6 s (4.5) | 759 / 1,148 | 8,527 (14.6 GB) | 9 % |
| turn 2 on the 2k context (38 new, 2,118 cached) | | 2.13 | 1.66 | 0.86 s (22.6) | 355 / 387 | | |
| turn 3 (36 new, 2,149 cached) | | 1.38 | 0.91 | 0.76 s (21.0) | 26 / 359 | | |

Decode: ≈ 0.4 s per request warm (8 tokens). Outside the generation span: 2.06 s
on a fresh server's first request, 0.02–0.08 s afterwards.

**What the rows say.**

1. **A first-turn prefill costs ≈ 4.5 s + 3.0 ms per token** (the warm intercept
   from the three pairs; cold adds ≈ 3.3 s: 2.06 outside the span plus ≈ 1 s of
   slower prefill). The intercept is the expert sweep: any chunk of a few hundred
   tokens routes to 190–250 of the 256 experts in every layer, so 6.9–8.5k
   expert fetches (12–15 GB) stream from the SSD at ≈ 3 GB/s with a 0–11 % hit
   rate — v12's alternating sweep (P15) only pays from a prompt's second chunk.
   GPU busy is 51–70 % of the span at these lengths: the routed path is
   fetch-bound outright (≈ 5 ms of fetch per 8-expert tile against 0.5–2.4 ms of
   GPU), which the two-tile bank (P16) cannot hide.
2. **Follow-up chat turns hit the prompt cache** (`prompt_cache hit tier=live`,
   `settle_restore`) and cost 1.4–2.1 s for ≈ 37 new tokens: 0.9–1.7 s of prefill
   = 40 layer-chunks of fixed work (the matrix kernels' per-dispatch floor at
   tiny row counts, 21–23 ms per new token; `shared→routed` 138 ms and
   `routed→routed` 355 ms of host on turn 2) plus ≈ 0.4 s of decode. A prompt
   that diverges INSIDE a stored entry gets no prefix (`settle_reset
   reason=no_prefix_snapshot`, `cached=0`, a full re-prefill): append-only turns
   — chat, tool rounds — are served; an edited document is not.
3. For the cards, the answers (600–800 tokens at 18–21 tok/s: 30–40 s) still
   dominate; first-turn and tool-round prefills (0.5–2k tokens: 6–11 s each) are
   second; follow-up-turn overhead (1–2 s) third; the fresh-server first request
   (+3.3 s) only after a restart.

## Levers, ranked for these shapes (modelled from step zero)

- **First-chunk hit rate** (the 4.5 s intercept is 12–15 GB of expert reads):
  carry the sweep parity ACROSS requests — a request's first chunk sweeps in the
  reverse of the previous request's last chunk, so the pool's 128 slots per
  layer serve the first tiles instead of missing them all. Modelled ≈ 50 % hits
  on the first chunk → −1.5 to −2 s per first-turn prefill and per tool round.
  Risk: decode between turns churns the pool (aging-LFU); a long answer between
  requests is an arm of the measurement.
- **Bytes per expert** (the same term): the drive runs at ≈ 3 GB/s here (QD 4
  inside a tile; 3.25 measured at QD4 in v10 P3); the reader's thread count as a
  knob (QD 8 unmeasured); two fetches in flight helps this mean-bound regime.
- **A small-row prefill path** for follow-up turns (≤ 64 new rows): the matrix
  kernels' per-dispatch floor costs 21–23 ms per new token over 40 layer-chunks;
  a decode-style or scalar path for tiny chunks could take ≈ 0.5–1 s off a
  1.4–2.1 s turn.
- **Decode** (v12 Task 17: hit-rate-bound, 22 / 18 / 12.5 tok/s by shape):
  prefetch accuracy on a tools context, cross-layer miss queue depth, the
  drive's latency (an external NVMe on the mini is a copy-the-model experiment).

Not levers here: the GPU kernels (busy ≤ 70 % of the span below 2k tokens); the
ANE (attention is 4 % of a 1.4k prefill); speculation (retired at v12 Task 17).

## Method

The v12 protocol carried over: mini-first verdicts (the M4 Pro is the check);
one send per server lifetime for whole-chunk rows and `settle_done` before a
warm second send; a distinct prompt per arm (no prompt a prefix of another);
`SHRIKE_RUNNER_STATS=1 SHRIKE_KERNEL_STATS=1`; the gap split read before any gap
is theorised about. Each task states its bars on this ledger's rows before it
runs, lands its knob either way (a measured null is a result), and reports the
same rows after. Scheduling-only changes are golden IDENTICAL on both boxes and
both profiles; anything that reorders arithmetic is qualified as v12 did.

## Numerics policy

Sweep order, prefetch and slot policy change which expert is fetched when, never
what any kernel computes: golden identical is the bar for them. A small-row
prefill path that runs different kernels on a chunk is a numerics change and
follows v12's policy (the 2e-2 bar against the fp32 reference, golden recaptured
once per box with the digests in the verdict).

## Out of scope

- Kernel work on the prefill GEMMs and attention (v12's follow-ons stay there).
- Speculative decoding (v12 Task 17).
- The prompt cache's interior snapshots (an edited document re-prefills; a
  cache-chapter item, recorded, not scheduled here).

## Risks

- The pool's contents after a long decode are the aging-LFU winners, not the
  sweep's tail; the parity lever may pay only for short answers — measured, not
  assumed.
- The mini is production: every arm restarts the server on 8081; Turbo on 8080
  is never touched; one model process at a time.
