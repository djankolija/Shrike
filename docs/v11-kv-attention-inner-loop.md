# v11: the KV-attention inner loop — ending the per-position stand-up

Design for the last large decode lever. Companion plan:
[v11-implementation-plan.md](v11-implementation-plan.md).

## Evidence chain (all 2026-09-01, receipts in v10's plan)

1. Role split: the entire +8.0 ms/1000-ctx depth tax lives in the 10 KV
   layers (`attn_layer_kv` 5.24 → 20.07 ms/token over ctx 48 → 1900);
   the 30 linear layers are flat. ~65× the per-layer KV-read roofline.
2. Chunk-count probe (16 vs 64): NULL at every depth — sequence
   parallelism is not the constraint.
3. Kernel read (`attention.metal:228-258`, `attention_decode_partial`):
   the per-position loop runs, for every KV position, a K-dot where each
   of 256 threads loads ONE element, then a full `block_reduce_sum`
   (simd reduction + threadgroup scratch + barriers + broadcast), then a
   V-accumulate. Two threadgroup barriers and a cross-simdgroup
   reduction per position, against ~2 loads + 2 FMAs of real work per
   thread. The serial barrier chain is the cost; the loads are noise.

## Design

**One SIMD group per position; no threadgroup barrier in the loop.**

- Layout: 256 threads = 8 simdgroups. Positions are strided across
  simdgroups (`p = p_start + sg_id`, step 8). Each simdgroup computes
  the full 256-dim dot itself: 32 lanes × 8 elements, reduced with
  `simd_sum` only — no threadgroup traffic.
- Each simdgroup maintains its own online-softmax state
  (m_sg, d_sg, o_sg[HD/32 per lane]) over its position subset.
- At chunk end, the 8 simdgroup partials merge with the standard
  rescale — the identical algebra the pass-2 combine already applies to
  chunk partials, applied one level down. One barrier + one small
  reduction per CHUNK instead of two per POSITION.
- Q stays in threadgroup memory as today (loaded once, read per lane).
- The GQA/SWA variant gets the same treatment if the twin confirms the
  shape (same loop body, different partial grouping).

## Numerics policy (explicit, for the sign-off ledger)

Not bitwise, by construction: positions accumulate per-simdgroup and
merge at chunk end, reordering the softmax summation — the same class of
change as a264b22 (better-conditioned pairwise-style merge) and the
chunk-probe sign-off. Digest WILL differ. Acceptance is therefore:
- a numeric arm vs the current kernel (max |Δ| within FP16-accumulation
  tolerance across the four shape classes), not a bitwise arm;
- the same-session twin (rig + depth ladder; the judgment metric is the
  warm `attn_layer_kv` slope, +8.0 → target ≤ +1.5 ms/1000 ctx, with
  wall/wait improvements to match);
- output sanity on real prompts, and a fresh golden baseline captured at
  T5 AFTER acceptance (baselines/ is still empty — capture once, on the
  machine that checks it, rig + ~2k prompt per the standing note).

## Expected win, honestly bounded

Barrier-bound serial chain (~0.4 µs/position/layer measured) collapses
toward load latency; the GQA 8× re-read of KV bytes remains (each Q head
still reads its KV head's chunk). Floor estimate at 1900 ctx: ~0.25
ms/layer vs today's ~2.0 — i.e., most of the depth tax, worth ~10+
ms/token at 2–3k ctx. If the twin shows less, the residual is load
latency, and a KV-head-major regrouping (share K/V reads across the 8 Q
heads of a KV head) is the follow-on, not a rewrite of this design.
