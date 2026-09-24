# v5: a second and third architecture — objective and steps

Status: objective agreed, implementation plan not yet written. **Done: both families are in `ModelFamily` (ModelTypes.swift:11-12) and served since the multi-model deploy (multi-model-serving.md:66-71, 163-168) (noted 2026-09-23).**

## Objective

Add **gpt-oss-20b** and **Kimi-Linear-48B-A3B** as loadable architectures alongside
the existing `qwen36` family, under the same bounded-memory contract, so the engine
runs recognizable third-party models rather than one architecture and its variants.

Both targets sit in the activation class the engine is built for (3.6B and 2.9B active),
both ship int4/group-64 affine MLX checkpoints, and neither is a Qwen derivative.

## Why these two

| | gpt-oss-20b | Kimi-Linear-48B-A3B |
| --- | --- | --- |
| Active / total | 3.6B / 21B | 2.9B / 48B |
| MLX 4-bit, group 64 | 11.8 GB | 27.6 GB |
| Router top-k | 4 | 8 |
| New attention operator | none | MLA (+ KDA) |
| Expert FFN | SwiGLU | SwiGLU |
| Expert container | `mlp.experts.*` | `mlp.switch_mlp.*` |

gpt-oss is the cheaper of the two — no new attention operator, and every blocker is a
conventional transformer variation. Kimi carries the higher learning value: KDA lands on
the gated-DeltaNet machinery already built for Qwen, and MLA's compressed KV bears
directly on the bounded-memory claim.

## What the fork starts from

This repo is `Pummelchen/NVMAI` at `4410d38`, itself a fork of `drumih/turbo-fieldfare`.
(Upstream keeps the NVMAI name; the rename to Shrike is ours alone, and `4410d38` no
longer resolves here — the imported history was squashed into the root commit.)
Commit `19aafd8` ("Qwen-only") removed the second-architecture support that upstream
still has: the `ChatDialect` enum, the Gemma family case, and the per-family branches in
`Model.swift`, `ArchInfo.swift` and `RepackPlanner.swift`. That commit sits at 70 of 310;
the v4.x engine — expert streaming, ANE prefill, tiled sampler, MTP — all came after it,
which is why the fork is the right base and upstream is not.

Recovery material for the removed abstractions is at `19aafd8~1` in `Pummelchen/NVMAI`.
It is **not** in this repository: the imported history was squashed to a single root commit
carrying only the fork-point tree, which is already after that removal.

## Steps

1. **Fork and clone.** — done. `djankolija/Shrike`, standalone (a GitHub fork was not
   possible: `djankolija/turbo-fieldfare` already occupies that fork network).
   `origin` = `djankolija/Shrike`, `upstream` = `Pummelchen/NVMAI`, both over
   `git@github-personal:`.

2. **Restore the dialect seam.** — done, minimally. `ChatDialect` is resolved from the
   tokenizer's framing tokens at load and dispatched in `applyChatTemplate`; a tokenizer
   matching no case is now rejected rather than rendered as ChatML by default. One case
   (`.chatml`), no behaviour change. Adding `.harmony` / `.kimi` is a case addition, and
   the exhaustive switch makes every render site that needs updating a compile error.
   Gemma's dialect was deliberately not restored — see *What was left out*.

3. **Generalize the router top-k.** `router_topk_select_k8` is the only selection kernel
   and `MoE.realDecodeTopK` is a static 8. gpt-oss needs 4. Kimi is already 8.

4. **Accept flat tensor names.** `RepackPlanner.classify` requires a `language_model.`
   prefix and `ArchInfo.load` requires a `text_config` block; both targets have neither.
   `ArchInfo.load` also hard-throws on any `model_type` outside
   `qwen3_5_moe` / `qwen3_5_mtp`.

5. **gpt-oss-20b.** Sliding-window revival (window 128, alternating S/F — the SWA
   scaffolding survives the purge but nothing exercises it), attention sinks
   (`self_attn.sinks`, a learned logit in the softmax denominator), additive biases on
   `q/k/v/o_proj`, `mlp.router.bias` and the expert FFNs, `mlp.experts.*` as the routed
   container, and the Harmony chat format — a multi-channel protocol
   (analysis / commentary / final), not a delimiter swap.
   **YaRN is already implemented** (`YaRNRoPEParameters.swift`, `rope_yarn_neox_subdim`).

6. **Kimi-Linear-48B-A3B.** MLA on 7 of 27 layers (`kv_lora_rank` 512, `qk_nope` 128,
   `qk_rope` 64) is the substantial new operator. KDA on the other 20 maps closely onto
   the existing GDN kernels — `A_log`, `dt_bias`, causal conv, low-rank gate projections
   and a gated output norm all have counterparts. Plus dense layer 0, one shared expert,
   and a sigmoid router with `e_score_correction_bias`.

Steps 2–4 are shared foundation and serve both targets.

## What was left out of step 2, and why

Step 2 was originally priced as a near-clean cherry-pick on the strength of net file
sizes (`Tokenizer.swift` 533 → 555 lines, `ModelTypes.swift` 410 → 396). That was wrong:
the purge rewrote 284 lines inside `Tokenizer.swift` and 101 inside `ModelTypes.swift`,
and ten commits have touched the tokenizer since (thinking controls, incremental
ByteLevel decode, Ornith support, the Sendable audit). It is a port forward, not a revert.

Most of those 284 lines are Gemma's template renderer and tool parser. Gemma will not be
run here, so restoring them would carry dead weight and a real merge against 240 commits
of drift to prove a seam that one case already establishes. The seam was taken; the Gemma
implementation was not.

The per-family branches removed from `Model.swift`, `ArchInfo.swift` and
`RepackPlanner.swift` were also left alone. Those are per-architecture code that gets
written fresh for gpt-oss and Kimi anyway; restoring Gemma's versions would not shorten
that work.

## Not carried forward

`FusedLayerTail.swift`, `FusedPostAttentionSetup.swift`, `PrefillLayerTail.swift` and
`PrefillPostAttentionSetup.swift` were Gemma's sandwich-norm fusions. Neither target has
sandwich norms; leave them deleted.

## One thing not to misread

`d14376f` ("Route every Top-K in 1...64 through the tiled sampler" — upstream's commit, in
`Pummelchen/NVMAI`, not resolvable here) is the **sampling** top-k over the vocabulary. It
is a different code path from the MoE **router** top-k, and does nothing for step 3.

## Kimi's decode, measured before the decode chapters

Measured 2026-08-27 on that day's build; no later Kimi decode measurement is recorded in
these docs.

- **On the mini Kimi is compute-bound, not IO-bound.** `kimi-linear-48b-a3b-4bit`, 32K
  context, `SHRIKE_RUNNER_STATS`, an expert-cache slots sweep: 6.05 tok/s at the standing
  64 slots per layer (`--ram-budget 6G`), rising monotonically to 6.18 at 96 against an
  extrapolated compute floor of about 6.35. At 96 slots misses, evictions and reloads are
  equal: every miss is a capacity miss, none compulsory in the warm window. `wait_ms`
  climbs at the top (70.2, 77.5, 95.9 at 32, 64, 96 slots) while `io_ms` falls, so
  pressure starts to show at 96; 128 slots (about 12.3 GB on the 16 GB box) was not run.
  An earlier slots matrix's "more slots is slower" did not reproduce; it is attributed to
  that run's 262,144-token KV reservation, which 32,768 does not approach.
- **On the MacBook, fully resident,** mlx runs Kimi at 48 to 81 tok/s where this engine ran
  24 to 30, at identical arithmetic.
- **Untried for Kimi's decode:** fusing the six-GEMV KDA f/g low-rank chains in
  `encodeKDADecode` (`RealForwardRunner.swift`). No commit since the Kimi bring-up touches
  them; the decode chapters targeted ornith15.
