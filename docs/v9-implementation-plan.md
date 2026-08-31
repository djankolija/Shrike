# v9 — Implementation plan

Design: [v9-speculative-routed-dispatch.md](v9-speculative-routed-dispatch.md).

Every stage lands independently behind gates: release build with zero
warnings, `swiftlint --strict` against the baseline, `swift test
--no-parallel`, the same suite under TSan, and an n=12 rig run on the mini
with the digest oracle unchanged (`494bab3edb62` for the fixed prompt).
Numbers cited as baselines are ornith15, n=12.

## S0 — prerequisites (no repo changes)

- [x] Validate GPU residency classification end-to-end on the target box
      (`SHRIKE_DECODE_EXPERT_EXECUTION=gpu-residency`, 2026-08-30): digest
      identical, 0 fail-closed mismatches, cost of validation mode +2.1 %.
- [x] Validate pool cache layout (`SHRIKE_EXPERT_CACHE_LAYOUT=pool`,
      2026-08-30): digest identical, allocation fine at `--ram-budget 6G`,
      and far from neutral — **body 100.83 → 77.13 ms/token (−23.5 %)**.
      The unattributed round-trip latency was Metal residency/hazard
      management over ~3400 per-slot buffers: the big gap halved
      (32.6 → 16.7 ms/token) and every CB got faster (GPU busy 66.2 → 59.8).
      Pool is now part of the deployed env. Follow-up recorded in S4: flipping
      the code default requires a graceful per-slot fallback, because pool
      allocation failure currently aborts startup (`StreamerError.allocFailed`).

## S1 — classifier writes the speculative dispatch surface

- [x] Extend the residency-classification kernel to write indirect dispatch
      arguments (full grid iff `missCount == 0`, zero otherwise) and keep
      `resolvedSlots`/`resolvedGenerations` as today.
      (`moe_classify_expert_residency_spec` + `MoE.SpeculativeDispatchArguments`,
      `f662d4a`.)
- [x] Unit tests for the argument encoding (CPU-readback comparison across
      hit/miss permutations; no model load).
- [x] Gates: build 0 warnings, lint clean, 1080 tests, TSan 0 races. No
      behavior change in any existing mode (base kernel untouched).

## S2 — pool-addressed spec kernels, still host-waited

- [x] Phase-1 gate/up variant taking `poolBase` + `poolSlotStride` +
      `resolvedSlots`; dispatched indirectly from S1's arguments. Byte-exact
      vs the production pipeline through scrambled pool slots, inert under
      zero grids (`98a8925`, MoEFusedFFNTests).
- [x] Phase-2 equivalent (same commit).
- [x] `SHRIKE_DECODE_EXPERT_EXECUTION=speculative` validation mode
      (`2625f5c`, ping-pong fix `0685249`): spec CB committed before the tail
      wait; memcmp cross-check vs the classic path on all-hit layers, fail
      closed. First run caught a validation-harness race (single scratch pair
      compared one layer late) — fixed with parity ping-pong; the fail-closed
      design worked as intended.
- [x] n=12 acceptance 2026-08-30: **body 77.13 → 62.60 (−18.8 %), wall
      17.19 → 14.33, digest identical, zero divergences (~68 k all-hit
      cross-checks).** The extra CBs are not a cost but a win: the spec CB
      fills the former idle gap, the queue never drains, and the
      idle-restart tax that inflated every CB disappears (attn 22.9 → 17.6,
      classic routed 11.6 → 8.5, shared 5.7 → 4.5; residual gaps ~5.8 + 3.2;
      busy_share_of_decode 76.4 %). The standing prediction from `a8168d9`
      is confirmed — the absorbed savings reappeared once the gap was filled.

## S3a — speculation authoritative, host still pacing (landed `317ed7c`)

- [x] Spec CB targets the real buffers (`moeActs`, `h2Buf`, `hidden` via a
      third indirect-gated dispatch reusing `residual_add_fp16`); on all-hit
      layers it IS the routed command and the classic path is not encoded.
      Miss layers: spec CB self-nullifies, classic path unchanged. S2's
      compare mode preserved as `speculative-validate`.
- [x] No eviction race by construction in this stage: per-layer host pacing
      means the next cache plan runs only after the spec CB completed.
- [x] Acceptance n=12 (2026-08-30): **body 62.60 → 59.46 (−5.0 %), wall
      13.31 s, digest byte-identical with speculation authoritative.**
      Classic routed work gone from all-hit layers (GPU busy 55.4 → 48.5
      ms/token); dominant remaining gap moved to
      `moe_spec_routed → attn_norm_qkv` (8.2 ms/token) — the host wake that
      S3b removes.

## S3b-lite — encode-ahead, commits-only critical path (landed `26ed857`)

- [x] `encodeLayerCommands` builds a routed layer's five CBs uncommitted
      (`HeldLayerCommands`); in speculative mode the loop pre-encodes layer
      L+1 during GPU-busy time and the post-readback critical path is commits
      only. Miss ordering is host-controlled by commit order (fixup before the
      held successor) — no events, no second queue, no eviction change.
- [x] Acceptance n=12 (2026-08-30): **body 59.46 → 55.03 (−7.5 %), wall
      12.58 s, digest byte-identical; spec→attn gap 8.2 → 4.2 ms/token.**
      Remaining gaps: miss fixup 6.5, spec→attn residual 4.2 (wake + plan —
      true event-machinery territory), spec→hit 3.2.

## S3b — event-gated successors, host off the per-layer critical path

- [x] Micro-test of the same-queue commit-order + cross-queue event
      assumptions — landed as a durable suite instead of a throwaway target
      (`CrossQueueSharedEventTests`, 4 tests, serial + TSan green 2026-08-31)
      so the assumptions stay pinned under both test gates.
- [ ] `layerDone` shared event; `attn(L+1)` waits; host signals on all-hit,
      fixup CB signals on miss (second queue).
- [ ] Eviction epochs: planner defers evictions of slots classified for
      in-flight layers until completion handlers retire them;
      generation cross-check retained in validation builds.
- [ ] TSan suite green (the epoch bookkeeping is the racy part; add targeted
      tests around retire-vs-plan).
- [ ] n=12 acceptance: digest identical.

## S4 — measurement and the standing prediction

- [ ] n=12 vs the pre-v9 baseline (**post-E5: body 77.13 sd 1.43, wall
      17.188 sd 0.293**, digest `494bab3edb62`).
- [ ] Consider flipping `ExpertCacheLayout`'s code default to pool (with a
      graceful per-slot fallback on allocation failure) — E5 leaves ~23 % on
      the table for any deployment that forgets the env var.
- [ ] Verify the shared_expert→routed gap collapses for all-hit layers in the
      per-role/gap report.
- [ ] Verify Davor's prediction: the win exceeds the gap reduction by roughly
      the ~1.4 ms absorbed at `a8168d9`. If absent, halt and re-measure before
      any further change.
- [ ] Decide default: flip `SHRIKE_DECODE_EXPERT_EXECUTION` to `speculative`
      only after a soak beyond the fixed rig prompt (longer contexts, MTP off,
      concurrent requests).
- [ ] Revisit the parked GPU-side items now that savings can cash out:
      shared-expert CB encoder merge (B1c), `fused_qkv_epilogue` on the gated
      path, norm→GEMV fusions.
