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

- [ ] Phase-1 gate/up variant taking `poolBase` + `poolSlotStride` +
      `resolvedSlots` instead of a CPU-encoded blob argument buffer; dispatched
      indirectly from S1's arguments. Same math, same tgmem staging — output
      must stay byte-identical.
- [ ] Phase-2 equivalent.
- [ ] New mode `SHRIKE_DECODE_EXPERT_EXECUTION=speculative` that runs the spec
      CBs but keeps today's host wait and, in this stage, cross-checks spec
      output against the classic path (fail closed on divergence).
- [ ] n=12 acceptance: digest identical; cost of the extra CBs measured.

## S3 — event-gated successors, fast path live

- [ ] Micro-test of the same-queue commit-order + cross-queue event
      assumptions (throwaway target, no model).
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
