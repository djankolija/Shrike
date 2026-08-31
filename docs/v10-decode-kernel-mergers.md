# v10 — decode kernel mergers (Phase 3 of the decode endgame)

Design for the last decode tier: multi-way kernel mergers on the attention
chain and the shared-expert chain, plus the residual wake investigation.
Companion plan: [v10-implementation-plan.md](v10-implementation-plan.md).

## Where this starts (checkpoint 349edc8, 2026-08-31 ~21:30)

Deployed best-known config: `--ram-budget 8G` (128 slots), S3a pacing
default, standard perf env. Mini rig **40.0 ms/token (25.0 tok/s)**, fresh
4-turn card **57.9 ms/token (17.3 tok/s)**, digest `494bab3edb62` exact
everywhere. M4 Pro converged rig 18.2 ms (55 tok/s).

Everything below rests on the measured laws from 2026-08-31 (handoff:
`~/.claude/handoffs/shrike-decode-bandwidth.md`):

- Empirical streaming roof **62.5 GB/s** (head and spec MoE sit on it).
- Intrinsic decode traffic **1,800.5 MB/token** (Opus-audited to the byte).
- **Boundaries cost wall time only on the host-waited critical path.**
  CB-count and single-seam fusions measured zero (Stage A, diet-1, Stage C).
- Miss-plan walls are **pure read time** (fetch = 99.9 % of load).
- GPU event-wake tax ≈ 162 µs/miss-layer on the dedicated queue (S3b,
  retired), ~1.75 ms/token residual on the main queue at card shape.

## The honest target arithmetic — read this before believing any estimate

The earlier "attn drain ~5.4 ms recoverable" over-counted: it priced every
small kernel at zero. A dispatch that touches 16–64 KB still costs a
~10–15 µs execution wall on the M1 (launch + memory latency; the diet-1
null is the proof that a *single* such wall is below measurement noise).
Re-derived per-GDN-layer floor with real walls:

| dispatch | bytes | honest floor |
|---|---|---|
| input rmsnorm | ~8 KB | ~10 µs |
| fused in-proj (qkv+z+a+b) | 14.2 MB | ~227 µs |
| conv_mix_decode | ~100 KB | ~10 µs |
| qk_norm | ~32 KB | ~10 µs |
| delta_step_decode | 4.2 MB state r+w | ~67 µs |
| gated_norm | ~24 KB | ~10 µs |
| o_proj GEMV | 4.7 MB | ~75 µs |
| **sum** | | **~409 µs** vs **486 measured** |

So the *real* recoverable pool in the GDN chain is ~77 µs/layer ≈
**~2.3 ms/token**, reachable only by collapsing several walls at once.
Phase 3's total honest envelope: ~1.5–2.5 (GDN mergers) + ~0.5–0.8
(shared-chain merger) + 0–1.5 (wake, uncertain) ≈ **2–4.5 ms/token**.
Card 57.9 − that ≈ 53.5–56 → **18–19 tok/s real**; 20 needs everything to
land at the top of its range plus further hit-rate luck. State that
plainly at the checkpoint rather than promising 20.

## Candidates, ranked

### C0 (prerequisite, mechanical): decompose the runner init
397/400 lint lines; any Phase 3 touch breaches. Extract buffer-allocation
blocks (`buf(...)` clusters) into focused helpers first, as its own commit.

### C1 — GDN mega-merge: conv + qk_norm + delta into one kernel (~30 layers)
Collapses three walls (~25–30 µs/layer) and the conv_out round-trip.
Bitwise analysis:
- conv per-channel math is order-preserving per element — safe.
- qk_norm's reduction must be replicated with the IDENTICAL summation tree
  (strided loop, `simd_sum`, ascending partial merge — `gdn.metal:417-431`).
  The merged kernel's threadgroup geometry differs from qk_norm's
  function-constant `tgThreads`, so the norm phase must mimic the original
  flatten (tid = y*32+x over the original stride pattern) — verify with a
  bytewise arm against the three-kernel reference before wiring.
- The rounded-through-half handoff matters: today delta reads
  `half(x*invRms*scale)` from conv_out; the merged kernel must round
  through half at the same point, not keep FP32.
- Grid shape: delta's (Hv, Dv/4)×(32,4) TGs each need conv output for one
  hk's q,k (256 elems) + one v slice. Conv-in-TG recompute per consumer is
  cheap (depthwise, per-element) but the TAIL SHIFT is a cross-TG write
  hazard — the tail update must move to a tiny separate kernel (1 wall
  survives) or the last TG per channel... simplest correct: keep
  `gdn_conv_tail_update`-style separate tail write, merge only the
  compute. Net: 4 walls → 2.
- Fallback if bitwise breaks irrecoverably: a264b22 precedent applies
  (better-conditioned order ≠ quality change), Davor signs off per-kernel,
  golden baseline re-captured after.

### C2 — shared-expert chain merger (~0.5–0.8 ms, all 40 layers)
gate+up are two 512×2048 GEMVs over the SAME input x → one 1024-row GEMV
call (row-wise concatenation; per-row dot order unchanged → bitwise safe).
The silu·mul + down + scalar-gate + sigmoid-mul tail can fold into the
down-GEMV epilogue. 5 dispatches → 2. Lives in the spec CB head post-Stage-C
(`encodeSharedExpertWork`).

### C3 — gated_norm placement (30 layers, ~10 µs/layer)
Folding into o_proj prologue is REJECTED (every o_proj TG would recompute
32 head-norms — adds work, breaks bitwise cheaply). Folding into delta's
epilogue needs cross-TG reduction over y per head (32 dv-TGs per head) —
only viable if C1's regrid lands a per-head TG shape. Treat as a C1
extension, not standalone.

### C4 — wake residual investigation (~1.75 ms/token card, uncertain yield)
The fixup CB still waits the MTLIO shared event (SHRIKE_EXPERT_IO_SYNC=event)
on the main queue. Alternative to test cheaply behind an env knob: host
spins on the I/O completion (the host is already in its router spin) and
commits the fixup CB without an encoded wait. Measure commit→GPU-start vs
signal→GPU-start; keep whichever is smaller. May be a null — budget one
A/B, not a campaign.

### C5 — spec-kernel occupancy audit (part of the 1.5 spec excess)
moe_spec_routed 11.69 post-Stage-C = shared (2.16) + spec (9.02) + fold
slack; spec alone ran ~95 % of roof pre-fold. After C2 shrinks the shared
half, re-measure before assuming anything remains here.

## Measurement discipline (unchanged, binding)

Same-session twins only (the mini drifts ~+1.4 ms/day). Fresh server per
arm; rig n=12 AND the 4-turn card per accepted candidate; digest
`494bab3edb62` before commit; bytewise test arm per merged kernel against
the multi-kernel reference (RMSNormTests pattern). Five gates per commit.

## Definition of "decode settled" (exit criteria)

1. C0–C2 landed or explicitly closed with data; C4 A/B'd once.
2. Golden baseline captured at last (baselines/ is still empty) — rig
   prompt AND a ~2k-token prompt, on the machine that will check it.
3. Default flips: pool/speculative/event/immediate promoted to code
   defaults (spin stays env until the thermal trial concludes), CLAUDE.md
   env paragraph shrunk accordingly.
4. S3b machinery removed or the opt-in documented as permanent.
5. Handoff closed with the final ledger; prefill quest opens.
