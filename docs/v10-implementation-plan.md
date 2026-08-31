# v10 implementation plan — decode kernel mergers

Status of record for [v10-decode-kernel-mergers.md](v10-decode-kernel-mergers.md).
Checkboxes here are the only status tracking; do not duplicate elsewhere.

Protocol per task: implement → bytewise arm vs the multi-kernel reference →
local digest (`494bab3edb62`) → five gates → deploy → same-session twin on
the mini (fresh rig n=12 + 4-turn card) → accept/revert on the twin, not
on hope. Baseline anchor at start of Phase 3: rig 40.0 / card 57.9
(tonight-terms, 349edc8, 8G/128 slots, S3a).

## Tasks

- [ ] **T0: decompose the runner init** (397/400 lint lines; mechanical,
      own commit, no behavior change; suite green is the only gate that
      matters here beyond lint).
- [ ] **T1: shared-chain merger (C2)** — gate+up as one 1024-row GEMV,
      tail ops folded into the down epilogue; 5 dispatches → 2 in
      `encodeSharedExpertWork`. Bitwise-safe by construction (per-row dot
      order unchanged); bytewise arm anyway. Expected −0.5–0.8 ms/token.
      Ordered first: smaller blast radius than T2, validates the merger
      workflow.
- [ ] **T2: GDN mega-merge (C1)** — conv compute + qk_norm + delta in one
      kernel; tail shift stays separate. The bytewise arm against the
      three-kernel reference is the acceptance gate; if the qk_norm
      summation tree cannot be replicated exactly, STOP and take the
      a264b22 route only with Davor's per-kernel sign-off + golden
      baseline re-capture. Expected −1–2 ms/token.
- [ ] **T3: wake A/B (C4)** — env-knobbed host-spin-commit alternative to
      the encoded I/O event wait on the fixup path. One A/B (rig + card),
      keep the winner, delete the loser's knob. Expected 0–1.5 ms/token.
- [ ] **T4: re-measure spec excess (C5)** after T1; open a task only if
      >0.5 ms remains above floor.
- [ ] **T5: settle** — golden baseline capture (rig + ~2k prompt),
      default flips (pool/speculative/event/immediate → code defaults;
      spin pending thermal verdict), CLAUDE.md env paragraph shrunk,
      S3b machinery removed or opt-in documented permanent, handoff
      closed with the final ledger.

## Explicitly out of scope (Davor's line)

Expert substitution on miss; FP16 GDN state; any change that alters
sampled output beyond the a264b22-class reduction-order exception, which
requires his explicit per-instance sign-off.
