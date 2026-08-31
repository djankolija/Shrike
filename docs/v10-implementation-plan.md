# v10 implementation plan — decode kernel mergers

Status of record for [v10-decode-kernel-mergers.md](v10-decode-kernel-mergers.md).
Checkboxes here are the only status tracking; do not duplicate elsewhere.

Protocol per task: implement → bytewise arm vs the multi-kernel reference →
local digest (`494bab3edb62`) → five gates → deploy → same-session twin on
the mini (fresh rig n=12 + 4-turn card) → accept/revert on the twin, not
on hope. Baseline anchor at start of Phase 3: rig 40.0 / card 57.9
(tonight-terms, 349edc8, 8G/128 slots, S3a).

## Tasks

- [x] **T0: decompose the runner init** (397/400 lint lines; mechanical,
      own commit, no behavior change; suite green is the only gate that
      matters here beyond lint). Landed: init 397 → ~100 lint lines via
      per-cluster static factories (kernels, prefill kernels, decode/
      residency/GDN/MLA/MTP scratch bundles, shared projections, router
      buffers); conditional clusters stored as bundles behind computed
      forwards, so no use site moved. Baseline 21 → 20 entries.
- [x] **T1: shared-chain merger (C2)** — LANDED 132448c, deployed,
      **wall-NEUTRAL on the twin** (rig 38.55→38.59, card 56.16→56.32,
      moe_spec_routed 12.83→12.69 ≈1σ; digest exact everywhere; kept as a
      simplification per the diet-1/Stage-C precedent). What shipped:
      silu·mul + down + sigmoid fused into `dequant_int4_shared_down_fused`
      (bitwise arms: gated/ungated/remainder), shape-specialized PSOs for
      the chain's GEMVs. Two design amendments: the gate+up 1024-row
      concatenation was replaced by two dispatches (no file-layout
      assumption), and the concurrent-encoder overlap is VETOED by an AGX
      driver segfault (see the code comment in `encodeSharedExpertWork`).
      Law refinement: the expected −0.5–0.8 was shadow — the shared chain's
      dispatch walls sit off the GPU-critical path on both shapes.
- [ ] **T2: GDN mega-merge (C1)** — conv compute + qk_norm + delta in one
      kernel; tail shift stays separate. The bytewise arm against the
      three-kernel reference is the acceptance gate; if the qk_norm
      summation tree cannot be replicated exactly, STOP and take the
      a264b22 route only with Davor's per-kernel sign-off + golden
      baseline re-capture. Expected −1–2 ms/token.
- [x] **T3: wake A/B (C4)** — RAN 2026-08-31/09-01, **NULL on the M1;
      winner = the encoded wait, host-spin knob deleted.** Four fresh-server
      arms on one binary (t3-747a18a): rig wall 9.308/9.307 s, card body
      57.69/58.45 (wait/host-spin); io_fixup_wake did not improve under
      host-spin (fresh-commit schedule ≈ parked wake on the M1). Digest
      exact in both modes. M4 local had shown −0.47 ms/token body — does
      not transfer; noted in 040bb22's message if M4-class hardware ever
      becomes a target. ⚠ wait_ms drops under host-spin are COLUMN
      SHIFTING into cb2_ms — judge any future wake idea on wall/body.
      Prerequisite paid: encodeDecodeRoutedMoE decomposed (398→~310,
      buildAndCommitMissFixupCommand extracted).
- [x] **T4: re-measure spec excess (C5)** — CLOSED with T1's twin as the
      measurement: reducing the shared half of moe_spec_routed was
      wall-neutral on both shapes, so the spec-CB excess above floor is
      shadowed/off the critical path; nothing >0.5 ms recoverable there.
      No task opened.
- [ ] **T5: settle** — golden baseline capture (rig + ~2k prompt),
      default flips (pool/speculative/event/immediate → code defaults;
      spin pending thermal verdict), CLAUDE.md env paragraph shrunk,
      S3b machinery removed or opt-in documented permanent, handoff
      closed with the final ledger.

## Close-out probes (added 2026-09-01 — part of the settled gate)

Each is "run once, record the verdict, close either way"; definitions in
[v10-decode-kernel-mergers.md](v10-decode-kernel-mergers.md).

- [x] **P1: machine-roof probe** — RAN 2026-09-01 (scratchpad `mtlbw`,
      runtime-compiled kernels, 1–2 GiB private buffers, 6 trials/arm).
      **M1 mini: machine ceiling ≈ 61 GB/s (blit copy r+w 61.1, best
      read kernel 60.5) — the big GEMVs' 62.5 role-stat rate is AT the
      machine roof (≤3 % method spread). Kernel bandwidth is NOT a lever
      on the mini; the 28.8 ms floor stands.** M4 Pro local: machine
      253 GB/s read (93 % of sticker) vs the head GEMV's measured 167 →
      ~⅓ kernel-side headroom exists on M4-class hardware only
      (occupancy/unpack tuning, not the machine). Out of scope for the
      mini chapter; recorded for any future M4-class work.
- [ ] **P2: attention-chain attribution** — per-kernel GPU times from
      the existing gputrace bundles vs the honest-floor table.
- [x] **P3: miss-read QD probe** — RAN 2026-09-01, both machines
      (scratchpad `p3ssd`: F_NOCACHE pread + MTLIO arms, 96 reads/arm,
      seeded picks over packed_experts). **Mini verdict: drive
      exonerated. Random ≡ sequential (0.79 ms p50 / 2.2 GB/s per
      1.77 MB expert read); QD4 lifts aggregate +48 % (3.25 GB/s).
      Production's 2.0 ms p50 decomposes: 0.79 drive + ~0.54 MTLIO
      single-load submission (probe 1-load-per-CB: 1.33 ms) + ~0.7
      in-engine queueing / GPU-contention residual. Batching 8 loads
      into one IO CB erases the submission overhead (0.81 ms p50).**
      Local M4 Pro / BuildSSD mirror: 0.55 drive, +0.30 MTLIO single,
      batch erases; page cache serves neither machine (cached ≈
      nocache). ⚠ Corrects the prefill-quest premise: the "0.8 GB/s
      random-read scheduling" gap is NOT drive random-read behavior —
      offset sorting buys nothing; the lever is batched submission +
      queue depth (Davor's idea, confirmed at the I/O layer).
      Follow-on (unscheduled): batch miss loads per discovery point,
      deepen in-flight QD; prefill batches whole tiles. Est. prize
      ~3–4 ms/token of real-shape exposed miss I/O + a large slice of
      prefill's 33 ms/token.

## Queued after T5 (Davor, 2026-08-31 — sequenced behind the original tasks)

- [ ] **Q1: between-token host overhead** (~5–7 ms/token for every request:
      decode window vs body_ms; GPU busy 77% of window on real turns vs 93%
      rig). First probe: stats-off A/B (RUNNER/KERNEL_STATS may tax the
      observed); then the per-token emit/detokenize/async-hop loop.
      Compare on WALL, never wait_ms.
- [ ] **Q2: context-depth tax — now sized as a genuine anomaly
      (2026-09-01).** Roofline for depth growth: only the 10 gated
      layers grow with context (30 GDN layers are constant-state);
      10 layers × 2 KV heads × 256 dim × K+V × int8 ≈ 10.2 MB per
      +1000 ctx ≈ **0.17 ms/1000 at the 61 GB/s roof — measured is
      7–9 ms/1000, ~40–50× over roofline.** Suspects: decode-attention
      kernel parallelism over context length (serial walk ⇒ latency-
      bound O(L)); the deep-prefill expert-cache sweep confound (split
      never measured); host-side O(L) work per token. Debug: fresh-
      server deterministic depth sweep (single_turn N ladder) with
      per-role stats — role growth localizes kernel vs io vs host.
      **MEASURED 2026-09-01 (ladder, warm arms = prompt-cached, hit
      0.90–0.94): the tax is in the attention-chain GPU role —
      attn_layer 21.68 → 34.97 ms/token over ctx 48 → 1900 = +7.2 ms/
      1000 ctx (~40× KV roofline); wait_ms slope only +0.7 because
      attention growth and declining miss exposure cancel (the
      confound that hid this). Cache-sweep cost is real but transient
      (cold arms +5–24 ms, hitD 0.82 vs 0.90 at t1). REMAINING: name
      the kernel — needs a gputrace captured at depth (~1900 ctx);
      folds into P2's analysis. Ladder logs:
      scratchpad ladder-runner.txt / ladder-roles.txt (session
      6dd4e253), mini /tmp/ornith.log (Server A PID 45374).**

## Quality-trading experiments (lane opened by Davor, 2026-09-01)

His rationale: int4 quantization is already an accepted quality trade,
and no public data exists for top-N-of-8 sensitivity on this model/
quant/hardware — so measure it. Everything here alters sampled output:
per-experiment sign-off stands, and no default flip without a quality-
battery verdict.

- [ ] **E0: router rank-mass instrumentation** — log mean routing-weight
      mass per rank (1..8) in runner stats (read-only, digest-neutral).
      Gates E1: if ranks 7–8 carry a few % of mass, dropping is
      plausible; if far more, stop here and record that.
- [ ] **E1: drop-bottom-miss experiment** — env-gated: on a miss whose
      normalized routing weight is below a threshold, drop the expert
      and renormalize over the executed set. Reproducible under the
      fresh-server rig protocol (deterministic cache trajectory), but in
      live traffic output varies with cache temperature — the model
      answers slightly differently when cold, exactly when it is also
      slowest (topic switches); flag this behavior explicitly at
      sign-off. Quality check: fixed prompt battery, side-by-side.
      Prize at 0.95 real-shape hit: a slice of the ~7–8 ms/token of
      exposed miss I/O.

## Explicitly out of scope (Davor's line)

FP16 GDN state; any change that alters sampled output beyond the
a264b22-class reduction-order exception and the E-lane above, which
require his explicit per-instance sign-off. (Expert substitution on
miss moved to the E-lane 2026-09-01 by Davor.)
