# v11 implementation plan — KV-attention inner loop

Status of record for [v11-kv-attention-inner-loop.md](v11-kv-attention-inner-loop.md).
Checkboxes here are the only status tracking.

Protocol per task: five gates per commit; numeric arm (tolerance, not
bitwise — see the design's numerics policy); same-session twin on the
mini (fresh rig n=6+ AND the warm depth ladder — judgment metric is the
warm `attn_layer_kv` slope); deploy = binary + bundles (new kernel!).

## Tasks

- [ ] **V0: numeric reference arm** — test comparing
      `attention_decode_partial` + combine against a scalar reference
      across the four shape classes (full, GQA/SWA, MLA excluded, sinks),
      tolerance-based; this arm then validates V1 unchanged.
- [ ] **V1: simdgroup-per-position partial kernel** — barrier-free
      position loop, per-simdgroup online softmax, chunk-end merge via
      the combine's rescale algebra. Behind
      `SHRIKE_ATTN_DECODE_LOOP=simdgroup` (default = current kernel)
      until V3 accepts.
- [ ] **V2: Swift wiring** — PSO selection by the knob; grid unchanged
      (the chunk geometry stays; the loop inside changes).
- [x] **V3: twin verdict — NULL, hypothesis falsified (2026-09-01
      ~14:00, twin on 77af089).** Warm attn_layer_kv at ctx 1900:
      20.35 (sg) vs 20.18 (off); slope +8/1k unchanged at every rung;
      rig 39.02 vs 38.73 (noise); rig digest incidentally identical;
      outputs sane. The cross-variant test proves the sg kernel ran —
      so the per-position barriers were NOT the cost, and with the
      chunk-probe null this exonerates chunks, barriers, AND
      threadgroup concurrency (8× more positions in flight changed
      nothing → the machine-wide limiter is per-byte work, not
      structure). Survivors: the int8 attn_load_kv dequant (per-element
      scale/bias loads + integer div/mod) and the 8× GQA re-read.
      Knob kept default-off while iteration continues.
- [x] **V3a: standalone microbench — MECHANISM NAMED: GQA READ
      AMPLIFICATION (2026-09-01 ~15:20, AttentionDepthBenchTests,
      M4 Pro, production encodeFull binding).** Depth slopes
      (µs/position, best-of-15): fp16-private 0.180, fp16-shared
      0.186, int8-private 0.205, int8-shared 0.204 — so the int8
      dequant/runtime-divisor theory (peer's #1) and the
      storageModeShared theory (mine) are BOTH exonerated; slopes
      also reproduce production (M1 0.43 ≈ machine ratio). The GQA
      arm decides it: at NKV=16 (amplification 1×) the slope
      collapses to 0.025 — 7.4× shallower than NKV=2 (amplification
      8×) while reading 8× more unique bytes. The 8-Q-heads-per-
      KV-head re-read IS the depth tax. Anomalies on record: NKV=4
      measured ≈ NKV=2 (not intermediate); this arm's absolute
      baselines run hot vs the first ladder (DVFS suspect) — slope
      comparisons within-run only. Fix direction: KV-head-shared
      threadgroups (read each KV chunk once, compute all 8 Q heads'
      dots against it) — changes lane assignment/reduction order,
      a264b22-class, NEEDS Davor's per-instance sign-off.
- [ ] **V4: (redefined after V3a) the kernel fix the microbench
      indicates, then twin, then default flip + golden baseline (rig +
      ~2k prompt) per the standing T5 note; v10's Q2 closes with a
      pointer here.
