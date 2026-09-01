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
- [ ] **V3a: standalone partial-kernel microbench** — dispatch
      attention_decode_partial alone on synthetic buffers, seqLen
      ladder 128→4096, one axis at a time: fp16 vs int8 KV, NKV 2 vs
      8 (GQA re-read), group_size variants. Names the per-byte
      limiter in one run; local M4 first, mini to confirm.
- [ ] **V4: (redefined after V3a) the kernel fix the microbench
      indicates, then twin, then default flip + golden baseline (rig +
      ~2k prompt) per the standing T5 note; v10's Q2 closes with a
      pointer here.
