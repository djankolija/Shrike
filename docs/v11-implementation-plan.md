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
- [x] **V3a: standalone microbench — MECHANISM: TRAFFIC-BOUND AT
      ~40 % OF ROOF, GQA-AMPLIFIED (corrected 2026-09-01 ~16:00).**
      Round 1 (clean): depth slopes fp16-private 0.180 / fp16-shared
      0.186 / int8-private 0.205 / int8-shared 0.204 µs/position —
      dequant/divides and storage mode both exonerated; production
      reproduced (M1 0.43 ≈ machine ratio). Round 2 ⚠ RETRACTED: the
      "7.4× NKV=16 slope collapse" came from a CONTAMINATED run (no
      pgrep before rerun; inflated T=1024 baselines; the parked NKV=4
      anomaly was the tell). Round 3 (drift-robust interleaved,
      pgrep-guarded): slopes 0.155/0.183/0.193/0.205 across NKV
      2/4/8/16 — flat, which is exactly right because that arm's
      total traffic (NQ×HD×T) is NKV-invariant: the kernel is
      DEVICE-TRAFFIC-BOUND, moving ~67 MB at 80–105 GB/s (~35–40 % of
      the M4 roof; consistent with 50 % line utilization of half-row
      slices). At ornith's NKV=2, 7/8 of that traffic is redundant
      GQA re-read. Fix unchanged in direction, sobered in size:
      KV-head-shared threadgroups cut traffic 8× → expected slope cut
      ~5–6×; full sharing required (pairing halves traffic only).
      Bench discipline now baked in: pgrep guard + interleaved
      3-round grid, global best per point.
- [x] **V4: KV-head-shared partial kernel — ACCEPTED ON THE TWIN
      (2026-09-01 ~18:30, df74636).** Per-simdgroup-per-head layout
      (peer-reviewed spec); shared path takes the full 64-chunk budget
      (first wall: TG count) and stages positions in 4-blocks (second
      wall: barrier cadence). Local M4 slopes: int8 0.207→0.069, fp16
      0.155→0.059 µs/pos. **Mini twin: warm attn_layer_kv slope
      +8.0 → +2.81 ms/1000 ctx (2.84×); deep card turns 35.9 → 33.2 s
      (~10 ms/token at ctx 1900); rig 38.02 (≤ baseline); outputs
      sane; rig digest incidentally byte-identical.** Live trial now
      runs the knob (PID 48356); default flip staged behind the
      golden-baseline ceremony below. Sign-off: Davor, 2026-09-01
      ("Proceed ;)").
- [x] **V4.1 (golf, bitwise-safe, no sign-off): fold the dequant
      index divides via KV function constants** — LANDED ca7881e,
      **ACCEPTED on twin 2026-09-01**. The shared partial had run fully
      generic (shape AND KV format as runtime buffer values); it now
      builds a shape+format-specialized PSO (FC 60-63 + new 96-99,
      cached per key; 80-83 were taken by fused.metal). Bitwise arms
      pin specialized ≡ generic byte-for-byte (fp16/int8/int4) and that
      the specialized PSO engages. Local: kvsh int8 slope 0.069 →
      0.054-0.058 µs/pos, int8-vs-fp16 gap closed (drift-guarded,
      base arms stable control). Mini twin (fresh-server arms, both
      kvshared): rig 37.83→37.77 sd 0.44 (neutral), digest exact,
      warm attn_layer_kv ladder 4.985/6.624/9.099/10.045 →
      4.799/6.147/8.264/8.878 ms/token over ctx 48→1912 — **slope
      2.71 → 2.19 ms/1k ctx (−19 %), −1.17 ms/token at depth**. Live
      trial now runs ca7881e (PID 48588); rollback .v4 + bundle
      .v4-bak staged.
- [ ] **V4.2: default flip + golden baseline** — kvShared becomes the
      code default for applicable shapes, knob deleted, golden
      baseline captured (rig + ~2k prompt, on the mini) per the
      standing T5 note; v10's Q2 closes with a pointer here.
- [ ] **V5 (unbundled, later, own sign-off): KV row-layout reorder**
      — the clean bench shows ~50 % line utilization (35–40 % of
      roof); a layout so one TG's walk touches full lines may hide a
      further ~2× behind V4's 8×. Layout change on top of a reduction
      change — deliberately NOT bundled with V4.
