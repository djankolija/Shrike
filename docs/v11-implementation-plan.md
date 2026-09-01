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
- [ ] **V3: twin verdict** — same-session three-point: current default
      vs knob-on, rig + full ladder; accept on warm slope ≤ +1.5 ms/1k
      and wall improvement at depth; revert on anything else. Output
      sanity read on real prompts (digest is expected to differ — the
      knob A/B was signed off as a264b22-class).
- [ ] **V4: default flip + golden baseline** — knob becomes default,
      knob deleted, CLAUDE.md env notes untouched (no new env), fresh
      golden baseline captured (rig + ~2k prompt) per the standing T5
      note; v10's Q2 closes with a pointer here.
