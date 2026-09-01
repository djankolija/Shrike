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
      **BACKEND A/B RAN 2026-09-01 ~04:30 — the dormant
      SHRIKE_EXPERT_IO_BACKEND=metal path (MetalExpertReader, batches
      per plan, exactly P3's winning shape) is NOT the vehicle:
      same-binary twin vs pread (b0f775b): rig wait 43.29 sd 9.2 % vs
      38.68 sd 2.2 % — slower AND noisy at rig shape (digest exact) —
      and the server DIED silently mid-1900-token prefill (no crash
      report, no fatal in log; OOM/jetsam suspected via the staging
      path; the header's never-run-A/B caution was justified twice
      over). Log: /tmp/ornith.log.metal-backend-ab. Verdict: pread
      stays production; the surviving lever is cheapening/batching the
      PREAD path itself — the ~1.2 ms/read gap between the streamer's
      2.0 ms p50 and raw pread's 0.79 (thread-pool dispatch + K12
      critical section + spin-core CPU contention are the suspects).
      **Follow-ups the same night, ending in a CORRECTION: (a)
      contention EXONERATED — raw pread under live decode load is
      unchanged (0.774 vs 0.789 idle); (b) worker-QoS knob (58298d2)
      A/B'd NULL (rig 38.50/38.43, cards 69.6/69.2, p50 identical)
      and reverted per the T3 precedent; (c) the histogram behind
      expert_load percentiles is POWER-OF-TWO UPPER BOUNDS
      (loadLatencyPercentile, PreadExpertStreamer.swift:93) — tonight's
      p50=1.000 means (0.5, 1.0] ms, which BRACKETS raw pread's 0.79:
      the production pread path has ≈ zero per-read software overhead
      left, and the era-2.0 figure was a coarser bucket on the old
      config. VERDICT REVISED: the miss path's exposed cost is drive
      physics + weak overlap (~30 % hidden), NOT submission overhead;
      the surviving lever is HIDING — deeper effective queue depth
      across in-flight layers (drive gives +48 % aggregate at QD4) —
      a scheduling-structure change, smaller and harder than the
      original ~3–4 ms batching estimate. MTLIO's +0.54 ms single-load
      overhead stands (probe-measured) but production never pays it.**

## Queued after T5 (Davor, 2026-08-31 — sequenced behind the original tasks)

- [ ] **Q1: between-token host overhead** (~5–7 ms/token for every request:
      decode window vs body_ms; GPU busy 77% of window on real turns vs 93%
      rig). First probe: stats-off A/B (RUNNER/KERNEL_STATS may tax the
      observed); then the per-token emit/detokenize/async-hop loop.
      Compare on WALL, never wait_ms.
      **FIRST PROBE RAN 2026-09-01: stats exonerated** — same-night
      fresh-server twins, wall 9.367 sd 0.141 (stats on) vs 9.292
      sd 0.122 (off), Δ 0.44 ms/token ≈ noise, digest exact both.
      Same arms re-bound the prize: with the prompt cache warm,
      wall − body − head ≈ **~10 ms/token of between-token host time
      on the rig** (54.8 wall vs 39.9 body + ~4.9 head). Narrowed
      suspects: host greedy/sampling over the 248,320 vocab (the fused
      greedy head the server never uses — head_fused_ms=0.000 for all
      server traffic), detokenize/emit, per-token async hops.
      **LOOP EXONERATED, MECHANISM FOUND (loop timers b0f775b,
      deployed): measured loop_sample 0.43 / loop_detok 0.002 /
      loop_progress 0.001 / loop_produce 45.18 (= body 39.75 + head
      4.85 + 0.58 async entry) — the token loop accounts for 45.6 of
      54.7 ms/token. The ~9 ms/token is PER-REQUEST work outside the
      loop: the v6 dialect-normalized-cache settle. Post-completion KV
      normalization (KVRewrite settle/dropEmission) re-prefills the
      conversation when no prefix snapshot exists (log: settle_reset
      reason=no_prefix_snapshot), and the NEXT request joins the
      pending rewrite (arbitrate decision=join). Measured: warm card
      with a 7-token suffix prefill took 67 s wall (~44 s = absorbed
      join of the prior settle); rig ~1 s/request. Later settles of
      the same prefix restore snapshots cheaply (settle_restore).
      FIX CANDIDATES — design question for Davor, spec is
      [v6-dialect-normalized-cache.md](v6-dialect-normalized-cache.md):
      (a) capture the boundary snapshot at decode START (≈ one ~85 MB
      state copy/request) so first-settle restores instead of
      resetting; (b) decouple the join so the next request does not
      stall behind normalization. Not touched overnight — spec'd,
      correctness-adjacent (dialect re-render strips reasoning).**
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
      (cold arms +5–24 ms, hitD 0.82 vs 0.90 at t1). Ladder logs:
      scratchpad ladder-runner.txt / ladder-roles.txt (session
      6dd4e253), mini /tmp/ornith.log (Server A PID 45374).**
      **FAMILY NAMED (role split c1db837, ladder rerun on fresh
      server): the 30 linear layers are FLAT (+0.09 ms/1k, 14.50 →
      14.66) — the ENTIRE tax is the 10 KV layers: attn_layer_kv
      5.24 → 20.07 ms/token over ctx 48 → 1900 = +8.0 ms/1000 ctx ≈
      65× the per-layer KV-read roofline (2.0 ms vs 0.031 at 1900;
      ~0.43 µs/position/layer — sequence-serial signature). Target:
      the gated full-attention decode path
      (encodeGatedFullAttentionDecode → its kernels). Fix shape:
      flash-decoding/split-K over context; prize at real depths
      (2–3k ctx) >10 ms/token — the largest remaining decode lever.
      Per-kernel confirmation available via Xcode replay of the deep
      capture: mini /tmp/gputrace/shrike-decode-1788221522.gputrace
      (17 GB, ctx 1900; a .gputrace stores commands + resources, NOT
      timings — profiling happens at replay). Side observation: a
      fresh server whose FIRST traffic is card-shaped runs card hitD
      0.94–0.97 (vs 0.82–0.94 after rig warmup) — cache trajectories
      are workload-seeded.**

## Quality-trading experiments (lane opened by Davor, 2026-09-01)

His rationale: int4 quantization is already an accepted quality trade,
and no public data exists for top-N-of-8 sensitivity on this model/
quant/hardware — so measure it. Everything here alters sampled output:
per-experiment sign-off stands, and no default flip without a quality-
battery verdict.

- [x] **E0: router rank-mass instrumentation** — LANDED 04ea4ad,
      deployed 2026-09-01 (digest exact, wait 38.54 ≡ pre-E0 38.65 —
      free). **Measured decode rank mass: rig
      0.183/0.146/0.130/0.119/0.112/0.107/0.103/0.100 (deterministic
      to 4 decimals across requests); fresh card turn-4 (1900 ctx)
      0.224/0.166/0.136/0.117/0.103/0.092/0.084/0.078. GATE VERDICT:
      NEGATIVE for E1 as premised — ranks 7–8 carry 8–10 % each
      (16 % together on real shape), not the hypothesized 2–3 %.**
      This router spreads mass unusually evenly (rank 1 only 18–22 %).
- [ ] **E1: drop-bottom-miss experiment — GATED NEGATIVE by E0's
      measurement (see above); do not build without Davor explicitly
      overriding the gate.** Original design: env-gated: on a miss whose
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
