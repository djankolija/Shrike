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
- [x] **T2: GDN mega-merge (C1) — RAN AND REVERTED 2026-09-01 (peer
      session built d89d172 on perf/t2-gdn-merge; twin verdict by the
      main session).** The implementation is CORRECT — bitwise arms
      pass on all four shapes, digest exact everywhere (including a
      subtle find: Metal fast-math elides half round-trips on
      register-resident values; the fused kernel needs a volatile
      thread slot — memorialized in project memory) — but the twin
      says SLOWER: rig wait 40.75 sd 0.38 vs 38.38/38.59 on the
      same-session sibling arms (+2.2 ms/token, ~6σ), attn_layer_linear
      +2 ms on cards. The boundary law extends: the three kernels
      already pipelined free inside the serial encoder, and the merge
      paid occupancy/register pressure instead. Reverted from the
      branch per accept-on-twin protocol; the commit survives on
      perf/t2-gdn-merge for a future occupancy-tuned attempt (M4-class
      behavior unmeasured). ⚠ Deploy-trap re-confirmed the hard way:
      the new .metal kernel 500'd the server until the resource
      bundles shipped alongside the binary. **Closed 2026-09-23 with no tt entry, by the owner's ruling: the occupancy-tuned attempt is retired, since v18 priced the wall between small kernels at about 3 µs (v18-implementation-plan.md:236-246) and its floor rule declined the similar merges T6.2 and T6.4 to T6.6, conv plus qk norm from this chain among them, leaving roughly 0.2 ms per token (an estimate, grade C), under the rig's noise; the branch conflicts with main. The branch perf/t2-gdn-merge is deleted under SHRIKE-14.**
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
- [x] **T5: settle — DONE f169f51, 2026-09-02.** All five perf
      winners are code defaults (pool/speculative/event/immediate/spin;
      env vars remain as explicit A/B overrides; spin's thermal gate
      lifted by Davor — the live deployment's bursty duty cycle was
      the trial). S3b machinery DELETED whole (git history preserves
      it; CrossQueueSharedEventTests re-chartered to the event IO sync
      it still pins). CLAUDE.md env paragraph shrunk to stats-only.
      Golden baselines captured V4.2 and used as gates since. Mini is
      a clean prod box: current binaries + bundles + models +
      baselines only — every staged rollback binary, nvmai-retired,
      the log rotations, and the 29 GB gputrace deleted per Davor's
      ruling (docs and git carry the history; ~235 GB freed). Bare
      launch digest 494bab3edb62 exact on both machines; golden
      --check identical on both; mini prod line is two stats vars.
      Handoff retired with this entry as the final ledger. FINAL
      CHAPTER ARITHMETIC: rig 111.6 → 38.1 ms/token (2.9×); depth
      tax +8.0 → +2.07 ms/1k ctx (3.9×); output byte-identical
      throughout.

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
- [x] **P2: CLOSED BY SUPERSESSION 2026-09-02** — the question ("where
      do the chain's ms go") was answered without the traces: the role
      split (c1db837) named the family, v11 V3a named the mechanism
      (traffic-bound GQA re-read), V4/V4.1/V5 fixed it. The parked
      gputrace bundles were deleted with the mini cleanup; recapture
      is one env var (SHRIKE_GPU_CAPTURE_DIR) if a per-kernel question
      ever returns.
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
      prefill's 33 ms/token. **Superseded: by its own verdict below (:190-205), v14 plan:855-859 and v15 plan:17 (noted 2026-09-23).**
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
      overhead stands (probe-measured) but production never pays it.** **Superseded: by the verdict at :190-205, v14 plan:855-859 and v15 plan:17 (noted 2026-09-23).**
      **P3 FOLLOW-ON RAN 2026-09-04 (Davor's SSD-fetch ideas, probed
      to exhaustion, all NULL on the mini).** Five standalone C probes
      on the mini's Apple-fabric SSD (APPLE SSD AP0512Q), 1.77 MB
      experts, F_NOCACHE, mirroring expert_io.c; archived at
      `~/.claude/handoffs/archive/shrike-ssd-split-probe/`.
      (1) Split one expert across 2/4/8 concurrent preads: NULL
      (0.77 ms p50 either way). The drive's rate is set by BYTES IN
      FLIGHT, not request count. (2) Split into 2..16 separate files
      read concurrently: NULL on the mini (flat ~0.64 ms). (3) Pad the
      read so the drive streams faster, discard the pad: NULL and
      worse. A page-tail watcher shows the payload's last page lands no
      sooner (0.77 alone vs 0.90 / 0.95 / 1.00 padded to
      3.5 / 7.1 / 14.2 MB) while occupancy balloons to 1.3 / 2.3 /
      4.2 ms; a single blocking pread cannot release partial data and
      macOS cannot cancel it. Concurrent pads and keep-hot background
      streams only slow the payload (shared aggregate bandwidth); mmap
      page-in is 2-3x worse. (4) Serial-ramp: serial small reads do NOT
      speed up over a long run (per-read p50 flat across all deciles at
      every size), confirming bytes-in-flight with no sustained-activity
      ramp. (5) A real but separate effect on the mini: a 5 ms idle gap
      doubles a 64 KB read (0.12 to 0.28 ms), with only a one-time
      cold-start warm-up (first ~100-200 reads, 0.22 to 0.12), no
      progressive ramp. The CPU-vs-drive discriminator (busy-spin gap
      == usleep gap, so the cause is I/O idle, not CPU P-state) was run
      on the external SN850X only (the mini arm was superseded by a
      lookup); with `disksleep=0` on the mini and NVMeFix documenting
      that Apple controllers use their own low-power path (LPSR) rather
      than generic APST, the cause is drive / PCIe-link power
      management, not OS disk sleep and not the kernel scheduler.
      DRIVE-DEPENDENT: the dev MacBook's external WD_BLACK SN850X DOES
      scale a single 1.77 MB read with concurrency (16-way =
      0.44 ms / 3.96 GB/s vs 0.58 / 3.0, -24 %); the external nearly
      saturates on one read, the mini needs many experts in flight. The
      mini is the deploy target, so this is not a production lever.
      **Verdict: the fetch-speed lever is shut on the deploy target.**
      The single 1.77 MB miss is at its floor (~0.77 ms); the only
      throughput lever is more experts in flight, which at decode
      requires next-layer miss prediction. That prediction lever is
      independently CLOSED (architecture.md distance experiment: the
      hidden state drifts 20-27 % per layer, k>=3 catches at most a
      third of misses at 8-28 wasted fetches each, cannot clear the
      +10 % bar), and every wasted prefetch now measurably steals
      bytes-in-flight from the real miss. So fetch-speed (shut) and
      predict-ahead (shut) both dead-end into the hardware. Davor's
      ruling 2026-09-04: not fighting the controller (no kext tweaking).
      Surviving miss levers unchanged: cost-side (keep the miss path
      warm to dodge the ~2x idle penalty, small) and policy-side (reduce
      miss COUNT via residency / cache). Prefill already exploits
      bytes-in-flight (batches whole tiles, near the aggregate ceiling);
      decode has ~1 miss per layer, so nothing to batch.

## Queued after T5 (Davor, 2026-08-31 — sequenced behind the original tasks)

- [x] **Q1: between-token host overhead** (~5–7 ms/token for every request:
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
      server traffic), detokenize/emit, per-token async hops. **Closed 2026-09-23 with no tt entry, by the owner's ruling: the fused greedy head stays not scheduled, since Pi's config sets no temperature, so its requests get Shrike's default 0.6 (`Sampler.swift:7`) and would never take a greedy path; read from config, not observed on the wire.**
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
      correctness-adjacent (dialect re-render strips reasoning).** **Superseded: by the owner's ruling at :260-265 (noted 2026-09-23).**
      **ROOT CAUSE PINNED 2026-09-01 morning (kv_tail diagnostic
      3bff0d4, ids decoded against the tokenizer): the live-vs-settled
      divergence on ornith/thinking-off is the EMPTY THINK BLOCK —
      the generation prompt prefills `<think>\\n\\n</think>\\n\\n`
      (ids 248068,271,248069,271; that block IS the template's
      thinking-suppression mechanism), and the settled history render
      strips it. Divergence sits at the START of the answer region,
      so every later position shifts and the re-prefill of that
      region is genuinely required — v6 is behaving as designed.
      Cost structure: steady state = restore prior snapshot + rebuild
      the new turn only (O(answer), a few s, background); the cliff
      is FIRST-settle on a server with no snapshot (full rebuild,
      44 s at 1900 ctx) and any join landing before a settle
      finishes. The boundary-snapshot amendment therefore targets
      first-settle; a dialect-level alternative (render thinking-off
      without the prefilled block) would zero the whole cost but
      changes what the model sees at generation time — Davor's call.**
      **RESOLVED: Davor ruled the settle a compensation layer for
      training-distribution mismatch — unnecessary for ornith-class
      models. Fix spun out as its own work item:
      [v6.1-reasoning-retention.md](v6.1-reasoning-retention.md)
      (reasoning-retention policy, as-generated default, Harmony
      forces stripped). Q1 closes when v6.1's R4 verdict lands.** **Ticked 2026-09-23:** v6.1's R4 verdict landed, CONFIRMED 2026-09-01 (v6.1-implementation-plan.md:31).
- [x] **Q2 CLOSED 2026-09-01 (v11 V4+V4.1, default via V4.2 b7aa00b): depth tax +8.0 → +2.19 ms/1k ctx; verdict trail in docs/v11-implementation-plan.md.** Original entry: **Q2: context-depth tax — now sized as a genuine anomaly
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
      **CHUNK-COUNT PROBE NULL (SHRIKE_ATTN_FULL_CHUNKS knob 6e783ea,
      A/B 16 vs 64 chunks with full ladders, Davor's digest sign-off):
      per-token attn_layer_kv identical-to-slightly-worse at every
      depth (warm t4 20.10 vs 20.63; slope unchanged; rig digest
      incidentally byte-identical — argmax robust to the combine-order
      shift on this prompt). 4× sequence parallelism bought ZERO →
      the geometry is acquitted and the INNER LOOP convicted: the
      per-position work inside the split-KV partial kernel (~0.4 µs/
      position/layer — dequant/barrier/softmax-chain) is the redesign
      target. Knob KEPT deliberately as the tuning surface for that
      redesign (delete at settle if unused — deviation from the T3
      delete-null-knobs precedent, stated reason).** **Done: the knob was deleted in v17 (v17-consolidation.md:67) (noted 2026-09-23).**

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
      overriding the gate.** (superseded: gated negative at :329-331; class 3 closed, v18-avenues.md:1162) Original design: env-gated: on a miss whose
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
