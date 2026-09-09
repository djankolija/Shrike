# v18 implementation plan: the quiet host

Companion to [v18-quiet-host.md](v18-quiet-host.md). Checkboxes here are the only
status tracking. Branch: `perf/v18-quiet-host` off `main` at `c503f7c`.

Protocol per task: the four gates per commit; the golden byte-identical on both
boxes; deploy to the mini (binary plus the `*.bundle` directories); the arms rig, two
production lifetimes per shape, read against the previous task's arms; the rows the
task pre-registered, moved or not, recorded in the design document's task record;
ThreadSanitizer once at the close.

Order after T2.0 (Davor's ruling, 2026-09-09): Task 4, then Task 3, then Task 5
decided on Task 3's arms; Task 2 skipped as a performance task, its mechanism folded
into T5.1. After Task 3 (Davor's ruling, 2026-09-09): Task 6, the walls, runs next,
and the fold's ruling follows it on the boundary costs Task 6 re-measures.

## Step zero: the board priced once (no runtime code)

- [x] **S0.1 The miss profile by layer** (A9 Q2): DONE 2026-09-09, pool mode (no
      ring) on the six T4 traces via v14's `step0-layer-misses.py`; U-shaped by
      depth, layers 0-3 22 % of misses with no lead, 30-39 25 %; recorded in A2/A9.
- [x] **S0.2 Routing locality** (A9 Q1): DONE 2026-09-09 (subagent, script and JSON
      in the scratchpad); last-occurrence recall 0.38, union of three 0.54 at 15
      predicted, identity 3× position at layers 0-3; recorded in A9.
- [x] **S0.3 The width sweep** (A0): DONE 2026-09-09, blocked beyond width eight (the
      captures hold the probe's top-8 only); coverage at eight 0.43 to 0.46 at
      distance one, 0.34 to 0.37 at two; a wider capture goes to v20's step zero.
- [x] **S0.4 The slot split and the policy** (A1, A2): DONE 2026-09-09 for the
      policies (seven, including Belady's bound at 2.5 to 2.8× below every online
      policy; SLRU and LRU 8 to 10 % better than aging-LFU on the longer shapes, none
      on the 300); the per-layer slot split deferred to v20's step zero (the replay
      takes one slot count); recorded in A2.
- [x] **S0.5 Experts per small dispatch and dispatches per GDN layer** (C1, D1): DONE
      2026-09-09; the hit dispatch at the roof (6.7 experts each), the fixup 1.75×;
      about 820 dispatches per token (11 per GDN layer, 14 per KV layer, 7 per
      speculative command on four encoders), consistent with a 12 to 14 µs wall; six
      class-1 merge candidates worth about 27 % of the dispatches; recorded in C1
      and D1.
- [x] **S0.6 The compression entropy** (H2): DONE 2026-09-09 (subagent); 11.8 to
      12.3 % fewer bytes on a typical layer and the head at an order-0 code, more on
      the shallow layers; the scales and biases are the second-order target; recorded
      in H2.
- [ ] **S0.7 The prefill outlier**: the 20:07 request's log against its neighbours;
      recorded in G.
- [x] **S0.8 The calibration probe** (B5): DONE 2026-09-09 on the mini through oMLX
      (Qwen3-14B, fp16 KV): the reference scan runs at 20 ns per KB against the 16 ns
      roof; ours at 200. The gap is our kernel; the scan rewrite has a demonstrated
      target and a prize of about 14 ms per token at 7k context; recorded in B5,
      the sequencing question raised in section 5.
- [x] **S0.9 The GPU's clock after an idle window** (I3): DONE 2026-09-09, closed;
      fixed-work kernels after a long idle run within 1 to 3 % of their busy-stretch
      duration; the fixup's growth with idle is its own miss count; recorded in I3.
- [x] **S0.10 The pricing recorded**: DONE 2026-09-09; every priced avenue updated
      in [v18-avenues.md](v18-avenues.md); v19 named in its section 5 as a
      recommendation pending Davor's ruling (the scan rewrite first, the SSD
      mechanism after) with the class-2-last alternative stated. Production restored
      on the mini after the two runs: the rebuilt `c503f7c` binary golden-identical on
      both profiles, the server relaunched on its production line and answering.

## Task 1: the hits in the speculative command (C5)

- [x] **T1.1 Read**: DONE 2026-09-09; the statement list is in the design doc's Task
      1 section. The speculative phase-1 kernel already skips a miss position per
      row by the classifier's sentinel, so the change is the classifier's grid rule
      and the host's deletion, no kernel change.
- [x] **T1.2 Build**: DONE 2026-09-09; the classifier publishes phase 1's full grid
      on every layer and zeroes only phase 2 and the tail on a miss layer; the hit
      split, its command buffer, its context fields, its four stats and their four
      runner-line fields deleted; the fixup takes `missesOnly` and always the reused
      argument buffer; the adopted path and the cross-check unchanged. Release build
      zero warnings, strict lint zero violations, links clean.
- [x] **T1.3 Tests**: DONE 2026-09-09; `speculativeDispatchArgumentsFollowResidency`
      expects phase 1 full and phase 2 and tail zero on a miss layer;
      `productionRoutedPipelineSpecHitsAndFixupMissesMatchReference` runs the hits
      through the speculative kernel at the full grid on a pool with the sentinel for
      the misses, checks the miss rows stayed untouched, runs the misses through the
      fixup's subset kernel and the reduce, and asserts the output bit-identical to
      the all-host pipeline and within tolerance of the reference. Both suites pass.
- [x] **T1.4 Gates and golden**: DONE 2026-09-09; the four gates (release zero
      warnings, lint zero, links, 1,234 tests in 170 suites in 202 s); the local
      golden identical on both profiles.
- [x] **T1.5 Deploy and arms**: DONE 2026-09-09; deployed (e6241a16f55feede), the
      mini's golden identical; two lifetimes per shape: the hit role and the submit
      gap to zero, the speculative command up by the hits' phase 1, the window up by
      the same amount, **the wall flat within the drift on all three shapes** (a
      measured null: the hit command sat in the read's shadow); the task record in
      the design doc. Kept and committed on Davor's ruling (2026-09-09).

## Task 2: the fixup as a speculative command (C6)

- [x] **T2.0 Price** (added 2026-09-09): DONE from Task 1's arms, no runtime code;
      the pre-issue chain on a miss layer is 61 of the word's visibility, at most 25
      of the host's plan, pin and submit, 26 of the reader's hand-off, then the flight
      and the 157 wake; C6 can touch only the 25, at most 0.34 ms per token, about
      0.2, under the drift; the record in the design doc's Task 2 section, the word to
      A8, the hand-off to A5, the join's order to C7, the shadow ledger on the board.
      **Davor's ruling (2026-09-09): skipped as a performance task.** T2.1 to T2.5
      below are not scheduled; they stay as the agreed-cell mechanism's step list for
      the fold's design note (T5.1).
- [ ] **T2.1 Read**: the ring's cell leases and the index swap (`PreadExpertStreamer`,
      `ExpertPrefetchRing`), the plan's swap and victim path, the fixup's encode; the
      agreed-cell contract written (miss i into cell i; the fallback when cells run
      out).
- [ ] **T2.2 Build**: the fixup encoded before the route with an indirect phase 1 over
      the classifier's miss list, the reduce, the residual, behind the event wait; the
      host's on-word path reduced to the reads' issue into agreed cells; the plan
      moved to the next wake; the fallback to the host-built fixup when a layer's
      misses exceed its free cells, counted.
- [ ] **T2.3 Tests**: the agreed-cell contract; the fallback; the plan-after ordering
      under the cache lock (the residency publish stays one release store per cell).
- [ ] **T2.4 Gates and golden.**
- [ ] **T2.5 Deploy and arms**: the host's path fields off the path, the fixup's
      commit latency gone, the window's latency term; misses per token recorded for
      drift; the task record.

## Task 3: one command per layer (prices E1)

- [x] **T3.1 Read**: DONE 2026-09-09; the served model folds attention and tail into
      one command already, the speculative command is the second, so the task is
      their merge with the fixup still separate; the assumption the task rests on
      (the host sees the classifier's word mid-command, not at the command's end)
      was uncited in the right direction and is now measured by
      `MidCommandVisibilityTests`: the word seen 42 to 45 µs after the command's
      GPU start, 29 to 31 ms before its end, three runs; the statement list and the
      rows in the design doc's Task 3 section. **Davor's go on T3.2 pending.**
- [x] **T3.2 Build**: DONE 2026-09-09; `encodeSpeculativeRouted` encodes into a given
      command and follows the tail in `attnCB` when the tail is folded (every layer of
      the served model), the split-tail path keeping its separate command;
      `HeldLayerCommands.specCB` optional with `routedCB` the carrier; the all-hit
      pending command is the merged command with no role of its own, the miss path's
      pending carries no separate speculative command; the merged command recorded
      once as `layer_linear` / `layer_kv`; `decode-rows.py`'s window regex takes the
      new names; no knob. The fixup stays a separate command (Task 2 skipped). The
      word wake's assumption measured by the probe (T3.1), not by the build.
- [x] **T3.3 Gates and golden**: DONE 2026-09-09; the four gates (the release build
      zero warnings, lint zero in 212 files, links clean, 1,243 tests in 171 suites in
      204 s, the probe among them); the golden identical on both profiles on both
      boxes (the mini on the deployed 80654748c95eeb44 with the server stopped).
- [x] **T3.4 Deploy and arms**: DONE 2026-09-09; deployed (80654748c95eeb44), two
      lifetimes per shape against Task 4's: the forty tail-to-speculative boundaries
      gone (about 1.4 ms of gaps) and the merged commands up by about 1.65 (the drain
      now an encoder boundary inside the command), `wait_ms` flat and the wake
      counter at zero (the word lands at the classifier inside the command); the
      wall +0.4 to +0.8 % on the 300 and the 1k's clean lifetimes, about 0.3 to 0.5
      ms per token, a third of the modelled 0.9; kept as simpler and non-negative;
      the fold's remaining prize re-examined at about 0.85 ms; the record in the
      design doc. **Davor's ruling on Task 5 pending.**

## Task 4: the sampler feeds the next embed (E2)

- [x] **T4.1 Read**: DONE 2026-09-09; the boundary is three synchronous command
      buffers (the head, the sample, the embed) with the token crossing by a
      `waitUntilCompleted` and a shared-memory load and entering the embed as a
      `setBytes` constant; the stop token is never embedded; the seed is a host
      constant known ahead; the penalty is host-side in place. The design changes:
      the stop check runs on the token's word (about 63 µs after the sample) and
      layer 0 is committed only when there is no stop, instead of one pass late with
      an extra pass to cancel (a pass mutates the GDN state in place, so cancelling
      it would need a state undo). The statement list S1 to S7 and the rows in the
      design doc's Task 4 section. **Davor's go on T4.2 pending.**
- [x] **T4.2 Build**: DONE 2026-09-09 (the stop check on the word, not one pass
      late; see T4.1); `BoundaryLogitProducer` with the two-step `produce` and
      `awaitBoundaryToken`; the runner's `emitBoundary` (the final norm, the lm_head
      GEMV, the caller's sampler and the word-fed embed in one command, the sentinel
      in the word before the commit), `holdLayerZero` after the cursor advances, the
      spin with the one-second fallback (`boundary_wake_fallbacks` on the runner
      line), the previous boundary waited on and recorded as `head_logits` at the
      end of the next pass; the two embed encoders' `tokenBuffer:` overloads (no
      Metal change); the loop's path chosen once per generation with the fallbacks
      (a penalty other than 1.0, the fused greedy head, the first token after
      prefill sampled as before).
- [x] **T4.3 Tests**: DONE 2026-09-09; six in `RawCompletionLoopTests+Boundary.swift`
      on a scripted boundary producer running the real sampler (the same tokens,
      deltas, reason, cursor and history as the synchronous path; every pass after
      the first continued; the stop token without another pass; max tokens; a stop
      string; the penalty fallback), three of them red with the path switched off
      and three invariants of both paths; two encoder tests (the buffer-fed lookup
      bit-identical to the constant-fed one, both kernels), red before the
      overloads existed.
- [x] **T4.4 Gates and golden**: DONE 2026-09-09; the four gates (the release build
      zero warnings, lint zero in 212 files, links clean, 1,242 tests in 170 suites in
      203 s); the local golden identical on both profiles.
- [x] **T4.5 Deploy and arms**: DONE 2026-09-09; deployed (6142e12205c5d3eb), the
      mini's golden identical on both profiles; two lifetimes per shape against Task
      1's arms: the three boundary gaps (0.83 to 0.90 ms per token) to one of 0.25 to
      0.27, the sample and embed roles folded into `head_logits` (+0.17 to 0.19),
      `loop_sample_ms` now the word's wait through the head, no wake fallbacks, the
      misses per token identical, the card's answer identical; the 300 +0.9 %, the
      card and the 1k mixed by a slow-drive box state (`prefetch_late` above zero),
      resolved by a same-box interleaved A/B on the 1k, four lifetimes each: Task 4
      wins every pair, +1.4 % on the clean lifetimes, about 0.8 ms per token; the
      task record in the design doc.

## Task 5: the fold (K)

- [ ] **T5.1 Design note**: the token's command layout, the two-in-flight protocol,
      the cancel path, the error surfacing per layer; reviewed before the build.
- [ ] **T5.2 Build**: one command per token, two in flight.
- [ ] **T5.3 Tests**: the cancel with two in flight; a failed read naming its layer.
- [ ] **T5.4 Gates and golden.**
- [ ] **T5.5 Deploy and arms**: the token against the v17 close on all three shapes
      and the turn rig's pair.

## Task 6: the walls (D1)

- [x] **T6.0 Price the wall by its kind**: DONE 2026-09-09, landed; the speculative
      command's seven dispatches on one encoder and the fixup's three on one, six
      kernel wrappers with `encoder:` variants, the one-encoder pipeline test
      bit-identical; the four gates (1,244 tests in 171 suites), the golden identical
      on both boxes (the mini on 83eddc36bd722e57); the arms against Task 3's: the
      GPU's role time down 3.3 ms per token over 147 encoder boundaries, **about 22
      µs a boundary**, the wall +1.6 to +2.8 % on the 300 and the 1k, about 1.0 to
      1.7 ms per token; the merges' 12 µs pricing stands for dispatch boundaries.
- [x] **T6.0b The boundary command on one encoder**: DONE 2026-09-09, landed; nine
      encoders to one, the sampler, four sampling kernels and two embed encoders with
      `encoder:` variants, the protocol's sample closure on the encoder, the
      one-encoder sampler-plus-embed test bit-identical; the gates (1,245 tests in
      172 suites), the golden identical on both boxes (the mini on e0f8bd17dc8ecbb0);
      the arms against T6.0's: `head_logits` down 0.06 to 0.09 ms per token, about 10
      µs a boundary between small kernels (half the speculative command's 22), the
      wall inside the drift; kept as simpler and non-negative.
- [ ] **T6.1 The shared gate and up GEMVs as one grid** (the cleanest; 40 walls a
      token, 26.5 on the path): the read, the merged kernel, the bitwise arm against
      the two it replaces, the gates, the golden, the arms.
- [ ] **T6.2 The scalar gate into that dispatch** (40; 26.5 on the path): the same
      steps.
- [ ] **T6.3 Speculative phase 2 plus its residual** (40; 26.5 on the path; the
      zero-grid miss behaviour preserved): the same steps.
- [ ] **T6.4 The top-k select plus the classifier** (40, every layer on the path):
      the same steps.
- [ ] **T6.5 Conv plus qk norm** (30 GDN layers, on the path): the same steps.
- [ ] **T6.6 The input norm into the in-projection** (30; the weakest, last): the
      same steps, or dropped if T6.5's arms say the GDN walls are not what D1
      counted.
- [ ] **T6.7 The record**: the task record in the design doc, D1 updated with what
      each wall cost, the fold re-priced for its ruling.

## Close

- [ ] ThreadSanitizer on the final tree (the suppressions file unchanged).
- [ ] The whole-branch review by a fresh reviewer; the fold into owning commits.
- [ ] The design document's closing block: the ledger at the close beside the v17
      close, what the chapter settled, what remains.
- [ ] The architecture document brought to the final tree (the decode path's
      command structure, the routed stage's steps, the token boundary).
- [ ] Merge to `main` on Davor's go; the branch deleted; the deploy restored.
