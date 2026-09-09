# v18 implementation plan: the quiet host

Companion to [v18-quiet-host.md](v18-quiet-host.md). Checkboxes here are the only
status tracking. Branch: `perf/v18-quiet-host` off `main` at `c503f7c`.

Protocol per task: the four gates per commit; the golden byte-identical on both
boxes; deploy to the mini (binary plus the `*.bundle` directories); the arms rig, two
production lifetimes per shape, read against the previous task's arms; the rows the
task pre-registered, moved or not, recorded in the design document's task record;
ThreadSanitizer once at the close.

## Step zero: the board priced once (no runtime code)

- [x] **S0.1 The miss profile by layer** (A9 Q2): DONE 2026-09-09, pool mode (no
      ring) on the six T4 traces via v14's `step0-layer-misses.py`; U-shaped by
      depth, layers 0-3 22 % of misses with no lead, 30-39 25 %; recorded in A2/A9.
- [x] **S0.2 Routing locality** (A9 Q1): DONE 2026-09-09 (subagent, script and JSON
      in the scratchpad); last-occurrence recall 0.38, union of three 0.54 at 15
      predicted, identity 3× position at layers 0-3; recorded in A9.
- [x] **S0.3 The width sweep** (A0): DONE 2026-09-09, blocked beyond width eight (the
      captures hold the probe's top-8 only); coverage at eight 0.43 to 0.46 at
      distance one, 0.34 to 0.37 at two; a wider capture goes to v19's step zero.
- [x] **S0.4 The slot split and the policy** (A1, A2): DONE 2026-09-09 for the
      policies (seven, including Belady's bound at 2.5 to 2.8× below every online
      policy; SLRU and LRU 8 to 10 % better than aging-LFU on the longer shapes, none
      on the 300); the per-layer slot split deferred to v19's step zero (the replay
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

- [ ] **T3.1 Read**: `encodeLayerCommands` and `commitHeldLayerCommands` (`:1885`,
      `:1875`), the tail's fold, the word wake (`:3052`).
- [ ] **T3.2 Build**: the attention and speculative commands as one command buffer per
      layer, the encoded fixup joining it on a miss layer; the word wake verified to
      land at the classifier's encoder completion.
- [ ] **T3.3 Gates and golden.**
- [ ] **T3.4 Deploy and arms**: the four transition rows against Task 2's arms; the
      pre-registered rule applied (a null keeps the merge only if free and simpler).

## Task 4: the sampler feeds the next embed (E2)

- [ ] **T4.1 Read**: the produce loop (`RawCompletion.swift`), the sampler's output
      buffer, the embed's input, the stop paths, the prompt cache's append.
- [ ] **T4.2 Build**: the embed reads the sampler's buffer; the next pass encoded
      before the sample completes; the token read back asynchronously; the stop check
      one pass late with the extra pass cancelled; a client seed still reproducing.
- [ ] **T4.3 Tests**: stop strings, end of turn, max tokens, the seeded reproduction,
      the stream's token order.
- [ ] **T4.4 Gates and golden.**
- [ ] **T4.5 Deploy and arms**: the three boundary rows and `loop_sample_ms`.

## Task 5: the fold (K)

- [ ] **T5.1 Design note**: the token's command layout, the two-in-flight protocol,
      the cancel path, the error surfacing per layer; reviewed before the build.
- [ ] **T5.2 Build**: one command per token, two in flight.
- [ ] **T5.3 Tests**: the cancel with two in flight; a failed read naming its layer.
- [ ] **T5.4 Gates and golden.**
- [ ] **T5.5 Deploy and arms**: the token against the v17 close on all three shapes
      and the turn rig's pair.

## Close

- [ ] ThreadSanitizer on the final tree (the suppressions file unchanged).
- [ ] The whole-branch review by a fresh reviewer; the fold into owning commits.
- [ ] The design document's closing block: the ledger at the close beside the v17
      close, what the chapter settled, what remains.
- [ ] The architecture document brought to the final tree (the decode path's
      command structure, the routed stage's steps, the token boundary).
- [ ] Merge to `main` on Davor's go; the branch deleted; the deploy restored.
