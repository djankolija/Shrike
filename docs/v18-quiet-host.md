# v18: the quiet host

The chapter that takes the CPU off the decoded token's critical path everywhere but
the one place it must stay, the issuing of a read. Companion plan:
[v18-implementation-plan.md](v18-implementation-plan.md), whose checkboxes are the
status of record. The board this chapter was chosen from, with every avenue, its
class and its pricing, is [v18-avenues.md](v18-avenues.md); the tree it starts from is
`main` at `c503f7c`, v17's close ([v17-consolidation.md](v17-consolidation.md)).

Every number in this document is labelled: **measured** (a counter or a clock on the
mini), **modelled** (arithmetic on measured inputs) or **remembered** (an earlier
chapter's record, cited, not re-run).

## The problem

The ledger at the v17 close ([v18-avenues.md](v18-avenues.md), section 2; measured
on the mini's Task 4 arms, 2026-09-08) puts 60.9 ms on a decoded token of the 300
shape. Six of those milliseconds are conversations between the CPU and the GPU:

| row (ms per token, the 300) | measured | what it is |
| --- | ---: | --- |
| the submit gap, speculative command to hit command | 1.84 | the GPU idle while the host reads the classifier's word, plans, encodes and commits the hit command, on 15.7 layers |
| the host's work and commit latency on miss layers | ~0.9 | the plan, the pin and submit, the argument buffer, the fixup encode, the driver's commit-to-kernel latency, inside the window |
| the token boundary | 0.86 | head to sample to embed to layer 0: the token crosses to the CPU, which encodes the next pass |
| the chain transitions | 2.64 | attention command to speculative command and back, 80 command boundaries per token |

A conversation costs about 100 µs regardless of payload, in two measured halves: a
GPU write becomes visible to a spinning host 63 µs after the command's GPU end
(v14, remembered), and a committed command starts on the GPU 25 to 27 µs later (the
T4 arms' `path_*_commit_to_kernel_ms`, measured). v9 removed the conversation from
the all-hit layers with the speculative command: encoded before the route is known,
sized on the GPU by the residency classifier. On the 24.3 all-hit layers per token
that command is the whole MoE block. On the 15.7 layers with a miss the classifier
zeroes the routed grids and the host builds two commands, the hits' phase 1 and the
fixup, and the GPU idles for the round trip in between. The first of those two
commands carries nothing the GPU did not already know: the classifier wrote the
hits' positions and their resolved cells and withheld only the permission to run
them.

The endpoint the chapter builds toward: **the whole token as one command buffer**,
forty layers encoded and committed once, every routed grid sized by the classifier,
every miss layer's fixup already encoded behind an event wait on reads into cells
agreed in advance, the sampler writing the token id into the buffer the next token's
embed reads, and the next token's command committed before this one finishes. The
CPU's job on the path is to see a miss layer's word and issue its reads. The GPU
waits on the drive and on nothing else.

## Evidence the endpoint is reachable (all at `c503f7c`)

- **The classifier's contract** (`moe_classify_expert_residency_spec`,
  `Metal/MoE/moe.metal:184`): for each of the router's top-k it reads the layer's
  residency table, writes the hit count and positions, the miss count, positions and
  expert ids, the resolved cells, the indirect dispatch arguments (the caller's full
  grids when every expert is resident, zero-width otherwise) and a tagged host
  readback word. Everything Task 1 needs is already written; the grid rule is the
  only change.
- **The speculative command** (`encodeSpeculativeRouted`, `RealForwardRunner.swift:5136`)
  encodes the shared-expert chain and the pool-addressed phase 1
  (`moe_phase1_gate_up_act_spec_u16load`, `moe.metal:1114`) and phase 2
  (`moe_phase2_down_reduce_spec_k8`, `:1166`) whose grids come from the classifier.
- **The fixup** (`buildAndCommitMissFixupCommand`, `:2943`): an event wait on the
  read batch's token, phase 1 for the misses (`..._subset_u16load`), the phase-2
  reduce over hits and misses (`moe_phase2_down_reduce_k8`), the residual. The
  kernels already guard on an I/O status word (`moe_io_ready`, `moe.metal:36`),
  fail-closed.
- **The routed stage's ten steps** (`encodeDecodeRoutedMoE`, `:5227`, the
  architecture document's account): the readback, the join, the plan, the pin and
  submit, the partition, the hit split (`encodeDecodeHitSplit`, `:5415`), the I/O
  acquisition, the all-hit hand-off (`:5541`), the fixup build (`:5574`), the
  hand-off. Tasks 1 and 2 remove steps from this list; they do not add any.
- **Host writes are visible to a running command.** v16's kernel-boundary probe
  (measured on both boxes, `v16-landing.md`): a later dispatch of a running command
  sees a host write immediately once the command has moved 1 MB through memory,
  never before. The attention command streams the projections and two router GEMVs
  before its classifier, so a residency publish made during the command is seen. A
  running kernel never sees one (v14): every wait in this chapter is an event wait
  between encoders, never a spin inside a kernel.
- **v16's index swap**: a ring cell becomes a pool slot by an index exchange, no
  bytes move. Task 2's "miss i into cell i" rests on it.
- **The sampler** (`Sampler.swift`) already lands the token in a one-element buffer
  on the GPU; the loop (`RawCompletion.swift`, the produce loop) reads it back and
  calls `produce` with it. Task 4 changes who reads that buffer first.
- **The caveat, from v10 (remembered):** command-buffer merges and single-seam fusions
  measured zero wall when the host round trip dominated the token ("GPU-side savings
  are worthless at the margin; only removing the round trip cashes them"). The round
  trips are mostly gone since v9, v14 and v15; Task 3 re-measures rather than cites.

## Step zero: the board priced once (measurement only, no runtime code)

Before Task 1, one session prices every avenue the board left priceable, so that
this chapter's plan and the next three chapters' choices rest on numbers. Everything
here reads archived data or runs a throwaway script; the two model runs are marked.

1. **The miss profile by layer** (A9 Q2): `tools/expert-pool-replay.py` at
   production's configuration over the T4 route traces
   (`~/.claude/handoffs/archive/shrike-v17-t4/v17t4-arms/route-v17t4-prod-*-*.trace`):
   misses per token per layer, the first-miss share per layer. Bounds every
   lead-based scheme; decides how much of the window sits in layers deep enough to
   have a lead.
2. **Routing locality** (A9 Q1): per layer, the top-8 overlap between two occurrences
   of the same token text and between consecutive positions, from the same traces
   joined to `tokens-v17t4-prod-*-*.json`. Decides whether a token-id table predicts
   routing and at which depths.
3. **The width sweep** (A0): `tools/prefetch-coverage.py --top-m 8 12 16 24` over
   `shrike-v14-t1/step1/t1-capture/prefetch-t1-d{1,2}-*.jsonl`: coverage, precision,
   wasted reads per token at each width and distance, against the drive's headroom
   (about 120 reads per token, modelled).
4. **The slot split and the policy** (A1, A2): the replay with a per-layer allocation
   drawn from item 1's profile, and with LRU and a longer-horizon frequency, against
   aging-LFU's misses per token.
5. **Experts per small dispatch** (C1) and **dispatches per GDN layer** (D1): reads of
   the encoders, counts per layer.
6. **The compression entropy** (H2): the 4-bit index distribution and the bf16 scale
   and bias distributions of ornith15's experts and head from the `.gturbo`,
   offline; the modelled bytes at an entropy code.
7. **The prefill outlier**: the 2026-09-08 20:07 request (1,133 new tokens, 15.7 s
   prefill, 6.0 s GPU) against its four neighbours at 0.1 to 2.3 s exposed; the log
   says what the drive was doing.
8. **The calibration probe** (B5; **a model run on the mini**, one lifetime, Shrike
   stopped, Davor's go per session): a reference decode-attention kernel (oMLX's
   Qwen3-14B at 4 bits is in the mini's cache) and its slope in µs per context
   position per layer scaled to a 1 KB row, against ours at 0.20 and the 0.017 roof.
   Decides whether the class-2 scan rewrite is in a later chapter's plan at all.
9. **The GPU's clock after an idle window** (I3; **a model run on the mini** with a
   per-command-buffer timestamp dump, a diagnostic and class 1): a kernel's duration
   against the idle length before it. A slope opens a class-1 lever for a later
   chapter; a flat line closes it.

Output: the board's pricing recorded in [v18-avenues.md](v18-avenues.md) beside each
avenue, the SSD mechanism named for v19, the class-2 block's go or no-go for v21.

## Tasks

Each task lands with the four gates, the golden byte-identical on both boxes, a
deploy to the mini and the arms rig (`tools/decode-rig.sh`, two production lifetimes
per shape, the card / the 300 / the 1k) read against the previous task's arms, never
against the v17 close. The rows each task is expected to move are named up front; a
task whose rows do not move is a measured null and is recorded, not defended.

### Task 1: the hits in the speculative command (C5)

**What.** On a miss layer the classifier sizes the speculative phase 1 to the hits
instead of zero, and zeroes only phase 2. The hit command (`encodeDecodeHitSplit`)
and its command buffer go; the fixup consumes the speculative phase 1's activations
for the hits and computes phase 1 for the misses and the reduce over all eight as
today. The adopted path (a landing the classifier did not see, swapped in by the
plan) stays as it is: the fixup computes those.

**The constraint that keeps the golden.** Phase 2's reduce order over the eight
experts is unchanged: the same kernel, the same inputs, the hits' activations
produced by the same phase-1 kernel from the same cells. Only who dispatched phase 1
changes.

**Rows expected to move (the 300, measured at the close):** the submit gap
`moe_spec_routed->moe_phase1_hit` (1.84) to zero, since the hit command no longer
exists; `moe_phase1_hit` (2.28) folds into `moe_spec_routed`; the miss window
(`moe_phase1_hit->moe_phase1_miss_fixup_phase2`, 13.13) is now measured from the
speculative command's end and should shrink by the hits' phase-1 time that now runs
inside the round trip. Modelled prize 1.8 ms per token plus what the hidden round
trip returns.

**T1.1, the read (2026-09-09), the statements that move.** The speculative phase-1
kernel already skips a miss position per row: it returns when
`resolved_slots[k] == 0xffffffff`, and the classifier writes that sentinel for every
miss (`moe_classify_residency_body`). So the kernel needs no change; the classifier
changes one rule, phase 1's grid is the full grid on every layer and only phase 2
and the tail go to zero on a miss layer. The hits' activations land in `moeActs` at
`k * F + f` exactly where the hit command's subset kernel wrote them, from the same
pool bytes through the same helper (`moe_int4_gate_up_rows_simd_tgmem_u16load`,
`moe_glu(moe_gate_up_bias(...))`), so the values are bit-identical by construction.
What goes: `encodeDecodeHitSplit` and its stage call; the context's `phase1HitCB`,
`phase1HitSplitArgBuf`, `hitCommitNanos`; the pending command's `phase1HitCB` and
`hitCommitNanos` with their waits, error check and `moe_phase1_hit` role recording;
the lease branch's wait on the hit command; `expectedOverlapCompletions` fixed at
one; the hit-split scratch arrays, `moeHitActiveSlots`, the four hit-split stats and
their four runner-line fields; `makeRoutedArgumentBuffer` if the hit split was its
only caller. What changes: `buildAndCommitMissFixupCommand` takes `missesOnly`
(the classifier ran and the plan has misses) instead of the hit command, and its
argument buffer is always the reused one over the plan's blobs; the fixup's
`kernelRole` keeps its names. The test
`productionRoutedPipelineAndHitSplitMatchReference` becomes the new pipeline
against the same reference, and the classifier's grid rule gets its own test.

**Risk.** The speculative phase 1 for hits on a miss layer reads cells the plan may
swap after the classifier ran (the adopted case). The classifier's cells are the
residency table's at classification time, which is what the hit split reads today
(`:5381` cross-checks the two sets fail-closed); the cross-check stays.

**Built and measured (2026-09-09).** Five files: the classifier's grid rule in
`moe.metal` (phase 1 always full; the doc comment), the runner 138 lines lighter
(the hit split, its command buffer, its context fields, the pending command's hit
fields, the lease wait, the scratch arrays, `moeHitActiveSlots`, the four stats), the
server's four runner-line fields, the two MoE test files. The four gates: the release
build with zero warnings, strict lint with zero violations, the links, the suite at
1,234 tests in 170 suites in 202 s. The local golden identical on both profiles. The
build deployed to the mini (binary e6241a16f55feede), the mini's golden identical on
both profiles, the arms two lifetimes per shape beside the v17 close's, the card's
answer identical to the archived turn, the misses per token identical to the tenth
(20.0 / 20.0 / 18.8), the turn rig's pair 300 at 3.26 s warm and 7.89 cold (v17's
close 3.17 and 7.36, the deploy's first lifetimes). Artefacts at
`~/.claude/handoffs/archive/shrike-v18-t1/`.

| shape | v17 close tok/s | Task 1 tok/s | spec ms | hit ms | fixup ms | window ms | submit ms | rows + gaps ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| the card | 15.57 / 15.62 | 15.53 / 15.24 | 11.4 to 13.4 | 2.1 to 0 | 2.6 to 2.6 | 13.0 to 14.7 / 15.3 | 1.7 to 0 | 63.8 to 64.1 / 65.3 |
| the 300 | 16.43 / 16.30 | 16.35 / 16.40 | 10.9 to 13.6 | 2.3 to 0 | 2.8 to 2.8 | 13.1 to 14.9 | 1.8 to 0 | 60.5 / 60.9 to 60.9 / 60.7 |
| the 1k | 16.14 / 16.16 | 16.23 / 16.18 | 11.6 to 13.7 | 2.1 to 0 | 2.5 to 2.5 | 12.3 to 14.0 | 1.7 to 0 | 61.6 to 61.4 / 61.5 |

**Reading: a measured null on the wall, and the mechanism.** The rows moved as
pre-registered and the wall did not. The hit role and the submit gap went to zero;
the speculative command grew by the hits' phase 1 plus about 0.4 ms of the full
grid's exiting rows; and the window grew by exactly what the two removed items had
occupied. On a miss layer the GPU's idle before the fixup is the read's flight,
whichever command precedes it. The stage order is the plan, then the pin and submit
of the reads to the storage threads, then the partition, then (until this task) the
hit command: the reads were already in flight when the hit command was built, so
that command and the 117 µs the GPU idled waiting for it sat inside the read's
shadow, not on the path. The window's start moved earlier by the same amount, the
read's landing did not move, and the fixup started when it always had. The design's
1.8 ms assumed the submit gap was serial; it was not.

**What the task settled.** The critical path on a miss layer is the read's
issue-to-landing plus the 155 µs wake; the only host work on it is what precedes the
issue: the word's 63 µs and the plan, pin and submit, about 25 µs, per miss layer.
That re-prices Task 2 from 0.9 ms to about 1.2 ms (13.6 layers of that pre-issue
latency, if the encoded fixup lets the host issue the reads the moment it sees the
word) and makes it the chain's one SSD-side lever; the chain's modelled total goes
from about 5 ms to about 3.5. Task 1 itself is a simplification with nothing lost:
one command buffer fewer per miss layer, one host path fewer, golden identical on
both boxes. Keep or revert was Davor's ruling; the recommendation was keep, because
it costs nothing measured, deletes a path, and Task 2 needs the speculative command
to own the hits. **Davor's ruling (2026-09-09): keep and commit.**

### Task 2: the fixup as a speculative command (C6)

**What.** The fixup is encoded before the route is known, like the speculative
command: an event wait, phase 1 for the misses as an indirect dispatch over the
classifier's miss list, the reduce over all eight, the residual. The misses are read
into ring cells agreed in advance, miss i into cell i, so the encode can name the
cells without the plan. On the word, the host's whole critical-path job is to call
`pread` for each miss into its agreed cell and hand the batch to the reader threads,
which signal the event. The plan (victims, the swap of the cells into the pool)
runs after the reads are issued, at the next wake, off the path.

**The constraint that keeps the golden.** Nothing about which experts compute or in
what order changes. Which pool slot an expert occupies afterwards may differ from
today because the plan runs later, so the misses per token may drift; the answer
does not, and the drift is measured.

**Rows expected to move:** the host's `path_hit_encode_ms`, `path_fixup_build_ms`,
`path_argbuf_ms`, `cache_plan_ms` off the path (they still run, later); the fixup's
commit-to-kernel latency gone (the command is already committed); the window's
per-layer latency term down by the host's ~40 µs and the 27 µs commit. Modelled 0.9
ms per token.

**Risk.** The ring's nine cells per layer bound the misses a layer can take through
the agreed-cell path; a layer with more misses than free cells falls back to today's
host-built fixup, fail-closed, and the counter says how often. The join (a
prediction's read landing within 400 µs) must keep its adoption semantics: the
classifier's miss list is authoritative for the encoded fixup, and an adopted
landing is handled by the plan exactly as Task 1 leaves it.

### Task 3: one command per layer (prices E1)

**What.** The attention command and the speculative command of a layer become one
command buffer; on a miss layer the encoded fixup joins it behind its event wait. The
per-layer boundary the architecture document names ("one submission and one boundary
per layer") becomes an encoder boundary inside one command.

**Rows expected to move:** the transitions `attn_layer_linear->moe_spec_routed`
(1.08), `moe_spec_routed->attn_layer_linear` (0.85), `attn_layer_kv->moe_spec_routed`
(0.36), `moe_spec_routed->attn_layer_kv` (0.35): 2.64 ms per token today at about 33
µs a boundary. If an encoder boundary costs what a dispatch does, about 0.8 remains.
This is the row v10 measured as null under different conditions; the task's rule is
pre-registered: the arms decide, and a null keeps the merge only if it costs nothing
and simplifies Task 5.

**Risk.** The word wake: the host spins on the classifier's word 63 µs after the
tail's GPU end; with the tail inside a longer command the word still lands when the
classifier's encoder completes, not at the command's end (v16's probe), and Task 1's
hidden round trip needs exactly that. Verify on the first build with
`path_router_wake_ms`.

### Task 4: the sampler feeds the next embed (E2)

**What.** The sampler's one-element token buffer is read by the next token's embed
kernel directly; the host encodes the next pass before the sample completes and
reads the token back asynchronously for detokenisation and streaming. The stop check
(end of turn, stop strings, max tokens) runs one pass late and cancels the extra
pass's output; one token's work is wasted per answer.

**Rows expected to move:** the token boundary `head_logits->sample` (0.31),
`sample->embed` (0.30), `embed->attn_layer_linear` (0.24): 0.86 ms to the cost of two
encoder boundaries. `loop_sample_ms` (0.44) off the path.

**Constraints.** The sampler's numerics are untouched (the same kernels, the same
seed derivation per position); the prompt cache and the route trace still see every
token in order; a client-supplied seed still reproduces. The fused greedy head, if
the server ever uses it (I1), writes the same buffer.

### Task 5: the fold (K)

**What.** One command buffer per token, forty layers, with two tokens in flight: the
next token's command committed before this one completes. The host's loop becomes:
issue reads on each miss layer's word as it lands; encode token t+2 while t+1 runs;
read tokens back for the stream.

**Rows expected to move:** whatever Tasks 3 and 4 left of the boundaries; the
`cb1_ms` encode (1.50) fully overlapped. The chapter's close measures the token
against the v17 close on all three shapes and the turn rig's pair.

**Risks.** Command-buffer size (about 600 dispatches per token; fine for Metal);
error surfacing when a whole token is one command (a failed read's fail-closed guard
must still name its layer); the stop check's late cancel with two in flight (at most
two wasted passes).

## Method

- The four gates per commit (release build with zero warnings, `swiftlint lint
  --strict`, the link check, `swift test --no-parallel`), ThreadSanitizer once at the
  close, the golden byte-identical on both boxes at every code commit.
- Every measurement on the mini (the target; M4 Pro numbers are iteration signal),
  two production lifetimes per shape through the arms rig, the first one or two
  lifetimes after a deploy discounted (they can run 2 to 4 % slow).
- One model process at a time on either box; deploy leave to the mini asked per
  session.
- Each task pre-registers the ledger rows it expects to move; a task that moves
  nothing is a measured null and is written down as one.
- Subagents run the gates and the rigs and return verbatim diagnostics; the
  reasoning stays in the session.

## Numerics policy

Class 1 throughout (section J of the avenues, Davor's ruling 2026-09-09): every
commit byte-identical on both golden profiles. No kernel's arithmetic changes; what
changes is who dispatches, when, and in which command. The one place numerics could
drift is Task 1's reduce, and it is pinned above.

## Out of scope, and where it went

- **The SSD mechanism** (A0's width, A9's lead, A1/A2's split and policy): v19,
  chosen by step zero's pricing.
- **Lossless compression** (H2): v20, scoped to the expert kernels and the head.
- **The class-2 block** (B6, J1, J2, G1): v21, with the forced-token instrument,
  behind the variance gate, one golden re-capture.
- **Speculation revisited** (H1) and **the draft in the spin-wait** (H3): after v21.
- **The fused greedy head in the server** (I1): only if a client sends temperature
  zero; not scheduled.

## Risks

- Task 3's prize may be null (v10's measurement), in which case the chapter's
  modelled 5 ms is 3.5.
- Two tokens in flight (Task 5) interacts with the prompt cache's settle and the
  streaming stop; the design keeps the host's view of the token order intact and
  the risk is in the cancel path, tested with stop strings and max tokens.
- The mini's first lifetimes after each deploy; the external build SSD dropping
  off (check the receipt before a local golden).
