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
from about 5 ms to about 3.5 (T2.0 corrected this the next day: the word's 63 is not
the host's to remove and C6 is at most 0.34 ms; see Task 2). Task 1 itself is a
simplification with nothing lost:
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

**T2.0, the pre-issue path priced (2026-09-09; Task 1's arms, no runtime code, no
model run).** The design above and Task 1's re-pricing counted the word's 63 µs as
part of the prize. It is not C6's: the miss list arrives with the word, so no
host-side restructuring issues a read before it. The pre-issue chain on a miss layer
from the runner line, the 300, lifetimes 1 / 2 (the 1k and the card agree to the
tenth; 13.5 miss layers per token):

| step | ms per token | µs per miss layer | C6 removes it |
| --- | ---: | ---: | --- |
| the word's visibility past the tail's GPU end (`path_router_wake_ms`, over 40 layers) | 2.43 / 2.45 | 61 | no |
| the ring's join, 1.7 per token, up to 400 µs each (`readyCells`) | uncounted | unknown | no |
| plan, pin and submit (`cache_plan_ms`, `path_pin_ms`, `path_submit_ms`, over 40 layers) | 0.34 / 0.34 | at most 25 | all but the pread's issue, about 5 |
| the hand-off to the reader thread (`io_queue_ms`, submit to `markInFlight`) | 0.35 / 0.35 | 26 | no |
| then the read's flight and the wake (the window, `io_fixup_wake_ms`) | 14.9, 2.1 | 1,100 and 157 | no |

C6's prize is bounded above by 0.34 ms per token (all 40 layers' plan, pin and
submit charged to the miss layers) and is about 0.2 with the all-hit layers' share
removed: under the mini's lifetime drift, a measured null before it is built. The
63 belongs to A8 (shrink it: untraced) or to prediction (hide it: A3, A9, H3). The
slice has shrinkers and no exposer (the board's shadow ledger, section 2), so 0.2 is
its ceiling. Two smalls surfaced beside it, both on the board: the hand-off's 26 µs
(A5's family) and the join's place before the issue (C7). The three tracks of a miss
layer, an all-hit layer and the boundary are drawn in the session's tracks page.
**Davor's ruling (2026-09-09): Task 2 skipped as a performance task; Task 4 (E2)
next, then Task 3 (one command per layer), the fold (Task 5) decided on Task 3's
arms.** The agreed-cell mechanism is not dropped: the endpoint's fixup behind an
event wait on reads into agreed cells is exactly it, so it moves into the fold's
design note (T5.1) as structure, not as a lever. The chapter's modelled prize after
T2.0 is E2's 0.7, the transitions' 0 to 1.8 and this slice's 0.2: 0.9 to 2.7 ms,
with Task 3's arms deciding which.

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

**T3.1 (2026-09-09), the read and the probe.** On the served model every layer
already folds its input norm, attention and tail (the router, the probe, the
classifier writing the tagged word) into one command (`attnCB`; `tailCB` is nil on
GDN, MLA and gated layers), and the speculative command (`specCB`: the shared
expert, phase 1, phase 2 and the residual, all indirect from the classifier's
arguments) is a second command committed right behind it in
`commitHeldLayerCommands`. Task 3 is the merge of those two into one command per
layer; the fixup stays a separate command on miss layers, host-built after the word
with Task 2 skipped. The task rested on an assumption the risk paragraph above cites
v16's probe for, but that probe measured the other direction (a host write seen by
a later dispatch): that the host sees the classifier's word when its encoder
completes rather than when the command ends. Measured now with
`MidCommandVisibilityTests` (a sampler kernel writes the word as a command's first
encoder, six blit encoders copying 1.5 GB follow, the host spins on the word): the
word is seen 42 to 45 µs after the command's GPU start and 29 to 31 ms before its
end, three runs on the M4 Pro. Mid-command visibility is immediate, tens of
microseconds after the write, and the merge is safe on this hardware; the mini's
first arms confirm it through `wait_ms`, flat if the word still lands at the
classifier and about 13 ms per token higher if it waited for the command.

What the merge changes in the instruments: the attention row and the speculative
row become one command and one role, so the ledger keeps their sum and the
context slope (the speculative work is context-free, so the merged row's slope is
the attention's) but not the split. The prefetch race counters key on the tail
command's GPU span, which becomes the whole layer's; they stay as diagnostics with
that caveat. `path_router_wake_ms` measures the wake past the command's end and
clamps to zero when the word lands inside it, so it becomes the count of wakes
that fell past the layer.

**The statement list (T3.2).**

- S1. `encodeSpeculativeRouted` encodes into a given command instead of making its
  own; when the tail is folded the speculative encoders follow the tail in
  `attnCB`; `HeldLayerCommands.specCB` becomes optional and
  `commitHeldLayerCommands` commits what exists. The split path (a separate tail
  command, gpt-oss and plain attention) keeps its speculative command as today.
- S2. The routed stage: the all-hit hand-off's pending command is the merged
  command; the miss path's pending carries no separate speculative command;
  `recordRoutedCommandTimings` and the diagnostics buffer list skip what is not
  there.
- S3. Roles: the merged command is recorded once as `layer_linear` or `layer_kv`,
  new names so the rows' change of meaning is explicit; `decode-rows.py`'s window
  regex takes the new names; `parse-kernel-stats.py` is generic.
- S4. The completion clock tracks the merged command; no new knob; a null reverts
  by git.
- Tests: the probe stays as the assumption's guard; the merge itself is
  structural and is covered by the golden and the arms, since the unit tests never
  load a model.

**Rows pre-registered** (Task 4's arms, ms per token): `attn_layer_linear->moe_spec_routed`
(about 1.0) and `attn_layer_kv->moe_spec_routed` (about 0.3) gone, about 1.3 of the
2.6; `moe_spec_routed->attn_layer_linear` (0.8 to 0.95) and `->attn_layer_kv` (0.3)
become `layer_*->layer_*` at the same cost, a command boundary still; the merged
role equal to the sum of the two it replaces; `wait_ms` flat; the window row
renamed. The wall by about 0.9 ms per token if an encoder boundary costs about 10
µs, a null if the command boundary's cost was never on the path (v10's finding
under different conditions). The rule stands: a null keeps the merge only if free
and simpler, and it is simpler, one command and one field fewer per layer.

**T3.2 to T3.4 (2026-09-09), the build, the gates and the arms.**
`encodeSpeculativeRouted` encodes into a given command and follows the tail in
`attnCB` when the tail is folded, which is every layer of the served model; the
split-tail path keeps its separate command. `HeldLayerCommands.specCB` is optional
with `routedCB` naming the carrier; the all-hit pending command is the merged
command and carries no role of its own (the layer's record covers it), the miss
path's pending carries no separate speculative command; the merged command is
recorded once as `layer_linear` or `layer_kv`; `decode-rows.py`'s window regex takes
the new names; no knob. The four gates: the release build with zero warnings, lint
zero in 212 files, links clean, 1,243 tests in 171 suites in 204 s with the probe
among them. The golden identical on both profiles on both boxes (the mini on the
deployed 80654748c95eeb44, the server stopped). The arms two lifetimes per shape
against Task 4's, the pair 300 (3.16 to 3.26 s warm, 7.88 cold), production
restored; the card's answer identical, the misses per token identical. The rows,
the 300, lifetimes 1 / 2, ms per token:

| row | Task 4 | Task 3 |
| --- | ---: | ---: |
| the four transitions, tail to speculative and back | 2.49 / 2.47 | gone |
| the layer-to-layer transitions | none | 1.08 / 1.15 |
| the attention roles plus the speculative role | 34.65 / 34.61 | 36.30 / 36.29 as `layer_linear` + `layer_kv` |
| `wait_ms` | 51.73 / 51.61 | 51.76 / 51.80 |
| `path_router_wake_ms` | 2.44 / 2.47 | 0.00 / 0.00 |
| decode tok/s | 16.50 / 16.54 | 16.61 / 16.60 |

**Reading.** The forty tail-to-speculative command boundaries are gone, about 1.4
ms of gaps, and the merged commands grew by about 1.65: the GPU still drains
between the tail's last kernel and the speculative work's first, now at an encoder
boundary inside the command, and that idle sits inside the role's span. The net is
the difference between a command boundary and an encoder boundary, about 10 µs a
layer, not the 25 the model assumed: the printed roles and gaps sum fell 0.2 to 0.4
ms per token on the 300 and the 1k; the wall +0.4 to +0.6 % on the 300 and +0.5 to
+0.8 % on the 1k against Task 4's clean same-day lifetimes (16.28 to 16.34 to
16.41 / 16.42), the card mixed by a slow-drive lifetime on each side. `wait_ms`
flat and `path_router_wake_ms` at zero: the word lands inside the command, at the
classifier, as the probe said. Not a null and a third of the modelled 0.9; kept,
being simpler and non-negative. The ledger's attention row now reads as the layer
(attention, tail, speculative work and the boundary between them); its context
slope survives, its split does not.

**What it says for the fold.** The forty layer-to-layer command boundaries that
remain, about 1.1 ms per token, would become encoder boundaries in one command
per token, worth about 0.4 by this measurement, plus Task 4's remaining boundary
gap (0.25) and C6's slice (0.2): about 0.85 ms per token, 1.4 %, for the
agreed-cell mechanism, two commands in flight and the cancel path. K's
re-examined prize; the ruling is Davor's.

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

**T4.1 (2026-09-09), the read.** The boundary today is three command buffers, each
committed and waited on synchronously on the one queue: the head (`emitHead`'s
`runSync`), the sample (its own command buffer in `sampleOnce`, `commit` then
`waitUntilCompleted`, the token id loaded from the shared 4-byte `outToken`), the
embed (`produceToken`'s `runSync`, the token id a `setBytes` constant to the lookup
kernel), then layer 0's commands. Nothing is encoded ahead of the sample. The three
gaps are, each, a completion mark (about 160 µs, the driver's), the host's work
between (the sampler's seven encoders, or the stop checks, detokenisation and the
stream's callback at 33 µs, or layer 0's encode) and a commit-to-kernel (25). The
seed is `seedFor(config, position: generated)`, host-computed, a kernel constant,
known before the head runs; the repetition penalty is host-side and in place on the
logits between the head and the sample (the server's default is 1.0, no penalty).
The stop token is never embedded, appended or streamed; it returns as
`uncommittedBoundaryTokenIDs`. Layer 0 is a routed GDN layer on the served model
(the gap row is `embed->attn_layer_linear`), so its commands can be held encoded
like any other layer's.

**What the read changes in the design.** The stop check does not need to run a pass
late. A pass mutates the GDN layers' recurrent state in place and writes the KV row,
so cancelling an extra pass would mean restoring state (a copy or a ping-pong per
touched layer) and rewinding the cursor, a cost the design above did not carry. It
is unnecessary: the token's word lands about 63 µs after the sample kernel ends,
long before layer 0 would start, so the host checks the stop on the word and commits
layer 0 only when there is none. The price is gap 3 at about 130 µs (the word's
wake, the host's checks, the commit) instead of an encoder boundary; the return is
no wasted token, no state to undo, the prompt cache's and the KV's view of the
answer unchanged.

**The statement list (T4.2).**

- S1. `outToken` becomes the token's word: the host writes a sentinel
  (`0xFFFFFFFF`, above any vocabulary) before the sample is committed, the sample
  kernel writes the id, the host spins on it with the router wake's 1 s fallback to
  `waitUntilCompleted`.
- S2. The sampler's encode is handed to the runner: `produce` takes an optional
  sample closure that the runner encodes into the head's command behind the lm_head
  GEMV; the head and the sample are one command buffer and the runner returns after
  the commit, without the head's wait.
- S3. The embed reads the token from `outToken`: a device-pointer variant of the two
  lookup kernels' Swift encoders (`EmbedLookupInt4`, `AffineQuantEmbeddingLookup`),
  the same lookup on the same table; encoded behind the sample in the same command
  buffer, no wait.
- S4. Layer 0's commands are held encoded during the head (`encodeLayerCommands`
  into the `heldNext` slot for `position + 1`, after `kv.advance()` and the
  reserve), so that on the word only their commit is left.
- S5. The loop's order on the word: spin; the stop-token check; detokenise and the
  stop-string matcher; max tokens; the progress callback and the caller's stop; then
  the held commit and the pass from layer 1 as today; then `history.append`, the
  counters, the position. On a stop nothing is committed.
- S6. `LogitProducer` grows the two-step shape with a default that keeps the
  synchronous path for the scripted test producer, the CLI and the app.
- S7. Fallbacks to today's path, no new knob: a repetition penalty other than 1.0,
  the first token after prefill (`prefillSeed == .logitsWritten`), the fused greedy
  head.
- Counters: `loop_sample_ms` becomes the word's wait and the runner line says so; the
  `sample` and `embed` kernel roles fold into `head_logits`, which grows by their
  time (about 0.17 ms).

**Rows pre-registered** (Task 1's arms, ms per token, the three shapes' two
lifetimes): `head_logits->sample` 0.31 to 0.38 and `sample->embed` 0.28 to an
encoder boundary each (about 0.01); `embed->attn_layer_linear` 0.23 to 0.25 to about
0.13; `loop_sample_ms` 0.44 to 0.47 to the word's wait; the wall by about 0.65 ms per
token, 1.1 % on the 300 (61.2 to about 60.5 ms), likely inside the lifetime drift on
the wall and unambiguous on the rows. Misses per token unchanged.

**T4.2 and T4.3 (2026-09-09), the build and the tests.** The boundary lives in
three places. `BoundaryLogitProducer` (`LogitProducer.swift`) adds the two-step
shape to the producer: a `produce` taking an optional token, the token word and the
sampler's encode closure, and `awaitBoundaryToken`. The runner conforms:
`produceToken` takes the held layer 0 on a continued pass instead of encoding an
embed, and ends with `emitBoundary`, one command carrying the final norm, the
lm_head GEMV, the caller's sampler and the word-fed embed, the sentinel written
into the word before the commit; then the cursor advances and `holdLayerZero`
encodes layer 0 for the next position into the held slot. `awaitBoundaryToken`
spins on the word with the router wake's one-second fallback, counted as
`boundary_wake_fallbacks` on the runner line. The previous boundary's command is
waited on and recorded as `head_logits` at the end of the next pass, when it has
long completed. The two embed encoders gained a `tokenBuffer:` overload binding the
word at the kernel's constant argument, no Metal change. The loop chooses the path
once per generation (a boundary producer, not the fused greedy head, the repetition
penalty at 1.0), samples the first token after prefill as before, and from then on
awaits the word, checks the stop token, detokenises, runs the stop-string matcher,
the max-tokens check and the progress callback, and only then calls the continued
produce with the sampler's encode at the next token's index. Tests: six in
`RawCompletionLoopTests+Boundary.swift` on a scripted boundary producer that runs
the real sampler on a command buffer (the same tokens, deltas, reason, cursor and
history as the synchronous path; every pass after the first continued; the stop
token without another pass; max tokens; a stop string; the penalty fallback), three
of which fail with the path switched off and three of which are invariants of both
paths; two in the encoder tests (the buffer-fed lookup bit-identical to the
constant-fed one, both kernels). The four gates: the release build with zero
warnings, lint zero in 212 files, links clean, 1,242 tests in 170 suites in 203 s.
The local golden identical on both profiles (T4.4).

**T4.5 (2026-09-09), the deploy and the arms.** Deployed to the mini (binary
6142e12205c5d3eb with its six bundles), the mini's golden identical on both
profiles with the server stopped, then the arms rig: two production lifetimes per
shape against Task 1's arms, the turn rig's pair 300, production restored. The
card's answer matched the archived turn on both lifetimes; misses per token
identical (20.0 / 20.0 / 18.8); `boundary_wake_fallbacks` zero everywhere. The
rows, the 300, lifetimes 1 / 2, ms per token:

| row | Task 1 | Task 4 |
| --- | ---: | ---: |
| `head_logits->sample` | 0.324 / 0.312 | gone |
| `sample->embed` | 0.277 / 0.279 | gone |
| `embed->attn_layer_linear` | 0.230 / 0.238 | gone |
| `head_logits->attn_layer_linear` | none | 0.259 / 0.253 |
| the `sample` and `embed` roles | 0.166 | 0.001 (the first token after prefill) |
| `head_logits` | 4.620 / 4.607 | 4.805 / 4.777 |
| `loop_sample_ms` | 0.445 / 0.436 | 4.987 / 4.959 (the word's wait through the head) |
| decode tok/s | 16.35 / 16.40 | 16.50 / 16.54 |

The same on the card and the 1k: the three gaps to one of 0.25 to 0.27, the head
up by the sample and the embed. The wall against the morning's Task 1 arms: the 300
+0.9 %, the card 15.53 / 15.24 to 15.44 / 15.38, the 1k 16.23 / 16.18 to 16.05 /
16.04, then 16.38 / 15.97 on two more lifetimes; the pair 300 at 3.17 s warm and
7.89 cold (Task 1's 3.26 and 7.89).

**The 1k's reading, and a noise source named.** The 1k's slow lifetimes carried
reads up 0.6 to 0.85 ms per token, the miss window up 0.8 to 1.1 and `prefetch_late`
at 67 to 92 where every Task 1 lifetime had zero, while its lifetime 3 showed the
boundary's gain cleanly. A same-box interleaved A/B on the 1k settled it: Task 1's
code rebuilt (8188af4a1ebbb14b) against Task 4's, two lifetimes each, twice:

| arm | tok/s | `prefetch_late` | the boundary's gaps ms/token |
| --- | --- | --- | --- |
| Task 1 | 16.09 / 16.15 / 16.05 / 15.72 | 0 / 0 / 0 / 74 | 0.86 / 0.85 / 0.90 / 0.85 |
| Task 4 | 16.20 / 16.33 / 16.28 / 16.34 | 41 / 0 / 0 / 0 | 0.24 / 0.26 / 0.27 / 0.27 |

Task 4 wins every pair; on the clean lifetimes 16.10 to 16.32 tok/s, +1.4 %, about
0.8 ms per token against the 0.65 modelled. The slow state hit Task 1's own fourth
lifetime, so it is the box's, not the change's: a lifetime with `prefetch_late`
above zero runs its reads 4 to 6 % slow and its window a millisecond wide,
whichever binary serves it. Two accounting notes for future readers: the kernel
stats print the twelve largest gaps only, so a row can appear or vanish because
other rows moved (the adopted fixup's gap surfaced this way; the adopted role's
count is the same in both arms); and the 0.11 to 0.13 ms `->head_logits` gap now in
the twelve is the sampler's seven encoders, moved from the old first gap to the
head's front. Artefacts at `~/.claude/handoffs/archive/shrike-v18-t4/`.

**What the task settled.** E2 is real and lands as modelled: the token boundary
went from three conversations with the host to one word wake, about 0.6 ms per
token on the rows and about 0.8 on the wall, golden identical on both boxes, no
kernel changed. The stop check stayed on time and nothing runs a pass late. The
remaining boundary cost is the one gap of 0.25 ms (the word's 63, the host's checks,
the commit) and the sampler's encode at the head's front; both are the fold's (Task
5) if it proceeds.

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

### Task 6: the walls (D1; scheduled by Davor's ruling, 2026-09-09, after Task 3)

**What.** The GPU's own boundaries. Task 3 measured the gap between two dependent
kernels at about 25 µs inside one command: the machine drains one kernel and
refills on the next, and Shrike's decode kernels are short enough that the drain is
as long as the work. About 820 dispatches per token pay it (D1). The task is D1's
six merges, each keeping every arithmetic operation and its order: the shared
expert's gate and up GEMVs as one grid over both row sets; the scalar gate into
that dispatch; speculative phase 2 plus its residual; the top-k select plus the
classifier; conv plus qk norm; the input norm into the in-projection. Before any
merge, T6.0 prices the wall by its kind, since D1's caveat is now the live question:
the speculative command spends four encoders on seven dispatches and the fixup
three on three, and if an encoder boundary is the 25 µs kind while a dispatch
boundary inside one encoder is v10's 12, putting those dispatches on one encoder
each is the cheapest move on the board and changes the merges' pricing.

**Rows expected to move,** with the slack rule applied up front: a merge inside the
speculative command counts only on the 26.5 all-hit layers per token, since on a
miss layer that command runs in the read's shadow; a merge in the tail or the GDN
block counts on every layer. At v10's 12 µs a wall: the gate and up GEMVs 0.32 ms
per token, the scalar gate 0.32, phase 2 plus residual 0.32, the select plus the
classifier 0.48, conv plus qk norm 0.36, the norm into the in-projection 0.36:
about 2.2 ms modelled, more if T6.0 finds the encoder boundary at 25. The rows are
`layer_linear` and `layer_kv` (the merged layer roles, down by the walls removed),
`moe_phase1_miss_fixup_phase2` if the fixup's encoders fold, and the wall.

**Constraints.** Class 1: the same arithmetic in the same order in every merge, and
each merge carries a bitwise arm against the two kernels it replaces, not the
golden alone, because a fused kernel can elide a half-precision rounding under fast
math on a register-resident value (the decode chapter's T2 found exactly that). The
misses per token unchanged. Each merge lands as its own commit with the four gates
and the golden; the arms per merge on the mini, or per pair once the first two
land as modelled.

**Order.** Task 6 runs before the fold's ruling: the fold is priced on the cost of
a command boundary against an encoder boundary, and Task 6 re-measures both.

**T6.0 (2026-09-09), the wall by its kind: landed.** The speculative command's
seven dispatches go on one serial encoder (the shared expert's chain, phase 1,
phase 2 and the residual, where they had four) and the fixup's three on one (where
they had three), no kernel change: six kernel wrappers gained `encoder:` variants
with the `commandBuffer:` overloads kept as thin wrappers, and a test runs the
production routed pipeline on one encoder and on separate encoders and asserts the
outputs bit-identical. The four gates (the release build with zero warnings, lint
zero in 212 files, links clean, 1,244 tests in 171 suites in 202 s), the golden
identical on both profiles on both boxes (the mini on 83eddc36bd722e57, the server
stopped). The arms, two lifetimes per shape against Task 3's; the card's answer
identical; the rows, the 300, lifetimes 1 / 2, ms per token:

| row | Task 3 | T6.0 |
| --- | ---: | ---: |
| `layer_linear` (30 GDN layers) | 27.19 / 27.18 | 25.12 / 25.13 |
| `layer_kv` (10 layers) | 9.12 / 9.11 | 8.45 / 8.46 |
| `moe_phase1_miss_fixup_phase2` | 2.47 / 2.47 | 1.92 / 1.92 |
| the layer-to-layer transitions | 1.08 / 1.15 | 1.77 / 1.52 |
| `wait_ms` | 51.76 / 51.80 | 50.98 / 50.35 |
| misses per token | 19.9 / 19.9 | 20.0 / 20.1 |
| decode tok/s | 16.61 / 16.60 | 16.87 / 17.06 |

The 1k 16.41 / 16.42 to 16.86 / 16.88, the card 15.66 / 15.32 to 15.80 / 15.89 (one
slow-drive lifetime on each side), the pair 300 at 3.13 to 3.16 s warm.

**Reading.** The GPU's role time fell by 3.3 ms per token: 2.07 on the GDN layers,
0.66 on the KV layers, 0.55 on the fixup, over the 120 speculative-command
boundaries and the 27 fixup boundaries that went, **about 22 µs an encoder
boundary**. On the path, by the slack rule, about 2.4 of the 3.3 (the
speculative command's share on the 13.5 miss layers sits in the read's shadow, and
the miss window duly widened by 0.6 to 0.8); the layer-to-layer transitions grew
by 0.4 to 0.7, unexplained and small; the wall +1.6 to +2.8 % on the 300 and +2.7
% on the 1k, about 1.0 to 1.7 ms per token, the misses per token up 0.5 % (the
faster pass leaves the ring's landings a little less time before the classifier).
Kept: class 1, no kernel changed, the largest single gain of the chapter so far.

**What it settles for the rest of Task 6.** The encoder boundary is the expensive
kind, about 22 µs; the dispatch boundary inside an encoder is v10's 12, so the six
merges' pricing at 12 µs a wall stands. One more encoder-only step is priced by the
same number and is the cheapest thing left: the boundary command of Task 4 holds
nine encoders (the final norm, the lm_head GEMV, the sampler's three softmax stages
and three top-k stages, the embed), eight boundaries at 22 µs, about 0.18 ms per
token, a T6.0b before the merges.

**T6.0b (2026-09-09), the boundary command on one encoder: landed.** The final
norm, the lm_head GEMV, the sampler's stages and the word-fed embed encode on one
serial encoder (the synchronous head path also takes one encoder for its two
kernels); the sampler and the four sampling kernels and the two embed encoders gained
`encoder:` variants with the `commandBuffer:` overloads as thin wrappers; the
producer protocol's sample closure takes the encoder; a test runs the sampler and
the embed on one encoder and on separate encoders at temperature 0 and at top-k 8
with a seed and asserts the token equal and the embed output bit-identical. The
four gates (1,245 tests in 172 suites in 203 s), the golden identical on both boxes
(the mini on e0f8bd17dc8ecbb0). The arms against T6.0's: `head_logits` 4.82 / 4.80
to 4.73 / 4.73 on the 300, 4.73 / 4.72 to 4.67 / 4.69 on the card, 4.76 / 4.76 to
4.73 / 4.70 on the 1k, so 0.06 to 0.09 ms per token for eight boundaries, **about
10 µs a boundary between the sampler's small kernels**, half the 22 measured between
the speculative command's larger ones; the wall 16.87 / 17.06 to 17.12 / 17.12 on the
300, 15.80 / 15.89 to 15.90 / 15.92 on the card, the 1k 16.86 / 16.88 to 16.58 /
16.88 with a slow-drive first lifetime (`prefetch_late` 81); the pair 300 at 3.13 s
warm. Inside the drift on the wall, the row moved as pre-registered in sign at
half the size; kept as simpler and non-negative. The encoder boundary's cost is
not one number: about 22 µs around the speculative command's indirect dispatches
and about 10 around the sampler's small kernels.

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
  nothing is a measured null and is written down as one, with the slack it sat
  behind and what would expose it.
- "X is hidden by Y" is a statement about today's dependency graph, not about X: it
  holds only while nothing shortens Y by more than X's slack. The board's shadow
  ledger (the avenues document, section 2) keeps every hidden item with its cost, what
  hides it, its slack and its exposer; a task that shortens a Y pre-registers the
  items it would expose and measures them in its own arms.
- A lifetime whose runner line shows `prefetch_late` above zero ran in the mini's
  slow-drive state (reads 4 to 6 % slow, the miss window a millisecond wide,
  either binary; Task 4's A/B): read it as noise, and settle a mixed shape with a
  same-box interleaved A/B rather than more lifetimes of one arm.
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
  modelled 2.7 ms after T2.0 is 0.9 and the fold (Task 5) is not worth its design
  note; the decision is taken on Task 3's arms.
- Two tokens in flight (Task 5) interacts with the prompt cache's settle and the
  streaming stop; the design keeps the host's view of the token order intact and
  the risk is in the cancel path, tested with stop strings and max tokens.
- The mini's first lifetimes after each deploy; the external build SSD dropping
  off (check the receipt before a local golden).
