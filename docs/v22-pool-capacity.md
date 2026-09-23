# v22: the pool's capacity

The chapter that gives the expert pool the RAM the mini already has: what the
box holds that decode never reads goes back to the pool as slots, and the pool
learns to outgrow one Metal buffer so it can take them. Companion plan:
[v22-implementation-plan.md](v22-implementation-plan.md), whose checkboxes are
the status of record. The chapter is the first of the three venues Davor ruled
at v21's close ([v21-compression.md](v21-compression.md), S0.5: the pool's
capacity, then speculation revisited, then the attention row's fixed part); the
tree it starts from is `main` at `57c7f2f`. **Speculation revisited is filed in tt as SHRIKE-16 and the attention row's fixed part as SHRIKE-19 (2026-09-23).**

Every number in this document is labelled **measured** (a counter, a clock or a
footprint on the mini), **modelled** (arithmetic on measured inputs) or
**remembered** (an earlier chapter's record, cited, not re-run), and carries the
grade of the Method (M, T, C, R) where it is load-bearing.

## The problem

The mini's token at the v20 close is 54 ms with 14 to 17 misses per token at
0.73 to 0.80 ms per read (measured), the token's largest row. v21's step zero
found, beside the avenue it closed, that the pool's misses fall about twice as
fast as its capacity grows: 8 % fewer at 4 % more slots, 25 % at 12 %
(modelled by replay, grade T, the replay that priced v20's slot table within
its measured move). Compression was one way to buy capacity and could not pay.
RAM is the other, and the question this chapter opened on was whether the box
had any.

## What the box holds (read on the mini, 2026-09-18)

**The ledger under load (measured: a 7k request through the production server,
`footprint` and `vm_stat` after it; the process at 11 GB, the box at 13 % free
with 1.85 GB in the compressor and 0.9 GB of swap).**

| holder | resident | reads it at decode |
| --- | ---: | --- |
| Shrike's pool | 9.05 GB | yes: 5,120 slots plus the ring's nine cells in one arena |
| Shrike's dense weights | 1.33 GB | yes, mapped from `model_weights.bin` |
| Shrike's prefill scratch | about 0.6 GB | no: the 4,096-token chunk's private buffers, allocated at the first prefill and never released |
| Shrike's decode buffers | about 0.23 GB | yes: the KV, the GDN states, the readbacks |
| oMLX, idle 21 days | 2.9 GB, plus 2.8 GB swapped | no: the reranker (1.69 GB) and the embedder (0.15 GB), last used 2026-09-10, no idle timeout; the rest the residue of earlier chat-model loads (peak 12 GB) |
| the OS and its daemons | about 1.1 GB summed, less in truth | no GUI session; the largest daemon 40 MB |

The paging counters across a full request do not move (547 decompressions, 47
page-ins, no page-outs, no swap traffic: measured), so the pressure is static
and freed RAM converts into slots, nothing else.

**The prize per slot (modelled by replay on twelve traces, the production
pool's table with SLRU, grade T; `v22-s01-slot-prize.py` at
`~/.claude/handoffs/archive/shrike-v22-step0/`).**

| RAM to the pool | slots | decode misses per position | the cut |
| ---: | ---: | ---: | ---: |
| today | 5,120 | 24.2 to 30.1 | |
| 600 MB | 5,478 | 21.0 to 25.9 | 12 to 15 % |
| 1,200 MB | 5,831 | 18.4 to 21.6 | 22 to 29 % |
| 2,000 MB | 6,304 | 14.9 to 17.7 | 35 to 44 % |
| 2,500 MB | 6,603 | 13.2 to 15.8 | 43 to 52 % |
| 3,000 MB | 6,899 | 11.7 to 14.3 | 48 to 58 % |

In production terms a 3 GB pool row is 14 to 17 misses per token to 7 to 9,
about 4.5 to 6 ms per token (modelled), a tenth of the token, and class 1: no
value changes anywhere.

**The wall (remembered, v16's measurement).** The arena is one Metal buffer so
that a landing changes owner at a swap by index without a byte moving, and the
mini's device caps a single buffer at 8.88 GiB (`maxBufferLength`; the M4 Pro's
is 28.08). The pool at 8G is 8.45 GiB with the ring, 0.43 to spare: the freed
RAM can add about 250 cells before the arena needs what v16 named as that
day's fallback, the kernels given a second base.

## The design

**Task 1, the prefill scratch released.** `prefillChunked` drops the scratch in
a `defer` and `ensurePrefillScratch` reallocates at the next prefill, the
settle rewrite included; a runner test that a prefill, a decode, a reset and a
second prefill match pure decode with the scratch gone after each. The cost is
one allocation and first touch per request, measured on the mini by the turn
rig. Built and committed 2026-09-18 (`9e02f3b`). **Keeping the scratch across a short idle window is filed in tt as SHRIKE-42 (2026-09-23).**

The release is bounded to the success path, and the bound is worth stating.
Every span drains before `prefillChunked` returns (each chunk's tail, the shared
command buffer and the final head are waited), so nothing references the scratch
when the `defer` fires. A prefill that throws does not drain:
`PrefillRoutedTileSequencer.run` abandons its begun fetches and rethrows without
the pending-batch drain the success path runs, so up to `maxPendingDepth` (2)
committed command buffers can still hold the scratch as the error returns. That
is not a use-after-free, since Metal retains a command buffer's resources, but it
means a failed prefill followed at once by a retry can transiently hold two
scratches, about 1.2 GB, until the abandoned buffers complete. Found by the
close's review (2026-09-19) and recorded as a bound rather than fixed: an error
path is not what a chapter's close should be changing.

**Task 2, the arena in chunks.** `ExpertCellArena` allocates in chunks under
the device's `maxBufferLength` (or a size the runtime configuration sets, for
the tests), each its own `posix_memalign` and buffer, cells numbered globally
as before; it publishes a `PoolBases` struct, the chunks' GPU addresses and the
cells per chunk, in a small buffer. Only two kernels compute an address from
the pool's base, the speculative phase-1 and phase-2; they take `PoolBases` at
buffer 0 and address a cell as its chunk's base plus its offset within the
chunk, and their encoders mark every chunk resident. The classifier, the ring's
swap and the residency tables name cells by global index and do not change;
the hit and fixup paths already carry a buffer and an offset per expert and
ask the arena which buffer. The streamer keeps a cell's global offset as the
slot's identity and binds the chunk offset. The allowed slot counts, capped at
128, open to 256. Tests: the arena over three chunks with every cell's buffer,
offsets and pointer, the refusal past eight chunks, and the toy runner with its pool
forced across chunks decoding as the pool in one; the golden byte-identical on
both boxes, one chunk on this box and two on the mini.

One failure mode is recorded rather than changed. The `PoolBases` padding
entries duplicate the last chunk's address, which `ExpertCellArenaTests` pins
at entry 7, so a cell index past the arena resolves through `pool_cell_base`
to a valid address inside the last chunk and reads another expert's bytes
where the single-buffer pool would have run past its end and faulted. Nothing
the classifier or the host writes can reach such an index today. Leaving the
unused entries null would restore the loud failure, but it moves it from a
wrong answer to an unmapped dereference inside a Metal kernel, which on this
hardware is a failed command buffer at best and a driver stall at worst, so
the trade is not obviously the right way round. Raised by the close's review
(2026-09-19) as an observation and left as one.

**Task 3, the mini's configuration.** oMLX restarted (Davor's ruling: a
no-brainer, with the scratch release), the box's free RAM read, the budget
raised to the slot count the headroom allows (`--ram-budget` snaps to the
nearest allowed count; the table scaled to that total with no layer above its
256 experts), the arms on the four shapes against today's production
configuration, memory pressure watched, the golden byte-identical, production
relaunched at the winning configuration and the launch line in the repo's
working instructions updated.

**The embedder and the reranker** stay resident in oMLX for now; Davor's ruling
is that Shrike will serve them itself in a chapter of its own (the right
architecture: one process owning the box's RAM), which frees the rest of this
table. The on-demand alternative was priced and declined: a reload during a
tool call would compress the pool's idle pages and re-fault them at the next
turn, about what the slots save. **Filed in tt as SHRIKE-43 (2026-09-23).**

## Step zero

- **S0.1 The RAM ledger** (measured on the mini): above.
- **S0.2 The prize per slot** (replay): above.
- **S0.3 The wall** (remembered): above.

## Tasks

- **Task 1** the prefill scratch released: done, `9e02f3b`.
- **Task 2** the arena in chunks: built 2026-09-18 as designed. `ExpertCellArena`
  takes a chunk size (the device's `maxBufferLength` unless the runtime
  configuration's `expertArenaChunkBytes` says otherwise, a test-only
  parameter, not a knob), allocates ⌈cells / cells-per-chunk⌉ chunks of its own
  `posix_memalign` and buffer, at most eight, and publishes `bases`, the
  `PoolBases` struct the two speculative kernels take at buffer 0 (eight GPU
  addresses written from `MTLBuffer.gpuAddress`, the cells per chunk after
  them). `pool_cell_base` in moe.metal is the one address line. The
  streamer's `slotBufferOffsets` stays the global offset (three lookups key on
  it) and `slotChunkOffsets` is what the GPU is handed; the residency
  resources carry `poolBases` and `poolChunks`, the encoders mark every chunk
  resident, prefill's pool residency includes every chunk. The allowed slot
  counts run to 256. Tests: the arena over three chunks (every cell's buffer,
  chunk offset, global offset, pointer; the `bases` words), nine chunks
  refused, the fused-FFN kernel tests over a `poolBases` for their one-buffer
  pool, the landing tests through the per-cell accessors, and the toy runner
  with its pool forced across four chunks decoding token for token as the
  pool in one and prefilling across the boundary. Gates: release build with
  zero warnings, lint, links, 1,297 tests in 176 suites; the golden
  byte-identical on all five profiles on this box (one chunk here; the mini's
  two-chunk golden is Task 3's).
- **Task 3** the mini's configuration and the arms (2026-09-18; the scripts,
  logs, rows and traces at `~/.claude/handoffs/archive/shrike-v22-t3/`). The T2
  build deployed; oMLX restarted (its two models unloaded, `loaded_count 0`; the
  box went from 13 % free under load with 0.9 GB of swap in use to 43 % free
  idle with 0.23 GB); then the arms through `tools/decode-rig.sh`, two arms
  interleaved per shape, two lifetimes each: `base`, today's production
  configuration (8G, the v20 table of 5,120 with SLRU, one arena chunk), and
  `big`, 160 slots per layer (`--ram-budget 11324620800`, which snaps to 160;
  the table scaled to 6,400 with layers 0 and 1 at their 256 experts; SLRU;
  the arena in two chunks on the mini). Measured, the cold 512-token answer of
  each shape, both lifetimes:

  | shape | tok/s base → big | misses per token | io ms per token | the token ms | prefill s |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | the card | 18.23 to 18.25 → 20.17 to 20.18 (+10.6 %) | 15.6 → 8.7 (−44 %) | 11.6 → 6.7 | 54.8 → 49.6 | 11.2 → 11.3 |
  | the 300 | 17.75 to 17.96 → 20.63 to 20.64 (+15.6 %) | 16.5 → 8.0 (−52 %) | 12.2 to 13.0 → 6.1 | 55.7 to 56.3 → 48.5 | 5.5 to 5.7 → 5.6 |
  | the 1k | 18.18 to 18.30 → 19.93 to 20.28 (+10.2 %) | 15.0 → 9.0 (−40 %) | 11.1 → 6.8 | 54.6 to 55.0 → 49.3 to 50.2 | 8.0 → 8.1 |
  | the 7k | 17.89 to 17.92 → 19.67 to 19.70 (+9.9 %) | 14.2 → 7.6 (−46 %) | 10.6 → 5.8 | 55.8 to 55.9 → 50.8 | 37.8 → 37.9 to 38.0 |

  The two lifetimes of every arm agree to the tenth. The hit rate 0.948 to
  0.956 → 0.972 to 0.976; the word clock's layers sum 49 to 50 → 43 to 45 ms
  per token, the boundary 5.5 → 5.3; memory free 82 to 85 % after every arm,
  the swap untouched at 0.23 GB. The replay had priced 6,400 slots at 35 to
  44 % fewer misses; the box gave 40 to 52 (the replay holds no ring and no
  landings, so its cut was the conservative one). Prefill moved by 0.05 to
  0.2 s per request, the scratch's reallocation and first touch, 0.5 to 2 % of
  a cold prefill; a short warm turn pays a larger share, noted as a follow-on
  (keep the scratch across a short idle window). The `base` arm on the T2
  build sits 1 to 3 % under the v20 close's arms, the box's drift; the
  one-chunk arena's indirection is not visible in the layers' sum. **The follow-on is filed in tt as SHRIKE-42 (2026-09-23).**

  The golden on the mini at the two-chunk arena (`--expert-cache-slots 160`
  through the golden script's new `CLI_EXTRA_ARGS`, 6,400 cells plus the ring's
  nine over the device's 8.88 GiB limit): byte-identical on all five profiles.
  Production relaunched at the `big` configuration (the launch line in the
  repo's working instructions updated; the box at 93 % free idle).

## Method

As v21's: the four gates per commit; the golden byte-identical on both boxes
for every commit; the arms rig, two production lifetimes per shape on the four
shapes, read against today's production configuration; every model run on the
mini under the session's deploy leave (given 2026-09-18), production restored
and verified before the session ends.

## Numerics policy

Class 1 throughout: the chapter changes where bytes sit and how many experts
the pool holds, never a value a kernel computes.

## Risks

- **Pressure.** The pool's slots are wired while the GPU runs; a budget past the
  headroom sends the box into the compressor, which the ledger found static
  today. The arms carry `memory_pressure` and the swap counters.
- **The reallocation per request.** A short warm turn pays the scratch's
  allocation and first touch; the turn rig measures it.
- **Two chunks on the mini, one here.** The chunk boundary is exercised by the
  toy test and by the mini's golden only; the arms are the production check.
