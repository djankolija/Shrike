# v9 — Speculative routed dispatch

Decode's largest remaining cost is not a kernel. Per token, the GPU sits idle
~32.6 ms — about a third of the 100.8 ms body — in one recurring hole: between
the shared-expert command buffer finishing and the routed-MoE command buffer
starting. The hole is the per-layer host round-trip: the router's top-k is born
on the GPU, the CPU reads it back, resolves expert blobs, encodes and commits
the routed CB, and only then does the GPU resume. ~0.8–1.0 ms × 40 layers,
paid even when every expert is cached (~83 % of layers).

This design removes the CPU from the critical path of those all-hit layers.
The cache-miss path keeps its current shape and stays CPU-driven.

## Evidence this is the binding constraint

All measurements: ornith15 on the M1 mini, n=12, fixed prompt, temperature 0,
digest oracle `494bab3edb62` (see the measurement rig in the session handoff).

- Per-role CB timing (`SHRIKE_KERNEL_STATS`) attributes 32.6 ms/token of GPU
  idle to `shared_expert → moe_phase1_2_routed`, dwarfing every kernel.
- Named host counters (readback, cache plan, encode, exposed IO) explain only
  ~half of it; the rest is wake-up, commit→schedule, and executor-hop latency —
  irreducible per round-trip, removable only by not round-tripping.
- The encoder-merge change (`a8168d9`) cut GPU busy by 2.2 ms/token but the
  token got only 0.7 ms faster: the gap absorbed the rest. GPU-side savings are
  currently worthless at the margin; only removing the round-trip cashes them.
- Standing prediction to verify on landing: the absorbed ~1.4–1.5 ms must
  reappear on top of the gap reduction itself. If it does not, the pipeline
  model behind this design is wrong — stop and re-measure.

## Existing machinery this builds on (all validated on the target box)

1. **GPU residency classification** (`moe.encodeResidencyClassification`,
   `SHRIKE_DECODE_EXPERT_EXECUTION=gpu-residency`): classifies the router's
   top-k against a GPU-resident residency table inside the tail CB, emitting
   hit/miss counts and positions plus `resolvedSlots`/`resolvedGenerations`.
   Validated 2026-08-30: 54 400 classifications per run, zero fail-closed
   mismatches against the CPU plan, digest identical.
2. **Pool cache layout** (`SHRIKE_EXPERT_CACHE_LAYOUT=pool`,
   `PreadExpertStreamer`): the whole slot cache in one `MTLBuffer`; a blob's
   address is `poolBase + slot × poolSlotStride` — computable in-shader from a
   slot index. Prerequisite; validated separately (E5).
3. **Event-driven expert IO** (`SHRIKE_EXPERT_IO_SYNC=event`, deployed): the
   host no longer blocks on miss IO, so the miss path already tolerates
   asynchronous completion.

## Design

### All-hit fast path

At layer L's encode time — before the router has even run — the host also
encodes and commits, in order, on the main queue:

- **spec-phase1**: the existing phase-1 gate/up kernel, addressed through the
  pool (`slot × stride`) using `resolvedSlots` written by the classifier, and
  dispatched **indirectly** from arguments the classifier writes: the full
  grid when `missCount == 0`, zero threadgroups otherwise.
- **spec-phase2**: the down-projection + reduce, same indirect zero-or-all
  predicate.

On an all-hit layer the GPU therefore flows router → classifier → shared
expert → spec-phase1 → spec-phase2 with no host involvement. The 32 bytes of
routing never leave the GPU.

### Successor gating and the miss path

`attn(L+1)` must not run before layer L's MoE output exists, and on a miss
layer that output comes from a CPU-driven fixup committed *later* than
`attn(L+1)`. Same-queue commit order cannot express that, so:

- `attn(L+1)` encodes a wait on `MTLSharedEvent layerDone == L`.
- All-hit: the host — woken by the tail wait as today, but now *off* the
  critical path because the spec CBs are already executing — reads
  `missCount == 0` and signals `layerDone = L` from the host. Queue order
  already serializes spec-phase2 before `attn(L+1)`, so the signal only
  needs to beat nothing; it releases the successor immediately.
- Miss: spec CBs self-nullify (zero-size). The host runs today's fixup path
  (fetch → phase-1 for missing experts → phase-2) on a **second command
  queue**, and the fixup CB signals `layerDone = L` via `encodeSignalEvent`
  after execution. Cross-queue write→read on the residual is ordered by that
  event, the sanctioned Metal mechanism.

Deadlock note: `layerDone = L` is always signaled by exactly one party (host
on all-hit, fixup CB on miss), decided by the same `missCount` readback.

### Eviction safety

A spec CB reads slots chosen by the GPU one CB earlier; the CPU planner must
not evict those slots in the window between classification and execution.
Conservative v1 rule: the planner may not evict any slot it marked resident
for the previous or current layer until that layer's completion handler
retires it (an epoch counter per layer, retired on CB completion). The
`resolvedGenerations` cross-check stays compiled in for validation builds and
the fail-closed mismatch guard is retained on the miss path, where the CPU
still authorities the partition.

### Out of scope

Prefill, MTP, the dense-prefix and MLA (kimi) paths, and the barrier/hit-fixup
execution modes are untouched; the new behavior sits behind a new
`SHRIKE_DECODE_EXPERT_EXECUTION=speculative` value, leaving `hit-fixup` the
default until acceptance.

## Expected effect

E5 (pool layout, a prerequisite of this design) landed first and took a large
bite by itself: the per-slot→pool flip halved the gap (32.6 → 16.7 ms/token)
and sped up every CB — the unattributed round-trip latency was mostly Metal
residency management over ~3400 slot buffers. Post-E5 baseline: body 77.13.

What remains for the speculative path: the ~16.7 ms all-hit gap, the ~6.9 ms
hit→fixup stall, plus the ~1.4 ms absorbed by `a8168d9` (the standing
prediction). Miss layers (~6.8/token) keep a reconcile, and per-layer host
work (cache plan ~2.8 ms/token) still runs — concurrently instead of inside
the hole. A realistic target is body 77 → low 60s ms/token.

## Risks and open questions

- **In-order queue assumption.** The design leans on same-queue commit-order
  execution for spec CBs vs `attn(L+1)`. Verified conceptually against Metal's
  documented in-order queues; stage S3 includes a micro-test before the event
  plumbing is trusted.
- **Pool allocation size.** One contiguous ~6 GB `posix_memalign` +
  `bytesNoCopy` wrap on a 16 GB box — validated by E5 (allocation, perf,
  digest, memory pressure all clean). Pool allocation failure aborts startup;
  a default flip needs a graceful per-slot fallback first.
- **Indirect-dispatch overhead.** The classifier gains a few extra stores;
  measured as part of S2's equivalence run.
- **Why per-slot is the default today** is not recorded anywhere. E5 answers
  the perf half (pool is far faster here); the remaining unknown is whether
  per-slot exists for boxes where a contiguous pool cannot be allocated.

Implementation stages, gates, and status of record:
[v9-implementation-plan.md](v9-implementation-plan.md).
