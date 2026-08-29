# Architecture

How the engine is built and why. Distilled from the v4 core design; every invariant
below was re-checked against `sources/` rather than carried on trust.

The product claim is bounded memory: run a mixture-of-experts model far larger than
RAM by streaming routed experts from disk, at a memory ceiling the operator declares
rather than discovers.

## 1. RAM budget is an input, not an outcome

The engine takes a declared budget and derives everything from it — slot counts per
layer, prefetch depth, KV reservation — and refuses a budget too small to hold the
resident tensors.

`--ram-budget <size>` (`ServerArguments.swift`) accepts `8G`, `512M`, or a byte
count, validated through `RuntimeConfiguration.parseBudgetBytes`. The server config
carries it as `ram_budget`.

This inverts the older design, where slot count was the knob and memory was whatever
fell out of it.

## 2. Streaming that genuinely uses the disk

Routed experts live on disk and are read on demand into a bounded per-layer slot
cache. Reads go through `ParallelExpertReader`, which fans out over
`DispatchQueue.concurrentPerform`.

**The often-quoted "0.62 GB/s against 3.2 available" is wrong and was retracted.**
It divided expert-read bytes by a whole phase that was mostly GPU tile execution
rather than fetch. The real rates are ~2.83 GB/s at prefill and ~2.6 GB/s at decode,
against a measured device ceiling of 3.92 GB/s — 66–72%, not 16%. There is roughly
1.4x of streaming headroom, not 6x. `ParallelExpertReader`'s own header states this.

Two consequences worth keeping in mind:

- Four reader threads is the knee; eight saturates at 3.92 GB/s.
- At decode's actual batch size the pool buys nothing. With 128 slots the hit rate is
  ~92%, so a layer fetches about one expert, and one read is one read (batch 1: pool
  3.41 GB/s vs serial 3.44). The pool earns its place at 4–8 misses per batch, which
  is what makes a *smaller* cache viable — a different claim from making today's
  configuration faster.

I/O threads are the one CPU load that is safe alongside decode: under a saturating
`pread` load GPU-busy per token rose 7.4%, against 44.9% under an equivalent CPU
compute load. They block in the kernel instead of burning ALU.

## 3. No per-layer CPU round trip in decode

The expensive transition is the CPU having to see routing before experts can be
dispatched. It is split in two:

- **Dispatch** needs no readback. The MoE kernel reads expert ids from the router's
  own output buffer through an argument buffer covering the resident slots.
- **Residency** cannot disappear while streaming — something must decide what to
  fetch — but it moves off the critical path. The kernel processes resident experts
  immediately (`moe_phase1_gate_up_act_subset_u16load`) and writes a miss list; the
  I/O pool services it asynchronously; a fixup pass completes the stragglers before
  the phase-2 reduce (`moe_phase2_down_reduce_k8`).

At a ~92% hit rate the synchronous stall becomes an 8%-of-layers event rather than
every layer. This is the hardest part of the design and the reason the engine was
rewritten rather than patched.

## 4. C99 for hot loops, Swift for structure

Swift owns lifetime, actors, and orchestration. Metal owns the GPU. `NVMAIKernelsC`
owns anything with a per-weight inner loop — it is deliberately small, 439 lines
across `int4_affine_gemv.c` and `expert_io.c`.

The split is measured, not stylistic: moving the int4 GEMV to C99/NEON was 2.9x, and
hoisting a redundant per-group sum added a further 16–20%. Swift's `SIMD8<Float>`
does not lower to vector loads. It is not a blanket rewrite — Swift costs ~4.6 ms of
a 47 ms token, and most of that is Metal API calls C would pay identically.

## Predictive prefetch: two different schemes, one of them disproven

The core design contains a section titled *Predictive prefetch: PROVABLY CANNOT
WORK*, while `ExpertPrefetchRing.swift`, `NVMAI_PREDICTIVE_PREFETCH`,
`NVMAI_PREFETCH_TOP_M` and `NVMAI_PREFETCH_TRACE` all exist. This is not a
contradiction — the proof and the code are about different predictors.

**What the proof disproves** is prefetching layer L's experts from the *previous
token's* routing at the same layer. One token touches 8 experts per layer, so a
cache of 16 or more can never be evicted by a single token; an expert used at layer
L in token N−1 is therefore still resident at token N. The predictable set and the
miss set are disjoint by construction. Replayed against a real 383-token routing
trace, the share of misses such prediction would catch is 0.00% at both 16 and 128
slots. The 38% token-to-token reuse is real but already fully exploited by the cache.

**What the code implements** is different: while executing layer L it loads layer
L+1's router and runs it on the current hidden state, staging the speculative top-k
in a ring *outside* the authoritative cache. Entries become resident only when the
exact router later selects them; wrong predictions are discarded without touching
cache mappings. Layer L+1's experts are a different set with no residency guarantee,
so the proof does not reach this scheme — and measurement agrees: the model-aware
next-router probe recalls 64.1% of actual nonresident misses at top-8 (56.4%
precision), against 3.25% for a transition-only fallback.

**It is off by default because it did not pay end-to-end**, not because it cannot
predict. The interleaved run gave −3.9% at 4-bit and +7.8% at 8-bit against a +10%
promotion bar. The limiting issue is lead time: a single next-layer prediction often
starts too late, and its reads can still consume SSD service while the following
demand work begins. The next eligible experiment is two-stage prediction — a smaller
earlier set, then refinement — measured with explicit prefetch completion timestamps
and demand-join telemetry. Do not promote the current path without that evidence.

Output correctness was verified when it was built: greedy 4-bit output matched the
disabled run byte-for-byte, and the 4-bit golden baseline passed.
