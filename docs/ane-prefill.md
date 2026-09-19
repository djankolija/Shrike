# Shrike ANE prefill (experimental, opt-in)

> ⚠ **Inherited document, not a Shrike record. Take every number here with a grain
> of salt.** This file came in with the fork from the turbo-fieldfare lineage and
> predates every chapter in this directory. It does not follow the
> `vN-<subject>.md` plus implementation-plan convention the real chapters use, no
> chapter's checkboxes cover it, and **nothing in it was measured on either of this
> project's boxes** — the result below is an M3 with 24 GB, while the deploy target
> is the M1 mini with 16 GB and this box is an M4 Pro.
>
> Read it as a list of hazards to re-measure, not as findings. Specifically
> unverified here: the 2.31x end-to-end and 26.7x offloadable-block speedups, the
> per-layer-kind time split, the one-resident arena policy with its decode-collapse
> figure, and the per-layer-chunk reload cost. The hazards themselves are still
> worth knowing: fixed enumerated shapes with chunk-aligned history, the Core ML
> arena's RAM cost, per-layer model reload, fp16 divergence from the GPU path, and
> the fused SDPA op's NaN/inf above sequence 2048.
>
> Any re-measurement has to account for v22
> ([v22-pool-capacity.md](v22-pool-capacity.md)): the mini now runs an 11.33 GB
> expert pool on a 16 GB box at 15 to 17 % free, so a ~1 GB Core ML arena comes
> straight out of expert slots. The trade that doc never had to price — attention
> time saved against pool slots lost — is the first thing an ANE chapter would owe.

Routes the prefill attention block of every full-attention layer through the
Neural Engine via a Core ML sidecar. GDN layers, the MoE, the `.gturbo`
format, the KV cache, the server API, and all of decode are untouched. Off by
default; nothing changes without `SHRIKE_PREFILL_ANE=on`.

On a real 6,103-token 4-bit prefill the ten full-attention layers cost 84.3 s
of 133.2 s (63.3%, growing quadratically with prompt length), the ANE runs
the same blocks 26.7x faster with the model's real weights, and the
inexpressible Gated-DeltaNet share is only 10%. The measurements behind those
numbers are in [Research](#research) below.

## Using it

```bash
# one-time export of the sidecar (~540 MB, fp16, all full-attention layers)
~/.venvs/coreml-py311/bin/python tools/export_ane_prefill.py \
    --model models/ornith-1.5_35B_A3B_4Bit --max-history 12288

SHRIKE_PREFILL_ANE=on .build/release/ShrikeServer --model ... # or ShrikeCLI
```

`SHRIKE_PREFILL_ANE` accepts `off|on` and fails closed on anything else. With
`on` and no sidecar present, the runner fails at load with the export
command. The first request per machine pays a one-time ANE specialization
per function (~130 s across all variants), cached by the OS thereafter;
the first request per *process* pays ~0.5 s per layer-chunk of model load.

## Measured result (M3, 24 GB, 4-bit, 6,103-token prompt, greedy, cache off)

Interleaved gpu/ane/ane/gpu, fresh server per run, one discarded warmup per
arm (`tools/ane-probes/shrike_ane_prefill_ab.py`):

| | prefill median | runs | decode after prefill |
| --- | ---: | --- | ---: |
| GPU path | 132.90 s | 132.85 / 132.95 | 8.70 tok/s |
| ANE path (warm OS cache) | **57.52 s** | 57.58 / 57.46 | 8.68 tok/s |
| | **2.31x** | | unchanged |

Each arm's greedy output is internally deterministic (stable digest across
its runs); the two arms differ as documented below, and both continuations
open with the same coherent sentence. Consistent with the rehearsal
projection (52.1 s) plus marshaling and per-layer model reloads. The speedup
grows with prompt length because the offloaded share is the quadratic one.

## Design

- One multifunction `.mlpackage` per full-attention layer; functions
  `h0/h4096/h8192/h12288` share a single fp16 weight blob (~53 MB/layer) and
  differ only in KV-history length. History is always chunk-aligned because
  the runtime's chunks are, so fixed enumerated shapes keep the ANE scheduler
  on its fast path.
- The block consumes the post-input-norm hidden chunk and token-major fp16
  K/V history, and returns the attention output plus the chunk's cache-layout
  K/V. `copyPrefillKVToCache` then quantizes exactly as on the GPU path, so
  decode sees an ordinary cache.
- Later chunks attend to an fp16 shadow of earlier chunks' K/V (allocated
  only for multi-chunk prompts, ~33 MB/layer), not to a re-dequantized cache.
- **Decomposed attention, never the fused SDPA op**: the fused
  `scaled_dot_product_attention` produces NaN/inf on this M3's ANE from
  sequence length 2048 even at score std 0.25. The causal mask is an input
  (a baked or generated constant gets const-folded to 64 MB per function),
  additive −30000 rather than −inf.
- **At most one Core ML model is resident at a time, and it is released
  before decode.** This is a hard requirement, not a preference: each loaded
  function pins an E5RT/ANE inference arena (~1 GB of score tensors for the
  h4096 variant), and an early build that kept all twenty resident pressured
  the 8 GiB expert slot cache out of RAM — the server's RSS fell to 2.85 GB
  and decode collapsed to 1.9 tok/s with 200 ms/token of expert I/O waits.
  With the one-resident policy the same request decodes at 9.6 tok/s and the
  reload costs ~0.5 s per layer-chunk, already included in the numbers above.

## Fallbacks (never silent mid-request; continuity is enforced)

The GPU path serves, unchanged: short single-chunk prompts (padding a 5-token
prompt to 4,096 costs more ANE time than the whole GPU prefill); prompts
beyond the exported history variants; non-4096 chunk configs; prompt-cache
resumes at unaligned positions; the MTP verify and MTP adapter chunks; and
any request whose earlier chunks already fell back (a chunk may only run on
the ANE if every prior chunk's shadow rows exist).

## Not byte-identical, by construction

The sidecar computes in fp16 with a different reduction order — measured ~1%
mean per-layer deviation against an fp32 reference of the same weights.
Greedy output on the smoke prompt matched the GPU path token-for-token, and
each arm is internally deterministic, but long generations can diverge in
low-probability positions. This is why the switch exists and defaults off,
and why `tools/golden-baseline.sh` runs with it off. Promotion to default
would require its own quality qualification, not just the speed number.

## Research

The measurements that retired every risk before the Swift integration was
written, on a real 6,103-token 4-bit prefill (133.2 s, 97.3% occupancy).

**The layer mix did not kill it — the opposite.** This architecture is 10
full-attention + 30 Gated-DeltaNet layers, and the original ANE analysis was
written for full-attention blocks only. Splitting the prefill instrumentation
by layer kind (`prefill_attn_router` vs `prefill_gdn_router`):

| | GPU s | share |
| --- | ---: | ---: |
| 10 full-attention layers (ANE-expressible) | **84.3** | **63.3%** |
| MoE tiles + shared + reduce (stays GPU by design) | 33.9 | 25.5% |
| 30 GDN layers (recurrent scan, not expressible) | 13.3 | 10.0% |

The quadratic SDPA makes the expressible share dominant *and growing with
prompt length*; the inexpressible share is 10%.

**The block is fully expressible and correct.** The complete block — packed
QKV with output gate, per-head q/k RMS norms, NeoX-subdim RoPE (64 of 256),
GQA 16/2 SDPA against KV history, sigmoid gate, O projection — built in MIL
(`tools/ane-probes/shrike_ane_attention_probe.py`) matches a float32 NumPy
reference at fp16-noise level.

**One real ANE defect found and routed around:** the fused
`scaled_dot_product_attention` op produces NaN/inf on this M3's ANE from
sequence length 2048, even at score std 0.25. Decomposed attention
(matmul+softmax+matmul) is clean and slightly faster (isolated A/B: rel err
inf vs 0.007, 58.4 vs 50.0 ms at 2048). The integration uses the decomposed
form.

**Long-sequence fp16 softmax precision is a non-issue on real
distributions.** The 8–12% rel err seen with uniform random attention at
seq 6144 collapses to **0.0002** with realistically peaked scores (std 3.7,
max 80).

**Real-weight rehearsal** (`tools/ane-probes/shrike_ane_realweight_rehearsal.py`):
the actual int4 affine weights of all 10 full-attention layers, dequantized
and baked into per-layer Core ML programs, replaying the exact 6,103-token
chunk sequence (4096:0 then 2007:4096 per layer), prediction wall including
marshaling:

| | 20 layer-chunks |
| --- | ---: |
| ANE (CPU_AND_NE, measured) | **3.15 s** |
| GPU (measured, same shapes) | 84.3 s |
| speedup on the offloadable block | **26.7x** |
| projected end-to-end prefill | 133.2 s → **52.1 s (2.56x)** |
| worst per-layer rel err vs fp32 | 0.0101, zero NaN/inf |

### Running the probes

`shrike_ane_attention_probe.py` and `shrike_ane_realweight_rehearsal.py` are
self-contained. `shrike_ane_prefill_ab.py` is **not**: it imports all of
`shrike_gate0_profile.py` plus five names from `shrike_profile.py`
(`DEFAULT_API_MODEL`, `ROOT`, `benchmark_log_path`, `server_command`,
`server_environment`), two harness modules that were not carried over.
Restore them from the import commit before running it:

```bash
git show <import-commit>:benchmark/nvmai_profile.py > tools/ane-probes/shrike_profile.py
git show <import-commit>:benchmark/nvmai_gate0_profile.py > tools/ane-probes/shrike_gate0_profile.py
```

They were left out deliberately: 24 KB of general gate-0 profiling machinery
for a script nobody runs today. Do not vendor a shim instead —
`gate0.preflight()` is the never-run-two-model-processes guard, and
reimplementing a safety check is worse than restoring the real one.

## Known costs and future work

- Sidecar: ~540 MB disk (fp16 weights ×4 function variants after dedup);
  not yet covered by the install receipt.
- Per-layer-chunk model reload (~0.5 s, serialized with compute). A
  next-model preload on a background thread while the GPU runs the MoE stage
  would hide most of it — unbuilt, worth ~15–20% of the remaining prefill.
- 8-bit models reuse the same exporter unchanged (attention weights
  dequantize to the same fp16 shapes); qualification for 8-bit not yet run.
- Short-prompt variants (chunk 1024) would extend coverage below 4,096
  tokens; unbuilt.
