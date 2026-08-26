# v5 implementation plan: gpt-oss-20b and Kimi-Linear-48B-A3B

> **For agentic workers:** execute task-by-task in order within a phase; phases A → B → C.
> Each task ends buildable (`swift build`) with its named tests passing
> (`swift test --no-parallel --filter <suite>`). Steps use checkbox syntax for tracking.

**Goal:** the engine loads and correctly runs gpt-oss-20b and Kimi-Linear-48B-A3B from their
affine int4/g64 MLX checkpoints, alongside the existing qwen36 family.

**Architecture:** flat checkpoint names are normalized at repack time into the engine's internal
`language_model.model.` convention (and Kimi's split KDA tensors are fused into the engine's
existing GDN layouts), so the runtime keeps one naming scheme; per-architecture behavior enters
through new `ArchConfig` baselines plus a small number of parameterized kernel variants (sinks,
additive biases, clamped SwiGLU, sigmoid routing, per-channel decay, MLA-as-MQA). Chat framing
enters exclusively through the `ChatDialect` seam added in step 2.

**Tech stack:** Swift 6.3 / SPM, Metal 4, swift-transformers Tokenizers, swift-testing.

**Spec:** `docs/v5-second-architecture-objective.md` (objective and step list),
plus the reference semantics pinned below — treat those as normative where the spec is silent.

## Global constraints

- `swift test --no-parallel`; unit tests never load a real model (AGENTS.md).
- `tools/lint.sh` gates: no `as!`/`try!` under `sources/` without `lint:allow-force`, no new
  function over 120 lines.
- CI builds `swift build -c release` warning-free.
- Existing qwen36 behavior must not change: the golden baseline
  (`tools/golden-baseline.sh --check 4`) is the final gate for any runtime-path change, run at
  deploy time on a machine with an installed model.
- Kernel changes ship with a differential test against a CPU reference in
  `NVMAIValidationSupport` (`RelError` vs a `Tolerance` constant), same shape as
  `RMSNormTests.swift`.

## Normative reference semantics (pinned from mlx-lm + checkpoint inspection)

These were verified against `mlx_lm/models/{gpt_oss,kimi_linear,gated_delta,rope_utils,
switch_layers,base}.py` and both checkpoints' `config.json` / safetensors indexes on 2026-08-26.

**gpt-oss-20b** (`InferenceIllusionist/gpt-oss-20b-MLX-4bit`, affine int4 g64 uniform):
24 layers, hidden 2880, 64 Q heads / 8 KV heads, head_dim 64, alternating
`sliding_attention`(window 128) / `full_attention` starting sliding at layer 0. 32 experts,
top-4, intermediate 2880. Additive biases on q/k/v/o, on the router, and on every expert
projection (`.bias` — distinct from the quant zero-point `.biases`). `self_attn.sinks` is a
learned per-Q-head logit: it joins the softmax as one extra term in max and denominator and
contributes no value row. Router: top-4 over raw logits (bias included), then softmax over the
selected 4. Activation: `out = (min(g,7)·σ(1.702·g)) · (clamp(u,±7)+1)` where g = gate_proj
half, u = up_proj half. RoPE: YaRN over the full head_dim — factor 32, original 4096, theta
150000, mscale = 0.1·ln 32 + 1 ≈ 1.3466 applied to q and k before rotation. rms_norm_eps 1e-5,
vocab 201 088, no shared expert, no QK-norm, no attention output gate. Sliding mask: key j
visible from query i iff j ≤ i and j > i − 128.

**Kimi-Linear-48B-A3B** (`mlx-community/Kimi-Linear-48B-A3B-Instruct-4bit`, affine int4 g64;
`mlp.gate` routers 8-bit g64): 27 layers, hidden 2304. Layer 0 dense (SwiGLU 2304→9216).
MLA on 1-indexed layers {4,8,12,16,20,24,27}; KDA on the other 20.
- MLA (NoPE — **no RoPE anywhere on these layers**): per head q = q_proj → [128 nope | 64 pe];
  kv_a_proj_with_mqa → [512 latent | 64 k_pe]; latent RMS-normed (eps 1e-5). Absorbed form:
  score(i,j) = (W_UKᵀ q_nope)·latent_j + q_pe·kpe_j, scale 192^−0.5; output = Σ p·latent_j then
  W_UV per head; o_proj. W_UK/W_UV come from splitting `kv_b_proj` rows per head at nope=128.
- KDA: q/k/v = 2304→4096 each; per-channel depthwise causal conv k=4 then SiLU; q,k
  RMS-normalized per head-dim (eps 1e-6/128) with q scaled 1/128 and k scaled 1/√128;
  `beta = σ(b_proj)` per head; decay `g = exp(−exp(A_log[h]) · softplus(a[h,dk] + dt_bias[h,dk]))`
  **per channel**, a = f_b_proj(f_a_proj(x)) (2304→128→4096); delta rule identical to the
  existing GDN kernel with `state[i] *= g[idx]` replacing the scalar decay; output
  `rmsnorm(y; o_norm.weight, per-head) · σ(g_b_proj(g_a_proj(x)))` — sigmoid gate, not SiLU;
  o_proj. State [32, 128, 128] fp32 — same as Qwen's layout.
- MoE (26 layers): 256 experts top-8, intermediate 1024; scores = σ(logits); selection by
  scores + `e_score_correction_bias` (f32); weights = **original** scores of the selected,
  ÷ (sum + 1e-20), × 2.446; plus one ungated shared expert (SwiGLU 2304→1024). Standard SiLU
  SwiGLU, no expert biases. n_group = topk_group = 1, so grouped top-k degenerates to plain.
- Tokenizer: **no `tokenizer.json` exists** upstream; `tiktoken.model` + custom class. A
  conversion artifact is required (Task C1).

**Harmony** (gpt-oss chat): system message fixed-form (identity, cutoff, date, "Reasoning:
{effort}", valid-channels line); user "system" role maps to the **developer** message
("# Instructions"); tools render as a TypeScript `namespace functions` block in the developer
message. Assistant turns: `<|start|>assistant<|channel|>final<|message|>…<|end|>` (previous
turns; thinking dropped), tool call
`<|start|>assistant to=functions.NAME<|channel|>commentary json<|message|>{args}<|call|>`
with its analysis (`<|channel|>analysis`) re-rendered while inside a tool loop; tool result
`<|start|>functions.NAME to=assistant<|channel|>commentary<|message|>` + content-as-JSON +
`<|end|>`. Generation prompt `<|start|>assistant`. Stop tokens: `<|return|>` (final answer)
and `<|call|>` (tool call); `<|end|>` is **not** a stop. Parser must accept `to=` on either
side of `<|channel|>`.

**Kimi dialect:** `<|im_user|>user<|im_middle|>…<|im_end|>`, assistant
`<|im_assistant|>assistant<|im_middle|>`, other roles `<|im_system|>{role}<|im_middle|>`;
tools declared as `<|im_system|>tool_declare<|im_middle|>{tools-json}<|im_end|>`; tool calls
`<|tool_calls_section_begin|><|tool_call_begin|>{id}<|tool_call_argument_begin|>{args}<|tool_call_end|>…<|tool_calls_section_end|>`;
tool results are role `tool` rendered as system with body `## Return of {id}\n{content}`.
No thinking channel. Stop: `<|im_end|>` / `[EOS]`.

## Locked decisions

1. **Normalize names at repack, not at runtime.** `RepackPlanner` rewrites flat `model.…` names
   to `language_model.model.…`; gpt-oss `mlp.experts.*` → the internal routed-expert container;
   Kimi KDA q/k/v (+convs) are row-concatenated into the engine's fused
   `linear_attn.in_proj_qkv` / `linear_attn.conv1d.weight` layouts; Kimi `b_proj` →
   `linear_attn.in_proj_b`; `kv_b_proj` is dequant-split-requantized into `embed_q` /
   `unembed_out` at repack. Kimi-only tensors keep their own (prefixed) names.
2. **MLA runs as MQA-576/512.** One cache row per token: [latent 512 | k_pe 64] fp16. Per-head
   query [W_UKᵀ·q_nope (512) | q_pe (64)]. Attention kernel variant with qkDim ≠ vDim where V is
   the 512-prefix of the K row. Absorbed form for decode *and* prefill (correctness first; the
   materialized prefill optimization can come later). MLA layers use fp16 KV rows regardless of
   the configured KV precision, and mask value `3` in the layer mask.
3. **Per-arch behavior via `ArchConfig`** literals (`.gptOss20b`, `.kimiLinear48bA3b`) selected
   through the manifest: `arch.family` becomes a real manifest field read by `ManifestReader`
   (heuristic retained as fallback for old packs). `expecting:` at server/CLI call sites is
   resolved via `peekIdentity`, not hardcoded.
4. **YaRN becomes arch-drivable**: an optional arch-level rope-scaling descriptor constructs
   `YaRNRoPEParameters` independently of the user context-extension mode (which remains for
   qwen36 unchanged).
5. **Additive biases** are new optional buffers with epilogue adds: an `Elementwise` bias-add
   for attention projections; three new per-expert pack roles (`gate_bias`/`up_bias`/
   `down_bias`) with adds in the MoE phase-1/phase-2 kernels; an `[E]` logit-bias read in both
   router kernels.
6. **Dialect detection is ordered most-specific-first**: `<|channel|>` → harmony;
   `<|im_middle|>` → kimi; `<|im_start|>` → chatml; anything else rejected at load. Required
   token sets and streaming-decoder marker sets become per-dialect. Harmony and Kimi rendering
   are hand renderers behind `applyChatTemplate`'s exhaustive switch (same as chatml), with
   golden tests derived from the shipped Jinja templates.
7. **Dense layer 0 (Kimi)** reuses the shared-expert kernel path with a per-layer intermediate
   size; the packed-experts layout gains explicit per-layer expert counts so a layer may have
   zero routed experts (format minor bump; old packs decode unchanged).
8. **Thinking replay (Harmony):** the server gains `reasoning_content` passthrough (out on
   responses, accepted on input assistant messages) so a client tool loop can replay analysis;
   the renderer uses it exactly where the Jinja template does.

---

## Phase A — shared foundation (spec steps 3–4)

### Task A1: config loading accepts both flat configs
**Files:** `sources/NVMAIRepack/Core/Format/ArchInfo.swift`,
`tests/NVMAIRepack/…ArchInfoTests` (extend).
`ArchInfo.load`: make `text_config` optional (fall back to root object); add `model_type`
branches `gpt_oss` and `kimi_linear` with their own field readers (gpt-oss: `layer_types`
including `sliding_attention`→0, `sliding_window`, `rope_scaling{factor,original…}`,
`swiglu_limit`; Kimi: `linear_attn_config` (1-indexed `full_attn_layers`/`kda_layers` → mask
values 3/2), `first_k_dense_replace`, MLA dims, `moe_*` router fields,
`routed_scaling_factor`, `num_shared_experts`). New `RepackModelFamily` cases `gptOss20b`,
`kimiLinear48b`. The qwen35 cross-check stays qwen-scoped.
**Test:** load fixture configs (committed JSON snippets of the two real config.json files) →
assert every derived field; assert an unknown model_type still throws.

### Task A2: classify + normalize flat names
**Files:** `sources/NVMAIRepack/Core/Planning/RepackPlanner.swift`, planner tests.
Per-family classify: for the two new families accept `model.…` (no prefix), classify
`mlp.experts.*`(gpt-oss) / `mlp.switch_mlp.*`(Kimi) as routed-expert roles — gpt-oss adds the
three additive-bias roles — everything else `.lmResident`; extend `residentDestinationName` to
prepend `language_model.` and apply the D1 renames. Per-family slot-rank tables for resident
ordering. Skip `model.mtp*` if present.
**Test:** synthetic name lists for both families classify with zero `.unknown`; destination
names carry the internal prefix; qwen names unchanged.

### Task A3: decode-path MoE top-k generalization
**Files:** `sources/NVMAI/Kernels/MoE/MoE.swift`, `sources/NVMAI/Metal/MoE/moe.metal`,
`tests/NVMAI/Core/Kernels/MoE/*`.
Selector kernel loops over runtime `top_k ≤ 8` (arrays stay `[8]`); `FC_ROUTER_TOP_K` actually
consumed; phase-2 reduce dispatches `topK` simdgroups (`topK*32` threads) with `partial[topK]`;
preconditions `== maxStreamedExperts` become `<= maxStreamedExperts`; `realDecodeTopK` comes
from `ArchConfig.topKExperts`. Also raise `kMoEXMaxD` 2816 → 2880 (gpt-oss hidden) with a host
precondition. Prefill router already generic — no change.
**Test:** differential router test at k=4 and k=8 vs a CPU top-k+softmax reference; phase-2
reduce at k=4 vs CPU.

### Task A4: manifest family + expecting-parameterization
**Files:** `sources/NVMAIFormat/GTurboManifestV1.swift`,
`sources/NVMAI/Infrastructure/ModelIO/ManifestReader.swift` (+`ModelTypes.swift` registry),
`sources/NVMAIServer/Core/ServerInference.swift`, `sources/NVMAICLI/…Run.swift`,
`sources/NVMAIApp/Core/Inference/RealInferenceClient.swift`, reader tests.
`ManifestArch` decodes `family` (optional; fallback = existing heuristic); mask alphabet
accepts `3`; `knownArchitectures` gains the two new baselines (added in B1/C2);
`peekIdentity` returns the family and every `expecting:` call site resolves the `ArchConfig`
from it instead of hardcoding `.qwen36_35B_A3B`.
**Test:** manifest fixtures with/without `family` decode to the right `ArchConfig`; qwen
fixtures unchanged.

### Task A5: per-layer expert counts in the pack format
**Files:** `sources/NVMAIFormat/GTurboPackedExpertsLayoutV1.swift`,
`sources/NVMAIFormat/GTurboManifestV1.swift`,
`sources/NVMAI/Infrastructure/ModelIO/PackedExpertsLayout.swift`,
`sources/NVMAI/Runtime/Inference/Model.swift` (`openLayerLocked`), format tests.
Reconcile the planner's empty-`LayerFilePlan` path with the validator: a layer entry may carry
zero experts and no layer file; `arch.numExperts == expertsPerLayer` assert becomes
per-populated-layer; `openLayerLocked` guards the empty case. Verify against the real
constraint found in code (two exploration reports disagreed) and keep the minimal change.
**Test:** layout fixture with one empty layer round-trips and validates; existing fixtures
unchanged.

### Task A6: dialect scaffolding made per-dialect
**Files:** `sources/NVMAI/Tokenization/Tokenizer.swift`,
`sources/NVMAI/Tokenization/Detokenizer.swift`,
`sources/NVMAI/Tokenization/StructuredAssistantDecoder.swift`, tokenizer tests.
`resolveDialect` ordered per D6. `resolveChatMLTokens`/`validateStreamingDecoder`/
`knownChatMLTokens`/`stopTokenIDs`/`generationSuffix` fallbacks become per-dialect (chatml sets
unchanged); `StructuredAssistantDecoder.consume` dispatches on dialect. Fix the padded
`vocabSize` floor to be dialect/arch-appropriate. `GFByteLevelDecoderConfiguration` unchanged
(all three tokenizers are ByteLevel-decodable once converted).
**Test:** existing ChatML suites stay green; a fixture tokenizer with none of the three marker
sets fails with the dialect error, not a ChatML-token error.

## Phase B — gpt-oss-20b (spec step 5)

### Task B1: `ArchConfig.gptOss20b` + repack end-to-end on a synthetic checkpoint
**Files:** `sources/NVMAI/Infrastructure/ModelIO/ModelTypes.swift`,
`sources/NVMAIRepack/Core/Remote/RemoteStreamingRepacker.swift` (quant-slot suffix table),
`tests/NVMAIRepack/Core/Support/SyntheticSnapshot.swift` (gpt-oss variant), repack tests.
Baseline per the pinned numbers (mask alternating 0/1, window 128, scale 0.125, no shared
expert, no output gate, sinks + biases flags, swiglu limit/alpha, yarn descriptor). Routed
blob layout grows to 12 slices (adds `gate_bias`/`up_bias`/`down_bias`, BF16). Synthetic
flat-name checkpoint repacks; layout + manifest validate.

### Task B2: expert additive biases + clamped SwiGLU in the MoE kernels
**Files:** `moe.metal`, `prefill.metal`, `MoE.swift`, `PrefillRouter.swift`/prefill MoE hosts,
`sources/NVMAI/Kernels/CPU/CPUExpertFFN.swift`, `ModelExpertIO.swift` (+`MoEExpertOffsets`),
`sources/NVMAIValidation/Support/Reference/MoE/Moe.swift`, MoE tests.
Three new offsets threaded to phase-1/phase-2 (decode and prefill); activation becomes an
FC-selected two-argument form implementing the pinned clamped formula; router kernels gain an
optional `[E]` additive logit bias. CPU references updated to match; differential tests at
gpt-oss shapes (D 2880, F 2880, E 32, k 4) and qwen shapes (regression).

### Task B3: attention — biases, sinks, SWA wiring, 64 Q heads
**Files:** `sources/NVMAI/Kernels/Attention/Attention.swift`, `attention.metal`,
`PrefillAttention.swift`, `prefill.metal`, `Elementwise.swift`+`utility.metal` (bias add),
`sources/NVMAI/Runtime/Inference/RealForwardRunner.swift` (new non-gated, non-QK-norm decode
branch), `Model.swift` (sinks + bias accessors, schema), KV/RealForwardRunner SWA dispatch,
Attention CPU reference, attention tests.
Scratch/preconditions sized from `cfg` (64 Q heads); sinks buffer bound into
`attention_decode_combine` (extends running max, adds `exp(sink−m)` to D, numerator untouched)
and into `attention_prefill_causal_tiled` (lift row_max, rescale, add to row_sum); new decode
branch: QKV GEMVs + bias adds + RoPE (YaRN per D4) + `encodeFull`/`encodeSWA` by mask + o_proj
+ bias. Prefill uses the existing per-row window clamp. ANE prefill stays qwen-gated.
**Test:** differential decode+prefill attention with sinks vs updated CPU reference (including
sink-dominates-max and window-128 cases); qwen shapes regression.

### Task B4: Harmony dialect — tokenizer, renderer, decoder, parser
**Files:** `Tokenizer.swift` (`.harmony` case: token resolution, stop set
{`<|return|>`,`<|call|>`}, renderer, generation suffix `<|start|>assistant`),
`StructuredAssistantDecoder.swift` (channel-header state machine; analysis → thinking channel,
commentary tool calls buffered, final → content), new `HarmonyToolCallParser.swift`
(`to=functions.NAME`, JSON args, both header orders), fixture tokenizer
`tests/NVMAI/Core/Tokenization/Fixtures/HarmonyTokenizer/` (+ server copy), golden tests
mirroring `ChatMLTemplateTests`.
Renderer per the pinned Harmony semantics, including the developer-message mapping, TS
namespace tool rendering, tool-result content-as-JSON, analysis replay inside tool loops, and
the current-date line.

### Task B5: server surface for Harmony
**Files:** `ServerInference.swift`, `HTTPServer.swift`/`OpenAIModels.swift`
(`reasoning_content` out + in), `ServerPromptCache.swift` (continuation bridges gated to
chatml), server tests with a scripted backend.
Structured decoder becomes unconditional for the harmony dialect (no think-leak); finish_reason
mapping from `<|call|>`/`<|return|>`; `apiModelID` for the family.

### Task B6: gpt-oss bring-up gate (deploy-time, not CI)
Repack the real checkpoint (`--input-snapshot`, unpinned source), greedy fixed-seed generation
sanity vs `mlx_lm.generate` on the same prompts, tool-loop smoke via the server. Documented
commands land in the plan’s companion section of the PR/commit message, not in AGENTS.md.

## Phase C — Kimi-Linear-48B-A3B (spec step 6)

### Task C1: tokenizer conversion artifact
**Files:** `tools/convert_kimi_tokenizer.py` (tiktoken.model + added tokens →
`tokenizer.json` with ByteLevel decoder; runs wherever Python with `tokenizers` exists),
fixture `KimiTokenizer/` synthetic equivalent for tests.

### Task C2: `ArchConfig.kimiLinear48bA3b` + repack fusions
**Files:** `ModelTypes.swift`, `RepackPlanner.swift` (KDA q/k/v + conv concat, `kv_b_proj`
dequant-split-requant into `embed_q`/`unembed_out`, dense layer 0 resident, per-layer expert
counts), `SyntheticSnapshot.swift` Kimi variant, repack tests.
Baseline per pinned numbers; mask {2,3} + dense-layer0 descriptor;
`LinearAttentionConfig(32,32,128,128,4)` + per-channel-decay flag; MLA dims struct; sigmoid
router descriptor (scaling 2.446, renormalize, correction bias); ungated shared expert.

### Task C3: KDA kernel variants
**Files:** `GDN.swift`, `gdn.metal`, `GDNStateManager.swift` (unchanged layout),
`RealForwardRunner.swift` (KDA layer encode: separate GEMVs incl. low-rank chains, buffers for
per-channel `a`), `GDNReference.swift`, GDN tests.
Per-channel decay variants of `gdn_delta_step_{decode,prefill}` (`g[Hv*Dk]`; `dt_bias[Hv*Dk]`,
`A_log[Hv]`); sigmoid-gate variant of `gdn_gated_norm`; qk-norm scale folding verified against
the pinned Kimi scaling; conv/qk-norm/state kernels reused as-is via the repack-time concat.
**Test:** differential vs an updated `GDNReference` with vectorized gating (port of the pinned
`gated_delta` ops semantics), plus scalar-decay regression at qwen shapes.

### Task C4: MLA — cache, kernels, runner branch
**Files:** `KVCacheManager.swift` (`LayerKind.mla`, 576-elem fp16 rows, V-as-prefix),
`attention.metal`/`Attention.swift` (qkDim/vDim split variant), prefill twins,
`RealForwardRunner.swift` (MLA decode+prefill encode: q_proj, split, per-head embed GEMVs,
kv_a proj + RMSNorm, cache append, attention, per-head unembed, o_proj), `Model.swift`
accessors/schema, new MLA CPU reference + tests.
Scale 192^−0.5; no RoPE. `attentionScale`/dims from the MLA descriptor. Prompt-cache snapshot
covers MLA KV through the existing KV segment path.

### Task C5: sigmoid router + shared expert + dense layer 0
**Files:** `moe.metal` (sigmoid/renorm/correction-bias selector variant; scaling factor),
`prefill.metal` router twin, `MoE.swift`/hosts, `RealForwardRunner.swift` (dense-0 via
shared-expert path with per-layer intermediate; ungated shared expert; skip-shared for
gpt-oss made explicit here if not already), CPU references, tests at E 256 / k 8 / F 1024.

### Task C6: Kimi dialect
**Files:** `Tokenizer.swift` (`.kimi` case: tokens, renderer per pinned template, stop set),
new `KimiToolCallParser.swift` (`<|tool_call_begin|>…` grammar), decoder dispatch, fixtures +
golden tests (both fixture locations).

### Task C7: Kimi bring-up gate (deploy-time)
Convert tokenizer, repack real checkpoint, greedy sanity vs `mlx_lm.generate`, server smoke.
27.6 GB source; target machine needs the disk and the run needs the streaming budget
(expert stride ≈ half of Qwen's — cache-slot formula re-checked in C5's review).

---

## Execution notes

- After each task: `swift build` + the task's filtered tests + affected existing suites; after
  each phase: full `swift test --no-parallel` + `tools/lint.sh`.
- Phase B before C everywhere they touch the same file (activation FC, router variants,
  attention variants) — C extends B's parameterization, never forks it.
- The qwen36 golden baseline and both bring-up gates need a machine with installed models —
  they are deploy-time, out of this working set.
