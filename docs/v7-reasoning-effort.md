# v7 — Harmony reasoning effort, and the diagnostics that precede it

Design record for the v7 batch: two failure-diagnostics knobs (Phase A), then a
reasoning-effort setting for Harmony models with per-request control (Phase B).
The implementation plan is [v7-implementation-plan.md](v7-implementation-plan.md);
its checkboxes are the status of record.

## Motivation

A benchmark run against gpt-oss-20b surfaced three findings (assessed 2026-08-30):

1. **`--thinking off` is a no-op on Harmony.** `harmonySystemBlock` hardcodes
   `Reasoning: medium` (Tokenizer.swift, `harmonySystemBlock`); the flag never
   reaches it. Cost observed in the field: 2,391 thinking tokens over 427s for a
   one-line factual question.
2. **A tool exchange dies on the second request** with
   `structured_output_failure … cause=unknown_tool` — but the failure log drops
   the offending tool name, so the mechanism cannot be pinned from outside. The
   name exists in `ToolCallParserError.unknownTool(name)`; classification
   flattens it away.
3. **Reasoning prose landed in visible content** after a mid-generation
   degeneration. Source analysis shows the streaming decoder cannot silently
   misroute a *valid* channel switch (out-of-place markers fail the request;
   in-place ones route correctly), so the text arrived either as plain
   final-channel tokens or as a recipient-less `commentary` block. Deciding
   which requires seeing the raw generated token IDs — and no knob dumps them.

Phase A closes the two evidence gaps (2, 3). Phase B fixes (1) properly, as a
first-class effort setting rather than a `--thinking` special case.

## What effort is, and which models have it

gpt-oss reads its effort level as literal prompt text — the word after
`Reasoning:` in its system block — and was trained to calibrate deliberation
length against it (`low`/`medium`/`high`). The upstream `chat_template.jinja`
models this as a `reasoning_effort` template variable defaulting to `"medium"`.

Effort exists on exactly one of the three supported families:

| Family | Thinking control | Effort levels |
| --- | --- | --- |
| gpt-oss (Harmony) | structural — analysis channel always exists | low / medium / high |
| Qwen 3.5/3.6 MoE (ChatML) | `<think>` block: off / on / adaptive | none — binary only |
| Kimi Linear (Kimi) | none — non-thinking model | none |

## Decisions

- **Effort is a Harmony-only setting, validated per model.** Sending
  `reasoning_effort` to a non-Harmony model is a 400, not a silent no-op; a
  value outside `low|medium|high` is a 400. Rationale: the original bug was a
  knob that silently did nothing; we do not build more of those. A benchmark
  arm labeled with an effort that measured nothing is worse than a failed arm.
- **Per-request plus server default.** The OpenAI Chat Completions
  `reasoning_effort` field is accepted per request; a `--reasoning-effort`
  launch flag (env: `SHRIKE_REASONING_EFFORT`) sets the server default;
  effective value is request → flag → `medium`. Chat Completions only — pi
  drives the `openai-completions` API; the Responses API surface is out of
  scope for now.
- **`--thinking off` on a Harmony model warns and lowers.** Harmony cannot
  disable thinking (the analysis channel is structural), so the launch line
  notes this and treats the flag as effort `low` — unless an explicit
  `--reasoning-effort` is also given, which wins. Existing launch commands keep
  working and get the closest honest behavior.
- **Effort threads through render calls as a parameter, not tokenizer state.**
  The tokenizer stays immutable; per-request values pass down the encode path.
  This is also the shape a later per-request ChatML thinking mode would take. **Filed in tt as SHRIKE-36 (2026-09-23).**
- **No forced-token injection.** A hard thinking budget needs the runtime to
  force a transition sequence mid-generation (`</think>`, or Harmony's
  `<|end|><|start|>assistant<|channel|>final<|message|>`). Declined for now:
  the decode loop has no injection mechanism, forced cuts are off-distribution
  and hurt answer quality, and effort covers the practical need.

### Client propagation (pi)

No wire-level convention exists for advertising effort support; OpenAI-compatible
`/v1/models` carries no capability metadata, and every harness resolves
capabilities from client-side model entries. pi (and opencode) use the
models.dev schema. The wiring is client config, not server code — in
`~/.pi/agent/models.json`, on the `shrike/gpt-oss-20b` entry:

```json
"reasoning": true,
"thinkingLevelMap": { "minimal": "low", "xhigh": "high" }
```

With `reasoning: true`, pi sends `reasoning_effort` on its
`openai-completions` API (verified in pi 0.84.3 source); `thinkingLevelMap`
remaps pi's wider effort vocabulary onto the three values Shrike accepts, so
the strict validation never trips on a legitimate pi session. Qwen and Kimi
entries stay `reasoning: false`, which disables pi's effort UI entirely.

### Prompt-cache interaction, accepted

`Reasoning: <effort>` sits in the first ~30 tokens of every Harmony prompt.
Changing effort mid-conversation therefore re-prefills the whole conversation
(and the salvage path truncates the old entry). This is inherent to where
gpt-oss keeps its knob, not a defect; benchmark arms hold effort constant per
conversation, where the cost is zero. Documented, not mitigated.

The v6 KV-settle re-render (`settledBoundaryTokens`/`settledFormRender`) also
re-renders the system block on every settle, and is effort-aware as of this
fix — it now renders at the request's own effort rather than unconditionally
at `medium`, so settle keeps matching a conversation running at `low`/`high`.

## Phase A — diagnostics

Two knobs, both unit-testable without loading a model:

1. **`unknown_tool_name=` in the failure log.** `StructuredOutputFailure`
   carries the name from `ToolCallParserError.unknownTool(name)` through to the
   `structured_output_failure` log line. One repro then shows whether the
   second-turn failure is namespace mangling, a hallucinated name, or something
   else — the evidence gate for any Bug-1 renderer change.
2. **`SHRIKE_GEN_DIAG=1` generated-token dump.** Mirrors `SHRIKE_CACHE_DIAG`:
   when set, every completion (and every structured-output failure) emits the
   generated token IDs to stderr as one `Shrike gen_diag …` line. This decides
   finding (3): a channel-marker token (`2000xx`) before the leaked reasoning
   text means one thing, its absence another.

## Phase B — reasoning effort

- `ReasoningEffort` enum (`low`/`medium`/`high`, string raw values) in the
  tokenization layer.
- `harmonySystemBlock`/`harmonyChatTemplate` parameterized on it, default
  `.medium` (byte-compatible with today's render); golden tests extended for
  `low` and `high`.
- `encodeToolChat`/the no-tools encode path accept an effort argument and pass
  it through for the Harmony dialect; other dialects ignore it.
- `--reasoning-effort low|medium|high` flag + `SHRIKE_REASONING_EFFORT` env in
  `ServerArguments`; the `--thinking off` Harmony mapping above; launch line
  prints the configured default (`auto` when unset).
- `reasoning_effort` parsed on Chat Completions requests, validated
  (`low|medium|high` else 400; non-Harmony model else 400), threaded through
  `ValidatedChatRequest` into the per-request render.

## Out of scope

- Per-request thinking control for ChatML models (deferred; needs a
  request-field convention decision — no OpenAI standard exists for on/off). **Filed in tt as SHRIKE-36 (2026-09-23).**
- Thinking budgets / forced-token injection.
- The Bug-1 renderer change (re-rendering assistant tool-call turns in the
  model's emission form rather than the upstream template's form) — awaits the
  evidence Phase A produces. **Done: in v8 (1da1c4c) (noted 2026-09-23).**
- Responses API effort surface.
- `/v1/models` capability enrichment for LLMBench.
- Gemma support.

## Verification

The five local gates (release build, swiftlint strict, markdown link check,
`swift test --no-parallel`, TSan suite) — all rendering and validation changes
are covered by unit tests against fixture tokenizers; no model run is required
for correctness. Live confirmation (pi sends the field, the rendered system
block reflects it, per-effort thinking-length differences) happens at the next
mini deploy, which independently carries the rename checklist from
[CLAUDE.md](../CLAUDE.md). **The live confirmation is filed in tt as SHRIKE-38 (2026-09-23); the rename checklist is done (the rename landed 2026-08-30).**

Adaptive thinking on qwen36 (`--thinking adaptive`, `4bc8d4e`) has never run against a live
model; its coverage is decoder-level (`ChatMLDecoderTests`, `ChatMLTemplateTests`). A live run
would check that the model's own `<think>` block streams as reasoning, not answer text.
