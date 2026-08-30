# v8 — Render Harmony tool calls in the model's emission form

A two-line renderer change with a day of evidence behind it. The change:
`harmonyAssistantBlock` re-renders a past assistant tool call as

```
<|start|>assistant<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{args}<|call|>
```

instead of the upstream `chat_template.jinja`'s

```
<|start|>assistant to=functions.NAME<|channel|>commentary json<|message|>{args}<|call|>
```

This is a deliberate deviation from the template our renderer otherwise ports
byte-exactly. The template's form does not round-trip what gpt-oss actually
emits, and that mismatch has two costs, both observed live on 2026-08-30
against gpt-oss-20b (4-bit, this machine, v7 build):

## Evidence

**Turn 1 of any tool exchange emits the form above.** Captured verbatim via
`SHRIKE_GEN_DIAG=1` and decoded:

```
⟦<|channel|>⟧analysis⟦<|message|>⟧…⟦<|end|>⟧⟦<|start|>⟧assistant⟦<|channel|>⟧commentary
 to=functions.WebSearch ⟦<|constrain|>⟧json⟦<|message|>⟧{"query":"bancor","limit":8}⟦<|call|>⟧
```

**Cost 1 — the template form corrupts the model's next call.** On turn 2 the
model reads its own call re-rendered in the template's order and blends the
two syntaxes at the exact seam where they differ (right after the tool name).
Captured failing generation, 32 tokens:

```
…⟦<|start|>⟧assistant⟦<|channel|>⟧commentary to=functions.WebFetch⟦<|channel|>⟧
```

— a legitimate tool name followed by a second `<|channel|>` token, which is
precisely the token that follows `functions.NAME` in the re-rendered form it
had just read. The strict decoder fails closed: 500
`structured_output_failure kind=decoder_consume cause=malformed`. The
original benchmark hit the same seam with a different mangling
(`cause=unknown_tool`). Same conditioning defect, run-dependent symptom;
every two-step tool exchange re-poisons the prompt.

**Cost 2 — every tool hop splits the KV prefix.** The v6 settle machinery
deliberately skips tool-call turns (`guard !emittedToolCalls`,
ServerPromptCache.swift), so the KV keeps the emission form while the
re-render produced the template form:

```
prompt_cache_diag lcp k=435 kv=458 fraction=0.949
s12_diverge … rendered=[…, 173781, 316, 28, 44580, …] cached=[…, 173781, 200005, 12606, 815, …]
```

With the emission-form render, render == KV for the call block natively —
no settle needed, full prefix reuse across tool hops.

## Deliberately unchanged

- **The tool-result line** (`<|start|>functions.NAME to=assistant…`): only
  the template defines it — models never emit tool results — so there is no
  emission to compare against and no evidence it is wrong.
- **The decoder**, which has always accepted the recipient on either side of
  `<|channel|>`; HarmonyDecoderTests keeps a template-order input case as
  coverage of that tolerance.
- **The v6 settle carve-out** for tool turns: still in place, now costless
  for this divergence.

## Acceptance — passed 2026-08-30

The exact repro that failed (pi + `harnesses/wikipedia.ts`, gpt-oss-20b,
"Look up the Wikipedia article on the bancor…") ran **three clean tool hops
plus a final answer** ("proposed by John Maynard Keynes", `finish=stop`)
where it previously died 500 on hop two. One hop matched the cache at
`fraction=1.0` with `cached_tokens=1055` — a byte-perfect prefix across a
tool hop; the final turn reused 1280. All five local gates green (1079
tests, TSan clean).

**Known residual divergence, benign:** the model's own header emission
varies under sampling — this run it wrote `to=functions.WebSearch
code<|message|>` (content-type " code", no `<|constrain|>`) where the
morning run wrote ` <|constrain|>json` — and it pretty-printed its call
arguments while clients echo them back compact. Both are outside the
renderer's control, are absorbed by the salvage path, and — unlike the
header-order mismatch this change removed — do not teach the model a
second syntax: subsequent calls stayed clean.
