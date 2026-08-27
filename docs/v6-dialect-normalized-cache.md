# v6: a dialect-normalized prompt cache — design

Status: design settled 2026-08-27. Implementation plan: `v6-implementation-plan.md`.

## Objective

Make the cached KV equal what the dialect's template will render, so that the prompt
cache is a byte comparison and nothing else. Normalize at write time; delete the
structural fallback that currently repairs the mismatch at match time.

## The defect

Two records of every assistant turn disagree, and always have:

| record | holds |
| --- | --- |
| the KV blob (`kvBackedTokenIDs`) | every token generated — reasoning included |
| the cache entry's message (`ServerPromptCache.publish`) | `content` + `toolCalls`, reasoning dropped |

The blob is what the model produced. The re-render is what the client sends back, put
through the template. Those differ whenever a template drops prior reasoning, so the
byte comparison (`S12`) misses and a structural match takes over: same input messages,
same assistant turn, splice a bridge onto the blob. `assistantMatches` compares content
and tool calls and never looks at thinking, which is what makes it succeed.

It works, and it costs two things:

- **Harmony cannot cache at all.** `publishCacheEntry` skips the publish outright,
  because the KV carries an analysis block every re-render drops. gpt-oss re-prefills
  the whole conversation on every turn.
- **Whether the model sees its own prior reasoning depends on a cache outcome.** A
  structural hit restores the blob, which still contains the thinking tokens. A miss
  prefills the client's render, which does not. Same conversation, same settings, two
  different contexts, decided by something the model cannot observe.

Measured, qwen36 with thinking on, temperature 0, 2026-08-27: across two plain
multi-turn runs `S12` hit 0 of 5 opportunities — every miss `s12_short`, the render
shorter than the blob by the retained thinking — and the structural path rescued every
valid turn, once restoring 901 KV-backed tokens against a 76-token render: the model's
own thinking, resurrected from state the client never sent. Mid-loop tool hops cached
zero until d7acd76 fixed the render boundary that dropped `reasoning_content` on the
tools path; after it, the mid-loop comparison is byte-exact — integer tool arguments
through `tojson` included — and the one remaining mismatch is the post-loop flip
(`s12_short rendered=395 kv=494`): `last_query_index` moves and the whole loop's
thinking leaves the render at once. That flip is the case normalization exists for.

## The rule the templates encode

Reasoning lives as long as the request it belongs to — across however many generations
that request takes. Every template read so far implements it, differently expressed:

| dialect | mechanism | keeps reasoning |
| --- | --- | --- |
| ChatML | `loop.index0 > ns.last_query_index` (fixture lines 100–101) | on turns after the last user message — i.e. the in-flight request, tool-loop hops included |
| Harmony | tool-call branch renders `<\|channel\|>analysis`, text branch does not (fixture lines 358–380) | on tool-call turns; dropped once a text turn ends the request |
| Gemma | position AND turn-type conjoined: `loop.index0 > last_user_idx and tool_calls` (gemma-4-12B-it, line 239) | on tool-call turns after the last user message only; a final text answer's reasoning never renders, and there is no `preserve_thinking` escape |
| Kimi | no thinking channel in the decoder at all | n/a — nothing to keep or drop |

Harmony states it in the file: *"CoT is dropped during all previous turns, so we never
render it for inference."*

This is not a quirk to work around. It is the reason a tool loop's steps can behave as
one continuous episode despite being separate generations over separate requests.

## Design

**Two zones, boundary at the last user prompt.**

```
[ ————— settled ————— ][ ————— live ————— ]
 sys u1 a1 u2 a2 … u_n │ a(tool) t a(tool) t a(final)
                        ↑ lastUserPromptEnd
```

- **Settled** — reasoning already dropped, render is stable, cacheable indefinitely.
  Nothing can move it again: those turns are at or below `last_query_index` and stay
  there.
- **Live** — reasoning retained because the template keeps it for the in-flight
  request. Rewritten wholesale when the request completes.

**One operation, run on every generation, with two rewind targets:**

```
if clean:    rewind to lastUserPromptEnd, re-prefill the live region in
             settled form, extend the entry
otherwise:   rewind to this generation's pre-suffix boundary, stop
```

The targets are not interchangeable. A clean completion settles the whole live region
at once — the loop's earlier hops sit in KV in live form, so settling only the final
turn would splice settled bytes after live ones, a mixed form no render ever produces.
A degenerate turn stays inside the live request, where the next render is still
live-form: rewind only past this generation's suffix and emission, and a five-hop loop
that dies on hop five keeps the four valid hops. The boundaries coincide exactly when
the request had no tool hops.

The live/settled boundary is the dialect's to define, not a role scan. ChatML walks the
messages backwards and skips user-role messages that are `<tool_response>` wrappers
(fixture lines 67–77), so "last message with role user" lands inside the live region in
exactly the multi-hop case; Harmony draws it structurally, at the text turn that ends
the request. The dialect owns the boundary the same way it owns the preserve/discard
rule.

**One match operation: longest common prefix, then rewind.**

The request's render is compared to the entry byte-for-byte and the entry rewinds to
the divergence point; prefill continues from there. Correctness is position-wise —
under causal attention a position's KV is a function of the tokens at or before it, and
RoPE keys off absolute position — so byte equality up to k is KV validity up to k,
wherever k falls, mid-turn included. `S12` is the k == entry-length special case. With
a single KV buffer this is also optimal: serving any divergent request overwrites the
tail regardless, so the LCP maximizes what is salvaged, and a new conversation under
the same system prompt warm-starts from the system-prompt KV.

Every request logs its matched fraction. Grading the match buys graceful misses and
spends the visible cliff a binary hit/miss gave — a fidelity regression that once
surfaced as a cold prefill now surfaces only as that number dropping.

`KVCacheManager.rewind(to:)` (line 359) is a pure cursor move, no copying, already
exercised by the MTP path. The expensive half is the re-prefill, and it runs eagerly in
the background during read time, under a lock. One active session is a published
limitation of this deployment. A request arriving mid-rewrite is arbitrated by the
rewrite's target sequence, which the normalizer computed before starting: if the target
is a byte prefix of the request's render, the rewrite *is* that request's prefill
already running — join it and continue from its end. Any other render aborts it, and
the LCP salvages whatever was written. No session identity is tested and none is
needed: a follow-up turn joins by construction, while regeneration and edits abort —
correctly, since the rewrite was producing bytes they cannot use. The check is one
CPU-side render and an array compare.

## Trigger

Normalize only when the **model itself signalled completion** and the **thought channel
closed**. Everything else is termination imposed from outside, leaving the turn in an
indeterminate state.

| stop reason | normalize |
| --- | --- |
| `.endOfTurn` — model emitted `endOfTurnID` | yes, if the thought channel closed |
| `.eos` | yes, if the thought channel closed |
| `.toolCalls` — model emitted `toolResponseID` | no, still live |
| `.maxTokens` | no |
| `.stopString` | no |
| `.external` | no |

Both signals are decoder state, not heuristics: the stop reason is a special token id
(`RawCompletion.swift:345-346`), and the thought channel is the running parse state
`StructuredAssistantDecoder` already keeps.

The second guard is not implied by the first: a stop token can land inside an unclosed
`<think>`, and the template's split is `if '</think>' in content` — no delimiter, no
split, so partial reasoning falls through into `content` as visible text. An unclosed
thought therefore takes the degenerate path whatever the stop reason says.
This is observed behaviour on these models, not a theoretical edge: LLMBench measured a
model burning its entire budget inside the thinking block and returning no text.

## Degenerate turns

Rewind, never recover. What the client will send back is its choice — it receives
`finish_reason: length` and may return the partial reasoning, drop it, or fold it into
content. Any normalization is a guess at that, and declining is the only thing that
cannot be wrong.

Rewinding discards **cache state, not the response**. Those tokens were already
streamed; the client keeps what it got.

Every degenerate path then converges on one failure mode: miss, cold prefill from the
last settled boundary, correct context. Never fuzzy — only fast or slow.

## What this deletes

`matchTextContinuation`, `matchToolContinuation`, `assistantMatches`, the entry's
`inputMessages` / `assistantTurn`, and `encodeToolResultContinuation`. The last is
already dead for every dialect: its only implementation throws unconditionally
(`Tokenizer.swift:1178–1190`) and the `try?` at the lone call site
(`ServerPromptCache.swift:300`) makes `matchToolContinuation` return nil every time it
is reached. Tool hops already live on byte fidelity with no fallback — the design
inherits that condition, it does not create it. The LCP match becomes the only path,
and the cache stops knowing anything about thinking, tool calls, or dialects.

## What this needs

**A settled-form renderer per dialect, byte-exact against the shipped template.**
`chatMLChatTemplate` implements only the live branch today. This is the bulk of the work
and where the bugs will be — the same "byte-exact renderer against the shipped template"
exercise already done once each for Harmony and Kimi.

**The live-region boundary, derived per dialect.** Nothing tracks `lastUserPromptEnd`
today. It re-derives without model work — render the message list truncated at the last
query, encode, count — and the result is a true prefix of the full render: the live
region contains no query, so `last_query_index` agrees between the two renders. The
derivation must run through whichever render path produced the prompt; the hand-written
`chatMLChatTemplate` (`Tokenizer.swift:700`) and the upstream Jinja path used when tools
are present (`Tokenizer.swift:1150`) are not interchangeable.

**On abort, the entry truncates to the cursor actually reached.** The rewrite's target
is what it was computing toward, not what it wrote; `rewind(to:)` and prefill both move
the live cursor, and that cursor is the truth. Publishing against the intended target
trips the `kvPosition == kvBackedTokenIDs.count` guard (held at publish and match,
`ServerPromptCache.swift:76,158`) and the entry is discarded whole — fail-safe, but it
forfeits exactly the salvage the abort path promises.

**Gemma only: pass tool-call arguments as a string.** Its template `dictsort`s a
mapping — reordering guaranteed — but renders a string argument verbatim; the string
branch is the only way to defeat the reorder. ChatML needs no such lever: its template
requires a mapping, and the round trip measured byte-exact through `orderedJinjaObject`
→ `|items` → per-value `tojson`, integer values included (qwen36, 2026-08-27).

## Cost

**Prefill ≈23.5 tok/s against decode ≈6.05 tok/s — a ratio of ≈3.9×.**

Measured 2026-08-27 on the mini, `kimi-linear-48b-a3b-4bit`, `--ram-budget 6G`,
`--max-context 32768`. Prefill is the slope between two fully uncached prompts (587 and
2603 tokens, `cached=0` on both), which cancels fixed request overhead and decode time;
each point cross-checks independently at 24.7 and 23.8 tok/s. Decode is the server's
own accounting (`NVMAI_RUNNER_STATS`, 256-token decode cells) at the same standing
config, from the expert-cache slots sweep.

⚠ Absolute rates are per-box and per-model — the mini is the IO-bound one. The **ratio**
is the transferable figure; it has not been re-measured on the MacBook or on a
dense-attention model.

At ≈3.9× prefill is not the cheap operation the usual batching argument assumes,
presumably because expert streaming gates both paths.

**This is the price of `preserve_thinking: false`, not a defect**, and there is no way
around it within the design nor should there be: a template that discards prior
reasoning requires the KV to be rewritten to match, and a rewrite costs a forward pass.
The number feeds exactly one decision — whether a given model's behaviour under `false`
is worth the rewrites — weighed per model against measurements, not settled here.

**Eager normalization cannot lose.** The rewrite is the same work the next turn's
prefill would do anyway, so partial progress is proportional saving: complete it during
read time and it is free, get interrupted at 60% and the next request pays the remaining
40%. Never worse than deferring. The only waste is an abandoned conversation, which
costs idle electricity on an idle box.

In practice read-think-type time absorbs most of it. A live region after one tool call
runs ~3,000 tokens — LLMBench measured a single Wikipedia lookup at 2,430 prompt tokens
on its own — which is **~2 minutes** at the rate above. A three-hop loop is closer to
~6 minutes and will outrun the human; a single lookup usually will not.

The wider consequence stands either way: the prompt cache is not an optimization on this
box, it is load-bearing for usability.

## `preserve_thinking`

The ChatML template exposes `preserve_thinking` (fixture line 100), which renders
reasoning on every assistant turn unconditionally. That makes the render a fold over the
message list again, so the whole conversation caches append-only, tool loops included.

Without it, the first user turn after a tool loop flips every assistant turn in that
loop to no-thinking at once, and the entire live region must be re-prefilled. At the
ratio above that is minutes, not milliseconds — which makes this an economic question,
not a matter of taste.

**The cache is indifferent either way.** The dialect declares whether reasoning is
preserved; the rewrite either triggers or it does not. The cost of the flip is a
behaviour measurement, not an implementation concern.

## Not established

- The prefill:decode ratio on any box or model other than the one in *Cost*.
- Whether replaying reasoning changes model behaviour. Two observations, neither
  established: LLMBench saw it restore gemma's tool use across turns (predicted by the
  mechanism, observed once, export lost), and the qwen36 fix-A run saw reasoning shorten
  on later hops once prior reasoning was replayed (228→150 and 341→300 chars — n=1,
  single seed, exported this time). The deliberate per-model run stands.
- Whether gemma-4-26b-a4b's chat template matches the gemma-4-12B-it one actually read
  (the boundary row in *The rule the templates encode*, the `dictsort` reorder, the
  `reasoning`/`reasoning_content` field). Same family, unverified; `gemma4.gturbo`
  exists on the mini if someone wants it checked.

## Stated limitations

Third-party clients (Codex, OpenCode) are not held to the contract and are not designed
for. A client that does not echo reasoning back simply misses and cold-prefills. This is
a deliberate scope decision for a single-user local deployment, not an oversight.

## One thing not to misread

**The thinking cannot be kept out of the KV in the first place.** Generation *is* KV
writing: each token is fed forward as it is produced because the next token attends to
it, and the visible answer was produced by attending to the reasoning. Removing a span
from the middle is not possible either — a token's key/value derive from a hidden state
that attended to everything before it, and RoPE keys off absolute position.

So the KV is a function of the token sequence. To hold state for sequence X, the model
must be run over X. Normalization is a rewrite, and its bill is one forward pass over
the rewritten span. What it does *not* change is the response: the emitted tokens are
kept exactly, and only their representation is recomputed under a context that lacks the
reasoning — which is precisely the claim the template makes anyway.
