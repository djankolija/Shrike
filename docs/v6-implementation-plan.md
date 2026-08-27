# v6 implementation plan: dialect-normalized prompt cache

> **For agentic workers:** execute task-by-task in order — the sequencing is load-bearing
> (see *Sequencing rationale*). Each task ends buildable (`swift build`) with its named
> tests passing (`swift test --no-parallel --filter <suite>`) and committed. One checkbox
> per task tracks completion.

**Goal:** the prompt cache becomes a byte comparison and nothing else — longest-common-prefix
matching with rewind, write-time normalization of the live region into settled form, and
deletion of the structural fallback.

**Architecture:** matching generalizes first to LCP + rewind (a pure cursor move,
`KVCacheManager.rewind(to:)`), landing *behind* the existing structural fallback so nothing
regresses; normalization then rewrites the live region into the dialect's settled render at
clean completion (stop token AND closed thought channel — both decoder state), with degenerate
turns rewinding to the generation boundary and entries always recording the achieved cursor.
All dialect knowledge — the live/settled boundary, the settled render — lives in the tokenizer
seam; the cache stays byte-blind. The structural machinery is deleted only after live
verification proves it no longer fires.

**Tech stack:** Swift 6.3 / SPM, swift-transformers Tokenizers, swift-testing.

**Spec:** `docs/v6-dialect-normalized-cache.md` — normative for every mechanism named here;
this plan only sequences it. Baseline already landed: `d7acd76` (tools-path
`reasoning_content` pass-through) — mid-loop ChatML tool hops are byte-exact as of it.

## Global constraints

- `swift test --no-parallel`; unit tests never load a real model (AGENTS.md).
- `tools/lint.sh` gates: no `as!`/`try!` under `sources/` without `lint:allow-force`, no new
  function over 120 lines. CI builds `swift build -c release` warning-free.
- The publish/match invariant `kvPosition == kvBackedTokenIDs.count`
  (`ServerPromptCache.swift:76,158`) holds at every commit — entries record what was
  **actually written**, never an intended target.
- The cache stays dialect-blind: after Task 7, `ServerPromptCache` knows nothing about
  thinking, tool calls, messages, or dialects — bytes and cursors only.
- Kimi is the regression canary: its S12 profile (~94–96% of prompt tokens served,
  2026-08-27) must not degrade at any task boundary.
- On-box probes: temperature 0, `NVMAI_CACHE_DIAG=1`, conditions recorded (model, thinking
  mode, `--max-context`, `--ram-budget`). Deploy is build-on-MacBook → binary to the mini's
  `nvmai-runtime/bin`; the pre-fix binary is preserved as `NVMAIServer.prefix-baseline`.
- `tools/golden-baseline.sh --check 4` at deploy time for any runtime-path change.
- No hit-rate targets anywhere: retained thinking inflates numerator and denominator both.
  The objective is eliminating zero-cache turns, not moving fractions.

## Sequencing rationale

Partial-prefix salvage must not fire before the structural fallback while that fallback still
carries traffic: with thinking on, the structural path rescues 100% of text turns today
(measured 2026-08-27), and its hit *restores the blob* — an early partial salvage would
truncate exactly the KV it restores, changing model input as a side effect of a cache change.
So the interim match order is **full-LCP hit → structural → partial-LCP salvage**, and the
structural deletion waits until live verification (Task 6) shows normalization has taken over.

Out of scope here, deliberately: the per-model `preserve_thinking` economics run and the
replaying-reasoning behaviour question (spec: *Cost*, *Not established*) — both are
measurement campaigns after v6 lands, not implementation.

---

## Task 1: LCP match + fraction logging

- [ ] implemented, tests green, committed

**Files:** `sources/NVMAIServer/Core/ServerPromptCache.swift:152-217` (the `match` path);
`tests/NVMAIServer/ServerPromptCacheTests.swift` (extend).

Compute the divergence index once per candidate entry — the `s12_diverge` diagnostic branch
already computes `firstDiff`; hoist it so every comparison yields `k` = length of the common
prefix of `renderedPromptIDs` and `entry.kvBackedTokenIDs`. Behavior:

1. `k == entry.kvPosition` and render ≥ entry → full hit, unchanged semantics (today's S12).
2. Otherwise try the structural paths exactly as today.
3. Only if structural fails: **partial salvage** — truncate the entry to `k`
   (`kvBackedTokenIDs.prefix(k)`, `kvPosition = k`; invariant intact) and return
   `(renderedPromptIDs, k)` so prefill continues from `k`. `k == 0` keeps today's total-miss
   semantics (entry untouched; replaced on next publish).

Every match attempt logs `lcp k=<k> kv=<kvPosition> fraction=<k/kv>` via `NVMAICacheDiag`,
alongside the existing `s12_short`/`s12_diverge` lines. The fraction is the instrument the
rest of this plan reads.

**Test:** three cases in `ServerPromptCacheTests`: (a) full hit unchanged; (b) a
thinking-retained text continuation (render shorter than entry) still routes to the structural
path *before* any salvage — assert the entry is not truncated; (c) a genuinely divergent tail
→ entry truncated to `k`, returned `cached == k`.

## Task 2: settled boundary per dialect

- [ ] implemented, tests green, committed

**Files:** `sources/NVMAI/Tokenization/Tokenizer.swift` (new API next to the render paths);
`tests/NVMAI/Core/Tokenization/` (per-dialect template test files, extend).

New tokenizer API: `settledBoundaryTokenCount(messages:tools:) throws -> Int` — the token
count of the render truncated at the last query, i.e. `lastUserPromptEnd`. Per dialect, per
its template: ChatML scans backwards past user-role messages whose trimmed content is a
`<tool_response>…</tool_response>` wrapper (fixture lines 67-77); Harmony and Kimi take the
last user message plainly. The derivation MUST run through whichever render path produced the
prompt — the hand-written `chatMLChatTemplate` (`Tokenizer.swift:700`) and the upstream Jinja
path used when tools are present (`Tokenizer.swift:1150`) are not interchangeable.

**Test:** the truncated render is a byte prefix of the full render — for a plain multi-turn
list and a tool-loop list, per dialect; the ChatML case includes a `<tool_response>`-wrapped
user message to pin the skip rule.

## Task 3: settled-form renderer per dialect

- [ ] implemented, tests green, committed

**Files:** `sources/NVMAI/Tokenization/Tokenizer.swift`; per-dialect template tests.

The bulk of the work (spec: *What this needs*). New API: `settledRender` of a completed
request's live region — the byte form every later request will send for those turns, i.e.
reasoning dropped per the dialect's rule (ChatML: turns at/below `last_query_index`; Harmony:
analysis dropped once the text turn ended; Kimi: identity).

**Byte-exactness property test, per dialect:** take a completed conversation `C`, append a
next user message `u'`; the shipped Jinja template's render of `C + u'` must contain this
settled render of `C`'s live region byte-for-byte at its position. Same golden-test shape
already used to validate the Harmony and Kimi hand renderers against their shipped templates.
The Kimi test asserts identity (settled == live), so normalization will no-op there.

## Task 4: normalization + Harmony publish

- [ ] implemented, tests green, committed

**Files:** `sources/NVMAIServer/Core/ServerPromptCache.swift` (publish + a new normalize
entry point); the generation-completion path that owns stop reasons
(`sources/NVMAI/Runtime/Generation/RawCompletion.swift:345-346` supplies them);
`tests/NVMAIServer/ServerPromptCacheTests.swift`.

Trigger, exactly the spec's table: normalize iff stop reason ∈ {`.endOfTurn`, `.eos`} AND the
thought channel closed — both decoder state (`StructuredAssistantDecoder`'s running parse),
never a string scan. Everything else is degenerate.

- **Clean:** rewind to `settledBoundaryTokenCount` (Task 2), prefill the settled render
  (Task 3) eagerly in the background under the session lock, extend the entry as tokens are
  written — so an interruption leaves the entry at the achieved cursor, by construction.
- **Degenerate:** rewind to this generation's pre-suffix boundary; truncate the entry to the
  achieved cursor. Never publish an intended target.

Remove the Harmony skip in `publishCacheEntry` — with the settled rewrite handling the
analysis drop, Harmony entries publish like every other dialect's.

**Test:** unit-level, no model: the trigger table exhaustively (each stop reason × thought
open/closed → normalize or degenerate); entry truncation on simulated interruption; the
Harmony publish path no longer skips. The prefill half is verified live in Task 6.

## Task 5: mid-rewrite arbitration

- [ ] implemented, tests green, committed

**Files:** `sources/NVMAIServer/Core/ServerPromptCache.swift` / server core;
`tests/NVMAIServer/ServerPromptCacheTests.swift`.

The rewrite records the target token sequence it is prefilling toward. A request arriving
mid-rewrite is arbitrated by that target (spec: *Design*): render the request CPU-side —
**before** waiting on the lock — and if the target is a byte prefix of the render, join (wait
for the rewrite, continue prefill from its end); any other render aborts the rewrite, the
entry truncates to the achieved cursor, and the request proceeds through Task 1's match,
salvaging what was written.

**Test:** stubbed slow rewrite, no model: the join case (follow-up render extends the
target), the abort case (divergent render → rewrite stopped, entry at achieved cursor, LCP
salvage from there), and the ordering (prefix check happens before lock wait).

## Task 6: live verification, structural still present

- [ ] all probes pass, results recorded

On-box, deploy per the constraints above. qwen36, thinking on, temperature 0, diag on:

- **Plain multi-turn ×2 (3 turns):** after each completed turn, normalization runs; every
  subsequent turn is a full-LCP hit (`fraction=1.0`), structural fires **zero** times.
  (Baseline being beaten: S12 0/5, structural 5/5.)
- **Tool loop** (the staged `chatml_toolloop_probe.py`): mid-loop byte-exact (as since
  `d7acd76`); the post-loop turn — the measured `s12_short rendered=395 kv=494` flip — is now
  a full hit against the normalized entry.
- **Degenerate:** a `max_tokens` cut mid-thinking → entry truncated to the settled boundary;
  next turn cold-prefills from there; no structural, no fuzz.
- **Kimi canary:** profile unchanged from 2026-08-27.
- **Harmony (gpt-oss):** caches for the first time — record its numbers as the new baseline.
- `tools/golden-baseline.sh --check 4` green.

Any structural hit in these runs is a finding, not noise — stop and diagnose before Task 7.

## Task 6a (unblocking — from Task 6 red): settle by reconstruction

- [ ] implemented, tests green, committed

Task 6 measured that no model on this box can run the settle: qwen36/ornith are
30-of-40 GDN layers, kimi is GDN, gpt-oss and gemma are ring-backed —
`supportsPartialRewind` is false everywhere that matters, normalization never ran, and
the structural fallback served ~97%. The capability check is right about *seeking*: a
recurrent state cannot go backwards and a wrapped ring's rows are gone. The mechanism
generalizes instead — the settle does not need to seek to a position, it needs the
state AT a prefix of its target, and `captureInferenceState`/`restoreInferenceState`
reconstruct state at any snapshotted position on every architecture; the prompt-cache
restore path already proves it on GDN.

**Contract:** the rewrite ("make the KV hold sequence X") gains a reconstruction path.
Rewind-capable runner: today's path (rewind to LCP(X, kv), prefill the remainder).
Otherwise: restore the best available snapshot whose bytes are a byte prefix of X, then
prefill `X[snapshot.position...]`; with no prefixing snapshot, reset and prefill X
whole — the eager-can't-lose argument prices that as the next request's work done
early. Settled renders are append-only across turns (Task 2's prefix property applied
to nested settled lists — pin it per dialect in a test, do not assume it), so
`finishRewrite`'s existing capture at the settled position IS the next turn's restore
source: steady-state settle cost = one restore + the new turn's delta. Normalization's
capability gate becomes "always", mechanism chosen per runner. `allowsPartialSalvage`
stays rewind-gated — mid-entry state cannot be reconstructed from a later snapshot —
and `dropEmission` stays rewind-gated (a degenerate turn on a non-rewindable runner
skips; the prior settled entries still carry the conversation). Arbitration, targets,
`.lost`, and every entry/snapshot pairing invariant are unchanged: reconstruction
failure lands exactly where prefill failure lands today.

**Tests:** the mechanism decision (rewind vs restore vs reset, given capability and
snapshot inventory) as a pure exhaustive function; the append-only property per
dialect; the actor wiring rides to the Task 6 re-run.

## Task 6b (from Task 6 red): verbatim non-scalar tool arguments

- [ ] implemented, tests green, committed

Measured: a nested tool call (objects, float arrays) breaks mid-loop byte-exactness —
flat mid-loop cached=486/506, nested cached=0/743 — because `args_value | tojson`
re-serializes what the model emitted. Fix: on the ChatML tools render path, carry
argument values as their raw source slices (strings), so the template's
`args_value | string` branch emits the model's original bytes for scalars and
non-scalars alike — the per-value cousin of gemma's whole-arguments string lever.
Verify against the shipped fixture's branch semantics and the existing goldens (scalar
bytes must not change), and confirm the settled render inherits the fix through the
shared render path. If the raw-slice representation cannot pass Jinja's `is string`
test, or a golden breaks in a way that reveals a real constraint, stop and report.

## Task 7: deletions + final verification

- [ ] deleted, probes re-pass, committed

**Gate hardened by Task 6 red:** do not delete until the Task 6 re-run shows
normalization *running* on qwen36 (`settle_done` lines) with structural hits at zero.
Task 6 measured structural carrying ~97% on qwen36; deletion before reconstruction
lands would replace it with ~3%.

**Files:** `sources/NVMAIServer/Core/ServerPromptCache.swift`,
`sources/NVMAI/Tokenization/Tokenizer.swift`, their tests.

Delete `matchTextContinuation`, `matchToolContinuation`, `assistantMatches`,
`encodeToolResultContinuation` (dead since before v6 — unconditional throw), and the entry's
`inputMessages` / `assistantTurn`. The entry is bytes and a cursor; the LCP path is the only
match. Re-run the full Task 6 probe set — identical results, now with no structural code to
fall back on. `rg` confirms none of the five symbols remain under `sources/`.

## Task 8 (gated — not scheduled): gemma

Only if gemma-4-26b-a4b is adopted — and the gate is **"does gemma have a cache path at
all"**, not adoption alone. Gemma is 25-of-30 sliding-window layers (turbo's ArchConfig;
verify on the real checkpoint), so the ring-backed storage that makes
`supportsPartialRewind` refuse gpt-oss refuses gemma harder — normalization cannot run —
while its template both drops reasoning (normalization mandatory) and exposes no
`preserve_thinking`-style replay flag (`enable_thinking` gates generation, not history).
With thinking on, that leaves no working cache path. Three options to cost at bring-up:
disable the fp16 ring so SWA storage stays linear and rewindable
(`KVCacheManager.swift:263`) — SWA layers then allocate against `maxContext` instead of
their window, a real memory price on a 16 GB box, and no launch flag currently exposes
the toggle; run gemma with thinking off — text turns then carry unmodified (the template
injects a closed empty thought block, so the blob holds no reasoning to drop) but tool
loops still need the string-arguments lever below, since `dictsort` reorders arguments
independently of thinking mode and NVMAI hands the template a mapping, so the string
branch never fires without it (whether the reorder bites depends on the model's emission
order — unknown, and the lever removes the need to find out); or accept cold prefills.
Then, as
before: verify the shipped template against the gemma-4-12B-it read (spec: *Not
established*), wire the conjunction boundary predicate into Task 2's scan, and pass
tool-call arguments as a string to defeat `dictsort` (spec: *What this needs*).
