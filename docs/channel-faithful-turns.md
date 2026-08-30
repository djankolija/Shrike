# Channel-faithful turns — design record (future work, unscheduled)

**Status: design only.** No implementation plan exists yet; per the repo
convention one gets written alongside this when the work is scheduled. Until
then this document records the decision, the evidence, and the open
questions, so the design survives the gap.

## The principle (settled 2026-08-30)

Two tools exist for keeping the KV cache, the re-render, and the model's
training distribution in agreement, and each has a domain:

- **The async settle rewrite is the right tool for *mandated deletions*** —
  prior-turn analysis must be dropped from history because the model was
  trained that way. That is not a hack and does not go away.
- **Faithful round-tripping is the right tool for everything that is
  supposed to survive.** [v8](v8-emission-form-tool-calls.md) applied this to
  tool calls after re-render infidelity was proven to corrupt the model's
  next call. This task extends the same principle to the remaining
  structure that today gets collapsed: recipient-less commentary blocks and
  multi-final (restart) turns.

## Motivation

gpt-oss restart turns are well-formed harmony:
`analysis → final #1 → commentary (reasoning prose) → final #2 → <|return|>`
(captured verbatim 2026-08-30; the leak analysis lives in the Bug-3 section
of the v7/v8 evidence). Today two lossy things happen to such a turn:

1. **On the wire**, the OpenAI chat-completions shape flattens three
   channels into two fields; recipient-less commentary is poured into
   `content` and its identity is erased. The harness cannot render it as
   thinking because the information no longer exists client-side.
2. **On replay**, the client echoes one concatenated `content` string and
   the re-render rebuilds it as a *single* final block — the model reads
   its own three-block turn collapsed into a shape it never emitted. This
   is the same infidelity class v8 fixed for tool calls, currently
   unproven as harmful (restarts observed twice, no downstream corruption
   seen) but structurally identical.

## Design sketch

- **Wire**: expose commentary as its own typed block. The Responses API is
  the natural vehicle — it is built around typed output items, so
  commentary becomes an item type rather than an invented field on chat
  completions. Chat-completions clients keep today's flattening.
- **Harness**: pi renders the commentary block in the TUI (distinct from
  thinking and answer) and replays it as a distinct block in follow-up
  requests. Whether pi's extension mechanism (`-e`) can reshape
  requests/responses or this needs upstream pi work is an open question.
- **Server**: accept the distinct block on replay; re-render restart turns
  as their true block sequence in emission form; the settle then preserves
  structure and drops only analysis. Result: render == KV natively for
  restart turns, no collapse anywhere, model always reads its own form.

## Interim option, deliberately not taken yet

A positional reroute in the decoder (recipient-less commentary *after* a
final block routes to `reasoning_content`; before one, stays visible) would
hide the leak today with zero client work and no cache impact — the settle
makes either bucket coherent. It remains available as a stopgap and is
reversible if this design lands. Not implemented on 2026-08-30 by explicit
choice: proper over patched.

## Open questions for the eventual plan

- Shrike's Responses API implementation completeness for typed items and
  for accepting them on replay.
- pi: extension-level reshaping vs upstream changes; TUI rendering design.
- Streaming semantics for the commentary block.
- Composition with tool-call turns (a restart turn that also calls tools).
- Whether chat-completions clients get the positional reroute as a
  side-door improvement or stay exactly as today.
- Whether replay-collapse is ever observed to harm the model (would raise
  this task's priority; the v8 acceptance methodology applies directly).

## Scheduling trigger

Next time a restart turn bites in a benchmark or real use, or whenever
Responses API work is otherwise scheduled — whichever comes first.
