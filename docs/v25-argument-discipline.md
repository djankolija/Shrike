# v25: flags are not a substitute for infrastructure

Opened at v24's close (2026-09-19), before step zero. This document states the
problem and the rule it is judged by. It does not yet state the fix: the design
belongs to step zero, and this chapter should not repeat v24's habit of writing
a solution into a spec before measuring the thing it solves.

## Why now

v24 trimmed 16 flags and kept 36. The count was defensible and every keep passed
a stated test, but the owner's reaction to the result is the reason this chapter
exists:

> Imagine if xcodebuild came with flags that were used solely by Apple during the
> development of Xcode.

And the rule that follows from it:

> Don't use args and flags as a lazy way to implement something that's a one
> shot. Some debug flags are fine. swiftc has things like dumping the AST, but a
> command that's only there because our deploy sucks is on us, and having an arg
> to help us is a shortcut.

v24's rule 1 ("it appears in a launch line, gate script or rig that runs on the
mini, which has no checkout") is the load-bearing defence for most of what
survived. Read again, **rule 1 is not a justification. It is a diagnosis.** It
says a flag must exist because we cannot get code onto the box where the
measurement has to happen. That is a statement about our deployment, dressed up
as a statement about the flag.

## The test this chapter applies

A flag is **legitimate** when it exposes a real capability of the tool to whoever
uses it. `swiftc -dump-ast` is the model: the compiler genuinely can show you its
AST, and that is useful to anyone reasoning about compilation, not only to the
people who wrote the compiler.

A flag is a **prosthetic** when it exists to work around our own infrastructure,
and the honest repair is to the infrastructure rather than to the surface. The
question is not "does a tool consume it" (v24 asked that, which is how flags with
exactly one consumer survived). The question is: **if our deployment and testing
were as good as they should be, would this flag still be here?**

## What is there today (read at `5ce5d2e`)

The clearest case is not a single flag. It is what the gate is.

`tools/golden-baseline.sh` is, per CLAUDE.md, the only check in the repository
that exercises real inference; the 1,159 unit tests never load a model. All five
of its profiles invoke **`shrike generate`**, the CLI. Production is
**`shrike serve`**, the server. Three flags exist to sustain that impersonation:

| flag | what it fakes | evidence |
| --- | --- | --- |
| `--logits-head` | the server's head path | `golden-baseline.sh:117`, `*-lh` profiles |
| `--follow-up` | the server's cached continuation | `golden-baseline.sh:112`; the only consumer in the repo |
| `--expert-cache-slots 160` | production's two-chunk arena | `CLI_EXTRA_ARGS`, required on the mini per CLAUDE.md |

`--follow-up` states its own purpose in `ShrikeCLI/Run.swift`:

> The boundary token the first answer sampled but never fed is re-fed here, **as
> the server's cached continuation does.**

So the CLI carries a second implementation of a behaviour the server already has,
and that second implementation exists so a gate can run in one deterministic
process on a box with no toolchain. Every part of that sentence is an
infrastructure fact, not a product requirement.

`--expert-cache-slots` is the same shape at a smaller scale: the CLI defaults to
64 slots while production serves 160, so the gate must pass the production value
by hand or it never crosses the arena chunk boundary that production serves at
(v22 Task 2). A default that disagrees with production is a defect; a flag that
lets the gate paper over it is the shortcut.

**Not every instrument is a prosthetic, and this chapter must not flatten them
together.** `--dump-logits` and `--dump-hidden` write the model's own internals
and are the direct analogue of `-dump-ast`: they expose something the engine
genuinely computes, useful to anyone reasoning about it. `--tokenize` answers
"what will you actually send to the model", which is a real question a user has.
Those are candidates to keep on their merits. The three in the table above are
candidates to delete **by fixing what forced them**.

## What step zero has to answer

1. **Can the gate drive the server instead of the CLI?** The server already does
   cached continuation, already has the head path, and already runs at
   production's pool. If the golden drove `shrike serve` over HTTP, all three
   prosthetics lose their reason to exist and the gate would test what we ship.
   The cost is determinism: the CLI is one process with a fixed seed, and a
   server adds a request path, a prompt cache and a queue. Whether a server-driven
   golden can be byte-identical run to run is the question the chapter turns on,
   and it is a measurement, not an opinion.
2. **What does the mini actually need?** "No toolchain" is the constraint behind
   rule 1. It is worth asking directly whether that is fixed or merely inherited,
   because most of this chapter disappears if a test harness can reach that box.
3. **Which of the survivors are genuinely `-dump-ast`?** Per the test above,
   judged one at a time, not as a class.

## Out of scope, and why it is recorded here rather than deferred silently

v24 deferred several things into its own "Out of scope" section, where they were
correctly written down and then structurally invisible. Two belong to this
chapter's subject and are restated so they are not rediscovered from scratch:

- **Per-model sampling defaults.** `GenerationDefaults` (`Sampler.swift:6`) is one
  global set of Qwen3's recommended values (temperature 0.6, top-k 20, top-p
  0.95) applied to all six installed bundles, `gpt-oss-20b` and
  `kimi-linear-48b` included, read both by generate for its flag defaults and by
  the server as its per-request fallback (`OpenAIModels.swift:361`). v24 called
  this "a correctness gap rather than a tidiness one" and did not fix it. It is
  the same family as `--expert-cache-slots`: a default that does not match what
  is served. It breaks no argv, so it does not need this chapter's break.
- **`ShrikeConfig` does not reject unknown keys**, so any key ever removed from it
  becomes a silent no-op, where the environment layer fails the launch loudly
  through `refuseUnknownEnvironment`. Recorded as a bound at v24's close.

## A note on how this document came to exist

This chapter was written at the moment the problem was understood, rather than
left as a line in a previous chapter's "Out of scope". That is deliberate, and it
is the second thing v24's close exposed: **this project has no task tracking, and
"Out of scope" is not a substitute for it.** A deferral that records the *topic*
needs a fresh investigation to act on; one that records the *conclusion* needs an
afternoon. Where this repository keeps that backlog, and in what form, is an open
question for the owner and is not settled by this document.
