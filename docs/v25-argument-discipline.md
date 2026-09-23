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
candidates to delete **by fixing what forced them**. **Retiring the three is filed in tt as SHRIKE-1 (2026-09-23).**

## What step zero has to answer

1. **Can the gate drive the server instead of the CLI?** The server already does
   cached continuation, already has the head path, and already runs at
   production's pool. If the golden drove `shrike serve` over HTTP, all three
   prosthetics lose their reason to exist and the gate would test what we ship.
   The cost is determinism: the CLI is one process with a fixed seed, and a
   server adds a request path, a prompt cache and a queue. Whether a server-driven
   golden can be byte-identical run to run is the question the chapter turns on,
   and it is a measurement, not an opinion. **Filed in tt as SHRIKE-2 (2026-09-23).** **Measured on the mini: it can, byte-identical run to run; the record is [Step zero: the golden through the server](#step-zero-the-golden-through-the-server).**
2. **What does the mini actually need?** "No toolchain" is the constraint behind
   rule 1. It is worth asking directly whether that is fixed or merely inherited,
   because most of this chapter disappears if a test harness can reach that box. **Filed in tt as SHRIKE-3 (2026-09-23).**
3. **Which of the survivors are genuinely `-dump-ast`?** Per the test above,
   judged one at a time, not as a class. **Filed in tt as SHRIKE-4 (2026-09-23).**

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
  is served. It breaks no argv, so it does not need this chapter's break. **Filed in tt as SHRIKE-12 (2026-09-23).**
- **`ShrikeConfig` does not reject unknown keys**, so any key ever removed from it
  becomes a silent no-op, where the environment layer fails the launch loudly
  through `refuseUnknownEnvironment`. Recorded as a bound at v24's close. **Filed in tt as SHRIKE-6 (2026-09-23).**

## A note on how this document came to exist

This chapter was written at the moment the problem was understood, rather than
left as a line in a previous chapter's "Out of scope". That is deliberate, and it
is the second thing v24's close exposed: **this project has no task tracking, and
"Out of scope" is not a substitute for it.** A deferral that records the *topic*
needs a fresh investigation to act on; one that records the *conclusion* needs an
afternoon. Where this repository keeps that backlog, and in what form, is an open
question for the owner and is not settled by this document. **Answered 2026-09-23: the owner chose `tt`, CLAUDE.md says so, and this chapter's own items are the SHRIKE ids above (SHRIKE-1 the chapter, SHRIKE-2 to SHRIKE-4 its step zero, SHRIKE-12 and SHRIKE-6 the two restated deferrals).**

## Step zero: the golden through the server

Measured 2026-09-23 for SHRIKE-2, with the instrument the golden's server profiles
(`tools/golden-baseline.sh`) productize: the same launch, requests and readiness check.
It runs on the box that serves, since the server binds 127.0.0.1 only. Each launch
is a fresh `shrike serve` at `tools/mini-production.sh`'s exact launch (160 slots,
the slot table, SLRU; the load line read `expert_slots=130..256 policy=slru:0.5`)
on port 8082, the binary built from `c3e25d7` (sha256 `105b1b58e13d048d…`, the
same file on both boxes). Each launch sends one sequence twice: the golden's short
and long prompts as one-message chats at temperature 0 (`max_tokens` 96 and 128),
each sent a second time at once, then the `turns` question answered to its stop
and the follow-up with that answer in the history. The mini ran five launches
(60 requests), the dev box (Mac16,7) three (36).

- **Byte-identical run to run.** Every request's text and usage row (prompt,
  cached and completion tokens, finish reason) is the same in all ten passes on
  the mini and all six on the dev box, across fresh processes and within one: the
  second pass, sent to a process that had served the whole sequence, repeats the
  first pass's rows, cached counts included.
- **An immediate replay is its own answer.** A prompt sent again right after a
  `max_tokens` stop resumes from the cached prefix (18 of 25 tokens cached for
  short, 3,749 of 3,756 for long) and prefills the last 7. Its greedy text departs
  from the first answer's (on the mini, short at character 310 of 417 and long at
  160 of 331) and is itself identical in every run. Production gives a retry after
  a `max_tokens` stop at temperature 0 a different answer from the first request
  (a retry after a stop token was not measured). The prompt reaches the
  model in a different chunk split, and a 7-row chunk is under
  `prefillMatrixMinRows` (16, `RealForwardRunner.swift:316`), so it runs the scalar
  kernels; which difference flips the pick is not isolated.
- **`turns-lh` does not test production's continuation.** The server's first turn
  matches the CLI baseline's on both boxes; its second does not (the mini's is a
  55-token answer on counting and binary semaphores where the baseline's is a
  one-clause contrast with a mutex; the dev box's differs in the last clause). The
  tokens are the same and the split is not: the server re-feeds the boundary token
  in the settle's 2-token prefill between requests, where `--follow-up` prefills it
  with the follow-up as one chunk of about 22 rows (`Run.swift:159-186`). Today's
  CLI still reproduces its own `turns-lh` baseline, so the difference is the path,
  not a stale file.
- **`short` and `long` run a path production never runs.** The server always takes
  the logits head (`ServerInference.swift:704`, `:816`), has no raw-prompt route and
  renders every prompt as ChatML, so four of the five CLI profiles have no server
  equivalent and a server golden needs new baselines.
- **The ready line is not readiness.** `shrike serve` prints it and answers
  `/v1/models` before any model loads; the model loads on the first request. The
  harness sends `POST /v1/models/load` and polls `/health` until `resident` names the
  model (3 s on either box).

So a golden driven through `shrike serve` is byte-identical run to run on the mini,
given a fresh process and a fixed request order, at production's exact launch. The
gate does not need `--logits-head`, `--follow-up` or `--expert-cache-slots`: the
server takes the logits head, continues a turn and sizes its pool from the launch
line, as production does, and outside the golden nothing in the repository passes
the three to `generate`. Whether any of them stays as a capability of `generate` is
SHRIKE-4's verdict, one flag at a time.
