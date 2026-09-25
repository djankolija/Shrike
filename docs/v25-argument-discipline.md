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
candidates to delete **by fixing what forced them**. **Retiring the three is filed in tt as SHRIKE-1 (2026-09-23).** **All three are deleted; see [Verdicts](#verdicts).**

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
   because most of this chapter disappears if a test harness can reach that box. **Filed in tt as SHRIKE-3 (2026-09-23).** **Answered: a harness reaches it, and rule 1 justifies no flag; the record is [Step zero: what the mini needs](#step-zero-what-the-mini-needs).**
3. **Which of the survivors are genuinely `-dump-ast`?** Per the test above,
   judged one at a time, not as a class. **Filed in tt as SHRIKE-4 (2026-09-23).** **Answered: every surviving flag has its verdict in [Verdicts](#verdicts).**

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
  with the follow-up as one chunk of about 22 rows (`Run.swift:159-186` at
  `c3e25d7`). Today's CLI still reproduces its own `turns-lh` baseline, so the
  difference is the path, not a stale file.
- **`short` and `long` run a path production never runs.** The server always takes
  the logits head (`ServerInference.swift:704`, `:816` at `c3e25d7`), has no
  raw-prompt route and renders every prompt as ChatML, so four of the five CLI
  profiles have no server equivalent and a server golden needs new baselines.
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

## Step zero: what the mini needs

Answered 2026-09-24 for SHRIKE-3.

**A test harness reaches the mini, and needs nothing built there.** The SHRIKE-2
run above is the demonstration: from the checkout, over `ssh macmini`, it stopped
production, copied its script, ran 60 requests against five fresh servers at
production's launch, fetched every output and relaunched production in about
twelve minutes, with nothing on the box but the deployed binary and the system's
`bash`, `curl`, `jq` and `python3`. `tools/decode-rig.sh` and `tools/turn-rig.sh`
drive it the same way.

**"No toolchain" is false as a fact about the box.** The mini carries Command Line
Tools 26.6, installed 2026-07-26 together with Homebrew (which oMLX came through),
and with them Swift 6.3.3, the dev box's version, and git 2.50.1. Shrike compiles
its Metal from source at runtime (`Package.swift` copies `Metal/`), so no Metal
compiler would be needed; no build was attempted there. What is true is a policy,
CLAUDE.md's "a deploy target, not a checkout", which keeps the box at current state
only.

**So rule 1 justifies no flag.** Its premise (`v24-unified-cli.md:74-83`) is that a
value the mini's gate or rig needs cannot be carried by a rebuild there. Nothing a
gate or rig does on the mini needs one: the harness and the binary both arrive from
the dev box, and a value that must change on the mini changes by deploy, a release
build here and a copy. The keeps that rest on rule 1 alone, with no input to name
(rule 2) and no per-invocation product choice (rule 3), lose their defence and go
to SHRIKE-4's test:

- generate's `--logits-head`, `--follow-up` and `--expert-cache-slots`, which the
  gate does not need (above);
- generate's `--dump-logits`, `--dump-hidden` and `--tokenize`, kept for their
  tools;
- bench's seven, `--arms`, `--positions`, `--repeats`, `--warmup`, `--layer`,
  `--experts` and `--batch`, kept because the benches run on the mini
  (`v24-unified-cli.md:157-175`).

The rest of v24's rule-1 keeps stand on another rule: generate's `--model`,
`--prompt` and `--messages-file` name inputs; its `--max-new`, `--temperature`,
`--seed`, `--thinking` and `--quiet` are per-invocation choices; and serve's
launch-line flags (`--port`, `--max-context`, `--thinking`, `--ram-budget`) are a
server's launch configuration, whose home is SHRIKE-5's question, not this
chapter's.

## Verdicts

Each flag judged on its own, per the test above. The first two are generate's,
deleted; the rest follow by command.

- **`--logits-head`, prosthetic, deleted.** It chose the logits head for a greedy
  run, which only the golden's imitation of the server wanted; the golden's
  server profiles now run the server's head itself, and `generate` still takes
  the logits head whenever it samples or dumps.
- **`--follow-up`, prosthetic, deleted.** It re-implemented the server's cached
  continuation, and the step-zero run measured that it is not the same
  computation: the same tokens in a different chunk split, a different second
  turn on both boxes. `serve-turns` checks the continuation production computes.

### generate

- **`--model`, legitimate.** It names the bundle to run, a path or an installed id.
- **`--prompt`, legitimate.** A raw-completion prompt, and the only route to one: the
  server renders every prompt as ChatML.
- **`--messages-file`, legitimate.** It names the chat to render, the input the
  server's `messages` carries over HTTP.
- **`--max-new`, legitimate.** The request's `max_tokens`, a per-run choice.
- **`--temperature`, legitimate.** The request's `temperature`; its default is the
  server's fallback (`OpenAIModels.swift:361`).
- **`--top-k`, legitimate.** The request's `top_k`, and one value more: 0 turns
  truncation off, which the request's 1...256 cannot (`OpenAIModels.swift:372`).
- **`--top-p`, legitimate.** The request's `top_p`.
- **`--repetition-penalty`, legitimate.** The request's `repetition_penalty`.
- **`--seed`, legitimate.** The request's `seed`, for a sampled run. At temperature 0
  on the fused head the sampler never draws, so the golden's `--seed` is inert.
- **`--stop`, legitimate.** The request's `stop`.
- **`--max-context`, legitimate.** A run's context is the user's choice. Its default
  of 4096 against the server's 262144 is two defaults for two jobs
  (`v23-argument-parsing.md:39-41`); the ranges the two commands accept differ,
  which SHRIKE-47 unifies.
- **`--rope-scaling`, legitimate.** The only way to reach YaRN's extended context, a
  path the engine carries and `YaRNRoPETests` covers. That nothing in the repository
  passes it describes our runs, not the capability.
- **`--kv-bits`, legitimate.** KV-cache precision trades memory against fidelity, a
  choice a user on a different box makes; production runs the default, 8 bits.
- **`--thinking`, legitimate.** The reasoning mode, a per-run choice.
- **`--quiet`, legitimate.** It suppresses the load line and the timing footer, a
  user's convenience. The golden passes it without needing it: both go to stderr
  (`Run.swift:237-238`, `:249-255`), which the golden already writes to its own file.
- **`--expert-cache-slots`, prosthetic, deleted.** Its default of 64 sits
  below the cliff the default budget was measured against, 9.91 tok/s at 64 slots
  and 18.91 at 128 (`RuntimeConfiguration.swift:93-94`), so a bare `generate`
  decodes at about half speed, and until `2721903` the golden passed 160 by hand to
  reach production's pool. Its one other use, running `generate` under production's
  slot table, whose total must equal the uniform count across the routed layers
  (`RuntimeConfiguration.swift:285-290`), a budget carries as well. `generate` takes
  serve's `--ram-budget` and its default in its place: one knob for both commands,
  the memory a user can give, with the slot count its outcome for each model.
- **`--dump-logits`, legitimate.** It writes the logits the engine computes at every
  position, the direct analogue of `-dump-ast`; `logit-compare.py` reads it.
- **`--dump-hidden`, legitimate.** It writes the residual before the final norm, the
  engine's own state. Its one reader, `q3-drafter-routes.py`, served closed work,
  which does not change what the flag exposes.
- **`--tokenize`, legitimate.** It shows what a run would send the model without
  loading it; `expert-pool-replay.py` and `q3-drafter-routes.py` parse its output.
- **`--force-tokens`, legitimate, and out of argv until it has a consumer.** It left
  in v24 with none (`ca5374f`); the runtime half remains, tested
  (`GenerationConfig.forcedTokens`, `Sampler.swift:33`). Feeding fixed ids in place
  of the sampler is how two builds' logits are compared position by position, which
  anyone checking numerics needs, and SHRIKE-20's gate is built on it. It returns
  as a `generate` flag beside `--dump-logits` when that gate is built.

### serve

- **`--model`, legitimate.** It names the one bundle to serve.
- **`--config`, legitimate.** It names the multi-model configuration file.
- **`--port`, legitimate.** Where the server listens, its launch configuration.
  Where launch configuration lives is SHRIKE-5's question.
- **`--max-context`, legitimate.** The context the server reserves; production
  passes 32768 against a default of 262144.
- **`--thinking`, legitimate.** The server's default reasoning mode.
- **`--reasoning-effort`, legitimate.** The server's default deliberation level for
  Harmony models, which a request's `reasoning_effort` overrides.
- **`--kv-bits`, legitimate.** As generate's.
- **`--rope-scaling`, legitimate.** As generate's.
- **`--ram-budget`, legitimate.** The memory the expert cache may use, the knob a
  user sizes for their box; slots derive from it and the model's expert stride. Its
  default of 8 GiB is the budget that first held the measured routing working set
  (`RuntimeConfiguration.swift:79-104`); production's 160 slots are the mini's own
  budget, passed at launch.
- **`--expert-cache-slots`, prosthetic, deleted.** A second spelling of what
  the budget derives, winning over it when both are given. Nothing passes it, and
  only a parse test names it.

### repack

- **`install --model`, legitimate.** It names the model to install.
- **`--output` on install, import-snapshot and discard-partial, legitimate.** It names
  the destination.
- **`import-snapshot --input-snapshot` and `--model-id`, legitimate.** They name the
  snapshot and its id. The subcommand exists to import Ornith's MTP draft
  (`ShrikeRepackCommand.swift:64-66`), which nothing consumes now
  (`ModelRoster.swift:83`); whether it stays is SHRIKE-18's question, and its flags
  go with it.
- **`verify-install --input-gturbo`, legitimate.** It names the install to re-attest.

### bench

The flags are what makes a bench a bench; the verb is the prosthetic. `bench` is in
the product binary only because the mini received one binary, and step zero found a
harness carries whatever a run needs there. It leaves `shrike` for a development
executable in the package, copied to the mini for a bench run, and its flags leave
`shrike`'s argv with it. Bench's `--seed` (a `BenchSeed`, which takes hex) and expert
bench's `--model` (a path, with no id resolution) are not generate's flags of those
names. Judged as a bench's flags:

- **attention `--arms`, legitimate.** The kernel variants to measure. Its default
  ladder includes `prodstream`, the runner's path (`Arms.swift:60`); `copy` names
  the kernel shipped before the streaming scan (`Arms.swift:61`), the state since
  the move.
- **attention `--positions`, legitimate.** The context lengths measured.
- **`--repeats`, legitimate.** The timed command buffers, whose median is reported.
- **`--warmup`, legitimate.** The untimed command buffers before them.
- **`--seed`, legitimate.** It fixes the synthetic rows or the activation vector.
- **expert `--model`, legitimate.** It names the bundle whose experts are read.
- **expert `--layer`, legitimate.** It names the layer read.
- **expert `--experts`, legitimate.** Every pass runs eight experts, the
  kernel's fixed top-k, repeating the last one loaded when fewer load
  (`Kernels.swift:37`); what it varies is how many distinct experts are read,
  which is what its help now says.
- **expert `--arms`, legitimate only while there are arms to choose between.**
  `coded` and `coded+aux` belong to v21's compression experiment, closed at its first
  measurement (`v21-compression.md:343-350`), and go with the move, which leaves
  `plain` alone; the flag goes with them, and a new variant brings both back.
- **expert `--batch`, legitimate.** Dispatches per command buffer, so the GPU holds
  its clock.
- **SHRIKE-52's run, recorded.** Measured 2026-09-26 on the dev box (Mac16,7), a
  release build of this chapter, `shrike-bench expert` read ornith15 at 4-bit, layer 20,
  eight experts, repeats 15, warmup 3, batch 20 dispatches per command buffer, and
  printed 1,179,648 bytes per expert, 51.8 µs per dispatch, 182.03 GB/s.
