# v24: one shrike, one verb tree

Five executables become one `shrike` carrying a verb tree, the flag surface
trimmed against evidence rather than against a target, and the model resolved
from configuration instead of named on every invocation. Companion plan:
[v24-implementation-plan.md](v24-implementation-plan.md), whose checkboxes are
the status of record. Starts from `main` at `8e50806`. Not a perf chapter;
nothing here should move a token.

## Why

Three changes, one argv break. All three are argv-visible, and argv here is a
production contract: the mini's launch line lives in this repo's CLAUDE.md, eight
tools shell out to these binaries, and every consumer is in this repo, which is
what makes changing it tractable at all. Done apart, they would move that launch
line three times.

They also inform each other. One root makes it visible which flags are shared
across verbs, which are per-verb, and which nobody names, and that is most of the
classification the trim needs. The trim in turn decides whether a flag survives
into the tree at all, so designing the tree first and trimming second would
design a tree around flags due for deletion.

## What is there today (read at `8e50806`)

Five binaries, each a `ParsableCommand` in a library target under a three-line
`@main extension` shim, which is what [v23](v23-argument-parsing.md) left behind
so that unifying is deleting the shims and adding a root.

| binary | flags | what it is |
| --- | --- | --- |
| `ShrikeCLI` | 24 | one generation from a prompt or messages file, plus the numerics instrument |
| `ShrikeServer` | 23 | the OpenAI-compatible HTTP API; the mini's production command |
| `ShrikeRepack` | 10 slots, 7 distinct | `install`, `import-snapshot`, `verify-install`, `discard-partial` |
| `ShrikeAttnBench` | 5 | the decode attention scan on synthetic rows, plus a `list` subcommand |
| `ShrikeExpertBench` | 8 | the decode phase-1 gate/up kernel over real experts |

Seventy flag slots, excluding `--help` on each. That total matches v23's opening
count by coincidence rather than by nothing having changed: v23 retired
AttnBench's `--list` into a subcommand, and Repack's nine flags became ten slots
once four subcommands each declared the ones they need, with `--output` and
`--overwrite` appearing in more than one.

A third of `ShrikeCLI`'s surface is not generation at all. `--tokenize` renders
the prompt and exits **without loading the model**; `--dump-logits`,
`--dump-hidden` and `--force-tokens` are the class-2 numerics instrument that
`tools/logit-compare.py` and the v19 and Q3 work were built on.

### The inventory (Task 1)

Seventy slots, **52 distinct flags**. Fourteen are declared by more than one
command, which is the 18 duplicate slots: `--model` four times (generate, serve,
bench expert, repack install), `--seed` and `--output` three times each, and
eleven more twice. Collapsing those into one definition each takes 70 to 52 and
removes no capability.

Consumer counts below are occurrences in `tools/`, `CLAUDE.md` and `README.md`.
Tests are counted separately and deliberately do not confer survival: a pin
exists because the flag exists, so it is evidence of coverage, not of use.

**A consumer count is not a `rg` count.** Four ambiguous hits were opened and
three were the consuming script's *own* flag, not an invocation of ours:
`--output` in `convert_kimi_tokenizer.py`, `--layer` in `expert-pool-replay.py`
and `--repeats` in `ane-probes/shrike_ane_attention_probe.py` are all
`parser.add_argument` declarations. Only `--prompt-cache-mode` in
`v6-probes/restore_fidelity_probe.py` was real, and that is prose in a docstring
describing a manual two-pass A/B rather than an automated invocation. The counts
below are after that correction.

### The test (owner's rulings, 2026-09-19)

A flag survives if **any** of:

1. it appears in a launch line, gate script or rig that runs on the mini, which
   has no checkout, so the value cannot be carried by a rebuild there;
2. it names what to operate on, an input the command cannot infer;
3. it is a per-invocation product choice rather than a setting.

Otherwise it becomes a constant. The reasoning behind rule 1: a release build is
33 seconds on the dev box, so a settled value costs nothing to carry in source,
while changing one on the deploy target costs a rebuild, a 22 MB copy with three
bundle directories, a relaunch and a golden re-run. The question each flag
answers is therefore *can this value be changed on the dev box alone?*

The rule that is **not** applied: moving settled values to configuration keys.
An earlier draft proposed it; the owner's ruling is that a value tweaked rarely
during development belongs in source, since a dev-only CLI flag buys no
flexibility that the `SHRIKE_*` environment layer (15 names, tripwired since
v17) does not already provide better. Configuration stays small: `models_dir`,
the default model, and the three keys it already has.

**generate** (24): 20 keep, 4 delete

| flag | what it does | consumers | disposition |
| --- | --- | --- | --- |
| `--model` | the `.gturbo` to load | 12 | keep, launch line |
| `--prompt` | raw-completion prompt | 8 | keep, gate |
| `--messages-file` | JSON chat messages | 1 | keep, gate |
| `--max-new` | generated-token limit | 1 | keep, gate |
| `--max-context` | native context limit | 4 | keep, and **unify** |
| `--temperature` | sampling temperature | 2 | keep, gate |
| `--top-k` | top-k truncation | 0 | keep, per-invocation |
| `--top-p` | nucleus truncation | 0 | keep, per-invocation |
| `--repetition-penalty` | repetition penalty | 0 | keep, per-invocation |
| `--seed` | deterministic sampling seed | 1 | keep, gate |
| `--stop` | stop substring, repeatable | 0 | keep, per-invocation |
| `--expert-cache-slots` | routed-expert slots per layer | 3 | keep, gate on the mini |
| `--kv-bits` | KV-cache precision | 1 | keep, owner's ruling |
| `--thinking` | reasoning mode | 5 | keep, launch line |
| `--quiet` | suppress the timing footer | 3 | keep, gate |
| `--logits-head` | force the logits head at temperature 0 | 2 | keep, gate |
| `--dump-logits` | fp16 logits per position | 3 | keep, `logit-compare.py` |
| `--dump-hidden` | fp16 residual before the final norm | 2 | keep, `q3-drafter-routes.py` |
| `--tokenize` | render the prompt and exit, no model load | 4 | keep, two tools |
| `--follow-up` | a second turn from the held state | 1 | keep, gate |
| `--rope-scaling` | none or yarn | 0 | **delete** |
| `--prefill-chunk` | prefill chunk tokens | 0 | **delete** |
| `--concise` | injects the concise-mode system prompt | 0 | **delete** |
| `--force-tokens` | feed fixed ids in place of the sampler | 0 | **delete** |

**serve** (23): 9 keep, 14 delete

| flag | what it does | consumers | disposition |
| --- | --- | --- | --- |
| `--model` | serve exactly this model | 12 | keep, launch line |
| `--config` | multi-model config file | 1 | keep, owner's ruling |
| `--port` | loopback port | 4 | keep, launch line |
| `--max-context` | native context | 4 | keep, launch line |
| `--kv-bits` | KV-cache precision | 1 | keep, owner's ruling |
| `--thinking` | reasoning mode | 5 | keep, launch line |
| `--reasoning-effort` | Harmony deliberation level | 1 | keep, owner's ruling |
| `--expert-cache-slots` | slots per layer | 3 | keep |
| `--ram-budget` | bytes the expert cache may use | 8 | keep, launch line |
| `--model-id` | API model identifier | 4 | **delete**, proven redundant |
| `--models-dir` | directory scanned for bundles | 0 | **delete** |
| `--preload` | load the default model at startup | 0 | **delete** |
| `--rope-scaling` | none or yarn | 0 | **delete** |
| `--queue-limit` | maximum queued requests | 0 | **delete**, no test either |
| `--prompt-cache-mode` | prefix reuse mode | 1 | **delete**, settled |
| `--prompt-cache-entries` | retained prefixes | 0 | **delete** |
| `--prompt-cache-memory-mib` | RAM snapshot budget | 0 | **delete** |
| `--prompt-cache-disk` | persistent SSD cache directory | 0 | **delete**, inherited |
| `--prompt-cache-disk-mib` | SSD snapshot budget | 0 | **delete**, inherited |
| `--prefill-chunk` | prefill chunk size | 0 | **delete** |
| `--reasoning-retention` | history render form | 0 | **delete**, three surfaces |
| `--lazy-load` | nothing | 0 | **delete**, dead |
| `--idle-unload-seconds` | release weights when idle | 0 | **delete** |

**repack** (7 distinct): 5 keep, 2 delete. `--model`, `--output`,
`--input-snapshot`, `--input-gturbo` and `--model-id` all name what to operate
on. `--overwrite` goes: refuse and let the caller remove. `--resume` goes and
takes its behaviour with it in the better direction: a partial download is
already saved, so resuming becomes automatic and `discard-partial` is how you
say "start over".

**bench** (7 distinct): all keep, **behind a debug-only verb**. `--arms`,
`--positions`, `--repeats`, `--warmup`, `--layer`, `--experts`, `--batch` have no
external consumer once the three false positives above are removed, but a bench
with hardcoded arms is not a bench. `mini-deploy.sh` already copies only
`ShrikeServer`, `ShrikeCLI` and `ShrikeRepack`, so the benches are already not
deployed; `bench` is compiled out of release builds and its flags stop counting
against the shipped surface.

### Two findings that are not flags

**`--model-id` on serve is redundant, and its help is wrong.**
`ModelRoster.single` derives the bundle name from the directory, stripping
`.gturbo`, and `resolve()` takes `override?.id ?? candidate.bundleName`
(`ModelRoster.swift:109`), so `--model ./models/ornith15.gturbo` already yields
the id `ornith15`. The mini's launch line passes `--model-id ornith15`, which is
exactly what the default produces. The help claims the default is "derived from
the installed model manifest", but the namespace only ever maps `entry.id` and
`entry.bundleName`, never `manifestModelID`, so a client cannot select by the
manifest id at all.

**The sole-bundle default already exists.** An earlier draft of this document
called "the models directory's only bundle, when it holds exactly one" the one
new rule this chapter adds. It is not new: `ModelRoster.swift:142` already does
`if defaultID == nil, entries.count == 1 { defaultID = entries[0].id }`. The
whole resolution chain exists; the only thing missing is that generate never
calls it.

### The result

| command | before | after |
| --- | ---: | ---: |
| generate | 24 | 20 |
| serve | 23 | 9 |
| repack | 7 | 5 |
| bench | 7 | 7, behind a debug verb |

Eighteen flags deleted, 34 distinct remaining, 27 in the shipped surface. Six of
serve's nine are in the mini's launch line today. The count is an output of the
test above, not a target: the aim discussed was roughly a dozen, and the evidence
did not support going below this without deleting working behaviour.

**The one dead flag.** `--lazy-load` documents itself as "This is the default;
the flag remains accepted for compatibility", and the code agrees: `lazyLoad`
occurs exactly twice in `sources/`, its declaration at
`ShrikeServerCommand.swift:157` and a branch at `:217` that exists only to reject
it when combined with `--preload`. It selects no behaviour. It goes with that
branch, which is v17's rule exactly.

**The drift.** `--max-context` is one flag with two contracts. The server
enforces membership in `RuntimeConfiguration.supportedContextTokens` and renders
both its help and its error from that constant (`ShrikeServerCommand.swift:272`),
which was v23's fix. Generate enforces a **range** instead,
`1...nativeMaximumContextTokens`, in `validateContext()`
(`ShrikeCLICommand.swift:227`), with the YaRN set when scaling is on. The
defaults differ too, 4096 against 262144.

A first draft of this section said generate "performs no argument-time validation
at all", inferred from `--max-context 1` parsing. That was wrong, and wrong in an
instructive way: the value parsed because it is legal under generate's own rule,
not because no rule ran. `ShrikeCLICore` genuinely never references
`supportedContextTokens`, but the conclusion drawn from that did not follow.
Generate's help is accurate about generate. The defect is that two siblings
enforce different contracts for one flag, and under one root they sit in one help
tree, so it is resolved rather than inherited.

### Who names these binaries

| consumer | what it names |
| --- | --- |
| `tools/golden-baseline.sh` | `ShrikeCLI` by path, plus a `pgrep -f` guard |
| `tools/mini-deploy.sh` | copies `ShrikeServer`, `ShrikeCLI`, `ShrikeRepack`; launches the server |
| `tools/decode-rig.sh`, `tools/turn-rig.sh` | launch `./bin/ShrikeServer`; `pkill -f`, `pgrep -x ShrikeServer` |
| `tools/ane-probes/shrike_ane_prefill_ab.py` | the `ShrikeServer` binary path |
| `CLAUDE.md` | the mini's production launch line, the `verify-install` command, the pgrep guard |
| `VerifiedInstallTool.swift:81` | writes `toolVersion: "ShrikeRepack verify-install"` into every receipt |
| `VerifiedInstallReceipt.swift:171` | prints `swift run -c release ShrikeRepack verify-install` as remediation |

The last two outlive the code: a receipt on disk carries the spelling it was
issued under, and the remediation text is what a user reads when a moved model
fails to load.

## Step zero, measured

Three questions were open at the chapter's start. Each was answered by running
something, not by reading more carefully, which is v23's first lesson.

**One executable links.** A throwaway root depending on all five command cores
built at 22 MB with zero errors and zero warnings, `ShrikeCLICore` and
`ShrikeServerCore` in one graph alongside NIO, the Hugging Face streaming stack
and ArgumentParser. No duplicate symbols and no conformance collisions, which is
`ShrikeArgumentSupport` doing the job v23 created it for.

**`Bundle.module` resolves per module.** Resource bundles are named
`Shrike_<Target>.bundle`, so a collision is not representable. Proven rather than
inferred: the attention bench ran from the unified binary and produced numbers,
and that path needs two modules' bundles in one process, since
`AttnBench/Runner.swift:21` builds Shrike's `MetalContext` (compiled from
`Shrike_Shrike.bundle`) while the bench's own kernels load from
`Shrike_ShrikeAttnBenchCore.bundle`. The third resource-bearing module,
`ShrikeExpertBenchCore`, needs real experts to run and was not executed; its
mechanism is the same, and the hardcoded cross-bundle path it uses was confirmed
to resolve (see below).

**A root with required options does not dispatch to subcommands.** With `--model`
declared required on the root, `probe serve --port 9000` fails with
`Missing expected argument '--model <model>'` at exit 64: ArgumentParser enforces
the root's required arguments before reaching the subcommand. Declared optional
and enforced in the root's own `run()`, every case behaves:

| invocation | result | exit |
| --- | --- | --- |
| `probe` | prints help | 0 |
| `probe --model X --prompt hi` | root ran | 0 |
| `probe serve --port 9000` | serve ran, root flags not demanded | 0 |
| `probe repack verify-install --input-gturbo m.gturbo` | ran, two levels deep | 0 |
| `probe srve` | `Error: Unexpected argument 'srve'` | 64 |

This is the constraint the whole shape rests on, and it has a cost recorded under
Declared changes below.

**The root's `validate()` runs even when a subcommand runs.** ArgumentParser
validates the whole command chain from root to leaf, so a probe invoked as
`probe serve --port 9000` executed the root's `validate()` *and* the `validate()`
of its `@OptionGroup` before serve's own `run()`. Both fire; the group's fires
first.

That is fatal to the obvious design, because `ShrikeCLICommand.validate()` throws
`"one of --prompt or --messages-file is required"` when both are absent
(`ShrikeCLICommand.swift:200`). Hung on the root as-is, it would fail every
`shrike serve`, every `shrike repack` and every `shrike bench`. So it is not only
`--model` that leaves ArgumentParser's hands: **all ten of generate's cross-flag
rules move into the root's `run()`**, and neither the root nor its option group
may declare a `validate()` at all.

The diagnostic contract survives that move intact, which was measured rather than
assumed. A `ValidationError` thrown from `run()` prints the same
`Error: <message>` line, the same usage block and the same footer, and exits 64,
identically to one thrown from `validate()`.

**And then a third finding killed that design outright.** Built for real, the
root carrying generation's options collided with its subcommands: a flag name
declared by both binds to the **root**, in either position, and the subcommand
never sees it. `shrike serve --model X` reached serve with `model` nil, so it
scanned the models directory instead, and `--max-context`, `--thinking`,
`--kv-bits`, `--expert-cache-slots`, `--rope-scaling` and `--prefill-chunk` were
all silently ignored on serve. That is the mini's production launch line failing
quietly, which is the exact failure mode this chapter exists to avoid.

**The resolution is `defaultSubcommand`.** Generation stays a leaf command,
`ShrikeGenerateCommand` with `commandName: "generate"`, and the root declares it
as the default so it is reached without being typed. Each command then owns its
own flags. Verified: `serve --model m.gturbo --max-context 32768 --thinking off
--kv-bits 4 --expert-cache-slots 160` lands every value on serve, pinned by
`serveKeepsEveryFlagItSharesWithGenerate`.

This supersedes the two findings above rather than building on them. Because
generation is no longer the root, its `validate()` stays exactly where v23 put
it, all ten cross-flag rules stay in it, and `drive()` and `run()` are unchanged.
The root declares no options at all.

Two costs of the resolution, both taken deliberately. `shrike --help` lists
subcommands with `generate (default)` rather than generation's flags, so
discovering them needs `shrike generate --help`; the alternative was the silent
break above. And a bare `shrike` is currently a **usage error at exit 64**,
printing `Error: Missing expected argument '--model <dir>'` above the root's
help, because generate's `--model` is still required. Task 3 makes it optional so
it can resolve from configuration, which is what turns a bare invocation into
something useful.

## The shape

Three verbs and a bare form. The bare form generates, following `claude -p` and
the tools it resembles, rather than spending a verb on the thing the binary is
named for.

```
shrike --model M --prompt "…"              generation, and the instrument flags
shrike serve  --model M --port 8081        the HTTP API; the mini's production command
shrike repack install | import-snapshot | verify-install | discard-partial
shrike bench  attention | expert
```

Naming convention: standard lowercase, which is what ArgumentParser already
generates from type names (`verify-install`, `import-snapshot`). No scheme is
invented. `serve`, `repack` and a nesting `bench` were the owner's own
illustrations when the chapter was ruled; the bare generation form and the two
bench children were settled at step zero.

## Model resolution

`--model` stops being required on the bare form, resolved instead in this order:

1. `--model <path or id>`, explicit, always wins.
2. The configuration file's entry marked `default: true`.
3. The models directory's only servable bundle, when it holds exactly one.
4. Otherwise an error naming the ids it found.

Rules 1 and 2 already exist for the server: `ModelRoster.defaultID` is set only
by an override carrying `default: true`, which is why v23 had to fix `--preload`
exiting 64 when no default was configured. Rule 3 is the one new rule, and it
means a machine with one model installed needs no configuration file at all.

A second piece follows from it: **generate's** `--model` accepts an id as well as
a path. A value naming an existing directory is taken as a path; anything else is
an id resolved against the models directory. Serve's `--model` deliberately stays
a path, because it pins one bundle and ignores any config or roster by design.

An earlier draft went further and had this replace the hardcoded candidate list
at `tools/golden-baseline.sh:52`, which walks
`/Volumes/BuildSSD/shrike/ornith15.gturbo` then
`$HOME/shrike-runtime/models/ornith15.gturbo` and takes the first holding a
`verified-install.json`. **That is reversed.** Id resolution reads the models
directory, which defaults to `~/shrike-runtime/models` and is otherwise set in
the config file. On the mini the models are there and `--model ornith15` would
work; on the dev box they are on `/Volumes/BuildSSD/shrike`, so the same
invocation would need a `~/.shrike/config.json` that is not in the repository. A
gate that passes only on a configured machine is worse than the loop it replaced.

That is the second correction this document has needed about
`golden-baseline.sh`, both for one reason: the baseline's bindings are not the
CLI's conveniences. A baseline is valid for one (machine, build, model) triple,
so the script names its own model and finds it its own way.

`ServerConfig` stops being server configuration the moment generate reads it, so
it moves out of `ShrikeServerCore` into a new `ShrikeCatalog` target that both
cores depend on, together with `ModelRoster`: 385 lines in two files. The type
follows the file, `ServerConfig` becoming `ShrikeConfig`, and
`~/.shrike/server.json` becoming `~/.shrike/config.json`. No such file exists on
either machine today, so nothing on disk migrates.

`ModelResolver` joins them, split so the half that matters is testable without
touching the filesystem: `resolve(requested:in:)` takes a roster and is pure,
while `resolve(requested:configPath:modelsDir:)` loads the config and scans
around it. Rules 1 and 3 turn out to need no new code at all, since
`ModelRoster.resolve` already sets `defaultID` from a `default: true` override
and already falls back to the sole entry.

## What is protected, and how

- **The golden baseline.** `tools/golden-baseline.sh --check` byte-identical on
  all five profiles, against a release build of the final code. Nothing here
  touches a runtime or kernel path, so a mismatch means the harness moved, not
  the numerics.
- **The invocation pins.** v23's argv tests assert that the command lines
  actually in use parse and that their values land where expected. Every one is
  rewritten against the new spelling in the task that changes it, with each
  message and exit code re-probed from the binary rather than assumed.
- **The four gates** per task, ThreadSanitizer once at the close.
- **A fresh-reader review of the whole branch at the close**, pointed at the
  documents as well as the code. At v23's close that review found two argv
  regressions that four green gates and 1,332 tests had not.

## Declared changes

1. **One executable.** `shrike`, with the five command types as a root plus three
   subcommand trees. The five old executable targets and their `@main` shims are
   deleted. Clean break: no forwarding shims, so anything still naming
   `./bin/ShrikeServer` fails loudly with file-not-found rather than quietly.
2. **The root declares no options.** Generation is `ShrikeGenerateCommand`, a
   leaf reached without typing via `defaultSubcommand`, because a flag shared
   between the root and a subcommand binds to the root and the subcommand never
   sees it (measured above). Nothing moves out of ArgumentParser's hands:
   generate keeps its `validate()` and all ten of its cross-flag rules.
3. **A bare `shrike` is a usage error at exit 64** until Task 3, naming
   `--model`. Pinned as such rather than aspirationally, with the pin carrying a
   note that Task 3 changes it.
4. **Model resolution** as above. `ShrikeConfig` and `ModelRoster` move to a new
   `ShrikeCatalog` target with `ModelResolver`, and the file becomes
   `config.json`. **`--model` stops being required on generate**, which changes a
   v23 pin rather than breaking one: `modelAndPromptAreRequired` asserted both
   halves, and only the prompt half survives, so it becomes
   `aPromptIsRequiredButTheModelNeedNotBeNamed` and additionally asserts that
   `--prompt` alone now parses with `model == nil`. A resolution failure surfaces
   at run rather than at parse, and exits 1 rather than 64, matching v23's ruling
   that a config failure is not a usage error.
5. **The flag trim: 18 flags deleted**, per the inventory's disposition. v17's
   rule holds throughout: the honest removal deletes a flag together with the
   code path it selected, so `--prompt-cache-disk` takes the disk snapshot path,
   `--concise` takes the prompt injection, `--rope-scaling` takes YaRN's
   reachability, and `--resume` is replaced by resuming automatically when a
   partial exists. `bench` is compiled out of release builds, taking seven more
   flags off the shipped surface without deleting them.
6. **The process name collapses**, and the tooling that identifies a Shrike
   process by name must move with it. `tools/decode-rig.sh:58` and
   `tools/turn-rig.sh:66` use `pgrep -x ShrikeServer` as their "did the old
   server actually die" check; with one binary every process is named `shrike`
   and `-x ShrikeServer` never matches again, so the check silently reports
   "not running" forever. The `pgrep -f` guards in CLAUDE.md and
   `tools/golden-baseline.sh:94` survive, because `shrike serve` is still
   distinguishable on a full command line, but their patterns are rewritten. This
   is the same silent-failure shape CLAUDE.md already records for `NVMAI_*`
   environment variables, so it is a task rather than a cleanup.
7. **`tools/mini-deploy.sh` copies one binary** instead of three, with the three
   `*.bundle` directories still required beside it.
8. **The receipt's `toolVersion` and the remediation text** move to the new
   spelling. Receipts already on disk keep the old string, which is provenance
   and correct.

## Out of scope

**Per-model sampling defaults.** `GenerationDefaults` (`Sampler.swift:6`) is one
global set, temperature 0.6, top-k 20, top-p 0.95, read both by generate for its
flag defaults (`ShrikeCLICommand.swift:82,86,90`) and by the server as its
per-request fallback (`OpenAIModels.swift:361`). Those are Qwen3's recommended
values and they are applied to all six installed bundles, `gpt-oss-20b` and
`kimi-linear-48b` included, so this is a correctness gap rather than a tidiness
one. The right shape is the precedence chain v23 already built, flag > config >
env > built-in, with configuration carrying a per-model default and the flag
overriding for one invocation: **not** a replacement for the flags, since
`golden-baseline.sh` passes `--temperature 0 --seed` to force determinism and
runs on a box with no checkout. Deferred to its own chapter because it breaks no
argv and so does not need this chapter's break (owner's ruling, 2026-09-19). Two
smaller drifts belong with it: `--repetition-penalty` defaults to an inline `1.0`
at `ShrikeCLICommand.swift:93` where its three siblings come from
`GenerationDefaults`, and `GenerationDefaults.presencePenalty` is exposed by no
flag at all.

**A further cut to the development flags.** The owner's standing reservation at
this chapter's opening is that several surviving flags still are not worth their
cost, kept "for the time being" rather than because the case for them is settled
(2026-09-19). The disposition above is therefore a floor, not a ceiling: a later
chapter revisiting it should start from the ones kept only by rule 3,
per-invocation choice, since rules 1 and 2 are evidence and rule 3 is judgement.

**`--tokenize` as its own verb.** It exits without loading the model, so it is
arguably not a generation flag at all, but whether it graduates to `shrike
tokenize` is a question the flag classification answers, not one to settle ahead
of it (owner's ruling, 2026-09-19).

**An interactive mode.** There is none today, and `shrike chat` is the obvious
name for one if it ever appears. Leaving the bare form as generation keeps that
name free, which is part of why the bare form wins over `shrike run`.

**The benches' table output.** Unchanged from v23's reasoning: no tool parses
that stdout, and a misaligned column is visible the moment the bench runs.

**`ShrikeExpertBench`'s cross-bundle reach.** `Kernels.swift:31` builds a path to
another module's resources off `Bundle.main.executableURL`:

```swift
let moeURL = executableDir.appendingPathComponent("Shrike_Shrike.bundle/Metal/MoE/moe.metal")
```

It survives unification, since that bundle sits beside whatever binary is
running, and the file was confirmed present at that path. But it hardcodes a
string encoding both the package name and a target name, and it would fail at
runtime rather than at compile time if either moved. Recorded here as a known
bound, not fixed in this chapter.
