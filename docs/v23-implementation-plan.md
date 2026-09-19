# v23 the argument surface: implementation plan

**Goal:** five hand-rolled parsers and their `main.swift` plumbing replaced by
`ParsableCommand`s that own their own entry points.

**Spec:** [v23-argument-parsing.md](v23-argument-parsing.md).

Six commits. The checkboxes here are the status of record.

## Constraints

- The four gates before any task is called done: `swift build -c release` (zero
  warnings), `swiftlint lint --strict`, `python3 tools/check-md-links.py`,
  `swift test --no-parallel`. ThreadSanitizer once at the close.
- No new `SHRIKE_*` variable. The registry holds 15 names and a test pins the
  count.
- `docs/v10-implementation-plan.md` is a peer's uncommitted edit. Never `git add`
  it, and never `git add docs/` wholesale.
- Commit subjects in the repo's style, ending `(v23 Tn)`. No `Co-Authored-By`.
- Comments only for a non-obvious why. Not for narrating the migration.

---

### Task 1: pin the invocations that exist

Before changing any parser, assert that the command lines actually in use parse.
Not a generated corpus: the real ones.

- [x] Collect every invocation from `tools/decode-rig.sh`, `tools/turn-rig.sh`,
      `tools/golden-baseline.sh`, `tools/mini-deploy.sh`,
      `tools/expert-pool-replay.py`, `tools/prefill-ledger.py`,
      `tools/logit-compare.py`, `tools/ane-probes/shrike_ane_prefill_ab.py`, plus
      the mini's launch line in `CLAUDE.md`.
- [x] Add them as cases in `tests/Shrike/Core/CLI/CLIArgumentsTests.swift` and a
      new `tests/ShrikeServer/ServerArgumentsTests.swift`, asserting each parses
      and that the values land where expected.
- [x] Gates, commit.

Repack and the two benches are executable targets, so their invocations cannot be
tested until their own tasks move the parsing into a command type. Pin them there.

---

### Task 2: ShrikeCLI

- [x] Add `swift-argument-parser` to `Package.swift` and to `ShrikeCLICore`.
- [x] `Args` becomes a `ParsableCommand` with `@main`: flags as declared
      properties, the cross-flag rules in `validate()`, `run()` calling the
      existing `run(args:)` in `Run.swift`.
- [x] `--top-k`'s "0 means off" and `--prefill-chunk`'s `auto` are expressed as
      the option's own type, not as sentinel values checked after the fact.
- [x] Delete `Args.usage`, `ParseContext`, `makeArgs`, `takeValue`, `takeInt`,
      `takeRawValue`.
- [x] The SIGINT cancellation bridge (`RunBox`, `drive`) survives, moved to
      `sources/ShrikeCLI/Drive.swift` beside the `run(args:)` it cancels: `Args`
      must stay in `ShrikeCLICore` for the tests, so its `run()` cannot reach a
      bridge left in the executable target. `Command/main.swift` and its
      parse-and-exit block are gone, replaced by `Command/ShrikeCLIMain.swift`
      holding `@main extension Args {}`, which a file of top-level code could
      not carry.
- [x] Task 1's tests still pass, unchanged. If one needs editing, the migration
      is wrong.
- [x] Look at `swift run ShrikeCLI --help` once. The allowed slot counts must end
      at 256 and nothing may run past the margin.
- [x] Gates, commit.
- [x] A second commit on this task, after the checkpoint above was recorded: the
      command type is `ShrikeCLICommand`, not `Args`, which named an argument bag
      it had stopped being; `--top-k` is a declared `TopKChoice` with no shadow
      property behind it, the call site reading `args.topK.tokens`; and the three
      `ExpressibleByArgument` conformances on Shrike's own enums move to a new
      `ShrikeArgumentSupport` target, since two modules conforming the same type
      collide the moment anything imports both. `<Binary>Command` is the naming
      convention for the rest of the chapter.

Watch: `Run.swift:60` declares a free `func run(args:)` and `ParsableCommand`
requires `run()`. Different labels, so no clash, but qualify if the compiler
disagrees.

---

### Task 3: ShrikeServer

- [x] `ServerArguments` becomes `ShrikeServerCommand`, an `AsyncParsableCommand`
      (its `run()` awaits) with `@main`, taking its enum conformances from
      `ShrikeArgumentSupport`; the 86-line body of
      `sources/ShrikeServer/Command/main.swift` moves into it as six stage methods
      over a `ResolvedRoster` context, per CLAUDE.md, rather than one long `run()`.
- [x] **Three** env vars, not two: parsing also read `SHRIKE_THINKING_MODE` via
      `ModelThinkingMode.resolved(environment:)`, so leaving it would have kept the
      environment parameter alive. All three move to `merging(configDefaults:
      environment:)`; `parse` takes argv only. A malformed value is still rejected
      before any config load or models scan, via `validateEnvironment` called at the
      top of `run()`, so launch-time diagnostics keep their old ordering.
- [x] `--max-context`'s help becomes the constraint the code enforces. The spec's
      diagnosis was wrong and is corrected there: the code never accepted `1`, it
      enforces membership in `supportedContextTokens`, so the help's range was
      *wider* than the code and `--max-context 50000` was refused as "not
      supported". Help and error now both render the set from the constant. The
      262144 default stands: it is the native maximum and correct for a server.
- [x] An unknown flag reports as unknown. Verified before and after against the
      binary: `--bogus requires a value` became `Unknown option '--bogus'`.
- [x] Task 1's server tests pass with only the type's name changed; every argv
      string and asserted value is untouched, the mini's launch line included.
- [x] Not in the plan but owed by it: both fixed defects are pinned, since a
      declared behaviour change with no test is the drift this chapter exists to
      stop. Three cases assert the unknown-flag message, that `--max-context`
      names every value it accepts, and that parsing reads no environment while
      `merging` resolves and rejects one.
- [x] Gates, commit.

---

### Task 4: ShrikeRepack, as four subcommands

- [ ] Move parsing out of `sources/ShrikeRepack/Command/main.swift` into
      `ShrikeRepackCore`, which `ShrikeRepackTests` already depends on.
- [ ] A root command with `subcommands: [Install, ImportSnapshot, VerifyInstall,
      DiscardPartial]`, each holding only the flags it needs and its own `run()`.
      The mode-validation block and the silent `return 2` both disappear.
- [ ] `--model` keeps its name inside `Install`, where a catalog name is the only
      thing it could mean.
- [ ] Update the three sites citing the old spelling: `CLAUDE.md:35`,
      `README.md:41`, `VerifiedInstallReceipt.swift:171`.
- [ ] Note Repack alone passes unstripped `CommandLine.arguments` today and skips
      element 0 itself. `@main` removes that entirely; make sure nothing else
      relies on it.
- [ ] Add its invocation tests.
- [ ] Gates, commit.

---

### Task 5: the two benches

- [ ] `ShrikeExpertBench` and `ShrikeAttnBench` each get a `ParsableCommand` with
      `@main`, named `ExpertBenchCommand` and `AttnBenchCommand`. Split a core library out of each executable target so the
      arguments are testable; leave `resources: [.copy("Metal")]` and the
      `Bundle.module` users in the executable.
- [ ] `--seed` parses the same way in both. AttnBench's hex form wins, since the
      defaults in both are written as hex literals. Neither help documents a
      format today, so this is tidying, not a fix.
- [ ] `--arms` and `--positions` stay comma-split single values, not repeated
      flags: an arm name cannot contain a comma and this is the existing surface.
- [ ] AttnBench's `--list` becomes a subcommand, since it makes the binary do
      something else and ignore everything.
- [ ] Add their invocation tests.
- [ ] Gates, commit.

---

### Task 6: the close

- [ ] `docs/architecture.md`: a v23 entry in History, and correct anything about
      how the binaries parse arguments.
- [ ] Tick this plan's boxes; record the close in the design doc.
- [ ] ThreadSanitizer once:
      `env TSAN_OPTIONS=suppressions=tsan-suppressions.txt swift test --no-parallel --sanitize=thread`
- [ ] `tools/golden-baseline.sh --check`. No kernel or runtime code changed, so it
      should be identical; run it anyway since it is the only check that
      exercises real inference. Counts as a model run, so `pgrep` first.
- [ ] A fresh-reader review of the whole branch against `main` before the merge.
      v20's close folded two real bugs out of its review and v22's five findings;
      a chapter that rewrites all five entry points does not skip it. Fold each
      finding into the commit that owns it, located with `git log -S` rather than
      the reviewer's attribution, then re-run the gates.
- [ ] Merge on Davor's go, delete the branch, update memory.
