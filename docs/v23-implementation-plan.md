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

Corrected at T4: half wrong. `RepackCLITests` already tested Repack's argv
end-to-end by spawning the binary and asserting on its exit code and stderr, so
that surface *was* pinned, in a file a search for the literal string
`ShrikeRepack ` could not find because it builds argv as an array. Task 1 should
have found it. When T4 changed the spelling, those five tests failed, which is
the pin working; they are rewritten against the new surface in the same commit.

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

- [x] Move parsing out of `sources/ShrikeRepack/Command/main.swift` into
      `ShrikeRepackCore`, which `ShrikeRepackTests` already depends on.
- [x] A root command with `subcommands: [Install, ImportSnapshot, VerifyInstall,
      DiscardPartial]`, each holding only the flags it needs and its own `run()`.
      The mode-validation block and the silent `return 2` both disappear, and with
      them the per-mode `guard` chains that existed only to reject another mode's
      flags: a subcommand cannot see them. No `defaultSubcommand`, so `install` is
      typed like the rest; see the spec's declared change 1 for why.
- [x] `--model` keeps its name inside `Install`, where a catalog name is the only
      thing it could mean. (An early design memo said it became `--model-name`;
      the spec and this plan both kept `--model`, and that memo is stale.)
- [x] Four sites, not three: `CLAUDE.md:35`, `README.md:41`,
      `VerifiedInstallReceipt.swift:171` and, unlisted,
      `RepackModelInstallerClient.swift:138`. `VerifiedInstallTool.swift:81`
      already wrote `ShrikeRepack verify-install` into every receipt it issued,
      so that string was ahead of the surface and is now correct.
- [x] Repack alone passed unstripped `CommandLine.arguments` and skipped element 0
      inside `parse` via `var index = 1`. `@main` removes both; nothing else read
      that array.
- [x] Add its invocation tests. Ten cases, including that the retired flag
      spelling no longer parses and that one subcommand refuses another's options.
      Note a root with subcommands *returns* ArgumentParser's help command where a
      leaf command throws; both exit zero, and the test asserts the real shape.
- [x] `RepackCLITests`, the pre-existing end-to-end suite, rewritten against the
      new spelling: every message and exit code re-probed from the binary first
      rather than assumed. Operational failures still exit 1; a parse failure
      moved from 2 to ArgumentParser's `validationFailure`, named as such in the
      test rather than written as a bare 64. One case lost its subject, since
      `--resume` and `--discard-partial` can no longer be combined to be rejected;
      it now asserts that unreachability instead.
- [x] Gates, commit.

---

### Task 5: the two benches

- [x] `ShrikeExpertBench` and `ShrikeAttnBench` each get a `ParsableCommand` with
      `@main`, named `ExpertBenchCommand` and `AttnBenchCommand`, with a core library
      split out of each so the arguments are testable. The resources do **not** stay
      in the executable as this bullet asked: `Bundle.module` resolves to its own
      module's bundle, so leaving `Metal/` behind would strand the three files that
      load it, and the command's `run()` could not reach the runner anyway. Both take
      the ShrikeCLI shape instead, the core holding everything and the executable a
      one-line `@main` shim, which is the correction T2 already made for `drive`.
- [x] `--seed` parses the same way in both, through one `BenchSeed` in
      `ShrikeArgumentSupport`: hex or decimal, and its `defaultValueDescription`
      renders the default back as `0x5EED0019`, the spelling the source writes.
- [x] `--arms` and `--positions` stay comma-split single values, not repeated
      flags, as `CommaSeparatedNames` and `CommaSeparatedCounts` beside `BenchSeed`,
      so the split is the option type's business rather than the runner's.
- [x] AttnBench's `--list` becomes the `list` subcommand. A root may carry both
      subcommands and its own `run()`: a bare invocation still runs the bench, which
      a test pins.
- [x] Add their invocation tests, in a new `ShrikeBenchTests` target over both
      cores. Nine cases; there was nothing to inherit, since the sweep found no
      script, tool or test that invokes either bench, and no tool parses their
      stdout either (the spec's out-of-scope note is corrected on that point).
- [x] `BenchError.help` dropped from both benches. Its only thrower was the
      `main.swift` this task deletes, so it was parsing plumbing left standing.
- [x] Gates, commit.

---

### Task 6: the close

- [x] `docs/architecture.md`: a v23 entry in History, and five stale references
      corrected: the RAM-budget invariant's parse site, the two reasoning knobs'
      validation sites (doubly stale, since parsing no longer reads the environment
      at all), the environment tripwire's first call site in a `main.swift` that no
      longer exists, and the long-functions table's row for
      `ServerArguments.ParseContext.apply(flag:value:)`, a 110-line flag switch the
      table called three flags from breaching the ceiling. It is gone, not shortened.
- [x] Tick this plan's boxes; record the close in the design doc.
- [x] ThreadSanitizer once: clean, zero `WARNING: ThreadSanitizer`, 1,333 tests in
      181 suites in 853 s. The suppressions file was not touched.
- [x] `tools/golden-baseline.sh --check`: byte-identical on all five profiles on
      this box (`short`, `long`, `short-lh`, `long-lh`, `turns-lh`), against a
      release build of the final code. `pgrep` clear before it, nothing else on the
      machine during it.
- [x] A fresh-reader review of the whole branch against `main`. **No real bug**;
      every validation in all five old parsers transfers, and the server's stage
      methods preserve the order of operations. Fifteen findings, nine acted on:
      two real regressions this chapter introduced, four wrong claims in the spec,
      three weak tests, two dead or over-broad lines. The two that mattered:
      - **An option value beginning with `-` was being refused.** ArgumentParser's
        default strategy rejects a dash-prefixed next token where every old parser
        took it unconditionally, so `--stop "-->"` broke. Verified against the
        binary before and after. Restored with `parsing: .unconditional` on the
        options whose value is user-supplied text (`--prompt`, `--stop`,
        `--follow-up`, both `--model-id`s) and deliberately not on those taking a
        path or a number, where a dash value was already invalid. Pinned by a test.
      - **`--preload` with no default model exited 64 and dumped the usage block**,
        because T3 threw `ValidationError` for what is a config failure while its
        siblings on that path exit 1. Now a plain `ServerLaunchError`, exit 1.
      The four spec corrections are in the design doc's declared changes, which grew
      from four items to eight. The three weak tests: `--layer -1` never reached
      `validate()` (the equals form does), the golden `turns-lh` pin carried a tidy
      follow-up where the tool issues one opening on a newline and carrying
      chat-template tokens, and four rejections that exist for a *reason* now assert
      on `message(for:)` rather than on any error at all.
- [x] **Deviation:** the findings are NOT folded into their owning commits. They
      belong across all five task commits, and this harness does not support
      `git rebase -i`, so an interactive fold is unavailable. They land as one
      labelled review-fold commit naming the owning commit per finding.
- [x] Merge on Davor's go, delete the branch, update memory. **Ticked 2026-09-23:** `ae18920` is on `main` and `origin/main`, and no v23 branch remains.
