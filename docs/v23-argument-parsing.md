# v23: the argument surface

Five hand-rolled argument parsers retired for `swift-argument-parser`, and the
binaries' entry points with them. Companion plan: `v23-implementation-plan.md`,
whose checkboxes are the status of record. Starts from `main` at `e9e6084`, the
v22 close. Not a perf chapter; nothing here should move a token.

## Why

At v22's close a `fixup!` corrected `--expert-cache-slots` help text in two
binaries that still said the allowed counts ended at 128, after v22 had raised
the ladder to 256. The fixup then introduced a second defect of the same kind,
running its replacement lines ten columns past a margin the rest of the block is
hand-aligned to. A flag's allowed values live once in code and are restated in
prose twice, and nothing checks that the restatements are true or even that they
fit. That is a shape that produces bugs, not a bug to fix once.

## What is there today (read at `e9e6084`)

Five binaries, 1,212 lines of parsing, **70 flags** plus help on all five. Each
is a `switch` over `case "--flag":` walking an index, beside a `static let usage`
string wrapped by hand to 80 columns, beside a `main.swift` that catches the
parse error, prints that string and picks an exit code.

| binary | parsing lines | flags |
| --- | --- | --- |
| `ShrikeServer/Core/ServerArguments.swift` | 448 | 23 |
| `ShrikeCLI/Args.swift` | 388 | 24 |
| `ShrikeRepack/Command/main.swift` | 241 | 9 |
| `ShrikeExpertBench/BenchArgs.swift` | 68 | 8 |
| `ShrikeAttnBench/BenchArgs.swift` | 67 | 6 |

Eight tools shell out to these binaries and the mini's launch line lives in the
repo's CLAUDE.md, so argv is a contract. Every consumer is in this repo, which is
what makes changing it tractable at all.

### What the full inventory actually exposes

Less than a first read suggests. Most apparent inconsistencies are deliberate,
and three candidates were checked and dropped: the CLI and server defaulting
`--max-context` to 4096 and 262144 is two correct defaults for two jobs, each
documented, with a config-file layer between default and flag on the server;
`--prefill-chunk auto` is a CLI convenience the server's help never claims; and
`--reasoning-retention` lowercasing its input makes it more lenient than its help
promises, which harms nobody.

Two real defects and one cosmetic one survive:

- **The server's help promises `--max-context` is `4096...262144`; the code
  enforces membership in a seven-value set.** Corrected at T3 after probing the
  binary: `--max-context 1` is *rejected*, not accepted, because `validate`
  checks `RuntimeConfiguration.supportedContextTokens`, which the flag's own
  `1...maximumContextTokens` guard hides on a first read. The defect is the
  other direction, and worse for it: `--max-context 50000` reads as legal
  against the help and is refused as "not supported", naming nothing. Help
  describing a surface the code does not implement, which is the chapter's own
  defect class.
- **`ShrikeServer --bogus` reports that the flag requires a value**, because
  `applyFlags` runs its value guard before it ever checks whether the flag is
  known.
- **`--seed` accepts `0x` hex in AttnBench and decimal only in ExpertBench.**
  Neither help documents a format, so neither lies; a sibling inconsistency worth
  fixing while the file is open, not a bug.

## The shape

Each binary's argument type becomes a `ParsableCommand` carrying `@main`, its
flags as declared properties, and a `run()` that does the work. The `usage`
string, the `ParseContext`, the hand-rolled `takeValue`, and `main.swift`'s
parse-and-exit plumbing all go. ArgumentParser owns the entry point, generates
the help, wraps it to the terminal and supplies `-h`.

`ShrikeRepack` becomes a root with four subcommands, because it has four
operations (`install`, `import-snapshot`, `verify-install`, `discard-partial`)
currently expressed as mutually exclusive flags guarded by a validation block
and a silent `return 2` when the combination is wrong.

The server's **three** environment-sourced settings move out of parsing to where
the server assembles its configuration: `SHRIKE_REASONING_EFFORT`,
`SHRIKE_REASONING_RETENTION`, and `SHRIKE_THINKING_MODE`, which this document
first missed because parsing read it indirectly through
`ModelThinkingMode.resolved(environment:)`. Leaving the third would have kept the
`environment:` parameter alive and missed the point. Parsing takes argv and
nothing else.

`ShrikeCLI`'s `run()` calls the existing `run(args:)` in `Run.swift` rather than
absorbing it. The SIGINT cancellation bridge is not argument parsing and
survives, beside the `run(args:)` it cancels rather than beside the entry point.

## What is protected, and how

Not by a generated corpus. By tests asserting that the invocations which actually
exist still parse: the eight tools' command lines and CLAUDE.md's launch line,
roughly fifteen cases written by hand.

## Declared changes

1. **Repack's modes become subcommands.** `ShrikeRepack --verify-install
   --input-gturbo X` becomes `ShrikeRepack verify-install --input-gturbo X`.
   All four modes move, not only that one, and there is no `defaultSubcommand`,
   so the bare install form becomes `ShrikeRepack install --output X` as well.
   The one in-repo issuer of Repack's argv is `RepackCLITests`, which spawns the
   binary; it exists to test that surface rather than to depend on it, and no
   script, tool or production path installs through the binary (the Mac app
   calls `ShrikeRepackCore` directly). So the choice is between a bare form kept
   alive for muscle memory and a `SUBCOMMANDS:` listing that names all four, and
   the listing wins. **Four** in-repo sites cite the old form, one more than this
   document first said: `CLAUDE.md`, `README.md`, `VerifiedInstallReceipt.swift`
   and `RepackModelInstallerClient.swift`. All four are updated in the task.
2. **`-h` gains the CLI and Repack.** The server and both benches already accept
   it; ArgumentParser supplies it everywhere.
3. **Error message text and the parse-failure exit code become ArgumentParser's.**
   Help still exits 0. A parse failure moves from **2 to 64**
   (`ExitCode.validationFailure`, `EX_USAGE`), so the first draft of this item,
   "exit codes stay", was wrong: only the help code stayed. No in-repo consumer
   distinguishes them; `golden-baseline.sh:130` tests `rc -ne 0`. The prefix also
   changes case, `error:` to `Error:`, and Repack loses its per-operation prefixes
   (`install failed:`, `verification failed:`), which the subcommand name now
   carries instead.
4. **The three drifts above are fixed** rather than preserved. (This item said
   "five", counting items 1 and 2 twice.)
5. **`ShrikeAttnBench --list` becomes `ShrikeAttnBench list`**, the same kind of
   change as item 1 and missed here at first. Nothing in the repo issues it.
6. **A bare `ShrikeRepack` prints help on stdout and exits 0**, where it used to
   print usage on stderr and exit 2, because a root with subcommands and no
   `run()` is a clean help request. A caller doing `ShrikeRepack … || handle`
   therefore no longer sees a failure on empty argv.
7. **An option value beginning with `-` is refused** where the hand-rolled parsers
   took the next token unconditionally. Restored with `parsing: .unconditional`
   on every option whose value is user-supplied text and may legitimately lead
   with a dash (`--prompt`, `--stop`, `--follow-up`, both `--model-id`s), so
   `--stop "-->"` works as before. It is **not** restored for options taking a
   path or a number, where a dash-leading value was already invalid; there the
   diagnostic changes from "invalid value" to "missing value", which is a worse
   message for the same rejection.
8. **`ShrikeCLI --bogus` alone reports the missing `--model` rather than the
   unknown flag**, since ArgumentParser reports a missing required argument
   first. With a complete command line it names the unknown option correctly.
   Noted because it is the inverse of the server defect item 4 fixes.

## Out of scope

Trimming the surface. Whether seventy flags should come down to roughly ten was
asked before the chapter started; the ruling is after the migration, not before
it (owner's ruling, 2026-09-19). Deleting a flag from a hand-rolled parser means
editing a `switch` and re-wrapping the usage block by hand, which is the defect
this chapter opens on; once a flag is a declared property, its help text goes
with it when it goes. The migration is behaviour-preserving and so checkable by
the golden and by the invocation tests above, while a trim is a behaviour change
that wants its own evidence; run together, a mismatch has two candidate causes.
The count is settled then rather than now, from what the pins and the migrated
properties show: the deletion list is the flags no in-repo consumer names and
that select between no live code paths, and v17's lesson is that the honest
removal takes a flag together with the path it selected.

One binary with a verb tree. The five executables stay five here. v24 unifies
them into a single `shrike` with subcommands, `shrike serve`, `shrike repack
verify-install`, `shrike bench attention` and so on, **together with the flag
trim above, as one chapter and one argv break** (owner's ruling, 2026-09-19):
both are argv-breaking, and doing them apart would move the mini's production
launch line twice. They also inform each other, since one root makes it visible
which flags are shared across verbs, which are per-verb and which nobody names,
and that is most of the classification the trim needs.

**The verbs themselves are not decided.** Only the naming convention is: standard
lowercase CLI naming, which is what ArgumentParser already generates from the
type names (`verify-install`, `import-snapshot`). Settle the tree at v24's step
zero, not before.

What this chapter leaves in place for it: every binary is a `ParsableCommand` in
a library target under a three-line `@main` shim, so unifying is deleting the
shims and adding a root. `ShrikeArgumentSupport` is load-bearing for it, since a
single executable links `ShrikeCLICore` and `ShrikeServerCore` into one graph and
conformances duplicated per parser would fail that link. Nothing links both
today, so **that is unverified**; step zero should prove the single executable
links at all, with NIO, the Hugging Face streaming stack and three separate Metal
resource bundles in one graph, and that `Bundle.module` still resolves per module.
A migration that breaks nothing is available if wanted: keep the five old binaries
for a chapter as one-line shims forwarding into the root, and delete them once the
rigs and CLAUDE.md have moved.

The benches' table output. `ShrikeExpertBench/Runner.swift:123` and `:152` pair a
header format string with a row format string that must agree column for column
by eye, which is the same shape as the help text. It stays out, but not for the reason
first given here: **no tool parses that stdout**, checked at T5 across the whole
repo, and neither bench is invoked by any script, tool or test. It stays out
because a misaligned column is visible the moment the bench runs and nothing
automated depends on it, so unlike help text it cannot lie silently. The fix, when it is wanted, is a
column-spec type from which both the header and the row derive.

## The close

Five hand-rolled parsers retired, five `main.swift` files gone, five binaries now
a `ParsableCommand` in a library target under a three-line `@main extension`.
Sources net −322 lines across 29 files; with the tests it is +2,063/−1,530 over
46, since the pins and the defect tests are new work rather than replacements.
No runtime or kernel code changed. ThreadSanitizer clean at the close, zero
warnings over 1,333 tests in 181 suites, and `tools/golden-baseline.sh --check`
byte-identical on all five profiles against a release build of the final code,
which is the only check here that exercises real inference.

### What this chapter got wrong about itself

Four claims in this document were false when written, and each was found by
running something rather than by reading more carefully.

- **`--max-context` "accepts `1`".** It never did. The flag's own
  `1...maximumContextTokens` guard sits above a `validate` that checks membership
  in a seven-value set, and stopping at the first guard gives exactly the wrong
  answer. The real defect pointed the other way and was worse: the help described
  a contiguous range the code does not implement, so `50000` read as legal.
- **"No in-repo consumer issues an install."** `RepackCLITests` spawns the binary
  and asserts on its exit codes. The search that missed it looked for the literal
  string `ShrikeRepack `, and that file builds argv as a Swift array. Five tests
  failing at T4 is how it surfaced.
- **"Three tools parse that stdout."** None do. Neither bench is invoked by any
  script, tool or test at all.
- **"Two environment-sourced settings."** Three. The third,
  `SHRIKE_THINKING_MODE`, was read indirectly through
  `ModelThinkingMode.resolved(environment:)`, so it did not appear in a search for
  the variable name near the parser.

The pattern is one thing: a claim about what a binary does costs one run of the
binary, and every one of these would have been caught by that.

### What the fresh-reader review caught that the gates could not

Two regressions passed four green gates and 1,332 tests.

- **An option value beginning with `-`** was refused, where every old parser took
  the next token unconditionally. `--stop "-->"` is the realistic break. No test
  would have caught it because no test used such a value, including the pin that
  was supposed to carry the awkward one: the `turns-lh` case passed a tidy
  `"And in one sentence?"` where the tool issues a string opening on a newline and
  carrying chat-template tokens. A pin is worth only the value it carries.
- **A config failure reported as a usage error.** `--preload` with no default model
  threw `ValidationError`, which exits 64 and prints the whole option list, while
  every sibling failure on that path exits 1.

Both are the same shape: the framework's defaults are not the old code's defaults,
and the differences surface in places no test would think to look. "Behaviour
preserving" is a claim to be tested, not a property the change has by being a
refactor.
