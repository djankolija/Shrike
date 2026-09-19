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

The server's two environment-sourced settings (`SHRIKE_REASONING_EFFORT`,
`SHRIKE_REASONING_RETENTION`) move out of parsing to where the server assembles
its configuration. Parsing takes argv and nothing else.

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
3. **Error message text becomes ArgumentParser's.** Exit codes stay: 0 on help,
   non-zero on a parse failure.
4. **The five drifts above are fixed** rather than preserved.

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

The benches' table output. `ShrikeExpertBench/Runner.swift:123` and `:152` pair a
header format string with a row format string that must agree column for column
by eye, which is the same shape as the help text. It stays out, but not for the reason
first given here: **no tool parses that stdout**, checked at T5 across the whole
repo, and neither bench is invoked by any script, tool or test. It stays out
because a misaligned column is visible the moment the bench runs and nothing
automated depends on it, so unlike help text it cannot lie silently. The fix, when it is wanted, is a
column-spec type from which both the header and the row derive.
