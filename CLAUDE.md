# Shrike — working instructions

This is a fork where active development happens. It was forked from `Pummelchen/NVMAI`,
which keeps that name — the rename to Shrike is ours alone. Upstream's posture — run and
report existing behaviour, don't edit source — **does not apply here**. Normal development
is expected.

What the project is, which models it supports and how to use it belong in
[README.md](README.md) and `docs/`. This file is only what an agent has to do differently.

## Never run two model processes

Before anything that loads a model — a server, the app, the CLI, a benchmark, or the golden
baseline — check:

```bash
pgrep -fl 'ShrikeServer|ShrikeMac|ShrikeDecodeService|ShrikeCLI|ShrikePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'
```

If something is already running, **stop and say so**. Never terminate a process you did not
start; another session may be mid-run. Run one model-using test at a time.

Also require macOS 26+, Swift 6.3+, enough disk, and an acceptable `memory_pressure -Q`
before a model run. If a check fails, report it and stop — do not reinstall or delete a
model to get past it.

## Installed models are bound to their path

A `.gturbo`'s `verified-install.json` receipt is bound to the absolute path it was
installed to, so **moving or renaming an installed model makes it fail to load** with
`trusted receipt invalid: model directory mismatch`. This is not corruption and does not
need a re-download. Re-issue the receipt in place:

```bash
swift run -c release ShrikeRepack --verify-install --input-gturbo <model.gturbo>
```

**Never hand-edit the receipt to match a new path.** The path binding is what detects a
moved or swapped directory; editing it forges the attestation instead of re-establishing it.

## Gates that must pass before calling work done

Five local gates. Nothing runs them for you — there is no CI — so run them yourself
before calling work done. All of them constrain how code gets written here:

1. **Release build with zero warnings.** A new warning fails the build.
2. **`swiftlint lint --strict --baseline .swiftlint-baseline.json`** — three rules:
   `force_cast`, `force_try`, and `function_body_length` (warn 120, error 400). A
   force cast or force try needs `// swiftlint:disable:next force_cast` (or
   `force_try`) on the line above it, with the reason stated in a comment. The 22
   functions already over 120 lines are recorded in the baseline; anything new fails.
   Regenerate with `swiftlint lint --write-baseline .swiftlint-baseline.json` when you
   legitimately fix one, or the gate fails on a stale entry. Decompose as you write.
3. **Markdown link check** — globs every `*.md` in the repo, so it binds on any document
   you add. Relative links must resolve.
4. **`swift test --no-parallel`** — serial, always. Pass `--filter` through as needed.
5. **The same suite under ThreadSanitizer.**

## Verifying a change that touches inference

**The unit tests never load a model.** They cannot catch a regression in the runtime or the
model-load path. The only check that exercises real inference is:

```bash
tools/golden-baseline.sh --check 4
```

It counts as a model run, so the process rules above apply first. Baselines are stored
in `baselines/`, which starts empty: the two that shipped with the fork were captured on
the original author's machine, and `d9b37b9` established that their own scope note rules
them out for this hardware. So `--check` has nothing to compare against until you capture
a baseline on the machine you intend to check.

A baseline is valid for one (machine, build, model) triple; re-capture only for a
deliberate numerics change, never to make a mismatch go away.

## Do not do these to get tests running

Do not download a full checkpoint, duplicate a `.gturbo`, create a worktree, or purge caches
just to run tests. If a test cannot run, report why.

## The Mac mini is a deploy target, not a checkout

There is no git on that box. Build release here and copy the binary over; never try to pull,
check out, or build there.

Reach it with `ssh macmini` — **never** the tailnet hostname, which is the HTTP endpoint
only and fails ssh with a misleading `Host key verification failed`. `sudo` there needs
`ssh -t`.

### The next deploy carries the rename, and all of it lands at once

The mini is still running a binary built before the rename to Shrike. That is fine —
a build artifact does not care what its source was called, and binary and service
config only couple at the next deploy. But at that deploy four things must change
together, or the service comes up subtly wrong:

1. **The executables.** `/Users/davor/nvmai-runtime/bin/` currently holds
   `NVMAIServer`, `NVMAICLI`, and `NVMAIRepack`. They become `ShrikeServer`,
   `ShrikeCLI`, `ShrikeRepack`.
2. **The deploy directory itself** — `~/nvmai-runtime/` → `~/shrike-runtime/`. The
   server's built-in models-directory default moved with the rename, so a server
   looking for `~/shrike-runtime/models` finds nothing if the directory still has its
   old name.
3. **Every `NVMAI_*` variable in the launchd plist** → `SHRIKE_*`. Also check for
   `TURBO_FIELDFARE_PHASES`, `TURBO_FIELDFARE_TOKENIZER_DIR` and `TURBO_FIELDFARE_MODEL`
   — inherited from the fork this project came from, and renamed to `SHRIKE_PHASES`,
   `SHRIKE_TOKENIZER_DIR` and `SHRIKE_MODEL`. They fail the same silent way.
4. **The plist's program path and label**, to match 1 and 2.

**The failure mode for 3 is the dangerous one: a stale `NVMAI_*` variable does not
error.** Nothing reads it, nothing complains, the built-in default is taken silently,
and carefully tuned configuration disappears into a performance regression noticed
days later with no obvious cause. Items 1 and 2 fail loudly; item 3 fails quietly.

## Where documents go

`docs/` pairs a design document with its implementation plan, named for the work rather than
dated — `v6-dialect-normalized-cache.md` alongside `v6-implementation-plan.md`. Follow that
shape for new work. The plan's own per-task checkboxes are the status of record; do not
duplicate status into other files.
