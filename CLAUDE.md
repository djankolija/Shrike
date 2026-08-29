# NVMAI — working instructions

This is a fork where active development happens. `origin` is `djankolija/NVMAI`; `upstream`
is `Pummelchen/NVMAI`. Upstream's posture — run and report existing behaviour, don't edit
source — **does not apply here**. Normal development is expected.

What the project is, which models it supports and how to use it belong in
[README.md](README.md) and `docs/`. This file is only what an agent has to do differently.

## Never run two model processes

Before anything that loads a model — a server, the app, the CLI, a benchmark, or the golden
baseline — check:

```bash
pgrep -fl 'NVMAIServer|NVMAIMac|NVMAIDecodeService|NVMAICLI|NVMAIPackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'
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
swift run -c release NVMAIRepack --verify-install --input-gturbo <model.gturbo>
```

**Never hand-edit the receipt to match a new path.** The path binding is what detects a
moved or swapped directory; editing it forges the attestation instead of re-establishing it.

## Gates that must pass before calling work done

CI runs five things. All of them constrain how code gets written here:

1. **Release build with zero warnings.** A new warning fails the build.
2. **`tools/lint.sh`** — two rules: no `as!` or `try!` under `sources/` without a
   `lint:allow-force <reason>` comment directly above it, and no *new* function over 120
   lines. Existing long ones are exempted in `tools/func-length-baseline.txt`; drop a row
   when its function shrinks, or the gate fails on the stale exemption. Decompose as you
   write rather than discovering this at CI.
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

It is not in CI, so it will not run unless you run it. It counts as a model run, so the
process rules above apply first. A baseline is valid for one (machine, build, model) triple;
re-capture only for a deliberate numerics change, never to make a mismatch go away.

## Do not do these to get tests running

Do not download a full checkpoint, duplicate a `.gturbo`, create a worktree, or purge caches
just to run tests. If a test cannot run, report why.

## The Mac mini is a deploy target, not a checkout

There is no git on that box. Build release here and copy the binary over; never try to pull,
check out, or build there.

Reach it with `ssh macmini` — **never** the tailnet hostname, which is the HTTP endpoint
only and fails ssh with a misleading `Host key verification failed`. `sudo` there needs
`ssh -t`.

## AGENTS.md is a benchmark fixture — needs deleting

`benchmark/coder_cli_benchmark.py` and `benchmark/nvmai_hit_fixup_ab.py` both read
`AGENTS.md` at runtime as their long-prompt input, so its **byte count is load-bearing**.
Editing it silently changes the measured prompt and breaks comparison with past runs —
`docs/v4.6-optimization-inventory.md` records that happening once already. Deleting it
breaks both scripts outright.

Its content is stale and its scope note is wrong for this fork; that is known, and
relocating it to a frozen fixture is deferred work. Until then, leave it alone.

## Where documents go

`docs/` pairs a design document with its implementation plan, named for the work rather than
dated — `v6-dialect-normalized-cache.md` alongside `v6-implementation-plan.md`. Follow that
shape for new work. The plan's own per-task checkboxes are the status of record; do not
duplicate status into other files.
