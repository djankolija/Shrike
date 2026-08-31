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
5. **The same suite under ThreadSanitizer**, run as
   `env TSAN_OPTIONS=suppressions=tsan-suppressions.txt swift test --no-parallel --sanitize=thread`
   from the repo root. The suppressions file silences only the known
   false-positive family from swift-nio's `EventLoopFuture.get()` continuation
   bridge (the happens-before edge lives in uninstrumented
   `libswift_Concurrency`); the file's header carries the analysis. A report
   that does not match that shape is real — fix it, never widen the file.

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

### The mini's layout (rename landed 2026-08-30)

The runtime is `~/shrike-runtime/` — `bin/` holds `ShrikeServer`, `ShrikeCLI`,
`ShrikeRepack` plus their resource bundles (a deploy copies the `*.bundle`
directories from `.build/release/` alongside the binaries, or resource lookups
fail at runtime); `models/` holds the six `.gturbo`s, receipts bound to the
`shrike-runtime` path. Old NVMAI artifacts are parked in `~/nvmai-retired/`,
kept only as rollback insurance.

There is **no launchd service** — the server is launched manually
(`cd ~/shrike-runtime && nohup ./bin/ShrikeServer … > /tmp/shrike-server.log 2>&1 &`),
usually serving one model on port 8081. Turbo (a separate project) serves on
8080; never touch it.

Configuration is `SHRIKE_*` env vars only. A resurrected old command or script
carrying `NVMAI_*` or `TURBO_FIELDFARE_*` names fails **silently** — nothing
reads those vars, the built-in defaults are taken, and tuned configuration
quietly vanishes. If a launch config ever graduates to a launchd plist, audit
every env var name against the current `SHRIKE_*` set first.

**The perf env set is load-bearing (2026-08-30, extended 2026-08-31): launching
ShrikeServer without `SHRIKE_EXPERT_IO_SYNC=event SHRIKE_EXPERT_IO_SUBMISSION=immediate
SHRIKE_EXPERT_CACHE_LAYOUT=pool SHRIKE_DECODE_EXPERT_EXECUTION=speculative
SHRIKE_HOST_WAIT=spin` silently costs large decode throughput** — the first four
roughly halve it; dropping `SHRIKE_HOST_WAIT=spin` costs a further ~11 % (rig) to
~16 % (real-shaped traffic) by paying a scheduler wake on every per-layer router
wait. These modes are deliberately not code defaults yet (pool needs an allocation
fallback; speculative needs an extended real-traffic trial beyond the fixed rig
prompt; spin busies one P-core during decode and its multi-hour thermal behavior
is unproven — see `docs/v9-implementation-plan.md`). Output is byte-identical
with or without them; only speed changes. Add
`SHRIKE_RUNNER_STATS=1 SHRIKE_KERNEL_STATS=1` when measuring with
`tools/decode-measure.sh` and the `tools/parse-*-stats.py` parsers.

## Where documents go

`docs/` pairs a design document with its implementation plan, named for the work rather than
dated — `v6-dialect-normalized-cache.md` alongside `v6-implementation-plan.md`. Follow that
shape for new work. The plan's own per-task checkboxes are the status of record; do not
duplicate status into other files.
