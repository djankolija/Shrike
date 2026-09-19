# Shrike — working instructions

This is a fork where active development happens. It was forked from `Pummelchen/NVMAI`,
which keeps that name — the rename to Shrike is ours alone. Upstream's posture — run and
report existing behaviour, don't edit source — **does not apply here**. Normal development
is expected.

What the project is, which models it supports and how to use it belong in
[README.md](README.md) and `docs/`. This file is only what an agent has to do differently.

## Never run two model processes

Before anything that loads a model — a server, the CLI, a benchmark, or the golden
baseline — check:

```bash
pgrep -fl 'shrike serve|shrike generate|ShrikePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'
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
swift run -c release shrike repack verify-install --input-gturbo <model.gturbo>
```

**Never hand-edit the receipt to match a new path.** The path binding is what detects a
moved or swapped directory; editing it forges the attestation instead of re-establishing it.

## Gates that must pass before calling work done

Five local gates, four per code commit and the fifth once per chapter at its close.
Nothing runs them for you — there is no CI — so run them yourself
before calling work done. All of them constrain how code gets written here:

1. **Release build with zero warnings.** A new warning fails the build.
2. **`swiftlint lint --strict`**: three rules, `force_cast`, `force_try`, and
   `function_body_length` (warn 120, error 400). A force cast or force try needs
   `// swiftlint:disable:next force_cast` (or `force_try`) on the line above it, with
   the reason stated in a comment. No function body is over 120 lines (v17 Task 4
   emptied the baseline and retired the file, 2026-09-08), so there is no baseline to
   pass and any new one fails the gate. Decompose as you write: a sequence of named
   stage methods over a small context struct, in the order the work runs.
3. **Markdown link check** — globs every `*.md` in the repo, so it binds on any document
   you add. Relative links must resolve.
4. **`swift test --no-parallel`** — serial, always. Pass `--filter` through as needed.
5. **The same suite under ThreadSanitizer — once per chapter, at its close before
   the merge to main, not per commit** (owner's ruling, 2026-09-04: the per-commit
   run cost ≈ 50 minutes, most of it the GPU-kernel reference suites that have no
   threads to check, and it held the build lock; a report found at close is fixed
   then). Run as
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
tools/golden-baseline.sh --check
```

It counts as a model run, so the process rules above apply first. Baselines are stored
in `baselines/`, tagged per machine (`short` and `long` ≈2k-token profiles); `--check`
compares only against files whose machine tag matches the box it runs on — capture on
the machine you intend to check. On the mini, run the script with its five env
overrides (header comment) since that box has no checkout. The fifth,
`CLI_EXTRA_ARGS="--expert-cache-slots 160"`, is what makes the run exercise the
two-chunk arena production serves at; without it the golden runs the CLI's default
64 slots, one chunk, and a regression at the chunk boundary passes unseen.

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

The runtime is `~/shrike-runtime/` — `bin/` holds the single `shrike` binary
plus its resource bundles (a deploy copies the `*.bundle`
directories from `.build/release/` alongside the binaries, or resource lookups
fail at runtime); `models/` holds the six `.gturbo`s, receipts bound to the
`shrike-runtime` path; `baselines/` holds the mini's golden-baseline files.
The box carries current state only — no staged rollback binaries, no retired
artifacts; git history and a fresh deploy are the rollback path (owner's
ruling, 2026-09-01).

There is **no launchd service** — the server is launched manually
(`cd ~/shrike-runtime && nohup ./bin/shrike serve … > /tmp/shrike-server.log 2>&1 &`),
usually serving one model on port 8081. Turbo (a separate project) serves on
8080; never touch it.

**Since v20 Task 1 (2026-09-17) the production launch carries two variables**, the
pool's per-layer slot allocation and its eviction policy (measured +4.4 to +7.8 % tok/s
on the four shapes, the record in `docs/v20-ssd-mechanism.md`), and **since v22
Task 3 (2026-09-18) the budget is 160 slots per layer** (the arena in two Metal
buffers, the prefill scratch released between requests, oMLX's models unloaded;
measured +9.9 to +15.6 % tok/s and 40 to 52 % fewer misses on the four shapes, the
record in `docs/v22-pool-capacity.md`):

```bash
SHRIKE_EXPERT_SLOT_TABLE=256,256,246,209,191,171,171,162,171,149,164,169,155,144,137,135,133,131,132,130,145,133,141,142,137,137,130,137,137,142,137,135,157,157,162,160,166,161,178,194 \
SHRIKE_EXPERT_POLICY=slru \
nohup ./bin/shrike serve --model ./models/ornith15.gturbo --model-id ornith15 --port 8081 --max-context 32768 --ram-budget 11324620800 --thinking off > /tmp/shrike-server.log 2>&1 &
```

The table is ornith15's (blend 0.3 of its production miss profile scaled to the
budget's 6,400, no layer above its 256 experts; the v20 table of 5,120 is
`~/.claude/handoffs/archive/shrike-v22-t3/v22-arms.sh`'s `base` arm); the budget is
given in bytes because `--ram-budget` snaps to the nearest allowed slot count (8, 16,
24, 32, 64, 96, 128, 160, 192, 224, 256 per layer) and 8G snapped to 128. A bare launch
without the variables runs the uniform 128 and aging-LFU. A table that does not match
the model's layers or the budget's total is refused at launch, loudly. The server's
load line names what it runs (`expert_slots=130..256 policy=slru:0.5`). **oMLX (port
8000, a LaunchDaemon `local.omlx`) must not hold models while Shrike serves at this
budget**: its reranker and embedder took 2.9 GB resident plus swap; a `kill` of its
`omlx-server` process respawns it empty.

Configuration is `SHRIKE_*` env vars only. A resurrected old command or script
carrying `NVMAI_*` or `TURBO_FIELDFARE_*` names fails **silently** — nothing
reads those vars, the built-in defaults are taken, and tuned configuration
quietly vanishes. If a launch config ever graduates to a launchd plist, audit
every env var name against the current `SHRIKE_*` set first.

**The measured perf winners are the only paths since v17 (2026-09-07)**: a bare
launch runs event IO sync, immediate submission, the pool cache layout, speculative
execution, the spin host wait and the word wake, and the A/B knobs that once
selected their losers (`SHRIKE_DECODE_EXPERT_EXECUTION`, `SHRIKE_EXPERT_IO_SYNC`,
`SHRIKE_EXPERT_IO_SUBMISSION`, `SHRIKE_SPEC_PHASE1`, `SHRIKE_ROUTER_WAKE`,
`SHRIKE_HOST_WAIT`) are gone with the losing code; git history is their record
(`docs/v17-consolidation.md`). Add
`SHRIKE_RUNNER_STATS=1 SHRIKE_KERNEL_STATS=1` when measuring with
`tools/decode-measure.sh` and the `tools/parse-*-stats.py` parsers.
**The mini's budget is 160 slots per layer since v22** (11.33 GB of cells; the
arena in two Metal buffers since the device caps one at 8.88 GiB; the box at 82
to 85 % free under load with oMLX empty and the prefill scratch released between
requests). `--ram-budget 8G`, 128 slots, was the optimum only while the arena
was one buffer. If the pool's allocation ever fails at startup, the error is
loud; the pool is the only layout since v17 (per-slot was v9's measured loss).

## Where documents go

`docs/` pairs a design document with its implementation plan, named for the work rather than
dated — `v6-dialect-normalized-cache.md` alongside `v6-implementation-plan.md`. Follow that
shape for new work. The plan's own per-task checkboxes are the status of record; do not
duplicate status into other files.
