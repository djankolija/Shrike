# Multi-model serving — implementation plan

> **For agentic workers:** execute task-by-task in order — the sequencing is load-bearing
> (see *Sequencing rationale*). Each task ends buildable (`swift build`) with its named
> tests passing (`swift test --no-parallel --filter <suite>`) and committed. One checkbox
> per task tracks completion.

**Goal:** one server process serves every installed bundle; the `model` field of an
ordinary OpenAI request selects which; one model resident at a time.

**Spec:** `docs/multi-model-serving.md` — normative for every mechanism named here; this
plan only sequences it. Baseline: the `feat/multi-model-serving` branch off `8f12be1`.

## Global constraints

- `swift test --no-parallel`; unit tests never load a real model.
- `tools/lint.sh` gates: no new function over 120 lines — decompose the swap path as it
  is written; `lint:allow-force` / `unchecked-invariant:` comments where applicable.
  Release build must be warning-free. Markdown links must resolve.
- All new code goes in `sources/NVMAIServer/Core` (`NVMAIServerCore`) so it is testable;
  `Command/main.swift` stays a thin assembly.
- Swift Testing (`@Suite`/`@Test`/`#expect`); suites that bind ports are `.serialized`.
- The registry preserves `ManagedModelBackend`'s invariants verbatim: load coalescing,
  in-flight drain before release, strict unload-before-load swap ordering, one
  process-wide `MetalContext`, reaper wakes once per deadline.
- Deploy (Task 9 only): build release on the MacBook, copy the binary to the mini's
  `nvmai-runtime/bin`, previous binary preserved beside it. The mini is shared — check
  nothing else is mid-run before starting anything that loads a model.

## Sequencing rationale

The registry lands beside `ManagedModelBackend`, not in place of it — every task ends
buildable, and the old type is deleted only in the task that rewires its last caller
(`main.swift`). The HTTP rewire (Task 4) is the widest diff, so everything it depends on
(roster, registry, coordinator affinity) exists and is tested first. `-fast` removal
(Task 5) follows the rewire because the validator change and the roster change must not
straddle a task boundary with one accepting what the other no longer advertises.

## Tasks

- [x] **Task 1 — config + roster resolution.** New `ServerConfig` (Codable: `models_dir`,
  `defaults{max_context, ram_budget, idle_unload_seconds}`, `models[{dir, id, default}]`;
  loaded from a `--config` path, `~` expanded) and `ModelRoster` (pure resolution over
  scan candidates `(bundleName, manifestModelID, family)`: canonical id = config `id`
  else bundle name minus `.gturbo`; bundle name kept as an accepted alias in the same
  namespace; non-servable families — `qwen36_mtp` — dropped by rule; a duplicate id or a
  second `default: true` fails with both claimants named). The disk scan
  (`ManifestReader.peekIdentity` per `*.gturbo`) stays separate from resolution so
  resolution tests need no disk. Tests: `ServerConfigTests`, `ModelRosterTests`.

- [x] **Task 2 — ModelRegistry.** New actor over the roster: immutable entries map
  (plan + facts + max context per id, `nonisolated` lookup), `acquire`/`release` with
  per-model load coalescing and a state loop safe under actor reentrancy, swap =
  drain → unload returns → load, `unload()` reporting the released id, `startLoad(_:)`
  (returns after resolution; the load runs detached through the same slot serialization;
  a failure surfaces on the next acquire), health snapshot (resident / loading / model
  count), idle reaper, one `MetalContext` threaded through `reusingContext`, injectable
  `Loader` (typealias unchanged). `ManagedModelBackend` stays untouched this task.
  Tests: `ModelRegistryTests` — port the ManagedModelBackend cases, add swap ordering,
  cross-model coalescing, unknown id, unload-reports-id, startLoad failure deferred to
  next acquire.

- [x] **Task 3 — coordinator model affinity.** `ServerCoordinator.run` takes the resolved
  model id; on release the next admit prefers the first waiter matching the
  last-admitted model, else the FIFO head. Queue-limit shedding unchanged. Tests:
  `ServerCoordinatorTests` — affinity batching, FIFO fallback when no waiter matches,
  shedding unchanged.

- [x] **Task 4 — HTTP rewire.** `NVMAIHTTPServer` holds the registry (no `modelID`, no
  `backend`); handlers resolve `request.model` against the roster before validation
  (unknown id errors naming the valid ids; omitted `model` resolves to the default entry
  or produces the same error), validate against that entry's bounds, and echo the
  canonical id at the seven envelope sites. `GET /v1/models` lists the roster;
  `GET /health` gains `resident`/`loading`/`models`; `POST /v1/models/unload` reports
  the released id; new `POST /v1/models/load`. Existing `HTTPServerTests` scenarios wrap
  their stub backends in one-entry registries (the `unloadEndpointReleasesTheModel`
  pattern). Tests: `HTTPServerTests` + load endpoint, health fields, unknown-id message,
  omitted-model behaviour, a two-model swap over HTTP with stub loaders.

- [x] **Task 5 — `-fast` removal.** `fastModelID` and `stripCLIPrompt` leave
  `OpenAIRequestValidator.validate` and `ValidatedChatRequest`; the `ServerInference`
  strip branch keys on `CLIStrip.isEnabled()` alone; `tools/server_launcher.sh` stops
  advertising the `-fast` id. Tests: `OpenAIValidationTests` (`-fast` now rejected as
  unknown), `ServerPromptCacheTests` adapted; `CLIStripTests` untouched — the operator
  env lever remains.

- [x] **Task 6 — arguments** *(landed in one commit with Task 7 — making `--model`
  optional spans both)*. `--config`, `--models-dir`, `--preload`; `--model` together
  with `--config` or `--models-dir` is a startup error; config defaults merge under
  flag > config > built-in, in memory, revalidating what they override; usage text
  updated. Tests: the arguments suite extended.

- [x] **Task 7 — main collapse.** One managed path: config load → bundle scan → roster →
  registry → server. `--model` builds a one-entry roster (`--model-id` as its canonical
  id); `--preload` fires `startLoad` on the default entry; the resolved roster is logged
  at startup. Delete: the eager branch, `ManagedModelBackend` and its test file (cases
  already ported), `ResidencyManaging` and the `handleUnload` cross-cast,
  `ServerModelIdentity` and `ModelIdentityTests` (dead once `previewFacts` takes the
  roster id as its override). Tests: the full `NVMAIServerTests` target green.

- [x] **Task 8 — gates.** `swift build -c release` warning-free, `tools/lint.sh`,
  `swift test --no-parallel` (1062 tests), the same suite under `--sanitize=thread`,
  markdown links resolve. Reconciled the spec's *What this deletes* against what was
  actually deleted. One TSan data-race report (`NVMAIHTTPServer.shutdown()` against an
  `EventLoopFuture.get()` continuation resume) appeared once in the first full TSan run
  and never again — four isolated re-runs and a full re-run all clean. Read as the
  continuation/task-allocator false-positive family; nothing was suppressed, so a
  recurrence will still fail CI.

- [x] **Task 9 — deploy + live verification** *(run 2026-08-28; the previous binary was
  deleted rather than preserved, per decision at deploy time)*. Pre/post A/B on the mini
  (ornith15, identical `--model` flags, temperature 0, seed 1234, 96 tokens, cold cache
  both sides): **byte-identical content, identical usage** — numerics and the load path
  unchanged. Config mode: 4 models resolved, both `-mtp` bundles excluded with logged
  notices; per-model generation, canonical-id echo, omitted-model default, unknown-id
  404 naming the valid ids, unload reporting the released id, and the load endpoint's
  accept-then-preload all verified over plain HTTP. Swap latency ~4–6 s (7–9 s swapping
  vs ~3 s resident) — batching is a convenience. `memory_pressure` never dropped below
  34% free across four swaps and the residency log shows strict unload-before-load
  throughout; oMLX on the same box stayed healthy. The mini was left as found (no
  NVMAIServer running); its `~/.nvmai/server.json` is written, so
  `nvmai-runtime/bin/NVMAIServer --port <p>` starts the full roster.
