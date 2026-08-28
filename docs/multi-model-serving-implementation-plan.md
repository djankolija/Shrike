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

- [ ] **Task 6 — arguments.** `--config`, `--models-dir`, `--preload`; `--model` together
  with `--config` or `--models-dir` is a startup error; config defaults merge under
  flag > config > built-in, in memory, revalidating what they override; usage text
  updated. Tests: the arguments suite extended.

- [ ] **Task 7 — main collapse.** One managed path: config load → bundle scan → roster →
  registry → server. `--model` builds a one-entry roster (`--model-id` as its canonical
  id); `--preload` fires `startLoad` on the default entry; the resolved roster is logged
  at startup. Delete: the eager branch, `ManagedModelBackend` and its test file (cases
  already ported), `ResidencyManaging` and the `handleUnload` cross-cast,
  `ServerModelIdentity` and `ModelIdentityTests` (dead once `previewFacts` takes the
  roster id as its override). Tests: the full `NVMAIServerTests` target green.

- [ ] **Task 8 — gates.** `swift build -c release` warning-free, `tools/lint.sh`,
  `swift test --no-parallel`, the same suite under `--sanitize=thread`, markdown links
  resolve. Reconcile the spec's *What this deletes* against what was actually deleted.

- [ ] **Task 9 — deploy + live verification (needs explicit go-ahead; the mini is
  shared).** Binary to `nvmai-runtime/bin` (previous preserved); write the mini's
  `~/.nvmai/server.json` for its installed bundles; start; `GET /v1/models`; one
  generation per model through an unmodified client; one forced swap watched with
  `memory_pressure`; pre/post A/B generation on the same machine and model
  (temperature 0, fixed seed) — the committed golden baseline is foreign hardware
  (v6 plan, *Global constraints*). Record swap latency once; it settles the batching
  question in the spec's *Not established*.
