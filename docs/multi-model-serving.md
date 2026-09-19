# Multi-model serving — design

The record of its own chapter, kept as written. Six of the flags it names went in
v24's trim: `--model-id`, `--models-dir`, `--preload`, `--idle-unload-seconds`,
`--prompt-cache-mode` and `--prompt-cache-disk` left argv; the roster, the idle
machinery and the prompt cache they selected did not, and the cache's settled
values are `ModelSessionPlan` defaults now. Its open question about the disk cache
rehydrating across a swap is therefore still open and now costs an edit to ask.
See [v24-unified-cli.md](v24-unified-cli.md).

## Objective

One server process serves every installed model, and the `model` field of an ordinary
OpenAI request selects which. No client needs patching: pi, opencode, Cursor and curl
all switch models through the API they already speak.

The capability belongs in the server rather than in a client extension. A per-client
extension has to be written once per client and rewritten whenever the client changes;
the request field is the same everywhere and always has been.

## The defect

The server is single-model down to the type level.

- `ShrikeHTTPServer` holds `private let modelID: String` (`HTTPServer.swift:144`), fixed
  for the process lifetime.
- `GET /v1/models` advertises exactly two ids: `modelID` and `modelID + "-fast"`
  (`HTTPServer.swift:263-277`).
- `OpenAIRequestValidator.validate` guards `request.model == modelID || stripCLIPrompt`
  and throws `unknownModel` otherwise (`OpenAIModels.swift:334`). A wrong model id errors;
  it does not silently answer from the wrong weights.
- `main.swift` builds one `ModelSessionPlan`, one `previewFacts`, one backend, and passes
  `facts.modelID` into the server.

The residency machinery that a swap needs already exists and is **not in use**.
`ManagedModelBackend` implements lazy load, idle unload, in-flight draining, coalesced
concurrent first-loads, and a manual unload that waits for drain. But
`ServerArguments.managesResidency` is `lazyLoad || idleUnloadSeconds > 0`, and the
deployed invocation passes neither — so `main.swift` takes the `else` branch, builds a
session eagerly, and never constructs it. The code is dormant in production.

## Model identity

Three identifiers exist, doing three different jobs, and the current code conflates two
of them.

| | job | written by | example |
|---|---|---|---|
| `manifest.modelID` | provenance — where the weights came from | ShrikeRepack at install | `mlx-community/Kimi-Linear-48B-A3B-Instruct-4bit` |
| `arch.family` | dispatch — which runtime path executes them | ShrikeRepack, from architecture | `kimi_linear_48b` |
| API model id | what a client types | the server | `kimi-linear-48b-a3b` |

`manifest.json` is Shrike's own artifact, not the provider's — the installer writes it
beside the weights. `ShrikeRepack/Core/Remote/SupportedModelSource.swift` holds a table of
recognized sources keyed by the SHA256 of the source safetensors index
(`SourceFingerprint.modelID(forIndexSha256:)`), and records a curated name when the
fingerprint matches. That is why provenance is clean for some models and an HF repo path
for others:

| bundle | `manifest.modelID` | `arch.family` |
|---|---|---|
| `qwen36.gturbo` | `qwen3.6-35b-a3b-4bit` | `qwen36` |
| `ornith15.gturbo` | `ornith-1.5-35b-a3b-4bit` | `qwen36` |
| `kimi-linear-48b-a3b-4bit.gturbo` | `mlx-community/Kimi-Linear-48B-A3B-Instruct-4bit` | `kimi_linear_48b` |
| `gpt-oss-20b-mlx-4bit.gturbo` | `InferenceIllusionist/gpt-oss-20b-MLX-4bit` | `gpt_oss_20b` |

`ServerModelIdentity.apiModelID` tries to derive the API id from provenance: strip
`-4bit`/`-8bit`/`-6bit`, else pass the string through, else fall back to a per-family
table. It is not an identity source, and multi-model breaks it two ways.

- Pass-through yields `mlx-community/Kimi-Linear-48B-A3B-Instruct` for kimi and
  `InferenceIllusionist/gpt-oss-20b-MLX` for gpt-oss. The `--model-id` override in the
  deployed invocation is load-bearing, not decoration.
- The family fallback **collides**. `ornith15` and `qwen36` are both `arch.family ==
  qwen36`, so once both are loadable the table maps them to one id. It is sound today
  only because one model is served at a time.

## Design

### Resolution

At startup, in order:

1. Scan the model root for `*.gturbo` bundles.
2. `ManifestReader.peekIdentity` on each — provenance id and family. Reads `manifest.json`
   only: no weights mapped, no Metal device, no measurable cost.
3. Drop any bundle whose family is not servable. With MTP withdrawn that is `qwen36_mtp`,
   which excludes `qwen36-mtp.gturbo` and `ornith15-mtp.gturbo` by rule rather than by
   name. Six bundles resolve to four models.
4. Assign the API id: the config entry's `id` if the bundle has one, else the bundle name
   minus `.gturbo`.
5. Fail startup on a duplicate id. Log the resolved roster, so a config typo surfaces at
   launch rather than as a 404 later.

A request naming either the configured id or the bundle name resolves to the same model.
Both forms share one namespace, which is the only way a duplicate can arise given the
filesystem already guarantees unique bundle names within a directory.

### Configuration

```json
{
  "models_dir": "~/shrike-runtime/models",
  "defaults": { "max_context": 32768, "ram_budget": "6G", "idle_unload_seconds": 0 },
  "models": [
    { "dir": "kimi-linear-48b-a3b-4bit.gturbo", "id": "kimi-linear-48b-a3b" },
    { "dir": "gpt-oss-20b-mlx-4bit.gturbo",     "id": "gpt-oss-20b", "default": true },
    { "dir": "qwen36.gturbo",                   "id": "qwen3.6-35b-a3b" },
    { "dir": "ornith15.gturbo",                 "id": "ornith-1.5-35b-a3b" }
  ]
}
```

`models` is a list of **overrides, not the roster**. A bundle with no entry is still
served, under its bundle name. Config is written only for models whose name needs fixing —
today, the two carrying HF repo paths.

`default: true` marks the model served when a request omits `model` entirely. With no such
entry and more than one model in the roster, an omitted `model` is an error naming the
valid ids — the same error an unknown id produces.

Default location `~/.shrike/server.json`, selected with `--config`. Precedence is
flag > config > built-in default, resolved in memory. **A flag never writes back into the
config file**: a launch argument that becomes a permanent setting defeats the purpose of
being an argument, and makes a value fixed live revert on the next restart.

`--model <dir>` with optional `--model-id` keeps working and means "serve exactly this
one, ignore any roster", so the deployed invocation runs unchanged. Passing both `--model`
and `--config` is a startup error rather than a silent precedence puzzle.

### Residency

One model resident at a time. The target is a 16 GiB M1 mini; every installed bundle is
larger than the machine's RAM on its own, and residency is bounded by `--ram-budget` and
`--expert-cache-slots` rather than by bundle size.

`ManagedModelBackend` generalizes into a registry over `[String: ModelSessionPlan]` with a
`residentID`. Its existing body survives: `loadTask` still coalesces concurrent first
requests, `inFlight` still blocks eviction, `unloadWaiters` still drains before release,
the reaper still wakes once on the deadline. A swap is `unload()` then `load(otherPlan)`,
and the ordering is strict: the load does not begin until `unload()` has returned. The
session is the sole owner of every large allocation — the weights' `MTLBuffer`s carry
`munmap` deallocators and the expert slots carry `free` deallocators, per
`ManagedModelBackend`'s header — so dropping it returns the memory before the next model
maps, and peak-of-swap is one resident model, not two.

`MetalContext` is built once and reused across swaps through the `reusingContext`
parameter `makeSession` already takes. This is what the in-process design buys over
restarting the process per switch: no shader-library recompile on every model change.

Validation happens before residency. Every model's facts derive from its manifest at
startup, so the registry exposes a `nonisolated` id → `ModelSessionFacts` lookup: resolve
the id, validate against that model's bounds, then acquire. No actor hop and no load, and
it preserves the current property that facts are correct before anything is resident.

Requests **batch by model**: when the slot frees, every pending request for the resident
model is served before a swap. Under alternating load that collapses into one swap per
batch rather than one per request. Measured at deploy (2026-08-28, the mini, 32k context,
6G budget): a swapping request completes in ~7–9 s against ~3 s resident, so a swap costs
roughly 4–6 s and batching is a convenience, not load-bearing. Across four swaps under a
`memory_pressure` watch, free memory never dropped below 34% and the residency log shows
every unload completing before the next load — the strict ordering holds live.

Idle unload stays as `--idle-unload-seconds`, default 0. Its purpose is coexistence — the
server otherwise holds roughly 5 GB of dense weights plus the expert budget for the
process lifetime, and that box also runs other inference servers. At 0 the behaviour
matches what is deployed today.

### HTTP surface

| route | change |
|---|---|
| `GET /v1/models` | lists every configured model. The `-fast` entry is gone. Payload stays `id`/`object`/`created`/`owned_by` — clients read `id` and nothing else |
| `GET /health` | gains `resident` (id or null), `loading` (bool), `models` (count) |
| `POST /v1/chat/completions`, `/v1/responses` | `model` selects. Resolve → validate against that model's bounds → acquire |
| `POST /v1/models/unload` | unchanged, bodyless. One slot makes it unambiguous; reports which model was released, or none |
| `POST /v1/models/load` | new. `{"model": "<id>"}`, loads without generating |

`load` takes a body rather than oMLX's `/v1/models/{id}/load` path form because the router
is a flat `switch (method, path)` over literal strings with no path-parameter support, and
one route does not justify adding prefix matching.

`load` answers after the cheap checks — the id resolves and the roster has it — and before
the weights load. The response confirms the request was accepted, not that the load
succeeded; a failed load surfaces on the first generate, exactly as a lazy load failure
does today. A `load` naming another model while one is loading does not block the caller;
it queues behind through the same slot serialization every request uses.

The `-fast` alias is removed from the HTTP surface; nothing deployed names it. What a
prompt contains is the client's decision; a server that silently rewrites it is answering a question nobody asked.
`fastModelID` and the `stripCLIPrompt` branch leave `validate`, and
`ValidatedChatRequest.stripCLIPrompt` goes with them — it would be permanently false.
`CLIStrip.isEnabled()` / `SHRIKE_STRIP_CLI_PROMPT` remains as an operator lever, which is
not something a client can reach.

## What this deletes

- `ServerModelIdentity.apiModelID` — string surgery standing in for identity.
- `fastModelID` and `stripCLIPrompt` from `OpenAIRequestValidator.validate`, and
  `stripCLIPrompt` from `ValidatedChatRequest`.
- The `-fast` entry in `GET /v1/models`.
- The eager branch in `main.swift`. Both paths become one managed path, and "eager"
  demotes to a `--preload` flag that changes *when* the first load happens rather than
  which code runs it. The residency code stops being dormant.
- `ManagedModelBackend` itself — its body survives inside `ModelRegistry`, so the type,
  the `ResidencyManaging` protocol, and the HTTP layer's runtime cross-cast to it all
  go, along with `ServerModelSession.defaultModelID` (identity lives in the roster).

## What this needs

- **New `ModelRegistry.swift`** — the actor: plans by id, `residentID`,
  `acquire(modelID:)`, the batching queue, one reaper, one `MetalContext`. `lint.sh`
  rejects any new function over 120 lines, so the swap path wants decomposing as it is
  written rather than after.
- **`ServerArguments.swift`** — `--config`, `--models-dir`, `--preload`, and a
  `ServerConfig` Codable. Bulk, but mechanical.
- **`HTTPServer.swift`** — `modelID` becomes per-request. Roughly fifteen call sites,
  nearly all echoing the id into a response envelope.
- **`main.swift`** — branch collapse and registry construction.
- **Tests.** `ManagedModelBackend.Loader` is an injectable typealias, documented as
  existing so residency can be tested against a stub with no model on disk. Swap ordering,
  drain-before-evict, load coalescing, batching and the reaper are all reachable without
  weights. Add: id resolution (config id, bundle-name fallback, duplicate fails startup,
  unknown id errors and lists valid ids), roster exclusion of `qwen36_mtp`, `/v1/models`
  contents, `-fast` rejected.
- **Live verification on the mini.** The unit tests never load a model, and this changes
  the model-load path. The committed golden baseline was captured on foreign hardware
  (v6 plan, *Global constraints*), so the check is a pre/post A/B generation on the same
  machine and model (temperature 0, fixed seed), plus a `memory_pressure` watch across
  one forced swap — the strict-ordering claim in Residency is checkable there and
  nowhere else.

`ServerPromptCache` needs nothing. `ServerPromptCacheDomain` already keys every entry by
`modelID` alongside the runtime profile, template hash and KV storage, so cross-model KV
restore is impossible by construction.

## Cost

Roughly a day for registry, routing and the `main.swift` collapse; a second for config,
arguments and tests. The residency machinery already existing is what makes it that rather
than a week. This is a shape, not a commitment: the server module has been read, but
`ServerInference` internals and the existing test suite have not.

## Not established

- **Whether the disk prompt cache rehydrates across a swap.** The `--idle-unload-seconds`
  help text states "Pair with `--prompt-cache-disk`, since unloading discards the
  in-memory prefix cache," and entries are domain-keyed by `modelID`, so a swap back
  should find its own entries. The load-side rehydrate path has not been traced. Verify
  during implementation; do not claim the benefit until then.
- **Whether two models could ever be co-resident on 16 GiB.** Two sets of dense weights
  plus two expert caches is not obviously impossible, and is not measured. Out of scope
  either way — the design assumes one slot.

## Stated limitations

- **One slot means alternating clients evict each other.** Batching bounds it to one swap
  per batch, not to zero. Accepted: the box has a single user, and `/health` makes the
  state visible when it is not.
- **Batching admits starvation.** A continuous stream for one model can hold the slot
  indefinitely. A max-wait bound would fix it and is deliberately not built; add it if
  starvation is ever observed rather than in anticipation.
- **`POST /v1/models/load` is unauthenticated**, like every other route — the server has
  no authentication or TLS. It adds a way for anyone who can reach the port to evict the
  resident model. That is a real widening of what a caller can do, bounded by the existing
  network posture rather than by anything in this change.
- **MTP is excluded, not removed.** Withdrawing it from `ServerModelSession.load`, the
  `qwen36_mtp` family, `RepackPlanner` and ShrikeBench is a separate change with its own
  blast radius. Here the family is simply not servable, so the two `-mtp` bundles never
  enter the roster and config never sets `mtpModelDirectory`.
- **Per-model overrides beyond `id` are not built.** The manifest already supplies what
  varies per model — `prefillChunkTokens` resolves from family inside `previewFacts`, and
  prompt cache mode resolves through `effectivePromptCacheMode`. The `defaults` block
  exists so a per-model override slot can be added later without reshaping the file.

## Two things not to misread

**Removing `-fast` is not a judgement on id-suffix variants.** oMLX exposes profiles as
`<model>:<profile>` ids and that mechanism is fine. What is being removed is the server
deciding what a client's prompt should contain. If a variant-id feature is ever wanted,
the pattern has prior art and nothing here forecloses it.

**`POST /v1/models/load` buys convenience, not portability.** No client in the wild calls
a load endpoint. LM Studio and llama-swap both load just-in-time from the `model` field;
Ollama's preload is a generate call with no prompt; llama-swap ships unload routes and no
load route, which is the asymmetry this server already had. The route exists for local
tooling and an eventual pi `model_select` handler. The `model` field is the load trigger,
and that is what makes every unmodified client work.
