# v24 one shrike, one verb tree: implementation plan

**Goal:** five executables replaced by one `shrike` carrying a verb tree, the
flag surface trimmed against evidence, and the model resolved from configuration
instead of named on every invocation.

**Spec:** [v24-unified-cli.md](v24-unified-cli.md).

Five commits. The checkboxes here are the status of record.

## Constraints

- The four gates before any task is called done: `swift build -c release` (zero
  warnings), `swiftlint lint --strict`, `python3 tools/check-md-links.py`,
  `swift test --no-parallel`. ThreadSanitizer once at the close.
- No new `SHRIKE_*` variable. Fifteen distinct names exist in `sources/` today.
- Never `git add docs/` wholesale.
- Commit subjects in the repo's style, ending `(v24 Tn)`. No `Co-Authored-By`.
- Comments only for a non-obvious why. Not for narrating the migration.
- `git rebase -i` is unavailable in this harness, so review findings cannot be
  folded into their owning commits. They land in one labelled fold commit at the
  close, naming the owning commit per finding, as v23 recorded as a deviation.
- Searching for consumers: grep the binary **name**, never a literal command
  string. v23's `RepackCLITests` builds argv as a Swift array and a search for
  `"ShrikeRepack "` missed it entirely.
- `pgrep` before anything that loads a model. Never terminate a process this
  session did not start.
- A claim about what a binary does costs one run of the binary. Four claims in
  v23's spec were false when written, and each died on one command.

---

### Task 1: the flag inventory and its classification

No code. Build the evidence the trim runs on, so that Task 2 does not wire flags
onto the root that Task 4 then deletes.

- [x] For all seventy flag slots, record: the flag, the command that declares it,
      every in-repo consumer that names it, and the code path it selects.
      **70 slots, 52 distinct**, 14 flags declared by more than one command.
      Ambiguous consumer hits are opened, not counted: three of four checked were
      the consuming script's own `parser.add_argument`, not an invocation.
- [x] Classify each against the test in the spec. **Result: 18 delete, 34 keep,
      7 of the keeps behind a debug-only `bench` verb.**
- [ ] For every deletion candidate, name the code path that goes with it. v17's
      rule: the honest removal deletes a flag together with the path it selected.
      A flag with no path to delete is a suspicious deletion, not a free one.
      **Partial**: named for `--lazy-load` (the `preload` conflict branch),
      `--prompt-cache-disk` (plumbed through five files into the runtime's
      snapshot store, and whether that store is shared with the in-memory cache
      is still open), `--concise`, `--rope-scaling` and `--resume`. The other
      thirteen are named in Task 4, before anything is cut.
- [x] Settle `--tokenize`: **stays a flag on the bare form.** Two tools read it,
      `expert-pool-replay.py` and `q3-drafter-routes.py`, so it survives rule 1.
      Graduating it to its own verb is recorded as out of scope.
- [x] **Sibling drift is part of the classification**, not only unnamed flags. A
      flag declared by more than one command is checked for one contract. One
      instance is already found: `--max-context` is enforced against
      `RuntimeConfiguration.supportedContextTokens` by the server
      (`ShrikeServerCommand.swift:272`, help and error rendered from the same
      constant, which was v23's fix) and against a **range** by generate,
      `1...nativeMaximumContextTokens` in `validateContext()`
      (`ShrikeCLICommand.swift:227`). Both validate; they disagree on the
      contract. The defaults differ too, 4096 against 262144. Under one root
      these become siblings in one help tree, so the drift is resolved rather
      than inherited.
- [x] Write the table into `v24-unified-cli.md` as an inventory section, with the
      resulting count stated as an output of the classification.
- [ ] Link check, commit. Text only.

The count is not a target. The aim discussed at the outset was roughly a dozen;
the evidence supports 27 in the shipped surface, and going below that would mean
deleting working behaviour rather than removing cost. The owner's reservation
that several survivors still are not worth their cost is recorded in the spec's
Out of scope, so a later chapter can reopen it from rule 3 rather than from
scratch.

---

### Task 2: the root and the tree

The structural change, with flags otherwise untouched, so that a golden mismatch
here can only mean the harness moved.

- [x] ~~Extract `ShrikeCLICommand`'s options into a `ParsableArguments` struct~~
      **Not done, and not needed.** It was the option-group design the collision
      finding killed; generation stays one leaf command.
- [x] New `ShrikeRootCore` library holding `ShrikeRootCommand`, an
      `AsyncParsableCommand` (serve's `run()` is async, so the root must be)
      over `subcommands: [generate, serve, repack, bench]` with
      `defaultSubcommand: ShrikeGenerateCommand.self` and no options of its own.
      New `sources/ShrikeRoot/Command` executable target holding the three-line
      `@main extension`, product `shrike`.
- [x] A `bench` parent command with `attention` and `expert` children;
      `AttnBenchCommand`'s existing `list` subcommand nests under `attention`.
- [x] Set each command's `commandName` to its verb: `generate`, `serve`,
      `repack`, `attention`, `expert`.
- [x] **The root declares no options at all**, and generation is reached through
      `defaultSubcommand`. The two designs this replaced both failed on measured
      behaviour: a required option on the root breaks subcommand dispatch at exit
      64, and the root's `validate()` runs even when a subcommand runs. Worse,
      a flag shared between the root and a subcommand binds to the **root**, so
      `shrike serve --model X` reached serve with `model` nil and scanned the
      models directory instead. `ShrikeCLICommand` becomes
      `ShrikeGenerateCommand`, a leaf with `commandName: "generate"`, keeping its
      `validate()` and all ten cross-flag rules unchanged.
- [x] Delete the five executable targets, their `@main` shims and their products.
- [x] Move every consumer in the table under "Who names these binaries" in the
      spec: `tools/golden-baseline.sh`, `tools/mini-deploy.sh` (one binary, three
      bundles), `tools/decode-rig.sh`, `tools/turn-rig.sh`,
      `tools/ane-probes/shrike_ane_prefill_ab.py`, `CLAUDE.md`, and the receipt
      strings in `VerifiedInstallTool.swift` and `VerifiedInstallReceipt.swift`.
- [x] **Rewrite the process-name checks.** `tools/decode-rig.sh:58` and
      `tools/turn-rig.sh:66` use `pgrep -x ShrikeServer`, which never matches
      again once every process is named `shrike`, and fails by reporting "not
      running" rather than by erroring. The `pgrep -f` guards in `CLAUDE.md` and
      `tools/golden-baseline.sh:94` keep working but need new patterns.
- [x] Rewrite v23's argv pins against the new spelling. `ShrikeCLICommand`
      renamed across three test files; `RepackCLITests` now spawns
      `.build/debug/shrike` with `repack` prepended, keeping every case's argv.
- [x] New pins in `tests/ShrikeRoot`, ten cases. The load-bearing one is
      `serveKeepsEveryFlagItSharesWithGenerate`, which is the collision
      regression. Also: generation needs no verb and is reachable by name, the
      mini's production launch line parses to the right values, serve still
      refuses an unsupported `--max-context`, `repack verify-install` resolves
      two levels, `bench` resolves either child, an unknown verb is rejected.
      **Corrected against the binary:** a bare `shrike` is a usage error at exit
      64 naming `--model`, not help at exit 0 as this plan first claimed. The 0
      was measured on a probe whose options were all optional. Task 3 changes
      it.
- [ ] Four gates, `tools/golden-baseline.sh --check` byte-identical, commit.

---

### Task 3: model resolution

- [x] Move `ServerConfig` and the roster out of `ShrikeServerCore` into a new
      `ShrikeCatalog` target that both it and `ShrikeCLICore` depend on, 385
      lines in two files. The type follows the file: `ServerConfig` becomes
      `ShrikeConfig`, since it stops being server configuration the moment
      generate reads it. `ModelRosterTests` and `ServerConfigTests` move with
      them into a new `ShrikeCatalogTests`.
- [x] Default config path becomes `~/.shrike/config.json`. No file exists on
      either machine, so nothing on disk migrates and `server.json` is not read.
- [x] `--model` on **generate** accepts an id as well as a path: a value naming
      an existing directory is taken as a path, anything else as an id. Serve's
      `--model` deliberately stays a path, because it pins one bundle and
      ignores any config or roster by design.
- [x] Implement the chain in `ModelResolver`, split so the roster-dependent
      half is testable without the filesystem: `resolve(requested:in:)` is pure,
      `resolve(requested:configPath:modelsDir:)` loads and scans around it.
      Eight cases in `ModelResolverTests` cover path-wins, id, configured
      default, sole bundle, several-with-no-default, unknown id, and a bundle
      name staying an alias for a configured id.
- [x] Replace Task 2's plain check with the chain. A bare `shrike` is still a
      usage error at exit 64, but the message improved from `--model` to `one of
      --prompt or --messages-file is required`, which is the thing that genuinely
      cannot be resolved. Pinned on the exact string, and on the absence of
      `--model` in it.
- [x] **REVERSED: the candidate loop stays.** Id resolution reads the models
      directory, which defaults to `~/shrike-runtime/models` and is otherwise set
      in the config file. On the mini the models are there and `--model ornith15`
      would work; on the dev box they are on `/Volumes/BuildSSD/shrike`, so the
      same invocation would need a `~/.shrike/config.json` that is not in the
      repository. A gate that only passes on a configured machine is worse than
      the loop it replaced, so `golden-baseline.sh` keeps locating ornith15 by
      path across both roots. This is the second time a plan step about
      golden-baseline needed correcting, both times for the same reason: the
      baseline's bindings are not the CLI's conveniences.
- [ ] Four gates, golden `--check` byte-identical, commit.

---

### Task 4: the trim

Seventeen flags, not eighteen: `--rope-scaling` is a keep, see the spec.

- [ ] **T4a, the ten self-contained flags.** `--lazy-load`, `--preload`,
      `--models-dir`, `--queue-limit`, `--model-id` (serve only; it stays on
      `repack import-snapshot`) and `--idle-unload-seconds` on the server;
      `--overwrite` and `--resume` on repack; `--concise` and `--force-tokens`
      on generate. Three knock-on deletions the inventory did not name:
      `unloadDiscardsWarmCache` and its startup warning die with
      `--idle-unload-seconds`, so does `ShrikeConfig.Defaults.idleUnloadSeconds`,
      and `SHRIKE_REASONING_RETENTION` must leave the environment registry when
      `--reasoning-retention` goes or it becomes the silent no-op CLAUDE.md
      warns about for `NVMAI_*` (15 names to 14; a test pins the count).
- [ ] **T4b, the seven that reach `ModelSessionPlan`.** Per the owner's ruling
      the argument goes and the code stays: give the `ModelSessionPlan` parameter
      a default and stop passing it from `ModelRegistry`, rather than freezing it
      into a constant. Nothing follows into `ServerInference`.
- [ ] Delete each flag's declaration, its `validate()` rule, and any path that
      exists only to serve it.
- [ ] Remove their help text, their tests, and every mention in `docs/` that
      describes them as live. Historical implementation plans keep theirs, as the
      record of their own chapters.
- [ ] Re-run Task 1's consumer sweep against the trimmed surface to confirm no
      consumer names a deleted flag.
- [ ] Four gates, golden `--check` byte-identical, commit.

---

### Task 5: the close

- [ ] ThreadSanitizer over the suite:
      `env TSAN_OPTIONS=suppressions=tsan-suppressions.txt swift test --no-parallel --sanitize=thread`.
- [ ] Fresh-reader review of the whole branch against `main`, pointed at the
      documents as well as the code. At v23's close this found two argv
      regressions that four green gates and 1,332 tests had not.
- [ ] Fold the findings in one labelled commit, naming the owning commit per
      finding. Record the deviation, as v23 did.
- [ ] `docs/architecture.md`: the v24 history entry, and every stale reference to
      five binaries or to the retired spellings.
- [ ] `README.md`: the product list becomes one binary.
- [ ] The design doc's close: what this chapter got wrong about itself, and what
      the review caught that the gates could not.
- [ ] Final four gates on the final state, plus `tools/golden-baseline.sh --check`
      byte-identical on all five profiles against a release build of that state.
- [ ] **Owner's go required:** deploy to the mini, relaunch production on the new
      launch line, and re-run the golden baseline there against the mini's own
      tagged baselines. The mini has no git; build release here and copy the
      binary with its three `*.bundle` directories.
- [ ] **Owner's go required:** merge to `main` and push.
