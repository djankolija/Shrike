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
      constant, which was v23's fix) and not validated at all by the CLI, whose
      help advertises `1...262144`. `--max-context 1` parses. The defaults differ
      too, 4096 against 262144. Under one root these become siblings in one help
      tree, so the drift has to be resolved rather than inherited.
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

- [ ] Extract `ShrikeCLICommand`'s options into a `public ParsableArguments`
      struct in `ShrikeCLICore`, so both the root and the tests can reach them
      without the root's module owning generation.
- [ ] New `ShrikeRootCore` library holding `ShrikeRootCommand`: an
      `AsyncParsableCommand` carrying that option group, `subcommands: [serve,
      repack, bench]`, and its own `run()` that dispatches into `ShrikeCLICore`'s
      runner. New `sources/ShrikeRoot/Command` executable target holding the
      three-line `@main extension`, product `shrike`.
- [ ] A `bench` parent command with `attention` and `expert` children;
      `AttnBenchCommand`'s existing `list` subcommand nests under `attention`.
- [ ] Set each command's `commandName` to its verb: `serve`, `repack`, `bench`.
      They are currently the legacy binary names.
- [ ] **The root's options are declared optional** and enforced in `run()`. A
      required option on the root breaks subcommand dispatch at exit 64, measured
      at step zero. At this task the enforcement is a plain check with a clear
      message; Task 3 replaces it with resolution.
- [ ] Delete the five executable targets, their `@main` shims and their products.
- [ ] Move every consumer in the table under "Who names these binaries" in the
      spec: `tools/golden-baseline.sh`, `tools/mini-deploy.sh` (one binary, three
      bundles), `tools/decode-rig.sh`, `tools/turn-rig.sh`,
      `tools/ane-probes/shrike_ane_prefill_ab.py`, `CLAUDE.md`, and the receipt
      strings in `VerifiedInstallTool.swift` and `VerifiedInstallReceipt.swift`.
- [ ] **Rewrite the process-name checks.** `tools/decode-rig.sh:58` and
      `tools/turn-rig.sh:66` use `pgrep -x ShrikeServer`, which never matches
      again once every process is named `shrike`, and fails by reporting "not
      running" rather than by erroring. The `pgrep -f` guards in `CLAUDE.md` and
      `tools/golden-baseline.sh:94` keep working but need new patterns.
- [ ] Rewrite v23's argv pins against the new spelling, re-probing each message
      and exit code from the binary rather than assuming them.
- [ ] New pins, each of which would otherwise break silently: a bare `shrike`
      prints help at exit 0; `shrike serve` does not demand the root's options;
      `shrike repack verify-install` resolves two levels deep; an unknown
      subcommand errors with `Unexpected argument` at exit 64; the mini's
      production launch line parses.
- [ ] Four gates, `tools/golden-baseline.sh --check` byte-identical, commit.

---

### Task 3: model resolution

- [ ] Move `ServerConfig` and the roster's default selection out of
      `ShrikeServerCore` into a target both it and `ShrikeCLICore` can depend on.
      It stops being server configuration the moment the bare form reads it.
- [ ] Default config path becomes `~/.shrike/config.json`. No file exists on
      either machine, so nothing on disk migrates and `server.json` is not read.
- [ ] `--model` accepts **an id as well as a path**, resolved against the models
      directory through the roster.
- [ ] Implement the chain: explicit `--model` wins; else the config entry marked
      `default: true`; else the models directory's only servable bundle when it
      holds exactly one; else an error naming the ids it found.
- [ ] Replace Task 2's plain check with the chain. Pin the error's message: a
      weakened diagnostic is the class v23's close caught twice.
- [ ] Replace `tools/golden-baseline.sh:52`'s hardcoded candidate loop with
      `--model ornith15`, resolved by id. **The baseline stays bound to
      ornith15**: a baseline is valid for one (machine, build, model) triple, so
      the script must keep naming its model. What the loop was doing was locating
      that model across two possible roots, and that is what id resolution
      replaces.
- [ ] Four gates, golden `--check` byte-identical, commit.

---

### Task 4: the trim

- [ ] Delete each flag Task 1 condemned, together with the code path it selected.
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
