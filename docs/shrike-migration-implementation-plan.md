# Shrike migration — implementation plan

Companion to [shrike-migration.md](shrike-migration.md). The checkboxes here are the
status of record.

Seven phases. Each ends at a gate that must pass before the next begins. Phases 1–6 are
ordinary commits that can be amended, reverted, or redone freely; Phase 7 is the only
irreversible step and is sequenced last for that reason.

**Deleting is not destroying.** The squashed import commit in Phase 7 carries upstream's
entire tree, and the bundle written at the start of that phase is a second copy outside
the repository. Anything on the delete-list is one `git show <import-commit>:<path>` from
returning. Phase 1 rescues only what belongs in the *working tree*; it is not an ark.

**Standing gate** (run at every gate marked *full*):

```bash
swift build -c release          # zero warnings
swiftlint lint --strict
swift test --no-parallel
swift test --no-parallel --sanitize=thread
```

Model-process check before anything that loads a model:

```bash
pgrep -fl 'NVMAIServer|NVMAIMac|NVMAIDecodeService|NVMAICLI|NVMAIPackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'
```

---

## Phase 1 — Rescue before subtracting

Nothing is deleted in this phase. It exists so that Phase 2 cannot destroy something we
decided to keep.

- [ ] Create `docs/ane-prefill.md` from `docs/v4.5-ane-prefill.md`
- [ ] Fold in `docs/v4.4-decode-width-plan.md` lines 487–579 (Track A: layer-mix table, MIL correctness validation, fp16-softmax precision finding, the `scaled_dot_product_attention` NaN/inf defect and its decomposed-attention workaround)
- [ ] Remove the now-internal link at old line 8 — the document must have no outbound reference to any deleted file
- [ ] `mkdir tools/ane-probes/` and move three files out of `benchmark/`:
  - [ ] `nvmai_ane_prefill_ab.py` — the interleaved A/B behind 2.31×
  - [ ] `nvmai_ane_attention_probe.py` — MIL block vs fp32 NumPy reference
  - [ ] `nvmai_ane_realweight_rehearsal.py` — real-int4 rehearsal (imports the probe above)
- [ ] Repoint every script path inside `ane-prefill.md` to `tools/ane-probes/`
- [ ] Delete `docs/v4.5-ane-prefill.md`
- [ ] Record in `ane-prefill.md` that `nvmai_ane_prefill_ab.py` needs two harness modules restored before it runs (see below)

**A known, documented rough edge.** `nvmai_ane_prefill_ab.py` imports 11 symbols from
`nvmai_profile.py` and `nvmai_gate0_profile.py`, both of which stay in `benchmark/` and are
deleted in Phase 2. So it will not run as-is. That is acceptable because the fix is one
command against history the migration deliberately preserves:

```bash
git show <import-commit>:benchmark/nvmai_profile.py > tools/ane-probes/nvmai_profile.py
git show <import-commit>:benchmark/nvmai_gate0_profile.py > tools/ane-probes/nvmai_gate0_profile.py
```

Rescuing them pre-emptively would drag 24 KB of general gate-0 profiling machinery into
`tools/` to serve a script nobody is running today. Vendoring a shim instead was rejected
outright: `gate0.preflight()` is the never-run-two-model-processes guard, and
reimplementing a safety check is worse than either option.

**Gate:** every relative link in `docs/ane-prefill.md` resolves; `nvmai_ane_attention_probe.py`
reaches `--help`. The A/B is expected to fail on import until the harness is restored, and
the document must say so.

---

## Phase 2 — The subtraction

The delete-list is **computed**, never hand-authored:

```bash
git ls-files > /tmp/all.txt        # keep-list per shrike-migration.md
# delete-list = all.txt minus keep-list, executed in one pass
```

- [ ] Write the keep-list to a file from the design document's table
- [ ] Generate the delete-list by subtraction; **review it before executing**
- [ ] `git rm` the delete-list — 70 files (plus 3 more in Phase 3, for 73 total):
  - [ ] `.github/` (6)
  - [ ] `benchmark/` (46 remaining after Phase 1)
  - [ ] `AGENTS.md`, `CONTRIBUTING.md`, `SECURITY.md`, `THIRD_PARTY_NOTICES.md` (4)
  - [ ] `assets/stats.png` (1)
  - [ ] `docs/`: `cpu-coexecution-plan.md`, `v4-core-design.md`, `v4.1`, `v4.2`, `v4.3`, `v4.4`, `v4.6` (7)
  - [ ] `docs/`: `v5-implementation-plan.md`, `v6-implementation-plan.md`, `multi-model-serving-implementation-plan.md` (3)
  - [ ] `tools/`: `release.sh`, `server_launcher.sh`, `cli_launcher.sh` (3) — `lint.sh` and the two baselines go in Phase 3
- [ ] `git add CLAUDE.md` — currently untracked
- [ ] `.gitignore`: drop the `benchmark/mock/` line
- [ ] `tools/golden-baseline.sh:23` — `OUT_DIR` default `$ROOT/benchmark/golden` → `$ROOT/baselines`
- [ ] Confirm `models/ornith-1.5_35B_A3B_8Bit.install.lock` (untracked, zero bytes, stale) — delete locally, no repo effect

**Before deleting, apply the operator-tool test** established by the
`prepare_ornith_mtp.py` near-miss: for any script on the delete-list, check for a consumer
of its *output*, not merely a caller. A grep for callers finds nothing for manual one-shot
tools.

**Gate:** *full*. Deleting documentation should not affect the build; if it does,
something referenced a deleted path at compile time.

---

## Phase 3 — Lint swap

- [ ] Create `.swiftlint.yml` at the repo root:

```yaml
only_rules:
  - force_cast
  - force_try
  - function_body_length

function_body_length:
  warning: 120
  error: 400

excluded:
  - tests
```

- [ ] Convert the two `lint:allow-force` annotations in `sources/` to `// swiftlint:disable:next force_cast` / `force_try` (1 each; SwiftLint does not understand the old syntax)
- [ ] `git rm tools/lint.sh tools/func-length-baseline.txt tools/unchecked-sendable-baseline.txt`
- [ ] Confirm `swiftlint lint --strict` exits 0 — expected: 22 `function_body_length` warnings, 0 errors

The 53 `unchecked-invariant:` comments stay in the source as documentation; they are no
longer mechanically checked, by decision. Tripadvisor's rule set can be bolted on later —
it produces 6,039 violations here because it assumes a companion SwiftFormat this
repository has never run.

**Gate:** *full*.

---

## Phase 4 — Documents

- [ ] **`README.md`** — full rewrite. Delete line 1 (image hotlinked from another user's GitHub attachment), line 82 (`assets/stats.png` → ornith.ai), line 91 (the "base 8-core M3 MacBook Pro with 24 GB" results, which are the upstream author's machine). Replace with: what this is, what it runs on, how to build, how to serve, on hardware that exists.
- [ ] **`docs/architecture.md`** — distil from `v4-core-design.md` before Phase 2 removes it (or from git history after). Roughly 80 lines. **Verify each invariant against `sources/` rather than carrying it on trust:**
  - [ ] RAM budget is an input, not an outcome
  - [ ] Streaming that genuinely uses the disk
  - [ ] No per-layer CPU round trip in decode
  - [ ] C99 for hot loops, Swift for structure
- [ ] **Resolve the predictive-prefetch contradiction.** v4-core-design §3 is titled `Predictive prefetch: PROVABLY CANNOT WORK`, yet `NVMAI_PREDICTIVE_PREFETCH`, `NVMAI_PREFETCH_TOP_M`, `NVMAI_PREFETCH_TRACE`, `ExpertPrefetchRing.swift` and `RealForwardRunner.swift:1237` all exist. Either the proof is wrong or the machinery is vestigial. Determine which; record the answer in `architecture.md`. Do not carry the contradiction forward.
- [ ] **`CLAUDE.md`** —
  - [ ] Delete the `AGENTS.md is a benchmark fixture` section
  - [ ] "CI runs five things" → "five local gates"; there is no CI and never was
  - [ ] Gate list: `tools/lint.sh` → `swiftlint lint --strict`
  - [ ] Update the golden-baseline note for the `baselines/` path and the recapture
- [ ] **Four dangling citations** (three others died with the implementation plans):
  - [ ] `tools/golden-baseline.sh:38` — `AGENTS.md` → `CLAUDE.md`
  - [ ] `tools/export_ane_prefill.py:12` — cites `docs/v4.4` Track A; inline the conclusion
  - [ ] `sources/NVMAI/Runtime/Generation/StreamingMTP.swift:144` — cites `docs/v4.4` Track B; inline the conclusion
  - [ ] `sources/NVMAI/Kernels/CPU/CPUExpertFFN.swift:22` — cites `docs/cpu-coexecution-plan.md`; inline the conclusion

A comment pointing at a deleted file is worse than no comment. Inline the one-line
conclusion the citation was reaching for; do not repoint at a document that will not
contain it.

**Gate:** every relative markdown link in the repository resolves. *full*.

---

## Phase 5 — The rename to Shrike

1,123 occurrences of `NVMAI` across 324 files, minus whatever Phase 2 removed. 13 products
and targets. `.gturbo` is **not** renamed.

- [ ] `sources/NVMAI*` and `tests/NVMAI*` directory names
- [ ] `Package.swift`: 13 products and targets (`NVMAI`, `NVMAIFormat`, `NVMAIKernelsC`, `NVMAIRepack`(+Core), `NVMAICLI`(+Core), `NVMAIAppCore`, `NVMAIMacPresentation`, `NVMAIDecodeProtocol`, `NVMAIDecodeService`, `NVMAIServerCore`, `NVMAIMac`, `NVMAIBench`)
- [ ] Swift type and symbol names
- [ ] The 37 `NVMAI_*` env vars → `SHRIKE_*`
- [ ] The 2 C include guards (`NVMAI_KERNELS_H`, `NVMAI_EXPERT_IO_H`) — no runtime risk
- [ ] Metal shader sources and any string literals
- [ ] `CLAUDE.md`, `README.md`, `docs/`, `tools/` references
- [ ] `pgrep` pattern in `CLAUDE.md` and `tools/golden-baseline.sh:38`
- [ ] Verify zero remaining matches: `rg -c 'NVMAI' --hidden -g '!.git'` returns nothing
- [ ] Verify `.gturbo` / `GTurbo` counts are **unchanged** (247 / 231) — the format name must survive

**Gate:** *full*. This is the phase with real risk; the TSan pass is not optional here.

---

## Phase 6 — Deploy checklist

The mini keeps running its current binary throughout. It is a build artifact and does not
care what its source was called. Binary and plist couple only at the next deploy.

- [ ] Add to `CLAUDE.md`'s deploy section: at next deploy the mini's service config must be
      updated in the same step — renamed executable **and** `NVMAI_*` → `SHRIKE_*` in the
      plist
- [ ] State the failure mode explicitly: a stale `NVMAI_*` variable does not error. It is
      silently never read, the default is taken, and tuned configuration disappears into an
      unexplained regression days later

---

## Phase 7 — History

Only irreversible step. Everything above must be committed and green first.

- [ ] `git bundle create ../shrike-prehistory.bundle --all` — parked outside the repo
- [ ] Create the squashed import commit: upstream's tree at the fork point, one commit, honestly labelled as an import of another codebase
- [ ] `git rebase --onto <import-commit> upstream/main HEAD` — replays our 62 commits plus the Phase 1–6 commits. Conflict-free: the base tree is exactly what those commits expect
- [ ] Delete `arch/restore-multi-family` and the old `main`
- [ ] `git remote remove upstream`; decide whether `origin` stays
- [ ] `git reflog expire --expire=now --all && git gc --prune=now --aggressive`
- [ ] **Verify the working tree is byte-identical to before surgery** — `git diff` against a pre-surgery stash or checkout must be empty
- [ ] `git count-objects -vH` — confirm the old objects are actually gone
- [ ] `git log --oneline | wc -l` — expect ~65

**Gate:** tree identical, object count reduced, `git log` shows the import commit as root
with our work above it, *full* suite green one final time.

---

## Phase 8 — Folder, remote, and everything keyed by path

Last, because renaming the working directory invalidates the cwd of any running session.
**Run 8.2 and 8.3 with Claude Code closed** — a live session rewrites `~/.claude.json`
from its own state on exit and will clobber a hand-edit.

### 8.1 — The local folder

- [ ] Confirm nothing has a cwd inside the repo (no shells, no editors, no sessions)
- [ ] `mv /Users/davorjankolija/Developer/NVMAI /Users/davorjankolija/Developer/Shrike`

### 8.2 — The remote

**A repository rename on GitHub is possible** — `gh repo rename`, or Settings → General.
It installs a redirect from the old URL, so existing clones keep working. Two options:

- **Rename**: one operation, keeps the repo, old URL redirects.
- **Fresh repo**: no redirect and no lingering `NVMAI` pointer. Also leaves behind no
  unreachable objects from the pre-rewrite history, which a force-push into the existing
  repo does leave on GitHub's side for a while.

Either way, **the account trap comes first**:

- [ ] `gh auth switch --hostname github.com --user djankolija` — the active `gh` account is
      `djankolija_tamg` (TripAdvisor). Creating or renaming without switching either fails
      or targets the wrong account. `djankolija` is a separate account, not an org, so the
      work token cannot act on it.
- [ ] Rename (`gh repo rename Shrike -R djankolija/NVMAI`) **or** create `djankolija/Shrike`
- [ ] Update the local remote — **must use the `github-personal` alias, never `github.com`**:
      `git remote set-url origin git@github-personal:djankolija/Shrike.git`
- [ ] Verify: `git remote -v` shows `github-personal`, and `ssh -T git@github-personal`
      greets `djankolija`, not `djankolija_tamg`
- [ ] Force-push the rewritten history from Phase 7
- [ ] `gh auth switch --hostname github.com --user djankolija_tamg` to restore the work
      default, if you want it back

The ssh config's own comment explains why the default is work: `Host github.com` uses the
TripAdvisor key because xcodebuild/SPM resolves `git@github.com:tripadv/` URLs baked into
`Package.swift` manifests and honors only that file. So the default cannot be flipped —
the personal alias must be explicit.

### 8.3 — State keyed by path

- [ ] `mv /Users/davorjankolija/.claude/projects/-Users-davorjankolija-Developer-NVMAI \
        /Users/davorjankolija/.claude/projects/-Users-davorjankolija-Developer-Shrike`
      — preserves 4 session transcripts and the `memory/` directory
- [ ] Edit `~/.claude.json`: the project key `"/Users/davorjankolija/Developer/NVMAI"` →
      `"/Users/davorjankolija/Developer/Shrike"` (one occurrence)
- [ ] `mv ~/.claude/handoffs/nvmai-decode-optimization.md \
        ~/.claude/handoffs/shrike-decode-optimization.md`
- [ ] Update inside that handoff: line 4 `repo:` path, line 9 `djankolija/NVMAI`, line 36
      the mini's `/Users/davor/nvmai-runtime/bin/` and the `NVMAIServer, NVMAICLI,
      NVMAIRepack` binary names
- [ ] Grep for stragglers: `rg -l 'Developer/NVMAI' ~/.claude --hidden -g '!*.jsonl'`

**Deliberately not migrated:** `~/.claude/file-history/` (content-addressed),
`~/.claude/backups/` (snapshots of prior state — rewriting them would defeat the point),
`~/.claude/logs/`, and the session/job state in `~/.claude/sessions/` and
`~/.claude/jobs/`. The scratchpad under `/private/tmp/claude-501/` is ephemeral.

`memory/` is currently **empty**, so nothing is lost there — but move the directory anyway
so future writes land in the right place.

### 8.4 — The mini

- [ ] Note in `CLAUDE.md` alongside the Phase 6 plist item: the deploy directory on the
      mini is `/Users/davor/nvmai-runtime/bin/` and holds `NVMAIServer`, `NVMAICLI`,
      `NVMAIRepack`. Renaming it is part of the same next-deploy step as the plist, or a
      deliberate decision to leave it — but it should not be discovered by surprise.

---

## Known gaps

**Real inference is not verified by any of this.** This Mac has no installed models, so
`tools/golden-baseline.sh` cannot run here. The rename touches identifiers, not numerics,
so no drift is expected — but expectation is not measurement. Inference stays unverified
until the next deploy to the mini. A green gate here does not mean otherwise.

**The golden baseline covers one model family of three.** It hardcodes
`models/ornith-1.5_35B_A3B_${q}Bit`. Kimi and gpt-oss — MLA, KDA kernels, the Harmony and
Kimi dialects, multi-family repack — are not exercised by the only real-inference check in
the repository. Out of scope here; worth its own work.

**ANE prefill remains off by default.** Promotion needs a quality qualification across a
real workload, not the speed number. Plus an unbuilt background preload worth a further
~15–20% of remaining prefill.

## Out of scope

- Any change to runtime behaviour, numerics, or architecture
- The `.gturbo` format and model re-attestation
- The Mac mini's running service
- SwiftFormat, and Tripadvisor's wider rule set
- The future clean-slate project
