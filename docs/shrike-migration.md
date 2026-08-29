# Shrike migration — design

Strip the inherited baggage from this repository, rename the project to **Shrike**, and
cut the imported history, so that what remains is only work we did and only claims that
are true of this machine.

## Why

This repository was cloned, not forked. `Pummelchen/NVMAI` is itself a fork of
`drumih/turbo-fieldfare`, and that upstream had already been forked into this account —
GitHub permits one fork per network per account, so a second was refused. The clone
brought a complete project with it — CI, contribution
policy, security contacts, benchmark instrumentation, agent instructions, and a README
about someone else's hardware. None of it was ever adapted, because none of it was ever
the point; the point was the runtime, which has since been rewritten twice.

The cost is not untidiness. It is that **agents read these files and act on them.**

Three observed instances:

- Sessions report on CI gates. No CI has ever run here. `CONTRIBUTING.md` states that
  "CI runs the same three, so a green run here is a green run there," which is false in
  both directions.
- Sessions cite benchmark figures from "the old MacBook." There is no old MacBook.
  `README.md:91` presents results from "a base 8-core M3 MacBook Pro with 24 GB" as this
  project's own, and it is the author's machine.
- The confusion has already cost commits. `36bf0a1` acted on the belief that the golden
  baseline came from retired hardware of ours; `9f18ea4` had to walk it back — *"the
  golden baseline's Mac15,3 is the upstream author's test machine, not retired hardware
  of ours."*

Every file on the delete-list below is judged by one question: **does it tell a session
something false about this setup?** That is the discriminator, not authorship. `LICENSE`
is inherited and stays, because it is inert and makes no claim about our machine.
`CONTRIBUTING.md` is inherited and goes, because it does.

## The name

`NVMAI` currently denotes five things in conversation: the original turbo implementation,
the pre-clone NVMAI, our clone, the current v6-era runtime, and the future clean-slate
project. Disambiguating them has required quoting binary SHAs.

The project becomes **Shrike** — a bird that impales prey on thorns to store it and
returns later to feed, which is what this runtime does with weights. It also continues a
motif already present in the lineage: *Fieldfare* is a thrush, and *Ornith* is Greek for
bird. The convention gives the future project its own name for free, so the ambiguity
cannot recur.

`.gturbo` is deliberately **kept**. It is the container format, it is genuinely inherited,
and carrying its name is attribution rather than baggage. Keeping it also removes all
model re-attestation from this work.

## Decisions

| Decision | Resolution |
|---|---|
| Selection procedure | Allowlist. Delete-list is computed, never hand-authored. |
| Name | Shrike. `.gturbo` unchanged. |
| Rename depth | Identifiers, targets, executables, env vars. Not the package format. |
| Live Mac mini | Untouched now; its plist updated at next deploy. |
| History | Squashed import commit, our 62 replayed on top, the other ~325 dropped. |
| `LICENSE` | Keep. |
| `benchmark/golden/` | Drop, recapture on real hardware. |
| `docs/v4-core-design.md` | Distil into `docs/architecture.md`, drop the original. |

## The selection procedure

The failure mode of a denylist is that it never terminates: anything not thought of
survives by default, and surfaces weeks later. An allowlist terminates by construction —
once the keep-list is complete, everything else is deleted without being individually
considered.

This does not require a fresh repository. It requires one subtraction:

```
delete-list = git ls-files  −  keep-list
```

executed in a single pass. The guarantee is identical to copying into an empty directory,
and nothing has to be remembered.

### Keep-list

| Entry | Reason |
|---|---|
| `sources/` (231 files) | The project. |
| `tests/` (193 files) | The project. |
| `Package.swift`, `Package.resolved` | Build definition. |
| `CLAUDE.md` | **Currently untracked.** Becomes tracked. |
| `.gitignore` | Minus its now-dead `benchmark/mock/` entry. |
| `docs/v5-second-architecture-objective.md` | Ours. The why behind v5. |
| `docs/v6-dialect-normalized-cache.md` | Ours. The why behind v6. |
| `docs/multi-model-serving.md` | Ours. The why behind multi-model. |
| `.swiftlint.yml` | **New.** Replaces the custom lint script. |
| `tools/golden-baseline.sh` | The only check exercising real inference. |
| `docs/ane-prefill.md` | **Rescued.** Live opt-in feature, self-contained. See below. |
| `tools/ane-probes/` (3 files) | **Rescued** from `benchmark/`. The instruments behind the 2.31×. |
| `tools/v6-probes/` | Ours. |
| `tools/convert_kimi_tokenizer.py` | Ours. |
| `tools/export_ane_prefill.py` | Inherited, but referenced by `ANEPrefillAttention.swift`. |
| `tools/prepare_ornith_mtp.py` | The only path to MTP on Ornith 1.5. See below. |
| `LICENSE` | Inert; makes no false claim. |

### Computed delete-list

Everything else. Enumerated here for review only — it is produced by subtraction, not by
selection.

| Entry | What it is |
|---|---|
| `.github/` (6 files) | CI for gates that never run; issue templates soliciting community benchmark reports; a PR template; a security contact pointing at `drumih/turbo-fieldfare`. |
| `AGENTS.md` | The previous author's agent-instructions file. Superseded by `CLAUDE.md` — `benchmark/nvmai_gate0_profile.py:158` still calls the never-run-two-models rule "the AGENTS.md rule," and that rule now lives in ours. |
| `benchmark/` (46 of 49 files) | Every file first-committed by "NVMAI Agent" or André Borchert. Referenced zero times by any document of ours. Optimization instrumentation for decisions v5 and v6 superseded. Three ANE probes are rescued first — see below. |
| `tools/server_launcher.sh` | A server has no business shipping its own launcher; if one should exist it belongs client-side. Also drifted — no config-mode awareness, and passes a `--max-time` the server no longer accepts. |
| `tools/cli_launcher.sh` | Pi is the client, already configured. |
| `benchmark/golden/` | Two stored baselines from the author's machine. `1e06d42` already established that the committed file's own scope note rules it out for our hardware. |
| `CONTRIBUTING.md` | Addressed to a contributor base that does not exist; links runtime-control docs to the Pummelchen wiki; asserts CI equivalence that is false. |
| `SECURITY.md` | Routes vulnerability reports to `drumih/turbo-fieldfare`. |
| `THIRD_PARTY_NOTICES.md` | A dependency review dated 2026-07-15, performed by someone else for a distribution that does not happen. |
| `assets/stats.png` | Published Ornith benchmark chart, linked to `ornith.ai`. Referenced only from `README.md:82`. |
| `docs/v4-core-design.md` | See below. |
| `docs/cpu-coexecution-plan.md`, `docs/v4.1`–`v4.3`, `v4.6` | v4-era planning for a runtime rewritten twice since. |
| `docs/v4.4-decode-width-plan.md` | 778 lines. Track A (~93) is extracted into `ane-prefill.md` first; the rest is superseded decode-width and MTP work. |
| `tools/release.sh` | A release process for a public repository we do not publish. |
| `tools/lint.sh` | Replaced by SwiftLint. See below. |
| `tools/func-length-baseline.txt`, `tools/unchecked-sendable-baseline.txt` | Both empty. Migration scaffolding whose migration is complete. |
| `docs/v5-implementation-plan.md` | Finished. Recoverable from history. |
| `docs/v6-implementation-plan.md` | Finished, 10/10. Recoverable from history. |
| `docs/multi-model-serving-implementation-plan.md` | Finished, 9/9. Recoverable from history. |

`AGENTS.md` is load-bearing today only by accident: `coder_cli_benchmark.py:143` and
`nvmai_hit_fixup_ab.py:43` read it at runtime as a long-prompt fixture, making its byte
count a measured variable. Nobody designed that — the scripts needed a large text blob
and it was the biggest file present. Both scripts are on the delete-list, so the coupling
dissolves rather than needing a frozen fixture.

### Couplings the subtraction breaks

**`tools/golden-baseline.sh` writes into `benchmark/`.** Line 23 defaults
`OUT_DIR="${OUT_DIR:-$ROOT/benchmark/golden}"` and line 60 writes
`$OUT_DIR/ornith-1.5-35b-a3b-${q}bit.txt`. Deleting `benchmark/` therefore breaks a tool
we are keeping. `OUT_DIR` is repointed to `$ROOT/baselines/` in the same change; the
directory is created fresh on recapture, since the two existing baselines are being
dropped anyway.

**Seven citations in kept files point at deleted ones.** All are one-line fixes, but they
dangle silently rather than failing loudly, so they are enumerated as tasks rather than
left to a sweep:

| Location | Cites | Fix |
|---|---|---|
| ~~`docs/v5-implementation-plan.md:24`~~ | `AGENTS.md` | Moot — file deleted |
| ~~`docs/v5-implementation-plan.md:259`~~ | `AGENTS.md` | Moot — file deleted |
| ~~`docs/v6-implementation-plan.md:29`~~ | `AGENTS.md` | Moot — file deleted |
| `tools/golden-baseline.sh:38` | `AGENTS.md` | → `CLAUDE.md` |
| `tools/export_ane_prefill.py:12` | `docs/v4.4-decode-width-plan.md` Track A | inline the conclusion |
| `sources/NVMAI/Runtime/Generation/StreamingMTP.swift:144` | `docs/v4.4` Track B | inline the conclusion |
| `sources/NVMAI/Kernels/CPU/CPUExpertFFN.swift:22` | `docs/cpu-coexecution-plan.md` | inline the conclusion |

Three of the four `AGENTS.md` citations were inside the implementation plans, which are
themselves now deleted, so only `golden-baseline.sh:38` needs the repoint. Worth recording
what those three showed before they go: the inherited file had propagated into documents
*we wrote*. It was never inert.

The three v4-doc citations are the harder case, because they point at *measurement
rationale* being deleted. Each explains a non-obvious numeric choice, which is precisely
the kind of comment worth keeping. The fix is to inline the one-line conclusion the
citation was reaching for, rather than repoint at a document that will not contain it —
a comment referring to a deleted file is worse than no comment.

Separately, six code comments name a v4.x *era* without citing a path (`ExpertLoadOperation.swift:75`
"the v4.1 fallback path", `RealForwardRunner.swift:4763` "v4.2 Phase B", and the v4.3
prefetch references). These do not dangle as links and are left alone. They are also
where the predictive-prefetch machinery shows up in the runtime, which is one more reason
that contradiction gets settled rather than inherited.

### The one entry a grep for callers gets wrong

`tools/prepare_ornith_mtp.py` is invoked by nothing — no source file, no test, no tool, no
document. On a naive reading of the allowlist test it fails, and an earlier draft of this
design kept it anyway on an invented justification. Both readings were wrong, in opposite
directions. The evidence:

- `SupportedModelSource.all` is `[qwen36, qwen36_8bit, ornith15, ornith15_8bit,
  qwen36MTP]`. **There is no Ornith MTP remote source** — the published MLX MTP build
  covers Qwen3.6 only, which is exactly the gap the script's docstring names.
- `NVMAIRepack/Command/main.swift:21` states it accepts "reproducibly derived sidecars
  such as Ornith's native MTP" — written to consume this script's output.
- `ServerArguments.swift:71` exposes `--mtp-model <dir>`.
- `SupportedModelSource.default` is `ornith15_8bit`, the model actually deployed.

The pipeline is `prepare_ornith_mtp.py` → MLX snapshot → `NVMAIRepack` → `--mtp-model`.
It is a manual one-shot whose output is handed on by path, so no caller exists to find.

**The general lesson for Phase 1:** "nothing references it" is a strong signal for
generated, imported, or programmatically invoked files, and a weak one for manual
operator tools. Anything on the delete-list that is a script an operator runs by hand
gets checked for a consumer of its *output* before deletion, not merely for a caller.

Compare `tools/export_ane_prefill.py`, which stays on ordinary evidence:
`sources/NVMAI/Runtime/Prefill/ANEPrefillAttention.swift:131` names it in an error
message telling the operator to run it.

## Documents rewritten, not kept or dropped

**`README.md`.** Line 1 hotlinks an image from another user's GitHub attachment upload;
line 82 points at `ornith.ai`'s published results; line 91 asserts the phantom MacBook.
Rewritten from scratch: what this is, what it runs on, how to build and serve, on the
hardware that actually exists.

**`docs/architecture.md`, distilled from `docs/v4-core-design.md`.** The original is 585
lines and disqualifies itself three times over:

1. *It retracts its own premise.* Line 17 opens `## CORRECTION: the premise this document
   was written on was wrong` — the core bet on ~6× SSD headroom was an arithmetic error;
   the real figure is ~1.4×.
2. *Its requirements have been overruled by us.* Line 448 begins `# Must survive the
   rewrite`; item 3 declares the `-fast` alias mandatory. Commit `adf574e`: *"delete the
   `-fast` alias — what a prompt contains is the client's decision."* Items 4 and 5 (idle
   unload by timer, unload by API) appear superseded by `ModelRegistry` in `2d8fc1e`,
   which replaced the model lifecycle wholesale — inferred from the commit, not verified
   line by line.
3. *Its measurements are the phantom machine.* The entire benchmark matrix.

Once the v4.x documents go, its only two inbound links go with them.

What survives is roughly forty lines of architectural spine, still true of `sources/`
today: RAM budget as an input rather than an outcome; streaming that genuinely uses the
disk; no per-layer CPU round trip in decode; C99 for hot loops with Swift for structure.
Each is verified against the code before being written down, not carried over on trust.

One contradiction must be **resolved rather than inherited**: section 3 is titled
`Predictive prefetch: PROVABLY CANNOT WORK`, yet `NVMAI_PREDICTIVE_PREFETCH`,
`NVMAI_PREFETCH_TOP_M`, and `NVMAI_PREFETCH_TRACE` all exist in the runtime and
`v4.3-predictive-prefetch-plan.md` planned the feature. Either the proof is wrong or the
machinery is vestigial. This is a live question about code still running.

**`CLAUDE.md`.** The `AGENTS.md` section is removed with its subject. "CI runs five
things" becomes "five local gates," since there is no CI and never was. The gate list
swaps `tools/lint.sh` for `swiftlint`. The golden-baseline note is updated for the
recapture, and the deploy section gains the plist checklist item.

Its description of `lint.sh` was also wrong: it documents "two rules" where the script
ran three — the `@unchecked Sendable` check was undocumented. Moot now that the script is
going, but a fair illustration of how fast this drifts even in a file nobody inherited.

## Linting: `lint.sh` → SwiftLint

`tools/lint.sh` is 285 lines of bash and ruby, inherited, enforcing three things. Two of
them — no `as!`/`try!`, no over-long functions — are stock SwiftLint rules reimplemented
by hand. The third checks that an `@unchecked Sendable` declaration carries an
`unchecked-invariant:` comment, which is not static analysis at all: `@unchecked` means
the compiler has already stopped checking, and a script cannot establish an invariant the
type system declined to. It greps for a magic string and confirms that *a* sentence was
written, not that the sentence is true.

It also shells out to bare `ruby`, which on this machine resolves through asdf shims. The
gate is coupled to a personal shell environment and breaks anywhere that is not
initialised.

**Replaced by a minimal opt-in `.swiftlint.yml`:**

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

Opt-in rather than opt-out, because SwiftLint's defaults produce 3,878 violations here —
half of them `identifier_name` objecting to `i`, `d`, `x`, `n` in kernel and tensor code
where those are the correct names. Tripadvisor's own 631-line config was evaluated and
produces 6,039, because it assumes a companion SwiftFormat this repository has never run;
roughly 80% of its violations are formatting. That rule set can be bolted on later once a
formatter decision is made.

`tests/` is excluded from the force rules, preserving the deliberate policy of the script
being replaced — its check was named "as! / try! outside tests."

The 120/400 split on function length keeps the 22 functions currently over 120 lines
**visible as warnings** while nothing blocks the gate. No baseline file, nothing hidden,
and the threshold tightens once the code has been read. The 53 existing
`unchecked-invariant:` comments stay in the source as documentation; they simply stop
being mechanically checked.

## The implementation plans

`v5-implementation-plan.md`, `v6-implementation-plan.md` (10/10) and
`multi-model-serving-implementation-plan.md` (9/9) are deleted. All three describe
finished work, and `CLAUDE.md` holds that a plan's checkboxes are its status of record —
once every box is ticked, the file's job is done. They stay fully recoverable from the
history this migration deliberately preserves.

The three design documents they pair with are kept, because *why* v5, v6 and multi-model
look as they do is not recoverable from a task list.

## ANE prefill: the one thing this migration nearly buried

`docs/v4.5-ane-prefill.md` was initially placed on the delete-list with the rest of the
v4.x set. That was a misclassification, and it is worth recording why, because it is the
sharpest test the discriminator got.

v4.1–v4.3 describe *superseded design*. v4.5 documents a **feature that still exists,
still works, and ships in the current binary** — an opt-in path that routes prefill
attention for full-attention layers through the Neural Engine via a Core ML sidecar,
measured at **2.31× end-to-end prefill** (132.90 s → 57.52 s on a 6,103-token prompt,
interleaved A/B, decode unchanged). It is off by default only because output is not
byte-identical: the sidecar computes fp16 with a different reduction order, ~1% mean
per-layer deviation against an fp32 reference.

By the discriminator this migration uses — *does this file tell a session something false
about this setup?* — it passes cleanly. Deleting it would have removed the only written
record of a shipped, qualified optimisation, leaving an error string in
`ANEPrefillAttention.swift` as the sole evidence the feature exists.

**It is made self-contained rather than kept as a cross-reference.** Track A of
`v4.4-decode-width-plan.md` (lines 487–579) holds the research the document currently
links to: the layer-mix measurement, the MIL correctness validation against a NumPy fp32
reference, the fp16-softmax precision investigation, and one real ANE hardware defect —
the fused `scaled_dot_product_attention` op emits NaN/inf from sequence length 2048, so
the integration must use decomposed attention. That section is folded in; the remaining
~685 lines of v4.4 are dropped.

The instruments move to `tools/ane-probes/`, following the precedent of `8f12be1` — *"the
instruments that caught the findings outlive the sweep"*:

- `nvmai_ane_prefill_ab.py` — the interleaved A/B that produced 2.31×
- `nvmai_ane_attention_probe.py` — MIL block validated against an fp32 NumPy reference
- `nvmai_ane_realweight_rehearsal.py` — real-int4-weight rehearsal

The A/B additionally imports 11 symbols from `nvmai_profile.py` and
`nvmai_gate0_profile.py`, which are not rescued — pulling them across would drag 24 KB of
general gate-0 profiling machinery into `tools/` for a script nobody runs today. So it
will not execute until they are restored, which is one `git show` against the import
commit. That is documented in `ane-prefill.md` rather than pre-empted.

This is the general principle for the whole migration: **the delete-list is not
destruction.** The squashed import commit carries upstream's entire tree, and
`git bundle` parks a second copy outside the repository, so anything removed here is one
command from coming back. Deletions are judged on what belongs in the working tree, not
on fear of losing something.

Renamed `docs/ane-prefill.md`, since it would otherwise be the lone survivor of a version
scheme whose other documents are gone.

**Flagged as follow-up, not folded in:** whether to promote ANE prefill toward default.
The document is explicit that this needs its own quality qualification rather than the
speed number, and the open question is not the 1% deviation — that is far smaller than the
error 4-bit quantisation already introduces — but whether long generations diverge in
low-probability positions across a real workload. There is also an unbuilt background
preload worth a further ~15–20% of remaining prefill.

## The rename

1,123 occurrences of `NVMAI` across 324 files; 13 products and targets; 39 `NVMAI_`
identifiers, of which `NVMAI_KERNELS_H` and `NVMAI_EXPERT_IO_H` are C include guards
rather than runtime configuration.

Two tiers, distinguished by risk:

**Compiler-verified.** Swift types, directories under `sources/` and `tests/`, target and
product names, documentation. If it builds clean, it is correct. No runtime exposure.

**Environment-coupled.** Executable names and the `NVMAI_*` env vars. Renaming the
executables makes the mini's service config point at a binary that no longer exists —
loud and immediate. The env vars are the sharper hazard: if the mini's plist sets
`NVMAI_EXPERT_CACHE_SLOTS` and the new binary reads only `SHRIKE_EXPERT_CACHE_SLOTS`,
nothing errors. The variable is silently never read, the default is taken, and tuned
configuration evaporates into an unexplained regression days later — precisely the class
of ghost this migration exists to eliminate.

A partial rename is not the safe option. Leaving `NVMAIServer` and `NVMAI_*` in place
would preserve the dead name in the two places it is actually typed.

### Against the live mini

The deployed binary is a build artifact. It does not care what its source was called, and
it keeps running self-consistently under the old names regardless of what happens here.
The two only couple at the **next deploy**, when binary and plist must change together.

So the mini is not touched by this work. The plist requirement is written into `CLAUDE.md`
as a deploy-step checklist item, which is what closes the env-var trap: it fires only if
the plist is forgotten, and the checklist is read at exactly the moment it matters.

## History

387 commits. 62 are ours; `origin/main` and `upstream/main` sit at the same commit, both
62 behind `HEAD`, so **none of our work is pushed anywhere.** Dropping history wholesale
would delete v5, v6, and the nine-part multi-model series outright; GitHub holds only the
~325 commits that were never ours.

The repository is 4.93 MiB packed, so size argues for nothing.

What we keep is the reasoning. Messages like *"Delete the structural fallback, since a
cache that matches on bytes has no use for a description of the turn that produced them"*
are the only record of **why** v5 and v6 look as they do; the tree shows what was decided
and never why. That is not reconstructable, so it is not squashed.

The procedure: one squashed import commit holding upstream's tree at the fork point, then
`git rebase --onto` replaying our commits on top. Conflict-free, because the base tree is
exactly what those commits expect. Old refs dropped, `gc --prune=now`, object count
verified.

Result: roughly 65 commits. Prehistory gone, `git blame` intact on our own work, commit 1
honestly labelled as an import of someone else's codebase.

**Sequenced last, deliberately.** It is the only irreversible step. Everything before it
is ordinary commits that can be amended or reverted, so the cleanup and the rename are
fully verified before any history is rewritten.

## Ordering

**Subtraction precedes rename.** Renaming `benchmark/`'s 49 files and then deleting them
would be wasted work; subtracting first materially cuts the rename surface.

**History surgery is last**, per above.

## Known gaps

**Real inference is not verified by this work.** This Mac has no installed models —
`models/` holds a single zero-byte lock file — so `tools/golden-baseline.sh` cannot run
here. The rename touches identifiers rather than numerics, so no drift is expected, but
expectation is not measurement. Inference stays unverified until the next deploy to the
mini, and the gates below must not be read as proving otherwise.

**The predictive-prefetch contradiction is opened, not closed**, by this design. It is
resolved during the `architecture.md` write-up.

## Gates

Unchanged from `CLAUDE.md`, minus the markdown link check that lived in the deleted CI
and is performed manually instead:

1. Release build, zero warnings
2. `swiftlint lint --strict` (replacing `tools/lint.sh`)
3. `swift test --no-parallel`
4. The same suite under ThreadSanitizer
5. Markdown links resolve

## Out of scope

- Any change to runtime behaviour, numerics, or architecture.
- The `.gturbo` format and model re-attestation.
- The Mac mini's running service.
- The future clean-slate project. This migration makes that project nameable; it does not
  begin it.
