# v25 flags are not a substitute for infrastructure: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the golden checks what production runs by driving `shrike serve` at
production's launch, and `generate` loses the two flags that only let it imitate
the server (SHRIKE-57).

**Architecture:** `tools/golden-baseline.sh` gains three server profiles that start
a fresh `shrike serve` from `tools/mini-production.sh`'s launch values on a spare
port and send the golden's prompts over HTTP; the CLI keeps `short` and `long` on
`generate`'s own fused head. A new `tools/mini-golden.sh` runs the golden on the
mini from the checkout, stopping and relaunching production around it, with the
repo's `baselines/*.mini.txt` as the only copy of the mini's baselines. Then
`--logits-head` and `--follow-up` leave `generate`.

**Tech Stack:** bash (the scripts; the mini's login shell is zsh, which runs the
remote strings), `curl` and `jq` (on both boxes at `/usr/bin`), Swift 6.3 with
swift-argument-parser and Swift Testing.

**Spec:** [v25-argument-discipline.md](v25-argument-discipline.md), its two step-zero
sections: the golden through the server (SHRIKE-2) and what the mini needs
(SHRIKE-3).

Three commits. The checkboxes here are the status of record.

## Global Constraints

- The four gates before any task is called done: `swift build -c release` (zero
  warnings), `swiftlint lint --strict`, `python3 tools/check-md-links.py`,
  `swift test --no-parallel`. ThreadSanitizer once at the chapter's close, not here.
- Before anything that loads a model: `pgrep -lx shrike; pgrep -fl
  'ShrikePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'`. Never terminate a
  process this session did not start.
- Production on the mini goes down only with the owner's go-ahead at that step;
  a go-ahead covers the step it was given for.
- A baseline is re-captured only for a deliberate numerics change. The `serve-*`
  files are first captures; `short` and `long` are never re-captured here, and a
  mismatch on them stops the task.
- No new `SHRIKE_*` variable. Production's launch stays written once, in
  `tools/mini-production.sh`.
- Stage by path, never `git add docs/` or `-a`: the tree carries another
  session's uncommitted edits to `docs/v5-*`, `docs/v6-*` and `docs/v7-*`.
- Commit subjects in the repo's style, at most 100 characters, ending
  `(SHRIKE-57)`. No `Co-Authored-By`.
- Comments only for a non-obvious why.
- A claim about what a script or binary does costs one run of it.

## Review Focus

- A `shrike` is already running when the golden starts: it refuses (exit 3) and
  kills nothing. Checked in Task 1, Step 6.
- A server profile's server never becomes ready, or the server rejects a request:
  the profile reports FAILED with the reason, captures nothing, and leaves no
  server running. Checked in Task 1, Step 6.
- `tools/mini-golden.sh` fails or is interrupted after stopping production:
  production is relaunched anyway, or the wrapper says loudly why not. Checked in
  Task 2, Step 5.
- A capture on the mini brings back exactly the profiles it captured; a `--check`
  writes nothing into `baselines/`. Checked in Task 2, Steps 6 and 7.
- The prompt cache stops resuming (a replay or a follow-up prefilled whole): the
  usage rows in each `serve-*` file carry the cached counts, so the gate fails
  even if the text survives. Checked in Task 1, Step 5.

---

### Task 1: the server profiles in the golden

**Files:**
- Modify: `tools/mini-production.sh` (the launch's two remaining literals become
  variables the golden reads)
- Modify: `tools/golden-baseline.sh` (whole file; the new version is below)
- Modify: `tests/Shrike/Core/CLI/CLIArgumentsTests+Invocations.swift` (the pins of
  the three retired invocations go)
- Test: `tests/ShrikeServer/OpenAIValidationTests.swift` (a pin of the server
  profiles' request body)
- Create: `baselines/ornith15-int4-serve-{short,long,turns}.Mac167.txt`
- Delete: `baselines/ornith15-int4-{short,long,turns}-lh.Mac167.txt`

**Interfaces:**
- Produces: `PRODUCTION_MAX_CONTEXT` and `PRODUCTION_THINKING` in
  `tools/mini-production.sh`; the golden's profile names `short`, `long`,
  `serve-short`, `serve-long`, `serve-turns`; its env `CLI`, `MODEL`, `OUT_DIR`,
  `MACHINE_TAG`, `SERVE_PORT`; it sources `mini-production.sh` from its own
  directory, so the two files travel together (Task 2 relies on that).

- [x] **Step 1: Write the request pin.** Add to `OpenAIValidationTests`, after
      `omittedSamplingControlsUseProductionDefaults`:

```swift
    @Test func goldenServeProfileRequestIsGreedy() throws {
        let data = Data(#"""
        {"model":"ornith15","max_tokens":96,"temperature":0,"seed":1234,"stream":false,
         "messages":[{"role":"user","content":"Explain what a mutex is and when you would use one."}]}
        """#.utf8)
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let validated = try OpenAIRequestValidator.validate(request)
        #expect(validated.generationConfig.isPureGreedy)
        #expect(validated.generationConfig.maxNewTokens == 96)
        #expect(validated.generationConfig.seed == 1_234)
        #expect(!validated.stream)
    }
```

- [x] **Step 2: Run it.** `swift test --no-parallel --filter goldenServeProfileRequestIsGreedy`.
      Expected: PASS (it pins today's behaviour, which the step-zero run used; a
      FAIL means the step-zero record and the code disagree, so stop and report).

- [x] **Step 3: Name the launch's remaining literals.** In `tools/mini-production.sh`,
      after `PRODUCTION_RAM_BUDGET=11324620800`, add:

```bash
PRODUCTION_MAX_CONTEXT=32768
PRODUCTION_THINKING=off
```

      and in `server_launch` replace `--max-context 32768` with
      `--max-context $PRODUCTION_MAX_CONTEXT` and `--thinking off` with
      `--thinking $PRODUCTION_THINKING`, keeping the argument order, so CLAUDE.md's
      copy of the launch line still reads the same when expanded. Update the
      file's header comment: it is also sourced by `tools/golden-baseline.sh`.

- [x] **Step 4: Replace `tools/golden-baseline.sh`** with:

```bash
#!/usr/bin/env bash
# Capture a deterministic generation baseline, so a refactor of the runtime
# can be checked against byte-identical output rather than "the tests still
# pass".
#
#   tools/golden-baseline.sh [profile ...]     # capture; default: all five
#   tools/golden-baseline.sh --check [profile ...]
#
# Profiles: short and long run `shrike generate` on its fused greedy head, the
# path a CLI user gets at temperature 0. serve-short, serve-long and serve-turns
# each start a fresh `shrike serve` at production's launch (tools/mini-production.sh)
# on SERVE_PORT and send the same prompts as chats over HTTP: serve-short and
# serve-long send theirs twice in a row, the second resuming from the prompt cache
# as a retry does; serve-turns answers a question to its stop, then sends the
# follow-up with that answer in the history, the continuation production computes.
# A server profile's file ends with its usage rows, so a change in what the cache
# resumes fails the check even when the text survives it.
#
# Determinism comes from greedy decoding: temperature 0 with a fixed seed and
# a fixed prompt. Greedy means the sampler never draws, so the only inputs are
# the weights and the kernels, exactly what a runtime refactor must not change.
# The `long` prompt pins a ~2k-word context: long-context near-tie picks are
# where reduction-order drift between binaries shows first (v6 numerics note).
# The server's answer also depends on how the cache splits a prompt into
# prefill chunks (docs/v25-argument-discipline.md), so its requests go in a
# fixed order to a fresh process.
#
# SCOPE: a baseline is valid for one (machine, build, model) triple. Metal
# reduction order is not guaranteed across GPU families, so a file captured on
# an M1 is not a reference for an M4; files carry a machine tag in their name.
# Re-capture after a deliberate, signed-off numerics change; a diff at any
# other time is a regression.
#
# On the mini, run tools/mini-golden.sh from the checkout; it sets CLI, MODEL,
# OUT_DIR and MACHINE_TAG. SERVE_PORT (default 8082) is never production's.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/mini-production.sh"
CLI="${CLI:-$ROOT/.build/arm64-apple-macosx/release/shrike}"
OUT_DIR="${OUT_DIR:-$ROOT/baselines}"
MACHINE_TAG="${MACHINE_TAG:-$(sysctl -n hw.model | tr -cd '[:alnum:]')}"
MAX_NEW="${MAX_NEW:-96}"
SEED="${SEED:-1234}"
SERVE_PORT="${SERVE_PORT:-8082}"
SERVE_URL="http://127.0.0.1:$SERVE_PORT"

if [ -z "${MODEL:-}" ]; then
  for candidate in /Volumes/BuildSSD/shrike/ornith15.gturbo \
                   "$HOME/shrike-runtime/models/ornith15.gturbo"; do
    if [ -f "$candidate/verified-install.json" ]; then MODEL="$candidate"; break; fi
  done
fi
if [ -z "${MODEL:-}" ] || [ ! -f "$MODEL/verified-install.json" ]; then
  echo "no verified ornith15.gturbo install found; set MODEL=" >&2
  exit 2
fi
MODEL_ID="$(basename "$MODEL" .gturbo)"

SHORT_PROMPT="Explain what a mutex is and when you would use one."
# ~2k words of deterministic context: a fixed ledger the model is asked to
# summarize. Built from literals only; never touch this construction, the
# stored baselines depend on its exact bytes.
long_prompt() {
  printf 'You are auditing a build ledger. Entries follow.\n'
  for i in $(seq 1 60); do
    printf 'Entry %d: commit c%04d built target shrike-core in %d ms with 0 warnings, ran 1108 tests in %d ms, linked 3 artifacts, and archived bundle b%03d to shelf s%d.\n' \
      "$i" $((i * 37)) $((1200 + i * 13)) $((80000 + i * 211)) "$i" $((i % 7))
  done
  printf 'Summarize: how many entries, which shelf received the most bundles, and the trend in build times.\n'
}
# The question must end at the model's stop token within max_tokens, or the
# follow-up never exercises the settled continuation.
TURN_QUESTION="In one sentence, what is a mutex?"
TURN_FOLLOW_UP="And a semaphore, in one sentence?"

mode=capture
if [ "${1:-}" = "--check" ]; then mode=check; shift; fi
profiles=("$@")
[ ${#profiles[@]} -eq 0 ] && profiles=(short long serve-short serve-long serve-turns)

if [ ! -x "$CLI" ]; then
  echo "missing $CLI; run: swift build -c release (or set CLI=)" >&2
  exit 2
fi
if ! command -v jq > /dev/null; then
  echo "jq is required for the server profiles" >&2
  exit 2
fi

# CLAUDE.md: never run alongside another model process, and never terminate one
# we did not start. Refuse rather than race.
if pgrep -x shrike >/dev/null 2>&1 \
   || pgrep -f 'ShrikePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm' >/dev/null 2>&1; then
  echo "a model process is already running; stop it yourself, then re-run" >&2
  exit 3
fi

server_pid=""
stop_server() {
  [ -n "$server_pid" ] || return 0
  kill "$server_pid" 2>/dev/null
  wait "$server_pid" 2>/dev/null
  server_pid=""
}
trap stop_server EXIT
trap 'exit 130' INT TERM HUP

# health_until <jq condition> <seconds>: polls /health while our server lives.
health_until() {
  local n=0
  until curl -sf -m 3 "$SERVE_URL/health" 2>/dev/null | jq -e "$1" > /dev/null 2>&1; do
    n=$((n + 1))
    if [ "$n" -gt "$2" ] || ! kill -0 "$server_pid" 2>/dev/null; then return 1; fi
    sleep 1
  done
}

# start_server <log>: a fresh server at production's launch with the model
# resident. The ready line and /v1/models answer before any model loads, so
# readiness is /health naming the model as resident.
start_server() {
  env $PRODUCTION_ENV "$CLI" serve --model "$MODEL" --port "$SERVE_PORT" \
      --max-context "$PRODUCTION_MAX_CONTEXT" --ram-budget "$PRODUCTION_RAM_BUDGET" \
      --thinking "$PRODUCTION_THINKING" > "$1" 2>&1 &
  server_pid=$!
  health_until '.status == "ok"' 60 || return 1
  curl -sf -m 10 "$SERVE_URL/v1/models/load" -H 'Content-Type: application/json' \
       -d "{\"model\":\"$MODEL_ID\"}" > /dev/null || return 1
  health_until ".resident == \"$MODEL_ID\" and .loading == false" 600
}

# chat_body <max_tokens> <role> <content> [<role> <content> ...]: a greedy,
# unstreamed request.
chat_body() {
  local max=$1
  shift
  jq -n --arg id "$MODEL_ID" --argjson max "$max" --argjson seed "$SEED" --args '
    {model: $id, max_tokens: $max, temperature: 0, seed: $seed, stream: false,
     messages: [$ARGS.positional as $p | range(0; $p | length; 2)
                | {role: $p[.], content: $p[. + 1]}]}' "$@"
}

# send <body> <answer file> <usage file> <label>: the answer's exact bytes, and
# one usage row.
send() {
  local resp
  resp="$(curl -s -m 1800 "$SERVE_URL/v1/chat/completions" \
          -H 'Content-Type: application/json' -d "$1")"
  if ! printf '%s' "$resp" | jq -e '.choices[0].message.content | strings' > /dev/null 2>&1; then
    echo "request $4 failed: $(printf '%s' "$resp" | head -c 300)" >&2
    return 1
  fi
  printf '%s' "$resp" | jq -j '.choices[0].message.content' > "$2"
  printf '%s' "$resp" | jq -r --arg l "$4" '"\($l) prompt=\(.usage.prompt_tokens) cached=\(.usage.prompt_tokens_details.cached_tokens // 0) completion=\(.usage.completion_tokens) finish=\(.choices[0].finish_reason)"' >> "$3"
}

# run_serve <profile> <work>: one server profile into <work>; diagnostics on stderr.
run_serve() {
  local profile=$1 work=$2 dir separator body rc
  dir="$(mktemp -d "${TMPDIR:-/tmp}/golden-serve.XXXXXX")"
  if ! start_server "$dir/server.log"; then
    echo "the server never had $MODEL_ID resident; its log ends:" >&2
    tail -5 "$dir/server.log" >&2
    stop_server; rm -rf "$dir"; return 1
  fi
  case "$profile" in
    serve-short|serve-long)
      separator=replay
      body="$(chat_body "$max_new" user "$prompt")"
      send "$body" "$dir/first" "$dir/usage" first \
        && send "$body" "$dir/second" "$dir/usage" replay ;;
    serve-turns)
      separator=follow-up
      send "$(chat_body "$max_new" user "$TURN_QUESTION")" "$dir/first" "$dir/usage" turn1 \
        && send "$(chat_body "$max_new" user "$TURN_QUESTION" assistant - user "$TURN_FOLLOW_UP" \
                   | jq --rawfile a "$dir/first" '.messages[1].content = $a')" \
                "$dir/second" "$dir/usage" turn2 ;;
  esac
  rc=$?
  stop_server
  if [ $rc -eq 0 ]; then
    { cat "$dir/first"; printf '\n--- %s ---\n' "$separator"; cat "$dir/second"
      printf '\n--- usage ---\n'; cat "$dir/usage"; } > "$work"
  fi
  rm -rf "$dir"
  return $rc
}

mkdir -p "$OUT_DIR"
status=0

for profile in "${profiles[@]}"; do
  case "$profile" in
    short|serve-short) prompt="$SHORT_PROMPT"; max_new="$MAX_NEW" ;;
    long|serve-long)   prompt="$(long_prompt)"; max_new=128 ;;
    serve-turns)       prompt="$TURN_QUESTION"; max_new=128 ;;
    *) echo "unknown profile: $profile (expected short, long, serve-short, serve-long or serve-turns)" >&2
       status=1; continue ;;
  esac
  file="$OUT_DIR/ornith15-int4-${profile}.${MACHINE_TAG}.txt"
  work="$(mktemp "${TMPDIR:-/tmp}/golden-baseline.XXXXXX")"

  echo "== $profile =="
  case "$profile" in
    serve-*)
      head="logits (shrike serve at tools/mini-production.sh's launch)"
      run_serve "$profile" "$work" 2>"$work.err" ;;
    *)
      head=fused
      # --quiet keeps the timing footer out of the compared text; only the
      # generated tokens are the contract. Timings vary run to run by design.
      "$CLI" generate --model "$MODEL" --prompt "$prompt" --max-new "$max_new" \
             --temperature 0 --seed "$SEED" --quiet > "$work" 2>"$work.err" ;;
  esac
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "  FAILED (exit $rc)"; sed 's/^/  /' "$work.err" | head -20
    rm -f "$work" "$work.err"; status=1; continue
  fi

  if [ "$mode" = capture ]; then
    {
      echo "# profile:     $profile (prompt $(printf '%s' "$prompt" | wc -c | tr -d ' ') bytes)"
      echo "# max-new:     $max_new"
      echo "# temperature: 0 (greedy)"
      echo "# head:        $head"
      echo "# seed:        $SEED"
      echo "# model:       $(basename "$MODEL")"
      echo "# captured-on: $(sysctl -n hw.model), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
      echo "# commit:      $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)$(git -C "$ROOT" diff --quiet 2>/dev/null || echo '-dirty')"
      echo "---"
      cat "$work"
    } > "$file"
    echo "  captured -> $file ($(wc -c < "$work" | tr -d ' ') bytes)"
  else
    if [ ! -f "$file" ]; then
      echo "  no baseline at $file; run without --check first"; status=1
    elif diff -q <(sed '1,/^---$/d' "$file") "$work" >/dev/null; then
      echo "  ok: output identical to baseline"
    else
      echo "  MISMATCH against $file:"
      diff <(sed '1,/^---$/d' "$file") "$work" | head -30 | sed 's/^/    /'
      status=1
    fi
  fi
  rm -f "$work" "$work.err"
done

exit $status
```

      Note the check's body split: `sed '1,/^---$/d'` deletes through the first
      line that is exactly `---`. The separators inside a server file are
      `--- replay ---`, `--- follow-up ---` and `--- usage ---`, never a bare
      `---`, so the body survives intact.

- [x] **Step 5: Capture and check on the dev box.** Run the process checks, then
      `tools/golden-baseline.sh serve-short serve-long serve-turns`. Expected: three
      `captured ->` lines. Open each file and confirm (a) the replay's and
      follow-up's text match the step-zero run's `Mac167-replay/launch1/pass1/`
      files byte for byte (`short-replay.txt`, `long-replay.txt`, `turn2.txt`), and
      (b) the usage rows show `cached=18` on the short replay, `cached=3749` on the
      long replay and a non-zero `cached=` on `turn2`. Then run
      `tools/golden-baseline.sh --check` twice. Expected: five `ok` lines each time.

- [x] **Step 6: Check the failure paths.**
      - Start a model-free stand-in named shrike, run the golden, expect refusal:
        compile `int main(void) { sleep(90); return 0; }` (with `<unistd.h>`) to a
        scratch path named `shrike` and run it in the background (a copied
        `/bin/sleep` is killed by macOS's code-signing check before it runs), then
        `tools/golden-baseline.sh --check short`. Expected: `a model process is
        already running`, exit 3. Then stop the stand-in (this session started it).
      - A rejected request: `env SEED=-1 tools/golden-baseline.sh --check serve-short`.
        Expected: `FAILED` with `request first failed:` and the server's error
        body (the seed is a `UInt64`), exit 1, and `pgrep -lx shrike` empty after.
      - A server that never becomes ready: point `CLI` at a binary that exits at
        once, `env CLI=/usr/bin/false tools/golden-baseline.sh --check serve-short`.
        Expected: `FAILED` with `the server never had ornith15 resident`, exit 1,
        nothing left running. (The short CLI profile is not run here: `false`
        would fail it for an unrelated reason.)

- [x] **Step 7: Retire the CLI's imitation profiles.** Delete
      `baselines/ornith15-int4-{short,long,turns}-lh.Mac167.txt` (`git rm`). In
      `CLIArgumentsTests+Invocations.swift`, delete `turnsFollowUp`,
      `goldenBaselineLogitsHeadProfileParses`, `goldenBaselineTurnsProfileParses` and
      `goldenBaselineExtraArgumentsParse`: they pin invocations the golden of
      Step 4 does not issue. `goldenBaselineShortProfileParses` stays as it is until Task 3.
      The `.mini` `-lh` files stay until Task 2 captures the mini's server profiles.

- [x] **Step 8: The four gates**, then commit `tools/mini-production.sh`,
      `tools/golden-baseline.sh`, the two test files, the three new and three
      deleted `.Mac167` baselines, and this plan with this task ticked:
      `tools: the golden drives shrike serve at production's launch (SHRIKE-57)`.

### Task 2: the golden on the mini, from the checkout

**Files:**
- Create: `tools/mini-golden.sh`
- Modify: `tools/mini-deploy.sh:43-45` (stops copying the golden to the mini)
- Modify: `CLAUDE.md` (the golden paragraph under "Verifying a change that touches
  inference", and "The mini's layout")
- Modify: `README.md:108-116` (the golden command and its framing)
- Create: `baselines/ornith15-int4-serve-{short,long,turns}.mini.txt`
- Delete: `baselines/ornith15-int4-{short,long,turns}-lh.mini.txt`; on the mini,
  `~/shrike-runtime/golden-baseline.sh` and `~/shrike-runtime/baselines/`

**Interfaces:**
- Consumes: `tools/golden-baseline.sh` and `tools/mini-production.sh` from Task 1,
  copied side by side; `server_launch`, `PRODUCTION_MODEL`, `PRODUCTION_PORT`,
  `PRODUCTION_RAM_BUDGET`, `SERVER_LOG`.
- Produces: `tools/mini-golden.sh [--check] [profile ...]`, exit status the golden's.

- [ ] **Step 1: Write `tools/mini-golden.sh`:**

```bash
#!/bin/bash
# tools/mini-golden.sh [--check] [profile ...]
# The golden on the mini, run from the checkout: the mini has no checkout, and
# the repo's baselines/*.mini.txt are the only copy of its baselines. Stops
# production's server, runs tools/golden-baseline.sh there from a scratch
# directory holding it, tools/mini-production.sh and (for --check) the mini's
# baselines, brings a capture's files back into baselines/, and relaunches
# production from tools/mini-production.sh whatever the golden's outcome.
# Refuses, stopping nothing, while any shrike other than production's server runs
# there. Arguments pass through to the golden.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/tools/mini-production.sh"
mode=capture
[ "${1:-}" = --check ] && mode=check
# The bracket keeps the pattern from matching the remote shell whose argv carries it.
PRODUCTION_MATCH="[b]in/shrike serve --model $PRODUCTION_MODEL --port $PRODUCTION_PORT"
scratch=""

relaunch() {
  local launch
  launch=$(server_launch "$PRODUCTION_MODEL" "$PRODUCTION_PORT" "$PRODUCTION_RAM_BUDGET")
  [ -n "$scratch" ] && ssh macmini "rm -rf $scratch"
  ssh macmini "
    n=0
    while pgrep -x shrike > /dev/null; do
      n=\$((n + 1))
      [ \$n -gt 10 ] && { echo 'a shrike process is still running; production NOT relaunched' >&2; exit 1; }
      sleep 1
    done
    cd ~/shrike-runtime
    [ -f $SERVER_LOG ] && mv -f $SERVER_LOG \"$SERVER_LOG.\$(date +%Y%m%d-%H%M%S)\"
    $launch
    tries=0
    until curl -sf -m 3 http://127.0.0.1:$PRODUCTION_PORT/v1/models > /dev/null 2>&1; do
      tries=\$((tries + 1))
      [ \$tries -gt 60 ] && { echo 'production never answered' >&2; exit 1; }
      sleep 2
    done
    echo \"production relaunched: \$(grep -a -m1 'ready at' $SERVER_LOG)\"
  "
}

ssh macmini "
  if [ \"\$(pgrep -x shrike | sort)\" != \"\$(pgrep -f '$PRODUCTION_MATCH' | sort)\" ]; then
    echo 'a shrike other than production is running on the mini; stopping nothing' >&2
    exit 1
  fi
  pkill -f '$PRODUCTION_MATCH' || true
  sleep 3
  if pgrep -x shrike > /dev/null; then echo 'production did not stop' >&2; exit 1; fi
" || exit 1
trap relaunch EXIT
trap 'exit 130' INT TERM HUP

scratch=$(ssh macmini 'mktemp -d /tmp/shrike-golden.XXXXXX') || exit 1
ssh macmini "mkdir $scratch/baselines" || exit 1
scp -q "$ROOT/tools/golden-baseline.sh" "$ROOT/tools/mini-production.sh" "macmini:$scratch/" || exit 1
if [ "$mode" = check ]; then
  scp -q "$ROOT"/baselines/*.mini.txt "macmini:$scratch/baselines/" || exit 1
fi
ssh macmini "env CLI=\$HOME/shrike-runtime/bin/shrike MODEL=\$HOME/shrike-runtime/models/ornith15.gturbo OUT_DIR=$scratch/baselines MACHINE_TAG=mini bash $scratch/golden-baseline.sh $*"
status=$?
if [ "$mode" = capture ] && [ $status -eq 0 ]; then
  scp -q "macmini:$scratch/baselines/*.mini.txt" "$ROOT/baselines/" || status=1
fi
exit $status
```

      If production was not running when the wrapper starts, it still relaunches
      it at the end: the mini's intended state is production serving.

- [ ] **Step 2: Stop the deploy copying the golden.** Delete `tools/mini-deploy.sh:43-45`
      (the comment and the `scp` of `golden-baseline.sh`).

- [ ] **Step 3: Syntax.** `bash -n tools/mini-golden.sh`. Then capture every string
      the wrapper sends and parse it as the mini's zsh would: in a scratch
      directory `stubs/`, write `ssh` as

```bash
#!/bin/bash
n=$(ls "$STUB_LOG" 2>/dev/null | wc -l | tr -d ' ')
printf '%s' "${@: -1}" > "$STUB_LOG/remote-$n.zsh"
```

      and `scp` as `#!/bin/bash` then `exit 0`, both executable; run
      `env PATH="<scratch>/stubs:/usr/bin:/bin" STUB_LOG=<scratch>/log tools/mini-golden.sh --check`
      with `<scratch>/log` created first, and `zsh -n` each `remote-*.zsh`.
      Expected: five strings (the stop, `mktemp`, `mkdir`, the golden's run and the
      relaunch; the scratch directory's `rm` is skipped since the stubbed `mktemp`
      answered nothing) and no syntax errors.

- [ ] **Step 4: The owner's go-ahead for production's downtime** (about five minutes
      for the capture below, plus under a minute for the failure check). Nothing on
      the mini before it.

- [ ] **Step 5: The failure path on the real box.** `tools/mini-golden.sh --check nosuchprofile`.
      Expected: the golden prints `unknown profile: nosuchprofile`, the wrapper exits
      1, and `production relaunched:` with the ready line. Confirm with
      `ssh macmini 'pgrep -lf "bin/shrike serve"'`: production's command line on 8081.

- [ ] **Step 6: Capture the mini's server profiles.**
      `tools/mini-golden.sh serve-short serve-long serve-turns`. Expected: three
      `captured ->` lines on the mini and three new files in `baselines/`; the
      replay's and follow-up's text match the step-zero mini run's
      `mini/launch1/pass1/` files; the usage rows show the cached counts;
      `git status --short baselines/` lists exactly the three new files.

- [ ] **Step 7: Check all five on the mini.** `tools/mini-golden.sh --check`.
      Expected: five `ok` lines, production relaunched, `git status --short baselines/`
      unchanged by the check. `short` and `long` now run at `generate`'s default 64
      slots where their baselines were captured at 160 (the retired
      `CLI_EXTRA_ARGS`); v22 recorded the golden identical across one and two arena
      chunks (`v22-pool-capacity.md:107-108`), so a mismatch here is a finding:
      stop and report, never re-capture.

- [ ] **Step 8: Retire the mini's own copies.** Delete
      `baselines/ornith15-int4-{short,long,turns}-lh.mini.txt` (`git rm`). On the
      mini, confirm `~/shrike-runtime/baselines/*.txt` match the repo's `.mini.txt`
      files of the same names (`ssh macmini 'shasum ~/shrike-runtime/baselines/*.txt'`
      against `shasum baselines/*.mini.txt`), then remove
      `~/shrike-runtime/golden-baseline.sh` and `~/shrike-runtime/baselines/`.

- [ ] **Step 9: The docs.** In CLAUDE.md's "Verifying a change that touches
      inference", replace the sentences from "Baselines are stored" to "passes
      unseen." with:

```markdown
Baselines are stored in `baselines/`, tagged per machine: `short` and `long` check
`shrike generate`'s fused head, and `serve-short`, `serve-long` and `serve-turns`
check a fresh `shrike serve` at production's launch over HTTP (the retry and the
second turn resuming from the prompt cache); `--check` compares only against files
whose machine tag matches the box it runs on, so capture on the machine you intend
to check. For the mini, run `tools/mini-golden.sh --check` from the checkout: it
stops production, runs the golden there against the repo's `baselines/*.mini.txt`
and relaunches production, so it needs the owner's go-ahead like any downtime.
```

      In "The mini's layout", delete the clause "`baselines/` holds the mini's
      golden-baseline files". In README.md, replace the line "To measure your own:"
      and the code block after it (`tools/golden-baseline.sh --check 4`: the golden
      checks output, not throughput, and `4` is not a profile) with:

````markdown
To check a build against real inference on your machine, capture its baselines
once, then check later builds against them:

```bash
tools/golden-baseline.sh          # capture
tools/golden-baseline.sh --check
```
````

      Run the link check.

- [ ] **Step 10: The four gates**, then commit `tools/mini-golden.sh`,
      `tools/mini-deploy.sh`, CLAUDE.md, README.md, the three new and three deleted
      `.mini` baselines, and this plan with this task ticked:
      `tools: tools/mini-golden.sh runs the golden on the mini from the checkout (SHRIKE-57)`.

### Task 3: `--logits-head` and `--follow-up` leave `generate`

**Files:**
- Modify: `Sources/ShrikeCLI/ShrikeGenerateCommand.swift:112-117` (`logitsHead`),
  `:145-151` (`followUp`)
- Modify: `Sources/ShrikeCLI/Run.swift:105` (the head choice), `:141-147` (the
  follow-up call), `:158-199` (`runFollowUp` and its doc comment)
- Modify: `tests/Shrike/Core/CLI/CLIArgumentsTests.swift` (the help inventory, the
  dash test, a retired-flags test), `CLIArgumentsTests+Instrument.swift`,
  `CLIArgumentsTests+Invocations.swift`
- Modify: `docs/v25-argument-discipline.md` (line 81's marker and a Verdicts section)

**Interfaces:**
- Consumes: nothing from Tasks 1 and 2 in code; their golden is this task's check.
- Produces: `generate` without the two flags. `buildRuntime(... logitsHead:)` and
  the runtime's `forceLogitsHead` stay: sampling and the two dumps still select
  the logits head. `RawDecodeResult.kvBackedTokenIDs` and
  `uncommittedBoundaryTokenIDs` stay: the server reads them.

- [ ] **Step 1: Write the failing test.** In `CLIArgumentsTests.swift`, after
      `unsupportedSelectorsAreRejectedWithANonZeroExit`:

```swift
    @Test func theGoldensRetiredImitationFlagsAreRejected() throws {
        for extra in [["--logits-head"], ["--follow-up", "and a semaphore?"]] {
            let error = #expect(throws: (any Error).self) {
                _ = try ShrikeGenerateCommand.parse(["--model", "m.gturbo", "--prompt", "hi"] + extra)
            }
            #expect(ShrikeGenerateCommand.exitCode(for: try #require(error)) != .success)
        }
    }
```

- [ ] **Step 2: Run it.** `swift test --no-parallel --filter theGoldensRetiredImitationFlagsAreRejected`.
      Expected: FAIL, both invocations parse.

- [ ] **Step 3: Delete the flags.** In `ShrikeGenerateCommand.swift`, delete the
      `@Flag ... public var logitsHead = false` block and the
      `@Option(parsing: .unconditional ...) public var followUp: String?` block. In
      `Run.swift`, change

```swift
        let logitsHead = !config.isPureGreedy || args.logitsHead
            || dump != nil || hiddenDump != nil
```

      to

```swift
        let logitsHead = !config.isPureGreedy || dump != nil || hiddenDump != nil
```

      and delete the `if let followUp = args.followUp { ... }` block and the whole
      `runFollowUp` function with its `///` line.

- [ ] **Step 4: Update the pins that named them.** In `CLIArgumentsTests.swift`,
      remove `"--logits-head"` and `"--follow-up"` from
      `helpListsExactlyThePublicOptions`, and in `anOptionValueMayBeginWithADash`
      remove `"--follow-up", "-- and again"` from the argv and its `#expect`
      (the prompt and `--stop` still pin a leading dash). In
      `CLIArgumentsTests+Instrument.swift`, remove `"--logits-head"` from the argv and
      both `logitsHead` expectations. In `CLIArgumentsTests+Invocations.swift`,
      remove `#expect(!arguments.logitsHead)` from
      `goldenBaselineShortProfileParses`.

- [ ] **Step 5: Run the CLI tests.** `swift test --no-parallel --filter CLIArgumentsTests`.
      Expected: PASS, including the new test.

- [ ] **Step 6: Record the verdicts.** Append to `docs/v25-argument-discipline.md`:

```markdown
## Verdicts

Each flag judged on its own, per the test above; SHRIKE-4 adds the rest.

- **`--logits-head`, prosthetic, deleted.** It chose the logits head for a greedy
  run, which only the golden's imitation of the server wanted; the golden's
  server profiles now run the server's head itself, and `generate` still takes
  the logits head whenever it samples or dumps.
- **`--follow-up`, prosthetic, deleted.** It re-implemented the server's cached
  continuation, and the step-zero run measured that it is not the same
  computation: the same tokens in a different chunk split, a different second
  turn on both boxes. `serve-turns` checks the continuation production computes.
```

      and on line 81, after `**Retiring the three is filed in tt as SHRIKE-1 (2026-09-23).**`,
      append ` **Two are deleted; see [Verdicts](#verdicts).**` on the same line.

- [ ] **Step 7: The four gates.**

- [ ] **Step 8: Check on the dev box.** `swift build -c release`, the process checks,
      `tools/golden-baseline.sh --check`. Expected: five `ok` lines (the deletion
      changes no computation the golden runs).

- [ ] **Step 9: Deploy and check on the mini**, with the owner's go-ahead:
      `tools/mini-deploy.sh --restart`, then `tools/mini-golden.sh --check`.
      Expected: five `ok` lines and production relaunched on the new binary.

- [ ] **Step 10: Commit** the two sources, the three test files, the v25 doc and
      this plan with this task ticked:
      `cli: generate loses --logits-head and --follow-up, the golden's imitation flags (SHRIKE-57)`.
