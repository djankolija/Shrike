# v25 flags are not a substitute for infrastructure: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the golden checks what production runs by driving `shrike serve` at
production's launch, and `generate` loses the two flags that only let it imitate
the server (SHRIKE-57); one knob, `--ram-budget`, sizes the expert pool on both
commands (SHRIKE-60); and the kernel benches leave the product binary for a
development executable, pruned to the kernels production runs (SHRIKE-61).

**Architecture:** `tools/golden-baseline.sh` gains three server profiles that start
a fresh `shrike serve` from `tools/mini-production.sh`'s launch values on a spare
port and send the golden's prompts over HTTP; the CLI keeps `short` and `long` on
`generate`'s own fused head. A new `tools/mini-golden.sh` runs the golden on the
mini from the checkout, stopping and relaunching production around it, with the
repo's `baselines/*.mini.txt` as the only copy of the mini's baselines. Then
`--logits-head` and `--follow-up` leave `generate`. `--ram-budget` replaces
`--expert-cache-slots` on both commands through one parser in
`ShrikeArgumentSupport` and one slot derivation from the model's manifest in
`RuntimeConfiguration`, and `generate` prints the load line `serve` logs, so the
slot count a budget gives is visible. The benches move to a `shrike-bench`
executable built beside `shrike`; the deploy stops shipping their bundles, every
guard that looks for a model process learns the second name, and the expert bench
loses v21's coded arms.

**Tech Stack:** bash (the scripts; the mini's login shell is zsh, which runs the
remote strings), `curl` and `jq` (on both boxes at `/usr/bin`), Swift 6.3 with
swift-argument-parser and Swift Testing.

**Spec:** [v25-argument-discipline.md](v25-argument-discipline.md), its two step-zero
sections, the golden through the server (SHRIKE-2) and what the mini needs
(SHRIKE-3), and its Verdicts (SHRIKE-4).

Six commits, one per task. The checkboxes here are the status of record.

## Global Constraints

- The four gates before any task is called done: `swift build -c release` (zero
  warnings), `swiftlint lint --strict`, `python3 tools/check-md-links.py`,
  `swift test --no-parallel`. ThreadSanitizer once at the chapter's close, not here.
- Before anything that loads a model: `pgrep -lx 'shrike(-bench)?'; pgrep -fl
  'ShrikePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'`. Never terminate a
  process this session did not start.
- Production on the mini goes down only with the owner's go-ahead at that step;
  a go-ahead covers the step it was given for.
- A baseline is re-captured only for a deliberate numerics change. The `serve-*`
  files are first captures; `short` and `long` are never re-captured here, and a
  mismatch on them stops the task.
- No new `SHRIKE_*` variable. Production's launch stays written once, in
  `tools/mini-production.sh`.
- Stage by path, never `git add docs/` or `-a`, so nothing unreviewed rides along.
- Commit subjects in the repo's style, at most 100 characters, ending in the
  task's entry: `(SHRIKE-57)` for Tasks 1 to 3, `(SHRIKE-60)` for Task 4,
  `(SHRIKE-61)` for Tasks 5 and 6. No `Co-Authored-By`.
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
- A `shrike-bench` is running when the golden, a rig, the deploy or the mini's
  golden starts: each refuses as it does for a `shrike`, since `shrike-bench
  expert` maps a real `.gturbo`. Checked in Task 5, Steps 7 and 8.
- A typed `--ram-budget` that is not a size (`0`, `8X`) is refused at parse with the
  message `serve` gives, on both commands. Checked in Task 4, Step 1.
- Production's launch still resolves to 160 slots per layer from its budget in bytes,
  so its slot table still totals: a drift in the derivation fails a test, not a
  launch. Checked in Task 4, Step 1.
- The golden's `short` and `long` now run `generate` at 128 slots instead of 64: a
  pool size must not change a greedy answer, so any mismatch stops Task 4. Checked
  in Task 4, Step 9, and on the mini in Task 6, Step 10.
- The deploy ships no bench bundle, and a deploy removes the mini's old ones as
  retired. Checked in Task 5, Step 9, and Task 6, Step 10.

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

- [x] **Step 1: Write `tools/mini-golden.sh`:**

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

- [x] **Step 2: Stop the deploy copying the golden.** Delete `tools/mini-deploy.sh:43-45`
      (the comment and the `scp` of `golden-baseline.sh`).

- [x] **Step 3: Syntax.** `bash -n tools/mini-golden.sh`. Then capture every string
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

- [x] **Step 4: The owner's go-ahead for production's downtime** (about five minutes
      for the capture below, plus under a minute for the failure check). Nothing on
      the mini before it.

- [x] **Step 5: The failure path on the real box.** `tools/mini-golden.sh --check nosuchprofile`.
      Expected: the golden prints `unknown profile: nosuchprofile`, the wrapper exits
      1, and `production relaunched:` with the ready line. Confirm with
      `ssh macmini 'pgrep -lf "bin/shrike serve"'`: production's command line on 8081.

- [x] **Step 6: Capture the mini's server profiles.**
      `tools/mini-golden.sh serve-short serve-long serve-turns`. Expected: three
      `captured ->` lines on the mini and three new files in `baselines/`; the
      replay's and follow-up's text match the step-zero mini run's
      `mini/launch1/pass1/` files; the usage rows show the cached counts;
      `git status --short baselines/` lists exactly the three new files.

- [x] **Step 7: Check all five on the mini.** `tools/mini-golden.sh --check`.
      Expected: five `ok` lines, production relaunched, `git status --short baselines/`
      unchanged by the check. `short` and `long` now run at `generate`'s default 64
      slots where their baselines were captured at 160 (the retired
      `CLI_EXTRA_ARGS`); v22 recorded the golden identical across one and two arena
      chunks (`v22-pool-capacity.md:107-108`), so a mismatch here is a finding:
      stop and report, never re-capture.

- [x] **Step 8: Retire the mini's own copies.** Delete
      `baselines/ornith15-int4-{short,long,turns}-lh.mini.txt` (`git rm`). On the
      mini, confirm `~/shrike-runtime/baselines/*.txt` match the repo's `.mini.txt`
      files of the same names (`ssh macmini 'shasum ~/shrike-runtime/baselines/*.txt'`
      against `shasum baselines/*.mini.txt`), then remove
      `~/shrike-runtime/golden-baseline.sh` and `~/shrike-runtime/baselines/`.

- [x] **Step 9: The docs.** In CLAUDE.md's "Verifying a change that touches
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

      In "The mini's layout", replace the clause "`baselines/` holds the mini's
      golden-baseline files" with "the mini's golden baselines live in the repo,
      not on the box", on the same line. In README.md, replace the line "To measure your own:"
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

- [x] **Step 10: The four gates**, then commit `tools/mini-golden.sh`,
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

- [x] **Step 1: Write the failing test.** In `CLIArgumentsTests.swift`, after
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

- [x] **Step 2: Run it.** `swift test --no-parallel --filter theGoldensRetiredImitationFlagsAreRejected`.
      Expected: FAIL, both invocations parse.

- [x] **Step 3: Delete the flags.** In `ShrikeGenerateCommand.swift`, delete the
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

- [x] **Step 4: Update the pins that named them.** In `CLIArgumentsTests.swift`,
      remove `"--logits-head"` and `"--follow-up"` from
      `helpListsExactlyThePublicOptions`, and in `anOptionValueMayBeginWithADash`
      remove `"--follow-up", "-- and again"` from the argv and its `#expect`
      (the prompt and `--stop` still pin a leading dash). In
      `CLIArgumentsTests+Instrument.swift`, remove `"--logits-head"` from the argv and
      both `logitsHead` expectations. In `CLIArgumentsTests+Invocations.swift`,
      remove `#expect(!arguments.logitsHead)` from
      `goldenBaselineShortProfileParses`.

- [x] **Step 5: Run the CLI tests.** `swift test --no-parallel --filter CLIArgumentsTests`.
      Expected: PASS, including the new test.

- [x] **Step 6: Record the verdicts.** Append to `docs/v25-argument-discipline.md`:

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

- [x] **Step 7: The four gates.**

- [x] **Step 8: Check on the dev box.** `swift build -c release`, the process checks,
      `tools/golden-baseline.sh --check`. Expected: five `ok` lines (the deletion
      changes no computation the golden runs).

- [x] **Step 9: Deploy and check on the mini**, with the owner's go-ahead:
      `tools/mini-deploy.sh --restart`, then `tools/mini-golden.sh --check`.
      Expected: five `ok` lines and production relaunched on the new binary.

- [x] **Step 10: Commit** the two sources, the three test files, the v25 doc and
      this plan with this task ticked:
      `cli: generate loses --logits-head and --follow-up, the golden's imitation flags (SHRIKE-57)`.

### Task 4: one knob sizes the expert pool, `--ram-budget` on both commands

SHRIKE-60. `generate`'s `--expert-cache-slots` defaulted to 64, below the cliff the
default budget was measured against, and `serve`'s was an override of the budget
that nothing passed (the v25 doc's Verdicts). Both go; `generate` takes `serve`'s
`--ram-budget`, and the two dead slot fields on the server side go with them.

**Files:**
- Modify: `Sources/ShrikeArgumentSupport/ShrikeArgumentConformances.swift` (the
  shared `--ram-budget` parser and help)
- Modify: `Sources/Shrike/Runtime/Configuration/RuntimeConfiguration.swift:131-149`
  (slots for a model directory, after the stride form)
- Modify: `Sources/ShrikeCLI/ShrikeGenerateCommand.swift:90-95` (the option),
  `:110` (`--quiet`'s help), `:167-171` (the slot validation)
- Modify: `Sources/ShrikeCLI/Run.swift:200-203`, `:232-236` (`buildRuntime`)
- Modify: `Sources/ShrikeServer/Core/ShrikeServerCommand.swift:68-90`, `:111-117`,
  `:127`, `:131-141`
- Modify: `Sources/ShrikeServer/Core/ServerInference.swift:587-589`, `:670`,
  `:715-719`, `:762`, `:774-798`, `:905`, `:923`
- Modify: `Sources/ShrikeServer/Core/ModelSessionPlan.swift`,
  `Sources/ShrikeServer/Core/ModelRegistry.swift:386`
- Test: `tests/Shrike/Core/Runtime/Configuration/ExpertCacheBudgetTests.swift`,
  `tests/Shrike/Core/CLI/CLIArgumentsTests.swift`,
  `tests/ShrikeServer/ServerArgumentsTests.swift`, `tests/ShrikeRoot/RootCommandTests.swift`
- Modify: the five `expertCacheSlots: nil` call sites,
  `tests/ShrikeServer/ModelRegistryTests.swift:117`, `HTTPServerTests.swift:205` and
  `:569`, `OpenAIValidationTests.swift:472`, `HTTPServerMultiModelTests.swift:48`
- Modify: `README.md` (the two flags worth knowing), `docs/multi-model-serving.md:140-141`,
  `docs/v25-argument-discipline.md` (line 81's marker and the two slot verdicts)

**Interfaces:**
- Consumes: nothing from Tasks 1 to 3 in code; the golden is this task's check.
- Produces: `ExpertCacheBudgetArgument.help: ArgumentHelp` and
  `ExpertCacheBudgetArgument.bytes(_: String) throws -> Int` in
  `ShrikeArgumentSupport`; `RuntimeConfiguration.expertCacheSlots(modelDirectory: URL,
  expecting: ArchConfig, budgetBytes: Int?) -> Int`;
  `ShrikeGenerateCommand.expertCacheBudgetBytes: Int?`. `ShrikeServerCommand` keeps
  `expertCacheBudgetBytes: Int?` and loses `expertCacheSlots`; `ModelSessionPlan.init`
  and `ServerModelSession.load` lose their `expertCacheSlots:` parameter.

- [x] **Step 1: Write the failing tests.** In `ExpertCacheBudgetTests.swift`, add
      `import Foundation` under `import Testing`, and append to the suite:

```swift
    @Test func productionsBudgetIs160SlotsAtTheFourBitStride() {
        #expect(RuntimeConfiguration.expertCacheSlots(
            expertStrideBytes: Self.stride4, layers: Self.layers,
            budgetBytes: 11_324_620_800) == 160)
    }

    @Test func aModelDirectorysSlotsFollowItsManifestAndTheBudget() throws {
        let (dir, toy) = try ManifestReaderTests.writeToyManifest(
            ["expertStride": Int(Self.stride4)])
        defer { try? FileManager.default.removeItem(at: dir) }
        let perSlot = Int(Self.stride4) * toy.numLayers
        #expect(RuntimeConfiguration.expertCacheSlots(
            modelDirectory: dir, expecting: toy, budgetBytes: 160 * perSlot) == 160)
        #expect(RuntimeConfiguration.expertCacheSlots(
            modelDirectory: dir, expecting: toy, budgetBytes: nil)
            == RuntimeConfiguration.expertCacheSlots(
                expertStrideBytes: Self.stride4, layers: toy.numLayers))
    }

    @Test func anUnreadableManifestTakesTheSmallestCount() {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("no-model-\(UUID().uuidString)")
        #expect(RuntimeConfiguration.expertCacheSlots(
            modelDirectory: missing, expecting: .qwenToy(), budgetBytes: nil) == 8)
    }
```

      In `CLIArgumentsTests.swift`, in `helpListsExactlyThePublicOptions` replace
      `"--expert-cache-slots",` with `"--ram-budget",`, and after it add:

```swift
    @Test func theRAMBudgetAloneSizesTheExpertPool() throws {
        #expect(try ShrikeGenerateCommand.parse(["--prompt", "hi"]).expertCacheBudgetBytes == nil)
        #expect(try ShrikeGenerateCommand.parse(["--prompt", "hi", "--ram-budget", "2G"])
            .expertCacheBudgetBytes == 2 << 30)
        #expect(try ShrikeGenerateCommand.parse(["--prompt", "hi", "--ram-budget", "11324620800"])
            .expertCacheBudgetBytes == 11_324_620_800)
        for size in ["0", "8X"] {
            #expect(try rejection(["--prompt", "hi", "--ram-budget", size])
                .contains("--ram-budget must be a positive size such as 2G, 512M or a byte count"))
        }
        #expect(throws: (any Error).self) {
            _ = try ShrikeGenerateCommand.parse(["--prompt", "hi", "--expert-cache-slots", "160"])
        }
    }
```

      In `ServerArgumentsTests.swift`, delete `#expect(arguments.expertCacheSlots == nil)`
      from `productionLaunchLineParses`, and append after the suite:

```swift
@Suite struct ServerPoolArgumentTests {
    @Test func theRAMBudgetIsTheOnlyPoolKnob() throws {
        #expect(throws: (any Error).self) {
            _ = try ShrikeServerCommand.parse(["--model", "m.gturbo", "--expert-cache-slots", "160"])
        }
        let error = #expect(throws: (any Error).self) {
            _ = try ShrikeServerCommand.parse(["--model", "m.gturbo", "--ram-budget", "8X"])
        }
        #expect(ShrikeServerCommand.message(for: try #require(error))
            .contains("--ram-budget must be a positive size such as 2G, 512M or a byte count"))
    }
}
```

      In `RootCommandTests.swift`, in `serveKeepsEveryFlagItSharesWithGenerate` replace
      `"--expert-cache-slots", "160",` with `"--ram-budget", "11324620800",` and
      `#expect(serve.expertCacheSlots == 160)` with
      `#expect(serve.expertCacheBudgetBytes == 11_324_620_800)`.

- [x] **Step 2: Run them.** `swift test --no-parallel --filter 'ExpertCacheBudgetTests|CLIArgumentsTests|ServerPoolArgumentTests'`.
      Expected: the build fails, `RuntimeConfiguration` has no
      `expertCacheSlots(modelDirectory:expecting:budgetBytes:)` and
      `ShrikeGenerateCommand` has no `expertCacheBudgetBytes`.

- [x] **Step 3: The shared parser.** Append to `ShrikeArgumentConformances.swift`:

```swift
public enum ExpertCacheBudgetArgument {
    public static var help: ArgumentHelp {
        ArgumentHelp("""
            Bytes the routed-expert cache may use, e.g. 8G, 2G, 512M. Slots are \
            derived from this and the model's expert stride, so this is the knob \
            and the slot count is the result. Default 8G, which holds the \
            measured routing working set; smaller budgets are markedly slower \
            because expert reads bypass the page cache and have no fallback.
            """,
            valueName: "size")
    }

    public static func bytes(_ value: String) throws -> Int {
        guard let parsed = RuntimeConfiguration.parseBudgetBytes(value) else {
            throw ValidationError(
                "--ram-budget must be a positive size such as 2G, 512M or a byte count")
        }
        return parsed
    }
}
```

      The help names no other flag: `helpListsExactlyThePublicOptions` collects every
      `--word` in `generate`'s help.

- [x] **Step 4: Slots for a model directory.** In `RuntimeConfiguration.swift`, after
      `expertCacheSlots(expertStrideBytes:layers:budgetBytes:)`, add:

```swift
    /// An unreadable manifest takes the smallest count: the load that follows
    /// fails with a better message than this could give.
    public static func expertCacheSlots(modelDirectory: URL,
                                        expecting arch: ArchConfig,
                                        budgetBytes: Int?) -> Int {
        guard let manifest = try? ManifestReader.load(directoryURL: modelDirectory,
                                                      expecting: arch) else {
            return allowedExpertCacheSlots.first ?? 8
        }
        return expertCacheSlots(expertStrideBytes: manifest.expertStride,
                                layers: manifest.arch.numLayers,
                                budgetBytes: budgetBytes ?? defaultExpertCacheBudgetBytes)
    }
```

- [x] **Step 5: `generate` takes the budget.** In `ShrikeGenerateCommand.swift`, replace
      the `@Option(help: ... Routed-expert cache slots per layer ...) public var
      expertCacheSlots = 64` block with:

```swift
    @Option(name: .customLong("ram-budget"),
            help: ExpertCacheBudgetArgument.help,
            transform: ExpertCacheBudgetArgument.bytes)
    public var expertCacheBudgetBytes: Int?
```

      delete the `guard RuntimeConfiguration.allowedExpertCacheSlots.contains(expertCacheSlots)`
      block from `validate()`, and change `--quiet`'s help to
      `"Suppress the load line and the timing footer."`. In `Run.swift`'s
      `buildRuntime`, replace `expertCacheSlots: args.expertCacheSlots,` in the first
      `RuntimeConfiguration(...)` with:

```swift
        expertCacheSlots: RuntimeConfiguration.expertCacheSlots(
            modelDirectory: modelURL, expecting: expectedArch,
            budgetBytes: args.expertCacheBudgetBytes),
```

      and after `let runner = try RealForwardRunner(...)` add:

```swift
    if !args.quiet {
        stderr.write(Data("\(runner.prefillDescription)\n".utf8))
    }
```

      It is the line `serve` logs at load (`ServerInference.swift:770`), ending
      `expert_slots=<count> policy=<policy>`: the budget is the knob, and this is where
      a user sees the slot count it gave.

- [x] **Step 6: `serve` loses its slot flag.** In `ShrikeServerCommand.swift`: delete
      the `@Option(name: .customLong("expert-cache-slots") ...) public var
      expertCacheSlots: Int?` block; replace the `--ram-budget` block, its `///`
      comment included, with:

```swift
    @Option(name: .customLong("ram-budget"),
            help: ExpertCacheBudgetArgument.help,
            transform: ExpertCacheBudgetArgument.bytes)
    public var expertCacheBudgetBytes: Int?
```

      delete `static func budgetBytes(_:)`; in `validate()` replace
      `try validateOptionalMemberships()` with:

```swift
        if let configPath, configPath.isEmpty {
            throw ValidationError("--config must not be empty")
        }
```

      and delete `validateOptionalMemberships()`. In `ServerInference.swift`: delete
      the session's `expertCacheSlots` property with its two `///` lines (`:587-589`),
      the `expertCacheSlots: loadSlots,` argument to `ServerModelSession(...)`
      (`:762`), the `expertCacheSlots: Int,` parameter of its `init` (`:905`) and
      `self.expertCacheSlots = expertCacheSlots` (`:923`), since nothing reads it; in
      `load(...)` delete the `expertCacheSlots requestedExpertCacheSlots: Int? = nil,`
      parameter and replace the `let loadSlots = resolveExpertCacheSlots(...)` call with:

```swift
        let loadSlots = RuntimeConfiguration.expertCacheSlots(
            modelDirectory: modelDirectory, expecting: expectedArch,
            budgetBytes: expertCacheBudgetBytes)
```

      then delete `resolveExpertCacheSlots` with its three `//` lines (`:774-798`). In
      `ModelSessionPlan.swift`: delete `ModelSessionFacts.expertCacheSlots` with its
      three `///` lines, its `expertCacheSlots: Int = 0` init parameter (the parameter
      before it now closes the list) and its assignment; delete
      `ModelSessionPlan.expertCacheSlots`, its `expertCacheSlots: Int?,` init parameter,
      its assignment, and `expertCacheSlots: expertCacheSlots,` in `makeSession`. In
      `ModelRegistry.swift` delete `expertCacheSlots: arguments.expertCacheSlots,`. At
      each of the five test call sites, delete the `expertCacheSlots: nil` argument and
      the comma before it.

- [x] **Step 7: Run the tests.** `swift test --no-parallel --filter 'ExpertCacheBudgetTests|CLIArgumentsTests|ServerPoolArgumentTests|ServerInvocationTests|RootCommandTests|ModelRegistryTests|HTTPServer|OpenAIValidationTests'`.
      Expected: PASS. If the two rejection messages do not contain the text, print
      one with `ShrikeGenerateCommand.message(for:)` and stop: ArgumentParser wraps a
      transform's error, and the test must pin the message users see.

- [x] **Step 8: The four gates.**

- [x] **Step 9: Check on the dev box.** The process checks, `swift build -c release`, then:

```bash
.build/release/shrike --model /Volumes/BuildSSD/shrike/ornith15.gturbo --prompt hi --max-new 4 2>&1 >/dev/null | grep -o 'expert_slots=[^ ]*'
.build/release/shrike --model /Volumes/BuildSSD/shrike/ornith15.gturbo --prompt hi --max-new 4 --ram-budget 11324620800 2>&1 >/dev/null | grep -o 'expert_slots=[^ ]*'
```

      Expected: `expert_slots=uniform:128`, then `expert_slots=uniform:160`. Then
      `tools/golden-baseline.sh --check`. Expected: five `ok` lines; `short` and `long`
      now run at 128 slots where they ran at 64, and a pool size changes no greedy
      answer, so a mismatch stops the task.

- [x] **Step 10: The documents.** In `README.md`, change `The two worth knowing first:`
      to ``The two worth knowing first, which `shrike generate` takes too:``. In
      `docs/multi-model-serving.md`, lines 140-141 become
      ``larger than the machine's RAM on its own, and residency is bounded by `--ram-budget` ``
      and `rather than by bundle size.`, keeping the line count. In
      `docs/v25-argument-discipline.md`, on line 81 change
      `**Two are deleted; see [Verdicts](#verdicts).**` to
      `**All three are deleted; see [Verdicts](#verdicts).**`, and change both
      `` **`--expert-cache-slots`, prosthetic, to be deleted.** `` to
      `` **`--expert-cache-slots`, prosthetic, deleted.** ``.

- [x] **Step 11: Commit** the sources, the tests, the three documents and this plan
      with this task ticked:
      `cli+server: one knob sizes the expert pool, --ram-budget on generate and serve (SHRIKE-60)`.

### Task 5: the benches leave `shrike` for `shrike-bench`

SHRIKE-61, the move. `bench` was a verb of the product binary only because the mini
received one binary (the v25 doc's Verdicts). It becomes a second executable built
beside `shrike` and never deployed with it.

**Files:**
- Create: `Sources/ShrikeBench/Core/ShrikeBenchCommand.swift`,
  `Sources/ShrikeBench/Command/ShrikeBenchMain.swift`
- Modify: `Package.swift` (a product, two targets, `ShrikeRootCore`'s and
  `ShrikeBenchTests`' dependencies)
- Modify: `Sources/ShrikeRoot/Core/ShrikeRootCommand.swift` (`BenchCommand` goes)
- Test: `tests/ShrikeRoot/RootCommandTests.swift`, `tests/ShrikeBench/BenchArgumentTests.swift`
- Modify: the guards, `tools/golden-baseline.sh:94`, `tools/decode-rig.sh:62`,
  `tools/turn-rig.sh:72`, `tools/mini-golden.sh:37` and `:56`, `tools/mini-deploy.sh:34`,
  `CLAUDE.md:17`
- Modify: `tools/mini-deploy.sh:3-4`, `:43-47` (no bench bundle ships)
- Modify: `README.md:23-24`, `docs/architecture.md:642-651`, `CLAUDE.md:108-110`

**Interfaces:**
- Consumes: nothing from Task 4.
- Produces: `ShrikeBenchCommand` (`commandName: "shrike-bench"`, subcommands
  `AttnBenchCommand` and `ExpertBenchCommand`) in the `ShrikeBenchCore` module; the
  `shrike-bench` executable product; the guard pattern `pgrep -x 'shrike(-bench)?'`.
  Task 6 edits the two bench modules, not this wiring.

- [x] **Step 1: Write the failing tests.** In `RootCommandTests.swift`, delete
      `import ShrikeAttnBenchCore` and `import ShrikeExpertBenchCore`, and replace
      `benchResolvesEitherChild` with:

```swift
    @Test func benchIsNotAVerbOfTheProductBinary() {
        #expect(throws: (any Error).self) {
            _ = try parse(["bench", "attention"])
        }
        #expect(!ShrikeRootCommand.helpMessage().contains("bench"))
    }
```

      In `BenchArgumentTests.swift`, add `import ShrikeBenchCore` after the two
      `@testable` imports, and append:

```swift
@Suite struct ShrikeBenchCommandTests {
    @Test func theBenchBinaryResolvesEitherChild() throws {
        #expect(try ShrikeBenchCommand.parseAsRoot(["attention"]) is AttnBenchCommand)
        #expect(try ShrikeBenchCommand.parseAsRoot(["expert", "--model", "m.gturbo"])
            is ExpertBenchCommand)
    }
}
```

- [x] **Step 2: Run them.** `swift test --no-parallel --filter 'RootCommandTests|ShrikeBenchCommandTests'`.
      Expected: the build fails, no module `ShrikeBenchCore`.

- [x] **Step 3: The bench executable.** Create
      `Sources/ShrikeBench/Core/ShrikeBenchCommand.swift`:

```swift
import ArgumentParser
import ShrikeAttnBenchCore
import ShrikeExpertBenchCore

public struct ShrikeBenchCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "shrike-bench",
        abstract: "Kernel benchmarks, a development tool built beside shrike.",
        subcommands: [AttnBenchCommand.self, ExpertBenchCommand.self])

    public init() {}
}
```

      and `Sources/ShrikeBench/Command/ShrikeBenchMain.swift`:

```swift
import ShrikeBenchCore

@main extension ShrikeBenchCommand {}
```

- [x] **Step 4: The package.** In `Package.swift`, add
      `.executable(name: "shrike-bench", targets: ["ShrikeBench"]),` after the `shrike`
      product; remove `"ShrikeAttnBenchCore",` and `"ShrikeExpertBenchCore",` from
      `ShrikeRootCore`'s dependencies; add `"ShrikeBenchCore",` to `ShrikeBenchTests`'
      dependencies; and after the `ShrikeRoot` executable target add:

```swift
        .target(
            name: "ShrikeBenchCore",
            dependencies: [
                "ShrikeAttnBenchCore",
                "ShrikeExpertBenchCore",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "sources/ShrikeBench/Core"
        ),
        .executableTarget(
            name: "ShrikeBench",
            dependencies: ["ShrikeBenchCore"],
            path: "sources/ShrikeBench/Command"
        ),
```

- [x] **Step 5: The root drops the verb.** In `ShrikeRootCommand.swift`, delete
      `import ShrikeAttnBenchCore`, `import ShrikeExpertBenchCore`, `BenchCommand.self,`
      from the subcommands, and the `BenchCommand` struct.

- [x] **Step 6: Run the tests.** `swift test --no-parallel --filter 'RootCommandTests|ShrikeBench'`.
      Expected: PASS.

- [x] **Step 7: Every guard learns the second name.** `shrike-bench expert` maps a real
      `.gturbo`, so it is a model process, and `pgrep -x shrike` matches the name
      exactly. Replace `pgrep -x shrike` with `pgrep -x 'shrike(-bench)?'` at
      `tools/golden-baseline.sh:94`, `tools/decode-rig.sh:62`, `tools/turn-rig.sh:72`,
      `tools/mini-golden.sh:37` and `:56` (inside `\"\$(...)\"`), and
      `tools/mini-deploy.sh:34`; on `CLAUDE.md:17` replace `pgrep -lx shrike;` with
      `pgrep -lx 'shrike(-bench)?';`. Then `rg -n "pgrep -l?x shrike" tools CLAUDE.md`
      (expected: no match) and `bash -n` on each of the five scripts.

- [x] **Step 8: Check the guards with stand-ins.** Write `/tmp/v25-guards.sh`:

```bash
#!/bin/bash
set -u
dir=/tmp/v25-guards
mkdir -p "$dir"
printf '#include <unistd.h>\nint main(void) { sleep(60); return 0; }\n' > "$dir/stub.c"
clang -o "$dir/shrike-bench" "$dir/stub.c"
clang -o "$dir/shrike-benchx" "$dir/stub.c"
"$dir/shrike-bench" & bench=$!
"$dir/shrike-benchx" & other=$!
sleep 1
echo "the pattern sees: $(pgrep -lx 'shrike(-bench)?' | tr '\n' ' ')"
tools/golden-baseline.sh --check; echo "golden exit $?"
kill "$bench" "$other"
```

      and run `bash /tmp/v25-guards.sh` from the repo root. Expected: the pattern sees
      the `shrike-bench` pid and not `shrike-benchx` (a copied system binary is killed
      at launch, hence compiled stand-ins), and the golden prints `a model process is
      already running` with `golden exit 3`. The mini-side guards run the same pattern
      through the same `pgrep`.

- [x] **Step 9: The deploy ships no bench bundle.** In `tools/mini-deploy.sh`, after
      `name=$(basename "$bundle")` in the bundle loop, add:

```bash
  case "$name" in *BenchCore.bundle) continue ;; esac
```

      and in the header change `Copies the release binary and its resource bundles`
      to `Copies the release binary and its resource bundles, not shrike-bench's,`.
      Then `swift build -c release` and write two stubs, `/tmp/v25-deploy-stubs/ssh` and
      `/tmp/v25-deploy-stubs/scp`, each:

```bash
#!/bin/bash
echo "$(basename "$0") $*" >> /tmp/v25-deploy-stubs/log
```

      `chmod +x` both, and run
      `env PATH="/tmp/v25-deploy-stubs:/usr/bin:/bin" bash tools/mini-deploy.sh`, then
      `grep -c BenchCore.bundle /tmp/v25-deploy-stubs/log` (expected: 0) and
      `grep -c Shrike_Shrike.bundle /tmp/v25-deploy-stubs/log` (expected: at least 1).
      `.build/release/` holds the bench bundles by now, so the first count is a real
      exclusion, not an absence.

- [x] **Step 10: The documents.** `README.md:23-24` become:

```markdown
`.build/release/` holds `shrike`, which generates once given a prompt, and `shrike-bench`,
the kernel benches, which the deploy does not ship; `shrike serve` and `shrike repack` are the other verbs.
```

      In `docs/architecture.md`, the attention bullet's five lines at `:642-646` become:

```markdown
- `shrike-bench attention`: the decode attention scan on synthetic rows at the served
  shape, the production pipeline through the wrapper, the shipped kernel's copy with one
  switch per function constant, and the streaming prototype. A development executable
  built beside `shrike` and never deployed with it; a run on the mini copies it and every
  `.bundle` from `.build/release/` into a scratch directory there (v25).
```

      and the expert bullet's first line at `:647` begins `` - `shrike-bench expert`: ``
      in place of `` - `shrike bench expert`: ``.

      In `CLAUDE.md:108-110`, change
      ``(a deploy copies the `*.bundle` directories from `.build/release/` alongside the binaries, or resource lookups fail at runtime)``
      to
      ``(a deploy copies `shrike`'s `*.bundle` directories from `.build/release/`, never `shrike-bench`'s, or resource lookups fail at runtime)``,
      keeping the line count.

- [x] **Step 11: The four gates.**

- [x] **Step 12: Commit** `Package.swift`, the two new sources, the root command, the
      two test files, the five scripts, `README.md`, `CLAUDE.md`, `docs/architecture.md`
      and this plan with this task ticked:
      `bench: the kernel benches leave shrike for a shrike-bench executable (SHRIKE-61)`.

### Task 6: the benches measure what production runs

SHRIKE-61, the pruning. The attention bench's default ladder leaves out the runner's
kernel and calls a pre-streaming copy the shipped one; the expert bench carries v21's
coded arms, closed at their first measurement, and a `--experts` whose help is wrong
(the v25 doc's Verdicts). With the coded arms gone `plain` is the only arm, so
`--arms` goes with them and the bench measures the production kernel alone.

**Files:**
- Modify: `Sources/ShrikeAttnBench/Arms.swift:52-55`, `:61`
- Modify: `Sources/ShrikeExpertBench/ExpertBenchCommand.swift` (whole file below),
  `Kernels.swift` (whole file below), `Runner.swift` (whole file below),
  `BenchError.swift` (whole file below)
- Delete: `Sources/ShrikeExpertBench/Coder.swift`, `Sources/ShrikeExpertBench/Metal/expert.metal`
- Modify: `Package.swift` (`ShrikeExpertBenchCore` loses its resources)
- Test: `tests/ShrikeBench/BenchArgumentTests.swift`
- Modify: `docs/architecture.md:642-651`

**Interfaces:**
- Consumes: Task 5's `shrike-bench` executable (the dev-box run) and deploy exclusion
  (the mini's retired bundles).
- Produces: `ExpertBenchCommand` without `arms`; `PlainOffsets` in `Kernels.swift`;
  `ExpertKernels` with `production` and `encodeProduction` only.

- [ ] **Step 1: Write the failing tests.** In `BenchArgumentTests.swift`, add to
      `AttnBenchArgumentTests`:

```swift
    @Test func theDefaultLadderMeasuresTheRunnersKernel() throws {
        #expect(try AttnBenchCommand.parse([]).arms.names.contains("prodstream"))
    }
```

      In `ExpertBenchArgumentTests`: delete `#expect(bench.arms.names == ["plain", "coded", "coded+aux"])`
      from `modelIsRequiredAndTheRestDefault`; in `everyOptionParses` delete
      `"--arms", "plain,coded",` from the argv and `#expect(bench.arms.names == ["plain", "coded"])`;
      and replace `anUnknownArmIsRejectedBeforeTheModelIsRead` with:

```swift
    @Test func armsIsNotAnOptionOnceProductionIsTheOnlyArm() {
        #expect(throws: (any Error).self) {
            _ = try ExpertBenchCommand.parse(["--model", "/m.gturbo", "--arms", "plain"])
        }
    }
```

- [ ] **Step 2: Run them.** `swift test --no-parallel --filter 'AttnBenchArgumentTests|ExpertBenchArgumentTests'`.
      Expected: FAIL, the ladder lacks `prodstream` and `--arms plain` parses.

- [ ] **Step 3: The attention bench.** In `Arms.swift`, the default ladder becomes:

```swift
    static let defaultLadder = [
        "prodstream", "prod", "copy", "qregs", "block8", "dbuf", "load8", "load16",
        "qregs+dbuf+load8", "nosoftmax", "nov", "loadonly", "loadonly+fullrow",
    ]
```

      and `copy`'s help becomes
      `"the ladder kernel with every switch at its default (the kernel shipped before the streaming scan)"`.

- [ ] **Step 4: The expert bench's command.** Replace `ExpertBenchCommand.swift` with:

```swift
import ArgumentParser
import Foundation
import ShrikeArgumentSupport

public struct ExpertBenchCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "expert",
        abstract: "The production decode phase-1 gate/up kernel over real experts.")

    @Option(help: ArgumentHelp("The .gturbo directory.", valueName: "dir"))
    public var model: String

    @Option(help: ArgumentHelp("The layer whose experts are read.", valueName: "n"))
    public var layer = 20

    @Option(help: ArgumentHelp("""
        Distinct experts read, 1 to 8. A pass always runs the routed top-k of eight, \
        repeating the last expert read.
        """,
        valueName: "n"))
    public var experts = 8

    @Option(help: ArgumentHelp("Timed command buffers; the median is reported.",
                               valueName: "n"))
    public var repeats = 15

    @Option(help: ArgumentHelp("Untimed command buffers before the repeats.", valueName: "n"))
    public var warmup = 3

    @Option(help: ArgumentHelp("""
        Dispatches per command buffer, so the GPU holds its clock; the time \
        reported is per dispatch.
        """,
        valueName: "n"))
    public var batch = 20

    @Option(help: ArgumentHelp("The activation vector's seed.", valueName: "n"))
    public var seed = BenchSeed(0x5EED_0021)

    public init() {}

    public func validate() throws {
        guard !model.isEmpty else {
            throw ValidationError("--model must not be empty")
        }
        guard layer >= 0 else {
            throw ValidationError("--layer must be zero or more")
        }
        guard (1...8).contains(experts) else {
            throw ValidationError("--experts is 1 to 8")
        }
        guard repeats > 0 else {
            throw ValidationError("--repeats needs a positive count")
        }
        guard warmup >= 0 else {
            throw ValidationError("--warmup needs a count of zero or more")
        }
        guard batch > 0 else {
            throw ValidationError("--batch needs a positive count")
        }
    }

    public func run() throws {
        try BenchRunner(args: self).run()
    }
}
```

- [ ] **Step 5: The production kernel alone.** Delete `Coder.swift` and
      `Metal/expert.metal`, remove `resources: [.copy("Metal")]` from
      `ShrikeExpertBenchCore` in `Package.swift`, and replace `Kernels.swift` with:

```swift
import Foundation
import Metal
import Shrike

final class ExpertKernels {
    static let hidden: UInt32 = 2048
    static let intermediate: UInt32 = 512
    static let topK: UInt32 = 8
    static let rowsPerThreadgroup = 16

    let context: MetalContext
    let production: MTLComputePipelineState
    let ioReady: MTLBuffer

    init(context: MetalContext) throws {
        self.context = context
        self.production = try context.pipeline("moe_phase1_gate_up_act_u16load", constants: [
            MetalFunctionConstant(index: 0, value: .uint32(Self.hidden)),
            MetalFunctionConstant(index: 1, value: .uint32(Self.intermediate)),
            MetalFunctionConstant(index: 2, value: .uint32(Self.topK)),
            MetalFunctionConstant(index: 3, value: .bool(true)),
            MetalFunctionConstant(index: 4, value: .bool(true)),
            MetalFunctionConstant(index: 6, value: .bool(true)),
        ])
        guard let ready = context.device.makeBuffer(length: 16, options: .storageModeShared) else {
            throw BenchError.allocation("io status")
        }
        ready.contents().storeBytes(of: UInt32(1), as: UInt32.self)
        self.ioReady = ready
    }

    static func argumentBuffer(device: MTLDevice, blobs: [MTLBuffer]) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: 8 * 8, options: .storageModeShared) else {
            throw BenchError.allocation("argument buffer")
        }
        let p = buffer.contents().assumingMemoryBound(to: UInt64.self)
        for i in 0..<8 { p[i] = blobs[min(i, blobs.count - 1)].gpuAddress }
        return buffer
    }

    func encodeProduction(_ cb: MTLCommandBuffer, arguments: MTLBuffer, blobs: [MTLBuffer],
                          offsets: PlainOffsets, x: MTLBuffer, acts: MTLBuffer) throws {
        guard let encoder = cb.makeComputeCommandEncoder() else { throw BenchError.encoder }
        defer { encoder.endEncoding() }
        encoder.setComputePipelineState(production)
        var d = Self.hidden, f = Self.intermediate, k = Self.topK
        encoder.setBuffer(arguments, offset: 0, index: 0)
        for blob in blobs { encoder.useResource(blob, usage: .read) }
        var o = offsets
        encoder.setBytes(&o, length: MemoryLayout<PlainOffsets>.stride, index: 1)
        encoder.setBuffer(x, offset: 0, index: 2)
        encoder.setBuffer(acts, offset: 0, index: 3)
        encoder.setBytes(&d, length: 4, index: 4)
        encoder.setBytes(&f, length: 4, index: 5)
        encoder.setBytes(&k, length: 4, index: 6)
        encoder.setBuffer(ioReady, offset: 0, index: 7)
        let rows = Int(Self.topK * Self.intermediate)
        encoder.dispatchThreadgroups(
            MTLSize(width: (rows + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
    }
}

/// Mirrors `ExpertOffsets` in moe.metal field for field.
struct PlainOffsets {
    var gateW: UInt32 = 0
    var gateS: UInt32 = 0
    var gateB: UInt32 = 0
    var upW: UInt32 = 0
    var upS: UInt32 = 0
    var upB: UInt32 = 0
    var downW: UInt32 = 0
    var downS: UInt32 = 0
    var downB: UInt32 = 0
    var gateAB: UInt32 = 0
    var upAB: UInt32 = 0
    var downAB: UInt32 = 0
}
```

      and `BenchError.swift` with:

```swift
enum BenchError: Error, CustomStringConvertible {
    case model(String)
    case allocation(String)
    case commandBuffer
    case encoder
    case gpu(String)

    var description: String {
        switch self {
        case .model(let text): return "model: \(text)"
        case .allocation(let what): return "allocation failed: \(what)"
        case .commandBuffer: return "command buffer creation failed"
        case .encoder: return "compute encoder creation failed"
        case .gpu(let text): return "GPU error: \(text)"
        }
    }
}
```

      `bind` and `dispatch` fold into `encodeProduction`, their only caller now.

- [ ] **Step 6: The runner.** Replace `Runner.swift` with:

```swift
import Foundation
import Metal
import Shrike

final class BenchRunner {
    private let args: ExpertBenchCommand
    private let context: MetalContext
    private let kernels: ExpertKernels
    private let stride: Int
    private let expertCount: Int
    private let blobs: [MTLBuffer]
    private let offsets: PlainOffsets
    private let arguments: MTLBuffer
    private let phase1Bytes: Int
    private let x: MTLBuffer
    private let acts: MTLBuffer

    init(args: ExpertBenchCommand) throws {
        self.args = args
        self.context = try MetalContext()
        self.kernels = try ExpertKernels(context: context)
        let loaded = try Experts.load(model: args.model, layer: args.layer, count: args.experts)
        self.stride = loaded.stride
        self.expertCount = loaded.experts.count
        let device = context.device

        func buffer(_ bytes: [UInt8], label: String) throws -> MTLBuffer {
            guard let b = device.makeBuffer(bytes: bytes, length: bytes.count, options: .storageModeShared) else {
                throw BenchError.allocation(label)
            }
            b.label = label
            return b
        }

        self.blobs = try loaded.experts.map { try buffer($0.bytes, label: "expert \($0.index)") }
        let first = loaded.experts[0]
        func off(_ name: String) throws -> UInt32 { UInt32(try first.tensor(name).offset) }
        self.offsets = PlainOffsets(
            gateW: try off("gate"), gateS: try off("gate_scales"), gateB: try off("gate_biases"),
            upW: try off("up"), upS: try off("up_scales"), upB: try off("up_biases"),
            downW: try off("down"), downS: try off("down_scales"), downB: try off("down_biases"))
        self.arguments = try ExpertKernels.argumentBuffer(device: device, blobs: blobs)
        self.phase1Bytes = try ["gate", "gate_scales", "gate_biases", "up", "up_scales", "up_biases"]
            .map { try first.tensor($0).size }.reduce(0, +)

        let d = Int(ExpertKernels.hidden)
        var state = args.seed.value
        var halves = [UInt16](repeating: 0, count: d)
        for i in 0..<d {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let unit = Float(Double(state >> 11) / Double(1 << 53))
            halves[i] = Float16(unit * 2 - 1).bitPattern
        }
        self.x = try buffer(halves.withUnsafeBufferPointer { Array(UnsafeRawBufferPointer($0)) }, label: "x")
        let actCount = Int(ExpertKernels.topK * ExpertKernels.intermediate)
        guard let acts = device.makeBuffer(length: actCount * 2, options: .storageModeShared) else {
            throw BenchError.allocation("acts")
        }
        self.acts = acts
    }

    func run() throws {
        print("device \(context.device.name); layer \(args.layer), \(expertCount) experts of \(stride) bytes; "
              + "repeats \(args.repeats), warmup \(args.warmup), \(args.batch) dispatches per command buffer")
        print(String(format: "%14@ %10@ %8@", "phase1 B/expert", "gpu_us", "GB/s"))
        _ = try Timing.medianGPUSeconds(context: context, warmup: 0, repeats: 20) { cb in
            try self.encode(cb)
        }
        let batch = args.batch
        let seconds = try Timing.medianGPUSeconds(context: context, warmup: args.warmup,
                                                  repeats: args.repeats) { cb in
            for _ in 0..<batch { try self.encode(cb) }
        } / Double(batch)
        print(String(format: "%14d %10.1f %8.2f", phase1Bytes, seconds * 1e6,
                     Double(phase1Bytes * expertCount) / seconds / 1e9))
    }

    private func encode(_ cb: MTLCommandBuffer) throws {
        try kernels.encodeProduction(cb, arguments: arguments, blobs: blobs,
                                     offsets: offsets, x: x, acts: acts)
    }
}
```

      The first timing call is the old `spinUp`, twenty untimed buffers that bring the
      GPU to its clock. The reference buffer and the per-arm comparison go: they held a
      variant to bit-identity against `plain`, and no variant is left.

- [ ] **Step 7: Run the bench tests.** `swift test --no-parallel --filter ShrikeBench`.
      Expected: PASS.

- [ ] **Step 8: The four gates.**

- [ ] **Step 9: Run both benches on the dev box.** The process checks, `swift build -c release`,
      then `.build/release/shrike-bench attention --positions 1024 --repeats 3 --warmup 1`
      (expected: a row per default arm, `prodstream` first) and
      `.build/release/shrike-bench expert --model /Volumes/BuildSSD/shrike/ornith15.gturbo`
      (expected: the device line, then one row of `phase1 B/expert`, `gpu_us` and `GB/s`).
      The expert run maps a real `.gturbo`, so it is a model run under the process rules.

- [ ] **Step 10: Deploy and check on the mini**, with the owner's go-ahead (about five
      minutes of production downtime): `tools/mini-deploy.sh --restart`, then
      `tools/mini-golden.sh --check`. Expected: the deploy prints `removed retired
      Shrike_ShrikeAttnBenchCore.bundle` and `removed retired
      Shrike_ShrikeExpertBenchCore.bundle` and relaunches production; the golden prints
      five `ok` lines, `short` and `long` at `generate`'s new 128 slots, and relaunches
      production.

- [ ] **Step 11: The documents.** In `docs/architecture.md`, the attention bullet's five
      lines at `:642-646` become:

```markdown
- `shrike-bench attention`: the decode attention scan on synthetic rows at the served
  shape, the production pipeline through the wrapper and on the streaming variant, the
  pre-streaming kernel's copy with one switch per function constant, and the streaming
  prototype. A development executable built beside `shrike` and never deployed with it;
  a run on the mini copies it and every `.bundle` from `.build/release/` there (v25).
```

      and the expert bullet's five lines at `:647-651` become:

```markdown
- `shrike-bench expert`: the production decode phase-1 gate/up kernel on up to eight
  real experts of a layer read from the `.gturbo`, a pass always the routed top-k of
  eight, timed with the GPU kept busy by a batch of dispatches per command buffer. It
  maps a real `.gturbo`, a model run. v21's coded arms, which closed the lossless-
  compression avenue on their numbers, are in git history (v25).
```

      Check that `docs/architecture.md:757` still reads as `v9-implementation-plan.md:133`
      cites it.

- [ ] **Step 12: Commit** `Package.swift`, the attention and expert sources and the two
      deletions, the test file, `docs/architecture.md` and this plan with this task
      ticked: `bench: the benches measure production's kernels, v21's coded arms gone (SHRIKE-61)`.
