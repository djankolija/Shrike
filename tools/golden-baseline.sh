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
if pgrep -x 'shrike(-bench)?' >/dev/null 2>&1 \
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
  local resp rc
  resp="$(curl -sS -m 1800 "$SERVE_URL/v1/chat/completions" \
          -H 'Content-Type: application/json' -d "$1" 2>&1)"
  rc=$?
  if [ $rc -ne 0 ] || ! printf '%s' "$resp" | jq -e '.choices[0].message.content | strings' > /dev/null 2>&1; then
    echo "request $4 failed (curl exit $rc): $(printf '%s' "$resp" | head -c 300)" >&2
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
  if [ $rc -ne 0 ]; then
    echo "the server's log ends:" >&2
    tail -5 "$dir/server.log" >&2
  fi
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
