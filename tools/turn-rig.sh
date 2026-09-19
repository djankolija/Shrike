#!/bin/bash
# turn-rig.sh <host> <port> <promptdir> <outdir> <tag> <phase> [arg]
#   phases: pair <300|1k|2k> | suffix | turns | turns-live [answer_max_tokens] | restore
# The v13 "the turn" chapter's request-shape rig. <host>:<port> is where this
# machine polls the server's HTTP API for readiness after a relaunch; the
# request sends run ON the mini over ssh instead (the payloads are large), so
# they target 127.0.0.1:<port> there regardless of <host>, since <host> is
# only this machine's name for the mini and does not resolve from inside it.
# Relaunching the server itself always goes over the ssh alias `macmini`,
# unrelated to <host>. <promptdir> holds the turn-prompts.py
# payloads (t300.json, t300b.json, ... tXp16.json, tXturn2.json, tXturn3.json);
# responses and the server log land in <outdir>, named with <tag>. `pair`: a
# fresh production server, the cold first request (t<arg>), wait for the
# prompt cache's settle to finish, then the warm second request with a
# different prompt of the same length (t<arg>b). `suffix`: a fresh server,
# tX (2k), then tX+4 (~300 new tokens on the cached prefix), then tX+16
# (~750 more). `turns`: tX, tXturn2, tXturn3 — a multi-turn conversation over
# the same cached prefix, tXturn2/tXturn3 sent as stored (the canned 8-token
# assistant answer from turn-prompts.py already baked in). `turns-live
# [answer_max_tokens]`: the same chain but with the card's actual follow-up
# shape — tX is sent with max_tokens <answer_max_tokens> (default 512) and
# its real response content becomes the assistant turn; turn 2 is built from
# that response plus the last user message of tXturn2.json (or USER_TURN2,
# below), sent with max_tokens 8 (or TURN2_MAX_TOKENS, below); turn 3 is
# built the same way from turn 2's own real response plus tXturn3.json's
# last user message. The built
# payloads land beside the responses in <outdir> as payload-<tag>-turn2.json
# and payload-<tag>-turn3.json. `restore`: relaunch production and stop.
# Every phase rotates the server's /tmp/ornith.log first and copies the
# whole log back afterwards. A missing payload or a `wait_settle` timeout
# aborts the phase with a non-zero exit rather than sending a row that would
# read as valid.
#
# MODEL (optional env, default ./models/ornith15.gturbo, matching
# tools/mini-deploy.sh's launch): the model the relaunched server serves.
# MODEL_ID (optional env, default ornith15): the id the server derives from it,
# used only to wait for readiness. MAX_TOKENS (optional env): overrides the phase's cold
# request's max_tokens (the long-decode arm raises it to 512). TURN2_MAX_TOKENS
# (optional env, `turns-live` only): overrides turn 2's max_tokens (default
# 8; the long-answer follow-up arm sets it to 512, e.g. `TURN2_MAX_TOKENS=512`).
# SERVER_ENV
# (optional env): prepended to the server launch's env assignments, e.g.
# SERVER_ENV="SHRIKE_PREFILL_ANE=on" for the A/B. REUSE=<dir> (optional
# env, `turns-live` only): instead of building turn 2 and turn 3 from this
# run's own live responses, copy the already-built payload-*-turn2.json and
# payload-*-turn3.json found in <dir> (another run's <outdir>) and send those
# — mandatory once the knob under test can change a completion, since a
# rebuilt turn 3 would then carry different bytes per arm and the pair would
# no longer be an A/B on identical requests; <dir> must hold exactly one of
# each and the reused turn 2's assistant turn must match this run's own live
# tX answer, or the phase aborts rather than send a silently mismatched pair.
# USER_TURN2=<payload> (optional
# env, `turns-live` only): read turn 2's live follow-up from this payload's
# last user message instead of tXturn2.json's — how a differently-shaped
# turn 2 (e.g. a padded one) gets substituted.
set -u
HOST="$1"; PORT="$2"; PDIR="$3"; ODIR="$4"; TAG="$5"; phase="$6"; arg="${7:-}"
MODEL="${MODEL:-./models/ornith15.gturbo}"
MODEL_ID="${MODEL_ID:-ornith15}"
mkdir -p "$ODIR"
LAUNCH="env ${SERVER_ENV:-} SHRIKE_RUNNER_STATS=1 SHRIKE_KERNEL_STATS=1 nohup ./bin/shrike serve --model $MODEL --port $PORT --max-context 32768 --ram-budget 8G --thinking off > /tmp/ornith.log 2>&1 &"

relaunch() {
  ssh macmini "
    pkill -f 'bin/shrike serve --model $MODEL' || true
    sleep 3
    if pgrep -f 'shrike serve' > /dev/null; then echo 'server still running' >&2; exit 1; fi
    cd ~/shrike-runtime
    [ -f /tmp/ornith.log ] && mv -f /tmp/ornith.log \"/tmp/ornith.log.\$(date +%Y%m%d-%H%M%S)\"
    $LAUNCH
    exit 0
  " || exit 1
  tries=0
  until curl -sf -m 3 "http://$HOST:$PORT/v1/models" 2>/dev/null | grep -q "$MODEL_ID"; do
    tries=$((tries + 1)); if [ "$tries" -gt 120 ]; then echo "server never listed the model" >&2; exit 1; fi; sleep 2
  done
  sleep 5
}

send() {  # $1 = payload label, $2 = optional max_tokens override, $3 = optional body path (default $PDIR/$1.json)
  local label="$1" override="${2:-}" body="${3:-$PDIR/$1.json}"
  if [ ! -f "$body" ]; then echo "missing payload: $body" >&2; exit 1; fi
  if [ -n "$override" ]; then
    local overridden="/tmp/turn-rig-$label-mt$override.json"
    python3 -c "import json; d = json.load(open('$body')); d['max_tokens'] = $override; json.dump(d, open('$overridden', 'w'))"
    body="$overridden"
  fi
  scp -q "$body" "macmini:/tmp/turn-$label.json"
  t=$(ssh macmini "curl -s -m 1800 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d @/tmp/turn-$label.json -o /tmp/turn-resp-$label.json -w '%{time_total}'")
  scp -q "macmini:/tmp/turn-resp-$label.json" "$ODIR/resp-$TAG-$label.json"
  u=$(python3 -c "import json; d=json.load(open('$ODIR/resp-$TAG-$label.json')); u=d.get('usage',{}); print(f\"prompt={u.get('prompt_tokens')} cached={u.get('prompt_tokens_details',{}).get('cached_tokens')} completion={u.get('completion_tokens')} finish={d['choices'][0].get('finish_reason')}\")" 2>&1)
  echo "$label wall=${t}s $u"
}

build_next() {  # $1 = previous payload, $2 = previous response, $3 = payload holding the desired next user turn, $4 = out
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import json, sys
prev, resp, rig, out = sys.argv[1:5]
p = json.load(open(prev)); r = json.load(open(resp)); g = json.load(open(rig))
answer = r["choices"][0]["message"]["content"]
user = [m for m in g["messages"] if m["role"] == "user"][-1]["content"]
p["messages"] = p["messages"] + [{"role": "assistant", "content": answer}, {"role": "user", "content": user}]
p["max_tokens"] = 8
json.dump(p, open(out, "w"))
print(f"built {out.rsplit('/',1)[-1]}: {len(p['messages'])} messages, answer {len(answer)} chars")
PY
}

reuse_single_match() {  # $1 = glob pattern (e.g. "$REUSE/payload-*-turn2.json"); echoes the one match or aborts
  local pattern="$1" matches count
  matches=$(ls $pattern 2>/dev/null)
  count=$(echo "$matches" | grep -ac .)
  if [ "$count" -eq 0 ]; then
    echo "REUSE has no built $pattern" >&2
    exit 1
  fi
  if [ "$count" -gt 1 ]; then
    echo "REUSE has more than one $pattern — point it at a single run's outdir" >&2
    exit 1
  fi
  echo "$matches"
}

verify_reuse_turn2() {  # $1 = reused turn2 payload, $2 = this run's tX response; aborts on a mismatched assistant turn
  python3 - "$1" "$2" <<'PY'
import json, os, sys
turn2, resp = sys.argv[1:3]
p = json.load(open(turn2))
r = json.load(open(resp))
reused = p["messages"][-2]["content"]
live = r["choices"][0]["message"]["content"]
if reused != live:
    print(f"REUSE turn2 payload {os.path.basename(turn2)} was built from a different tX "
          "answer than this run's live tX", file=sys.stderr)
    sys.exit(1)
PY
}

wait_settle() {  # wait until the last request's settle_done, or fail after 90 s
  ssh macmini 'n=0; until grep -a -q "settle_done" /tmp/ornith.log && [ "$(grep -a -c "settle_done" /tmp/ornith.log)" -ge '"$1"' ]; do n=$((n+1)); [ $n -gt 45 ] && { echo "settle wait timed out" >&2; exit 1; }; sleep 2; done; echo "settled after $((n*2))s"'
}

fetch_log() {
  sleep 2
  scp -q macmini:/tmp/ornith.log "$ODIR/server-mini-$TAG.log"
  echo "log: server-mini-$TAG.log ($(wc -l < "$ODIR/server-mini-$TAG.log") lines)"
}

case "$phase" in
  pair)
    relaunch
    send "t$arg" "${MAX_TOKENS:-}"; wait_settle 1 || exit 1
    send "t${arg}b"; wait_settle 2 || exit 1
    fetch_log
    ;;
  suffix)
    relaunch
    send "tX" "${MAX_TOKENS:-}"; wait_settle 1 || exit 1
    send "tXp4"; wait_settle 2 || exit 1
    send "tXp16"; wait_settle 3 || exit 1
    fetch_log
    ;;
  turns)
    relaunch
    send "tX" "${MAX_TOKENS:-}"; wait_settle 1 || exit 1
    send "tXturn2"; wait_settle 2 || exit 1
    send "tXturn3"; wait_settle 3 || exit 1
    fetch_log
    ;;
  turns-live)
    relaunch
    turn2_payload="$ODIR/payload-$TAG-turn2.json"
    turn3_payload="$ODIR/payload-$TAG-turn3.json"
    send "tX" "${arg:-512}"; wait_settle 1 || exit 1
    if [ -n "${REUSE:-}" ]; then
      turn2_src=$(reuse_single_match "$REUSE/payload-*-turn2.json") || exit 1
      turn3_src=$(reuse_single_match "$REUSE/payload-*-turn3.json") || exit 1
      verify_reuse_turn2 "$turn2_src" "$ODIR/resp-$TAG-tX.json" || exit 1
      cp "$turn2_src" "$turn2_payload"
      cp "$turn3_src" "$turn3_payload"
    else
      build_next "$PDIR/tX.json" "$ODIR/resp-$TAG-tX.json" "${USER_TURN2:-$PDIR/tXturn2.json}" "$turn2_payload"
    fi
    send "turn2" "${TURN2_MAX_TOKENS:-}" "$turn2_payload"; wait_settle 2 || exit 1
    if [ -z "${REUSE:-}" ]; then
      build_next "$turn2_payload" "$ODIR/resp-$TAG-turn2.json" "$PDIR/tXturn3.json" "$turn3_payload"
    fi
    send "turn3" "" "$turn3_payload"; wait_settle 3 || exit 1
    fetch_log
    ;;
  restore)
    relaunch; echo "production restored"
    ;;
  *) echo "unknown phase $phase" >&2; exit 2 ;;
esac
