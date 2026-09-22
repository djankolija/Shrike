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
# and payload-<tag>-turn3.json. `restore`: relaunch production
# (tools/mini-production.sh, whatever the env below says) and stop.
# Every phase rotates the server's log first and copies the
# whole log back afterwards. A missing payload or a `wait_settle` timeout
# aborts the phase with a non-zero exit rather than sending a row that would
# read as valid.
#
# MODEL (optional env, default production's): the model the relaunched server
# serves; readiness waits for the id the server derives from its name.
# RAM_BUDGET (optional env, default production's): the launch's --ram-budget; an
# arm changing it gives SERVER_ENV a SHRIKE_EXPERT_SLOT_TABLE summing to what it
# snaps to, or an empty one for the uniform pool. MAX_TOKENS (optional env): overrides the phase's cold
# request's max_tokens (the long-decode arm raises it to 512). TURN2_MAX_TOKENS
# (optional env, `turns-live` only): overrides turn 2's max_tokens (default
# 8; the long-answer follow-up arm sets it to 512, e.g. `TURN2_MAX_TOKENS=512`).
# SERVER_ENV
# (optional env): layered on production's env assignments, e.g.
# SERVER_ENV="SHRIKE_PREFILL_ANE=on" for the A/B; a production value is
# overridden by assigning it again. REUSE=<dir> (optional
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
source "$(dirname "$0")/mini-production.sh"
MODEL="${MODEL:-$PRODUCTION_MODEL}"
RAM_BUDGET="${RAM_BUDGET:-$PRODUCTION_RAM_BUDGET}"
mkdir -p "$ODIR"

relaunch() {  # $1 = model, $2 = port, $3 = ram budget, $4 = env assignments layered on production's
  local launch id tries=0
  launch=$(server_launch "$1" "$2" "$3" "$4")
  ssh macmini "
    pkill -f 'bin/shrike serve --model' || true
    sleep 3
    if pgrep -x shrike > /dev/null; then echo 'a shrike process is still running' >&2; exit 1; fi
    cd ~/shrike-runtime
    [ -f $SERVER_LOG ] && mv -f $SERVER_LOG \"$SERVER_LOG.\$(date +%Y%m%d-%H%M%S)\"
    $launch
    exit 0
  " || exit 1
  id=$(basename "$1" .gturbo)
  until curl -sf -m 3 "http://$HOST:$2/v1/models" 2>/dev/null | grep -q "$id"; do
    tries=$((tries + 1)); if [ "$tries" -gt 120 ]; then echo "server never listed $id" >&2; exit 1; fi; sleep 2
  done
  sleep 5
}

relaunch_arm() {
  relaunch "$MODEL" "$PORT" "$RAM_BUDGET" "${SERVER_ENV:-}"
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
  ssh macmini 'n=0; until grep -a -q "settle_done" '"$SERVER_LOG"' && [ "$(grep -a -c "settle_done" '"$SERVER_LOG"')" -ge '"$1"' ]; do n=$((n+1)); [ $n -gt 45 ] && { echo "settle wait timed out" >&2; exit 1; }; sleep 2; done; echo "settled after $((n*2))s"'
}

fetch_log() {
  sleep 2
  scp -q "macmini:$SERVER_LOG" "$ODIR/server-mini-$TAG.log"
  echo "log: server-mini-$TAG.log ($(wc -l < "$ODIR/server-mini-$TAG.log") lines)"
}

case "$phase" in
  pair)
    relaunch_arm
    send "t$arg" "${MAX_TOKENS:-}"; wait_settle 1 || exit 1
    send "t${arg}b"; wait_settle 2 || exit 1
    fetch_log
    ;;
  suffix)
    relaunch_arm
    send "tX" "${MAX_TOKENS:-}"; wait_settle 1 || exit 1
    send "tXp4"; wait_settle 2 || exit 1
    send "tXp16"; wait_settle 3 || exit 1
    fetch_log
    ;;
  turns)
    relaunch_arm
    send "tX" "${MAX_TOKENS:-}"; wait_settle 1 || exit 1
    send "tXturn2"; wait_settle 2 || exit 1
    send "tXturn3"; wait_settle 3 || exit 1
    fetch_log
    ;;
  turns-live)
    relaunch_arm
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
    relaunch "$PRODUCTION_MODEL" "$PRODUCTION_PORT" "$PRODUCTION_RAM_BUDGET" ""
    echo "production restored"
    ;;
  *) echo "unknown phase $phase" >&2; exit 2 ;;
esac
