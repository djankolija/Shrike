#!/bin/bash
# turn-rig.sh <host> <port> <promptdir> <outdir> <tag> <phase> [arg]
#   phases: pair <300|1k|2k> | suffix | turns | restore
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
# the same cached prefix. `restore`: relaunch production and stop. Every
# phase rotates the server's /tmp/ornith.log first and copies the whole log
# back afterwards. A missing payload or a `wait_settle` timeout aborts the
# phase with a non-zero exit rather than sending a row that would read as
# valid.
#
# MODEL / MODEL_ID (optional env, default ./models/ornith15.gturbo /
# ornith15, matching tools/mini-deploy.sh's launch): the model the relaunched
# server serves. MAX_TOKENS (optional env): overrides the phase's cold
# request's max_tokens (the long-decode arm raises it to 512). SERVER_ENV
# (optional env): prepended to the server launch's env assignments, e.g.
# SERVER_ENV="SHRIKE_PREFILL_SWEEP=carry" for the A/B.
set -u
HOST="$1"; PORT="$2"; PDIR="$3"; ODIR="$4"; TAG="$5"; phase="$6"; arg="${7:-}"
MODEL="${MODEL:-./models/ornith15.gturbo}"
MODEL_ID="${MODEL_ID:-ornith15}"
mkdir -p "$ODIR"
LAUNCH="env ${SERVER_ENV:-} SHRIKE_RUNNER_STATS=1 SHRIKE_KERNEL_STATS=1 nohup ./bin/ShrikeServer --model $MODEL --model-id $MODEL_ID --port $PORT --max-context 32768 --ram-budget 8G --thinking off > /tmp/ornith.log 2>&1 &"

relaunch() {
  ssh macmini "
    pkill -f 'bin/ShrikeServer --model $MODEL' || true
    sleep 3
    if pgrep -x ShrikeServer > /dev/null; then echo 'server still running' >&2; exit 1; fi
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

send() {  # $1 = payload label, $2 = optional max_tokens override
  local label="$1" override="${2:-}" body
  body="$PDIR/$label.json"
  if [ ! -f "$body" ]; then echo "missing payload: $body" >&2; exit 1; fi
  if [ -n "$override" ]; then
    body="/tmp/turn-rig-$label-mt$override.json"
    python3 -c "import json; d = json.load(open('$PDIR/$label.json')); d['max_tokens'] = $override; json.dump(d, open('$body', 'w'))"
  fi
  scp -q "$body" "macmini:/tmp/turn-$label.json"
  t=$(ssh macmini "curl -s -m 1800 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d @/tmp/turn-$label.json -o /tmp/turn-resp-$label.json -w '%{time_total}'")
  scp -q "macmini:/tmp/turn-resp-$label.json" "$ODIR/resp-$TAG-$label.json"
  u=$(python3 -c "import json; d=json.load(open('$ODIR/resp-$TAG-$label.json')); u=d.get('usage',{}); print(f\"prompt={u.get('prompt_tokens')} cached={u.get('prompt_tokens_details',{}).get('cached_tokens')} completion={u.get('completion_tokens')} finish={d['choices'][0].get('finish_reason')}\")" 2>&1)
  echo "$label wall=${t}s $u"
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
  restore)
    relaunch; echo "production restored"
    ;;
  *) echo "unknown phase $phase" >&2; exit 2 ;;
esac
