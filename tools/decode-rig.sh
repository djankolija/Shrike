#!/bin/bash
# decode-rig.sh <host> <port> <promptdir> <outdir> <tag> <shape>...
#   shapes: card | d512-300 | d512-1k | restore
# The decode pass II chapter's request-shape rig: a fresh server per shape,
# the cold request STREAMED so every token's arrival time is recorded beside
# the route trace (tools/decode-stream-client.py runs on the mini), then the
# shape's follow-ups. `card`: tX (the 2k card) answered at MAX_TOKENS, then the
# archived turn 2 / turn 3 payloads from REUSE at max_tokens 8 (the live answer
# is checked against the payload's assistant turn first; a mismatch is reported
# and the turns still run, on a cached prefix that no longer matches, so their
# rows read as a full re-prefill rather than a follow-up). `d512-300` /
# `d512-1k`: the 300- / 1k-token prompt answered at MAX_TOKENS, the settle
# awaited, then the warm same-length second prompt (t300b / t1kb) at 8.
# `restore`: relaunch the bare production server and stop.
#
# <host>:<port> is where this machine polls the server's HTTP API for
# readiness after a relaunch; the requests run ON the mini over ssh and target
# 127.0.0.1:<port> there. Relaunching goes over the ssh alias `macmini`.
# <promptdir> holds the turn-prompts.py payloads (tX.json, t300.json,
# t300b.json, t1k.json, t1kb.json). Outputs land in <outdir>, named
# <tag>-<shape>: tokens-*.json (the streamed arrivals), route-*.trace,
# prefetch-*.jsonl (with PREFETCH_TRACE=1), resp-*.json, server-mini-*.log,
# and one row per request from tools/decode-rows.py.
#
# Env: SERVER_ENV (prepended to the server launch, e.g.
# SERVER_ENV="SHRIKE_PREFILL_ANE=on" for an A/B arm; every launch also carries
# SHRIKE_RUNNER_STATS=1 SHRIKE_KERNEL_STATS=1 and SHRIKE_ROUTE_TRACE);
# PREFETCH_TRACE=1 adds SHRIKE_PREFETCH_TRACE (the next-layer router probe's
# top-8 is logged per decode layer); NO_TURNS=1 skips the follow-up requests;
# MAX_TOKENS (default 512) the cold request's answer length; MODEL / MODEL_ID
# (default ./models/ornith15.gturbo / ornith15, matching tools/mini-deploy.sh);
# REUSE=<dir> (card only) the directory holding payload-*-turn2.json and
# payload-*-turn3.json. A missing payload or a settle timeout aborts the shape
# with a non-zero exit rather than sending a row that would read as valid.
set -u
if [ $# -lt 6 ]; then
  sed -n '2,3p' "$0" >&2
  exit 2
fi
HOST=$1; PORT=$2; PROMPTS=$3; OUT=$4; TAG=$5; shift 5
MODEL=${MODEL:-./models/ornith15.gturbo}; MODEL_ID=${MODEL_ID:-ornith15}
SERVER_ENV=${SERVER_ENV:-}; MAX_TOKENS=${MAX_TOKENS:-512}
PREFETCH_TRACE=${PREFETCH_TRACE:-0}; NO_TURNS=${NO_TURNS:-0}; REUSE=${REUSE:-}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
scp -q "$ROOT/tools/decode-stream-client.py" macmini:/tmp/decode-stream-client.py || exit 1

need_payload() {
  if [ ! -f "$1" ]; then echo "missing payload $1" >&2; exit 1; fi
}

relaunch() {  # $1 = extra env assignments for this shape (traces)
  ssh macmini "
    pkill -f 'bin/ShrikeServer --model $MODEL' || true
    sleep 3
    if pgrep -x ShrikeServer > /dev/null; then echo 'server still running' >&2; exit 1; fi
    cd ~/shrike-runtime
    [ -f /tmp/ornith.log ] && mv -f /tmp/ornith.log \"/tmp/ornith.log.\$(date +%Y%m%d-%H%M%S)\"
    env $SERVER_ENV $1 SHRIKE_RUNNER_STATS=1 SHRIKE_KERNEL_STATS=1 nohup ./bin/ShrikeServer --model $MODEL --model-id $MODEL_ID --port $PORT --max-context 32768 --ram-budget 8G --thinking off > /tmp/ornith.log 2>&1 &
    exit 0
  " || exit 1
  tries=0
  until curl -sf -m 3 "http://$HOST:$PORT/v1/models" 2>/dev/null | grep -q "$MODEL_ID"; do
    tries=$((tries + 1))
    if [ "$tries" -gt 120 ]; then echo "server never listed the model" >&2; exit 1; fi
    sleep 2
  done
  sleep 5
}

wait_settle() {  # $1 = the settle_done count to wait for (one per request sent so far)
  ssh macmini 'n=0; until [ "$(grep -a -c "settle_done" /tmp/ornith.log)" -ge '"$1"' ]; do n=$((n+1)); [ $n -gt 150 ] && { echo "settle wait timed out" >&2; exit 1; }; sleep 2; done; echo "settled after $((n*2))s"' || exit 1
}

send_plain() {  # $1 = label, $2 = local payload path, $3 = max_tokens, $4 = run tag
  need_payload "$2"
  python3 -c "import json; d=json.load(open('$2')); d['max_tokens']=$3; json.dump(d, open('/tmp/decode-$1.json','w'))"
  scp -q "/tmp/decode-$1.json" "macmini:/tmp/decode-$1.json"
  t=$(ssh macmini "curl -s -m 1800 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d @/tmp/decode-$1.json -o /tmp/decode-resp-$1.json -w '%{time_total}'")
  scp -q "macmini:/tmp/decode-resp-$1.json" "$OUT/resp-$4-$1.json"
  u=$(python3 -c "import json; d=json.load(open('$OUT/resp-$4-$1.json')); u=d.get('usage',{}); print(f\"prompt={u.get('prompt_tokens')} cached={u.get('prompt_tokens_details',{}).get('cached_tokens')} completion={u.get('completion_tokens')} finish={d['choices'][0].get('finish_reason')}\")" 2>&1)
  echo "$1 wall=${t}s $u"
}

send_stream() {  # $1 = label, $2 = local payload path, $3 = max_tokens, $4 = run tag
  need_payload "$2"
  scp -q "$2" "macmini:/tmp/decode-$1.json"
  ssh macmini "python3 /tmp/decode-stream-client.py /tmp/decode-$1.json $3 /tmp/decode-tokens-$1.json $PORT" || exit 1
  scp -q "macmini:/tmp/decode-tokens-$1.json" "$OUT/tokens-$4-$1.json"
}

for shape in "$@"; do
  run="$TAG-$shape"
  echo "=== $shape ($run) $(date +%H:%M:%S) [$SERVER_ENV] ==="
  if [ "$shape" = restore ]; then
    SERVER_ENV="" relaunch ""
    echo "production restored at the bare launch"
    continue
  fi
  traces="SHRIKE_ROUTE_TRACE=/tmp/route-$run.trace"
  if [ "$PREFETCH_TRACE" = 1 ]; then
    traces="$traces SHRIKE_PREFETCH_TRACE=/tmp/prefetch-$run.jsonl"
  fi
  ssh macmini "rm -f /tmp/route-$run.trace /tmp/prefetch-$run.jsonl"
  case "$shape" in
    card)
      relaunch "$traces"
      send_stream tX "$PROMPTS/tX.json" "$MAX_TOKENS" "$run"
      wait_settle 1
      if [ "$NO_TURNS" != 1 ]; then
        if [ -z "$REUSE" ]; then echo "card needs REUSE=<dir> for the turn payloads" >&2; exit 1; fi
        turn2=$(ls "$REUSE"/payload-*-turn2.json 2>/dev/null | head -1)
        turn3=$(ls "$REUSE"/payload-*-turn3.json 2>/dev/null | head -1)
        need_payload "$turn2"; need_payload "$turn3"
        python3 - "$OUT/tokens-$run-tX.json" "$turn2" <<'PY'
import json, sys
live = json.load(open(sys.argv[1]))["text"]
reused = json.load(open(sys.argv[2]))["messages"]
assistant = [m for m in reused if m["role"] == "assistant"][-1]["content"]
print("reuse check:", "answer matches the archived assistant turn" if live == assistant
      else f"MISMATCH (live {len(live)} chars vs archived {len(assistant)})")
PY
        send_plain turn2 "$turn2" 8 "$run"
        wait_settle 2
        send_plain turn3 "$turn3" 8 "$run"
        wait_settle 3
      fi
      ;;
    d512-300|d512-1k)
      p="${shape#d512-}"
      relaunch "$traces"
      send_stream "t$p" "$PROMPTS/t$p.json" "$MAX_TOKENS" "$run"
      wait_settle 1
      if [ "$NO_TURNS" != 1 ]; then
        send_plain "t${p}b" "$PROMPTS/t${p}b.json" 8 "$run"
        wait_settle 2
      fi
      ;;
    *) echo "unknown shape $shape" >&2; exit 2 ;;
  esac
  sleep 3
  scp -q "macmini:/tmp/ornith.log" "$OUT/server-mini-$run.log" || exit 1
  scp -q "macmini:/tmp/route-$run.trace" "$OUT/route-$run.trace" || exit 1
  if [ "$PREFETCH_TRACE" = 1 ]; then
    scp -q "macmini:/tmp/prefetch-$run.jsonl" "$OUT/prefetch-$run.jsonl" || exit 1
    echo "prefetch trace: $(wc -l < "$OUT/prefetch-$run.jsonl") lines"
  fi
  echo "log: $(wc -l < "$OUT/server-mini-$run.log") lines; trace: $(wc -l < "$OUT/route-$run.trace") lines"
  tokens=( "$OUT"/tokens-"$run"-*.json )
  [ -f "${tokens[0]}" ] || tokens=()
  python3 "$ROOT/tools/decode-rows.py" "$OUT/server-mini-$run.log" "${tokens[@]}"
done
echo "=== done $(date +%H:%M:%S) ==="
