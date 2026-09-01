#!/bin/bash
# prefill-measure.sh <host> <port> <promptdir> <outdir> <tag> <label>...
# The decode-measure.sh twin for prefill: waits for a running server, posts
# each prompt-<label>.json (from prefill-prompts.py) once, prints wall time and
# usage, and pauses PAUSE seconds (default 45) between prompts so a post-request
# cache settle cannot leak into the next request's kernel stats. Read the roles
# afterwards with prefill-ledger.py on the server's log.
set -u
HOST="$1"; PORT="$2"; PDIR="$3"; ODIR="$4"; TAG="$5"; shift 5
URL="http://$HOST:$PORT/v1/chat/completions"
mkdir -p "$ODIR"
tries=0
until curl -s -m 3 "http://$HOST:$PORT/v1/models" > /dev/null 2>&1; do
  tries=$((tries + 1))
  if [ "$tries" -gt 300 ]; then echo "server on $HOST:$PORT never became ready"; exit 1; fi
  sleep 2
done
echo "server ready on $HOST:$PORT after $((tries * 2))s"
for label in "$@"; do
  body="$PDIR/prompt-$label.json"
  out="$ODIR/resp-$TAG-$label.json"
  t=$(curl -s -m 3600 "$URL" -H 'Content-Type: application/json' -d @"$body" -o "$out" -w '%{time_total}')
  echo "$label wall=${t}s"
  python3 - "$out" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print("  response unreadable:", e); sys.exit(0)
if "error" in d:
    print("  ERROR:", d["error"]); sys.exit(0)
u = d.get("usage", {})
c = d["choices"][0]["message"]["content"]
print(f"  prompt_tokens={u.get('prompt_tokens')} completion_tokens={u.get('completion_tokens')} text={c[:60]!r}")
PY
  sleep "${PAUSE:-45}"
done
