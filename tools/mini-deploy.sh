#!/bin/bash
# mini-deploy.sh [--restart]
# Copies the release binary and its resource bundles from .build/release to
# the mini's ~/shrike-runtime/bin. Each binary and bundle is staged remotely as
# <name>.staging first; only after every copy succeeds does one ssh command
# rm -rf the old bundle (mv -f for a binary) and swap the staged copy into
# place, so a dropped connection mid-copy can never leave a half-written
# artifact live. Anything else in bin/ is removed then: the box carries the
# current deploy only, and a retired binary left there escapes every guard
# that looks for a running `shrike`.
#
# By default this only copies — the mini's server is left running. The mini's
# owner approved restarts and deploy actions on 2026-09-01, but --restart is
# still an explicit act: pass it to also kill the running server, relaunch it
# with the production launch (tools/mini-production.sh), and poll for readiness.
#
# Run from the repo root after `swift build -c release`.
set -euo pipefail
source "$(dirname "$0")/mini-production.sh"
BIN=.build/release
RESTART=0
if [ "${1:-}" = "--restart" ]; then
  RESTART=1
fi

for f in shrike; do
  [ -x "$BIN/$f" ] || { echo "missing $BIN/$f — build release first" >&2; exit 1; }
done

if [ "$RESTART" -eq 1 ]; then
  ssh macmini "
    pkill -f 'bin/shrike serve --model $PRODUCTION_MODEL' || true
    sleep 3
    if pgrep -x shrike > /dev/null; then echo 'a shrike process is still running' >&2; exit 1; fi
  "
fi

shipped=""
for f in shrike; do
  scp -q "$BIN/$f" "macmini:shrike-runtime/bin/$f.staging"
  shipped="$shipped $f"
done
# The mini runs the golden from its own copy; deploy it with the binary so its
# process guard cannot go stale against a renamed binary.
scp -q tools/golden-baseline.sh "macmini:shrike-runtime/golden-baseline.sh"
for bundle in "$BIN"/*.bundle; do
  name=$(basename "$bundle")
  scp -q -r "$bundle" "macmini:shrike-runtime/bin/$name.staging"
  shipped="$shipped $name"
done

ssh macmini '
  set -eu
  cd ~/shrike-runtime/bin
  for f in shrike; do
    mv -f "$f.staging" "$f"
  done
  for staged in *.bundle.staging; do
    [ -e "$staged" ] || continue
    name="${staged%.staging}"
    rm -rf "$name"
    mv -f "$staged" "$name"
  done
  shipped="'"$shipped"' "
  for entry in *; do
    case "$shipped" in
      *" $entry "*) ;;
      *) rm -rf "$entry"; echo "removed retired $entry" ;;
    esac
  done
'
echo "copied binaries + bundles"

if [ "$RESTART" -eq 0 ]; then
  exit 0
fi

LAUNCH=$(server_launch "$PRODUCTION_MODEL" "$PRODUCTION_PORT" "$PRODUCTION_RAM_BUDGET")
ssh macmini "
  set -eu
  cd ~/shrike-runtime
  if curl -sf -m 3 http://127.0.0.1:$PRODUCTION_PORT/v1/models > /dev/null 2>&1; then
    echo 'something already answers on $PRODUCTION_PORT' >&2; exit 1
  fi
  [ -f $SERVER_LOG ] && mv -f $SERVER_LOG \"$SERVER_LOG.\$(date +%Y%m%d-%H%M%S)\"
  $LAUNCH
  server_pid=\$!
  tries=0
  until curl -sf -m 3 http://127.0.0.1:$PRODUCTION_PORT/v1/models > /dev/null 2>&1; do
    kill -0 \"\$server_pid\" 2>/dev/null || {
      echo 'the server this script launched exited' >&2; tail -n 5 $SERVER_LOG; exit 1; }
    tries=\$((tries + 1))
    if [ \"\$tries\" -gt 60 ]; then echo 'server never became ready' >&2; tail -n 5 $SERVER_LOG; exit 1; fi
    sleep 3
  done
  echo \"server ready after \$((tries * 3))s (pid \$server_pid)\"
  logged_bits=''
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    logged_bits=\$(grep -a -o 'prefill_router_bits=[0-9]*' $SERVER_LOG | head -n 1)
    [ -n \"\$logged_bits\" ] && break
    sleep 2
  done
  echo \"\${logged_bits:-prefill_router_bits=NOT-LOGGED}\"
"
