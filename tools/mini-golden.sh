#!/bin/bash
# tools/mini-golden.sh [--check] [profile ...]
# The golden on the mini, run from the checkout: the mini has no checkout, and
# the repo's baselines/*.mini.txt are the only copy of its baselines. Stops
# production's server, runs tools/golden-baseline.sh there from a scratch
# directory holding it, tools/mini-production.sh and (for --check) the mini's
# baselines, brings a capture's files back into baselines/, and relaunches
# production from tools/mini-production.sh whatever the golden's outcome.
# Refuses with exit 3, stopping nothing, while any shrike other than production's
# server runs there. Once it has tried to stop production it either relaunches it
# or exits 4 saying production was NOT relaunched; the relaunch first stops the
# golden and its server, which an interrupted ssh leaves running on the mini
# (a remote command without a pty is never told its client died, and -tt does
# not change that for a killed client). Arguments pass through to the golden.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/tools/mini-production.sh"
mode=capture
[ "${1:-}" = --check ] && mode=check
PRODUCTION_MATCH="bin/shrike serve --model $PRODUCTION_MODEL --port $PRODUCTION_PORT"
GOLDEN_PORT=8082
scratch=""

relaunch() {
  local launch cleanup=""
  launch=$(server_launch "$PRODUCTION_MODEL" "$PRODUCTION_PORT" "$PRODUCTION_RAM_BUDGET")
  # The golden first, so a pending TERM stops it before it can start the next
  # profile's server once its current request fails.
  if [ -n "$scratch" ]; then
    cleanup="pkill -f '$scratch/golden-baseline.sh'
    pkill -f \"shrike serve --model \$HOME/shrike-runtime/models/ornith15.gturbo --port $GOLDEN_PORT\"
    rm -rf $scratch"
  fi
  ssh macmini "
    $cleanup
    n=0
    while pgrep -x 'shrike(-bench)?' > /dev/null; do
      n=\$((n + 1))
      [ \$n -gt 30 ] && { echo 'a shrike process is still running' >&2; exit 1; }
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
  if [ \"\$(pgrep -x 'shrike(-bench)?' | sort)\" != \"\$(pgrep -f '$PRODUCTION_MATCH' | sort)\" ]; then
    echo 'a shrike other than production is running on the mini; stopping nothing' >&2
    exit 3
  fi
"
refused=$?
[ $refused -eq 0 ] || exit $refused
trap 'relaunch || { echo "production NOT relaunched on the mini" >&2; exit 4; }' EXIT
trap 'exit 130' INT TERM HUP
ssh macmini "
  pkill -f '$PRODUCTION_MATCH' || true
  n=0
  while pgrep -f '$PRODUCTION_MATCH' > /dev/null; do
    n=\$((n + 1))
    [ \$n -gt 30 ] && { echo 'production did not stop' >&2; exit 1; }
    sleep 1
  done
" || exit 1

scratch=$(ssh macmini 'mktemp -d /tmp/shrike-golden.XXXXXX') || exit 1
ssh macmini "mkdir $scratch/baselines" || exit 1
scp -q "$ROOT/tools/golden-baseline.sh" "$ROOT/tools/mini-production.sh" "macmini:$scratch/" || exit 1
if [ "$mode" = check ]; then
  scp -q "$ROOT"/baselines/*.mini.txt "macmini:$scratch/baselines/" || exit 1
fi
ssh macmini "env CLI=\$HOME/shrike-runtime/bin/shrike MODEL=\$HOME/shrike-runtime/models/ornith15.gturbo OUT_DIR=$scratch/baselines MACHINE_TAG=mini SERVE_PORT=$GOLDEN_PORT bash $scratch/golden-baseline.sh $*"
status=$?
if [ "$mode" = capture ] && [ $status -eq 0 ]; then
  scp -q "macmini:$scratch/baselines/*.mini.txt" "$ROOT/baselines/" || status=1
fi
exit $status
