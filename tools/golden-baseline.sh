#!/usr/bin/env bash
# Capture a deterministic generation baseline, so a refactor of the runtime
# can be checked against byte-identical output rather than "the tests still
# pass".
#
#   tools/golden-baseline.sh [short|long ...]     # capture; default: both
#   tools/golden-baseline.sh --check [short|long ...]
#
# Determinism comes from greedy decoding: --temperature 0 with a fixed seed and
# a fixed prompt. Greedy means the sampler never draws, so the only inputs are
# the weights and the kernels — exactly what a runtime refactor must not change.
# The `long` profile pins a ~2k-token context: long-context near-tie picks are
# where reduction-order drift between binaries shows first (v6 numerics note).
#
# SCOPE: a baseline is valid for one (machine, build, model) triple. Metal
# reduction order is not guaranteed across GPU families, so a file captured on
# an M1 is not a reference for an M4; files carry a machine tag in their name.
# Re-capture after a deliberate, signed-off numerics change; a diff at any
# other time is a regression.
#
# Env overrides (the mini has no repo checkout — run with all four):
#   CLI=~/shrike-runtime/bin/ShrikeCLI
#   MODEL=~/shrike-runtime/models/ornith15.gturbo
#   OUT_DIR=~/shrike-runtime/baselines
#   MACHINE_TAG=mini
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CLI="${CLI:-$ROOT/.build/arm64-apple-macosx/release/ShrikeCLI}"
OUT_DIR="${OUT_DIR:-$ROOT/baselines}"
MACHINE_TAG="${MACHINE_TAG:-$(sysctl -n hw.model | tr -cd '[:alnum:]')}"
MAX_NEW="${MAX_NEW:-96}"
SEED="${SEED:-1234}"

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

SHORT_PROMPT="Explain what a mutex is and when you would use one."
# ~2k tokens of deterministic context: a fixed ledger the model is asked to
# summarize. Built from literals only — never touch this construction, the
# stored baselines depend on its exact bytes.
long_prompt() {
  printf 'You are auditing a build ledger. Entries follow.\n'
  for i in $(seq 1 60); do
    printf 'Entry %d: commit c%04d built target shrike-core in %d ms with 0 warnings, ran 1108 tests in %d ms, linked 3 artifacts, and archived bundle b%03d to shelf s%d.\n' \
      "$i" $((i * 37)) $((1200 + i * 13)) $((80000 + i * 211)) "$i" $((i % 7))
  done
  printf 'Summarize: how many entries, which shelf received the most bundles, and the trend in build times.\n'
}

mode=capture
if [ "${1:-}" = "--check" ]; then mode=check; shift; fi
profiles=("$@"); [ ${#profiles[@]} -eq 0 ] && profiles=(short long)

if [ ! -x "$CLI" ]; then
  echo "missing $CLI — run: swift build -c release (or set CLI=)" >&2
  exit 2
fi

# CLAUDE.md: never run alongside another model process, and never terminate one
# we did not start. Refuse rather than race.
if pgrep -f 'ShrikeServer|ShrikeMac|ShrikeDecodeService|ShrikeCLI|ShrikePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm' >/dev/null 2>&1; then
  echo "a model process is already running; stop it yourself, then re-run" >&2
  exit 3
fi

mkdir -p "$OUT_DIR"
status=0

for profile in "${profiles[@]}"; do
  case "$profile" in
    short) prompt="$SHORT_PROMPT"; max_new="$MAX_NEW" ;;
    long)  prompt="$(long_prompt)"; max_new=128 ;;
    *) echo "unknown profile: $profile (expected short or long)" >&2
       status=1; continue ;;
  esac
  file="$OUT_DIR/ornith15-int4-${profile}.${MACHINE_TAG}.txt"
  work="$(mktemp "${TMPDIR:-/tmp}/golden-baseline.XXXXXX")"

  echo "== $profile =="
  # --quiet keeps the timing footer out of the compared text; only the
  # generated tokens are the contract. Timings vary run to run by design.
  "$CLI" --model "$MODEL" --prompt "$prompt" --max-new "$max_new" \
         --temperature 0 --seed "$SEED" --quiet > "$work" 2>"$work.err"
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
      echo "# seed:        $SEED"
      echo "# model:       $(basename "$MODEL")"
      echo "# captured-on: $(sysctl -n hw.model), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
      echo "# commit:      $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
      echo "---"
      cat "$work"
    } > "$file"
    echo "  captured -> $file ($(wc -c < "$work" | tr -d ' ') bytes)"
  else
    if [ ! -f "$file" ]; then
      echo "  no baseline at $file; run without --check first"; status=1
    elif diff -q <(sed '1,/^---$/d' "$file") "$work" >/dev/null; then
      echo "  ok — output identical to baseline"
    else
      echo "  MISMATCH against $file:"
      diff <(sed '1,/^---$/d' "$file") "$work" | head -30 | sed 's/^/    /'
      status=1
    fi
  fi
  rm -f "$work" "$work.err"
done

exit $status
