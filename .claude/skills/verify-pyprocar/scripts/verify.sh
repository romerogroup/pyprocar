#!/usr/bin/env bash
# Harness for verifying pyprocar end to end. See ../SKILL.md.
#   verify.sh doctor
#   verify.sh fetch <relpath>...            e.g. data/examples/bands/non-spin-polarized
#   verify.sh run <name> <fixture-relpath> <driver.py>
#   verify.sh clean <run-dir>
#   verify.sh gc [hours]                    remove work/ of runs older than hours (default 24)
set -euo pipefail

REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
RUNS="$REPO/data/verify-runs"
MIN_FREE_MB="${VERIFY_MIN_FREE_MB:-2048}"
abs() { (cd "$(dirname "$1")" && echo "$PWD/$(basename "$1")"); }
[ "${1:-}" = run ] && [ $# -ge 4 ] && set -- "$1" "$2" "$3" "$(abs "$4")"
cd "$REPO"

py() { pixi run -q -e default python "$@"; }

free_mb() { df -Pk "$1" | awk 'NR == 2 { print int($4 / 1024) }'; }

require_free() {
  local need_mb="$1" dir free
  for dir in "$RUNS" "${TMPDIR:-/tmp}"; do
    free="$(free_mb "$dir")"
    if [ "$free" -lt "$need_mb" ]; then
      echo "refusing: $dir has ${free} MB free, the run needs ${need_mb} MB (fixture + VERIFY_MIN_FREE_MB=$MIN_FREE_MB)." >&2
      echo "free space with: $0 gc, or $0 clean <run-dir>" >&2
      exit 3
    fi
  done
}

scrub() { rm -rf "$1/work" "$1/.start"; }

case "${1:-}" in
doctor)
  echo "branch:  $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
  echo "dirty:   $(git status --porcelain --untracked-files=no | wc -l) tracked file(s) modified"
  py -W ignore -c '
import sys, pyprocar, matplotlib, pyvista, vtk
print("python: ", sys.version.split()[0])
print("pyprocar:", pyprocar.__version__, "from", pyprocar.__file__)
print("pyvista: ", pyvista.__version__, "vtk", vtk.vtkVersion.GetVTKVersion())
'
  echo "fixtures:"; ls -d data/examples/*/* 2>/dev/null | sed 's/^/  /' || echo "  (none; run: verify.sh fetch <relpath>)"
  ;;
fetch)
  shift
  for rel in "$@"; do
    py -W ignore -c 'import sys; from pathlib import Path; import pyprocar
pyprocar.download_from_hf(relpath=sys.argv[1], output_path=Path(".").resolve())' "$rel"
  done
  ;;
run)
  name="$2" fixture="$3" driver="$4"
  [ -d "$fixture" ] || { echo "missing fixture $fixture; run: verify.sh fetch $fixture" >&2; exit 2; }
  mkdir -p "$RUNS"
  require_free $(( $(du -sm "$fixture" | cut -f1) + MIN_FREE_MB ))
  run="$RUNS/$(date +%Y%m%d-%H%M%S)-$name"
  mkdir -p "$run/evidence" "$run/work/tmp"
  cp -r --reflink=auto "$fixture" "$run/work/calc"
  cp "$driver" "$run/evidence/driver.py"
  touch "$run/.start"
  set +e
  TMPDIR="$run/work/tmp" CALC="$run/work/calc" EVIDENCE="$run/evidence" REPO="$REPO" MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true \
    PYTHONPATH="$REPO/.claude/skills/verify-pyprocar/scripts/lib${PYTHONPATH:+:$PYTHONPATH}" \
    pixi run -q -e default python "$run/evidence/driver.py" >"$run/evidence/run.log" 2>&1
  code=$?
  set -e
  echo "$code" >"$run/evidence/exit_code"
  # Side effects: files the run created or modified inside the calc copy.
  (cd "$run/work/calc" && find . -newer "$run/.start" -type f) >"$run/evidence/side_effects.txt"
  echo "run:        $run"
  echo "exit code:  $code"
  echo "evidence:"; ls -1 "$run/evidence" | sed 's/^/  /'
  echo "side effects in calc dir:"; sed 's/^/  /' "$run/evidence/side_effects.txt"
  exit "$code"
  ;;
clean)
  run="$(realpath "$2")"
  case "$run" in "$RUNS"/*) ;; *) echo "refusing: $run is not under $RUNS" >&2; exit 2 ;; esac
  scrub "$run"
  echo "removed scratch; evidence kept at $run/evidence"
  ;;
gc)
  hours="${2:-24}"
  [[ "$hours" =~ ^[0-9]+$ ]] || { echo "gc: hours must be a whole number, got $hours" >&2; exit 2; }
  [ -d "$RUNS" ] || exit 0
  find "$RUNS" -mindepth 2 -maxdepth 2 -name work -type d -mmin +$((hours * 60)) -print0 |
    while IFS= read -r -d '' work; do
      echo "removing $work"
      scrub "$(dirname "$work")"
    done
  echo "free under $RUNS: $(free_mb "$RUNS") MB"
  ;;
*)
  sed -n '2,7p' "${BASH_SOURCE[0]}"; exit 2 ;;
esac
