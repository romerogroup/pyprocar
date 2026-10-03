#!/usr/bin/env bash
# Harness for verifying pyprocar end to end. See ../SKILL.md.
#   verify.sh doctor
#   verify.sh fetch <relpath>...            e.g. data/examples/bands/non-spin-polarized
#   verify.sh run <name> <fixture-relpath> <driver.py>
#   verify.sh clean <run-dir>
set -euo pipefail

REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
RUNS="$REPO/data/verify-runs"
abs() { (cd "$(dirname "$1")" && echo "$PWD/$(basename "$1")"); }
[ "${1:-}" = run ] && [ $# -ge 4 ] && set -- "$1" "$2" "$3" "$(abs "$4")"
cd "$REPO"

py() { pixi run -q -e default python "$@"; }

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
  run="$RUNS/$(date +%Y%m%d-%H%M%S)-$name"
  mkdir -p "$run/evidence" "$run/work/tmp"
  cp -r "$fixture" "$run/work/calc"
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
  rm -rf "$run/work" "$run/.start"
  echo "removed scratch; evidence kept at $run/evidence"
  ;;
*)
  sed -n '2,6p' "${BASH_SOURCE[0]}"; exit 2 ;;
esac
