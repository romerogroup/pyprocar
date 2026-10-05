#!/usr/bin/env bash
# Harness for verifying pyprocar end to end. See ../SKILL.md.
#   verify.sh doctor
#   verify.sh fetch <relpath>...            download if missing, then make read-only
#   verify.sh run <name> <fixture-relpath> <driver.py>
#   verify.sh clean <run-dir>
#   verify.sh gc [hours]                    remove work/ of runs older than hours (default 24)
#   verify.sh worktree-setup                link data/, copy _version.py into a linked worktree
#   verify.sh exec <cmd>...                 run a command (a CI gate) in the project env
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
MAIN="$(dirname "$(git -C "$REPO" rev-parse --path-format=absolute --git-common-dir)")"
SHARED_ENV="$MAIN/.pixi/envs/dev/bin"
abs() { (cd "$(dirname "$1")" && echo "$PWD/$(basename "$1")"); }
[ "${1:-}" = run ] && [ $# -ge 4 ] && set -- "$1" "$2" "$3" "$(abs "$4")"
cd "$REPO"

# A linked worktree borrows the main checkout's env, because pixi would build a multi-GB env per worktree.
in_worktree() { [ "$REPO" != "$MAIN" ]; }
shared_env() {
  PATH="$SHARED_ENV:$PATH" PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" "$@"
}
py() { if in_worktree; then shared_env python "$@"; else pixi run -q --locked -e default python "$@"; fi; }
require_shared_env() {
  [ -x "$SHARED_ENV/python" ] || { echo "missing $SHARED_ENV/python; run 'pixi install -e dev' in $MAIN" >&2; exit 2; }
}
case "${1:-}" in doctor | run | exec | worktree-setup) ! in_worktree || require_shared_env ;; esac

whole_number() {
  [[ "$2" =~ ^[0-9]+$ ]] || { echo "$1 must be a whole number, got '$2'" >&2; exit 2; }
  echo $((10#$2))
}

free_mb() {
  local dir="$1"
  while [ ! -e "$dir" ]; do dir="$(dirname "$dir")"; done
  df -Pk "$dir" | awk 'NR == 2 { print int($4 / 1024) }'
}

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

PLAIN_NAME='[A-Za-z0-9_][A-Za-z0-9_.-]*'

physical_dir() {
  local dir
  dir="$(CDPATH='' cd -P -- "$1" 2>/dev/null && pwd -P)" && [[ "$dir" =~ ^/+(.*)$ ]] || return 1
  printf '/%s\n' "${BASH_REMATCH[1]}"
}

real_path() {
  local head="$1" tail="" name="" dir out
  while [ ! -e "$head" ]; do
    [ ! -L "$head" ] || return 1
    tail="/${head##*/}$tail" head="$(dirname "$head")"
  done
  if [ ! -d "$head" ]; then
    { [ -L "$head" ] || [ -n "$tail" ]; } && return 1
    name="/${head##*/}" head="$(dirname "$head")"
  fi
  dir="$(physical_dir "$head")" || return 1
  out="${dir%/}$name$tail"
  printf '%s\n' "${out:-/}"
}

is_below() {
  local child="$1" parent="$2" c p
  [[ "$child" =~ ^(/[^/]+)+$ && "$parent" =~ ^(/[^/]+)+$ ]] || return 1
  while [ -n "$parent" ]; do
    p="${parent#/}" c="${child#/}"
    [ "${p%%/*}" = "${c%%/*}" ] || return 1
    parent="${p#"${p%%/*}"}" child="${c#"${c%%/*}"}"
  done
  [ -n "$child" ]
}

DATA="" RUNS=""
require_data() {
  local main
  if main="$(physical_dir "$MAIN")" && DATA="$(real_path data)" && [ "$DATA" = "${main%/}/data" ] &&
    RUNS="$(real_path "$DATA/verify-runs")" && [ "$RUNS" = "$DATA/verify-runs" ]; then
    return 0
  fi
  echo "refusing: $REPO/data must be the directory $MAIN/data, or in a linked worktree a symlink to it" \
    "(run: verify.sh worktree-setup), and data/verify-runs a directory in it" >&2
  return 2
}

is_fixture_rel() { case "$1/" in examples/*/*/*) ;; examples/* | verify-runs/*) return 1 ;; esac; }

fixture_path() {
  local rel="${1%/}" real
  [[ "$rel" =~ ^data(/$PLAIN_NAME)+$ ]] && real="$(real_path "$rel")" && is_below "$real" "$DATA" &&
    is_fixture_rel "${real#"$DATA"/}" && echo "$real"
}

require_resolved_below() {
  local real
  real="$(real_path "$1")" && [ "$real" = "$1" ] && is_below "$real" "$2" ||
    { echo "refusing: $1 is not a resolved path below $2" >&2; exit 2; }
}

chmod_below_data() {
  require_resolved_below "$2" "$DATA"
  find -P "$2" ! -type l \( -type d -o -links 1 \) -exec chmod "$1" {} +
}

fixture_report() (
  shopt -s nullglob dotglob
  for d in data/examples/*/* data/*; do
    is_fixture_rel "${d#data/}" || continue
    if real="$(fixture_path "$d")"; then
      find -P "$real" ! -type l \( -perm -200 -o -perm -020 -o -perm -002 \) | sed 's/^/writable /'
    else
      printf 'unlockable %s\n' "$d"
    fi
  done | LC_ALL=C sort -u
)

print_list() {
  local label="$1" lines="$2" limit="$3" hint="$4" n
  n="$(printf '%s' "$lines" | grep -c . || true)"
  echo "$label: $n"
  [ "$n" -eq 0 ] || { printf '%s\n' "$lines" | sed -n "1,${limit}s/^/  /p"; echo "  $hint"; }
}

scrub() {
  require_resolved_below "$1" "$RUNS"
  [ ! -d "$1/work" ] || [ -L "$1/work" ] || chmod_below_data u+w "$1/work"
  rm -rf "$1/work" "$1/.start" "$1/.pid"
}

live() { [ -f "$1/.pid" ] && ps -p "$(cat "$1/.pid")" >/dev/null 2>&1; }

case "${1:-}" in
doctor)
  echo "branch:  $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
  echo "dirty:   $(GIT_OPTIONAL_LOCKS=0 git status --porcelain --untracked-files=no | wc -l) tracked file(s) modified"
  py -W ignore -c '
import sys, pyprocar, matplotlib, pyvista, vtk
print("python: ", sys.version.split()[0])
print("pyprocar:", pyprocar.__version__, "from", pyprocar.__file__)
print("pyvista: ", pyvista.__version__, "vtk", vtk.vtkVersion.GetVTKVersion())
'
  echo "fixtures:"; ls -d data/examples/*/* 2>/dev/null | sed 's/^/  /' || echo "  (none; run: verify.sh fetch <relpath>)"
  require_data
  report="$(fixture_report)"
  print_list "writable fixture paths" "$(printf '%s\n' "$report" | sed -n 's/^writable //p')" 5 \
    "lock each fixture with: verify.sh fetch <relpath>"
  print_list "unlockable fixture roots" "$(printf '%s\n' "$report" | sed -n 's/^unlockable //p')" '$' \
    "fetch refuses these: rename each to plain names inside data/, or remove it"
  ;;
fetch)
  shift
  require_data
  for rel in "$@"; do
    rel="${rel%/}"
    if ! [[ "$rel" =~ ^data/(examples/[^/]+/)?[^/]+$ ]] || ! real="$(fixture_path "$rel")"; then
      echo "refusing: $rel is not data/examples/<category>/<name> or data/<name> with plain names, inside data/" >&2
      exit 2
    fi
    if [ ! -e "$real" ]; then
      ! in_worktree || require_shared_env
      HF_XET_LOG_DIR_DISABLE_CLEANUP=1 py -W ignore -c 'import sys; from pathlib import Path; import pyprocar
pyprocar.download_from_hf(relpath=sys.argv[1], output_path=Path(".").resolve())' "$rel"
      real="$(fixture_path "$rel")" && [ -e "$real" ] || { echo "the download did not create $rel" >&2; exit 1; }
    fi
    chmod_below_data a-w "$real"
    echo "read-only: $rel"
    linked="$(find -P "$real" ! -type l ! -type d -links +1)"
    [ -z "$linked" ] || printf 'left writable, hard-linked (another link may be outside data/):\n%s\n' "$linked" >&2
  done
  ;;
run)
  name="$2" driver="$4"
  require_data
  [[ "$name" =~ ^$PLAIN_NAME$ ]] || { echo "refusing: run name '$name' is not one plain name" >&2; exit 2; }
  if ! fixture="$(fixture_path "$3")" || [ ! -d "$fixture" ]; then
    echo "refusing: $3 is not a fixture directory inside data/; get one with: verify.sh fetch <relpath>" >&2
    exit 2
  fi
  MIN_FREE_MB="$(whole_number VERIFY_MIN_FREE_MB "${VERIFY_MIN_FREE_MB:-2048}")"
  mkdir -p "$RUNS"
  require_free $(( $(du -sm "$fixture" | cut -f1) + MIN_FREE_MB ))
  run="$RUNS/$(date +%Y%m%d-%H%M%S)-$name"
  mkdir "$run" ||
    { echo "refusing: $run already exists; rerun to get the next second's name" >&2; exit 2; }
  require_resolved_below "$run" "$RUNS"
  mkdir "$run/evidence" "$run/work" "$run/work/tmp"
  (set -C; echo $$ >"$run/.pid")
  trap 'rm -f "$run/.pid"' EXIT
  cp -RL --reflink=auto "$fixture" "$run/work/calc"
  chmod_below_data u+w "$run/work/calc"
  cp "$driver" "$run/evidence/driver.py"
  touch "$run/.start"
  set +e
  TMPDIR="$run/work/tmp" CALC="$run/work/calc" EVIDENCE="$run/evidence" REPO="$REPO" MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true \
    PYTHONPATH="$REPO/.claude/skills/verify-pyprocar/scripts/lib${PYTHONPATH:+:$PYTHONPATH}" \
    py "$run/evidence/driver.py" >"$run/evidence/run.log" 2>&1
  code=$?
  set -e
  echo "$code" >"$run/evidence/exit_code"
  # Side effects: files the run created or modified inside the calc copy.
  (cd "$run/work/calc" && find . -newer "$run/.start" -type f) \
    >"$run/evidence/side_effects.txt"
  echo "run:        $run"
  echo "exit code:  $code"
  echo "evidence:"; ls -1 "$run/evidence" | sed 's/^/  /'
  echo "side effects in calc dir:"; sed 's/^/  /' "$run/evidence/side_effects.txt"
  exit "$code"
  ;;
clean)
  require_data
  run="$(real_path "$2")" && is_below "$run" "$RUNS" && [ -d "$run" ] && [[ "${run#"$RUNS"/}" =~ ^$PLAIN_NAME$ ]] ||
    { echo "refusing: $2 is not a run directory in $RUNS" >&2; exit 2; }
  scrub "$run"
  echo "removed scratch; evidence kept at $run/evidence"
  ;;
gc)
  hours="$(whole_number "gc hours" "${2:-24}")"
  require_data
  [ -d "$RUNS" ] || exit 0
  find "$RUNS" -mindepth 2 -maxdepth 2 -name work -type d -mmin +$((hours * 60)) -print0 |
    while IFS= read -r -d '' work; do
      run="$(dirname "$work")"
      if live "$run"; then echo "keeping $work (run in progress, pid $(cat "$run/.pid"))"; continue; fi
      echo "removing $work"
      scrub "$run"
    done
  echo "free under $RUNS: $(free_mb "$RUNS") MB"
  ;;
worktree-setup)
  in_worktree || { echo "refusing: $REPO is the main checkout; run this in a linked worktree" >&2; exit 2; }
  [ -e "$MAIN/pyprocar/_version.py" ] ||
    { echo "missing $MAIN/pyprocar/_version.py; run 'pixi install -e dev' in $MAIN" >&2; exit 2; }
  if [ -e data ] && [ ! -L data ]; then
    echo "refusing: $REPO/data is a real directory; move it out of the way and rerun" >&2; exit 2
  fi
  cp "$MAIN/pyprocar/_version.py" pyprocar/_version.py
  mkdir -p "$MAIN/data" .tmp
  ln -sfn "$MAIN/data" data
  base="$(git merge-base HEAD origin/dev 2>/dev/null || true)"
  if [ -z "$base" ]; then
    echo "warning: no merge base with origin/dev; run 'git fetch origin dev' to check pixi.lock" >&2
  elif ! git diff --quiet "$base" -- pixi.lock pixi.toml; then
    echo "warning: this branch changes pixi.lock or pixi.toml relative to origin/dev; the shared env does not match it, so build this worktree's own env with 'pixi run --locked'" >&2
  elif ! cmp -s "$MAIN/pixi.lock" pixi.lock; then
    echo "warning: the main checkout's pixi.lock differs from this branch's, which matches origin/dev; local gates may drift, so CI is the final gate" >&2
  fi
  echo "ready: data -> $MAIN/data, pyprocar/_version.py copied, env $SHARED_ENV"
  ;;
exec)
  shift
  if in_worktree; then
    mkdir -p .tmp
    TMPDIR="$REPO/.tmp" shared_env "$@"
  else
    pixi run -q --locked -e dev "$@"
  fi
  ;;
*)
  sed -n '2,9p' "${BASH_SOURCE[0]}"; exit 2 ;;
esac
