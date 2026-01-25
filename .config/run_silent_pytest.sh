#!/usr/bin/env bash
set -euo pipefail

# Usage: run_silent_pytest.sh [test_pattern] [--verbose|-v]
# If no test_pattern provided, runs all tests
# If --verbose/-v provided, shows full output instead of summary

test_pattern=""
verbose=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --verbose|-v)
            verbose=true
            ;;
        *)
            if [[ -z "$test_pattern" ]]; then
                test_pattern="$arg"
            fi
            ;;
    esac
done

# Build the command
command="pytest -s --numprocesses=auto tests --config-file $PIXI_PROJECT_ROOT/.config/.pytest.toml"
if [[ -n "$test_pattern" ]]; then
    command="$command -k '$test_pattern'"
fi

tmp_file=$(mktemp)

if eval "$command" > "$tmp_file" 2>&1; then
    printf "  ✓ running tests\n"
    rm -f "$tmp_file"
    exit 0
else
    exit_code=$?
    printf "  ✗ running tests\n"

    if $verbose; then
        # Verbose mode: show full output
        cat "$tmp_file"
    else
        # Summary mode: extract stats and first failure
        awk '
            /workers \[[0-9]+ items\]/ {
                match($0, /\[([0-9]+) items\]/, items)
                total = items[1]
            }
            /^=+ .+ failed.*passed/ {
                match($0, /([0-9]+) failed, ([0-9]+) passed/, stats)
                failed = stats[1]
                passed = stats[2]
            }
            /stopping after [0-9]+ failures/ {
                match($0, /stopping after ([0-9]+) failures/, stop)
                stopped = stop[1]
            }
            END {
                if (total && failed && passed) {
                    printf "%s total tests. Stopped after %s failures. %s failed, %s passed.\n\n", total, stopped ? stopped : failed, failed, passed
                    printf "1st failure:\n\n"
                }
            }
        ' "$tmp_file"

        # Extract and format first failure
        gawk '
            /^_+ .+ _+$/ {
                if (found) exit
                found = 1
                # Extract test name between underscores
                gsub(/^_+ /, "")
                gsub(/ _+$/, "")
                test_name = $0
                next
            }
            /^=+ short test summary/ { exit }
            /^\[gw[0-9]+\]/ { next }
            # Capture first test file location (tests/*.py:line:)
            found && !file_path && /^tests\/.*\.py:[0-9]+:/ {
                split($0, parts, ":")
                file_path = parts[1]
                line_num = parts[2]
                next
            }
            # Capture the E error line (comes after location in parallel mode)
            found && /^E +[A-Za-z]+[A-Za-z]:/ {
                # Remove leading "E" and spaces
                line = $0
                sub(/^E +/, "", line)
                # Split on first colon
                idx = index(line, ":")
                error_type = substr(line, 1, idx - 1)
                error_msg = substr(line, idx + 2)

                # Remove path before src/ in parentheses
                gsub(/\([^)]*\/src\//, "(src/", error_msg)

                printf "%s (%s:%s) - %s: %s\n", test_name, file_path, line_num, error_type, error_msg
                exit
            }
        ' "$tmp_file"
    fi

    rm -f "$tmp_file"
    exit 0
fi
