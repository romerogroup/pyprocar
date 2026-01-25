#!/usr/bin/env bash
set -euo pipefail

# Usage: run_silent_ruff_lint.sh [file_path] [--verbose|-v] [--src|--tests]
# If no file_path provided, runs on all files
# If --verbose/-v provided with a specific file, shows full output instead of summary
# If --src or --tests provided, uses respective config

file_path=""
verbose=false
config=""

# Parse arguments
for arg in "$@"; do
    case $arg in
        --verbose|-v)
            verbose=true
            ;;
        --src)
            config="--config $PIXI_PROJECT_ROOT/.config/.ruff-src.toml"
            ;;
        --tests)
            config="--config $PIXI_PROJECT_ROOT/.config/.ruff-tests.toml"
            ;;
        *)
            if [[ -z "$file_path" ]]; then
                file_path="$arg"
            fi
            ;;
    esac
done

# Only allow verbose for specific files, not directories
if $verbose && [[ -z "$file_path" || -d "$file_path" ]]; then
    verbose=false
fi

# Build the command
command="ruff check --fix --exit-non-zero-on-fix --force-exclude $config $file_path"

tmp_file=$(mktemp)

if eval "$command" > "$tmp_file" 2>&1; then
    printf "  ✓ linting\n"
    rm -f "$tmp_file"
    exit 0
else
    printf "  ✗ linting\n"

    if $verbose; then
        # Verbose mode: show full output
        cat "$tmp_file"
    else
        # Summary mode: extract total error count and first file
        total_errors=$(grep -oP 'Found \K[0-9]+(?= errors?)' "$tmp_file" || echo "0")

        if [ "$total_errors" -gt 0 ]; then
            printf "Found %s errors. Showing first file with issues:\n\n" "$total_errors"

            # Extract first file and its errors, grouped
            awk '
                /^[^:]+:[0-9]+:[0-9]+:/ {
                    if (!found) {
                        match($0, /^([^:]+):/, file_match)
                        first_file = file_match[1]
                        found = 1
                        print first_file ":"
                    }
                    if (index($0, first_file) == 1) {
                        # Remove filepath from line and print with indentation
                        sub(first_file ":", "", $0)
                        print "  " $0
                    }
                }
            ' "$tmp_file"
        else
            cat "$tmp_file"
        fi
    fi

    rm -f "$tmp_file"
    exit 0
fi
