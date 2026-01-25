#!/usr/bin/env bash
set -euo pipefail

# Usage: run_silent_ruff_format.sh [file_path] [--verbose|-v] [--src|--tests]
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
command="ruff format --exit-non-zero-on-format --force-exclude $config $file_path"

tmp_file=$(mktemp)

if eval "$command" > "$tmp_file" 2>&1; then
    printf "  ✓ formatting\n"
    rm -f "$tmp_file"
    exit 0
else
    printf "  ✗ formatting\n"

    if $verbose; then
        # Verbose mode: show full output
        cat "$tmp_file"
    else
        # Summary mode: show count and first file
        file_count=$(grep -c 'Would reformat:' "$tmp_file" 2>/dev/null || wc -l < "$tmp_file")

        if [ "$file_count" -gt 0 ]; then
            printf "%s files need formatting. Showing first:\n\n" "$file_count"
            head -5 "$tmp_file"
        else
            cat "$tmp_file"
        fi
    fi

    rm -f "$tmp_file"
    exit 0
fi
