#!/usr/bin/env bash
set -euo pipefail

# Usage: run_silent_typecheck.sh [file_path] [--verbose|-v]
# If no file_path provided, runs on all files
# If --verbose/-v provided with a specific file, shows full output instead of summary

file_path=""
verbose=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --verbose|-v)
            verbose=true
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
command="basedpyright --project pyrightconfig.json"
if [[ -n "$file_path" ]]; then
    command="$command $file_path"
fi

tmp_file=$(mktemp)

if eval "$command" > "$tmp_file" 2>&1; then
    printf "  ✓ type checking\n"
    rm -f "$tmp_file"
    exit 0
else
    printf "  ✗ type checking\n"

    if $verbose; then
        # Verbose mode: show full output
        cat "$tmp_file"
    else
        # Summary mode: extract and show summary with first error
        summary=$(grep -E '^[0-9]+ errors?, [0-9]+ warnings?, [0-9]+ notes?$' "$tmp_file" || true)

        if [[ -n "$summary" ]]; then
            errors=$(echo "$summary" | sed -E 's/^([0-9]+) errors?.*/\1/')
            warnings=$(echo "$summary" | sed -E 's/.*[^0-9]([0-9]+) warnings?.*/\1/')
            notes=$(echo "$summary" | sed -E 's/.*[^0-9]([0-9]+) notes?$/\1/')

            # Extract first error with its continuation line
            first_error=$(awk '
                /^\s*\/.*:[0-9]+:[0-9]+ - error:/ {
                    if (found_first == 0) {
                        found_first = 1
                        first_error = $0
                        next
                    }
                    if (found_first) {
                        print first_error
                        printed = 1
                        exit
                    }
                }
                found_first && /^\s+[^\/]/ {
                    first_error = first_error "\n" $0
                    print first_error
                    printed = 1
                    exit
                }
                END {
                    if (found_first && first_error && printed == 0) {
                        print first_error
                    }
                }
            ' "$tmp_file")

            printf "%s errors, %s warnings, %s notes. Only showing first error:\n" "$errors" "$warnings" "$notes"
            if [[ -n "$first_error" ]]; then
                echo "$first_error"
            fi
        else
            # No summary found, show raw output
            cat "$tmp_file"
        fi
    fi

    rm -f "$tmp_file"
    exit 0
fi
