#!/usr/bin/env bash

set -u

# --- Load Profile ---
ci_profile="${CI_PRECOMMIT_PROFILE:-}"

if [ -z "$ci_profile" ] && [ -f .env ]; then
    ci_profile="$(
        grep -E '^[[:space:]]*(export[[:space:]]+)?CI_PRECOMMIT_PROFILE[[:space:]]*=' .env 2>/dev/null \
            | tail -n 1 \
            | cut -d '=' -f 2- \
            | sed -e 's/#.*$//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'$/\1/" \
            | tr -d '\r' \
            || true
    )"
fi

ci_profile="${ci_profile:-False}"

echo "Using precommit profile: $ci_profile"

# --- Return on default profile ---
case "$ci_profile" in
    [Tt][Rr][Uu][Ee]|1) ;;
    *) exit 0 ;;
esac

uv sync --frozen

targets=()
for f in "$@"; do
    if [ -f "$f" ]; then
        case "$f" in
            *.py) targets+=("$f") ;;
        esac
    fi
done
if [ "${#targets[@]}" -eq 0 ]; then
    targets=("./src")
fi

exec uv run pylint -j 0 --output-format=colorized "${targets[@]}"
