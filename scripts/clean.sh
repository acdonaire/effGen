#!/usr/bin/env bash
################################################################################
# Repository cleanup — remove generated/test/cache artifacts from the tree.
#
# Everything removed here is already gitignored; this script just makes a tidy
# tree a one-liner so build/test runs don't leave litter behind.
#
#   scripts/clean.sh           # remove caches, coverage, build artifacts, runtime state
#   scripts/clean.sh --dry-run # show what would be removed without deleting
#
# Never touches tracked source, .git/, or the .env file.
################################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

run() {
    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "[clean] would run: $*"
    else
        eval "$@"
    fi
}

[ "${DRY_RUN}" -eq 1 ] && LABEL=" (dry-run)" || LABEL=""
echo "[clean] tidying ${REPO_ROOT}${LABEL}"

# Coverage + test/type/lint caches and build outputs.
run "rm -rf .coverage .coverage.* coverage.xml htmlcov .hypothesis \
    .pytest_cache .ruff_cache .mypy_cache .dmypy.json dist build *.egg-info \
    gitleaks-report.json"

# Runtime state written under the working directory.
run "rm -rf checkpoints .effgen"

# __pycache__ / *.pyc anywhere in the tree (never under .git or site-packages).
# Use `-exec rm` rather than `-delete`: `-delete` implies `-depth`, which
# silently disables the `./.git` prune (GNU find only warns, but bfs errors out
# and then deletes nothing). The prune + `-exec` form is the portable one.
run "find . -path ./.git -prune -o -type d -name __pycache__ \
    -not -path '*/site-packages/*' -exec rm -rf {} + 2>/dev/null || true"
run "find . -path ./.git -prune -o -type f -name '*.pyc' \
    -not -path '*/site-packages/*' -exec rm -f {} + 2>/dev/null || true"

echo "[clean] done"
