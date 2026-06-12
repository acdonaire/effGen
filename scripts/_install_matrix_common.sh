#!/usr/bin/env bash
################################################################################
# Shared helpers for the install-matrix check scripts
# (check_install_cpu.sh / check_install_cu124.sh / check_install_vllm_cu124.sh /
#  check_install_server.sh).
#
# Each matrix script creates a FRESH virtual environment, installs effGen from a
# built wheel (not the editable tree), and then runs the same battery of checks:
#   pip check  ->  import smoke  ->  CLI smoke  ->  minimal inference.
#
# This file is sourced, never run directly. It must stay POSIX-bash and not
# depend on the repo being installed.
################################################################################

set -euo pipefail

# Resolve the repo root from this file's location (scripts/ -> repo root).
MATRIX_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${MATRIX_SCRIPT_DIR}/.." && pwd)"

# Colors (fall back to empty when not a TTY).
if [ -t 1 ]; then
    C_GREEN='\033[0;32m'; C_RED='\033[0;31m'; C_YELLOW='\033[1;33m'
    C_BLUE='\033[0;34m'; C_BOLD='\033[1m'; C_NC='\033[0m'
else
    C_GREEN=''; C_RED=''; C_YELLOW=''; C_BLUE=''; C_BOLD=''; C_NC=''
fi

log()    { printf "${C_BLUE}[matrix]${C_NC} %s\n" "$*"; }
ok()     { printf "${C_GREEN}[ ok  ]${C_NC} %s\n" "$*"; }
warn()   { printf "${C_YELLOW}[warn ]${C_NC} %s\n" "$*"; }
fail()   { printf "${C_RED}[fail ]${C_NC} %s\n" "$*" >&2; }

# Python interpreter used to BUILD the wheel and CREATE venvs. Override with
# MATRIX_PYTHON=/path/to/python3.11. Must be >= 3.10.
MATRIX_PYTHON="${MATRIX_PYTHON:-python3}"

# Where to keep the scratch env + built wheel. Defaults to a fresh mktemp dir
# that is removed on exit unless MATRIX_KEEP=1.
MATRIX_WORKDIR="${MATRIX_WORKDIR:-$(mktemp -d -t effgen-matrix-XXXXXX)}"
VENV_DIR="${MATRIX_WORKDIR}/venv"
WHEEL_DIR="${MATRIX_WORKDIR}/wheel"
VENV_PY="${VENV_DIR}/bin/python"

cleanup_matrix() {
    local rc=$?
    if [ "${MATRIX_KEEP:-0}" = "1" ]; then
        warn "MATRIX_KEEP=1 — leaving workdir ${MATRIX_WORKDIR}"
    else
        rm -rf "${MATRIX_WORKDIR}" 2>/dev/null || true
    fi
    return $rc
}
trap cleanup_matrix EXIT

require_python_310() {
    if ! "${MATRIX_PYTHON}" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3,10) else 1)'; then
        fail "${MATRIX_PYTHON} is < 3.10; set MATRIX_PYTHON to a 3.10+ interpreter."
        exit 2
    fi
}

# Build the wheel ONCE into WHEEL_DIR (or reuse one passed via MATRIX_WHEEL).
build_wheel() {
    if [ -n "${MATRIX_WHEEL:-}" ]; then
        log "Using pre-built wheel: ${MATRIX_WHEEL}"
        EFFGEN_WHEEL="${MATRIX_WHEEL}"
        return 0
    fi
    log "Building effGen wheel from ${REPO_ROOT} ..."
    mkdir -p "${WHEEL_DIR}"
    "${MATRIX_PYTHON}" -m pip install --quiet --upgrade build >/dev/null 2>&1 || true
    ( cd "${REPO_ROOT}" && "${MATRIX_PYTHON}" -m build --wheel --outdir "${WHEEL_DIR}" ) \
        >"${MATRIX_WORKDIR}/build.log" 2>&1 || {
            fail "wheel build failed; see ${MATRIX_WORKDIR}/build.log"; tail -20 "${MATRIX_WORKDIR}/build.log"; exit 1; }
    EFFGEN_WHEEL="$(ls -1 "${WHEEL_DIR}"/effgen-*.whl | head -1)"
    [ -n "${EFFGEN_WHEEL}" ] || { fail "no wheel produced"; exit 1; }
    ok "Built wheel: $(basename "${EFFGEN_WHEEL}")"
}

create_venv() {
    log "Creating fresh venv at ${VENV_DIR} ..."
    "${MATRIX_PYTHON}" -m venv "${VENV_DIR}"
    "${VENV_PY}" -m pip install --quiet --upgrade pip setuptools wheel >/dev/null 2>&1
    ok "venv ready ($("${VENV_PY}" --version 2>&1))"
}

# install_torch <index>  e.g. install_torch cpu | install_torch cu124
install_torch() {
    local index="$1"
    log "Installing torch from index ${index} ..."
    "${VENV_PY}" -m pip install --quiet "torch>=2.0,<3" \
        --index-url "https://download.pytorch.org/whl/${index}" \
        >"${MATRIX_WORKDIR}/torch.log" 2>&1 || {
            fail "torch (${index}) install failed; see ${MATRIX_WORKDIR}/torch.log"; exit 1; }
    ok "torch installed (${index})"
}

# install_wheel "<extras>"   e.g. install_wheel "[local]"  (empty for base)
install_wheel() {
    local extras="${1:-}"
    log "Installing effGen wheel ${extras:-(base)} ..."
    "${VENV_PY}" -m pip install --quiet "${EFFGEN_WHEEL}${extras}" \
        >"${MATRIX_WORKDIR}/install.log" 2>&1 || {
            fail "wheel install failed; see ${MATRIX_WORKDIR}/install.log"; tail -25 "${MATRIX_WORKDIR}/install.log"; exit 1; }
    ok "effGen installed from wheel"
}

# ---- the shared check battery -------------------------------------------------

check_pip_check() {
    log "pip check ..."
    if "${VENV_PY}" -m pip check >"${MATRIX_WORKDIR}/pipcheck.log" 2>&1; then
        ok "pip check clean"
    else
        fail "pip check reported conflicts:"; cat "${MATRIX_WORKDIR}/pipcheck.log"; return 1
    fi
}

check_import_smoke() {
    log "import smoke ..."
    "${VENV_PY}" - <<'PY'
import effgen
from effgen import Agent, load_model, create_agent  # core surface
from effgen.tools import get_registry
n = len(get_registry().list_tools())
assert n > 0, "tool registry empty after lazy discovery"
print(f"effgen {effgen.__version__} import OK; tools discovered: {n}")
PY
    ok "import smoke OK"
}

check_cli_smoke() {
    log "CLI smoke ..."
    "${VENV_DIR}/bin/effgen" --version >/dev/null
    "${VENV_DIR}/bin/effgen" --help    >/dev/null
    "${VENV_DIR}/bin/effgen" doctor     >"${MATRIX_WORKDIR}/doctor.log" 2>&1 || true
    ok "CLI smoke OK ($("${VENV_DIR}/bin/effgen" --version 2>&1 | head -1))"
}

# Minimal local inference on a tiny model. Skips gracefully if torch/cuda absent
# and the caller asked for a GPU run. cpu runs always attempt the tiny model.
check_minimal_inference() {
    local require_gpu="${1:-0}"
    log "minimal inference (tiny model) ..."
    MATRIX_REQUIRE_GPU="${require_gpu}" "${VENV_PY}" - <<'PY'
import os, sys
require_gpu = os.environ.get("MATRIX_REQUIRE_GPU") == "1"
try:
    import torch
    has_cuda = torch.cuda.is_available()
except Exception as e:  # noqa: BLE001
    print(f"torch import failed: {e}"); sys.exit(1 if require_gpu else 0)
if require_gpu and not has_cuda:
    print("FAIL: GPU run requested but torch.cuda.is_available() is False")
    sys.exit(1)
print(f"torch.cuda.is_available() = {has_cuda}")
from effgen import load_model
model = load_model("Qwen/Qwen2.5-0.5B-Instruct", engine="transformers")
out = model.generate("Reply with the single word: ready", max_tokens=8, temperature=0.0)
text = getattr(out, "text", out)
print(f"inference output: {text!r}")
assert text and str(text).strip(), "empty inference output"
PY
    ok "minimal inference OK"
}

print_header() {
    printf "\n${C_BOLD}=== effGen install-matrix: %s ===${C_NC}\n" "$1"
    log "workdir: ${MATRIX_WORKDIR}"
}

print_footer() {
    printf "\n${C_GREEN}${C_BOLD}PASS${C_NC} — %s install matrix check complete.\n" "$1"
}
