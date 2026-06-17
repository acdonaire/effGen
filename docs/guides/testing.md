# Testing, coverage, and the release install matrix

This guide explains how effGen's test suite is organized, how to run coverage
without corrupting data, the test markers and how they map to CI jobs, and the
release scripts that validate the install matrix on real hardware.

## Running the suite

```bash
# Fast offline lane (what most contributors run)
pytest tests/unit tests/integration -m "not gpu and not api and not live and not docker"

# A single directory or file
pytest tests/unit/test_cost_tracker.py -q
```

Coverage is **off by default**. The default pytest configuration deliberately
does not pass `--cov`, because enabling it globally:

1. made `pytest --collect-only` spin up the coverage machinery and write a
   `.coverage` file during pure collection, and
2. made two parallel pytest shards race the single default `.coverage` SQLite
   database, crashing a shard with `coverage.exceptions.DataError: ... no such
   table: meta`.

So enable coverage explicitly when you want it (see below). Because coverage is
no longer in the default config, plain `pytest --collect-only` is coverage-free.
(Don't combine `--cov` with `--collect-only` yourself — pytest-cov will still
instrument collection; there's no reason to measure coverage while only listing
tests.)

## Coverage

For a single-process report:

```bash
scripts/run_coverage.sh tests/unit
```

For a sharded, parallel report (each shard gets its own `COVERAGE_FILE`, then the
data is combined — no corruption):

```bash
scripts/run_coverage.sh --shards 4 tests/unit tests/integration
```

Both write `coverage.xml` + `htmlcov/` at the repo root. Under the hood,
`[tool.coverage.run]` sets `parallel = true` so each process writes a uniquely
suffixed data file, and the script runs `coverage combine` before reporting.

If you invoke pytest directly with coverage, do it per shard with a unique data
file and combine afterwards — never point two concurrent runs at the same
`.coverage`:

```bash
COVERAGE_FILE=.cov/a pytest tests/unit       --cov=effgen --cov-report= &
COVERAGE_FILE=.cov/b pytest tests/integration --cov=effgen --cov-report= &
wait
coverage combine .cov/a .cov/b && coverage report
```

## Markers and how they map to CI jobs

Declared in `pyproject.toml` (`[tool.pytest.ini_options].markers`). Markers under
`tests/security/` and `tests/deploy/` are applied automatically by directory in
`tests/conftest.py`, so `-m security` / `-m deployment` always select the right
suites.

| Marker | Selects | CI job |
|---|---|---|
| `unit` | pure unit tests | `test-unit` (matrix: 3.10–3.13) |
| `integration` | cross-module, no network/GPU | `test-integration` |
| `live` | real provider API calls (needs keys) | live-providers (opt-in) |
| `api` | needs a provider key set | live-providers (opt-in) |
| `gpu` | needs a CUDA GPU | gpu-local (self-hosted) |
| `docker` | needs a Docker daemon | sandbox/docker (full image) |
| `deployment` | Docker/Helm/Lambda/Cloudflare/probe recipes | deployment (full image) |
| `security` | sandbox, SSRF, secret-scan, vuln-audit, supply-chain | `security` |
| `slow` | long-but-bounded tests | nightly |
| `expensive` | soak/stress/large GPU runs | nightly/on-demand |
| `reliability` | retry/timeout/circuit/bulkhead primitives | `test-unit` |
| `fuzz` | hypothesis-driven fuzz harnesses | nightly |
| `cookbook` | docs/cookbook snippet smokes | docs |

Select or deselect with `-m`, e.g. `pytest -m "security"` or
`pytest -m "not gpu and not live"`.

## Order-independence (the single-process lane)

The per-marker CI jobs above each run **one directory** (`tests/unit`,
`tests/integration`, `tests/security`, …). That sharding is fast, but it hides a
class of bug: a test that only fails depending on **what ran before it** in the
same process. Two examples that slipped past the sharded jobs:

- a model test reset the global `ProviderRegistry` in its teardown without
  restoring it, so a later `tests/core` capability check saw an empty registry;
- a `tests/e2e` module put `tests/` on `sys.path` at import time, so a later
  `tests/deploy` lambda import resolved `deploy` to `tests/deploy/` instead of
  the repo-root `deploy/` namespace package.

Both **passed in isolation** and only failed in a full run. To catch this class,
run the whole offline suite in a **single process**:

```bash
# Order-independence guard — one process, whole offline suite
pytest tests -m "not gpu and not api and not live and not docker and not expensive" -q
```

This lane runs in CI as the `test-order-independence` job. Install
`pytest-randomly` (in the `dev` extra) to also shuffle the run order and surface
ordering coupling that a fixed order would hide:

```bash
pip install -e ".[dev]"
# Random order; the printed seed reproduces a failing order exactly
pytest tests -m "not gpu and not api and not live and not docker and not expensive" -p randomly -q
# Reproduce a specific order (use the seed printed in the failing run's header)
pytest tests -p randomly --randomly-seed=12345 -m "not gpu and not api and not live and not docker and not expensive"
```

The other lanes opt out of shuffling with `-p no:randomly` so their order stays
fixed and reproducible. Global isolation is enforced in `tests/conftest.py`: an
autouse fixture restores the full `ProviderRegistry` before every test, so a
teardown that empties it can no longer poison later tests.

### "Green offline" is not "release ready"

A green `pytest -m "not live and not gpu"` run only proves the offline surface.
It intentionally **skips** Docker, ffmpeg, Tesseract OCR, gitleaks, `pip-audit`,
live-provider, and GPU tests. A release must additionally run the live, GPU,
security, and deployment lanes — see the full CI image below.

## Install-matrix scripts

These build the wheel, create a **fresh** virtual environment, install from the
wheel (not the editable tree), and run the same battery — `pip check`, import
smoke, CLI smoke, and minimal inference. No release should ship unless each of
these passes for its lane.

| Script | Lane | Proves |
|---|---|---|
| `scripts/check_install_cpu.sh` | CPU base + `[local]` | clean install + CPU inference anywhere |
| `scripts/check_install_cu124.sh` | CUDA 12.4 + `[local]` | `torch.cuda.is_available()` True + tiny model on GPU |
| `scripts/check_install_vllm_cu124.sh` | `[vllm]` | vLLM/torch ABI is internally consistent + imports |
| `scripts/check_install_server.sh` | `[server]` | server boots, probes 200, auth on by default |

Common knobs (see `scripts/_install_matrix_common.sh`):
`MATRIX_PYTHON` (interpreter, ≥3.10), `MATRIX_WHEEL` (reuse a pre-built wheel),
`MATRIX_KEEP=1` (keep the scratch dir for debugging).

## GPU shard runner

`scripts/gpu_shard_runner.sh` fans a GPU test selection across the host's free
GPUs. It reads `nvidia-smi`, treats a GPU as free below `EFFGEN_GPU_MEM_MB` MiB
used (default 1000) so it never overcommits a busy GPU, splits the selected tests
round-robin into one shard per GPU, pins each with `CUDA_VISIBLE_DEVICES`, runs
them in parallel with a unique per-shard `COVERAGE_FILE`, and kills any orphan
GPU processes it spawned.

```bash
scripts/gpu_shard_runner.sh                    # runs `-m gpu` across free GPUs
scripts/gpu_shard_runner.sh tests/e2e          # shard a directory
EFFGEN_SHARD_COV=1 scripts/gpu_shard_runner.sh # also collect + combine coverage
EFFGEN_MAX_SHARDS=4 scripts/gpu_shard_runner.sh
```

## Release artifacts

`scripts/release_artifacts.sh [OUTPUT_DIR]` captures the dependency and
environment diagnostics that should accompany every release: `pip freeze`,
`pip check`, `pip-audit`, `nvidia-smi`, torch CUDA diagnostics, and a
provider-readiness summary (key **presence** and SDK importability only — never
secret values).

## The full CI image

The offline lane runs on a stock runner. The live/security/deployment lanes need
a richer image. A "full CI image" should provide:

- **Docker** daemon access (sandbox + deployment recipe tests)
- **ffmpeg** (audio/video tools)
- **Tesseract** + language packs (OCR tools)
- **gitleaks** (secret-scan tests under `tests/security/`)
- **pip-audit** (dependency vulnerability gate)
- sample **media fixtures** for the multimodal/OCR suites

When any of these is missing, the corresponding tests skip with a clear reason;
the release job must treat those skips as failures (no skipped required checks).
