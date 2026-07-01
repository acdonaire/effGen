# Supply-Chain Security

effGen applies defence-in-depth to its software supply chain: pinned
dependencies with hashes, automated vulnerability scanning, and a runtime
integrity check that warns operators if installed packages drift from the
locked set.

---

## Dependency lock file

`requirements-lock.txt` is a hash-pinned snapshot of every transitive
dependency.  It is generated with [`pip-compile`](https://pip-tools.readthedocs.io/):

```bash
pip install pip-tools
pip-compile \
    --generate-hashes \
    --strip-extras \
    --output-file requirements-lock.txt \
    pyproject.toml
```

### Reproducible installs

```bash
# Install using exact pinned versions and verified hashes
pip install effgen -r requirements-lock.txt
```

> **Note:** Editable installs (`pip install -e .`) do not support hash
> checking.  Use a built wheel for production/CI deployments.

### Regenerating the lockfile

When you update a dependency in `pyproject.toml`, regenerate the lockfile
before opening a pull request:

```bash
pip-compile --generate-hashes --strip-extras \
    --output-file requirements-lock.txt pyproject.toml
```

The `deps-audit.yml` CI workflow runs `pip-audit` against the lockfile on
every push to `main` and daily at 06:00 UTC.

---

## Vulnerability scanning (`pip-audit`)

effGen uses [`pip-audit`](https://github.com/pypa/pip-audit) to detect CVEs
in installed packages.

```bash
pip install pip-audit
pip-audit --format json --output pip-audit-report.json
```

### CI gate (`.github/workflows/deps-audit.yml`)

The `deps-audit` workflow:

1. Installs effGen core + dev extras.
2. Runs `pip-audit --format json`.
3. Fails the build if any **non-exempt** package has a known CVE.
4. Uploads the full report as a CI artifact (retained 30 days).

**Exempt packages** (documented in `tests/security/test_vuln_audit.py`):

| Package | Reason |
|---------|--------|
| `pip` / `setuptools` / `wheel` | Packaging toolchain, not shipped at runtime |
| `py` / `pytest-forked` | Dev-only test utilities |
| `vllm` / `torch` / `transformers` | Optional extras; upstream CVEs in non-default codepaths |
| `xgrammar` | Optional vLLM dependency |
| `diskcache` | No upstream fix for CVE-2025-69872; effGen does not use pickle deserialization paths |

---

## Runtime integrity verification (`EFFGEN_VERIFY_HASHES=1`)

On startup, if `EFFGEN_VERIFY_HASHES=1` is set, effGen compares every
installed package's version against the pinned version in
`requirements-lock.txt`.  The result is logged on the `effgen.security.supply_chain`
logger:

- A clean environment emits a single `hash_verification: ok` line.
- Drift emits `hash_verification: drift` naming the affected packages, **and**
  raises a `HashDriftWarning`.

The application continues running either way — the operator is alerted but
startup is never blocked.

```bash
EFFGEN_VERIFY_HASHES=1 python -c "import effgen"
```

The build-fingerprint snapshot used to detect silent wheel substitution is
stored at `~/.effgen/supply_chain/installed_hashes.json` (created on the first
run with the env var set).

### What is checked

1. **Version pinning** — the installed version must exactly match the pinned
   version in `requirements-lock.txt`.
2. **Build fingerprint** — a SHA-256 fingerprint of each distribution's
   `WHEEL` metadata file is recorded on the first run and compared on
   subsequent runs.  A fingerprint change with the same version indicates a
   silent wheel substitution.

### Python API

```python
from effgen.security.supply_chain import verify_installed_hashes, HashDriftWarning
import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = verify_installed_hashes()

if result.drifted:
    print("Drift detected:", result.drifted)
else:
    print(f"All {result.ok} locked packages verified clean.")
```

### `VerificationResult` fields

| Field | Type | Description |
|-------|------|-------------|
| `checked` | `int` | Total distributions checked |
| `ok` | `int` | Distributions matching the lockfile version |
| `drifted` | `list[str]` | Package names with version or build drift |
| `not_in_lockfile` | `list[str]` | Installed but not in the lockfile (extras, dev tools) |
| `skipped` | `int` | Distributions where metadata was unavailable |
| `clean` | `bool` | `True` when `drifted` is empty |

---

## Build reproducibility (Sigstore / cosign)

effGen wheels are signed with [Sigstore](https://www.sigstore.dev/) `cosign`
at release time.  Signing happens in the `release.yml` GitHub Actions workflow
using the OIDC identity of the GitHub Actions runner (keyless signing).

### Verifying a release wheel

```bash
pip install cosign            # or use the binary from https://github.com/sigstore/cosign

# Download wheel and its .sig / .bundle from the GitHub release
cosign verify-blob \
    --certificate-identity "https://github.com/ctrl-gaurav/effGen/.github/workflows/release.yml@refs/tags/v0.3.1" \
    --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
    --bundle effgen-0.3.1.whl.sigstore.bundle \
    effgen-0.3.1-py3-none-any.whl
```

> Sigstore signing is documented here but the step is optional in the current
> release workflow.  Check the `release.yml` file for the exact signing
> invocation when it is enabled.

---

## pyproject.toml hardening

`pyproject.toml` satisfies the following supply-chain requirements:

- **`license`** — SPDX identifier (`Apache-2.0`).
- **`authors` / `maintainers`** — named with email contacts.
- **`[project.urls]`** — `Homepage`, `Documentation`, `Repository`,
  `Bug Tracker`, `Changelog`.
- **`[build-system]`** — pinned `setuptools>=69.0.0` and `wheel>=0.42.0`
  with `setuptools.build_meta` backend.
- **`protobuf>=5.29.5`** — fixes CVE-2025-4565 and CVE-2026-0994.
- **`starlette>=1.0.1`** — fixes PYSEC-2026-161 (host-header injection).

---

## See also

- [`docs/security/secrets.md`](secrets.md) — secret scanning with gitleaks
- [`docs/security/sbom.md`](sbom.md) — SBOM generation with CycloneDX
- [`tests/security/test_vuln_audit.py`](../../tests/security/test_vuln_audit.py)
- [`tests/security/test_supply_chain.py`](../../tests/security/test_supply_chain.py)
