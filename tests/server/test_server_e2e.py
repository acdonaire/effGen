"""End-to-end tests for the effGen API server: auth + RBAC + budget + audit.

These drive the full FastAPI app (auth, RBAC/budget, audit, production
middleware all wired) through Starlette's TestClient against a local fake
OIDC issuer (an in-process RSA key pair). No network calls, no live models —
a stub runner stands in for the model so the access-control mechanics are
exercised deterministically.
"""
from __future__ import annotations

import glob
import json
import time
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("jwt")
pytest.importorskip("cryptography")


# ---------------------------------------------------------------------------
# Fake OIDC issuer (in-process RSA keypair)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _keypair():
    from cryptography.hazmat.primitives.asymmetric import rsa

    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture()
def _issuer(_keypair, monkeypatch):
    """Configure the app to trust the local fake issuer and seed the JWKS cache."""
    import jwt as pyjwt  # noqa: F401
    from jwt.algorithms import RSAAlgorithm

    jwk = json.loads(RSAAlgorithm.to_jwk(_keypair.public_key()))
    jwk.update({"kid": "k1", "alg": "RS256", "use": "sig"})

    import effgen.server.auth as auth

    auth._jwks_cache.keys = {"k1": jwk}
    auth._jwks_cache.fetched_at = time.time() + 9999

    monkeypatch.setenv("EFFGEN_DEV_MODE", "0")
    monkeypatch.setenv("EFFGEN_OIDC_ISSUER", "https://fake.issuer")
    monkeypatch.setenv("EFFGEN_OIDC_CLIENT_ID", "effgen-client")
    monkeypatch.setenv("EFFGEN_OIDC_JWKS_URI", "https://fake.issuer/jwks")
    monkeypatch.delenv("EFFGEN_RBAC_POLICY_FILE", raising=False)
    auth._DEV_MODE_WARNED = False
    yield
    auth._jwks_cache.keys = {}
    auth._jwks_cache.fetched_at = 0.0


def _token(keypair, roles: list[str], *, sub: str, aud: str = "effgen-client",
           iss: str = "https://fake.issuer", exp_offset: int = 3600) -> str:
    import jwt as pyjwt

    payload = {
        "sub": sub,
        "iss": iss,
        "aud": aud,
        "exp": int(time.time()) + exp_offset,
        "iat": int(time.time()),
        "roles": roles,
        "email": f"{sub}@example.com",
    }
    return pyjwt.encode(payload, keypair, algorithm="RS256", headers={"kid": "k1"})


def _stub_runner(prompt: str, *, model: str, tools: Any = None, stream: bool = False, **_: Any) -> str:
    return f"stub-reply::{model}"


@pytest.fixture()
def client(_issuer, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("EFFGEN_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setenv("EFFGEN_BUDGET_DIR", str(tmp_path / "budget"))
    monkeypatch.setenv("EFFGEN_PER_CALL_COST_USD", "0.01")

    # Reset module singletons so the temp dirs take effect.
    from effgen.server import audit as _audit
    from effgen.server import budget as _budget
    from effgen.server import rbac as _rbac

    _audit._AUDIT_DIR = None
    _budget._BUDGET_DIR = None
    _budget.reset()
    _rbac.reset_registry(None)

    from effgen.server.app import create_app

    app = create_app(runner=_stub_runner)
    return TestClient(app, raise_server_exceptions=False)


CHAT_BODY = {"model": "llama3.1-8b", "messages": [{"role": "user", "content": "hi"}]}


# ---------------------------------------------------------------------------
# Spec items 2, 3, 4, 5, 6, 7
# ---------------------------------------------------------------------------


def test_health_is_public(client):
    assert client.get("/health").status_code == 200


def test_unauthenticated_chat_rejected_401(client):
    # Spec item 2
    r = client.post("/v1/chat/completions", json=CHAT_BODY)
    assert r.status_code == 401


def test_valid_jwt_chat_succeeds_200(client, _keypair):
    # Spec item 3
    h = {"Authorization": f"Bearer {_token(_keypair, ['researcher'], sub='alice')}"}
    r = client.post("/v1/chat/completions", headers=h, json=CHAT_BODY)
    assert r.status_code == 200
    assert "stub-reply" in r.json()["choices"][0]["message"]["content"]


def test_reader_role_tool_use_forbidden_403(client, _keypair):
    # Spec item 4
    h = {"Authorization": f"Bearer {_token(_keypair, ['reader'], sub='bob')}"}
    body = dict(CHAT_BODY, tools=[{"type": "function", "function": {"name": "web_search"}}])
    r = client.post("/v1/chat/completions", headers=h, json=body)
    assert r.status_code == 403
    # RBAC rejections share the OpenAI error envelope (uniform with model errors).
    assert "role reader does not permit tool web_search" in r.json()["error"]["message"]


def test_reader_chat_without_tools_allowed(client, _keypair):
    """Reader may still chat (read-only) when no tools are requested."""
    h = {"Authorization": f"Bearer {_token(_keypair, ['reader'], sub='bob2')}"}
    r = client.post("/v1/chat/completions", headers=h, json=CHAT_BODY)
    assert r.status_code == 200


def test_cost_cap_triggers_429_budget_exceeded(client, _keypair):
    # Spec item 5 — custom $0.01/day role.
    from effgen.server import rbac as _rbac

    _rbac.reset_registry([_rbac.Role("tightbudget", max_cost_per_day=0.01)])
    h = {"Authorization": f"Bearer {_token(_keypair, ['tightbudget'], sub='carol')}"}
    r1 = client.post("/v1/chat/completions", headers=h, json=CHAT_BODY)
    r2 = client.post("/v1/chat/completions", headers=h, json=CHAT_BODY)
    assert r1.status_code == 200
    assert r2.status_code == 429
    assert "BudgetExceeded" in r2.json()["error"]["message"]
    _rbac.reset_registry(None)


def test_audit_log_records_each_request_redacted(client, _keypair, tmp_path):
    # Spec item 6
    raw_jwt = _token(_keypair, ["researcher"], sub="dave")
    client.get("/v1/chat/completions")  # 405/401 path still audited
    client.post("/v1/chat/completions", json=CHAT_BODY)  # unauth → 401
    client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {raw_jwt}"},
        json=CHAT_BODY,
    )  # ok

    time.sleep(0.5)  # let the fire-and-forget audit writes flush
    files = glob.glob(str(tmp_path / "audit" / "*.jsonl"))
    assert files, "no audit file written"
    blob = "".join(open(f).read() for f in files)
    records = [json.loads(line) for line in blob.splitlines() if line.strip()]

    # One record per request, principals resolved (not all "anonymous").
    principals = {r["principal"] for r in records}
    assert "dave" in principals
    assert "anonymous" in principals  # the unauth request

    # No raw JWT and no Authorization header value leaked anywhere.
    assert raw_jwt not in blob
    assert "eyJ" not in blob  # JWT segments start with eyJ
    for rec in records:
        assert set(rec) >= {
            "ts", "principal", "roles", "endpoint",
            "request_summary", "response_summary", "outcome",
        }


def test_dev_mode_allows_unauth_with_warning(_keypair, tmp_path, monkeypatch):
    # Spec item 7
    monkeypatch.setenv("EFFGEN_DEV_MODE", "1")
    monkeypatch.setenv("EFFGEN_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setenv("EFFGEN_BUDGET_DIR", str(tmp_path / "budget"))
    import effgen.server.auth as auth

    auth._DEV_MODE_WARNED = False
    from effgen.server.app import create_app

    app = create_app(dev_mode=True, runner=_stub_runner)

    from fastapi.testclient import TestClient

    c = TestClient(app, raise_server_exceptions=False)
    # The loud dev-mode warning fires on the first request that hits auth.
    with pytest.warns(UserWarning, match="EFFGEN_DEV_MODE"):
        r = c.get("/whoami")
    assert r.status_code == 200
    assert r.json()["sub"] == "dev-user"


# ---------------------------------------------------------------------------
# Default runner: OpenAI-style model-id normalization (deterministic, no API)
# ---------------------------------------------------------------------------


class TestModelIdNormalization:
    """The default runner must route OpenAI-style ``provider/model`` ids to the
    right effGen adapter (colon syntax), not the local Transformers path.
    """

    def test_known_provider_slash_becomes_colon(self) -> None:
        from effgen.server.app import _normalize_model_id

        assert _normalize_model_id("cerebras/llama3.1-8b") == "cerebras:llama3.1-8b"
        assert _normalize_model_id("openai/gpt-4o-mini") == "openai:gpt-4o-mini"

    def test_bare_model_id_untouched(self) -> None:
        from effgen.server.app import _normalize_model_id

        assert _normalize_model_id("llama3.1-8b") == "llama3.1-8b"

    def test_colon_form_untouched(self) -> None:
        from effgen.server.app import _normalize_model_id

        assert _normalize_model_id("cerebras:llama3.1-8b") == "cerebras:llama3.1-8b"

    def test_unknown_prefix_left_as_hf_repo(self) -> None:
        # Looks like an HF org/repo (not a known provider) → must be left alone.
        from effgen.server.app import _normalize_model_id

        assert _normalize_model_id("meta-llama/Llama-3.1-8B") == "meta-llama/Llama-3.1-8B"

    def test_non_string_passthrough(self) -> None:
        from effgen.server.app import _normalize_model_id

        sentinel = object()
        assert _normalize_model_id(sentinel) is sentinel  # type: ignore[arg-type]
