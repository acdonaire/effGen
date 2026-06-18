"""OpenAI-compatible API + secure-server contract tests.

These exercise the HTTP/auth/usage/streaming *contract* of the server via a
stub runner (an in-process callable) — they do not mock live model behaviour,
which the project forbids. Live model coverage through the server is in the
integration/evidence runs.

Covers the OpenAI-compatible server API:
- static API-key auth (Bearer + X-API-Key), fail-closed
- /openapi.json public even with auth on
- real usage from a RunnerResult (not len//4)
- documented alias shim → ``effgen`` resolved-model metadata
- SSE streaming works through the full create_app middleware stack (the F15
  regression: it used to hang) with incremental chunks + final usage chunk
- structured, redacted OpenAI-style error envelopes (bad model → 404)
- honest mid-stream error event
"""
from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.skipif(
    not all(__import__("importlib").util.find_spec(m) for m in ("fastapi", "starlette")),
    reason="fastapi not installed",
)

try:  # module-level so a route's ``ws: WebSocket`` annotation resolves here too
    from fastapi import WebSocket
except Exception:  # pragma: no cover - skipped when fastapi is absent
    WebSocket = None


def _client(api_key=None, runner=None):
    from starlette.testclient import TestClient

    from effgen.server.app import create_app

    return TestClient(create_app(api_key=api_key, runner=runner))


def _ok_runner(prompt, *, model, tools=None, stream=False, temperature=None, **kw):
    from effgen.api.openai_compat import RunnerResult

    if stream:
        def g():
            yield from ["Hello", ", ", "world", "!"]
        return g()
    return RunnerResult(
        text="Hello, world!",
        prompt_tokens=11,
        completion_tokens=4,
        resolved_model=model,
        finish_reason="stop",
    )


class TestStaticApiKeyAuth:
    def test_no_key_rejected(self):
        c = _client(api_key="secret", runner=_ok_runner)
        r = c.post("/v1/chat/completions",
                   json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 401

    def test_wrong_key_rejected(self):
        c = _client(api_key="secret", runner=_ok_runner)
        r = c.post("/v1/chat/completions", headers={"X-API-Key": "nope"},
                   json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 401

    def test_bearer_key_accepted(self):
        c = _client(api_key="secret", runner=_ok_runner)
        r = c.post("/v1/chat/completions", headers={"Authorization": "Bearer secret"},
                   json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200

    def test_x_api_key_accepted(self):
        c = _client(api_key="secret", runner=_ok_runner)
        r = c.post("/v1/chat/completions", headers={"X-API-Key": "secret"},
                   json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200

    def test_openapi_public_with_auth(self):
        c = _client(api_key="secret", runner=_ok_runner)
        assert c.get("/openapi.json").status_code == 200
        assert c.get("/health").status_code == 200


class TestUsageAndAliasMetadata:
    def test_real_usage_passthrough(self):
        c = _client(api_key="k", runner=_ok_runner)
        r = c.post("/v1/chat/completions", headers={"X-API-Key": "k"},
                   json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]})
        usage = r.json()["usage"]
        # Exactly the RunnerResult values — not a len//4 estimate.
        assert usage == {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}

    def test_alias_metadata(self):
        c = _client(api_key="k", runner=_ok_runner)
        r = c.post("/v1/chat/completions", headers={"X-API-Key": "k"},
                   json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]})
        body = r.json()
        meta = body.get("effgen")
        assert meta["requested_model"] == "gpt-4"
        assert meta["resolved_model"] == "Qwen/Qwen2.5-7B-Instruct"
        assert meta["alias_applied"] is True
        # The response 'model' reports what actually ran.
        assert body["model"] == "Qwen/Qwen2.5-7B-Instruct"

    def test_no_alias_metadata(self):
        c = _client(api_key="k", runner=_ok_runner)
        r = c.post("/v1/chat/completions", headers={"X-API-Key": "k"},
                   json={"model": "Qwen/Qwen2.5-1.5B-Instruct",
                         "messages": [{"role": "user", "content": "hi"}]})
        assert r.json()["effgen"]["alias_applied"] is False


class TestStreamingThroughFullStack:
    """Regression for F15: SSE used to hang through the create_app stack."""

    def test_streaming_incremental_and_usage(self):
        c = _client(api_key="k", runner=_ok_runner)
        deltas, usage, first_meta = [], None, None
        with c.stream("POST", "/v1/chat/completions", headers={"X-API-Key": "k"},
                      json={"model": "gpt-4", "stream": True,
                            "stream_options": {"include_usage": True},
                            "messages": [{"role": "user", "content": "hi"}]}) as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                if line.startswith("data: ") and "[DONE]" not in line:
                    d = json.loads(line[6:])
                    if d.get("choices") and d["choices"][0]["delta"].get("content"):
                        deltas.append(d["choices"][0]["delta"]["content"])
                    if d.get("usage"):
                        usage = d["usage"]
                    if d.get("effgen") and first_meta is None:
                        first_meta = d["effgen"]
        # Incremental: more than one content chunk, not one buffered blob.
        assert deltas == ["Hello", ", ", "world", "!"]
        assert usage is not None and usage["total_tokens"] > 0
        assert first_meta and first_meta["alias_applied"] is True

    def test_mid_stream_error_is_terminal_event(self):
        from effgen.api.openai_compat import RunnerResult  # noqa: F401

        def boom_runner(prompt, *, model, tools=None, stream=False, **kw):
            def g():
                yield "partial "
                raise RuntimeError("boom sk-secret123 leaked")
            return g()

        c = _client(api_key="k", runner=boom_runner)
        saw_error = False
        with c.stream("POST", "/v1/chat/completions", headers={"X-API-Key": "k"},
                      json={"model": "gpt-4", "stream": True,
                            "messages": [{"role": "user", "content": "hi"}]}) as resp:
            for line in resp.iter_lines():
                if line.startswith("data: ") and "[DONE]" not in line:
                    d = json.loads(line[6:])
                    if d.get("error"):
                        saw_error = True
                        assert "sk-secret123" not in d["error"]["message"]
        assert saw_error


class TestStructuredErrors:
    def test_bad_model_404_envelope(self):
        from effgen.models.errors import ModelNotFoundError

        def err_runner(prompt, *, model, tools=None, stream=False, **kw):
            raise ModelNotFoundError("no such model", model_name=model)

        c = _client(api_key="k", runner=err_runner)
        r = c.post("/v1/chat/completions", headers={"X-API-Key": "k"},
                   json={"model": "ghost", "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 404
        err = r.json()["error"]
        assert err["type"] == "model_not_found"
        assert err["code"] == "model_not_found"

    def test_error_message_redacted(self):
        def leak_runner(prompt, *, model, tools=None, stream=False, **kw):
            raise RuntimeError("failed; Authorization: Bearer sk-abcdef123456 nope")

        c = _client(api_key="k", runner=leak_runner)
        r = c.post("/v1/chat/completions", headers={"X-API-Key": "k"},
                   json={"model": "x", "messages": [{"role": "user", "content": "hi"}]})
        assert "sk-abcdef123456" not in json.dumps(r.json())

    def test_missing_upstream_key_is_503_not_401(self):
        """A server-side missing provider key is the server's problem, not the
        caller's. With the client correctly authenticated, it must surface as a
        gateway error (503), never 401 — 401 would wrongly blame the caller's
        credentials."""
        from effgen.models.errors import ModelAuthError

        def no_upstream_key(prompt, *, model, tools=None, stream=False, **kw):
            raise ModelAuthError("Groq API key not found. Set the GROQ_API_KEY variable")

        c = _client(api_key="k", runner=no_upstream_key)
        r = c.post("/v1/chat/completions", headers={"X-API-Key": "k"},
                   json={"model": "groq:llama-3.1-8b-instant",
                         "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 503, r.status_code
        assert r.json()["error"]["type"] == "upstream_unavailable"

    def test_rejected_upstream_key_is_502_not_401(self):
        """A present-but-rejected upstream key is a bad-gateway condition (502),
        not a client-auth failure (401)."""
        from effgen.models.errors import ModelAuthError

        def rejected_upstream(prompt, *, model, tools=None, stream=False, **kw):
            raise ModelAuthError("Incorrect API key provided by the upstream provider")

        c = _client(api_key="k", runner=rejected_upstream)
        r = c.post("/v1/chat/completions", headers={"X-API-Key": "k"},
                   json={"model": "groq:llama-3.1-8b-instant",
                         "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 502, r.status_code

    def test_forged_client_token_still_401(self):
        """Phase-18 regression guard: the server's own client-auth rejection
        stays 401 even though upstream auth failures now map to 502/503."""
        c = _client(api_key="secret", runner=_ok_runner)
        r = c.post("/v1/chat/completions", headers={"Authorization": "Bearer forged"},
                   json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 401, r.status_code


class TestWebSocketAuth:
    """Static API-key auth must apply to websocket handshakes too, and reject
    them cleanly (ASGI ``websocket.close``) instead of crashing with an HTTP
    response message on a websocket scope."""

    def _app_with_ws(self):
        from effgen.server.app import create_app

        app = create_app(api_key="secret")

        @app.websocket("/wsauth")
        async def _wsauth(ws: WebSocket) -> None:
            await ws.accept()
            await ws.send_json({"ok": True})
            await ws.close()

        return app

    def test_ws_valid_key_accepts(self):
        from starlette.testclient import TestClient

        c = TestClient(self._app_with_ws())
        with c.websocket_connect("/wsauth", headers={"Authorization": "Bearer secret"}) as ws:
            assert ws.receive_json() == {"ok": True}

    def test_ws_missing_key_rejected_cleanly(self):
        from starlette.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect

        c = TestClient(self._app_with_ws())
        with pytest.raises(WebSocketDisconnect) as exc:
            with c.websocket_connect("/wsauth"):
                pass
        assert exc.value.code == 1008  # policy violation, not a 500 crash


class TestConvenienceWebSocketRoute:
    """The CLI ``/ws`` route's ``ws: WebSocket`` annotation must resolve from
    module scope; otherwise FastAPI treats ``ws`` as a required query param and
    rejects the handshake (the same module-vs-local class trap as the /run body
    model). Guard against a regression without needing a live model."""

    def test_ws_route_has_no_spurious_query_param(self):
        from effgen.cli._main import CLIInterface
        from effgen.server.app import create_app

        app = create_app(api_key="secret")
        cli = CLIInterface()
        app.state.cli = cli
        cli._register_convenience_routes(app)

        ws_route = next((r for r in app.router.routes if getattr(r, "path", "") == "/ws"), None)
        assert ws_route is not None, "/ws route was not registered"
        # If the WebSocket annotation failed to resolve, FastAPI would expose a
        # 'ws' query parameter here and reject every handshake.
        assert [p.name for p in ws_route.dependant.query_params] == []


class TestTextCompletions:
    def test_completions_runner_result(self):
        c = _client(api_key="k", runner=_ok_runner)
        r = c.post("/v1/completions", headers={"X-API-Key": "k"},
                   json={"model": "gpt-4", "prompt": "hi", "max_tokens": 8})
        assert r.status_code == 200
        body = r.json()
        assert body["choices"][0]["text"] == "Hello, world!"
        assert body["usage"]["completion_tokens"] == 4
