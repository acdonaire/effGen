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
- surfaced mid-stream error event
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


class TestModelsListDiscoverability:
    """``GET /v1/models`` must list what the server actually serves, not just
    the 6 legacy OpenAI-flagship aliases."""

    def test_lists_legacy_aliases_by_default(self):
        c = _client(api_key="k", runner=_ok_runner)
        r = c.get("/v1/models", headers={"X-API-Key": "k"})
        ids = {m["id"] for m in r.json()["data"]}
        assert {"gpt-4", "gpt-4o", "gpt-3.5-turbo"} <= ids

    def test_lists_actually_served_models_alongside_aliases(self):
        from starlette.testclient import TestClient

        from effgen.server.app import create_app

        served = ["openai:gpt-5-nano", "groq:llama-3.1-8b-instant"]
        c = TestClient(create_app(
            api_key="k", runner=_ok_runner, extra_models=lambda: served,
        ))
        r = c.get("/v1/models", headers={"X-API-Key": "k"})
        ids = {m["id"] for m in r.json()["data"]}
        # Real served ids are present without displacing the legacy aliases.
        assert "openai:gpt-5-nano" in ids
        assert "groq:llama-3.1-8b-instant" in ids
        assert "gpt-4" in ids

    def test_extra_models_failure_does_not_break_listing(self):
        from starlette.testclient import TestClient

        from effgen.server.app import create_app

        def _boom():
            raise RuntimeError("pool lock unavailable")

        c = TestClient(create_app(api_key="k", runner=_ok_runner, extra_models=_boom))
        r = c.get("/v1/models", headers={"X-API-Key": "k"})
        assert r.status_code == 200
        assert {m["id"] for m in r.json()["data"]} >= {"gpt-4"}


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


class TestClientDefinedToolsRejected:
    """A tool the server does not host must be rejected with a clear 400, never
    silently dropped — a silent drop returns prose and no ``tool_calls``, quietly
    breaking any client that expects OpenAI function-calling."""

    def test_resolve_tools_rejects_unhosted(self):
        from effgen.api.openai_compat import UnknownToolError
        from effgen.server.app import _resolve_tools

        with pytest.raises(UnknownToolError) as exc:
            _resolve_tools([{"type": "function", "function": {"name": "get_weather"}}])
        assert "get_weather" in str(exc.value)
        assert exc.value.tool_names == ["get_weather"]

    def test_resolve_tools_accepts_builtin(self):
        from effgen.server.app import _resolve_tools

        resolved = _resolve_tools([{"type": "function", "function": {"name": "calculator"}}])
        assert len(resolved) == 1  # the built-in calculator resolves and runs server-side

    def test_classify_unknown_tool_is_400(self):
        from effgen.api.openai_compat import UnknownToolError, _classify_http

        status, etype, code = _classify_http(UnknownToolError(["foo"]))
        assert status == 400 and etype == "invalid_request_error" and code == "unknown_tool"

    def test_unhosted_tool_returns_400_envelope(self):
        from effgen.api.openai_compat import RunnerResult
        from effgen.server.app import _resolve_tools

        def tool_runner(prompt, *, model, tools=None, stream=False, **kw):
            _resolve_tools(tools)  # raises UnknownToolError for an unhosted tool
            return RunnerResult(text="ok")

        c = _client(api_key="k", runner=tool_runner)
        r = c.post("/v1/chat/completions", headers={"X-API-Key": "k"}, json={
            "model": "x", "tool_choice": "required",
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            "messages": [{"role": "user", "content": "weather?"}]})
        assert r.status_code == 400
        err = r.json()["error"]
        assert err["code"] == "unknown_tool"
        assert "get_weather" in err["message"]


class TestCostInExtension:
    """Per-call ``cost_usd`` is surfaced in the ``effgen`` extension when the
    model is priced, and omitted (not a misleading 0) when it is not."""

    def test_cost_usd_surfaced(self):
        from effgen.api.openai_compat import RunnerResult

        def priced(prompt, *, model, tools=None, stream=False, **kw):
            return RunnerResult(text="hi", prompt_tokens=1, completion_tokens=1,
                                resolved_model=model, cost_usd=0.00042)

        c = _client(api_key="k", runner=priced)
        r = c.post("/v1/chat/completions", headers={"X-API-Key": "k"},
                   json={"model": "x", "messages": [{"role": "user", "content": "hi"}]})
        assert r.json()["effgen"]["cost_usd"] == 0.00042

    def test_cost_omitted_when_unpriced(self):
        # _ok_runner sets no cost (cost_usd=None) → key absent, not a fake 0.0.
        c = _client(api_key="k", runner=_ok_runner)
        r = c.post("/v1/chat/completions", headers={"X-API-Key": "k"},
                   json={"model": "x", "messages": [{"role": "user", "content": "hi"}]})
        assert "cost_usd" not in r.json()["effgen"]


class TestEmptyMessagesRejected:
    """An empty ``messages`` array is a 400 (matching OpenAI), not a 200 with an
    invented answer."""

    def test_empty_messages_400(self):
        c = _client(api_key="k", runner=_ok_runner)
        r = c.post("/v1/chat/completions", headers={"X-API-Key": "k"},
                   json={"model": "x", "messages": []})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "empty_messages"


class TestUnifiedErrorEnvelope:
    """Auth (401), validation (422), and rate-limit (429) rejections speak the
    same OpenAI ``{"error": {...}}`` envelope as model errors, so a client can
    branch on ``err.type``/``err.code`` uniformly."""

    def test_401_uses_error_envelope(self):
        c = _client(api_key="secret", runner=_ok_runner)
        r = c.post("/v1/chat/completions",
                   json={"model": "x", "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 401
        err = r.json()["error"]
        assert err["type"] == "invalid_request_error" and err["code"] == "invalid_api_key"
        # The helpful header hint survives (not scrambled by the secret scrubber).
        assert "Authorization" in err["message"]

    def test_422_uses_error_envelope(self):
        c = _client(api_key="k", runner=_ok_runner)
        r = c.post("/v1/chat/completions", headers={"X-API-Key": "k"}, json={"model": "x"})
        assert r.status_code == 422
        err = r.json()["error"]
        assert err["type"] == "invalid_request_error"
        assert "messages" in err["message"]

    def test_429_rate_limit_uses_error_envelope(self):
        from starlette.testclient import TestClient

        from effgen.server.app import create_app

        c = TestClient(create_app(api_key="k", rate_limit_per_minute=1, runner=_ok_runner))
        h = {"X-API-Key": "k"}
        body = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
        c.post("/v1/chat/completions", headers=h, json=body)
        r = c.post("/v1/chat/completions", headers=h, json=body)
        assert r.status_code == 429
        assert r.json()["error"]["type"] == "rate_limit_exceeded"
        assert r.headers.get("retry-after")


class TestRateLimitHeaders:
    """Standard ``RateLimit-*`` headers let a client pace itself before it
    ever hits 429, not just learn about the breach after the fact."""

    def test_headers_present_and_counting_down_on_allowed_requests(self):
        from starlette.testclient import TestClient

        from effgen.server.app import create_app

        c = TestClient(create_app(api_key="k", rate_limit_per_minute=3, runner=_ok_runner))
        h = {"X-API-Key": "k"}
        body = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}

        r1 = c.post("/v1/chat/completions", headers=h, json=body)
        assert r1.status_code == 200
        assert r1.headers["ratelimit-limit"] == "3"
        assert r1.headers["ratelimit-remaining"] == "2"

        r2 = c.post("/v1/chat/completions", headers=h, json=body)
        assert r2.headers["ratelimit-remaining"] == "1"

    def test_headers_present_on_throttled_429(self):
        from starlette.testclient import TestClient

        from effgen.server.app import create_app

        c = TestClient(create_app(api_key="k", rate_limit_per_minute=1, runner=_ok_runner))
        h = {"X-API-Key": "k"}
        body = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
        c.post("/v1/chat/completions", headers=h, json=body)
        r = c.post("/v1/chat/completions", headers=h, json=body)
        assert r.status_code == 429
        assert r.headers["ratelimit-limit"] == "1"
        assert r.headers["ratelimit-remaining"] == "0"
        assert r.headers["ratelimit-reset"]
        assert r.headers["ratelimit-reset"] == r.headers["retry-after"]


class TestDefaultRunnerFailClosed:
    """The production runner (``_build_default_runner``, wired in when no
    custom ``runner=`` is supplied) must not turn a failed generation into a
    200 with the error text as the answer — it has to raise so the route maps
    it to a real HTTP status, the same as every other runner failure."""

    def _app_with_fake_model(self, monkeypatch, api_key="k", *, exc=None, text="PONG"):
        import effgen.server.app as app_mod
        from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount

        class _FakeModel(BaseModel):
            def __init__(self):
                super().__init__(model_name="fake-model", model_type=ModelType.OPENAI)

            def load(self):
                pass

            def generate(self, prompt, config=None, **kwargs):
                if exc is not None:
                    raise exc
                return GenerationResult(
                    text=text, tokens_used=3, finish_reason="stop", model_name=self.model_name,
                )

            def generate_stream(self, prompt, config=None, **kwargs):  # pragma: no cover
                yield text

            def count_tokens(self, text):  # pragma: no cover
                return TokenCount(count=len(text.split()), model_name=self.model_name)

            def get_context_length(self):  # pragma: no cover
                return 4096

            def unload(self):  # pragma: no cover
                pass

            def generate_batch(self, prompts, config=None, **kwargs):  # pragma: no cover
                return [self.generate(p, config=config) for p in prompts]

            def generate_with_tools(self, prompt, tools, config=None, **kwargs):  # pragma: no cover
                return self.generate(prompt, config=config)

            def supports_function_calling(self):
                return False

            def supports_tool_calling(self):
                return False

        monkeypatch.setattr(app_mod, "_get_pooled_model", lambda resolved_model: _FakeModel())

        from starlette.testclient import TestClient

        return TestClient(app_mod.create_app(api_key=api_key))

    def test_failed_generation_is_not_200(self, monkeypatch):
        from effgen.models.errors import ModelNotFoundError

        c = self._app_with_fake_model(
            monkeypatch, exc=ModelNotFoundError("openai", "gpt-does-not-exist-999", "no such model")
        )
        r = c.post("/v1/chat/completions", headers={"X-API-Key": "k"},
                   json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 404, r.status_code
        err = r.json()["error"]
        assert err["type"] == "model_not_found"
        assert "no such model" in err["message"]
        # Not stacked twice ("openai error (model=...)" appearing only once).
        assert err["message"].count("openai error") == 1

    def test_successful_generation_still_200(self, monkeypatch):
        c = self._app_with_fake_model(monkeypatch, text="PONG")
        r = c.post("/v1/chat/completions", headers={"X-API-Key": "k"},
                   json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "PONG"
        assert r.json()["choices"][0]["finish_reason"] == "stop"

    def test_unknown_provider_prefix_is_400_not_500(self):
        # No monkeypatching: exercises the real production path — an unknown
        # "provider:model" prefix must fail loading with a clear 4xx before it
        # ever reaches a network call, not fall through to the local loader
        # (which used to leak an "invalid repo id" 500).
        from starlette.testclient import TestClient

        from effgen.server.app import create_app

        c = TestClient(create_app(api_key="k"))
        r = c.post("/v1/chat/completions", headers={"X-API-Key": "k"},
                   json={"model": "nonexistent:foo-model",
                         "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 400, r.status_code
        err = r.json()["error"]
        assert err["type"] == "invalid_request_error"
        assert "nonexistent" in err["message"]
