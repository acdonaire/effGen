# effGen Local Dashboard

The effGen local dashboard is a lightweight single-page web app served by the API server at `/dashboard`.  It gives a real-time view into a running effGen deployment.

## Panels

| Panel | Description |
|-------|-------------|
| **Summary cards** | Total requests, errors, average latency, estimated daily cost, and total tokens used. |
| **SLO burn rates** | Visual progress bars showing how close p99 latency, error rate, and availability are to their SLO thresholds. |
| **Request latency chart** | Rolling line chart of average latency over recent polling intervals, drawn on a canvas. |
| **Recent agent runs** | Table of the last 50 agent runs with model, token counts, cost, duration, and success/error badge. |
| **Agent topology** | Team and workflow executions as a node-link graph: agents (the manager marked apart) and the tools they reached as nodes, delegation, handoff and tool use as edges. Status is carried by a glyph and a text label as well as color; nodes are keyboard-focusable and open a detail panel. |
| **Live span stream** | Real-time feed of trace spans via Server-Sent Events (SSE).  Includes a pause toggle and clear button. |
| **Prometheus metrics (raw)** | Sortable table of all registered Prometheus metric names and their current values. |

## Starting the server

```bash
# Development mode — auth disabled
EFFGEN_DEV_MODE=1 uvicorn effgen.server.app:create_app --factory --host 0.0.0.0 --port 8080
```

Then open **[http://localhost:8080/dashboard](http://localhost:8080/dashboard)** in a browser.

## Data endpoint

The dashboard polls `/dashboard/data.json` every 5 seconds.  You can also query it directly:

```bash
curl http://localhost:8080/dashboard/data.json | python -m json.tool
```

Response structure:

```json
{
  "ts": "2026-05-26T12:00:00Z",
  "metrics": {
    "total_requests": 142,
    "total_errors": 3,
    "avg_latency_s": 0.7213,
    "total_tokens": 18500,
    "daily_cost_usd": 0.014200
  },
  "slo": {
    "p99_latency_burn": 0.36,
    "error_rate_burn": 0.21,
    "availability": 0.979
  },
  "recent_runs": [
    {
      "ts": "12:00:01",
      "model": "cerebras:gpt-oss-120b",
      "input_tokens": 120,
      "output_tokens": 340,
      "duration_s": 0.543,
      "cost_usd": 0.00001,
      "error": null
    }
  ],
  "recent_spans": [...],
  "raw_metrics": {...}
}
```

## SSE span stream

Trace spans are pushed over SSE at `/dashboard/spans`.  Each event is a JSON object:

```json
{
  "ts": "12:00:01",
  "name": "effgen.model.call cerebras:gpt-oss-120b",
  "kind": "model",
  "agent": null,
  "tool": null,
  "model": "cerebras:gpt-oss-120b",
  "duration_ms": 543.2,
  "status": "ok",
  "error": null,
  "run_id": "9f2c1d40ab77",
  "offset_ms": 12.4,
  "execution_id": "b25fed1b57d3",
  "execution_kind": "team",
  "execution_name": "newsroom",
  "parent_agent": "lead",
  "role": "worker"
}
```

`kind` is one of `agent`, `model`, `tool` or `router`, and the matching
`agent`/`tool`/`model` field names what the span timed — read those rather than
parsing `name`, which is the display label. `status` is `ok`, `error` or
`skipped`; a run that reports a failure without raising still records it here.
The `execution_*` fields group the spans of one team or workflow run.

## Topology endpoint

`GET /dashboard/topology.json?limit=6` returns recent team and workflow
executions as node-link graphs:

```json
{
  "executions": [
    {
      "id": "b25fed1b57d3",
      "kind": "team",
      "name": "newsroom",
      "status": "ok",
      "cost_usd": 0.000042,
      "tokens": 1234,
      "nodes": [
        {"id": "lead", "type": "manager", "status": "ok", "model": "llama-3.1-8b-instant",
         "runs": 2, "cost_usd": 0.00002, "tokens": 800, "duration_s": 1.2}
      ],
      "edges": [{"source": "lead", "target": "researcher", "kind": "delegation", "count": 1}]
    }
  ],
  "count": 1
}
```

It is built from the durable run store plus the buffered spans, so a team or
workflow run from a script or the CLI appears here too, not only work done
inside the server process. `executions` is empty when nothing multi-agent has
run yet.

## Authentication

The static SPA shell (HTML/JS/CSS) is public so the page can load and prompt for
a key. The data endpoints — `/dashboard/data.json`, `/dashboard/spans`,
`/dashboard/catalog.json`, `/dashboard/history.json` and
`/dashboard/topology.json` — require authentication by default and return a
typed `invalid_api_key` envelope without one. Set `EFFGEN_PUBLIC_DASHBOARD=1` to
open them for local viewing, and restrict access at the network/ingress level in
a shared deployment.

## Static files

The SPA consists of three files shipped with the `effgen` package:

| File | Purpose |
|------|---------|
| `effgen/dashboard/static/index.html` | Dashboard HTML shell |
| `effgen/dashboard/static/app.js` | All dashboard JavaScript (polling, charts, SSE) |
| `effgen/dashboard/static/style.css` | Dark and light theme styles |

Every asset is served from the package: the page references no external host, so
it renders the same in an air-gapped deployment. The charts are drawn on a
canvas and the topology graph is inline SVG built in `app.js` — no chart or
graph library is involved.

## Adding run records

To populate the "Recent Agent Runs" panel, call `record_run` from your code:

```python
from effgen.observability.run_log import record_run

record_run(
    model="cerebras:zai-glm-4.7",
    input_tokens=250,
    output_tokens=80,
    duration_s=1.12,
    cost_usd=0.000032,
)
```

The `Agent` class calls this automatically on every run (best-effort — it never
breaks a run if the dashboard machinery is unavailable), so manual calls are only
needed for custom integrations that bypass `Agent`.
