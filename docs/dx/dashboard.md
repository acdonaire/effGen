# effGen Local Dashboard

The effGen local dashboard is a lightweight single-page web app served by the API server at `/dashboard`.  It gives a real-time view into a running effGen deployment.

## Panels

| Panel | Description |
|-------|-------------|
| **Summary cards** | Total requests, errors, average latency, estimated daily cost, and total tokens used. |
| **SLO burn rates** | Visual progress bars showing how close p99 latency, error rate, and availability are to their SLO thresholds. |
| **Request latency chart** | Rolling Chart.js line chart of average latency over recent polling intervals. |
| **Recent agent runs** | Table of the last 50 agent runs with model, token counts, cost, duration, and success/error badge. |
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
  "duration_ms": 543.2,
  "error": null
}
```

## Authentication

By default the dashboard is **public** (no JWT required) — this is intentional for local developer use.  The dashboard itself contains no secrets; it reads only aggregated metrics.

In production, restrict access at the network/ingress level rather than adding JWT requirements to the dashboard.

## Static files

The SPA consists of three files shipped with the `effgen` package:

| File | Purpose |
|------|---------|
| `effgen/dashboard/static/index.html` | Dashboard HTML shell |
| `effgen/dashboard/static/app.js` | All dashboard JavaScript (polling, charts, SSE) |
| `effgen/dashboard/static/style.css` | Dark-mode styles |

Chart.js is loaded from a CDN (`cdn.jsdelivr.net`).  For air-gapped deployments, download `chart.umd.min.js` and serve it locally, updating the `<script>` src in `index.html`.

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
