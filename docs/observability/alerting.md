# Alerting

effGen ships a complete Alertmanager-compatible alert rule pack and an
`AlertWebhook` class that dispatches alert payloads to **Slack** or **Discord**
webhooks.

---

## Alert rule pack

The bundled rule file lives at:

```
docs/observability/alert_rules.yaml
```

It defines six alerts across five rule groups:

| Alert | Severity | Condition | Window |
|---|---|---|---|
| `HighErrorRate` | critical | Error rate > 5% | 10 min |
| `HighP95Latency` | warning | p95 latency > 10 s | 5 min |
| `CostBurnHigh` | warning | Est. cost > $10/day | instant |
| `SLOFastBurn` | critical | Burn rate > 14.4× | instant |
| `SLOSlowBurn` | warning | Burn rate > 3× | 60 min |
| `CircuitBreakerOpen` | warning | Breaker OPEN > 1 min | 1 min |

### Load into Prometheus

```yaml
# prometheus.yml
rule_files:
  - "docs/observability/alert_rules.yaml"
```

### Validate

If [promtool](https://prometheus.io/docs/prometheus/latest/command-line/promtool/)
is installed:

```bash
promtool check rules docs/observability/alert_rules.yaml
```

Or validate programmatically from Python:

```python
from pathlib import Path
from effgen.observability.alerting import validate_alert_rules_yaml

ok, errors = validate_alert_rules_yaml(Path("docs/observability/alert_rules.yaml"))
assert ok, errors
```

---

## AlertWebhook

`AlertWebhook` posts an `Alert` payload to a Slack or Discord webhook URL.
The URL type is auto-detected.

### Setup

```python
from effgen.observability.alerting import AlertWebhook, Alert, AlertSeverity

hook = AlertWebhook("https://hooks.slack.com/services/T.../B.../TOKEN")
# or
hook = AlertWebhook("https://discord.com/api/webhooks/WEBHOOK_ID/WEBHOOK_TOKEN")
```

### Fire an alert

```python
alert = Alert(
    name="HighErrorRate",
    severity=AlertSeverity.CRITICAL,
    summary="Error rate exceeded 5% for 10 minutes",
    value=0.08,
    threshold=0.05,
    labels={"provider": "cerebras", "model": "gpt-oss-120b"},
)
result = hook.fire(alert)
# {"ok": True, "webhook": "https://hooks.slack.com/***", "status": 200, ...}
```

### Fire multiple alerts

```python
results = hook.fire_many([alert1, alert2, alert3])
```

### Non-raising design

`AlertWebhook.fire()` **never raises**.  Network errors, HTTP errors, and
payload issues are caught, logged at `ERROR` level, and returned as:

```python
{"ok": False, "error": "<description>", "webhook": "<redacted>"}
```

This guarantees that failed alert delivery never blocks the agent loop.

---

## URL redaction

Webhook URLs contain secret tokens in the path.  effGen **never logs the full
URL**.  All log lines show only the origin (scheme + host):

```
Input  : https://hooks.slack.com/services/T.../B.../TOKEN
Logged : https://hooks.slack.com/***
```

This is enforced at the `AlertWebhook` layer (before the URL is passed to any
logger) and also covered by the global `Redactor` in structured log lines.

---

## Integration with SLO tracker

```python
from effgen.observability import get_slo_tracker
from effgen.observability.alerting import AlertWebhook, Alert, AlertSeverity

tracker = get_slo_tracker()
webhook = AlertWebhook("https://hooks.slack.com/services/...")

burn = tracker.burn_rate("model_call_success")
if burn > 14.4:
    webhook.fire(Alert(
        name="SLOFastBurn",
        severity=AlertSeverity.CRITICAL,
        summary=f"SLO burn rate {burn:.1f}× (fast-burn threshold: 14.4×)",
        value=burn,
        threshold=14.4,
    ))
```

---

## Configuring Alertmanager

Add a webhook receiver to your Alertmanager configuration:

```yaml
# alertmanager.yml
receivers:
  - name: effgen-slack
    webhook_configs:
      - url: "https://hooks.slack.com/services/..."
        send_resolved: true
```

For programmatic firing from effGen code, use `AlertWebhook.fire()` directly —
you do not need a running Alertmanager instance.
