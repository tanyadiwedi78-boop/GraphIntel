"""Offline HTML presentation only; data aggregation stays in reporting.py."""

import html
import json
from typing import Any, List


def display(value: Any) -> str:
    if value is None:
        return "Not measured"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.4f}"
    return html.escape(str(value))


def run_identity(summary: dict) -> dict:
    result = {}
    for key in ("run_id", "trace_id", "data_kind"):
        result[key] = summary.get(key)
    return result


def attempt_rows(summary: dict) -> str:
    rows = []
    for attempt in summary["attempts"][-100:]:
        cells = []
        for key in (
            "chunk_id",
            "provider",
            "model",
            "stage",
            "accepted",
            "category",
            "reason",
            "request_seconds",
            "tokens",
        ):
            value = attempt.get(key)
            rendered = (
                "None"
                if key in ("category", "reason") and value is None
                else display(value)
            )
            cells.append(
                '<td data-label="'
                + html.escape(key.replace("_", " "))
                + '">'
                + rendered
                + "</td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "".join(rows)


def span_rows(summary: dict) -> str:
    span_rows = []
    for span in summary.get("spans", [])[-100:]:
        span_rows.append(
            "<tr><td>"
            + display(span.get("name"))
            + "</td><td>"
            + display(span.get("status"))
            + "</td><td>"
            + display(span.get("duration_seconds"))
            + "</td><td>"
            + display(span.get("parent_span_id"))
            + "</td></tr>"
        )
    return "".join(span_rows)


def event_rows(events: List[dict]) -> str:
    timeline = []
    for event in events[-100:]:
        timeline.append(
            "<li><strong>"
            + display(event["event"])
            + "</strong> "
            + display(event.get("chunk_id", ""))
            + " "
            + display(event.get("category", ""))
            + " <small>"
            + display(event.get("span_id"))
            + "</small></li>"
        )
    return "".join(timeline)


def metric_cards(summary: dict) -> str:
    metrics = []
    for key in (
        "attempts_started",
        "attempts_finished",
        "unfinished_attempts",
        "accepted_attempts",
        "schema_validity",
        "schema_denominator",
        "retry_rate",
        "retry_denominator",
        "retry_attempts",
        "fallback_attempts",
        "cache_hits",
        "checkpoint_skips",
        "mean_request_seconds",
        "p95_request_seconds",
        "latency_sample_count",
        "cost_usd",
    ):
        metrics.append(
            "<div><dt>"
            + html.escape(key.replace("_", " "))
            + "</dt><dd>"
            + display(summary[key])
            + "</dd></div>"
        )
    return "".join(metrics)


def provider_cards(summary: dict) -> str:
    cards = []
    for name, row in summary.get("providers", {}).items():
        cards.append(
            "<section><h3>"
            + display(name)
            + "</h3><p>Finished attempts: "
            + display(row["attempts"])
            + "; accepted: "
            + display(row["accepted"])
            + "; schema-valid: "
            + display(row["schema_valid"])
            + "/"
            + display(row["attempts"])
            + "</p><p>Mean request seconds: "
            + display(row["mean_seconds"])
            + "; timing samples: "
            + display(row["duration_count"])
            + "</p></section>"
        )
    return "".join(cards)


def dashboard_html(summary: dict, events: List[dict], quality: dict) -> str:
    data_kind = summary.get("data_kind", "unknown")
    notice = (
        "OFFLINE FIXTURE: demonstration only, not real model results."
        if data_kind == "offline_fixture"
        else "Run data: " + str(data_kind)
    )
    return (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="10">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>GraphIntel run report</title><style>
body{font:16px/1.55 system-ui,sans-serif;margin:0;background:#f5f4ef;color:#262c35}
main{max-width:1100px;margin:auto;padding:28px}h1{font-size:30px}h2{margin-top:36px}.notice{border-left:4px solid #ad6b14;padding:12px;background:#fff3d9;font-weight:600}
dl{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}
dl div,section{background:white;padding:18px;border:1px solid #d8dce0;border-radius:10px}
dt{font-size:14px;color:#505965}dd{font-size:23px;margin:8px 0 0}.scroll{overflow:auto}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:12px;text-align:left;border-bottom:1px solid #ddd}
@media(max-width:600px){.attempts thead{display:none}.attempts tr{display:block;border-bottom:2px solid #ddd;margin-bottom:18px}.attempts td{display:block;border:0;padding:7px 0;overflow-wrap:anywhere}.attempts td:before{content:attr(data-label);display:block;font-weight:600;color:#505965}}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}small{color:#59636f}li{padding:7px 0;overflow-wrap:anywhere}
</style><main><h1>GraphIntel run report</h1><p class="notice">"""
        + html.escape(notice)
        + """</p>
<p>Refreshes every 10 seconds while the local file is open. Local evidence only. No external dashboard, tracking scripts or automatic graph writes.</p>
<p>"""
        + html.escape(summary["basis"])
        + "</p><dl>"
        + metric_cards(summary)
        + "</dl>"
        + """
<h2>Run identity</h2><section><pre>"""
        + display(run_identity(summary))
        + """</pre></section>
<h2>Speed and resource use</h2><section><p>Latency is request time, including failed attempts, not whole-corpus runtime.
P95 is the nearest-rank sample statistic, not a stable service guarantee.</p><pre>"""
        + display(summary["tokens"])
        + "</pre>"
        + """
<p>Schema validity and retry rate are fractions, with finished-attempt denominators shown above. Unknown usage stays unknown. Known subtotals are not total token usage. Costs need billing reconciliation.</p></section>
<h2>Quality, separate from speed</h2><section><p>Schema/evidence checks do not prove factual accuracy.
Precision and recall need source-checked reference labels.</p><pre>"""
        + html.escape(json.dumps(quality, indent=2))
        + "</pre></section>"
        + """
<h2>Attempts and fallback</h2><p>Latest 100 rows shown in each detailed section; JSON and JSONL files retain full run evidence.</p><p>"""
        + html.escape(summary["comparison_warning"])
        + "</p>"
        + provider_cards(summary)
        + "<section class='scroll'><table class='attempts'>"
        + """
<thead><tr><th>Chunk</th><th>Provider</th><th>Model</th><th>Stage</th><th>Accepted</th><th>Category</th><th>Reason</th><th>Seconds</th><th>Tokens</th></tr></thead>
<tbody>"""
        + (
            attempt_rows(summary)
            or "<tr><td colspan='9'>No finished attempts in this run.</td></tr>"
        )
        + "</tbody></table></section>"
        + """
<h2>Local spans</h2><section class="scroll"><table><tr><th>Stage</th><th>Status</th><th>Seconds</th><th>Parent span</th></tr>"""
        + span_rows(summary)
        + """</table></section>
<h2>Correlated lifecycle</h2><section><p>Trace and span identifiers correlate local events with spans.jsonl.
A started attempt without a finish needs reconciliation, not blind replay.</p><ol>"""
        + event_rows(events)
        + "</ol></section></main></html>"
    )
