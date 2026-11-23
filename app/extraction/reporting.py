"""Render local evidence as separate operational and quality reports."""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List

from app.extraction.dashboard import dashboard_html
from app.extraction.observability import scrub
from app.extraction.pipeline import Pipeline, check_label_binding, load


def read_events(path: Path) -> List[dict]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def token_counts(usage: Any) -> dict:
    if not isinstance(usage, dict):
        return {"input": None, "output": None}
    result = {}
    for name, alternatives in (
        ("input", ("promptTokenCount", "prompt_tokens", "input_tokens", "input")),
        (
            "output",
            ("candidatesTokenCount", "completion_tokens", "output_tokens", "output"),
        ),
    ):
        result[name] = None
        for key in alternatives:
            value = usage.get(key)
            if type(value) is int and value >= 0:
                result[name] = value
                break
    return result


def operational_report(events: List[dict]) -> Dict[str, Any]:
    attempts = []
    cache_hits = 0
    checkpoints = 0
    started = 0
    for event in events:
        if event["event"] == "attempt_started":
            started += 1
        if event["event"] == "cache_hit":
            cache_hits += 1
        if event["event"] == "checkpoint_skipped":
            checkpoints += 1
        if event["event"] == "attempt_finished":
            saved = dict(event)
            saved["tokens"] = token_counts(event.get("usage"))
            attempts.append(saved)
    latencies = []
    accepted = 0
    schema_valid = 0
    retries = 0
    fallback = 0
    seen = set()
    tokens = {"input": 0, "output": 0}
    known = {"input": 0, "output": 0}
    for attempt in attempts:
        accepted += int(attempt.get("accepted") is True)
        schema_valid += int(attempt.get("schema_pass") is True)
        fallback += int(attempt.get("stage") == "fallback")
        key = (attempt["chunk_id"], attempt["provider"])
        if key in seen:
            retries += 1
        seen.add(key)
        seconds = attempt.get("request_seconds")
        if type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0:
            latencies.append(seconds)
        for name in tokens:
            value = attempt["tokens"][name]
            if value is not None:
                known[name] += 1
                tokens[name] += value
    ordered = sorted(latencies)
    return {
        "basis": "This run only. All finished attempts include failures; cache/checkpoint skips excluded.",
        "comparison_warning": "Fallback sees selected failures, not a matched provider comparison.",
        "attempts_started": started,
        "attempts_finished": len(attempts),
        "unfinished_attempts": started - len(attempts),
        "accepted_attempts": accepted,
        "schema_valid_attempts": schema_valid,
        "schema_denominator": len(attempts),
        "schema_validity": schema_valid / len(attempts) if attempts else None,
        "retry_attempts": retries,
        "fallback_attempts": fallback,
        "cache_hits": cache_hits,
        "checkpoint_skips": checkpoints,
        "mean_request_seconds": sum(latencies) / len(latencies) if latencies else None,
        "p95_request_seconds": ordered[math.ceil(len(ordered) * 0.95) - 1]
        if ordered
        else None,
        "latency_sample_count": len(latencies),
        "retry_rate": retries / len(attempts) if attempts else None,
        "retry_denominator": len(attempts),
        "token_basis": "Provider-reported fields only; output reasoning inclusion depends on the provider. No tokenizer estimates are presented as usage.",
        "tokens": {
            "known_subtotal": tokens,
            "known_attempts": known,
            "total": tokens
            if attempts and all(count == len(attempts) for count in known.values())
            else None,
        },
        "cost_usd": None,
        "cost_status": "Not reconciled with billing.",
        "attempts": attempts,
    }


def atomic_report(target: Path, content: str) -> None:
    """Only derived reports in this run folder are replaced, never job evidence."""
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def write_dashboard(
    pipeline: Pipeline, folder: Path, labels: Any = None, evidence_reviews: Any = None
) -> Path:
    manifest = load(folder / "manifest.json")
    if manifest.get("plan_hash") != pipeline.plan["plan_hash"]:
        raise ValueError("Report folder belongs to a different immutable job.")
    events = read_events(folder / "events.jsonl")
    for event in events:
        if (
            event.get("run_id") != manifest["run_id"]
            or event.get("trace_id") != manifest["trace_id"]
        ):
            raise ValueError("Report event correlation differs from run manifest.")
    summary = operational_report(events)
    summary["spans"] = read_events(folder / "spans.jsonl")
    quality = {
        "status": "Not measured",
        "reason": "No source-reviewed reference labels supplied.",
    }
    if labels is not None:
        if not isinstance(labels, dict):
            raise ValueError("Source-bound labels must be a mapping.")
        check_label_binding(pipeline.plan["chunks"], labels)
        report = pipeline.report(labels=labels, evidence_reviews=evidence_reviews)
        quality = {
            "scope": "Whole immutable job, not this run only.",
            "metrics": report,
            "label_provenance": "See source-bound label file; user confirmation not inferred.",
        }
    summary.update(
        run_id=manifest["run_id"],
        trace_id=manifest["trace_id"],
        data_kind=pipeline.plan["primary"].get("execution_mode", "unknown"),
    )
    atomic_report(
        folder / "summary.json", json.dumps(scrub(summary), indent=2, allow_nan=False)
    )
    if labels is not None or not (folder / "quality.json").exists():
        atomic_report(
            folder / "quality.json",
            json.dumps(scrub(quality), indent=2, allow_nan=False),
        )
    quality = load(folder / "quality.json")
    target = folder / "dashboard.html"
    atomic_report(target, dashboard_html(scrub(summary), scrub(events), scrub(quality)))
    return target
