"""Local correlated event/span evidence. No network exporter or background service."""

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

KEY_NAMES = ("GEMINI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY")


def redact(text: str) -> str:
    text = str(text)
    for name in KEY_NAMES:
        value = os.environ.get(name, "").strip()
        if len(value) >= 8:
            text = text.replace(value, "[redacted-key]")
    text = re.sub(
        r"(?i)((?:key|api_key|apikey|access_token|token)=|bearer\s+)[A-Za-z0-9._\-]{4,}",
        r"\1[redacted]",
        text,
    )
    return re.sub(
        r"(?<![A-Za-z0-9_\-])(?:sk-or-[A-Za-z0-9_\-]{4,}|sk-[A-Za-z0-9_\-]{8,}|AIza[A-Za-z0-9_\-]{8,})",
        "[redacted-key]",
        text,
    )


def scrub(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        cleaned = []
        for item in value:
            cleaned.append(scrub(item))
        return cleaned
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            cleaned[redact(str(key))] = scrub(item)
        return cleaned
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LocalTelemetry:
    """One run directory, one writer; raw source and provider text stay in the job."""

    def __init__(self, reports_root: Path, job: Path, plan_hash: str) -> None:
        self.run_id = uuid.uuid4().hex
        self.trace_id = uuid.uuid4().hex
        self.folder = reports_root / self.run_id
        self.folder.mkdir(parents=True, exist_ok=False)
        with (self.folder / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(
                {
                    "run_id": self.run_id,
                    "trace_id": self.trace_id,
                    "plan_hash": plan_hash,
                    "job": str(job),
                },
                handle,
            )
        self.started = time.monotonic()
        self.root_span = uuid.uuid4().hex[:16]
        self.event("run_started", job=str(job), plan_hash=plan_hash)

    def append(self, filename: str, value: Dict[str, Any]) -> None:
        with (self.folder / filename).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(scrub(value), allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def event(self, event: str, span_id: Optional[str] = None, **fields: Any) -> None:
        entry = {
            "ts": utc_now(),
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "span_id": span_id or self.root_span,
            "event": event,
        }
        entry.update(fields)
        self.append("events.jsonl", entry)

    def start_span(
        self, name: str, parent_id: Optional[str] = None, **attributes: Any
    ) -> Dict[str, Any]:
        span = {
            "name": name,
            "span_id": uuid.uuid4().hex[:16],
            "parent_span_id": parent_id or self.root_span,
            "started_at": utc_now(),
            "start_clock": time.monotonic(),
            "attributes": attributes,
        }
        self.event(
            "span_started",
            span["span_id"],
            name=name,
            parent_span_id=span["parent_span_id"],
            attributes=attributes,
        )
        return span

    def end_span(
        self, span: Dict[str, Any], status: str = "OK", **attributes: Any
    ) -> None:
        saved = dict(span)
        saved["attributes"] = dict(span["attributes"])
        saved["attributes"].update(attributes)
        saved["duration_seconds"] = time.monotonic() - saved.pop("start_clock")
        saved.update(
            ended_at=utc_now(),
            status=status,
            run_id=self.run_id,
            trace_id=self.trace_id,
        )
        self.append("spans.jsonl", saved)
        self.event("span_finished", span["span_id"], name=span["name"], status=status)

    def finish(self, status: str) -> None:
        self.append(
            "spans.jsonl",
            {
                "name": "extraction_run",
                "span_id": self.root_span,
                "parent_span_id": None,
                "trace_id": self.trace_id,
                "run_id": self.run_id,
                "status": status,
                "duration_seconds": time.monotonic() - self.started,
            },
        )
        self.event("run_finished", status=status)
