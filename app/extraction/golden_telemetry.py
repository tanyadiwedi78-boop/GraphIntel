"""Optional OpenTelemetry tracing for golden-20 runs. Off by default.

When configs/golden_extraction.json sets telemetry.enabled true AND the
observability extras are installed (requirements-observability.txt), every
chunk attempt becomes a span exported over OTLP HTTP - typically to a local
Phoenix started with `phoenix serve` (UI and OTLP HTTP on
http://localhost:6006, traces at /v1/traces). Phoenix is a viewer only:
extraction never depends on it.

Hard rules:
- Setup or export failure must never erase or block extraction output. Any
  problem prints a warning and the run continues; metrics.jsonl stays the
  durable record.
- Spans carry metadata only: ids, category, counts, tokens, cost, latency,
  provider, mechanical check tallies and a redacted error label. Never
  chunk text, prompt text, answer text or keys.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from app.extraction.chunk_metrics import UNAVAILABLE
from app.extraction.observability import redact

DEFAULT_ENDPOINT = 'http://localhost:6006/v1/traces'
SERVICE_NAME = 'graphintel-golden20'


class _WatchingExporter:
    """Wraps an exporter to notice failures the SDK only logs internally."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.failed = False
        self.reported = False

    def export(self, spans: Any) -> Any:
        try:
            result = self.inner.export(spans)
        except Exception:
            self.failed = True
            self._warn()
            raise
        if getattr(result, 'name', '') != 'SUCCESS':
            self.failed = True
            self._warn()
        return result

    def _warn(self) -> None:
        if not self.reported:
            self.reported = True
            print('Telemetry: span export failed; is Phoenix running (`phoenix serve`)? '
                  'Run continues; metrics.jsonl is unaffected.')

    def shutdown(self) -> Any:
        return self.inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> Any:
        return self.inner.force_flush(timeout_millis)


class GoldenTelemetry:
    """Small wrapper so the runner never touches OTel objects directly."""

    def __init__(self, provider: Any, tracer: Any) -> None:
        self.provider = provider
        self.tracer = tracer

    def record_chunk(self, row: Dict[str, Any], started_at: float,
                     ended_at: float, checks: Optional[Dict[str, Any]] = None) -> None:
        """Export one span for one chunk attempt, from its metrics row."""
        attributes: Dict[str, Any] = {
            'chunk.id': str(row.get('chunk_id')),
            'chunk.category': str(row.get('chunk_category') or 'unassigned'),
            'llm.model': str(row.get('model')),
            'llm.provider.served': str(row.get('served_provider') or UNAVAILABLE),
            'schema.valid': bool(row.get('schema_valid')),
        }
        numeric = {
            'llm.tokens.input': row.get('prompt_tokens'),
            'llm.tokens.output': row.get('completion_tokens'),
            'llm.tokens.cached': row.get('cached_tokens'),
            'llm.tokens.reasoning': row.get('reasoning_tokens'),
            'llm.cost_usd': row.get('cost_usd'),
            'llm.latency_ms': row.get('latency_ms'),
            'extraction.entities': row.get('entity_count'),
            'extraction.relationships': row.get('edge_count'),
        }
        for key, value in numeric.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                attributes[key] = value
        if checks:
            # Mechanical validation tallies only. Semantic grounding stays a
            # reviewed-label question and is never asserted here.
            attributes['extraction.mechanical_problems'] = len(checks.get('problems') or [])
            attributes['extraction.grounding_holds'] = len(checks.get('grounding_holds') or [])
            attributes['extraction.cautionary_holds'] = len(checks.get('cautionary_holds') or [])
            attributes['extraction.semantic_review'] = 'not_reviewed'
        error = row.get('error')
        ok = bool(row.get('schema_valid')) and not error
        if error:
            # Classified label only: a raw validation message can quote answer
            # text, so spans keep the kind of failure, never the message body.
            text = redact(str(error))
            label = text if text.startswith('http ') else text.split(':')[0]
            attributes['error.type'] = label[:80]
        try:
            from opentelemetry.trace import Status, StatusCode
            span = self.tracer.start_span(
                'golden_extraction.chunk',
                start_time=int(started_at * 1e9),
                attributes=attributes)
            span.set_status(Status(StatusCode.OK if ok else StatusCode.ERROR,
                                   None if ok else attributes.get('error.type', 'chunk failed')))
            span.end(end_time=int(ended_at * 1e9))
        except Exception as exc:  # export problems never break extraction
            print('Telemetry: span export failed (' + redact(str(exc))[:120] + '); run continues.')

    def close(self) -> None:
        """Flush and shut the exporter down; failures only warn."""
        try:
            self.provider.force_flush(timeout_millis=5000)
        except Exception as exc:
            print('Telemetry: flush failed (' + redact(str(exc))[:120] + '); saved rows are intact.')
        try:
            self.provider.shutdown()
        except Exception as exc:
            print('Telemetry: shutdown failed (' + redact(str(exc))[:120] + ').')


def start(config: Dict[str, Any], exporter: Any = None) -> Optional[GoldenTelemetry]:
    """Build telemetry when enabled and installed; otherwise None.

    exporter is injectable so tests can capture spans without a server.
    """
    settings = config.get('telemetry') or {}
    if not settings.get('enabled'):
        return None
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    except ImportError:
        print('Telemetry is enabled but the observability extras are not installed.')
        print('Install them with: pip install -r requirements-observability.txt')
        print('Continuing without traces; metrics.jsonl is unaffected.')
        return None
    if exporter is None:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            endpoint = str(settings.get('otlp_endpoint') or DEFAULT_ENDPOINT)
            exporter = OTLPSpanExporter(endpoint=endpoint, timeout=5)
        except Exception as exc:
            print('Telemetry setup failed (' + redact(str(exc))[:120] + '); continuing without traces.')
            return None
    try:
        provider = TracerProvider(resource=Resource.create({'service.name': SERVICE_NAME}))
        provider.add_span_processor(SimpleSpanProcessor(_WatchingExporter(exporter)))
        tracer = trace.get_tracer('graphintel.golden20', tracer_provider=provider)
        return GoldenTelemetry(provider, tracer)
    except Exception as exc:
        print('Telemetry setup failed (' + redact(str(exc))[:120] + '); continuing without traces.')
        return None
