"""Bounded async execution over the existing immutable extraction contract."""

import asyncio
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from app.evaluation.extraction_benchmark import identity
from app.extraction.failures import Failure, classify_failure
from app.extraction.observability import LocalTelemetry
from app.extraction.pipeline import Pipeline, load, money, save_new, validated_record
from app.extraction.reporting import token_counts, write_dashboard


class AsyncRunner:
    """One in-flight request: a stop cannot race a second chunk into a provider."""

    def __init__(self, pipeline: Pipeline, reports_root: Path) -> None:
        self.pipeline = pipeline
        self.reports_root = reports_root
        self.telemetry: Optional[LocalTelemetry] = None
        self.last_start: Dict[str, float] = {}
        self.token_events: Dict[str, list] = {}
        self.used = 0
        self.stopped = False
        self.current_chunk: Optional[str] = None

    def emit(self, event: str, **fields: Any) -> None:
        if self.telemetry is not None:
            self.telemetry.event(event, **fields)

    def validate_limits(self, max_calls: int, limits: Dict[str, dict]) -> None:
        if type(max_calls) is not int or max_calls < 0:
            raise ValueError("Nonnegative total call budget required.")
        for name, limit in limits.items():
            rpm = limit.get("rpm")
            tpm = limit.get("tpm")
            if (
                isinstance(rpm, bool)
                or not isinstance(rpm, (int, float))
                or not math.isfinite(rpm)
                or rpm <= 0
            ):
                raise ValueError("Positive finite RPM required for " + name)
            if tpm is not None and (type(tpm) is not int or tpm <= 0):
                raise ValueError("Positive TPM or explicitly unavailable TPM required.")
        if self.pipeline.primary.recipe["provider"] not in limits:
            raise ValueError("Primary limits missing.")

    async def pace(self, provider: Any, chunk: dict, limits: Dict[str, dict]) -> None:
        name = provider.recipe["provider"]
        limit = limits[name]
        reserved = (
            len(self.pipeline.prompt(chunk).encode())
            + len(json.dumps(self.pipeline.plan["schema"]).encode())
            + 4096
            + provider.recipe.get("max_output_tokens", 0)
        )
        tpm = limit.get("tpm")
        if tpm is not None and reserved > tpm:
            raise ValueError("One request exceeds the conservative token window.")
        now = time.monotonic()
        wait = max(0, self.last_start.get(name, now - 60) + 60 / limit["rpm"] - now)
        events = self.token_events.setdefault(name, [])
        if tpm is not None:
            events[:] = [
                (started, amount) for started, amount in events if now - started < 60
            ]
            while events and sum(amount for _, amount in events) + reserved > tpm:
                wait = max(wait, 60 - (now - events[0][0]))
                break
        if wait > 0:
            self.emit("pacing_wait", provider=name, seconds=wait)
            await asyncio.sleep(wait)
        now = time.monotonic()
        events[:] = [
            (started, amount) for started, amount in events if now - started < 60
        ]
        if tpm is not None and sum(amount for _, amount in events) + reserved > tpm:
            raise ValueError("Token window did not clear.")
        self.last_start[name] = now
        events.append((now, reserved))

    async def await_transport(
        self, provider: Any, chunk: dict, reservation: Optional[dict], span: dict
    ) -> dict:
        # Cancellation waits for the bounded request rather than leaving a
        # worker sending data after the orchestration has reported a stop.
        task = asyncio.create_task(
            asyncio.to_thread(
                self.pipeline.attempt, provider, chunk, self.used, reservation
            )
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            self.emit("cancellation_wait", chunk_id=chunk["chunk_id"])
            try:
                await task
            finally:
                self.telemetry.end_span(span, "CANCELLED")
            raise
        except BaseException as error:
            self.telemetry.end_span(span, "ERROR", error_type=type(error).__name__)
            raise

    async def call(
        self,
        provider: Any,
        chunk: dict,
        parent_span: str,
        stage: str,
        limits: Dict[str, dict],
    ) -> tuple:
        await self.pace(provider, chunk, limits)
        reservation = self.pipeline.reservation(provider, chunk)
        if (
            reservation
            and reservation["local_input_byte_estimate"]
            > reservation["input_units_reserved"]
        ):
            raise ValueError(
                "Conservative input-byte bound exceeds verified model context; review chunk size."
            )
        self.pipeline.check_reserved([reservation])
        self.used += 1
        span = self.telemetry.start_span(
            "llm." + provider.recipe["provider"],
            parent_span,
            chunk_id=chunk["chunk_id"],
            model=provider.recipe["model"],
            stage=stage,
            attempt=self.used,
        )
        self.emit(
            "attempt_started",
            span_id=span["span_id"],
            chunk_id=chunk["chunk_id"],
            attempt=self.used,
            provider=provider.recipe["provider"],
            model=provider.recipe["model"],
            stage=stage,
        )
        result = await self.await_transport(provider, chunk, reservation, span)
        return self.finish_attempt(result, provider, chunk, parent_span, stage, span)

    def finish_attempt(
        self,
        result: dict,
        provider: Any,
        chunk: dict,
        parent_span: str,
        stage: str,
        span: dict,
    ) -> tuple:
        validation_span = self.telemetry.start_span(
            "validate_source", parent_span, chunk_id=chunk["chunk_id"]
        )
        record = validated_record(result.get("raw_text"), chunk, self.pipeline.ontology)
        failure = classify_failure(result, record)
        self.telemetry.end_span(
            validation_span,
            "OK" if record["accepted"] and failure is None else "ERROR",
            accepted=record["accepted"],
        )
        self.telemetry.end_span(
            span,
            "OK" if failure is None else "ERROR",
            category=failure.category if failure else None,
            request_seconds=result.get("request_seconds"),
        )
        self.emit(
            "attempt_finished",
            span_id=span["span_id"],
            chunk_id=chunk["chunk_id"],
            attempt=self.used,
            provider=provider.recipe["provider"],
            model=provider.recipe["model"],
            stage=stage,
            accepted=record["accepted"],
            schema_pass=record.get("schema_pass"),
            http_status=result.get("http_status"),
            category=failure.category if failure else None,
            reason=failure.reason if failure else None,
            usage=token_counts(result.get("usage")),
            request_seconds=result.get("request_seconds"),
        )
        write_dashboard(self.pipeline, self.telemetry.folder)
        return result, record, failure

    def dead_letter(self, chunk: dict, failure: Failure) -> None:
        self.telemetry.append(
            "dead-letter.jsonl",
            {
                "run_id": self.telemetry.run_id,
                "chunk_id": chunk["chunk_id"],
                "plan_hash": self.pipeline.plan["plan_hash"],
                "category": failure.category,
                "reason": failure.reason,
                "replayable": failure.retryable and not failure.systemic,
                "ambiguous": failure.ambiguous,
                "evidence_job": str(self.pipeline.job),
            },
        )

    def restore_chunk(self, chunk: dict, prior: list, record_path: Path) -> bool:
        cache_path, key = self.pipeline.cache_path(chunk)
        if cache_path and cache_path.exists() and not prior:
            cached = load(cache_path)
            record = validated_record(
                cached["record"]["raw_text"], chunk, self.pipeline.ontology
            )
            if (
                cached.get("cache_key") != key
                or identity(cached["record"]) != cached.get("record_hash")
                or not record["accepted"]
            ):
                raise ValueError("Cache identity or source validation failed.")
            record.update(
                cache_hit=True, origin_provider=cached["record"].get("provider")
            )
            save_new(record_path, record)
            self.emit("cache_hit", chunk_id=chunk["chunk_id"])
            return True
        if not prior:
            return False
        latest = prior[-1]
        checked = validated_record(
            latest.get("raw_text"), chunk, self.pipeline.ontology
        )
        failure = classify_failure(latest, checked)
        if failure is None:
            self.pipeline.checkpoint(chunk, latest, recovered=True)
            self.emit("checkpoint_recovered", chunk_id=chunk["chunk_id"])
            return True
        if not failure.retryable or failure.systemic:
            self.dead_letter(chunk, failure)
            self.stopped = True
            self.emit(
                "run_stopped",
                reason="Prior failure requires review.",
                category=failure.category,
            )
        return False

    async def retry_wait(self, prior: list, index: int) -> bool:
        if not prior:
            return True
        delay = prior[-1].get("retry_after")
        policy = self.pipeline.plan["retry_policy"]
        if delay is not None and delay > policy["max_wait_seconds"]:
            self.stopped = True
            self.emit("run_stopped", reason="Retry-After exceeds bounded wait.")
            return False
        wait = min(
            policy["max_wait_seconds"], policy["base_seconds"] * 2 ** max(0, index - 1)
        )
        self.emit("retry_wait", seconds=max(wait, delay or 0))
        await asyncio.sleep(max(wait, delay or 0))
        return True

    async def extract_chunk(
        self,
        chunk: dict,
        prior: list,
        max_calls: int,
        limits: Dict[str, dict],
        fallback_authorized: bool,
    ) -> None:
        record_path = self.pipeline.job / (
            "chunk-" + identity(chunk["chunk_id"]) + ".json"
        )
        if record_path.exists():
            saved = load(record_path)
            self.emit(
                "checkpoint_skipped",
                chunk_id=chunk["chunk_id"],
                accepted=saved["accepted"],
            )
            return
        chunk_span = self.telemetry.start_span(
            "extract_chunk", chunk_id=chunk["chunk_id"]
        )
        status = "ERROR"
        started = time.monotonic()
        try:
            if self.restore_chunk(chunk, prior, record_path):
                status = "OK"
                return
            if self.stopped:
                return
            accepted = await self.extract_attempts(
                chunk,
                prior,
                max_calls,
                limits,
                fallback_authorized,
                chunk_span["span_id"],
                started,
            )
            status = "OK" if accepted else "ERROR"
        finally:
            self.telemetry.end_span(chunk_span, status)

    async def provider_attempts(
        self,
        provider: Any,
        stage: str,
        chunk: dict,
        prior: list,
        max_calls: int,
        limits: Dict[str, dict],
        chunk_span_id: str,
        started: float,
    ) -> bool:
        previous_count = 0
        for attempt in prior:
            if attempt["provider"] == provider.recipe["provider"]:
                previous_count += 1
        retry_count = 1 + self.pipeline.plan["retry_policy"]["retries"]
        for index in range(previous_count, retry_count):
            if self.used >= max_calls:
                self.stopped = True
                self.emit("run_stopped", reason="Total job call budget reached.")
                return False
            if stage == "fallback" and index == 0:
                self.emit(
                    "fallback_switched",
                    chunk_id=chunk["chunk_id"],
                    reason="Eligible primary attempts exhausted.",
                )
            if not await self.retry_wait(prior, index):
                return False
            result, _record, failure = await self.call(
                provider, chunk, chunk_span_id, stage, limits
            )
            prior.append(result)
            if failure is None:
                self.pipeline.checkpoint(
                    chunk, result, chunk_wall_seconds=time.monotonic() - started
                )
                self.emit(
                    "chunk_accepted",
                    chunk_id=chunk["chunk_id"],
                    provider=provider.recipe["provider"],
                )
                return True
            if failure.systemic or not failure.retryable:
                self.dead_letter(chunk, failure)
                self.stopped = True
                self.emit(
                    "run_stopped", category=failure.category, reason=failure.reason
                )
                return False
        return False

    async def extract_attempts(
        self,
        chunk: dict,
        prior: list,
        max_calls: int,
        limits: Dict[str, dict],
        fallback_authorized: bool,
        chunk_span_id: str,
        started: float,
    ) -> bool:
        providers = [(self.pipeline.primary, "primary")]
        if self.pipeline.fallback and fallback_authorized:
            providers.append((self.pipeline.fallback, "fallback"))
        for provider, stage in providers:
            accepted = await self.provider_attempts(
                provider, stage, chunk, prior, max_calls, limits, chunk_span_id, started
            )
            if accepted:
                return True
            if self.stopped:
                return False
        if prior:
            last_result = prior[-1]
            failure = classify_failure(
                last_result,
                validated_record(
                    last_result.get("raw_text"), chunk, self.pipeline.ontology
                ),
            )
            if failure is not None:
                self.pipeline.checkpoint(
                    chunk, last_result, chunk_wall_seconds=time.monotonic() - started
                )
                self.dead_letter(chunk, failure)
                self.stopped = True
                self.emit(
                    "run_stopped",
                    reason="Chunk exhausted eligible attempts; review before continuing.",
                )
        return False

    def preflight(
        self,
        max_calls: int,
        limits: Dict[str, dict],
        live_authorized: bool,
        fallback_authorized: bool,
    ) -> list:
        span = self.telemetry.start_span("preflight")
        status = "ERROR"
        try:
            self.validate_limits(max_calls, limits)
            pipe = self.pipeline
            pipe.preflight(live_authorized, fallback_authorized)
            self.verify_free_scope(fallback_authorized)
            if (
                pipe.primary.recipe.get("execution_mode") != "offline_fixture"
                and limits[pipe.primary.recipe["provider"]].get("tpm") is None
            ):
                raise ValueError("Verified primary TPM is required.")
            if pipe.plan.get("execution_mode") != "sync":
                raise ValueError(
                    "Async orchestration requires a sync-transport job, not native batch."
                )
            if (
                fallback_authorized
                and pipe.fallback
                and pipe.fallback.recipe["provider"] not in limits
            ):
                raise ValueError("Separate fallback limits required.")
            prior_attempts, unknown_claims = pipe.attempts()
            if unknown_claims:
                raise ValueError("Unreconciled attempt: no automatic resubmission.")
            pipe.records()
            pipe.reserved_totals = None
            self.used = len(prior_attempts)
            self.restore_pacing(limits)
            status = "OK"
            return prior_attempts
        finally:
            self.telemetry.end_span(span, status)

    async def run(
        self,
        max_calls: int,
        limits: Dict[str, dict],
        live_authorized: bool = False,
        fallback_authorized: bool = False,
        concurrency: int = 1,
    ) -> Dict[str, Any]:
        if concurrency != 1:
            raise ValueError(
                "Only concurrency 1 is enabled until verified account-wide quota and stop-policy approval."
            )
        pipe = self.pipeline
        self.telemetry = LocalTelemetry(
            self.reports_root, pipe.job, pipe.plan["plan_hash"]
        )
        lock = pipe.job / "run-lock.json"
        locked = False
        status = "ERROR"
        try:
            save_new(lock, {"run_id": self.telemetry.run_id})
            locked = True
            prior_attempts = self.preflight(
                max_calls, limits, live_authorized, fallback_authorized
            )
            for chunk in pipe.plan["chunks"]:
                self.current_chunk = chunk["chunk_id"]
                prior = [
                    item
                    for item in prior_attempts
                    if item["chunk_id"] == chunk["chunk_id"]
                ]
                await self.extract_chunk(
                    chunk, prior, max_calls, limits, fallback_authorized
                )
                if self.stopped:
                    break
            status = "STOPPED" if self.stopped else "OK"
            return {
                "run_id": self.telemetry.run_id,
                "reports": str(self.telemetry.folder),
                "status": status,
                "job_attempts": self.used,
            }
        except BaseException as error:
            self.emit(
                "run_failed",
                error_type=type(error).__name__,
                chunk_id=self.current_chunk,
            )
            raise
        finally:
            active_error = sys.exc_info()[0] is not None
            try:
                self.telemetry.finish(status)
                write_dashboard(pipe, self.telemetry.folder)
            except Exception:
                if not active_error:
                    raise
                # Keep the original extraction failure rather than replacing
                # it with a secondary report failure. Durable claims remain.
            finally:
                if locked:
                    lock.unlink()

    def verify_free_scope(self, fallback_authorized: bool) -> None:
        providers = [self.pipeline.primary]
        if self.pipeline.fallback and fallback_authorized:
            providers.append(self.pipeline.fallback)
        for provider in providers:
            if provider.recipe.get("execution_mode") == "offline_fixture":
                continue
            price = self.pipeline.plan.get("pricing", {}).get(
                provider.recipe["provider"], {}
            )
            if (
                price.get("free_tier_no_billing_verified") is not True
                or money(price.get("input_per_million_usd")) != 0
                or money(price.get("output_per_million_usd")) != 0
            ):
                raise ValueError(
                    "This async path is free-only; verified no-billing pricing required."
                )
            if provider.recipe["provider"] == "openrouter" and not provider.recipe[
                "model"
            ].endswith(":free"):
                raise ValueError(
                    "OpenRouter async execution requires an explicit :free model."
                )

    def restore_pacing(self, limits: Dict[str, dict]) -> None:
        for path in self.pipeline.job.glob("attempt-*-claim.json"):
            claim = load(path)
            name = claim["provider"]
            if name not in limits:
                continue
            timestamp = claim.get("started_at")
            if not timestamp:
                raise ValueError("Prior claim has no pacing timestamp.")
            age = (
                datetime.now(timezone.utc) - datetime.fromisoformat(timestamp)
            ).total_seconds()
            if age < 0:
                raise ValueError("Prior claim timestamp is in the future.")
            if age >= 60:
                continue
            started = time.monotonic() - age
            self.last_start[name] = max(self.last_start.get(name, started), started)
            chunk = next(
                item
                for item in self.pipeline.plan["chunks"]
                if item["chunk_id"] == claim["chunk_id"]
            )
            reserved = (
                len(self.pipeline.prompt(chunk).encode())
                + 4096
                + len(json.dumps(self.pipeline.plan["schema"]).encode())
                + claim["recipe"].get("max_output_tokens", 0)
            )
            self.token_events.setdefault(name, []).append((started, reserved))
