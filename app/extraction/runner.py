"""Run extraction over chunks:
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List

from app.extraction.errors import ProviderError, RETRY_LATER, classify_status, wait_before_retry
from app.extraction.ledger import already_done, make_work_key, save_result
from app.extraction.prompt import PROMPT_VERSION, SCHEMA_VERSION, build_prompt
from app.extraction.schema import validate_extraction
from app.graph.ontology import Ontology


@dataclass(frozen=True)
class RunLimits:
    """Proposed starting values. Tune them from a small pilot, not from guesses."""
    max_calls: int = 100                 # whole run; every attempt counts. MAX_EXTRACTION_CALLS 
    max_attempts_per_chunk: int = 3      # TOTAL calls for one chunk, so at most 2 waits
    base_wait_seconds: float = 1.0
    max_wait_seconds: float = 60.0       
    jitter_fraction: float = 0.1
    stop_after_failed_chunks: int = 5    


def run_extraction(
    chunks: List[dict],
    call_model: Callable[[str], str],
    ontology: Ontology,
    model_name: str,
    results_dir: str,
    limits: RunLimits,
    sleep: Callable[[float], None] = time.sleep,
    save: Callable[[str, str, str, dict], None] = save_result,
    clock: Callable[[], float] = time.perf_counter,
    log_event=None,
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "total_chunks": len(chunks),
        "skipped_done": 0,
        "valid": 0,
        "invalid": 0,
        "failed": 0,
        "calls_used": 0,
        "stopped_early": False,
        "stop_reason": "",
        "elapsed_seconds": 0.0,
        "processed_chunks": 0,
        "chunks_per_minute": 0.0,
        "chunk_median_seconds": None,
        "chunk_p95_seconds": None,
        "retry_wait_seconds": 0.0,
    }
    fingerprint = ontology.fingerprint()
    failed_in_a_row = 0
    run_start = clock()
    chunk_times: List[float] = []

    def finish() -> Dict[str, object]:
        elapsed = clock() - run_start
        summary["elapsed_seconds"] = round(elapsed, 3)
        fresh = summary["valid"] + summary["invalid"] + summary["failed"]
        summary["processed_chunks"] = fresh
        if elapsed > 0:
            summary["chunks_per_minute"] = round(fresh / elapsed * 60, 2)
        if len(chunk_times) > 0:
            ordered = sorted(chunk_times)
            middle = len(ordered) // 2
            if len(ordered) % 2 == 1:
                median = ordered[middle]
            else:
                median = (ordered[middle - 1] + ordered[middle]) / 2
            p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
            summary["chunk_median_seconds"] = round(median, 3)
            summary["chunk_p95_seconds"] = round(ordered[p95_index], 3)
        return summary


    def emit(kind, chunk, **fields):
        if log_event is None:
            return
        event = {"event": kind, "chunk_id": chunk["chunk_id"],
                "document_id": chunk.get("accession"), "section": chunk.get("item_id"),
                "form": chunk.get("form"), "start": chunk.get("start"), "end": chunk.get("end"),
                "model": model_name, "prompt_version": PROMPT_VERSION,
                "schema_version": SCHEMA_VERSION}
        event.update(fields)
        log_event(event)

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        work_key = make_work_key(chunk, fingerprint, SCHEMA_VERSION, PROMPT_VERSION, model_name)

        if already_done(results_dir, chunk_id, work_key):
            summary["skipped_done"] = summary["skipped_done"] + 1
            emit("chunk_final", chunk, status="skipped", result_reused=True,
                attempts=0, valid_first_try=None, retries=0)
            continue

        prompt = build_prompt(chunk, ontology)

        attempt = 0
        outcome = "failed"
        problems: List[str] = []
        edges: List[dict] = []
        raw_text = ""
        last_error_text = ""
        chunk_start = clock()
        chunk_retry_wait = 0.0

        while attempt < limits.max_attempts_per_chunk:
            if summary["calls_used"] >= limits.max_calls:
                summary["stopped_early"] = True
                summary["stop_reason"] = "call cap reached"
                chunk_seconds = clock() - chunk_start
                emit("chunk_final", chunk, status="stopped", result_reused=False,
                    attempts=attempt, retries=max(0, attempt - 1), reason="call cap reached",
                    chunk_seconds=round(chunk_seconds, 3), retry_wait_seconds=round(chunk_retry_wait, 3))
                return finish()

            attempt = attempt + 1
            summary["calls_used"] = summary["calls_used"] + 1
            call_start = clock()
            try:
                raw_text = call_model(prompt)
            except ProviderError as error:
                last_error_text = str(error)
                outcome = "failed"
                metrics = dict(getattr(call_model, "last_metrics", {}))
                metrics.setdefault("latency_ms", round((clock() - call_start) * 1000, 3))
                metrics.setdefault("error_type", "http_" + str(error.status_code))
                emit("attempt", chunk, attempt=attempt, status="failed", metrics=metrics)
                if classify_status(error.status_code) != RETRY_LATER:
                    summary["stopped_early"] = True
                    summary["stop_reason"] = "setup problem, not retried: " + last_error_text
                    chunk_seconds = clock() - chunk_start
                    emit("chunk_final", chunk, status="stopped", attempts=attempt,
                        retries=attempt - 1, result_reused=False, reason=summary["stop_reason"],
                        chunk_seconds=round(chunk_seconds, 3), retry_wait_seconds=round(chunk_retry_wait, 3))
                    return finish()
                if attempt >= limits.max_attempts_per_chunk:
                    break
                if error.retry_after_seconds is not None and error.retry_after_seconds > limits.max_wait_seconds:
                    summary["stopped_early"] = True
                    summary["stop_reason"] = "provider asked for a wait longer than the limit"
                    chunk_seconds = clock() - chunk_start
                    emit("chunk_final", chunk, status="stopped", attempts=attempt,
                        retries=attempt - 1, result_reused=False, reason=summary["stop_reason"],
                        chunk_seconds=round(chunk_seconds, 3), retry_wait_seconds=round(chunk_retry_wait, 3))
                    return finish()
                wait = wait_before_retry(
                    attempt, error, limits.base_wait_seconds, limits.max_wait_seconds, limits.jitter_fraction
                )
                sleep(wait)
                chunk_retry_wait = chunk_retry_wait + wait
                summary["retry_wait_seconds"] = round(summary["retry_wait_seconds"] + wait, 3)
                continue

            is_valid, problems, edges = validate_extraction(raw_text, chunk, ontology)
            metrics = dict(getattr(call_model, "last_metrics", {}))
            metrics.setdefault("latency_ms", round((clock() - call_start) * 1000, 3))
            validation_status = "invalid"
            if not is_valid:
                metrics["error_type"] = "validation_failed"
            if is_valid:
                validation_status = "valid"
                last_error_text = ""
            emit("attempt", chunk, attempt=attempt, status=validation_status,
                metrics=metrics, problems=problems)
            if is_valid:
                outcome = "valid"
                break
            outcome = "invalid"   # bad answer: try again at once, no wait


        record = {
            "status": outcome,
            "chunk_id": chunk_id,
            "work_key": work_key,
            "model": model_name,
            "schema_version": SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
            "ontology_fingerprint": fingerprint,
            "attempts": attempt,
            "problems": problems,
            "last_error": last_error_text,
            "edges": edges,
        }
        if outcome == "invalid":
            record["raw_response"] = raw_text
        save(results_dir, chunk_id, work_key, record)
        names = set()
        for edge in edges:
            names.add((edge["source_type"], edge["source_name"]))
            names.add((edge["target_type"], edge["target_name"]))
        chunk_seconds = clock() - chunk_start
        chunk_times.append(chunk_seconds)
        emit("chunk_final", chunk, status=outcome, result_reused=False,
            attempts=attempt, retries=max(0, attempt - 1),
            valid_first_try=(outcome == "valid" and attempt == 1),
            empty=(outcome == "valid" and len(edges) == 0),
            edges=len(edges), derived_endpoint_entities=len(names),
            chunk_seconds=round(chunk_seconds, 3), retry_wait_seconds=round(chunk_retry_wait, 3))
        if outcome == "valid":
            summary["valid"] = summary["valid"] + 1
            failed_in_a_row = 0
        elif outcome == "invalid":
            summary["invalid"] = summary["invalid"] + 1
        else:
            summary["failed"] = summary["failed"] + 1
            failed_in_a_row = failed_in_a_row + 1
            if failed_in_a_row >= limits.stop_after_failed_chunks:
                summary["stopped_early"] = True
                summary["stop_reason"] = "too many chunks in a row ran out of attempts"
                return finish()

    return finish()

