"""Run extraction over chunks and save one validated JSON result per chunk."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from app.extraction.client import ChunkFailedError, ProviderError, RETRY_LATER, classify_status, wait_before_retry
from app.extraction.validate import validate_extraction
from app.graph.ontology import Ontology

# Bump SCHEMA_VERSION when the saved record shape changes and PROMPT_VERSION when the
# prompt text changes. Both are part of the work key, so old results are never
# mistaken for results of a new recipe.
SCHEMA_VERSION = "s3"
PROMPT_VERSION = "p2-rules-v1"


def build_prompt(chunk: dict, ontology: Ontology) -> str:
    lines = []
    lines.append("Extract entities and relationships from the SEC filing text below.")
    lines.append('Reply with JSON only: {"entities": [...], "relationships": [...]}.')
    lines.append("Each entity has: name, type.")
    lines.append("Allowed entity types: " + ", ".join(ontology.entity_types) + ".")
    lines.append("Each relationship has: relation, source, target, evidence_quote, confidence (0 to 1).")
    lines.append("source and target must be names from your own entities list.")
    lines.append("Allowed relations (source type -> target type):")
    for name in sorted(ontology.relations.keys()):
        source_type, target_type = ontology.relations[name]
        lines.append(f"- {name}: {source_type} -> {target_type}")
    lines.append("List every entity worth remembering, even one that has no relationship.")
    lines.append("evidence_quote must be copied word for word from the text. Do not guess.")
    lines.append("If nothing fits, return empty lists.")
    lines.append("")
    lines.append("TEXT:")
    lines.append(chunk["text"])
    return "\n".join(lines)


def make_work_key(chunk: dict, ontology_fingerprint: str, schema_version: str, prompt_version: str, model_name: str) -> str:
    text_hash = hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()[:16]
    pieces = [text_hash, ontology_fingerprint, schema_version, prompt_version, model_name]
    key = "_".join(pieces)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def result_path(results_dir: str, chunk_id: str, work_key: str) -> str:
    return os.path.join(results_dir, chunk_id + "__" + work_key + ".json")


def load_result(results_dir: str, chunk_id: str, work_key: str) -> Optional[dict]:
    path = result_path(results_dir, chunk_id, work_key)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def already_done(results_dir: str, chunk_id: str, work_key: str) -> bool:
    saved = load_result(results_dir, chunk_id, work_key)
    if saved is None:
        return False
    return saved.get("status") == "valid"


def save_result(results_dir: str, chunk_id: str, work_key: str, record: dict) -> None:
    os.makedirs(results_dir, exist_ok=True)
    final_path = result_path(results_dir, chunk_id, work_key)
    temp_path = final_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, final_path)


class JsonlEvents:
    """Append-only JSONL event log. One writer per log file."""

    def __init__(self, path, run_id):
        self.path = Path(path)
        self.run_id = run_id

    def __call__(self, event):
        record = dict(event)
        record["run_id"] = self.run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


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
        relationships: List[dict] = []
        entities: List[dict] = []
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
            except ChunkFailedError as error:
                # One chunk cannot complete (for example the answer was cut off at
                # the output token limit). Retrying the identical request would
                # fail the same way, so fail the chunk and continue the run.
                last_error_text = str(error)
                outcome = "failed"
                metrics = dict(getattr(call_model, "last_metrics", {}))
                metrics.setdefault("latency_ms", round((clock() - call_start) * 1000, 3))
                metrics.setdefault("error_type", "chunk_failed")
                emit("attempt", chunk, attempt=attempt, status="failed", metrics=metrics)
                break
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

            is_valid, problems, relationships, entities = validate_extraction(raw_text, chunk, ontology)
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

        # One record per chunk and recipe: run metadata once, then the validated
        # entities and relationships in the model's own shape.
        record = {
            "status": outcome,
            "chunk_id": chunk_id,
            "work_key": work_key,
            "model": model_name,
            "schema_version": SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
            "ontology_fingerprint": fingerprint,
            "filing_accession": chunk["accession"],
            "form_type": chunk["form"],
            "period": chunk["period"],
            "attempts": attempt,
            "problems": problems,
            "last_error": last_error_text,
            "entities": entities,
            "relationships": relationships,
        }
        if outcome == "invalid":
            record["raw_response"] = raw_text
        save(results_dir, chunk_id, work_key, record)
        chunk_seconds = clock() - chunk_start
        chunk_times.append(chunk_seconds)
        emit("chunk_final", chunk, status=outcome, result_reused=False,
            attempts=attempt, retries=max(0, attempt - 1),
            valid_first_try=(outcome == "valid" and attempt == 1),
            empty=(outcome == "valid" and len(relationships) == 0),
            relationships=len(relationships), entities=len(entities),
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
