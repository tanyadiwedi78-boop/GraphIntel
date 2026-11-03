"""Run extraction over chunks: save as you go, stop cleanly, resume later.
"""

from __future__ import annotations

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
    }
    fingerprint = ontology.fingerprint()
    failed_in_a_row = 0

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        work_key = make_work_key(chunk, fingerprint, SCHEMA_VERSION, PROMPT_VERSION, model_name)

        if already_done(results_dir, chunk_id, work_key):
            summary["skipped_done"] = summary["skipped_done"] + 1
            continue

        prompt = build_prompt(chunk, ontology)
        attempt = 0
        outcome = "failed"
        problems: List[str] = []
        edges: List[dict] = []
        raw_text = ""
        last_error_text = ""

        while attempt < limits.max_attempts_per_chunk:
            if summary["calls_used"] >= limits.max_calls:
                summary["stopped_early"] = True
                summary["stop_reason"] = "call cap reached"
                return summary

            attempt = attempt + 1
            summary["calls_used"] = summary["calls_used"] + 1
            try:
                raw_text = call_model(prompt)
            except ProviderError as error:
                last_error_text = str(error)
                outcome = "failed"
                if classify_status(error.status_code) != RETRY_LATER:
                    summary["stopped_early"] = True
                    summary["stop_reason"] = "setup problem, not retried: " + last_error_text
                    return summary
                if attempt >= limits.max_attempts_per_chunk:
                    break
                if error.retry_after_seconds is not None and error.retry_after_seconds > limits.max_wait_seconds:
                    summary["stopped_early"] = True
                    summary["stop_reason"] = "provider asked for a wait longer than the limit"
                    return summary
                wait = wait_before_retry(
                    attempt, error, limits.base_wait_seconds, limits.max_wait_seconds, limits.jitter_fraction
                )
                sleep(wait)
                continue

            is_valid, problems, edges = validate_extraction(raw_text, chunk, ontology)
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
                return summary

    return summary
