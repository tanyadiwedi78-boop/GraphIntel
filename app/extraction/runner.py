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


# Plain-English meaning for each entity type, shown in the prompt when the type
# exists in the ontology. The ontology stays the source of truth: a type missing
# from configs/ontology.json never appears in the prompt, and a type without a
# meaning here is listed without one.
TYPE_MEANINGS = {
    "Company": "a business acting for itself, such as the filing company or a business it names",
    "Subsidiary": "a company owned or controlled by another company",
    "Industry": "a business sector or market, not one company",
    "Technology": "a specific technology, standard or system",
    "Risk": "a risk, threat or uncertainty the text describes",
    "FinancialMetric": "a financial measure such as revenue, net income or par value per share, whether or not the text states its number; the name of a security itself, such as Common Stock, is not one",
    "Product": "a product or service a company offers",
}

PROMPT_RULES = """RULES:
1. Use only the allowed entity types and relations below.
2. Give an entity a type only when that type truly fits. If no allowed type fits, leave the item out instead of forcing it into the nearest type.
3. A short list, or an empty list, can be the correct answer. Cover pages and boilerplate often contain nothing worth extracting.
4. List every entity that fits a type, even one that has no relationship.
5. A relationship must use two names from your own entities list and follow its direction: relation (source type -> target type).
6. evidence_quote must be copied word for word from the text AND state the relationship you claim. A quote that only repeats the same words is not evidence. A table row naming a stock class does not show that a company reports a financial metric.
7. confidence is your honest estimate, from 0 to 1, that the text supports the item. Do not guess.
"""


def build_prompt(chunk: dict, ontology: Ontology) -> str:
    types_text = ", ".join(ontology.entity_types)

    meaning_lines = []
    for name in ontology.entity_types:
        if name in TYPE_MEANINGS:
            meaning_lines.append("- " + name + ": " + TYPE_MEANINGS[name])
    meanings_text = "\n".join(meaning_lines)

    relation_lines = []
    for name in sorted(ontology.relations.keys()):
        source_type, target_type = ontology.relations[name]
        relation_lines.append("- " + name + ": " + source_type + " -> " + target_type)
    relations_text = "\n".join(relation_lines)

    return f"""Extract entities and relationships from the SEC filing text below.
Reply with JSON only: {{"entities": [...], "relationships": [...]}}.
Each entity has: name, type.
Each relationship has: relation, source, target, evidence_quote, confidence (0 to 1).

{PROMPT_RULES}

Allowed entity types: {types_text}.
{meanings_text}

Allowed relations (source type -> target type):
{relations_text}

TEXT:
{chunk["text"]}"""


def build_retry_prompt(base_prompt: str, problems: List[str]) -> str:
    """Second-and-later attempt for the same chunk: same task, plus what failed.

    Without this the retry sends the identical prompt, and with a fixed seed the
    model tends to repeat the identical violation - each repeat still costs one
    call. Naming the validator's problems gives the model the information it
    needs to correct the answer instead of reproducing it.
    """
    lines = []
    for problem in problems:
        lines.append("- " + problem)
    return (base_prompt
            + "\n\nYour previous answer was rejected for these reasons:\n"
            + "\n".join(lines)
            + "\nFix these problems; the task and the text above are unchanged."
            + " If fixing a problem would require inventing something the text does"
            + " not say, leave that item out instead - a shorter or empty list can"
            + " be the correct answer.")


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
    give_up_after_runs: int = 3          # CONTENT failures in this many separate runs: skip the chunk
    retry_given_up: bool = False         # True: include given-up chunks again (records are kept, never deleted)


def find_compatible_valid(results_dir: str, chunk: dict, ontology_fingerprint: str,
                          current_model: str) -> Optional[dict]:
    """A valid saved result for the same chunk text and recipe from ANY provider.

    Work keys include the model name, so each provider's results are saved as
    separate files and results never mix. But a chunk that ANY provider already
    extracted validly under the same chunk text, ontology, schema and prompt is
    done: never redo finished work. Identity is recomputed, not trusted: the
    record must carry a complete identity (model, work_key, ontology
    fingerprint, schema and prompt versions, status valid), its work key must
    equal make_work_key(this chunk's text, this fingerprint, the current schema
    and prompt versions, the record's own model), and the filename must match.
    Legacy or incomplete records are ignored. Provenance is preserved: the
    accepted record stays on disk under its own provider's key and the skip
    event names the provider that produced it.
    """
    prefix = chunk["chunk_id"] + "__"
    if not os.path.isdir(results_dir):
        return None
    newest = None
    newest_mtime = -1.0
    for filename in os.listdir(results_dir):
        if not filename.startswith(prefix) or not filename.endswith(".json"):
            continue
        path = os.path.join(results_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                record = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict) or record.get("status") != "valid":
            continue
        model = record.get("model")
        if not isinstance(model, str) or model == "" or model == current_model:
            continue
        if record.get("ontology_fingerprint") != ontology_fingerprint:
            continue
        if record.get("schema_version") != SCHEMA_VERSION:
            continue
        if record.get("prompt_version") != PROMPT_VERSION:
            continue
        if not isinstance(record.get("work_key"), str):
            continue
        expected_key = make_work_key(chunk, ontology_fingerprint, SCHEMA_VERSION,
                                     PROMPT_VERSION, model)
        if record["work_key"] != expected_key:
            continue
        if filename != chunk["chunk_id"] + "__" + expected_key + ".json":
            continue
        mtime = os.path.getmtime(path)
        if mtime > newest_mtime:
            newest_mtime = mtime
            newest = record
    return newest


def give_up(saved: Optional[dict], give_up_after_runs: int) -> bool:
    """True when a chunk's own content keeps failing across enough separate runs.

    Only CONTENT failures count: the answer was cut off or never validated, so
    retrying the identical request cannot help. Transient failures (rate limit,
    quota, network, provider 5xx) never count: they are the run's problem, not
    the chunk's, and the chunk must stay pending for a later run.
    """
    if saved is None:
        return False
    status = saved.get("status")
    if status == "invalid":
        kind = "content"  # an invalid answer is always a content problem
    elif status == "failed":
        kind = saved.get("failure_kind")
        if kind is None:
            kind = "content"  # records from before failure_kind existed: old code only saved content failures and exhausted-retry failures; the next attempt under this code re-tags them, so at most stale runs are miscounted
        if kind != "content":
            return False
    else:
        return False
    runs = saved.get("runs")
    if not isinstance(runs, int) or isinstance(runs, bool):
        runs = 1  # records from before this counter existed count as one run
    return runs >= give_up_after_runs


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
        "gave_up": 0,
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
        # A router can fail over to another provider mid-run. Name the ACTIVE
        # provider for this chunk's work key, events and saved record, so
        # provenance always says exactly which provider produced the result.
        # For a single client this never changes.
        model_name = getattr(call_model, "model_name", model_name)
        work_key = make_work_key(chunk, fingerprint, SCHEMA_VERSION, PROMPT_VERSION, model_name)

        saved_record = load_result(results_dir, chunk_id, work_key)
        accepted_record = None
        accepted_from_model = None
        if saved_record is not None and saved_record.get("status") == "valid":
            accepted_record = saved_record
        else:
            # Done is done, whichever provider did it: a compatible valid
            # result (same chunk text, ontology, schema and prompt, identity
            # recomputed) is never reprocessed on another provider.
            compatible = find_compatible_valid(results_dir, chunk, fingerprint, model_name)
            if compatible is not None:
                accepted_record = compatible
                accepted_from_model = compatible.get("model")
        if accepted_record is not None:
            summary["skipped_done"] = summary["skipped_done"] + 1
            emit("chunk_final", chunk, status="skipped", result_reused=True,
                attempts=0, valid_first_try=None, retries=0,
                accepted_from_model=accepted_from_model)
            continue
        if not limits.retry_given_up and give_up(saved_record, limits.give_up_after_runs):
            summary["gave_up"] = summary["gave_up"] + 1
            emit("chunk_final", chunk, status="gave_up", result_reused=True,
                attempts=0, valid_first_try=None, retries=0,
                reason="content failed or invalid in " + str(limits.give_up_after_runs) + " separate runs")
            continue

        base_prompt = build_prompt(chunk, ontology)
        prompt = base_prompt

        attempt = 0
        outcome = "failed"
        failure_kind = "content"
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
            # Snapshot the provider BEFORE the call: a router can fail over
            # inside the call, and this attempt was still served by the one
            # that received it.
            attempt_model = getattr(call_model, "model_name", model_name)
            route_events_before = len(getattr(call_model, "route_events", []))
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
                emit("attempt", chunk, attempt=attempt, status="failed", metrics=metrics,
                    model=attempt_model)
                break
            except ProviderError as error:
                last_error_text = str(error)
                outcome = "failed"
                metrics = dict(getattr(call_model, "last_metrics", {}))
                metrics.setdefault("latency_ms", round((clock() - call_start) * 1000, 3))
                metrics.setdefault("error_type", "http_" + str(error.status_code))
                emit("attempt", chunk, attempt=attempt, status="failed", metrics=metrics,
                    model=attempt_model)
                failure_kind = "transient"  # rate limit, quota, network, 5xx: the run's problem, not the chunk's
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
                routed_away = len(getattr(call_model, "route_events", [])) > route_events_before
                if routed_away:
                    # The router just confirmed an outage and moved the run to
                    # a fallback: retry at once on the new provider instead of
                    # waiting for the one that is down, and never stop for the
                    # old provider's long requested wait.
                    continue
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
                metrics=metrics, problems=problems,
                retry_with_feedback=(prompt != base_prompt), model=attempt_model)
            if is_valid:
                outcome = "valid"
                break
            outcome = "invalid"   # bad answer: try again at once, no wait
            if len(problems) > 0:
                # Show the next attempt what was rejected; otherwise the
                # identical prompt with a fixed seed just repeats the violation.
                prompt = build_retry_prompt(base_prompt, problems)

        # One record per chunk and recipe: run metadata once, then the validated
        # entities and relationships in the model's own shape. Name the provider
        # that produced the FINAL outcome of this chunk: a router may have failed
        # over mid-chunk, and provenance says who actually produced the result.
        model_name = getattr(call_model, "model_name", model_name)
        served_work_key = make_work_key(chunk, fingerprint, SCHEMA_VERSION, PROMPT_VERSION, model_name)
        if served_work_key != work_key:
            # A router failed over mid-chunk: this record belongs to the
            # provider that actually served it, so its work key embeds THAT
            # provider's name and the file lands under that provider's key.
            # Give-up history comes from the serving provider's own earlier
            # record, not the one the run started with.
            saved_record = load_result(results_dir, chunk_id, served_work_key)
            work_key = served_work_key
        previous_runs = 0
        if saved_record is not None:
            saved_runs = saved_record.get("runs")
            if isinstance(saved_runs, int) and not isinstance(saved_runs, bool):
                previous_runs = max(0, saved_runs)
            else:
                previous_runs = 1  # record from before the counter existed
        runs_count = previous_runs + 1
        if outcome == "valid" or (outcome == "failed" and failure_kind == "transient"):
            runs_count = previous_runs  # only separate runs of CONTENT failure count toward give-up
        record = {
            "status": outcome,
            "runs": runs_count,
            "failure_kind": failure_kind,
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
            if failure_kind == "transient":
                # A provider-wide failure (rate limit, quota, network, server
                # error) means later chunks would very likely fail the same
                # way. Stop now instead of burning the call cap on them. The
                # chunk's record above is transient, so it stays pending and
                # rerunning the same command resumes exactly here.
                summary["stopped_early"] = True
                summary["stop_reason"] = ("provider-side failure (rate limit, quota, network or server error); "
                                          "stopped so later chunks are not charged against the call cap. "
                                          "This chunk stays pending: rerun the same command when the provider recovers.")
                return finish()
            if failed_in_a_row >= limits.stop_after_failed_chunks:
                summary["stopped_early"] = True
                summary["stop_reason"] = "too many chunks in a row ran out of attempts"
                return finish()

    return finish()
