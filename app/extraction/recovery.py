"""Prepare a reviewed failed-chunk subset as a new job, never erase old evidence."""

import copy
from pathlib import Path
from typing import Any, Dict, List

from app.evaluation.extraction_benchmark import identity
from app.extraction.failures import classify_failure
from app.extraction.pipeline import Pipeline, prepare, validated_record
from app.extraction.reporting import read_events


def reviewed_chunks(
    pipeline: Pipeline, dead_letter: Path, reviews: Dict[str, Any]
) -> List[dict]:
    attempts, unknown = pipeline.attempts()
    if unknown:
        raise ValueError("Unreconciled claims block replay preparation.")
    pipeline.records()
    selected = []
    seen = set()
    by_id = {}
    for chunk in pipeline.plan["chunks"]:
        by_id[chunk["chunk_id"]] = chunk
    for entry in read_events(dead_letter):
        chunk_id = entry.get("chunk_id")
        if chunk_id in seen:
            continue
        if (
            entry.get("plan_hash") != pipeline.plan["plan_hash"]
            or chunk_id not in by_id
        ):
            raise ValueError("Dead-letter entry is not bound to this immutable job.")
        if entry.get("replayable") is not True or entry.get("ambiguous") is not False:
            raise ValueError(
                "Systemic or ambiguous failures cannot be replayed automatically."
            )
        prior = [attempt for attempt in attempts if attempt["chunk_id"] == chunk_id]
        if not prior:
            raise ValueError("Dead-letter entry has no saved attempt evidence.")
        latest = prior[-1]
        failure = classify_failure(
            latest,
            validated_record(
                latest.get("raw_text"), by_id[chunk_id], pipeline.ontology
            ),
        )
        if (
            failure is None
            or failure.systemic
            or failure.ambiguous
            or not failure.retryable
        ):
            raise ValueError("Saved attempt is not eligible for reviewed replay.")
        if entry.get("category") != failure.category:
            raise ValueError("Dead-letter category differs from saved evidence.")
        review = reviews.get(chunk_id, {})
        chunk = by_id[chunk_id]
        expected = identity([pipeline.plan["plan_hash"], chunk_id, chunk["text"]])
        if (
            review.get("reviewed") is not True
            or review.get("source_binding") != expected
        ):
            raise ValueError(
                "Every replay chunk requires an explicit source-bound review."
            )
        selected.append(chunk)
        seen.add(chunk_id)
    if not selected:
        raise ValueError("No reviewed replayable chunks selected.")
    return selected


def prepare_replay(
    pipeline: Pipeline, dead_letter: Path, reviews: Dict[str, Any], new_job: Path
) -> dict:
    chunks = reviewed_chunks(pipeline, dead_letter, reviews)
    plan = pipeline.plan
    if new_job.exists():
        raise ValueError("Replay target must be a new directory.")
    if new_job.resolve() == pipeline.job.resolve():
        raise ValueError("Replay requires a new job directory.")
    cost_limits = copy.deepcopy(plan.get("cost_limits"))
    if cost_limits:
        documents = {}
        for chunk in chunks:
            document = chunk.get("document_hash") or chunk.get("source_sha256")
            if document in cost_limits["documents"]:
                documents[document] = cost_limits["documents"][document]
        cost_limits["documents"] = documents
    return prepare(
        new_job,
        chunks,
        pipeline.ontology,
        plan["primary"],
        plan["fallback"],
        policy=plan["retry_policy"],
        cost_limits=cost_limits,
        pricing=plan.get("pricing"),
        purpose=plan["purpose"],
        execution_mode="sync",
    )
