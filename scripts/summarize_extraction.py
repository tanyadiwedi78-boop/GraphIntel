"""Summarize what the extraction runs actually did, from their event logs.

Reads every data/extraction_logs/*.jsonl and prints measured totals: chunk
outcomes, run time, chunk timing median/p95, retry and pacing waits, token
usage (where the provider reported it) and the cost basis, plus per-attempt
API reliability (validation pass, content rejections, transient failures,
cut-offs, setup problems), attempts per chunk, first-try valid rate, empty
answers and provider routing (failovers, attempts per provider, outcomes per
model). Every number here is measured from a real run's log; nothing is estimated. Cost is reported as
unknown unless the logged attempts carry a verified price - the client writes
estimated_cost_usd as unknown on purpose, so unknown is never printed as $0.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ConfigError, load_settings


def median_and_p95(values: list):
    if len(values) == 0:
        return None, None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        median = ordered[middle]
    else:
        median = (ordered[middle - 1] + ordered[middle]) / 2
    p95_index = max(0, int(0.95 * len(ordered) + 0.999999) - 1)
    return round(median, 3), round(ordered[p95_index], 3)


def add_tokens(totals: dict, metrics: dict) -> None:
    for key in ["prompt_tokens", "completion_tokens", "thought_tokens", "cached_tokens"]:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            totals[key] = totals[key] + value
            totals[key + "_attempts"] = totals[key + "_attempts"] + 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize extraction runs from their event logs.")
    args = parser.parse_args()

    try:
        settings = load_settings()
    except ConfigError as error:
        print(f"Settings problem: {error}")
        return 1

    logs_dir = str(settings.data_dir / "extraction_logs")
    paths = sorted(glob.glob(os.path.join(logs_dir, "*.jsonl")))
    if len(paths) == 0:
        print(f"No event logs in {logs_dir}. Run scripts/run_extraction.py first.")
        return 1

    run_count = 0
    status_counts = {}
    chunk_seconds = []
    attempt_status = {}
    attempt_errors = {}
    attempt_providers = {}
    failover_count = 0
    fallback_served = 0
    chunk_attempt_counts = {}
    first_try_valid = 0
    empty_valid = 0
    model_outcomes = {}
    retry_wait_total = 0.0
    paced_wait_total = 0.0
    elapsed_total = 0.0
    attempt_count = 0
    cost_basis_seen = set()
    doc_attempts = {}
    doc_status = {}
    doc_seconds = {}
    doc_tokens = {}
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "thought_tokens": 0, "cached_tokens": 0,
              "prompt_tokens_attempts": 0, "completion_tokens_attempts": 0,
              "thought_tokens_attempts": 0, "cached_tokens_attempts": 0}

    for path in paths:
        run_count = run_count + 1
        run_id = os.path.basename(path)
        run_status = {}
        run_elapsed = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                event = json.loads(line)
                kind = event.get("event")
                if kind == "attempt":
                    attempt_count = attempt_count + 1
                    metrics = event.get("metrics", {})
                    add_tokens(totals, metrics)
                    outcome = str(event.get("status"))
                    attempt_status[outcome] = attempt_status.get(outcome, 0) + 1
                    provider = metrics.get("provider")
                    if isinstance(provider, str) and provider != "":
                        attempt_providers[provider] = attempt_providers.get(provider, 0) + 1
                    if metrics.get("failover_to"):
                        failover_count = failover_count + 1
                    if metrics.get("served_after_failover"):
                        fallback_served = fallback_served + 1
                    if outcome == "failed":
                        error_type = metrics.get("error_type")
                        if not isinstance(error_type, str) or error_type == "":
                            error_type = "unknown"
                        attempt_errors[error_type] = attempt_errors.get(error_type, 0) + 1
                    doc = event.get("document_id")
                    if isinstance(doc, str) and doc != "":
                        doc_attempts[doc] = doc_attempts.get(doc, 0) + 1
                        if doc not in doc_tokens:
                            doc_tokens[doc] = {"prompt_tokens": 0, "completion_tokens": 0,
                                               "thought_tokens": 0, "cached_tokens": 0,
                                               "prompt_tokens_attempts": 0, "completion_tokens_attempts": 0,
                                               "thought_tokens_attempts": 0, "cached_tokens_attempts": 0}
                        add_tokens(doc_tokens[doc], metrics)
                    paced = metrics.get("paced_wait_seconds")
                    if isinstance(paced, (int, float)) and not isinstance(paced, bool):
                        paced_wait_total = paced_wait_total + paced
                    basis = metrics.get("cost_basis")
                    if isinstance(basis, str) and basis != "":
                        cost_basis_seen.add(basis)
                elif kind == "chunk_final":
                    status = str(event.get("status"))
                    status_counts[status] = status_counts.get(status, 0) + 1
                    run_status[status] = run_status.get(status, 0) + 1
                    if status in ("valid", "invalid", "failed"):
                        n_attempts = event.get("attempts")
                        if isinstance(n_attempts, int) and not isinstance(n_attempts, bool) and n_attempts > 0:
                            chunk_attempt_counts[n_attempts] = chunk_attempt_counts.get(n_attempts, 0) + 1
                        if event.get("valid_first_try") is True:
                            first_try_valid = first_try_valid + 1
                        if status == "valid" and event.get("empty") is True:
                            empty_valid = empty_valid + 1
                        model = event.get("model")
                        if isinstance(model, str) and model != "":
                            if model not in model_outcomes:
                                model_outcomes[model] = {}
                            model_outcomes[model][status] = model_outcomes[model].get(status, 0) + 1
                    seconds = event.get("chunk_seconds")
                    if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
                        chunk_seconds.append(seconds)
                    doc = event.get("document_id")
                    if isinstance(doc, str) and doc != "":
                        if doc not in doc_status:
                            doc_status[doc] = {}
                        doc_status[doc][status] = doc_status[doc].get(status, 0) + 1
                        if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
                            if doc not in doc_seconds:
                                doc_seconds[doc] = []
                            doc_seconds[doc].append(seconds)
                    waited = event.get("retry_wait_seconds")
                    if isinstance(waited, (int, float)) and not isinstance(waited, bool):
                        retry_wait_total = retry_wait_total + waited
                elif kind == "run_final":
                    summary = event.get("summary", {})
                    elapsed = summary.get("elapsed_seconds")
                    if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
                        run_elapsed = elapsed
                        elapsed_total = elapsed_total + elapsed
        line_text = "run " + run_id + ":"
        for status in sorted(run_status.keys()):
            line_text = line_text + " " + status + "=" + str(run_status[status])
        if run_elapsed is not None:
            line_text = line_text + " (" + str(run_elapsed) + " s)"
        print(line_text)

    print("=" * 60)
    print(f"Runs: {run_count}")
    total_chunks = 0
    for status in sorted(status_counts.keys()):
        print(f"Chunks {status}: {status_counts[status]}")
        total_chunks = total_chunks + status_counts[status]
    print(f"Chunks total: {total_chunks}")
    fresh = status_counts.get("valid", 0) + status_counts.get("invalid", 0) + status_counts.get("failed", 0)
    print(f"Fresh chunks (not skipped): {fresh}")
    print(f"Model attempts: {attempt_count}")
    print(f"Run time total: {round(elapsed_total, 3)} s (sum of run times; time between runs is not counted)")
    if elapsed_total > 0:
        print(f"Throughput: {round(fresh / elapsed_total * 60, 2)} fresh chunks per minute of run time")
    median, p95 = median_and_p95(chunk_seconds)
    if median is not None:
        print(f"Chunk time median/p95: {median} s / {p95} s")
    all_docs = []
    for doc in list(doc_status.keys()) + list(doc_attempts.keys()):
        if doc not in all_docs:
            all_docs.append(doc)
    if len(all_docs) > 0:
        print("Per-document measured totals (the per-document budget inputs):")
        for doc in sorted(all_docs):
            line_text = "  " + doc + ": calls=" + str(doc_attempts.get(doc, 0))
            for status in sorted(doc_status.get(doc, {}).keys()):
                line_text = line_text + " " + status + "=" + str(doc_status[doc][status])
            median, p95 = median_and_p95(doc_seconds.get(doc, []))
            if median is not None:
                line_text = line_text + " chunk-time-median=" + str(median) + "s"
            tokens = doc_tokens.get(doc)
            if tokens is not None and tokens["prompt_tokens_attempts"] > 0:
                line_text = line_text + " prompt-tokens=" + str(tokens["prompt_tokens"])
            print(line_text)
        print("Cost per document in dollars: unknown until a verified price source exists.")
        print("The calls, tokens and chunk times above are the measured per-document budget inputs.")
    print(f"Retry backoff wait total: {round(retry_wait_total, 3)} s")
    print(f"Rate-limit pacing wait total: {round(paced_wait_total, 3)} s")
    if attempt_count > 0:
        print("API reliability (per attempt, measured from the logs):")
        answered_valid = attempt_status.get("valid", 0)
        answered_invalid = attempt_status.get("invalid", 0)
        print(f"  Answered and passed validation: {answered_valid} "
              f"({round(100 * answered_valid / attempt_count, 1)}% of attempts)")
        print(f"  Answered but rejected by validation (content problem): {answered_invalid} "
              f"({round(100 * answered_invalid / attempt_count, 1)}% of attempts)")
        # Failed attempts grouped by cause. The bucket boundaries are fixed so
        # a transient provider problem is never confused with a content or
        # setup problem.
        buckets = {"transient": 0, "cut_off": 0, "setup": 0, "cost_blocked": 0, "other": 0}
        other_names = {}
        for error_type in sorted(attempt_errors.keys()):
            count = attempt_errors[error_type]
            bucket = "other"
            if error_type in ("output_cut_off", "chunk_failed"):
                bucket = "cut_off"
            elif error_type == "cost_blocked":
                bucket = "cost_blocked"
            elif error_type == "transport_or_timeout":
                bucket = "transient"
            elif error_type.startswith("http_") and error_type[5:].isdigit():
                code = int(error_type[5:])
                if code == 429 or code == 408 or code >= 500:
                    bucket = "transient"
                else:
                    bucket = "setup"
            buckets[bucket] = buckets[bucket] + count
            if bucket == "other":
                other_names[error_type] = count
        print(f"  Failed, transient provider problem (rate limit, quota, network, server): {buckets['transient']}")
        print(f"  Failed, answer cut off at the output token limit: {buckets['cut_off']}")
        print(f"  Failed, setup problem (key, model id, billing; not retried): {buckets['setup']}")
        print(f"  Failed, blocked by the cost gate before any request: {buckets['cost_blocked']}")
        for error_type in sorted(other_names.keys()):
            print(f"  Failed, other ({error_type}): {other_names[error_type]}")
        if len(chunk_attempt_counts) > 0:
            parts = []
            for n in sorted(chunk_attempt_counts.keys()):
                parts.append(str(n) + " attempt(s): " + str(chunk_attempt_counts[n]))
            print("  Attempts per fresh chunk: " + ", ".join(parts))
        valid_chunks = status_counts.get("valid", 0)
        if valid_chunks > 0:
            print(f"  Valid on the first try: {first_try_valid} of {valid_chunks} valid chunks")
            print(f"  Valid answers with zero relationships (empty answers): {empty_valid} of {valid_chunks}")
        print("Provider routing:")
        if len(attempt_providers) > 0:
            for provider in sorted(attempt_providers.keys()):
                print(f"  Attempts by provider: {provider}: {attempt_providers[provider]}")
        print(f"  Failovers to a fallback provider (route-eligible provider failures): {failover_count}")
        print(f"  Attempts served after failover: {fallback_served}")
        for model in sorted(model_outcomes.keys()):
            outcomes = model_outcomes[model]
            line_text = "  Chunks by model " + model + ":"
            for status in sorted(outcomes.keys()):
                line_text = line_text + " " + status + "=" + str(outcomes[status])
            print(line_text)
    print("Token usage (provider-reported):")
    for key in ["prompt_tokens", "completion_tokens", "thought_tokens", "cached_tokens"]:
        attempts = totals[key + "_attempts"]
        if attempts > 0:
            print(f"  {key}: {totals[key]} across {attempts} attempts with data")
        else:
            print(f"  {key}: no attempts reported it")
    if len(cost_basis_seen) > 0:
        print("Cost basis seen in logs: " + ", ".join(sorted(cost_basis_seen)))
    print("Cost: unknown. The logs record estimated_cost_usd as unknown on purpose;")
    print("unknown is not $0. Add a verified price source before printing dollars.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
