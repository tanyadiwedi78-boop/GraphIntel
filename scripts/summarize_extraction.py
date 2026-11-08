"""Summarize what the extraction runs actually did, from their event logs.

Reads every data/extraction_logs/*.jsonl and prints measured totals: chunk
outcomes, run time, chunk timing median/p95, retry and pacing waits, token
usage (where the provider reported it) and the cost basis. Every number here
is measured from a real run's log; nothing is estimated. Cost is reported as
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
    retry_wait_total = 0.0
    paced_wait_total = 0.0
    elapsed_total = 0.0
    attempt_count = 0
    cost_basis_seen = set()
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
                    seconds = event.get("chunk_seconds")
                    if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
                        chunk_seconds.append(seconds)
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
    print(f"Retry backoff wait total: {round(retry_wait_total, 3)} s")
    print(f"Rate-limit pacing wait total: {round(paced_wait_total, 3)} s")
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
