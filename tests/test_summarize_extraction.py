"""summarize_extraction.py aggregates real event logs into campaign totals.

No network and no real API key: the tests write small fake .jsonl event logs
into a temp data dir and check the printed aggregates. Numbers in the fake
logs stand in for real run output only inside these tests.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def write_log(data_dir, name, events):
    logs_dir = data_dir / "extraction_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    with open(logs_dir / name, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


def run_tool(data_dir):
    args = [sys.executable, "scripts/summarize_extraction.py"]
    env = dict(os.environ)
    env["DATA_DIR"] = str(data_dir)
    env.pop("GEMINI_API_KEY", None)  # tests never see a real key
    return subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, env=env)


def attempt_event(prompt_tokens=None, completion_tokens=None, paced_wait=0.0, cost_basis="assumed_free_unverified"):
    return {"run_id": "run-x", "event": "attempt", "metrics": {
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "thought_tokens": None, "cached_tokens": None,
        "paced_wait_seconds": paced_wait, "cost_basis": cost_basis,
        "estimated_cost_usd": None, "error_type": None}}


def chunk_final(status, chunk_seconds=None, retry_wait=None):
    event = {"run_id": "run-x", "event": "chunk_final", "chunk_id": "c-1", "status": status}
    if chunk_seconds is not None:
        event["chunk_seconds"] = chunk_seconds
    if retry_wait is not None:
        event["retry_wait_seconds"] = retry_wait
    return event


def run_final(elapsed_seconds):
    return {"run_id": "run-x", "event": "run_final",
            "summary": {"elapsed_seconds": elapsed_seconds}}


def test_no_logs_fails_cleanly(tmp_path):
    result = run_tool(tmp_path)
    assert result.returncode == 1
    assert "No event logs" in result.stdout


def test_counts_statuses_and_fresh_chunks(tmp_path):
    write_log(tmp_path, "run-a.jsonl", [
        chunk_final("valid", chunk_seconds=10.0),
        chunk_final("invalid", chunk_seconds=20.0),
        chunk_final("failed", chunk_seconds=30.0),
        chunk_final("skipped"),
        run_final(60.0),
    ])
    result = run_tool(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Chunks valid: 1" in result.stdout
    assert "Chunks invalid: 1" in result.stdout
    assert "Chunks failed: 1" in result.stdout
    assert "Chunks skipped: 1" in result.stdout
    assert "Chunks total: 4" in result.stdout
    assert "Fresh chunks (not skipped): 3" in result.stdout


def test_aggregates_across_runs_and_reports_throughput(tmp_path):
    write_log(tmp_path, "run-a.jsonl", [
        chunk_final("valid", chunk_seconds=10.0), run_final(30.0)])
    write_log(tmp_path, "run-b.jsonl", [
        chunk_final("valid", chunk_seconds=20.0),
        chunk_final("valid", chunk_seconds=30.0), run_final(90.0)])
    result = run_tool(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Runs: 2" in result.stdout
    assert "Chunks valid: 3" in result.stdout
    assert "Run time total: 120.0 s" in result.stdout
    # 3 fresh chunks over 120 s = 1.5 per minute
    assert "1.5 fresh chunks per minute" in result.stdout


def test_chunk_time_median_and_p95(tmp_path):
    events = []
    for seconds in [1.0, 2.0, 3.0, 4.0]:
        events.append(chunk_final("valid", chunk_seconds=seconds))
    events.append(run_final(10.0))
    write_log(tmp_path, "run-a.jsonl", events)
    result = run_tool(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Chunk time median/p95: 2.5 s / 4.0 s" in result.stdout


def test_token_totals_only_from_attempts_that_report(tmp_path):
    write_log(tmp_path, "run-a.jsonl", [
        attempt_event(prompt_tokens=100, completion_tokens=50, paced_wait=2.0),
        attempt_event(prompt_tokens=None, completion_tokens=10, paced_wait=1.0),
        chunk_final("valid", chunk_seconds=5.0),
        run_final(10.0),
    ])
    result = run_tool(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "prompt_tokens: 100 across 1 attempts with data" in result.stdout
    assert "completion_tokens: 60 across 2 attempts with data" in result.stdout
    assert "thought_tokens: no attempts reported it" in result.stdout
    assert "Rate-limit pacing wait total: 3.0 s" in result.stdout
    assert "Cost basis seen in logs: assumed_free_unverified" in result.stdout


def test_cost_is_reported_unknown_never_zero(tmp_path):
    write_log(tmp_path, "run-a.jsonl", [
        attempt_event(prompt_tokens=10, completion_tokens=5),
        chunk_final("valid", chunk_seconds=1.0),
        run_final(1.0),
    ])
    result = run_tool(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Cost: unknown" in result.stdout
    assert "unknown is not $0" in result.stdout
    assert "$0.0" not in result.stdout


def test_retry_wait_is_summed(tmp_path):
    write_log(tmp_path, "run-a.jsonl", [
        chunk_final("valid", chunk_seconds=8.0, retry_wait=2.0),
        chunk_final("valid", chunk_seconds=9.0, retry_wait=4.0),
        run_final(17.0),
    ])
    result = run_tool(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Retry backoff wait total: 6.0 s" in result.stdout
