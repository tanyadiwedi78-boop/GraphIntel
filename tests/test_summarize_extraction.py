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


def attempt_outcome(status, provider="gemini", error_type=None, failover_to=None,
                    served_after_failover=False, chunk_id="c-1"):
    metrics = {"prompt_tokens": None, "completion_tokens": None, "thought_tokens": None,
               "cached_tokens": None, "paced_wait_seconds": 0.0,
               "cost_basis": "assumed_free_unverified", "estimated_cost_usd": None,
               "error_type": error_type, "provider": provider}
    if failover_to is not None:
        metrics["failover_to"] = failover_to
    if served_after_failover:
        metrics["served_after_failover"] = True
    return {"run_id": "run-x", "event": "attempt", "status": status, "chunk_id": chunk_id,
            "metrics": metrics}


def chunk_final_full(status, attempts=1, model="gemini:m:t", valid_first_try=None, empty=None,
                     chunk_id="c-1"):
    event = {"run_id": "run-x", "event": "chunk_final", "chunk_id": chunk_id, "status": status,
             "attempts": attempts, "model": model}
    if valid_first_try is not None:
        event["valid_first_try"] = valid_first_try
    if empty is not None:
        event["empty"] = empty
    return event


def test_api_reliability_counts_attempt_outcomes(tmp_path):
    write_log(tmp_path, "run-a.jsonl", [
        attempt_outcome("valid"),
        attempt_outcome("invalid"),
        attempt_outcome("failed", error_type="http_429"),
        attempt_outcome("failed", error_type="output_cut_off"),
        attempt_outcome("failed", error_type="http_401"),
        chunk_final_full("valid"),
        run_final(10.0),
    ])
    result = run_tool(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Answered and passed validation: 1 (20.0% of attempts)" in result.stdout
    assert "Answered but rejected by validation (content problem): 1 (20.0% of attempts)" in result.stdout
    assert "Failed, transient provider problem (rate limit, quota, network, server): 1" in result.stdout
    assert "Failed, answer cut off at the output token limit: 1" in result.stdout
    assert "Failed, setup problem (key, model id, billing; not retried): 1" in result.stdout
    assert "Failed, blocked by the cost gate before any request: 0" in result.stdout


def test_failover_and_fallback_served_are_measured(tmp_path):
    write_log(tmp_path, "run-a.jsonl", [
        attempt_outcome("failed", provider="gemini", error_type="http_503",
                        failover_to="openai:gpt-6-luna:tag"),
        attempt_outcome("valid", provider="openai", served_after_failover=True),
        chunk_final_full("valid", attempts=2, model="openai:gpt-6-luna:tag"),
        run_final(10.0),
    ])
    result = run_tool(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Attempts by provider: gemini: 1" in result.stdout
    assert "Attempts by provider: openai: 1" in result.stdout
    assert "Failovers to a fallback provider (route-eligible provider failures): 1" in result.stdout
    assert "Attempts served after failover: 1" in result.stdout
    assert "Chunks by model openai:gpt-6-luna:tag: valid=1" in result.stdout


def test_attempts_per_chunk_first_try_and_empty_answers(tmp_path):
    write_log(tmp_path, "run-a.jsonl", [
        attempt_outcome("valid"),
        chunk_final_full("valid", attempts=1, valid_first_try=True, empty=False),
        chunk_final_full("valid", attempts=2, valid_first_try=False, empty=True),
        chunk_final_full("invalid", attempts=3),
        run_final(10.0),
    ])
    result = run_tool(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Attempts per fresh chunk: 1 attempt(s): 1, 2 attempt(s): 1, 3 attempt(s): 1" in result.stdout
    assert "Valid on the first try: 1 of 2 valid chunks" in result.stdout
    assert "Valid answers with zero relationships (empty answers): 1 of 2" in result.stdout


def test_retry_wait_is_summed(tmp_path):
    write_log(tmp_path, "run-a.jsonl", [
        chunk_final("valid", chunk_seconds=8.0, retry_wait=2.0),
        chunk_final("valid", chunk_seconds=9.0, retry_wait=4.0),
        run_final(17.0),
    ])
    result = run_tool(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Retry backoff wait total: 6.0 s" in result.stdout


def test_fallback_rate_reports_both_denominators(tmp_path):
    # Her decision rule is fallback chunks / all chunks; the second rate keeps
    # the same count over only chunks where the primary failed route-eligibly.
    write_log(tmp_path, "run-a.jsonl", [
        # c-1: primary rate-limited, failover, fallback serves the chunk valid.
        attempt_outcome("failed", provider="gemini", error_type="http_429",
                        failover_to="openai:gpt-6-luna:tag", chunk_id="c-1"),
        attempt_outcome("valid", provider="openai", served_after_failover=True, chunk_id="c-1"),
        chunk_final_full("valid", attempts=2, model="openai:gpt-6-luna:tag", chunk_id="c-1"),
        # c-2: clean on the primary, no failure at all.
        attempt_outcome("valid", provider="gemini", chunk_id="c-2"),
        chunk_final_full("valid", chunk_id="c-2"),
        # c-3: a route-eligible primary failure, but no fallback was configured.
        attempt_outcome("failed", provider="gemini", error_type="http_503", chunk_id="c-3"),
        attempt_outcome("valid", provider="gemini", chunk_id="c-3"),
        chunk_final_full("valid", attempts=2, chunk_id="c-3"),
        {"run_id": "run-x", "event": "run_final",
         "summary": {"elapsed_seconds": 10.0, "parked_transient": 2, "circuit_trips": 1}},
    ])
    result = run_tool(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Chunks served by a fallback provider: 1" in result.stdout
    assert "Fallback rate over all processed chunks: 1 of 3 (33.3%)" in result.stdout
    assert "Fallback rate over chunks with a route-eligible primary failure: 1 of 2 (50.0%)" in result.stdout
    assert "Chunks parked for provider-wide failure (pending; replay on rerun): 2" in result.stdout
    assert "Provider circuit openings: 1" in result.stdout


def test_openai_rate_denominator_includes_all_setup_chunks(tmp_path):
    events = [
        attempt_outcome("valid", provider="openai", served_after_failover=True, chunk_id="c-1"),
        chunk_final_full("valid", model="openai:gpt-6-luna:tag", chunk_id="c-1"),
        chunk_final_full("skipped", attempts=0, chunk_id="c-2"),
        {"event": "run_final", "summary": {"total_chunks": 4}},
    ]
    write_log(tmp_path, "run-a.jsonl", events)
    result = run_tool(tmp_path)
    assert "OpenAI use rate over all setup chunks: 1 of 4 (25.0%)" in result.stdout
