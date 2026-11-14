"""CLI wiring for the current pipeline entry point. No network, no keys.

The retired script's paid-submission flags and Anthropic plan flow are gone;
these tests pin the current CLI's gates and offline behavior through real
subprocesses.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def write_chunks(data_dir, count):
    chunks_dir = data_dir / "chunks"
    chunks_dir.mkdir(parents=True)
    with open(chunks_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        number = 0
        while number < count:
            number = number + 1
            chunk = {
                "chunk_id": f"chunk-{number}",
                "text": f"Item 1A. Acme builds the Widget Engine number {number} for customers.",
                "document_hash": "%064x" % number,
                "accession": "0000000000-26-000001",
                "form": "10-K",
                "period": "2026-01-31",
            }
            f.write(json.dumps(chunk) + "\n")
    return chunks_dir / "chunks.jsonl"


def run_cli(arguments, env_extra=None):
    import os
    env = dict(os.environ)
    # Never inherit a real provider key into these tests.
    env.pop('GEMINI_API_KEY', None)
    env.pop('OPENAI_API_KEY', None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run([sys.executable, "scripts/run_extraction.py", "phase1"] + arguments,
                          cwd=REPO_ROOT, capture_output=True, text=True, env=env)


def test_bare_script_starts_nothing_and_explains():
    result = subprocess.run([sys.executable, "scripts/run_extraction.py"],
                            cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0
    assert "No extraction started." in result.stdout


def test_historical_provider_commands_are_refused():
    result = subprocess.run([sys.executable, "scripts/run_extraction.py", "anthropic"],
                            cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 2
    assert "Historical provider commands are not part of this entry point." in result.stdout


def test_help_lists_current_stages():
    result = run_cli(["--help"])
    assert result.returncode == 0
    for stage in ("prepare", "run", "report", "batch-submit", "batch-collect", "demo"):
        assert stage in result.stdout


def test_missing_chunks_file_explains_what_to_do(tmp_path):
    result = run_cli(["prepare", "--job", str(tmp_path / "job"),
                      "--chunks", str(tmp_path / "missing.jsonl"),
                      "--gemini-model", "m1", "--max-output-tokens", "128",
                      "--output", str(tmp_path / "out.json")])
    assert result.returncode == 2
    assert "chunks" in result.stderr.lower()


def test_run_without_authorization_is_refused(tmp_path):
    chunks = write_chunks(tmp_path, 2)
    job = tmp_path / "job"
    result = run_cli(["prepare", "--job", str(job), "--chunks", str(chunks),
                      "--gemini-model", "m1", "--max-output-tokens", "128",
                      "--output", str(tmp_path / "out.json")])
    assert result.returncode == 0, result.stderr
    # A dummy key gets us past the key check to the actual authorization gate;
    # nothing is sent because the refusal happens before any request.
    result = run_cli(["run", "--job", str(job), "--max-calls", "2",
                      "--rpm", "60", "--tpm", "100000",
                      "--output", str(tmp_path / "run-out.json")],
                     {"GEMINI_API_KEY": "dummy-not-real"})
    assert result.returncode == 2
    assert "not authorized" in result.stderr


def test_run_needs_explicit_limits_and_budget(tmp_path):
    chunks = write_chunks(tmp_path, 1)
    job = tmp_path / "job"
    run_cli(["prepare", "--job", str(job), "--chunks", str(chunks),
             "--gemini-model", "m1", "--max-output-tokens", "128",
             "--output", str(tmp_path / "out.json")])
    result = run_cli(["run", "--job", str(job), "--execute-authorized",
                      "--output", str(tmp_path / "run-out.json")],
                     {"GEMINI_API_KEY": "dummy-not-real"})
    assert result.returncode == 2  # missing max-calls/rpm/tpm is an error, not a default


def test_demo_stage_runs_offline_without_keys(tmp_path):
    result = run_cli(["demo", "--output", str(tmp_path / "demo-out.json")])
    assert result.returncode == 0, result.stderr


def test_stage_validation_errors_are_clean_exits_not_traces(tmp_path):
    chunks = write_chunks(tmp_path, 1)
    job = tmp_path / "job"
    run_cli(["prepare", "--job", str(job), "--chunks", str(chunks),
             "--gemini-model", "m1", "--max-output-tokens", "128",
             "--output", str(tmp_path / "out.json")])
    result = run_cli(["run", "--job", str(job), "--max-calls", "-1",
                      "--rpm", "60", "--tpm", "100000", "--execute-authorized",
                      "--output", str(tmp_path / "run-out.json")])
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
