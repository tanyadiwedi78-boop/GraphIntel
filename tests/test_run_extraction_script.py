import json
import os
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
                "accession": "0000000000-26-000001",
                "form": "10-K",
                "period": "2026-01-31",
            }
            f.write(json.dumps(chunk) + "\n")


def run_script(data_dir, extra_args=None, extra_env=None):
    args = [sys.executable, "scripts/run_extraction.py"]
    if extra_args is not None:
        args = args + extra_args
    env = dict(os.environ)
    env["DATA_DIR"] = str(data_dir)
    env["MAX_EXTRACTION_CALLS"] = "100"
    if extra_env is not None:
        env.update(extra_env)
    return subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, env=env)


def count_results(data_dir):
    results_dir = data_dir / "extraction"
    if not results_dir.exists():
        return 0
    names = []
    for name in os.listdir(results_dir):
        if name.endswith(".json"):
            names.append(name)
    return len(names)


def test_first_run_extracts_and_second_run_skips(tmp_path):
    write_chunks(tmp_path, 4)
    first = run_script(tmp_path)
    assert first.returncode == 0, first.stderr
    assert "New valid results:     4" in first.stdout
    assert count_results(tmp_path) == 4

    second = run_script(tmp_path)
    assert second.returncode == 0, second.stderr
    assert "Already done, skipped: 4" in second.stdout
    assert "New valid results:     0" in second.stdout
    assert count_results(tmp_path) == 4


def test_missing_chunks_file_explains_what_to_do(tmp_path):
    result = run_script(tmp_path)
    assert result.returncode == 1
    assert "build_chunks" in result.stdout


def test_max_chunks_limits_how_many_are_read(tmp_path):
    write_chunks(tmp_path, 6)
    result = run_script(tmp_path, ["--max-chunks", "2"])
    assert result.returncode == 0, result.stderr
    assert "Chunks read:           2" in result.stdout
    assert count_results(tmp_path) == 2


def test_max_extraction_calls_from_env_caps_the_run(tmp_path):
    write_chunks(tmp_path, 5)
    result = run_script(tmp_path, ["--max-chunks", "5"], {"MAX_EXTRACTION_CALLS": "2"})
    assert result.returncode == 0, result.stderr
    assert "Model calls used:      2" in result.stdout
    assert "Stopped early: call cap reached" in result.stdout
    assert count_results(tmp_path) == 2


def test_zero_call_cap_refuses_to_run_and_saves_nothing(tmp_path):
    write_chunks(tmp_path, 3)
    result = run_script(tmp_path, None, {"MAX_EXTRACTION_CALLS": "0"})
    assert result.returncode == 1
    assert "no calls are allowed" in result.stdout
    assert count_results(tmp_path) == 0
