"""CLI wiring for the Anthropic-only extraction script. No network, no real API key.
"""
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
    env.pop("ANTHROPIC_API_KEY", None)   # tests never see a real key
    if extra_env is not None:
        env.update(extra_env)
    return subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, env=env)



def prepare(data_dir):
    folder = data_dir / "pilot"
    result = run_script(data_dir, ["prepare", "--job-dir", str(folder), "--max-chunks", "2"])
    assert result.returncode == 0, result.stdout + result.stderr
    return folder, json.loads((folder / "plan.json").read_text())["plan_id"]


def submission_args(folder, plan_id):
    return ["submit", "--job-dir", str(folder), "--approve-plan", plan_id, "--budget-usd", "100"]


def test_missing_chunks_file_explains_what_to_do(tmp_path):
    result = run_script(tmp_path, ["prepare", "--job-dir", str(tmp_path / "pilot")])
    assert result.returncode == 1
    assert "chunks" in result.stdout
    assert "Traceback" not in result.stderr


def test_zero_call_cap_refuses_to_run(tmp_path):
    write_chunks(tmp_path, 3)
    folder, plan_id = prepare(tmp_path)
    result = run_script(tmp_path, submission_args(folder, plan_id) + ["--allow-paid"], {"MAX_EXTRACTION_CALLS":"0"})
    assert result.returncode == 1
    assert "0 permits none" in result.stdout
    assert not (folder / "submission-claim.json").exists()


def test_live_calls_need_explicit_paid_confirmation(tmp_path):
    write_chunks(tmp_path, 3)
    folder, plan_id = prepare(tmp_path)
    result = run_script(tmp_path, submission_args(folder, plan_id))
    assert result.returncode == 1
    assert "explicit paid approval" in result.stdout
    assert not (folder / "submission-claim.json").exists()


def test_live_calls_need_reviewed_plan_and_sufficient_budget(tmp_path):
    write_chunks(tmp_path, 3)
    folder, plan_id = prepare(tmp_path)
    flags = submission_args(folder, "wrong") + ["--allow-paid"]
    result = run_script(tmp_path, flags)
    assert result.returncode == 1 and "plan ID" in result.stdout
    flags = submission_args(folder, plan_id) + ["--allow-paid"]
    flags[flags.index("100")] = "0.01"
    result = run_script(tmp_path, flags)
    assert result.returncode == 1 and "exceeds approved job budget" in result.stdout
    assert not (folder / "submission-claim.json").exists()


def test_missing_api_key_is_a_setup_problem_not_a_crash(tmp_path):
    write_chunks(tmp_path, 3)
    folder, plan_id = prepare(tmp_path)
    result = run_script(tmp_path, submission_args(folder, plan_id) + ["--allow-paid"])
    assert result.returncode == 1
    assert "ANTHROPIC_API_KEY is missing" in result.stdout
    assert "Traceback" not in result.stderr
    assert not (folder / "submission-claim.json").exists()
