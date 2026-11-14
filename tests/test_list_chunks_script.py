import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_chunks_folder(tmp_path):
    data_dir = tmp_path / "data"
    chunks_dir = data_dir / "chunks"
    chunks_dir.mkdir(parents=True)
    chunks = []
    number = 0
    for accession in ("0001-26-000001", "0002-26-000002", "0003-26-000003"):
        for item_id, is_table in (("1", False), ("1A", False), ("2", True)):
            number = number + 1
            chunks.append({
                "chunk_id": f"chunk-{number:03d}",
                "ticker": "TEST",
                "form": "10-K",
                "accession": accession,
                "filed": "2026-01-01",
                "period": "2025",
                "item_id": item_id,
                "is_table": is_table,
                "est_tokens": 100,
                "text": f"source text {number}",
            })
    chunks_path = chunks_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as out:
        for chunk in chunks:
            out.write(json.dumps(chunk) + "\n")
    return data_dir, chunks


def run_list_chunks(data_dir, arguments):
    env = {**os.environ, "DATA_DIR": str(data_dir)}
    return subprocess.run(
        [sys.executable, "scripts/list_chunks.py"] + arguments,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )


def printed_sample_ids(output):
    ids = []
    for line in output.splitlines():
        line = line.strip()
        for marker in ("1. ", "2. ", "3. ", "4. ", "5. "):
            if line.startswith(marker) and line[len(marker):].startswith("chunk-"):
                ids.append(line[len(marker):].split()[0])
    return ids


def test_write_subset_creates_file_and_prints_matching_commands(tmp_path):
    data_dir, chunks = make_chunks_folder(tmp_path)
    subset = tmp_path / "data" / "chunks" / "sample5.jsonl"
    result = run_list_chunks(data_dir, ["--sample", "5", "--write-subset", str(subset)])
    assert result.returncode == 0, result.stderr + result.stdout
    ids = printed_sample_ids(result.stdout)
    assert len(ids) == 5
    written = [json.loads(line) for line in subset.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["chunk_id"] for row in written] == ids
    assert f"--chunks {subset}" in result.stdout
    assert "--chunks data/chunks/chunks.jsonl" not in result.stdout
    assert "run_extraction.py phase1 compare-prepare" in result.stdout
    assert "--labels data/labels_sample.json" in result.stdout
    assert "--output data/compare-sample-prepare.json" in result.stdout
    assert "Do NOT run this yet" in result.stdout
    assert "Fallback stays OFF" in result.stdout
    assert "python -m app.extraction.pipeline" not in result.stdout


def test_write_subset_refuses_to_overwrite_existing_file(tmp_path):
    data_dir, chunks = make_chunks_folder(tmp_path)
    subset = tmp_path / "data" / "chunks" / "sample5.jsonl"
    subset.write_text("keep me\n", encoding="utf-8")
    result = run_list_chunks(data_dir, ["--sample", "5", "--write-subset", str(subset)])
    assert result.returncode == 1
    assert "already exists" in result.stdout
    assert subset.read_text(encoding="utf-8") == "keep me\n"


def test_sample_without_write_subset_prints_hint_and_writes_nothing(tmp_path):
    data_dir, chunks = make_chunks_folder(tmp_path)
    result = run_list_chunks(data_dir, ["--sample", "5"])
    assert result.returncode == 0, result.stderr + result.stdout
    assert "--write-subset" in result.stdout
    assert "too large for one job" in result.stdout
    assert not (tmp_path / "data" / "chunks" / "sample5.jsonl").exists()


def test_write_subset_without_sample_is_rejected(tmp_path):
    data_dir, chunks = make_chunks_folder(tmp_path)
    subset = tmp_path / "data" / "chunks" / "sample5.jsonl"
    result = run_list_chunks(data_dir, ["--write-subset", str(subset)])
    assert result.returncode == 1
    assert "--sample" in result.stdout
    assert not subset.exists()


def test_sample_pick_is_deterministic(tmp_path):
    data_dir, chunks = make_chunks_folder(tmp_path)
    first = tmp_path / "data" / "chunks" / "first.jsonl"
    second = tmp_path / "data" / "chunks" / "second.jsonl"
    run_first = run_list_chunks(data_dir, ["--sample", "5", "--write-subset", str(first)])
    run_second = run_list_chunks(data_dir, ["--sample", "5", "--write-subset", str(second)])
    assert run_first.returncode == 0 and run_second.returncode == 0
    assert first.read_bytes() == second.read_bytes()


def make_grounded_subset(tmp_path):
    data_dir, chunks = make_chunks_folder(tmp_path)
    for chunk in chunks:
        chunk["source_sha256"] = "ab" * 32
    chunks_path = data_dir / "chunks" / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as out:
        for chunk in chunks:
            out.write(json.dumps(chunk) + "\n")
    subset = data_dir / "chunks" / "sample5.jsonl"
    result = run_list_chunks(data_dir, ["--sample", "5", "--write-subset", str(subset)])
    assert result.returncode == 0, result.stderr + result.stdout
    return subset


def test_printed_prepare_command_runs_end_to_end_offline(tmp_path):
    subset = make_grounded_subset(tmp_path)
    job = tmp_path / "sample-job"
    plan_out = tmp_path / "sample-job-prepare.json"
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_extraction.py"), "phase1", "prepare",
         "--job", str(job), "--chunks", str(subset),
         "--gemini-model", "unverified-placeholder-model", "--max-output-tokens", "4096",
         "--output", str(plan_out)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert (job / "plan.json").exists()
    assert plan_out.exists()
    plan = json.loads((job / "plan.json").read_text(encoding="utf-8"))
    statuses = json.dumps(plan)
    assert "Not priced; execution blocked" in statuses


def test_module_entry_point_is_not_a_silent_noop_guard(tmp_path):
    subset = make_grounded_subset(tmp_path)
    job = tmp_path / "module-job"
    result = subprocess.run(
        [sys.executable, "-m", "app.extraction.pipeline", "prepare",
         "--job", str(job), "--chunks", str(subset),
         "--gemini-model", "unverified-placeholder-model", "--max-output-tokens", "4096",
         "--output", str(tmp_path / "out.json")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert not job.exists(), "python -m app.extraction.pipeline must not be the documented entry point"
