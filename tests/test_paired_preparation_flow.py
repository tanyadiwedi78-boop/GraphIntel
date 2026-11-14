"""Offline paired-comparison preparation: subset helper -> source-bound labels -> compare-prepare."""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_corpus(tmp_path):
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
                "text": f"Acme source text {number}.",
                "source_sha256": "ab" * 32,
            })
    with (chunks_dir / "chunks.jsonl").open("w", encoding="utf-8") as out:
        for chunk in chunks:
            out.write(json.dumps(chunk) + "\n")
    return data_dir, chunks


def run_script(data_dir, script, arguments):
    env = {**os.environ, "DATA_DIR": str(data_dir)}
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script)] + arguments,
        cwd=REPO_ROOT, capture_output=True, text=True, env=env)


def make_subset_and_ids(tmp_path):
    data_dir, chunks = make_corpus(tmp_path)
    subset = data_dir / "chunks" / "sample5.jsonl"
    result = run_script(data_dir, "list_chunks.py", ["--sample", "5", "--write-subset", str(subset)])
    assert result.returncode == 0, result.stderr + result.stdout
    rows = [json.loads(line) for line in subset.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row["chunk_id"] for row in rows]
    assert len(ids) == 5
    return data_dir, chunks, subset, ids


def write_labels(data_dir, ids, labels_path):
    return run_script(data_dir, "review_extraction.py",
                      ["--chunk-ids", ",".join(ids), "--write-labels", str(labels_path)])


def review_everything(labels_path):
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    for entry in labels.values():
        entry["reviewed"] = True
    labels_path.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
    return labels


def test_labels_file_is_source_bound_and_starts_unreviewed(tmp_path):
    data_dir, chunks, subset, ids = make_subset_and_ids(tmp_path)
    labels_path = tmp_path / "labels.json"
    result = write_labels(data_dir, ids, labels_path)
    assert result.returncode == 0, result.stderr + result.stdout
    from app.evaluation.extraction_benchmark import identity
    text_by_id = {chunk["chunk_id"]: chunk["text"] for chunk in chunks}
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    assert set(labels) == set(ids)
    for chunk_id, entry in labels.items():
        assert entry["reviewed"] is False
        assert entry["source_hash"] == identity(text_by_id[chunk_id])
        assert entry["entities"] == [] and entry["relationships"] == []


def test_write_labels_rejects_unknown_ids_and_writes_nothing(tmp_path):
    data_dir, chunks, subset, ids = make_subset_and_ids(tmp_path)
    labels_path = tmp_path / "labels.json"
    result = write_labels(data_dir, ids + ["no-such-chunk"], labels_path)
    assert result.returncode == 1
    assert "no-such-chunk" in result.stdout
    assert "No labels file written" in result.stdout
    assert not labels_path.exists()


def test_write_labels_never_overwrites_filled_entries(tmp_path):
    data_dir, chunks, subset, ids = make_subset_and_ids(tmp_path)
    labels_path = tmp_path / "labels.json"
    assert write_labels(data_dir, ids[:4], labels_path).returncode == 0
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    labels[ids[0]]["reviewed"] = True
    labels[ids[0]]["entities"] = [{"canonical_name": "Acme", "entity_type": "Company"}]
    labels_path.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
    result = write_labels(data_dir, ids, labels_path)
    assert result.returncode == 0, result.stderr + result.stdout
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    assert labels[ids[0]]["reviewed"] is True
    assert labels[ids[0]]["entities"] == [{"canonical_name": "Acme", "entity_type": "Company"}]
    assert ids[4] in labels and labels[ids[4]]["reviewed"] is False


def run_compare_prepare(tmp_path, subset, labels_path, root, output_name="compare-prepare-out.json"):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_extraction.py"), "phase1", "compare-prepare",
         "--job", str(root), "--chunks", str(subset),
         "--gemini-model", "gemini-test-model", "--openai-model", "openai-test-model",
         "--max-output-tokens", "4096", "--labels", str(labels_path),
         "--output", str(tmp_path / output_name)],
        cwd=REPO_ROOT, capture_output=True, text=True)


def test_full_flow_reviewed_labels_reach_two_isolated_job_manifests(tmp_path):
    data_dir, chunks, subset, ids = make_subset_and_ids(tmp_path)
    labels_path = tmp_path / "labels.json"
    assert write_labels(data_dir, ids, labels_path).returncode == 0
    review_everything(labels_path)
    root = tmp_path / "compare-sample"
    result = run_compare_prepare(tmp_path, subset, labels_path, root)
    assert result.returncode == 0, result.stderr + result.stdout
    manifest = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
    assert [p["provider"] for p in manifest["providers"]] == ["gemini", "openai"]
    plans = {}
    for provider in ("gemini", "openai"):
        plan_path = root / provider / "plan.json"
        assert plan_path.exists(), provider
        plans[provider] = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plans["gemini"]["purpose"] == "comparison"
    assert plans["openai"]["purpose"] == "comparison"
    assert plans["gemini"]["fallback"] is None
    assert plans["openai"]["fallback"] is None
    assert plans["gemini"]["chunks"] == plans["openai"]["chunks"]
    assert plans["gemini"]["schema"] == plans["openai"]["schema"]
    assert "Not priced; execution blocked" in json.dumps(plans["gemini"])
    assert "Not priced; execution blocked" in json.dumps(plans["openai"])


def test_compare_prepare_rejects_unreviewed_labels(tmp_path):
    data_dir, chunks, subset, ids = make_subset_and_ids(tmp_path)
    labels_path = tmp_path / "labels.json"
    assert write_labels(data_dir, ids, labels_path).returncode == 0
    result = run_compare_prepare(tmp_path, subset, labels_path, tmp_path / "compare-sample")
    assert result.returncode != 0
    assert "source-bound reviewed labels required" in result.stderr + result.stdout


def test_compare_prepare_refuses_existing_folder(tmp_path):
    data_dir, chunks, subset, ids = make_subset_and_ids(tmp_path)
    labels_path = tmp_path / "labels.json"
    assert write_labels(data_dir, ids, labels_path).returncode == 0
    review_everything(labels_path)
    root = tmp_path / "compare-sample"
    assert run_compare_prepare(tmp_path, subset, labels_path, root).returncode == 0
    second = run_compare_prepare(tmp_path, subset, labels_path, root, output_name="compare-prepare-out-2.json")
    assert second.returncode != 0
    assert "preserve it and choose a new folder" in second.stderr + second.stdout
