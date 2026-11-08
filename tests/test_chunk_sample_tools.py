"""Chunk sample tools: list_chunks, review_extraction, and --chunk-ids selection.

No network and no real API key: selection errors are checked before any model
client is built, the review tools only read local files, and the cap/resume
test drives run_extraction with the fake model.
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from app.extraction.runner import RunLimits, run_extraction
from app.graph.ontology import load_ontology
from tests.fake_model import FakeModel, good_answer

REPO_ROOT = Path(__file__).resolve().parent.parent

GOOD_FLAGS = ["--gemini-assume-free", "--gemini-max-rpm", "100", "--gemini-max-tpm", "100000"]


def no_sleep(seconds):
    return None


def write_chunks(data_dir, specs):
    chunks_dir = data_dir / "chunks"
    chunks_dir.mkdir(parents=True)
    with open(chunks_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for spec in specs:
            chunk = {
                "chunk_id": spec["chunk_id"],
                "ticker": spec.get("ticker", "ACME"),
                "form": spec.get("form", "10-K"),
                "accession": spec.get("accession", "0000000000-26-000001"),
                "period": spec.get("period", "2026-01-31"),
                "item_id": spec.get("item_id", "item_1"),
                "is_table": spec.get("is_table", False),
                "est_tokens": spec.get("est_tokens", 100),
                "text": spec["text"],
            }
            f.write(json.dumps(chunk) + "\n")


def sample_specs():
    return [
        {"chunk_id": "chunk-1", "item_id": "cover", "text": "Acme Inc. cover page."},
        {"chunk_id": "chunk-2", "item_id": "item_1a", "text": "Acme Inc. faces supply risk."},
        {"chunk_id": "chunk-3", "item_id": "item_1a", "text": "Acme depends on one supplier."},
    ]


def three_doc_specs():
    specs = []
    for doc in range(3):
        accession = "0000000000-26-00000" + str(doc + 1)
        for item in ["cover", "item_1", "item_1a"]:
            specs.append({
                "chunk_id": "doc" + str(doc) + "-" + item,
                "ticker": "T" + str(doc),
                "accession": accession,
                "item_id": item,
                "is_table": item == "item_1a" and doc == 2,
                "text": "Text for doc " + str(doc) + " " + item + ".",
            })
    return specs


def run_tool(script, data_dir, extra_args=None, extra_env=None):
    args = [sys.executable, "scripts/" + script]
    if extra_args is not None:
        args = args + extra_args
    env = dict(os.environ)
    env["DATA_DIR"] = str(data_dir)
    env.pop("GEMINI_API_KEY", None)  # tests never see a real key
    if extra_env is not None:
        env.update(extra_env)
    return subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, env=env)


def test_list_chunks_summary_groups_by_section(tmp_path):
    write_chunks(tmp_path, sample_specs())
    result = run_tool("list_chunks.py", tmp_path)
    assert result.returncode == 0
    assert "3 chunks" in result.stdout
    assert "cover | 1 chunks | first: chunk-1" in result.stdout
    assert "item_1a | 2 chunks | first: chunk-2" in result.stdout


def test_list_chunks_show_prints_text_and_unknown_id_fails(tmp_path):
    write_chunks(tmp_path, sample_specs())
    shown = run_tool("list_chunks.py", tmp_path, ["--show", "chunk-2"])
    assert shown.returncode == 0
    assert "Acme Inc. faces supply risk." in shown.stdout
    missing = run_tool("list_chunks.py", tmp_path, ["--show", "chunk-99"])
    assert missing.returncode == 1
    assert "chunk-99" in missing.stdout


def test_sample_picks_varied_chunks_and_prints_real_commands(tmp_path):
    write_chunks(tmp_path, three_doc_specs())
    result = run_tool("list_chunks.py", tmp_path, ["--sample", "6"])
    assert result.returncode == 0
    assert "T0" in result.stdout
    assert "T1" in result.stdout
    assert "T2" in result.stdout
    assert "table: True" in result.stdout
    for line in result.stdout.splitlines():
        if line.startswith("python scripts/run_extraction.py"):
            ids_part = line.split("--chunk-ids ", 1)[1].split(" ", 1)[0]
            ids = ids_part.split(",")
            assert len(ids) == 6
            for chunk_id in ids:
                assert chunk_id.startswith("doc")


def test_sample_is_deterministic(tmp_path):
    write_chunks(tmp_path, three_doc_specs())
    first = run_tool("list_chunks.py", tmp_path, ["--sample", "6"])
    second = run_tool("list_chunks.py", tmp_path, ["--sample", "6"])
    assert first.returncode == 0
    assert first.stdout == second.stdout


def test_load_chunks_selects_only_requested_ids(tmp_path):
    write_chunks(tmp_path, sample_specs())
    spec = importlib.util.spec_from_file_location("run_extraction_script",
                                                  REPO_ROOT / "scripts" / "run_extraction.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    chunks_path = tmp_path / "chunks" / "chunks.jsonl"
    picked = module.load_chunks(chunks_path, 10, ["chunk-2"])
    assert len(picked) == 1
    assert picked[0]["chunk_id"] == "chunk-2"
    first_two = module.load_chunks(chunks_path, 2)
    assert len(first_two) == 2
    assert first_two[0]["chunk_id"] == "chunk-1"


def test_run_extraction_rejects_unknown_chunk_id_before_any_call(tmp_path):
    write_chunks(tmp_path, sample_specs())
    args = ["--chunk-ids", "chunk-1,chunk-99"] + GOOD_FLAGS
    env = {"MAX_EXTRACTION_CALLS": "100", "GEMINI_API_KEY": "fake-key"}
    result = run_tool("run_extraction.py", tmp_path, args, env)
    assert result.returncode == 1
    assert "chunk-99" in result.stdout


def test_run_extraction_rejects_empty_chunk_ids(tmp_path):
    write_chunks(tmp_path, sample_specs())
    args = ["--chunk-ids", " , "] + GOOD_FLAGS
    env = {"MAX_EXTRACTION_CALLS": "100", "GEMINI_API_KEY": "fake-key"}
    result = run_tool("run_extraction.py", tmp_path, args, env)
    assert result.returncode == 1
    assert "no ids" in result.stdout


def make_runner_chunk(chunk_id, quote):
    return {
        "chunk_id": chunk_id,
        "text": "Item 1. Acme Corp develops the Widget Engine. " + quote,
        "accession": "0000000000-26-000001",
        "form": "10-K",
        "period": "2026-01-31",
        "item_id": "item_1",
        "start": 0,
        "end": 60,
    }


def test_call_cap_batches_resume_without_repeating_work(tmp_path):
    ontology = load_ontology(REPO_ROOT / "configs" / "ontology.json")
    quote = "develops the Widget Engine"
    chunks = []
    for number in range(6):
        chunks.append(make_runner_chunk("chunk-" + str(number + 1), quote))
    results_dir = str(tmp_path / "extraction")
    limits = RunLimits(max_calls=3)

    first_model = FakeModel([good_answer(quote), good_answer(quote), good_answer(quote)])
    first = run_extraction(chunks, first_model, ontology, "fake-model", results_dir, limits,
                           sleep=no_sleep)
    assert first["valid"] == 3
    assert first["stopped_early"] is True
    assert first["stop_reason"] == "call cap reached"
    assert len(first_model.calls) == 3

    second_model = FakeModel([good_answer(quote), good_answer(quote), good_answer(quote)])
    second = run_extraction(chunks, second_model, ontology, "fake-model", results_dir, limits,
                            sleep=no_sleep)
    assert second["skipped_done"] == 3
    assert second["valid"] == 3
    assert second["stopped_early"] is False
    assert len(second_model.calls) == 3
    assert len(os.listdir(results_dir)) == 6


def test_write_labels_adds_entries_and_keeps_filled_ones(tmp_path):
    write_chunks(tmp_path, sample_specs())
    labels_path = tmp_path / "labels.json"
    first = run_tool("review_extraction.py", tmp_path,
                     ["--chunk-ids", "chunk-1,chunk-2", "--write-labels", str(labels_path)])
    assert first.returncode == 0
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    assert labels["chunk-1"] == {"entities": [], "relationships": [], "notes": ""}
    labels["chunk-1"]["entities"] = [{"name": "Acme Inc.", "type": "Company"}]
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f)
    second = run_tool("review_extraction.py", tmp_path,
                      ["--chunk-ids", "chunk-1,chunk-3", "--write-labels", str(labels_path)])
    assert second.returncode == 0
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    assert labels["chunk-1"]["entities"] == [{"name": "Acme Inc.", "type": "Company"}]
    assert "chunk-3" in labels


def test_review_sheet_shows_source_and_saved_result(tmp_path):
    write_chunks(tmp_path, sample_specs())
    results_dir = tmp_path / "extraction"
    results_dir.mkdir(parents=True)
    record = {
        "status": "valid",
        "chunk_id": "chunk-2",
        "prompt_version": "p2-rules-v1",
        "schema_version": "s3",
        "model": "fake-model",
        "entities": [{"name": "Acme Inc.", "type": "Company"}],
        "relationships": [],
    }
    with open(results_dir / "chunk-2__fakekey.json", "w", encoding="utf-8") as f:
        json.dump(record, f)
    result = run_tool("review_extraction.py", tmp_path, ["--chunk-ids", "chunk-2"])
    assert result.returncode == 0
    assert "Acme Inc. faces supply risk." in result.stdout
    assert "EXTRACTION from" in result.stdout
    assert "Acme Inc. (Company)" in result.stdout


def test_score_reports_wrong_and_missed_items(tmp_path):
    write_chunks(tmp_path, sample_specs())
    results_dir = tmp_path / "extraction"
    results_dir.mkdir(parents=True)
    record = {
        "status": "valid",
        "chunk_id": "chunk-1",
        "prompt_version": "p2-rules-v1",
        "schema_version": "s3",
        "model": "fake-model",
        "entities": [
            {"name": "Acme Inc.", "type": "Company"},
            {"name": "Common Stock", "type": "FinancialMetric"},
        ],
        "relationships": [
            {"relation": "REPORTS", "source": "Acme Inc.", "target": "Common Stock",
             "evidence_quote": "Acme Inc. cover page.", "confidence": 0.8},
        ],
    }
    with open(results_dir / "chunk-1__fakekey.json", "w", encoding="utf-8") as f:
        json.dump(record, f)
    labels_path = tmp_path / "labels.json"
    labels = {
        "chunk-1": {
            "entities": [{"name": "Acme Inc.", "type": "Company"}],
            "relationships": [],
            "notes": "",
        }
    }
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f)
    result = run_tool("review_extraction.py", tmp_path, ["--labels", str(labels_path)])
    assert result.returncode == 0
    assert "precision 1/2 = 0.50" in result.stdout
    assert "recall 1/1 = 1.00" in result.stdout
    assert "wrong extraction: Common Stock (FinancialMetric)" in result.stdout
    assert "wrong extraction: Acme Inc. -REPORTS-> Common Stock" in result.stdout
    assert "precision 0/1 = 0.00" in result.stdout
