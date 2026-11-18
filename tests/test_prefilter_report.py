"""Offline checks for the read-only prefilter measurement script."""
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def write_chunks(tmp_path, rows):
    chunks_path = tmp_path / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    return chunks_path


def run_report(arguments):
    return subprocess.run(
        [sys.executable, "scripts/prefilter_report.py"] + arguments,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def sample_rows():
    return [
        {"chunk_id": "c-names", "is_table": False,
         "text": "Apple Inc. develops Widget Pro in California."},
        {"chunk_id": "c-quiet", "is_table": True,
         "text": "table of contents 1 2 3 4 5 6 7 8 9"},
        {"chunk_id": "c-one", "is_table": False,
         "text": "the company announced record results for Apple."},
    ]


def test_skip_counts_match_name_candidates(tmp_path):
    chunks_path = write_chunks(tmp_path, sample_rows())
    result = run_report(["--chunks", str(chunks_path)])
    assert result.returncode == 0, result.stderr
    assert "chunks read: 3" in result.stdout
    # Only the all-lowercase table chunk has zero name candidates.
    assert "would skip: 1 of 3 (33.3%)" in result.stdout
    assert "of those, table chunks: 1" in result.stdout
    # The one-name chunk joins it under the fewer-than-two gate.
    assert "would skip: 2 of 3 (66.7%)" in result.stdout
    assert "heuristic measurement only" in result.stdout


def test_missing_chunks_file_exits_loudly(tmp_path):
    result = run_report(["--chunks", str(tmp_path / "absent.jsonl")])
    assert result.returncode == 1
    assert "chunks file not found" in result.stdout


def test_labels_cross_check_counts_lost_reviewed_facts(tmp_path):
    chunks_path = write_chunks(tmp_path, sample_rows())
    labels = {"c-quiet": {"entities": [{"canonical_name": "Acme"}],
                          "relationships": []},
              "c-absent": {"entities": [{"canonical_name": "Ghost"}],
                           "relationships": []}}
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    result = run_report(["--chunks", str(chunks_path), "--labels", str(labels_path)])
    assert result.returncode == 0, result.stderr
    assert "labeled chunks: 2   not present in this chunks file: 1" in result.stdout
    assert "labeled chunks gate B would skip despite reviewed facts: 1" in result.stdout
    assert "reviewed entities lost: 1   reviewed relationships lost: 0" in result.stdout


def test_real_sample5_chunks_are_never_skipped(tmp_path):
    """The reviewed sample5 set must survive both gates on real data."""
    fixture = json.loads(
        (REPO_ROOT / "tests" / "fixtures" / "sample5-chunks.json").read_text(encoding="utf-8"))
    rows = []
    for chunk_id, text in fixture.items():
        rows.append({"chunk_id": chunk_id, "is_table": False, "text": text})
    chunks_path = write_chunks(tmp_path, rows)
    labels_path = REPO_ROOT / "tests" / "fixtures" / "labels-sample5.json"
    result = run_report(["--chunks", str(chunks_path), "--labels", str(labels_path)])
    assert result.returncode == 0, result.stderr
    assert "would skip: 0 of 5 (0.0%)" in result.stdout
    assert "labeled chunks gate B would skip despite reviewed facts: 0" in result.stdout
    assert "reviewed entities lost: 0   reviewed relationships lost: 0" in result.stdout
