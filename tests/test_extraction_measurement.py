"""Review and scoring mechanics on the current job checkpoint layout.

Retired pieces (Anthropic usage/cost chains, planning estimates, the socket
network guard) are quarantined in legacy_anthropic_tests/; the current
architecture has no repair chains or socket guard to measure - fixture
providers and injected openers make tests offline structurally instead.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.extraction.contracts import CONTRACT_VERSION
from scripts import review_extraction as review

SOURCE = "Acme develops the Widget Engine. Acme mentions Other Co."
CHUNK = {"chunk_id": "c1", "text": SOURCE, "document_hash": "a" * 64,
         "accession": "a1", "form": "10-K", "period": "2026"}
ENTITIES = [{"canonical_name": "Acme", "entity_type": "Company"},
            {"canonical_name": "Widget Engine", "entity_type": "Technology"}]
RELATIONS = [{"relationship_type": "DEVELOPS", "source_entity": "Acme",
              "target_entity": "Widget Engine", "evidence": "Acme develops the Widget Engine.",
              "confidence": 0.9}]


def job_dir(tmp_path, record):
    job = tmp_path / "job"
    job.mkdir()
    if record is not None:
        record = dict(record, chunk_id="c1", source_hash="x", raw_text="{}",
                      schema_pass=True, semantic_evidence_status="Not reviewed",
                      contract_version=CONTRACT_VERSION)
        (job / "chunk-abc123.json").write_text(json.dumps(record))
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    (chunks / "chunks.jsonl").write_text(json.dumps(CHUNK) + "\n")
    return SimpleNamespace(data_dir=tmp_path), job


def labels_file(tmp_path, entries):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(entries))
    return path


def good_record():
    return {"accepted": True, "entities": ENTITIES, "relationships": RELATIONS,
            "provider": "gemini", "terminal_outcome": "returned"}


def reviewed_labels(entities=None, relationships=None):
    return {"c1": {"reviewed": True,
                   "entities": ENTITIES if entities is None else entities,
                   "relationships": RELATIONS if relationships is None else relationships}}


def test_reviewed_empty_false_positive_is_scored(tmp_path, capsys):
    settings, job = job_dir(tmp_path, good_record())
    labels = labels_file(tmp_path, reviewed_labels(entities=[], relationships=[]))
    assert review.run_score(settings, labels, str(job)) == 0
    out = capsys.readouterr().out
    assert '"reviewed_empty": 1' in out
    assert "precision 0/1 = 0.00" in out  # the output is a real false positive


def test_reviewed_empty_correct_output_is_in_coverage(tmp_path, capsys):
    record = {"accepted": True, "entities": [], "relationships": [],
              "provider": "gemini", "terminal_outcome": "returned"}
    settings, job = job_dir(tmp_path, record)
    labels = labels_file(tmp_path, reviewed_labels(entities=[], relationships=[]))
    assert review.run_score(settings, labels, str(job)) == 0
    out = capsys.readouterr().out
    assert "Accepted output coverage: 1/1" in out


@pytest.mark.parametrize("entry", [
    {"entities": [], "relationships": []},                  # legacy: no reviewed flag
    {"reviewed": False, "entities": [], "relationships": []},
    {"reviewed": "yes", "entities": [], "relationships": []}])
def test_unreviewed_and_legacy_entries_are_not_assumed_reviewed(tmp_path, capsys, entry):
    settings, job = job_dir(tmp_path, good_record())
    labels = labels_file(tmp_path, {"c1": entry})
    assert review.run_score(settings, labels, str(job)) == 1
    out = capsys.readouterr().out
    assert "not reviewed" in out or "reviewed must be true or false" in out
    assert "Overall across" not in out


def test_failed_output_counts_as_missed_recall(tmp_path, capsys):
    record = {"accepted": False, "entities": [], "relationships": [],
              "provider": "gemini", "terminal_outcome": "http_failure"}
    settings, job = job_dir(tmp_path, record)
    labels = labels_file(tmp_path, reviewed_labels())
    assert review.run_score(settings, labels, str(job)) == 0
    out = capsys.readouterr().out
    assert "invalid_or_failed" in out
    assert "end-to-end recall 0/2 = 0.00" in out  # expected items are missed, not hidden


def test_missing_output_counts_as_missed_recall(tmp_path, capsys):
    settings, job = job_dir(tmp_path, None)
    labels = labels_file(tmp_path, reviewed_labels())
    assert review.run_score(settings, labels, str(job)) == 0
    out = capsys.readouterr().out
    assert "missing" in out
    assert "end-to-end recall 0/2 = 0.00" in out


def test_incompatible_contract_is_not_scored_as_valid(tmp_path, capsys):
    record = good_record()
    settings, job = job_dir(tmp_path, record)
    # Rewrite the checkpoint with a retired contract version.
    path = next(job.glob("chunk-*.json"))
    data = json.loads(path.read_text())
    data["contract_version"] = "extraction-contract-v1-retired"
    path.write_text(json.dumps(data))
    labels = labels_file(tmp_path, reviewed_labels())
    assert review.run_score(settings, labels, str(job)) == 0
    out = capsys.readouterr().out
    assert "incompatible" in out
    assert '"valid": 0' in out


def test_label_template_starts_unreviewed_and_preserves_completed_labels(tmp_path, capsys):
    settings, job = job_dir(tmp_path, None)
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"c1": {"reviewed": True, "entities": ENTITIES,
                                       "relationships": RELATIONS, "notes": "done"}}))
    assert review.run_write_labels(settings, ["c1", "c2"], path) == 0
    written = json.loads(path.read_text())
    assert written["c1"]["notes"] == "done"  # completed work never overwritten
    assert written["c2"] == {"entities": [], "relationships": [], "notes": "", "reviewed": False}


def test_canonical_labels_do_not_use_fuzzy_alias_scoring(tmp_path, capsys):
    predicted = good_record()
    predicted["entities"] = [{"canonical_name": "Acme Incorporated", "entity_type": "Company"},
                             ENTITIES[1]]
    settings, job = job_dir(tmp_path, predicted)
    labels = labels_file(tmp_path, reviewed_labels())
    assert review.run_score(settings, labels, str(job)) == 0
    out = capsys.readouterr().out
    # "Acme Incorporated" is not fuzzy-matched to "Acme": wrong AND missed.
    assert "wrong extraction: Acme Incorporated" in out
    assert "missed: Acme" in out


def test_case_and_space_only_matching(tmp_path, capsys):
    predicted = good_record()
    predicted["entities"] = [{"canonical_name": "  ACME ", "entity_type": "Company"},
                             ENTITIES[1]]
    settings, job = job_dir(tmp_path, predicted)
    labels = labels_file(tmp_path, reviewed_labels())
    assert review.run_score(settings, labels, str(job)) == 0
    out = capsys.readouterr().out
    assert "precision 2/2 = 1.00" in out
