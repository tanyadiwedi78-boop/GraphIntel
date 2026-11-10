"""Shared validation and error classification retained across the batch migration.

Legacy synchronous job tests are replaced by the batch contract tests below and
in test_anthropic_batch. Historical source remains in Git before this commit.
"""
import json
import pytest
from app.extraction.client import backoff_seconds, classify_status, RETRY_LATER, STOP_RUN
from app.extraction.validate import validate_extraction
from app.graph.ontology import load_ontology
from tests.fake_model import good_answer
QUOTE = "Acme develops the Widget Engine"
def make_chunk(number):
    return {"chunk_id": "chunk-" + str(number), "text": "Item " + str(number) + ". " + QUOTE + " for customers.",
            "accession": "a1", "form": "10-K", "period": "2026"}
@pytest.fixture
def ontology():
    return load_ontology("configs/ontology.json")

# ---------- validation: provenance, not truth ----------

def test_valid_relationships_keep_the_validated_model_shape(ontology):
    ok, problems, relationships, entities = validate_extraction(good_answer(QUOTE), make_chunk(1), ontology)
    assert ok, problems
    assert relationships == [{
        "relation": "DEVELOPS",
        "source": "Acme Corp",
        "target": "Widget Engine",
        "evidence_quote": QUOTE,
        "confidence": 0.9,
    }]
    assert entities == [{"name": "Acme Corp", "type": "Company"},
                        {"name": "Widget Engine", "type": "Technology"}]


def test_quote_spacing_differences_are_allowed(ontology):
    ok = validate_extraction(good_answer("Acme develops  the Widget Engine"), make_chunk(1), ontology)[0]
    assert ok


def test_quote_not_in_chunk_is_invalid(ontology):
    ok, problems, relationships, entities = validate_extraction(good_answer("Acme invented time travel"), make_chunk(1), ontology)
    assert not ok
    assert relationships == []
    assert entities == []
    assert "evidence_quote does not appear" in problems[0]


def test_relation_outside_ontology_is_invalid(ontology):
    bad = json.loads(good_answer(QUOTE))
    bad["entities"][1]["type"] = "Company"   # DEVELOPS must end at Technology
    ok, problems, _, _ = validate_extraction(json.dumps(bad), make_chunk(1), ontology)
    assert not ok
    assert "not in the ontology" in problems[0]


def test_not_json_and_wrong_shape_are_invalid(ontology):
    assert not validate_extraction("sorry, here you go", make_chunk(1), ontology)[0]
    assert not validate_extraction('{"things": []}', make_chunk(1), ontology)[0]


def test_confidence_must_be_between_0_and_1(ontology):
    bad = json.loads(good_answer(QUOTE))
    bad["relationships"][0]["confidence"] = 7
    assert not validate_extraction(json.dumps(bad), make_chunk(1), ontology)[0]


def test_empty_lists_are_valid(ontology):
    ok, _, relationships, entities = validate_extraction('{"entities": [], "relationships": []}', make_chunk(1), ontology)
    assert ok and relationships == [] and entities == []


def test_standalone_entity_without_relationship_is_kept(ontology):
    answer = '{"entities": [{"name": "Acme Corp", "type": "Company"}], "relationships": []}'
    ok, problems, relationships, entities = validate_extraction(answer, make_chunk(1), ontology)
    assert ok, problems
    assert relationships == []
    assert entities == [{"name": "Acme Corp", "type": "Company"}]


def test_relationship_must_reference_listed_entities(ontology):
    bad = json.loads(good_answer(QUOTE))
    bad["relationships"][0]["source"] = "Ghost Corp"
    ok, problems, _, _ = validate_extraction(json.dumps(bad), make_chunk(1), ontology)
    assert not ok
    assert "not in the entities list" in problems[0]


def test_duplicate_entity_names_are_invalid(ontology):
    bad = json.loads(good_answer(QUOTE))
    bad["entities"].append({"name": "Acme Corp", "type": "Company"})
    ok, problems, _, _ = validate_extraction(json.dumps(bad), make_chunk(1), ontology)
    assert not ok
    assert "duplicate entity name" in problems[0]


def test_entity_type_outside_ontology_is_invalid(ontology):
    bad = json.loads(good_answer(QUOTE))
    bad["entities"][0]["type"] = "Planet"
    ok, problems, _, _ = validate_extraction(json.dumps(bad), make_chunk(1), ontology)
    assert not ok
    assert "not an allowed entity type" in problems[0]


def test_extra_keys_are_rejected_not_silently_kept(ontology):
    """A relationship carrying year or amount is not in the ontology: reject, never keep quietly."""
    bad = json.loads(good_answer(QUOTE))
    bad["relationships"][0]["year"] = 2017
    bad["relationships"][0]["amount"] = "13.7 billion USD"
    ok, problems, _, _ = validate_extraction(json.dumps(bad), make_chunk(1), ontology)
    assert not ok
    assert "unexpected keys" in problems[0]


def test_missing_relationship_key_is_invalid(ontology):
    bad = json.loads(good_answer(QUOTE))
    del bad["relationships"][0]["evidence_quote"]
    ok, problems, _, _ = validate_extraction(json.dumps(bad), make_chunk(1), ontology)
    assert not ok
    assert "missing evidence_quote" in problems[0]


# ---------- errors and waits ----------

def test_which_statuses_are_retried():
    assert classify_status(429) == RETRY_LATER
    assert classify_status(408) == RETRY_LATER
    assert classify_status(503) == RETRY_LATER
    assert classify_status(400) == STOP_RUN
    assert classify_status(403) == STOP_RUN


def test_backoff_doubles_and_is_capped():
    assert backoff_seconds(1, 1.0, 60.0) == 1.0
    assert backoff_seconds(2, 1.0, 60.0) == 2.0
    assert backoff_seconds(3, 1.0, 60.0) == 4.0
    assert backoff_seconds(30, 1.0, 60.0) == 60.0



from tests.test_anthropic_batch import job, FakeAPI, receipt, results
from app.extraction.anthropic_batch import (submit_job, import_results, collect_job,
    write_new, prepare_retry, prepare_plan, digest)

@pytest.mark.parametrize("error", [TimeoutError(), ConnectionError(), OSError("temporary")])
def test_submit_transport_errors_never_auto_retry(job, error):
    folder, plan = job
    api = FakeAPI([error])
    with pytest.raises(type(error)):
        submit_job(folder, plan["plan_id"], "100", True, api, max_calls=2)
    assert len(api.calls) == 1
    assert (folder / "submission-claim.json").exists()
    with pytest.raises(FileExistsError):
        submit_job(folder, plan["plan_id"], "100", True, api, max_calls=2)
    assert len(api.calls) == 1

@pytest.mark.parametrize("cap", [0, 1, -1, True])
def test_call_cap_blocks_entire_batch_before_side_effect(job, cap):
    folder, plan = job
    api = FakeAPI([])
    with pytest.raises(ValueError):
        submit_job(folder, plan["plan_id"], "100", True, api, max_calls=cap)
    assert api.calls == []
    assert not (folder / "submission-claim.json").exists()

@pytest.mark.parametrize("reply", ["not json", '{"id":"wrong"}'])
def test_unknown_receipt_is_retained_and_claim_blocks_repost(job, reply):
    folder, plan = job
    api = FakeAPI([reply])
    with pytest.raises(ValueError):
        submit_job(folder, plan["plan_id"], "100", True, api, max_calls=2)
    assert (folder / "submission-response.txt").read_text() == reply
    with pytest.raises(FileExistsError):
        submit_job(folder, plan["plan_id"], "100", True, api, max_calls=2)
    assert len(api.calls) == 1


def test_invalid_result_keeps_raw_and_requires_separate_repair(job):
    folder, plan = job
    receipt(folder)
    rows = results(plan, "bad JSON")
    raw = "\n".join(json.dumps(row) for row in rows)
    (folder / "provider-results.jsonl").write_text(raw)
    summary = import_results(folder, rows)
    write_new(folder / "summary.json", summary)
    repair = prepare_retry(folder)
    assert summary["counts"] == {"invalid": 2}
    assert repair["round"] == 1
    assert repair["plan_id"] != plan["plan_id"]
    assert (folder / "provider-results.jsonl").read_text() == raw
    assert not (folder / "submission-claim.json").exists()


def test_crash_during_local_import_recovers_without_second_paid_call(job, monkeypatch):
    import app.extraction.anthropic_batch as batch
    folder, plan = job
    receipt(folder)
    raw = "\n".join(json.dumps(row) for row in results(plan))
    api = FakeAPI([{"id": "msgbatch_test", "processing_status": "ended"}, raw])
    original = batch.import_results
    def crash(*args):
        raise KeyboardInterrupt("local import interruption")
    monkeypatch.setattr(batch, "import_results", crash)
    with pytest.raises(KeyboardInterrupt):
        collect_job(folder, api)
    assert (folder / "provider-results.jsonl").read_text() == raw
    monkeypatch.setattr(batch, "import_results", original)
    again = FakeAPI([{"id": "msgbatch_test", "processing_status": "ended"}])
    assert collect_job(folder, again)["counts"] == {"valid": 2}
    assert len(again.calls) == 1 and again.calls[0][0] == "GET"


def test_old_results_and_partial_files_stay_untouched(job, ontology, tmp_path):
    folder, plan = job
    cache = tmp_path / "old-cache"
    cache.mkdir()
    old = cache / "c1-old.json"
    old.write_text('{"status":"valid","schema_version":"s1"}')
    partial = cache / "unfinished.json.tmp"
    partial.write_text("{half a file")
    prepared = prepare_plan(tmp_path / "chunks.jsonl", ontology, 2, cache_dir=cache)
    assert len(prepared["requests"]) == 2
    assert old.read_text() == '{"status":"valid","schema_version":"s1"}'
    assert partial.read_text() == "{half a file"


def test_terminal_errors_do_not_hide_other_valid_chunks(job):
    folder, plan = job
    receipt(folder)
    rows = results(plan)
    rows[0]["result"] = {"type":"expired"}
    summary = import_results(folder, rows)
    assert summary["counts"] == {"expired":1, "valid":1}
    assert len(list((folder / "extractions").glob("*.json"))) == 1


def test_completed_collection_repeats_without_any_transport(job):
    folder, plan = job
    receipt(folder)
    api = FakeAPI([{"id":"msgbatch_test", "processing_status":"ended"},
                   "\n".join(json.dumps(row) for row in results(plan))])
    first = collect_job(folder, api)
    again = FakeAPI([])
    assert collect_job(folder, again) == first
    assert again.calls == []
