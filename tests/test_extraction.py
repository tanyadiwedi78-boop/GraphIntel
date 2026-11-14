"""Shared validation and error classification on the current contract.

Migrated from the retired app.extraction.validate / client modules to
app.extraction.pipeline.validated_record (Pydantic expanded contract) and
Pipeline.retryable / retry_policy. Native-batch lifecycle coverage that used
to live here moved to tests/test_anthropic_batch.py with the new provider.
"""
import json

import pytest

from app.extraction.pipeline import retry_policy, validated_record, Pipeline, prepare
from app.extraction.providers import CallFailure
from tests.conftest import FakeProvider, demo_chunks, payload_json

QUOTE = "Acme develops the Widget Engine"


def make_chunk(number):
    return {"chunk_id": "chunk-" + str(number), "text": "Item " + str(number) + ". " + QUOTE + " for customers.",
            "document_hash": "%064x" % number, "accession": "a1", "form": "10-K", "period": "2026"}


def good_payload(quote=QUOTE, chunk_id="chunk-1"):
    return json.dumps({
        "entities": [
            {"canonical_name": "Acme Corp", "entity_type": "Company", "aliases": [],
             "description": "A company.", "source_chunk_id": chunk_id,
             "confidence": 0.9, "evidence": quote},
            {"canonical_name": "Widget Engine", "entity_type": "Technology", "aliases": [],
             "description": "A technology.", "source_chunk_id": chunk_id,
             "confidence": 0.9, "evidence": quote}],
        "relationships": [{
            "relationship_type": "DEVELOPS",
            "source_entity": "Acme Corp",
            "target_entity": "Widget Engine",
            "source_chunk_id": chunk_id,
            "evidence": quote,
            "confidence": 0.9,
        }],
    })


# ---------- validation: provenance, not truth ----------

def test_valid_relationships_keep_the_validated_model_shape(ontology):
    record = validated_record(good_payload(), make_chunk(1), ontology)
    assert record["accepted"], record.get("problems")
    assert record["relationships"] == [{
        "relationship_type": "DEVELOPS",
        "source_entity": "Acme Corp",
        "target_entity": "Widget Engine",
        "source_chunk_id": "chunk-1",
        "evidence": QUOTE,
        "confidence": 0.9,
    }]
    assert record["entities"][0]["canonical_name"] == "Acme Corp"
    assert record["entities"][0]["entity_type"] == "Company"
    assert record["semantic_evidence_status"] == "Not reviewed"
    assert record["graph_write_permitted"] is False


def test_quote_spacing_differences_are_allowed(ontology):
    record = validated_record(good_payload("Acme develops  the Widget Engine"), make_chunk(1), ontology)
    assert record["accepted"], record.get("problems")


def test_quote_not_in_chunk_is_invalid(ontology):
    record = validated_record(good_payload("Acme invented time travel"), make_chunk(1), ontology)
    assert not record["accepted"]
    assert record["entities"] == []
    assert record["relationships"] == []
    assert "evidence is absent from the original chunk" in record["problems"][0]


def test_relation_outside_ontology_is_invalid(ontology):
    bad = json.loads(good_payload())
    bad["entities"][1]["entity_type"] = "Company"   # DEVELOPS must end at Technology
    record = validated_record(json.dumps(bad), make_chunk(1), ontology)
    assert not record["accepted"]
    # The frozen Literal contract rejects the wrong type at schema level, or
    # the source check rejects the direction: either way it is refused.
    assert record["schema_pass"] is False or any("ontology" in p for p in record["problems"])


def test_not_json_and_wrong_shape_are_invalid(ontology):
    assert not validated_record("sorry, here you go", make_chunk(1), ontology)["accepted"]
    assert not validated_record('{"things": []}', make_chunk(1), ontology)["accepted"]


def test_confidence_must_be_between_0_and_1(ontology):
    bad = json.loads(good_payload())
    bad["relationships"][0]["confidence"] = 7
    assert not validated_record(json.dumps(bad), make_chunk(1), ontology)["accepted"]


def test_empty_lists_are_valid(ontology):
    record = validated_record('{"entities": [], "relationships": []}', make_chunk(1), ontology)
    assert record["accepted"] and record["relationships"] == [] and record["entities"] == []


def test_standalone_entity_without_relationship_is_kept(ontology):
    answer = json.dumps({"entities": [{
        "canonical_name": "Acme Corp", "entity_type": "Company", "aliases": [],
        "description": "A company.", "source_chunk_id": "chunk-1",
        "confidence": 0.9, "evidence": QUOTE}], "relationships": []})
    record = validated_record(answer, make_chunk(1), ontology)
    assert record["accepted"], record.get("problems")
    assert record["relationships"] == []
    assert [e["canonical_name"] for e in record["entities"]] == ["Acme Corp"]


def test_relationship_must_reference_listed_entities(ontology):
    bad = json.loads(good_payload())
    bad["relationships"][0]["source_entity"] = "Ghost Corp"
    record = validated_record(json.dumps(bad), make_chunk(1), ontology)
    assert not record["accepted"]
    assert "relationship endpoint is absent from extracted entities" in record["problems"]


def test_duplicate_entity_names_are_invalid(ontology):
    bad = json.loads(good_payload())
    bad["entities"].append(dict(bad["entities"][0]))
    record = validated_record(json.dumps(bad), make_chunk(1), ontology)
    assert not record["accepted"]
    assert "duplicate entity name" in record["problems"][0]


def test_entity_type_outside_ontology_is_invalid(ontology):
    bad = json.loads(good_payload())
    bad["entities"][0]["entity_type"] = "Planet"
    record = validated_record(json.dumps(bad), make_chunk(1), ontology)
    assert not record["accepted"]


def test_extra_keys_are_rejected_not_silently_kept(ontology):
    """A relationship carrying year or amount is not in the ontology: reject, never keep quietly."""
    bad = json.loads(good_payload())
    bad["relationships"][0]["year"] = 2017
    bad["relationships"][0]["amount"] = "13.7 billion USD"
    record = validated_record(json.dumps(bad), make_chunk(1), ontology)
    assert not record["accepted"]
    assert record["schema_pass"] is False  # extra='forbid' on the strict contract


def test_missing_relationship_key_is_invalid(ontology):
    bad = json.loads(good_payload())
    del bad["relationships"][0]["evidence"]
    record = validated_record(json.dumps(bad), make_chunk(1), ontology)
    assert not record["accepted"]
    assert record["schema_pass"] is False


# ---------- errors and waits ----------

def pipe_for_retry(tmp_path, ontology, values):
    chunk = demo_chunks(1)[0]
    primary = FakeProvider(values)
    job = tmp_path / "job"
    prepare(job, [chunk], ontology, primary.recipe, policy=retry_policy(2))
    return Pipeline(job, ontology, primary, sleep=lambda seconds: None), chunk


def test_which_statuses_are_retried(tmp_path, ontology):
    pipe, chunk = pipe_for_retry(tmp_path, ontology, [])
    for status in (429, 500, 502, 503, 504):
        result = {"outcome": "http_failure", "http_status": status, "raw_text": None}
        assert pipe.retryable(result, chunk) is True, status
    for status in (400, 401, 402, 403, 404, 408):
        result = {"outcome": "http_failure", "http_status": status, "raw_text": None}
        assert pipe.retryable(result, chunk) is False, status
    # A provider outage whose outcome is unknown is never resent blindly.
    assert pipe.retryable({"outcome": "transport_unknown", "raw_text": None}, chunk) is False
    # A retryable batch item error (gRPC 8/13/14) is retried in a later wave.
    assert pipe.retryable({"outcome": "batch_error_retryable", "raw_text": None}, chunk) is True
    # A returned but invalid answer earns a bounded same-prompt retry.
    assert pipe.retryable({"outcome": "returned", "raw_text": "not json"}, chunk) is True
    # An accepted answer is never resent.
    assert pipe.retryable({"outcome": "returned", "raw_text": payload_json(chunk)}, chunk) is False


def test_backoff_doubles_and_is_capped(tmp_path, ontology):
    chunk = demo_chunks(1)[0]
    waits = []

    def always_503(prompt, schema):
        raise CallFailure("http_failure", 503)

    class FailingProvider(FakeProvider):
        def call(self, prompt, schema):
            self.calls += 1
            raise CallFailure("http_failure", 503)

    primary = FailingProvider([])
    job = tmp_path / "job"
    prepare(job, [chunk], ontology, primary.recipe, policy=retry_policy(3, base_seconds=1, max_wait_seconds=60))
    pipe = Pipeline(job, ontology, primary, sleep=waits.append)
    pipe.run(max_calls=10, requests_per_minute=100000, live_authorized=True)
    assert waits[1:] == [1, 2, 4]  # after the initial pacing sleep, backoff doubles
    assert primary.calls == 4  # first try plus three bounded retries

    capped = []
    primary2 = FailingProvider([])
    job2 = tmp_path / "job2"
    prepare(job2, [chunk], ontology, primary2.recipe, policy=retry_policy(3, base_seconds=30, max_wait_seconds=60))
    Pipeline(job2, ontology, primary2, sleep=capped.append).run(max_calls=10, requests_per_minute=100000, live_authorized=True)
    assert capped[1:] == [30, 60, 60]  # capped at max_wait_seconds=60


def test_retry_policy_bounds_are_validated():
    assert retry_policy(2) == {"retries": 2, "base_seconds": 2, "max_wait_seconds": 120}
    for bad in (-1, 5, True, "2"):
        with pytest.raises(ValueError):
            retry_policy(bad)
    with pytest.raises(ValueError):
        retry_policy(1, base_seconds=0)
    with pytest.raises(ValueError):
        retry_policy(1, base_seconds=10, max_wait_seconds=5)
