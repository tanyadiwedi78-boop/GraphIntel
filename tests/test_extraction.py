"""Offline tests for the resumable extraction job. No network, no API key.
"""

import json
import os

import pytest

from app.extraction.client import ChunkFailedError, ProviderError, backoff_seconds, classify_status, RETRY_LATER, STOP_RUN
from app.extraction.runner import (PROMPT_VERSION, SCHEMA_VERSION, RunLimits, already_done,
                                   load_result, make_work_key, result_path, run_extraction, save_result)
from app.extraction.validate import validate_extraction
from app.graph.ontology import load_ontology
from tests.fake_model import FakeModel, good_answer

QUOTE = "Acme develops the Widget Engine"


@pytest.fixture
def ontology():
    return load_ontology("configs/ontology.json")


def make_chunk(number):
    return {
        "chunk_id": f"chunk-{number}",
        "text": f"Item {number}. {QUOTE} for\n customers.",
        "accession": "0000000000-26-000001",
        "form": "10-K",
        "period": "2026-01-31",
    }


def no_sleep_recorder():
    waits = []

    def sleep(seconds):
        waits.append(seconds)

    return sleep, waits


def quick_limits(**changes):
    values = dict(max_calls=100, max_attempts_per_chunk=3, base_wait_seconds=1.0,
                  max_wait_seconds=60.0, jitter_fraction=0.0, stop_after_failed_chunks=5)
    values.update(changes)
    return RunLimits(**values)


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


# ---------- the job ----------

def test_three_attempts_means_two_waits_then_chunk_is_failed(ontology, tmp_path):
    model = FakeModel([ProviderError(429), ProviderError(429), ProviderError(429)])
    sleep, waits = no_sleep_recorder()
    summary = run_extraction([make_chunk(1)], model, ontology, "m", str(tmp_path), quick_limits(), sleep)
    assert len(model.calls) == 3
    assert waits == [1.0, 2.0]
    assert summary["failed"] == 1
    assert summary["valid"] == 0


def test_retry_after_is_honored_when_longer_than_backoff(ontology, tmp_path):
    model = FakeModel([ProviderError(429, retry_after_seconds=20), good_answer(QUOTE)])
    sleep, waits = no_sleep_recorder()
    summary = run_extraction([make_chunk(1)], model, ontology, "m", str(tmp_path), quick_limits(), sleep)
    assert waits == [20]
    assert summary["valid"] == 1


def test_retry_after_longer_than_limit_stops_the_run(ontology, tmp_path):
    model = FakeModel([ProviderError(429, retry_after_seconds=600)])
    sleep, waits = no_sleep_recorder()
    summary = run_extraction([make_chunk(1), make_chunk(2)], model, ontology, "m", str(tmp_path), quick_limits(), sleep)
    assert summary["stopped_early"]
    assert "longer than the limit" in summary["stop_reason"]
    assert waits == []
    assert len(model.calls) == 1


def test_setup_error_is_not_retried_and_stops(ontology, tmp_path):
    model = FakeModel([ProviderError(403, "bad key")])
    sleep, waits = no_sleep_recorder()
    summary = run_extraction([make_chunk(1), make_chunk(2)], model, ontology, "m", str(tmp_path), quick_limits(), sleep)
    assert summary["stopped_early"]
    assert len(model.calls) == 1
    assert waits == []


def test_invalid_answer_is_retried_at_once_without_waiting(ontology, tmp_path):
    model = FakeModel(["not json", good_answer(QUOTE)])
    sleep, waits = no_sleep_recorder()
    summary = run_extraction([make_chunk(1)], model, ontology, "m", str(tmp_path), quick_limits(), sleep)
    assert summary["valid"] == 1
    assert waits == []
    assert len(model.calls) == 2


def test_retry_prompt_carries_the_validation_problems(ontology, tmp_path):
    bad = json.dumps({"entities": [{"name": "Acme Corp", "type": "Company"}],
                      "relationships": [{"relation": "OWNS", "source": "Acme Corp",
                                          "target": "Acme Corp",
                                          "evidence_quote": QUOTE, "confidence": 0.9}]})
    model = FakeModel([bad, good_answer(QUOTE)])
    sleep, _ = no_sleep_recorder()
    summary = run_extraction([make_chunk(1)], model, ontology, "m", str(tmp_path), quick_limits(), sleep)
    assert summary["valid"] == 1
    assert len(model.calls) == 2
    first_prompt, retry_prompt = model.calls
    assert "previous answer was rejected" in retry_prompt
    assert "OWNS" in retry_prompt   # the validator's problem text reaches the model
    assert "previous answer was rejected" not in first_prompt
    assert retry_prompt.startswith(first_prompt)   # same task and text above the feedback


def test_first_try_prompt_is_unchanged_by_feedback_logic(ontology, tmp_path):
    from app.extraction.runner import build_prompt
    model = FakeModel([good_answer(QUOTE)])
    sleep, _ = no_sleep_recorder()
    summary = run_extraction([make_chunk(1)], model, ontology, "m", str(tmp_path), quick_limits(), sleep)
    assert summary["valid"] == 1
    assert len(model.calls) == 1
    chunk = make_chunk(1)
    assert model.calls[0] == build_prompt(chunk, ontology)  # exact base prompt


def test_invalid_after_all_attempts_is_saved_with_raw_response(ontology, tmp_path):
    model = FakeModel(["bad1", "bad2", "bad3"])
    sleep, _ = no_sleep_recorder()
    summary = run_extraction([make_chunk(1)], model, ontology, "m", str(tmp_path), quick_limits(), sleep)
    assert summary["invalid"] == 1
    files = os.listdir(tmp_path)
    saved = json.load(open(os.path.join(tmp_path, files[0])))
    assert saved["status"] == "invalid"
    assert saved["raw_response"] == "bad3"


def test_call_cap_stops_run_then_second_run_finishes_without_repeats(ontology, tmp_path):
    chunks = [make_chunk(1), make_chunk(2), make_chunk(3), make_chunk(4)]
    sleep, _ = no_sleep_recorder()

    first_model = FakeModel([good_answer(QUOTE), good_answer(QUOTE)])
    first = run_extraction(chunks, first_model, ontology, "m", str(tmp_path), quick_limits(max_calls=2), sleep)
    assert first["valid"] == 2
    assert first["stopped_early"] and first["stop_reason"] == "call cap reached"

    second_model = FakeModel([good_answer(QUOTE), good_answer(QUOTE)])
    second = run_extraction(chunks, second_model, ontology, "m", str(tmp_path), quick_limits(), sleep)
    assert second["skipped_done"] == 2
    assert second["valid"] == 2
    assert len(second_model.calls) == 2          # only the two unfinished chunks
    assert not second["stopped_early"]


def test_finished_run_repeated_makes_no_calls(ontology, tmp_path):
    chunks = [make_chunk(1), make_chunk(2)]
    sleep, _ = no_sleep_recorder()
    run_extraction(chunks, FakeModel([good_answer(QUOTE)] * 2), ontology, "m", str(tmp_path), quick_limits(), sleep)
    again = FakeModel([])
    summary = run_extraction(chunks, again, ontology, "m", str(tmp_path), quick_limits(), sleep)
    assert again.calls == []
    assert summary["skipped_done"] == 2


def test_done_chunk_is_not_reprocessed_on_another_provider(ontology, tmp_path):
    """A chunk any provider extracted validly under the same recipe is done:
    rerunning with a different model makes zero calls and says who did the work."""
    sleep, _ = no_sleep_recorder()
    first = FakeModel([good_answer(QUOTE)])
    run_extraction([make_chunk(1)], first, ontology, "model-a", str(tmp_path), quick_limits(), sleep)
    second = FakeModel([])
    events = []
    summary = run_extraction([make_chunk(1)], second, ontology, "model-b", str(tmp_path),
                             quick_limits(), sleep, log_event=events.append, approved_models=["model-a", "model-b"])
    assert summary["skipped_done"] == 1
    assert summary["calls_used"] == 0
    assert second.calls == []
    assert events[0]["status"] == "skipped"
    assert events[0]["accepted_from_model"] == "model-a"
    # The original record stays on disk under its own provider's key.
    assert len(list(tmp_path.glob("chunk-1__*.json"))) == 1


def test_legacy_result_without_full_identity_is_not_accepted(ontology, tmp_path):
    """An old record with a valid status but no recomputed identity is ignored."""
    sleep, _ = no_sleep_recorder()
    legacy = {"status": "valid", "entities": [], "relationships": []}
    (tmp_path / "chunk-1__legacy.json").write_text(json.dumps(legacy), encoding="utf-8")
    model = FakeModel([good_answer(QUOTE)])
    summary = run_extraction([make_chunk(1)], model, ontology, "model-b", str(tmp_path),
                             quick_limits(), sleep)
    assert summary["calls_used"] == 1
    assert summary["valid"] == 1


def test_result_with_mismatched_identity_is_not_accepted(ontology, tmp_path):
    """A record whose work key does not recompute from THIS chunk, ontology,
    schema, prompt and its own model is ignored, never trusted."""
    sleep, _ = no_sleep_recorder()
    record = {"status": "valid", "model": "model-a", "work_key": "x" * 20,
              "ontology_fingerprint": "wrong-fingerprint",
              "schema_version": SCHEMA_VERSION, "prompt_version": PROMPT_VERSION,
              "entities": [], "relationships": []}
    (tmp_path / ("chunk-1__" + "x" * 20 + ".json")).write_text(json.dumps(record), encoding="utf-8")
    model = FakeModel([good_answer(QUOTE)])
    summary = run_extraction([make_chunk(1)], model, ontology, "model-b", str(tmp_path),
                             quick_limits(), sleep)
    assert summary["calls_used"] == 1
    assert summary["valid"] == 1


def test_crash_between_answer_and_save_repeats_that_one_call(ontology, tmp_path):
    """The honest limit: at-least-once. The provider answered, we died before saving."""
    chunks = [make_chunk(1), make_chunk(2), make_chunk(3)]
    sleep, _ = no_sleep_recorder()
    saves = []

    def crashing_save(results_dir, chunk_id, work_key, record):
        if chunk_id == "chunk-2":
            raise KeyboardInterrupt("simulated crash before the file was written")
        save_result(results_dir, chunk_id, work_key, record)
        saves.append(chunk_id)

    first_model = FakeModel([good_answer(QUOTE), good_answer(QUOTE)])
    with pytest.raises(KeyboardInterrupt):
        run_extraction(chunks, first_model, ontology, "m", str(tmp_path), quick_limits(), sleep, crashing_save)
    assert len(first_model.calls) == 2           # chunk-2 WAS answered by the provider

    second_model = FakeModel([good_answer(QUOTE), good_answer(QUOTE)])
    summary = run_extraction(chunks, second_model, ontology, "m", str(tmp_path), quick_limits(), sleep)
    assert len(second_model.calls) == 2          # chunk-2 paid for twice, chunk-3 once
    assert summary["valid"] == 2 and summary["skipped_done"] == 1


def test_leftover_temp_file_is_ignored(ontology, tmp_path):
    chunk = make_chunk(1)
    key = make_work_key(chunk, ontology.fingerprint(), SCHEMA_VERSION, PROMPT_VERSION, "m")
    open(result_path(str(tmp_path), "chunk-1", key) + ".tmp", "w").write("{half a file")
    assert not already_done(str(tmp_path), "chunk-1", key)


def test_changing_any_part_of_the_recipe_changes_the_key(ontology):
    chunk = make_chunk(1)
    fp = ontology.fingerprint()
    base = make_work_key(chunk, fp, "s1", "p1", "m1")
    assert base == make_work_key(chunk, fp, "s1", "p1", "m1")
    assert base != make_work_key(chunk, fp, "s1", "p2", "m1")
    assert base != make_work_key(chunk, fp, "s2", "p1", "m1")
    assert base != make_work_key(chunk, fp, "s1", "p1", "m2")
    assert base != make_work_key(chunk, "otherfp", "s1", "p1", "m1")
    changed = dict(chunk)
    changed["text"] = chunk["text"] + " more"
    assert base != make_work_key(changed, fp, "s1", "p1", "m1")


def test_new_model_name_accepts_the_compatible_valid_record(ontology, tmp_path):
    """Done is done, whichever model did it: another model under the same
    recipe (chunk text, ontology, schema, prompt) accepts the valid record
    and makes no call. The accepting run names where the record came from,
    and the original provider's file is preserved untouched. To compare
    providers on the same chunks, run each with its own results directory."""
    chunks = [make_chunk(1)]
    sleep, _ = no_sleep_recorder()
    run_extraction(chunks, FakeModel([good_answer(QUOTE)]), ontology, "model-a", str(tmp_path), quick_limits(), sleep)
    files_before = sorted(os.listdir(str(tmp_path)))
    other = FakeModel([good_answer(QUOTE)])
    events = []
    summary = run_extraction(chunks, other, ontology, "model-b", str(tmp_path), quick_limits(), sleep,
                             log_event=events.append, approved_models=["model-a", "model-b"])
    assert len(other.calls) == 0
    assert summary["skipped_done"] == 1
    assert events[0]["status"] == "skipped"
    assert events[0]["accepted_from_model"] == "model-a"
    assert sorted(os.listdir(str(tmp_path))) == files_before  # per-provider provenance preserved


def test_stops_after_too_many_failed_chunks_in_a_row(ontology, tmp_path):
    # The in-a-row breaker is for CONTENT failures (the chunk's own problem):
    # each one fails fast and the run moves on until too many fail in a row.
    # Provider-wide transient failures stop the run immediately instead; that
    # is covered by test_transient_failures_stop_the_run_before_later_chunks.
    chunks = [make_chunk(1), make_chunk(2), make_chunk(3)]
    model = FakeModel([ChunkFailedError(400, "cut off"), ChunkFailedError(400, "cut off")])
    sleep, _ = no_sleep_recorder()
    summary = run_extraction(
        chunks, model, ontology, "m", str(tmp_path),
        quick_limits(max_attempts_per_chunk=2, stop_after_failed_chunks=2), sleep,
    )
    assert summary["failed"] == 2
    assert summary["stopped_early"]
    assert len(model.calls) == 2


def test_transient_failures_park_chunks_and_the_run_continues(ontology, tmp_path):
    # A provider-wide failure (persistent 503) parks the chunk - its record
    # stays pending and never counts toward give-up - and the run CONTINUES
    # with the later chunks. Rerunning the same command replays the parked
    # chunks. This replaced the old stop-the-run behavior: parked work is
    # recoverable, and the call cap still bounds the total.
    chunks = [make_chunk(1), make_chunk(2), make_chunk(3)]
    model = FakeModel([ProviderError(503)] * 6)
    sleep, _ = no_sleep_recorder()
    summary = run_extraction(
        chunks, model, ontology, "m", str(tmp_path),
        quick_limits(max_attempts_per_chunk=2, stop_after_failed_chunks=2), sleep,
    )
    assert summary["failed"] == 3
    assert summary["stopped_early"] is False
    assert summary["parked_transient"] == 3
    assert summary["circuit_trips"] == 1
    assert len(model.calls) == 6


def test_max_retries_two_means_three_attempts_per_chunk(ontology, tmp_path):
    # The config's max_retries: 2 is retries AFTER the first try: exactly 3
    # attempts per chunk. The record keeps both numbers distinct.
    chunks = [make_chunk(1)]
    model = FakeModel([ProviderError(503)] * 3)
    sleep, _ = no_sleep_recorder()
    events = []
    summary = run_extraction(chunks, model, ontology, "m", str(tmp_path),
                             quick_limits(max_attempts_per_chunk=3), sleep,
                             log_event=events.append)
    assert len(model.calls) == 3  # 1 first try + 2 retries
    assert summary["failed"] == 1
    final = events[-1]
    assert final["event"] == "chunk_final"
    assert final["attempts"] == 3
    assert final["retries"] == 2


def test_single_provider_cools_down_after_five_failed_attempts(ontology, tmp_path):
    # After 5 failed provider calls in a row, the circuit
    # opens: no calls for the 120 s cooldown, then the run resumes. Nothing is
    # lost: every parked chunk is pending for the next run.
    chunks = [make_chunk(number) for number in range(1, 7)]
    model = FakeModel([ProviderError(503)] * 12)
    sleep, waits = no_sleep_recorder()
    events = []
    summary = run_extraction(chunks, model, ontology, "m", str(tmp_path),
                             quick_limits(max_attempts_per_chunk=2), sleep,
                             log_event=events.append)
    assert summary["parked_transient"] == 6
    assert summary["circuit_trips"] == 2
    assert summary["stopped_early"] is False
    assert len(model.calls) == 12
    assert waits.count(120.0) == 2  # the cooldown; the other waits are retry backoff
    assert len(waits) == 8
    circuit_events = []
    for event in events:
        if event["event"] == "circuit_open":
            circuit_events.append(event)
    assert len(circuit_events) == 2
    assert circuit_events[0]["consecutive_failures"] == 5
    assert circuit_events[0]["cooldown_seconds"] == 120.0
    # Replay recovers the parked chunks.
    healthy = FakeModel([good_answer(QUOTE)] * 6)
    replay = run_extraction(chunks, healthy, ontology, "m", str(tmp_path),
                            quick_limits(), sleep)
    assert replay["valid"] == 6
    assert replay["parked_transient"] == 0


def test_failed_and_invalid_results_are_tried_again_next_run(ontology, tmp_path):
    chunks = [make_chunk(1), make_chunk(2)]
    sleep, _ = no_sleep_recorder()
    first = FakeModel([ChunkFailedError(400, "cut off"), "bad", "bad", "bad"])
    summary = run_extraction(chunks, first, ontology, "m", str(tmp_path), quick_limits(), sleep)
    assert summary["failed"] == 1 and summary["invalid"] == 1

    second = FakeModel([good_answer(QUOTE), good_answer(QUOTE)])
    again = run_extraction(chunks, second, ontology, "m", str(tmp_path), quick_limits(), sleep)
    assert again["valid"] == 2
    assert again["skipped_done"] == 0
    assert len(second.calls) == 2


def test_old_schema_results_are_not_reused_and_stay_on_disk(ontology, tmp_path):
    chunk = make_chunk(1)
    old_key = make_work_key(chunk, ontology.fingerprint(), "s1", "p1", "m")
    save_result(str(tmp_path), "chunk-1", old_key, {"status": "valid", "edges": [{"relation": "DEVELOPS"}]})
    old_file = result_path(str(tmp_path), "chunk-1", old_key)

    sleep, _ = no_sleep_recorder()
    model = FakeModel([good_answer(QUOTE)])
    summary = run_extraction([chunk], model, ontology, "m", str(tmp_path), quick_limits(), sleep)
    assert len(model.calls) == 1          # re-extracted under the new schema
    assert summary["skipped_done"] == 0
    assert summary["valid"] == 1
    assert os.path.exists(old_file)       # the old file is left alone


def test_saved_record_and_every_edge_have_provenance(ontology, tmp_path):
    sleep, _ = no_sleep_recorder()
    run_extraction([make_chunk(1)], FakeModel([good_answer(QUOTE)]), ontology, "m", str(tmp_path), quick_limits(), sleep)
    files = os.listdir(tmp_path)
    assert len(files) == 1
    raw = open(os.path.join(tmp_path, files[0])).read()
    assert chr(10) in raw                 # indented, not one long line
    saved = json.loads(raw)
    assert saved["status"] == "valid"
    assert saved["schema_version"] == SCHEMA_VERSION
    # Source metadata stays on the record; provider/model/prompt are also on each edge.
    assert saved["filing_accession"] == "0000000000-26-000001"
    assert saved["form_type"] == "10-K"
    assert saved["period"] == "2026-01-31"
    assert saved["entities"] == [{"name": "Acme Corp", "type": "Company"},
                                 {"name": "Widget Engine", "type": "Technology"}]
    assert saved["relationships"] == [{
        "relation": "DEVELOPS",
        "source": "Acme Corp",
        "target": "Widget Engine",
        "evidence_quote": QUOTE,
        "confidence": 0.9,
        "provider": "m",
        "model": "m",
        "prompt_version": PROMPT_VERSION,
    }]
    assert "edges" not in saved


def test_unconfigured_model_cannot_supply_accepted_skip(ontology, tmp_path):
    sleep, _ = no_sleep_recorder()
    run_extraction([make_chunk(1)], FakeModel([good_answer(QUOTE)]), ontology,
                   "unapproved:old:r", str(tmp_path), quick_limits(), sleep)
    model = FakeModel([good_answer(QUOTE)])
    summary = run_extraction([make_chunk(1)], model, ontology, "approved:new:r",
                             str(tmp_path), quick_limits(), sleep)
    assert summary["skipped_done"] == 0
    assert summary["valid"] == 1
    assert len(model.calls) == 1
    assert len(list(tmp_path.glob("*.json"))) == 2  # originals preserved


def test_changed_recipe_tag_cannot_supply_accepted_skip(ontology, tmp_path):
    sleep, _ = no_sleep_recorder()
    run_extraction([make_chunk(1)], FakeModel([good_answer(QUOTE)]), ontology,
                   "gemini:model:old-recipe", str(tmp_path), quick_limits(), sleep)
    model = FakeModel([good_answer(QUOTE)])
    summary = run_extraction([make_chunk(1)], model, ontology, "gemini:model:new-recipe",
                             str(tmp_path), quick_limits(), sleep,
                             approved_models=["gemini:model:new-recipe", "openai:other:recipe"])
    assert summary["skipped_done"] == 0
    assert len(model.calls) == 1
