"""Offline provider fixtures only. Never use credentials or live APIs."""
import copy
import json
from decimal import Decimal
from pathlib import Path

import pytest

from app.extraction.anthropic_batch import (BATCH_PATH, MODEL, canonical, collect_job,
    digest, import_results, message_params, money, prepare_plan, submit_job,
    usage_cost, verify_plan, write_new)
from app.extraction.runner import build_prompt
from app.graph.ontology import load_ontology

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def ontology():
    return load_ontology(ROOT / "configs" / "ontology.json")


@pytest.fixture
def job(tmp_path, ontology):
    chunks = [{"chunk_id": "c1", "text": "Acme develops Widget Engine.",
               "accession": "a1", "form": "10-K", "period": "2026"},
              {"chunk_id": "c2", "text": "A cover page.", "accession": "a1",
               "form": "10-K", "period": "2026"}]
    source = tmp_path / "chunks.jsonl"
    source.write_text("\n".join(json.dumps(chunk) for chunk in chunks), encoding="utf-8")
    plan = prepare_plan(source, ontology, 2)
    folder = tmp_path / "job"
    folder.mkdir()
    write_new(folder / "plan.json", plan)
    return folder, plan


class FakeAPI:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def request(self, method, path, body=None):
        self.calls.append((method, path, body))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)


def usage():
    return {"input_tokens": 100, "output_tokens": 200,
            "cache_creation_input_tokens": 1000, "cache_read_input_tokens": 500,
            "cache_creation": {"ephemeral_5m_input_tokens": 400,
                               "ephemeral_1h_input_tokens": 600}}


def results(plan, text=None, stop="end_turn", used_model=MODEL):
    if text is None:
        text = '{"entities":[],"relationships":[]}'
    rows = []
    for custom_id in plan["chunks"]:
        rows.append({"custom_id": custom_id, "result": {"type": "succeeded", "message": {
            "id": "msg_test", "model": used_model, "stop_reason": stop,
            "content": [{"type": "thinking", "thinking": "private reasoning"},
                        {"type": "text", "text": text}], "usage": usage()}}})
    return rows


def receipt(folder):
    write_new(folder / "receipt.json", {"id": "msgbatch_test"})


def test_prompt_exact_words_and_prefix_stable(job, ontology):
    folder, plan = job
    params = [request["params"] for request in plan["requests"]]
    assert params[0]["messages"][0]["content"][0] == params[1]["messages"][0]["content"][0]
    for request in plan["requests"]:
        blocks = request["params"]["messages"][0]["content"]
        assert "".join(block["text"] for block in blocks) == build_prompt(plan["chunks"][request["custom_id"]], ontology)
        assert blocks[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
        assert "cache_control" not in blocks[1]
        assert set(request["params"]) == {"model", "max_tokens", "messages", "output_config", "inference_geo"}
        assert request["params"]["model"] == MODEL
    verify_plan(plan)


@pytest.mark.parametrize("change", ["prompt", "price", "model", "tokens", "digest"])
def test_tampered_plans_rejected(job, change):
    folder, plan = job
    if change == "prompt":
        plan["requests"][0]["params"]["messages"][0]["content"][0]["text"] += "different"
    elif change == "price":
        plan["strict_reservation_usd"] = "0"
    elif change == "model":
        plan["model"] = "other-model"
    elif change == "tokens":
        plan["max_tokens"] = 0
    else:
        plan["plan_id"] = "wrong"
    with pytest.raises(ValueError):
        verify_plan(plan)


@pytest.mark.parametrize("budget,allowed,approved", [("100",False,True),("100",True,False),("0.01",True,True),
                                                   ("NaN",True,True),("-1",True,True),("Infinity",True,True)])
def test_submission_gates_before_network(job, budget, allowed, approved):
    folder, plan = job
    api = FakeAPI([])
    with pytest.raises(ValueError):
        submit_job(folder, plan["plan_id"] if approved else "wrong", budget, allowed, api, max_calls=2)
    assert api.calls == []
    assert not (folder / "submission-claim.json").exists()


def test_exact_conservative_reservation(job):
    folder, plan = job
    assert Decimal(plan["strict_reservation_usd"]) == 2 * (Decimal("16") + 4096 * Decimal("0.000020"))


def test_single_submission_and_no_duplicate(job):
    folder, plan = job
    api = FakeAPI([{"id":"msgbatch_test"}])
    got = submit_job(folder, plan["plan_id"], "40", True, api, max_calls=2)
    assert got["id"] == "msgbatch_test"
    assert api.calls[0][0:2] == ("POST", BATCH_PATH)
    assert api.calls[0][2] == {"requests": plan["requests"]}
    with pytest.raises(FileExistsError):
        submit_job(folder, plan["plan_id"], "40", True, api, max_calls=2)
    assert len(api.calls) == 1


def test_unknown_submit_holds_claim_and_never_retries(job):
    folder, plan = job
    api = FakeAPI([TimeoutError()])
    with pytest.raises(TimeoutError):
        submit_job(folder, plan["plan_id"], "40", True, api, max_calls=2)
    assert (folder / "submission-claim.json").exists()
    with pytest.raises(FileExistsError):
        submit_job(folder, plan["plan_id"], "40", True, api, max_calls=2)
    assert len(api.calls) == 1


def test_cost_stack_exact_and_thinking_not_double_counted():
    assert Decimal(usage_cost(usage())) == Decimal("0.01125")
    u = usage()
    u["output_tokens"] = 1200
    assert Decimal(usage_cost(u)) == Decimal("0.03125")


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"])
def test_missing_usage_unknown(field):
    u = usage()
    del u[field]
    assert usage_cost(u) is None


def test_missing_write_ttl_unknown():
    u = usage()
    del u["cache_creation"]
    assert usage_cost(u) is None


@pytest.mark.parametrize("kind", ["errored", "canceled", "expired"])
def test_terminal_provider_errors_preserved_not_retried(job, kind):
    folder, plan = job
    receipt(folder)
    rows = [{"custom_id": key, "result": {"type": kind, "error": {"type":"example"}}} for key in plan["chunks"]]
    got = import_results(folder, rows)
    assert got["counts"] == {kind: 2}
    assert got["calculated_cost_usd"] == "0"
    assert not (folder / "extractions").exists()


def test_shared_validator_and_source_provenance(job):
    folder, plan = job
    receipt(folder)
    text = json.dumps({"entities":[{"name":"Acme","type":"Company"},{"name":"Widget Engine","type":"Product"}],
        "relationships":[{"source":"Acme","target":"Widget Engine","relation":"OFFERS",
                          "evidence_quote":"Acme develops Widget Engine.","confidence":0.9}]})
    # Find an allowed Company -> Product relation from actual ontology.
    ontology = verify_plan(plan)
    relation = next(key for key, value in ontology.relations.items() if value == ("Company","Product"))
    data = json.loads(text)
    data["relationships"][0]["relation"] = relation
    rows = results(plan)
    rows[0]["result"]["message"]["content"] = [{"type":"text","text":json.dumps(data)}]
    summary = import_results(folder, list(reversed(rows)))
    assert summary["counts"] == {"valid": 2}
    records = [json.loads(p.read_text()) for p in (folder / "extractions").glob("*.json")]
    record = next(record for record in records if record["chunk_id"] == "c1")
    edge = record["relationships"][0]
    assert edge["source_chunk_id"] == "c1"
    assert edge["filing_accession"] == "a1"
    assert edge["provider"] == "anthropic"
    assert record["entities"][0]["canonical_name"] == "Acme"
    assert record["cost_usd"] is None
    assert record["calculated_cost_usd"] == usage_cost(usage())


@pytest.mark.parametrize("text,stop,model", [('{"entities":[],"relationships":[]}',"max_tokens",MODEL),
    ('{"entities":[],"relationships":[]}',"refusal",MODEL),
    ('{"entities":[],"relationships":[]}',"end_turn","wrong"),
    ('not json',"end_turn",MODEL),
    ('{"entities":[],"relationships":[],"extra":1}',"end_turn",MODEL)])
def test_invalid_refusal_truncation_never_valid(job, text, stop, model):
    folder, plan = job
    receipt(folder)
    got = import_results(folder, results(plan,text,stop,model))
    assert got["counts"] == {"invalid":2}
    assert Decimal(got["calculated_cost_usd"]) > 0
    assert len(list((folder / "extractions").glob("*.json"))) == 2


@pytest.mark.parametrize("kind", ["missing","unknown","duplicate"])
def test_bad_results_no_completion_claim(job, kind):
    folder, plan = job
    receipt(folder)
    rows = results(plan)
    if kind == "missing": rows.pop()
    if kind == "unknown": rows[0]["custom_id"] = "unknown"
    if kind == "duplicate": rows[1] = rows[0]
    with pytest.raises(ValueError): import_results(folder, rows)
    assert not (folder / "summary.json").exists()


def test_unknown_cost_propagates_to_document_and_total(job):
    folder, plan = job
    receipt(folder)
    rows = results(plan)
    del rows[0]["result"]["message"]["usage"]["input_tokens"]
    got = import_results(folder, rows)
    assert got["calculated_cost_usd"] is None
    assert got["unknown_cost_count"] == 1
    assert got["documents"]["a1"]["calculated_cost_usd"] is None


def test_collect_pending_reads_once_no_submit(job):
    folder, plan = job
    receipt(folder)
    api = FakeAPI([{"id":"msgbatch_test","processing_status":"in_progress"}])
    got = collect_job(folder, api)
    assert got["processing_status"] == "in_progress"
    assert len(api.calls) == 1
    assert api.calls[0][0] == "GET"


def test_collect_saves_raw_then_summary_and_is_idempotent(job):
    folder, plan = job
    receipt(folder)
    raw = "\n".join(json.dumps(row) for row in results(plan))
    api = FakeAPI([{"id":"msgbatch_test","processing_status":"ended"},raw])
    got = collect_job(folder, api)
    assert got["counts"] == {"valid":2}
    assert (folder / "provider-results.jsonl").read_text() == raw
    assert collect_job(folder, FakeAPI([])) == got
    assert len(api.calls) == 2


def test_strict_wire_schema_and_local_ranges(job):
    folder, plan = job
    schema = plan["requests"][0]["params"]["output_config"]["format"]["schema"]
    assert schema["additionalProperties"] is False
    for key in ("entities", "relationships"):
        assert schema["properties"][key]["items"]["additionalProperties"] is False
    confidence = schema["properties"]["relationships"]["items"]["properties"]["confidence"]
    assert "minimum" not in confidence and "maximum" not in confidence
    assert "checked locally" in confidence["description"]


def alias_text():
    return ('The condensed consolidated financial statements include the accounts of Apple Inc. '
            'and its wholly owned subsidiaries (collectively “Apple” or the “Company”). '
            'Apple offers iPhone. Apple Inc. offers iPhone.')


def alias_answer():
    return {"entities": [{"name":"Apple Inc.","type":"Company"},
                         {"name":"Apple","type":"Company"},
                         {"name":"Company","type":"Subsidiary"},
                         {"name":"iPhone","type":"Product"}],
            "relationships": [{"source":"Apple Inc.","target":"Company","relation":"OWNS",
                               "evidence_quote": alias_text().split(').')[0] + ').', "confidence":0.9}]}


def test_apple_explicit_collective_alias_not_fake_subsidiary(ontology):
    from app.extraction.anthropic_batch import normalize_response, source_aliases
    aliases, evidence = source_aliases(alias_text())
    assert aliases["Apple"] == "Apple Inc."
    assert aliases["Company"] == "Apple Inc."
    assert evidence[0]["source_quote"] in alias_text()
    raw = json.dumps(alias_answer())
    text, audit = normalize_response(raw, {"text":alias_text()}, ontology)
    data = json.loads(text)
    assert data["entities"] == [{"name":"Apple Inc.","type":"Company"},{"name":"iPhone","type":"Product"}]
    assert data["relationships"] == []
    assert any(event["reason"] == "alias_self_ownership_removed" for event in audit)
    assert json.loads(raw) == alias_answer()  # raw stays untouched


def test_second_actual_apple_definition_form(ontology):
    from app.extraction.anthropic_batch import source_aliases
    text = ('Each of the terms the “Company” and “Apple” as used herein refers collectively '
            'to Apple Inc. and its wholly owned subsidiaries, unless otherwise stated.')
    aliases, audit = source_aliases(text)
    assert aliases["Apple"] == "Apple Inc."
    assert aliases["Company"] == "Apple Inc."


def test_no_alias_merge_without_source_definition(ontology):
    from app.extraction.anthropic_batch import normalize_response
    raw = json.dumps({"entities":[{"name":"Apple Inc.","type":"Company"},
                                 {"name":"Apple Records","type":"Company"},
                                 {"name":"Apple","type":"Company"}],"relationships":[]})
    normalized, audit = normalize_response(raw,{"text":"Apple Inc., Apple and Apple Records are named."},ontology)
    assert len(json.loads(normalized)["entities"]) == 3


def test_whitespace_entity_and_relation_dedup_is_conservative(ontology):
    from app.extraction.anthropic_batch import normalize_response
    relation = next(key for key,value in ontology.relations.items() if value == ("Company","Product"))
    raw = {"entities":[{"name":"Acme  Corp","type":"Company"},
                       {"name":"Acme Corp","type":"Company"},
                       {"name":"Widget","type":"Product"}],
           "relationships":[{"source":"Acme  Corp","target":"Widget","relation":relation,
                             "evidence_quote":"Acme Corp offers Widget.","confidence":0.9},
                            {"source":"Acme Corp","target":"Widget","relation":relation,
                             "evidence_quote":"Acme Corp offers Widget.","confidence":0.7}]}
    normalized, audit = normalize_response(json.dumps(raw),{"text":"Acme Corp offers Widget."},ontology)
    data = json.loads(normalized)
    assert len(data["entities"]) == 2
    assert len(data["relationships"]) == 1
    assert data["relationships"][0]["confidence"] == 0.7
    assert any(event["reason"] == "duplicate_edge" for event in audit)


def test_invalid_duplicate_cannot_hide_behind_valid_one(ontology):
    from app.extraction.anthropic_batch import normalize_response
    relation = next(key for key,value in ontology.relations.items() if value == ("Company","Product"))
    raw = {"entities":[{"name":"Acme","type":"Company"},{"name":"Widget","type":"Product"}],
           "relationships":[{"source":"Acme","target":"Widget","relation":relation,
                             "evidence_quote":"Acme offers Widget.","confidence":0.8},
                            {"source":"Acme","target":"Widget","relation":relation,
                             "evidence_quote":"invented quote", "confidence":0.8}]}
    source = json.dumps(raw)
    normalized, audit = normalize_response(source,{"text":"Acme offers Widget."},ontology)
    assert normalized == source
    assert audit[-1]["reason"] == "invalid_duplicate_not_hidden"


def test_shape_failure_never_normalized_into_valid(ontology):
    from app.extraction.anthropic_batch import normalize_response
    raw = json.dumps({"entities":[],"relationships":[],"unexpected":True})
    assert normalize_response(raw,{"text":alias_text()},ontology) == (raw,[])


def test_type_conflicts_stay_rejected(ontology):
    from app.extraction.anthropic_batch import normalize_response
    raw = json.dumps({"entities":[{"name":"Acme","type":"Company"},
                                 {"name":"Acme","type":"Product"}],"relationships":[]})
    normalized, audit = normalize_response(raw,{"text":"Acme"},ontology)
    assert normalized == raw
    assert audit[-1]["reason"] == "conflicting_types_no_merge"


def test_call_cap_blocks_before_submission(job):
    folder, plan = job
    api = FakeAPI([])
    with pytest.raises(ValueError, match="MAX_EXTRACTION_CALLS"):
        submit_job(folder,plan["plan_id"],"100",True,api)
    assert not api.calls


def test_active_cli_does_not_accept_old_provider_or_fallback_flags(tmp_path):
    import subprocess
    import sys
    for flags in (["--provider","gemini"],["--fallback-provider","openai"],[]):
        got = subprocess.run([sys.executable,"scripts/run_extraction.py"]+flags,
                             cwd=ROOT,capture_output=True,text=True)
        assert got.returncode != 0
        assert "prepare,prepare-retry,submit,collect" in got.stderr


def test_active_offline_prepare_needs_no_key_and_changes_no_source(tmp_path):
    import os
    import subprocess
    import sys
    data = tmp_path / "data"
    (data / "chunks").mkdir(parents=True)
    source = data / "chunks" / "chunks.jsonl"
    raw = json.dumps({"chunk_id":"c","text":"Cover.","accession":"a","form":"10-Q","period":"2022"})+"\n"
    source.write_text(raw)
    env = dict(os.environ, DATA_DIR=str(data), ANTHROPIC_API_KEY="", MAX_EXTRACTION_CALLS="0")
    got = subprocess.run([sys.executable,"scripts/run_extraction.py","prepare",
                          "--job-dir",str(tmp_path / "job"),"--max-chunks","1"],
                         cwd=ROOT,capture_output=True,text=True,env=env)
    assert got.returncode == 0, got.stdout+got.stderr
    assert source.read_text() == raw
    assert "strict_reservation_usd" in got.stdout
    assert not (tmp_path / "job" / "submission-claim.json").exists()


def test_resume_cache_skips_valid_without_skipping_new_chunks(tmp_path, ontology):
    from app.extraction.anthropic_batch import compatible_cache
    chunks = [{"chunk_id":"c"+str(i),"text":"Cover.","accession":"a","form":"10-K","period":"2026"} for i in range(3)]
    source = tmp_path / "chunks.jsonl"
    source.write_text("\n".join(json.dumps(chunk) for chunk in chunks))
    cache = tmp_path / "cache"
    plan = prepare_plan(source,ontology,1,cache_dir=cache)
    job = tmp_path / "one"
    job.mkdir()
    write_new(job / "plan.json",plan)
    receipt(job)
    import_results(job,results(plan))
    second = prepare_plan(source,ontology,1,cache_dir=cache)
    assert [c["chunk_id"] for c in second["chunks"].values()] == ["c1"]
    assert second["skipped_cached_chunks"] == ["c0"]
    chunks[2]["text"] = "Changed elsewhere in the same filing."
    source.write_text("\n".join(json.dumps(chunk) for chunk in chunks))
    changed = prepare_plan(source,ontology,1,cache_dir=cache)
    assert [c["chunk_id"] for c in changed["chunks"].values()] == ["c0"]


def test_all_cached_plan_is_noop_and_cannot_submit(job, ontology, tmp_path):
    folder, plan = job
    plan["cache_dir"] = str(tmp_path / "cache")
    plan.pop("plan_id"); plan["plan_id"] = digest(plan)
    (folder / "plan.json").write_text(json.dumps(plan))
    receipt(folder)
    import_results(folder,results(plan))
    source = tmp_path / "chunks.jsonl"
    # prepare recomputes a document hash, so reuse the exact source content/order.
    chunks = list(plan["chunks"].values())
    source.write_text("\n".join(json.dumps(chunk) for chunk in chunks))
    new = prepare_plan(source,ontology,2,cache_dir=plan["cache_dir"])
    assert new["requests"] == []
    assert new["strict_reservation_usd"] == "0.00000"
    verify_plan(new)
    other = tmp_path / "other"; other.mkdir(); write_new(other / "plan.json",new)
    with pytest.raises(ValueError,match="no batch"):
        submit_job(other,new["plan_id"],"100",True,FakeAPI([]),max_calls=10)


def test_repair_plan_only_invalid_and_same_cached_prefix(job):
    from app.extraction.anthropic_batch import prepare_retry
    folder,plan=job
    receipt(folder)
    rows=results(plan)
    rows[1]["result"]["message"]["content"]=[{"type":"text","text":"bad json"}]
    raw="\n".join(json.dumps(row) for row in rows)
    (folder / "provider-results.jsonl").write_text(raw)
    summary=import_results(folder,rows)
    write_new(folder / "summary.json",summary)
    retry=prepare_retry(folder)
    verify_plan(retry)
    assert len(retry["requests"]) == 1
    assert retry["round"] == 1
    original=plan["requests"][1]["params"]["messages"][0]["content"]
    repaired=retry["requests"][0]["params"]["messages"][0]["content"]
    assert repaired[0] == original[0]
    assert "previous answer was rejected" in repaired[1]["text"]
    assert "not valid JSON" in repaired[1]["text"]


def test_truncated_answer_needs_review_not_identical_retry(job):
    from app.extraction.anthropic_batch import prepare_retry
    folder,plan=job;receipt(folder)
    rows=results(plan,stop="max_tokens")
    (folder / "provider-results.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    write_new(folder / "summary.json",import_results(folder,rows))
    with pytest.raises(ValueError,match="output cap"):
        prepare_retry(folder)


def test_setup_failure_does_not_keep_spending_retries(job):
    from app.extraction.anthropic_batch import prepare_retry
    folder,plan=job;receipt(folder)
    rows=[{"custom_id":key,"result":{"type":"errored","error":{"error":{"type":"authentication_error"}}}} for key in plan["chunks"]]
    (folder / "provider-results.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    write_new(folder / "summary.json",import_results(folder,rows))
    with pytest.raises(ValueError,match="setup error"):
        prepare_retry(folder)


def test_shared_claim_blocks_another_job_before_post(job,tmp_path):
    folder,plan=job
    plan["cache_dir"] = str(tmp_path / "cache")
    plan.pop("plan_id");plan["plan_id"]=digest(plan)
    (folder / "plan.json").write_text(json.dumps(plan))
    api=FakeAPI([{"id":"msgbatch_test"}])
    submit_job(folder,plan["plan_id"],"100",True,api,max_calls=2)
    other=tmp_path / "copy";other.mkdir();write_new(other / "plan.json",plan)
    with pytest.raises(ValueError,match="submission claim"):
        submit_job(other,plan["plan_id"],"100",True,api,max_calls=2)
    assert len(api.calls)==1


def test_existing_unknown_submission_is_not_imported_as_zero_cost(job):
    folder,plan=job
    write_new(folder / "submission-claim.json",{"state":"possibly_submitted_do_not_retry"})
    with pytest.raises(FileNotFoundError):
        collect_job(folder,FakeAPI([]))
    assert not (folder / "summary.json").exists()


def test_whole_actual_apple_chunk_normalization(ontology):
    from app.extraction.anthropic_batch import source_aliases
    # Exact first source sentence retained from the attached real filing chunk.
    text=('The condensed consolidated financial statements include the accounts of Apple Inc. '
          'and its wholly owned subsidiaries (collectively “Apple” or the “Company”). '
          'Intercompany accounts and transactions have been eliminated.')
    aliases,audit=source_aliases(text)
    assert set(aliases.values())=={"Apple Inc."}
    assert all(event["source_quote"] in text for event in audit)


def test_collection_reimports_partial_local_work_deterministically(job):
    folder,plan=job;receipt(folder)
    rows=results(plan,text="not json")
    raw="\n".join(json.dumps(row) for row in rows)
    (folder / "provider-results.jsonl").write_text(raw)
    first=import_results(folder,rows)
    second=import_results(folder,rows)
    assert first == second
    assert len(list((folder / "extractions").glob("*.json")))==2


def test_saved_raw_precedes_validation_failure(job):
    folder,plan=job;receipt(folder)
    rows=results(plan)
    rows[1]["custom_id"]="unknown"
    raw="\n".join(json.dumps(row) for row in rows)
    api=FakeAPI([{"id":"msgbatch_test","processing_status":"ended"},raw])
    with pytest.raises(ValueError,match="unknown"):
        collect_job(folder,api)
    assert (folder / "provider-results.jsonl").read_text()==raw
    assert not (folder / "summary.json").exists()


@pytest.mark.parametrize("confidence", [True,"0.8",float("nan"),float("inf"),-0.1,1.1])
def test_local_confidence_bounds_even_if_provider_shape_succeeds(ontology,confidence):
    from app.extraction.validate import validate_extraction
    relation=next(key for key,value in ontology.relations.items() if value==("Company","Product"))
    data={"entities":[{"name":"Acme","type":"Company"},{"name":"Widget","type":"Product"}],
          "relationships":[{"source":"Acme","target":"Widget","relation":relation,
                            "evidence_quote":"Acme offers Widget.","confidence":confidence}]}
    assert not validate_extraction(json.dumps(data),{"text":"Acme offers Widget."},ontology)[0]


def test_streamed_results_download_and_partial_recovery(tmp_path,monkeypatch):
    from app.extraction.anthropic_batch import BatchAPI
    monkeypatch.setenv("ANTHROPIC_API_KEY","fake-test-key")
    class Response:
        def __init__(self,fail=False): self.calls=0;self.fail=fail
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self,size):
            self.calls+=1
            if self.calls==1:return b'{"example":1}\n'
            if self.fail:raise TimeoutError()
            return b''
    class Opener:
        def __init__(self,fail=False):self.fail=fail
        def open(self,request,timeout):
            assert request.full_url.startswith("https://api.anthropic.com/")
            assert request.get_method()=="GET"
            return Response(self.fail)
    api=BatchAPI();api.opener=Opener(True)
    path=tmp_path / "results.jsonl"
    with pytest.raises(TimeoutError):api.download_results(BATCH_PATH+"/msgbatch_test/results",path)
    assert not path.exists()
    assert len(list(tmp_path.glob("*.partial-*")))==1
    api.opener=Opener(False)
    api.download_results(BATCH_PATH+"/msgbatch_test/results",path)
    assert path.read_bytes()==b'{"example":1}\n'
    assert len(list(tmp_path.glob("*.partial-*")))==1  # evidence retained, never deleted


def test_missing_key_does_not_create_a_submission_claim(job,monkeypatch):
    folder,plan=job
    monkeypatch.setenv("ANTHROPIC_API_KEY","")
    with pytest.raises(ValueError,match="ANTHROPIC_API_KEY"):
        submit_job(folder,plan["plan_id"],"100",True,max_calls=2)
    assert not (folder / "submission-claim.json").exists()


def test_support_scripts_have_no_deleted_extraction_dependencies(tmp_path):
    import os
    import subprocess
    import sys
    data=tmp_path / "data";(data / "chunks").mkdir(parents=True)
    source=data / "chunks" / "chunks.jsonl"
    source.write_text(json.dumps({"chunk_id":"c1","text":"Cover.","accession":"a","form":"10-K","period":"2026",
                                 "ticker":"ACME","item_id":"cover","is_table":False,"est_tokens":2})+"\n")
    env=dict(os.environ,DATA_DIR=str(data),MAX_EXTRACTION_CALLS="0",ANTHROPIC_API_KEY="")
    commands=[(["scripts/list_chunks.py","--sample","1"],"prepare --job-dir"),
              (["scripts/review_extraction.py","--chunk-ids","c1","--results-dir",str(tmp_path / "outputs")],"Cover."),
              (["scripts/summarize_extraction.py","--job-dir",str(tmp_path / "unknown")],"do not resubmit")]
    for command,expected in commands:
        got=subprocess.run([sys.executable]+command,cwd=ROOT,env=env,capture_output=True,text=True)
        assert expected in got.stdout,got.stdout+got.stderr
        assert "Traceback" not in got.stderr


def test_sample_score_uses_its_job_results(capsys):
    from scripts.list_chunks import print_sample
    print_sample([{"chunk_id": "example", "accession": "a"}], [0])
    output = capsys.readouterr().out
    assert "--results-dir data/anthropic-sample/extractions --labels data/labels_sample.json" in output
