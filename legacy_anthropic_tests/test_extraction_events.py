"""Durable asynchronous audit records replace synchronous callback events."""
import json
import pytest
from app.extraction.anthropic_batch import collect_job, submit_job, digest, write_new
from tests.test_anthropic_batch import job, ontology, FakeAPI, receipt, results


def test_claim_response_and_receipt_bind_exact_batch_before_collection(job):
    folder, plan = job
    reply = {"id":"msgbatch_test", "created_at":"2026-11-10T05:00:00Z"}
    api = FakeAPI([reply])
    submit_job(folder, plan["plan_id"], "100", True, api, max_calls=2)
    claim = json.loads((folder / "submission-claim.json").read_text())
    assert claim["plan_id"] == plan["plan_id"]
    assert claim["budget_usd"] == "100"
    assert json.loads((folder / "submission-response.txt").read_text()) == reply
    assert json.loads((folder / "receipt.json").read_text()) == reply
    assert len(api.calls) == 1


def test_pending_snapshots_are_immutable_and_do_not_submit(job):
    folder, plan = job
    receipt(folder)
    one = {"id":"msgbatch_test", "processing_status":"in_progress", "request_counts":{"processing":2}}
    two = {"id":"msgbatch_test", "processing_status":"in_progress", "request_counts":{"processing":1}}
    api = FakeAPI([one, two, two])
    for unused in range(3):
        assert collect_job(folder, api)["processing_status"] == "in_progress"
    snapshots = list(folder.glob("status-*.json"))
    assert len(snapshots) == 2
    assert {json.loads(p.read_text())["request_counts"]["processing"] for p in snapshots} == {1,2}
    assert all(call[0] == "GET" for call in api.calls)
    assert not (folder / "summary.json").exists()


def test_zero_cap_stops_before_claim_or_network(job):
    folder, plan = job
    api = FakeAPI([])
    with pytest.raises(ValueError, match="0 permits none"):
        submit_job(folder, plan["plan_id"], "100", True, api, max_calls=0)
    assert api.calls == []
    assert not (folder / "submission-claim.json").exists()


def test_completion_retains_one_final_record_for_each_source(job):
    folder, plan = job
    receipt(folder)
    api = FakeAPI([{"id":"msgbatch_test", "processing_status":"ended"},
                   "\n".join(json.dumps(row) for row in results(plan))])
    summary = collect_job(folder, api)
    assert summary["counts"] == {"valid":2}
    saved = [json.loads(p.read_text()) for p in (folder / "extractions").glob("*.json")]
    assert {row["chunk_id"] for row in saved} == {"c1", "c2"}
    assert len(saved) == 2
    assert all(row["batch_id"] == "msgbatch_test" for row in saved)


def test_resume_does_not_create_new_audit_or_network_records(job):
    folder, plan = job
    receipt(folder)
    api = FakeAPI([{"id":"msgbatch_test", "processing_status":"ended"},
                   "\n".join(json.dumps(row) for row in results(plan))])
    collect_job(folder, api)
    before = {str(p):p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    again = FakeAPI([])
    collect_job(folder, again)
    assert again.calls == []
    assert before == {str(p):p.read_bytes() for p in folder.rglob("*") if p.is_file()}
