"""Regression tests for truthful pilot measurement. No credentials or network."""
import copy
import json
import socket
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.extraction.anthropic_batch import (digest, import_results, prepare_plan,
    prepare_retry, planning_estimate, submit_job, verify_plan, write_new)
from app.extraction.runner import PROMPT_VERSION, SCHEMA_VERSION
from scripts import review_extraction as review
from tests.test_anthropic_batch import job, ontology, receipt, results, usage, FakeAPI


def score(tmp_path, monkeypatch, capsys, entries, saved):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(entries))
    monkeypatch.setattr(review, "find_result", lambda folder, cid: ("result.json", saved.get(cid)))
    code = review.run_score(SimpleNamespace(data_dir=tmp_path), path)
    out = capsys.readouterr().out
    coverage = next(json.loads(line[len("Coverage: "):]) for line in out.splitlines() if line.startswith("Coverage: "))
    return code, out, coverage


def label(entities=None, reviewed=True):
    return {"reviewed": reviewed, "entities": entities or [], "relationships": []}


def accepted(entities=None):
    return {"status":"valid", "prompt_version":PROMPT_VERSION, "schema_version":SCHEMA_VERSION,
            "entities":entities or [], "relationships":[]}


def test_reviewed_empty_false_positive_is_scored(tmp_path, monkeypatch, capsys):
    code, out, coverage = score(tmp_path, monkeypatch, capsys,
        {"cover":label()}, {"cover":accepted([{"name":"Imaginary", "type":"Company"}])})
    assert code == 0 and coverage["reviewed_empty"] == 1 and coverage["valid"] == 1
    assert "precision 0/1 = 0.00" in out
    assert "wrong extraction: Imaginary" in out


def test_reviewed_empty_correct_output_is_in_coverage(tmp_path, monkeypatch, capsys):
    code, out, coverage = score(tmp_path, monkeypatch, capsys,{"cover":label()},{"cover":accepted()})
    assert code == 0 and coverage["reviewed"] == coverage["valid"] == 1
    assert "n/a (none)" in out  # No invented perfect precision with zero predictions.


@pytest.mark.parametrize("entry", [label(reviewed=False), {"entities":[],"relationships":[]}])
def test_unreviewed_and_legacy_entries_are_not_assumed_reviewed(tmp_path, monkeypatch, capsys, entry):
    code, out, coverage = score(tmp_path, monkeypatch, capsys,{"cover":entry},{"cover":accepted()})
    assert code == 1 and coverage["reviewed"] == 0 and coverage["unreviewed"] == 1


@pytest.mark.parametrize("kind", ["missing", "invalid_or_failed", "incompatible"])
def test_failed_or_missing_output_counts_as_missed_recall(tmp_path, monkeypatch, capsys, kind):
    expected = [{"name":"Acme", "type":"Company"}]
    row = accepted()
    if kind == "invalid_or_failed":
        row["status"] = "invalid"
    elif kind == "incompatible":
        row["schema_version"] = "old"
    else:
        row = None
    code, out, coverage = score(tmp_path, monkeypatch, capsys,{"one":label(expected)},{"one":row})
    assert code == 0 and coverage[kind] == 1 and coverage["valid"] == 0
    assert "end-to-end recall 0/1 = 0.00" in out
    assert "no accepted output" in out


def test_mixed_coverage_does_not_hide_failed_chunks(tmp_path, monkeypatch, capsys):
    entity = [{"name":"Acme", "type":"Company"}]
    code, out, coverage = score(tmp_path, monkeypatch, capsys,
        {"good":label(entity), "missing":label(entity), "negative":label(), "todo":label(reviewed=False)},
        {"good":accepted(entity), "negative":accepted(entity)})
    assert code == 0
    assert coverage == {"reviewed":3,"unreviewed":1,"valid":2,"missing":1,"invalid_or_failed":0,"incompatible":0,"reviewed_empty":1}
    assert "precision 1/2 = 0.50, end-to-end recall 1/2 = 0.50" in out


def test_label_template_starts_unreviewed_and_preserves_completed_labels(tmp_path, capsys):
    path = tmp_path / "labels.json"
    first = {"done":label()}
    path.write_text(json.dumps(first))
    assert review.run_write_labels(SimpleNamespace(), ["done","new"], path) == 0
    saved = json.loads(path.read_text())
    assert saved["done"] == first["done"]
    assert saved["new"]["reviewed"] is False
    assert review.check_labels_file({"bad":label(reviewed="yes")})


def test_canonical_labels_do_not_use_fuzzy_alias_scoring():
    correct, wrong, missed = review.match_items([{"name":"Apple Inc.","type":"Company"}],
        [{"name":"Apple Records","type":"Company"}], review.ENTITY_FIELDS)
    assert correct == 0 and len(wrong) == len(missed) == 1


def test_known_usage_survives_missing_cost_ttl(job):
    folder, plan = job; receipt(folder)
    rows = results(plan)
    for row in rows:
        del row["result"]["message"]["usage"]["cache_creation"]
    summary = import_results(folder, rows)
    assert summary["calculated_cost_usd"] is None
    assert summary["reported_total_input_tokens"] == 3200
    assert summary["reported_output_tokens"] == 400
    assert summary["cache_read_input_tokens"] == 1000
    assert summary["cache_creation_input_tokens"] == 2000
    assert summary["reported_usage"]["input_tokens"] == 200
    assert summary["documents"]["a1"]["reported_usage"]["output_tokens"] == 400
    assert summary["usage_complete"] is False
    assert summary["usage_missing_fields"]["ephemeral_1h_input_tokens"] == 2
    assert summary["chain_documents"]["a1"]["calculated_cost_usd"] is None


@pytest.mark.parametrize("field", ["input_tokens","output_tokens","cache_read_input_tokens","cache_creation_input_tokens"])
def test_partially_missing_usage_has_known_subtotals_and_completeness(job, field):
    folder, plan = job; receipt(folder)
    rows = results(plan)
    del rows[0]["result"]["message"]["usage"][field]
    summary = import_results(folder, rows)
    assert summary["calculated_cost_usd"] is None
    assert summary["reported_usage"][field] == usage()[field]
    assert summary["usage_missing_fields"][field] == 1
    assert summary["usage_complete"] is False
    assert summary["unknown_cost_count"] == 1


def test_all_zero_usage_without_ttl_is_complete_and_free_calculation(job):
    folder, plan = job; receipt(folder)
    rows = results(plan)
    for row in rows:
        row["result"]["message"]["usage"] = dict.fromkeys(
            ["input_tokens","output_tokens","cache_creation_input_tokens","cache_read_input_tokens"],0)
    summary = import_results(folder, rows)
    assert summary["calculated_cost_usd"] == "0.00"
    assert summary["usage_complete"] is True


def save_completed(folder, rows):
    (folder / "provider-results.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    summary = import_results(folder, rows)
    write_new(folder / "summary.json",summary)
    return summary


def test_document_chain_includes_three_rounds_and_preserves_earlier_documents(job, tmp_path):
    folder, plan = job; receipt(folder)
    # Put the second request in another document, and bind plan again.
    custom_ids = list(plan["chunks"])
    plan["chunks"][custom_ids[1]]["accession"] = "a2"
    # Easier regenerate from source so the custom IDs remain correct.
    source = tmp_path / "two-docs.jsonl"
    source.write_text("\n".join(json.dumps(chunk) for chunk in plan["chunks"].values()))
    from app.graph.ontology import parse_ontology
    plan = prepare_plan(source, parse_ontology(plan["ontology"]), 2)
    (folder / "plan.json").write_text(json.dumps(plan))
    rows = results(plan)
    rows[1]["result"]["message"]["content"] = [{"type":"text","text":"bad"}]
    root = save_completed(folder, rows)
    repair1 = prepare_retry(folder)
    one = tmp_path / "repair-one";one.mkdir();write_new(one / "plan.json",repair1);receipt(one)
    mid = save_completed(one,results(repair1,"bad"))
    repair2 = prepare_retry(one)
    two = tmp_path / "repair-two";two.mkdir();write_new(two / "plan.json",repair2);receipt(two)
    end = save_completed(two,results(repair2))
    assert end["chain_documents_complete"] is True
    assert end["chain_documents"]["a1"]["requests"] == 1
    assert end["chain_documents"]["a2"]["requests"] == 3
    assert Decimal(end["chain_documents"]["a2"]["calculated_cost_usd"]) == Decimal("0.01125") * 3
    assert sum(Decimal(doc["calculated_cost_usd"]) for doc in end["chain_documents"].values()) == Decimal(end["chain_calculated_cost_usd"])
    assert end["chain_documents"]["a2"]["reported_usage"]["output_tokens"] == 600
    with pytest.raises(ValueError,match="two repair"):
        prepare_retry(two)


def test_unknown_parent_document_cost_stays_unknown_with_known_subtotal(job, tmp_path):
    folder, plan = job;receipt(folder)
    rows = results(plan,"bad")
    del rows[0]["result"]["message"]["usage"]["output_tokens"]
    save_completed(folder, rows)
    retry = prepare_retry(folder)
    one = tmp_path / "repair";one.mkdir();write_new(one / "plan.json",retry);receipt(one)
    summary = save_completed(one,results(retry))
    assert summary["chain_calculated_cost_usd"] is None
    doc = summary["chain_documents"]["a1"]
    assert doc["calculated_cost_usd"] is None and Decimal(doc["known_calculated_cost_usd"]) > 0
    assert doc["unknown_cost_count"] == 1 and doc["usage_complete"] is False


def test_request_specific_estimate_changes_with_payload_not_strict_bound(job):
    folder, plan = job
    estimate = plan["planning_estimate"]
    assert estimate["can_authorize_submit"] is False
    assert estimate["kind"] == "offline_heuristic_not_a_bound"
    assert len(estimate["requests"]) == 2
    assert Decimal(estimate["estimated_no_hit_cost_usd"]) < Decimal(plan["strict_reservation_usd"])
    changed = copy.deepcopy(plan["requests"])
    changed[0]["params"]["messages"][0]["content"][1]["text"] += "Longer source. " * 100
    other = planning_estimate(changed,plan["chunks"])
    assert Decimal(other["estimated_no_hit_cost_usd"]) > Decimal(estimate["estimated_no_hit_cost_usd"])
    assert sum(Decimal(value) for value in estimate["documents"].values()) == Decimal(estimate["estimated_no_hit_cost_usd"])


def test_low_estimate_is_never_authority_to_submit(job):
    folder,plan=job
    api=FakeAPI([])
    with pytest.raises(ValueError,match="not guaranteed bounds"):
        submit_job(folder,plan["plan_id"],"1",True,api,max_calls=2)
    assert api.calls == [] and not (folder / "submission-claim.json").exists()


def test_estimate_tampering_rejected_even_with_rehashed_plan(job):
    folder,plan=job
    plan["planning_estimate"]["can_authorize_submit"] = True
    plan.pop("plan_id");plan["plan_id"]=digest(plan)
    with pytest.raises(ValueError,match="planning estimate"):
        verify_plan(plan)


def test_network_guard_blocks_external_dns_and_connection_before_traffic():
    with pytest.raises(RuntimeError,match="external DNS"):
        socket.getaddrinfo("api.anthropic.com",443)
    with socket.socket() as sock:
        with pytest.raises(RuntimeError,match="external network"):
            sock.connect(("203.0.113.1",443))
        with pytest.raises(RuntimeError,match="external network"):
            sock.connect_ex(("203.0.113.1",443))


def test_python_subprocess_inherits_network_guard():
    import subprocess
    import sys
    code = "import socket; socket.getaddrinfo('api.anthropic.com',443)"
    result = subprocess.run([sys.executable,"-c",code],capture_output=True,text=True)
    assert result.returncode != 0
    assert "Offline tests forbid external DNS" in result.stderr


def test_summary_cli_names_partial_usage_and_document_chains(job, monkeypatch, capsys):
    from scripts import summarize_extraction
    folder, plan = job;receipt(folder)
    rows=results(plan)
    del rows[0]["result"]["message"]["usage"]["cache_creation"]
    save_completed(folder,rows)
    monkeypatch.setattr("sys.argv",["summarize_extraction.py","--job-dir",str(folder)])
    assert summarize_extraction.main() == 0
    out=capsys.readouterr().out
    assert "Known reported input-token subtotal: 3200" in out
    assert "Calculated token cost USD: unknown" in out
    assert "Missing usage fields by request:" in out
    assert "Per-document repair-chain totals" in out
    assert "a1:" in out


def test_legacy_multiround_document_history_is_not_called_complete(job, tmp_path):
    folder,plan=job;receipt(folder)
    save_completed(folder,results(plan,"bad"))
    retry=prepare_retry(folder)
    one=tmp_path / "old-repair";one.mkdir();write_new(one / "plan.json",retry);receipt(one)
    middle=save_completed(one,results(retry,"bad"))
    del middle["chain_documents"]
    (one / "summary.json").write_text(json.dumps(middle))
    retry2=prepare_retry(one)
    two=tmp_path / "new-repair";two.mkdir();write_new(two / "plan.json",retry2);receipt(two)
    summary=save_completed(two,results(retry2))
    assert summary["chain_documents_complete"] is False
    assert summary["chain_documents"]["a1"]["calculated_cost_usd"] is None
    assert summary["chain_documents"]["a1"]["usage_complete"] is False


def test_repair_estimate_binds_actual_feedback_payload(job, tmp_path):
    folder,plan=job;receipt(folder)
    save_completed(folder,results(plan,"bad"))
    retry=prepare_retry(folder)
    verify_plan(retry)
    assert retry["planning_estimate"] == planning_estimate(retry["requests"],retry["chunks"])
    assert retry["planning_estimate"]["requests"][0]["request_utf8_bytes"] > plan["planning_estimate"]["requests"][0]["request_utf8_bytes"]
    assert retry["planning_estimate"]["can_authorize_submit"] is False
