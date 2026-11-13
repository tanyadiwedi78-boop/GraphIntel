"""Async batch wall time replaces synchronous per-chunk latency statistics.

Native created_at to ended_at includes provider queue time. No median/p95 or
backoff time can be inferred for individual chunks from this API.
"""
import json
import pytest
from app.extraction.anthropic_batch import batch_timing, collect_job
from tests.test_anthropic_batch import job, ontology, FakeAPI, receipt, results


def test_provider_wall_time_has_explicit_boundaries():
    timing = batch_timing({"created_at":"2026-11-10T05:00:00Z", "ended_at":"2026-11-10T05:02:30Z"})
    assert timing["elapsed_seconds"] == 150
    assert "including queue time" in timing["scope"]
    assert "not per-chunk inference latency" in timing["scope"]
    assert "chunk_median_seconds" not in timing
    assert "chunk_p95_seconds" not in timing


@pytest.mark.parametrize("status", [{}, {"created_at":"bad", "ended_at":"bad"},
    {"created_at":"2026-11-10T05:00:00", "ended_at":"2026-11-10T05:01:00"},
    {"created_at":"2026-11-10T05:02:00Z", "ended_at":"2026-11-10T05:00:00Z"}])
def test_missing_or_invalid_native_times_stay_unknown(status):
    assert batch_timing(status)["elapsed_seconds"] is None


def test_collection_persists_native_wall_time_and_resume_does_not_remeasure(job):
    folder, plan = job
    receipt(folder)
    status = {"id":"msgbatch_test", "processing_status":"ended",
              "created_at":"2026-11-10T05:00:00Z", "ended_at":"2026-11-10T05:03:00Z"}
    api = FakeAPI([status, "\n".join(json.dumps(row) for row in results(plan))])
    summary = collect_job(folder, api)
    assert summary["batch_timing"]["elapsed_seconds"] == 180
    assert collect_job(folder, FakeAPI([]))["batch_timing"] == summary["batch_timing"]
    assert len(api.calls) == 2
    assert all(call[0] == "GET" for call in api.calls)


@pytest.mark.parametrize('elapsed', [None, 0, 180])
def test_summary_cli_reports_native_time_or_unknown(job, monkeypatch, capsys, elapsed):
    from scripts import summarize_extraction
    folder, plan = job
    receipt(folder)
    status = {'id':'msgbatch_test', 'processing_status':'ended'}
    if elapsed is not None:
        status.update(created_at='2026-11-10T05:00:00Z',
                      ended_at='2026-11-10T05:03:00Z' if elapsed else '2026-11-10T05:00:00Z')
    collect_job(folder, FakeAPI([status, '\n'.join(json.dumps(row) for row in results(plan))]))
    monkeypatch.setattr('sys.argv', ['summarize_extraction.py', '--job-dir', str(folder)])
    assert summarize_extraction.main() == 0
    output = capsys.readouterr().out
    expected = 'unknown' if elapsed is None else str(float(elapsed))
    assert 'Provider batch wall time seconds: ' + expected in output
    assert 'not per-chunk inference latency' in output
