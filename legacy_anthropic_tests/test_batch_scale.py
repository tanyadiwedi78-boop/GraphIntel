"""Offline scale/bounds/recovery tests. All provider answers are synthetic."""
import copy
import json
from pathlib import Path

import pytest

from app.extraction import anthropic_batch as batch
from app.extraction import batch_storage as storage
from tests.test_anthropic_batch import job, ontology, receipt, results, FakeAPI


def write_source(path, count, text_size=2000):
    with path.open("w", encoding="utf-8") as handle:
        for number in range(count):
            chunk = {"chunk_id": "chunk-" + str(number),
                     "text": "Acme develops Widget Engine. " + "a" * text_size,
                     "accession": "filing-" + str(number // 10),
                     "form": "10-K", "period": "2026"}
            handle.write(json.dumps(chunk) + "\n")


def test_one_thousand_requests_keep_separate_source_ids(tmp_path, ontology):
    path = tmp_path / "chunks.jsonl"
    write_source(path, 1000)
    plan = batch.prepare_plan(path, ontology, 1000)
    assert len(plan["requests"]) == len(plan["chunks"]) == 1000
    assert len({row["custom_id"] for row in plan["requests"]}) == 1000
    assert len({chunk["chunk_id"] for chunk in plan["chunks"].values()}) == 1000
    batch.verify_plan(plan)
    with pytest.raises(ValueError, match="1 to 1000"):
        batch.prepare_plan(path, ontology, 1001)


def test_prepare_rejects_oversized_batch_without_truncating(tmp_path, ontology, monkeypatch):
    path = tmp_path / "chunks.jsonl"
    write_source(path, 2)
    monkeypatch.setattr(batch, "MAX_BATCH_BYTES", 100)
    with pytest.raises(ValueError, match="batch exceeds local byte limit"):
        batch.prepare_plan(path, ontology, 2)


def test_bounded_line_and_file_readers(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{}\n{}\n')
    assert list(storage.json_lines(path, 3)) == [{}, {}]
    with pytest.raises(ValueError, match="line exceeds"):
        list(storage.json_lines(path, 2))
    with pytest.raises(ValueError, match="file exceeds"):
        list(storage.json_lines(path, 3, 5))
    path.write_bytes(b'{"x":"' + b'a' * 50 + b'"}')
    with pytest.raises(ValueError, match="plan exceeds"):
        old = storage.MAX_PLAN_BYTES
        try:
            storage.MAX_PLAN_BYTES = 10
            storage.read_plan(path)
        finally:
            storage.MAX_PLAN_BYTES = old


def test_oversized_plan_never_leaves_partial_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "MAX_PLAN_BYTES", 10)
    path = tmp_path / "plan.json"
    with pytest.raises(ValueError, match="plan exceeds"):
        batch.write_new(path, {"large": "a" * 20})
    assert not path.exists()


@pytest.mark.parametrize("problem", ["duplicate", "missing", "unknown", "bad_type"])
def test_bad_result_coverage_has_no_partial_extractions(job, problem):
    folder, plan = job
    receipt(folder)
    rows = results(plan)
    if problem == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif problem == "missing":
        rows.pop()
    elif problem == "unknown":
        rows[1]["custom_id"] = "not-in-plan"
    else:
        rows[1]["result"]["type"] = "unknown"
    with pytest.raises(ValueError):
        batch.import_results(folder, iter(rows))
    assert not (folder / "extractions").exists()
    assert not (folder / "summary.json").exists()


def test_result_readers_total_and_per_row_limits(job, monkeypatch):
    folder, plan = job
    receipt(folder)
    monkeypatch.setattr(storage, "MAX_RESULTS_BYTES", 10)
    with pytest.raises(ValueError, match="results exceed"):
        batch.import_results(folder, iter(results(plan)))
    monkeypatch.setattr(storage, "MAX_RESULTS_BYTES", 100000)
    monkeypatch.setattr(storage, "MAX_RESULT_LINE_BYTES", 10)
    with pytest.raises(ValueError, match="result exceeds"):
        batch.import_results(folder, iter(results(plan)))
    assert not (folder / "extractions").exists()


def test_collect_recovers_after_interrupted_import(job, monkeypatch):
    folder, plan = job
    receipt(folder)
    raw = folder / "provider-results.jsonl"
    raw.write_text("\n".join(json.dumps(row) for row in reversed(results(plan))), encoding="utf-8")
    original = batch.write_new
    def interrupted(path, value):
        if Path(path).parent.name == "extractions":
            original(path, value)
            raise OSError("simulated disk interruption after first result")
        return original(path, value)
    monkeypatch.setattr(batch, "write_new", interrupted)
    api = FakeAPI([{"id": "msgbatch_test", "processing_status": "ended"}])
    with pytest.raises(OSError, match="simulated"):
        batch.collect_job(folder, api)
    assert not (folder / "summary.json").exists()
    assert len(list((folder / "extractions").glob("*.json"))) == 1
    before = raw.read_bytes()
    monkeypatch.setattr(batch, "write_new", original)
    api = FakeAPI([{"id": "msgbatch_test", "processing_status": "ended"}])
    summary = batch.collect_job(folder, api)
    assert summary["counts"] == {"valid": 2}
    assert raw.read_bytes() == before
    assert len(list((folder / "extractions").glob("*.json"))) == 2
    assert all(call[0] == "GET" for call in api.calls)


def test_download_limit_preserves_partial_not_a_complete_raw_file(tmp_path, monkeypatch):
    import io
    class OfflineOpener:
        def open(self, request, timeout):
            return io.BytesIO(b"a" * 50)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-offline-only")
    api = batch.BatchAPI()
    api.opener = OfflineOpener()
    monkeypatch.setattr(batch, "MAX_RESULTS_BYTES", 10)
    destination = tmp_path / "provider-results.jsonl"
    with pytest.raises(ValueError, match="partial download preserved"):
        api.download_results("/v1/messages/batches/msgbatch_test/results", destination)
    assert not destination.exists()
    assert len(list(tmp_path.glob("provider-results.jsonl.partial-*"))) == 1


def test_low_disk_stops_download_without_marking_it_complete(tmp_path, monkeypatch):
    import io
    from types import SimpleNamespace
    class OfflineOpener:
        def open(self, request, timeout):
            return io.BytesIO(b"a" * 50)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-offline-only")
    api = batch.BatchAPI()
    api.opener = OfflineOpener()
    monkeypatch.setattr(batch.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    destination = tmp_path / "provider-results.jsonl"
    with pytest.raises(ValueError, match="not enough free disk"):
        api.download_results("/v1/messages/batches/msgbatch_test/results", destination)
    assert not destination.exists()
    assert len(list(tmp_path.glob("provider-results.jsonl.partial-*"))) == 1


def test_old_small_plan_still_passes_unchanged_contract(job):
    folder, plan = job
    original = json.dumps(plan, sort_keys=True)
    batch.verify_plan(plan)
    assert json.dumps(plan, sort_keys=True) == original


def test_single_use_result_generator_is_not_restarted(job):
    folder, plan = job
    receipt(folder)
    expected = results(plan)
    visited = []
    def once():
        for row in reversed(expected):
            visited.append(row["custom_id"])
            yield row
    summary = batch.import_results(folder, once())
    assert summary["counts"] == {"valid": 2}
    assert len(visited) == 2


def test_saved_jsonl_uses_no_temporary_result_copy(job, monkeypatch):
    folder, plan = job
    receipt(folder)
    path = folder / "provider-results.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in reversed(results(plan))), encoding="utf-8")
    def forbidden(*args, **kwargs):
        raise AssertionError("saved results must not be copied to temporary storage")
    monkeypatch.setattr(storage.tempfile, "TemporaryDirectory", forbidden)
    summary = batch.import_results(folder, path)
    assert summary["counts"] == {"valid": 2}
    assert [item["custom_id"] for item in summary["requests"]] == list(plan["chunks"])


def test_changed_raw_file_between_passes_stops_before_extraction(job, monkeypatch):
    folder, plan = job
    receipt(folder)
    path = folder / "provider-results.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in results(plan)), encoding="utf-8")
    original = storage.json_lines
    passes = []
    def changing(source, *limits):
        passes.append(1)
        if len(passes) == 2:
            with path.open("a", encoding="utf-8") as handle:
                handle.write("\n")
        yield from original(source, *limits)
    monkeypatch.setattr(storage, "json_lines", changing)
    with pytest.raises(ValueError, match="raw results changed"):
        batch.import_results(folder, path)
    assert not (folder / "extractions").exists()


def test_incomplete_saved_jsonl_stops_before_extraction(job):
    folder, plan = job
    receipt(folder)
    path = folder / "provider-results.jsonl"
    path.write_text(json.dumps(results(plan)[0]), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        batch.import_results(folder, path)
    assert not (folder / "extractions").exists()


def test_changed_raw_file_during_import_never_creates_summary(job, monkeypatch):
    folder, plan = job
    receipt(folder)
    path = folder / "provider-results.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in results(plan)), encoding="utf-8")
    original = batch.write_new
    def change_after_save(target, value):
        original(target, value)
        if Path(target).parent.name == "extractions":
            with path.open("a", encoding="utf-8") as handle:
                handle.write("\n")
    monkeypatch.setattr(batch, "write_new", change_after_save)
    api = FakeAPI([{"id": "msgbatch_test", "processing_status": "ended"}])
    with pytest.raises(ValueError, match="raw results changed"):
        batch.collect_job(folder, api)
    assert not (folder / "summary.json").exists()
    assert len(list((folder / "extractions").glob("*.json"))) == 1
