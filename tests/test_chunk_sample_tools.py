"""Chunk sample tools: list_chunks, review_extraction, and --chunk-ids selection.

No network and no real API key: selection errors are checked before any model
client is built, the review tools only read local files, and the cap/resume
test drives run_extraction with the fake model.
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from app.extraction.runner import RunLimits, run_extraction
from app.extraction.client import ProviderError
from app.graph.ontology import load_ontology
from tests.fake_model import FakeModel, good_answer

REPO_ROOT = Path(__file__).resolve().parent.parent

GOOD_FLAGS = ["--gemini-assume-free", "--gemini-max-rpm", "100", "--gemini-max-tpm", "100000"]


def no_sleep(seconds):
    return None


def write_chunks(data_dir, specs):
    chunks_dir = data_dir / "chunks"
    chunks_dir.mkdir(parents=True)
    with open(chunks_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for spec in specs:
            chunk = {
                "chunk_id": spec["chunk_id"],
                "ticker": spec.get("ticker", "ACME"),
                "form": spec.get("form", "10-K"),
                "accession": spec.get("accession", "0000000000-26-000001"),
                "period": spec.get("period", "2026-01-31"),
                "item_id": spec.get("item_id", "item_1"),
                "is_table": spec.get("is_table", False),
                "est_tokens": spec.get("est_tokens", 100),
                "text": spec["text"],
            }
            f.write(json.dumps(chunk) + "\n")


def sample_specs():
    return [
        {"chunk_id": "chunk-1", "item_id": "cover", "text": "Acme Inc. cover page."},
        {"chunk_id": "chunk-2", "item_id": "item_1a", "text": "Acme Inc. faces supply risk."},
        {"chunk_id": "chunk-3", "item_id": "item_1a", "text": "Acme depends on one supplier."},
    ]


def three_doc_specs():
    specs = []
    for doc in range(3):
        accession = "0000000000-26-00000" + str(doc + 1)
        for item in ["cover", "item_1", "item_1a"]:
            specs.append({
                "chunk_id": "doc" + str(doc) + "-" + item,
                "ticker": "T" + str(doc),
                "accession": accession,
                "item_id": item,
                "is_table": item == "item_1a" and doc == 2,
                "text": "Text for doc " + str(doc) + " " + item + ".",
            })
    return specs


def run_tool(script, data_dir, extra_args=None, extra_env=None):
    args = [sys.executable, "scripts/" + script]
    if extra_args is not None:
        args = args + extra_args
    env = dict(os.environ)
    env["DATA_DIR"] = str(data_dir)
    env.pop("GEMINI_API_KEY", None)  # tests never see a real key
    if extra_env is not None:
        env.update(extra_env)
    return subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, env=env)


def test_list_chunks_summary_groups_by_section(tmp_path):
    write_chunks(tmp_path, sample_specs())
    result = run_tool("list_chunks.py", tmp_path)
    assert result.returncode == 0
    assert "3 chunks" in result.stdout
    assert "cover | 1 chunks | first: chunk-1" in result.stdout
    assert "item_1a | 2 chunks | first: chunk-2" in result.stdout


def test_list_chunks_show_prints_text_and_unknown_id_fails(tmp_path):
    write_chunks(tmp_path, sample_specs())
    shown = run_tool("list_chunks.py", tmp_path, ["--show", "chunk-2"])
    assert shown.returncode == 0
    assert "Acme Inc. faces supply risk." in shown.stdout
    missing = run_tool("list_chunks.py", tmp_path, ["--show", "chunk-99"])
    assert missing.returncode == 1
    assert "chunk-99" in missing.stdout


def test_sample_picks_varied_chunks_and_prints_real_commands(tmp_path):
    write_chunks(tmp_path, three_doc_specs())
    result = run_tool("list_chunks.py", tmp_path, ["--sample", "6"])
    assert result.returncode == 0
    assert "T0" in result.stdout
    assert "T1" in result.stdout
    assert "T2" in result.stdout
    assert "table: True" in result.stdout
    for line in result.stdout.splitlines():
        if line.startswith("python scripts/run_extraction.py"):
            ids_part = line.split("--chunk-ids ", 1)[1].split(" ", 1)[0]
            ids = ids_part.split(",")
            assert len(ids) == 6
            for chunk_id in ids:
                assert chunk_id.startswith("doc")


def test_sample_is_deterministic(tmp_path):
    write_chunks(tmp_path, three_doc_specs())
    first = run_tool("list_chunks.py", tmp_path, ["--sample", "6"])
    second = run_tool("list_chunks.py", tmp_path, ["--sample", "6"])
    assert first.returncode == 0
    assert first.stdout == second.stdout


def test_load_chunks_selects_only_requested_ids(tmp_path):
    write_chunks(tmp_path, sample_specs())
    spec = importlib.util.spec_from_file_location("run_extraction_script",
                                                  REPO_ROOT / "scripts" / "run_extraction.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    chunks_path = tmp_path / "chunks" / "chunks.jsonl"
    picked = module.load_chunks(chunks_path, 10, ["chunk-2"])
    assert len(picked) == 1
    assert picked[0]["chunk_id"] == "chunk-2"
    first_two = module.load_chunks(chunks_path, 2)
    assert len(first_two) == 2
    assert first_two[0]["chunk_id"] == "chunk-1"


def test_run_extraction_rejects_unknown_chunk_id_before_any_call(tmp_path):
    write_chunks(tmp_path, sample_specs())
    args = ["--chunk-ids", "chunk-1,chunk-99"] + GOOD_FLAGS
    env = {"MAX_EXTRACTION_CALLS": "100", "GEMINI_API_KEY": "fake-key"}
    result = run_tool("run_extraction.py", tmp_path, args, env)
    assert result.returncode == 1
    assert "chunk-99" in result.stdout


def test_run_extraction_rejects_empty_chunk_ids(tmp_path):
    write_chunks(tmp_path, sample_specs())
    args = ["--chunk-ids", " , "] + GOOD_FLAGS
    env = {"MAX_EXTRACTION_CALLS": "100", "GEMINI_API_KEY": "fake-key"}
    result = run_tool("run_extraction.py", tmp_path, args, env)
    assert result.returncode == 1
    assert "no ids" in result.stdout


def make_runner_chunk(chunk_id, quote):
    return {
        "chunk_id": chunk_id,
        "text": "Item 1. Acme Corp develops the Widget Engine. " + quote,
        "accession": "0000000000-26-000001",
        "form": "10-K",
        "period": "2026-01-31",
        "item_id": "item_1",
        "start": 0,
        "end": 60,
    }


def test_call_cap_batches_resume_without_repeating_work(tmp_path):
    ontology = load_ontology(REPO_ROOT / "configs" / "ontology.json")
    quote = "develops the Widget Engine"
    chunks = []
    for number in range(6):
        chunks.append(make_runner_chunk("chunk-" + str(number + 1), quote))
    results_dir = str(tmp_path / "extraction")
    limits = RunLimits(max_calls=3)

    first_model = FakeModel([good_answer(quote), good_answer(quote), good_answer(quote)])
    first = run_extraction(chunks, first_model, ontology, "fake-model", results_dir, limits,
                           sleep=no_sleep)
    assert first["valid"] == 3
    assert first["stopped_early"] is True
    assert first["stop_reason"] == "call cap reached"
    assert len(first_model.calls) == 3

    second_model = FakeModel([good_answer(quote), good_answer(quote), good_answer(quote)])
    second = run_extraction(chunks, second_model, ontology, "fake-model", results_dir, limits,
                            sleep=no_sleep)
    assert second["skipped_done"] == 3
    assert second["valid"] == 3
    assert second["stopped_early"] is False
    assert len(second_model.calls) == 3
    assert len(os.listdir(results_dir)) == 6


def test_write_labels_adds_entries_and_keeps_filled_ones(tmp_path):
    write_chunks(tmp_path, sample_specs())
    labels_path = tmp_path / "labels.json"
    first = run_tool("review_extraction.py", tmp_path,
                     ["--chunk-ids", "chunk-1,chunk-2", "--write-labels", str(labels_path)])
    assert first.returncode == 0
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    assert labels["chunk-1"] == {"entities": [], "relationships": [], "notes": ""}
    labels["chunk-1"]["entities"] = [{"name": "Acme Inc.", "type": "Company"}]
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f)
    second = run_tool("review_extraction.py", tmp_path,
                      ["--chunk-ids", "chunk-1,chunk-3", "--write-labels", str(labels_path)])
    assert second.returncode == 0
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    assert labels["chunk-1"]["entities"] == [{"name": "Acme Inc.", "type": "Company"}]
    assert "chunk-3" in labels


def test_review_sheet_shows_source_and_saved_result(tmp_path):
    write_chunks(tmp_path, sample_specs())
    results_dir = tmp_path / "extraction"
    results_dir.mkdir(parents=True)
    record = {
        "status": "valid",
        "chunk_id": "chunk-2",
        "prompt_version": "p2-rules-v1",
        "schema_version": "s3",
        "model": "fake-model",
        "entities": [{"name": "Acme Inc.", "type": "Company"}],
        "relationships": [],
    }
    with open(results_dir / "chunk-2__fakekey.json", "w", encoding="utf-8") as f:
        json.dump(record, f)
    result = run_tool("review_extraction.py", tmp_path, ["--chunk-ids", "chunk-2"])
    assert result.returncode == 0
    assert "Acme Inc. faces supply risk." in result.stdout
    assert "EXTRACTION from" in result.stdout
    assert "Acme Inc. (Company)" in result.stdout


def test_score_reports_wrong_and_missed_items(tmp_path):
    write_chunks(tmp_path, sample_specs())
    results_dir = tmp_path / "extraction"
    results_dir.mkdir(parents=True)
    record = {
        "status": "valid",
        "chunk_id": "chunk-1",
        "prompt_version": "p2-rules-v1",
        "schema_version": "s3",
        "model": "fake-model",
        "entities": [
            {"name": "Acme Inc.", "type": "Company"},
            {"name": "Common Stock", "type": "FinancialMetric"},
        ],
        "relationships": [
            {"relation": "REPORTS", "source": "Acme Inc.", "target": "Common Stock",
             "evidence_quote": "Acme Inc. cover page.", "confidence": 0.8},
        ],
    }
    with open(results_dir / "chunk-1__fakekey.json", "w", encoding="utf-8") as f:
        json.dump(record, f)
    labels_path = tmp_path / "labels.json"
    labels = {
        "chunk-1": {
            "entities": [{"name": "Acme Inc.", "type": "Company"}],
            "relationships": [],
            "notes": "",
        }
    }
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f)
    result = run_tool("review_extraction.py", tmp_path, ["--labels", str(labels_path)])
    assert result.returncode == 0
    assert "precision 1/2 = 0.50" in result.stdout
    assert "recall 1/1 = 1.00" in result.stdout
    assert "wrong extraction: Common Stock (FinancialMetric)" in result.stdout
    assert "wrong extraction: Acme Inc. -REPORTS-> Common Stock" in result.stdout
    assert "precision 0/1 = 0.00" in result.stdout


def test_repeated_command_visits_every_chunk_exactly_once(tmp_path):
    # Proof for the whole-corpus flow: 7 chunks, cap 3. Each invocation does
    # only unfinished chunks; a final invocation does nothing at all.
    ontology = load_ontology(REPO_ROOT / "configs" / "ontology.json")
    quote = "develops the Widget Engine"
    chunks = []
    for number in range(7):
        chunks.append(make_runner_chunk("chunk-" + str(number + 1), quote))
    results_dir = str(tmp_path / "extraction")
    limits = RunLimits(max_calls=3)

    plan = [3, 3, 1, 0]  # calls each of the four invocations must make
    summaries = []
    for expected_calls in plan:
        steps = []
        for _ in range(expected_calls):
            steps.append(good_answer(quote))
        model = FakeModel(steps)
        summary = run_extraction(chunks, model, ontology, "fake-model", results_dir, limits,
                                 sleep=no_sleep)
        assert len(model.calls) == expected_calls
        summaries.append(summary)
    assert summaries[0]["valid"] == 3 and summaries[0]["stop_reason"] == "call cap reached"
    assert summaries[1]["valid"] == 3
    assert summaries[2]["valid"] == 1 and summaries[2]["stopped_early"] is False
    assert summaries[3]["skipped_done"] == 7 and summaries[3]["calls_used"] == 0
    assert len(os.listdir(results_dir)) == 7


def test_failing_chunk_gives_up_and_the_campaign_still_advances(tmp_path):
    # 5 chunks, cap 3, chunk-2 always fails with a per-chunk error. The runs
    # must keep visiting the unfinished chunks around it, and after 3 separate
    # failing runs chunk-2 is left alone instead of burning the cap forever.
    from app.extraction.client import ChunkFailedError
    ontology = load_ontology(REPO_ROOT / "configs" / "ontology.json")
    quote = "develops the Widget Engine"
    chunks = []
    for number in range(5):
        chunks.append(make_runner_chunk("chunk-" + str(number + 1), quote))
    results_dir = str(tmp_path / "extraction")
    limits = RunLimits(max_calls=3)

    # Run 1: chunk-1 ok, chunk-2 fails (1 call, no retry), chunk-3 ok; cap hit at chunk-4.
    model1 = FakeModel([good_answer(quote), ChunkFailedError(400, "cut off"), good_answer(quote)])
    first = run_extraction(chunks, model1, ontology, "fake-model", results_dir, limits, sleep=no_sleep)
    assert first["valid"] == 2 and first["failed"] == 1 and first["stop_reason"] == "call cap reached"

    # Run 2: chunk-2 fails again, chunk-4 and chunk-5 finish; everything else resumes.
    model2 = FakeModel([ChunkFailedError(400, "cut off"), good_answer(quote), good_answer(quote)])
    second = run_extraction(chunks, model2, ontology, "fake-model", results_dir, limits, sleep=no_sleep)
    assert second["valid"] == 2 and second["failed"] == 1 and second["skipped_done"] == 2

    # Run 3: only chunk-2 is unfinished; it fails a third separate run.
    model3 = FakeModel([ChunkFailedError(400, "cut off")])
    third = run_extraction(chunks, model3, ontology, "fake-model", results_dir, limits, sleep=no_sleep)
    assert third["failed"] == 1 and third["calls_used"] == 1

    # Run 4: chunk-2 is given up without any model call; nothing else is reprocessed.
    model4 = FakeModel([])
    fourth = run_extraction(chunks, model4, ontology, "fake-model", results_dir, limits, sleep=no_sleep)
    assert len(model4.calls) == 0
    assert fourth["gave_up"] == 1
    assert fourth["skipped_done"] == 4
    assert fourth["calls_used"] == 0


def test_invalid_answer_gives_up_after_three_separate_runs(tmp_path):
    # A chunk whose answer never validates is retried across runs, then left
    # alone: three separate runs of attempts, no fourth.
    ontology = load_ontology(REPO_ROOT / "configs" / "ontology.json")
    quote = "develops the Widget Engine"
    chunks = [make_runner_chunk("chunk-1", quote), make_runner_chunk("chunk-2", quote)]
    results_dir = str(tmp_path / "extraction")
    limits = RunLimits(max_calls=10)

    good_run = FakeModel([good_answer(quote)])
    run_extraction([chunks[0]], good_run, ontology, "fake-model", results_dir, limits, sleep=no_sleep)

    for expected_runs in [1, 2, 3]:
        bad_run = FakeModel(["not json at all", "still not json", "nope"])
        summary = run_extraction(chunks, bad_run, ontology, "fake-model", results_dir, limits,
                                 sleep=no_sleep)
        assert summary["invalid"] == 1
        assert summary["skipped_done"] == 1

    fourth_model = FakeModel([])
    fourth = run_extraction(chunks, fourth_model, ontology, "fake-model", results_dir, limits,
                            sleep=no_sleep)
    assert len(fourth_model.calls) == 0
    assert fourth["gave_up"] == 1
    assert fourth["skipped_done"] == 1


def test_record_from_before_the_runs_counter_counts_as_one_run(tmp_path):
    # Records saved by the old code have no "runs" field: they must count as
    # one run, so they are retried twice more before any give-up.
    from app.extraction.client import ChunkFailedError
    from app.extraction.runner import make_work_key
    ontology = load_ontology(REPO_ROOT / "configs" / "ontology.json")
    quote = "develops the Widget Engine"
    chunk = make_runner_chunk("chunk-1", quote)
    results_dir = str(tmp_path / "extraction")
    os.makedirs(results_dir)
    key = make_work_key(chunk, ontology.fingerprint(), "s3", "p2-rules-v1", "fake-model")
    old_record = {"status": "failed", "chunk_id": "chunk-1", "work_key": key}
    with open(os.path.join(results_dir, "chunk-1__" + key + ".json"), "w", encoding="utf-8") as f:
        json.dump(old_record, f)

    limits = RunLimits(max_calls=3)
    model = FakeModel([ChunkFailedError(400, "cut off")])
    summary = run_extraction([chunk], model, ontology, "fake-model", results_dir, limits, sleep=no_sleep)
    assert len(model.calls) == 1  # retried, not given up
    assert summary["failed"] == 1
    with open(os.path.join(results_dir, "chunk-1__" + key + ".json"), "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["runs"] == 2


def write_result_record(results_dir, chunk_id, status="valid",
                        prompt_version="p2-rules-v1", schema_version="s3"):
    results_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "status": status,
        "chunk_id": chunk_id,
        "prompt_version": prompt_version,
        "schema_version": schema_version,
        "model": "fake-model",
        "entities": [{"name": "Acme Inc.", "type": "Company"}],
        "relationships": [],
    }
    with open(results_dir / (chunk_id + "__fakekey.json"), "w", encoding="utf-8") as f:
        json.dump(record, f)


def write_labels_file(labels_path, labels):
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f)


def test_score_fails_clearly_on_malformed_labels_json(tmp_path):
    labels_path = tmp_path / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        f.write("{not json")
    result = run_tool("review_extraction.py", tmp_path, ["--labels", str(labels_path)])
    assert result.returncode == 1
    assert "not valid JSON" in result.stdout


def test_score_fails_clearly_on_bad_label_shape(tmp_path):
    labels_path = tmp_path / "labels.json"
    write_labels_file(labels_path, {"chunk-1": {"entities": []}})  # no relationships list
    result = run_tool("review_extraction.py", tmp_path, ["--labels", str(labels_path)])
    assert result.returncode == 1
    assert 'missing the "relationships" list' in result.stdout
    assert "nothing" not in result.stdout.lower() or "before anything is scored" in result.stdout


def test_score_excludes_empty_labels_entry(tmp_path):
    write_chunks(tmp_path, sample_specs())
    write_result_record(tmp_path / "extraction", "chunk-2")
    labels_path = tmp_path / "labels.json"
    write_labels_file(labels_path, {
        "chunk-1": {"entities": [], "relationships": [], "notes": ""},
        "chunk-2": {"entities": [{"name": "Acme Inc.", "type": "Company"}],
                    "relationships": [], "notes": ""},
    })
    result = run_tool("review_extraction.py", tmp_path, ["--labels", str(labels_path)])
    assert result.returncode == 0
    assert "chunk-1: labels are empty" in result.stdout
    assert "Overall across 1 scored chunks" in result.stdout
    assert "1 labelled chunks were excluded" in result.stdout


def test_score_excludes_old_version_results(tmp_path):
    write_chunks(tmp_path, sample_specs())
    write_result_record(tmp_path / "extraction", "chunk-1", prompt_version="p1-old")
    labels_path = tmp_path / "labels.json"
    write_labels_file(labels_path, {
        "chunk-1": {"entities": [{"name": "Acme Inc.", "type": "Company"}],
                    "relationships": [], "notes": ""},
    })
    result = run_tool("review_extraction.py", tmp_path, ["--labels", str(labels_path)])
    assert result.returncode == 1
    assert "not scored - this result used prompt p1-old" in result.stdout
    assert "Nothing could be scored" in result.stdout


def test_score_excludes_non_valid_results(tmp_path):
    write_chunks(tmp_path, sample_specs())
    write_result_record(tmp_path / "extraction", "chunk-1", status="invalid")
    labels_path = tmp_path / "labels.json"
    write_labels_file(labels_path, {
        "chunk-1": {"entities": [{"name": "Acme Inc.", "type": "Company"}],
                    "relationships": [], "notes": ""},
    })
    result = run_tool("review_extraction.py", tmp_path, ["--labels", str(labels_path)])
    assert result.returncode == 1
    assert 'status "invalid", not valid' in result.stdout
    assert "Nothing could be scored" in result.stdout


def test_transient_failures_never_give_up_and_stay_pending(tmp_path):
    # A chunk failing with retry-later provider errors (429: rate limit or
    # quota) must NEVER be given up: that is the run's problem, not the
    # chunk's. Four separate failing runs, zero give-ups, chunk still pending.
    ontology = load_ontology(REPO_ROOT / "configs" / "ontology.json")
    quote = "develops the Widget Engine"
    chunk = make_runner_chunk("chunk-1", quote)
    results_dir = str(tmp_path / "extraction")
    limits = RunLimits(max_calls=3)
    for round_number in range(4):
        model = FakeModel([ProviderError(429), ProviderError(429), ProviderError(429)])
        summary = run_extraction([chunk], model, ontology, "fake-model", results_dir, limits,
                                 sleep=no_sleep)
        assert summary["failed"] == 1
        assert summary["gave_up"] == 0
        assert summary["calls_used"] == 3  # in-run retries count against the cap
    from app.extraction.runner import make_work_key
    key = make_work_key(chunk, ontology.fingerprint(), "s3", "p2-rules-v1", "fake-model")
    with open(os.path.join(results_dir, "chunk-1__" + key + ".json"), "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["failure_kind"] == "transient"
    assert saved["runs"] == 0  # transient failures never count toward give-up


def test_retry_given_up_flag_reincludes_chunks_without_deleting_records(tmp_path):
    from app.extraction.client import ChunkFailedError
    ontology = load_ontology(REPO_ROOT / "configs" / "ontology.json")
    quote = "develops the Widget Engine"
    chunk = make_runner_chunk("chunk-1", quote)
    results_dir = str(tmp_path / "extraction")
    limits = RunLimits(max_calls=3)
    for round_number in range(3):
        model = FakeModel([ChunkFailedError(400, "cut off")])
        run_extraction([chunk], model, ontology, "fake-model", results_dir, limits, sleep=no_sleep)
    # Given up now: a normal run makes no calls.
    quiet = run_extraction([chunk], FakeModel([]), ontology, "fake-model", results_dir, limits,
                           sleep=no_sleep)
    assert quiet["gave_up"] == 1 and quiet["calls_used"] == 0
    # Recovery is a flag, not a deletion: the record stays on disk and the chunk succeeds.
    retry_limits = RunLimits(max_calls=3, retry_given_up=True)
    model = FakeModel([good_answer(quote)])
    retried = run_extraction([chunk], model, ontology, "fake-model", results_dir, retry_limits,
                             sleep=no_sleep)
    assert retried["valid"] == 1 and retried["gave_up"] == 0
    assert len(os.listdir(results_dir)) == 1  # same record file, now valid


def test_give_up_after_runs_is_configurable(tmp_path):
    from app.extraction.client import ChunkFailedError
    ontology = load_ontology(REPO_ROOT / "configs" / "ontology.json")
    quote = "develops the Widget Engine"
    chunk = make_runner_chunk("chunk-1", quote)
    results_dir = str(tmp_path / "extraction")
    strict = RunLimits(max_calls=3, give_up_after_runs=1)
    model = FakeModel([ChunkFailedError(400, "cut off")])
    run_extraction([chunk], model, ontology, "fake-model", results_dir, strict, sleep=no_sleep)
    second = run_extraction([chunk], FakeModel([]), ontology, "fake-model", results_dir, strict,
                            sleep=no_sleep)
    assert second["gave_up"] == 1


def test_record_from_an_older_recipe_is_ignored_not_counted(tmp_path):
    # A failed record under a different model/prompt/schema has a different
    # work key: it must not count toward give-up or block a fresh attempt.
    from app.extraction.client import ChunkFailedError
    from app.extraction.runner import make_work_key
    ontology = load_ontology(REPO_ROOT / "configs" / "ontology.json")
    quote = "develops the Widget Engine"
    chunk = make_runner_chunk("chunk-1", quote)
    results_dir = str(tmp_path / "extraction")
    os.makedirs(results_dir)
    old_key = make_work_key(chunk, ontology.fingerprint(), "s2-old", "p1-old", "old-model")
    with open(os.path.join(results_dir, "chunk-1__" + old_key + ".json"), "w", encoding="utf-8") as f:
        json.dump({"status": "failed", "runs": 99, "chunk_id": "chunk-1", "work_key": old_key}, f)
    limits = RunLimits(max_calls=3)
    model = FakeModel([good_answer(quote)])
    summary = run_extraction([chunk], model, ontology, "fake-model", results_dir, limits, sleep=no_sleep)
    assert len(model.calls) == 1  # fresh attempt under the current recipe
    assert summary["valid"] == 1 and summary["gave_up"] == 0


def load_run_extraction_module():
    spec = importlib.util.spec_from_file_location("run_extraction_script",
                                                  REPO_ROOT / "scripts" / "run_extraction.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_max_documents_scopes_the_run_to_the_first_documents(tmp_path):
    write_chunks(tmp_path, three_doc_specs())
    module = load_run_extraction_module()
    chunks_path = tmp_path / "chunks" / "chunks.jsonl"
    chunks = module.load_chunks(chunks_path, 100)
    one_doc = module.limit_documents(chunks, 1)
    assert len(one_doc) == 3
    for chunk in one_doc:
        assert chunk["accession"] == "0000000000-26-000001"
    two_docs = module.limit_documents(chunks, 2)
    assert len(two_docs) == 6
    all_docs = module.limit_documents(chunks, 99)
    assert len(all_docs) == 9


def test_persistent_server_error_parks_chunks_and_continues(tmp_path):
    # A provider-wide failure (persistent 503: network or server) parks the
    # chunk and the run CONTINUES: the parked chunk's record stays pending
    # (transient, runs not counted) and later chunks are still visited, so a
    # rerun replays everything parked. The circuit breaker (5 parked chunks in
    # a row) is not reached here, so no cooldown happens.
    ontology = load_ontology(REPO_ROOT / "configs" / "ontology.json")
    quote = "develops the Widget Engine"
    chunks = []
    for number in range(3):
        chunks.append(make_runner_chunk("chunk-" + str(number + 1), quote))
    results_dir = str(tmp_path / "extraction")
    limits = RunLimits(max_calls=10)
    model = FakeModel([ProviderError(503)] * 9)
    summary = run_extraction(chunks, model, ontology, "fake-model", results_dir, limits,
                             sleep=no_sleep)
    assert summary["calls_used"] == 9  # 3 chunks, 3 attempts each
    assert summary["stopped_early"] is False
    assert summary["parked_transient"] == 3
    assert summary["circuit_trips"] == 0
    assert len(model.calls) == 9  # every chunk was visited
    saved_files = sorted(os.listdir(results_dir))
    assert len(saved_files) == 3  # all three chunks parked: pending, transient
    for filename in saved_files:
        with open(os.path.join(results_dir, filename), "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["failure_kind"] == "transient"
        assert saved["runs"] == 0


def test_auth_error_stops_run_immediately(tmp_path):
    # 401/403 is a setup problem: one call, immediate stop, nothing written,
    # later chunks never visited. Everything stays pending for a fixed setup.
    for status_code in [401, 403]:
        ontology = load_ontology(REPO_ROOT / "configs" / "ontology.json")
        quote = "develops the Widget Engine"
        chunks = []
        for number in range(3):
            chunks.append(make_runner_chunk("chunk-" + str(number + 1), quote))
        results_dir = str(tmp_path / ("extraction" + str(status_code)))
        limits = RunLimits(max_calls=10)
        model = FakeModel([ProviderError(status_code)])
        summary = run_extraction(chunks, model, ontology, "fake-model", results_dir, limits,
                                 sleep=no_sleep)
        assert summary["calls_used"] == 1
        assert summary["stopped_early"] is True
        assert "setup problem" in summary["stop_reason"]
        assert len(model.calls) == 1


def test_daily_quota_429_stops_the_run_now(tmp_path):
    # A 429 whose requested wait is longer than the allowed wait (daily quota
    # exhausted) stops the run at once instead of retrying or moving on.
    ontology = load_ontology(REPO_ROOT / "configs" / "ontology.json")
    quote = "develops the Widget Engine"
    chunks = []
    for number in range(3):
        chunks.append(make_runner_chunk("chunk-" + str(number + 1), quote))
    results_dir = str(tmp_path / "extraction")
    limits = RunLimits(max_calls=10)
    model = FakeModel([ProviderError(429, retry_after_seconds=90000)])
    summary = run_extraction(chunks, model, ontology, "fake-model", results_dir, limits,
                             sleep=no_sleep)
    assert summary["calls_used"] == 1
    assert summary["stopped_early"] is True
    assert "wait longer than the limit" in summary["stop_reason"]
    assert len(model.calls) == 1


def test_summarize_reports_per_document_measured_totals(tmp_path):
    logs_dir = tmp_path / "extraction_logs"
    logs_dir.mkdir(parents=True)
    run_log = [
        {"event": "attempt", "chunk_id": "c1", "document_id": "doc-a",
         "metrics": {"prompt_tokens": 100, "completion_tokens": 10}},
        {"event": "chunk_final", "chunk_id": "c1", "document_id": "doc-a",
         "status": "valid", "chunk_seconds": 2.0},
        {"event": "attempt", "chunk_id": "c2", "document_id": "doc-a",
         "metrics": {"prompt_tokens": 120, "completion_tokens": 12}},
        {"event": "chunk_final", "chunk_id": "c2", "document_id": "doc-a",
         "status": "failed", "chunk_seconds": 4.0},
        {"event": "attempt", "chunk_id": "c3", "document_id": "doc-b",
         "metrics": {"prompt_tokens": 200, "completion_tokens": 20}},
        {"event": "chunk_final", "chunk_id": "c3", "document_id": "doc-b",
         "status": "valid", "chunk_seconds": 6.0},
        {"event": "run_final", "summary": {"elapsed_seconds": 12.0}},
    ]
    with open(logs_dir / "run1.jsonl", "w", encoding="utf-8") as f:
        for event in run_log:
            f.write(json.dumps(event) + "\n")
    result = run_tool("summarize_extraction.py", tmp_path)
    assert result.returncode == 0
    assert "Per-document measured totals" in result.stdout
    assert "doc-a: calls=2 failed=1 valid=1" in result.stdout
    assert "doc-b: calls=1 valid=1" in result.stdout
    assert "prompt-tokens=220" in result.stdout
    assert "Cost per document in dollars: unknown" in result.stdout


def test_pilot_scope_and_full_scope_coverage_under_the_cap(tmp_path):
    # The pilot (default scope: 1 document) finishes document 1 and never
    # touches the others. The full-corpus route (--max-documents covering all
    # documents) advances through every document under the 3-call cap:
    # repeated identical runs visit the unfinished chunks in order until all
    # documents are valid.
    ontology = load_ontology(REPO_ROOT / "configs" / "ontology.json")
    quote = "develops the Widget Engine"
    module = load_run_extraction_module()
    chunks = []
    for doc in range(3):
        accession = "0000000000-26-00000" + str(doc + 1)
        for number in range(2):
            chunk = make_runner_chunk("doc" + str(doc) + "-chunk-" + str(number), quote)
            chunk["accession"] = accession
            chunks.append(chunk)
    results_dir = str(tmp_path / "extraction")
    limits = RunLimits(max_calls=3)

    # Pilot scope: document 1 only, done in one run; the rest are untouched.
    pilot = module.limit_documents(chunks, 1)
    assert len(pilot) == 2
    first = run_extraction(pilot, FakeModel([good_answer(quote), good_answer(quote)]),
                           ontology, "fake-model", results_dir, limits, sleep=no_sleep)
    assert first["valid"] == 2
    again = run_extraction(pilot, FakeModel([]), ontology, "fake-model", results_dir, limits,
                           sleep=no_sleep)
    assert again["skipped_done"] == 2 and again["calls_used"] == 0
    assert len(os.listdir(results_dir)) == 2  # only document 1 has records

    # Full-corpus route: repeated capped runs advance through documents 2
    # and 3 until all six chunks are valid - no document is stranded.
    full = module.limit_documents(chunks, 99)
    assert len(full) == 6
    run_extraction(full, FakeModel([good_answer(quote)] * 3), ontology, "fake-model",
                   results_dir, limits, sleep=no_sleep)
    run_extraction(full, FakeModel([good_answer(quote)] * 3), ontology, "fake-model",
                   results_dir, limits, sleep=no_sleep)
    final = run_extraction(full, FakeModel([]), ontology, "fake-model", results_dir, limits,
                           sleep=no_sleep)
    assert final["skipped_done"] == 6
    assert final["valid"] == 0 and final["calls_used"] == 0
    assert len(os.listdir(results_dir)) == 6


def document_only_specs():
    # The shape of the real corpus sample: one section id (DOCUMENT) for
    # every chunk, two filings, mostly tables - the case that starved the
    # old selector.
    specs = []
    specs.append({"chunk_id": "doc-a-0", "accession": "0000000000-26-000010",
                  "item_id": "DOCUMENT", "is_table": True, "text": "Filing A cover."})
    specs.append({"chunk_id": "doc-a-1", "accession": "0000000000-26-000010",
                  "item_id": "DOCUMENT", "is_table": True, "text": "Filing A body."})
    for number in range(6):
        specs.append({"chunk_id": "doc-b-" + str(number),
                      "accession": "0000000000-26-000011",
                      "item_id": "DOCUMENT", "is_table": number % 3 != 2,
                      "text": "Filing B chunk " + str(number) + "."})
    return specs


def test_sample_fills_requested_count_on_document_only_corpus(tmp_path):
    # Every chunk shares one section id: section metadata alone cannot
    # diversify, and the sample must still reach the requested count.
    write_chunks(tmp_path, document_only_specs())
    result = run_tool("list_chunks.py", tmp_path, ["--sample", "4"])
    assert result.returncode == 0
    assert "Sample of 4 chunks" in result.stdout
    assert "Could only pick" not in result.stdout
    ids = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped[:2] in ["1.", "2.", "3.", "4."]:
            ids.append(stripped.split()[1])
    assert len(ids) == 4
    has_a = False
    has_b = False
    for chunk_id in ids:
        if chunk_id.startswith("doc-a"):
            has_a = True
        if chunk_id.startswith("doc-b"):
            has_b = True
    assert has_a and has_b  # both documents represented
    assert "table: True" in result.stdout and "table: False" in result.stdout


def test_sample_covers_every_row_when_size_equals_corpus(tmp_path):
    write_chunks(tmp_path, document_only_specs())
    result = run_tool("list_chunks.py", tmp_path, ["--sample", "8"])
    assert result.returncode == 0
    assert "Sample of 8 chunks" in result.stdout
    for spec in document_only_specs():
        count = 0
        for line in result.stdout.splitlines():
            if spec["chunk_id"] in line and line.strip()[:2].rstrip(".") in \
                    ["1", "2", "3", "4", "5", "6", "7", "8"]:
                count = count + 1
        assert count == 1  # each chunk picked exactly once, no repeats


def test_sample_never_exceeds_available_rows(tmp_path):
    write_chunks(tmp_path, sample_specs())
    result = run_tool("list_chunks.py", tmp_path, ["--sample", "10"])
    assert "only 3 chunks exist" in result.stdout
