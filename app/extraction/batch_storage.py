"""Bounded batch-file readers. Two JSONL passes, no database or provider calls."""
import json
import hashlib
import tempfile
from contextlib import contextmanager
from pathlib import Path

MAX_REQUESTS = 1000
MAX_BATCH_BYTES = 16 * 1024 * 1024
MAX_PLAN_BYTES = 32 * 1024 * 1024
MAX_SOURCE_LINE_BYTES = 1024 * 1024
MAX_RESULT_LINE_BYTES = 4 * 1024 * 1024
MAX_SOURCE_CHUNKS = 1_000_000
MAX_RESULTS_BYTES = 256 * 1024 * 1024


def json_lines(path, line_limit, total_limit=None):
    """Read at most one bounded UTF-8 line at a time, including blank lines."""
    total = 0
    with Path(path).open("rb") as handle:
        while True:
            line = handle.readline(line_limit + 1)
            if not line:
                return
            total += len(line)
            if total_limit is not None and total > total_limit:
                raise ValueError("JSONL file exceeds local byte limit")
            if len(line) > line_limit:
                raise ValueError("JSONL line exceeds local byte limit")
            if line.strip():
                yield json.loads(line.decode("utf-8"))


def read_plan(path):
    """Reject oversized plans before loading or decoding the whole file."""
    with Path(path).open("rb") as handle:
        data = handle.read(MAX_PLAN_BYTES + 1)
    if len(data) > MAX_PLAN_BYTES:
        raise ValueError("plan exceeds local byte limit")
    return json.loads(data.decode("utf-8"))


def file_state(path):
    """Detect replacement or ordinary edits between and during the two passes."""
    state = Path(path).stat()
    return (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns)


def check_result(row, expected_ids, seen):
    custom_id = row.get("custom_id")
    if not isinstance(custom_id, str) or custom_id not in expected_ids or custom_id in seen:
        raise ValueError("unknown or duplicate batch result custom_id")
    if row.get("result", {}).get("type") not in ("succeeded", "errored", "canceled", "expired"):
        raise ValueError("unknown batch result type")
    seen.add(custom_id)


def result_hash_row(digest, row):
    text = json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
    digest.update(text.encode("utf-8") + b"\n")


def checked_file_rows(path, expected_ids):
    """First check every ID; then reopen and yield one answer at a time.

    Only IDs and a digest stay in RAM. The saved raw JSONL is not copied.
    Raw files must not be edited during collection. Ordinary edits/replacements
    stop collection; the final digest also prevents a changed file's summary.
    """
    before = file_state(path)
    seen = set()
    first_hash = hashlib.sha256()
    for row in json_lines(path, MAX_RESULT_LINE_BYTES, MAX_RESULTS_BYTES):
        check_result(row, expected_ids, seen)
        result_hash_row(first_hash, row)
    if len(seen) != len(expected_ids):
        raise ValueError("incomplete batch results; no success summary written")
    if file_state(path) != before:
        raise ValueError("raw results changed during collection; preserve the job")
    seen = set()
    second_hash = hashlib.sha256()
    for row in json_lines(path, MAX_RESULT_LINE_BYTES, MAX_RESULTS_BYTES):
        if file_state(path) != before:
            raise ValueError("raw results changed during collection; preserve the job")
        check_result(row, expected_ids, seen)
        result_hash_row(second_hash, row)
        yield row
    if file_state(path) != before or second_hash.digest() != first_hash.digest():
        raise ValueError("raw results changed during collection; no summary written")


@contextmanager
def checked_results(rows, expected_ids):
    """Production passes a saved path. Offline fixtures may pass a one-shot iterator.

    Fixture rows are first written to a bounded temporary JSONL, never kept in a
    list. Real collection reads its original raw file directly without this copy.
    """
    if isinstance(rows, (str, Path)):
        yield checked_file_rows(Path(rows), expected_ids)
        return
    with tempfile.TemporaryDirectory(prefix="graphintel-fixture-") as folder:
        path = Path(folder) / "results.jsonl"
        total = 0
        with path.open("wb") as handle:
            for row in rows:
                body = (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
                total += len(body)
                if total > MAX_RESULTS_BYTES:
                    raise ValueError("results exceed local byte limit")
                if len(body) > MAX_RESULT_LINE_BYTES:
                    raise ValueError("result exceeds local byte limit")
                handle.write(body)
        yield checked_file_rows(path, expected_ids)
