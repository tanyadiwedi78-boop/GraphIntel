"""Offline synthetic scale check. No key, API, GPU or corpus data is used.

Run from the repository root. This measures THIS process on Linux/macOS, not
Windows laptop capacity. --memory-limit-mib adds a Linux address-space limit.
Temporary synthetic input, raw answers and results live outside your data folder.
"""
import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import network_guard
network_guard.install()
from app.extraction.anthropic_batch import (MODEL, prepare_plan, write_new,
                                           collect_job, canonical)
from app.graph.ontology import load_ontology


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-chars", type=int, default=2000)
    parser.add_argument("--answer-padding-chars", type=int, default=1000)
    parser.add_argument("--memory-limit-mib", type=int, default=0)
    args = parser.parse_args()
    try:
        import resource
    except ImportError:
        print("This measurement needs Python resource support (Linux/macOS). No API was called.")
        return 1
    if args.memory_limit_mib:
        if platform.system() != "Linux":
            print("The optional address-space limit is supported here only on Linux.")
            return 1
        limit = args.memory_limit_mib * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="graphintel-benchmark-") as folder:
        root = Path(folder)
        source = root / "chunks.jsonl"
        with source.open("w", encoding="utf-8") as handle:
            for number in range(1000):
                chunk = {"chunk_id": "chunk-" + str(number),
                         "text": "Acme develops Widget Engine. " + "a" * args.chunk_chars,
                         "accession": "filing-" + str(number // 10),
                         "form": "10-K", "period": "2026"}
                handle.write(json.dumps(chunk) + "\n")
        ontology = load_ontology(Path(__file__).resolve().parent.parent / "configs" / "ontology.json")
        try:
            plan = prepare_plan(source, ontology, 1000)
        except ValueError as error:
            print(json.dumps({"outcome": "safely_refused", "reason": str(error),
                              "chunk_chars": args.chunk_chars,
                              "elapsed_seconds": round(time.perf_counter() - started, 3)}))
            return 0
        job = root / "job"
        job.mkdir()
        write_new(job / "plan.json", plan)
        write_new(job / "receipt.json", {"id": "msgbatch_offline"})
        raw = job / "provider-results.jsonl"
        with raw.open("w", encoding="utf-8") as handle:
            for custom_id in reversed(list(plan["chunks"])):
                row = {"custom_id": custom_id, "result": {"type": "succeeded", "message": {
                    "id": "msg_offline", "model": MODEL, "stop_reason": "end_turn",
                    "content": [{"type": "thinking", "thinking": "x" * args.answer_padding_chars},
                                {"type": "text", "text": '{"entities":[],"relationships":[]}'}],
                    "usage": {"input_tokens": 100, "output_tokens": 20,
                              "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}}}
                handle.write(json.dumps(row) + "\n")
        batch_bytes = len(canonical({"requests": plan["requests"]}).encode("utf-8"))
        del plan
        class OfflineStatus:
            def request(self, method, path, body=None):
                if method != "GET":
                    raise AssertionError("benchmark must never submit")
                return json.dumps({"id": "msgbatch_offline", "processing_status": "ended"})
        summary = collect_job(job, OfflineStatus())
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mib = peak / (1024 * 1024 if platform.system() == "Darwin" else 1024)
        print(json.dumps({"outcome": "completed_offline", "synthetic_requests": 1000,
                          "chunk_chars": args.chunk_chars,
                          "answer_padding_chars": args.answer_padding_chars,
                          "source_bytes": source.stat().st_size,
                          "batch_bytes": batch_bytes,
                          "plan_bytes": (job / "plan.json").stat().st_size,
                          "raw_result_bytes": raw.stat().st_size,
                          "saved_results": len(list((job / "extractions").glob("*.json"))),
                          "counts": summary["counts"],
                          "peak_rss_mib": round(peak_mib, 2),
                          "address_space_limit_mib": args.memory_limit_mib,
                          "elapsed_seconds": round(time.perf_counter() - started, 3),
                          "platform": platform.platform(),
                          "caveat": "synthetic offline Linux/macOS measurement, not real model quality or a Windows 4GB guarantee"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
