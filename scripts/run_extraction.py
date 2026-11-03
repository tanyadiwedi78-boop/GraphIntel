"""Run the extraction job over the saved chunks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ConfigError, load_settings
from app.extraction.runner import RunLimits, run_extraction
from app.graph.ontology import load_ontology


class EchoClient:

    model_name = "echo-placeholder"

    def __call__(self, prompt: str) -> str:
        marker = "TEXT:\n"
        if marker in prompt:
            text = prompt.split(marker, 1)[1]
        else:
            text = ""
        words = text.split()
        quote = " ".join(words[:8])
        if quote == "":
            quote = "placeholder quote"
        relation = {
            "relation": "DEVELOPS",
            "source_name": "Placeholder Company",
            "source_type": "Company",
            "target_name": "Placeholder Technology",
            "target_type": "Technology",
            "evidence_quote": quote,
            "confidence": 0.0,
        }
        return json.dumps({"relations": [relation]})


def load_chunks(chunks_path: Path, max_chunks: int) -> list:
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            chunks.append(json.loads(line))
            if len(chunks) >= max_chunks:
                break
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Run extraction over saved chunks (offline echo client for now).")
    parser.add_argument("--max-chunks", type=int, default=10, help="how many chunks to read this run (default 10)")
    args = parser.parse_args()

    try:
        settings = load_settings()
    except ConfigError as error:
        print(f"Settings problem: {error}")
        return 1

    chunks_path = settings.data_dir / "chunks" / "chunks.jsonl"
    if not chunks_path.exists():
        print(f"No chunks at {chunks_path}. Run scripts/build_chunks.py first.")
        return 1

    chunks = load_chunks(chunks_path, args.max_chunks)
    if len(chunks) == 0:
        print(f"{chunks_path} has no chunks.")
        return 1

    ontology = load_ontology(settings.ontology_path)

    if settings.max_extraction_calls <= 0:
        print("MAX_EXTRACTION_CALLS is 0, so no calls are allowed.")
        print("Set it to a positive whole number in .env for this run, for example 20 for a first offline try.")
        print("0 is the safe default: nothing can be called until you choose a cap.")
        return 1
    limits = RunLimits(max_calls=settings.max_extraction_calls)

    results_dir = settings.data_dir / "extraction"
    client = EchoClient()
    summary = run_extraction(chunks, client, ontology, client.model_name, str(results_dir), limits)

    print(f"Chunks read:           {summary['total_chunks']} (from {chunks_path})")
    print(f"Already done, skipped: {summary['skipped_done']}")
    print(f"New valid results:     {summary['valid']}")
    print(f"Invalid answers:       {summary['invalid']}")
    print(f"Failed chunks:         {summary['failed']}")
    print(f"Model calls used:      {summary['calls_used']}")
    if summary["stopped_early"]:
        print(f"Stopped early: {summary['stop_reason']}")
    print(f"Results saved in {results_dir}")
    print("Run the same command again: finished chunks are skipped, nothing is processed twice.")
    print("Reminder: the echo client is a placeholder. It proves save/validate/resume, not extraction quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
