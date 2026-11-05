"""Run the extraction job over the saved chunks.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ConfigError, load_settings
from app.extraction.events import JsonlEvents
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
        answer = {
            "entities": [
                {"name": "Placeholder Company", "type": "Company"},
                {"name": "Placeholder Technology", "type": "Technology"},
            ],
            "relationships": [{
                "relation": "DEVELOPS",
                "source": "Placeholder Company",
                "target": "Placeholder Technology",
                "evidence_quote": quote,
                "confidence": 0.0,
            }],
        }
        return json.dumps(answer)


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
    parser = argparse.ArgumentParser(description="Run extraction over saved chunks.")
    parser.add_argument("--max-chunks", type=int, default=10, help="how many chunks to read this run (default 10)")
    parser.add_argument("--provider", choices=["echo", "gemini"], default="echo",
                        help="echo = offline placeholder (default), gemini = real model, needs GEMINI_API_KEY in the environment")
    parser.add_argument("--gemini-assume-free", action="store_true",
                        help="you checked Google AI Studio and this project has NO billing account "
                             "(free tier). The code cannot check that for you. Required for any "
                             "live call; paid calls are not implemented in this build.")
    parser.add_argument("--gemini-max-rpm", type=int, default=None,
                        help="requests per minute limit from YOUR AI Studio rate limit page")
    parser.add_argument("--gemini-max-tpm", type=int, default=None,
                        help="input tokens per minute limit from YOUR AI Studio rate limit page")
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
    if args.provider == "gemini":
        if not args.gemini_assume_free:
            print("Live Gemini calls stay blocked until you pass --gemini-assume-free.")
            print("That flag means: you checked Google AI Studio (API keys / Projects page)")
            print("and this project has NO billing account linked, so it stays on the free tier.")
            print("The code cannot check that for you. Paid calls are not implemented in this build.")
            return 1
        if args.gemini_max_rpm is None or args.gemini_max_tpm is None:
            print("Live Gemini calls also need --gemini-max-rpm and --gemini-max-tpm:")
            print("your account's requests-per-minute and input-tokens-per-minute limits,")
            print("copied from the AI Studio rate limit page. Limits are per project and per")
            print("model and can change with your tier, so this build invents no numbers.")
            return 1
        from app.extraction.gemini_client import GeminiClient
        try:
            client = GeminiClient(ontology, assume_free=True,
                                  max_requests_per_minute=args.gemini_max_rpm,
                                  max_tokens_per_minute=args.gemini_max_tpm)
        except ValueError as error:
            print(f"Gemini setup problem: {error}")
            return 1
    else:
        client = EchoClient()
    run_id = uuid.uuid4().hex
    log_path = settings.data_dir / "extraction_logs" / (run_id + ".jsonl")
    events = JsonlEvents(log_path, run_id)
    summary = run_extraction(chunks, client, ontology, client.model_name, str(results_dir), limits,
                             log_event=events)
    events({"event": "run_final", "summary": summary})
    print(f"Event log: {log_path}")

    print(f"Chunks read:           {summary['total_chunks']} (from {chunks_path})")
    print(f"Already done, skipped: {summary['skipped_done']}")
    print(f"New valid results:     {summary['valid']}")
    print(f"Invalid answers:       {summary['invalid']}")
    print(f"Failed chunks:         {summary['failed']}")
    print(f"Model calls used:      {summary['calls_used']}")
    print(f"Run time:              {summary['elapsed_seconds']} s (wall clock: calls, waiting, validation, saving)")
    print(f"Chunks per minute:     {summary['chunks_per_minute']} (fresh chunks only; skipped chunks are not counted)")
    print(f"Chunk time median/p95: {summary['chunk_median_seconds']} s / {summary['chunk_p95_seconds']} s (fresh chunks only)")
    print(f"Retry backoff wait:    {summary['retry_wait_seconds']} s total (provider pacing waits are per attempt in the event log)")
    if summary["stopped_early"]:
        print(f"Stopped early: {summary['stop_reason']}")
    print(f"Results saved in {results_dir}")
    print("Run the same command again: finished chunks are skipped, nothing is processed twice.")
    if args.provider == "gemini":
        print("Cost this run: unknown (" + client.cost_mode() + "). Measured token usage is in the event log.")
        print("Reminder: real model answers still need hand-checking against the source text before you trust them.")
    else:
        print("Reminder: the echo client is a placeholder. It proves save/validate/resume, not extraction quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
