"""Run the extraction job over the saved chunks with a real provider API.

Gemini is the default provider; --provider groq uses the same runner,
validator, cache and review tooling with a Groq model instead.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ConfigError, load_settings
from app.extraction.client import GeminiClient
from app.extraction.groq_client import GroqClient
from app.extraction.runner import JsonlEvents, RunLimits, run_extraction
from app.graph.ontology import load_ontology


def load_chunks(chunks_path: Path, max_chunks: int, wanted_ids=None) -> list:
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            chunk = json.loads(line)
            if wanted_ids is not None:
                if chunk.get("chunk_id") not in wanted_ids:
                    continue
            chunks.append(chunk)
            if wanted_ids is None and len(chunks) >= max_chunks:
                break
    return chunks


def limit_documents(chunks: list, max_documents: int) -> list:
    """Keep only the chunks of the first `max_documents` distinct documents.

    Budget the extraction cost per document before running the full corpus:
    the default scope is one document, so widening the scope is always an
    explicit choice made after reading measured per-document totals.
    """
    seen = []
    kept = []
    for chunk in chunks:
        accession = chunk.get("accession")
        if accession not in seen:
            if len(seen) >= max_documents:
                break
            seen.append(accession)
        kept.append(chunk)
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description="Run extraction over saved chunks with the Gemini API.")
    parser.add_argument("--max-chunks", type=int, default=10, help="how many chunks to read this run (default 10)")
    parser.add_argument("--chunk-ids", default=None,
                        help="comma-separated chunk ids to extract instead of the first --max-chunks chunks")
    parser.add_argument("--model", default="gemini-3.8-flash", help="Gemini model id (default: gemini-3.8-flash)")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "groq"],
                        help="which provider to call (default: gemini)")
    parser.add_argument("--groq-model", default="openai/gpt-oss-20b", dest="groq_model",
                        help="Groq model id, used with --provider groq (default: openai/gpt-oss-20b)")
    parser.add_argument("--gemini-assume-free", "--assume-free", dest="gemini_assume_free",
                        action="store_true",
                        help="you checked the provider console and this account has NO paid plan "
                             "or billing linked (free tier). The code cannot check that for you; "
                             "the flag is an attestation, not billing protection. Required for any "
                             "live call; paid calls are not implemented in this build.")
    parser.add_argument("--gemini-max-rpm", "--max-rpm", dest="gemini_max_rpm",
                        type=int, default=None,
                        help="requests per minute limit from YOUR provider's rate limit page")
    parser.add_argument("--gemini-max-tpm", "--max-tpm", dest="gemini_max_tpm",
                        type=int, default=None,
                        help="input tokens per minute limit from YOUR provider's rate limit page")
    parser.add_argument("--max-documents", type=int, default=1,
                        help="how many documents this run may touch (default 1). A scope limiter, "
                             "not a cost budget: the default keeps the first runs a measured "
                             "one-document pilot. Read the per-document totals from "
                             "summarize_extraction.py, then rerun with a bigger number.")
    parser.add_argument("--give-up-after-runs", type=int, default=3,
                        help="skip a chunk after this many separate runs where its own content "
                             "failed or stayed invalid (default 3). Rate-limit, quota and network "
                             "failures never count toward this.")
    parser.add_argument("--retry-given-up", action="store_true",
                        help="include given-up chunks again this run; their saved records stay on disk")
    args = parser.parse_args()

    if args.max_documents < 1:
        print("--max-documents must be at least 1.")
        return 1

    try:
        settings = load_settings()
    except ConfigError as error:
        print(f"Settings problem: {error}")
        return 1

    chunks_path = settings.data_dir / "chunks" / "chunks.jsonl"
    if not chunks_path.exists():
        print(f"No chunks at {chunks_path}. Run scripts/build_chunks.py first.")
        return 1

    wanted_ids = None
    if args.chunk_ids is not None:
        wanted_ids = []
        for piece in args.chunk_ids.split(","):
            piece = piece.strip()
            if piece != "":
                wanted_ids.append(piece)
        if len(wanted_ids) == 0:
            print("--chunk-ids was given but it has no ids in it.")
            return 1

    chunks = load_chunks(chunks_path, args.max_chunks, wanted_ids)
    all_documents = []
    if wanted_ids is None:
        for chunk in chunks:
            accession = chunk.get("accession")
            if accession not in all_documents:
                all_documents.append(accession)
        chunks = limit_documents(chunks, args.max_documents)
    scoped_documents = []
    for chunk in chunks:
        accession = chunk.get("accession")
        if accession not in scoped_documents:
            scoped_documents.append(accession)
    if len(chunks) == 0 and wanted_ids is None:
        print(f"{chunks_path} has no chunks.")
        return 1
    if wanted_ids is not None:
        found_ids = []
        for chunk in chunks:
            found_ids.append(chunk["chunk_id"])
        missing = []
        for wanted in wanted_ids:
            if wanted not in found_ids:
                missing.append(wanted)
        if len(missing) > 0:
            print(f"These chunk ids are not in {chunks_path}:")
            for wanted in missing:
                print("  " + wanted)
            return 1

    ontology = load_ontology(settings.ontology_path)

    if settings.max_extraction_calls <= 0:
        print("MAX_EXTRACTION_CALLS is 0, so no calls are allowed.")
        print("Set it to a positive whole number in .env for this run, for example 10 for a first pilot.")
        print("0 is the safe default: nothing can be called until you choose a cap.")
        return 1
    limits = RunLimits(max_calls=settings.max_extraction_calls,
                       give_up_after_runs=args.give_up_after_runs,
                       retry_given_up=args.retry_given_up)

    if args.provider == "groq":
        provider_name = "Groq"
        limits_page = "the Groq console limits page"
        billing_check = "you checked the Groq console (Billing / Plans page)"
    else:
        provider_name = "Gemini"
        limits_page = "the AI Studio rate limit page"
        billing_check = "you checked Google AI Studio (API keys / Projects page)"
    if not args.gemini_assume_free:
        print(f"Live {provider_name} calls stay blocked until you pass --assume-free")
        print("(the older --gemini-assume-free name also works).")
        print(f"That flag means: {billing_check}")
        print("and this account has NO paid plan or billing linked, so it stays on the free tier.")
        print("The code cannot check that for you; the flag is an attestation, not billing")
        print("protection. Paid calls are not implemented in this build.")
        return 1
    if args.gemini_max_rpm is None or args.gemini_max_tpm is None:
        print(f"Live {provider_name} calls also need --max-rpm and --max-tpm")
        print("(older names --gemini-max-rpm and --gemini-max-tpm also work):")
        print("your account's requests-per-minute and input-tokens-per-minute limits,")
        print(f"copied from {limits_page}. Limits are per account and per")
        print("model and can change with your tier, so this build invents no numbers.")
        return 1

    if args.provider == "groq":
        try:
            client = GroqClient(ontology, model_id=args.groq_model, assume_free=True,
                                max_requests_per_minute=args.gemini_max_rpm,
                                max_tokens_per_minute=args.gemini_max_tpm)
        except ValueError as error:
            print(f"Groq setup problem: {error}")
            return 1
    else:
        try:
            client = GeminiClient(ontology, model_id=args.model, assume_free=True,
                                  max_requests_per_minute=args.gemini_max_rpm,
                                  max_tokens_per_minute=args.gemini_max_tpm)
        except ValueError as error:
            print(f"Gemini setup problem: {error}")
            return 1

    results_dir = settings.data_dir / "extraction"
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
    print(f"Gave up on:            {summary['gave_up']} (own-content failures in 3 separate runs; rerun with --retry-given-up to include them again - records stay on disk)")
    print(f"Model calls used:      {summary['calls_used']}")
    print(f"Run time:              {summary['elapsed_seconds']} s (wall clock: calls, waiting, validation, saving)")
    print(f"Chunks per minute:     {summary['chunks_per_minute']} (fresh chunks only; skipped chunks are not counted)")
    print(f"Chunk time median/p95: {summary['chunk_median_seconds']} s / {summary['chunk_p95_seconds']} s (fresh chunks only)")
    print(f"Retry backoff wait:    {summary['retry_wait_seconds']} s total (provider pacing waits are per attempt in the event log)")
    if summary["stopped_early"]:
        print(f"Stopped early: {summary['stop_reason']}")
    print(f"Results saved in {results_dir}")
    if wanted_ids is None:
        print(f"Documents this run: {len(scoped_documents)} of {len(all_documents)} in the loaded chunks "
              f"(scope limiter: --max-documents {args.max_documents}).")
        if len(scoped_documents) < len(all_documents):
            print("Pilot scope: read the measured per-document totals from summarize_extraction.py,")
            print("then rerun with --max-documents and a bigger number to widen the scope.")
    unfinished = summary["total_chunks"] - summary["skipped_done"] - summary["valid"]
    if unfinished > 0:
        print(f"NOT fully extracted: {unfinished} of {summary['total_chunks']} loaded chunks are still pending, invalid, failed or given up.")
        print("Transient failures (rate limit, quota, network) never count toward give-up and stay pending for a later run.")
    print("Run the same command again: finished chunks are skipped, given-up chunks are left alone, nothing is processed twice.")
    print("Cost this run: unknown (" + client.cost_mode() + "). Measured token usage is in the event log.")
    print("Reminder: real model answers still need hand-checking against the source text before you trust them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
