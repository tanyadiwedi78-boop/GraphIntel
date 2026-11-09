"""Run the extraction job over the saved chunks with a real provider API.

Gemini is the default provider; --provider groq or --provider openai uses the
same runner, validator, cache and review tooling with a Groq or OpenAI model
instead. OpenAI has no free tier: it needs --openai-model (this build invents
no model id) and --openai-allow-paid, which records the user's approval of
paid usage; it is not a spending limit and does not cap dollars.

--fallback-provider keeps a run working when the primary provider is down or
rate-limited: provider-wide problems (rate limit, quota, server or network
failure) fail over, and an answer that is still invalid after one repair
attempt also moves to the fallback, which faces the same prompt, schema and
validation. Setup problems (key, model id, billing) never fail over: the run
stops so you can fix them. Saved results and event logs always name the
provider that produced each answer.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ConfigError, load_settings
from app.extraction.client import GeminiClient
from app.extraction.groq_client import GroqClient
from app.extraction.openai_client import OpenAIClient
from app.extraction.router import ProviderRouter
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


PROVIDER_NAMES = {"gemini": "Gemini", "groq": "Groq", "openai": "OpenAI"}

DEFAULT_CONFIG_PATH = "configs/extraction.yaml"

KNOWN_FALLBACK_CAUSES = ["rate_limit", "server_error", "timeout", "schema_invalid_after_repair"]


def load_extraction_config(config_arg):
    """Read the extraction config yaml. Returns (config_dict, error_text).

    The default path may be absent (built-in defaults then apply); an
    explicitly passed --config path must exist.
    """
    config_path = Path(config_arg)
    if not config_path.exists():
        if config_arg != DEFAULT_CONFIG_PATH:
            return None, f"Config file not found: {config_path}"
        return {}, None
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        return None, f"Config problem in {config_path}: {error}"
    if raw is None:
        return {}, None
    if not isinstance(raw, dict):
        return None, f"Config problem in {config_path}: the top level must be a mapping."
    return raw, None


def provider_config_entries(extraction_config):
    """The providers list as {name: entry}; entries without a name are ignored."""
    entries = {}
    for entry in extraction_config.get("providers") or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if isinstance(name, str) and name.strip() != "":
            entries[name.strip()] = entry
    return entries


def configured_model(provider_entries, provider):
    """The exact model id for one provider from the config file, or None."""
    entry = provider_entries.get(provider) or {}
    model = entry.get("model")
    if isinstance(model, str) and model.strip() != "":
        return model.strip()
    return None


def configured_positive_number(provider_entries, provider, extraction_config, key, default):
    """A per-provider numeric setting (timeout_s), falling back to the section
    value, then the default. Returns (value, error_text)."""
    entry = provider_entries.get(provider) or {}
    value = entry.get(key, extraction_config.get(key, default))
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None, f"Config problem: {key} for {provider} must be a positive number, got {value!r}."
    return float(value), None


def configured_max_retries(provider_entries, provider, extraction_config):
    """Retries AFTER the first try for one provider (default 2, so at
    most 3 attempts per chunk). Returns (value, error_text)."""
    entry = provider_entries.get(provider) or {}
    value = entry.get("max_retries", extraction_config.get("max_retries", 2))
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None, f"Config problem: max_retries for {provider} must be a whole number 0 or more, got {value!r}."
    return value, None


def gate_provider(provider: str, args, max_rpm, max_tpm) -> bool:
    """Print why this provider may not be called and return False, or return True.

    Every configured provider (primary or fallback) passes the same gates:
    its own cost approval and explicit per-minute rate limits.
    """
    if provider == "groq":
        provider_name = "Groq"
        limits_page = "the Groq console limits page"
        billing_check = "you checked the Groq console (Billing / Plans page)"
    elif provider == "openai":
        provider_name = "OpenAI"
        limits_page = "your organization's limits page in the OpenAI platform"
        billing_check = ""
    else:
        provider_name = "Gemini"
        limits_page = "the AI Studio rate limit page"
        billing_check = "you checked Google AI Studio (API keys / Projects page)"

    if provider == "openai":
        if args.openai_model is None or args.openai_model.strip() == "":
            print("--provider openai needs the exact model id: --openai-model on the command line")
            print("or an openai model entry in configs/extraction.yaml. Copy the id from the OpenAI")
            print("platform docs - for example the model line from a script that already runs for")
            print("you. This build invents no model id.")
            return False
        if not args.openai_allow_paid:
            print("Live OpenAI calls stay blocked until you pass --openai-allow-paid")
            print("That flag means: you approved paid usage and set a spending limit in the")
            print("OpenAI platform. The flag records your approval; it does not cap dollars.")
            print("OpenAI has no free tier, so --assume-free does not apply to OpenAI.")
            return False
    elif not args.gemini_assume_free:
        print(f"Live {provider_name} calls stay blocked until you pass --assume-free")
        print("(the older --gemini-assume-free name also works).")
        print(f"That flag means: {billing_check}")
        print("and this account has NO paid plan or billing linked, so it stays on the free tier.")
        print("The code cannot check that for you; the flag is an attestation, not billing")
        print("protection. Paid calls are not implemented in this build.")
        return False
    if max_rpm is None or max_tpm is None:
        print(f"Live {provider_name} calls also need --max-rpm and --max-tpm")
        print("(older names --gemini-max-rpm and --gemini-max-tpm also work):")
        print("your account's requests-per-minute and input-tokens-per-minute limits,")
        print(f"copied from {limits_page}. Limits are per account and per")
        print("model and can change with your tier, so this build invents no numbers.")
        return False
    return True


def build_client(provider: str, args, ontology, max_rpm, max_tpm, timeout_seconds):
    """Construct one provider's client after its gates passed."""
    if provider == "groq":
        return GroqClient(ontology, model_id=args.groq_model, assume_free=True,
                          max_requests_per_minute=max_rpm, max_tokens_per_minute=max_tpm,
                          timeout_seconds=timeout_seconds)
    if provider == "openai":
        return OpenAIClient(ontology, model_id=args.openai_model, allow_paid=True,
                            max_requests_per_minute=max_rpm, max_tokens_per_minute=max_tpm,
                            timeout_seconds=timeout_seconds)
    return GeminiClient(ontology, model_id=args.model, assume_free=True,
                        max_requests_per_minute=max_rpm, max_tokens_per_minute=max_tpm,
                        timeout_seconds=timeout_seconds)


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
    parser.add_argument("--model", default=None,
                        help="Gemini model id (default: configs/extraction.yaml, else gemini-3.8-flash)")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "groq", "openai"],
                        help="which provider to call (default: gemini)")
    parser.add_argument("--groq-model", default=None, dest="groq_model",
                        help="Groq model id, used with --provider groq "
                             "(default: configs/extraction.yaml, else openai/gpt-oss-20b)")
    parser.add_argument("--openai-model", default=None, dest="openai_model",
                        help="OpenAI model id, required with --provider openai unless configs/extraction.yaml "
                             "gives one. No built-in default: copy the exact id from the OpenAI platform "
                             "docs or from a script that already runs for you. This build invents no model id.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                        help=f"extraction settings file (default: {DEFAULT_CONFIG_PATH}). CLI flags win "
                             "over this file; the file wins over built-in defaults.")
    parser.add_argument("--openai-allow-paid", dest="openai_allow_paid", action="store_true",
                        help="you approved paid OpenAI usage and set a spending limit in the OpenAI "
                             "platform. The flag records your approval; it does not cap dollars. "
                             "Required for any live OpenAI call; OpenAI has no free tier.")
    parser.add_argument("--fallback-provider", default=None, choices=["gemini", "groq", "openai"],
                        help="provider the run continues on when the primary hits a provider-wide "
                             "problem (rate limit, quota, server or network failure). Setup problems "
                             "(key, model id, billing) never fail over: the run stops so you can fix "
                             "them. Each configured provider passes its own cost and rate limit gates; "
                             "an OpenAI fallback stays blocked until --openai-allow-paid and "
                             "--openai-model are given.")
    parser.add_argument("--fallback-max-rpm", type=int, default=None,
                        help="requests per minute limit for the fallback provider "
                             "(default: the --max-rpm value)")
    parser.add_argument("--fallback-max-tpm", type=int, default=None,
                        help="input tokens per minute limit for the fallback provider "
                             "(default: the --max-tpm value)")
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

    config, config_error = load_extraction_config(args.config)
    if config_error is not None:
        print(config_error)
        return 1
    extraction_config = config.get("extraction") or {}
    if not isinstance(extraction_config, dict):
        print("Config problem: the 'extraction' section must be a mapping.")
        return 1
    provider_entries = provider_config_entries(extraction_config)
    fallback_on = extraction_config.get("fallback_on")
    if fallback_on is None:
        fallback_on = list(KNOWN_FALLBACK_CAUSES)
    for cause in fallback_on:
        if cause not in KNOWN_FALLBACK_CAUSES:
            print(f"Config problem: fallback_on cause {cause!r} is not one of {KNOWN_FALLBACK_CAUSES}.")
            return 1

    # Model ids: a CLI flag wins over the config file, which wins over the
    # built-in default. The exact OpenAI id comes from one of those two
    # explicit sources only; this build invents no model id.
    if args.model is None:
        args.model = configured_model(provider_entries, "gemini") or "gemini-3.8-flash"
    if args.groq_model is None:
        args.groq_model = configured_model(provider_entries, "groq") or "openai/gpt-oss-20b"
    if args.openai_model is None:
        args.openai_model = configured_model(provider_entries, "openai")

    max_retries, retries_error = configured_max_retries(provider_entries, args.provider, extraction_config)
    if retries_error is not None:
        print(retries_error)
        return 1
    circuit = extraction_config.get("circuit_breaker") or {}
    circuit_failures = circuit.get("consecutive_failures", 5)
    circuit_cooldown = circuit.get("cooldown_s", 120)
    if not isinstance(circuit_failures, int) or isinstance(circuit_failures, bool) or circuit_failures < 1:
        print(f"Config problem: circuit_breaker.consecutive_failures must be a whole number 1 or more, got {circuit_failures!r}.")
        return 1
    if isinstance(circuit_cooldown, bool) or not isinstance(circuit_cooldown, (int, float)) or circuit_cooldown <= 0:
        print(f"Config problem: circuit_breaker.cooldown_s must be a positive number, got {circuit_cooldown!r}.")
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
                       max_attempts_per_chunk=max_retries + 1,
                       give_up_after_runs=args.give_up_after_runs,
                       retry_given_up=args.retry_given_up,
                       circuit_consecutive_failures=circuit_failures,
                       circuit_cooldown_seconds=float(circuit_cooldown))
    print(f"Settings: at most {max_retries + 1} attempts per chunk ({max_retries} retries), "
          f"circuit breaker {circuit_failures} consecutive eligible failed calls per provider / {circuit_cooldown}s cooldown, "
          f"fallback on {fallback_on} (from {args.config}).")

    providers = [args.provider]
    if args.fallback_provider is not None:
        if args.fallback_provider == args.provider:
            print("--fallback-provider must name a different provider than --provider.")
            return 1
        providers.append(args.fallback_provider)

    clients = []
    provider_budgets = {}
    position = 0
    for provider in providers:
        if position == 0:
            max_rpm = args.gemini_max_rpm
            max_tpm = args.gemini_max_tpm
        else:
            max_rpm = args.fallback_max_rpm
            if max_rpm is None:
                max_rpm = args.gemini_max_rpm
            max_tpm = args.fallback_max_tpm
            if max_tpm is None:
                max_tpm = args.gemini_max_tpm
        if not gate_provider(provider, args, max_rpm, max_tpm):
            return 1
        timeout_seconds, timeout_error = configured_positive_number(
            provider_entries, provider, extraction_config, "timeout_s", 60)
        if timeout_error is not None:
            print(timeout_error)
            return 1
        try:
            one_client = build_client(provider, args, ontology, max_rpm, max_tpm, timeout_seconds)
        except ValueError as error:
            print(f"{PROVIDER_NAMES[provider]} setup problem: {error}")
            return 1
        retries, retries_error = configured_max_retries(provider_entries, provider, extraction_config)
        if retries_error is not None:
            print(retries_error)
            return 1
        provider_budgets[one_client.model_name] = retries + 1
        clients.append(one_client)
        position = position + 1

    if len(clients) == 1:
        client = clients[0]
    else:
        client = ProviderRouter(clients, failover_after=max_retries + 1,
                                consecutive_failures=circuit_failures,
                                cooldown_seconds=float(circuit_cooldown),
                                max_attempts=provider_budgets, fallback_on=fallback_on)
        primary_name = PROVIDER_NAMES[providers[0]]
        fallback_name = PROVIDER_NAMES[providers[1]]
        print(f"Fallback route: if {primary_name} hits a provider-wide problem (rate limit, quota,")
        print(f"server or network failure) or stays invalid after one repair, the run continues on")
        print(f"{fallback_name}, with the same prompt, schema and validation.")
        print("Primary is retried on later chunks unless its circuit is open; after cooldown it is probed again.")
        print("Setup problems (key, model id, billing) never fail over: the run stops")
        print("so you can fix them. Saved results and event logs always name the provider that")
        print("actually produced each answer.")

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
    print(f"Valid chunks/minute:   {summary['valid_chunks_per_minute']} (new validator-passing results, not factual accuracy)")
    print(f"Processed chunks/min:  {summary['chunks_per_minute']} (includes invalid/failed chunks; skips excluded)")
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
