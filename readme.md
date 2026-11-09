# GraphIntel - Enterprise Knowledge GraphRAG (SEC filings)

Hybrid knowledge-graph + vector RAG over real SEC EDGAR filings (10-K, 10-Q, 8-K).
The design and phase plan live in [architecture.md](architecture.md).

## Current state

Corpus foundation in progress: SEC EDGAR download plan, filing parser (text,
tables, captions, footnotes), structure-aware chunking, project structure checks.
Extraction into Neo4j, retrieval, answering and evaluation arrive in later phases.

## Setup

```
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env         # then put your name and email in EDGAR_USER_AGENT
```

## Check it works

```
python scripts/check_setup.py
python -m pytest tests/
```

See [IMPROVE.md](IMPROVE.md) for the running list of changes.


## Extraction job (multi-provider: Gemini, Groq, OpenAI)

`app/extraction/` holds the resumable extraction job: one saved result per chunk
and recipe, finished chunks are skipped on the next run, a call cap stops runaway
spend. Done is done, whichever provider did it: a valid result from ANY provider
is accepted when the recipe (chunk text, ontology, schema and prompt versions) is
identical. Acceptance is recomputed, not trusted - the record must carry its full
identity (model, work key, ontology fingerprint, schema and prompt versions,
status valid) and the stored work key must match a fresh computation from those
parts; legacy or incomplete records are never accepted. The event log names the
provider that produced an accepted record (`accepted_from_model`), and the
original provider's file stays untouched, so per-provider provenance is preserved.
To compare providers on the same chunks, give each provider its own results
directory. Every provider uses the same prompt, the same response schema and the
same local validation; saved results and event logs always name the provider and
model that actually produced each answer, so results from different providers
never mix.

The model answers with two lists: `entities` (name and type, including entities
that have no relationship) and `relationships` (relation, source, target, evidence
quote, confidence; source and target must name entities from its own list). The
answer is checked against a strict JSON schema at the API, then again locally
against the ontology, the entity references and the chunk text before it is saved.
Saved results are readable, indented JSON; the chunk and filing provenance is added
by our code, never by the model.

```
python scripts/run_extraction.py --provider gemini --assume-free \
    --max-rpm <from AI Studio> --max-tpm <from AI Studio> --max-chunks 5
python scripts/run_extraction.py --provider gemini --fallback-provider groq \
    --assume-free --max-rpm <from AI Studio> --max-tpm <from AI Studio> --max-chunks 5
```

- Live Gemini or Groq needs the provider's key in the environment, `MAX_EXTRACTION_CALLS`
  above 0 in `.env`, `--assume-free` (your own check in the provider console that this
  account has NO paid plan or billing linked), and your own RPM/TPM limits.
  Cost is reported as unknown, never $0.
- OpenAI (`--provider openai`, or `--fallback-provider openai`) has no free tier: it also
  needs the exact model id (`--openai-model`, or the openai entry in
  `configs/extraction.yaml` - this build invents no model id) and `--openai-allow-paid`.
  That flag records your approval of paid usage; it is not a spending limit and does
  not cap dollars.
- `configs/extraction.yaml` holds the exact model ids, `max_retries` (retries AFTER the
  first try: 2 means at most 3 attempts per chunk ON EACH provider), `timeout_s`, the `fallback_on`
  causes and the circuit breaker. Precedence: a CLI flag wins over the file, the file
  wins over the built-in defaults. Keys and the call cap stay in `.env`; per-minute
  rate limits stay CLI flags. No value lives in two places.
- `--fallback-provider` uses the same prompt, validator and ontology checks for
  every provider. A short first 429 waits on that SAME provider with backoff.
  With `max_retries: 2`, persistent 429/5xx/timeouts use at most 3 calls on the
  primary before fallback; a requested wait beyond the wait limit can route
  sooner. One rejected schema repair also routes. Each configured provider has
  its own retry budget and timeout (2 retries = 3 calls each; at most 6 calls per
  chunk for a two-provider chain). Setup errors (key/model/billing) stop, never
  route to paid usage; blocked-key demonstrations are MOCK tests, not live proof.
- Provider circuits count consecutive eligible failed CALLS per provider, not
  parked chunks. A schema response counts only when its one repair was rejected.
  At 5 failures, only that provider opens for 120 seconds. Healthy fallback keeps
  serving without a global sleep. Later chunks prefer primary while its circuit
  is closed; after cooldown a half-open primary probe tests recovery. Valid
  success resets only that provider's streak; failed probes reopen its circuit.
  If every provider is open, the run waits until the earliest cooldown expires.
- Exhausted transient chunks are parked with reasons and never count toward
  content give-up. The run continues and a rerun replays them; files are kept.
- Cross-provider saved-result reuse is restricted to the CONFIGURED chain's
  exact provider/model/recipe identities, plus recomputed chunk text hash,
  ontology, schema/prompt versions and matching work key/filename. An arbitrary
  valid-shaped model in the results directory is not approved for reuse. A
  single-provider run approves only its own model identity. Use separate data
  directories for Gemini-only, OpenAI-only and fallback comparison runs.
- Every saved relationship includes provider, model and prompt_version. Chunk
  records also include provider_used, attempt_number, fallback_reason,
  primary_error_type, valid_first_try and attempt_records. Each attempt includes
  latency_ms, tokens_in, tokens_out and cost_usd; missing usage/cost stays null.
  The model response schema is unchanged: provenance is added AFTER validation.
- Event log per run: `data/extraction_logs/<run_id>.jsonl` (attempts, tokens, latency,
  per-chunk result, run summary, provider per attempt, failovers, circuit openings).
- The script prints run time, chunks per minute, median and p95 chunk time, and total
  retry wait. `scripts/summarize_extraction.py` adds per-attempt API reliability
  (validation pass, content rejections, transient failures, cut-offs, setup problems),
  attempts per chunk, first-try valid rate, empty answers, and provider routing
  (failovers, attempts per provider, chunk outcomes per model, fallback rate over
  all processed chunks AND over only the chunks with a route-eligible primary
  failure, OpenAI-attempted chunks / ALL setup chunks, parked chunks and provider
  circuit openings). The setup denominator includes skipped/unvisited loaded chunks;
  benchmarks must start with separate clean results directories.
- Delivery is at least once: a crash after the model answered but before the save repeats that one call.
- The evidence-quote check proves the quote appears in the chunk. It does not prove the fact is true.
- Results from the old relations-only schema stay on disk but are never reused:
  the schema and prompt versions are part of each result's save key.
- Not done yet: entity resolution, Neo4j writes, precision/recall on a reviewed labeled set, benchmark table.

```
python -m pytest
```

Throughput labels distinguish new validator-passing results from all processed
chunks (including failed/invalid). Neither proves factual accuracy. A mixed
validation/provider failure keeps rejected_responses and full attempt_records;
parked transient work does not count toward content give-up.
