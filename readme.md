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
  first try: 2 means at most 3 attempts per chunk), `timeout_s`, the `fallback_on`
  causes and the circuit breaker. Precedence: a CLI flag wins over the file, the file
  wins over the built-in defaults. Keys and the call cap stay in `.env`; per-minute
  rate limits stay CLI flags. No value lives in two places.
- `--fallback-provider` keeps a run working when the primary provider is down:
  the run continues on the fallback for the rest of the run. Only provider-wide
  problems (rate limit, quota, server or network failure) can fail over, and a
  first short failure never does - a per-minute rate limit waits it out on the
  SAME provider with the normal backoff. The router moves only on a confirmed
  outage: the provider asked for a wait longer than the wait limit (a daily
  quota or equivalent long stop), or it failed provider-wide twice in a row
  (a success resets the count). An answer that is still invalid after ONE repair
  attempt also moves to the fallback (`schema_invalid_after_repair`): the fallback
  faces the same prompt, schema and validation, and the record says why it moved
  (`fallback_reason`). Without a fallback, an invalid chunk keeps repairing on the
  same provider. A paid fallback is never touched by a transient blip. Setup
  problems (key, model id, billing) never fail over: the run stops so you can fix
  them. Each provider passes its own cost and rate-limit gates.
- Provider-wide failure parks, never loses: a chunk that keeps failing transiently
  is parked (its record stays pending) and the run CONTINUES with the next chunk.
  After 5 parked chunks in a row the circuit breaker stops calling for 120 seconds,
  then resumes. Rerunning the same command replays every parked chunk.
- Event log per run: `data/extraction_logs/<run_id>.jsonl` (attempts, tokens, latency,
  per-chunk result, run summary, provider per attempt, failovers, circuit openings).
- The script prints run time, chunks per minute, median and p95 chunk time, and total
  retry wait. `scripts/summarize_extraction.py` adds per-attempt API reliability
  (validation pass, content rejections, transient failures, cut-offs, setup problems),
  attempts per chunk, first-try valid rate, empty answers, and provider routing
  (failovers, attempts per provider, chunk outcomes per model, fallback rate over
  all processed chunks AND over only the chunks with a route-eligible primary
  failure, parked chunks, circuit cooldowns).
- Delivery is at least once: a crash after the model answered but before the save repeats that one call.
- The evidence-quote check proves the quote appears in the chunk. It does not prove the fact is true.
- Results from the old relations-only schema stay on disk but are never reused:
  the schema and prompt versions are part of each result's save key.
- Not done yet: entity resolution, Neo4j writes, precision/recall on a reviewed labeled set, benchmark table.

```
python -m pytest
```
