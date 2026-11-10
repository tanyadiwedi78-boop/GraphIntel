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


## Extraction job (Gemini adapter + timing)

`app/extraction/` holds the resumable extraction job: one saved result per chunk,
finished chunks are skipped on the next run, a call cap stops runaway spend.

The model answers with two lists: `entities` (name and type, including entities
that have no relationship) and `relationships` (relation, source, target, evidence
quote, confidence; source and target must name entities from its own list). The
answer is checked against a strict JSON schema at the API, then again locally
against the ontology, the entity references and the chunk text before it is saved.
Saved results are readable, indented JSON; the chunk and filing provenance is added
by our code, never by the model.

```
python scripts/run_extraction.py --max-chunks 5                 # offline echo placeholder
python scripts/run_extraction.py --provider gemini --gemini-assume-free \
    --gemini-max-rpm <from AI Studio> --gemini-max-tpm <from AI Studio> --max-chunks 5
```

- `echo` is the default: no model, no key, no cost. It proves save/validate/resume, not extraction quality.
- Live Gemini needs `GEMINI_API_KEY` in the environment, `MAX_EXTRACTION_CALLS` above 0 in `.env`,
  `--gemini-assume-free` (your own check in Google AI Studio that this project has NO billing account),
  and your own RPM/TPM limits. Paid calls are not implemented. Cost is reported as unknown, never $0.
- Event log per run: `data/extraction_logs/<run_id>.jsonl` (attempts, tokens, latency, per-chunk result, run summary).
- The script prints run time, chunks per minute, median and p95 chunk time, and total retry wait.
- Delivery is at least once: a crash after the model answered but before the save repeats that one call.
- The evidence-quote check proves the quote appears in the chunk. It does not prove the fact is true.
- Results from the old relations-only schema stay on disk but are never reused:
  the schema and prompt versions are part of each result's save key.
- Not done yet: entity resolution, Neo4j writes, precision/recall on a labeled set, benchmark table.

```
python -m pytest
```

## Anthropic-only extraction rebuild - 10 November 2026

The active extraction command is now scripts/run_extraction.py with prepare,
submit, collect and prepare-retry. It uses only claude-fable-5-2 through Message
Batches with an explicit shared-prefix prompt cache. Earlier multi-provider
instructions above are historical, not this command's runtime configuration.
configs/extraction.yaml is a reference, not a source of paid authority.

The generated JSON schema uses ontology enums and closed objects. Local strict
validation still checks confidence, endpoint types and verbatim evidence.
Source-defined Apple collective aliases and exact duplicate entities/edges are
normalized with an audit; distinct companies are not fuzzy-merged. Cross-document
entity resolution, embedding thresholds and Neo4j MERGE remain separate work.

Prepare is offline. Submit needs a reviewed immutable plan ID, an approved USD
job budget, positive MAX_EXTRACTION_CALLS and ANTHROPIC_API_KEY in local .env.
There is no fallback or automatic paid retry. Prepare-retry makes an offline
feedback plan, at most two repair rounds, for separately approved submission.
Do not blindly repeat submit after a timeout; preserve its claim and reconcile
with the Console. Collect resumes the existing batch and keeps raw results.

Accepted results are cached by available document-chunk hash plus ontology,
source metadata, schema, prompt and model recipe. Old results are retained,
not rewritten. Each microbatch is at most 100 uncached chunks; repeated prepare
advances past accepted cached chunks. API cache hits are best-effort and require
at least 512 shared-prefix tokens for this model. Usage records distinguish
uncached input, cache writes/reads and output including thinking.

Costs are calculated from reported usage at documented batch/cache rates, not
invoiced account totals. Missing usage stays unknown. Strict admission uses an
intentionally conservative no-hit reservation, not an expected charge. Prices,
account adjustments/taxes, model access and actual source quality need checking
before any paid pilot or full corpus run. No live-model quality claim is made.

Offline rebuild check: python -m pytest tests/test_anthropic_batch.py -q

### Main integration and test migration

The existing main branch still contained synchronous Gemini-runner tests. They
are migrated to the Anthropic batch contract, not excluded from collection.
Shared validation and standalone historical adapter tests remain active. All
189 current tests pass offline with no skipped or hidden legacy failures.

Batch claims, receipts, raw results and immutable provider status snapshots
replace synchronous attempt callbacks. A timed-out POST is never automatically
repeated. Repairs require a separately reviewed plan and spending approval.
Provider created_at to ended_at gives batch elapsed wall time, including queue
time. Missing native times remain unknown. It does not measure individual chunk
inference latency, median/p95 latency or synchronous retry/backoff duration.

This does not authorize a real-key pilot. Keep existing local changes backed up
and use a separate clean checkout rather than resetting a modified worktree.
