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
- Not done yet: entity resolution, Neo4j writes, precision/recall on a labeled set, benchmark table.

```
python -m pytest
```
