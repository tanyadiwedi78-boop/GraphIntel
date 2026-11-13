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

## 2026-11-10 - Honest pilot measurement follow-up

- Labels now require an explicit reviewed=true after source review. Correctly
  empty chunks are scored, so a false extraction on boilerplate is a false
  positive. Existing labels are not silently approved. Keep supported canonical
  names consistent in labels; only case and spaces are ignored, not fuzzy aliases.
- Scoring shows reviewed/unreviewed, missing, invalid/failed and incompatible
  coverage. End-to-end recall includes expected items on unavailable outputs as
  missed. Precision describes accepted predictions, not the success rate of the
  whole pipeline. Reviewed-empty chunks remain in accepted-output coverage.
- Known input/output/cache fields survive incomplete billing usage. Summaries
  name missing fields, show known subtotals and retain unknown calculated cost.
- chain_documents includes per-document usage/cost across repair rounds, including
  documents completed before the current round. Missing historical per-document
  ancestry is marked incomplete and does not become an invented complete cost.
- Plans include request-specific offline planning_estimate values with exact
  request byte sizes, a stated four-UTF-8-bytes/token heuristic, selected cache
  write pricing, full capped output and no assumed hits. This is uncalibrated
  against Fable and is NOT a tokenizer count, hard bound or submit authority.
  The strict admission reservation still applies. The low-budget pilot gate is
  NOT solved: do not approve more spending merely to bypass it. Anthropic's own
  token counter is an estimate without a documented numeric error bound.
- Current official rates were checked on 10 November 2026. No token-count or
  inference request was made to implement this change; no paid run was approved.
- Replaced tests/test_model.py's import-time Gemini model-list HTTP probe with
  actual offline model-selection tests. The earlier 189-test result did not prove
  network isolation. Tests now block external DNS/socket connections before
  collection and in Python subprocesses; localhost fake transports remain allowed.

Official policy sources:
https://platform.claude.com/docs/en/build-with-claude/token-counting
https://platform.claude.com/docs/en/about-claude/pricing
https://platform.claude.com/docs/en/build-with-claude/batch-processing
https://platform.claude.com/docs/en/build-with-claude/prompt-caching

## 1,000-request batch support - 13 November 2026

This revision keeps one model request and one source ID per chunk. It supports
up to 1,000 requests in ONE batch submission, not 1,000 chunks in one prompt.
Every request has its own token charge. The default remains 10 chunks.

Local limits protect memory: 16 MiB serialized batch body, 32 MiB saved plan,
1 MiB source JSONL line, 4 MiB result JSONL line, 256 MiB downloaded results,
and 1,000,000 source chunks. These are deliberately below the provider limits.
A large job is refused, never silently truncated into an incomplete batch.
Use fewer/smaller chunks when a byte limit is reached. Source files are retained.

Raw results download to a uniquely named partial file, then become a complete
JSONL only on successful download. Collection reads this saved JSONL twice.
The first pass checks every ID and terminal outcome, holding only seen IDs and
a digest in RAM. The second pass validates and saves one answer at a time using
its custom_id to find the source, never its row position. Summary metadata is
put back into plan order. There is no SQLite/database or extra raw-result copy.
Raw file state and content digest checks detect changes during collection.
Do not edit raw results while collecting. If import is interrupted, run collect
on that same unchanged job; never resubmit it.
A partial raw download is kept for diagnosis and is not treated as complete.

This does not change spending authority. The conservative strict reservation
remains in force; the offline estimate cannot authorize a low-budget submission.
No live extraction, factual-quality result or actual cache saving is claimed.

Offline checks:
```
python -m pytest -q
python scripts/benchmark_batch_memory.py --chunk-chars 2000 --answer-padding-chars 1000 --memory-limit-mib 512
python scripts/benchmark_batch_memory.py --chunk-chars 9000 --answer-padding-chars 100000 --memory-limit-mib 512
```
The benchmark needs Linux/macOS resource support; the optional memory limit is
Linux-only. It uses temporary synthetic files and blocks external networking.
It is not a Windows laptop or real-provider benchmark. The runtime code uses
only standard-library additions and retains the existing project dependencies.

### One original request with accepted actual charge - 13 November 2026

The normal `submit --budget-usd` path still requires the conservative strict
reservation. A separate `--single-request-actual-charge` option is for an
explicitly approved single original request only. It does NOT enforce a dollar
maximum. An estimate is not a cap or proof of authorization.

This option needs `--allow-paid`, the exact saved `--approve-plan`, and
`--approve-chunk-id` matching that plan's one source chunk. It cannot be combined
with `--budget-usd`, multiple requests or repair plans. Keep the original plan
unchanged. Current cache and job/global submission claims are checked before
network. Any prior claim or submission/output state blocks this pilot; reconcile
rather than deleting claims or retrying. The durable claim records the admission
mode and absence of a hard USD cap. No automatic retries, repairs or fallback.

This feature does not grant standing spending permission. A person's approval
must cover the exact request and actual-charge risk before using it. A timeout
may already have submitted the request: preserve the job and check the Console,
never repeat submit. `collect --job-dir ...` only follows the existing batch.
The answer, validation outcome, reported tokens, batch timing and usage-based
calculated cost can then be reviewed; one answer is not an accuracy benchmark.
Offline check: 260 passed, including 24 new one-request admission tests. No paid
request was sent during implementation or testing.
