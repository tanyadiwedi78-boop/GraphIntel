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

### Extraction recovery and measurement limits

New HTTP failures retain only the status, a known error-type enum, a validated
request ID, an optional known spend-limit code and observation time. A failed
submit saves those fields in `submission-error.json` when the disk permits it.
Server error prose, request headers, API keys and prompts are not printed or
saved in that diagnostic. Unknown fields are omitted rather than echoed. The
safe status still prints if writing the diagnostic fails. Existing claims stay
locked, and no HTTP failure triggers another submission. A 400 or 500 diagnostic
alone is not a billing reconciliation. A failed GET is not a new submission.
This cannot reconstruct an HTTP response lost by an older version.

The summary command prints stored provider batch timestamps and elapsed wall
time, including queue/processing. Missing timestamps remain unknown; this is not
per-chunk latency or a p95 benchmark. One collected chunk can verify the pipeline,
but cannot establish quality, cache-hit savings or 1,000-request throughput.

`architecture.md` places global entity resolution and idempotent Neo4j `MERGE`
writes inside Phase 1. They are not implemented by this extraction module.
Source-local alias normalization and saved provenance are input to that work,
not a replacement for embedding-threshold evaluation or graph ingestion.

## Prepared Phase 1 path: Gemini primary, OpenAI fallback

This local revision adds a separate, connected path. The older Anthropic command
and its saved claims remain untouched. It does not reuse an old failed claim as
permission to submit anything again.

```
chunks.jsonl -> immutable plan -> one Gemini request -> strict common contract
                                  |
                     known 429 only, separate authority + remaining call budget
                                  -> optional OpenAI request
validated checkpoint -> benchmark / source review -> identity review -> graph plan
```

Normal work uses one inference per uncached chunk. All later stages reuse that
output, without an LLM judge or repair pass. Local result caching is not provider
prompt caching. Processing many selected chunks locally is not an async provider
Batch API request and does not promise a batch discount or free paid features.

The existing `scripts/run_extraction.py` accepts the new `phase1` subcommand:

```
python scripts/run_extraction.py phase1 prepare --chunks data/chunks.jsonl --job data/phase1-v2 --gemini-model VERIFIED_MODEL_ID --max-output-tokens VERIFIED_OUTPUT_LIMIT --output data/phase1-plan-summary.json
python scripts/run_extraction.py phase1 report --job data/phase1-v2 --output data/phase1-unmeasured-report.json
python scripts/run_extraction.py phase1 resolve --job data/phase1-v2 --output data/phase1-resolution.json
```

`VERIFIED_MODEL_ID` and `VERIFIED_OUTPUT_LIMIT` are placeholders, not chosen
settings. Preparing, reporting and resolving do not contact a provider or DB.
`prepare` snapshots every selected JSONL chunk, the ontology, prompt/schema
versions and explicit provider recipe. It refuses to overwrite an existing plan.

A future authorized run needs `--execute-authorized`, an explicit total-job
`--max-calls`, verified `--rpm` and `--tpm`, and the provider key in the environment.
No live run was used to prepare this revision. Flags record a caller decision;
they do not verify account eligibility or create spending permission. Call count
is not a dollar cap. OpenAI is opt-in at prepare time via `--openai-model`, then
requires separate `--paid-fallback-authorized` at run time. It is never attempted
on an unknown POST outcome, an invalid answer, or an auth/billing/config failure.
Only a known HTTP 429 can trigger it, and only within the shared call budget.

The runner is deliberately sequential. It spaces requests and reserves a
conservative UTF-8-byte input allowance plus the output ceiling in a local
one-minute token window. This is not exact billing-token accounting or an
account-wide quota guarantee. Other processes, daily limits and provider-specific
reasoning/token rules still require checking before a live run. Restarting waits
conservatively rather than bursting. `--cache PATH` enables exact-result reuse.

### Quality and failure criteria

- All required entity and relationship fields use one strict Pydantic schema.
- Ontology, directed endpoints, source IDs and quote containment are checked
  locally. Rejection does not silently prune an answer and call it correct.
- Quote containment is not semantic evidence accuracy. Aliases, descriptions,
  identity and claims remain unreviewed until source-bound review.
- Each request gets an exclusive durable claim before sending and a saved raw
  response/usage/status after it. Crashes and ambiguous outcomes never resubmit
  automatically. A saved returned response can be recovered without a call.
- A stale run lock, incomplete write, corrupt cache, changed plan/ontology/recipe,
  or changed accepted checkpoint fails closed. Do not delete claims to retry.
- Total attempt budget includes failures and fallback, across restarts. Account
  or config failures stop the run. Long `Retry-After` instructions stop fallback
  rather than being truncated into an early request.
- Resume skips completed checkpoints. Cache identity includes original content,
  document/provenance, prompt, schema, ontology and full provider recipe. A cache
  hit is not a newly measured latency or evidence of factual correctness.

### Seven benchmark groups, one set of saved outputs

1. Raw strict JSON schema pass rate, with returned-output coverage.
2. Entity precision/recall on source-hash-bound reviewed labels.
3. Relationship precision/recall on the same reviewed chunks.
4. Semantic evidence accuracy on explicit exact-prediction reviews, separately
   from mechanical quote containment and review coverage.
5. Per-attempt mean/p50/p95 request latency, including failed attempts; per-chunk
   wall time separately includes pacing/fallback. Cache reads are not requests.
6. Cost per 1,000 planned sample chunks only after all attempted charges are
   known. This remains Not measured without reconciled cost, not `$0`.
7. Known HTTP 429 responses divided by all submission attempts. Unknown status
   keeps the result incomplete instead of quietly counting as a non-429.

Use `report --labels PATH --evidence-reviews PATH` for reviewed results. Label
files map chunk IDs to `reviewed`, `source_hash`, `entities`, `relationships`.
`source_hash` is the SHA-256 of the canonical JSON string of original text, via
`app.evaluation.extraction_benchmark.identity`. Evidence review files map chunk
IDs to exact `identity(entity_or_relationship)` keys, each with `reviewed: true`,
`source_hash`, and explicit boolean `supported`. Stale review hashes are rejected.
No labels, cost, quota, latency, or quality numbers are invented when absent.
Offline fixture reports identify themselves as `offline_fixture`, not a live
Gemini/OpenAI benchmark or evidence that quality stayed unchanged.

### Resolution and graph construction stay separate

Resolution assigns source-local mention IDs, emits normalized name/alias match
candidates, and does not merge them automatically. Reviewed merge files are
lists of `{reviewed: true, mention_ids: [...]}`. Unknown, stale, overlapping and
cross-type merge sets are rejected. This conservative implementation can leave
cross-chunk duplicates until reviewed. It is not tuned embedding-based identity
resolution; no arbitrary cosine threshold is presented as an identity proof.

`phase1 graph --job ... --evidence-reviews ... --reviewed-merges ... --output ...`
prepares parameterized Cypher with uniqueness constraints, source-local mentions,
reviewed entity mappings, typed directed edges, and source-bound citation nodes.
Entity reviews cover description/aliases as well as existence; every graph edge
also needs a positive semantic evidence review. Unreviewed claims are blocked.
The command saves a plan only. It does not connect to Neo4j or write anything.
The global embedding resolver and live driver connection are described below.
Actual threshold calibration, rollback/read-back verification on the chosen DB,
and real model benchmark comparison remain before production Phase 1 is verified.

Current API references for the prepared adapters:
- https://ai.google.dev/gemini-api/docs/structured-output?authuser=2
- https://developers.openai.com/api/docs/guides/structured-outputs
- https://developers.openai.com/api/docs/api-reference/responses/create
- https://neo4j.com/docs/cypher-manual/current/clauses/merge/
- https://neo4j.com/docs/cypher-manual/current/syntax/parameters/

### Full global-resolution and Neo4j connection

The source-local `resolve` command remains a clearly marked preview, not the full ER.

1. Install a small local SentenceTransformer model separately. Official examples
   use `sentence-transformers/all-MiniLM-L6-v2`; no model was downloaded or run for
   this build. `LocalEmbedder` requires an existing directory, disables remote
   code/downloads, and fingerprints its files. It is optional until global ER.
2. `phase1 calibrate --identity-pairs PAIRS --embedding-model-path MODEL_DIR ...`
   builds a threshold from reviewed same/different identity examples. Each pair
   includes `left`, `right`, `same_entity`, `reviewed`, and `pair_hash` equal to
   `identity([left, right])`. Keep original source provenance in both entity
   records. Both classes are required; overlapping scores block automatic ER.
   Sample separation is not a promise about unseen entities. Keep held-out
   identity labels for a real quality comparison before rollout.
3. `phase1 registry --neo4j-uri URI --database DB ...` reads the chosen graph's
   canonical nodes and registry version. An existing graph without this registry
   needs migration review, not an assumed-empty local snapshot.
4. `phase1 resolve --registry SNAPSHOT --calibration CALIBRATION
   --embedding-model-path MODEL_DIR ...` embeds local mentions and reuses a
   unique above-threshold canonical match. Type fences remain strict. New IDs
   come from the resolver, never the LLM. Name/alias collisions or multiple
   matches stop for identity review instead of producing another node.
   `--identity-reviews PATH` can supply explicit `{reviewed: true, mention_id,
   entity_id}` decisions. `entity_id: "new"` is an explicit reviewed distinct
   identity, not an automatic escape from an ambiguous match.
5. `phase1 graph --global-resolution RESOLUTION --evidence-reviews REVIEWS ...`
   exports the reviewed global plan, including canonical aliases and typed edges.
6. `phase1 write --global-resolution RESOLUTION --evidence-reviews REVIEWS
   --neo4j-uri URI --database DB --write-authorized ...` is the explicit driver
   connection. Credentials are read from `NEO4J_USERNAME`/`NEO4J_PASSWORD`, never
   written into artifacts. Do not use it until target/access and write authority
   are verified. `--schema-authorized` separately permits required constraints.

The writer checks exact uniqueness-constraint definitions, target identity, and
registry version. It rebuilds Cypher from reviewed records instead of accepting
model-generated queries. An explicit non-retried transaction locks the registry,
rejects a stale snapshot, writes MERGEs, and increments its version. Read-back
checks canonical properties/aliases and typed edge endpoints/IDs. An unknown
commit acknowledgement or read-back mismatch requires reconciliation, not retry.
DDL may remain after a rolled-back data transaction; it is separately authorized.
The same canonical IDs are reused on re-ingestion after a fresh registry read.

These connections are implemented and offline exercised, not live-DB verified.
No real embedding calibration, account quota verification, provider comparison,
Neo4j commit or rollback proof has been manufactured. They are the remaining
real-run checks, not extra architecture phases.

### Reconciled cost without changing raw outputs

`report --cost-ledger PATH` accepts a separate JSON mapping of exact
`identity(saved_attempt_result)` to `{reviewed: true, cost_usd: "0.00",
billing_reference: "actual provider billing record"}`. Include every attempt,
not just successes. Use a zero only when the actual charge is confirmed zero.
Missing, stale or unreconciled entries leave cost Not measured. This ledger is
measurement evidence, not spending permission, and never edits raw checkpoints.

Identity threshold calibration uses same-type matches and nonmatches. Cross-type
pairs are not useful calibration negatives because the resolver filters types
before similarity. The verified package ranges are SentenceTransformers3.4.1
within major3 and Neo4j5.28.8 within major5. CPU-only Torch installation is a
separate environment choice; do not assume a generic install is small or that a
pretrained model has the tiny smoke model's memory footprint.
