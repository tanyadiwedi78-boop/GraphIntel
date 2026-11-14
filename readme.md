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

## Current prepared Phase 1: final two-provider design

This section supersedes the earlier Phase1 behavior descriptions. Older commands,
raw jobs, claims, results and local stashes remain preserved. No old failed job
is silently migrated, resubmitted or repurposed. New work uses a NEW job folder.
Main remains e81aa5b until separately approved placement; this revision is local.

```
Original SEC chunks + document SHA256 + fixed ontology
 -> immutable plan + retry policy + verified prices + document/total USD limits
 -> Gemini extraction -> validation -> bounded retries/backoff
 -> optional authorized OpenAI only after retry exhaustion
 -> source review -> saved-output diagnostics -> tuned identity resolution
 -> reviewed Neo4j MERGE plan -> separately authorized write/read-back

Independent benchmark:
same chunks / prompt / schema / labels / output ceiling / retry policy / mode
 -> Gemini-only job               -> OpenAI-only job
 -> separate reviewed results    -> paired report (no fallback or cache reuse)
```

The common strict schema includes all requested fields. Extraction NEVER directly
creates graph nodes or writes Cypher. No Qwen or third provider is on this route.
Normal uncached output costs one inference. Default retries=2 means at most3
requests per provider per chunk, up to6 with authorized fallback. --max-calls is
TOTAL attempts across restarts, not a dollar limit. Every failure is retained.

Known-invalid returned JSON/schema/ontology/quote output can retry the SAME
prompt/schema, so the comparison inputs stay identical. Known HTTP429/500/502/
503/504 can retry with exponential backoff2,4,...seconds, bounded at120seconds,
respecting Retry-After. A longer/unparseable Retry-After pauses rather than being
ignored. Only after primary retry exhaustion may OpenAI run with separate paid
scope and its own quotas. Auth/billing/config failures, refusals/incomplete
outputs and unknown transport/POST outcomes DO NOT trigger retries or fallback.
A transport timeout can mean the request was charged; reconcile, never erase its
claim. Known HTTP5xx retries may also cost money, so all are reserved separately.
No automatic LLM-judge or hidden SDK retries. End-to-end quality still needs review.

### Document hash, cost budget and cache contract

Every original chunk must carry a full64hex document_hash OR source_sha256 from
the original document. Both, when present, must agree. Group budgets by that
hash, not accession. No missing source hash is guessed. Exact-result caching
also binds the full chunk, provenance, prompt, schema, ontology, both recipes,
policy and execution mode. Document/source or recipe changes miss the cache.
Checkpoint identity is revalidated on resume. Old jobs are report-only.

Before execution, supply a verified pricing JSON snapshot by provider. No price,
model, billing state, quota or spending authority has been invented for you:

```json
{
  "gemini": {
    "model": "VERIFIED_MODEL_ID", "verified": true,
    "source": "OFFICIAL_PRICING_URL", "account_reference": "VERIFIED_PROJECT",
    "currency": "USD", "mode": "standard", "checked_at": "UTC_ISO_TIMESTAMP",
    "input_per_million_usd": "VERIFIED_HIGHEST_APPLICABLE_RATE",
    "output_per_million_usd": "VERIFIED_HIGHEST_APPLICABLE_RATE",
    "max_input_tokens": 0, "max_output_tokens": 0,
    "free_tier_no_billing_verified": false
  }
}
```

Replace zero model ceilings with verified positive exact-model limits; these are
NON-EXECUTABLE placeholders, not selected defaults. Include the OpenAI entry too
if configured. Use the highest applicable prompt-size pricing tier, including
billed reasoning output. The pricing snapshot expires after7days. Zero rates
require explicit verified free-tier/no-billing state. Batch reservations use
standard rates and never presume discount/free eligibility. Account verification
and original authority references must be real; booleans/CLI flags do not create
permission. Input/output model limits must cover ALL billed text/reasoning; no
media/tools/search/cached-discount pricing is assumed.

Cost-limit JSON for one job:
```json
{"authority_reference":"ORIGINAL_APPROVAL_REFERENCE","total_usd":"APPROVED_USD",
 "documents":{"ORIGINAL_DOCUMENT_SHA256":"APPROVED_DOCUMENT_USD"}}
```

The pre-run maximum reserves full verified model input AND output ceilings for
every allowed retry/fallback attempt. This is intentionally conservative, not a
cheap token estimate. It can reject a small budget even when likely spend would
be less. Durable claims reserve each attempt before network, including unknown
outcomes, and reservation totals survive restarts. It is a LOCAL admission guard,
not a merchant/provider hard cap or a guarantee against changed prices/incorrect
configuration. Actual charges remain separate, reviewed ledger evidence. Do not
relax this guard to make a budget fit. Tighten only after verified provider token
counting and exact-model output semantics are available. No such extra API call
is silently introduced by this version.

### Connected command order (PowerShell, original folder)

Stay in C:\Users\dell\Multi_hop_Graphrag. Preserve .env/data/stashes; don't pop
old edits, reset, delete claims or copy complete files over newer unreviewed work.
Use .\.venv\Scripts\python.exe in this SAME folder. Model IDs, limits, key
access, filenames and reviews must be resolved before live commands. Keys belong
in your existing private environment, never a page/chat/output file.

Offline production preparation:
```
.\.venv\Scripts\python.exe scripts/run_extraction.py phase1 prepare --job data/phase1-final-new --chunks data/pilot-candidates.jsonl --gemini-model VERIFIED_GEMINI_MODEL --openai-model VERIFIED_OPENAI_MODEL --max-output-tokens VERIFIED_LIMIT --retries 2 --pricing data/verified-pricing.json --cost-limits data/approved-production-limits.json --output data/final-plan-summary.json
```
Omit --openai-model to prepare a primary-only job until fallback pricing/access is
ready; that does not remove the selected production architecture. A plan without
prices/limits can be inspected offline but WILL NOT EXECUTE. --max-output-tokens
must be common across comparison providers. No actual model has been selected.

Offline paired preparation, both independent:
```
.\.venv\Scripts\python.exe scripts/run_extraction.py phase1 compare-prepare --job data/paired-final-new --chunks data/pilot-candidates.jsonl --gemini-model VERIFIED_GEMINI_MODEL --openai-model VERIFIED_OPENAI_MODEL --max-output-tokens VERIFIED_LIMIT --retries 2 --pricing data/verified-pricing.json --cost-limits data/approved-paired-limits.json --labels data/reviewed-labels.json --output data/paired-manifest-summary.json
```
Paired cost-limits JSON is {"gemini": SINGLE_JOB_LIMITS, "openai": SINGLE_JOB_LIMITS}.
Labels are frozen before both runs. Retries repeat the same prompt/schema; record
model-specific default reasoning differences rather than pretending internals
are identical. No sampling parameters are secretly translated between APIs.

After separate execution/spending approval ONLY, run each job with its VERIFIED
local limits and key. Example template, not an instruction to spend now:
```
.\.venv\Scripts\python.exe scripts/run_extraction.py phase1 run --job data/paired-final-new/gemini --max-calls APPROVED_TOTAL_ATTEMPTS --rpm VERIFIED_RPM --tpm VERIFIED_TPM --execute-authorized --output data/gemini-run-new.json
.\.venv\Scripts\python.exe scripts/run_extraction.py phase1 run --job data/paired-final-new/openai --max-calls APPROVED_TOTAL_ATTEMPTS --rpm VERIFIED_RPM --tpm VERIFIED_TPM --execute-authorized --output data/openai-run-new.json
```
Production run uses the production job instead. Add --paid-fallback-authorized
ONLY under actual paid scope, plus --fallback-rpm/--fallback-tpm separately.
No fallback flag in either comparison job. Comparison forbids --cache; production
sync jobs may use --cache data/phase1-response-cache. RPM/TPM are local pacing
bounds, not account-wide RPD/quota guarantees. No full-corpus start yet.

Review exact entity/relationship predictions against original text, not only
copied-quote containment. Supply each provider's reviews/actual-charge ledger
under its own key before the paired report:
```
.\.venv\Scripts\python.exe scripts/run_extraction.py phase1 compare --job data/paired-final-new --evidence-reviews data/paired-evidence-reviews.json --cost-ledger data/paired-actual-costs.json --output data/paired-report-new.json
```
Missing results count as recall misses; unreviewed evidence and unknown costs
remain Not measured. First-attempt metrics, each-attempt raw schema rates and
final post-retry results stay distinct. Do not pick a winner from JSON validity
alone or claim a universal best provider from one small sample.

### Native provider batch mode, not packed prompts

Prepare the same production or paired job with --execution-mode native_batch.
Each chunk remains its OWN inference with the same prompt/schema. Gemini uses
bounded inline generateContent Batch API; OpenAI uploads JSONL requests for its
Responses Batch API. This is not Interactions. No SDK, extra worker or queue
service is added. Up to1000requests and local15MiB serialized wave, bounded further
by verified provider request/enqueue allowances. Large groups must be prepared as
smaller immutable jobs; no source is silently truncated.

After actual batch/spending authorization ONLY:
```
.\.venv\Scripts\python.exe scripts/run_extraction.py phase1 batch-submit --job data/JOB --max-calls APPROVED_TOTAL_ATTEMPTS --batch-max-requests VERIFIED_LOCAL_REQUEST_LIMIT --batch-max-input-units VERIFIED_LOCAL_ENQUEUE_TOKEN_ALLOWANCE --execute-authorized --output data/batch-wave1-submit-new.json
.\.venv\Scripts\python.exe scripts/run_extraction.py phase1 batch-collect --job data/JOB --wave 1 --output data/batch-wave1-collect-new.json
```
Collection makes ONE read-only status check, no polling loop or implicit inference
retry. Recheck no more often than hourly or use the provider's completion alert.
If pending, later collect uses a NEW output filename. Once collected, invoke
batch-submit again only for remaining known failures within approved call/dollar
limits. Backoff applies between waves. Fallback uses a separate provider wave
only after primary exhaustion; add paid-fallback scope explicitly. A lost create
receipt or missing item stays unresolved and blocks new submission. If OpenAI
upload succeeded but create failed, reconcile provider resources; no file delete
is automatic. Known-invalid results can retry; refusal/auth/unknown does not.
Native batch reports collection turnaround separately; per-chunk latency and
HTTP429frequency remain unmeasured when the service provides no such timestamps/
HTTPstatuses. Never divide turnaround by chunk count and call it inference time.
To compare latency, use the identical synchronous mode for both providers.

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


### Careful integration of the later extraction guide

The existing 7 entity / 9 relationship SEC ontology is frozen in one config.
The common contract now derives local Pydantic Literal types AND provider JSON
schema from that snapshot. No replacement Person/Organization ontology, SQLite,
Redis, LiteLLM or extra provider was added. evidence is the existing exact-quote
field; relationships already carry source/target/type/source_chunk_id/evidence.
Literal/schema checks limit type names, not hallucinated values or factual truth.
Contract v3 changes require a NEW plan; old jobs remain readable, not runnable.

Default is still 2 retries. Use --retries 1 at NEW preparation if selecting the
pasted guide's one-retry policy; both independent comparison jobs use the same
bound. Known invalid output already triggers retries then separately authorized
fallback. Empty output can be correct even at >150 words. quality_review reports
long_empty_output and minimum_self_reported_confidence without declaring error,
changing acceptance, forcing new facts or spending on another request. There is
no calibrated confidence cutoff yet and no hidden LLM judge call. Semantic quote
support still requires source review, not a claimed 90% detection guarantee.

Reports now include entity_f1, relationship_f1, exact_quote_containment_rate
(raw case-sensitive substring), the existing whitespace-normalized containment,
and source-reviewed evidence_accuracy. F1 uses counts directly, so a completely
missing output scores zero recall/F1 against nonempty reviewed truth even when
precision has no denominator. No labels still means Not measured, not zero.

Optional source-reviewed name equivalences are frozen in each label object:
  "name_groups": [["Acme", "Acme Corp"]]
This applies only on a label marked reviewed:true with the exact source_hash.
Groups cannot overlap, contain duplicates or empty names. They apply only to
entity/endpoint names, never types, relation direction or quotes. No automatic
suffix stripping/fuzzy equivalence/LLM opinion is counted as ground truth.
A judge can suggest adjudication later, under separate approved call budgets,
but cannot silently rewrite these labels or certify its own answer. No judge
has run or been measured. Freeze/adjudicate labels before both provider runs.

Calibrate on reviewed matches AND nonmatches with the installed local embedder.
No separating score means blocked; do not fall back to 0.92. Both raw names and
aliases remain intact. Then test disjoint reviewed identity pairs with the same
model/calibration through the ACTUAL canonical-ID resolver, not just cosine:

.\.venv\Scripts\python.exe scripts/run_extraction.py phase1 identity-eval `
  --job "$newJob" --identity-pairs "$heldoutPairs" `
  --embedding-model-path "$localModel" --calibration "$calibration" `
  --output "$identityEvalOutput"

Use NEW output paths. The four variables are existing selected input/output
paths, not files supplied by this command. Identity-eval makes local CPU calls,
not provider calls or DB requests. Requires calibration pair_hashes, refuses
repeated/reversed calibration/heldout pairs, requires both classes, and reports
canonicalization_accuracy, pair precision/recall/F1 and unresolved coverage.
Pair-hash separation is not proof of unseen entities/documents or corpus-level
cluster accuracy. Keep calibration and held-out entity/document groups separate
when building a real quality benchmark, not just different pair combinations.

The 20 original chunks are a review candidate pool, mostly one issuer and many
financial tables, NOT a balanced certified golden set. The review packet binds
all20 source hashes/text and marks only the2existing assistant-reviewed labels
reviewed:true. The other18 are UNREVIEWED, with null expected lists rather than
invented gold. Before quality claims, complete ideal labels for 15-20 genuinely
representative chunks, including pronouns, defined aliases, nested entities,
negation, potential-vs-current risk, tables and reviewed-empty controls. Add
negative identity pairs for similarly named distinct companies. Reviewer identity
and label scope stay explicit. Do not just mark empty arrays reviewed to finish.
No 10-second/$0.05/runtime/accuracy guarantee is asserted. Exact current model
choices, availability, rates and relative quality must be checked and measured.


### Fresh extraction after your local deletion: no restoration required

This route does NOT import the deleted extraction runner.py, client.py,
validate.py, anthropic_batch.py or batch_storage.py. Its extraction runtime is
contracts.py + providers.py + pipeline.py. The entry point routes phase1 first,
without importing any old module. Optional historical commands are not needed
and are not instructions to restore old files. The expanded benchmark also
avoids the historical validator; only explicit old-format reports can use it.
Do not overwrite a surviving ontology/graph file simply because it appears here.
Full non-extraction dependencies are included for inspection/completeness; match
her actual local Git status before placement, keeping chunks/.env/stashes intact.

Folder tree of this complete route (all ten complete files appear below):
app/
  extraction/
    contracts.py
    providers.py
    pipeline.py
  evaluation/
    extraction_benchmark.py
  graph/
    ontology.py
    construction.py
    resolution.py
    neo4j_store.py
configs/
  ontology.json
scripts/
  run_extraction.py

Existing package __init__.py files can remain; these directories also work as
Python namespace packages in the deletion-simulation test. No old extraction
files are needed for Phase1 help, demo, prepare or the expanded report route.
Core extraction needs Python3.10+ and Pydantic2; live .env loading needs the
existing python-dotenv dependency. Local identity and Neo4j stages separately
need their already-listed sentence-transformers/neo4j packages. No automatic
installation, download, model call or DB connection happens when importing.

AFTER matching/placing the complete prepared route, the first output command is:

.\.venv\Scripts\python.exe scripts/run_extraction.py phase1 demo `
  --output phase1-demo-review-1.json
Get-Content .\phase1-demo-review-1.json

This creates ONE new file in the current original project folder. If that file
exists, pick a new numbered output; do not delete/overwrite it. It runs no API,
uses no key, spends nothing and touches no database. The source is the synthetic
sentence "Acme develops Widget." The returned JSON clearly calls the response a
HAND-WRITTEN fixture, not extraction by an LLM. It shows entities, directed
relation, copied evidence, successful Pydantic/source checks and rejection of
an invented quote. Model-proposed confidence is a fixture number, not measured
accuracy. This verifies local wiring only. Real extraction then uses the saved
SEC source plan, explicitly verified model/account and authorized budget.

These commands are NOT claimed to run on the laptop right now. Tanya deleted
old files; her exact remaining local paths/version are still unverified. The
local Git check is placement safety only and does not block building this route.
The new route was separately tested in a tree containing ONLY these ten files,
with every historical extraction module absent: help, demo and prepare passed.
All stage commands in this guide use the phase1 prefix. Live stages load .env
from the original project root, without overriding existing environment values.
Never paste .env contents or keys into chat or this File.


### Entry-point correction after the partial local edit

Replace the WHOLE scripts/run_extraction.py from the clearly labelled copy button
near First visible output, not only the Anthropic import line. No historical
provider import remains anywhere in this small entry point. With no arguments it
prints safe guidance before importing the pipeline. To actually test local wiring:
.\.venv\Scripts\python.exe scripts/run_extraction.py phase1 --help
No model or DB calls happen. Help can still expose mixed/missing local dependencies;
that is an installation issue to inspect, not permission to restore old providers.
Do not commit/Continue/stage the merge until local files and intended deletions are
reviewed. The verified backup stays untouched. Tests of historical modules now call
those modules explicitly, not the new two-provider-only entry point; none were skipped.
