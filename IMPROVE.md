# IMPROVE.md

Running list of changes, newest first.

## 2026-11-05 - extraction schema: entities + relationships (main)

- `app/extraction/prompt.py` (p2), `app/extraction/schema.py` (s2),
  `app/extraction/gemini_client.py`: the extraction answer is now
  `{"entities": [...], "relationships": [...]}` instead of relations only.
  Entities with no relationship are kept, so later phases can use mentions.
  Relationships reference entities by name. The validator rejects unknown
  references, duplicate entity names, types outside the ontology, relation and
  type combinations the ontology does not allow, quotes missing from the chunk,
  and unexpected keys (year or amount are not in the ontology, so they are
  rejected, never silently stored).
- `app/extraction/ledger.py`: saved results are indented, readable JSON.
- `app/extraction/runner.py`: the saved record carries the entities list; the
  final chunk event reports `entities` instead of derived endpoint names.
- Old s1/p1 results stay on disk but are never reused: schema and prompt
  versions are part of the work key, so the next run re-extracts under s2/p2.
- Suite after this commit: 117 passed (Linux, Python 3.10), including new
  migration and validation tests. Windows and a live Gemini run are untested.

## 2026-10-31 - failing tests and import bugs (branch real-filing-html-tests, commit 2)

- `app/corpus/chunking.py`, `scripts/build_chunks.py`: fixed stale import
  `app.corpus.filing_parser` -> `app.corpus.filling_parser`. Importing the
  chunking module crashed with ModuleNotFoundError on a fresh clone.
- `tests/test_chunking.py`: was an empty placeholder. Now 12 real tests of the
  implemented chunking contract: packing under the token limit, paragraph
  splits at line boundaries, oversized lines/rows kept whole and flagged,
  table splits between rows with caption + header repeated on every part,
  footnotes travelling with the last part, stable namespaced chunk ids, and
  the config fingerprint changing with any config change.
- `tests/test_offsets.py`: 5 real tests pinning the offset contract: block and
  chunk start/end index the parser's normalized text, a packed chunk's text is
  exactly `normalized[start:end]`, table part offsets cover only their own
  rows (footnotes on the last part), and `offset_basis` is stamped
  `normalized_text` on every chunk.
- `app/structure.py`: the structure checklist now expects `.env.example`
  (committed template) instead of `.env`. `.env` is gitignored by design - it
  holds your private EDGAR identity - so a fresh clone could never pass the
  old check. `scripts/check_setup.py` still validates your real local `.env`.
- Added `.env.example`: documents every setting in `app/config.py`
  (EDGAR_USER_AGENT, DATA_DIR, COMPANIES_PATH, ONTOLOGY_PATH,
  MAX_EXTRACTION_CALLS, LOG_LEVEL).
- Added `readme.md`: setup and test commands.
- Suite after this commit: 42 passed, 0 failed (Linux, Python 3.10.12).

## 2026-10-30 - parser on real SEC filing HTML (branch real-filing-html-tests, commit 1)

- `app/corpus/filling_parser.py` (+31/-3):
  - `to_blocks` removes only the hidden inline-XBRL containers (`ix:header`,
    `ix:hidden`, `ix:references`, `ix:resources`) before extraction. Visible
    `ix:nonfraction` / `ix:nonnumeric` values (the real numbers) are untouched.
  - New `_drop_page_furniture`: removes only the full pipe-separated running
    footer shape `issuer | year Form 10-K | page` (anchored, up to 60 chars).
    Sentences that merely name a form, bare numbers, bullets and repeated
    disclosures are kept.
  - New `_clean`: collapses non-breaking spaces and space runs in paragraph
    and table-cell text.
- `tests/test_real_filing_html.py`: 10 new regression tests on verbatim Apple
  10-K excerpts (`tests/fixtures/aapl_10k_2023_head_excerpt.html`,
  `tests/fixtures/aapl_10k_2023_table_excerpt.html`).
- Measured on full real filings downloaded from SEC EDGAR (2026-10-30/31):
  Apple FY2023 10-K: 23/23 Item sections, 58 tables, 54 standalone footer
  paragraphs removed; Microsoft FY2026 10-K: 23/23 items, 88 tables, 0 removed
  (no footers match the strict shape - its page noise is kept deliberately).

## 2026-11-05 - completing the extraction slice on main

- `app/extraction/runner.py`: restored lines lost in a partial paste (`attempt = 0`, `raw_text`,
  `last_error_text`, the `invalid` outcome, per-attempt events, call timing) and finished the
  timing changes (`chunk_seconds`, `retry_wait_seconds`, `finish()` on every exit path).
- `app/extraction/schema.py`: the ontology check had no body (IndentationError, nothing could import).
  It now reports "not in the ontology" again.
- `scripts/run_extraction.py`: now the complete Gemini + timing version (`--provider echo|gemini`,
  `--gemini-assume-free`, `--gemini-max-rpm`, `--gemini-max-tpm`, timing lines).
- New tests: `tests/test_extraction_events.py`, `tests/test_extraction_timing.py`; restored
  `test_echo_results_are_never_reused_by_a_real_model` in `tests/test_extraction.py`.
- Result: 109 passed on Linux, Python 3.10. Windows run pending. No live Gemini call has been made.

## 2026-11-10 - Fresh Anthropic-only extraction rebuild

- Reconstructed the extraction package, active CLI, source review/sample helpers,
  summary script and independent offline tests without legacy provider imports.
- Fable 5.2 only, Message Batches plus a shared explicit prompt-cache prefix;
  no Gemini/OpenAI/Groq fallback in the active flow.
- Strict JSON output schema plus local ontology/reference/evidence/confidence
  validation. Source-defined Apple alias and false self-ownership regressions,
  local exact duplicate prevention and normalization audit.
- Offline immutable plans, guarded one-time submissions, terminal result import,
  document-content cache, separately approved bounded feedback-repair batches,
  preserved raw failures and honest per-document/chain token-cost summaries.
- The 66 fresh-reconstruction offline tests pass without real credentials or API
  calls. This is not evidence of live accuracy, cache-hit rate, account billing,
  corpus throughput, global entity resolution or a completed graph/vector RAG.
- Paid pilot and full-corpus spending remain unapproved. The reservation is a
  deliberately conservative token-cost bound; it is not expected spend, a cash
  deposit, or an account-wide/tax-inclusive spending guarantee.

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

## 2026-11-13 - Bounded 1,000-request support

- Raise the application batch count limit to 1,000 separate chunk requests,
  subject to stricter explicit byte limits. No packed 1,000-chunk prompt.
- Keep the existing schema, prompt, ontology, evidence checks, normalization,
  deduplication, immutable plans, approval gate, cache and cost accounting.
- Read source/results with bounded lines. Bound saved plan and batch body bytes.
- Read saved raw JSONL twice without a database. Check complete ID coverage
  before writing any extraction, then process one raw answer at a time by ID.
  Keep deterministic summary order and detect changes between/during passes.
- Bound result download size and stop before exhausting free disk. Preserve
  partial downloads, raw results and previous outputs for recovery.
- Add offline bounds, out-of-order coverage, interrupted-import recovery tests
  and a separate-process synthetic memory benchmark. These prove mechanics,
  not extraction quality, real-key access or a Windows 4GB capacity guarantee.
- Paid pilot still blocked pending spending authority/admission policy. No
  estimate-based override or provider call is introduced by this revision.

### 13 November 2026 - explicit one-request actual-charge pilot

- Added mutually exclusive pilot admission, keeping ordinary strict budget
  admission unchanged. Exact plan/chunk approvals, one original request, no
  repairs, shared claim/cache checks, durable no-hard-cap audit fields.
- Kept the saved plan unchanged; the estimate is not a cap. No paid call,
  automatic retry, new job, key change or legacy cleanup in this revision.
- All 260 offline tests passed: the existing 236 plus 24 pilot tests for
  admission scope, altered plans, prior state, cache, timeout and one POST only.

### 13 November 2026 - preserve safe HTTP diagnostics and show native timing

- Confirmed bug: the CLI hid HTTP status/error type/request ID after a failed
  submission. Preserve only fixed known enums and validated IDs with status and
  observation time. Never echo server prose or secret-bearing bodies. Save a
  separate diagnostic without overwriting claims, receipts or old evidence.
- Keep claims locked, no automatic POST retry, no plan/recipe/ontology changes.
  If diagnostic persistence fails, report the safe status and save-failed flag.
  This improves future failures; it cannot recover an older discarded response
  or establish the outcome/billing of the attempted local pilot.
- Print already-stored native batch timing, with unknowns and queue-time scope.
- All 281 offline tests pass: 260 existing plus 21 diagnostic/timing regressions.
  No live provider requests were used for this patch. Tests are not extraction
  benchmarks. Account access, real output review, reviewed-label quality metrics,
  global entity resolution and Neo4j writes remain unverified or unfinished.

## Phase 1 alignment of the prepared Gemini/OpenAI revision

The implementation follows only architecture.md Phase 1:
- Reuses SEC corpus/chunks and the fixed ontology.
- Uses the requested expanded common Pydantic contract and Gemini primary /
  separately authorized OpenAI fallback instead of the older Claude direction.
- Separates extraction, mechanical checks, source review, global resolution and
  Neo4j construction, all from one immutable saved output set.
- Adds local embedding similarity with reviewed threshold calibration, persistent
  canonical IDs, alias storage, ambiguity review and existing-node reuse.
- Connects Neo4j registry read, version-guarded MERGE transaction and read-back.
- Snapshots document-level call-path budgets and unknown cost explicitly; exact
  document/prompt/schema/model cache identity avoids re-extraction. A call limit
  is not a money cap. Actual pricing/account scope must be checked before runs.
- Computes the requested extraction benchmark groups without further LLM calls.

No Phase 2 vectors/pgvector, Phase 3 query router, Phase 4 answer generation or
Phase 5 multi-hop benchmark was added. Offline extraction/identity fixtures are
regression checks, not real extraction-quality, latency or cost results. The
old failed Anthropic claim and all original saved data stay intact.


## 2026-11-13 - final two-provider Phase1 revision, prepared not on main

Reconcile architecture retry/per-document cost/document-hash gaps. Same-template
known-output retries and bounded transient backoff; optional OpenAI after Gemini
retry exhaustion only. Durable model-ceiling USD reservations and explicit
source_sha256 mapping. No unknown-POST resend. Fair paired jobs forbid cache and
fallback, preserve raw-first/final metrics and relation recall. Native async batch
waves retain IDs/claims/missing outputs; no made-up per-chunk batch latency.
Gemini generateContent responseFormat matches current official REST examples.
Existing legacy jobs remain preserved/read-only, separate from the new route.
No provider calls, account changes, database execution or push in this revision.


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
