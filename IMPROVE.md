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
