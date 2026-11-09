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

## 2026-11-09 - multi-provider extraction: OpenAI adapter, fallback routing, reliability metrics

- `app/extraction/openai_client.py` (new): OpenAI chat-completions client with the same
  contract as the Gemini and Groq clients (same prompt, schema, validator, cache; its own
  work keys so results never mix). The model id is REQUIRED - this build invents no model
  id. No free tier, so no assume_free mode: every call needs `allow_paid=True` plus explicit
  per-minute limits, and the gate runs before any request is built. The flag records the
  user's approval of paid usage; it is not a spending limit and does not cap dollars.
  `reasoning_effort` is sent only when configured - no per-model defaults.
- `app/extraction/router.py` (new): `ProviderRouter` wraps a primary client and optional
  fallbacks behind the same callable contract. Only provider-wide problems fail over
  (429, 408, 5xx, network); setup problems (key, model id, billing - any other 4xx) and
  content failures (cut-off answers) NEVER fail over, so a misconfiguration stops the run
  before any fallback - including a paid one - is touched. Failover is one-way per run.
  The failed attempt is re-raised so the runner's attempt accounting stays exact.
- `app/extraction/runner.py`: work keys, attempt events and saved records name the ACTIVE
  provider per chunk and per attempt, so provenance survives a mid-run failover.
  Single-client runs are unchanged.
- `scripts/run_extraction.py`: `--provider openai` with `--openai-model` (no default) and
  `--openai-allow-paid`; `--fallback-provider` with `--fallback-max-rpm/--fallback-max-tpm`.
  Every configured provider passes the same cost and rate-limit gates.
- `scripts/summarize_extraction.py`: new measured sections - API reliability per attempt
  (validation pass, content rejections, transient provider failures, output cut-offs,
  setup problems, cost-gate blocks), attempts per fresh chunk, first-try valid rate,
  empty answers, and provider routing (attempts per provider, failovers, attempts served
  after failover, chunk outcomes per model).
- Tests: `tests/test_openai_client.py`, `tests/test_router.py`, plus CLI gate and
  summarizer tests. 208 offline tests pass on Linux (fake HTTP servers and fake clients
  only; no network, no real keys). No live OpenAI or fallback call has been made.
