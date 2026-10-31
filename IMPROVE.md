# IMPROVE.md

Running list of changes, newest first.

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
