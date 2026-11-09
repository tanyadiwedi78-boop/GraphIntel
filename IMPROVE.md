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
  fallbacks behind the same callable contract. Only provider-wide problems are
  route-eligible (429, 408, 5xx, network), and a first short failure never routes: a
  per-minute rate limit waits it out on the SAME provider with the runner's normal
  backoff. The router fails over only on a confirmed outage - the provider asked for a
  wait longer than the wait limit (a daily quota or equivalent long stop), or it raised
  route-eligible errors `failover_after` times in a row (default 2; a success resets the
  count). Setup problems (key, model id, billing - any other 4xx) and content failures
  (cut-off answers) NEVER fail over, so a misconfiguration stops the run before any
  fallback - including a paid one - is touched. Failover is one-way per run; the next
  run starts on the primary again. The failed attempt is re-raised so the runner's
  attempt accounting stays exact.
- `app/extraction/runner.py`: work keys, attempt events and saved records name the ACTIVE
  provider per chunk and per attempt, and the work key is recomputed at save time when a
  failover happened mid-chunk, so a saved record's key always embeds the provider that
  actually produced it and the next run finds it again. New `find_compatible_valid`:
  done is done, whichever provider did it - a valid record from any provider is accepted
  for the same chunk text, ontology, schema and prompt, but only after recomputing its
  identity (complete identity fields required, stored work key must match a fresh
  computation, filename must match); legacy or incomplete records are never accepted.
  The skip event names the producing provider (`accepted_from_model`) and the original
  file stays untouched. Single-client runs are unchanged.
- Behavior change, by design: running the same chunks with a DIFFERENT model no longer
  re-extracts chunks that already have a compatible valid record (done chunks are never
  redone). The old `test_new_model_name_reruns_the_chunk` was replaced by
  `test_new_model_name_accepts_the_compatible_valid_record`. To compare providers on
  the same chunks, give each provider its own results directory.
- `scripts/run_extraction.py`: `--provider openai` with `--openai-model` (no default) and
  `--openai-allow-paid`; `--fallback-provider` with `--fallback-max-rpm/--fallback-max-tpm`.
  Every configured provider passes the same cost and rate-limit gates.
- `scripts/summarize_extraction.py`: new measured sections - API reliability per attempt
  (validation pass, content rejections, transient provider failures, output cut-offs,
  setup problems, cost-gate blocks), attempts per fresh chunk, first-try valid rate,
  empty answers, and provider routing (attempts per provider, failovers, attempts served
  after failover, chunk outcomes per model).
## 2026-11-09 (later) - her fallback spec aligned: repair-then-route, park-and-continue, circuit breaker, config file

Her routing/eval spec (17:53) changed three behaviors and asked for the exact
model ids in a config file. What changed in code, each with tests:
- `schema_invalid_after_repair` is now failover-eligible: an answer that fails
  validation, then fails its ONE repair attempt, moves the chunk to the fallback
  (same prompt, schema and validation there). Without a fallback the chunk keeps
  repairing as before. Cut-off answers (ChunkFailedError) still never route.
  Records and chunk_final events now carry `fallback_reason` and
  `primary_error_type`.
- Transient failures park, never stop the run: the chunk's record stays pending
  (transient, never counts toward give-up) and the run continues. After 5 parked
  chunks in a row the circuit breaker stops calling for 120 s, then resumes
  (`circuit_open` event; both numbers from config). A rerun replays every parked
  chunk. This supersedes the old stop-on-transient behavior; the two old tests
  became park-and-continue tests. Setup problems (401/403/other 4xx) still stop
  the run immediately and never route - her blocked-key failure test stays a
  MOCK test: a real bad key never silently spends on a paid fallback.
- `configs/extraction.yaml` (new): exact model ids (gemini-3.5-flash-lite,
  gpt-6-luna), `max_retries: 2` (retries AFTER the first try = at most 3
  attempts; the runner's `max_attempts_per_chunk` = max_retries + 1),
  `timeout_s: 60` (now actually passed to every client), `fallback_on`,
  `circuit_breaker`. Precedence: CLI flag > this file > built-in default. Keys
  and the call cap stay in `.env`; per-minute limits stay CLI flags - no value
  lives in two places. Unknown fallback causes and bad values are rejected.
- `scripts/summarize_extraction.py`: the fallback rate is printed with BOTH
  denominators - over all processed chunks (her decision-rule shape) and over
  only chunks with a route-eligible primary failure - plus parked-chunk and
  circuit-cooldown totals. Her <=10% criterion is reported, not reinterpreted.
- Tests: 225 pass. New: repair-then-route attempt accounting (1 base + 1 repair
  on the primary, then the fallback's first try with the same base prompt),
  no-fallback keeps repairing, both-providers-blocked parks and replay recovers,
  circuit trips once at 5 parked chunks with exactly one 120 s cooldown,
  max_retries 2 = 3 attempts exactly, transient park-and-continue, config
  supplies the OpenAI model id, unknown fallback cause and missing config file
  rejected, fallback-rate denominators. Still offline only: no live OpenAI or
  fallback call has been made; paid approval and a spend limit are still open.

## 2026-11-09 - multi-provider extraction: OpenAI adapter, fallback routing, reliability metrics

- Tests: `tests/test_openai_client.py`, `tests/test_router.py` (first short 429 stays
  same-provider, repeated failure routes, long requested wait routes at once, success
  resets the count, setup and content failures never route, mid-chunk failover saves
  under the serving provider's key and a rerun on the original primary makes ZERO
  calls), plus accepted-skip tests (cross-provider zero-call skip, legacy record
  without identity not accepted, mismatched work key not accepted), CLI gate and
  summarizer tests. 214 offline tests pass on Linux (fake HTTP servers and fake clients
  only; no network, no real keys). No live OpenAI or fallback call has been made.

## 2026-11-09 (follow-up) - provider-scoped breaker and configured reuse

This supersedes the earlier parked-chunk/global-sleep breaker and one-way-route
notes above. No files or results were deleted; the same extraction flow is extended.
- ProviderRouter now counts eligible failed calls separately per provider, opens
  only the failing provider after 5, skips it for 120 seconds while healthy
  fallback serves, and half-opens primary on cooldown expiry. A valid success
  resets only its own streak; failed probes reopen. New chunks prefer primary
  while closed. A rejected schema repair counts once, not every invalid draft.
  If both circuits are open, only the earliest remaining cooldown is awaited.
- Reuse accepts only the configured chain's exact model identities (including
  provider and recipe tag), then checks all existing fingerprint/version/key
  fields. Single-provider runs do not approve arbitrary other models. Separate
  results directories remain required for provider-comparison benchmarks.
- The CLI reads EACH provider's max_retries and timeout. Two retries means three
  calls on each provider, not three shared calls across the chain; two configured
  providers allow at most six calls/chunk, still under the whole-run call cap.
- Saved relationships now include provider/model/prompt_version AFTER the same
  existing validator. Chunk records preserve attempt records and requested field
  aliases. Existing review/scoring reads named relationship fields and tolerates
  this additive metadata (full suite passes). No entity/ontology semantics changed.
- Summarizer explicitly reports OpenAI-attempted chunks / all loaded setup chunks
  (including skipped/unvisited chunks), alongside the older generic fallback and
  eligible-failure rates. Unknown token counts/costs stay null, never fake zero.
- 240 offline tests pass. Added manual-clock checks for exact expiry, failed
  half-open, provider-local reset, fallback without global pause, both-circuit
  unavailability, 20 parked chunks + replay, configured-model rejection,
  per-provider retry budgets, edge provenance/record aliases and all-setup denominator.
- No live OpenAI or paid fallback call. No live 200-chunk benchmark, reviewed edge
  precision/recall/agreement or full-corpus completion. Paid approval + enforceable
  spend cap, laptop verification and reviewed labels remain required. Existing
  explicit validator is shared across providers; it is not a Pydantic validator.
