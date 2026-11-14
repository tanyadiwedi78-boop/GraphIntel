# Legacy Anthropic-only tests (evidence, not collected)

These files are the ORIGINAL versions of the retired Anthropic-era tests,
copied verbatim from `main` before the suite was migrated to the current
Gemini/OpenAI pipeline. They are NOT collected by pytest (pytest only looks
in `tests/`) and they cannot run: the modules they import
(`app.extraction.runner`, `app.extraction.client`, `app.extraction.anthropic_batch`,
`app.extraction.batch_storage`, `app.extraction.validate`, `app.extraction.safe_http_failure`,
`app.extraction.prepare_retry`, `app.extraction.pilot`, `app.extraction.planning_estimate`)
were deleted from the repo by the project owner.

They are kept as evidence of what the old suite asserted, so no safety
behavior disappears silently. Disposition (keep as evidence / delete relying
on git history / port selected items deeper) is an open decision for the
project owner; nothing here has been deleted by the migration.

## Groups with NO current equivalent (retired-vendor-only behavior)

A. Anthropic POST-HOC usage/cost arithmetic (~10 tests, in
   test_anthropic_batch.py and test_extraction_measurement.py): computing
   realized dollar cost from Anthropic usage fields - token-cost stacks,
   thinking-token accounting, cache read/write token costs, repair-chain USD
   totals. This is retired-vendor accounting, not the cost guard. The current
   architecture's cost guard is PRE-EXECUTION conservative admission:
   verified exact-model pricing snapshots, per-document-hash and total USD
   budgets with an authority reference, and durable dollar reservations that
   survive restarts. That admission path EXISTS and is actively tested in
   tests/test_phase1_pipeline.py (document budgets block before claims,
   prices cannot be omitted, zero price needs a verified free account, stale
   pricing is blocked, reservations cannot be reset by restart, fallback
   needs its own budget and authority). Realized per-request spend stays
   honestly unknown (cost_usd is None, never zero) until reconciled against
   account billing - unknown realized spend is deliberate, not a missing
   guard.

B. prepare_retry repair loop (3 tests, test_anthropic_batch.py:
   test_repair_plan_only_invalid_and_same_cached_prefix and neighbors): the
   current architecture has no repair stage by design; invalid output is
   refused, never auto-repaired (see tests/test_single_request_pilot.py::
   test_repairs_are_not_a_stage for the current negative guarantee).

C. BatchAPI streaming downloads (3-4 tests, test_batch_scale.py):
   download limits, partial-file handling, low-disk behavior for the retired
   Anthropic result-download flow. The current native batch reads provider
   responses through JSONProvider.batch_collect; there is no file-download
   flow.

D. batch_storage bounded readers / two-pass import (~6 tests,
   test_batch_scale.py): line/file byte caps, changed-file-between-passes
   detection, single-use generators. That storage module no longer exists;
   current jobs keep one JSON file per claim/receipt/record.

E. safe_http_failure diagnostics + pilot CLI (5-6 tests,
   test_http_diagnostics.py): diagnostic-file rendering, whitelisted fields,
   failed-diagnostic-write handling. The current JSONProvider raises
   CallFailure carrying only kind/status/retry_after; the surviving safety
   intent is tested in tests/test_http_diagnostics.py.

F. source_aliases / normalize_response parsing (3-4 tests,
   test_anthropic_batch.py): the old normalization module is gone; the
   current contract validates the strict Pydantic schema directly and alias
   intent is covered in tests/test_anthropic_batch.py (alias section) and
   tests/test_extraction.py.

G. pilot registry / admission flags (3-4 tests, test_single_request_pilot.py):
   the Anthropic-claims registry and its CLI admission modes do not exist in
   the current pipeline; the surviving single-flight intent is covered in
   tests/test_single_request_pilot.py.

H. planning_estimate (3 tests, test_extraction_measurement.py): pre-flight
   USD estimate documents and tamper checks for the retired planner. The
   current pipeline reserves conservative token windows, not dollars, and
   requires a verified pricing snapshot before any run.

I. Socket-level network guard: NOT retired. The current suite keeps an
   active guard in tests/network_guard.py, installed before collection by
   tests/conftest.py and inherited by Python child processes via an injected
   sitecustomize (loopback fake servers remain allowed; proxy env vars are
   stripped). Active proof lives in tests/test_http_diagnostics.py::
   test_suite_network_guard_blocks_external_sockets and
   test_suite_network_guard_reaches_python_subprocesses. The two old tests
   are preserved here as history only because their module retired.
