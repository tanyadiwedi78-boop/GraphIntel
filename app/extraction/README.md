# Extraction run observability

The architecture remains extraction, separate entity resolution, then source-reviewed Neo4j loading. No graph writes happen in the extraction runner.

## Modules

- `providers.py`: bounded synchronous HTTP transport, no hidden SDK retries.
- `pipeline.py`: existing immutable job contract, validation, budgets and CLI dispatch.
- `async_runner.py`: async orchestration around the bounded transport using `asyncio.to_thread`; one request in flight. This is not a parallel worker pool or native async HTTP client.
- `failures.py`: categories and retry/stop decisions.
- `observability.py`: local durable correlated event/span records and tested secret scrubbing. This is not the OpenTelemetry SDK or OTLP export.
- `reporting.py`: evidence aggregation and derived report refresh.
- `dashboard.py`: local HTML presentation, no external resources.
- `recovery.py`: source-reviewed replay preparation into a new immutable job.

## Reports, separate from code

Each invocation of the new CLI `run` path creates a new `reports/<run-id>/` directory:

- `manifest.json`: run/trace identity and source job binding.
- `events.jsonl`: append-only lifecycle events.
- `spans.jsonl`: append-only completed local spans with parents, status and duration.
- `dead-letter.jsonl`: failure category and pointers to saved job evidence, when a failure occurs. It deliberately does not duplicate source text or provider responses.
- `summary.json`: this invocation's operational counters, explicit denominators, token coverage and latency sample statistics.
- `quality.json`: not measured unless source-bound reviewed labels are supplied. Job quality and this invocation's speed have different scopes.
- `dashboard.html`: open in a local browser; refreshes every ten seconds. Derived files refresh after attempts and run termination. No network server or backend is needed.

Job evidence remains in `data/<job>/` or the job path explicitly selected by the user. Old evidence is never moved or overwritten. An absent dead-letter file means no dead-letter entry was written, not proof the run succeeded. A hard process kill may leave an attempt started without a finish. Disk failures can prevent logging; no lossless guarantee is made.

The dashboard's detailed lists show the latest 100 rows, while JSON/JSONL files retain the full run. Raw source/output stays in the immutable job and is not exported by this code. Scrubbing masks configured keys and common key shapes, not every possible secret encoding. Do not share raw job evidence without review.

## Retry and stop policy

- HTTP 429, auth/access denial, refusal/safety blocks and ambiguous transport outcomes stop immediately. No automatic fallback around them.
- Selected known 5xx failures and schema/evidence-invalid returned text can use the immutable job's bounded retries, then explicitly authorized fallback.
- Empty structured results can be valid. Source length is not a truth label.
- Missing/unusable output and provider-declared output truncation stop for review, rather than hiding a cause with a new provider.
- `Retry-After` above the configured bounded wait stops; unparseable values do not permit retries.
- Total call cap includes saved previous claims, retries and fallback. Reservations and pacing are restored across restarts.
- Cancellation waits for an active bounded transport to finish before releasing the job lock. Force-killing the process is different; unknown claims block resubmission.
- The new async path only accepts verified zero-price/no-billing snapshots; OpenRouter requires an explicit `:free` model. This is a local guard, not a substitute for verifying the actual account or a provider-side billing cap.

Concurrency is intentionally one. Higher concurrency is rejected until account-wide quotas and stop behavior are verified. No quota, daily-cap reset, price, or total runtime is inferred from example config files.

## CLI integration

Keep the existing entry point: `python scripts/run_extraction.py phase1 ...`.

- `run` uses the async runner. Existing prepare/report/budget/resolve/graph/write stages remain separate.
- `--reports-root` selects the generated reports root (default `reports`).
- `dashboard --job ... --run-report ... --labels ... --output ...` can refresh derived views with optional source-bound labels. The output path must be new, just like other CLI outputs.
- `replay-prepare --job ... --dead-letter ... --replay-reviews ... --new-job ... --output ...` only prepares a new job, no inference. The target job directory must not already exist.

Replay reviews are a JSON mapping keyed by chunk ID. Each selected entry needs `reviewed: true` and `source_binding` equal to `identity([original_plan_hash, chunk_id, original_source_text])`, using the project's identity function. The command rechecks the saved actual attempt, not just editable dead-letter flags. Systemic/ambiguous failures and unresolved claims cannot pass. Replay retains pricing/cost ceilings but does not create new spending approval; a later run still needs current account, quota and authorization checks.

## What metrics mean

Request latency includes known finished failures. Cache hits and checkpoint skips have separate counts and are excluded from request latency. Missing durations are not zero. Schema validity is a fraction over finished attempts and does not mean factual accuracy. Retry rate counts additional same-provider attempts within this invocation over finished attempts. Cross-invocation retries remain visible in durable attempt numbers, not silently assigned to this run.

Provider-reported usage fields only are counted. Missing token fields stay null, with known subtotals and coverage displayed separately. Provider-specific reasoning-token inclusion may differ. Costs stay unreconciled without billing evidence. Averages or p95 from a small sample do not establish exact full-corpus cost/runtime. Fallback is a selected hard subset, not a fair model comparison. Existing matched-comparison jobs remain independent.

## Validation before a future real run

Run offline tests first. Verify laptop file state before applying replacements. Then verify exact current model/account quotas, free-tier/no-billing state, token ceilings and new-job pricing. Use a small approved pilot followed by a separately approved canary only after review. Do not force live failures with bad credentials or tiny timeouts. No full-corpus execution, hosted backend, telemetry export, model change or graph loading is implied by this implementation.
