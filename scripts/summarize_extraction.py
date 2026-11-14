"""Read a pipeline job's durable state. No API requests or spending.

Prints what the job directory actually records: plan identity, submitted
waves, per-chunk checkpoint outcomes, and honest unknowns (cost is never
invented; native batch has no per-request latency).
"""
import argparse
import glob
import json
from pathlib import Path


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Read a pipeline job's recorded state, not an invoice or corpus forecast.")
    parser.add_argument("--job-dir", required=True)
    args = parser.parse_args()
    job = Path(args.job_dir)
    plan_path = job / "plan.json"
    if not plan_path.exists():
        print(f"No plan at {plan_path}. Prepare the job first; do not resubmit anything.")
        return 1
    plan = read_json(plan_path)
    print("Plan:", plan.get("plan_hash", "unknown")[:16], " contract:", plan.get("version"),
          " prompt:", plan.get("prompt_version"), " mode:", plan.get("execution_mode"))
    print("Planned chunks:", len(plan.get("chunks", [])))

    waves = sorted(glob.glob(str(job / "batch-*-claim.json")))
    for claim_path in waves:
        stem = claim_path[:-len("-claim.json")]
        claim = read_json(claim_path)
        state = "claimed, no receipt (reconcile before any resubmission)"
        if Path(stem + "-receipt.json").exists():
            state = "submitted, not collected"
        if Path(stem + "-collected.json").exists():
            collected = read_json(stem + "-collected.json")
            state = "collected at " + str(collected.get("finished_at"))
        print("Wave", Path(stem).name + ":", claim.get("provider"), len(claim.get("items", [])),
              "items -", state)
    print("Native batch waves have no per-request latency; wall time is not inference latency.")

    records = []
    for path in sorted(glob.glob(str(job / "chunk-*.json"))):
        records.append(read_json(path))
    accepted = len([r for r in records if r.get("accepted") is True])
    print("Chunk checkpoints:", len(records), " accepted:", accepted,
          " not accepted:", len(records) - accepted)
    outcomes = {}
    for record in records:
        outcome = str(record.get("terminal_outcome", "unknown"))
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    if outcomes:
        print("Outcomes:", json.dumps(outcomes, sort_keys=True))
    cached = len([r for r in records if r.get("cache_hit") is True])
    if cached:
        print("Cache hits (no provider call):", cached)
    print("Cost: unknown. This tool never computes dollars; reconcile account billing separately.")
    print("Validation acceptance is not factual accuracy. Review sources and label a varied sample.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
