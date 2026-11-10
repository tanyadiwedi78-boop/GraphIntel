"""Read a completed Anthropic batch summary. No API requests or spending."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Read measured Anthropic batch usage and cost, not an invoice or corpus forecast.")
    parser.add_argument("--job-dir", required=True)
    args = parser.parse_args()
    path = Path(args.job_dir) / "summary.json"
    if not path.exists():
        print("No completed summary. Use collect on the existing job; do not resubmit it.")
        return 1
    summary = json.loads(path.read_text(encoding="utf-8"))
    print("Model:", summary["model"])
    print("Batch:", summary["batch_id"])
    print("Outcomes:", json.dumps(summary["counts"], sort_keys=True))
    print("Reported total input tokens:", summary["reported_total_input_tokens"])
    print("Reported output tokens (includes thinking):", summary["reported_output_tokens"])
    print("Cache reads / writes:", summary["cache_read_input_tokens"], "/", summary["cache_creation_input_tokens"])
    cost = summary["calculated_cost_usd"]
    print("Calculated token cost USD:", "unknown" if cost is None else cost)
    print("Known subtotal USD:", summary["known_calculated_cost_usd"])
    print("Requests with unknown cost:", summary["unknown_cost_count"])
    print("Calculated chain token cost USD:", summary.get("chain_calculated_cost_usd"))
    print("Cost basis:", summary["cost_basis"])
    print("Price source:", summary["price_source"])
    print("Per-document measurements for this batch:")
    for document, totals in summary["documents"].items():
        print(document + ": " + json.dumps(totals, sort_keys=True))
    print("Validator success is not factual accuracy. Review each source and label a varied sample.")
    print("Token-derived costs exclude unknown account adjustments/taxes; reconcile account billing separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
