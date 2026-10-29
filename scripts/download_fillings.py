from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ConfigError, load_settings
from app.corpus.data_limits import PlanError, load_plan
from app.corpus.edgar import EdgarClient, EdgarError, save_filing, select_filings


def main() -> int:
    parser = argparse.ArgumentParser(description="Download SEC filings for the corpus plan.")
    parser.add_argument("--ticker", help="one ticker only, for a quick try")
    parser.add_argument("--forms", nargs="+", help="forms to take, for example: 10-K")
    parser.add_argument("--max-per-form", type=int, default=None,
                        help="at most this many filings per form per company (default: all in range)")
    parser.add_argument("--delay", type=float, default=0.2,
                        help="seconds between requests (default: 0.2, EDGAR allows 10/second)")
    args = parser.parse_args()

    try:
        settings = load_settings()
        plan = load_plan(settings.companies_path)
        user_agent = settings.require_user_agent()
    except (ConfigError, PlanError) as exc:
        print(f"PROBLEM: {exc}")
        return 1

    companies = plan.companies
    if args.ticker:
        matching = []
        for company in companies:
            if company.ticker == args.ticker.upper():
                matching.append(company)
        companies = tuple(matching)
        if not companies:
            print(f"PROBLEM: {args.ticker} missing in configs/companies.json")
            return 1
    if args.forms:
        forms = tuple(args.forms)
    else:
        forms = plan.forms

    saved = 0
    try:
        with EdgarClient(user_agent, delay_seconds=args.delay) as client:
            for company in companies:
                if not company.cik:
                    print(f"skip {company.ticker}: ")
                    continue
                recent = client.recent_submissions(company.cik)
                chosen = select_filings(
                    recent, forms, plan.first_year, plan.last_year,
                    company.cik, company.ticker, per_form_limit=args.max_per_form,
                )
                print(f"{company.ticker}: {len(chosen)} filings match")
                for meta in chosen:
                    path = save_filing(settings.data_dir, meta, client.download(meta))
                    saved += 1
                    print(f"  saved {meta.form} filed {meta.filed} -> {path}")
    except EdgarError as exc:
        print(f"PROBLEM: {exc}")
        return 1

    print(f"Done. {saved} filings saved under {settings.data_dir}/raw/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
