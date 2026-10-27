from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

ALLOWED_FORMS = ("10-K", "10-Q", "8-K")
MAX_COMPANIES = 20


class PlanError(ValueError):
    """The corpus plan file is missing or breaks a rule."""
    pass


@dataclass(frozen=True)
class Company:
    name: str
    ticker: str
    cik: Optional[str]  # digits only, or None until looked up


@dataclass(frozen=True)
class CorpusPlan:
    forms: Tuple[str, ...]
    first_year: int
    last_year: int
    companies: Tuple[Company, ...]

    @property
    def years(self) -> int:
        return self.last_year - self.first_year + 1

    def missing_ciks(self) -> Tuple[str, ...]:
        return tuple(c.ticker for c in self.companies if c.cik is None)


def parse_plan(raw: dict) -> CorpusPlan:
    try:
        forms = tuple(raw["forms"])
        first_year, last_year = int(raw["first_year"]), int(raw["last_year"])
        companies_raw = raw["companies"]
    except (KeyError, TypeError, ValueError):
        raise PlanError('plan needs "forms", "first_year", "last_year" and "companies"') from None

    if not forms or any(f not in ALLOWED_FORMS for f in forms):
        raise PlanError(f"forms must be chosen from {ALLOWED_FORMS}")
    if first_year > last_year:
        raise PlanError("first_year must not be after last_year")
    if not 1 <= len(companies_raw) <= MAX_COMPANIES:
        raise PlanError(f"plan needs 1 to {MAX_COMPANIES} companies, got {len(companies_raw)}")

    companies = []
    seen = set()
    for item in companies_raw:
        name = str(item.get("name", "")).strip()
        ticker = str(item.get("ticker", "")).strip().upper()
        cik = item.get("cik")
        if not name or not ticker:
            raise PlanError("every company needs a name and a ticker")
        if ticker in seen:
            raise PlanError(f"ticker {ticker} appears twice")
        seen.add(ticker)
        if cik is not None:
            cik = str(cik).strip()
            if not cik.isdigit():
                raise PlanError(f"cik for {ticker} must be digits only, got {cik!r}")
        companies.append(Company(name, ticker, cik))
    return CorpusPlan(forms, first_year, last_year, tuple(companies))


def load_plan(path: Path) -> CorpusPlan:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PlanError(f"plan file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise PlanError(f"plan file is not valid JSON: {exc}") from None
    return parse_plan(raw)
