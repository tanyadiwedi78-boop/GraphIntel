#Download filings from SEC EDGAR.
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import httpx

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_number}/{accession_flat}/{document}"
DEFAULT_DELAY_SECONDS = 0.2  # stays far under the 10 requests/second limit


class EdgarError(RuntimeError):
    """EDGAR answered with an error or something unexpected."""
    pass


@dataclass(frozen=True)
class FilingMeta:
    """One filing chosen from a company's submissions list."""
    cik: str
    ticker: str
    form: str
    accession: str          #  0001045810-25-000023
    filed: str              # filing date, YYYY-MM-DD
    period: str             # reporting period, YYYY-MM-DD
    primary_document: str   

    @property
    def accession_flat(self) -> str:
        return self.accession.replace("-", "")

    @property
    def url(self) -> str:
        return ARCHIVES_URL.format(
            cik_number=int(self.cik),
            accession_flat=self.accession_flat,
            document=self.primary_document,
        )

    @property
    def suffix(self) -> str:
        return Path(self.primary_document).suffix.lower() or ".txt"


def select_filings(
    recent: dict,
    forms: Sequence[str],
    first_year: int,
    last_year: int,
    cik: str,
    ticker: str,
    per_form_limit: Optional[int] = None,
) -> Tuple[FilingMeta, ...]:
    """Pick filings from the 'filings.recent' 
    """
    try:
        accession_numbers = recent["accessionNumber"]
        form_names = recent["form"]
        filing_dates = recent["filingDate"]
        report_dates = recent["reportDate"]
        primary_documents = recent["primaryDocument"]
    except KeyError as exc:
        raise EdgarError(f"submissions file is missing the field {exc}") from None

    wanted = set(forms)
    chosen = []
    per_form_count = {}
    for accession, form, filed, period, document in zip(
        accession_numbers, form_names, filing_dates, report_dates, primary_documents
    ):
        if form not in wanted:
            continue
        year = int(filed[:4])
        if not first_year <= year <= last_year:
            continue
        if per_form_limit is not None and per_form_count.get(form, 0) >= per_form_limit:
            continue
        per_form_count[form] = per_form_count.get(form, 0) + 1
        chosen.append(FilingMeta(cik, ticker, form, accession, filed, period, document))
    return tuple(chosen)


class EdgarClient:
    """Small EDGAR downloader with a polite delay between requests."""

    def __init__(self, user_agent: str, delay_seconds: float = DEFAULT_DELAY_SECONDS):
        if "@" not in user_agent:
            raise EdgarError("EDGAR_USER_AGENT must contain your name and email")
        self._delay = delay_seconds
        self._client = httpx.Client(
            headers={"User-Agent": user_agent},
            follow_redirects=True,
            timeout=30.0,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "EdgarClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _get(self, url: str) -> httpx.Response:
        time.sleep(self._delay)
        response = self._client.get(url)
        if response.status_code != 200:
            raise EdgarError(f"EDGAR answered {response.status_code} for {url}")
        return response

    def recent_submissions(self, cik: str) -> dict:
        """Return the 'filings.recent' mapping for one company."""
        url = SUBMISSIONS_URL.format(cik=cik.zfill(10))
        data = self._get(url).json()
        try:
            return data["filings"]["recent"]
        except KeyError:
            raise EdgarError(f"no 'filings.recent' {cik}") from None

    def download(self, meta: FilingMeta) -> str:
        """Download one filing's primary document as text."""
        return self._get(meta.url).text


def save_filing(data_dir: Path, meta: FilingMeta, text: str) -> Path:
    folder = Path(data_dir) / "raw" / meta.ticker
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{meta.accession_flat}{meta.suffix}"
    path.write_text(text, encoding="utf-8")
    sidecar = folder / f"{meta.accession_flat}.meta.json"
    sidecar.write_text(json.dumps({
        "cik": meta.cik,
        "ticker": meta.ticker,
        "form": meta.form,
        "accession": meta.accession,
        "filed": meta.filed,
        "period": meta.period,
        "primary_document": meta.primary_document,
        "url": meta.url,
    }, indent=2), encoding="utf-8")
    return path


def load_saved_filings(data_dir: Path) -> Tuple[Tuple[Path, FilingMeta], ...]:
    raw_dir = Path(data_dir) / "raw"
    if not raw_dir.exists():
        return ()
    found = []
    for sidecar in sorted(raw_dir.rglob("*.meta.json")):
        info = json.loads(sidecar.read_text(encoding="utf-8"))
        meta = FilingMeta(
            cik=info["cik"],
            ticker=info["ticker"],
            form=info["form"],
            accession=info["accession"],
            filed=info["filed"],
            period=info["period"],
            primary_document=info["primary_document"],
        )
        body = sidecar.with_name(sidecar.name.replace(".meta.json", meta.suffix))
        if body.exists():
            found.append((body, meta))
    return tuple(found)
