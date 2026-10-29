"""EDGAR selection and saving, without any network."""
from app.corpus.edgar import FilingMeta, load_saved_filings, save_filing, select_filings

RECENT = {
    "accessionNumber": ["0001045810-25-000023", "0001045810-24-000029", "0001045810-21-000010"],
    "form": ["10-K", "10-Q", "10-K"],
    "filingDate": ["2025-02-26", "2024-05-22", "2021-02-26"],
    "reportDate": ["2025-01-26", "2024-03-31", "2021-01-31"],
    "primaryDocument": ["nvda-20250126.htm", "nvda-20240331.htm", "nvda-20210131.htm"],
}


def test_select_filings_keeps_forms_and_year_range():
    chosen = select_filings(RECENT, ("10-K", "10-Q"), 2022, 2025, "0001045810", "NVDA")
    forms = []
    for filing in chosen:
        forms.append(filing.form)
    assert forms == ["10-K", "10-Q"]  # the 2021 10-K falls out of range


def test_select_filings_respects_per_form_limit():
    chosen = select_filings(RECENT, ("10-K",), 2021, 2025, "0001045810", "NVDA", per_form_limit=1)
    assert len(chosen) == 1
    assert chosen[0].filed == "2025-02-26"  # newest first, as EDGAR lists them


def test_filing_url_uses_integer_cik_and_flat_accession():
    meta = select_filings(RECENT, ("10-K",), 2022, 2025, "0001045810", "NVDA")[0]
    assert meta.url == (
        "https://www.sec.gov/Archives/edgar/data/1045810/"
        "000104581025000023/nvda-20250126.htm"
    )


def test_save_and_load_roundtrip_with_metadata(tmp_path):
    meta = select_filings(RECENT, ("10-K",), 2022, 2025, "0001045810", "NVDA")[0]
    save_filing(tmp_path, meta, "raw filing text")
    loaded = load_saved_filings(tmp_path)
    assert len(loaded) == 1
    path, loaded_meta = loaded[0]
    assert path.read_text(encoding="utf-8") == "raw filing text"
    assert loaded_meta == meta
