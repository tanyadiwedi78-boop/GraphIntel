"""Parser behavior on REAL SEC filing HTML (trimmed excerpts, sources in fixture headers).

These complement the small inline-string tests: the fixtures are verbatim
excerpts from Apple Inc.'s real 10-K (accession 0000320193-23-000106),
so they pin how the parser treats actual filing HTML - hidden XBRL
metadata, page footers, non-breaking spaces, units captions.
"""

from pathlib import Path

from app.corpus.filling_parser import load_section_rules, parse_filing, to_blocks

FIXTURES = Path(__file__).resolve().parent / "fixtures"
RULES = load_section_rules(Path(__file__).resolve().parent.parent / "configs" / "filling_sections.yaml")


def test_real_filing_has_no_hidden_xbrl_metadata_in_text():
    raw = (FIXTURES / "aapl_10k_2023_head_excerpt.html").read_text(encoding="utf-8")
    _, text = to_blocks(raw, ".htm")
    assert "fasb.org" not in text
    assert "0000320193 P1Y" not in text


def test_real_filing_item_heading_found_with_normalized_spaces():
    raw = (FIXTURES / "aapl_10k_2023_head_excerpt.html").read_text(encoding="utf-8")
    blocks, _ = to_blocks(raw, ".htm")
    headings = [b.text for b in blocks if b.text.startswith("Item 1.")]
    assert "Item 1. Business" in headings


def test_real_filing_page_footers_removed():
    raw = (FIXTURES / "aapl_10k_2023_head_excerpt.html").read_text(encoding="utf-8")
    _, text = to_blocks(raw, ".htm")
    assert "Apple Inc. | 2023 Form 10-K |" not in text


def test_real_filing_keeps_visible_page_text():
    raw = (FIXTURES / "aapl_10k_2023_head_excerpt.html").read_text(encoding="utf-8")
    _, text = to_blocks(raw, ".htm")
    assert "UNITED STATES" in text
    assert "FORM 10-K" in text


def test_real_financial_table_keeps_units_caption_and_header():
    raw = (FIXTURES / "aapl_10k_2023_table_excerpt.html").read_text(encoding="utf-8")
    blocks, _ = to_blocks(raw, ".htm")
    tables = [b for b in blocks if b.kind == "table"]
    assert len(tables) == 1
    table = tables[0]
    assert table.caption.startswith("(In millions")
    assert "Years ended" in table.header
    assert len(table.rows) >= 20
    assert any("Total net sales" in " ".join(row) for row in table.rows)


def test_real_filing_excerpt_items_1_and_1a_sectioned():
    raw = (FIXTURES / "aapl_10k_2023_head_excerpt.html").read_text(encoding="utf-8")
    parsed = parse_filing(raw, ".htm", "10-K", RULES)
    by_id = {s.item_id: s for s in parsed.sections}
    assert "1" in by_id and "1A" in by_id
    assert by_id["1"].text_in(parsed.text).strip().startswith("Item 1. Business")
    assert by_id["1A"].text_in(parsed.text).strip().startswith("Item 1A. Risk Factors")
    assert by_id["1"].end - by_id["1"].start > 1000

def test_repeated_legitimate_short_sentences_survive():
    # repeated real disclosures must never be treated as page furniture
    html = "<html><body>" + "<p>Not applicable.</p>" * 6 + "<p>None.</p>" * 6 + "</body></html>"
    _, text = to_blocks(html, ".htm")
    assert text.count("Not applicable.") == 6
    assert text.count("None.") == 6


def test_bare_numbers_and_bullets_are_kept():
    # a standalone number can be a year or a financial value; a bullet can be
    # real structure - neither is safe to classify as furniture
    html = "<html><body><p>2023</p><p>•</p><p>298,085</p><p>Real sentence stays.</p></body></html>"
    _, text = to_blocks(html, ".htm")
    assert "2023" in text
    assert "•" in text
    assert "298,085" in text
    assert "Real sentence stays." in text


def test_fixture_provenance_comment_is_not_parsed_as_content():
    raw = (FIXTURES / "aapl_10k_2023_head_excerpt.html").read_text(encoding="utf-8")
    _, text = to_blocks(raw, ".htm")
    assert "Accession" not in text
    assert "sec.gov" not in text

def test_sentence_mentioning_a_form_survives():
    # a short substantive sentence that names a form is not page furniture;
    # only the full pipe-separated footer shape is removed
    html = "<html><body><p>See Form 10-K for 2023</p><p>Apple Inc. | 2023 Form 10-K | 19</p></body></html>"
    _, text = to_blocks(html, ".htm")
    assert "See Form 10-K for 2023" in text
    assert "Apple Inc. | 2023 Form 10-K | 19" not in text