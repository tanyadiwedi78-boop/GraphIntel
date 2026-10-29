"""Section detection, typed blocks, and the honest fallbacks."""
from pathlib import Path

from app.corpus.filling_parser import (
    detect_sections,
    load_chunking_version,
    parse_filing,
    to_blocks,
)

RULES = {"10-K": ("1", "1A", "7", "8"), "8-K": ()}

MINI_10K = """
FORM 10-K

Table of Contents

Item 1. Business ........ page 3
Item 1A. Risk Factors ... page 9

Item 1. Business
We make chips and software for gaming and data centers.

Item 1A. Risk Factors
Our business depends on supply of advanced wafers.

Item 7. Management's Discussion and Analysis
Revenue grew across all reportable segments.

Item 8. Financial Statements and Supplementary Data
See the consolidated statements that follow.
"""

HTML_TABLE = """
<html><body>
<p>Item 1. Business</p>
<p>We make chips.</p>
<p>Consolidated results (in millions)</p>
<table>
<tr><th>Year</th><th>Revenue</th></tr>
<tr><td>2024</td><td>60,922</td></tr>
<tr><td>2023</td><td>26,974</td></tr>
</table>
<p>(1) Reflects an acquisition charge.</p>
<p>More prose after the table.</p>
</body></html>
"""


def test_10k_items_found_after_the_table_of_contents():
    parsed = parse_filing(MINI_10K, ".txt", "10-K", RULES)
    item_ids = []
    for section in parsed.sections:
        item_ids.append(section.item_id)
    assert item_ids == ["1", "1A", "7", "8"]
    business = parsed.sections[0].text_in(parsed.text)
    assert "We make chips" in business
    assert "page 3" not in business  # the contents entry is not the section


def test_sections_cover_the_document_in_order():
    parsed = parse_filing(MINI_10K, ".txt", "10-K", RULES)
    for earlier, later in zip(parsed.sections, parsed.sections[1:]):
        assert earlier.end == later.start
    assert parsed.sections[-1].end == len(parsed.text)
    # every block after the preamble belongs to exactly one section
    body_count = 0
    for block in parsed.blocks:
        if block.start >= parsed.sections[0].start:
            body_count += 1
    total = 0
    for section in parsed.sections:
        total += len(parsed.blocks_in(section))
    assert total == body_count


def test_normalized_text_is_the_blocks_joined():
    blocks, text = to_blocks(MINI_10K, ".txt")
    block_texts = []
    for block in blocks:
        block_texts.append(block.text)
    assert text == "\n\n".join(block_texts)
    for block in blocks:
        assert text[block.start:block.end] == block.text


def test_form_without_rules_becomes_one_document_section():
    parsed = parse_filing("An 8-K about an event.", ".txt", "8-K", RULES)
    assert len(parsed.sections) == 1
    assert parsed.sections[0].item_id == "DOCUMENT"
    assert parsed.sectioned is False


def test_too_few_headings_falls_back_to_document():
    parsed = parse_filing("Item 1. Business\nOnly one heading here.", ".txt", "10-K", RULES)
    assert parsed.sectioned is False


def test_html_paragraphs_keep_the_words_lose_the_tags():
    blocks, text = to_blocks("<html><body><p>Item 1. Business</p><p>We make chips.</p></body></html>", ".htm")
    assert "<p>" not in text
    found_chips = False
    for block in blocks:
        if block.kind == "paragraph" and block.text == "We make chips.":
            found_chips = True
    assert found_chips


def test_html_table_keeps_its_structure_caption_and_footnotes():
    blocks, _ = to_blocks(HTML_TABLE, ".htm")
    tables = []
    for block in blocks:
        if block.kind == "table":
            tables.append(block)
    assert len(tables) == 1
    table = tables[0]
    assert table.header == ("Year", "Revenue")
    assert ("2024", "60,922") in table.rows
    assert table.caption == "Consolidated results (in millions)"
    assert table.footnotes == ("(1) Reflects an acquisition charge.",)
    # caption and footnote left the paragraph flow and travel with the table
    paragraphs = []
    for block in blocks:
        if block.kind == "paragraph":
            paragraphs.append(block.text)
    for paragraph in paragraphs:
        assert "in millions" not in paragraph
        assert "acquisition charge" not in paragraph
    found_prose = False
    for paragraph in paragraphs:
        if paragraph == "More prose after the table.":
            found_prose = True
    assert found_prose


def test_plain_text_fixed_width_table_becomes_a_table_block():
    text = "Intro line.\n\nSegment    Revenue\nGaming     9,160\nOther      500\n\nClosing line."
    blocks, _ = to_blocks(text, ".txt")
    tables = []
    for block in blocks:
        if block.kind == "table":
            tables.append(block)
    assert len(tables) == 1
    assert tables[0].header == ("Segment    Revenue",)
    assert len(tables[0].rows) == 2


def test_chunking_version_comes_from_config():
    version = load_chunking_version(Path("configs/filling_sections.yaml"))
    assert version == "v1"
