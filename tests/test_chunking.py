"""Regression tests for the implemented chunking behavior in app/corpus/chunking.py.

The contracts pinned here come straight from the module docstring: paragraphs
are kept whole when they fit, splits happen at line or row boundaries only,
oversized pieces are kept whole and flagged, table parts repeat caption and
header, and chunk ids are stable, namespaced and config-stamped.
"""

from app.corpus.chunking import (
    DEFAULT_TOKEN_LIMIT,
    chunk_blocks,
    chunk_section,
    chunking_fingerprint,
)
from app.corpus.filling_parser import Block


def paragraph(text: str) -> Block:
    return Block("paragraph", text, 0, len(text))


def test_chunking_module_imports():
    # regression: this module used to import the misspelled `filing_parser` and crash
    import app.corpus.chunking  # noqa: F401


def test_small_blocks_pack_into_one_chunk():
    blocks = (paragraph("First short paragraph."), paragraph("Second short paragraph."))
    chunks = chunk_blocks(blocks, token_limit=DEFAULT_TOKEN_LIMIT)
    assert len(chunks) == 1
    assert chunks[0].text == "First short paragraph.\n\nSecond short paragraph."
    assert not chunks[0].is_table
    assert not chunks[0].oversized


def test_packing_respects_token_limit():
    # each block is 20 chars -> 5 estimated tokens; limit 12 fits two, not three
    blocks = tuple(paragraph("w" * 20) for _ in range(3))
    chunks = chunk_blocks(blocks, token_limit=12)
    assert [len(c.text) for c in chunks] == [42, 20]  # 20 + "\n\n" + 20, then 20


def test_oversize_paragraph_splits_at_line_boundaries():
    lines = ["line number one", "line number two", "line number three"]
    block = paragraph("\n".join(lines))
    pieces = chunk_blocks((block,), token_limit=5)  # ~20 chars: one line per chunk
    assert [p.text for p in pieces] == lines  # every line whole, never cut mid-word
    assert all(not p.oversized for p in pieces)


def test_single_line_over_limit_kept_whole_and_flagged():
    big = "word " * 40  # 200 chars -> 50 estimated tokens, no line break to split at
    (piece,) = chunk_blocks((paragraph(big.strip()),), token_limit=10)
    assert piece.text == big.strip()  # kept whole, retention is explicit
    assert piece.oversized


def make_table(row_count: int, with_footnote: bool = False) -> Block:
    lines = ["Table 9. Results", "Year | Revenue"]
    rows = []
    for i in range(row_count):
        lines.append(f"202{i} | {100 + i}")
        rows.append((f"202{i}", str(100 + i)))
    footnotes = ("(1) In millions.",) if with_footnote else ()
    if with_footnote:
        lines.append("(1) In millions.")
    return Block(
        "table", "\n".join(lines), 0, len("\n".join(lines)),
        header=("Year", "Revenue"), rows=tuple(rows),
        caption="Table 9. Results", footnotes=footnotes,
    )


def test_table_splits_between_rows_and_repeats_caption_and_header():
    # prefix is 31 chars -> 8 tokens; rows are 10 chars -> 3 tokens each;
    # limit 15 fits the prefix plus two rows per part
    pieces = chunk_blocks((make_table(5),), token_limit=15)
    assert len(pieces) == 3
    for index, piece in enumerate(pieces):
        assert piece.is_table and piece.header_repeated
        assert piece.part_index == index and piece.part_count == 3
        assert piece.text.startswith("Table 9. Results\nYear | Revenue\n")
        assert not piece.oversized
    # no row was cut: every row appears whole in exactly one part
    bodies = [p.text.split("\n", 2)[2] for p in pieces]
    for i in range(5):
        row = f"202{i} | {100 + i}"
        assert sum(row in body for body in bodies) == 1


def test_table_row_over_limit_kept_whole_and_flagged():
    row = "2023 | " + "9" * 100  # 107 chars -> 27 tokens, over the limit alone
    text = "Year | Revenue\n" + row
    block = Block("table", text, 0, len(text), header=("Year", "Revenue"), rows=(("2023", "9" * 100),))
    (piece,) = chunk_blocks((block,), token_limit=15)
    assert row in piece.text  # never split mid-row
    assert piece.oversized


def test_table_fitting_limit_stays_one_unsplit_piece():
    (piece,) = chunk_blocks((make_table(2),), token_limit=DEFAULT_TOKEN_LIMIT)
    assert not piece.header_repeated
    assert piece.part_count == 1
    assert piece.text == make_table(2).text


def test_footnotes_travel_with_the_last_table_part():
    pieces = chunk_blocks((make_table(3, with_footnote=True),), token_limit=15)
    assert len(pieces) == 2
    assert pieces[-1].text.endswith("(1) In millions.")
    assert "(1) In millions." not in pieces[0].text


SECTION_META = dict(
    cik="0000320193", ticker="AAPL", form="10-K", accession="0000320193-23-000106",
    filed="2023-11-03", period="2023-09-30", item_id="1A", sectioned=True,
    source_sha256="ab" * 32, chunk_version="v1", chunk_config="cf123456",
)


def test_chunk_ids_are_stable_namespaced_and_config_stamped():
    blocks = (paragraph("Risk factors text."), paragraph("More risk text."))
    first = chunk_section(blocks, **SECTION_META)
    second = chunk_section(blocks, **SECTION_META)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]  # identical reruns
    chunk = first[0]
    assert chunk.chunk_id.startswith("000032019323000106-" + "ab" * 8 + "-cf123456-1a-")
    assert chunk.chunk_id.endswith("-000")
    assert chunk.offset_basis == "normalized_text"
    assert chunk.chunk_version == "v1" and chunk.chunk_config == "cf123456"


def test_chunking_fingerprint_tracks_the_configuration():
    rules = {"10-K": ("1", "1A", "7"), "10-Q": ("1", "2")}
    base = chunking_fingerprint(chunk_version="v1", parser_version="p1", token_limit=800, section_rules=rules)
    assert chunking_fingerprint(chunk_version="v1", parser_version="p1", token_limit=800, section_rules=rules) == base
    assert chunking_fingerprint(chunk_version="v1", parser_version="p1", token_limit=800, section_rules={"10-Q": ("1", "2"), "10-K": ("1", "1A", "7")}) == base  # order-independent
    assert chunking_fingerprint(chunk_version="v2", parser_version="p1", token_limit=800, section_rules=rules) != base
    assert chunking_fingerprint(chunk_version="v1", parser_version="p2", token_limit=800, section_rules=rules) != base
    assert chunking_fingerprint(chunk_version="v1", parser_version="p1", token_limit=400, section_rules=rules) != base
    assert len(base) == 8
