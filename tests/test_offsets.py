"""Pin the offset contract between the parser and the chunker.

Every block and chunk start/end is an offset into the parser's NORMALIZED
text (never the raw filing bytes): a packed chunk's text is exactly
normalized[start:end], and a table split part's offsets cover only its own
rows (plus footnotes on the last part) while its text repeats the header.
"""

from app.corpus.chunking import chunk_blocks, chunk_section
from app.corpus.filling_parser import Block, to_blocks


def placed_blocks(*texts):
    """Blocks laid out exactly like filling_parser._with_offsets does it."""
    blocks, pos = [], 0
    for text in texts:
        blocks.append(Block("paragraph", text, pos, pos + len(text)))
        pos += len(text) + 2  # the "\n\n" between blocks
    return tuple(blocks), "\n\n".join(texts)


def test_block_offsets_index_the_normalized_text():
    blocks, normalized = placed_blocks("Alpha paragraph.", "Beta paragraph.", "Gamma paragraph.")
    for block in blocks:
        assert normalized[block.start:block.end] == block.text


def test_packed_chunk_text_equals_its_normalized_slice():
    blocks, normalized = placed_blocks("Alpha paragraph.", "Beta paragraph.", "Gamma paragraph.")
    (chunk,) = chunk_blocks(blocks, token_limit=800)
    assert normalized[chunk.start:chunk.end] == chunk.text


def test_split_paragraph_chunks_point_at_their_own_lines():
    lines = ["line number one", "line number two", "line number three"]
    blocks, normalized = placed_blocks("\n".join(lines))
    chunks = chunk_blocks(blocks, token_limit=5)
    assert [c.text for c in chunks] == lines
    for chunk in chunks:
        assert normalized[chunk.start:chunk.end] == chunk.text


def test_table_split_offsets_cover_own_rows_and_last_part_carries_footnotes():
    lines = ["Table 9. Results", "Year | Revenue", "2020 | 100", "2021 | 101", "2022 | 102", "(1) In millions."]
    text = "\n".join(lines)
    block = Block(
        "table", text, 0, len(text),
        header=("Year", "Revenue"),
        rows=(("2020", "100"), ("2021", "101"), ("2022", "102")),
        caption="Table 9. Results", footnotes=("(1) In millions.",),
    )
    pieces = chunk_blocks((block,), token_limit=15)
    assert len(pieces) == 2
    # offsets cover only the part's own rows (footnotes on the last part)...
    assert text[pieces[0].start:pieces[0].end] == "2020 | 100\n2021 | 101"
    assert text[pieces[1].start:pieces[1].end] == "2022 | 102\n(1) In millions."
    # ...while the text repeats caption and header above that slice
    for piece in pieces:
        assert piece.text == "Table 9. Results\nYear | Revenue\n" + text[piece.start:piece.end]


def test_chunk_offsets_stay_ordered_and_within_bounds():
    blocks, normalized = placed_blocks(*[f"paragraph {i} " + "x" * 30 for i in range(6)])
    chunks = chunk_blocks(blocks, token_limit=12)
    assert len(chunks) > 1
    previous_end = 0
    for chunk in chunks:
        assert 0 <= chunk.start < chunk.end <= len(normalized)
        assert chunk.start >= previous_end
        previous_end = chunk.end


def test_real_parser_output_flows_through_the_chunker_with_exact_offsets():
    raw = "<html><body><p>First visible paragraph.</p><p>Second visible paragraph.</p></body></html>"
    blocks, normalized = to_blocks(raw, ".htm")
    assert blocks, "parser produced no blocks"
    chunks = chunk_section(
        blocks,
        cik="0000000000", ticker="TEST", form="10-K", accession="0000000000-26-000001",
        filed="2026-01-01", period="2025-12-31", item_id="1", sectioned=True,
        source_sha256="cd" * 32, chunk_version="v1", chunk_config="cf123456",
    )
    for chunk in chunks:
        assert chunk.offset_basis == "normalized_text"
        assert normalized[chunk.start:chunk.end] == chunk.text
