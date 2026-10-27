"""Pack a filing's typed blocks into chunks under the token limit.

Rules, all deliberate:
- Paragraphs are kept whole when they fit. An oversize paragraph splits
  at line boundaries, between words, and any piece still over the limit
  is flagged oversized, never cut mid-word.
- Tables split between rows, never mid-row. Every part repeats the
  caption and the header row, so column names and units survive the
  split. A single row over the limit is kept whole and flagged
  oversized - retention is explicit, not silent.
- Chunk ids are namespaced so they can never collide across sources or
  chunking rules: accession + a 16-hex tag of the raw filing's SHA-256 +
  an 8-hex fingerprint of the canonical chunking configuration +
  item + number. The fingerprint is computed, not remembered: it covers
  the chunking version, the parser version, the token limit and the
  section rules, so ANY config change makes new ids automatically.
  Identical reruns produce identical ids. Graph edges and vector rows
  must point at these same ids. build_chunks.py refuses to write
  duplicate ids, as a collision guard.
- start/end are offsets into the parser's NORMALIZED text (see
  filing_parser), and every chunk says so: offset_basis is always
  "normalized_text". A packed chunk's text is exactly
  normalized[start:end]. A table split part repeats its header lines,
  so its text is those repeated lines plus normalized[start:end] of its
  own rows (and footnotes, on the last part); it is marked
  header_repeated with part_index/part_count.
- Token counts are estimates (characters / 4) until the embedding
  model's real tokenizer is chosen in a later phase.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence, Tuple

from app.corpus.filing_parser import Block

DEFAULT_TOKEN_LIMIT = 800
CHARS_PER_TOKEN = 4
OFFSET_BASIS = "normalized_text"
SOURCE_TAG_CHARS = 16   # 64 bits of the raw filing's SHA-256
CONFIG_TAG_CHARS = 8    # 32 bits of the canonical config fingerprint


def chunking_fingerprint(
    *,
    chunk_version: str,
    parser_version: str,
    token_limit: int,
    section_rules: Mapping[str, Sequence[str]],
) -> str:
    """An 8-hex fingerprint of the canonical chunking configuration.

    Computed, not remembered: any change to the chunking version, the
    parser version, the token limit or the section rules changes the
    fingerprint, so chunk ids change automatically with the config.
    Identical configuration means an identical fingerprint."""
    rules_canonical = {}
    for form in sorted(section_rules.keys()):
        item_ids = []
        for item in section_rules[form]:
            item_ids.append(str(item))
        rules_canonical[str(form)] = item_ids
    config = {
        "chunk_version": str(chunk_version),
        "parser_version": str(parser_version),
        "token_limit": int(token_limit),
        "section_rules": rules_canonical,
    }
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:CONFIG_TAG_CHARS]


@dataclass(frozen=True)
class Chunk:
    """One retrievable piece of one filing, with its evidence metadata."""
    chunk_id: str
    cik: str
    ticker: str
    form: str
    accession: str
    filed: str
    period: str
    item_id: str
    sectioned: bool            
    chunk_index: int
    start: int               
    end: int
    offset_basis: str          # always "normalized_text", never the raw bytes
    source_sha256: str       
    chunk_version: str        
    chunk_config: str          
    est_tokens: int
    is_table: bool
    oversized: bool        
    header_repeated: bool      
    part_index: int        
    part_count: int            
    text: str = field(compare=False)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "cik": self.cik,
            "ticker": self.ticker,
            "form": self.form,
            "accession": self.accession,
            "filed": self.filed,
            "period": self.period,
            "item_id": self.item_id,
            "sectioned": self.sectioned,
            "chunk_index": self.chunk_index,
            "start": self.start,
            "end": self.end,
            "offset_basis": self.offset_basis,
            "source_sha256": self.source_sha256,
            "chunk_version": self.chunk_version,
            "chunk_config": self.chunk_config,
            "est_tokens": self.est_tokens,
            "is_table": self.is_table,
            "oversized": self.oversized,
            "header_repeated": self.header_repeated,
            "part_index": self.part_index,
            "part_count": self.part_count,
            "text": self.text,
        }


def estimate_tokens(text: str) -> int:
    """Rough token count."""
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


@dataclass(frozen=True)
class Piece:
    """One chunkable unit:"""
    text: str
    block_seq: int
    is_table: bool
    oversized: bool
    start: int
    end: int
    header_repeated: bool = False
    part_index: int = 0
    part_count: int = 1


def _paragraph_pieces(block: Block, block_seq: int, token_limit: int) -> Tuple[Piece, ...]:
    lines = []
    for line in block.text.splitlines():
        if line.strip():
            lines.append(line)
    spans = []
    cursor = block.start
    for line in lines:
        at = block.text.find(line, cursor - block.start)
        spans.append((block.start + at, block.start + at + len(line)))
        cursor = block.start + at + len(line)
    groups = []
    current = []
    current_tokens = 0
    for index in range(len(lines)):
        line = lines[index]
        start, end = spans[index]
        line_tokens = estimate_tokens(line)
        if current and current_tokens + line_tokens > token_limit:
            groups.append(current)
            current = []
        current.append((line, start, end))
        current_tokens += line_tokens
    if current:
        groups.append(current)
    pieces = []
    for group in groups:
        group_lines = []
        group_oversized = False
        for line, _start, _end in group:
            group_lines.append(line)
            if estimate_tokens(line) > token_limit:
                group_oversized = True
        first_line, first_start, _first_end = group[0]
        _last_line, _last_start, last_end = group[-1]
        pieces.append(Piece(
            "\n".join(group_lines),
            block_seq,
            False,
            group_oversized,
            first_start,
            last_end,
        ))
    return tuple(pieces)


def _split_table(block: Block, block_seq: int, token_limit: int) -> Tuple[Piece, ...]:
    prefix_lines = []
    if block.caption:
        prefix_lines.append(block.caption)
    if block.header:
        header_line = " | ".join(block.header)
        prefix_lines.append(header_line)
    prefix = "\n".join(prefix_lines)
    if prefix:
        prefix_tokens = estimate_tokens(prefix)
    else:
        prefix_tokens = 0

    rows = []
    for row in block.rows:
        rows.append(" | ".join(row))
    row_spans, cursor = [], 0
    for row in rows:
        at = block.text.find(row, cursor)
        row_spans.append((at, at + len(row)))
        cursor = at + len(row)
    footnote_text = "\n".join(block.footnotes)
    footnote_span = None
    if footnote_text:
        at = block.text.find(footnote_text, cursor)
        footnote_span = (at, at + len(footnote_text))

    parts = []   # each: {"rows": [row indices], "oversized": bool}
    current, current_tokens = [], prefix_tokens
    for index, row in enumerate(rows):
        row_tokens = estimate_tokens(row)
        if prefix_tokens + row_tokens > token_limit:
            # one row alone breaks the limit: kept whole, flagged
            if current:
                parts.append({"rows": current, "oversized": False})
                current, current_tokens = [], prefix_tokens
            parts.append({"rows": [index], "oversized": True})
            continue
        if current and current_tokens + row_tokens > token_limit:
            parts.append({"rows": current, "oversized": False})
            current, current_tokens = [], prefix_tokens
        current.append(index)
        current_tokens += row_tokens
    if current:
        parts.append({"rows": current, "oversized": False})

    if len(parts) <= 1:
        # the table fits (or cannot be split) as one piece after all
        is_oversized = estimate_tokens(block.text) > token_limit
        return (Piece(block.text, block_seq, True, is_oversized, block.start, block.end),)

    pieces = []
    last_part = len(parts) - 1
    for part_index, part in enumerate(parts):
        first, last_row = part["rows"][0], part["rows"][-1]
        local_start = row_spans[first][0]
        local_end = row_spans[last_row][1]
        if part_index == last_part and footnote_span is not None:
            local_end = footnote_span[1]  # footnotes travel with the last part
        body = block.text[local_start:local_end]
        if prefix:
            text = prefix + "\n" + body
        else:
            text = body
        is_oversized = part["oversized"] or estimate_tokens(text) > token_limit
        pieces.append(Piece(
            text, block_seq, True,
            is_oversized,
            block.start + local_start, block.start + local_end,
            header_repeated=True, part_index=part_index, part_count=len(parts),
        ))
    return tuple(pieces)


def _block_pieces(block: Block, block_seq: int, token_limit: int) -> Tuple[Piece, ...]:
    if block.kind == "table":
        if estimate_tokens(block.text) > token_limit:
            return _split_table(block, block_seq, token_limit)
        return (Piece(block.text, block_seq, True, False, block.start, block.end),)
    if estimate_tokens(block.text) > token_limit:
        return _paragraph_pieces(block, block_seq, token_limit)
    return (Piece(block.text, block_seq, False, False, block.start, block.end),)


def chunk_blocks(blocks: Tuple[Block, ...], token_limit: int = DEFAULT_TOKEN_LIMIT) -> Tuple[Piece, ...]:
    
    pieces = []
    for seq, block in enumerate(blocks):
        pieces.extend(_block_pieces(block, seq, token_limit))

    chunks: list[Piece] = []
    current: list[Piece] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current_tokens
        if not current:
            return
        text_parts = []
        for i, piece in enumerate(current):
            if i > 0:
                if piece.block_seq == current[i - 1].block_seq:
                    text_parts.append("\n")
                else:
                    text_parts.append("\n\n")
            text_parts.append(piece.text)
        solo = len(current) == 1
        has_table = False
        has_oversized = False
        for piece in current:
            if piece.is_table:
                has_table = True
            if piece.oversized:
                has_oversized = True
        if solo:
            header_repeated = current[0].header_repeated
            part_index = current[0].part_index
            part_count = current[0].part_count
        else:
            header_repeated = False
            part_index = 0
            part_count = 1
        chunks.append(Piece(
            "".join(text_parts),
            current[0].block_seq,
            has_table,
            has_oversized,
            current[0].start,
            current[-1].end,
            header_repeated=header_repeated,
            part_index=part_index,
            part_count=part_count,
        ))
        current.clear()
        current_tokens = 0

    for piece in pieces:
        if piece.header_repeated:
            flush()
            current.append(piece)
            flush()
            continue
        piece_tokens = estimate_tokens(piece.text)
        if current and current_tokens + piece_tokens > token_limit:
            flush()
        current.append(piece)
        current_tokens += piece_tokens
    flush()
    return tuple(chunks)


def chunk_section(
    blocks: Tuple[Block, ...],
    *,
    cik: str,
    ticker: str,
    form: str,
    accession: str,
    filed: str,
    period: str,
    item_id: str,
    sectioned: bool,
    source_sha256: str,
    chunk_version: str,
    chunk_config: str,
    token_limit: int = DEFAULT_TOKEN_LIMIT,
) -> Tuple[Chunk, ...]:
    """Chunk one section's blocks and stamp every chunk with stable,
    namespaced metadata."""
    accession_flat = accession.replace("-", "")
    source_tag = source_sha256[:SOURCE_TAG_CHARS]
    item_slug = item_id.lower().replace(".", "-")
    if not item_slug:
        item_slug = "doc"
    chunks = []
    for index, piece in enumerate(chunk_blocks(blocks, token_limit)):
        chunks.append(Chunk(
            chunk_id=f"{accession_flat}-{source_tag}-{chunk_config}-{item_slug}-{index:03d}",
            cik=cik,
            ticker=ticker,
            form=form,
            accession=accession,
            filed=filed,
            period=period,
            item_id=item_id,
            sectioned=sectioned,
            chunk_index=index,
            start=piece.start,
            end=piece.end,
            offset_basis=OFFSET_BASIS,
            source_sha256=source_sha256,
            chunk_version=chunk_version,
            chunk_config=chunk_config,
            est_tokens=estimate_tokens(piece.text),
            is_table=piece.is_table,
            oversized=piece.oversized,
            header_repeated=piece.header_repeated,
            part_index=piece.part_index,
            part_count=piece.part_count,
            text=piece.text,
        ))
    return tuple(chunks)
