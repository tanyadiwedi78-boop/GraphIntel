from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import yaml
from bs4 import BeautifulSoup, Tag

PARSER_VERSION = "p1"  # bump when block extraction or normalization changes
WHOLE_DOCUMENT = "DOCUMENT"
MIN_ITEMS_FOR_SPLIT = 3
CAPTION_MAX_CHARS = 300
# real filings print a running footer on every page, e.g. "Apple Inc. | 2023 Form 10-K | 19".
# Only that full footer shape (issuer | year form | page, pipe-separated, anchored)
# is removed; a sentence that merely mentions a form ("See Form 10-K for 2023")
# always survives, as do genuine repeated disclosures ("Not applicable.", "None.").
PAGE_FOOTER_MAX_CHARS = 60
PAGE_FOOTER_FORM = re.compile(r"^.*\|\s*\d{4}\s+form\s+(10-K|10-Q|8-K)\s*\|\s*\d+\s*$", re.IGNORECASE)
FOOTNOTE_MARKER = re.compile(r"^\s*(\(\d{1,2}\)|\([a-z]\)|[*\u2020\u2021#])\s")


class ParseError(ValueError):
    """The filing text or the section rules broke a parser assumption."""


@dataclass(frozen=True)
class Block:
    
    kind: str                       # "paragraph" or "table"
    text: str
    start: int                      # offset into the normalized text
    end: int
    header: Tuple[str, ...] = ()
    rows: Tuple[Tuple[str, ...], ...] = ()
    caption: str = ""
    footnotes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class FilingSection:
    """One Item section of one filing, as offsets into normalized text."""
    item_id: str
    title_line: str
    start: int
    end: int

    def text_in(self, document: str) -> str:
        return document[self.start:self.end]


@dataclass(frozen=True)
class ParsedFiling:
    """The parse result: sections, typed blocks and the normalized text."""
    sections: Tuple[FilingSection, ...]
    blocks: Tuple[Block, ...]
    text: str

    @property
    def sectioned(self) -> bool:
        """False when the filing stayed one DOCUMENT section."""
        return not (len(self.sections) == 1 and self.sections[0].item_id == WHOLE_DOCUMENT)

    def blocks_in(self, section: FilingSection) -> Tuple[Block, ...]:
        belonging = []
        for block in self.blocks:
            if section.start <= block.start < section.end:
                belonging.append(block)
        return tuple(belonging)


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if "|" in stripped:
        return True
    return stripped.count("  ") >= 2 and not stripped.endswith((".", ",", ";", ":"))


def _table_block(
    header: Tuple[str, ...],
    rows: Tuple[Tuple[str, ...], ...],
    caption: str,
    footnotes: Tuple[str, ...],
) -> Block:
    lines = []
    if caption:
        lines.append(caption)
    if header:
        lines.append(" | ".join(header))
    for row in rows:
        lines.append(" | ".join(row))
    for footnote in footnotes:
        lines.append(footnote)
    return Block("table", "\n".join(lines), 0, 0, header, rows, caption, footnotes)


def _text_blocks(text: str) -> Tuple[Block, ...]:
    """Paragraph and fixed-width-table blocks from a plain-text filing."""
    lines = text.splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        j = i
        if _is_table_line(lines[i]):
            while j < len(lines) and _is_table_line(lines[j]):
                j += 1
            table_rows = []
            for line in lines[i + 1:j]:
                table_rows.append((line,))
            blocks.append(_table_block((lines[i],), tuple(table_rows), "", ()))
        else:
            while j < len(lines) and lines[j].strip() and not _is_table_line(lines[j]):
                j += 1
            blocks.append(Block("paragraph", "\n".join(lines[i:j]), 0, 0))
        i = j
    return tuple(blocks)


def _clean(text: str) -> str:
    # real filings use non-breaking spaces and long space runs for alignment;
    # browsers collapse them, so evidence text should read like the page
    return re.sub(r"\s+", " ", text.replace("\xa0", " "))


def _walk(element: Tag, items: list) -> None:
    for child in element.children:
        if not isinstance(child, Tag):
            continue
        if child.name == "table":
            items.append(("table", child))
        elif child.find(["table", "p", "div", "h1", "h2", "h3", "h4", "h5", "h6"]):
            _walk(child, items)
        else:
            text = _clean(child.get_text(" ", strip=True))
            if text:
                items.append(("paragraph", text))


def _html_table(tag: Tag) -> Optional[Tuple[Tuple[str, ...], Tuple[Tuple[str, ...], ...]]]:
    
    rows = []
    for tr in tag.find_all("tr"):
        cell_tags = tr.find_all(["td", "th"], recursive=False)
        if not cell_tags:
            cell_tags = tr.find_all(["td", "th"])
        cells = []
        for cell_tag in cell_tags:
            cells.append(_clean(cell_tag.get_text(" ", strip=True)))
        has_text = False
        for cell in cells:
            if cell:
                has_text = True
        if has_text:
            rows.append(tuple(cells))
    if not rows:
        return None
    header = rows[0]
    body = tuple(rows[1:])
    return header, body


def _associate(items: list) -> Tuple[Block, ...]:
    """Attach captions and footnotes to their tables.
        without losing the their meaning """
    blocks: list = []
    i = 0
    while i < len(items):
        kind, payload = items[i]
        if kind != "table":
            blocks.append(Block("paragraph", payload, 0, 0))
            i += 1
            continue
        parsed = _html_table(payload)
        i += 1
        if parsed is None:
            continue
        header, rows = parsed
        caption = ""
        if blocks and blocks[-1].kind == "paragraph" and len(blocks[-1].text) <= CAPTION_MAX_CHARS:
            caption = blocks.pop().text
        footnotes = []
        while i < len(items) and items[i][0] == "paragraph" and FOOTNOTE_MARKER.match(items[i][1]):
            footnotes.append(items[i][1])
            i += 1
        blocks.append(_table_block(header, rows, caption, tuple(footnotes)))
    return tuple(blocks)


def _with_offsets(blocks: Tuple[Block, ...]) -> Tuple[Tuple[Block, ...], str]:
    placed, parts, pos = [], [], 0
    for block in blocks:
        placed.append(replace(block, start=pos, end=pos + len(block.text)))
        parts.append(block.text)
        pos += len(block.text) + 2  # the "\n\n" between blocks
    return tuple(placed), "\n\n".join(parts)


def _drop_page_furniture(blocks: Tuple[Block, ...]) -> Tuple[Block, ...]:
    """Remove only explicit running page footers that name the form
    ("... | 2023 Form 10-K | 19"). Bare numbers, bullets and everything
    else are kept - they can be real content."""
    kept = []
    for b in blocks:
        if b.kind == "paragraph":
            if len(b.text) <= PAGE_FOOTER_MAX_CHARS and PAGE_FOOTER_FORM.search(b.text):
                continue
        kept.append(b)
    return tuple(kept)


def to_blocks(raw: str, suffix: str) -> Tuple[Tuple[Block, ...], str]:
    """Typed blocks and the normalized text from one raw filing body."""
    if suffix in {".htm", ".html"}:
        # some filings start with an XML declaration; they are still HTML
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            soup = BeautifulSoup(raw, "lxml")
        # real filings hide XBRL metadata in ix: tags; it is machine data, not page text
        for junk in soup.find_all(re.compile(r"^ix:(header|hidden|references|resources)$")):
            junk.decompose()
        items: list = []
        _walk(soup.body or soup, items)
        blocks = _drop_page_furniture(_associate(items))
    else:
        blocks = _text_blocks(raw)
    return _with_offsets(blocks)


def load_section_rules(path: Path) -> Mapping[str, Sequence[str]]:
    """Read configs/filing_sections.yaml into {form: [item ids]}."""
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ParseError(f"section rules not found: {path}") from None
    except yaml.YAMLError as exc:
        raise ParseError(f"section rules are not valid YAML: {exc}") from None
    try:
        forms = raw["forms"]
    except (AttributeError, TypeError, KeyError):
        raise ParseError('section rules need a "forms" mapping with an "items" list per form') from None
    rules = {}
    for form, rule in forms.items():
        item_ids = []
        for item in rule.get("items") or ():
            item_ids.append(str(item))
        rules[str(form)] = tuple(item_ids)
    return rules


def load_chunking_version(path: Path) -> str:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ParseError(f"section rules not found: {path}") from None
    except yaml.YAMLError as exc:
        raise ParseError(f"section rules are not valid YAML: {exc}") from None
    return str((raw.get("chunking") or {}).get("version", "v1"))


def _section_start(section: FilingSection) -> int:
    """Sort key for sections: where the section begins in the text."""
    return section.start


def _heading_pattern(item_id: str) -> "re.Pattern":
    # Matches a line like "Item 1A. Risk Factors" or "ITEM 7. MANAGEMENT'S..."
    return re.compile(rf"^\s*item\s+{re.escape(item_id)}[\s.:)\-\u2014]", re.IGNORECASE | re.MULTILINE)


def detect_sections(text: str, item_ids: Sequence[str]) -> Tuple[FilingSection, ...]:
    
    found = []
    for item_id in item_ids:
        matches = list(_heading_pattern(item_id).finditer(text))
        if not matches:
            continue
        last = matches[-1]
        line_end = text.find("\n", last.start())
        if line_end == -1:
            line_end = len(text)
        title_line = text[last.start():line_end].strip()
        found.append(FilingSection(item_id, title_line, last.start(), -1))
    if len(found) < MIN_ITEMS_FOR_SPLIT:
        return ()
    found.sort(key=_section_start)
    sections = []
    for index, section in enumerate(found):
        if index + 1 < len(found):
            end = found[index + 1].start
        else:
            end = len(text)
        sections.append(FilingSection(section.item_id, section.title_line, section.start, end))
    return tuple(sections)


def parse_filing(raw: str, suffix: str, form: str, section_rules: Mapping[str, Sequence[str]]) -> ParsedFiling:
    
    
    blocks, text = to_blocks(raw, suffix)
    sections = detect_sections(text, section_rules.get(form, ()))
    if not sections:
        sections = (FilingSection(WHOLE_DOCUMENT, f"{form} (whole document)", 0, len(text)),)
    else:
        # snap each section start to the block holding its heading, so no
        # block is ever split between two sections
        snapped = []
        for section in sections:
            for block in blocks:
                if block.start <= section.start < block.end:
                    section = FilingSection(section.item_id, section.title_line, block.start, section.end)
                    break
            snapped.append(section)
        with_ends = []
        for i, section in enumerate(snapped):
            if i + 1 < len(snapped):
                end = snapped[i + 1].start
            else:
                end = len(text)
            with_ends.append(FilingSection(section.item_id, section.title_line, section.start, end))
        sections = tuple(with_ends)
    return ParsedFiling(sections, blocks, text)
