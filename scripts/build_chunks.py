""" SEC fillings 
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ConfigError, load_settings
from app.corpus.chunking import DEFAULT_TOKEN_LIMIT, chunk_section, chunking_fingerprint
from app.corpus.edgar import load_saved_filings
from app.corpus.filling_parser import PARSER_VERSION, load_chunking_version, load_section_rules, parse_filing


def main() -> int:
    parser = argparse.ArgumentParser(description="Build structure-aware chunks from downloaded filings.")
    parser.add_argument("--token-limit", type=int, default=DEFAULT_TOKEN_LIMIT,
                        help=f"estimated tokens per chunk (default: {DEFAULT_TOKEN_LIMIT})")
    args = parser.parse_args()


    try:

        settings = load_settings()
        rules_path = Path("configs/filing_sections.yaml")
        rules = load_section_rules(rules_path)
        chunk_version = load_chunking_version(rules_path)
        chunk_config = chunking_fingerprint(
            chunk_version=chunk_version, parser_version=PARSER_VERSION,
            token_limit=args.token_limit, section_rules=rules,
        )
    except ConfigError as exc:
        print(f"PROBLEM: {exc}")
        return 1

    filings = load_saved_filings(settings.data_dir)
    if not filings:
        print(f"No filings under {settings.data_dir}/raw/ - run scripts/download_filings.py first")
        return 1

    out_path = Path(settings.data_dir) / "chunks" / "chunks.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    table_chunks = 0
    oversized = 0
    whole_document = 0
    seen_ids = set()  # collision guard: refuse to write a duplicate id
    with out_path.open("w", encoding="utf-8") as out:
        for path, meta in filings:
            raw_bytes = path.read_bytes()
            source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
            parsed = parse_filing(raw_bytes.decode("utf-8", errors="replace"), meta.suffix, meta.form, rules)
            if not parsed.sectioned:
                whole_document += 1
            section_chunks = 0
            section_tables = 0
            for section in parsed.sections:
                for chunk in chunk_section(
                    parsed.blocks_in(section),
                    cik=meta.cik, ticker=meta.ticker, form=meta.form,
                    accession=meta.accession, filed=meta.filed, period=meta.period,
                    item_id=section.item_id, sectioned=parsed.sectioned,
                    source_sha256=source_sha256, chunk_version=chunk_version,
                    chunk_config=chunk_config, token_limit=args.token_limit,
                ):
            
                    if chunk.chunk_id in seen_ids:
                        print(f"PROBLEM: duplicate chunk id {chunk.chunk_id} - "
                            f"refusing to write. Please report this collision.")
                        return 1
                    seen_ids.add(chunk.chunk_id)
                    out.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
                    total_chunks += 1
                    section_chunks += 1
                    if chunk.is_table:
                        table_chunks += 1
                        section_tables += 1
                    if chunk.oversized:
                        oversized += 1
            print(f"{meta.ticker} {meta.form} filed {meta.filed}: "
                f"{len(parsed.sections)} sections, {section_chunks} chunks "
                f"({section_tables} table chunks)")


    print(f"Done. {total_chunks} chunks -> {out_path}")
    print(f"  table chunks (structure kept): {table_chunks}")
    print(f"  oversized chunks (kept whole, over the limit): {oversized}")
    print(f"  filings kept as one DOCUMENT section (sectioned=false): {whole_document}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
