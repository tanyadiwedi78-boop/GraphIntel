"""Measure what a pre-API name filter would skip. Read-only, changes nothing.

This script does NOT filter the extraction pipeline. It answers one question
with real corpus numbers: if chunks with no detectable entity names never
reached the model, how many chunks would be skipped, and would any reviewed
labels be lost?

The name search is a simple capitalized-word heuristic, not the ontology: the
ontology lists entity TYPES (Company, Risk, ...), not real names, so no exact
vocabulary exists to match against. The heuristic can miss real entities
written as "the Company" or in lowercase (false skips), and it sees capital
letters where no extractable fact exists (false keeps). A skip rate from this
script is a measurement on this corpus, never a guaranteed saving.

Usage:
    python scripts/prefilter_report.py [--chunks PATH] [--labels PATH]

--chunks defaults to data/chunks/chunks.jsonl. --labels is an optional
reviewed answer key (JSON object keyed by chunk id, like
tests/fixtures/labels-sample5.json) used to count facts the filter would
have lost.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Two or more capitalized words in a row, or one capitalized word, allowing
# digits and a few name characters (Apple, COVID-19, iPhone 13, AT&T).
NAME_PATTERN = re.compile(r"[A-Z][A-Za-z0-9&.'-]+(?:\s+[A-Z][A-Za-z0-9&.'-]+)*")


def candidate_names(text):
    """Distinct capitalized-name candidates, in order of first appearance."""
    names = []
    for match in NAME_PATTERN.finditer(text):
        name = match.group(0)
        if name not in names:
            names.append(name)
    return names


def read_chunks(chunks_path):
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            chunks.append(json.loads(line))
    return chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunks", default="data/chunks/chunks.jsonl",
                        help="path to chunks.jsonl (default: data/chunks/chunks.jsonl)")
    parser.add_argument("--labels", default=None,
                        help="optional reviewed labels JSON keyed by chunk id")
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        print("chunks file not found: " + str(chunks_path))
        sys.exit(1)
    chunks = read_chunks(chunks_path)
    if not chunks:
        print("no chunks read from: " + str(chunks_path))
        sys.exit(1)

    zero_name = []
    under_two = []
    zero_name_tables = 0
    for chunk in chunks:
        names = candidate_names(chunk.get("text", ""))
        if len(names) == 0:
            zero_name.append(chunk)
            if chunk.get("is_table"):
                zero_name_tables += 1
        if len(names) < 2:
            under_two.append(chunk)

    total = len(chunks)
    print("file: " + str(chunks_path))
    print("chunks read: " + str(total))
    print("")
    print("gate A (skip chunks with zero name candidates):")
    print("  would skip: " + str(len(zero_name)) + " of " + str(total)
          + " (" + str(round(100 * len(zero_name) / total, 1)) + "%)")
    print("  of those, table chunks: " + str(zero_name_tables))
    print("gate B (skip chunks with fewer than two name candidates):")
    print("  would skip: " + str(len(under_two)) + " of " + str(total)
          + " (" + str(round(100 * len(under_two) / total, 1)) + "%)")

    if args.labels is not None:
        labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
        texts = {}
        for chunk in chunks:
            texts[chunk["chunk_id"]] = chunk.get("text", "")
        lost_chunks = 0
        lost_entities = 0
        lost_relationships = 0
        missing = 0
        for chunk_id, label in labels.items():
            if chunk_id not in texts:
                missing += 1
                continue
            names = candidate_names(texts[chunk_id])
            if len(names) < 2:
                entities = label.get("entities", [])
                relationships = label.get("relationships", [])
                if entities or relationships:
                    lost_chunks += 1
                    lost_entities += len(entities)
                    lost_relationships += len(relationships)
        print("")
        print("label cross-check (" + str(args.labels) + "):")
        print("  labeled chunks: " + str(len(labels))
              + "   not present in this chunks file: " + str(missing))
        print("  labeled chunks gate B would skip despite reviewed facts: " + str(lost_chunks))
        print("  reviewed entities lost: " + str(lost_entities)
              + "   reviewed relationships lost: " + str(lost_relationships))

    print("")
    print("remember: heuristic measurement only. The filter is not wired into")
    print("the pipeline, and a skip rate here is not a guaranteed time or cost")
    print("saving. Lowercase names and words like 'the Company' are invisible")
    print("to this search.")


if __name__ == "__main__":
    main()
