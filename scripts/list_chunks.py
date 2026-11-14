"""List the saved chunks, or pick a small varied sample for hand-checking.

Default: one summary line per filing section (ticker, form, accession, item),
with the chunk count, the first chunk id, and the token range.

--sample N picks N chunks spread across different filings and different
sections, with a table chunk included when the corpus has any, and prints the
exact commands for the extraction run, the review and the scoring, with the
real chunk ids already filled in. The pick is deterministic: the same
chunks.jsonl always gives the same sample.

--show CHUNK_ID prints one chunk's metadata and full text.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ConfigError, load_settings


def read_chunks(chunks_path: Path) -> list:
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            chunks.append(json.loads(line))
    return chunks


def pick_sample(chunks: list, size: int) -> list:
    """Positions of `size` chunks, spread across documents, sections and kinds.

    Deterministic, no randomness. Documents rotate round-robin so every
    filing is represented. Inside each turn, prefer a section not yet in
    the sample, then the table/text kind that is underrepresented so far,
    then simply the next chunk - so the sample always fills up to
    min(size, len(chunks)) even when every chunk shares one section id.
    """
    queues = {}
    docs = []
    for position in range(len(chunks)):
        accession = str(chunks[position].get("accession"))
        if accession not in queues:
            queues[accession] = []
            docs.append(accession)
        queues[accession].append(position)

    picked = []
    used_items = set()
    picked_tables = 0
    picked_text = 0
    while len(picked) < size:
        progressed = False
        for accession in docs:
            if len(picked) >= size:
                break
            queue = queues[accession]
            if len(queue) == 0:
                continue
            chosen = None
            for position in queue:
                if str(chunks[position].get("item_id")) not in used_items:
                    chosen = position
                    break
            if chosen is None:
                want_table = picked_tables <= picked_text
                for position in queue:
                    if bool(chunks[position].get("is_table")) == want_table:
                        chosen = position
                        break
            if chosen is None:
                chosen = queue[0]
            queue.remove(chosen)
            used_items.add(str(chunks[chosen].get("item_id")))
            if chunks[chosen].get("is_table"):
                picked_tables = picked_tables + 1
            else:
                picked_text = picked_text + 1
            picked.append(chosen)
            progressed = True
        if not progressed:
            break
    return picked


def print_sample(chunks: list, picked: list, subset_path) -> None:
    ids = []
    print(f"Sample of {len(picked)} chunks:")
    number = 0
    for position in picked:
        number = number + 1
        chunk = chunks[position]
        ids.append(str(chunk.get("chunk_id")))
        print(f"  {number}. {chunk.get('chunk_id')}  {chunk.get('ticker')} {chunk.get('form')}"
              f" {chunk.get('item_id')}  table: {chunk.get('is_table')}, {chunk.get('est_tokens')} est tokens")
    if subset_path is None:
        print("")
        print("To prepare an offline plan for these chunks, write them to a subset file first:")
        print("")
        print(f"python scripts/list_chunks.py --sample {len(picked)} "
              "--write-subset data/chunks/sample" + str(len(picked)) + ".jsonl")
        print("")
        print("The full chunks.jsonl is too large for one job; the plan step needs the subset file.")
        return
    joined = ",".join(ids)
    print("")
    print(f"Wrote {len(picked)} chunks to {subset_path}")
    print("")
    print("Prepare an OFFLINE plan (no provider calls) for a sample job:")
    print("")
    print("python -m app.extraction.pipeline prepare --job data/sample-job "
          f"--chunks {subset_path} --gemini-model MODEL --max-output-tokens 4096")
    print("")
    print("Review the immutable plan before approving any execution; a run needs")
    print("--execute-authorized plus verified --rpm/--tpm limits. Never repeat a batch")
    print("submission after a timeout; reconcile its saved claim first.")
    print("")
    print("Then read each result next to its source text:")
    print("")
    print("python scripts/review_extraction.py --results-dir data/sample-job --chunk-ids " + joined)
    print("")
    print("Create the labels file, fill it in completely, then score it:")
    print("")
    print("python scripts/review_extraction.py --chunk-ids " + joined + " --write-labels data/labels_sample.json")
    print("python scripts/review_extraction.py --results-dir data/sample-job --labels data/labels_sample.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="List saved chunks, or pick a varied hand-check sample.")
    parser.add_argument("--document", default=None,
                        help="only sections whose ticker or accession contains this text")
    parser.add_argument("--show", default=None, metavar="CHUNK_ID",
                        help="print one chunk's metadata and full text")
    parser.add_argument("--sample", type=int, default=0, metavar="N",
                        help="pick N varied chunks and print the exact next commands")
    parser.add_argument("--write-subset", default=None, metavar="PATH",
                        help="with --sample: write the picked chunks to PATH (must be a new file)")
    args = parser.parse_args()

    if args.write_subset is not None and args.sample <= 0:
        print("--write-subset only works together with --sample N.")
        return 1

    try:
        settings = load_settings()
    except ConfigError as error:
        print(f"Settings problem: {error}")
        return 1

    chunks_path = settings.data_dir / "chunks" / "chunks.jsonl"
    if not chunks_path.exists():
        print(f"No chunks at {chunks_path}. Run scripts/build_chunks.py first.")
        return 1

    chunks = read_chunks(chunks_path)
    if len(chunks) == 0:
        print(f"{chunks_path} has no chunks.")
        return 1

    if args.sample > 0:
        if args.sample > len(chunks):
            print(f"Asked for a sample of {args.sample} but only {len(chunks)} chunks exist.")
            return 1
        picked = pick_sample(chunks, args.sample)
        if len(picked) < args.sample:
            print(f"Could only pick {len(picked)} varied chunks out of {args.sample} requested.")
            print("The sample is smaller than asked for; the commands below use the smaller sample.")
        subset_path = None
        if args.write_subset is not None:
            subset_path = Path(args.write_subset)
            if subset_path.exists():
                print(f"Not writing {subset_path}: the file already exists. Choose a new name.")
                return 1
            parent = subset_path.parent
            if not parent.exists():
                print(f"Not writing {subset_path}: folder {parent} does not exist.")
                return 1
            with subset_path.open("w", encoding="utf-8") as out:
                for position in picked:
                    out.write(json.dumps(chunks[position], ensure_ascii=False) + "\n")
        print_sample(chunks, picked, subset_path)
        return 0

    if args.show is not None:
        found = None
        for chunk in chunks:
            if chunk.get("chunk_id") == args.show:
                found = chunk
        if found is None:
            print(f"No chunk with id {args.show} in {chunks_path}.")
            return 1
        print("chunk_id: " + str(found.get("chunk_id")))
        print("ticker: " + str(found.get("ticker")) + "  form: " + str(found.get("form"))
              + "  accession: " + str(found.get("accession")))
        print("item: " + str(found.get("item_id")) + "  period: " + str(found.get("period"))
              + "  table: " + str(found.get("is_table")) + "  est_tokens: " + str(found.get("est_tokens")))
        print("-" * 60)
        print(found.get("text", ""))
        return 0

    groups = []
    index = {}
    for chunk in chunks:
        if args.document is not None:
            haystack = str(chunk.get("ticker")) + " " + str(chunk.get("accession"))
            if args.document.lower() not in haystack.lower():
                continue
        key = (str(chunk.get("ticker")), str(chunk.get("form")),
               str(chunk.get("accession")), str(chunk.get("item_id")))
        if key not in index:
            index[key] = len(groups)
            groups.append({"key": key, "count": 0, "first_id": "", "tokens_min": None, "tokens_max": 0})
        group = groups[index[key]]
        group["count"] = group["count"] + 1
        if group["first_id"] == "":
            group["first_id"] = str(chunk.get("chunk_id"))
        tokens = chunk.get("est_tokens")
        if isinstance(tokens, int):
            if group["tokens_min"] is None or tokens < group["tokens_min"]:
                group["tokens_min"] = tokens
            if tokens > group["tokens_max"]:
                group["tokens_max"] = tokens

    print(f"{len(chunks)} chunks in {chunks_path}")
    if args.document is not None:
        print(f"Filtered by --document {args.document}")
    print("One line per filing section: ticker form accession item | chunks | first chunk id | token range")
    for group in groups:
        ticker, form, accession, item_id = group["key"]
        print(f"{ticker} {form} {accession} {item_id} | {group['count']} chunks"
              f" | first: {group['first_id']} | tokens {group['tokens_min']}-{group['tokens_max']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
