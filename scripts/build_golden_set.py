"""Pick the golden-20 evaluation chunks from data/chunks/chunks.jsonl.

One command:

    python scripts/build_golden_set.py

The selection rule stays inside this file, next to the output:

    form is 10-K, filed in 2023-2025, section is Item 1, 1A or 7.

The 20 slots follow the requested mix:

    8 clean prose, 6 dense, 4 messy/tables, 2 boilerplate.

Categories come from fields every chunk already carries, nothing guessed:

    messy       is_table is true
    boilerplate the text carries cautionary safe-harbor boilerplate
                ("forward-looking statements"), or is a cover/header chunk
    dense       not a table, est_tokens >= 800
    clean       not a table, no boilerplate marker, est_tokens < 800

The pick is deterministic: chunks sort by ticker, accession, section and
chunk id, then rotate round-robin across filings so no filing dominates.
Chunks already used in earlier samples or in the few-shot examples are
excluded, so the evaluation set stays disjoint from everything the prompt
shows the model.

Output is a NEW file every run: data/golden/golden20-chunks-<date>.json, and
the script refuses to overwrite an existing golden set. If a category has
fewer chunks than its slots, the shortfall is printed and kept - the mix is
never padded from another category.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_FOLDER))

CHUNKS_PATH = PROJECT_FOLDER / 'data' / 'chunks' / 'chunks.jsonl'
OUTPUT_FOLDER = PROJECT_FOLDER / 'data' / 'golden'

WANTED = {'clean': 8, 'dense': 6, 'messy': 4, 'boilerplate': 2}
ALLOWED_FORM = '10-K'
ALLOWED_YEARS = {2023, 2024, 2025}
ALLOWED_ITEMS = {'1', '1A', '7'}
DENSE_MIN_TOKENS = 800

BOILERPLATE_MARKERS = ('forward-looking statements', 'forward looking statements',
                       'private securities litigation reform act')
COVER_MARKERS = ('united states securities and exchange commission',
                 'securities and exchange commission washington')

EXCLUDE_FILES = ['tests/fixtures/sample5-chunks.json',
                 'tests/fixtures/positive-sample-chunks.json']


def read_chunks(path: Path) -> list:
    chunks = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def filed_year(chunk: dict):
    filed = str(chunk.get('filed') or '')
    try:
        return int(filed[:4])
    except ValueError:
        return None


def category_of(chunk: dict) -> str:
    text = str(chunk.get('text') or '').casefold()
    if chunk.get('is_table'):
        return 'messy'
    for marker in BOILERPLATE_MARKERS + COVER_MARKERS:
        if marker in text:
            return 'boilerplate'
    tokens = chunk.get('est_tokens')
    if isinstance(tokens, (int, float)) and not isinstance(tokens, bool):
        if tokens >= DENSE_MIN_TOKENS:
            return 'dense'
    return 'clean'


def matches_rule(chunk: dict) -> bool:
    if str(chunk.get('form')) != ALLOWED_FORM:
        return False
    if filed_year(chunk) not in ALLOWED_YEARS:
        return False
    if str(chunk.get('item_id')) not in ALLOWED_ITEMS:
        return False
    if not str(chunk.get('text') or '').strip():
        return False
    return True


def excluded_ids() -> set:
    excluded = set()
    for relative in EXCLUDE_FILES:
        path = PROJECT_FOLDER / relative
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(data, dict):
            excluded.update(str(key) for key in data)
        elif isinstance(data, list):
            excluded.update(str(item) for item in data)
    return excluded


def spread_pick(candidates: list, size: int) -> list:
    """Round-robin across filings; deterministic order inside each filing."""
    by_filing = {}
    order = []
    for chunk in candidates:
        accession = str(chunk.get('accession'))
        if accession not in by_filing:
            by_filing[accession] = []
            order.append(accession)
        by_filing[accession].append(chunk)
    picked = []
    while len(picked) < size and order:
        for accession in list(order):
            if len(picked) >= size:
                break
            queue = by_filing[accession]
            if not queue:
                order.remove(accession)
                continue
            picked.append(queue.pop(0))
    return picked


def main() -> int:
    if not CHUNKS_PATH.exists():
        print('No chunk corpus found at ' + str(CHUNKS_PATH))
        print('Build or copy chunks.jsonl there first, then rerun.')
        return 2

    excluded = excluded_ids()
    pool = []
    for chunk in read_chunks(CHUNKS_PATH):
        if not matches_rule(chunk):
            continue
        if str(chunk.get('chunk_id')) in excluded:
            continue
        pool.append(chunk)

    pool.sort(key=lambda c: (str(c.get('ticker')), str(c.get('accession')),
                             str(c.get('item_id')), str(c.get('chunk_id'))))

    by_category = {'clean': [], 'dense': [], 'messy': [], 'boilerplate': []}
    for chunk in pool:
        by_category[category_of(chunk)].append(chunk)

    picked = []
    shortfalls = []
    for category in ('clean', 'dense', 'messy', 'boilerplate'):
        chosen = spread_pick(by_category[category], WANTED[category])
        for chunk in chosen:
            chunk = dict(chunk)
            chunk['golden_category'] = category
            picked.append(chunk)
        missing = WANTED[category] - len(chosen)
        if missing:
            shortfalls.append(category + ': only ' + str(len(chosen))
                              + ' of ' + str(WANTED[category]) + ' available')

    stamp = datetime.now(timezone.utc).strftime('%Y%m%d')
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    target = OUTPUT_FOLDER / ('golden20-chunks-' + stamp + '.json')
    if target.exists():
        print('Refusing to overwrite ' + str(target))
        print('A golden set is immutable evidence. Pick a new name or a new day.')
        return 2

    document = {
        'purpose': 'Golden-20 evaluation set. Fixed once written; re-runs of the evaluation read this file.',
        'selection_rule': {'form': ALLOWED_FORM, 'filed_years': sorted(ALLOWED_YEARS),
                           'items': sorted(ALLOWED_ITEMS)},
        'requested_mix': WANTED,
        'shortfalls': shortfalls,
        'excluded_source_files': EXCLUDE_FILES,
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'chunks': picked,
    }
    target.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')

    counts = {}
    for chunk in picked:
        counts[chunk['golden_category']] = counts.get(chunk['golden_category'], 0) + 1
    print('Wrote ' + str(len(picked)) + ' chunks to ' + str(target))
    for category in ('clean', 'dense', 'messy', 'boilerplate'):
        print('  ' + category + ': ' + str(counts.get(category, 0))
              + ' of ' + str(WANTED[category]) + ' wanted')
    for shortfall in shortfalls:
        print('  shortfall - ' + shortfall)
    print('Point chunks_path at this file in configs/golden_extraction.json before the run.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
