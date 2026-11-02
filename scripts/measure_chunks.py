"""Read-only audit of data/chunks/chunks.jsonl.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/chunks/chunks.jsonl")
DATA_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data")

rows = []
bad_lines = 0
with PATH.open("r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad_lines += 1

print(f"file: {PATH}")
print(f"chunks read: {len(rows)}   unreadable lines: {bad_lines}")
if not rows:
    sys.exit(1)

# 1. distinct filings per form
filings = {(r["accession"], r["form"], r["ticker"], r["filed"]) for r in rows}
per_form_filings = Counter(f[1] for f in filings)
print(f"\ndistinct filings processed: {len(filings)}")
for form, n in sorted(per_form_filings.items()):
    print(f"  {form}: {n}")

# 2. sectioned=false per form, and any 10-K fallback
fallback = defaultdict(set)
for r in rows:
    if not r["sectioned"]:
        fallback[r["form"]].add((r["ticker"], r["filed"], r["accession"]))
print("\nfilings kept as one DOCUMENT section (sectioned=false):")
for form, s in sorted(fallback.items()):
    print(f"  {form}: {len(s)}")
tenk_fallback = sorted(fallback.get("10-K", ()))
if tenk_fallback:
    print("  10-Ks that missed their Item headings (worth a look):")
    for tk, filed, acc in tenk_fallback:
        print(f"    {tk} filed {filed} ({acc})")
else:
    print("  no 10-K fell back - every 10-K found its Item headings")

# 3. oversized chunks
over = [r for r in rows if r["oversized"]]
print(f"\noversized chunks (kept whole, over the token limit): {len(over)}")
if over:
    by_form = Counter(r["form"] for r in over)
    print("  per form:", dict(sorted(by_form.items())))
    ests = sorted(r["est_tokens"] for r in over)
    print(f"  est_tokens: smallest {ests[0]}, median {ests[len(ests)//2]}, largest {ests[-1]}")
    print("  5 largest:")
    for r in sorted(over, key=lambda r: -r["est_tokens"])[:5]:
        print(f"    {r['ticker']} {r['form']} filed {r['filed']} section {r['item_id']}"
            f" est_tokens={r['est_tokens']} table={r['is_table']} id={r['chunk_id']}")

# 4. empty chunks
empty = [r["chunk_id"] for r in rows if not r["text"].strip()]
print(f"\nempty or whitespace-only chunks: {len(empty)}")
for cid in empty[:10]:
    print(f"  {cid}")

# 5. fast offset sanity (length screen only)
bad_basis = Counter(r["offset_basis"] for r in rows if r["offset_basis"] != "normalized_text")
bad_span = [r["chunk_id"] for r in rows
            if not r["header_repeated"] and r["end"] - r["start"] != len(r["text"])]
bad_parts = [r["chunk_id"] for r in rows
            if r["header_repeated"] and not (0 <= r["part_index"] < r["part_count"])]
dupes = [cid for cid, n in Counter(r["chunk_id"] for r in rows).items() if n > 1]
print(f"\noffset_basis values other than normalized_text: {dict(bad_basis) or 'none'}")
print(f"ordinary chunks failing the fast length screen (end - start != len(text)): {len(bad_span)}")
for cid in bad_span[:10]:
    print(f"  {cid}")
print(f"table parts with broken part_index/part_count: {len(bad_parts)}")
print(f"duplicate chunk_ids: {len(dupes)}")

# 6. deep offset proof on a small sample: rebuild the normalized source text
#    with the repo's own parser and compare the actual characters.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path.cwd()))
try:
    from app.corpus.filling_parser import to_blocks
except ImportError as exc:
    print(f"\ndeep offset check skipped (run from the repo root with the project venv): {exc}")
    sys.exit(0)

sample = []
for pred, n in ((lambda r: r["oversized"], 3),
                (lambda r: r["header_repeated"], 3),
                (lambda r: r["is_table"] and not r["oversized"] and not r["header_repeated"], 3),
                (lambda r: not r["is_table"], 3)):
    sample.extend([r for r in rows if pred(r)][:n])

print(f"\ndeep offset check on {len(sample)} sampled chunks (re-parses the source filing):")
ok = fail = missing = 0
norm_cache = {}
for r in sample:
    acc_flat = r["accession"].replace("-", "")
    src = next((p for p in (DATA_DIR / "raw" / r["ticker"]).glob(acc_flat + ".*")
                if p.suffix != ".json"), None)
    if src is None:
        print(f"  MISSING SOURCE  {r['ticker']} {r['form']} {r['filed']} ({acc_flat}.* not found)")
        missing += 1
        continue
    if src not in norm_cache:
        raw = src.read_text(encoding="utf-8", errors="replace")
        _, norm_cache[src] = to_blocks(raw, src.suffix.lower())
    expected = norm_cache[src][r["start"]:r["end"]]
    if r["header_repeated"]:
        good = r["text"].endswith(expected) and len(r["text"]) > len(expected)
    else:
        good = r["text"] == expected
    label = "OK     " if good else "MISMATCH"
    print(f"  {label} {r['ticker']} {r['form']} {r['filed']} section {r['item_id']}"
        f" chunk {r['chunk_index']} table={r['is_table']} header_repeated={r['header_repeated']}")
    if good:
        ok += 1
    else:
        fail += 1
print(f"deep check: {ok} ok, {fail} mismatched, {missing} missing source")
