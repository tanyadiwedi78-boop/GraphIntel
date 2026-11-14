"""Hand-check extraction results against the source chunk text.

Three uses:

  python scripts/review_extraction.py --chunk-ids id1,id2
      Print each chunk's source text and its saved extraction result together,
      so you can compare them by eye.

  python scripts/review_extraction.py --chunk-ids id1,id2 --write-labels data/labels_sample.json
      Create a labels file with empty lists for these chunks, or add missing
      chunks to a labels file you already started. Entries you filled in are
      never overwritten.

  python scripts/review_extraction.py --labels data/labels_sample.json
      Score the saved results against your labels: precision and recall for
      entities and relationships, with the wrong and the missed items listed.

Fill the labels file by reading each chunk's text, then set reviewed=true.
A reviewed empty answer is a real negative example, not unfinished work.
Unreviewed or legacy entries are never assumed reviewed. "entities" lists what the
source really supports (canonical_name and entity_type); "relationships" lists what
the source really states (relationship_type, source_entity, target_entity). Matching ignores upper/lower case
and extra spaces, nothing else: if the model's name differs from yours, it
counts as wrong and yours counts as missed. Use the canonical company name
only when an explicit source-local alias definition supports that normalization;
otherwise keep the source name. Never fuzzy-match distinct companies.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ConfigError, load_settings
from app.extraction.contracts import CONTRACT_VERSION
from app.extraction.pipeline import PROMPT_VERSION

ENTITY_FIELDS = ["canonical_name", "entity_type"]
RELATIONSHIP_FIELDS = ["relationship_type", "source_entity", "target_entity"]


def read_chunks(chunks_path: Path) -> list:
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            chunks.append(json.loads(line))
    return chunks


def parse_chunk_ids(raw: str) -> list:
    ids = []
    for piece in raw.split(","):
        piece = piece.strip()
        if piece != "":
            ids.append(piece)
    return ids


def find_result(results_dir: str, chunk_id: str):
    """The saved checkpoint record for this chunk in a job directory, or (None, None)."""
    paths = glob.glob(os.path.join(results_dir, "chunk-*.json"))
    for path in sorted(paths):
        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)
        if record.get("chunk_id") == chunk_id:
            return path, record
    return None, None


def normalize(value) -> str:
    return " ".join(str(value).lower().split())


def match_items(predicted: list, expected: list, fields: list):
    """One-to-one matching. Returns (correct, wrong_predictions, missed_expected)."""
    remaining = []
    for item in expected:
        remaining.append(item)
    correct = 0
    wrong = []
    for pred in predicted:
        pred_key = []
        for field_name in fields:
            pred_key.append(normalize(pred.get(field_name, "")))
        hit = None
        for exp in remaining:
            exp_key = []
            for field_name in fields:
                exp_key.append(normalize(exp.get(field_name, "")))
            if exp_key == pred_key:
                hit = exp
        if hit is not None:
            remaining.remove(hit)
            correct = correct + 1
        else:
            wrong.append(pred)
    return correct, wrong, remaining


def ratio_text(correct: int, total: int) -> str:
    if total == 0:
        return "n/a (none)"
    return f"{correct}/{total} = {correct / total:.2f}"


def print_sheet(chunk, result_path, record) -> None:
    print("=" * 60)
    print("chunk_id: " + str(chunk.get("chunk_id")))
    print("ticker: " + str(chunk.get("ticker")) + "  form: " + str(chunk.get("form"))
          + "  accession: " + str(chunk.get("accession")))
    print("item: " + str(chunk.get("item_id")) + "  period: " + str(chunk.get("period")))
    print("-" * 60)
    print("SOURCE TEXT:")
    print(chunk.get("text", ""))
    print("-" * 60)
    if record is None:
        print("EXTRACTION: no saved result for this chunk yet. Run the pipeline prepare/run stages first.")
        return
    print("EXTRACTION from " + str(result_path))
    print("accepted: " + str(record.get("accepted")) + "  contract: " + str(record.get("contract_version"))
          + "  provider: " + str(record.get("provider")) + "  outcome: " + str(record.get("terminal_outcome")))
    print("entities:")
    for entity in record.get("entities", []):
        print("  - " + str(entity.get("canonical_name")) + " (" + str(entity.get("entity_type")) + ")")
    print("relationships:")
    for rel in record.get("relationships", []):
        print("  - " + str(rel.get("source_entity")) + " -" + str(rel.get("relationship_type"))
              + "-> " + str(rel.get("target_entity")))
        print("    evidence: \"" + str(rel.get("evidence")) + "\"")
        print("    confidence: " + str(rel.get("confidence")))


def run_sheets(settings, chunk_ids: list, results_dir=None) -> int:
    chunks_path = settings.data_dir / "chunks" / "chunks.jsonl"
    if not chunks_path.exists():
        print(f"No chunks at {chunks_path}. Run scripts/build_chunks.py first.")
        return 1
    chunks = read_chunks(chunks_path)
    results_dir = str(results_dir or settings.data_dir / "extraction")
    missing = 0
    for chunk_id in chunk_ids:
        found = None
        for chunk in chunks:
            if chunk.get("chunk_id") == chunk_id:
                found = chunk
        if found is None:
            print(f"No chunk with id {chunk_id} in {chunks_path}.")
            missing = missing + 1
            continue
        result_path, record = find_result(results_dir, chunk_id)
        print_sheet(found, result_path, record)
    if missing > 0:
        return 1
    return 0


def run_write_labels(settings, chunk_ids: list, labels_path: Path) -> int:
    labels = {}
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        if not isinstance(labels, dict):
            print(f"{labels_path} is not a JSON object. Fix or move it first.")
            return 1
    added = 0
    for chunk_id in chunk_ids:
        if chunk_id not in labels:
            labels[chunk_id] = {"entities": [], "relationships": [], "notes": "", "reviewed": False}
            added = added + 1
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
        f.write("\n")
    print(f"Labels file: {labels_path} ({added} chunk entries added, {len(labels)} total)")
    print("Fill each entry, then set reviewed to true, including reviewed empty answers. Unreviewed entries never score.")
    print("Use supported source-defined canonical names consistently in labels; no fuzzy alias matching is applied.")
    print("Use canonical names only for explicit source-defined aliases; otherwise use source names. Matching ignores only case/spaces, never fuzzy company identity.")
    return 0


def check_labels_file(labels) -> list:
    """Shape problems in the labels file, as a list of messages (empty = fine)."""
    problems = []
    if not isinstance(labels, dict):
        problems.append("the file must be a JSON object like {\"chunk-id\": {\"entities\": [...], \"relationships\": [...]}}")
        return problems
    if len(labels) == 0:
        problems.append("the file has no chunk entries - create them with --write-labels")
        return problems
    for chunk_id in labels:
        entry = labels[chunk_id]
        if not isinstance(entry, dict):
            problems.append(f"{chunk_id}: the entry must be an object with entities and relationships lists")
            continue
        if "reviewed" in entry and type(entry["reviewed"]) is not bool:
            problems.append(f"{chunk_id}: reviewed must be true or false")
        for field in ["entities", "relationships"]:
            if field not in entry:
                problems.append(f"{chunk_id}: missing the \"{field}\" list")
            elif not isinstance(entry[field], list):
                problems.append(f"{chunk_id}: \"{field}\" must be a list")
    return problems


def run_score(settings, labels_path: Path, results_dir=None) -> int:
    if not labels_path.exists():
        print(f"No labels file at {labels_path}. Create one with --write-labels.")
        return 1
    with open(labels_path, "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        labels = json.loads(raw)
    except json.JSONDecodeError as error:
        print(f"Labels file is not valid JSON: {error}. Fix it or recreate it with --write-labels.")
        return 1
    problems = check_labels_file(labels)
    if len(problems) > 0:
        print("Labels file needs fixing before anything is scored:")
        for problem in problems:
            print("  " + problem)
        return 1
    results_dir = str(results_dir or settings.data_dir / "extraction")
    totals = {"e_pred": 0, "e_correct": 0, "e_expected": 0,
              "r_pred": 0, "r_correct": 0, "r_expected": 0}
    unscored = 0
    scored = 0
    coverage = {"reviewed": 0, "unreviewed": 0, "valid": 0, "missing": 0,
                "invalid_or_failed": 0, "incompatible": 0, "reviewed_empty": 0}
    for chunk_id in labels:
        entry = labels[chunk_id]
        if entry.get("reviewed") is not True:
            print(f"{chunk_id}: not reviewed; set reviewed=true only after checking the source. Not scored.")
            unscored += 1
            coverage["unreviewed"] += 1
            continue
        coverage["reviewed"] += 1
        totals["e_expected"] += len(entry["entities"])
        totals["r_expected"] += len(entry["relationships"])
        if not entry["entities"] and not entry["relationships"]:
            coverage["reviewed_empty"] += 1
        result_path, record = find_result(results_dir, chunk_id)
        reason = None
        if record is None:
            reason = "missing"
        elif record.get("contract_version") != CONTRACT_VERSION:
            reason = "incompatible"
        elif record.get("accepted") is not True:
            reason = "invalid_or_failed"
        if reason is not None:
            coverage[reason] += 1
            print(f"{chunk_id}: {reason}; no accepted output, expected items count as missed in end-to-end recall.")
            continue
        coverage["valid"] += 1
        scored = scored + 1
        print(f"{chunk_id}: scored against {result_path}")
        predicted_entities = record.get("entities", [])
        expected_entities = entry.get("entities", [])
        e_correct, e_wrong, e_missed = match_items(predicted_entities, expected_entities, ENTITY_FIELDS)
        predicted_rels = record.get("relationships", [])
        expected_rels = entry.get("relationships", [])
        r_correct, r_wrong, r_missed = match_items(predicted_rels, expected_rels, RELATIONSHIP_FIELDS)
        print(f"  entities: precision {ratio_text(e_correct, len(predicted_entities))}, "
              f"recall {ratio_text(e_correct, len(expected_entities))}")
        for pred in e_wrong:
            print(f"    wrong extraction: {pred.get('canonical_name')} ({pred.get('entity_type')})")
        for exp in e_missed:
            print(f"    missed: {exp.get('canonical_name')} ({exp.get('entity_type')})")
        print(f"  relationships: precision {ratio_text(r_correct, len(predicted_rels))}, "
              f"recall {ratio_text(r_correct, len(expected_rels))}")
        for pred in r_wrong:
            print(f"    wrong extraction: {pred.get('source_entity')} -{pred.get('relationship_type')}-> {pred.get('target_entity')}")
        for exp in r_missed:
            print(f"    missed: {exp.get('source_entity')} -{exp.get('relationship_type')}-> {exp.get('target_entity')}")
        totals["e_pred"] = totals["e_pred"] + len(predicted_entities)
        totals["e_correct"] = totals["e_correct"] + e_correct
        totals["r_pred"] = totals["r_pred"] + len(predicted_rels)
        totals["r_correct"] = totals["r_correct"] + r_correct
    print("=" * 60)
    print("Coverage:", json.dumps(coverage, sort_keys=True))
    if coverage["reviewed"] == 0:
        print("Nothing reviewed; no quality score. Legacy labels must be explicitly marked reviewed after checking.")
        return 1
    print(f"Accepted output coverage: {coverage['valid']}/{coverage['reviewed']} reviewed chunks")
    print("Precision counts accepted predictions, including false positives on reviewed-empty chunks.")
    print("End-to-end recall includes expected items on missing, failed and incompatible outputs as missed.")
    print(f"Overall across {coverage['reviewed']} reviewed chunks ({scored} accepted outputs):")
    print(f"  entities: precision {ratio_text(totals['e_correct'], totals['e_pred'])}, "
          f"end-to-end recall {ratio_text(totals['e_correct'], totals['e_expected'])}")
    print(f"  relationships: precision {ratio_text(totals['r_correct'], totals['r_pred'])}, "
          f"end-to-end recall {ratio_text(totals['r_correct'], totals['r_expected'])}")
    print("Precision: of what the model extracted, how much is right. "
          "Recall: of what the source really says, how much the model found.")
    print("These numbers describe only the chunks you labeled - not the whole corpus.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Hand-check extraction results against source chunk text.")
    parser.add_argument("--chunk-ids", default=None,
                        help="comma-separated chunk ids to show or to add to a labels file")
    parser.add_argument("--write-labels", default=None, metavar="PATH",
                        help="create or extend a labels JSON file for --chunk-ids")
    parser.add_argument("--labels", default=None, metavar="PATH",
                        help="score saved results against this labels JSON file")
    parser.add_argument("--results-dir", default=None, help="pipeline job directory; default data/extraction")
    args = parser.parse_args()

    try:
        settings = load_settings()
    except ConfigError as error:
        print(f"Settings problem: {error}")
        return 1

    if args.labels is not None:
        return run_score(settings, Path(args.labels), args.results_dir)

    if args.chunk_ids is None:
        print("Give --chunk-ids to review chunks, or --labels to score a labels file.")
        return 1
    chunk_ids = parse_chunk_ids(args.chunk_ids)
    if len(chunk_ids) == 0:
        print("--chunk-ids was given but it has no ids in it.")
        return 1

    if args.write_labels is not None:
        return run_write_labels(settings, chunk_ids, Path(args.write_labels))
    return run_sheets(settings, chunk_ids, args.results_dir)


if __name__ == "__main__":
    raise SystemExit(main())
