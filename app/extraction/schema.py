"""Check one model response before it is saved as valid."""

from __future__ import annotations

import json
import math
from typing import Dict, List, Tuple

from app.graph.ontology import Ontology


def squash_spaces(text: str) -> str:
    return " ".join(text.split())


def validate_extraction(raw_text: str, chunk: dict, ontology: Ontology) -> Tuple[bool, List[str], List[dict]]:
    """Return (is_valid, problems, edges). `edges` is empty unless valid."""
    problems: List[str] = []

    try:
        data = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return False, ["response is not valid JSON"], []

    if not isinstance(data, dict) or not isinstance(data.get("relations"), list):
        return False, ['response must be an object with a "relations" list'], []

    chunk_text = squash_spaces(chunk["text"])
    edges: List[dict] = []
    position = 0
    for item in data["relations"]:
        position = position + 1
        label = f"relation {position}"
        if not isinstance(item, dict):
            problems.append(f"{label} is not an object")
            continue

        needed = ["relation", "source_name", "source_type", "target_name", "target_type", "evidence_quote", "confidence"]
        missing = []
        for key in needed:
            if key not in item:
                missing.append(key)
        if len(missing) > 0:
            problems.append(f"{label} is missing {', '.join(missing)}")
            continue

        for key in ["source_name", "target_name", "evidence_quote"]:
            if not isinstance(item[key], str) or item[key].strip() == "":
                problems.append(f"{label}: {key} must be a non-empty string")
        for key in ["relation", "source_type", "target_type"]:
            if not isinstance(item[key], str):
                problems.append(f"{label}: {key} must be a string")
        if len(problems) > 0:
            continue
        confidence = item["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            problems.append(f"{label}: confidence must be a number")
        elif not math.isfinite(confidence) or confidence < 0 or confidence > 1:
            problems.append(f"{label}: confidence must be between 0 and 1")

        if not ontology.allows(item["relation"], item["source_type"], item["target_type"]):

        quote = item["evidence_quote"]
        if isinstance(quote, str) and quote.strip() != "":
            if squash_spaces(quote) not in chunk_text:
                problems.append(f"{label}: evidence_quote does not appear in the chunk")

        if len(problems) == 0:
            edge: Dict[str, object] = dict(item)
            # These come from OUR chunk record, never from the model.
            edge["source_chunk_id"] = chunk["chunk_id"]
            edge["filing_accession"] = chunk["accession"]
            edge["form_type"] = chunk["form"]
            edge["period"] = chunk["period"]
            edges.append(edge)

    if len(problems) > 0:
        return False, problems, []
    return True, [], edges
