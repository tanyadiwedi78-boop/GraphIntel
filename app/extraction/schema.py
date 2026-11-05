"""Check one model response before it is saved as valid."""

from __future__ import annotations

import json
import math
from typing import Dict, List, Tuple

from app.graph.ontology import Ontology


def squash_spaces(text: str) -> str:
    return " ".join(text.split())


def validate_extraction(raw_text: str, chunk: dict, ontology: Ontology) -> Tuple[bool, List[str], List[dict], List[dict]]:
    """Return (is_valid, problems, edges, entities). `edges` and `entities` are empty unless valid."""
    problems: List[str] = []

    try:
        data = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return False, ["response is not valid JSON"], [], []

    if not isinstance(data, dict) or not isinstance(data.get("entities"), list) or not isinstance(data.get("relationships"), list):
        return False, ['response must be an object with "entities" and "relationships" lists'], [], []

    extra_top = sorted(set(data.keys()) - {"entities", "relationships"})
    if len(extra_top) > 0:
        problems.append(f"unexpected top-level keys: {', '.join(extra_top)}")

    # Entities first: relationships reference them by name.
    entities: List[dict] = []
    by_name: Dict[str, dict] = {}
    position = 0
    for item in data["entities"]:
        position = position + 1
        label = f"entity {position}"
        if not isinstance(item, dict):
            problems.append(f"{label} is not an object")
            continue
        unexpected = sorted(set(item.keys()) - {"name", "type"})
        if len(unexpected) > 0:
            problems.append(f"{label} has unexpected keys: {', '.join(unexpected)}")
            continue
        name = item.get("name")
        entity_type = item.get("type")
        if not isinstance(name, str) or name.strip() == "":
            problems.append(f"{label}: name must be a non-empty string")
            continue
        if not isinstance(entity_type, str) or entity_type not in ontology.entity_types:
            problems.append(f"{label}: type {entity_type!r} is not an allowed entity type")
            continue
        if name in by_name:
            problems.append(f"{label}: duplicate entity name {name!r}")
            continue
        entity = {"name": name, "type": entity_type}
        by_name[name] = entity
        entities.append(entity)

    chunk_text = squash_spaces(chunk["text"])
    edges: List[dict] = []
    position = 0
    for item in data["relationships"]:
        position = position + 1
        label = f"relationship {position}"
        if not isinstance(item, dict):
            problems.append(f"{label} is not an object")
            continue
        allowed = {"relation", "source", "target", "evidence_quote", "confidence"}
        unexpected = sorted(set(item.keys()) - allowed)
        if len(unexpected) > 0:
            problems.append(f"{label} has unexpected keys: {', '.join(unexpected)}")
            continue
        needed = ["relation", "source", "target", "evidence_quote", "confidence"]
        missing = []
        for key in needed:
            if key not in item:
                missing.append(key)
        if len(missing) > 0:
            problems.append(f"{label} is missing {', '.join(missing)}")
            continue

        before = len(problems)
        relation = item["relation"]
        if not isinstance(relation, str):
            problems.append(f"{label}: relation must be a string")
        for key in ["source", "target", "evidence_quote"]:
            if not isinstance(item[key], str) or item[key].strip() == "":
                problems.append(f"{label}: {key} must be a non-empty string")
        confidence = item["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            problems.append(f"{label}: confidence must be a number")
        elif not math.isfinite(confidence) or confidence < 0 or confidence > 1:
            problems.append(f"{label}: confidence must be between 0 and 1")
        if len(problems) > before:
            continue

        # Cross-reference and ontology checks need well-typed fields.
        source = item["source"]
        target = item["target"]
        if source not in by_name:
            problems.append(f"{label}: source {source!r} is not in the entities list")
        if target not in by_name:
            problems.append(f"{label}: target {target!r} is not in the entities list")
        if len(problems) > before:
            continue
        source_type = by_name[source]["type"]
        target_type = by_name[target]["type"]
        if not ontology.allows(relation, source_type, target_type):
            problems.append(
                f"{label}: {source_type} -{relation}-> {target_type} is not in the ontology"
            )

<<<<<<< HEAD
            quote = item["evidence_quote"]
        if isinstance(quote, str) and quote.strip() != "":
            if squash_spaces(quote) not in chunk_text:
                problems.append(f"{label}: evidence_quote does not appear in the chunk")
=======
        quote = item["evidence_quote"]
        if squash_spaces(quote) not in chunk_text:
            problems.append(f"{label}: evidence_quote does not appear in the chunk")
>>>>>>> 6a7897f27694b77c8356d82ed1769bb2ddd5d8ac

        if len(problems) == before:
            edge: Dict[str, object] = {
                "relation": relation,
                "source_name": source,
                "source_type": source_type,
                "target_name": target,
                "target_type": target_type,
                "evidence_quote": quote,
                "confidence": confidence,
            }
            # These come from OUR chunk record, never from the model.
            edge["source_chunk_id"] = chunk["chunk_id"]
            edge["filing_accession"] = chunk["accession"]
            edge["form_type"] = chunk["form"]
            edge["period"] = chunk["period"]
            edges.append(edge)

    if len(problems) > 0:
        return False, problems, [], []
    return True, [], edges, entities
