from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

MIN_ENTITY_TYPES, MAX_ENTITY_TYPES = 5, 10
MIN_RELATION_TYPES, MAX_RELATION_TYPES = 8, 15


class OntologyError(ValueError):
    """The ontology file is missing, malformed or outside the limits."""


@dataclass(frozen=True)
class Ontology:
    entity_types: Tuple[str, ...]
    relations: Dict[str, Tuple[str, str]]  # name -> (source type, target type)
    edge_fields: Tuple[str, ...]

    def allows(self, relation: str, source_type: str, target_type: str) -> bool:
        
        return self.relations.get(relation) == (source_type, target_type)

    def fingerprint(self) -> str:
        """Changes whenever the ontology changes. Later used in cache keys,
        so an edited ontology never reuses stale results."""
        canonical = json.dumps(
            {
                "e": sorted(self.entity_types),
                "r": sorted(self.relations.items()),
                "f": sorted(self.edge_fields),
            }
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def parse_ontology(raw: dict) -> Ontology:
    try:
        entity_types = tuple(raw["entity_types"])
        relations_raw = raw["relations"]
        edge_fields = tuple(raw["edge_fields"])
    except (KeyError, TypeError):
        raise OntologyError(
            'ontology needs "entity_types", "relations" and "edge_fields"'
        ) from None

    if not all(isinstance(t, str) and t.strip() for t in entity_types):
        raise OntologyError("entity type names must be non-empty strings")
    if len(set(entity_types)) != len(entity_types):
        raise OntologyError("entity type names must be unique")
    if not MIN_ENTITY_TYPES <= len(entity_types) <= MAX_ENTITY_TYPES:
        raise OntologyError(
            f"need {MIN_ENTITY_TYPES} to {MAX_ENTITY_TYPES} entity types, got {len(entity_types)}"
        )

    if not isinstance(relations_raw, dict):
        raise OntologyError('"relations" must be an object')
    relations: Dict[str, Tuple[str, str]] = {}
    for name, ends in relations_raw.items():
        if not isinstance(ends, dict) or "source" not in ends or "target" not in ends:
            raise OntologyError(f"relation {name} needs a source and a target")
        if name != name.upper() or " " in name:
            raise OntologyError(f"relation name {name!r} must be UPPER_CASE with no spaces")
        for end in (ends["source"], ends["target"]):
            if end not in entity_types:
                raise OntologyError(f"relation {name} uses {end!r}, which is not an entity type")
        relations[name] = (ends["source"], ends["target"])
    if not MIN_RELATION_TYPES <= len(relations) <= MAX_RELATION_TYPES:
        raise OntologyError(
            f"need {MIN_RELATION_TYPES} to {MAX_RELATION_TYPES} relation types, got {len(relations)}"
        )

    if not edge_fields or not all(isinstance(f, str) and f for f in edge_fields):
        raise OntologyError("edge_fields must be a non-empty list of names")
    return Ontology(entity_types, relations, edge_fields)


def load_ontology(path: Path) -> Ontology:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise OntologyError(f"ontology file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise OntologyError(f"ontology file is not valid JSON: {exc}") from None
    return parse_ontology(raw)
