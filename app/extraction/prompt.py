"""The extraction prompt. Change the text -> change PROMPT_VERSION, so old
saved results are never mistaken for results of the new prompt."""

from __future__ import annotations

from app.graph.ontology import Ontology

PROMPT_VERSION = "p2"
SCHEMA_VERSION = "s2"


def build_prompt(chunk: dict, ontology: Ontology) -> str:
    lines = []
    lines.append("Extract entities and relationships from the SEC filing text below.")
    lines.append('Reply with JSON only: {"entities": [...], "relationships": [...]}.')
    lines.append("Each entity has: name, type.")
    lines.append("Allowed entity types: " + ", ".join(ontology.entity_types) + ".")
    lines.append("Each relationship has: relation, source, target, evidence_quote, confidence (0 to 1).")
    lines.append("source and target must be names from your own entities list.")
    lines.append("Allowed relations (source type -> target type):")
    for name in sorted(ontology.relations.keys()):
        source_type, target_type = ontology.relations[name]
        lines.append(f"- {name}: {source_type} -> {target_type}")
    lines.append("List every entity worth remembering, even one that has no relationship.")
    lines.append("evidence_quote must be copied word for word from the text. Do not guess.")
    lines.append("If nothing fits, return empty lists.")
    lines.append("")
    lines.append("TEXT:")
    lines.append(chunk["text"])
    return "\n".join(lines)
