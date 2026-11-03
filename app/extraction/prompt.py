"""The extraction prompt. Change the text -> change PROMPT_VERSION, so old
saved results are never mistaken for results of the new prompt."""

from __future__ import annotations

from app.graph.ontology import Ontology

PROMPT_VERSION = "p1"
SCHEMA_VERSION = "s1"


def build_prompt(chunk: dict, ontology: Ontology) -> str:
    lines = []
    lines.append("Extract relationships from the SEC filing text below.")
    lines.append("Reply with JSON only: {\"relations\": [ ... ]}.")
    lines.append("Each relation has: relation, source_name, source_type, target_name, target_type, evidence_quote, confidence (0 to 1).")
    lines.append("Allowed relations (source type -> target type):")
    for name in sorted(ontology.relations.keys()):
        source_type, target_type = ontology.relations[name]
        lines.append(f"- {name}: {source_type} -> {target_type}")
    lines.append("evidence_quote must be copied word for word from the text. Do not guess. If nothing fits, return an empty list.")
    lines.append("")
    lines.append("TEXT:")
    lines.append(chunk["text"])
    return "\n".join(lines)
