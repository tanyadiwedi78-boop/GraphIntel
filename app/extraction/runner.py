"""Shared prompt, content identity and validated records. No API calls here."""
from __future__ import annotations
import hashlib
import json
from app.graph.ontology import Ontology
from app.extraction.validate import validate_extraction

SCHEMA_VERSION = "s4-source-contract-v1"
PROMPT_VERSION = "p3-alias-evidence-v1"


# Plain-English meaning for each entity type, shown in the prompt when the type
# exists in the ontology. The ontology stays the source of truth: a type missing
# from configs/ontology.json never appears in the prompt, and a type without a
# meaning here is listed without one.
TYPE_MEANINGS = {
    "Company": "a business acting for itself, such as the filing company or a business it names",
    "Subsidiary": "a company owned or controlled by another company",
    "Industry": "a business sector or market, not one company",
    "Technology": "a specific technology, standard or system",
    "Risk": "a risk, threat or uncertainty the text describes",
    "FinancialMetric": "a financial measure such as revenue, net income or par value per share, whether or not the text states its number; the name of a security itself, such as Common Stock, is not one",
    "Product": "a product or service a company offers",
}

PROMPT_RULES = """RULES:
1. Use only the allowed entity types and relations below.
2. Give an entity a type only when that type truly fits. If no allowed type fits, leave the item out instead of forcing it into the nearest type.
3. A short list, or an empty list, can be the correct answer. Cover pages and boilerplate often contain nothing worth extracting.
4. List every entity that fits a type, even one that has no relationship.
5. A relationship must use two names from your own entities list and follow its direction: relation (source type -> target type).
6. evidence_quote must be copied word for word from the text AND state the relationship you claim. A quote that only repeats the same words is not evidence. A table row naming a stock class does not show that a company reports a financial metric.
7. confidence is your honest estimate, from 0 to 1, that the text supports the item. Do not guess.
8. A definition saying names refer to the same company or collectively to a company and its subsidiaries is not an ownership link between those names. Use one company name for aliases; never turn the company itself into a Subsidiary to make OWNS fit. Do not invent named subsidiaries from an unnamed collective.
9. A specific potential risk can be a Risk even in forward-looking text, but its evidence must explicitly connect that risk to the company and keep words such as potential or may. Do not turn a possible impact into a realized fact. A heading alone states no relationship.
"""


def build_prompt(chunk: dict, ontology: Ontology) -> str:
    types_text = ", ".join(ontology.entity_types)

    meaning_lines = []
    for name in ontology.entity_types:
        if name in TYPE_MEANINGS:
            meaning_lines.append("- " + name + ": " + TYPE_MEANINGS[name])
    meanings_text = "\n".join(meaning_lines)

    relation_lines = []
    for name in sorted(ontology.relations.keys()):
        source_type, target_type = ontology.relations[name]
        relation_lines.append("- " + name + ": " + source_type + " -> " + target_type)
    relations_text = "\n".join(relation_lines)

    return f"""Extract entities and relationships from the SEC filing text below.
Reply with JSON only: {{"entities": [...], "relationships": [...]}}.
Each entity has: name, type.
Each relationship has: relation, source, target, evidence_quote, confidence (0 to 1).

{PROMPT_RULES}

Allowed entity types: {types_text}.
{meanings_text}

Allowed relations (source type -> target type):
{relations_text}

TEXT:
{chunk["text"]}"""


def build_retry_prompt(base_prompt: str, problems: List[str]) -> str:
    """Second-and-later attempt for the same chunk: same task, plus what failed.

    Without this the retry sends the identical prompt, and with a fixed seed the
    model tends to repeat the identical violation - each repeat still costs one
    call. Naming the validator's problems gives the model the information it
    needs to correct the answer instead of reproducing it.
    """
    lines = []
    for problem in problems:
        lines.append("- " + problem)
    return (base_prompt
            + "\n\nYour previous answer was rejected for these reasons:\n"
            + "\n".join(lines)
            + "\nFix these problems; the task and the text above are unchanged."
            + " If fixing a problem would require inventing something the text does"
            + " not say, leave that item out instead - a shorter or empty list can"
            + " be the correct answer.")


def make_work_key(chunk: dict, ontology_fingerprint: str, schema_version: str, prompt_version: str, model_name: str) -> str:
    text_hash = hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()[:16]
    pieces = [text_hash, ontology_fingerprint, schema_version, prompt_version, model_name]
    # Cache identity includes deterministic provenance as well as content.
    pieces.append(json.dumps([chunk.get("chunk_id"), chunk.get("accession"),
                              chunk.get("form"), chunk.get("period")]))
    document_hash = chunk.get("document_hash")
    if document_hash is not None:
        pieces.append(document_hash)
    key = "_".join(pieces)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]



def build_record(raw_text, chunk, ontology, model_name, attempt_number=1):
    valid, problems, relationships, entities = validate_extraction(raw_text, chunk, ontology)
    for entity in entities:
        entity["canonical_name"] = entity["name"]
    for edge in relationships:
        edge["source_chunk_id"] = chunk["chunk_id"]
        edge["filing_accession"] = chunk["accession"]
        edge["form_type"] = chunk["form"]
        edge["period"] = chunk["period"]
        edge["provider"] = "anthropic"
        edge["model"] = model_name
        edge["prompt_version"] = PROMPT_VERSION
    key = make_work_key(chunk, ontology.fingerprint(), SCHEMA_VERSION, PROMPT_VERSION, model_name)
    return {"status": "valid" if valid else "invalid", "provider_used": "anthropic",
            "model": model_name, "work_key": key, "chunk_id": chunk["chunk_id"],
            "document_hash": chunk.get("document_hash"),
            "filing_accession": chunk["accession"], "form_type": chunk["form"], "period": chunk["period"],
            "schema_version": SCHEMA_VERSION, "prompt_version": PROMPT_VERSION,
            "ontology_fingerprint": ontology.fingerprint(), "attempt_number": attempt_number,
            "problems": problems, "entities": entities, "relationships": relationships,
            "cost_usd": None, "raw_response": raw_text if not valid else None}
