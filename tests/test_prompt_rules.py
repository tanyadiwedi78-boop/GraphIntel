"""Tests for the extraction prompt rules. No network, no API key."""

from app.extraction.runner import PROMPT_VERSION, build_prompt, make_work_key
from app.graph.ontology import load_ontology


def make_chunk():
    return {
        "chunk_id": "chunk-1",
        "text": "Acme develops the Widget Engine for customers.",
        "accession": "0000000000-26-000001",
        "form": "10-K",
        "period": "2026-01-31",
    }


def test_prompt_contains_rules_and_chunk_text():
    ontology = load_ontology("configs/ontology.json")
    prompt = build_prompt(make_chunk(), ontology)
    assert "RULES:" in prompt
    assert "leave the item out" in prompt
    assert "state the relationship" in prompt
    assert make_chunk()["text"] in prompt


def test_prompt_lists_types_and_relations_from_ontology():
    ontology = load_ontology("configs/ontology.json")
    prompt = build_prompt(make_chunk(), ontology)
    for name in ontology.entity_types:
        assert f"- {name}: " in prompt
    for name in sorted(ontology.relations.keys()):
        source_type, target_type = ontology.relations[name]
        assert f"- {name}: {source_type} -> {target_type}" in prompt


def test_prompt_type_meanings_follow_ontology_not_hardcoding():
    class SmallOntology:
        entity_types = ["Company"]
        relations = {}

        def fingerprint(self):
            return "small"

    prompt = build_prompt(make_chunk(), SmallOntology())
    assert "Allowed entity types: Company." in prompt
    assert "FinancialMetric" not in prompt


def test_prompt_version_bumped_for_new_rules():
    # The p2 prompt produced the saved baseline results. The rules edit must
    # use a new version so those results are never mistaken for p3 output.
    assert PROMPT_VERSION != "p2"


def test_work_key_changes_with_prompt_version():
    ontology = load_ontology("configs/ontology.json")
    chunk = make_chunk()
    old_key = make_work_key(chunk, ontology.fingerprint(), "s3", "p2", "m")
    new_key = make_work_key(chunk, ontology.fingerprint(), "s3", PROMPT_VERSION, "m")
    assert old_key != new_key
