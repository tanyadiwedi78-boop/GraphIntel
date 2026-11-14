"""Tests for the extraction prompt rules. No network, no API key.

Migrated from the retired Anthropic runner module to the current
app.extraction.pipeline prompt builder. The old work-key function is gone;
the current guarantee is that the prompt version is stamped into every
immutable plan and changes the plan identity.
"""
import json

from app.extraction.pipeline import PROMPT_VERSION, build_prompt, prepare, retry_policy
from app.extraction.providers import JSONProvider


def make_chunk():
    return {
        "chunk_id": "chunk-1",
        "text": "Acme develops the Widget Engine for customers.",
        "document_hash": "a" * 64,
        "accession": "0000000000-26-000001",
        "form": "10-K",
        "period": "2026-01-31",
    }


def test_prompt_contains_rules_and_chunk_text(ontology):
    prompt = build_prompt(make_chunk(), ontology)
    assert "RULES:" in prompt
    assert "leave the item out" in prompt
    assert "state the relationship" in prompt
    assert make_chunk()["text"] in prompt


def test_prompt_lists_types_and_relations_from_ontology(ontology):
    prompt = build_prompt(make_chunk(), ontology)
    for name in ontology.entity_types:
        assert name + ": " in prompt
    assert "Relations and direction: " + json.dumps(ontology.relations, sort_keys=True) in prompt


def test_prompt_type_meanings_follow_ontology_not_hardcoding():
    class SmallOntology:
        entity_types = ["Company"]
        relations = {}

        def fingerprint(self):
            return "small"

    prompt = build_prompt(make_chunk(), SmallOntology())
    assert "Company" in prompt
    assert "FinancialMetric" not in prompt


def test_prompt_version_bumped_for_new_rules():
    # The p2 prompt produced the saved baseline results. The rules edit must
    # use a new version so those results are never mistaken for p3 output.
    assert PROMPT_VERSION != "p2"


def test_plan_identity_changes_with_prompt_version(tmp_path, ontology):
    # Old work keys are replaced by the immutable plan: prompt_version is
    # stamped into plan.json and plan_hash changes if it changes, so output
    # from another prompt version can never share this plan's identity.
    recipe = JSONProvider('gemini', 'gemini-fixture-model', 1024).recipe
    job = tmp_path / "job"
    plan = prepare(job, [make_chunk()], ontology, recipe, policy=retry_policy(0))
    assert plan["prompt_version"] == PROMPT_VERSION
    from app.evaluation.extraction_benchmark import identity
    changed = dict(plan)
    changed["prompt_version"] = "p2"
    changed.pop("plan_hash")
    original = dict(plan)
    original.pop("plan_hash")
    assert identity(changed) != identity(original)
