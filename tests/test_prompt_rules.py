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


def test_stricter_risk_rule_bumps_version_and_keeps_v2_frozen():
    # The approved stricter disclaimer-risk policy is rule 10 in a NEW prompt
    # version. The v2 template stays byte-frozen so saved v2 jobs (including
    # their pending recovery) rebuild exactly the prompts they were run with.
    from app.extraction.pipeline import PROMPT_RULES_BY_VERSION
    assert PROMPT_VERSION == "expanded-source-v3"
    assert set(PROMPT_RULES_BY_VERSION) == {"expanded-source-v2", "expanded-source-v3"}
    v2 = PROMPT_RULES_BY_VERSION["expanded-source-v2"]
    v3 = PROMPT_RULES_BY_VERSION["expanded-source-v3"]
    assert v3.startswith(v2)  # rules 1-9 unchanged; v3 only appends
    assert "substantive, company-specific statement" in v3
    assert "substantive, company-specific statement" not in v2


def test_build_prompt_defaults_to_current_and_honours_explicit_version(ontology):
    current = build_prompt(make_chunk(), ontology)
    assert "substantive, company-specific statement" in current
    old = build_prompt(make_chunk(), ontology, version="expanded-source-v2")
    assert "substantive, company-specific statement" not in old
    assert "RULES:" in old


def test_build_prompt_rejects_unknown_version(ontology):
    import pytest
    with pytest.raises(ValueError):
        build_prompt(make_chunk(), ontology, version="no-such-version")


def test_pipeline_rebuilds_the_plan_version_prompt(tmp_path, ontology):
    # A resume of a job frozen under v2 must send the v2 prompt even though
    # the current default is v3; a v3 plan sends the v3 prompt.
    import json as jsonlib
    from app.evaluation.extraction_benchmark import identity
    from app.extraction.pipeline import Pipeline
    recipe = JSONProvider('gemini', 'gemini-fixture-model', 1024).recipe
    job = tmp_path / "job"
    prepare(job, [make_chunk()], ontology, recipe, policy=retry_policy(0))
    provider = JSONProvider('gemini', 'gemini-fixture-model', 1024)
    v3_pipe = Pipeline(job, ontology, provider)
    assert "substantive, company-specific statement" in v3_pipe.prompt(make_chunk())
    plan_path = job / "plan.json"
    plan = jsonlib.loads(plan_path.read_text(encoding="utf-8"))
    plan["prompt_version"] = "expanded-source-v2"
    plan.pop("plan_hash")
    plan["plan_hash"] = identity(plan)
    plan_path.write_text(jsonlib.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    v2_pipe = Pipeline(job, ontology, provider)
    assert "substantive, company-specific statement" not in v2_pipe.prompt(make_chunk())
    assert "A specific potential risk can be a Risk" in v2_pipe.prompt(make_chunk())
