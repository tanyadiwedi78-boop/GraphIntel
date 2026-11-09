"""CLI wiring for the Gemini-only extraction script. No network, no real API key.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

GOOD_FLAGS = ["--gemini-assume-free", "--gemini-max-rpm", "100", "--gemini-max-tpm", "100000"]


def write_chunks(data_dir, count):
    chunks_dir = data_dir / "chunks"
    chunks_dir.mkdir(parents=True)
    with open(chunks_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        number = 0
        while number < count:
            number = number + 1
            chunk = {
                "chunk_id": f"chunk-{number}",
                "text": f"Item 1A. Acme builds the Widget Engine number {number} for customers.",
                "accession": "0000000000-26-000001",
                "form": "10-K",
                "period": "2026-01-31",
            }
            f.write(json.dumps(chunk) + "\n")


def run_script(data_dir, extra_args=None, extra_env=None):
    args = [sys.executable, "scripts/run_extraction.py"]
    if extra_args is not None:
        args = args + extra_args
    env = dict(os.environ)
    env["DATA_DIR"] = str(data_dir)
    env["MAX_EXTRACTION_CALLS"] = "100"
    # Empty string, not a missing variable: the script loads .env in the repo
    # root, and dotenv fills a MISSING variable back in from a developer's real
    # .env but never overrides one that is already set. Tests must never see a
    # real key, or the "no key" test would make live calls on a keyed machine.
    env["GEMINI_API_KEY"] = ""
    env["GROQ_API_KEY"] = ""
    env["OPENAI_API_KEY"] = ""
    if extra_env is not None:
        env.update(extra_env)
    return subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, env=env)


def test_missing_chunks_file_explains_what_to_do(tmp_path):
    result = run_script(tmp_path, GOOD_FLAGS, {"GEMINI_API_KEY": "fake-key"})
    assert result.returncode == 1
    assert "build_chunks" in result.stdout


def test_zero_call_cap_refuses_to_run(tmp_path):
    write_chunks(tmp_path, 3)
    result = run_script(tmp_path, GOOD_FLAGS, {"MAX_EXTRACTION_CALLS": "0", "GEMINI_API_KEY": "fake-key"})
    assert result.returncode == 1
    assert "no calls are allowed" in result.stdout


def test_live_calls_need_the_assume_free_confirmation(tmp_path):
    write_chunks(tmp_path, 3)
    flags = ["--gemini-max-rpm", "100", "--gemini-max-tpm", "100000"]
    result = run_script(tmp_path, flags, {"GEMINI_API_KEY": "fake-key"})
    assert result.returncode == 1
    assert "--gemini-assume-free" in result.stdout


def test_live_calls_need_explicit_rate_limits(tmp_path):
    write_chunks(tmp_path, 3)
    result = run_script(tmp_path, ["--gemini-assume-free"], {"GEMINI_API_KEY": "fake-key"})
    assert result.returncode == 1
    assert "--gemini-max-rpm" in result.stdout


def test_missing_api_key_is_a_setup_problem_not_a_crash(tmp_path):
    write_chunks(tmp_path, 3)
    result = run_script(tmp_path, GOOD_FLAGS)
    assert result.returncode == 1
    assert "Gemini setup problem" in result.stdout
    assert "GEMINI_API_KEY" in result.stdout
    assert "Traceback" not in result.stderr


def test_groq_provider_needs_its_own_key(tmp_path):
    write_chunks(tmp_path, 3)
    result = run_script(tmp_path, GOOD_FLAGS + ["--provider", "groq"])
    assert result.returncode == 1
    assert "Groq setup problem" in result.stdout
    assert "GROQ_API_KEY" in result.stdout
    assert "Traceback" not in result.stderr


def test_groq_provider_uses_the_same_cost_gates(tmp_path):
    write_chunks(tmp_path, 3)
    flags = ["--gemini-max-rpm", "100", "--gemini-max-tpm", "100000", "--provider", "groq"]
    result = run_script(tmp_path, flags, {"GROQ_API_KEY": "fake-key"})
    assert result.returncode == 1
    assert "Live Groq calls stay blocked" in result.stdout
    assert "--gemini-assume-free" in result.stdout


def test_openai_provider_needs_its_own_key(tmp_path):
    write_chunks(tmp_path, 3)
    flags = GOOD_FLAGS + ["--provider", "openai", "--openai-model", "test-model-1",
                          "--openai-allow-paid"]
    result = run_script(tmp_path, flags)
    assert result.returncode == 1
    assert "OpenAI setup problem" in result.stdout
    assert "OPENAI_API_KEY" in result.stdout
    assert "Traceback" not in result.stderr


def test_openai_provider_needs_paid_approval_not_assume_free(tmp_path):
    write_chunks(tmp_path, 3)
    flags = ["--gemini-assume-free", "--gemini-max-rpm", "100", "--gemini-max-tpm", "100000",
             "--provider", "openai", "--openai-model", "test-model-1"]
    result = run_script(tmp_path, flags, {"OPENAI_API_KEY": "fake-key"})
    assert result.returncode == 1
    assert "Live OpenAI calls stay blocked" in result.stdout
    assert "--openai-allow-paid" in result.stdout


def test_openai_provider_needs_an_explicit_model_id(tmp_path):
    # No --openai-model flag and no config entry: the run must refuse. An
    # empty config file stands in so the repo's real config cannot supply one.
    write_chunks(tmp_path, 3)
    empty_config = tmp_path / "empty-config.yaml"
    empty_config.write_text("{}", encoding="utf-8")
    flags = ["--gemini-max-rpm", "100", "--gemini-max-tpm", "100000",
             "--provider", "openai", "--openai-allow-paid",
             "--config", str(empty_config)]
    result = run_script(tmp_path, flags, {"OPENAI_API_KEY": "fake-key"})
    assert result.returncode == 1
    assert "--openai-model" in result.stdout


def test_fallback_must_differ_from_primary(tmp_path):
    write_chunks(tmp_path, 3)
    result = run_script(tmp_path, GOOD_FLAGS + ["--fallback-provider", "gemini"],
                        {"GEMINI_API_KEY": "fake-key"})
    assert result.returncode == 1
    assert "different provider" in result.stdout


def test_fallback_provider_is_gated_and_keyed_too(tmp_path):
    write_chunks(tmp_path, 3)
    flags = GOOD_FLAGS + ["--fallback-provider", "groq"]
    result = run_script(tmp_path, flags, {"GEMINI_API_KEY": "fake-key"})
    assert result.returncode == 1
    assert "Groq setup problem" in result.stdout
    assert "GROQ_API_KEY" in result.stdout


def test_openai_fallback_still_needs_paid_approval(tmp_path):
    write_chunks(tmp_path, 3)
    flags = GOOD_FLAGS + ["--fallback-provider", "openai", "--openai-model", "gpt-6-luna"]
    result = run_script(tmp_path, flags, {"GEMINI_API_KEY": "fake-key"})
    assert result.returncode == 1
    assert "Live OpenAI calls stay blocked" in result.stdout
    assert "--openai-allow-paid" in result.stdout


def test_unknown_provider_is_rejected(tmp_path):
    write_chunks(tmp_path, 3)
    result = run_script(tmp_path, GOOD_FLAGS + ["--provider", "other"])
    assert result.returncode != 0


def test_config_file_can_supply_the_openai_model_id(tmp_path):
    # The exact OpenAI model id may come from configs/extraction.yaml instead of
    # --openai-model. Proof: with the id only in the config, the model gate
    # passes and the run stops at the NEXT gate (paid approval) instead.
    write_chunks(tmp_path, 3)
    config = tmp_path / "extraction-config.yaml"
    config.write_text("extraction:\n  providers:\n    - {name: openai, model: gpt-6-luna}\n",
                      encoding="utf-8")
    flags = ["--gemini-max-rpm", "100", "--gemini-max-tpm", "100000",
             "--provider", "openai", "--config", str(config)]
    result = run_script(tmp_path, flags, {"OPENAI_API_KEY": "fake-key"})
    assert result.returncode == 1
    assert "--openai-allow-paid" in result.stdout
    assert "needs the exact model id" not in result.stdout


def test_unknown_fallback_cause_in_config_is_rejected(tmp_path):
    write_chunks(tmp_path, 3)
    config = tmp_path / "bad-config.yaml"
    config.write_text("extraction:\n  fallback_on: [rate_limit, bad_cause]\n", encoding="utf-8")
    result = run_script(tmp_path, GOOD_FLAGS + ["--config", str(config)], {"GEMINI_API_KEY": "fake-key"})
    assert result.returncode == 1
    assert "bad_cause" in result.stdout


def test_missing_explicit_config_file_is_rejected(tmp_path):
    write_chunks(tmp_path, 3)
    result = run_script(tmp_path, GOOD_FLAGS + ["--config", str(tmp_path / "not-there.yaml")],
                        {"GEMINI_API_KEY": "fake-key"})
    assert result.returncode == 1
    assert "Config file not found" in result.stdout
