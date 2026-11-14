"""New two-provider entrypoint wiring. Offline: no API key, no network, no Neo4j."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_cli(arguments):
    return subprocess.run(
        [sys.executable, "scripts/run_extraction.py"] + arguments,
        cwd=ROOT, capture_output=True, text=True)


def test_no_arguments_prints_guidance_without_running():
    result = run_cli([])
    assert result.returncode == 0
    assert "No extraction started." in result.stdout
    assert "phase1 --help" in result.stdout
    assert "Traceback" not in result.stderr


def test_phase1_help_lists_stages_without_api_or_db():
    result = run_cli(["phase1", "--help"])
    assert result.returncode == 0
    assert "demo" in result.stdout
    assert "identity-eval" in result.stdout
    assert "compare-prepare" in result.stdout
    assert "batch-collect" in result.stdout
    assert "Traceback" not in result.stderr
