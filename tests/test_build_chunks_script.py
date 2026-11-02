import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_build_chunks_default_rules_path_exists():
    source = (REPO_ROOT / "scripts" / "build_chunks.py").read_text(encoding="utf-8")
    match = re.search(r'Path\("([^"]*sections[^"]*\.yaml)"\)', source)
    assert match, "build_chunks.py should name its section rules file"
    assert (REPO_ROOT / match.group(1)).exists(), f"missing config file: {match.group(1)}"


def test_build_chunks_entrypoint_gets_past_config_loading(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/build_chunks.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "DATA_DIR": str(tmp_path)},
    )
    assert "section rules not found" not in result.stderr
    assert "No filings under" in result.stdout
    assert result.returncode == 1
