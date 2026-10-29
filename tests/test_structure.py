from pathlib import Path

from app.structure import EXPECTED_PATHS, missing_paths

ROOT = Path(__file__).resolve().parent.parent


def test_every_expected_path_exists():
    assert missing_paths(ROOT) == ()


def test_missing_paths_reports_what_is_gone(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "__init__.py").write_text("", encoding="utf-8")
    missing = missing_paths(tmp_path)
    assert "app/config.py" in missing
    assert "app/__init__.py" not in missing
    assert len(missing) == len(EXPECTED_PATHS) - 1
