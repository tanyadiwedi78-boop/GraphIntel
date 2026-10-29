from __future__ import annotations

from pathlib import Path
from typing import Tuple

EXPECTED_PATHS: Tuple[str, ...] = (
    "app/__init__.py",
    "app/config.py",
    "app/structure.py",
    "app/corpus/__init__.py",
    "app/corpus/data_limits.py",
    "app/corpus/edgar.py",
    "app/corpus/filling_parser.py",
    "app/corpus/chunking.py",
    "app/graph/__init__.py",
    "app/graph/ontology.py",
    "app/extraction/__init__.py",
    "app/retrieval/__init__.py",
    "app/answering/__init__.py",
    "app/api/__init__.py",
    "app/evaluation/__init__.py",
    "app/updates/__init__.py",
    "app/community/__init__.py",
    "configs/companies.json",
    "configs/ontology.json",
    "configs/filling_sections.yaml",
    "scripts/check_setup.py",
    "scripts/download_fillings.py",
    "scripts/build_chunks.py",
    "tests/test_structure.py",
    "tests/test_edgar.py",
    "tests/test_filling_parser.py",
    "tests/test_chunking.py",
    "tests/test_offsets.py",
    "tests/fixtures/table_heavy.html",
    "architecture.md",
    "requirements.txt",
    "pytest.ini",
    ".gitignore",
    "readme.md",
    ".env",
)


def missing_paths(root: Path) -> Tuple[str, ...]:
    """Expected paths that do not exist under root."""
    root = Path(root)
    missing = []
    for path in EXPECTED_PATHS:
        if not (root / path).exists():
            missing.append(path)
    return tuple(missing)


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    missing = missing_paths(root)
    if missing:
        print("Missing from the project structure:")
        for path in missing:
            print(f"  {path}")
        return 1
    print(f"Structure OK: all {len(EXPECTED_PATHS)} expected files and folders exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
