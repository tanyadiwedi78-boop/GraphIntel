"""Anthropic-only Fable 5.2 extraction. prepare is offline; submit requires approval."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.extraction.anthropic_batch import main

if __name__ == "__main__":
    raise SystemExit(main())
