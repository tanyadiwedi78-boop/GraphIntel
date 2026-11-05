"""Append-only JSONL events. One writer per log file in this first build.
"""
import json
import os
from pathlib import Path


class JsonlEvents:
    def __init__(self, path, run_id):
        self.path = Path(path)
        self.run_id = run_id

    def __call__(self, event):
        record = dict(event)
        record["run_id"] = self.run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
