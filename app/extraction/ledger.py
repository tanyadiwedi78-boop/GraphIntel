"""Saved extraction results, one small file per chunk and recipe.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional


def make_work_key(chunk: dict, ontology_fingerprint: str, schema_version: str, prompt_version: str, model_name: str) -> str:
    text_hash = hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()[:16]
    pieces = [text_hash, ontology_fingerprint, schema_version, prompt_version, model_name]
    key = "_".join(pieces)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def result_path(results_dir: str, chunk_id: str, work_key: str) -> str:
    return os.path.join(results_dir, chunk_id + "__" + work_key + ".json")


def load_result(results_dir: str, chunk_id: str, work_key: str) -> Optional[dict]:
    path = result_path(results_dir, chunk_id, work_key)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def already_done(results_dir: str, chunk_id: str, work_key: str) -> bool:
    saved = load_result(results_dir, chunk_id, work_key)
    if saved is None:
        return False
    return saved.get("status") == "valid"


def save_result(results_dir: str, chunk_id: str, work_key: str, record: dict) -> None:
    os.makedirs(results_dir, exist_ok=True)
    final_path = result_path(results_dir, chunk_id, work_key)
    temp_path = final_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(record, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, final_path)
