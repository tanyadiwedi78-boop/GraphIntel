"""Anthropic-only extraction: offline plan, one guarded batch, resumed collection.

No other provider, automatic repair, submission retry or hidden pre-warm call.
Prices are token-derived USD calculations, not an account billing statement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import unicodedata
import uuid
from decimal import Decimal, InvalidOperation
from datetime import datetime
from pathlib import Path
from urllib.request import HTTPRedirectHandler, Request, build_opener

from app.extraction.batch_storage import (MAX_REQUESTS, MAX_BATCH_BYTES,
    MAX_PLAN_BYTES, MAX_SOURCE_LINE_BYTES, MAX_RESULT_LINE_BYTES, MAX_SOURCE_CHUNKS, MAX_RESULTS_BYTES,
    json_lines, read_plan, checked_results)
from app.config import load_settings
from app.extraction.validate import ExtractionResponse
from app.extraction.runner import (PROMPT_VERSION, SCHEMA_VERSION,
                                   build_prompt, build_retry_prompt, build_record, make_work_key)
from app.graph.ontology import load_ontology, parse_ontology

MODEL = "claude-fable-5-2"
ORIGIN = "https://api.anthropic.com"
BATCH_PATH = "/v1/messages/batches"
VERSION = "2026-10-21"
# Official standard global prices, with 50% batch discount, USD per million.
RATES = {"input_tokens": Decimal("4"), "output_tokens": Decimal("20"),
         "ephemeral_5m_input_tokens": Decimal("5"),
         "ephemeral_1h_input_tokens": Decimal("8"),
         "cache_read_input_tokens": Decimal("0.10")}
PRICE_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"
CONTEXT_CEILING = 2_000_000  # intentionally exceeds the published 1M context, not an estimate
RECIPE = "anthropic-batch-cache-normalize-v2"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_new(path, value):
    """Never overwrite a plan, claim, receipt, snapshot or measured result."""
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if Path(path).name == "plan.json" and len(text.encode("utf-8")) > MAX_PLAN_BYTES:
        raise ValueError("plan exceeds local byte limit")
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def money(value):
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        raise ValueError("budget must be a positive USD amount") from None
    if not number.is_finite() or number <= 0:
        raise ValueError("budget must be a positive USD amount")
    return number


def model_name(max_tokens, ttl):
    return "anthropic:" + MODEL + ":" + digest([RECIPE, max_tokens, ttl])[:16]


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        raise ValueError("API redirect refused; key was not forwarded")


def anthropic_schema(ontology):
    """Closed JSON output objects, ontology enums, local confidence bounds."""
    entity = {"type": "object", "additionalProperties": False,
              "properties": {"name": {"type": "string"},
                             "type": {"type": "string", "enum": list(ontology.entity_types)}},
              "required": ["name", "type"]}
    edge = {"type": "object", "additionalProperties": False,
            "properties": {"relation": {"type": "string", "enum": sorted(ontology.relations)},
                           "source": {"type": "string"}, "target": {"type": "string"},
                           "evidence_quote": {"type": "string"},
                           "confidence": {"type": "number", "description": "Finite confidence from 0 to 1 inclusive; checked locally."}},
            "required": ["relation", "source", "target", "evidence_quote", "confidence"]}
    return {"type": "object", "additionalProperties": False,
            "properties": {"entities": {"type": "array", "items": entity},
                           "relationships": {"type": "array", "items": edge}},
            "required": ["entities", "relationships"]}


def clean_name(name):
    # No fuzzy similarity, suffix stripping or cross-company/cross-document merge.
    return " ".join(unicodedata.normalize("NFC", name).split())


def source_aliases(text):
    """Two explicit SEC collective-definition forms, only in this input chunk.

    A collective alias is a local filing-company reference, NOT a named
    subsidiary and not proof of an ownership relationship between alias names.
    Unrecognized wording stays unchanged for review, rather than guessed.
    """
    company = r"[A-Z][A-Za-z0-9 &'.,-]{0,100}?(?:Inc\.|Corp\.|Corporation|Ltd\.|Limited)"
    quote = r'[“"]'
    endquote = r'[”"]'
    patterns = [
        r"accounts of (?P<company>" + company + r") and its wholly owned subsidiaries \(collectively "
        + quote + r"(?P<alias>[^”\"]+)" + endquote + r" or the " + quote + r"Company" + endquote + r"\)",
        r"Each of the terms the " + quote + r"Company" + endquote + r" and "
        + quote + r"(?P<alias>[^”\"]+)" + endquote
        + r" as used herein refers collectively to (?P<company>" + company + r") and its wholly owned subsidiaries",
    ]
    aliases, evidence = {}, []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            canonical_name = clean_name(match.group("company"))
            names = [canonical_name, clean_name(match.group("alias")), "Company", "the Company"]
            for name in names:
                if name in aliases and aliases[name] != canonical_name:
                    # Ambiguous chunk definition: do not normalize anything.
                    return {}, [{"reason": "ambiguous_source_aliases_no_merge"}]
                aliases[name] = canonical_name
            evidence.append({"reason": "source_alias_definition", "canonical_name": canonical_name, "alias": match.group("alias"),
                             "source_quote": match.group(0)})
    return aliases, evidence


def normalize_response(raw_text, chunk, ontology):
    """Validate shape FIRST, then normalize supported local identities only.

    Shared ontology, endpoint, type and evidence validation still runs afterward.
    Raw provider output is kept unchanged in provider-results.jsonl.
    """
    try:
        data = ExtractionResponse.model_validate_json(raw_text).model_dump()
    except Exception:
        return raw_text, []  # shared validator will preserve the exact rejection
    aliases, audit = source_aliases(chunk["text"])
    names, entities, by_name = {}, [], {}
    for entity in data["entities"]:
        original = entity["name"]
        name = clean_name(original)
        kind = entity["type"]
        if name in aliases and kind in ("Company", "Subsidiary"):
            name = aliases[name]
            kind = "Company"
        if name in by_name and by_name[name]["type"] != kind:
            # Conflicting types are not repaired by guesswork.
            return raw_text, audit + [{"reason": "conflicting_types_no_merge", "name": name}]
        names[original] = name
        if name not in by_name:
            fixed = {"name": name, "type": kind}
            by_name[name] = fixed
            entities.append(fixed)
        else:
            audit.append({"reason": "duplicate_entity", "name": original, "canonical_name": name})
        if name != original or kind != entity["type"]:
            audit.append({"reason": "source_alias_or_spacing", "name": original,
                          "canonical_name": name, "type": kind})
    edges, by_edge = [], {}
    for edge in data["relationships"]:
        # Do not invent a missing endpoint. Shared validator must reject it.
        edge["source"] = names.get(edge["source"], edge["source"])
        edge["target"] = names.get(edge["target"], edge["target"])
        if edge["relation"] == "OWNS" and edge["source"] == edge["target"] and edge["source"] in aliases.values():
            audit.append({"reason": "alias_self_ownership_removed", "edge": dict(edge)})
            continue
        key = (edge["relation"], edge["source"], edge["target"])
        if key in by_edge:
            kept = by_edge[key]
            # Both quotes must still pass the shared evidence validator, even
            # when only one edge is saved. Never hide an invalid duplicate.
            from app.extraction.validate import validate_extraction
            single = json.dumps({"entities": entities, "relationships": [edge]})
            if not validate_extraction(single, chunk, ontology)[0]:
                return raw_text, audit + [{"reason": "invalid_duplicate_not_hidden"}]
            kept["confidence"] = min(kept["confidence"], edge["confidence"])
            audit.append({"reason": "duplicate_edge", "edge": dict(edge)})
        else:
            by_edge[key] = edge
            edges.append(edge)
    return json.dumps({"entities": entities, "relationships": edges}), audit


def message_params(chunk, ontology, max_tokens, ttl, feedback=None):
    # Preserve the exact p3 prompt words. Split only at the invariant TEXT fence.
    full_prompt = build_prompt(chunk, ontology)
    prefix, text = full_prompt.split("\nTEXT:\n", 1)
    if feedback:
        repaired = build_retry_prompt(full_prompt, feedback)
        text += repaired[len(full_prompt):]
    return {"model": MODEL, "max_tokens": max_tokens, "inference_geo": "global",
            "output_config": {"format": {"type": "json_schema", "schema": anthropic_schema(ontology)}},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prefix + "\nTEXT:\n",
                 "cache_control": {"type": "ephemeral", "ttl": ttl}},
                {"type": "text", "text": text}]}]}


def planning_estimate(requests, chunks):
    """Offline heuristic for comparison ONLY, never paid admission authority.

    Four UTF-8 bytes per token is an explicit rough planning assumption, not
    Claude's tokenizer or a bound. Request JSON includes schema and formatting.
    No cache hits are assumed; highest selected write rate prices all input.
    """
    items, documents = [], {}
    total = Decimal(0)
    for request in requests:
        params = request["params"]
        ttl = params["messages"][0]["content"][0]["cache_control"]["ttl"]
        rate = RATES["ephemeral_1h_input_tokens"] if ttl == "1h" else RATES["ephemeral_5m_input_tokens"]
        byte_count = len(canonical(params).encode("utf-8"))
        tokens = (byte_count + 3) // 4
        cost = (tokens * rate + params["max_tokens"] * RATES["output_tokens"]) / Decimal(1000000)
        document = chunks[request["custom_id"]]["accession"]
        items.append({"custom_id": request["custom_id"], "document": document,
                      "request_utf8_bytes": byte_count, "estimated_input_tokens": tokens,
                      "output_token_cap": params["max_tokens"], "estimated_no_hit_cost_usd": str(cost)})
        documents[document] = str(Decimal(documents.get(document, "0")) + cost)
        total += cost
    return {"kind": "offline_heuristic_not_a_bound", "can_authorize_submit": False,
            "basis": "ceil(request JSON UTF-8 bytes / 4), all input at selected cache-write rate, full output cap, no cache hits; uncalibrated for this model",
            "estimated_no_hit_cost_usd": str(total), "documents": documents, "requests": items,
            "strict_admission_note": "Estimate never replaces strict reservation; low-budget submission remains blocked. Token-count endpoint is also an estimate, not a documented error bound."}


def prepare_plan(chunks_path, ontology, max_chunks, max_tokens=4096, ttl="1h", wanted_ids=None, cache_dir=None):
    if isinstance(max_chunks, bool) or max_chunks < 1 or max_chunks > MAX_REQUESTS:
        raise ValueError("max_chunks must be from 1 to 1000 (subject to local byte limits)")
    if isinstance(max_tokens, bool) or not 1 <= max_tokens <= 128000:
        raise ValueError("max_tokens must be from 1 to 128000")
    if ttl not in ("5m", "1h"):
        raise ValueError("cache TTL must be 5m or 1h")
    if wanted_ids is not None and (not wanted_ids or len(set(wanted_ids)) != len(wanted_ids)):
        raise ValueError("chunk IDs must be nonempty and unique")
    source_before = Path(chunks_path).stat()
    hashes, seen = {}, set()
    for chunk in json_lines(chunks_path, MAX_SOURCE_LINE_BYTES):
        for field in ("chunk_id", "text", "accession", "form", "period"):
            if not isinstance(chunk.get(field), str) or not chunk[field]:
                raise ValueError("chunk needs a nonempty string " + field)
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,180}", chunk["chunk_id"]):
            raise ValueError("unsafe chunk ID for a result filename")
        if chunk["chunk_id"] in seen:
            raise ValueError("duplicate source chunk ID: " + chunk["chunk_id"])
        if len(seen) >= MAX_SOURCE_CHUNKS:
            raise ValueError("source exceeds local chunk-count limit")
        seen.add(chunk["chunk_id"])
        accession = chunk["accession"]
        if accession not in hashes:
            hashes[accession] = hashlib.sha256()
        content = json.dumps([chunk["chunk_id"], chunk["text"]], ensure_ascii=False)
        hashes[accession].update(content.encode("utf-8") + b"\n")
    if not seen:
        raise ValueError("no source chunks")
    if wanted_ids is not None:
        missing = set(wanted_ids) - seen
        if missing:
            raise ValueError("selected chunk IDs missing: " + ", ".join(sorted(missing)))
    name = model_name(max_tokens, ttl)
    requests, mapping, skipped = [], {}, []
    request_bytes = len(canonical({"requests": []}).encode("utf-8"))
    for chunk in json_lines(chunks_path, MAX_SOURCE_LINE_BYTES):
        if wanted_ids is not None and chunk["chunk_id"] not in wanted_ids:
            continue
        chunk["document_hash"] = hashes[chunk["accession"]].hexdigest()
        if cache_dir is not None and compatible_cache(cache_dir, chunk, ontology, name):
            skipped.append(chunk["chunk_id"])
            continue
        if len(requests) >= max_chunks:
            if wanted_ids is not None:
                raise ValueError("selected uncached chunk IDs exceed max_chunks")
            break
        key = make_work_key(chunk, ontology.fingerprint(), SCHEMA_VERSION, PROMPT_VERSION, name)
        custom_id = "chunk_" + digest([chunk["chunk_id"], key])[:48]
        request = {"custom_id": custom_id,
                   "params": message_params(chunk, ontology, max_tokens, ttl)}
        request_bytes += len(canonical(request).encode("utf-8")) + (1 if requests else 0)
        if request_bytes > MAX_BATCH_BYTES:
            raise ValueError("batch exceeds local byte limit; select fewer or smaller chunks")
        requests.append(request)
        mapping[custom_id] = chunk
    source_after = Path(chunks_path).stat()
    if (source_before.st_size, source_before.st_mtime_ns) != (source_after.st_size, source_after.st_mtime_ns):
        raise ValueError("source changed during preparation; no plan was created")
    body = {"requests": requests}
    if len(canonical(body).encode("utf-8")) > MAX_BATCH_BYTES:
        raise ValueError("batch exceeds local byte limit")
    # Token counting is only an estimate. For the strict admission reservation,
    # assume the entire documented context is input at the highest chosen cache
    # write rate on EVERY request, plus full capped output. No assumed cache hits.
    highest_input_rate = RATES["ephemeral_1h_input_tokens"] if ttl == "1h" else RATES["ephemeral_5m_input_tokens"]
    per_request = (CONTEXT_CEILING * highest_input_rate + max_tokens * RATES["output_tokens"]) / Decimal(1000000)
    plan = {"recipe": RECIPE, "model": MODEL, "model_name": name,
            "schema_version": SCHEMA_VERSION, "prompt_version": PROMPT_VERSION,
            "ontology": {"entity_types": list(ontology.entity_types),
                         "relations": {key: {"source": ends[0], "target": ends[1]}
                                       for key, ends in ontology.relations.items()},
                         "edge_fields": list(ontology.edge_fields)},
            "requests": requests, "chunks": mapping, "ttl": ttl,
            "cache_dir": str(Path(cache_dir).resolve()) if cache_dir else None,
            "skipped_cached_chunks": skipped, "round": 0, "feedback": {},
            "max_tokens": max_tokens, "price_source": PRICE_SOURCE,
            "strict_reservation_usd": str(per_request * len(requests)),
            "budget_scope": "this immutable job only, not the account or other jobs",
            "pricing_checked_on": "2026-11-10",
            "cache_note": "512-token minimum; hits best-effort, writes may cost more; inspect returned usage"}
    document_reservations = {}
    for chunk in mapping.values():
        accession = chunk["accession"]
        document_reservations[accession] = str(Decimal(document_reservations.get(accession, "0")) + per_request)
    plan["document_reservations_usd"] = document_reservations
    plan["planning_estimate"] = planning_estimate(requests, mapping)
    plan["plan_id"] = digest(plan)
    return plan


def compatible_cache(cache_dir, chunk, ontology, name):
    key = make_work_key(chunk, ontology.fingerprint(), SCHEMA_VERSION, PROMPT_VERSION, name)
    path = Path(cache_dir) / (chunk["chunk_id"] + "__" + key + ".json")
    if not path.exists():
        return False
    try:
        record = read_json(path)
    except (OSError, ValueError):
        return False
    expected = {"status": "valid", "provider_used": "anthropic", "model": name,
                "chunk_id": chunk["chunk_id"], "work_key": key,
                "ontology_fingerprint": ontology.fingerprint(),
                "schema_version": SCHEMA_VERSION, "prompt_version": PROMPT_VERSION}
    return all(record.get(field) == value for field, value in expected.items())


def prepare_retry(parent_dir):
    """Offline, bounded repair plan. Never submits or spends automatically."""
    parent = Path(parent_dir)
    plan = read_plan(parent / "plan.json")
    ontology = verify_plan(plan)
    summary = read_json(parent / "summary.json")
    if plan.get("round", 0) >= 2:
        raise ValueError("two repair rounds exhausted; review source instead of retrying forever")
    next_plan = dict(plan)
    next_plan.pop("plan_id")
    next_plan["parent_job"] = str(parent.resolve())
    next_plan["parent_plan_id"] = plan["plan_id"]
    next_plan["root_plan_id"] = plan.get("root_plan_id", plan["plan_id"])
    next_plan["round"] = plan.get("round", 0) + 1
    next_plan["requests"], next_plan["chunks"], next_plan["feedback"] = [], {}, {}
    setup_errors = ("authentication_error", "permission_error", "invalid_request_error", "billing_error")
    summary_by_id = {}
    for result in summary["requests"]:
        summary_by_id[result["custom_id"]] = result
    raw_path = parent / "provider-results.jsonl"
    with checked_results(raw_path, plan["chunks"]) as rows:
        for row in rows:
            custom_id = row["custom_id"]
            result = summary_by_id[custom_id]
            if result["status"] in ("valid", "valid_reused"):
                continue
            raw = row["result"]
            if raw["type"] == "errored":
                error = raw.get("error", {})
                detail = error.get("error", error)
                if detail.get("type") in setup_errors:
                    raise ValueError("batch setup error; fix it before preparing another paid job")
            chunk = plan["chunks"][custom_id]
            feedback = result.get("problems") or []
            if result.get("stop_reason") == "max_tokens":
                # Same cap would likely cut off again; do not hide a budget increase.
                raise ValueError("output cap reached; review token cap and budget before a new plan")
            next_plan["chunks"][custom_id] = chunk
            next_plan["feedback"][custom_id] = feedback
            next_plan["requests"].append({"custom_id": custom_id, "params": message_params(
                chunk, ontology, plan["max_tokens"], plan["ttl"], feedback)})
    if not next_plan["requests"]:
        raise ValueError("nothing needs a repair batch")
    next_plan["strict_reservation_usd"] = str(Decimal(plan["strict_reservation_usd"]) *
        len(next_plan["requests"]) / len(plan["requests"]))
    unit = Decimal(plan["strict_reservation_usd"]) / len(plan["requests"])
    docs = {}
    for chunk in next_plan["chunks"].values():
        doc = chunk["accession"]
        docs[doc] = str(Decimal(docs.get(doc, "0")) + unit)
    next_plan["document_reservations_usd"] = docs
    next_plan["planning_estimate"] = planning_estimate(next_plan["requests"], next_plan["chunks"])
    next_plan["plan_id"] = digest(next_plan)
    return next_plan


def verify_plan(plan):
    supplied_id = plan.get("plan_id")
    unsigned = dict(plan)
    unsigned.pop("plan_id", None)
    if supplied_id != digest(unsigned):
        raise ValueError("plan changed: prepare and review a new job")
    if type(plan.get("round")) is not int or not 0 <= plan["round"] <= 2:
        raise ValueError("invalid repair round")
    if plan.get("recipe") != RECIPE or plan.get("model") != MODEL:
        raise ValueError("unsupported plan recipe/model")
    if plan.get("schema_version") != SCHEMA_VERSION or plan.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("plan contract differs from current code")
    if type(plan.get("max_tokens")) is not int or not 1 <= plan["max_tokens"] <= 128000:
        raise ValueError("invalid output cap")
    if plan.get("ttl") not in ("5m", "1h"):
        raise ValueError("invalid cache TTL")
    ontology = parse_ontology(plan["ontology"])
    if plan.get("model_name") != model_name(plan["max_tokens"], plan["ttl"]):
        raise ValueError("model recipe mismatch")
    requests = plan["requests"]
    if len(canonical({"requests": requests}).encode("utf-8")) > MAX_BATCH_BYTES:
        raise ValueError("batch exceeds local byte limit")
    if len(requests) > MAX_REQUESTS or len(requests) != len(plan["chunks"]):
        raise ValueError("bad request mapping")
    ids = set()
    for request in requests:
        custom_id = request["custom_id"]
        if custom_id in ids or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", custom_id):
            raise ValueError("bad custom_id")
        ids.add(custom_id)
        chunk = plan["chunks"][custom_id]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,180}", chunk.get("chunk_id", "")):
            raise ValueError("unsafe chunk ID")
        key = make_work_key(chunk, ontology.fingerprint(), SCHEMA_VERSION, PROMPT_VERSION, plan["model_name"])
        if custom_id != "chunk_" + digest([chunk["chunk_id"], key])[:48]:
            raise ValueError("custom_id does not bind the selected source")
        if request["params"] != message_params(chunk, ontology, plan["max_tokens"], plan["ttl"], plan.get("feedback", {}).get(custom_id)):
            raise ValueError("request differs from shared prompt contract")
    rate = RATES["ephemeral_1h_input_tokens"] if plan["ttl"] == "1h" else RATES["ephemeral_5m_input_tokens"]
    reserve = (CONTEXT_CEILING * rate + plan["max_tokens"] * RATES["output_tokens"]) * len(requests) / Decimal(1000000)
    if reserve != Decimal(plan["strict_reservation_usd"]):
        raise ValueError("reservation mismatch")
    if "planning_estimate" in plan and plan["planning_estimate"] != planning_estimate(requests, plan["chunks"]):
        raise ValueError("planning estimate differs from exact requests")
    return ontology


class BatchAPI:
    """No redirect, automatic retry, secret logging or alternate provider."""
    def __init__(self):
        self.key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not self.key:
            raise ValueError("ANTHROPIC_API_KEY is missing; keep it in local .env, never chat")
        self.opener = build_opener(NoRedirect())

    def request(self, method, path, body=None):
        if not path.startswith(BATCH_PATH) and path != "/v1/messages/count_tokens":
            raise ValueError("unexpected API path")
        data = None if body is None else canonical(body).encode("utf-8")
        request = Request(ORIGIN + path, data=data, method=method,
                          headers={"x-api-key": self.key, "anthropic-version": VERSION,
                                   "Content-Type": "application/json"})
        with self.opener.open(request, timeout=120) as response:
            return response.read().decode("utf-8")


    def download_results(self, path, destination):
        if not re.fullmatch(r"/v1/messages/batches/msgbatch_[A-Za-z0-9]+/results", path):
            raise ValueError("unexpected results path")
        request = Request(ORIGIN + path, headers={"x-api-key": self.key,
                          "anthropic-version": VERSION}, method="GET")
        partial = Path(str(destination) + ".partial-" + uuid.uuid4().hex)
        with self.opener.open(request, timeout=120) as response:
            with partial.open("xb") as handle:
                downloaded = 0
                while True:
                    data = response.read(65536)
                    if not data:
                        break
                    downloaded += len(data)
                    if downloaded > MAX_RESULTS_BYTES:
                        raise ValueError("results exceed local byte limit; partial download preserved")
                    if shutil.disk_usage(partial.parent).free < len(data) + 64 * 1024 * 1024:
                        raise ValueError("not enough free disk; partial download preserved")
                    handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        if Path(destination).exists():
            raise ValueError("results already downloaded; partial preserved")
        partial.rename(destination)


def submit_job(job_dir, approved_plan, budget_usd, allow_paid, api=None, max_calls=0):
    job = Path(job_dir)
    plan = read_plan(job / "plan.json")
    verify_plan(plan)
    if not allow_paid or approved_plan != plan["plan_id"]:
        raise ValueError("submission needs explicit paid approval of this plan ID")
    if not plan["requests"]:
        raise ValueError("all selected chunks are cached; no batch is needed")
    if type(max_calls) is not int or max_calls < len(plan["requests"]):
        raise ValueError("MAX_EXTRACTION_CALLS must cover every request in this job; 0 permits none")
    ontology = parse_ontology(plan["ontology"])
    if plan.get("cache_dir"):
        for chunk in plan["chunks"].values():
            if compatible_cache(plan["cache_dir"], chunk, ontology, plan["model_name"]):
                raise ValueError("a planned chunk is already valid in cache; prepare a new plan instead of paying twice")
    budget = money(budget_usd)
    reserve = money(plan["strict_reservation_usd"])
    if reserve > budget:
        raise ValueError("strict no-hit reservation $" + str(reserve) + " exceeds approved job budget; request-specific estimates are not guaranteed bounds and cannot authorize submission. Do not raise approval merely to bypass this gate.")
    api = api or BatchAPI()
    if plan.get("cache_dir"):
        claims = Path(plan["cache_dir"]) / "anthropic_claims"
        claims.mkdir(parents=True, exist_ok=True)
        for custom_id, chunk in plan["chunks"].items():
            claim_path = claims / (custom_id + ".json")
            if claim_path.exists():
                old_claim = read_json(claim_path)
                if old_claim.get("plan_id") not in (plan.get("parent_plan_id"), plan.get("root_plan_id")):
                    raise ValueError("chunk already has a submission claim; reconcile before a new paid job")
            else:
                write_new(claim_path, {"plan_id": plan["plan_id"], "job_dir": str(job.resolve())})
            if plan.get("round", 0) > 0:
                write_new(claims / (custom_id + "-round-" + str(plan["round"]) + ".json"),
                          {"plan_id": plan["plan_id"], "job_dir": str(job.resolve())})
    # Exclusive durable claim BEFORE network. Any interruption/unknown response
    # blocks resubmission. Reconcile with Console, never blindly repeat POST.
    write_new(job / "submission-claim.json", {"plan_id": approved_plan,
              "budget_usd": str(budget), "reserved_usd": str(reserve),
              "state": "possibly_submitted_do_not_retry"})
    raw = api.request("POST", BATCH_PATH, {"requests": plan["requests"]})
    with (job / "submission-response.txt").open("x", encoding="utf-8") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    receipt = json.loads(raw)
    batch_id = receipt.get("id", "")
    if not re.fullmatch(r"msgbatch_[A-Za-z0-9]+", batch_id):
        raise ValueError("unexpected receipt; reconcile Console before any resubmission")
    write_new(job / "receipt.json", receipt)
    return receipt


def usage_cost(usage):
    """Missing, malformed or incompatible usage is unknown, never zero."""
    if not isinstance(usage, dict):
        return None
    fields = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    for key in fields:
        value = usage.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
    creation = usage.get("cache_creation")
    if not isinstance(creation, dict):
        if usage["cache_creation_input_tokens"] != 0:
            return None
        creation = {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0}
    for key in ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"):
        value = creation.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
    if creation["ephemeral_5m_input_tokens"] + creation["ephemeral_1h_input_tokens"] != usage["cache_creation_input_tokens"]:
        return None
    cost = Decimal(0)
    for key in ("input_tokens", "output_tokens", "cache_read_input_tokens"):
        cost += usage[key] * RATES[key]
    for key in ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"):
        cost += creation[key] * RATES[key]
    return str(cost / Decimal(1000000))


def import_results(job_dir, rows):
    """Import complete results out of order by custom_id, using the SAME validator.

    Raw results must already be saved before this function is called. No network.
    Every terminal error is preserved; no automatic resubmission or repair.
    """
    job = Path(job_dir)
    plan = read_plan(job / "plan.json")
    ontology = verify_plan(plan)
    with checked_results(rows, plan["chunks"]) as checked_rows:
        return import_checked_results(job, plan, ontology, checked_rows)


def import_checked_results(job, plan, ontology, rows):
    """The existing validator/accounting flow, with only one raw answer in RAM."""
    totals = {"model": MODEL, "batch_id": read_json(job / "receipt.json")["id"],
              "calculated_cost_usd": "0", "unknown_cost_count": 0,
              "cost_basis": "documented batch rates applied to reported usage, not invoice",
              "price_source": PRICE_SOURCE, "documents": {}, "counts": {},
              "requests": [], "cache_read_input_tokens": 0,
              "cache_creation_input_tokens": 0, "reported_total_input_tokens": 0,
              "reported_output_tokens": 0, "reported_usage": {}, "usage_missing_fields": {}}
    cost_sum = Decimal(0)
    for row in rows:
        custom_id = row["custom_id"]
        chunk = plan["chunks"][custom_id]
        result = row["result"]
        outcome = result["type"]
        item = {"custom_id": custom_id, "chunk_id": chunk["chunk_id"],
                "status": outcome, "calculated_cost_usd": None}
        document = chunk["accession"]
        doc = totals["documents"].setdefault(document, {"requests": 0, "calculated_cost_usd": "0",
                                                       "unknown_cost_count": 0, "reported_usage": {}, "usage_missing_fields": {}})
        doc["requests"] += 1
        cost = "0"  # provider docs: errored/canceled/expired requests are not billed
        if outcome == "succeeded":
            message = result["message"]
            cost = usage_cost(message.get("usage"))
            item["usage"] = message.get("usage")
            reported = message.get("usage")
            fields = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
            for field in fields:
                value = reported.get(field) if isinstance(reported, dict) else None
                if type(value) is int and value >= 0:
                    doc["reported_usage"][field] = doc["reported_usage"].get(field, 0) + value
                    totals["reported_usage"][field] = totals["reported_usage"].get(field, 0) + value
                else:
                    for target in (doc, totals):
                        target["usage_missing_fields"][field] = target["usage_missing_fields"].get(field, 0) + 1
            creation = reported.get("cache_creation") if isinstance(reported, dict) else None
            for field in ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"):
                value = creation.get(field) if isinstance(creation, dict) else None
                if type(value) is int and value >= 0:
                    for target in (doc, totals):
                        target["reported_usage"][field] = target["reported_usage"].get(field, 0) + value
                elif isinstance(reported, dict) and reported.get("cache_creation_input_tokens") == 0:
                    for target in (doc, totals):
                        target["reported_usage"].setdefault(field, 0)
                else:
                    for target in (doc, totals):
                        target["usage_missing_fields"][field] = target["usage_missing_fields"].get(field, 0) + 1
            item["stop_reason"] = message.get("stop_reason")
            item["message_id"] = message.get("id")
            valid_stop = message.get("stop_reason") == "end_turn"
            correct_model = message.get("model") == MODEL
            text = "".join(block["text"] for block in message.get("content", []) if block.get("type") == "text")
            # Refusal/truncation/wrong-model replies must not become valid even
            # if a partial text block accidentally contains parseable JSON.
            original_text = text
            if not valid_stop or not correct_model:
                text = ""
            normalized, normalization = normalize_response(text, chunk, ontology)
            usage = message.get("usage") or {}
            record = build_record(normalized, chunk, ontology, plan["model_name"], plan.get("round", 0) + 1)
            record["normalization"] = normalization
            for entity in record["entities"]:
                aliases = [entity["name"]]
                for event in normalization:
                    if event.get("canonical_name") == entity["name"]:
                        for field in ("name", "alias"):
                            if event.get(field) and event[field] not in aliases:
                                aliases.append(event[field])
                entity["aliases"] = aliases
            record["raw_provider_text"] = original_text
            record["batch_id"] = totals["batch_id"]
            record["custom_id"] = custom_id
            record["anthropic_usage"] = message.get("usage")
            record["calculated_cost_usd"] = cost
            record["cost_basis"] = totals["cost_basis"]
            record["provider_stop_reason"] = message.get("stop_reason")
            record["provider_message_id"] = message.get("id")
            record["timing_basis"] = "asynchronous batch; no per-request latency measurement"
            directory = job / "extractions"
            directory.mkdir(exist_ok=True)
            path = directory / (chunk["chunk_id"] + "__" + record["work_key"] + ".json")
            if path.exists():
                if read_json(path) != record:
                    raise ValueError("existing result differs; nothing overwritten")
            else:
                write_new(path, record)
            if record["status"] == "valid" and plan.get("cache_dir"):
                cache = Path(plan["cache_dir"])
                cache.mkdir(parents=True, exist_ok=True)
                cache_path = cache / path.name
                if not cache_path.exists():
                    write_new(cache_path, record)
                elif not compatible_cache(cache, chunk, ontology, plan["model_name"]):
                    raise ValueError("cache path conflict; old result preserved")
            item["status"] = record["status"]
            item["problems"] = record["problems"]
        item["calculated_cost_usd"] = cost
        if cost is None:
            totals["unknown_cost_count"] += 1
            doc["unknown_cost_count"] += 1
        else:
            cost_sum += Decimal(cost)
            doc["calculated_cost_usd"] = str(Decimal(doc["calculated_cost_usd"]) + Decimal(cost))
        status = item["status"]
        totals["counts"][status] = totals["counts"].get(status, 0) + 1
        totals["requests"].append(item)
    # Provider results can arrive out of order; summary metadata follows the plan.
    items_by_id = {}
    for item in totals["requests"]:
        items_by_id[item["custom_id"]] = item
    totals["requests"] = []
    for custom_id in plan["chunks"]:
        totals["requests"].append(items_by_id[custom_id])
    claim_path = job / "submission-claim.json"
    claim = read_json(claim_path) if claim_path.exists() else {}
    totals["approved_job_budget_usd"] = claim.get("budget_usd")
    totals["reserved_token_cost_usd"] = claim.get("reserved_usd")
    totals["known_calculated_cost_usd"] = str(cost_sum)
    parent_total = "0"
    if plan.get("parent_job"):
        parent_summary = read_json(Path(plan["parent_job"]) / "summary.json")
        parent_total = parent_summary.get("chain_calculated_cost_usd")
    totals["chain_calculated_cost_usd"] = (str(Decimal(parent_total) + cost_sum)
        if parent_total is not None and totals["unknown_cost_count"] == 0 else None)
    totals["calculated_cost_usd"] = str(cost_sum) if totals["unknown_cost_count"] == 0 else None
    for doc in totals["documents"].values():
        doc["known_calculated_cost_usd"] = doc["calculated_cost_usd"]
        if doc["unknown_cost_count"]:
            doc["calculated_cost_usd"] = None
    # Known usage is independent from whether every billing field is present.
    known = totals["reported_usage"]
    totals["reported_total_input_tokens"] = sum(known.get(field, 0) for field in
        ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    totals["reported_output_tokens"] = known.get("output_tokens", 0)
    totals["cache_read_input_tokens"] = known.get("cache_read_input_tokens", 0)
    totals["cache_creation_input_tokens"] = known.get("cache_creation_input_tokens", 0)
    for target in [totals] + list(totals["documents"].values()):
        target["usage_complete"] = not bool(target["usage_missing_fields"])
    parent_docs = {}
    chain_complete = True
    if plan.get("parent_job"):
        if "chain_documents" in parent_summary:
            parent_docs = parent_summary["chain_documents"]
            chain_complete = parent_summary.get("chain_documents_complete", True)
        elif plan.get("round") == 1:
            parent_docs = parent_summary["documents"]
        else:
            # Historical multi-round summary has no per-document ancestry.
            parent_docs = parent_summary["documents"]
            chain_complete = False
    totals["chain_documents"] = merge_document_totals(parent_docs, totals["documents"], chain_complete)
    totals["chain_documents_complete"] = chain_complete
    return totals


def merge_document_totals(previous, current, complete=True):
    merged = {}
    for document in sorted(set(previous) | set(current)):
        rows = [source[document] for source in (previous, current) if document in source]
        known = sum((Decimal(row.get("known_calculated_cost_usd", "0")) for row in rows), Decimal(0))
        unknown = sum(row.get("unknown_cost_count", 0) for row in rows)
        item = {"requests": sum(row.get("requests", 0) for row in rows),
                "known_calculated_cost_usd": str(known), "unknown_cost_count": unknown,
                "calculated_cost_usd": str(known) if complete and unknown == 0 and all(row.get("calculated_cost_usd") is not None for row in rows) else None,
                "reported_usage": {}, "usage_missing_fields": {}}
        for row in rows:
            for key in ("reported_usage", "usage_missing_fields"):
                for field, value in row.get(key, {}).items():
                    item[key][field] = item[key].get(field, 0) + value
        item["usage_complete"] = complete and all(row.get("usage_complete", not row.get("usage_missing_fields")) for row in rows)
        merged[document] = item
    return merged


def batch_timing(status):
    """Provider batch wall time includes queue/processing, not chunk latency."""
    timing = {"created_at": status.get("created_at"), "ended_at": status.get("ended_at"),
              "elapsed_seconds": None,
              "scope": "provider batch created_at to ended_at, including queue time; not per-chunk inference latency"}
    try:
        start = datetime.fromisoformat(timing["created_at"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(timing["ended_at"].replace("Z", "+00:00"))
        if start.tzinfo is not None and end.tzinfo is not None and end >= start:
            timing["elapsed_seconds"] = (end - start).total_seconds()
    except (AttributeError, TypeError, ValueError):
        pass  # Missing/malformed native timestamps are unknown, never invented.
    return timing


def collect_job(job_dir, api=None):
    job = Path(job_dir)
    if (job / "summary.json").exists():
        return read_json(job / "summary.json")
    receipt = read_json(job / "receipt.json")
    batch_id = receipt["id"]
    if not re.fullmatch(r"msgbatch_[A-Za-z0-9]+", batch_id):
        raise ValueError("invalid batch ID")
    api = api or BatchAPI()
    status = json.loads(api.request("GET", BATCH_PATH + "/" + batch_id))
    if status.get("id") != batch_id:
        raise ValueError("status returned wrong batch")
    # Content-addressed snapshots avoid overwriting earlier observations.
    snapshot = job / ("status-" + digest(status)[:16] + ".json")
    if not snapshot.exists():
        write_new(snapshot, status)
    if status.get("processing_status") != "ended":
        return {"processing_status": status.get("processing_status"), "batch_id": batch_id,
                "next_step": "collect again later; no new batch was submitted"}
    raw_path = job / "provider-results.jsonl"
    if not raw_path.exists():
        if hasattr(api, "download_results"):
            api.download_results(BATCH_PATH + "/" + batch_id + "/results", raw_path)
        else:
            # Injected offline test transport only.
            raw = api.request("GET", BATCH_PATH + "/" + batch_id + "/results")
            if len(raw.encode("utf-8")) > MAX_RESULTS_BYTES:
                raise ValueError("results exceed local byte limit")
            with raw_path.open("x", encoding="utf-8") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
    summary = import_results(job, raw_path)
    summary["batch_timing"] = batch_timing(status)
    write_new(job / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Anthropic-only Fable 5.2 batch extraction. prepare is offline; submit spends only with approval.")
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare", help="offline immutable plan, no key or API needed")
    prepare.add_argument("--job-dir", required=True)
    prepare.add_argument("--max-chunks", type=int, default=10)
    prepare.add_argument("--chunk-ids")
    prepare.add_argument("--max-output-tokens", type=int, default=4096)
    prepare.add_argument("--cache-ttl", choices=["5m", "1h"], default="1h")
    retry = sub.add_parser("prepare-retry", help="offline feedback repair plan, at most two rounds; separate paid approval")
    retry.add_argument("--parent-job", required=True)
    retry.add_argument("--job-dir", required=True)
    submit = sub.add_parser("submit", help="one paid batch; no auto retry, repair or fallback")
    submit.add_argument("--job-dir", required=True)
    submit.add_argument("--approve-plan", required=True)
    submit.add_argument("--budget-usd", required=True)
    submit.add_argument("--allow-paid", action="store_true")
    collect = sub.add_parser("collect", help="read batch state, preserve raw results, validate and measure")
    collect.add_argument("--job-dir", required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            settings = load_settings()
            ids = None
            if args.chunk_ids is not None:
                ids = [piece.strip() for piece in args.chunk_ids.split(",") if piece.strip()]
            plan = prepare_plan(settings.data_dir / "chunks" / "chunks.jsonl",
                                load_ontology(settings.ontology_path), args.max_chunks,
                                args.max_output_tokens, args.cache_ttl, ids, settings.data_dir / "extraction")
            job = Path(args.job_dir)
            job.mkdir(parents=True, exist_ok=False)
            write_new(job / "plan.json", plan)
            print(json.dumps({"plan_id": plan["plan_id"], "requests": len(plan["requests"]),
                              "strict_reservation_usd": plan["strict_reservation_usd"],
                              "budget_scope": plan["budget_scope"], "cache_note": plan["cache_note"],
                              "document_reservations_usd": plan["document_reservations_usd"],
                              "skipped_cached_chunks": plan["skipped_cached_chunks"],
                              "planning_estimate": plan["planning_estimate"]}, indent=2))
        elif args.action == "prepare-retry":
            plan = prepare_retry(args.parent_job)
            job = Path(args.job_dir)
            job.mkdir(parents=True, exist_ok=False)
            write_new(job / "plan.json", plan)
            print(json.dumps({"plan_id": plan["plan_id"], "requests": len(plan["requests"]),
                              "strict_reservation_usd": plan["strict_reservation_usd"],
                              "round": plan["round"], "planning_estimate": plan["planning_estimate"]}, indent=2))
        else:
            # Only the normal local settings loader reads .env. Never output it.
            settings = load_settings()
            if args.action == "submit":
                result = submit_job(args.job_dir, args.approve_plan, args.budget_usd, args.allow_paid,
                                    max_calls=settings.max_extraction_calls)
            else:
                result = collect_job(args.job_dir)
            print(json.dumps(result, indent=2))
    except Exception as error:
        # Do not print HTTP response bodies, request headers or secret values.
        if isinstance(error, (ValueError, FileNotFoundError, FileExistsError)):
            print("Stopped:", error)
        else:
            print("Stopped:", type(error).__name__, ". Preserve the job; reconcile any submission claim before retrying.")
        return 1
    return 0
