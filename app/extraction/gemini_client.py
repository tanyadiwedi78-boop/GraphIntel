from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from app.extraction.errors import ProviderError

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"
KEY_ENV_VAR = "GEMINI_API_KEY"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def number_or_none(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


def extraction_json_schema(ontology):
    """JSON schema for generation, using only fields the API documents.

    Strictness lives in our validator (schema.py). This schema is the first
    fence, not the last one.
    """
    fields = {
        "relation": {"type": "string", "enum": sorted(ontology.relations.keys())},
        "source_name": {"type": "string"},
        "source_type": {"type": "string", "enum": list(ontology.entity_types)},
        "target_name": {"type": "string"},
        "target_type": {"type": "string", "enum": list(ontology.entity_types)},
        "evidence_quote": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }
    relation = {"type": "object", "properties": fields, "required": list(fields.keys())}
    return {
        "type": "object",
        "properties": {"relations": {"type": "array", "items": relation}},
        "required": ["relations"],
    }


class MinuteLimiter:
    """Proactive pacing for ONE process: wait before sending so the chosen
    per-minute limits hold.

    Tracks request starts and estimated input tokens in a rolling 60-second
    window. It cannot see other apps, other machines or earlier runs, and it
    does not track per-day quotas.
    """

    def __init__(self, max_requests_per_minute, max_tokens_per_minute,
                 sleep_func=time.sleep, now_func=time.monotonic):
        if max_requests_per_minute <= 0 or max_tokens_per_minute <= 0:
            raise ValueError("per-minute limits must be positive")
        self.max_requests = max_requests_per_minute
        self.max_tokens = max_tokens_per_minute
        self.sleep = sleep_func
        self.now = now_func
        self.events = []  # (timestamp, estimated input tokens)

    def estimate_input_tokens(self, prompt):
        # Approximate guess only: 2 characters per token is a rough heuristic,
        # not a guaranteed bound (dense JSON or non-English text can tokenize
        # worse). A 429 is still possible; the retry policy handles that.
        # Measured tokens are logged after each real response.
        return max(1, len(prompt) // 2)

    def wait_for_slot(self, estimated_tokens):
        if estimated_tokens > self.max_tokens:
            raise ValueError(
                "one prompt's estimated input tokens exceed your per-minute token "
                "limit, so pacing could never pass: use a smaller chunk or set the "
                "correct limit from your AI Studio rate limit page")
        waited = 0.0
        while True:
            now = self.now()
            self.events = [(t, tok) for (t, tok) in self.events if now - t < 60]
            used_tokens = sum(tok for (_, tok) in self.events)
            if len(self.events) < self.max_requests and used_tokens + estimated_tokens <= self.max_tokens:
                self.events.append((now, estimated_tokens))
                return waited
            self.sleep(1)
            waited = waited + 1


class GeminiClient:
    

    def __init__(self, ontology, model_id="gemini-3.8-flash", timeout_seconds=120,
                 max_output_tokens=1024, assume_free=False,
                 max_requests_per_minute=None, max_tokens_per_minute=None,
                 sleep_func=time.sleep, now_func=time.monotonic):
        api_key = os.environ.get(KEY_ENV_VAR, "").strip()
        if api_key == "":
            raise ValueError(
                KEY_ENV_VAR + " is not set. Create a key in Google AI Studio, then set it "
                "as an environment variable. Never paste it into chat or code."
            )
        if not model_id.strip() or model_id == "echo-placeholder":
            raise ValueError("model_id must name a real Gemini model, not the echo client")
        if timeout_seconds <= 0 or max_output_tokens <= 0:
            raise ValueError("timeout and max_output_tokens must be positive")
        if assume_free and (max_requests_per_minute is None or max_tokens_per_minute is None):
            raise ValueError(
                "live calls need explicit per-minute limits: set max_requests_per_minute "
                "and max_tokens_per_minute from your account's rate limit page in AI Studio")
        self.api_key = api_key
        self.model_id = model_id
        self.schema = extraction_json_schema(ontology)
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.assume_free = assume_free
        if max_requests_per_minute is not None and max_tokens_per_minute is not None:
            self.limiter = MinuteLimiter(max_requests_per_minute, max_tokens_per_minute,
                                        sleep_func, now_func)
        else:
            self.limiter = None
        recipe = json.dumps([model_id, self.schema, max_output_tokens, "gemini-interactions-v1"],
                            sort_keys=True)
        tag = hashlib.sha256(recipe.encode("utf-8")).hexdigest()[:16]
        self.model_name = "gemini:" + model_id + ":" + tag
        self.last_metrics = {}
        self.opener = build_opener(NoRedirect())

    def cost_mode(self):
        if self.assume_free:
            return "assumed_free_unverified"
        return "unpriced_unknown"

    def check_call_allowed(self):
        """Fail-closed gate. Runs before any request is built or sent."""
        if not self.assume_free:
            raise ProviderError(
                402, "cost is not controlled: pass assume_free=True only after checking in "
                "Google AI Studio that this project has no billing account. Paid calls are "
                "not implemented in this build")
        if self.limiter is None:
            raise ProviderError(
                402, "no per-minute limits configured: pass max_requests_per_minute and "
                "max_tokens_per_minute from your account's rate limit page in AI Studio")

    def __call__(self, prompt):
        self.last_metrics = {"provider": "gemini", "model_id": self.model_id,
                            "prompt_tokens": None, "completion_tokens": None,
                            "thought_tokens": None, "cached_tokens": None,
                            "output_tokens_per_sec_end_to_end": None,
                            "cost_basis": self.cost_mode(),
                            "estimated_cost_usd": None,  # no verified prices: unknown, never fake $0
                            "error_type": None}
        self.last_metrics["paced_wait_seconds"] = 0.0
        try:
            self.check_call_allowed()
        except ProviderError as error:
            self.last_metrics["error_type"] = "cost_blocked"
            raise
        estimated_input = self.limiter.estimate_input_tokens(prompt)
        self.last_metrics["estimated_input_tokens"] = estimated_input
        self.last_metrics["paced_wait_seconds"] = self.limiter.wait_for_slot(estimated_input)
        payload = {
            "model": self.model_id,
            "input": prompt,
            "response_format": {"type": "text", "mime_type": "application/json",
                                "schema": self.schema},
            "generation_config": {"seed": 0, "thinking_level": "minimal",
                                "max_output_tokens": self.max_output_tokens},
            "store": False,
        }
        request = Request(ENDPOINT, data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json",
                                "x-goog-api-key": self.api_key}, method="POST")
        start = time.perf_counter()
        try:
            try:
                with self.opener.open(request, timeout=self.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
            except HTTPError as error:
                self.last_metrics["error_type"] = "http_" + str(error.code)
                wait = None
                value = error.headers.get("Retry-After")
                if value is not None:
                    try:
                        wait = number_or_none(float(value))
                    except ValueError:
                        pass
                # Do not put arbitrary server bodies into logs or exception messages.
                raise ProviderError(error.code, "Gemini API HTTP failure", wait) from None
            except (URLError, TimeoutError, socket.timeout) as error:
                self.last_metrics["error_type"] = "transport_or_timeout"
                raise ProviderError(503, "Gemini API unavailable or request timed out") from None
            data = None
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                data = None
            if not isinstance(data, dict):
                self.last_metrics["error_type"] = "response_envelope"
                raise ProviderError(502, "invalid API response envelope") from None
            if data.get("status") == "requires_continuation":
                self.last_metrics["error_type"] = "output_limit"
                raise ProviderError(400, "generation token limit reached; review limits")
            if data.get("status") != "completed" or not isinstance(data.get("steps"), list):
                self.last_metrics["error_type"] = "response_envelope"
                raise ProviderError(502, "invalid API response envelope") from None
            text = None
            for step in data["steps"]:
                if not isinstance(step, dict) or step.get("type") != "model_output":
                    continue
                for part in step.get("content", []):
                    if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                        text = part["text"]
            if text is None:
                self.last_metrics["error_type"] = "response_envelope"
                raise ProviderError(502, "invalid API response envelope") from None
            usage = data.get("usage") or {}
            if not isinstance(usage, dict):
                usage = {}
            metrics = self.last_metrics
            metrics["prompt_tokens"] = number_or_none(usage.get("total_input_tokens"))
            metrics["completion_tokens"] = number_or_none(usage.get("total_output_tokens"))
            metrics["thought_tokens"] = number_or_none(usage.get("total_thought_tokens"))
            metrics["cached_tokens"] = number_or_none(usage.get("total_cached_tokens"))
            latency_ms = (time.perf_counter() - start) * 1000
            if metrics["completion_tokens"] and latency_ms > 0:
                # Output tokens over the WHOLE request time (queue + reading the
                # prompt + generating). Not the model's own decode speed.
                metrics["output_tokens_per_sec_end_to_end"] = round(
                    metrics["completion_tokens"] / (latency_ms / 1000), 2)
            return text
        finally:
            self.last_metrics["latency_ms"] = round((time.perf_counter() - start) * 1000, 3)
