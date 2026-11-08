"""Talk to the Gemini API: one prompt in, JSON text out, with cost gates and pacing."""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import random
import time
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"
KEY_ENV_VAR = "GEMINI_API_KEY"

RETRY_LATER = "retry_later"   # 429, 408, 5xx: wait, then try again
STOP_RUN = "stop_run"         # 400, 401, 402, 403 and similar: fix the setup first


class ProviderError(Exception):
    """One failed model call."""

    def __init__(self, status_code: int, message: str = "", retry_after_seconds: Optional[float] = None):
        super().__init__(f"provider error {status_code}: {message}")
        self.status_code = status_code
        self.message = message
        self.retry_after_seconds = retry_after_seconds


class ChunkFailedError(ProviderError):
    """This chunk's answer cannot succeed by retrying the same request.

    The runner fails this one chunk and moves on: it does not burn the
    remaining attempts on an identical request, and it does not stop the
    whole run for a problem that belongs to one chunk.
    """


def classify_status(status_code: int) -> str:
    if status_code == 429 or status_code == 408:
        return RETRY_LATER
    if status_code >= 500:
        return RETRY_LATER
    return STOP_RUN


def backoff_seconds(failed_attempt_number: int, base_seconds: float, cap_seconds: float) -> float:
    seconds = base_seconds
    count = 1
    while count < failed_attempt_number:
        seconds = seconds * 2
        count = count + 1
    if seconds > cap_seconds:
        seconds = cap_seconds
    return seconds


def wait_before_retry(
    failed_attempt_number: int,
    error: ProviderError,
    base_seconds: float,
    cap_seconds: float,
    jitter_fraction: float,
) -> float:
    seconds = backoff_seconds(failed_attempt_number, base_seconds, cap_seconds)
    if error.retry_after_seconds is not None:
        if error.retry_after_seconds > seconds:
            seconds = error.retry_after_seconds
    if jitter_fraction > 0:
        seconds = seconds + seconds * jitter_fraction * random.random()
    return seconds


def number_or_none(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


def extraction_json_schema(ontology):
    """JSON schema for generation, using only fields the API documents.

    Strictness lives in our validator (validate.py). This schema is the first
    fence, not the last one: the provider schema cannot express "source must
    name an entity in the list" or "the quote must appear in the chunk", so
    those checks stay local.
    """
    entity = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "type": {"type": "string", "enum": list(ontology.entity_types)},
        },
        "required": ["name", "type"],
    }
    relationship = {
        "type": "object",
        "properties": {
            "relation": {"type": "string", "enum": sorted(ontology.relations.keys())},
            "source": {"type": "string"},
            "target": {"type": "string"},
            "evidence_quote": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["relation", "source", "target", "evidence_quote", "confidence"],
    }
    return {
        "type": "object",
        "properties": {
            "entities": {"type": "array", "items": entity},
            "relationships": {"type": "array", "items": relationship},
        },
        "required": ["entities", "relationships"],
    }


class MinuteLimiter:
    """Stay under the account's per-minute request and token limits."""

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


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class GeminiClient:
    """Callable client: client(prompt) returns the model's JSON text."""

    def __init__(self, ontology, model_id="gemini-3.5-flash-lite", timeout_seconds=120,
                max_output_tokens=1024, assume_free=False,
                max_requests_per_minute=None, max_tokens_per_minute=None,
                sleep_func=time.sleep, now_func=time.monotonic):
        api_key = os.environ.get(KEY_ENV_VAR, "").strip()
        if api_key == "":
            raise ValueError(
                KEY_ENV_VAR + " is not set. Create a key in Google AI Studio, then set it "
                "as an environment variable. Never paste it into chat or code.")
        if not model_id.strip():
            raise ValueError("model_id must name a real Gemini model")
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
        except ProviderError:
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
            "generation_config": {"seed": 0, "thinking_level": "low",
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
            except (OSError, http.client.HTTPException):
                # URLError, TimeoutError and dropped connections are all OSError;
                # a half-read response raises http.client.IncompleteRead.
                self.last_metrics["error_type"] = "transport_or_timeout"
                raise ProviderError(503, "Gemini API unavailable or request timed out") from None
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                data = None
            if not isinstance(data, dict):
                self.last_metrics["error_type"] = "response_envelope"
                raise ProviderError(502, "invalid API response envelope") from None
            status = data.get("status")
            if status == "incomplete":
                # Official meaning: completed but cut off, e.g. hitting max_tokens.
                self.last_metrics["error_type"] = "output_cut_off"
                raise ChunkFailedError(
                    400, "answer cut off at the output token limit; raise "
                    "max_output_tokens or use a smaller chunk")
            if status == "requires_continuation":
                # Official meaning: needs another request with continuation_token,
                # which this build does not support. Not an output-limit signal.
                self.last_metrics["error_type"] = "needs_continuation"
                raise ChunkFailedError(
                    400, "provider asked for a continuation step this build does not support")
            if status != "completed" or not isinstance(data.get("steps"), list):
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
