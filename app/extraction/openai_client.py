"""OpenAI client for extraction calls through the chat completions API.

Same contract as GeminiClient and GroqClient: client(prompt) returns the
model's JSON text, client.last_metrics carries the last call's measurements,
and client.model_name feeds the work key, so OpenAI results never mix with
other providers' results. The runner, validator, cache and review tooling
see no difference between the clients.

Two rules make this client different from the free-tier providers:

- The model id is REQUIRED. This build invents no model id: pass the exact
  id from the OpenAI platform docs, for example the model line from a script
  that already runs for you.
- OpenAI API calls bill real money. There is no free tier, so this client
  has no assume_free mode: every call requires allow_paid=True, which the
  CLI only passes after the --openai-allow-paid flag. That flag records the
  user's approval of paid usage; it is not a spending limit and does not
  cap dollars.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import time
from urllib.error import HTTPError
from urllib.request import Request, build_opener

from app.extraction.client import (ChunkFailedError, MinuteLimiter, NoRedirect,
                                   ProviderError, extraction_json_schema,
                                   number_or_none)

ENDPOINT = "https://api.openai.com/v1/chat/completions"
KEY_ENV_VAR = "OPENAI_API_KEY"


class OpenAIClient:
    """Callable client: client(prompt) returns the model's JSON text."""

    def __init__(self, ontology, model_id, timeout_seconds=120,
                 max_output_tokens=1024, allow_paid=False, reasoning_effort=None,
                 max_requests_per_minute=None, max_tokens_per_minute=None,
                 sleep_func=time.sleep, now_func=time.monotonic):
        api_key = os.environ.get(KEY_ENV_VAR, "").strip()
        if api_key == "":
            raise ValueError(
                KEY_ENV_VAR + " is not set. Create a key in the OpenAI platform, then set it "
                "as an environment variable or a line in .env. Never paste it into chat or code.")
        if not model_id.strip():
            raise ValueError("model_id must name a real OpenAI model")
        if timeout_seconds <= 0 or max_output_tokens <= 0:
            raise ValueError("timeout and max_output_tokens must be positive")
        if allow_paid and (max_requests_per_minute is None or max_tokens_per_minute is None):
            raise ValueError(
                "live calls need explicit per-minute limits: set max_requests_per_minute "
                "and max_tokens_per_minute from your organization's limits page in the "
                "OpenAI platform")
        self.api_key = api_key
        self.model_id = model_id
        self.schema = extraction_json_schema(ontology)
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.allow_paid = allow_paid
        # Optional: some OpenAI models accept a reasoning effort setting.
        # Left out of the request unless the caller sets it, because models
        # differ and this build invents no per-model defaults.
        self.reasoning_effort = reasoning_effort
        if max_requests_per_minute is not None and max_tokens_per_minute is not None:
            self.limiter = MinuteLimiter(max_requests_per_minute, max_tokens_per_minute,
                                         sleep_func, now_func)
        else:
            self.limiter = None
        recipe = json.dumps([model_id, self.schema, max_output_tokens, "openai-chat-v1"],
                            sort_keys=True)
        tag = hashlib.sha256(recipe.encode("utf-8")).hexdigest()[:16]
        self.model_name = "openai:" + model_id + ":" + tag
        self.last_metrics = {}
        self.opener = build_opener(NoRedirect())

    def cost_mode(self):
        if self.allow_paid:
            return "paid_approved_by_flag"
        return "paid_not_approved"

    def check_call_allowed(self):
        """Fail-closed gate. Runs before any request is built or sent."""
        if not self.allow_paid:
            raise ProviderError(
                402, "OpenAI calls bill real money and are not approved: pass "
                "allow_paid=True only after the user approves paid usage and a spending "
                "limit. The flag records approval; it does not cap dollars")
        if self.limiter is None:
            raise ProviderError(
                402, "no per-minute limits configured: pass max_requests_per_minute and "
                "max_tokens_per_minute from your organization's limits page in the "
                "OpenAI platform")

    def __call__(self, prompt):
        self.last_metrics = {"provider": "openai", "model_id": self.model_id,
                             "prompt_tokens": None, "completion_tokens": None,
                             "thought_tokens": None, "cached_tokens": None,
                             "output_tokens_per_sec_end_to_end": None,
                             "cost_basis": self.cost_mode(),
                             "estimated_cost_usd": None,  # usage tokens are recorded; no price math in code
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
            "messages": [{"role": "user", "content": prompt}],
            # Best-effort structured output (strict False): the local validator is
            # the real fence, so a schema the provider cannot enforce strictly must
            # still produce answerable JSON rather than a refusal.
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "extraction_result",
                                                "strict": False,
                                                "schema": self.schema}},
            "max_completion_tokens": self.max_output_tokens,
            "stream": False,
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        request = Request(ENDPOINT, data=json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json",
                                   "Authorization": "Bearer " + self.api_key}, method="POST")
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
                raise ProviderError(error.code, "OpenAI API HTTP failure", wait) from None
            except (OSError, http.client.HTTPException):
                # URLError, TimeoutError and dropped connections are all OSError;
                # a half-read response raises http.client.IncompleteRead.
                self.last_metrics["error_type"] = "transport_or_timeout"
                raise ProviderError(503, "OpenAI API unavailable or request timed out") from None
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                data = None
            if not isinstance(data, dict) or not isinstance(data.get("choices"), list):
                self.last_metrics["error_type"] = "response_envelope"
                raise ProviderError(502, "invalid API response envelope") from None
            text = None
            finish_reason = None
            for choice in data["choices"]:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    text = message["content"]
                    finish_reason = choice.get("finish_reason")
                    break
            if text is None:
                self.last_metrics["error_type"] = "response_envelope"
                raise ProviderError(502, "invalid API response envelope") from None
            if finish_reason == "length":
                # The answer was cut off at max_completion_tokens; retrying the same
                # request cannot fix that.
                self.last_metrics["error_type"] = "output_cut_off"
                raise ChunkFailedError(
                    400, "answer cut off at the output token limit; raise "
                    "max_output_tokens or use a smaller chunk")
            usage = data.get("usage") or {}
            if not isinstance(usage, dict):
                usage = {}
            metrics = self.last_metrics
            metrics["prompt_tokens"] = number_or_none(usage.get("prompt_tokens"))
            metrics["completion_tokens"] = number_or_none(usage.get("completion_tokens"))
            details = usage.get("completion_tokens_details")
            if isinstance(details, dict):
                metrics["thought_tokens"] = number_or_none(details.get("reasoning_tokens"))
            cached = usage.get("prompt_tokens_details")
            if isinstance(cached, dict):
                metrics["cached_tokens"] = number_or_none(cached.get("cached_tokens"))
            latency_ms = (time.perf_counter() - start) * 1000
            if metrics["completion_tokens"] and latency_ms > 0:
                # Output tokens over the WHOLE request time (queue + reading the
                # prompt + generating). Not the model's own decode speed.
                metrics["output_tokens_per_sec_end_to_end"] = round(
                    metrics["completion_tokens"] / (latency_ms / 1000), 2)
            return text
        finally:
            self.last_metrics["latency_ms"] = round((time.perf_counter() - start) * 1000, 3)
