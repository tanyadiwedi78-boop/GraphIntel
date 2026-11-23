"""Classify saved outcomes without treating unknown requests as safe retries."""

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

SAFETY_REASONS = ("SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "RECITATION", "SPII")
RETRYABLE_HTTP = (500, 502, 503, 504)


@dataclass(frozen=True)
class Failure:
    category: str
    reason: str
    retryable: bool = False
    systemic: bool = False
    ambiguous: bool = False


def envelope_failure(result: Dict[str, Any]) -> Optional[Failure]:
    raw_envelope = result.get("raw_envelope")
    if not isinstance(raw_envelope, str):
        return None
    try:
        envelope = json.loads(raw_envelope)
    except (ValueError, TypeError):
        return Failure("schema_error", "Provider envelope was not valid JSON.")
    if not isinstance(envelope, dict):
        return Failure("schema_error", "Provider envelope was not an object.")

    feedback = envelope.get("promptFeedback")
    if isinstance(feedback, dict) and feedback.get("blockReason"):
        return Failure("safety_block", "Provider blocked the request.", systemic=True)
    candidates = envelope.get("candidates") or []
    choices = envelope.get("choices") or []
    if not isinstance(candidates, list) or not isinstance(choices, list):
        return Failure("schema_error", "Provider candidates or choices were malformed.")
    for candidate in candidates + choices:
        if not isinstance(candidate, dict):
            continue
        finish_reason = candidate.get("finishReason") or candidate.get("finish_reason")
        message = candidate.get("message") or {}
        refused = isinstance(message, dict) and message.get("refusal")
        if (
            finish_reason in SAFETY_REASONS
            or finish_reason == "content_filter"
            or refused
        ):
            return Failure(
                "safety_block",
                "Provider refused or filtered the output.",
                systemic=True,
            )
        if finish_reason in ("MAX_TOKENS", "length"):
            return Failure(
                "truncated_output",
                "Provider reported an output limit; review token settings.",
            )
    return None


def transport_failure(result: Dict[str, Any]) -> Optional[Failure]:
    status = result.get("http_status")
    outcome = str(result.get("outcome", ""))
    if status in (401, 403):
        return Failure(
            "authentication_error",
            "Credentials or access were rejected.",
            systemic=True,
        )
    if status == 429:
        return Failure(
            "rate_limit",
            "HTTP 429: stop this run, do not switch providers.",
            systemic=True,
        )
    if outcome == "http_failure_retry_after_unparseable":
        return Failure(
            "request_error",
            "Retry-After could not be interpreted safely.",
            systemic=True,
        )
    if "transport" in outcome:
        category = "timeout" if "timeout" in outcome else "unknown_error"
        return Failure(
            category,
            "Transport outcome needs reconciliation before another call.",
            systemic=True,
            ambiguous=True,
        )
    if status in RETRYABLE_HTTP:
        return Failure(
            "server_error",
            "Provider returned HTTP " + str(status) + ".",
            retryable=True,
        )
    if status is not None and status >= 400:
        return Failure(
            "request_error",
            "Provider rejected the request with HTTP " + str(status) + ".",
            systemic=True,
        )
    return None


def classify_failure(
    result: Dict[str, Any], record: Dict[str, Any]
) -> Optional[Failure]:
    transport = transport_failure(result)
    if transport is not None:
        return transport
    provider_failure = envelope_failure(result)
    if provider_failure is not None:
        return provider_failure
    if record.get("accepted"):
        return None
    raw_text = result.get("raw_text")
    if not isinstance(raw_text, str) or not raw_text.strip():
        return Failure(
            "empty_output",
            "No usable output text was returned; review the saved response.",
        )
    return Failure(
        "schema_error",
        "Returned text failed schema or source-evidence validation.",
        retryable=True,
    )
