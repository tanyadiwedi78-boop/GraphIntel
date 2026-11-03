"""What to do when a model call fails.
"""

from __future__ import annotations

import random
from typing import Optional

RETRY_LATER = "retry_later"   # 429, 408, 5xx: wait, then try again
STOP_RUN = "stop_run"         # 400, 401, 402, 403 and similar: fix the setup first


class ProviderError(Exception):
    """One failed model call."""

    def __init__(self, status_code: int, message: str = "", retry_after_seconds: Optional[float] = None):
        super().__init__(f"provider error {status_code}: {message}")
        self.status_code = status_code
        self.message = message
        self.retry_after_seconds = retry_after_seconds


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
