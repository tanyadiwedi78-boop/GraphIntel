"""A pretend model for offline tests"""

from __future__ import annotations

import json
from typing import List, Union

from app.extraction.errors import ProviderError


class FakeModel:
    def __init__(self, steps: List[Union[str, ProviderError]]):
        self.steps = steps
        self.calls: List[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        position = len(self.calls) - 1
        if position >= len(self.steps):
            raise AssertionError("FakeModel was called more times than the test planned")
        step = self.steps[position]
        if isinstance(step, ProviderError):
            raise step
        return step


def good_answer(quote: str) -> str:
    """A valid answer for the standard test ontology, quoting `quote`."""
    answer = {
        "entities": [
            {"name": "Acme Corp", "type": "Company"},
            {"name": "Widget Engine", "type": "Technology"},
        ],
        "relationships": [{
            "relation": "DEVELOPS",
            "source": "Acme Corp",
            "target": "Widget Engine",
            "evidence_quote": quote,
            "confidence": 0.9,
        }],
    }
    return json.dumps(answer)
