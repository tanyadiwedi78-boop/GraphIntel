"""Provider router: keep the extraction run working when a provider goes down.

The router wraps a primary client and optional fallback clients and presents
the SAME callable contract as a single client: router(prompt) returns the
model's JSON text, router.last_metrics carries the last call's measurements,
and router.model_name names the ACTIVE provider, so work keys, event logs and
saved results always say exactly which provider produced them.

Routing rules, kept deliberately strict:

- Only a provider-wide problem is route-eligible: rate limit (429), request
  timeout (408), server errors (5xx) and network failures (the clients map
  those to 503). That is the same RETRY_LATER set the runner already retries
  with backoff.
- A setup problem is NEVER routed: a bad key (401), the cost gate (402), a
  wrong model id (404) or any other 4xx stops the run before any fallback -
  including a paid one - is touched. Failing over would hide the real
  problem and could spend money against a misconfiguration.
- A content failure is never routed: an answer cut off at the output limit
  (ChunkFailedError, status 400) fails the chunk, not the provider.
- Failover is one-way for the rest of the run. The router does not probe the
  primary to "see if it is back"; the next run starts on the primary again.

The router re-raises the failed attempt's error instead of silently retrying
on the fallback, so the runner's attempt accounting stays exact: attempt 1
failed on the primary, attempt 2 is served by the fallback. The retry backoff
between them is the runner's normal wait.
"""

from __future__ import annotations

from app.extraction.client import RETRY_LATER, ProviderError, classify_status


class ProviderRouter:
    """Callable client: tries the active provider, fails over once per outage."""

    def __init__(self, clients):
        if len(clients) == 0:
            raise ValueError("ProviderRouter needs at least one client")
        self.clients = list(clients)
        self.active_index = 0
        # One entry per failover: which provider, to which, and why. The
        # fallback rate is measured from these plus the attempt events.
        self.route_events = []

    @property
    def model_name(self):
        """The ACTIVE provider's model name; feeds this chunk's work key."""
        return self.clients[self.active_index].model_name

    @property
    def last_metrics(self):
        return self.clients[self.active_index].last_metrics

    def cost_mode(self):
        return self.clients[self.active_index].cost_mode()

    def __call__(self, prompt):
        active = self.clients[self.active_index]
        try:
            text = active(prompt)
        except ProviderError as error:
            if self.active_index > 0:
                active.last_metrics["served_after_failover"] = True
            route_eligible = classify_status(error.status_code) == RETRY_LATER
            has_fallback = self.active_index + 1 < len(self.clients)
            if route_eligible and has_fallback:
                fallback = self.clients[self.active_index + 1]
                self.route_events.append({
                    "from_model": active.model_name,
                    "to_model": fallback.model_name,
                    "reason": "provider error " + str(error.status_code),
                })
                # The failed attempt's event shows where the run moved to.
                active.last_metrics["failover_to"] = fallback.model_name
                self.active_index = self.active_index + 1
            # Always re-raise: the runner's retry loop makes the next attempt,
            # and that attempt lands on the now-active provider.
            raise
        if self.active_index > 0:
            active.last_metrics["served_after_failover"] = True
        return text
