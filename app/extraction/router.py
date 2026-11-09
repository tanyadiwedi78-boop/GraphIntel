"""Provider router: keep the extraction run working when a provider goes down.

The router wraps a primary client and optional fallback clients and presents
the SAME callable contract as a single client: router(prompt) returns the
model's JSON text, router.last_metrics carries the measurements of the client
that served the last call, and router.model_name names the ACTIVE provider,
so work keys, event logs and saved results always say exactly which provider
produced each answer.

Routing rules, kept deliberately strict:

- Only a provider-wide problem is route-eligible: rate limit (429), request
  timeout (408), server errors (5xx) and network failures (the clients map
  those to 503). That is the same RETRY_LATER set the runner already retries
  with backoff.
- A first short failure is NOT enough. A per-minute rate limit clears after a
  small wait on the same provider, and a paid fallback must never absorb a
  transient blip. The router fails over only on a confirmed outage:
    * the provider asked for a wait longer than max_wait_seconds (a daily
      quota or equivalent long stop), or
    * it raised route-eligible errors failover_after times in a row
      (default 2; a success resets the count).
- A setup problem is NEVER routed: a bad key (401), the cost gate (402), a
  wrong model id (404) or any other 4xx stops the run before any fallback -
  including a paid one - is touched. Failing over would hide the real
  problem and could spend money against a misconfiguration.
- A content failure is never routed: an answer cut off at the output limit
  (ChunkFailedError, status 400) fails the chunk, not the provider.
- Failover is one-way for the rest of the run. The router does not probe the
  primary to "see if it is back"; the next run starts on the primary again.

The router re-raises the failed attempt's error instead of silently retrying
on the fallback, so the runner's attempt accounting stays exact: attempts 1
and 2 failed on the primary, attempt 3 is served by the fallback. Same-provider
retries use the runner's normal backoff; the attempt right after a failover
goes to the fallback at once - waiting for the provider that is down helps no
one, and the fallback client paces itself.
"""

from __future__ import annotations

from app.extraction.client import RETRY_LATER, ProviderError, classify_status


class ProviderRouter:
    """Callable client: stays on the active provider until a confirmed outage."""

    def __init__(self, clients, failover_after=2, max_wait_seconds=60.0):
        if len(clients) == 0:
            raise ValueError("ProviderRouter needs at least one client")
        if failover_after < 1:
            raise ValueError("failover_after must be at least 1")
        if max_wait_seconds <= 0:
            raise ValueError("max_wait_seconds must be positive")
        self.clients = list(clients)
        self.active_index = 0
        self.failover_after = failover_after
        self.max_wait_seconds = max_wait_seconds
        # One entry per failover: which provider, to which, and why. The
        # fallback rate is measured from these plus the attempt events.
        self.route_events = []
        # Route-eligible errors in a row from the active provider. A success
        # resets it; a short per-minute 429 that clears never routes.
        self.eligible_in_a_row = 0
        # The client that served the most recent call. After a failover this
        # still points at the client that failed, so the failed attempt's
        # metrics (error type, tokens, latency) are the ones that get logged.
        self.served_index = 0

    @property
    def model_name(self):
        """The ACTIVE provider's model name; feeds this chunk's work key."""
        return self.clients[self.active_index].model_name

    @property
    def last_metrics(self):
        """Metrics of the client that served the last call, failed or not."""
        return self.clients[self.served_index].last_metrics

    def cost_mode(self):
        return self.clients[self.active_index].cost_mode()

    def request_failover(self, reason):
        """Content-driven failover, asked for by the runner: the active
        provider's answers keep failing validation (schema-invalid after one
        repair attempt). Move to the next provider for the rest of the run.
        Returns True when a fallback took over, False when there is none.
        The same prompt, schema and validation apply on the new provider."""
        if self.active_index + 1 >= len(self.clients):
            return False
        active = self.clients[self.active_index]
        fallback = self.clients[self.active_index + 1]
        self.route_events.append({
            "from_model": active.model_name,
            "to_model": fallback.model_name,
            "reason": reason,
        })
        self.active_index = self.active_index + 1
        self.eligible_in_a_row = 0
        return True

    def __call__(self, prompt):
        self.served_index = self.active_index
        active = self.clients[self.active_index]
        try:
            text = active(prompt)
        except ProviderError as error:
            if self.active_index > 0:
                active.last_metrics["served_after_failover"] = True
            eligible = classify_status(error.status_code) == RETRY_LATER
            if not eligible:
                raise
            self.eligible_in_a_row = self.eligible_in_a_row + 1
            long_wait = (error.retry_after_seconds is not None
                         and error.retry_after_seconds > self.max_wait_seconds)
            has_fallback = self.active_index + 1 < len(self.clients)
            if has_fallback and (long_wait or self.eligible_in_a_row >= self.failover_after):
                fallback = self.clients[self.active_index + 1]
                if long_wait:
                    reason = ("provider asked for a wait longer than "
                              + str(self.max_wait_seconds) + " s")
                else:
                    reason = ("provider-wide failure " + str(self.eligible_in_a_row)
                              + " times in a row")
                self.route_events.append({
                    "from_model": active.model_name,
                    "to_model": fallback.model_name,
                    "reason": reason,
                })
                # The failed attempt's event shows where the run moved to.
                active.last_metrics["failover_to"] = fallback.model_name
                self.active_index = self.active_index + 1
                self.eligible_in_a_row = 0
            # Always re-raise: the runner's retry loop makes the next attempt.
            # Same-provider retries wait with backoff; after a failover the
            # next attempt goes to the new provider at once.
            raise
        self.eligible_in_a_row = 0
        if self.active_index > 0:
            active.last_metrics["served_after_failover"] = True
        return text
