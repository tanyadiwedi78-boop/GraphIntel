"""Bounded retries and provider-scoped circuit breakers for extraction.

A short first failure stays on the same provider. Persistent failures route
within a chunk. A new chunk prefers the primary unless its circuit is open.
Each provider's five consecutive eligible failed CALLS open only its circuit
for 120 seconds; fallback success never clears primary failure history.
After cooldown the primary is half-open: one call tests recovery. Success
closes it; an eligible failure reopens it. Setup and cut-off errors never route.
"""
from __future__ import annotations

import time
from app.extraction.client import RETRY_LATER, ProviderError, classify_status


class ProviderRouter:
    def __init__(self, clients, failover_after=2, max_wait_seconds=60.0,
                 consecutive_failures=5, cooldown_seconds=120.0,
                 clock=time.monotonic, max_attempts=None, fallback_on=None):
        if len(clients) == 0:
            raise ValueError("ProviderRouter needs at least one client")
        if failover_after < 1 or consecutive_failures < 1:
            raise ValueError("failure thresholds must be at least 1")
        if max_wait_seconds <= 0 or cooldown_seconds <= 0:
            raise ValueError("wait and cooldown must be positive")
        self.clients = list(clients)
        self.max_attempts = max_attempts
        self.fallback_on = fallback_on

        self.active_index = 0
        self.served_index = 0
        self.failover_after = failover_after
        self.max_wait_seconds = max_wait_seconds
        self.consecutive_failures = consecutive_failures
        self.cooldown_seconds = cooldown_seconds
        self.clock = clock
        self.route_events = []
        self.circuit_events = []
        self.failures = [0] * len(clients)
        self.open_until = [0.0] * len(clients)
        self.half_open = [False] * len(clients)
        self.failures_before_response = 0

    @property
    def model_name(self):
        return self.clients[self.active_index].model_name

    @property
    def served_model_name(self):
        return self.clients[self.served_index].model_name

    @property
    def approved_models(self):
        return [client.model_name for client in self.clients]

    @property
    def last_metrics(self):
        return self.clients[self.served_index].last_metrics

    def cost_mode(self):
        return self.clients[self.active_index].cost_mode()

    def available(self, index):
        return self.open_until[index] <= self.clock()

    def begin_chunk(self):
        # Prefer primary at chunk boundaries, but never call an open provider.
        for index in range(len(self.clients)):
            if self.available(index):
                self.active_index = index
                return

    def prepare_call(self, exhausted_models=None):
        # Cooldown expiry permits a primary probe even mid-chunk.
        if exhausted_models is None:
            exhausted_models = []
        if (self.open_until[0] > 0 and self.available(0)
                and self.clients[0].model_name not in exhausted_models):
            self.active_index = 0
        if not self.available(self.active_index):
            for index in range(len(self.clients)):
                if self.available(index):
                    self.active_index = index
                    return 0.0
            # All providers are open. No call is made until the earliest can
            # be probed; the runner owns the wait and still parks exhausted work.
            return max(0.0, min(self.open_until) - self.clock())
        return 0.0

    def finish_cooldown_wait(self):
        # Sleep was fulfilled by the caller. Also supports injected no-sleep
        # clocks in tests without sending a call through an open circuit.
        index = self.open_until.index(min(self.open_until))
        self.open_until[index] = self.clock()
        self.active_index = index

    def request_failover(self, reason):
        if (reason == "schema_invalid_after_repair" and self.fallback_on is not None
                and reason not in self.fallback_on):
            return False
        for index in range(self.active_index + 1, len(self.clients)):
            if self.available(index):
                self.route_events.append({
                    "from_model": self.model_name,
                    "to_model": self.clients[index].model_name,
                    "reason": reason,
                })
                self.active_index = index
                return True
        return False

    def validated_response(self, valid, repair_exhausted=False):
        # HTTP success alone is not extraction success. The runner supplies
        # validation; only a rejected repair is a schema failover failure.
        index = self.served_index
        if valid:
            self.failures[index] = 0
            return
        self.failures[index] = self.failures_before_response
        if repair_exhausted:
            self.failures[index] += 1
            if self.failures[index] >= self.consecutive_failures:
                self.open_until[index] = self.clock() + self.cooldown_seconds
                self.circuit_events.append({
                    "model": self.clients[index].model_name,
                    "consecutive_failures": self.failures[index],
                    "cooldown_seconds": self.cooldown_seconds,
                    "state": "open",
                })

    def __call__(self, prompt):
        if self.prepare_call() > 0:
            raise RuntimeError("All provider circuits are open; prepare_call must be awaited")
        index = self.active_index
        self.served_index = index
        active = self.clients[index]
        probing = self.open_until[index] > 0
        self.half_open[index] = probing
        try:
            text = active(prompt)
        except ProviderError as error:
            if index > 0:
                active.last_metrics["served_after_failover"] = True
            cause = "server_error"
            if error.status_code == 429:
                cause = "rate_limit"
            elif error.status_code == 408:
                cause = "timeout"
            if classify_status(error.status_code) != RETRY_LATER:
                raise
            self.failures[index] = self.failures[index] + 1
            if probing or self.failures[index] >= self.consecutive_failures:
                self.open_until[index] = self.clock() + self.cooldown_seconds
                self.half_open[index] = False
                self.circuit_events.append({
                    "model": active.model_name,
                    "consecutive_failures": self.failures[index],
                    "cooldown_seconds": self.cooldown_seconds,
                    "state": "open",
                })
                active.last_metrics["circuit_open"] = True
            long_wait = (error.retry_after_seconds is not None
                         and error.retry_after_seconds > self.max_wait_seconds)
            allowed = self.fallback_on is None or cause in self.fallback_on
            if allowed and (long_wait or probing or self.failures[index] >= self.failover_after):
                if long_wait:
                    reason = "provider asked for a wait longer than " + str(self.max_wait_seconds) + " s"
                else:
                    reason = "provider-wide failure " + str(self.failures[index]) + " times in a row"
                if self.request_failover(reason):
                    active.last_metrics["failover_to"] = self.model_name
            raise
        self.failures_before_response = self.failures[index]
        self.failures[index] = 0
        self.open_until[index] = 0.0
        self.half_open[index] = False
        if index > 0:
            active.last_metrics["served_after_failover"] = True
        return text
