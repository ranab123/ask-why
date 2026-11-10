"""Random baseline: same budget, same notion of decision points and same hard rules as the scored selector,
but WHICH decision points get a question is chosen uniformly at random (online sampling without replacement:
at each decision point fire with probability budget_left / decisions_remaining)."""
from __future__ import annotations

import random

from .base import Decision, ObserverOut, Selector
from .scored import criticality, is_decision_point
from ..config import TauParams
from ..recorder import StepRecord


class RandomSelector(Selector):
    def __init__(self, tau: TauParams, seed: int = 0):
        self.tau_p = tau
        self.rng = random.Random(seed)

    def score(self, step: StepRecord, o: ObserverOut | None) -> Decision:
        C = criticality(step)
        S = 0.5
        d = Decision(C=C, S=S, predicted=o.predicted if o else [])
        d.is_decision = is_decision_point(step, S, C)
        d.question_focus = "N"  # generic "what were you trying to accomplish?"
        if not d.is_decision:
            d.reason = "not a decision point"
        elif self.budget_left <= 0:
            d.reason = "budget exhausted"
        elif self.decisions_seen < self.tau_p.warmup and C < self.tau_p.warmup_c:
            d.reason = "warmup"
        elif (remaining := max(1, self.est_decisions - self.decisions_seen)) <= self.budget_left:
            d.fire, d.reason = True, "force-fire: remaining decisions <= budget"
        elif self.decisions_seen - self.d_last_q < self.tau_p.min_gap:
            d.reason = "min gap"
        else:
            p = min(1.0, self.budget_left / remaining)
            d.fire = self.rng.random() < p
            d.reason = f"random p={p:.2f}"
        return self._commit(step, d)
