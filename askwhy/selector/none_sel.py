from __future__ import annotations

from .base import Decision, ObserverOut, Selector
from ..recorder import StepRecord


class NoneSelector(Selector):
    def score(self, step: StepRecord, obs_out: ObserverOut | None) -> Decision:
        return self._commit(step, Decision(reason="selector=none"))
