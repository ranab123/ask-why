"""The budgeted selector:  g_t = λs·S + λu·U + λa·A + λn·N + λc·C − λr·R,  fire when g_t ≥ τ_t at a decision point.

Signals (all in [0,1]):
  S surprise       — observer's blind prediction vs. what the teacher actually did
  U uncertainty    — observer's low confidence in its own explanation of the action
  A ambiguity      — several rationales are consistent with the action
  N novelty        — first time this (action_type, page) occurs / a new activity segment
  C criticality    — rule-based: irreversible / state-changing actions score high (+ observer hint)
  R redundancy     — similarity to questions already asked, plus a recency penalty

The budget applies to *meaningful decision points* (segment boundaries, state-changing or surprising actions),
not to every click. The threshold is paced over the expected number of decision points (ThriftyDAgger-style):
ahead of pace → τ rises; behind pace → τ drops.
"""
from __future__ import annotations

import math
import re

from .base import Decision, ObserverOut, Selector
from ..config import Lambdas, TauParams
from ..env_factory import ACTION_TYPES_FORM, ACTION_TYPES_NAV
from ..recorder import StepRecord

_CRITICAL = re.compile(r"delete|submit|update|save|resolve|close|confirm|^ok$|assign|approve|reject", re.I)
_SEARCH = re.compile(r"search|filter|run|go|find|keyword", re.I)


def criticality(step: StepRecord) -> float:
    name = step.target_name or ""
    at = step.action_type
    if at in ("send_msg_to_user", "report_infeasible"):
        return 1.0
    if at == "oracle":
        lbl = (step.activity_label or "").lower()
        if any(k in lbl for k in ("delete", "edit", "update", "assign", "private")):
            return 1.0
        if "filter" in lbl or "search" in lbl:
            return 0.4
        return 0.2
    if at in ("click", "dblclick") and _CRITICAL.search(name):
        return 1.0
    if at in ACTION_TYPES_FORM:
        return 0.4 if _SEARCH.search(name) else 0.6
    if at == "press":
        return 0.4
    if at in ("click", "dblclick"):
        return 0.4 if _SEARCH.search(name) else 0.2
    if at in ACTION_TYPES_NAV:
        return 0.2
    return 0.0  # scroll, noop, hover, ...


def is_decision_point(step: StepRecord, S: float, C: float) -> bool:
    if step.action_type in ("noop", "scroll", "hover"):
        return False
    return (
        step.action_type in ("oracle", "send_msg_to_user", "report_infeasible")
        or step.boundary
        or C >= 0.6
        or S >= 0.8
    )


def _norm(a: str) -> str:
    return re.sub(r"\s+", "", a or "").lower()


class ScoredSelector(Selector):
    def __init__(self, lam: Lambdas, tau: TauParams):
        self.lam = lam
        self.tau_p = tau
        self.seen_keys: set[tuple[str, str]] = set()
        self.seen_pages: set[str] = set()

    # ---------------------------------------------------------------- signals
    def surprise(self, step: StepRecord, o: ObserverOut | None) -> float:
        if not o or not o.ok or not o.predicted:
            return 0.5
        want_name = _norm(step.target_name)
        for p in o.predicted:
            fn = p.get("action_type") or ""
            if fn == step.action_type and (
                (p.get("bid") and p.get("bid") == step.target_bid) or (want_name and _norm(p.get("target", "")) == want_name)
            ):
                return max(0.0, 1.0 - float(p.get("prob", 0.0)))
        return 1.0

    def uncertainty(self, o: ObserverOut | None) -> float:
        if not o or not o.interp_ok:
            return 0.5
        return 1.0 - float(o.confidence)

    def ambiguity(self, o: ObserverOut | None) -> float:
        if not o or not o.interp_ok:
            return 0.5
        return min(1.0, len(o.alt_rationales) / 3.0)

    def novelty(self, step: StepRecord) -> float:
        key = (step.action_type, step.page_sig)
        n = 0.0 if key in self.seen_keys else (1.0 if step.page_sig not in self.seen_pages else 0.5)
        self.seen_keys.add(key)
        self.seen_pages.add(step.page_sig)
        if step.boundary:
            n = max(n, 0.8)
        return n

    def redundancy(self, step: StepRecord) -> float:
        r = 0.0
        for q in self.asked:
            if (q.action_type, q.page_sig, q.activity_label) == (step.action_type, step.page_sig, step.activity_label):
                r = max(r, 1.0)
            elif q.activity_label and q.activity_label == step.activity_label:
                r = max(r, 0.5)
        if self.asked:
            r += 0.5 * math.exp(-(self.decisions_seen - self.d_last_q) / 2.0)
        return min(1.0, r)

    # ---------------------------------------------------------------- threshold
    def threshold(self) -> float:
        p = self.tau_p
        remaining = max(1, self.est_decisions - self.decisions_seen)
        pace_now = self.budget_left / remaining
        pace_0 = self.budget / max(1, self.est_decisions)
        rho = pace_now / pace_0 if pace_0 > 0 else 1.0  # >1 behind pace (budget piling up), <1 ahead
        return min(p.hi, max(p.lo, p.t0 + p.k * (1.0 - rho)))

    # ---------------------------------------------------------------- main
    def score(self, step: StepRecord, o: ObserverOut | None) -> Decision:
        S = self.surprise(step, o)
        U = self.uncertainty(o)
        A = self.ambiguity(o)
        N = self.novelty(step)
        C = min(1.0, criticality(step) + (0.2 * o.criticality_hint if o and o.interp_ok else 0.0))
        R = self.redundancy(step)
        lam = self.lam
        g = min(1.0, max(0.0, lam.S * S + lam.U * U + lam.A * A + lam.N * N + lam.C * C - lam.R * R))
        tau = self.threshold()
        d = Decision(S=S, U=U, A=A, N=N, C=C, R=R, g=round(g, 3), tau=round(tau, 3), predicted=o.predicted if o else [])
        d.is_decision = is_decision_point(step, S, C)

        p = self.tau_p
        remaining = max(1, self.est_decisions - self.decisions_seen)
        if not d.is_decision:
            d.reason = "not a decision point"
        elif self.budget_left <= 0:
            d.reason = "budget exhausted"
        elif self.decisions_seen < p.warmup and C < p.warmup_c:
            d.reason = "warmup"
        elif remaining <= self.budget_left and g >= p.force_g:
            d.fire, d.reason = True, "force-fire: remaining decisions <= budget"
        elif self.decisions_seen - self.d_last_q < p.min_gap:
            d.reason = "min gap"
        elif g >= tau:
            d.fire, d.reason = True, f"g={g:.2f} >= tau={tau:.2f}"
        else:
            d.reason = f"g={g:.2f} < tau={tau:.2f}"

        comps = {"S": lam.S * S, "U": lam.U * U, "A": lam.A * A, "N": lam.N * N + (0.05 if step.boundary else 0), "C": lam.C * C}
        d.question_focus = max(comps, key=comps.get)
        return self._commit(step, d)
