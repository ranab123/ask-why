from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..recorder import StepRecord


@dataclass
class ObserverOut:
    """What the observer agent produced for this step."""

    # blind prediction (before the action was known)
    predicted: list[dict] = field(default_factory=list)  # [{"action_type","bid","target","prob"}]
    activity_label: str = ""
    ok: bool = False
    latency_s: float = 0.0
    # interpretation (after the action was known)
    rationale: str = ""
    confidence: float = 0.5
    alt_rationales: list[str] = field(default_factory=list)
    subgoal: str = ""
    is_new_subgoal: bool = False
    criticality_hint: float = 0.0
    interp_ok: bool = False
    interp_latency_s: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Decision:
    S: float = 0.0  # surprise
    U: float = 0.0  # rationale uncertainty
    A: float = 0.0  # ambiguity (several rationales fit)
    N: float = 0.0  # novelty
    C: float = 0.0  # criticality
    R: float = 0.0  # redundancy
    g: float = 0.0
    tau: float = 1.0
    is_decision: bool = False  # a "meaningful decision point" (budget applies to these, not every click)
    fire: bool = False
    question_focus: str = ""  # S | U | A | N | C
    budget_left_after: int = 0
    decisions_seen: int = 0
    predicted: list[dict] = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class Selector:
    """Online question gate. Sees only (step, observer_out); never the task's oracle state.

    `est_decisions` is the expected number of meaningful decision points in the demonstration
    (≈ the number of sub-tasks), not the number of clicks."""

    def start(self, est_decisions: int, budget: int) -> None:
        self.est_decisions = est_decisions
        self.budget = budget
        self.budget_left = budget
        self.decisions_seen = 0
        self.asked: list[StepRecord] = []
        self.t_last_q = -10**6
        self.d_last_q = -10**6  # decision index of last question

    def score(self, step: StepRecord, obs_out: ObserverOut | None) -> Decision:  # pragma: no cover - interface
        raise NotImplementedError

    def observe_outcome(self, step: StepRecord) -> None:
        # Keep the horizon honest if the run has more decision points than expected.
        self.est_decisions = max(self.est_decisions, self.decisions_seen + 1)

    def _commit(self, step: StepRecord, d: Decision) -> Decision:
        if d.is_decision:
            self.decisions_seen += 1
        if d.fire:
            self.budget_left -= 1
            self.asked.append(step)
            self.t_last_q = step.t
            self.d_last_q = self.decisions_seen
        d.budget_left_after = self.budget_left
        d.decisions_seen = self.decisions_seen
        return d
