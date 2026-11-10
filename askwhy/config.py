from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Expected number of *meaningful decision points* per demonstration (≈ sub-tasks); used to pace the budget.
# In oracle mode the exact number of sub-tasks is used instead.
EST_DECISIONS = {
    "basic-expense-management-medium-l3": 8,
    "date-based-expense-management-medium-l3": 8,
    "amount-based-expense-management-medium-l3": 8,
    "priority-assignment-medium-l3": 13,
}
DEFAULT_EST_DECISIONS = 10

# Default question budgets from the research plan: 2 for expense-management, 3 for priority-assignment.
DEFAULT_BUDGET = {
    "basic-expense-management-medium-l3": 2,
    "date-based-expense-management-medium-l3": 2,
    "amount-based-expense-management-medium-l3": 2,
    "priority-assignment-medium-l3": 3,
}

EVAL_SEEDS = range(0, 10)  # reserved for held-out evaluation; never collect demonstrations on them


class Lambdas(BaseModel):
    S: float = 0.25  # surprise
    U: float = 0.20  # rationale uncertainty
    A: float = 0.15  # ambiguity (several rationales fit)
    N: float = 0.10  # novelty
    C: float = 0.30  # criticality
    R: float = 0.25  # redundancy (subtracted)


class TauParams(BaseModel):
    t0: float = 0.55  # base threshold when on pace
    k: float = 0.25  # how strongly pace moves the threshold
    lo: float = 0.25
    hi: float = 0.85
    min_gap: int = 1  # min number of decision points between two questions
    warmup: int = 1  # no questions before this many decision points unless C >= warmup_c
    warmup_c: float = 0.9
    force_g: float = 0.15  # force-fire floor when remaining decision points <= remaining budget


class RunConfig(BaseModel):
    task: str = "basic-expense-management-medium-l3"
    seed: int | None = None
    mode: Literal["teacher", "oracle"] = "teacher"
    selector: Literal["scored", "random", "none"] = "scored"
    budget: int | None = None  # None → per-task default (DEFAULT_BUDGET)
    answerer: Literal["llm", "human"] = "llm"
    allow_eval_seeds: bool = False

    # Models (OpenRouter ids). The teacher is the frontier web agent; in teacher mode it also answers ask_why itself,
    # from its own memory. The observer is a separate, cheap vision model. Students are trained later, not here.
    teacher_model: str = "anthropic/claude-opus-4.7"
    teacher_reasoning_tokens: int = 2000  # extended-thinking budget per teacher call (0 = off); private, never in the trace
    observer_model: str | None = "google/gemini-3.1-flash-lite"
    explainer_model: str | None = None  # oracle mode only (no LLM teacher to ask): model that answers ask_why post hoc
    save_teacher_memory: bool = True  # write the teacher's private notes to teacher_private.jsonl (audit only; never exported)
    temperature: float = 0.0

    max_steps: int = 60
    max_teacher_failures: int = 3  # consecutive unparsable/empty teacher replies before the run aborts
    est_decisions: int | None = None
    lambdas: Lambdas = Field(default_factory=Lambdas)
    tau: TauParams = Field(default_factory=TauParams)

    vision: bool = True  # teacher/observer/explainer see the screenshot as well as the accessibility tree
    headless: bool = False  # run the browser without a window (for machines with no display)
    obs_max_chars: int = 40_000
    slow_mo: int | None = None  # ms between browser actions; None keeps the task's default (1000 ms)
    panel: bool = False
    panel_port: int = 8765
    panel_pos: tuple[int, int] | None = None  # window position in screen points; None = right edge of the screen
    record: bool | None = None  # record the whole screen to runs/<run_id>/screen.mov (macOS); None = on when --panel is on
    record_browser: bool = False  # also save Playwright's own video of the task browser (video/*.webm), no permissions needed
    out_dir: Path = Path("runs")
    run_id: str | None = None

    @model_validator(mode="after")
    def _defaults(self):
        if self.budget is None:
            self.budget = DEFAULT_BUDGET.get(self.task, 2)
        if self.record is None:
            self.record = self.panel
        if self.seed is not None and self.seed in EVAL_SEEDS and not self.allow_eval_seeds:
            raise ValueError(f"seed {self.seed} is reserved for held-out evaluation (seeds 0-9). Pass --allow-eval-seeds to override.")
        return self

    def est(self) -> int:
        return self.est_decisions or EST_DECISIONS.get(self.task, DEFAULT_EST_DECISIONS)

    def observer(self) -> str:
        return self.observer_model or self.teacher_model

    def explainer(self) -> str:
        """Oracle mode only: the model that answers ask_why (in teacher mode the teacher answers itself)."""
        return self.explainer_model or self.observer()
