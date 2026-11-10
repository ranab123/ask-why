"""The observer agent (distinct from the teacher and from the later students).

Two LLM calls per step:
  predict()   — BLIND: before the action is known, predicts the next action  → surprise S
  interpret() — after the action is known, infers the rationale and its own confidence,
                lists alternative rationales, labels the sub-goal           → uncertainty U, ambiguity A,
                                                                             boundary and criticality hints
"""
from __future__ import annotations

from pathlib import Path

from .env_factory import user_content
from .llm import LLM, parse_json
from .selector.base import ObserverOut

PROMPTS = Path(__file__).parent / "prompts"


class Observer:
    def __init__(self, llm: LLM, model: str, obs_max_chars: int = 25_000):
        self.llm = llm
        self.model = model
        self.system_predict = (PROMPTS / "observer.md").read_text()
        self.system_interpret = (PROMPTS / "observer_interpret.md").read_text()
        self.obs_max_chars = obs_max_chars
        self.current_subgoal = ""  # running interpretation of what the teacher is doing

    def predict(self, ticket: str, obs_text: str, history: list[tuple[str, str]], t: int, url: str = "", png: bytes | None = None) -> ObserverOut:
        hist = "\n".join(f"- {a}" for a, _ in history[-8:]) or "(none yet)"
        user = (
            f"# Ticket\n{ticket or '(not yet read)'}\n\n# Recent actions\n{hist}\n\n# URL\n{url}\n\n"
            f"# Current page (accessibility tree; screenshot attached when available)\n{obs_text[: self.obs_max_chars]}\n\nReturn the JSON now."
        )
        res = self.llm.chat(
            self.model,
            [{"role": "system", "content": self.system_predict}, {"role": "user", "content": user_content(user, png)}],
            role="observer", t=t, max_tokens=300, temperature=0.0, json_mode=True, timeout_s=25,
        )
        data = parse_json(res.text) if res.ok else None
        if not data:
            return ObserverOut(ok=False, latency_s=res.latency_s)
        preds = []
        for p in (data.get("predicted") or [])[:3]:
            if isinstance(p, dict):
                preds.append({
                    "action_type": str(p.get("action_type", "")).strip(),
                    "bid": (str(p["bid"]).strip() if p.get("bid") not in (None, "", "null") else None),
                    "target": str(p.get("target", "")).strip(),
                    "prob": _f(p.get("prob"), 0.0),
                })
        return ObserverOut(predicted=preds, activity_label=str(data.get("activity_label", ""))[:60], ok=True, latency_s=res.latency_s)

    def interpret(self, out: ObserverOut, ticket: str, protocol_excerpt: str, snippet: str, action_desc: str,
                  history: list[tuple[str, str]], t: int, png: bytes | None = None) -> ObserverOut:
        hist = "\n".join(f"- {a}" for a, _ in history[-8:]) or "(none yet)"
        user = (
            f"# Ticket\n{ticket or '(not yet read)'}\n\n# Protocol text seen so far\n{protocol_excerpt[:2000] or '(none yet)'}\n\n"
            f"# Recent actions\n{hist}\n\n# Your current interpretation of the sub-goal\n{self.current_subgoal or '(none yet)'}\n\n"
            f"# Page region around the element\n{snippet[:4000]}\n\n# The action the expert just took\n{action_desc}\n\nReturn the JSON now."
        )
        res = self.llm.chat(
            self.model,
            [{"role": "system", "content": self.system_interpret}, {"role": "user", "content": user_content(user, png)}],
            role="observer", t=t, max_tokens=300, temperature=0.0, json_mode=True, timeout_s=25,
        )
        data = parse_json(res.text) if res.ok else None
        out.interp_latency_s = res.latency_s
        if not data:
            out.interp_ok = False
            return out
        alts = data.get("alternative_rationales") or []
        out.rationale = str(data.get("rationale", "")).strip()[:300]
        out.confidence = _f(data.get("confidence"), 0.5)
        out.alt_rationales = [str(a)[:200] for a in alts if str(a).strip()][:5] if isinstance(alts, list) else []
        out.subgoal = str(data.get("subgoal", "")).strip()[:60]
        out.is_new_subgoal = bool(data.get("is_new_subgoal", False))
        out.criticality_hint = _f(data.get("criticality"), 0.0)
        out.interp_ok = True
        if out.subgoal:
            self.current_subgoal = out.subgoal
        return out


def _f(v, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(v)))
    except (TypeError, ValueError):
        return default
