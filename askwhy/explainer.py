"""ask_why: question templates and the three answerers.

  TeacherAnswerer   — the method: the teacher is paused and answers from its own memory (teacher mode)
  LLMAnswerer       — oracle mode only: a separate LLM explains the scripted solver's action post hoc
  HumanChatAnswerer — experimental: a person types the answer in BrowserGym's chat window
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from .env_factory import user_content
from .llm import LLM, parse_json

PROMPTS = Path(__file__).parent / "prompts"
TEMPLATES: dict = yaml.safe_load((PROMPTS / "question_templates.yaml").read_text())


def build_question(focus: str, prediction: str | None = None) -> tuple[str, str]:
    tpl = TEMPLATES.get(focus) or TEMPLATES["N"]
    pred = prediction or "what an observer expected"
    return tpl["id"], tpl["text"].format(prediction=pred)


@dataclass
class ExplainCtx:
    ticket: str
    protocol_excerpt: str
    recent_actions: list[str]
    axtree_snippet: str
    action: str
    action_desc: str  # e.g. "click on button 'Delete'" or the oracle subtask label
    png: bytes | None = None  # screenshot of the page the action was taken on


@dataclass
class Explanation:
    goal: str = ""
    evidence: str = ""
    alternatives: str = ""
    would_differ: str = ""
    raw: str = ""
    ok: bool = False
    latency_s: float = 0.0
    answerer: str = "llm"
    model: str | None = None

    def one_line(self) -> str:
        return f"{self.goal} — {self.evidence}".strip(" —")

    def as_dict(self) -> dict:
        return asdict(self)


class LLMAnswerer:
    def __init__(self, llm: LLM, model: str):
        self.llm = llm
        self.model = model
        self.system = (PROMPTS / "ask_why.md").read_text()

    def ask(self, question: str, ctx: ExplainCtx, t: int) -> Explanation:
        hist = "\n".join(f"- {a}" for a in ctx.recent_actions) or "(none)"
        user = (
            f"# Your ticket\n{ctx.ticket or '(unknown)'}\n\n"
            f"# Protocol text you have seen (may be empty)\n{ctx.protocol_excerpt[:2500] or '(none yet)'}\n\n"
            f"# Your recent actions\n{hist}\n\n"
            f"# Page region around the element you acted on\n{ctx.axtree_snippet[:4000]}\n\n"
            f"# The action you just took\n{ctx.action_desc}\n`{ctx.action}`\n\n"
            f"# Question from the observer\n{question}\n\nAnswer with the JSON object."
        )
        res = self.llm.chat(
            self.model,
            [{"role": "system", "content": self.system}, {"role": "user", "content": user_content(user, ctx.png)}],
            role="explainer",
            t=t,
            max_tokens=300,
            temperature=0.2,
            json_mode=False,  # frontier teachers via OpenRouter don't all accept response_format; the prompt asks for JSON
        )
        data = parse_json(res.text) if res.ok else None
        if not data:
            return Explanation(raw=res.text, ok=False, latency_s=res.latency_s, model=self.model)
        return Explanation(
            goal=str(data.get("goal", "")).strip(),
            evidence=str(data.get("evidence", "")).strip(),
            alternatives=str(data.get("alternatives", "")).strip(),
            would_differ=str(data.get("would_differ", "")).strip(),
            raw=json.dumps(data),
            ok=True,
            latency_s=res.latency_s,
            model=self.model,
        )


class HumanChatAnswerer:
    """Posts the question in BrowserGym's chat window and blocks until the person types an answer."""

    def __init__(self, env):
        self.env = env

    def ask(self, question: str, ctx: ExplainCtx, t: int) -> Explanation:
        import time

        n_before = len(self.env.chat.messages)
        self.env.chat.add_message("assistant", f"❓ {question}")
        t0 = time.time()
        self.env.chat.wait_for_user_message()
        new = [m for m in self.env.chat.messages[n_before:] if m["role"] == "user"]
        text = new[-1]["message"] if new else ""
        return Explanation(goal=text, raw=text, ok=bool(text), latency_s=time.time() - t0, answerer="human")


class TeacherAnswerer:
    """The teacher answers ask_why itself: it is paused and asked inside its own conversation, so the answer
    comes from its actual memory and reasoning. Only the 4-field answer is recorded in questions.jsonl."""

    def __init__(self, teacher):
        self.teacher = teacher

    def ask(self, question: str, ctx: "ExplainCtx", t: int) -> "Explanation":
        data, raw, latency, _reasoning = self.teacher.answer_why(question, t)
        if not data:
            return Explanation(raw=raw, ok=False, latency_s=latency, answerer="teacher", model=self.teacher.model)
        return Explanation(
            goal=str(data.get("goal", "")).strip(),
            evidence=str(data.get("evidence", "")).strip(),
            alternatives=str(data.get("alternatives", "")).strip(),
            would_differ=str(data.get("would_differ", "")).strip(),
            raw=json.dumps(data),
            ok=True,
            latency_s=latency,
            answerer="teacher",
            model=self.teacher.model,
        )
