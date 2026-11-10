"""The teacher: a frontier LLM web agent with its own private memory and reasoning.

The teacher is a stateful agent. Its conversation (its thinking, notes and plan at every step, plus any
interruptions it answered) is its memory and stays private: it is never shown to the observer and never enters
the training trace. The observer sees only browser states and actions. When the observer's question gate fires,
the teacher is paused and asked `ask_why` *inside its own conversation*, so it answers from its actual memory and
reasoning; then it continues the task.

Its private reasoning is written only to `teacher_private.jsonl` in the run folder, for auditing explanation
faithfulness; `export.py` never reads that file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .env_factory import ACTION_SET, chat_history_text, parse_action, target_bid, user_content
from .llm import LLM, parse_json

PROMPTS = Path(__file__).parent / "prompts"
_ACTION_RE = re.compile(r"ACTION:\s*(.+?)\s*$", re.S | re.M)


@dataclass
class TeacherOut:
    action: str
    ok: bool
    latency_s: float
    parse_error: str | None = None


@dataclass
class _Turn:
    """One exchange in the teacher's memory: what it saw (condensed) and what it said."""

    t: int
    user_brief: str  # condensed observation kept in memory (URL + error), not the full page
    assistant: str  # full reply: reasoning + ACTION line (private)
    extra: list[dict] = field(default_factory=list)  # interruptions answered at this step


class Teacher:
    def __init__(self, llm: LLM, model: str, temperature: float = 0.0,
                 reasoning_tokens: int | None = 2000, memory_steps: int = 25, private_log=None):
        self.llm = llm
        self.model = model
        self.temperature = temperature
        self.reasoning_tokens = reasoning_tokens
        self.memory_steps = memory_steps  # most recent steps kept verbatim; older ones are summarised as actions
        self.private_log = private_log or (lambda rec: None)
        self.system = (PROMPTS / "teacher_system.md").read_text().replace(
            "{action_space}", ACTION_SET.describe(with_long_description=False, with_examples=True)
        )
        self.memory: list[_Turn] = []
        self._current: tuple[int, str | list, str] | None = None  # (t, full user content, assistant reply) of this step

    # ------------------------------------------------------------------ prompt building
    def _observation(self, obs: dict, obs_text: str) -> str:
        err = obs.get("last_action_error") or ""
        return (
            f"# Goal\n{obs.get('goal','')}\n\n"
            f"# Chat\n{chat_history_text(obs) or '(empty)'}\n\n"
            f"# Result of your last action\n{('ERROR: ' + err[:300]) if err else 'ok'}\n\n"
            f"# Current URL\n{obs.get('url','')}\n\n"
            f"# Page (accessibility tree; screenshot attached)\n{obs_text}\n\n"
            "Think, then end with one line: ACTION: <action>"
        )

    @staticmethod
    def _brief(obs: dict, note: str = "") -> str:
        err = obs.get("last_action_error") or ""
        return (f"[step observation, page omitted from memory] URL: {obs.get('url','')}"
                + (f" | last action ERROR: {err[:200]}" if err else "") + (f" | {note}" if note else ""))

    def _history_messages(self) -> list[dict]:
        msgs: list[dict] = []
        first, rest = self.memory[:1], self.memory[1:]
        older = rest[: -self.memory_steps] if len(rest) > self.memory_steps else []
        recent = first + rest[-self.memory_steps:]
        if older:
            summary = "\n".join(f"step {m.t}: {self._action_of(m.assistant)}" for m in older)
            older_block = [{"role": "user", "content": f"[memory: steps {older[0].t}-{older[-1].t}, condensed to the actions you took]\n{summary}"},
                           {"role": "assistant", "content": "Noted."}]
        else:
            older_block = []
        for i, m in enumerate(recent):
            if i == 1:
                msgs.extend(older_block)
            msgs.append({"role": "user", "content": m.user_brief})
            msgs.append({"role": "assistant", "content": m.assistant})
            msgs.extend(m.extra)
        return msgs

    @staticmethod
    def _action_of(reply: str) -> str:
        m = _ACTION_RE.findall(reply or "")
        return m[-1].strip() if m else "(no action)"

    # ------------------------------------------------------------------ acting
    def act(self, obs: dict, obs_text: str, t: int, png: bytes | None = None,
            note: str = "") -> TeacherOut:
        user_full = user_content(self._observation(obs, obs_text) + (f"\n\nNote: {note}" if note else ""), png)
        msgs = [{"role": "system", "content": self.system}, *self._history_messages(), {"role": "user", "content": user_full}]
        res = self.llm.chat(self.model, msgs, role="teacher", t=t, max_tokens=900, temperature=self.temperature,
                            keep_bodies=False, reasoning_tokens=self.reasoning_tokens)
        reply = self._compose(res.reasoning, res.text)
        latency = res.latency_s
        action, perr = self.parse(res.text)

        # repair 1: no parsable action
        if perr and res.ok:
            msgs += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": f"Your reply had no valid action ({perr}). Reply with ONLY one line: ACTION: <action>"}]
            res2 = self.llm.chat(self.model, msgs, role="teacher", t=t, max_tokens=120, temperature=0.0, keep_bodies=False)
            action, perr = self.parse(res2.text)
            reply += f"\n[corrected] {res2.text.strip()}"
            latency += res2.latency_s
        # repair 2: element id not on this page
        if action and not perr:
            bad = self.missing_bid(action, obs_text)
            if bad:
                msgs += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": bad + " Copy the id exactly as written. Reply with ONLY one line: ACTION: <action>"}]
                res3 = self.llm.chat(self.model, msgs, role="teacher", t=t, max_tokens=120, temperature=0.0, keep_bodies=False)
                a3, p3 = self.parse(res3.text)
                latency += res3.latency_s
                if a3 and not p3 and not self.missing_bid(a3, obs_text):
                    action = a3
                    reply += f"\n[corrected] {res3.text.strip()}"
                else:
                    action, perr = None, f"unknown element id in {action!r}"

        # the very first page (the ticket) is kept verbatim, like keeping the assigned ticket open in a tab
        brief = self._observation(obs, obs_text) if not self.memory else self._brief(obs, note)
        self.memory.append(_Turn(t=t, user_brief=brief, assistant=reply))
        self._current = (t, user_full, reply)
        self.private_log({"t": t, "kind": "step", "reasoning": res.reasoning, "reply": res.text, "action": action, "error": perr})
        if perr or not action:
            return TeacherOut(action="noop(1000)", ok=False, latency_s=latency, parse_error=perr or res.error or "no action")
        return TeacherOut(action=action, ok=True, latency_s=latency)

    @staticmethod
    def _compose(reasoning: str, text: str) -> str:
        """What the teacher 'remembers' saying: its private thinking plus its visible reply."""
        reasoning = (reasoning or "").strip()
        return (f"<my_private_thinking>\n{reasoning}\n</my_private_thinking>\n" if reasoning else "") + (text or "").strip()

    # ------------------------------------------------------------------ being interrupted
    def answer_why(self, question: str, t: int) -> tuple[dict | None, str, float, str]:
        """Pause: answer the observer's question from inside the teacher's own conversation (full memory),
        then the exchange stays in memory and the teacher continues. Returns (fields, raw, latency, reasoning)."""
        prompt = (
            "[PAUSE: an observer who is learning from your work has a question about the action you just chose]\n"
            f"Question: {question}\n\n"
            "Answer from your actual reasons, grounded in the ticket, the protocol and what is on screen. Keep it short "
            "and concrete; do not narrate your thinking process. Do not take a new action. Return only JSON:\n"
            '{"goal": "...", "evidence": "...", "alternatives": "...", "would_differ": "..."}\n'
            "(each a single sentence of at most 30 words; use \"\" if not applicable)"
        )
        msgs = [{"role": "system", "content": self.system}, *self._history_messages()[:-2]] if self.memory else [{"role": "system", "content": self.system}]
        # replace the condensed copy of the current step with the full observation the teacher actually saw
        if self._current and self._current[0] == t:
            msgs += [{"role": "user", "content": self._current[1]}, {"role": "assistant", "content": self._current[2]}]
        msgs.append({"role": "user", "content": prompt})
        res = self.llm.chat(self.model, msgs, role="teacher_explain", t=t, max_tokens=500, temperature=0.2,
                            keep_bodies=False, reasoning_tokens=self.reasoning_tokens)
        data = parse_json(res.text) if res.ok else None
        # the interruption becomes part of the teacher's memory, as it would for a person
        if self.memory and self.memory[-1].t == t:
            self.memory[-1].extra += [{"role": "user", "content": f"[observer asked] {question}"},
                                      {"role": "assistant", "content": res.text.strip() or "(no answer)"}]
        self.private_log({"t": t, "kind": "ask_why", "question": question, "reasoning": res.reasoning, "reply": res.text})
        return data, res.text, res.latency_s, res.reasoning

    # ------------------------------------------------------------------ parsing / checks
    @staticmethod
    def missing_bid(action: str, obs_text: str) -> str | None:
        """Return a correction hint if the action targets an id that is not on the page, else None."""
        at, args = parse_action(action)
        bid = target_bid(at, args)
        if not bid or f"[{bid}]" in obs_text:
            return None
        digits = re.sub(r"\D", "", bid)
        cands = re.findall(rf"^\s*(\[[a-zA-Z]*{digits}\][^\n]*)$", obs_text, re.M)[:3] if digits else []
        hint = f"Element id '{bid}' is not on the current page."
        if cands:
            hint += " Similar ids on this page: " + "; ".join(c.strip()[:90] for c in cands) + "."
        return hint

    @staticmethod
    def parse(text: str) -> tuple[str | None, str | None]:
        body = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
        matches = _ACTION_RE.findall(body)
        if not matches:
            return None, "no ACTION line"
        action = matches[-1].strip().strip("`").splitlines()[0].strip()
        # models often copy the bracketed id from the tree: click('[78]') → click('78'); click([78]) → click('78')
        action = re.sub(r"""(['"])\[([a-zA-Z0-9]+)\]\1""", r"\1\2\1", action)
        action = re.sub(r"\(\[([a-zA-Z0-9]+)\]", r"('\1'", action)
        try:
            ACTION_SET.to_python_code(action)
        except Exception as e:  # noqa: BLE001
            return None, f"invalid action {action!r}: {e}"[:200]
        return action, None
