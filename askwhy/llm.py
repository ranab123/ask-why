"""Thin OpenRouter client (OpenAI-compatible). Logs every call's usage/latency; never stores teacher bodies."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from openai import OpenAI

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


@dataclass
class LLMResult:
    text: str
    ok: bool
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    error: str | None = None
    reasoning: str = ""  # model's extended-thinking text, when the provider returns it
    raw: dict = field(default_factory=dict)


class LLM:
    def __init__(self, log: Callable[[dict], None] | None = None, api_key: str | None = None):
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("OPENROUTER_API_KEY is not set. Put your key from https://openrouter.ai/keys in the .env file in the repository root.")
        self.client = OpenAI(base_url=OPENROUTER_BASE, api_key=key, timeout=120, max_retries=2)
        self.log = log or (lambda rec: None)

    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        role: str,
        t: int | None = None,
        max_tokens: int = 600,
        temperature: float = 0.0,
        json_mode: bool = False,
        keep_bodies: bool = True,
        reasoning_tokens: int | None = None,  # enable extended thinking with this budget
        timeout_s: float | None = None,  # per-call timeout (no retries) for latency-sensitive roles
    ) -> LLMResult:
        t0 = time.time()
        kwargs: dict[str, Any] = dict(model=model, messages=messages, max_tokens=max_tokens)
        kwargs["extra_body"] = {"usage": {"include": True}}
        if reasoning_tokens:
            kwargs["extra_body"]["reasoning"] = {"max_tokens": reasoning_tokens}
            kwargs["max_tokens"] = max_tokens + reasoning_tokens  # thinking counts against max_tokens
        else:
            kwargs["temperature"] = temperature  # extended thinking requires the provider's default temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            client = self.client.with_options(timeout=timeout_s, max_retries=0) if timeout_s else self.client
            resp = client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message if resp.choices else None
            text = (msg.content or "") if msg else ""
            reasoning = (getattr(msg, "reasoning", None) or (msg.model_extra or {}).get("reasoning") or "") if msg else ""
            finish = resp.choices[0].finish_reason if resp.choices else None
            usage = getattr(resp, "usage", None)
            res = LLMResult(
                text=text,
                ok=bool(text.strip()),
                latency_s=time.time() - t0,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                cost_usd=_cost(usage),
                error=None if text.strip() else f"empty completion (finish_reason={finish})",
                reasoning=reasoning or "",
            )
        except Exception as e:  # noqa: BLE001
            res = LLMResult(text="", ok=False, latency_s=time.time() - t0, error=f"{type(e).__name__}: {e}"[:300])

        rec = {
            "t": t,
            "role": role,
            "model": model,
            "ok": res.ok,
            "error": res.error,
            "latency_s": round(res.latency_s, 2),
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
            "cost_usd": res.cost_usd,
        }
        if keep_bodies and role != "teacher":
            rec["messages"] = _strip_images(messages)  # screenshots are saved per step; don't duplicate base64 here
            rec["response"] = res.text
        self.log(rec)
        return res


def _cost(usage) -> float | None:
    if usage is None:
        return None
    c = getattr(usage, "cost", None)
    if c is None and hasattr(usage, "model_extra"):
        c = (usage.model_extra or {}).get("cost")
    try:
        return float(c) if c is not None else None
    except (TypeError, ValueError):
        return None


def parse_json(text: str) -> dict | None:
    """Tolerant JSON extraction: handles ```json fences and leading prose."""
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _strip_images(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c = [p if p.get("type") != "image_url" else {"type": "image_url", "image_url": "<screenshot omitted>"} for p in c]
        out.append({**m, "content": c})
    return out
