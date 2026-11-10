"""Writes one run folder: run.json, steps.jsonl, questions.jsonl, llm_calls.jsonl, obs/step_XXX.txt."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import RunConfig


def _jsonable(o: Any) -> Any:
    if is_dataclass(o):
        return asdict(o)
    if hasattr(o, "model_dump"):
        return o.model_dump(mode="json")
    if isinstance(o, Path):
        return str(o)
    return str(o)


@dataclass
class StepRecord:
    t: int
    url_before: str
    page_sig: str
    action: str
    action_type: str
    target_bid: str | None
    target_name: str
    activity_label: str = ""
    boundary: bool = False
    micro_actions: list | None = None
    obs_path: str = ""
    screenshot_path: str = ""
    ts: float = field(default_factory=time.time)
    selector: dict | None = None
    observer: dict | None = None  # observer's inferred rationale / confidence / alternatives
    after: dict | None = None
    latency: dict = field(default_factory=dict)


class Recorder:
    def __init__(self, cfg: RunConfig, task_id: str, seed: int):
        short = task_id.split(".")[-1].replace("-medium-l3", "")
        rid = cfg.run_id or f"{datetime.now():%Y%m%d-%H%M%S}_{short}_s{seed}_{cfg.mode}_{cfg.selector}_b{cfg.budget}"
        self.dir = Path(cfg.out_dir) / rid
        (self.dir / "obs").mkdir(parents=True, exist_ok=True)
        self.run_id = rid
        self.t_start = time.time()
        self.cost = 0.0
        self._run: dict = {
            "run_id": rid,
            "task_id": task_id,
            "seed": seed,
            "config": cfg.model_dump(mode="json"),
            "result": None,
        }
        self._flush_run()

    # ---- run.json
    def set_run_meta(self, **kw):
        self._run.update(kw)
        self._flush_run()

    def _flush_run(self):
        (self.dir / "run.json").write_text(json.dumps(self._run, indent=2, default=_jsonable))

    def finalize(self, success: bool, reward: float, steps: int, questions_fired: int, error: str | None = None):
        self._run["result"] = {
            "success": success,
            "reward": reward,
            "steps": steps,
            "questions_fired": questions_fired,
            "wall_s": round(time.time() - self.t_start, 1),
            "cost_usd": round(self.cost, 4),
            "error": error,
        }
        self._flush_run()

    # ---- per-step files
    def save_obs(self, t: int, obs_text: str) -> str:
        p = self.dir / "obs" / f"step_{t:03d}.txt"
        p.write_text(obs_text)
        return str(p.relative_to(self.dir))

    def save_screenshot(self, t: int, png: bytes | None) -> str:
        if not png:
            return ""
        p = self.dir / "obs" / f"step_{t:03d}.png"
        p.write_bytes(png)
        return str(p.relative_to(self.dir))

    def save_snippet(self, t: int, text: str) -> str:
        p = self.dir / "obs" / f"q_{t:03d}_snippet.txt"
        p.write_text(text)
        return str(p.relative_to(self.dir))

    def _append(self, name: str, obj: Any):
        with open(self.dir / name, "a") as f:
            f.write(json.dumps(obj, default=_jsonable) + "\n")

    def write_step(self, step: StepRecord):
        self._append("steps.jsonl", step)

    def write_question(self, q: dict):
        self._append("questions.jsonl", q)

    def write_llm_call(self, rec: dict):
        self.cost += rec.get("cost_usd") or 0.0
        self._append("llm_calls.jsonl", rec)
