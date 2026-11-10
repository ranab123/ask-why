"""Paired training-data export: the same demonstration as action-only and as rationale-enriched examples.

    python -m askwhy.export runs/<run_id> [more runs...] [--include-failed]

Writes <run>/export/action_only.jsonl and <run>/export/rationale.jsonl, line-aligned by step (plus
rationale_only_asked.jsonl with just the questioned steps). Each line is a chat-format next-action example built
from a trajectory prefix:
  system    = the student prompt (prompts/student_system.md)
  user      = task goal, recent actions, [rationale version only: explanations collected at EARLIER decision points],
              current URL and page (accessibility tree; the screenshot path is in the record's "screenshot" field)
  assistant = "ACTION: ..."   (rationale version, on a questioned step: "RATIONALE: ...", then "ACTION: ...")
The two files contain identical states and actions; they differ only in the explanation text.

Only teacher demonstrations that passed the WorkArena++ validator are exported (--include-failed overrides the
success check). Oracle-mode runs are always refused. The teacher's private memory (teacher_private.jsonl) is never read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .env_factory import ACTION_SET

SYSTEM = (Path(__file__).parent / "prompts" / "student_system.md").read_text().replace(
    "{action_space}", ACTION_SET.describe(with_long_description=False, with_examples=True)
)


def _fmt_qa(q: dict) -> str:
    a = q["answer"]
    return (f"- (step {q['t']}) Q: {q['question']}\n  A: goal: {a['goal']} | evidence: {a['evidence']} | "
            f"alternatives: {a['alternatives']} | would differ: {a['would_differ']}")


def _user_msg(step: dict, obs_text: str, history: list[str], prior_qas: list[dict] | None) -> str:
    hist = "\n".join(f"{i+1}. {a}" for i, a in enumerate(history[-8:])) or "(none yet)"
    parts = [f"# Goal\nPlease complete the following task.", f"# Recent actions\n{hist}"]
    if prior_qas is not None:
        parts.append("# Explanations the teacher gave earlier in this task\n" + ("\n".join(_fmt_qa(q) for q in prior_qas) or "(none yet)"))
    parts += [f"# Current URL\n{step['url_before']}", f"# Page (accessibility tree)\n{obs_text}", "Reply with ACTION: <action>"]
    return "\n\n".join(parts)


def export_run(run_dir: Path, include_failed: bool = False) -> tuple[int, int]:
    run = json.loads((run_dir / "run.json").read_text())
    if run.get("config", {}).get("mode") != "teacher":
        raise ValueError(f"mode={run.get('config', {}).get('mode')!r} is not a teacher demonstration (oracle runs are a development aid and are never exported)")
    if not include_failed and not (run.get("result") or {}).get("success"):
        return 0, 0
    steps = [json.loads(l) for l in (run_dir / "steps.jsonl").read_text().splitlines() if l.strip()]
    qpath = run_dir / "questions.jsonl"
    questions = {q["t"]: q for q in (json.loads(l) for l in qpath.read_text().splitlines() if l.strip()) if q.get("ok", True)} if qpath.exists() else {}

    out = run_dir / "export"
    out.mkdir(exist_ok=True)
    fa = open(out / "action_only.jsonl", "w")
    fr = open(out / "rationale.jsonl", "w")
    fq = open(out / "rationale_only_asked.jsonl", "w")
    history: list[str] = []
    prior: list[dict] = []
    n_asked = 0
    for s in steps:
        obs_text = (run_dir / s["obs_path"]).read_text() if s.get("obs_path") else ""
        base = {"t": s["t"], "run_id": run["run_id"], "task_id": run["task_id"], "screenshot": s.get("screenshot_path", "")}
        action_line = f"ACTION: {s['action']}"
        # action-only
        fa.write(json.dumps({**base, "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": _user_msg(s, obs_text, history, None)},
            {"role": "assistant", "content": action_line}]}) + "\n")
        # rationale-enriched: earlier Q&A in the context; the questioned step also teaches its own rationale
        q = questions.get(s["t"])
        target = action_line
        if q:
            a = q["answer"]
            target = (f"RATIONALE: goal={a['goal']}; evidence={a['evidence']}; alternatives={a['alternatives']}; "
                      f"would_differ={a['would_differ']}\n{action_line}")
        rec = {**base, "asked": bool(q), "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": _user_msg(s, obs_text, history, prior)},
            {"role": "assistant", "content": target}]}
        fr.write(json.dumps(rec) + "\n")
        if q:
            n_asked += 1
            fq.write(json.dumps({**rec, "question": q["question"]}) + "\n")
            prior.append(q)
        history.append(s["action"])
    for f in (fa, fr, fq):
        f.close()
    return len(steps), n_asked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--include-failed", action="store_true")
    args = ap.parse_args()
    for r in args.runs:
        if not (r / "run.json").exists() or not (r / "steps.jsonl").exists():
            print(f"{r}: skipped (not a completed run folder)")
            continue
        try:
            n, k = export_run(r, args.include_failed)
        except ValueError as e:
            print(f"{r}: skipped ({e})")
            continue
        print(f"{r}: {n} steps exported, {k} with rationale → {r / 'export'}" if n else f"{r}: skipped (task not solved)")


if __name__ == "__main__":
    main()
