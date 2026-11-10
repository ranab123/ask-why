"""Main loop. The teacher (a frontier LLM) drives the browser; the observer watches states and actions; the question
gate decides when to pause the teacher and ask why. `--mode oracle` swaps the teacher for WorkArena's scripted solver
(a free development aid for testing the observer/gate/recording pipeline; not part of the method)."""
from __future__ import annotations

import faulthandler
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

from .config import RunConfig
from .env_factory import (
    axtree_snippet,
    element_line,
    element_name,
    extract_ticket_text,
    make_env_for_class,
    obs_to_text,
    page_signature,
    parse_action,
    resolve_task,
    screenshot_png,
    target_bid,
)
from .explainer import ExplainCtx, Explanation, HumanChatAnswerer, LLMAnswerer, TeacherAnswerer, build_question
from .oracle import macro_action_str, run_subtask, subtask_label
from .recorder import Recorder, StepRecord
from .recording import ScreenRecorder
from .selector import make_selector
from .selector.base import ObserverOut
from .selector.segmenter import Segmenter

app = typer.Typer(add_completion=False, help="ask-why: budgeted rationale elicitation on WorkArena++")
console = Console()


class KBMemory:
    """Remembers the most recent Knowledge-Base article text the agent has had on screen (for the explainer)."""

    def __init__(self):
        self.text = ""

    def update(self, obs: dict, obs_text: str):
        if re.search(r"kb_view|kb_knowledge|knowledge", obs.get("url", ""), re.I):
            m = re.search(r"(?:article|document)\b.*", obs_text, re.S | re.I)
            self.text = (m.group(0) if m else obs_text)[:6000]


class Run:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.cls, default_seed = resolve_task(cfg.task)
        self.seed = cfg.seed if cfg.seed is not None else default_seed
        self.task_id = self.cls.get_task_id()
        needs_llm = cfg.mode == "teacher" or cfg.selector == "scored" or (cfg.selector != "none" and cfg.budget > 0 and cfg.answerer == "llm")
        if needs_llm and not os.environ.get("OPENROUTER_API_KEY"):
            raise SystemExit("OPENROUTER_API_KEY is not set. Put your key from https://openrouter.ai/keys in the .env file in the repository root.")
        self.rec = Recorder(cfg, self.task_id, self.seed)
        self.panel = None
        if cfg.panel:
            from .panel.server import PanelClient

            self.panel = PanelClient(cfg.panel_port, position=cfg.panel_pos)
            self.panel.start()
        self.llm = None
        if needs_llm:
            from .llm import LLM

            self.llm = LLM(log=self.rec.write_llm_call)
        self.history: list[tuple[str, str]] = []
        self.ticket = ""
        self.ticket_source = ""
        self.kb = KBMemory()
        self.questions_fired = 0

    # ------------------------------------------------------------------ helpers
    def push(self, event: str, **data):
        if self.panel:
            self.panel.push(event, **data)

    def log(self, msg: str):
        console.print(msg)

    def make_step(self, t: int, obs: dict, obs_text: str, action: str) -> StepRecord:
        at, args = parse_action(action)
        bid = target_bid(at, args)
        line = element_line(obs_text, bid)
        return StepRecord(
            t=t,
            url_before=obs.get("url", ""),
            page_sig=page_signature(obs.get("url", "")),
            action=action,
            action_type=at,
            target_bid=bid,
            target_name=element_name(line) or (str(args[0])[:40] if at in ("goto",) and args else ""),
        )

    def ask(self, t: int, step: StepRecord, decision, obs_text: str, action_desc: str, png: bytes | None = None) -> Explanation:
        pred = None
        if decision.predicted:
            top = max(decision.predicted, key=lambda p: p.get("prob", 0))
            pred = f"{top.get('action_type','')} on '{top.get('target','')}'".strip()
        template_id, question = build_question(decision.question_focus, pred)
        self.log(f"[bold yellow]❓ ask_why (t={t}, focus={decision.question_focus}):[/] {question}")
        self.env.chat.add_message("info", f"❓ ask_why: {question}")
        self.push("question_asked", t=t, question=question, focus=decision.question_focus, budget_left=decision.budget_left_after)

        snippet = axtree_snippet(obs_text, step.target_bid)
        ctx = ExplainCtx(
            ticket=self.ticket,
            protocol_excerpt=self.kb.text,
            recent_actions=[a for a, _ in self.history[-6:]] + [step.action],
            axtree_snippet=snippet,
            action=step.action,
            action_desc=action_desc,
            png=png,
        )
        expl = self.answerer.ask(question, ctx, t)
        self.env.chat.add_message("info", f"💬 {expl.one_line() or expl.raw[:200]}")
        self.log(f"[green]💬 {expl.goal}[/]\n   evidence: {expl.evidence}\n   alternatives: {expl.alternatives}\n   would differ: {expl.would_differ}")
        self.rec.write_question(
            {
                "t": t,
                "question_focus": decision.question_focus,
                "template_id": template_id,
                "question": question,
                "answerer": expl.answerer,
                "model": expl.model,
                "selector": decision.as_dict(),
                "ctx": {
                    "action": step.action,
                    "action_desc": action_desc,
                    "recent_actions": ctx.recent_actions,
                    "axtree_snippet_path": self.rec.save_snippet(t, snippet),
                    "protocol_excerpt_chars": len(ctx.protocol_excerpt),
                },
                "answer": {"goal": expl.goal, "evidence": expl.evidence, "alternatives": expl.alternatives, "would_differ": expl.would_differ},
                "raw": expl.raw,
                "ok": expl.ok,
                "latency_s": round(expl.latency_s, 2),
            }
        )
        self.questions_fired += 1
        self.push("question_answered", t=t, answer=expl.as_dict())
        return expl

    def read_ticket(self, obs: dict):
        txt = extract_ticket_text(obs)
        if txt:
            self.ticket, self.ticket_source = txt, "axtree"
        else:
            self.ticket, self.ticket_source = getattr(self.env.task, "task_description", ""), "task_object"

    # ------------------------------------------------------------------ main
    def run(self) -> dict:
        cfg = self.cfg
        self.log(f"[bold]ask-why[/] task={self.task_id} seed={self.seed} mode={cfg.mode} selector={cfg.selector} B={cfg.budget}")
        self.env = make_env_for_class(self.cls, slow_mo=cfg.slow_mo, headless=cfg.headless,
                                       record_video_dir=str(self.rec.dir / "video") if cfg.record_browser else None)
        reward, terminated, t, error, hang_log = 0.0, False, 0, None, None
        screen = ScreenRecorder(self.rec.dir / "screen.mov") if cfg.record else None
        try:
            t0 = time.time()
            obs, info = self.env.reset(seed=self.seed)
            self.log(f"env ready in {time.time()-t0:.0f}s → {obs['url']}")
            self.read_ticket(obs)
            n_sub = len(self.env.task)
            est = n_sub if cfg.mode == "oracle" else cfg.est()  # expected number of decision points
            self.rec.set_run_meta(
                ticket_text=self.ticket,
                ticket_source=self.ticket_source,
                num_subtasks=n_sub,
                subtask_classes=[type(s).__name__ for s in self.env.task.subtasks],
                est_decisions=est,
                models={"teacher": cfg.teacher_model if cfg.mode == "teacher" else "oracle (scripted solver)", "observer": cfg.observer(),
                        "answers_ask_why": "human" if cfg.answerer == "human" else (cfg.teacher_model if cfg.mode == "teacher" else cfg.explainer())},
                instance_url=self.env.task.instance.snow_url,
            )
            self.selector = make_selector(cfg.selector, cfg, est_decisions=est, budget=cfg.budget)
            self.segmenter = Segmenter()
            # answerer: a human typing, the teacher itself (teacher mode), or a separate LLM (oracle dev mode only)
            if cfg.answerer == "human":
                self.answerer = HumanChatAnswerer(self.env)
            elif cfg.mode == "teacher":
                self.answerer = None  # set to TeacherAnswerer once the teacher exists (below)
            else:
                self.answerer = LLMAnswerer(self.llm, cfg.explainer()) if self.llm else None
            observer = None
            if cfg.selector == "scored" and self.llm:
                from .observer import Observer

                observer = Observer(self.llm, cfg.observer())
            teacher = None
            if cfg.mode == "teacher":
                from .teacher import Teacher

                teacher = Teacher(self.llm, cfg.teacher_model, cfg.temperature,
                                  reasoning_tokens=cfg.teacher_reasoning_tokens or None,
                                  private_log=(lambda rec: self.rec._append("teacher_private.jsonl", rec)) if cfg.save_teacher_memory else None)
                if cfg.answerer == "llm":
                    self.answerer = TeacherAnswerer(teacher)
            self.push(
                "run_start",
                task=self.task_id.split(".")[-1],
                seed=self.seed,
                mode=cfg.mode,
                selector=cfg.selector,
                budget=cfg.budget,
                model=cfg.teacher_model if cfg.mode == "teacher" else "oracle (task.cheat)",
                est_decisions=est,
                ticket=self.ticket[:600],
            )
            self.env.chat.add_message("info", "👀 ask-why observer attached. I may interrupt to ask why.")
            if self.panel:
                self.panel.open_window()  # fresh panel window, opened when the agent starts (not during setup)
                time.sleep(3)
            if screen:
                if screen.start():
                    self.log(f"🎥 screen recording → {screen.out_path}")
                else:
                    self.log("[yellow]🎥 screen recording could not start. On macOS allow your terminal app under "
                             "System Settings → Privacy & Security → Screen Recording, then rerun. "
                             "(--record-browser saves the browser-only video without permissions.)[/]")
                    screen = None

            pool = ThreadPoolExecutor(max_workers=2)
            teacher_failures = 0
            # Hang watchdog: if a browser step blocks for >3 min, dump all thread stacks to hang_traces.txt (repeats).
            hang_log = open(self.rec.dir / "hang_traces.txt", "a")
            faulthandler.dump_traceback_later(180, repeat=True, file=hang_log)
            max_steps = n_sub if cfg.mode == "oracle" else cfg.max_steps
            for t in range(max_steps):
                obs_text = obs_to_text(obs, cfg.obs_max_chars)
                obs_path = self.rec.save_obs(t, obs_text)
                png = screenshot_png(obs)
                shot_path = self.rec.save_screenshot(t, png)
                png_llm = png if cfg.vision else None
                self.kb.update(obs, obs_text)
                if not self.ticket:
                    self.read_ticket(obs)
                url = obs.get("url", "")
                self.push("step_start", t=t, url=url, page_sig=page_signature(url))

                # ---- decide the action (teacher or oracle); observer predicts blind, concurrently
                obs_fut = pool.submit(observer.predict, self.ticket, obs_text, self.history, t, url, png_llm) if observer else None
                lat: dict = {}
                if cfg.mode == "oracle":
                    label = subtask_label(self.env.task.subtasks[t])
                    self.push("status", text=f"oracle executing subtask {t+1}/{n_sub}: {label}")
                    t1 = time.time()
                    micro, (obs_after, reward, terminated, truncated, info) = run_subtask(self.env, t)
                    lat["env_s"] = round(time.time() - t1, 2)
                    action = macro_action_str(micro)
                    step = StepRecord(
                        t=t, url_before=url, page_sig=page_signature(url), action=action, action_type="oracle",
                        target_bid=None, target_name=label, activity_label=label, micro_actions=micro,
                    )
                    action_desc = f"completed sub-step: {label}"
                    executed = True
                else:
                    self.push("status", text="teacher thinking…")
                    last_note = self.history[-1][1] if self.history and self.history[-1][1].startswith("(this exact action") else ""
                    out = teacher.act(obs, obs_text, t, png_llm, note=last_note)
                    lat["teacher_s"] = round(out.latency_s, 2)
                    action = out.action
                    step = self.make_step(t, obs, obs_text, action)
                    if out.parse_error:
                        step.latency["parse_error"] = out.parse_error
                        teacher_failures += 1
                        if teacher_failures >= cfg.max_teacher_failures:
                            raise RuntimeError(
                                f"teacher failed {teacher_failures} times in a row (last: {out.parse_error}); aborting run. "
                                "Check the teacher model / prompt (see llm_calls.jsonl)."
                            )
                    else:
                        teacher_failures = 0
                    action_desc = f"{step.action_type} on {element_line(obs_text, step.target_bid) or step.action}"
                    executed = False

                obs_out: ObserverOut | None = None
                if obs_fut:
                    obs_out = obs_fut.result()
                    lat["observer_predict_s"] = round(obs_out.latency_s, 2)
                    # post-action interpretation: the observer infers the rationale and its own confidence (U, A)
                    obs_out = observer.interpret(
                        obs_out, self.ticket, self.kb.text, axtree_snippet(obs_text, step.target_bid), action_desc, self.history, t, png_llm
                    )
                    lat["observer_interpret_s"] = round(obs_out.interp_latency_s, 2)
                    step.observer = {k: v for k, v in obs_out.as_dict().items() if k not in ("latency_s", "interp_latency_s")}
                step.obs_path = obs_path
                step.screenshot_path = shot_path
                step.latency.update(lat)
                boundary, label = self.segmenter.update(step, obs_out.is_new_subgoal if obs_out else False)
                step.boundary = boundary
                if step.action_type != "oracle":
                    step.activity_label = (obs_out.activity_label if obs_out and obs_out.ok and obs_out.activity_label else label)

                if step.latency.get("parse_error"):
                    step.boundary = False  # a fallback noop is not the teacher's decision; never ask about it
                    obs_out = None
                decision = self.selector.score(step, obs_out)
                step.selector = decision.as_dict()
                self.log(
                    f"[dim]t={t:02d}[/] {step.activity_label[:28]:<28} {action[:56]:<56} "
                    f"{'◆' if decision.is_decision else '·'} g={decision.g:.2f} τ={decision.tau:.2f} {'⚡' if decision.fire else ''}"
                    + (f"\n      [dim]observer thinks:[/] {obs_out.rationale[:110]} (conf {obs_out.confidence:.1f})" if obs_out and obs_out.interp_ok else "")
                )
                self.push("scored", t=t, action=action, action_type=step.action_type, target=step.target_name, label=step.activity_label,
                          boundary=boundary, selector=step.selector,
                          observer={"rationale": obs_out.rationale, "confidence": obs_out.confidence, "alternatives": obs_out.alt_rationales} if obs_out and obs_out.interp_ok else None)

                if decision.fire and self.answerer:
                    self.push("status", text="PAUSED — asking why")
                    self.ask(t, step, decision, obs_text, action_desc, png_llm)

                # ---- execute (teacher mode) and record outcome
                if not executed:
                    self.push("status", text=f"executing {action[:50]}")
                    t1 = time.time()
                    obs_after, reward, terminated, truncated, info = self.env.step(action)
                    lat["env_s"] = round(time.time() - t1, 2)
                    err = obs_after.get("last_action_error", "")
                    reps = sum(1 for a, _ in self.history[-6:] if a == action)
                    if reps >= 1 and obs_after.get("url", "") == url and not err:
                        err = f"(this exact action was already taken {reps} time(s) recently and the page did not change; try a different element or approach)"
                    self.history.append((action, err))
                else:
                    self.history.append((f"[{label}]", ""))
                faulthandler.cancel_dump_traceback_later()
                faulthandler.dump_traceback_later(180, repeat=True, file=hang_log)  # re-arm per step
                step.after = {
                    "url": obs_after.get("url", ""),
                    "last_action_error": obs_after.get("last_action_error", ""),
                    "reward": reward,
                    "terminated": terminated,
                    "valid_index": getattr(self.env.task, "valid_index", None),  # post-hoc eval only
                }
                self.rec.write_step(step)
                self.selector.observe_outcome(step)
                self.push("step_done", t=t, error=step.after["last_action_error"], reward=reward)
                obs = obs_after
                if terminated or truncated:
                    break
            pool.shutdown(wait=False)
            steps_done = t + 1
        except KeyboardInterrupt:
            error = "interrupted"
            steps_done = t
        except Exception:  # noqa: BLE001
            error = traceback.format_exc()[-2000:]
            steps_done = t
            self.log(f"[red]{error}[/]")
        finally:
            faulthandler.cancel_dump_traceback_later()
            if hang_log:  # keep hang_traces.txt only if the watchdog actually fired
                hang_log.close()
                hang_path = self.rec.dir / "hang_traces.txt"
                if hang_path.exists() and hang_path.stat().st_size == 0:
                    hang_path.unlink()
            success = reward >= 1.0
            self.rec.finalize(success=success, reward=reward, steps=steps_done, questions_fired=self.questions_fired, error=error)
            self.push("run_end", success=success, reward=reward, steps=steps_done, questions=self.questions_fired, cost=self.rec.cost)
            if screen:
                time.sleep(2)  # let the panel show the final state before the recording stops
                mov = screen.stop()
                self.rec.set_run_meta(screen_recording=str(mov.relative_to(self.rec.dir)) if mov else None)
                self.log(f"🎥 saved {mov}" if mov else "[yellow]🎥 no recording was saved[/]")
            self.log(f"[bold]{'✅ SUCCESS' if success else '❌ not solved'}[/] steps={steps_done} questions={self.questions_fired} "
                     f"cost=${self.rec.cost:.3f} → {self.rec.dir}")
            if cfg.panel:
                self.log("panel stays open; press Enter to close everything")
                try:
                    input()
                except EOFError:
                    pass
            try:
                self.env.close()
            except Exception:  # noqa: BLE001
                pass
            if self.panel:
                self.panel.close_window()
        return self.rec._run


@app.command()
def run(
    task: str = typer.Option("basic-expense-management-medium-l3", help="L3 task short name"),
    seed: int = typer.Option(None),
    mode: str = typer.Option("teacher", help="teacher (the method) | oracle (scripted solver, dev aid)"),
    selector: str = typer.Option("scored", help="scored (the gate) | random (baseline) | none (no questions)"),
    budget: int = typer.Option(None, "--budget", "-b", help="question budget (default: 2 expense / 3 priority)"),
    allow_eval_seeds: bool = typer.Option(False, help="allow seeds 0-9 (reserved for evaluation)"),
    answerer: str = typer.Option("llm", help="llm (the teacher answers) | human (experimental: you type the answer in the chat window)"),
    teacher_model: str = typer.Option(None, help="OpenRouter model id for the teacher"),
    observer_model: str = typer.Option(None, help="OpenRouter model id for the observer"),
    explainer_model: str = typer.Option(None, help="oracle mode only: model that answers ask_why"),
    max_steps: int = typer.Option(60),
    slow_mo: int = typer.Option(None, help="ms between browser actions (default: task's own, 1000)"),
    panel: bool = typer.Option(False, "--panel/--no-panel", help="open the live Ask-Why panel window"),
    vision: bool = typer.Option(True, "--vision/--no-vision", help="attach the screenshot to teacher/observer/explainer calls"),
    record: bool = typer.Option(None, "--record/--no-record", help="record the screen to runs/<run_id>/screen.mov (default: on with --panel)"),
    record_browser: bool = typer.Option(False, "--record-browser", help="also save Playwright video of the task browser (no permissions needed)"),
    headless: bool = typer.Option(False, "--headless", help="run the browser without a window (machines with no display)"),
    save_teacher_memory: bool = typer.Option(True, "--save-teacher-memory/--no-save-teacher-memory", help="write the teacher's private notes to teacher_private.jsonl (audit only; never exported)"),
    panel_port: int = typer.Option(8765),
    out_dir: Path = typer.Option(Path("runs")),
    run_id: str = typer.Option(None),
):
    """Run one episode."""
    load_dotenv()
    kw = {k: v for k, v in dict(
        task=task, seed=seed, mode=mode, selector=selector, budget=budget, answerer=answerer, allow_eval_seeds=allow_eval_seeds,
        teacher_model=teacher_model, observer_model=observer_model, explainer_model=explainer_model,
        max_steps=max_steps, slow_mo=slow_mo, panel=panel, panel_port=panel_port, vision=vision, record=record, record_browser=record_browser, headless=headless, save_teacher_memory=save_teacher_memory,
        out_dir=out_dir, run_id=run_id,
    ).items() if v is not None}
    cfg = RunConfig(**kw)
    Run(cfg).run()


@app.command()
def tasks():
    """List the L3 tasks ask-why knows about."""
    from browsergym.workarena import get_all_tasks_agents

    for cls, seed in get_all_tasks_agents(filter="l3"):
        print(cls.get_task_id().split(".")[-1], seed)


if __name__ == "__main__":
    app()
