"""Oracle executor: drives the browser with the task's own cheat(), one subtask at a time, while
recording the Playwright micro-actions it performs so each subtask becomes one (obs, macro-action) step."""
from __future__ import annotations

import contextlib
import logging
import re
from typing import Any

from playwright.sync_api import Frame, Locator, Page

_PATCHED = ["click", "dblclick", "fill", "type", "press", "select_option", "check", "uncheck", "hover", "set_input_files"]


def _loc_selector(loc: Locator) -> str:
    m = re.search(r"selector='(.*)'>?$", repr(loc))
    return m.group(1) if m else repr(loc)


@contextlib.contextmanager
def capture_micro_actions(sink: list[dict[str, Any]]):
    """Monkeypatch Locator/Page/Frame methods so cheat()'s Playwright calls are logged into `sink`."""
    originals: list[tuple[type, str, Any]] = []

    def wrap(cls, name):
        orig = getattr(cls, name)

        def patched(self, *args, **kwargs):
            target = _loc_selector(self) if isinstance(self, Locator) else getattr(self, "url", "")
            sink.append({"method": name, "target": target[:200], "args": [str(a)[:120] for a in args]})
            return orig(self, *args, **kwargs)

        setattr(cls, name, patched)
        originals.append((cls, name, orig))

    for n in _PATCHED:
        if hasattr(Locator, n):
            wrap(Locator, n)
    for cls in (Page, Frame):
        if hasattr(cls, "goto"):
            wrap(cls, "goto")
    try:
        yield
    finally:
        for cls, name, orig in originals:
            setattr(cls, name, orig)


def subtask_label(subtask) -> str:
    name = type(subtask).__name__.replace("Task", "")
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    cfg = getattr(subtask, "fixed_config", None) or {}
    if isinstance(cfg, dict):
        extra = cfg.get("module") or cfg.get("short_description") or ""
        if extra:
            return f"{name}: {extra}"[:80]
    return name[:80]


def macro_action_str(micro: list[dict]) -> str:
    parts = [f"{m['method']}({m['target'][:40]!r}{', ' + repr(m['args'][0][:30]) if m['args'] else ''})" for m in micro[:6]]
    more = f", ... +{len(micro) - 6}" if len(micro) > 6 else ""
    return "oracle([" + ", ".join(parts) + more + "])"


def run_subtask(env, idx: int) -> tuple[list[dict], tuple]:
    """Execute subtask idx via cheat(), then run BrowserEnv.post_step to validate and observe."""
    micro: list[dict] = []
    try:
        with capture_micro_actions(micro):
            env.task.cheat(env.page, env.chat.messages, idx)
    except Exception as e:  # noqa: BLE001  (WorkArena's menu cheats occasionally time out; retry once from home)
        logging.warning("cheat(%d) failed (%s); retrying once from the home page", idx, type(e).__name__)
        env.page.goto(env.task.instance.snow_url + "/now/nav/ui/home")
        env.page.wait_for_timeout(3000)
        micro.append({"method": "retry", "target": "home", "args": []})
        with capture_micro_actions(micro):
            env.task.cheat(env.page, env.chat.messages, idx)
    obs, reward, terminated, truncated, info = env.post_step({}, validate=True)
    return micro, (obs, reward, terminated, truncated, info)
