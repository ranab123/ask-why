"""BrowserGym / WorkArena plumbing: task lookup, env creation, observation → text, action parsing."""
from __future__ import annotations

import ast
import base64
import io
import re
from functools import lru_cache
from urllib.parse import unquote

from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.core.env import BrowserEnv
from browsergym.utils.obs import flatten_axtree_to_str

ACTION_SET = HighLevelActionSet(
    subsets=["chat", "infeas", "bid", "nav", "tab"],
    multiaction=False,
    strict=False,
)

# ---------------------------------------------------------------- tasks


@lru_cache(maxsize=1)
def _l3_catalog() -> list[tuple[type, int]]:
    from browsergym.workarena import get_all_tasks_agents

    return get_all_tasks_agents(filter="l3")


def resolve_task(name: str) -> tuple[type, int]:
    """'priority-assignment-medium-l3' → (TaskClass, default_seed). Accepts full ids too."""
    short = name.split(".")[-1]
    for cls, seed in _l3_catalog():
        if cls.get_task_id().endswith(short):
            return cls, seed
    known = sorted({c.get_task_id().split(".")[-1] for c, _ in _l3_catalog()})
    raise SystemExit(f"Unknown L3 task '{name}'. Known: {known[:60]}")


def make_env_for_class(cls: type, slow_mo: int | None = None, record_video_dir: str | None = None, headless: bool = False) -> BrowserEnv:
    # WorkArena tasks ship their own viewport / slow_mo / timeout; only override slow_mo when asked.
    kw = {"slow_mo": slow_mo} if slow_mo is not None else {}
    if record_video_dir:
        kw["record_video_dir"] = record_video_dir
    return BrowserEnv(
        task_entrypoint=cls,
        headless=headless,
        resizeable_window=not headless,
        action_mapping=ACTION_SET.to_python_code,
        **kw,
    )


# ---------------------------------------------------------------- observation → text

_STEP_LEAK = re.compile(r"^Step \d+", re.I)


def chat_history_text(obs: dict) -> str:
    """Chat messages visible to the agent, minus validator leaks ('Step k has been completed')."""
    lines = []
    for m in obs.get("chat_messages", []):
        if m["role"] == "user" and _STEP_LEAK.match(m["message"] or ""):
            continue
        if m["role"] in ("user", "assistant"):
            lines.append(f"{m['role']}: {m['message']}")
    return "\n".join(lines)


def obs_to_text(obs: dict, max_chars: int = 40_000) -> str:
    try:
        ax = flatten_axtree_to_str(
            obs["axtree_object"],
            extra_properties=obs.get("extra_element_properties"),
            with_clickable=True,
            with_visible=False,
            filter_visible_only=True,
            skip_generic=True,
        )
    except Exception as e:  # noqa: BLE001
        ax = f"<axtree unavailable: {e}>"
    if len(ax) > max_chars:
        ax = ax[:max_chars] + f"\n... [truncated {len(ax) - max_chars} chars]"
    return ax


def page_signature(url: str) -> str:
    """Collapse a ServiceNow URL into a coarse page token, e.g. 'fm_expense_line_list', 'kb_view', 'vtb_task'."""
    u = unquote(unquote(url or ""))
    m = re.search(r"target/([a-zA-Z0-9_]+)\.do", u)
    if m:
        return m.group(1)
    m = re.search(r"/([a-zA-Z0-9_]+)\.do", u)
    if m:
        return m.group(1)
    m = re.search(r"/now/nav/ui/([a-zA-Z0-9_/\-]+)", u)
    if m:
        return m.group(1).strip("/").replace("/", ":")[:40]
    m = re.search(r"https?://[^/]+/([^?#]*)", u)
    return (m.group(1) if m else u)[:40] or "root"


def extract_ticket_text(obs: dict) -> str:
    """Best effort: pull the private-task Description value out of the accessibility tree."""
    txt = obs_to_text(obs, max_chars=200_000)
    # textbox 'Description' value='...' (value may be long and quoted with single or double quotes)
    m = re.search(r"textbox 'Description'[^\n]*?value='((?:[^'\\]|\\.)*)'", txt, re.S)
    if m:
        return m.group(1).replace("\\n", "\n").replace("\\'", "'")
    m = re.search(r"Referring to company protocol.*?(?=\n\[|\Z)", txt, re.S)
    return m.group(0) if m else ""


# ---------------------------------------------------------------- actions

ACTION_TYPES_FORM = {"fill", "select_option", "clear", "check", "uncheck", "upload_file"}
ACTION_TYPES_NAV = {"goto", "go_back", "go_forward", "new_tab", "tab_close", "tab_focus"}


def parse_action(action: str) -> tuple[str, list]:
    """'click(\"a12\")' → ('click', ['a12']). Unparseable → (raw-head, [])."""
    action = action.strip()
    try:
        node = ast.parse(action, mode="eval").body
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            args = []
            for a in node.args:
                try:
                    args.append(ast.literal_eval(a))
                except Exception:  # noqa: BLE001
                    args.append(ast.unparse(a))
            return node.func.id, args
    except SyntaxError:
        pass
    head = re.match(r"[a-zA-Z_]+", action)
    return (head.group(0) if head else "unknown"), []


def target_bid(action_type: str, args: list) -> str | None:
    if action_type in {"click", "dblclick", "hover", "fill", "select_option", "clear", "check", "uncheck", "focus", "press", "upload_file"}:
        if args and isinstance(args[0], str):
            return args[0]
    return None


def element_line(obs_text: str, bid: str | None) -> str:
    """The axtree line for a bid, e.g. \"[a318] button 'Delete'\"."""
    if not bid:
        return ""
    m = re.search(rf"^\s*\[{re.escape(bid)}\]\s*(.*)$", obs_text, re.M)
    return m.group(1).strip() if m else ""


def element_name(line: str) -> str:
    m = re.search(r"'((?:[^'\\]|\\.)*)'", line)
    return m.group(1) if m else ""


def axtree_snippet(obs_text: str, bid: str | None, radius: int = 25) -> str:
    lines = obs_text.splitlines()
    if not bid:
        return "\n".join(lines[:2 * radius])
    for i, ln in enumerate(lines):
        if f"[{bid}]" in ln:
            return "\n".join(lines[max(0, i - radius): i + radius])
    return "\n".join(lines[:2 * radius])


# ---------------------------------------------------------------- screenshots


def screenshot_png(obs: dict) -> bytes | None:
    """obs['screenshot'] (H,W,3 uint8) → PNG bytes."""
    arr = obs.get("screenshot")
    if arr is None:
        return None
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def image_part(png: bytes | None, detail: str = "auto") -> dict | None:
    """OpenAI-style image content part for a chat message."""
    if not png:
        return None
    b64 = base64.b64encode(png).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": detail}}


def user_content(text: str, png: bytes | None) -> str | list:
    part = image_part(png)
    return [{"type": "text", "text": text}, part] if part else text
