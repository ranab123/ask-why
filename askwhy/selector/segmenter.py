"""TMI-style segmentation of low-level browser events into activities and decision boundaries (rule-based v1)."""
from __future__ import annotations

import re

from ..recorder import StepRecord

_LABELS = [
    (re.compile(r"kb_|knowledge", re.I), "read policy (knowledge base)"),
    (re.compile(r"vtb_task|task\.do", re.I), "work on assigned ticket"),
    (re.compile(r"expense", re.I), "manage expense lines"),
    (re.compile(r"incident", re.I), "handle incidents"),
    (re.compile(r"task_list|my work", re.I), "review my work"),
    (re.compile(r"home|nav/ui$|^root$", re.I), "navigate"),
]


def label_for(page_sig: str, action_type: str, target_name: str) -> str:
    name = (target_name or "").lower()
    if action_type in ("fill", "press") and ("search" in name or "filter" in name or "keyword" in name):
        return "search / filter records"
    if action_type == "click" and re.search(r"delete", name):
        return "delete record"
    if action_type in ("click",) and re.search(r"update|save|submit|close", name):
        return "save / submit record"
    if action_type in ("fill", "select_option") and name:
        return f"edit field: {target_name[:30]}"
    for pat, lbl in _LABELS:
        if pat.search(page_sig or ""):
            return lbl
    return "navigate"


class Segmenter:
    def __init__(self):
        self.prev_sig: str | None = None
        self.prev_label: str | None = None

    def update(self, step: StepRecord, observer_new_subgoal: bool = False) -> tuple[bool, str]:
        if step.action_type == "oracle" and step.activity_label:
            label = step.activity_label  # subtask class name already describes the activity
        else:
            label = label_for(step.page_sig, step.action_type, step.target_name)
        boundary = (
            self.prev_sig is not None and step.page_sig != self.prev_sig
        ) or (self.prev_label is not None and label != self.prev_label) or observer_new_subgoal
        self.prev_sig, self.prev_label = step.page_sig, label
        return boundary, label
