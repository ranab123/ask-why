"""Teacher action parsing and the paired export, on synthetic data (no browser, no LLM)."""
import json

import pytest

from askwhy.config import RunConfig
from askwhy.export import export_run
from askwhy.teacher import Teacher

PAGE = "\t[78] button 'All', clickable\n\t\t[321] link 'Knowledge', clickable\n\t[a56] button 'menu'\n"


def test_parse_takes_the_last_action_line_and_strips_brackets():
    assert Teacher.parse("NOTES: read the ticket\nACTION: click('78')") == ("click('78')", None)
    assert Teacher.parse("ACTION: click('[78]')")[0] == "click('78')"  # id copied with brackets from the tree
    assert Teacher.parse("ACTION: click([78])")[0] == "click('78')"
    assert Teacher.parse("ACTION: fill('a56', 'text with [brackets]')")[0] == "fill('a56', 'text with [brackets]')"


def test_parse_rejects_replies_without_a_valid_action():
    assert Teacher.parse("I will think about it")[1] == "no ACTION line"
    assert Teacher.parse("ACTION: launch_missiles()")[0] is None


def test_missing_bid_suggests_the_element_with_the_same_number():
    hint = Teacher.missing_bid("click('a321')", PAGE)  # wrong frame prefix
    assert "not on the current page" in hint and "[321] link 'Knowledge'" in hint
    assert Teacher.missing_bid("click('321')", PAGE) is None
    assert Teacher.missing_bid("noop(1000)", PAGE) is None


def test_eval_seeds_are_refused_for_collection():
    with pytest.raises(ValueError):
        RunConfig(seed=3)
    assert RunConfig(seed=3, allow_eval_seeds=True).seed == 3
    assert RunConfig(task="priority-assignment-medium-l3").budget == 3
    assert RunConfig(task="basic-expense-management-medium-l3").budget == 2


def _fake_run(tmp_path, mode="teacher", success=True):
    run = tmp_path / "run"
    (run / "obs").mkdir(parents=True)
    steps = []
    for t, action in enumerate(["click('78')", "click('a349')", "click('a108')"]):
        (run / "obs" / f"step_{t:03d}.txt").write_text(f"page {t}")
        steps.append({"t": t, "url_before": f"https://example/{t}", "action": action, "obs_path": f"obs/step_{t:03d}.txt",
                      "screenshot_path": f"obs/step_{t:03d}.png"})
    (run / "steps.jsonl").write_text("\n".join(json.dumps(s) for s in steps) + "\n")
    q = {"t": 1, "question": "Why this record?", "ok": True,
         "answer": {"goal": "delete the duplicate", "evidence": "protocol keeps the task-linked one", "alternatives": "", "would_differ": ""}}
    (run / "questions.jsonl").write_text(json.dumps(q) + "\n")
    (run / "teacher_private.jsonl").write_text(json.dumps({"t": 1, "reply": "SECRET PRIVATE NOTES"}) + "\n")
    (run / "run.json").write_text(json.dumps({"run_id": "r", "task_id": "task", "config": {"mode": mode}, "result": {"success": success}}))
    return run


def test_export_pairs_are_aligned_and_differ_only_in_explanations(tmp_path):
    run = _fake_run(tmp_path)
    assert export_run(run) == (3, 1)
    a = [json.loads(l) for l in (run / "export" / "action_only.jsonl").read_text().splitlines()]
    r = [json.loads(l) for l in (run / "export" / "rationale.jsonl").read_text().splitlines()]
    assert [x["t"] for x in a] == [x["t"] for x in r] == [0, 1, 2]
    # identical demonstrated actions in both versions
    assert [x["messages"][-1]["content"].splitlines()[-1] for x in r] == [x["messages"][-1]["content"] for x in a]
    # the questioned step teaches its rationale; later steps see the earlier explanation in their input
    assert r[1]["messages"][-1]["content"].startswith("RATIONALE: goal=delete the duplicate")
    assert "protocol keeps the task-linked one" in r[2]["messages"][1]["content"]
    assert "protocol keeps the task-linked one" not in r[1]["messages"][1]["content"]
    assert "Explanations" not in a[2]["messages"][1]["content"]
    # the teacher's private memory never reaches the training data
    assert "SECRET" not in (run / "export" / "rationale.jsonl").read_text()


def test_export_refuses_oracle_and_unsuccessful_runs(tmp_path):
    with pytest.raises(ValueError):
        export_run(_fake_run(tmp_path / "a", mode="oracle"))
    assert export_run(_fake_run(tmp_path / "b", success=False)) == (0, 0)
