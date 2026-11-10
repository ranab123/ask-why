"""
Play WorkArena++ L3 tasks by hand.

Usage:
    python scripts/play_l3.py                                # cycles through the 4 tasks below
    python scripts/play_l3.py priority-assignment-medium-l3  # only tasks whose id contains this text

For each task a real Chromium window opens on a ServiceNow instance.
Do the task yourself in that window, then come back to this terminal and
press Enter. The script checks your work and prints PASS/FAIL.
"""
import sys

from browsergym.core.env import BrowserEnv
from browsergym.workarena import get_all_tasks_agents

WANTED = [
    "basic-expense-management-medium-l3",
    "date-based-expense-management-medium-l3",
    "amount-based-expense-management-medium-l3",
    "priority-assignment-medium-l3",
]
if len(sys.argv) > 1:
    WANTED = [w for w in WANTED if sys.argv[1] in w] or [sys.argv[1]]

# (task_class, seed) pairs for every L3 task
l3 = get_all_tasks_agents(filter="l3")

# keep one seed per wanted task
picked = {}
for task_cls, seed in l3:
    tid = task_cls.get_task_id()
    for w in WANTED:
        if tid.endswith(w) and w not in picked:
            picked[w] = (task_cls, seed)

missing = [w for w in WANTED if w not in picked]
if missing:
    print("Not found in the L3 list:", missing)
    print("Some L3 ids that do exist:")
    for t, _ in l3[:40]:
        print("  ", t.get_task_id())

for name, (task_cls, seed) in picked.items():
    print(f"\n=== {name} (seed {seed}) ===")
    env = BrowserEnv(task_entrypoint=task_cls, headless=False, timeout=30000)
    obs, info = env.reset(seed=seed)
    print("GOAL (also shown in the chat panel in the browser):\n", obs.get("goal"))
    input("\nDo the task in the browser window, then press Enter here to check it...")
    reward, done, message, info = env.task.validate(env.page, env.chat.messages)
    print("PASS ✅" if reward == 1 else f"FAIL ❌  {info.get('message', '') or message}")
    input("Press Enter to close this task and go to the next one...")
    env.close()
