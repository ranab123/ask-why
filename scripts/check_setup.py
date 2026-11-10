"""
Check that everything needed to run ask-why is in place, and optionally pin one ServiceNow instance.

    python scripts/check_setup.py          # run all checks
    python scripts/check_setup.py --pin    # also write the fastest healthy instance into .env (SNOW_INSTANCE_*)

Checks, in order:
  1. OpenRouter API key in .env (validated against OpenRouter; no tokens are spent)
  2. Hugging Face login
  3. Access to the gated dataset ServiceNow/WorkArena-Instances
  4. Each ServiceNow instance in the pool: HTTP status, latency, hibernation
"""
import argparse
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def openrouter_ok() -> bool:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        print("[1/4] OpenRouter key MISSING. Put OPENROUTER_API_KEY=sk-or-... in .env (https://openrouter.ai/keys)")
        return False
    try:
        r = requests.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"}, timeout=20)
    except requests.RequestException as e:
        print(f"[1/4] OpenRouter could not be reached: {type(e).__name__}")
        return False
    if r.status_code != 200:
        print(f"[1/4] OpenRouter key REJECTED (HTTP {r.status_code}). Check the key in .env.")
        return False
    print("[1/4] OpenRouter key OK")
    return True


def hf_login_ok() -> bool:
    try:
        from huggingface_hub import whoami

        info = whoami()
        print(f"[2/4] Hugging Face login OK: {info.get('name')}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[2/4] Hugging Face login FAILED: {e}")
        print("      Fix: run `hf auth login` in this terminal.")
        return False


def load_pool():
    from browsergym.workarena.instance import fetch_instances

    try:
        instances = fetch_instances()
    except Exception as e:  # noqa: BLE001
        print(f"[3/4] Instance pool FAILED: {e}")
        print(
            "      Most likely your access request on\n"
            "      https://huggingface.co/datasets/ServiceNow/WorkArena-Instances\n"
            "      has not been approved yet. Check the page: it should say you have access."
        )
        return None
    print(f"[3/4] Instance pool OK: {len(instances)} instances listed")
    return instances


def probe(url: str, timeout: float = 30.0) -> dict:
    t0 = time.time()
    try:
        r = requests.get(url, timeout=timeout)
        dt = time.time() - t0
        hib = "hibernating" in r.text.lower()
        return {"ok": r.status_code < 400 and not hib, "status": r.status_code, "latency": dt, "hibernating": hib}
    except requests.exceptions.Timeout:
        return {"ok": False, "status": "timeout", "latency": time.time() - t0, "hibernating": False}
    except requests.exceptions.ConnectionError as e:
        return {"ok": False, "status": f"conn-error ({type(e.__cause__).__name__ if e.__cause__ else 'n/a'})", "latency": time.time() - t0, "hibernating": False}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", action="store_true", help="write SNOW_INSTANCE_* for the fastest healthy instance into .env")
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args()

    pinned = all(k in os.environ for k in ("SNOW_INSTANCE_URL", "SNOW_INSTANCE_UNAME", "SNOW_INSTANCE_PWD"))
    if pinned:
        print(f"Note: SNOW_INSTANCE_URL is set ({os.environ['SNOW_INSTANCE_URL']}); WorkArena will use it, not the pool.")

    key_ok = openrouter_ok()
    if not hf_login_ok():
        return 1
    instances = load_pool()
    if not instances:
        return 1

    print("[4/4] Probing instances...")
    results = []
    for inst in instances:
        res = probe(inst["url"], args.timeout)
        results.append((inst, res))
        flag = "OK " if res["ok"] else ("HIB" if res["hibernating"] else "BAD")
        print(f"   {flag}  {inst['url']:<55} status={res['status']!s:<22} {res['latency']:5.1f}s")

    healthy = sorted([(i, r) for i, r in results if r["ok"]], key=lambda x: x[1]["latency"])
    if not healthy:
        print("\nNo healthy instance right now. Hibernating instances wake up if you open their URL in a browser;")
        print("timeouts/connection errors usually mean the pool is down or your network blocks it. Retry later.")
        return 2

    best, res = healthy[0]
    print(f"\nHealthy instances: {len(healthy)}/{len(results)}. Fastest: {best['url']} ({res['latency']:.1f}s)")
    lines = [
        f"SNOW_INSTANCE_URL={best['url']}",
        "SNOW_INSTANCE_UNAME=admin",
        f"SNOW_INSTANCE_PWD={best['password']}",
    ]
    if args.pin:
        env_path = ROOT / ".env"
        existing = env_path.read_text() if env_path.exists() else ""
        kept = [l for l in existing.splitlines() if not l.startswith("SNOW_INSTANCE_")]
        env_path.write_text("\n".join(kept + lines) + "\n")
        print(f"Pinned it in {env_path} (SNOW_INSTANCE_*).")
    else:
        print("Runs pick a random instance from the pool. To always use this one: python scripts/check_setup.py --pin")
    print("\nAll checks passed. You can run: python -m askwhy run" if key_ok else "\nServiceNow access is fine, but fix the OpenRouter key above before running.")
    return 0 if key_ok else 1


if __name__ == "__main__":
    sys.exit(main())
