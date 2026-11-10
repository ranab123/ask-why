"""Ask-Why panel: a tiny local web app (FastAPI in a daemon thread) that the runner pushes events to,
plus a helper that opens it as its own chromeless Chromium window for screen recording."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

STATIC = Path(__file__).parent / "static"


class PanelClient:
    def __init__(self, port: int = 8765, window: bool = True, position: tuple[int, int] | None = None, size: tuple[int, int] = (600, 980)):
        self.port = port
        self.events: list[dict] = []
        self.lock = threading.Lock()
        self.window = window
        self.position = position or _default_position(size)
        self.size = size
        self._proc: subprocess.Popen | None = None

    # ---- producer side
    def push(self, event: str, **data):
        with self.lock:
            self.events.append({"i": len(self.events), "event": event, "ts": time.time(), **data})

    # ---- server
    def _app(self) -> FastAPI:
        app = FastAPI()

        nocache = {"Cache-Control": "no-store"}

        @app.get("/", response_class=HTMLResponse)
        def index():
            return HTMLResponse((STATIC / "index.html").read_text(), headers=nocache)

        @app.get("/state")
        def state(since: int = 0):
            with self.lock:
                return JSONResponse({"events": self.events[since:], "total": len(self.events)}, headers=nocache)

        return app

    def start(self):
        config = uvicorn.Config(self._app(), host="127.0.0.1", port=self.port, log_level="warning")
        server = uvicorn.Server(config)
        threading.Thread(target=server.run, daemon=True).start()
        for _ in range(50):
            if server.started:
                break
            time.sleep(0.1)
        # the window itself is opened by the runner when the agent is about to start (open_window)

    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def open_window(self):
        if not self.window or self._proc:
            return
        subprocess.run(["pkill", "-f", "askwhy-panel-"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # stale panels
        exe = _chromium_executable()
        if exe:
            profile = tempfile.mkdtemp(prefix="askwhy-panel-")
            args = [
                exe,
                f"--app={self.url()}",
                f"--window-size={self.size[0]},{self.size[1]}",
                f"--window-position={self.position[0]},{self.position[1]}",
                f"--user-data-dir={profile}",
                "--no-first-run",
                "--no-default-browser-check",
                # keep the page live when it is behind other windows (otherwise Chromium throttles its timers to
                # about once a minute and the panel looks frozen on the recording)
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
                "--disable-backgrounding-occluded-windows",
                "--disable-features=CalculateNativeWinOcclusion,IntensiveWakeUpThrottling",
            ]
            env = {**os.environ, "GOOGLE_API_KEY": "no", "GOOGLE_DEFAULT_CLIENT_ID": "no", "GOOGLE_DEFAULT_CLIENT_SECRET": "no"}
            self._proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        else:
            webbrowser.open(self.url())

    def close_window(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()


def _chromium_executable() -> str | None:
    """Playwright's bundled Chromium if available, else Google Chrome on macOS."""
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            exe = p.chromium.executable_path
        if exe and os.path.exists(exe):
            return exe
    except Exception:  # noqa: BLE001
        pass
    for cand in (
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        shutil.which("chromium"),
        shutil.which("google-chrome"),
    ):
        if cand and os.path.exists(cand):
            return cand
    return None


def _default_position(size: tuple[int, int]) -> tuple[int, int]:
    """Right edge of the main screen (so the panel sits beside the task browser), else a safe fallback."""
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        root.destroy()
        return (max(0, w - size[0]), 0)
    except Exception:  # noqa: BLE001
        return (900, 0)
