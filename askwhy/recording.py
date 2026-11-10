"""Automatic screen recording of a run (macOS), saved next to the trajectory as runs/<run_id>/screen.mov.

Uses the built-in `screencapture -v` (no extra dependencies). The first time, macOS asks to allow the terminal /
app that launched the run to record the screen (System Settings → Privacy & Security → Screen Recording).
"""
from __future__ import annotations

import logging
import platform
import shutil
import signal
import subprocess
import time
from pathlib import Path

log = logging.getLogger(__name__)


class ScreenRecorder:
    def __init__(self, out_path: Path, region: tuple[int, int, int, int] | None = None, show_clicks: bool = True):
        self.out_path = Path(out_path)
        self.region = region  # (x, y, w, h) in screen points; None = whole main display
        self.show_clicks = show_clicks
        self.proc: subprocess.Popen | None = None
        self.started_at: float | None = None

    @staticmethod
    def available() -> bool:
        return platform.system() == "Darwin" and shutil.which("screencapture") is not None

    def start(self) -> bool:
        if not self.available():
            log.warning("screen recording is only supported on macOS (screencapture not found)")
            return False
        args = ["screencapture", "-v", "-x"]  # -v video, -x no sound effect
        if self.show_clicks:
            args.append("-k")
        if self.region:
            x, y, w, h = self.region
            args.append(f"-R{x},{y},{w},{h}")
        args.append(str(self.out_path))
        self.proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.started_at = time.time()
        time.sleep(1.0)
        if self.proc.poll() is not None:  # died immediately → most likely no permission
            err = (self.proc.stderr.read() or b"").decode(errors="ignore").strip()
            log.warning("screen recording failed to start: %s", err or "screencapture exited")
            self.proc = None
            return False
        return True

    def stop(self) -> Path | None:
        if not self.proc:
            return None
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)  # screencapture finalises the .mov on SIGINT
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
        self.proc = None
        return self.out_path if self.out_path.exists() else None
