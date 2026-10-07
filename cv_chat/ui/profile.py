"""Opt-in timing of each page run. Start the app with CV_PROFILE=1 and the terminal shows where the time went, for example:

    page run 1830 ms | login check 4 | account menu 2 | imports 1210 | prepare workspace 590 | sidebar 18 | cv list 5 | view 12

When CV_PROFILE is not set, none of this does anything.
"""
import os
import sys
import threading
import time
from contextlib import contextmanager

ENABLED = os.environ.get("CV_PROFILE") == "1"
_local = threading.local()  # several users' page runs happen at once, each in its own thread


def start() -> None:
    if ENABLED:
        _local.started = time.perf_counter()
        _local.phases = []


@contextmanager
def phase(name: str):
    if not ENABLED:
        yield
        return
    begin = time.perf_counter()
    try:
        yield
    finally:
        _local.phases.append((name, (time.perf_counter() - begin) * 1000))


def finish() -> None:
    if ENABLED and getattr(_local, "phases", None) is not None:
        total = (time.perf_counter() - _local.started) * 1000
        parts = " | ".join(f"{name} {ms:.0f}" for name, ms in _local.phases)
        print(f"page run {total:.0f} ms | {parts}", file=sys.stderr, flush=True)
