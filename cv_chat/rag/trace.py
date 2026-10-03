"""A record of what happened while one question was answered: each stage, what it used and how long it took.

The chat shows it under the answer. Stages can run in parallel threads (the searches of query expansion), so adding
an event is thread-safe.
"""
import threading
import time
from contextlib import contextmanager

from cv_chat import config


class Trace:
    def __init__(self, expand: bool = False, cache_answers: bool = False):
        self._start = time.perf_counter()
        self._lock = threading.Lock()
        self._events: list[dict] = []
        self.route = ""
        self.total_ms: float | None = None
        self.first_token_ms: float | None = None
        self.settings = {"query_expansion": expand, "answer_cache": cache_answers}

    def now_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000

    def add(self, stage: str, label: str, ms: float, **info) -> None:
        """stage: route | expand | search | agent_model | tool | cache | filter | generate."""
        with self._lock:
            self._events.append({"stage": stage, "label": label, "at_ms": round(self.now_ms() - ms, 1), "ms": round(ms, 1), "info": info})

    @contextmanager
    def timed(self, stage: str, label: str, **info):
        """Time a block. The block may add to the yielded dict; it is stored with the event."""
        extra = dict(info)
        start = time.perf_counter()
        try:
            yield extra
        finally:
            self.add(stage, label, (time.perf_counter() - start) * 1000, **extra)

    def first_token(self) -> None:
        if self.first_token_ms is None:
            self.first_token_ms = round(self.now_ms(), 1)

    def finish(self) -> None:
        self.total_ms = round(self.now_ms(), 1)
        if self.first_token_ms is not None:
            with self._lock:
                busy_until = max((e["at_ms"] + e["ms"] for e in self._events), default=0)
            if self.first_token_ms > busy_until + 1:  # the model took this long to start writing
                self.add("model", config.CHAT_DEPLOYMENT, self.first_token_ms - busy_until, note="until its first word")
            self.add("generate", config.CHAT_DEPLOYMENT, self.total_ms - self.first_token_ms)

    def to_dict(self) -> dict:
        """Plain data (kept with the chat message and drawn by ui/details.py)."""
        with self._lock:
            events = sorted(self._events, key=lambda event: event["at_ms"])
        return {
            "route": self.route, "total_ms": self.total_ms, "first_token_ms": self.first_token_ms, "events": events,
            "settings": self.settings,
            "models": {"chat": config.CHAT_DEPLOYMENT, "embedding": config.EMBEDDING_DEPLOYMENT},
        }
