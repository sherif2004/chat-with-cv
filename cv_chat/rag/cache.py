"""In-memory cache for repeated questions: router results, search results and (optionally) final answers.

Each user has their own cache (see cache_for), so one user's answer can never be served to another.
Chat replies (greetings, thanks) are cached by their text alone, because they do not depend on the CVs or the chat.
Everything is cleared when the user's CVs change. An entry computed while a CV was being changed is not stored (see token).
"""
import threading
from collections import OrderedDict
from collections.abc import Hashable
from typing import Any

from cv_chat import config
from cv_chat.workspace import Workspace

ROUTE, SEARCH, ANSWER, CHAT = "route", "search", "answer", "chat"


class Cache:
    def __init__(self, size: int):
        self._size = size
        self._lock = threading.Lock()
        self._data: dict[str, OrderedDict[Hashable, Any]] = {}
        self._generation = 0

    def token(self) -> int:
        """Take before computing a value and pass to put(): it is dropped if the cache was cleared in between."""
        with self._lock:
            return self._generation

    def get(self, namespace: str, key: Hashable) -> Any | None:
        with self._lock:
            entries = self._data.get(namespace)
            if entries is None or key not in entries:
                return None
            entries.move_to_end(key)
            return entries[key]

    def put(self, namespace: str, key: Hashable, value: Any, token: int) -> None:
        with self._lock:
            if token != self._generation:  # the CVs changed while this value was computed: it may be out of date
                return
            entries = self._data.setdefault(namespace, OrderedDict())
            entries[key] = value
            entries.move_to_end(key)
            while len(entries) > self._size:
                entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._generation += 1

    def __len__(self) -> int:
        with self._lock:
            return sum(len(entries) for entries in self._data.values())


_caches: dict[str, Cache] = {}
_caches_lock = threading.Lock()


def cache_for(ws: Workspace) -> Cache:
    """The cache of one user."""
    with _caches_lock:
        if ws.user_id not in _caches:
            _caches[ws.user_id] = Cache(config.CACHE_SIZE)
        return _caches[ws.user_id]
