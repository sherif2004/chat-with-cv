import uuid

import pytest

from cv_chat.ui import library
from cv_chat.workspace import Workspace


class Queue:
    def __init__(self, active):
        self._active = active

    def active(self):
        return self._active


@pytest.fixture
def ws():
    return Workspace(str(uuid.uuid4()))


@pytest.fixture
def fragments(monkeypatch):
    """Stands in for st.fragment: records run_every for each fragment the page defines and runs nothing."""
    seen = []

    def fragment(run_every=None):
        seen.append(run_every)
        return lambda function: (lambda *args, **kwargs: None)

    monkeypatch.setattr(library.st, "fragment", fragment)
    return seen


def processing(monkeypatch, active):
    monkeypatch.setattr(library.jobs, "queue_for", lambda ws: Queue(active))


def test_while_cvs_are_processing_the_page_checks_every_two_seconds(ws, monkeypatch):
    processing(monkeypatch, True)
    assert library.poll_every(ws) == 2


def test_when_nothing_is_processing_the_page_does_not_poll(ws, monkeypatch):
    processing(monkeypatch, False)
    assert library.poll_every(ws) is None


@pytest.mark.parametrize("active, expected", [(True, 2), (False, None)])
def test_the_sidebar_watch_polls_only_while_something_is_processing(ws, monkeypatch, fragments, active, expected):
    processing(monkeypatch, active)
    library.watch(ws)
    assert fragments == [expected]


@pytest.mark.parametrize("active, expected", [(True, 2), (False, None)])
def test_the_library_progress_polls_only_while_something_is_processing(ws, monkeypatch, fragments, active, expected):
    processing(monkeypatch, active)
    library.progress(ws)
    assert fragments == [expected]
