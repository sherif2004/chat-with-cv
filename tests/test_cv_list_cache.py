import uuid

import pytest

from cv_chat.rag import ingest
from cv_chat.rag.cache import cache_for
from cv_chat.workspace import Workspace


@pytest.fixture
def ws():
    return Workspace(str(uuid.uuid4()))


class Blobs:
    def __init__(self, names):
        self.names, self.listed = names, 0

    def list_files(self, ws):
        self.listed += 1
        return list(self.names)


def install(monkeypatch, names):
    blobs = Blobs(names)
    monkeypatch.setattr(ingest.blob_storage, "list_files", blobs.list_files)
    return blobs


def test_asking_again_does_not_list_the_container_again(monkeypatch, ws):
    blobs = install(monkeypatch, ["a.pdf", "b.pdf"])
    assert ingest.list_cvs(ws) == ["a.pdf", "b.pdf"] and ingest.list_cvs(ws) == ["a.pdf", "b.pdf"]
    assert blobs.listed == 1


def test_changing_the_cvs_clears_the_list(monkeypatch, ws):
    blobs = install(monkeypatch, ["a.pdf"])
    ingest.list_cvs(ws)
    blobs.names = ["a.pdf", "b.pdf"]
    cache_for(ws).clear()  # what process_cv and delete_cv do
    assert ingest.list_cvs(ws) == ["a.pdf", "b.pdf"]


def test_the_list_is_asked_for_again_after_it_has_aged(monkeypatch, ws):
    blobs = install(monkeypatch, ["a.pdf"])
    clock = [1000.0]
    monkeypatch.setattr(ingest.time, "monotonic", lambda: clock[0])
    ingest.list_cvs(ws)
    clock[0] += ingest.settings.cache.cv_list_seconds + 1  # a CV deleted in the portal shows up as gone
    blobs.names = []
    assert ingest.list_cvs(ws) == []
    assert blobs.listed == 2


def test_each_user_has_their_own_list(monkeypatch):
    blobs = install(monkeypatch, ["a.pdf"])
    ingest.list_cvs(Workspace(str(uuid.uuid4())))
    ingest.list_cvs(Workspace(str(uuid.uuid4())))
    assert blobs.listed == 2


def test_changing_the_returned_list_does_not_change_the_cached_one(monkeypatch, ws):
    install(monkeypatch, ["a.pdf"])
    ingest.list_cvs(ws).append("junk.pdf")
    assert ingest.list_cvs(ws) == ["a.pdf"]
