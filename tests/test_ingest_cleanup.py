import uuid

import pytest

from cv_chat.rag import ingest
from cv_chat.rag.cache import ANSWER, cache_for
from cv_chat.services import search_index
from cv_chat.workspace import Workspace


@pytest.fixture
def ws():
    return Workspace(str(uuid.uuid4()))


class Recorder:
    def __init__(self):
        self.calls = []

    def index_delete(self, ws, name):
        self.calls.append(("index", name))

    def blob_delete(self, ws, name):
        self.calls.append(("blob", name))


@pytest.fixture
def recorder(monkeypatch):
    recorder = Recorder()
    monkeypatch.setattr(ingest.search_index, "delete_file_chunks", recorder.index_delete)
    monkeypatch.setattr(ingest.blob_storage, "delete_file", recorder.blob_delete)
    return recorder


def remember_something(ws):
    cache = cache_for(ws)
    cache.put(ANSWER, "q", "an answer", cache.token())
    return cache


# --- deleting a CV removes it from the search index and from Blob Storage ---------------------------------------------

def test_deleting_a_cv_removes_it_from_the_index_first_and_then_from_blob_storage(ws, recorder):
    ingest.delete_cv(ws, "a.pdf")
    assert recorder.calls == [("index", "a.pdf"), ("blob", "a.pdf")]


def test_deleting_a_cv_forgets_what_was_cached_about_it(ws, recorder):
    cache = remember_something(ws)
    ingest.delete_cv(ws, "a.pdf")
    assert cache.get(ANSWER, "q") is None


def test_if_the_blob_cannot_be_deleted_the_error_is_raised_but_the_cache_is_still_cleared(ws, recorder, monkeypatch):
    def broken(ws, name):
        raise RuntimeError("storage down")

    monkeypatch.setattr(ingest.blob_storage, "delete_file", broken)
    cache = remember_something(ws)
    with pytest.raises(RuntimeError):
        ingest.delete_cv(ws, "a.pdf")
    assert cache.get(ANSWER, "q") is None


def test_if_the_index_cannot_be_cleaned_the_original_is_kept(ws, recorder, monkeypatch):
    def broken(ws, name):
        raise RuntimeError("search down")

    monkeypatch.setattr(ingest.search_index, "delete_file_chunks", broken)
    with pytest.raises(RuntimeError):
        ingest.delete_cv(ws, "a.pdf")
    assert recorder.calls == []  # the blob was not touched: the CV can be deleted again


# --- a CV deleted outside the app (the portal) leaves no chunks behind -------------------------------------------------

def stored(monkeypatch, in_blob, in_index):
    monkeypatch.setattr(ingest.blob_storage, "list_files", lambda ws: in_blob)
    monkeypatch.setattr(ingest.search_index, "list_profiles", lambda ws: {name: {} for name in in_index})


def test_only_the_cvs_whose_file_is_gone_are_removed_from_the_index(ws, recorder, monkeypatch):
    stored(monkeypatch, in_blob=["a.pdf", "b.pdf"], in_index=["a.pdf", "b.pdf", "gone.pdf"])
    ingest.remove_deleted_cvs(ws)
    assert recorder.calls == [("index", "gone.pdf")]


def test_nothing_is_removed_and_the_cache_is_kept_when_nothing_is_gone(ws, recorder, monkeypatch):
    stored(monkeypatch, in_blob=["a.pdf"], in_index=["a.pdf"])
    cache = remember_something(ws)
    ingest.remove_deleted_cvs(ws)
    assert recorder.calls == [] and cache.get(ANSWER, "q") == "an answer"


def test_removing_a_gone_cv_forgets_what_was_cached(ws, recorder, monkeypatch):
    stored(monkeypatch, in_blob=[], in_index=["gone.pdf"])
    cache = remember_something(ws)
    ingest.remove_deleted_cvs(ws)
    assert cache.get(ANSWER, "q") is None


def test_it_reads_one_row_per_cv_and_never_every_chunk(ws, recorder, monkeypatch):
    stored(monkeypatch, in_blob=[], in_index=[])
    assert not hasattr(search_index, "list_chunks") and not hasattr(search_index, "delete_chunks")
    ingest.remove_deleted_cvs(ws)
