import uuid

import pytest

from cv_chat.rag import ingest
from cv_chat.services import search_index
from cv_chat.workspace import Workspace


class FakeIndexClient:
    """Stands in for SearchIndexClient: keeps at most one index and records what was done to it."""

    def __init__(self, existing=None):
        self.index, self.calls = existing, []

    def get_index(self, name):
        if self.index is None:
            raise search_index.ResourceNotFoundError("missing")
        return self.index

    def delete_index(self, name):
        self.calls.append("delete")
        self.index = None

    def create_or_update_index(self, index):
        self.calls.append("create")
        self.index = index


@pytest.fixture
def ws():
    return Workspace(str(uuid.uuid4()))


def install(monkeypatch, existing=None) -> FakeIndexClient:
    fake = FakeIndexClient(existing)
    monkeypatch.setattr(search_index, "_index_client", fake)
    return fake


def test_a_new_user_gets_an_index_and_it_needs_filling(monkeypatch, ws):
    fake = install(monkeypatch)
    assert search_index.ensure_index(ws, 3) is True
    assert fake.calls == ["create"]


def test_an_index_that_matches_is_left_alone(monkeypatch, ws):
    fake = install(monkeypatch, search_index.desired_index(ws, 3))
    assert search_index.ensure_index(ws, 3) is False
    assert fake.calls == []


def test_an_index_made_by_an_older_version_is_replaced(monkeypatch, ws):
    older = search_index.desired_index(ws, 3)
    older.fields = [field for field in older.fields if not field.name.startswith(search_index.MARKER_PREFIX)]  # no marker: made before migrations
    fake = install(monkeypatch, older)
    assert search_index.ensure_index(ws, 3) is True
    assert fake.calls == ["delete", "create"]  # the old one goes first: Azure cannot change an analyzer or a key in place


def test_a_different_embedding_size_replaces_the_index_instead_of_failing(monkeypatch, ws):
    fake = install(monkeypatch, search_index.desired_index(ws, 3))
    assert search_index.ensure_index(ws, 1536) is True
    assert fake.calls == ["delete", "create"]


def test_the_marker_depends_on_the_vector_size_and_the_index_version(monkeypatch, ws):
    three, more = search_index.desired_index(ws, 3), search_index.desired_index(ws, 4)
    marker = lambda index: next(field.name for field in index.fields if field.name.startswith(search_index.MARKER_PREFIX))
    assert marker(three) != marker(more)
    assert marker(three) == marker(search_index.desired_index(ws, 3))
    bumped = search_index.settings.model_copy(update={"index": search_index.settings.index.model_copy(update={"version": 99})})
    monkeypatch.setattr(search_index, "settings", bumped)
    assert marker(search_index.desired_index(ws, 3)) != marker(three)


def test_prepare_returns_the_stored_cvs_when_the_index_was_rebuilt(monkeypatch, ws):
    monkeypatch.setattr(ingest.blob_storage, "ensure_container", lambda ws: None)
    monkeypatch.setattr(ingest.openai_service, "embedding_dimensions", lambda: 3)
    monkeypatch.setattr(ingest.blob_storage, "list_files", lambda ws: ["a.pdf", "b.pdf"])
    install(monkeypatch, None)
    assert ingest.prepare(ws) == ["a.pdf", "b.pdf"]
    assert ingest.prepare(ws) == []  # the index is current now: nothing to redo
