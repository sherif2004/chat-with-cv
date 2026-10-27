import uuid

import pytest

from cv_chat.services import blob_storage
from cv_chat.workspace import Workspace


@pytest.fixture
def ws():
    return Workspace(str(uuid.uuid4()))


@pytest.fixture
def signing(monkeypatch):
    """Counts how many links are signed, and lets a test move the clock."""
    state = {"signed": 0, "now": 1000.0}

    def sign(**kwargs):
        state["signed"] += 1
        return f"token-{state['signed']}"

    monkeypatch.setattr(blob_storage, "generate_blob_sas", sign)
    monkeypatch.setattr(blob_storage.time, "monotonic", lambda: state["now"])
    monkeypatch.setattr(blob_storage, "_links", {})
    return state


def test_the_same_link_is_reused_instead_of_signing_it_on_every_click(ws, signing):
    first = blob_storage.read_link(ws, "a.pdf")
    assert blob_storage.read_link(ws, "a.pdf") == first
    assert signing["signed"] == 1


def test_a_new_link_is_signed_once_the_old_one_is_older_than_the_cache_time(ws, signing):
    first = blob_storage.read_link(ws, "a.pdf")
    signing["now"] += blob_storage.settings.cache.link_seconds + 1
    assert blob_storage.read_link(ws, "a.pdf") != first
    assert signing["signed"] == 2


def test_every_cv_and_every_user_has_their_own_link(ws, signing):
    other = Workspace(str(uuid.uuid4()))
    links = {blob_storage.read_link(ws, "a.pdf"), blob_storage.read_link(ws, "b.pdf"), blob_storage.read_link(other, "a.pdf")}
    assert len(links) == 3 and signing["signed"] == 3


def test_the_cache_does_not_grow_without_limit(ws, signing, monkeypatch):
    monkeypatch.setattr(blob_storage, "settings", blob_storage.settings.model_copy(update={"cache": blob_storage.settings.cache.model_copy(update={"links": 3})}))
    for n in range(10):
        blob_storage.read_link(ws, f"cv{n}.pdf")
    assert len(blob_storage._links) == 3


def test_a_link_stays_valid_longer_than_it_is_cached():
    assert blob_storage.settings.cache.link_seconds < 60 * 60  # links are signed for an hour
