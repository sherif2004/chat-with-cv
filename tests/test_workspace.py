"""One user must never reach another user's CVs, index, cache or job list."""
import threading
import uuid

import pytest

from cv_chat.rag import cache as cache_module
from cv_chat.rag import ingest, jobs
from cv_chat.services import blob_storage, search_index
from cv_chat.workspace import Workspace

A = Workspace(str(uuid.uuid4()))
B = Workspace(str(uuid.uuid4()))


def test_names_are_per_user_and_valid_for_azure():
    assert A.container != B.container and A.index != B.index
    assert A.container == Workspace(A.user_id).container
    for name in (A.container, B.container):
        assert 3 <= len(name) <= 63 and name == name.lower() and "_" not in name
    assert len(A.index) <= 128 and A.index == A.index.lower()
    assert A.user_id.replace("-", "") in A.container and A.user_id.replace("-", "") in A.index


@pytest.mark.parametrize("bad", ["", "../etc", "cvs-index", "x" * 40, "123e4567-e89b-12d3-a456-42661417400g"])
def test_a_workspace_cannot_be_built_from_free_text(bad):
    with pytest.raises(ValueError):
        Workspace(bad)


def test_blob_calls_use_only_the_users_own_container(monkeypatch):
    used = []

    class Fake:
        def list_blobs(self):
            return []

    monkeypatch.setattr(blob_storage._service, "get_container_client", lambda name: used.append(name) or Fake())
    blob_storage.list_files(A)
    blob_storage.list_files(B)
    assert used == [A.container, B.container]


def test_search_calls_use_only_the_users_own_index(monkeypatch):
    used = []

    class Fake:
        def search(self, *args, **kwargs):
            return []

    monkeypatch.setattr(search_index, "_search_client_for", lambda index: used.append(index) or Fake())
    search_index.list_chunks(A)
    search_index.get_hash(B, "f")
    search_index.hybrid_search(A, "q", [0.0], 3)
    assert used == [A.index, B.index, A.index]


def test_every_search_index_function_takes_a_workspace():
    import inspect

    public = [f for n, f in inspect.getmembers(search_index, inspect.isfunction) if not n.startswith("_") and f.__module__ == search_index.__name__]
    for function in public:
        if function.__name__ == "metadata_filter":  # builds a filter string, touches no index
            continue
        assert next(iter(inspect.signature(function).parameters)) == "ws", function.__name__


def test_caches_are_separate_and_cleared_separately():
    a, b = cache_module.cache_for(A), cache_module.cache_for(B)
    assert a is cache_module.cache_for(A) and a is not b
    a.put(cache_module.ANSWER, "same question", "A's answer", a.token())
    assert b.get(cache_module.ANSWER, "same question") is None
    b.put(cache_module.ANSWER, "same question", "B's answer", b.token())
    a.clear()
    assert a.get(cache_module.ANSWER, "same question") is None
    assert b.get(cache_module.ANSWER, "same question") == "B's answer"


@pytest.fixture
def fake_ingest(monkeypatch):
    calls = {"prepare": [], "sweep": [], "process": []}
    monkeypatch.setattr(ingest, "prepare", lambda ws: calls["prepare"].append(ws))
    monkeypatch.setattr(ingest, "remove_deleted_cvs", lambda ws: calls["sweep"].append(ws))
    monkeypatch.setattr(ingest, "process_cv", lambda ws, name, data, force=False, on_stage=lambda s: None: calls["process"].append((ws, name)) or (3, False))
    return calls


def _wait(queue):
    deadline = threading.Event()
    for _ in range(200):
        if not queue.active():
            return
        deadline.wait(0.01)
    raise AssertionError("jobs did not finish")


def test_each_user_has_their_own_job_queue(fake_ingest):
    qa, qb = jobs.queue_for(A), jobs.queue_for(B)
    assert qa is jobs.queue_for(A) and qa is not qb
    qa.submit([("ada.pdf", b"x")])
    _wait(qa)
    assert [job.file_name for job in qa.snapshot()] == ["ada.pdf"]
    assert qb.snapshot() == []
    assert fake_ingest["process"] == [(A, "ada.pdf")]  # processed in A's workspace, not B's
    assert fake_ingest["prepare"] == [A] and fake_ingest["sweep"] == [A]


def test_clean_up_does_not_run_while_a_file_is_in_progress(fake_ingest):
    queue = jobs.IngestQueue(A, jobs._pool)
    queue._set(jobs.Job("busy.pdf", "processing", stage="embedding"))
    queue.submit([])
    assert fake_ingest["sweep"] == []
