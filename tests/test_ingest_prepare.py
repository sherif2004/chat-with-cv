import uuid

from cv_chat.rag import ingest, jobs
from cv_chat.workspace import Workspace


def fingerprint_with(monkeypatch, **changes) -> bytes:
    changed = ingest.settings.model_copy(update=changes)
    monkeypatch.setattr(ingest, "settings", changed)
    ingest._pipeline_fingerprint.cache_clear()
    try:
        return ingest._pipeline_fingerprint()
    finally:
        ingest._pipeline_fingerprint.cache_clear()


def test_the_fingerprint_is_stable_for_the_same_settings(monkeypatch):
    assert fingerprint_with(monkeypatch) == fingerprint_with(monkeypatch)


def test_a_chunking_setting_changes_the_fingerprint(monkeypatch):
    base = fingerprint_with(monkeypatch)
    chunking = ingest.settings.chunking.model_copy(update={"max_chars": 2500})
    assert fingerprint_with(monkeypatch, chunking=chunking) != base


def test_bumping_the_pipeline_version_changes_the_fingerprint(monkeypatch):
    base = fingerprint_with(monkeypatch)
    bumped = ingest.settings.ingest.model_copy(update={"pipeline_version": 2})
    assert fingerprint_with(monkeypatch, ingest=bumped) != base


def test_a_tunable_that_does_not_shape_the_chunks_leaves_the_fingerprint_alone(monkeypatch):
    base = fingerprint_with(monkeypatch)
    retrieval = ingest.settings.retrieval.model_copy(update={"top_k": 5})
    assert fingerprint_with(monkeypatch, retrieval=retrieval) == base


class InlinePool:
    def __init__(self):
        self.submitted = []

    def submit(self, function, *args):
        self.submitted.append(args)


def queue_with(monkeypatch, rebuilt):
    monkeypatch.setattr(jobs.ingest, "prepare", lambda ws: rebuilt)
    monkeypatch.setattr(jobs.ingest, "remove_deleted_cvs", lambda ws: None)
    pool = InlinePool()
    return jobs.IngestQueue(Workspace(str(uuid.uuid4())), pool), pool


def test_cvs_returned_by_a_rebuild_are_queued_with_the_uploaded_files(monkeypatch):
    queue, pool = queue_with(monkeypatch, ["old1.pdf", "old2.pdf"])
    queue.submit([("new.pdf", b"bytes")])
    assert pool.submitted == [("new.pdf", b"bytes", False), ("old1.pdf", None, False), ("old2.pdf", None, False)]


def test_a_cv_that_is_both_uploaded_and_stored_is_queued_once_with_the_new_bytes(monkeypatch):
    queue, pool = queue_with(monkeypatch, ["cv.pdf"])
    queue.submit([("cv.pdf", b"new bytes")])
    assert pool.submitted == [("cv.pdf", b"new bytes", False)]


def test_nothing_extra_is_queued_when_the_index_is_current(monkeypatch):
    queue, pool = queue_with(monkeypatch, [])
    queue.submit([("new.pdf", b"bytes")])
    assert len(pool.submitted) == 1
