import uuid

from cv_chat.services import search_index
from cv_chat.workspace import Workspace


class FakeClient:
    def __init__(self, ids):
        self.ids, self.filters, self.deleted = ids, [], []

    def search(self, text, filter=None, select=None, **kwargs):
        self.filters.append(filter)
        return [{"id": chunk_id, "content_hash": "h"} for chunk_id in self.ids]

    def delete_documents(self, documents):
        self.deleted += [document["id"] for document in documents]


def install(monkeypatch, ids):
    fake = FakeClient(ids)
    monkeypatch.setattr(search_index, "_client", lambda ws: fake)
    return fake, Workspace(str(uuid.uuid4()))


def test_stale_chunks_are_found_by_file_name_so_chunks_with_an_older_id_scheme_go_too(monkeypatch):
    fake, ws = install(monkeypatch, ["new-0", "old-md5-0", "old-md5-1"])
    search_index.delete_stale_chunks(ws, "o'brien.pdf", {"new-0"})
    assert fake.filters == ["file_name eq 'o''brien.pdf'"]
    assert fake.deleted == ["old-md5-0", "old-md5-1"]


def test_deleting_a_cv_removes_every_chunk_with_its_name(monkeypatch):
    fake, ws = install(monkeypatch, ["a-0", "legacy-0"])
    search_index.delete_file_chunks(ws, "cv.pdf")
    assert fake.deleted == ["a-0", "legacy-0"]


def test_the_stored_hash_is_looked_up_by_file_id_only(monkeypatch):
    fake, ws = install(monkeypatch, ["a-0"])
    assert search_index.get_hash(ws, "cv.pdf") == "h"
    assert fake.filters == ["file_id eq 'cv.pdf'"]
