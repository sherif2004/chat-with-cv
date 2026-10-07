"""The stored CV address, the expiring read link, and the file-name check."""
import uuid
from types import SimpleNamespace

import pytest

from cv_chat.rag import ingest
from cv_chat.services import blob_storage, search_index
from cv_chat.workspace import Workspace

A = Workspace(str(uuid.uuid4()))
B = Workspace(str(uuid.uuid4()))


@pytest.mark.parametrize("name", ["ada.pdf", "Ada Lovelace (CV) 2024.docx", "سيرة ذاتية.pdf", "a" * 200])
def test_plain_file_names_are_accepted(name):
    ingest.validate_file_name(name)


@pytest.mark.parametrize(
    "name", ["", "   ", "..", ".", "../x.pdf", "a/b.pdf", "a\\b.pdf", "a\x00b.pdf", "a\nb.pdf", "a\x7fb.pdf", "a" * 201]
)
def test_unsafe_file_names_are_refused(name):
    with pytest.raises(ValueError):
        ingest.validate_file_name(name)


def test_an_unsafe_name_is_refused_before_anything_reaches_azure(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("Azure must not be called")

    monkeypatch.setattr(ingest, "extract_cv", boom)
    monkeypatch.setattr(blob_storage, "upload_file", boom)
    monkeypatch.setattr(search_index, "get_hash", boom)
    with pytest.raises(ValueError, match="must not contain"):
        ingest.process_cv(A, "../../other-user.pdf", b"data")


def test_blob_url_points_into_the_users_own_container():
    url = blob_storage.blob_url(A, "Ada Lovelace (1).pdf")
    assert f"/{A.container}/" in url and B.container not in url
    assert "Ada%20Lovelace%20%281%29.pdf" in url  # encoded, so a name cannot add to the address
    assert "?" not in url  # nothing secret is stored in the index


def test_read_link_is_signed_read_only_expiring_and_for_the_users_container():
    link = blob_storage.read_link(A, "ada.pdf")
    path, query = link.split("?", 1)
    assert f"/{A.container}/ada.pdf" in path
    params = dict(part.split("=", 1) for part in query.split("&"))
    assert params["sp"] == "r" and "se" in params and "sig" in params  # read only, with an expiry and a signature
    assert blob_storage.read_link(B, "ada.pdf").split("?")[0] != path


def test_no_read_link_without_an_account_key(monkeypatch):
    monkeypatch.setattr(blob_storage._service, "credential", SimpleNamespace(account_key=None))
    assert blob_storage.read_link(A, "ada.pdf") is None


def test_every_indexed_chunk_carries_the_cv_address(monkeypatch):
    uploaded = []
    document = SimpleNamespace(pages=[(1, "Ada Lovelace\nPython engineer")], headers=[])
    monkeypatch.setattr(ingest, "extract_cv", lambda name, data: document)
    monkeypatch.setattr(ingest.metadata, "extract_metadata", lambda pages: {
        "candidate_name": "Ada", "job_title": "Engineer", "years_experience": 3.0, "email": "", "phone": "", "location": ""})
    monkeypatch.setattr(ingest.openai_service, "embed", lambda texts: [[0.0] for _ in texts])
    monkeypatch.setattr(search_index, "get_hash", lambda ws, file_id: None)
    monkeypatch.setattr(search_index, "upload_chunks", lambda ws, docs: uploaded.extend(docs))
    monkeypatch.setattr(search_index, "delete_stale_chunks", lambda *args: None)
    monkeypatch.setattr(blob_storage, "upload_file", lambda *args: None)

    chunks, skipped = ingest.process_cv(A, "ada.pdf", b"data", force=True)

    assert chunks == len(uploaded) > 0 and not skipped
    assert all(doc["file_url"] == blob_storage.blob_url(A, "ada.pdf") and doc["file_name"] == "ada.pdf" for doc in uploaded)


def test_search_results_carry_the_cv_address(monkeypatch):
    class Fake:
        def search(self, *args, **kwargs):
            assert "file_url" in kwargs["select"]
            return [{"file_name": "ada.pdf", "file_url": "https://x/ada.pdf", "section": "Skills", "page": 1, "content": "Python"}]

    monkeypatch.setattr(search_index, "_search_client_for", lambda index: Fake())
    [chunk] = search_index.hybrid_search(A, "python", [0.0], 3)
    assert chunk["file_url"] == "https://x/ada.pdf"


def test_the_address_is_not_shown_to_the_model():
    from cv_chat.rag import retrieval

    chunk = {"file_name": "ada.pdf", "file_url": "https://secret.example/ada.pdf", "section": "Skills", "page": 1, "content": "Python"}
    assert "https://" not in retrieval.format_excerpts([chunk])


def test_the_fingerprint_changes_with_the_index_version(monkeypatch):
    before = ingest._pipeline_fingerprint()
    ingest._pipeline_fingerprint.cache_clear()
    monkeypatch.setattr(ingest, "INDEX_VERSION", ingest.INDEX_VERSION + 1)
    try:
        assert ingest._pipeline_fingerprint() != before  # CVs indexed by the older version are re-indexed
    finally:
        ingest._pipeline_fingerprint.cache_clear()
