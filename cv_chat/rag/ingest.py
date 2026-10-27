"""Ingestion pipeline: CV file -> searchable chunks in Azure AI Search, original file in Blob Storage.

CVs are processed in parallel threads: the work is I/O-bound (HTTP calls to Azure).
"""
import hashlib
from functools import lru_cache
from pathlib import Path
from collections.abc import Callable

from cv_chat.config import settings
from cv_chat.processing import chunking, entities, extract, sections
from cv_chat.processing.chunking import chunk_cv
from cv_chat.processing.extract import extract_cv
from cv_chat.processing.sections import section_type
from cv_chat.rag.cache import cache_for
from cv_chat.services import blob_storage, openai_service, search_index
from cv_chat.workspace import Workspace


INDEX_VERSION = 3  # bump when the stored chunk fields change: older CVs then show up as outdated and are re-indexed


def validate_file_name(file_name: str) -> None:
    """Upload names come from the browser, so refuse anything that is not a plain file name. Raises ValueError.
    The name is the CV's id everywhere (search index, filters), so it is kept short enough to make a valid search key."""
    limit = settings.ingest.max_name_bytes
    if not file_name or not file_name.strip() or len(file_name.encode()) > limit:
        raise ValueError(f"The file name must be 1 to {limit} bytes long (about {limit // 2} letters in Arabic or Chinese).")
    if "/" in file_name or "\\" in file_name or file_name.strip() in (".", ".."):
        raise ValueError("The file name must not contain / or \\ or be . or ..")
    if any(ord(char) < 32 or ord(char) == 127 for char in file_name):
        raise ValueError("The file name must not contain control characters.")


@lru_cache(maxsize=1)
def _pipeline_fingerprint() -> bytes:
    """Changes whenever the extraction or chunking code or its settings change, so those CVs get re-indexed by themselves."""
    digest = hashlib.md5(f"{settings.chunking.max_chars}/{settings.chunking.overlap}/{INDEX_VERSION}".encode())
    for module in (extract, chunking, sections, entities):
        digest.update(Path(module.__file__).read_bytes())
    return digest.digest()


def process_cv(
    ws: Workspace, file_name: str, data: bytes, force: bool = False, on_stage: Callable[[str], None] = lambda stage: None
) -> tuple[int, bool]:
    """Store the original file, then index the CV. Returns (chunks, skipped); skipped means it was already indexed.

    on_stage is told what the CV is doing, for the live status in the sidebar.
    """
    validate_file_name(file_name)
    content_hash = hashlib.md5(data + _pipeline_fingerprint()).hexdigest()
    if not force and search_index.get_hash(ws, file_name) == content_hash:
        return 0, True

    on_stage("saving original")
    blob_storage.upload_file(ws, file_name, data)  # first, so a failure in any later step still leaves the file for Re-index
    on_stage("reading layout")
    document = extract_cv(file_name, data)
    if not any(text.strip() for _, text in document.pages):
        raise ValueError("No text found (scanned PDF?)")

    chunks = chunk_cv(document.pages, document.headers)
    on_stage("reading name, title and contact")
    meta = entities.extract(chunks)
    on_stage("embedding")
    # Embed each chunk together with its file name, so chunks from later pages still point to the candidate.
    who = f"\nCandidate: {meta['candidate_name']}, {meta['job_title']}" if meta["candidate_name"] else ""
    vectors = openai_service.embed([f"CV: {file_name}{who}\n{chunk.text}" for chunk in chunks])

    file_url = blob_storage.blob_url(ws, file_name)
    docs = [
        {
            "id": search_index.chunk_key(file_name, i), "file_id": file_name, "file_name": file_name, "file_url": file_url,
            "section": chunk.section, "section_type": section_type(chunk.section), "page": chunk.page,
            "content": chunk.text, "content_hash": content_hash, "content_vector": vector, **meta,
        }
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    on_stage("saving")
    search_index.upload_chunks(ws, docs)  # upload first, then drop leftovers, so the CV is never missing from the index
    search_index.delete_stale_chunks(ws, file_name, {doc["id"] for doc in docs})
    cache_for(ws).clear()  # cached searches and answers may not know this CV's new content
    return len(chunks), False


def delete_cv(ws: Workspace, file_name: str) -> None:
    """Remove a CV everywhere. The index goes first: a CV left in the index but not in storage would still be quoted."""
    try:
        search_index.delete_file_chunks(ws, file_name)
        blob_storage.delete_file(ws, file_name)
    finally:  # even a half-finished delete changes what a search returns
        cache_for(ws).clear()


def prepare(ws: Workspace) -> None:
    """Create the user's container and search index, or bring an older index up to the current fields, before anything uses them."""
    blob_storage.ensure_container(ws)
    search_index.ensure_index(ws, openai_service.embedding_dimensions())


def destroy(ws: Workspace) -> None:
    """Delete everything a user has in Azure: their container (the original CVs) and their search index."""
    search_index.delete_index(ws)
    blob_storage.delete_container(ws)
    cache_for(ws).clear()


def remove_deleted_cvs(ws: Workspace) -> None:
    """Delete the index chunks of CVs whose file is no longer in Blob Storage (for example deleted in the portal)."""
    files = set(blob_storage.list_files(ws))
    search_index.delete_chunks(ws, [chunk_id for chunk_id, name in search_index.list_chunks(ws) if name not in files])


def list_cvs(ws: Workspace) -> list[str]:
    return blob_storage.list_files(ws)
