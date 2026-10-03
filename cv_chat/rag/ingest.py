"""Ingestion pipeline: CV file -> searchable chunks in Azure AI Search, original file in Blob Storage.

CVs are processed in parallel threads: the work is I/O-bound (HTTP calls to Azure).
"""
import hashlib
from functools import lru_cache
from pathlib import Path
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

from cv_chat import config
from cv_chat.processing import chunking, extract, sections
from cv_chat.processing.chunking import chunk_cv
from cv_chat.processing.extract import extract_cv
from cv_chat.processing.sections import section_type
from cv_chat.rag import metadata
from cv_chat.rag.cache import cache
from cv_chat.services import blob_storage, openai_service, search_index


@lru_cache(maxsize=1)
def _pipeline_fingerprint() -> bytes:
    """Changes whenever the extraction or chunking code or its settings change, so those CVs get re-indexed by themselves."""
    digest = hashlib.md5(f"{config.CHUNK_SIZE}/{config.CHUNK_OVERLAP}".encode())
    for module in (extract, chunking, sections, metadata):
        digest.update(Path(module.__file__).read_bytes())
    return digest.digest()


def file_id_for(file_name: str) -> str:
    """Stable id per file: processing the same file again overwrites its chunks instead of duplicating them."""
    return hashlib.md5(file_name.encode()).hexdigest()


def process_cv(
    file_name: str, data: bytes, force: bool = False, on_stage: Callable[[str], None] = lambda stage: None
) -> tuple[int, bool]:
    """Index one CV and store the original file. Returns (chunks, skipped); skipped means it was already indexed.

    on_stage is told what the CV is doing, for the live status in the sidebar.
    """
    file_id = file_id_for(file_name)
    content_hash = hashlib.md5(data + _pipeline_fingerprint()).hexdigest()
    if not force and search_index.get_hash(file_id) == content_hash:
        return 0, True

    on_stage("reading layout")
    document = extract_cv(file_name, data)
    if not any(text.strip() for _, text in document.pages):
        raise ValueError("No text found (scanned PDF?)")

    on_stage("reading name, title and experience")
    meta = metadata.extract_metadata(document.pages)
    chunks = chunk_cv(document.pages, document.headers, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    on_stage("embedding")
    # Embed each chunk together with its file name, so chunks from later pages still point to the candidate.
    who = f"\nCandidate: {meta['candidate_name']}, {meta['job_title']}" if meta["candidate_name"] else ""
    vectors = openai_service.embed([f"CV: {file_name}{who}\n{chunk.text}" for chunk in chunks])

    docs = [
        {
            "id": f"{file_id}-{i}", "file_id": file_id, "file_name": file_name,
            "section": chunk.section, "section_type": section_type(chunk.section), "page": chunk.page,
            "content": chunk.text, "content_hash": content_hash, "content_vector": vector, **meta,
        }
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    on_stage("saving")
    search_index.upload_chunks(docs)  # upload first, then drop leftovers, so the CV is never missing from the index
    search_index.delete_stale_chunks(file_id, {doc["id"] for doc in docs})
    blob_storage.upload_file(file_name, data)
    cache.clear()  # cached searches and answers may not know this CV's new content
    return len(chunks), False


def delete_cv(file_name: str) -> None:
    """Remove a CV everywhere. The index goes first: a CV left in the index but not in storage would still be quoted."""
    try:
        search_index.delete_file_chunks(file_id_for(file_name))
        blob_storage.delete_file(file_name)
    finally:  # even a half-finished delete changes what a search returns
        cache.clear()


def prepare() -> None:
    """Create the search index, or bring an older one up to the current fields, before anything searches it."""
    search_index.ensure_index(openai_service.embedding_dimensions())


def process_cvs(files: list[tuple[str, bytes]]) -> Iterator[dict]:
    """Process (file_name, data) pairs in parallel and yield each file's result as soon as it finishes."""
    blob_storage.ensure_container()
    prepare()
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = {pool.submit(process_cv, name, data): name for name, data in files}
        for future in as_completed(futures):
            try:
                chunks, skipped = future.result()
                result = {"file": futures[future], "chunks": chunks, "skipped": skipped, "error": None}
            except Exception as error:  # one bad file never stops the others
                result = {"file": futures[future], "chunks": 0, "skipped": False, "error": str(error)}
            yield result


def list_cvs() -> list[str]:
    return blob_storage.list_files()
