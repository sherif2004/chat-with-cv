"""Ingestion pipeline: CV file -> searchable chunks in Azure AI Search, original file in Blob Storage.

CVs are processed in parallel threads: the work is I/O-bound (HTTP calls to Azure).
"""
import hashlib
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

from cv_chat import config
from cv_chat.processing.chunking import chunk_cv
from cv_chat.processing.extract import extract_cv
from cv_chat.processing.sections import section_type
from cv_chat.services import blob_storage, openai_service, search_index


def process_cv(file_name: str, data: bytes) -> tuple[int, bool]:
    """Index one CV and store the original file. Returns (chunks, skipped); skipped means it was already indexed."""
    # Stable ids per file: processing the same file again overwrites its chunks instead of duplicating them.
    file_id = hashlib.md5(file_name.encode()).hexdigest()
    content_hash = hashlib.md5(data + str(config.PIPELINE_VERSION).encode()).hexdigest()
    if search_index.get_hash(file_id) == content_hash:
        return 0, True

    document = extract_cv(file_name, data)
    if not any(text.strip() for _, text in document.pages):
        raise ValueError("No text found (scanned PDF?)")

    chunks = chunk_cv(document.pages, document.headers, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    # Embed each chunk together with its file name, so chunks from later pages still point to the candidate.
    vectors = openai_service.embed([f"CV: {file_name}\n{chunk.text}" for chunk in chunks])

    docs = [
        {
            "id": f"{file_id}-{i}", "file_id": file_id, "file_name": file_name,
            "section": chunk.section, "section_type": section_type(chunk.section), "page": chunk.page,
            "content": chunk.text, "content_hash": content_hash, "content_vector": vector,
        }
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    search_index.upload_chunks(docs)  # upload first, then drop leftovers, so the CV is never missing from the index
    search_index.delete_stale_chunks(file_id, {doc["id"] for doc in docs})
    blob_storage.upload_file(file_name, data)
    return len(chunks), False


def process_cvs(files: list[tuple[str, bytes]]) -> Iterator[dict]:
    """Process (file_name, data) pairs in parallel and yield each file's result as soon as it finishes."""
    blob_storage.ensure_container()
    search_index.ensure_index()
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
