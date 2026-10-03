"""Ingestion pipeline: CV file -> searchable chunks in Azure AI Search, original file in Blob Storage.

CVs are processed in parallel threads: the work is I/O-bound (HTTP calls to Azure).
"""
import hashlib
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

from cv_chat import config
from cv_chat.processing.chunking import chunk_text
from cv_chat.processing.extract import extract_text
from cv_chat.services import blob_storage, openai_service, search_index


def process_cv(file_name: str, data: bytes) -> int:
    """Index one CV and store the original file. Returns the number of chunks."""
    text = extract_text(file_name, data)
    if not text.strip():
        raise ValueError("No text found (scanned PDF?)")

    chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    # Embed each chunk together with its file name, so chunks from later pages still point to the candidate.
    vectors = openai_service.embed([f"CV: {file_name}\n{chunk}" for chunk in chunks])

    # Stable ids per file: processing the same file again overwrites its chunks instead of duplicating them.
    file_id = hashlib.md5(file_name.encode()).hexdigest()
    search_index.upload_chunks([
        {"id": f"{file_id}-{i}", "file_name": file_name, "content": chunk, "content_vector": vector}
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ])
    blob_storage.upload_file(file_name, data)
    return len(chunks)


def process_cvs(files: list[tuple[str, bytes]]) -> Iterator[dict]:
    """Process (file_name, data) pairs in parallel and yield each file's result as soon as it finishes."""
    blob_storage.ensure_container()
    search_index.ensure_index()
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = {pool.submit(process_cv, name, data): name for name, data in files}
        for future in as_completed(futures):
            try:
                result = {"file": futures[future], "chunks": future.result(), "error": None}
            except Exception as error:  # one bad file never stops the others
                result = {"file": futures[future], "chunks": 0, "error": str(error)}
            yield result


def list_cvs() -> list[str]:
    return blob_storage.list_files()
