"""Ingestion pipeline: CV file -> searchable chunks in Azure AI Search, original file in Blob Storage.

CVs are processed in parallel threads: the work is I/O-bound (HTTP calls to Azure).
"""
import hashlib
from functools import lru_cache
from pathlib import Path
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

from cv_chat import config
from cv_chat.processing import chunking, extract, sections
from cv_chat.processing.chunking import chunk_cv
from cv_chat.processing.extract import extract_cv
from cv_chat.processing.sections import section_type
from cv_chat.services import blob_storage, openai_service, search_index


@lru_cache(maxsize=1)
def _pipeline_fingerprint() -> bytes:
    """Changes whenever the extraction or chunking code or its settings change, so those CVs get re-indexed by themselves."""
    digest = hashlib.md5(f"{config.CHUNK_SIZE}/{config.CHUNK_OVERLAP}".encode())
    for module in (extract, chunking, sections):
        digest.update(Path(module.__file__).read_bytes())
    return digest.digest()


def file_id_for(file_name: str) -> str:
    """Stable id per file: processing the same file again overwrites its chunks instead of duplicating them."""
    return hashlib.md5(file_name.encode()).hexdigest()


def process_cv(file_name: str, data: bytes, force: bool = False) -> tuple[int, bool]:
    """Index one CV and store the original file. Returns (chunks, skipped); skipped means it was already indexed."""
    file_id = file_id_for(file_name)
    content_hash = hashlib.md5(data + _pipeline_fingerprint()).hexdigest()
    if not force and search_index.get_hash(file_id) == content_hash:
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


def delete_cv(file_name: str) -> None:
    """Remove a CV everywhere. The index goes first: a CV left in the index but not in storage would still be quoted."""
    search_index.delete_file_chunks(file_id_for(file_name))
    blob_storage.delete_file(file_name)


def reindex_cv(file_name: str) -> int:
    """Process a stored CV again from its original file (for example after the pipeline changed). Returns the chunks."""
    chunks, _ = process_cv(file_name, blob_storage.download_file(file_name), force=True)
    return chunks


def process_cvs(files: list[tuple[str, bytes]]) -> Iterator[dict]:
    """Process (file_name, data) pairs in parallel and yield each file's result as soon as it finishes."""
    blob_storage.ensure_container()
    search_index.ensure_index(openai_service.embedding_dimensions())
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
