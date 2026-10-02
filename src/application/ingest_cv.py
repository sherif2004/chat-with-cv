"""Ingest CVs: Blob -> extract -> chunk -> embed -> index, in parallel across CVs."""
import hashlib
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from src.application.chunking import chunk_cv
from src.domain.models import Chunk
from src.domain.ports import BlobStore, Embedder, PdfExtractor, VectorIndex


log = logging.getLogger(__name__)


class IngestError(Exception):
    pass


@dataclass(frozen=True)
class Deps:
    blob: BlobStore
    extractor: PdfExtractor
    embedder: Embedder
    index: VectorIndex


@dataclass(frozen=True)
class IngestResult:
    chunks: int
    skipped: bool = False


def cv_id_for(filename: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", Path(filename).stem).strip("_").lower()[:40]
    return f"{slug}_{hashlib.sha1(filename.encode()).hexdigest()[:6]}"


def ingest_cv(deps: Deps, filename: str, data: bytes, force: bool = False) -> IngestResult:
    cv_id = cv_id_for(filename)
    content_hash = hashlib.md5(data).hexdigest()
    if not force and deps.index.get_hash(cv_id) == content_hash:
        log.info("skip %s: unchanged", filename)
        return IngestResult(chunks=0, skipped=True)
    started = time.perf_counter()

    try:
        document = deps.extractor.extract(data, filename)
    except Exception as e:
        raise IngestError(f"cannot read PDF: {e}") from e
    pages = document.pages
    if not any(text.strip() for _, text in pages):
        raise IngestError("no text found (scanned PDF? needs OCR)")

    raw = chunk_cv(pages, document.headers)
    contents = [c.text for c in raw]
    vectors = deps.embedder.embed(contents)
    chunks = [
        Chunk(f"{cv_id}_{i}", cv_id, filename, c.section, c.page, content, content_hash, vec)
        for i, (c, content, vec) in enumerate(zip(raw, contents, vectors))
    ]

    # Upload first (same ids overwrite), then drop leftovers, so the CV is never missing from the index.
    deps.index.upload(chunks)
    deps.index.delete_cv(cv_id, keep_ids={c.id for c in chunks})
    log.info("indexed %s: %d chunks, %.1fs", filename, len(chunks), time.perf_counter() - started)
    return IngestResult(chunks=len(chunks))


@dataclass(frozen=True)
class JobStatus:
    filename: str
    state: str  # queued | processing | indexed | skipped | failed
    chunks: int = 0
    error: str = ""


class IngestionService:
    """Runs one ingest job per CV on a bounded pool; a failing CV never affects the others."""

    def __init__(self, deps: Deps, workers: int = 4):
        self._deps = deps
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ingest")
        self._lock = threading.Lock()
        self._jobs: dict[str, JobStatus] = {}
        self._version = 0  # bumped whenever indexed data may have changed

    def _set(self, status: JobStatus) -> None:
        with self._lock:
            self._jobs[status.filename] = status
            if status.state in ("indexed", "skipped", "failed"):
                self._version += 1

    def submit(self, files: list[tuple[str, bytes | None]], force: bool = False) -> None:
        """files: (filename, bytes); bytes=None means download the existing blob (re-index)."""
        for filename, data in files:
            self._set(JobStatus(filename, "queued"))
            self._pool.submit(self._run, filename, data, force)

    def _run(self, filename: str, data: bytes | None, force: bool) -> None:
        self._set(JobStatus(filename, "processing"))
        try:
            if data is None:
                data = self._deps.blob.download(filename)
            else:
                self._deps.blob.upload(filename, data)  # store first, so the file is kept even if indexing fails
            result = ingest_cv(self._deps, filename, data, force)
            self._set(JobStatus(filename, "skipped" if result.skipped else "indexed", result.chunks))
        except Exception as e:
            log.warning("ingest failed for %s: %s", filename, e, exc_info=not isinstance(e, IngestError))
            self._set(JobStatus(filename, "failed", error=str(e)))

    def reindex(self, filename: str) -> None:
        self.submit([(filename, None)], force=True)

    def delete(self, filename: str) -> None:
        self._deps.blob.delete(filename)
        self._deps.index.delete_cv(cv_id_for(filename))
        with self._lock:
            self._jobs.pop(filename, None)
            self._version += 1
        log.info("deleted %s", filename)

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def snapshot(self) -> list[JobStatus]:
        with self._lock:
            return [replace(j) for j in self._jobs.values()]

    def active(self) -> bool:
        with self._lock:
            return any(j.state in ("queued", "processing") for j in self._jobs.values())
