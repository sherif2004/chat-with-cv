"""Background ingestion queue: CVs are processed in parallel while the app stays usable, and every file's state can be read.

The state of each file is queued, processing (with its current stage), indexed, skipped or failed. The sidebar polls it.
"""
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from cv_chat import config
from cv_chat.rag import ingest
from cv_chat.services import blob_storage

log = logging.getLogger(__name__)
FINISHED = ("indexed", "skipped", "failed")


@dataclass(frozen=True)
class Job:
    file_name: str
    state: str  # queued | processing | indexed | skipped | failed
    stage: str = ""  # while processing: what this file is doing right now
    chunks: int = 0
    error: str = ""


class IngestQueue:
    """Runs one job per CV on a bounded pool; a failing CV never affects the others."""

    def __init__(self, workers: int):
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ingest")
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}

    def _set(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.file_name] = job

    def submit(self, files: list[tuple[str, bytes | None]], force: bool = False) -> None:
        """files: (file name, bytes). bytes=None means use the stored original (re-index). May raise if Azure setup fails."""
        blob_storage.ensure_container()
        ingest.prepare()
        if not self.active():  # a file in progress is in the index before it reaches Blob Storage, so never sweep while one runs
            ingest.remove_deleted_cvs()
        with self._lock:
            self._jobs = {name: job for name, job in self._jobs.items() if job.state not in FINISHED}
        for name, data in files:
            self._set(Job(name, "queued"))
            self._pool.submit(self._run, name, data, force)

    def _run(self, name: str, data: bytes | None, force: bool) -> None:
        self._set(Job(name, "processing", stage="starting"))
        try:
            if data is None:
                data = blob_storage.download_file(name)
            chunks, skipped = ingest.process_cv(
                name, data, force, on_stage=lambda stage: self._set(Job(name, "processing", stage=stage))
            )
            self._set(Job(name, "skipped" if skipped else "indexed", chunks=chunks))
        except Exception as error:  # one bad file never stops the others
            log.warning("could not process %s: %s", name, error, exc_info=not isinstance(error, ValueError))
            self._set(Job(name, "failed", error=str(error)))

    def forget(self, name: str) -> None:
        with self._lock:
            self._jobs.pop(name, None)

    def clear_finished(self) -> None:
        with self._lock:
            self._jobs = {name: job for name, job in self._jobs.items() if job.state not in FINISHED}

    def snapshot(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def active(self) -> bool:
        return any(job.state not in FINISHED for job in self.snapshot())


queue = IngestQueue(config.MAX_WORKERS)
