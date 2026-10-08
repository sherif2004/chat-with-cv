"""Candidate details read from a CV by a generic named-entity model (GLiNER). No model call to Azure, no patterns, no word
lists and no fallback: the model is asked for each detail by its plain-language label and the best-scoring answer is kept.

Years of experience is not read at all: it is not written in the text of most CVs, it has to be worked out from which dates
are jobs, and a generic NER cannot tell those from a degree's dates.

If the model cannot be loaded, reading a CV fails with a clear message instead of quietly producing something worse.
"""
import logging
import threading
from functools import lru_cache

from cv_chat import config

log = logging.getLogger(__name__)

LABELS = {  # the label the model is asked for, one at a time (asking for all together found much less)
    "candidate_name": "person name",
    "job_title": "job title",
    "email": "email address",
    "phone": "phone number",
    "location": "city, country",
}
EMPTY = {"candidate_name": "", "job_title": "", "email": "", "phone": "", "location": ""}
PAGES = 1  # the details are on the first page: reading the second only added wrong answers (measured) and doubled the time
WINDOW_WORDS = 150  # the model reads about 384 tokens at a time
PAGE_CHARS = 8000  # the most of one page that is read
THRESHOLD = 0.3  # the lowest confidence the model's answer may have

_lock = threading.Lock()  # one reading at a time: the ingest threads share the model


@lru_cache(maxsize=1)
def _model():
    """The GLiNER model, downloaded on first use. A failure is raised, not hidden."""
    try:
        from gliner import GLiNER

        import torch

        return GLiNER.from_pretrained(config.NER_MODEL, map_location="cuda" if torch.cuda.is_available() else "cpu")  # a GPU, if there is one
    except Exception as error:
        raise RuntimeError(
            f"The NER model '{config.NER_MODEL}' could not be loaded ({str(error).splitlines()[0] if str(error) else type(error).__name__}). "
            "Install the packages in requirements.txt and check the internet connection: the model is downloaded the first time."
        ) from error


def warm_up() -> None:
    """Load the model now, so the first CV does not wait for it."""
    _model()


def _windows(text: str) -> list[str]:
    """The text cut into pieces of about WINDOW_WORDS words, at line breaks."""
    pieces, current, words = [], [], 0
    for line in text.split("\n"):
        size = len(line.split())
        if current and words + size > WINDOW_WORDS:
            pieces.append("\n".join(current))
            current, words = [], 0
        current.append(line)
        words += size
    if current:
        pieces.append("\n".join(current))
    return pieces


def _clean(value: str, limit: int = 100) -> str:
    """One short line of plain text: no line breaks and no angle brackets, so a value cannot carry markup."""
    return " ".join(str(value or "").replace("<", " ").replace(">", " ").split())[:limit]


def extract(pages: list[tuple[int, str]]) -> dict:
    """The candidate's name, job title, email, phone and location from the first pages of a CV."""
    model = _model()
    best: dict[str, tuple[str, float]] = {}
    windows = [window for _, text in pages[:PAGES] for window in _windows(text[:PAGE_CHARS])]
    for key, label in LABELS.items():
        for window in windows:
            with _lock:
                found = model.predict_entities(window, [label], threshold=THRESHOLD)
            for entity in found:
                if key not in best or entity["score"] > best[key][1]:
                    best[key] = (entity["text"], entity["score"])
    return {**EMPTY, **{key: _clean(text) for key, (text, _) in best.items()}}
