"""Candidate details read from a CV by a generic named-entity model (GLiNER). No model call to Azure, no patterns, no word
lists: the model is asked for each detail by its plain-language label.

The CV is read in order, the way a person would: the first block (the header) first, then the next section, and so on. The
first answer found for a detail is kept and never replaced, and reading stops as soon as every detail is filled or the CV
ends. There are no rules about where a detail may be or what it looks like: only the labels and the threshold in the settings.

Years of experience is not read at all: it is not written in the text of most CVs, it has to be worked out from which dates
are jobs, and a generic NER cannot tell those from a degree's dates.

If the model cannot be loaded, reading a CV fails with a clear message instead of quietly producing something worse.
"""
import logging
import threading
from functools import lru_cache

from cv_chat.config import settings
from cv_chat.processing.chunking import Chunk

log = logging.getLogger(__name__)

LABELS = settings.entities.labels  # detail -> the label the model is asked for, one at a time (all together found much less)
EMPTY = {key: "" for key in LABELS}

_lock = threading.Lock()  # one reading at a time: the ingest threads share the model


def _device(torch) -> str:
    """A GPU only if this PyTorch build has kernels for it; an older card (e.g. sm_61) is present but cannot run them, so the CPU is used."""
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if f"sm_{major}{minor}" in torch.cuda.get_arch_list():
            return "cuda"
    return "cpu"


@lru_cache(maxsize=1)
def _model():
    """The GLiNER model, downloaded on first use. A failure is raised, not hidden."""
    try:
        from gliner import GLiNER

        import torch

        return GLiNER.from_pretrained(settings.entities.model, map_location=_device(torch))
    except Exception as error:
        raise RuntimeError(
            f"The NER model '{settings.entities.model}' could not be loaded ({str(error).splitlines()[0] if str(error) else type(error).__name__}). "
            "Install the packages in requirements.txt and check the internet connection: the model is downloaded the first time."
        ) from error


def warm_up() -> None:
    """Load the model now, so the first CV does not wait for it."""
    _model()


def _windows(text: str) -> list[str]:
    """The text cut into pieces of about window_words words (what the model reads at a time), at line breaks."""
    pieces, current, words = [], [], 0
    for line in text.split("\n"):
        size = len(line.split())
        if current and words + size > settings.entities.window_words:
            pieces.append("\n".join(current))
            current, words = [], 0
        current.append(line)
        words += size
    if current:
        pieces.append("\n".join(current))
    return pieces


def _clean(value: str) -> str:
    """One short line of plain text: no line breaks and no angle brackets, so a value cannot carry markup."""
    return " ".join(str(value or "").replace("<", " ").replace(">", " ").split())[: settings.entities.max_value_chars]


def _best(model, text: str, label: str) -> dict | None:
    """The model's most confident answer for a label in one text, or None if it has none above the threshold."""
    with _lock:
        answers = model.predict_entities(text, [label], threshold=settings.entities.threshold)
    return max(answers, key=lambda entity: entity["score"], default=None)


def extract(chunks: list[Chunk], model=None) -> dict:
    """The candidate's name, job title, email, phone and location, with no rules of its own: the CV is read from the start to
    the end, one piece after another, and each detail is asked of the model until it has an answer. The first piece that has
    one decides (its most confident answer), and a detail that is found is never replaced by a later one. Reading stops as soon
    as every detail is filled; if one never is, the whole CV is read."""
    model = model or _model()
    found: dict[str, str] = {}
    for chunk in chunks:
        for window in _windows(chunk.text):
            for key in [key for key in LABELS if key not in found]:
                answer = _best(model, window, LABELS[key])
                if answer:
                    found[key] = answer["text"]
            if len(found) == len(LABELS):
                return {**EMPTY, **{key: _clean(text) for key, text in found.items()}}
    return {**EMPTY, **{key: _clean(text) for key, text in found.items()}}
