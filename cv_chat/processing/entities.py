"""Candidate details read from a CV by a generic named-entity model (GLiNER). No model call to Azure, no patterns, no word
lists: the model is asked for each detail by its plain-language label.

The CV is read in order, the way a person would: the first block (the header) first, then the next section, and so on. The
first answer found for a detail is kept, and reading stops as soon as every detail is filled or the CV ends.

Years of experience is not read at all: it is not written in the text of most CVs, it has to be worked out from which dates
are jobs, and a generic NER cannot tell those from a degree's dates.

If the model cannot be loaded, reading a CV fails with a clear message instead of quietly producing something worse.
"""
import logging
import threading
from functools import lru_cache

from cv_chat.config import settings
from cv_chat.processing.chunking import Chunk
from cv_chat.processing.sections import section_type

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


def _find(model, text: str, label: str, threshold: float) -> list[dict]:
    """The model's answers for one label in one text, in reading order."""
    with _lock:
        return sorted(model.predict_entities(text, [label], threshold=threshold), key=lambda entity: entity["start"])


def _best(model, text: str, label: str, threshold: float) -> dict | None:
    """The model's most confident answer for a label in one text."""
    with _lock:
        answers = model.predict_entities(text, [label], threshold=threshold)
    return max(answers, key=lambda entity: entity["score"], default=None)


def _read_window(model, window: str, wanted: list[str], header: bool) -> dict[str, str]:
    """The first answer for each wanted detail in one piece of text. In the header the location label is "location"; further
    down that label also finds universities, so there the narrower labels (city, country) are used."""
    cfg = settings.entities
    lines = [line.strip() for line in window.split("\n") if line.strip()]

    def line_of(text: str) -> str:
        return next((line for line in lines if text in line), "")

    def widen(text: str) -> str:  # the model sometimes cuts a word short: keep the whole word it was cut from
        return next((w for line in lines for w in line.replace("|", " ").replace("·", " ").replace("•", " ").split() if text in w), text)

    first: dict[str, str] = {}
    if "email" in wanted:  # the email label also fires on phone numbers: the surest answer wins
        email = _best(model, window, LABELS["email"], cfg.email_threshold)
        if email:
            first["email"] = widen(email["text"])

    name = _find(model, window, LABELS["candidate_name"], cfg.threshold) if {"candidate_name", "job_title"} & set(wanted) else []
    name_text = name[0]["text"] if name else ""
    after_name = name[0]["start"] if name else 0  # a title comes after the name: a job line above it belongs to the experience
    title = None
    if "job_title" in wanted:
        title = next((e for e in _find(model, window, LABELS["job_title"], cfg.threshold) if e["text"] not in name_text and e["start"] >= after_name), None)
        if title:
            line = line_of(title["text"])
            rest = line.replace(name_text, "").strip(" |-,") if name_text and name_text in line else line  # name and title can share a line
            first["job_title"] = rest if rest and len(rest.split()) <= cfg.title_line_words else title["text"]
    if "candidate_name" in wanted and name_text:
        line = line_of(name_text)
        alone = not (title and title["text"] in line)  # "Rawan Tarek Taha AI Engineer": the title is not part of the name
        first["candidate_name"] = line if alone and len(line.split()) <= cfg.name_line_words else name_text

    if "phone" in wanted:
        phones = [e for e in _find(model, window, LABELS["phone"], cfg.threshold) if "@" not in e["text"] and e["text"] not in first.get("email", "")]
        if phones:
            first["phone"] = phones[0]["text"]
    if "location" in wanted:
        labels = [LABELS["location"]] if header else cfg.location_fallback_labels
        places = [entity for label in labels for entity in _find(model, window, label, cfg.threshold)]
        if places:
            first["location"] = min(places, key=lambda entity: entity["start"])["text"]
    return first


def extract(chunks: list[Chunk], model=None) -> dict:
    """The candidate's name, job title, email, phone and location: the first answer in the CV, reading the sections in order.
    A location is taken only from the header block and the sections named in settings.entities.location_sections."""
    model = model or _model()
    found: dict[str, str] = {}
    for position, chunk in enumerate(chunks):
        header = position == 0
        may_have_place = header or section_type(chunk.section) in settings.entities.location_sections
        for window in _windows(chunk.text):
            wanted = [key for key in LABELS if key not in found and (key != "location" or may_have_place)]
            if not wanted:
                break
            for key, value in _read_window(model, window, wanted, header).items():
                found.setdefault(key, value)
        if all(key in found for key in LABELS):
            break
    return {**EMPTY, **{key: _clean(text) for key, text in found.items()}}
