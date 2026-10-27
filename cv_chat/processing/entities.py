"""Read candidate details locally, using contact context to disambiguate GLiNER entities.

Email and phone have format checks; names and cities are selected near contact details,
not from publication authors or employment history. Missing details remain empty.
"""
import logging
import re
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
    lines = []
    for line in text.split("\n"):
        words_in_line = line.split()
        limit = settings.entities.window_words
        lines.extend(" ".join(words_in_line[i:i + limit]) for i in range(0, len(words_in_line), limit))
    for line in lines:
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


def extract(chunks: list[Chunk], model=None) -> dict:
    """Use unchunked page text when available, preserving the candidate's contact context."""
    model = model or _model()
    text = "\n".join(chunk.text for chunk in chunks)
    lines = text.splitlines()
    found = dict(EMPTY)
    email = re.search(r"[\w.+%-]+@[\w.-]+\.[A-Za-z]{2,}", text)
    # Phone numbers must contain 8-15 digits and cannot be a DOI, date or decimal.
    phones = [m for m in re.finditer(r"(?<![\w./])\+?\d[\d ()-]{6,}\d(?![\w./])", text)
              if 8 <= sum(c.isdigit() for c in m.group()) <= 15
              and not re.search(r"\d{4}\s*-\s*\d{4}", m.group())]
    anchor = email or (phones[0] if phones else None)
    if anchor:
        line_no = text[:anchor.start()].count("\n")
        start, end = line_no, line_no + 1
        while start > max(0, line_no - 12) and len(lines[start - 1].split()) <= 18:
            start -= 1
        while end < min(len(lines), line_no + 4) and len(lines[end].split()) <= 18:
            end += 1
        contact = "\n".join(lines[start:end])
    else:
        # With no contact anchor, only the opening block can identify the candidate.
        contact = "\n".join(lines[:4])
    if email:
        found["email"] = email.group()
    contact_phone = next((m.group() for m in phones if m.group() in contact), "")
    found["phone"] = contact_phone
    name_line = ""
    for key in ("candidate_name", "location", "job_title"):
        answers = []
        offset = 0
        for window in _windows(contact):
            with _lock:
                predictions = model.predict_entities(window, [LABELS[key]], threshold=settings.entities.threshold)
            answers.extend({**a, "start": a.get("start", 0) + offset} for a in predictions)
            offset += len(window) + 1
        if answers:
            answer = (min(answers, key=lambda a: a.get("start", 0)) if key == "job_title"
                      else max(answers, key=lambda a: a["score"]))
            found[key] = answer["text"]
            if key == "candidate_name":
                contact_lines = contact.splitlines()
                index = next((i for i, line in enumerate(contact_lines) if answer["text"] in line), None)
                if index is not None:
                    name_line = contact_lines[index].strip()
                    # Discard employment/publication text preceding the identity line.
                    contact = "\n".join(contact_lines[index:])
    if (name_line and 2 <= len(name_line.split()) <= 6
            and all(c.isalpha() or c in " '-" for c in name_line)
            and (not found["job_title"] or found["job_title"] not in name_line)):
        found["candidate_name"] = name_line
    if found["job_title"]:
        # Keep qualifiers such as Junior when the model returns only part of a short title line.
        for line in contact.splitlines():
            if (found["job_title"] in line and found["candidate_name"] not in line
                    and len(line.split()) <= 8 and all(c.isalpha() or c in " &/-" for c in line)):
                found["job_title"] = line.strip()
                break
    if not found["job_title"]:
        for window in _windows(text):
            with _lock:
                answers = model.predict_entities(window, [LABELS["job_title"]], threshold=settings.entities.threshold)
            if answers:
                found["job_title"] = min(answers, key=lambda a: a.get("start", 0))["text"]
                break
    return {key: _clean(value) for key, value in found.items()}
