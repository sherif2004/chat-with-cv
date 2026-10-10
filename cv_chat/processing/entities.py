"""Candidate details read from a CV by a generic named-entity model (GLiNER). No model call to Azure, no patterns, no word
lists and no fallback: the model is asked for each detail by its plain-language label and the first answer in the CV (reading order) is kept.

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
    "location": "location",
}
FALLBACK_LABELS = {"location": ["city", "country"]}  # below the header the label "location" also finds universities
EMPTY = {"candidate_name": "", "job_title": "", "email": "", "phone": "", "location": ""}
WINDOW_WORDS = 150  # the model reads about 384 tokens at a time
PAGE_CHARS = 8000  # the most of one page that is read
THRESHOLD = 0.3  # the lowest confidence the model's answer may have

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

        return GLiNER.from_pretrained(config.NER_MODEL, map_location=_device(torch))
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


HEADER_BEFORE, HEADER_AFTER = 10, 1  # lines around the email line that make up the CV's header block
HEADER_FALLBACK = 8  # without an email: the first lines
NAME_LINE_WORDS = 4
TITLE_LINE_WORDS = 6  # a title found inside a line this short is taken as the whole line ("Junior AI Engineer", not "AI Engineer")


def _find(model, texts: list[str], label: str, threshold: float) -> list[dict]:
    """The model's answers for one label, from the first text that has any, in reading order."""
    for text in texts:
        with _lock:
            found = model.predict_entities(text, [label], threshold=threshold)
        if found:
            return sorted(found, key=lambda entity: entity["start"])
    return []


def _best(model, texts: list[str], label: str, threshold: float) -> dict | None:
    """The model's most confident answer for a label across all the texts."""
    answers = []
    for text in texts:
        with _lock:
            answers += model.predict_entities(text, [label], threshold=threshold)
    return max(answers, key=lambda entity: entity["score"], default=None)


def _header(model, lines: list[str], windows: list[str]) -> str:
    """The lines around the email address, where a CV keeps name, title, place and contacts. The layout reader can place this block
    anywhere on the page (even after the experience), so it is found by the email and not by its position."""
    email = _best(model, windows, LABELS["email"], THRESHOLD / 3)
    at = next((n for n, line in enumerate(lines) if email and email["text"] in line), None)
    if at is None:
        return "\n".join(lines[:HEADER_FALLBACK])
    return "\n".join(lines[max(at - HEADER_BEFORE, 0): at + HEADER_AFTER + 1])


def extract(pages: list[tuple[int, str]]) -> dict:
    """The candidate's name, job title, email, phone and location: the first answer in the CV's header block."""
    model = _model()
    page = pages[0][1][:PAGE_CHARS] if pages else ""
    lines = [line for line in page.split("\n") if line.strip()]
    header = _header(model, lines, _windows(page))
    header_lines = header.split("\n")

    def line_of(text: str) -> str:
        return next((l.strip() for l in header_lines if text in l), "")

    def widen(text: str) -> str:  # the model sometimes cuts a word short: keep the whole word it was cut from
        return next((w for line in header_lines for w in line.replace("|", " ").replace("·", " ").replace("•", " ").split() if text in w), text)

    first: dict[str, str] = {}
    email = _best(model, [header], LABELS["email"], THRESHOLD / 3)  # the email label also fires on phone numbers: the surest answer wins
    if email:
        first["email"] = widen(email["text"])

    name = _find(model, [header], LABELS["candidate_name"], THRESHOLD)
    name_text = name[0]["text"] if name else ""
    after_name = name[0]["start"] if name else 0  # a title comes after the name: a job line above it belongs to the experience
    title = next((e for e in _find(model, [header], LABELS["job_title"], THRESHOLD) if e["text"] not in name_text and e["start"] >= after_name), None)
    if title:
        line = line_of(title["text"])
        rest = line.replace(name_text, "").strip(" |-,") if name_text and name_text in line else line  # name and title can share a line
        first["job_title"] = rest if rest and len(rest.split()) <= TITLE_LINE_WORDS else title["text"]
    if name_text:
        line = line_of(name_text)
        alone = not (title and title["text"] in line)  # "Rawan Tarek Taha AI Engineer": the title is not part of the name
        first["candidate_name"] = line if alone and len(line.split()) <= NAME_LINE_WORDS else name_text

    for key in ("location",):
        found = _find(model, [header], LABELS[key], THRESHOLD)
        if found:
            first[key] = found[0]["text"]
    phones = [e for e in _find(model, [header], LABELS["phone"], THRESHOLD) if "@" not in e["text"] and e["text"] not in first.get("email", "")]
    if phones:
        first["phone"] = phones[0]["text"]

    missing = [key for key in ("candidate_name", "job_title", "location", "phone") if key not in first]
    if missing:  # not in the header: the next sections, in reading order, and the first answer found there
        in_header = set(header_lines)
        rest = [line for _, text in pages for line in text.split("\n") if line.strip() and line not in in_header]
        windows = _windows("\n".join(rest))
        for key in missing:
            labels = FALLBACK_LABELS.get(key, [LABELS[key]])
            for window in windows:
                found = [entity for label in labels for entity in _find(model, [window], label, THRESHOLD)]
                if found:
                    first[key] = min(found, key=lambda entity: entity["start"])["text"]
                    break
    return {**EMPTY, **{key: _clean(text) for key, text in first.items()}}
