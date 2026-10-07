"""Candidate details (name, job title, years of experience, contact) read from a CV's text. No model call, no Azure.

- name and location: spaCy named entity recognition on the top of the CV (PERSON and GPE entities), with a plain
  rule for a name written in capitals or when the spaCy model is not installed;
- email and phone: patterns;
- job title: the line under the name, or the first job listed, when it contains a job word (NER does not label titles);
- years of experience: the dated jobs in the Experience section, counted as the time covered by their date ranges
  (overlapping jobs are counted once, "Present" is today).

The values are stored on every chunk of the CV, so searches can boost, filter and show them. Anything that cannot be
read is left empty, and a failure here never stops a CV from being indexed.
"""
import logging
import re
import threading
from datetime import date
from functools import lru_cache

from cv_chat import config
from cv_chat.processing.chunking import Chunk
from cv_chat.processing.sections import section_type

log = logging.getLogger(__name__)
EMPTY = {"candidate_name": "", "job_title": "", "years_experience": None, "email": "", "phone": "", "location": ""}

HEADER_LINES = 12  # the top of the first page, where the name, title and contact line are
NAME_LINES = 6  # the name is within the first few lines
HEADER_CHARS = 800  # how much of the top is read by the NER model
SKIPPED_FOR_YEARS = {"education", "certifications", "languages", "contact"}  # dates in these are not jobs

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?<![\w/])(\+?\(?\d[\d\s().-]{6,}\d)(?![\w/])")
_YEAR_RANGE = re.compile(r"^\(?(?:19|20)\d{2}\)?\s*[-–—]\s*\(?(?:19|20)\d{2}\)?$")
_NOT_A_NAME = {
    "curriculum vitae", "resume", "résumé", "cv", "profile", "summary", "contact", "personal information",
    "personal details", "objective", "about me", "career objective", "professional summary",
}
_NAME = re.compile(r"^[^\W\d_][\w'’.-]*(?: [^\W\d_][\w'’.-]*){1,3}$")  # two to four words, no digits
_TITLE_WORDS = re.compile(
    r"\b(?:engineer|developer|programmer|architect|manager|director|analyst|scientist|consultant|designer|administrator|"
    r"specialist|lead|head|officer|coordinator|technician|researcher|accountant|auditor|teacher|lecturer|professor|"
    r"nurse|physician|doctor|intern|trainee|assistant|associate|executive|supervisor|planner|strategist|"
    r"marketer|recruiter|editor|writer|translator|paralegal|lawyer|attorney|founder|owner|president|vp|cto|ceo|cfo|coo|"
    r"devops|sre|tester|qa)\b",
    re.I,
)
_TITLE_SPLIT = re.compile(r"\s*(?:[,|•·@–—]|\s-\s|\bat\b)\s*", re.I)

_MONTH = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_DATE = rf"(?:{_MONTH}\.?,?\s+(?:19|20)\d{{2}}|(?:0?[1-9]|1[0-2])\s*[/.-]\s*(?:19|20)\d{{2}}|(?:19|20)\d{{2}})"
_RANGE = re.compile(rf"({_DATE})\s*(?:-|–|—|to|until)\s*({_DATE}|present|current|now|today|ongoing|till date|to date)", re.I)
_STATED = re.compile(r"(\d{1,2})\s*\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:professional\s+|work\s+|relevant\s+|industry\s+)?experience", re.I)
_MONTH_NUMBER = {name: number for number, name in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}

_lock = threading.Lock()  # one NER call at a time: the ingest threads share the model


@lru_cache(maxsize=1)
def _nlp():
    """The spaCy model, or None (with one warning) when spaCy or the model is not installed."""
    try:
        import spacy

        return spacy.load(config.NER_MODEL, disable=["parser", "lemmatizer"])
    except Exception as error:
        log.warning(
            "spaCy model %r is not available (%s): names are read by a plain rule and locations are left empty. "
            "Install the packages in requirements.txt.", config.NER_MODEL, str(error).splitlines()[0],
        )
        return None


def _entities(text: str) -> list[tuple[str, str, int, int]]:
    """(text, label, start, end) of the named entities in `text`."""
    nlp = _nlp()
    if nlp is None:
        return []
    with _lock:
        doc = nlp(text)
    return [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]


def _clean(value, limit: int) -> str:
    """One short line of plain text: no line breaks and no angle brackets, so a value cannot carry markup."""
    return " ".join(str(value or "").replace("<", " ").replace(">", " ").split())[:limit]


def _is_name(text: str) -> bool:
    text = text.strip()
    return (
        bool(_NAME.match(text)) and text.lower() not in _NOT_A_NAME and not _TITLE_WORDS.search(text) and "@" not in text
        and section_type(text) == "other"  # "Work Experience" and "Technical Skills" are headings, not people
    )


def _tidy_name(text: str) -> str:
    text = _clean(text, 100)
    return text.title() if text.isupper() else text


def _name(lines: list[str], entities: list[tuple[str, str, int, int]], offsets: list[int]) -> tuple[str, int]:
    """(name, index of the line it is on). NER first (a PERSON entity inside one of the first lines), then a plain rule."""
    for text, label, start, end in entities:
        if label != "PERSON":
            continue
        line_index = max(i for i, offset in enumerate(offsets) if offset <= start)
        line = lines[line_index]
        if line_index < NAME_LINES and _is_name(text) and len(text.split()) >= 2 and text in line:
            return _tidy_name(text), line_index
    for index, line in enumerate(lines[:NAME_LINES]):
        if _is_name(line):
            return _tidy_name(line), index
    return "", -1


def _phone(text: str) -> str:
    for match in _PHONE.finditer(text):
        candidate = " ".join(match.group(1).split())
        digits = re.sub(r"\D", "", candidate)
        if 8 <= len(digits) <= 15 and not _YEAR_RANGE.match(candidate) and not re.fullmatch(r"(?:19|20)\d{2}[-/.]\d{2}[-/.]\d{2}", candidate):
            return candidate[:40]
    return ""


def _location(text: str, entities: list[tuple[str, str, int, int]]) -> str:
    """The first place named at the top, with the place right after it when they are written as "City, Country"."""
    places = [(t, s, e) for t, label, s, e in entities if label == "GPE"]
    for index, (first, _, end) in enumerate(places):
        parts = [first]
        for nxt, nxt_start, nxt_end in places[index + 1 :]:
            if text[end:nxt_start].strip() == ",":
                parts.append(nxt)
                end = nxt_end
            else:
                break
        return _clean(", ".join(parts), 100)
    return ""


def _title_from(line: str) -> str:
    """The part of a line that is a job title, or "". Lines with dates, numbers or an address are not titles."""
    line = _RANGE.sub("", line).strip(" |•·-–—")
    if "@" in line or "http" in line.lower() or re.search(r"\d{3,}", line) or len(line) > 90:
        return ""
    for part in _TITLE_SPLIT.split(line):
        part = part.strip(" .:;")
        if part and len(part.split()) <= 7 and _TITLE_WORDS.search(part):
            return _clean(part, 100)
    return ""


def _job_title(lines: list[str], name_index: int, experience: list[str]) -> str:
    for line in (lines[name_index + 1 :] if name_index >= 0 else lines):
        if title := _title_from(line):
            return title
    for line in experience:  # the first job listed is usually the latest one
        if title := _title_from(line):
            return title
    return ""


def _month_index(text: str, today: date) -> tuple[int, bool] | None:
    """(months since year 0, whether the month was written) for one end of a date range."""
    text = text.strip().lower()
    if text in {"present", "current", "now", "today", "ongoing", "till date", "to date"}:
        return today.year * 12 + today.month, True
    year = int(re.search(r"(?:19|20)\d{2}", text).group())
    month_name = re.match(_MONTH, text)
    if month_name:
        return year * 12 + _MONTH_NUMBER[month_name.group()[:3]], True
    numeric = re.match(r"(0?[1-9]|1[0-2])\s*[/.-]", text)
    if numeric:
        return year * 12 + int(numeric.group(1)), True
    return year * 12 + 7, False  # only a year was written: assume the middle of it


def _years(text: str, today: date) -> float | None:
    """Time covered by the dated ranges in `text`, overlaps counted once. Falls back to a stated "N years of experience"."""
    spans = []
    for start_text, end_text in _RANGE.findall(text):
        start, end = _month_index(start_text, today), _month_index(end_text, today)
        if start is None or end is None or start[0] // 12 < 1950:
            continue
        low, high = start[0], end[0] + (1 if start[1] and end[1] else 0)  # Jan 2018 to Dec 2018 is twelve months
        if low < high <= today.year * 12 + today.month + 1 and high - low <= 50 * 12:
            spans.append((low, high))
    spans.sort()
    months, reach = 0, None
    for low, high in spans:
        if reach is None or low > reach:
            months, reach = months + (high - low), high
        elif high > reach:
            months, reach = months + (high - reach), high
    years = round(months / 12, 1) if months else None
    if years is None:
        stated = [int(n) for n in _STATED.findall(text)]
        years = float(max(stated)) if stated else None
    return years if years is not None and 0 < years <= config.MAX_YEARS else None


def extract(pages: list[tuple[int, str]], chunks: list[Chunk], today: date | None = None) -> dict:
    """The candidate's details from a CV's pages and chunks. Never raises."""
    try:
        return _extract(pages, chunks, today or date.today())
    except Exception as error:
        log.warning("could not read CV details: %s", str(error).splitlines()[0] if str(error) else type(error).__name__)
        return dict(EMPTY)


def _extract(pages: list[tuple[int, str]], chunks: list[Chunk], today: date) -> dict:
    first_page = pages[0][1] if pages else ""
    lines = [" ".join(line.split()) for line in first_page.splitlines() if line.strip()][:HEADER_LINES]
    header, offsets = "", []
    for line in lines:  # the top of the CV as one text, remembering where each line starts
        offsets.append(len(header))
        header += line + "\n"
    header = header[:HEADER_CHARS]
    entities = _entities(header)
    full_text = "\n".join(text for _, text in pages)

    name, name_index = _name(lines, entities, offsets)
    experience_chunks = [c for c in chunks if section_type(c.section) == "experience"]
    dated_chunks = experience_chunks or [c for c in chunks if section_type(c.section) not in SKIPPED_FOR_YEARS]
    experience_lines = [line for c in experience_chunks for line in c.text.splitlines()[1:]]  # [1:] skips the heading
    dated_text = "\n".join(c.text for c in dated_chunks) if chunks else full_text

    email = _EMAIL.search(header) or _EMAIL.search(full_text)
    return {
        "candidate_name": name,
        "job_title": _job_title(lines, name_index, experience_lines),
        "years_experience": _years(dated_text, today),
        "email": _clean(email.group(), 100) if email else "",
        "phone": _phone(header) or _phone(full_text),
        "location": _location(header, entities),
    }
