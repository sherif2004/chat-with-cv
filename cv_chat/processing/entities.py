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
FULL_TEXT_CHARS = 50_000  # the patterns never scan more than this much of a CV, however long it is
PAGE_NER_CHARS = 6000  # how much of a page is read by the NER model when the name is not at the top
SKIPPED_FOR_YEARS = {"education", "certifications", "languages", "contact"}  # dates in these are not jobs

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?<![\w/])(\+?\(?\d[\d\s().-]{6,22}\d)(?![\w/])")
_YEAR_RANGE = re.compile(r"^\(?(?:19|20)\d{2}\)?\s*[-–—]\s*\(?(?:19|20)\d{2}\)?$")
_NOT_A_NAME = {
    "curriculum vitae", "resume", "résumé", "cv", "profile", "summary", "contact", "personal information",
    "personal details", "objective", "about me", "career objective", "professional summary",
}
_PARTICLES = {"de", "del", "della", "di", "da", "dos", "du", "van", "von", "der", "den", "ter", "ten", "bin", "bint", "ibn", "abu", "al", "el", "la", "le", "ben", "mac", "st"}
_NAME_WORD = re.compile(r"[^\W\d_][^\W\d_]*(?:['’-][^\W\d_]+)*")  # letters, with an inner apostrophe or hyphen: O'Brien, Anne-Marie
_NOT_NAME_WORDS = {  # words that make a line something other than a person's name
    "computer", "science", "sciences", "engineering", "technology", "technologies", "university", "college", "school", "institute",
    "academy", "faculty", "bachelor", "bachelors", "master", "masters", "degree", "diploma", "certificate", "certification",
    "certifications", "skills", "skill", "experience", "education", "summary", "profile", "objective", "projects", "project",
    "languages", "language", "courses", "course", "training", "internship", "management", "development", "software", "hardware",
    "data", "systems", "system", "network", "networks", "web", "mobile", "design", "analysis", "analytics", "quality", "assurance",
    "testing", "business", "administration", "information", "security", "cloud", "artificial", "intelligence", "machine",
    "learning", "english", "arabic", "french", "german", "spanish", "native", "fluent", "intermediate", "beginner", "microsoft",
    "google", "amazon", "linkedin", "github", "email", "phone", "address", "reference", "references", "available", "request",
    "personal", "details", "information", "contact", "professional", "technical", "work", "history", "achievements", "awards",
    "interests", "hobbies", "volunteer", "activities", "publications", "gpa", "grade", "graduate", "graduated", "present",
    "company", "limited", "ltd", "inc", "corp", "department", "digital", "global", "solutions", "services", "group",
}
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

_NOT_A_PERSON_LOCAL = {
    "info", "contact", "hr", "admin", "mail", "email", "cv", "resume", "jobs", "job", "career", "careers", "hello", "office",
    "support", "sales", "team", "me", "my", "the", "recruitment", "apply", "enquiries", "noreply", "no", "reply",
}
_NOT_A_PERSON_FILE = re.compile(
    r"\b(?:cv|resume|résumé|curriculum|vitae|copy|final|new|updated|latest|profile|candidate|application|draft|scan|scanned|doc|document)\b", re.I
)

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


def warm_up() -> None:
    """Load the spaCy model now, so the first CV does not wait for it."""
    _nlp()


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


def _looks_like_name(text: str) -> bool:
    """Two to four words, each a capitalised word, an initial ("A." or "A") or a name particle ("de", "bin"): nothing else.
    "B.Sc. in Computer Engineering" and "Machine Learning" are not names, "Ada de Lovelace" and "J. R. Smith" are."""
    words = text.split()
    if not 2 <= len(words) <= 4:
        return False
    real = 0
    for word in words:
        if re.fullmatch(r"[^\W\d_]\.?", word) or word.lower() in _PARTICLES:
            continue
        if not (_NAME_WORD.fullmatch(word) and word[0].isupper()):
            return False
        real += 1
    return real >= 1 and not {word.lower().strip(".") for word in words} & _NOT_NAME_WORDS


def _is_name(text: str) -> bool:
    text = text.strip()
    return (
        _looks_like_name(text) and text.lower() not in _NOT_A_NAME and not _TITLE_WORDS.search(text) and "@" not in text
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


def _name_from_email(email: str) -> tuple[str, str]:
    """(a name read from the part before the @, that part reduced to letters). "ada.lovelace2024@x.com" gives
    ("Ada Lovelace", "adalovelace"). The name is "" unless the address is made of two or three plain words."""
    local = email.split("@")[0].lower()
    letters = re.sub(r"[^a-z]", "", local)
    parts = [part for part in re.split(r"[^a-z]+", local) if part]
    if 2 <= len(parts) <= 3 and all(len(part) >= 2 for part in parts) and not set(parts) & _NOT_A_PERSON_LOCAL:
        return " ".join(part.capitalize() for part in parts), letters
    return "", letters


def _name_from_file(file_name: str) -> tuple[str, str]:
    """(a name read from the file name, the file name reduced to letters). "Ada_Lovelace_CV.pdf" gives ("Ada Lovelace", ...)."""
    stem = re.sub(r"\.[A-Za-z0-9]{2,5}$", "", file_name.strip())
    letters = re.sub(r"[^a-z]", "", stem.lower())
    spaced = re.sub(r"[\d_.()+\-]+", " ", re.sub(r"(?<=[a-z])(?=[A-Z])", " ", stem))  # "Ada_Lovelace-CV2024" -> "Ada Lovelace CV"
    words = _NOT_A_PERSON_FILE.sub(" ", spaced).split()  # then leave out words such as CV and resume
    if 2 <= len(words) <= 4 and all(re.fullmatch(r"[^\W\d_][\w'’]+", word) for word in words):
        return " ".join(word.capitalize() for word in words), letters
    return "", letters


def _agrees(name: str, *evidence: str) -> bool:
    """True if the name matches the email address or file name it is checked against: one word of four letters or more
    appears in it, or two shorter ones do. This is what keeps a name-shaped line such as "Machine Learning" from being taken
    for a person."""
    words = [word for word in re.findall(r"[a-z]+", name.lower()) if len(word) >= 3]
    hits = [word for word in words if any(word in text for text in evidence if text)]
    return any(len(word) >= 4 for word in hits) or len(hits) >= 2


def _name_elsewhere(pages: list[tuple[int, str]], email_name: str, email_letters: str, file_name_name: str, file_letters: str) -> str:
    """The name when it is not in the first lines (a CV that opens with a long summary, or whose layout put the name later).
    Candidates are people spaCy finds on the first two pages and lines that are just a name. One is taken only when the email
    address or the file name backs it up. Failing that, a name read from the email address or the file name is used, and
    last of all a line that is just a name and that spaCy also calls a person."""
    people: list[str] = []
    name_lines: list[str] = []
    for _, text in pages[:2]:
        text = text[:PAGE_NER_CHARS]
        for entity, label, _, _ in _entities(text):
            candidate = " ".join(entity.split())
            if label == "PERSON" and 2 <= len(candidate.split()) <= 4 and _is_name(candidate):
                people.append(candidate)
        name_lines += [line for line in (" ".join(raw.split()) for raw in text.splitlines()) if _is_name(line)]
    candidates = list(dict.fromkeys(people + name_lines))
    for candidate in candidates:
        if _agrees(candidate, email_letters, file_letters):
            return _tidy_name(candidate)
    if email_name:
        return email_name
    if file_name_name:
        return file_name_name
    for line in name_lines:
        if line in people:
            return _tidy_name(line)
    return ""


def _phone(text: str) -> str:
    for match in _PHONE.finditer(text):
        candidate = " ".join(match.group(1).split())
        digits = re.sub(r"\D", "", candidate)
        if 8 <= len(digits) <= 15 and not _YEAR_RANGE.match(candidate) and not re.fullmatch(r"(?:19|20)\d{2}[-/.]\d{2}[-/.]\d{2}", candidate):
            return candidate[:40]
    return ""


_CONTACT_SIGNAL = re.compile(r"@|https?://|www\.|linkedin|github", re.I)
_SEGMENT_SPLIT = re.compile(r"\s*[|·•●▪◦]\s*|\s+[–—-]\s+|\s{3,}")
_PLACE_LABEL = re.compile(r"(?i)^\s*(?:address|location|based in|city|country)\s*[:\-]\s*(.+)$")
_PLACE_PART = re.compile(r"[A-Z][A-Za-z.'’-]*(?: [A-Z][A-Za-z.'’-]*){0,2}")
_COUNTRY_CODES = {"US", "USA", "UK", "UAE", "KSA", "EU", "UN"}  # short all-capital places that are real; others (AI, ML, IT...) are not


def _is_real_place(text: str) -> bool:
    """False for 2 or 3 capital letters that are not a known country code: the small spaCy model tags "AI" or "QA" as places."""
    return not (text.isupper() and len(text) <= 3 and text not in _COUNTRY_CODES)


def _place_from_segment(segment: str) -> str:
    """A written place such as "Cairo, Egypt" or "Berlin", from one piece of a contact line. "" if it is something else."""
    segment = segment.strip(" ,;:")
    if not segment or re.search(r"[\d@/\\]|\.(?:com|net|org|io|dev)\b", segment):
        return ""
    parts = [part.strip() for part in segment.split(",")]
    if not 1 <= len(parts) <= 4 or not all(_PLACE_PART.fullmatch(part) for part in parts):
        return ""
    if _nlp() is not None:  # a skill list such as "Python, SQL" looks the same: at least one part must be a place
        if not any(label in ("GPE", "LOC") for _, label, _, _ in _entities(segment)):
            return ""
    elif len(parts) < 2:
        return ""  # without the model only a pair like "Cairo, Egypt" is trusted
    return _clean(", ".join(parts), 100)


def _location_from_contact(lines: list[str]) -> str:
    """The place written on the contact line (the line with the email, phone or links), wherever it is on the page: in many
    CVs it sits at the bottom, far from the top. Also understands "Address: ..." and "Location: ..." lines."""
    for line in lines:
        if label := _PLACE_LABEL.match(line):
            if place := _place_from_segment(label.group(1)):
                return place
        if _CONTACT_SIGNAL.search(line) or _phone(line):
            for segment in _SEGMENT_SPLIT.split(line):
                if place := _place_from_segment(segment):
                    return place
    return ""


def _location(text: str, entities: list[tuple[str, str, int, int]]) -> str:
    """The first place spaCy names at the top, with the place right after it when they are written as "City, Country"."""
    places = [(t, s, e) for t, label, s, e in entities if label == "GPE" and _is_real_place(t)]
    for index, (first, _, end) in enumerate(places):
        parts = [first]
        for nxt, nxt_start, nxt_end in places[index + 1 :]:
            if text[end:nxt_start].strip() == "," and _is_real_place(nxt):
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


_WORK_WORDS = re.compile(
    r"\b(?:intern|interns|internship|engineer|developer|programmer|analyst|instructor|trainer|tutor|teacher|assistant|agent|manager|"
    r"officer|consultant|freelance|freelancer|specialist|designer|administrator|technician|coordinator|supervisor|associate|"
    r"executive|employee|full[- ]time|part[- ]time|contract)\b", re.I,
)
_EDU_WORDS = re.compile(
    r"\b(?:university|college|faculty|school|bachelor|bachelors|b\.?sc\.?|m\.?sc\.?|degree|gpa|cgpa|graduat\w*|undergraduate|"
    r"postgraduate|diploma|thesis|major|student)\b", re.I,
)
_TRAIN_WORDS = re.compile(
    r"\b(?:bootcamp|boot camp|course|courses|training|program|programme|academy|initiative|scholarship|workshop|certification|"
    r"certificate|learning|camp|institute|nanodegree|level \d)\b", re.I,
)


def _evidence(context: str) -> tuple[int, int, int]:
    """How many different education, training and work words are in the text around a date range."""
    return tuple(len({m.lower() for m in rx.findall(context)}) for rx in (_EDU_WORDS, _TRAIN_WORDS, _WORK_WORDS))


def _kind(context: str) -> str:
    """What a date range is for, judged by the words around it: "edu" (a degree), "train" (a course or training program),
    "work" (a job or internship), or "" when nothing says. Only jobs and internships count as experience."""
    edu, train, work = _evidence(context)
    if edu and edu >= work:
        return "edu"
    if train > work:
        return "train"
    return "work" if work else ""


def _spans(ranges: list[tuple[str, str]], today: date) -> list[tuple[int, int]]:
    """Month spans (start, end) for date ranges written as text, leaving out any that make no sense."""
    spans = []
    for start_text, end_text in ranges:
        start, end = _month_index(start_text, today), _month_index(end_text, today)
        if start is None or end is None or start[0] // 12 < 1950:
            continue
        low, high = start[0], end[0] + (1 if start[1] and end[1] else 0)  # Jan 2018 to Dec 2018 is twelve months
        if low < high <= today.year * 12 + today.month + 1 and high - low <= 50 * 12:
            spans.append((low, high))
    return spans


def _counted_ranges(chunks: list[Chunk]) -> list[tuple[str, str]]:
    """The date ranges that are work. In an Experience section a range counts unless the lines around it say it is a degree
    or a course. Anywhere else it counts only if the lines around it name a job or an internship. Education, certification,
    language and contact sections are skipped, and so are projects and training programs."""
    counted = []
    for chunk in chunks:
        kind = section_type(chunk.section)
        if kind in SKIPPED_FOR_YEARS:
            continue
        in_experience = kind == "experience"
        lines = chunk.text.splitlines()
        for index, line in enumerate(lines):
            found = _RANGE.findall(line)
            if not found:
                continue
            around = "\n".join(lines[max(0, index - 1) : index + 1] if in_experience else lines[max(0, index - 2) : index + 3])
            if in_experience:
                if _kind(around) in ("work", ""):
                    counted += found
                continue
            # Outside an Experience section the words must clearly say job: more work words than training words, and a range
            # of more than three years (which is much more likely a degree) needs two different work words.
            edu, train, work = _evidence(around)
            for start, end in found:
                span = _spans([(start, end)], date.today())
                long_range = bool(span) and span[0][1] - span[0][0] > 36
                if not edu and work > train and (work >= 2 or not long_range):
                    counted.append((start, end))
    return counted


def _years(chunks: list[Chunk], full_text: str, today: date) -> float | None:
    """Years of work: the time covered by the job and internship date ranges, overlaps counted once. Falls back to a stated
    "N years of experience" when no dated job is found."""
    ranges = _counted_ranges(chunks) if chunks else _RANGE.findall(full_text)
    spans = sorted(_spans(ranges, today))
    months, reach = 0, None
    for low, high in spans:
        if reach is None or low > reach:
            months, reach = months + (high - low), high
        elif high > reach:
            months, reach = months + (high - reach), high
    years = int(months / 12 * 10 + 0.5) / 10 if months else None  # rounded half up: three months is 0.3 years, not 0.2
    if years is None:
        stated = [int(n) for n in _STATED.findall(full_text)]
        years = float(max(stated)) if stated else None
    return years if years is not None and 0 < years <= config.MAX_YEARS else None


def extract(pages: list[tuple[int, str]], chunks: list[Chunk], today: date | None = None, file_name: str = "") -> dict:
    """The candidate's details from a CV's pages and chunks (and its file name, which can confirm the name). Never raises."""
    try:
        return _extract(pages, chunks, today or date.today(), file_name)
    except Exception as error:
        log.warning("could not read CV details: %s", str(error).splitlines()[0] if str(error) else type(error).__name__)
        return dict(EMPTY)


def _extract(pages: list[tuple[int, str]], chunks: list[Chunk], today: date, file_name: str = "") -> dict:
    first_page = pages[0][1] if pages else ""
    lines = [" ".join(line.split()) for line in first_page.splitlines() if line.strip()][:HEADER_LINES]
    page_lines = [" ".join(line.split()) for line in first_page.splitlines() if line.strip()][:HEADER_LINES * 8]  # the first page, not only its top
    header, offsets = "", []
    for line in lines:  # the top of the CV as one text, remembering where each line starts
        offsets.append(len(header))
        header += line + "\n"
    header = header[:HEADER_CHARS]
    entities = _entities(header)
    full_text = "\n".join(text for _, text in pages)[:FULL_TEXT_CHARS]
    email = _EMAIL.search(header) or _EMAIL.search(full_text)

    email_name, email_letters = _name_from_email(email.group()) if email else ("", "")
    file_name_name, file_letters = _name_from_file(file_name)
    name, name_index = _name(lines, entities, offsets)
    if name and name_index >= 2 and (email_name or file_name_name) and not _agrees(name, email_letters, file_letters):
        name, name_index = email_name or file_name_name, -1  # a name a few lines down that the email or file name contradicts is probably not the owner
    if not name:  # the name is not in the first lines: look further, with the email address and file name as a check
        name = _name_elsewhere(pages, email_name, email_letters, file_name_name, file_letters)
    experience_lines = [line for c in chunks if section_type(c.section) == "experience" for line in c.text.splitlines()[1:]]  # [1:] skips the heading

    return {
        "candidate_name": name,
        "job_title": _job_title(lines, name_index, experience_lines),
        "years_experience": _years(chunks, full_text, today),
        "email": _clean(email.group(), 100) if email else "",
        "phone": _phone(header) or _phone(full_text),
        "location": _location_from_contact(page_lines) or _location(header, entities),
    }
