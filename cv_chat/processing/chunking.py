"""Section-aware chunking: one chunk per section of the CV, windowed only when a section is too long."""
from dataclasses import dataclass

from cv_chat.config import settings


@dataclass(frozen=True)
class Chunk:
    section: str  # the heading as written in the CV; "Other" before the first heading
    page: int
    text: str  # starts with the heading, so every chunk says which section it belongs to


def _clean(line: str) -> str:
    return " ".join(line.split())


def _is_heading(line: str, layout_headers: set[str], caps_headings: bool) -> bool:
    """A line the layout model labelled as a heading, or (if caps_headings) a short ALL-CAPS line."""
    line = _clean(line)
    if not line or len(line) > settings.chunking.heading_max_chars:
        return False
    return line in layout_headers or (caps_headings and len(line.split()) <= settings.chunking.caps_max_words and line.isupper())


def _windows(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            cut = text.rfind("\n", start + max_chars // 2, end)  # prefer a line break
            if cut != -1:
                end = cut
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [w for w in out if w]


def chunk_cv(
    pages: list[tuple[int, str]], layout_headers: list[str], max_chars: int = settings.chunking.max_chars,
    overlap: int = settings.chunking.overlap, caps_headings: bool = settings.chunking.caps_headings,
) -> list[Chunk]:
    """One chunk per section: the heading and everything under it. Only a section longer than max_chars is split."""
    headers = {_clean(h) for h in layout_headers}
    blocks: list[tuple[str, int, str]] = []  # (heading, page, body)
    heading, buf, buf_page = "Other", [], 1

    def emit() -> None:
        text = "\n".join(buf).strip()
        if text:  # a heading with no body under it makes no chunk
            blocks.append((heading, buf_page, text))

    for page_no, text in pages:
        for line in text.splitlines():
            if _is_heading(line, headers, caps_headings):
                emit()
                heading, buf, buf_page = _clean(line), [], page_no
                continue
            if not buf:
                buf_page = page_no
            buf.append(line)
    emit()

    return [
        Chunk(head, page, window if head == "Other" else f"{head}\n{window}")
        for head, page, text in blocks
        for window in _windows(text, max_chars, overlap)
    ]
