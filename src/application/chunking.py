"""Generic CV chunker: split on headings (whatever the CV calls them), then window long sections."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RawChunk:
    section: str  # the heading text as written in the CV; "Other" before the first heading
    page: int
    text: str


def _clean(line: str) -> str:
    return " ".join(line.split())


def _is_heading(line: str, layout_headers: set[str]) -> bool:
    """A line the layout model labelled as a heading, or a short ALL-CAPS line."""
    line = _clean(line)
    if not line or len(line) > 40:
        return False
    return line in layout_headers or (len(line.split()) <= 5 and line.isupper())


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
    pages: list[tuple[int, str]],
    layout_headers: list[str] = (),
    max_chars: int = 3200,
    overlap: int = 400,
) -> list[RawChunk]:
    headers = {_clean(h) for h in layout_headers}
    blocks: list[tuple[str, int, str]] = []  # (heading, page, body)
    heading, buf, buf_page = "Other", [], 1

    def emit() -> None:
        text = "\n".join(buf).strip()
        if text:  # a heading with no body under it makes no chunk
            blocks.append((heading, buf_page, text))

    for page_no, text in pages:
        for line in text.splitlines():
            if _is_heading(line, headers):
                emit()
                heading, buf, buf_page = _clean(line), [], page_no
                continue
            if not buf:
                buf_page = page_no
            buf.append(line)
    emit()

    # The heading line stays in the text of every chunk of its section.
    return [
        RawChunk(head, page, window if head == "Other" else f"{head}\n{window}")
        for head, page, text in blocks
        for window in _windows(text, max_chars, overlap)
    ]
