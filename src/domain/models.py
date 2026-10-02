from dataclasses import dataclass

@dataclass(frozen=True)
class ExtractedCV:
    """Text per page in reading order, plus the lines the layout model labelled as headings."""

    pages: list[tuple[int, str]]
    headers: list[str]


@dataclass(frozen=True)
class Chunk:
    id: str
    cv_id: str
    filename: str
    section: str
    page: int
    content: str
    content_hash: str
    vector: list[float]


@dataclass(frozen=True)
class CVInfo:
    cv_id: str
    filename: str
    chunks: int
