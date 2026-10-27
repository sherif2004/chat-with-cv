"""Layout-aware extraction with Azure Document Intelligence (prebuilt-layout): reading order across columns, headings, tables. Handles PDF and DOCX."""
import hashlib
import json
import logging
import threading
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

from cv_chat.config import settings

log = logging.getLogger(__name__)
SUPPORTED = (".pdf", ".docx")
_HEADING_ROLES = ("title", "sectionHeading")


@dataclass(frozen=True)
class ExtractedCV:
    """Text per page in reading order, plus the lines the layout model labelled as headings."""

    pages: list[tuple[int, str]]
    headers: list[str]


_lock_for_flag = threading.Lock()
_client = None  # a DocumentIntelligenceClient, created on first use
_warm_up_started = False


def _get_client():
    global _client
    if _client is None:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential

        _client = DocumentIntelligenceClient(settings.azure.docintel_endpoint, AzureKeyCredential(settings.azure.docintel_key.get_secret_value()))
    return _client


def cache_path(data: bytes, version: int) -> Path:
    """Where the extraction of this file is cached. The version is part of the name, so changing what is kept from a CV
    (a new version number in the settings) never serves the output of the old rules."""
    return Path(settings.extraction.cache_dir) / f"{hashlib.md5(data).hexdigest()}-v{version}.json"


def extract_cv(file_name: str, data: bytes) -> ExtractedCV:
    """Azure Document Intelligence output, cached on disk by file content: changing the chunking never calls the service again."""
    if not file_name.lower().endswith(SUPPORTED):
        raise ValueError(f"Unsupported file type: {file_name} (expected PDF or DOCX)")
    cache_file = cache_path(data, settings.extraction.version)
    try:
        cached = json.loads(cache_file.read_text())
        return ExtractedCV(pages=[(page, text) for page, text in cached["pages"]], headers=cached["headers"])
    except (OSError, ValueError, KeyError):
        pass  # no usable cache entry: convert
    extracted = _convert(file_name, data)
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(f".{threading.get_ident()}.tmp")  # write-then-rename: parallel CVs never see half a file
        tmp.write_text(json.dumps({"pages": extracted.pages, "headers": extracted.headers}))
        tmp.replace(cache_file)
    except OSError:
        log.warning("could not write the extraction cache for %s", file_name)
    return extracted


def warm_up_in_background() -> None:
    """Start loading the NER model in a background thread, once per process, so the first CV does not wait for it.
    Called when the Library is first opened; people who only chat never pay for it."""
    global _warm_up_started
    with _lock_for_flag:
        if _warm_up_started:
            return
        _warm_up_started = True
    threading.Thread(target=_warm_up, name="ner-warm-up", daemon=True).start()


def _warm_up() -> None:
    try:
        from cv_chat.processing import entities

        entities.warm_up()  # the NER model (downloaded the first time)
    except Exception as error:  # the first CV will try again and report the real problem
        log.warning("could not preload the NER model: %s", str(error).splitlines()[0] if str(error) else type(error).__name__)


def _table_markdown(table) -> str:
    rows: dict[int, dict[int, str]] = defaultdict(dict)
    for cell in table.cells:
        rows[cell.row_index][cell.column_index] = " ".join((cell.content or "").split()).replace("|", "/")
    width = table.column_count
    lines = ["| " + " | ".join(row.get(c, "") for c in range(width)) + " |" for _, row in sorted(rows.items())]
    if len(lines) > 1:
        lines.insert(1, "|" + " --- |" * width)
    return "\n".join(lines)


def _convert(file_name: str, data: bytes) -> ExtractedCV:
    from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

    result = _get_client().begin_analyze_document("prebuilt-layout", AnalyzeDocumentRequest(bytes_source=data)).result()
    extracted = assemble(result)
    log.info("document intelligence: %s -> %d page(s), %d header(s)", file_name, len(extracted.pages), len(extracted.headers))
    return extracted


def _page_at(result, offset: int) -> int:
    for page in result.pages or []:
        if any(span.offset <= offset < span.offset + span.length for span in page.spans or []):
            return page.page_number
    return 1


def _uncovered(content: str, covered: list[tuple[int, int]]) -> list[tuple[int, str]]:
    """(offset, text) of the stretches of the document text that no paragraph or table accounts for, such as a checkbox state."""
    found, position = [], 0
    for start, end in sorted(covered) + [(len(content), len(content))]:
        if start > position and content[position:start].strip():
            found.append((position, content[position:start]))
        position = max(position, end)
    return found


def assemble(result) -> ExtractedCV:
    """The text of every page in reading order. Nothing is dropped: page numbers, footers, and text that belongs to no
    paragraph are kept, and a table becomes one markdown block in place of its cell paragraphs."""
    # (offset in the document text, page, text, is a heading): sorting by offset gives reading order
    items: list[tuple[int, int, str, bool]] = []
    table_spans: list[tuple[int, int]] = []
    for table in result.tables or []:
        if not table.spans or not table.bounding_regions:
            continue
        table_spans.append((table.spans[0].offset, table.spans[0].offset + table.spans[0].length))
        items.append((table.spans[0].offset, table.bounding_regions[0].page_number, _table_markdown(table), False))
    covered = list(table_spans)
    for paragraph in result.paragraphs or []:
        if not paragraph.spans:
            continue
        offset = paragraph.spans[0].offset
        covered.append((offset, offset + paragraph.spans[0].length))
        if any(start <= offset < end for start, end in table_spans):
            continue  # already part of the table
        page = paragraph.bounding_regions[0].page_number if paragraph.bounding_regions else _page_at(result, offset)
        items.append((offset, page, " ".join(paragraph.content.split()), paragraph.role in _HEADING_ROLES))
    for offset, text in _uncovered(result.content or "", covered):
        items.append((offset, _page_at(result, offset), " ".join(text.split()), False))

    lines: dict[int, list[str]] = defaultdict(list)
    headers: list[str] = []
    for _, page, text, is_heading in sorted(items, key=lambda item: item[0]):
        if not text.strip():
            continue
        if is_heading:
            headers.append(text)
        lines[page].append(text)

    return ExtractedCV(pages=[(page, "\n".join(texts)) for page, texts in sorted(lines.items())], headers=headers)
