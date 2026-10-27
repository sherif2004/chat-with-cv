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
_SKIPPED_ROLES = ("pageNumber", "pageFooter")  # repeated on every page, not CV content


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


def extract_cv(file_name: str, data: bytes) -> ExtractedCV:
    """Azure Document Intelligence output, cached on disk by file content: changing the chunking never calls the service again."""
    if not file_name.lower().endswith(SUPPORTED):
        raise ValueError(f"Unsupported file type: {file_name} (expected PDF or DOCX)")
    cache_file = Path(settings.extraction.cache_dir) / f"{hashlib.md5(data).hexdigest()}.json"
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

    # (offset in the document text, page, text, is a heading): sorting by offset gives reading order
    items: list[tuple[int, int, str, bool]] = []
    table_spans: list[tuple[int, int]] = []
    for table in result.tables or []:
        if not table.spans or not table.bounding_regions:
            continue
        table_spans.append((table.spans[0].offset, table.spans[0].offset + table.spans[0].length))
        items.append((table.spans[0].offset, table.bounding_regions[0].page_number, _table_markdown(table), False))
    for paragraph in result.paragraphs or []:
        if paragraph.role in _SKIPPED_ROLES or not paragraph.spans:
            continue
        offset = paragraph.spans[0].offset
        if any(start <= offset < end for start, end in table_spans):
            continue  # already part of the table
        page = paragraph.bounding_regions[0].page_number if paragraph.bounding_regions else 1
        items.append((offset, page, " ".join(paragraph.content.split()), paragraph.role in _HEADING_ROLES))

    lines: dict[int, list[str]] = defaultdict(list)
    headers: list[str] = []
    for _, page, text, is_heading in sorted(items, key=lambda item: item[0]):
        if not text.strip():
            continue
        if is_heading:
            headers.append(text)
        lines[page].append(text)

    pages = [(page, "\n".join(texts)) for page, texts in sorted(lines.items())]
    log.info("document intelligence: %s -> %d page(s), %d header(s)", file_name, len(pages), len(headers))
    return ExtractedCV(pages=pages, headers=headers)
