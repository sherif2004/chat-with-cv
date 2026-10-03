"""Layout-aware extraction with Docling: reading order across columns, headings, tables. Handles PDF and DOCX."""
import hashlib
import io
import json
import logging
import os
import threading
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

# torch.compile adds ~40 s to the first conversion for no benefit on one-off CV pages.
# Must be set before torch is imported.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from docling.datamodel.base_models import DocumentStream
from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DocItemLabel, TableItem
from docling.datamodel.base_models import InputFormat

from cv_chat import config

log = logging.getLogger(__name__)
SUPPORTED = (".pdf", ".docx")


@dataclass(frozen=True)
class ExtractedCV:
    """Text per page in reading order, plus the lines the layout model labelled as headings."""

    pages: list[tuple[int, str]]
    headers: list[str]


_lock = threading.Lock()  # the layout models are not thread-safe: one conversion at a time
_converter: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    """Loaded on first use (the first call downloads/loads the models). Call with _lock held."""
    global _converter
    if _converter is None:
        options = PdfPipelineOptions(accelerator_options=AcceleratorOptions(device=config.DOCLING_DEVICE))
        _converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
        _converter.initialize_pipeline(InputFormat.PDF)
    return _converter


def extract_cv(file_name: str, data: bytes) -> ExtractedCV:
    """Docling output, cached on disk by file content: changing the chunking never re-runs the layout models."""
    if not file_name.lower().endswith(SUPPORTED):
        raise ValueError(f"Unsupported file type: {file_name} (expected PDF or DOCX)")
    cache_file = Path(config.EXTRACT_CACHE_DIR) / f"{hashlib.md5(data).hexdigest()}.json"
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


def _convert(file_name: str, data: bytes) -> ExtractedCV:
    with _lock:
        document = _get_converter().convert(DocumentStream(name=file_name, stream=io.BytesIO(data))).document

    lines: dict[int, list[str]] = defaultdict(list)
    headers: list[str] = []
    for item, _ in document.iterate_items():
        page = item.prov[0].page_no if item.prov else 1  # DOCX has no pages
        if isinstance(item, TableItem):
            text = item.export_to_markdown(doc=document)
        else:
            text = " ".join(getattr(item, "text", "").split())
            if getattr(item, "label", None) == DocItemLabel.SECTION_HEADER and text:
                headers.append(text)
        if text.strip():
            lines[page].append(text)

    pages = [(page, "\n".join(texts)) for page, texts in sorted(lines.items())]
    log.info("docling: %s -> %d page(s), %d header(s)", file_name, len(pages), len(headers))
    return ExtractedCV(pages=pages, headers=headers)
