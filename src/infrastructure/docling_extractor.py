import io
import logging
import os
import threading
import time
from collections import defaultdict

# torch.compile adds ~40 s to the first conversion for no benefit on one-off CV pages.
# Must be set before torch is imported.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DocItemLabel, TableItem

from src.domain.models import ExtractedCV

log = logging.getLogger(__name__)


class DoclingExtractor:
    """Layout-aware extraction: reading order across columns, headings, tables, and OCR for scans."""

    def __init__(self, device: str = "cpu"):
        options = PdfPipelineOptions(accelerator_options=AcceleratorOptions(device=device))
        self._converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
        self._lock = threading.Lock()  # one conversion at a time: the models are not thread-safe

    def warm_up(self) -> None:
        """Load the layout models now so the first upload doesn't pay for it."""
        started = time.perf_counter()
        with self._lock:
            self._converter.initialize_pipeline(InputFormat.PDF)
        log.info("docling models loaded in %.1fs", time.perf_counter() - started)

    def extract(self, data: bytes, filename: str) -> ExtractedCV:
        started = time.perf_counter()
        with self._lock:
            document = self._converter.convert(DocumentStream(name=filename, stream=io.BytesIO(data))).document

        lines: dict[int, list[str]] = defaultdict(list)
        headers: list[str] = []
        for item, _ in document.iterate_items():
            page = item.prov[0].page_no if item.prov else 1
            if isinstance(item, TableItem):
                text = item.export_to_markdown(doc=document)
            else:
                text = " ".join(getattr(item, "text", "").split())
                if getattr(item, "label", None) == DocItemLabel.SECTION_HEADER and text:
                    headers.append(text)
            if text.strip():
                lines[page].append(text)

        pages = [(page, "\n".join(texts)) for page, texts in sorted(lines.items())]
        log.info("docling: %s -> %d page(s), %d header(s), %.1fs", filename, len(pages), len(headers), time.perf_counter() - started)
        return ExtractedCV(pages=pages, headers=headers)
