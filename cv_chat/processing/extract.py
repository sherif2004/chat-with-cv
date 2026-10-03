"""Extract plain text from CV files: PDF with PyMuPDF, DOCX with python-docx."""
import io

import docx
import pymupdf
from docx.table import Table


def extract_text(file_name: str, data: bytes) -> str:
    name = file_name.lower()
    if name.endswith(".pdf"):
        return _pdf_text(data)
    if name.endswith(".docx"):
        return _docx_text(data)
    raise ValueError(f"Unsupported file type: {file_name} (expected PDF or DOCX)")


def _pdf_text(data: bytes) -> str:
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        return "\n\n".join(page.get_text() for page in pdf)


def _docx_text(data: bytes) -> str:
    lines = []
    for block in docx.Document(io.BytesIO(data)).iter_inner_content():
        if isinstance(block, Table):
            for row in block.rows:
                cells = dict.fromkeys(cell.text.strip() for cell in row.cells)  # merged cells repeat their text
                lines.append(" | ".join(cell for cell in cells if cell))
        else:
            lines.append(block.text)
    return "\n".join(lines)
