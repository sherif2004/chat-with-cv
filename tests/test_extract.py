from types import SimpleNamespace as NS

from cv_chat.processing import extract


def span(offset: int, length: int):
    return NS(offset=offset, length=length)


def paragraph(content: str, offset: int, page: int = 1, role: str | None = None):
    return NS(content=content, role=role, spans=[span(offset, len(content))], bounding_regions=[NS(page_number=page)])


def result(content: str, paragraphs=(), tables=(), pages=()):
    return NS(content=content, paragraphs=list(paragraphs), tables=list(tables), pages=list(pages))


def one_page(length: int, number: int = 1, offset: int = 0):
    return NS(page_number=number, spans=[span(offset, length)])


def test_page_numbers_and_footers_are_kept():
    text = "Jane Doe\nPage 1 of 2\nConfidential"
    found = extract.assemble(result(text, [
        paragraph("Jane Doe", 0),
        paragraph("Page 1 of 2", 9, role="pageNumber"),
        paragraph("Confidential", 21, role="pageFooter"),
    ], pages=[one_page(len(text))]))
    assert found.pages == [(1, "Jane Doe\nPage 1 of 2\nConfidential")]


def test_titles_and_section_headings_are_reported_as_headers():
    text = "Jane Doe\nEXPERIENCE\nBuilt things"
    found = extract.assemble(result(text, [
        paragraph("Jane Doe", 0, role="title"),
        paragraph("EXPERIENCE", 9, role="sectionHeading"),
        paragraph("Built things", 20),
    ], pages=[one_page(len(text))]))
    assert found.headers == ["Jane Doe", "EXPERIENCE"]


def test_reading_order_follows_the_document_offset_and_pages_stay_separate():
    text = "second\nfirst\nthird"
    found = extract.assemble(result(text, [
        paragraph("second", 0, page=1),
        paragraph("first", 7, page=1),
        paragraph("third", 13, page=2),
    ], pages=[one_page(13, 1, 0), one_page(5, 2, 13)]))
    assert found.pages == [(1, "second\nfirst"), (2, "third")]


def test_a_table_is_kept_once_as_markdown_and_its_cell_paragraphs_are_not_repeated():
    text = "Skill Level Python Expert"
    cells = [
        NS(row_index=0, column_index=0, content="Skill"), NS(row_index=0, column_index=1, content="Level"),
        NS(row_index=1, column_index=0, content="Python"), NS(row_index=1, column_index=1, content="Expert"),
    ]
    table = NS(cells=cells, column_count=2, spans=[span(0, len(text))], bounding_regions=[NS(page_number=1)])
    found = extract.assemble(result(text, [paragraph("Skill", 0), paragraph("Level", 6), paragraph("Python", 12), paragraph("Expert", 19)],
                                    [table], [one_page(len(text))]))
    assert found.pages == [(1, "| Skill | Level |\n| --- | --- |\n| Python | Expert |")]


def test_text_that_is_in_no_paragraph_is_kept_too():
    text = "Jane Doe\n:selected: Willing to relocate"
    found = extract.assemble(result(text, [paragraph("Jane Doe", 0)], pages=[one_page(len(text))]))
    assert found.pages == [(1, "Jane Doe\n:selected: Willing to relocate")]


def test_the_cache_file_depends_on_the_content_and_the_extraction_version():
    assert extract.cache_path(b"cv", 1) != extract.cache_path(b"cv", 2)
    assert extract.cache_path(b"cv", 1) != extract.cache_path(b"other", 1)
    assert extract.cache_path(b"cv", 1) == extract.cache_path(b"cv", 1)
