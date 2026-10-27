from cv_chat.processing.chunking import chunk_cv

CV = [(1, "Jane Doe\njane@example.com\nEXPERIENCE\nBuilt APIs\nLed a team\nSKILLS\nPython, SQL")]


def chunks(pages=CV, headers=(), max_chars=4000, overlap=400, caps_headings=True):
    return chunk_cv(pages, list(headers), max_chars, overlap, caps_headings)


def test_each_section_is_one_chunk_that_starts_with_its_heading():
    found = chunks()
    assert [(c.section, c.text) for c in found] == [
        ("Other", "Jane Doe\njane@example.com"),
        ("EXPERIENCE", "EXPERIENCE\nBuilt APIs\nLed a team"),
        ("SKILLS", "SKILLS\nPython, SQL"),
    ]


def test_layout_headings_count_even_when_they_are_not_all_caps():
    found = chunks([(1, "Work History\nBuilt APIs")], headers=["Work History"])
    assert [(c.section, c.text) for c in found] == [("Work History", "Work History\nBuilt APIs")]


def test_the_all_caps_fallback_can_be_switched_off():
    found = chunks(caps_headings=False)
    assert [c.section for c in found] == ["Other"]  # nothing is a heading without the layout model's help


def test_a_long_caps_line_is_not_a_heading():
    line = "THIS IS A VERY LONG LINE THAT SHOUTS BUT IS NOT A HEADING"
    assert [c.section for c in chunks([(1, f"{line}\nbody")])] == ["Other"]


def test_a_heading_with_nothing_under_it_makes_no_chunk():
    assert [c.section for c in chunks([(1, "EXPERIENCE\nSKILLS\nPython")])] == ["SKILLS"]


def test_a_chunk_has_the_page_where_its_section_starts():
    pages = [(1, "EXPERIENCE\nline one"), (2, "line two\nSKILLS\nPython")]
    found = chunks(pages)
    assert [(c.section, c.page) for c in found] == [("EXPERIENCE", 1), ("SKILLS", 2)]
    assert found[0].text == "EXPERIENCE\nline one\nline two"


def test_a_section_longer_than_the_limit_is_split_at_line_breaks_and_each_part_keeps_the_heading():
    body = "\n".join(f"line number {n:03d}" for n in range(60))
    found = chunks([(1, f"EXPERIENCE\n{body}")], max_chars=300, overlap=60)
    assert len(found) > 1
    assert all(c.section == "EXPERIENCE" and c.text.startswith("EXPERIENCE\n") for c in found)
    assert all(len(c.text) <= 300 + len("EXPERIENCE\n") for c in found)
    joined = "\n".join(c.text for c in found)
    assert all(f"line number {n:03d}" in joined for n in range(60))  # nothing is lost


def test_no_pages_gives_no_chunks():
    assert chunks([]) == []
