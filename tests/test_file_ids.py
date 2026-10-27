import re

import pytest

from cv_chat.rag import ingest
from cv_chat.services import search_index

AZURE_KEY = re.compile(r"^[A-Za-z0-9_\-=]+$")  # what Azure AI Search allows in a document key
NAMES = ["cv.pdf", "Ali Hassan (final).docx", "o'brien, jane.pdf", "سيرة ذاتية.pdf", "a" * 200 + ".pdf"]


@pytest.mark.parametrize("name", NAMES)
def test_chunk_key_only_uses_characters_azure_allows(name):
    assert AZURE_KEY.match(search_index.chunk_key(name, 7))


def test_chunk_key_ends_with_the_chunk_number_and_is_unique_per_name_and_number():
    keys = {search_index.chunk_key(name, n) for name in NAMES for n in range(3)}
    assert len(keys) == len(NAMES) * 3
    assert search_index.chunk_key("cv.pdf", 12).rsplit("-", 1)[1] == "12"
    assert search_index.chunk_key("cv.pdf", 1) == search_index.chunk_key("cv.pdf", 1)


def test_the_longest_allowed_name_still_makes_a_key_under_azures_limit():
    name = "ب" * 127 + "a"  # 255 bytes of UTF-8
    assert len(name.encode()) == 255
    assert len(search_index.chunk_key(name, 999)) <= 1024


def test_odata_string_doubles_single_quotes():
    assert search_index.odata_string("o'brien.pdf") == "'o''brien.pdf'"


def test_one_file_id_is_matched_with_eq():
    assert search_index.file_id_filter(["o'brien.pdf"]) == "file_id eq 'o''brien.pdf'"


def test_several_file_ids_use_search_in_with_a_delimiter_that_is_in_none_of_them():
    text = search_index.file_id_filter(["a|b.pdf", "c.pdf"])
    assert text == "search.in(file_id, 'a|b.pdf;c.pdf', ';')"


def test_when_every_delimiter_is_taken_the_filter_falls_back_to_eq_clauses():
    taken = ["x" + "".join(search_index.DELIMITERS) + ".pdf", "y.pdf"]
    text = search_index.file_id_filter(taken)
    assert text.startswith("(file_id eq ") and " or file_id eq 'y.pdf'" in text


def test_no_file_ids_gives_no_filter():
    assert search_index.file_id_filter([]) is None


@pytest.mark.parametrize("name", ["cv.pdf", "Ali Hassan (final).docx", "o'brien, jane.pdf", "سيرة ذاتية.pdf", "ب" * 127 + "a"])
def test_plain_file_names_are_accepted(name):
    ingest.validate_file_name(name)


@pytest.mark.parametrize("name", ["", "   ", "a/b.pdf", "a\\b.pdf", "..", ".", "bad\x00.pdf", "ب" * 128])
def test_unsafe_or_too_long_names_are_refused(name):
    with pytest.raises(ValueError):
        ingest.validate_file_name(name)
