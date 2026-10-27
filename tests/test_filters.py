import uuid

import pytest

from cv_chat.rag import filters
from cv_chat.workspace import Workspace

PROFILES = {
    "ali.pdf": {"job_title": "Senior Data Engineer", "location": "Cairo, Egypt"},
    "mona.pdf": {"job_title": "Data Analyst", "location": "Alexandria"},
    "omar.pdf": {"job_title": "Backend Engineers Lead", "location": "Cairo"},
    "none.pdf": {"job_title": "", "location": None},  # nothing could be read from this CV
}


@pytest.fixture
def ws():
    return Workspace(str(uuid.uuid4()))


@pytest.fixture(autouse=True)
def profiles(monkeypatch):
    monkeypatch.setattr(filters.search_index, "list_profiles", lambda ws: PROFILES)


def names(ws, **kwargs):
    return filters.find(ws, **kwargs).names


def test_a_job_title_matches_when_every_word_of_it_is_in_the_cvs_title(ws):
    assert names(ws, job_title="data engineer") == ["ali.pdf"]


def test_word_order_and_case_do_not_matter(ws):
    assert names(ws, job_title="ENGINEER data") == ["ali.pdf"]


def test_a_plural_matches_the_singular_and_the_other_way_round(ws):
    assert names(ws, job_title="backend engineer") == ["omar.pdf"]  # the title says "Engineers"
    assert names(ws, job_title="data analysts") == ["mona.pdf"]  # the title says "Analyst"


def test_a_location_matches_when_every_word_of_it_is_in_the_cvs_location(ws):
    assert names(ws, location="cairo") == ["ali.pdf", "omar.pdf"]
    assert names(ws, location="Egypt") == ["ali.pdf"]


def test_job_and_location_must_both_match(ws):
    assert names(ws, job_title="engineer", location="egypt") == ["ali.pdf"]
    assert names(ws, job_title="analyst", location="cairo") == []


def test_a_cv_whose_detail_could_not_be_read_never_matches_a_filter_on_it(ws):
    assert "none.pdf" not in names(ws, job_title="engineer") + names(ws, location="cairo")


def test_without_a_filter_every_cv_matches(ws):
    assert names(ws) == sorted(PROFILES) and names(ws, job_title="  ", location="") == sorted(PROFILES)


def test_a_filter_with_no_usable_word_is_no_filter(ws):
    assert names(ws, job_title="!!!") == sorted(PROFILES)


def test_the_search_can_be_limited_to_some_cvs(ws):
    assert names(ws, job_title="engineer", scope=["omar.pdf"]) == ["omar.pdf"]
    assert names(ws, job_title="engineer", scope=["mona.pdf"]) == []


def test_the_titles_and_locations_that_exist_are_returned_for_the_cvs_in_scope(ws):
    found = filters.find(ws, job_title="astronaut", scope=["ali.pdf", "mona.pdf"])
    assert found.names == [] and found.titles == ["Data Analyst", "Senior Data Engineer"] and found.locations == ["Alexandria", "Cairo, Egypt"]


def test_the_message_names_each_filter_that_was_set_and_what_exists(ws):
    found = filters.find(ws, job_title="astronaut", location="mars")
    text = filters.missing_message("astronaut", "mars", found)
    assert "**astronaut**" in text and "**mars**" in text and "`Data Analyst`" in text and "`Alexandria`" in text


def test_the_message_mentions_only_the_filter_that_was_set(ws):
    text = filters.missing_message("", "mars", filters.find(ws, location="mars"))
    assert "mars" in text and "job title" not in text.split("The ")[0]


def test_text_from_cvs_and_from_the_user_cannot_add_markup_to_the_message(ws):
    found = filters.Found([], ["Dev`; <b>x</b>"], ["Cairo"])
    text = filters.missing_message("**bold", "", found)
    assert text.count("**") == 2 and text.count("`") % 2 == 0  # only the message's own bold pair
