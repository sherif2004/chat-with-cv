import uuid

import pytest
from azure.core.exceptions import HttpResponseError

from cv_chat.rag import roles
from cv_chat.services import search_index
from cv_chat.workspace import Workspace


def result(name="a.pdf", score=2.5):
    return {"file_name": name, "file_url": "", "section": "Skills", "page": 1, "content": "text", "@search.captions": [],
            "@search.reranker_score": score, "@search.score": 1.0, "candidate_name": "", "job_title": "", "email": "", "phone": "", "location": ""}


class FakeClient:
    """Records every request; `semantic` False makes a semantic request fail the way a service without the ranker does."""

    def __init__(self, rows=None, semantic=True):
        self.rows, self.semantic, self.requests = rows or [], semantic, []

    def search(self, text=None, **kwargs):
        self.requests.append({"text": text, **kwargs})
        if kwargs.get("query_type") == "semantic" and not self.semantic:
            raise HttpResponseError("Semantic search is not enabled for this service")
        return list(self.rows)


@pytest.fixture
def ws():
    return Workspace(str(uuid.uuid4()))


@pytest.fixture(autouse=True)
def semantic_available(monkeypatch):
    monkeypatch.setattr(search_index, "_semantic_available", True)


def install(monkeypatch, client):
    monkeypatch.setattr(search_index, "_client", lambda ws: client)
    return client


def test_a_search_is_one_request_with_keyword_vector_and_semantic_ranking_together(monkeypatch, ws):
    client = install(monkeypatch, FakeClient([result()]))
    found = search_index.hybrid_search(ws, "python", [0.1, 0.2], 50)
    assert len(client.requests) == 1
    request = client.requests[0]
    assert request["search_text"] == "python" and request["query_type"] == "semantic" and request["vector_queries"] and request["top"] == 50
    assert found[0]["file_name"] == "a.pdf" and found[0]["score"] == 2.5


def test_the_sections_the_router_picked_boost_the_ranking_instead_of_filtering_the_search(monkeypatch, ws):
    client = install(monkeypatch, FakeClient())
    search_index.hybrid_search(ws, "python", [0.1], 50, boost_sections=["education"])
    request = client.requests[0]
    assert request["scoring_parameters"] == ["boostSections-education"]
    assert request["filter"] is None  # a CV with unusual headings is still found


def test_without_picked_sections_the_default_boosted_sections_are_used(monkeypatch, ws):
    client = install(monkeypatch, FakeClient())
    search_index.hybrid_search(ws, "python", [0.1], 50)
    assert client.requests[0]["scoring_parameters"] == ["boostSections-experience,skills"]


def test_cvs_and_an_extra_condition_are_filters_of_the_same_request(monkeypatch, ws):
    client = install(monkeypatch, FakeClient())
    search_index.hybrid_search(ws, "python", [0.1], 50, file_ids=["a.pdf"], where="search.ismatch('x', 'job_title')")
    assert client.requests[0]["filter"] == "file_id eq 'a.pdf' and (search.ismatch('x', 'job_title'))"


def test_a_service_without_the_semantic_ranker_costs_one_extra_request_once_not_on_every_search(monkeypatch, ws):
    client = install(monkeypatch, FakeClient([result()], semantic=False))
    search_index.hybrid_search(ws, "first", [0.1], 50)
    assert [r.get("query_type") for r in client.requests] == ["semantic", None]
    client.requests.clear()
    search_index.hybrid_search(ws, "second", [0.1], 50)
    assert [r.get("query_type") for r in client.requests] == [None]


def test_another_error_is_not_swallowed(monkeypatch, ws):
    class Broken(FakeClient):
        def search(self, text=None, **kwargs):
            raise HttpResponseError("the index is gone")

    install(monkeypatch, Broken())
    with pytest.raises(HttpResponseError):
        search_index.hybrid_search(ws, "x", [0.1], 50)


# --- one profile per CV --------------------------------------------------------------------------------------------

def test_profiles_are_read_from_the_first_chunk_of_each_cv_only(monkeypatch, ws):
    rows = [{"file_name": "a.pdf", "candidate_name": "A", "job_title": "AI Engineer", "email": "", "phone": "", "location": ""},
            {"file_name": "b.pdf", "candidate_name": "B", "job_title": "Data Analyst", "email": "", "phone": "", "location": ""}]
    client = install(monkeypatch, FakeClient(rows))
    profiles = search_index.list_profiles(ws)
    assert client.requests[0]["filter"] == "chunk_no eq 0"
    assert set(profiles) == {"a.pdf", "b.pdf"} and profiles["b.pdf"]["job_title"] == "Data Analyst"


def test_the_chunks_of_a_cv_are_returned_in_reading_order(monkeypatch, ws):
    rows = [{"id": "x-2", "chunk_no": 2, "file_name": "a.pdf", "content": "third"}, {"id": "x-0", "chunk_no": 0, "file_name": "a.pdf", "content": "first"},
            {"id": "x-1", "chunk_no": 1, "file_name": "a.pdf", "content": "second"}]
    install(monkeypatch, FakeClient(rows))
    assert [c["content"] for c in search_index.get_cv_chunks(ws, "a.pdf")] == ["first", "second", "third"]


# --- the role check uses the job title ------------------------------------------------------------------------------

def profiles(monkeypatch, **titles):
    monkeypatch.setattr(roles.search_index, "list_profiles", lambda ws: {name: {"job_title": title} for name, title in titles.items()})


def test_a_cv_has_the_role_when_every_word_of_it_is_in_its_job_title(monkeypatch, ws):
    profiles(monkeypatch, **{"a.pdf": "Senior Data Engineer", "b.pdf": "Data Analyst", "c.pdf": ""})
    have, titles = roles.find(ws, "data engineer")
    assert have == ["a.pdf"] and titles == ["Data Analyst", "Senior Data Engineer"]


def test_the_role_check_can_be_limited_to_the_chosen_cvs(monkeypatch, ws):
    profiles(monkeypatch, **{"a.pdf": "Data Engineer", "b.pdf": "Data Engineer"})
    assert roles.find(ws, "data engineer", ["b.pdf"])[0] == ["b.pdf"]


def test_an_empty_role_matches_every_cv(monkeypatch, ws):
    profiles(monkeypatch, **{"a.pdf": "x", "b.pdf": "y"})
    assert roles.find(ws, "   ")[0] == ["a.pdf", "b.pdf"]
