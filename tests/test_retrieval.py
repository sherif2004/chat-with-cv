import uuid

import pytest

from cv_chat.rag import retrieval
from cv_chat.workspace import Workspace


def hit(name: str, score: float, text: str = "excerpt", page: int = 1) -> dict:
    return {"file_name": name, "file_url": "", "section": "Skills", "page": page, "content": f"{name}: {text}", "caption": "c", "score": score}


def full_cv(name: str, count: int) -> list[dict]:
    return [{"file_name": name, "file_url": "", "section": f"S{n}", "page": 1, "content": f"{name} chunk {n}", "caption": ""} for n in range(count)]


class FakeIndex:
    """Stands in for services.search_index: hybrid_search answers from `answers` in order, get_cv_chunks from `cvs`."""

    def __init__(self, answers: list[list[dict]], cvs: dict[str, list[dict]]):
        self.answers, self.cvs, self.searches = list(answers), cvs, []

    def hybrid_search(self, ws, text, vector, k, boost_sections=None, file_ids=None, where=None):
        self.searches.append({"text": text, "file_ids": file_ids})
        return self.answers.pop(0) if self.answers else []

    def get_cv_chunks(self, ws, file_id):
        return self.cvs[file_id]


@pytest.fixture
def ws():
    return Workspace(str(uuid.uuid4()))


def install(monkeypatch, index: FakeIndex) -> None:
    monkeypatch.setattr(retrieval.search_index, "hybrid_search", index.hybrid_search)
    monkeypatch.setattr(retrieval.search_index, "get_cv_chunks", index.get_cv_chunks)
    monkeypatch.setattr(retrieval.openai_service, "embed", lambda texts: [[0.0] for _ in texts])


def ids(*names: str) -> list[str]:
    return list(names)  # the file id is the file name


def test_best_cvs_orders_by_each_cvs_best_chunk_and_keeps_k():
    results = [hit("a", 0.9), hit("b", 0.5), hit("a", 0.4), hit("c", 0.7)]
    assert retrieval.best_cvs(results, 2) == ["a", "c"]


def test_best_cvs_with_no_scores_follows_the_search_order():
    results = [{**hit("a", 0), "score": None}, {**hit("b", 0), "score": None}]
    assert retrieval.best_cvs(results, 5) == ["a", "b"]


def test_every_chunk_of_the_top_k_cvs_reaches_the_model_and_no_other_cv(ws, monkeypatch):
    index = FakeIndex([[hit("b", 0.5), hit("a", 0.9)]], {"a": full_cv("a", 3), "b": full_cv("b", 2), "c": full_cv("c", 4)})
    install(monkeypatch, index)
    found = retrieval.retrieve(ws, ["q-every-chunk"], [], ids("a", "b", "c"), k=2)
    assert [chunk["file_name"] for chunk in found.context] == ["a", "a", "a", "b", "b"]
    assert [chunk["file_name"] for chunk in found.matched] == ["a", "b"]  # the sources are what the search matched


def test_k_limits_how_many_cvs_are_read(ws, monkeypatch):
    index = FakeIndex([[hit("a", 0.9), hit("b", 0.8), hit("c", 0.7)]], {n: full_cv(n, 1) for n in "abc"})
    install(monkeypatch, index)
    found = retrieval.retrieve(ws, ["q-k-limit"], [], ids("a", "b", "c"), k=1)
    assert {chunk["file_name"] for chunk in found.context} == {"a"}


def test_a_scoped_search_is_limited_to_the_given_cvs(ws, monkeypatch):
    index = FakeIndex([[hit("a", 0.9)]], {"a": full_cv("a", 1)})
    install(monkeypatch, index)
    retrieval.retrieve(ws, ["q-scoped"], [], ids("a"), k=3, scoped=True)
    assert index.searches[0]["file_ids"] == ["a"]


def test_an_unscoped_search_does_not_filter_by_cv(ws, monkeypatch):
    index = FakeIndex([[hit("a", 0.9)]], {"a": full_cv("a", 1), "b": full_cv("b", 1)})
    install(monkeypatch, index)
    retrieval.retrieve(ws, ["q-unscoped"], [], ids("a", "b"), k=1)
    assert index.searches[0]["file_ids"] is None


def test_results_from_cvs_outside_the_given_set_are_dropped(ws, monkeypatch):
    index = FakeIndex([[hit("x", 0.99), hit("a", 0.5)]], {"a": full_cv("a", 1)})
    install(monkeypatch, index)
    found = retrieval.retrieve(ws, ["q-outside"], [], ids("a"), k=2)
    assert [chunk["file_name"] for chunk in found.context] == ["a"]


def test_cvs_the_first_search_missed_are_searched_again(ws, monkeypatch):
    index = FakeIndex([[hit("a", 0.9)], [hit("b", 0.4)]], {"a": full_cv("a", 1), "b": full_cv("b", 1)})
    install(monkeypatch, index)
    found = retrieval.retrieve(ws, ["q-cover"], [], ids("a", "b"), k=2)
    assert index.searches[1]["file_ids"] == ["b"]
    assert {chunk["file_name"] for chunk in found.context} == {"a", "b"}


def test_no_hits_gives_no_context(ws, monkeypatch):
    install(monkeypatch, FakeIndex([[]], {}))
    found = retrieval.retrieve(ws, ["q-nothing"], [], ids("a"), k=2)
    assert found.context == [] and found.matched == []
