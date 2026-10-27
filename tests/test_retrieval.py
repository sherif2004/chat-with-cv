import uuid

import pytest

from cv_chat.rag import retrieval
from cv_chat.workspace import Workspace


def hit(name: str, score: float, text: str = "excerpt", page: int = 1, chunk_no: int = 0, section: str = "Skills") -> dict:
    return {"file_name": name, "file_url": "", "section": section, "page": page, "chunk_no": chunk_no, "content": f"{name}: {text}", "caption": "c", "score": score}


class FakeIndex:
    """Stands in for services.search_index: hybrid_search answers from `answers` in order. get_cv_chunks is not faked on
    purpose: the question flow must not read whole CVs."""

    def __init__(self, answers: list[list[dict]]):
        self.answers, self.searches = list(answers), []

    def hybrid_search(self, ws, text, vector, k, boost_sections=None, file_ids=None, where=None):
        self.searches.append({"text": text, "file_ids": file_ids, "boost": boost_sections})
        return self.answers.pop(0) if self.answers else []


@pytest.fixture
def ws():
    return Workspace(str(uuid.uuid4()))


def install(monkeypatch, index: FakeIndex) -> None:
    monkeypatch.setattr(retrieval.search_index, "hybrid_search", index.hybrid_search)
    monkeypatch.setattr(retrieval.openai_service, "embed", lambda texts: [[0.0] for _ in texts])
    monkeypatch.setattr(retrieval.search_index, "get_cv_chunks", lambda ws, name: pytest.fail("a whole CV was read"))


def test_best_cvs_orders_by_each_cvs_best_chunk_and_keeps_k():
    results = [hit("a", 0.9), hit("b", 0.5), hit("a", 0.4), hit("c", 0.7)]
    assert retrieval.best_cvs(results, 2) == ["a", "c"]


def test_best_cvs_with_no_scores_follows_the_search_order():
    results = [{**hit("a", 0), "score": None}, {**hit("b", 0), "score": None}]
    assert retrieval.best_cvs(results, 5) == ["a", "b"]


def test_the_model_reads_only_the_sections_the_search_found_in_the_best_cvs_in_reading_order(ws, monkeypatch):
    index = FakeIndex([[hit("a", 0.9, "skills", chunk_no=3), hit("b", 0.6, "intro", chunk_no=0), hit("a", 0.5, "education", chunk_no=1), hit("c", 0.1)]])
    install(monkeypatch, index)
    found = retrieval.retrieve(ws, "q-sections", [], ["a", "b", "c"], k=2)
    assert [(c["file_name"], c["chunk_no"]) for c in found.context] == [("a", 1), ("a", 3), ("b", 0)]  # best CV first, reading order inside it


def test_the_sources_are_the_same_chunks_cut_short(ws, monkeypatch):
    long_text = "x" * 5000
    install(monkeypatch, FakeIndex([[hit("a", 0.9, long_text)]]))
    found = retrieval.retrieve(ws, "q-short", [], ["a"], k=1)
    limit = retrieval.settings.retrieval.source_chars
    assert len(found.context[0]["content"]) > 5000 and len(found.sources[0]["content"]) == limit + 1  # the ellipsis
    assert found.sources[0]["file_name"] == "a" and found.sources[0]["section"] == "Skills"
    assert found.context[0]["content"].endswith("x")  # the model's copy is untouched


def test_a_short_chunk_is_not_changed_in_the_sources(ws, monkeypatch):
    install(monkeypatch, FakeIndex([[hit("a", 0.9, "short")]]))
    found = retrieval.retrieve(ws, "q-short-chunk", [], ["a"], k=1)
    assert found.sources == found.context


def test_k_limits_how_many_cvs_are_read(ws, monkeypatch):
    install(monkeypatch, FakeIndex([[hit("a", 0.9), hit("b", 0.8), hit("c", 0.7)]]))
    found = retrieval.retrieve(ws, "q-k-limit", [], ["a", "b", "c"], k=1)
    assert {chunk["file_name"] for chunk in found.context} == {"a"}


def test_the_sections_the_router_picked_are_passed_on_as_a_boost(ws, monkeypatch):
    index = FakeIndex([[hit("a", 0.9)]])
    install(monkeypatch, index)
    retrieval.retrieve(ws, "q-boost", ["education"], ["a"], k=1)
    assert index.searches[0]["boost"] == ["education"]


def test_a_scoped_search_is_limited_to_the_given_cvs(ws, monkeypatch):
    index = FakeIndex([[hit("a", 0.9)]])
    install(monkeypatch, index)
    retrieval.retrieve(ws, "q-scoped", [], ["a"], k=3, scoped=True)
    assert index.searches[0]["file_ids"] == ["a"]


def test_an_unscoped_search_does_not_filter_by_cv(ws, monkeypatch):
    index = FakeIndex([[hit("a", 0.9)]])
    install(monkeypatch, index)
    retrieval.retrieve(ws, "q-unscoped", [], ["a", "b"], k=1)
    assert index.searches[0]["file_ids"] is None


def test_results_from_cvs_outside_the_given_set_are_dropped(ws, monkeypatch):
    install(monkeypatch, FakeIndex([[hit("x", 0.99), hit("a", 0.5)]]))
    found = retrieval.retrieve(ws, "q-outside", [], ["a"], k=2)
    assert [chunk["file_name"] for chunk in found.context] == ["a"]


def test_cvs_the_first_search_missed_are_searched_again_so_a_question_about_everyone_covers_them(ws, monkeypatch):
    index = FakeIndex([[hit("a", 0.9)], [hit("b", 0.4)]])
    install(monkeypatch, index)
    found = retrieval.retrieve(ws, "q-cover", [], ["a", "b"], k=2)
    assert index.searches[1]["file_ids"] == ["b"]
    assert {chunk["file_name"] for chunk in found.context} == {"a", "b"}


def test_no_hits_gives_no_context(ws, monkeypatch):
    install(monkeypatch, FakeIndex([[]]))
    found = retrieval.retrieve(ws, "q-nothing", [], ["a"], k=2)
    assert found.context == [] and found.sources == []
