import uuid

import pytest

from cv_chat.config import settings
from cv_chat.rag import qa, retrieval
from cv_chat.workspace import Workspace


@pytest.fixture
def ws():
    return Workspace(str(uuid.uuid4()))


def chunk(name: str, text: str) -> dict:
    return {"file_name": name, "section": "Skills", "page": 1, "content": text}


class Spy:
    def __init__(self):
        self.calls = []

    def retrieve(self, ws, query, sections, names, trace=None, k=None, scoped=False):
        self.calls.append({"query": query, "cvs": names, "k": k, "scoped": scoped})
        return retrieval.Retrieved([], [])


@pytest.fixture
def spy(monkeypatch):
    spy = Spy()
    monkeypatch.setattr(qa.retrieval, "retrieve", spy.retrieve)
    monkeypatch.setattr(qa.ingest, "list_cvs", lambda ws: ["a.pdf", "b.pdf", "c.pdf"])
    return spy


def read(ws, route, scope=()):
    return qa._read_cvs(ws, route, list(scope), lambda step: None, qa.Trace())


def test_a_question_about_everyone_reads_top_k_cvs(ws, spy):
    read(ws, qa.Route("simple", "compare skills", [], everyone=True))
    assert spy.calls[0]["k"] == settings.retrieval.top_k


def test_any_other_question_reads_top_k_focused_cvs(ws, spy):
    read(ws, qa.Route("simple", "Ali's phone", []))
    assert spy.calls[0]["k"] == settings.retrieval.top_k_focused


def test_the_question_is_searched_as_the_router_rewrote_it(ws, spy):
    read(ws, qa.Route("simple", "python developer in cairo", []))
    assert spy.calls[0]["query"] == "python developer in cairo"


def test_the_chosen_cvs_limit_what_can_be_read(ws, spy):
    read(ws, qa.Route("simple", "q", []), ["b.pdf"])
    assert spy.calls[0]["cvs"] == ["b.pdf"] and spy.calls[0]["scoped"] is True


def test_with_no_choice_every_cv_can_be_read_and_the_search_is_not_scoped(ws, spy):
    read(ws, qa.Route("simple", "q", []))
    assert spy.calls[0]["cvs"] == ["a.pdf", "b.pdf", "c.pdf"] and spy.calls[0]["scoped"] is False


def test_no_cvs_means_no_search(ws, spy, monkeypatch):
    monkeypatch.setattr(qa.ingest, "list_cvs", lambda ws: [])
    found = read(ws, qa.Route("simple", "q", []))
    assert found == retrieval.Retrieved([], []) and spy.calls == []


def test_the_content_filter_drops_the_flagged_excerpt_from_the_context_and_the_sources(monkeypatch):
    def stream(messages):
        if "BAD" in messages[-1]["content"]:
            raise qa.openai_service.ContentFilterError("refused")
        yield "answer"

    monkeypatch.setattr(qa.openai_service, "chat_stream", stream)
    chunks = [chunk("a.pdf", "fine"), chunk("b.pdf", "BAD text"), chunk("c.pdf", "also fine")]
    sources = [retrieval.shorten(dict(c, caption="")) for c in chunks]  # the Sources list holds shortened copies
    messages_for = lambda chosen: [{"role": "user", "content": retrieval.format_excerpts(chosen)}]

    text = "".join(qa._guarded(chunks, messages_for, None, sources))

    assert text.startswith("answer") and "b.pdf" in text  # the reply names the CV that was left out
    assert [c["file_name"] for c in chunks] == ["a.pdf", "c.pdf"]
    assert [s["file_name"] for s in sources] == ["a.pdf", "c.pdf"]


def test_there_is_no_query_expansion_left():
    assert not hasattr(qa, "_expand") and not hasattr(qa, "EXPAND_PROMPT")
    assert not hasattr(settings.retrieval, "expanded_queries")
