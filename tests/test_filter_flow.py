import json
import uuid

import pytest

from cv_chat.processing.sections import SECTION_TYPES
from cv_chat.rag import agent, filters, qa, retrieval
from cv_chat.rag.cache import cache_for
from cv_chat.workspace import Workspace

PROFILES = {
    "ali.pdf": {"job_title": "Senior Data Engineer", "location": "Cairo, Egypt", "candidate_name": "Ali", "email": "", "phone": ""},
    "mona.pdf": {"job_title": "Data Analyst", "location": "Alexandria", "candidate_name": "Mona", "email": "", "phone": ""},
}


@pytest.fixture
def ws():
    return Workspace(str(uuid.uuid4()))


@pytest.fixture(autouse=True)
def profiles(monkeypatch):
    monkeypatch.setattr(filters.search_index, "list_profiles", lambda ws: PROFILES)


# --- the router -----------------------------------------------------------------------------------------------------------

def route_for(monkeypatch, ws, reply: dict):
    monkeypatch.setattr(qa.openai_service, "chat", lambda messages, json_mode=False, deployment=None: json.dumps(reply))
    return qa._route(cache_for(ws), f"question {uuid.uuid4()}", [], None)


BASE = {"route": "simple", "query": "q", "sections": [], "role": "", "everyone": False, "ask": ""}


def test_the_router_can_ask_for_a_location_and_for_sections_only(monkeypatch, ws):
    route = route_for(monkeypatch, ws, {**BASE, "sections": ["experience"], "role": "data engineer", "location": "Cairo", "section_only": True})
    assert (route.role, route.location, route.sections, route.section_only) == ("data engineer", "Cairo", ["experience"], True)


def test_sections_only_means_nothing_without_sections(monkeypatch, ws):
    assert route_for(monkeypatch, ws, {**BASE, "sections": [], "section_only": True}).section_only is False


def test_unknown_sections_are_dropped_and_then_sections_only_is_off(monkeypatch, ws):
    route = route_for(monkeypatch, ws, {**BASE, "sections": ["hobbies"], "section_only": True})
    assert route.sections == [] and route.section_only is False


def test_a_reply_without_the_new_fields_still_works(monkeypatch, ws):
    route = route_for(monkeypatch, ws, BASE)
    assert route.location == "" and route.section_only is False


def test_the_router_prompt_explains_the_new_fields():
    assert '"location"' in qa.ROUTER_PROMPT and '"section_only"' in qa.ROUTER_PROMPT


# --- narrowing the CVs before the search ----------------------------------------------------------------------------------

def narrow(ws, route, scope=()):
    return qa._filter_cvs(ws, route, list(scope), lambda step: None)


def test_with_no_job_or_location_the_chosen_cvs_are_left_alone(ws, monkeypatch):
    monkeypatch.setattr(filters.search_index, "list_profiles", lambda ws: pytest.fail("profiles were read for nothing"))
    assert narrow(ws, qa.Route("simple", "q", []), ["x.pdf"]) == (["x.pdf"], None)


def test_a_job_and_a_location_narrow_the_cvs(ws):
    assert narrow(ws, qa.Route("simple", "q", [], role="engineer", location="cairo")) == (["ali.pdf"], None)


def test_when_nothing_matches_the_answer_says_what_exists(ws):
    scope, message = narrow(ws, qa.Route("simple", "q", [], location="mars"))
    assert "mars" in message and "Alexandria" in message and "Cairo, Egypt" in message


def test_a_failing_check_never_blocks_the_answer(ws, monkeypatch):
    monkeypatch.setattr(filters.search_index, "list_profiles", lambda ws: (_ for _ in ()).throw(RuntimeError("search down")))
    assert narrow(ws, qa.Route("simple", "q", [], role="engineer"), ["x.pdf"]) == (["x.pdf"], None)


# --- the section filter in retrieval ---------------------------------------------------------------------------------------

def hit(name, chunk_no=0, section_type="experience"):
    return {"file_name": name, "file_url": "", "section": section_type.title(), "page": 1, "chunk_no": chunk_no, "content": f"{name} {section_type}", "caption": "", "score": 1.0}


class Index:
    def __init__(self, answers):
        self.answers, self.calls = list(answers), []

    def hybrid_search(self, ws, text, vector, k, boost_sections=None, file_ids=None, section_types=None):
        self.calls.append({"file_ids": file_ids, "section_types": section_types, "boost": boost_sections})
        return self.answers.pop(0) if self.answers else []


def install(monkeypatch, index):
    monkeypatch.setattr(retrieval.search_index, "hybrid_search", index.hybrid_search)
    monkeypatch.setattr(retrieval.openai_service, "embed", lambda texts: [[0.0] for _ in texts])


def test_only_sections_turns_the_router_sections_into_a_filter(ws, monkeypatch):
    index = Index([[hit("ali.pdf")]])
    install(monkeypatch, index)
    retrieval.retrieve(ws, "q-only-1", ["experience"], ["ali.pdf"], k=1, only_sections=True)
    assert index.calls[0]["section_types"] == ["experience"] and index.calls[0]["boost"] == ["experience"]


def test_without_only_sections_they_are_only_a_boost(ws, monkeypatch):
    index = Index([[hit("ali.pdf")]])
    install(monkeypatch, index)
    retrieval.retrieve(ws, "q-only-2", ["experience"], ["ali.pdf"], k=1)
    assert index.calls[0]["section_types"] is None


def test_if_the_section_filter_finds_nothing_the_search_is_repeated_without_it(ws, monkeypatch):
    index = Index([[], [hit("ali.pdf", section_type="projects")]])
    install(monkeypatch, index)
    found = retrieval.retrieve(ws, "q-only-3", ["experience"], ["ali.pdf"], k=1, only_sections=True)
    assert [c["file_name"] for c in found.context] == ["ali.pdf"]
    assert [call["section_types"] for call in index.calls] == [["experience"], None]


def test_the_cover_searches_keep_the_section_filter(ws, monkeypatch):
    index = Index([[hit("ali.pdf")], [hit("mona.pdf")]])
    install(monkeypatch, index)
    retrieval.retrieve(ws, "q-only-4", ["experience"], ["ali.pdf", "mona.pdf"], k=2, only_sections=True)
    assert index.calls[1]["file_ids"] == ["mona.pdf"] and index.calls[1]["section_types"] == ["experience"]


def test_read_cvs_passes_the_routers_choice_on(ws, monkeypatch):
    seen = {}
    monkeypatch.setattr(qa.ingest, "list_cvs", lambda ws: ["ali.pdf"])
    monkeypatch.setattr(qa.retrieval, "retrieve", lambda *args, **kwargs: seen.update(kwargs) or retrieval.Retrieved([], []))
    qa._read_cvs(ws, qa.Route("simple", "q", ["experience"], section_only=True), [], lambda step: None, qa.Trace())
    assert seen["only_sections"] is True


# --- the agent -------------------------------------------------------------------------------------------------------------

class Search:
    def __init__(self):
        self.calls = []

    def __call__(self, ws, query, sections, file_ids=None, only_sections=False, trace=None):
        self.calls.append({"query": query, "sections": sections, "file_ids": file_ids, "only_sections": only_sections})
        return [hit(name) for name in (file_ids or list(PROFILES))]


@pytest.fixture
def search(monkeypatch):
    search = Search()
    monkeypatch.setattr(agent.retrieval, "search", search)
    monkeypatch.setattr(agent.ingest, "list_cvs", lambda ws: list(PROFILES))
    return search


def run(ws, scope=()):
    return agent._Run(ws, lambda step: None, scope=list(scope))


def test_the_agent_filters_by_job_title_through_the_real_profiles(ws, search):
    text = run(ws).call("search_cvs", json.dumps({"query": "pipelines", "job_title": "data engineers"}))
    assert search.calls[0]["file_ids"] == ["ali.pdf"] and "1 CV matched" in text


def test_the_agent_filters_by_location(ws, search):
    run(ws).call("search_cvs", json.dumps({"query": "pipelines", "location": "alexandria"}))
    assert search.calls[0]["file_ids"] == ["mona.pdf"]


def test_the_agent_filters_by_section(ws, search):
    run(ws).call("search_cvs", json.dumps({"query": "compare", "sections": ["experience"]}))
    assert search.calls[0]["only_sections"] is True and search.calls[0]["sections"] == ["experience"]


def test_without_a_section_filter_the_search_is_not_restricted(ws, search):
    run(ws).call("search_cvs", json.dumps({"query": "compare"}))
    assert search.calls[0]["only_sections"] is False


def test_a_filter_that_matches_nothing_tells_the_agent_what_exists_and_does_not_search(ws, search):
    text = run(ws).call("search_cvs", json.dumps({"query": "x", "job_title": "astronaut"}))
    assert search.calls == [] and "astronaut" in text and "Data Analyst" in text and text.startswith("<cv_excerpt>")


def test_the_chosen_cvs_limit_what_a_filter_can_find(ws, search):
    text = run(ws, scope=["mona.pdf"]).call("search_cvs", json.dumps({"query": "x", "job_title": "data engineer"}))
    assert search.calls == [] and "No CV matches" in text


def test_unknown_section_types_are_refused(ws, search):
    text = run(ws).call("search_cvs", json.dumps({"query": "x", "sections": ["hobbies"]}))
    assert text.startswith("Error") and search.calls == []


def test_list_cvs_shows_the_titles_and_locations_a_filter_can_use(ws, search):
    text = run(ws).call("list_cvs", "{}")
    assert "Senior Data Engineer" in text and "Cairo, Egypt" in text


def test_the_tool_schema_offers_all_three_filters():
    properties = next(t["function"] for t in agent.TOOLS if t["function"]["name"] == "search_cvs")["parameters"]["properties"]
    assert {"cvs", "job_title", "location", "sections"} <= set(properties)
    assert properties["sections"]["items"]["enum"] == SECTION_TYPES
