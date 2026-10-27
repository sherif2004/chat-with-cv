import uuid

import pytest

from cv_chat.processing.sections import SECTION_TYPES
from cv_chat.rag import agent
from cv_chat.workspace import Workspace


def chunk(no, heading, kind, text, name="a.pdf"):
    return {"file_name": name, "file_url": "", "section": heading, "section_type": kind, "chunk_no": no, "page": 1,
            "content": f"{heading}\n{text}", "caption": ""}


CV = [chunk(0, "Other", "other", "Jane Doe"), chunk(1, "EXPERIENCE", "experience", "Built APIs at Acme"),
      chunk(2, "EDUCATION", "education", "BSc Computer Science"), chunk(3, "SKILLS", "skills", "Python, SQL")]


@pytest.fixture
def run(monkeypatch):
    monkeypatch.setattr(agent.search_index, "get_cv_chunks", lambda ws, name: CV if name == "a.pdf" else [])
    return agent._Run(Workspace(str(uuid.uuid4())), lambda step: None, scope=[])


def test_get_cv_does_not_read_the_whole_cv_by_default_it_lists_the_sections(run):
    text = run.call("get_cv", '{"file_name": "a.pdf"}')
    assert "EXPERIENCE" in text and "education" in text.lower()
    assert "Built APIs" not in text and "Python, SQL" not in text  # none of the CV's text
    assert run.sources == {}  # nothing was shown to the model as an excerpt


def test_get_cv_reads_only_the_sections_asked_for(run):
    text = run.call("get_cv", '{"file_name": "a.pdf", "sections": ["skills", "experience"]}')
    assert "Python, SQL" in text and "Built APIs" in text
    assert "BSc Computer Science" not in text and "Jane Doe" not in text


def test_only_what_was_shown_is_listed_as_a_source(run):
    run.call("get_cv", '{"file_name": "a.pdf", "sections": ["education"]}')
    assert [source["section"] for source in run.sources.values()] == ["EDUCATION"]


def test_unknown_section_types_are_refused_with_the_valid_ones(run):
    text = run.call("get_cv", '{"file_name": "a.pdf", "sections": ["hobbies"]}')
    assert text.startswith("Error") and "experience" in text


def test_a_cv_that_does_not_exist_is_an_error(run):
    assert run.call("get_cv", '{"file_name": "missing.pdf"}').startswith("Error")


def test_a_cv_outside_the_chosen_ones_cannot_be_read(monkeypatch):
    monkeypatch.setattr(agent.search_index, "get_cv_chunks", lambda ws, name: CV)
    scoped = agent._Run(Workspace(str(uuid.uuid4())), lambda step: None, scope=["b.pdf"])
    assert scoped.call("get_cv", '{"file_name": "a.pdf"}').startswith("Error")


def test_the_tool_schema_offers_sections_and_not_a_whole_cv():
    tool = next(t["function"] for t in agent.TOOLS if t["function"]["name"] == "get_cv")
    assert tool["parameters"]["required"] == ["file_name"]
    assert tool["parameters"]["properties"]["sections"]["items"]["enum"] == SECTION_TYPES
