"""Main area: chat with the indexed CVs."""
from pathlib import Path

import streamlit as st

from cv_chat import config
from cv_chat.rag import qa
from cv_chat.ui import details

ROUTE_NOTES = {
    "chat": "Answered without searching the CVs",
    "simple": "Simple question · one search",
    "complex": "Complex question · answered by the search agent",
}
AVATARS = {"user": ":material/person:", "assistant": ":material/auto_awesome:"}
SUGGESTIONS = [
    "Who has strong Python experience?",
    "Compare cloud and DevOps skills",
    "Who has the most work experience?",
    "Summarize everyone's education",
]


def render(has_cvs: bool) -> None:
    st.html(Path(__file__).with_name("styles.css"))
    question = st.chat_input("Ask about skills, experience, education...", disabled=not has_cvs)
    question = question or st.session_state.pop("suggested_question", None)

    if not st.session_state.messages and not question:
        _welcome(has_cvs)
    for message in st.session_state.messages:
        _show(message)
    if question:
        _answer(question)


def _welcome(has_cvs: bool) -> None:
    st.html(
        '<div class="cv-hero">'
        '<span class="cv-hero-eyebrow">Azure AI Search · Azure OpenAI</span>'
        '<div class="cv-hero-title">Chat with CVs</div>'
        '<p class="cv-hero-subtitle">Ask anything about your candidates. '
        "Every answer comes from the uploaded CVs, with its sources.</p>"
        "</div>"
    )
    if not has_cvs:
        st.info(f"Process at least {config.MIN_CVS} CVs from the sidebar to start chatting.", icon=":material/info:")
        return
    columns = st.columns(2)
    for i, suggestion in enumerate(SUGGESTIONS):
        columns[i % 2].button(
            suggestion,
            icon=":material/arrow_outward:",
            width="stretch",
            on_click=_suggest,
            args=(suggestion,),
        )


def _suggest(question: str) -> None:
    st.session_state.suggested_question = question


def _show(message: dict) -> None:
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        st.markdown(message["content"])
        if message.get("trace"):
            details.render(message["trace"], message.get("sources", []))
        else:
            _route_note(message.get("route"))
        _sources(message.get("sources", []))


def _answer(question: str) -> None:
    history = list(st.session_state.messages)
    _show({"role": "user", "content": question})
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        try:
            with st.status("Reading your question...", expanded=True) as status:
                stream, sources, route, trace = qa.ask(
                    question, history, expand=st.session_state.get("expand_queries", False),
                    cache_answers=st.session_state.get("cache_answers", False), on_step=st.write
                )
                status.update(label="Done", state="complete", expanded=False)
            answer = st.write_stream(stream)
        except Exception as error:
            st.error(f"Could not answer: {str(error).splitlines()[0]}", icon=":material/error:")
            return
        record = trace.to_dict()  # complete now that the answer has been written
        details.render(record, sources)
        _sources(sources)
    st.session_state.messages += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer, "sources": sources, "route": route, "trace": record},
    ]


def _route_note(route: str | None) -> None:
    if route in ROUTE_NOTES:
        st.caption(f":material/alt_route: {ROUTE_NOTES[route]}")


def _sources(sources: list[dict]) -> None:
    """List the CVs an answer was based on, with a short excerpt from each."""
    if not sources:
        return
    excerpts: dict[str, list[str]] = {}
    for source in sources:
        excerpts.setdefault(source["file_name"], []).append(source.get("caption") or source["content"])
    with st.expander(f"Sources · {len(excerpts)} CV{'' if len(excerpts) == 1 else 's'}", icon=":material/menu_book:"):
        for file_name, contents in excerpts.items():
            st.markdown(f":material/description: **{file_name}** · {len(contents)} excerpt{'' if len(contents) == 1 else 's'}")
            snippet = " ".join(contents[0].split())
            st.caption(snippet[:240] + ("..." if len(snippet) > 240 else ""))
