"""Main area: chat with the indexed CVs."""
import streamlit as st

from cv_chat import history
from cv_chat.rag import qa
from cv_chat.services import blob_storage
from cv_chat.ui import candidates, chats, details, safe, states, suggestions
from cv_chat.workspace import Workspace

ROUTE_NOTES = {
    "chat": "Answered without searching the CVs",
    "simple": "Simple question · one search",
    "complex": "Complex question · answered by the search agent",
}
AVATARS = {"user": ":material/person:", "assistant": ":material/auto_awesome:"}


SHOWN = 20  # messages drawn at first in a long chat, and added by each "Load earlier messages"


def render(ws: Workspace, cvs: list[str], has_cvs: bool) -> None:
    question = st.chat_input("Ask about skills, experience, education...", disabled=not has_cvs)
    question = question or st.session_state.pop("suggested_question", None)

    _header(ws)
    _scope(cvs)
    messages = st.session_state.messages
    if not messages and not question:
        _welcome(ws, cvs, has_cvs)
    hidden = max(len(messages) - (st.session_state.get("shown_messages") or SHOWN), 0)
    if hidden:  # a long chat draws only its latest messages, so every click stays fast
        st.button(f"Load earlier messages ({hidden})", icon=":material/history:", type="tertiary", on_click=_load_earlier)
    for index in range(hidden, len(messages)):
        _show(ws, messages[index], index, last=index == len(messages) - 1)
    if question:
        _answer(ws, question)


def _scope(cvs: list[str]) -> None:
    """Which CVs the chat answers from. Nothing selected means all of them."""
    if not cvs:
        return
    if "chat_scope" not in st.session_state:  # Streamlit forgets a widget's value while another view is shown: restore it
        st.session_state.chat_scope = [name for name in st.session_state.get("scope_saved", []) if name in cvs]
    st.session_state.chat_scope = [name for name in st.session_state.chat_scope if name in cvs]  # drop deleted CVs
    chosen = st.session_state.chat_scope
    label = "All CVs" if not chosen else (safe.esc(chosen[0]) if len(chosen) == 1 else f"{len(chosen)} CVs")
    with st.popover(f"Chat with: {label}", icon=":material/filter_list:"):
        st.multiselect(
            "Chat with", cvs, key="chat_scope", placeholder="All CVs", label_visibility="collapsed",
            help="Pick one or more CVs to answer only from them. Leave empty to search all CVs.",
        )
    st.session_state.scope_saved = list(st.session_state.chat_scope)


def _load_earlier() -> None:
    st.session_state.shown_messages = (st.session_state.get("shown_messages") or SHOWN) + SHOWN


def _header(ws: Workspace) -> None:
    """The title of the open chat, with a way to rename it."""
    conversation_id = st.session_state.get("conversation_id")
    title = history.title(ws.user_id, conversation_id) if conversation_id else None
    if title is None:
        return
    title_column, rename_column = st.columns([12, 1], vertical_alignment="center")
    title_column.markdown(f"#### {safe.esc(title)}")
    with rename_column.popover("", icon=":material/edit:", width="content"):
        key = f"header_name_{conversation_id}"
        st.text_input("Chat name", value=title, key=key, max_chars=history.TITLE_LENGTH)
        st.button("Rename", key=f"header_rename_{conversation_id}", icon=":material/check:", width="stretch",
                  on_click=chats.rename_chat, args=(ws, conversation_id, key))


def _welcome(ws: Workspace, cvs: list[str], has_cvs: bool) -> None:
    st.html(
        '<div class="cv-hero">'
        '<span class="cv-hero-eyebrow">Azure AI Search · Azure OpenAI</span>'
        '<div class="cv-hero-title">Chat with CVs</div>'
        '<p class="cv-hero-subtitle">Ask anything about your candidates. '
        "Every answer comes from the uploaded CVs, with its sources.</p>"
        "</div>"
    )
    _checklist(ws, len(cvs))
    if not has_cvs:
        states.empty("Add your CVs in the Library to start chatting.", "Open the Library", _open_library, icon=":material/upload_file:")
        return
    columns = st.columns(2)
    for i, suggestion in enumerate(_starter_questions(ws, cvs)):
        columns[i % 2].button(
            safe.esc(suggestion), icon=":material/arrow_outward:", width="stretch", on_click=_suggest, args=(suggestion,), key=f"starter_{i}",
        )


def _starter_questions(ws: Workspace, cvs: list[str]) -> list[str]:
    """Questions built from these CVs' titles and places; the fixed ones if that is not possible."""
    try:
        return suggestions.build(candidates.candidate_list(ws, cvs))
    except Exception:  # starter questions are a nicety: never let them break the welcome screen
        return suggestions.FIXED


def _checklist(ws: Workspace, cv_count: int) -> None:
    """Three first steps, shown until the user has had a chat."""
    try:
        if history.list_conversations(ws.user_id, 1):
            return
    except Exception:
        return
    steps = [
        (cv_count >= 1, f"Add your CVs in the Library ({cv_count} so far)"),
        (False, "Ask your first question below"),
        (False, "Open **Sources** and **Details** under the answer to check it"),
    ]
    with st.container(border=True):
        st.markdown("**Getting started**")
        for done, text in steps:
            st.markdown((":green[:material/check_circle:]" if done else ":gray[:material/radio_button_unchecked:]") + f" {text}")


def _open_library() -> None:
    st.session_state.view = "Library"


def _suggest(question: str) -> None:
    st.session_state.suggested_question = question


def _show(ws: Workspace, message: dict, index: int | None = None, last: bool = False) -> None:
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        st.markdown(message["content"])
        if message["role"] != "assistant":
            return
        if message.get("sources"):
            _chips(ws, message["sources"])
        _actions(ws, message, index, last)
        if message.get("trace"):
            _details(f"details_{st.session_state.get('conversation_id')}_{index}", message["trace"], message.get("sources", []))
        else:
            _route_note(message.get("route"))
        _sources(ws, message.get("sources", []))


def _chips(ws: Workspace, sources: list[dict]) -> None:
    """The CVs an answer is based on, as links that open the original file."""
    names = list(dict.fromkeys(source["file_name"] for source in sources))
    shown = [f"[{safe.esc(name)}]({link})" if (link := _link(ws, name)) else safe.esc(name) for name in names[:6]]
    more = f" · +{len(names) - 6} more" if len(names) > 6 else ""
    st.caption(":material/description: " + " · ".join(shown) + more)


def _actions(ws: Workspace, message: dict, index: int | None, last: bool) -> None:
    """Thumbs up or down, copy, and (on the latest answer) regenerate."""
    message_id = message.get("id")
    feedback_column, copy_column, regenerate_column, _ = st.columns([1.2, 1, 2.4, 5], vertical_alignment="center", gap="xsmall")
    if message_id is not None:
        key = f"feedback_{message_id}"
        saved = message.get("feedback")
        with feedback_column:
            st.feedback("thumbs", key=key, default=None if saved is None else (1 if saved == 1 else 0), on_change=_feedback, args=(ws, message_id, key))
    with copy_column.popover("", icon=":material/content_copy:", width="content", help="Copy the answer"):
        st.code(message["content"], language=None, wrap_lines=True)
    if last:
        regenerate_column.button("Regenerate", icon=":material/refresh:", type="tertiary", on_click=_regenerate, help="Ask the same question again")


def _feedback(ws: Workspace, message_id: int, key: str) -> None:
    value = st.session_state.get(key)  # 0 is thumbs down, 1 thumbs up, None cleared
    history.set_feedback(ws.user_id, message_id, None if value is None else (1 if value == 1 else -1))


def _regenerate() -> None:
    """Ask the last question again. The answer cache is skipped once, so a repeated question really is answered again."""
    questions = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
    if questions:
        st.session_state.suggested_question = questions[-1]
        st.session_state.skip_cache_once = True


def _details(key: str, trace: dict, sources: list[dict]) -> None:
    """Details are built only when asked for: drawing the tabs of every answer on every click made long chats slow."""
    if st.toggle(details.title(trace), key=key):
        with st.container(border=True):
            details.render(trace, sources)


def _answer(ws: Workspace, question: str) -> None:
    history = list(st.session_state.messages)
    _show(ws, {"role": "user", "content": question})
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        try:
            with st.status("Reading your question...", expanded=True) as status:
                stream, sources, route, trace = qa.ask(
                    ws, question, history, expand=st.session_state.get("expand_queries", False),
                    cache_answers=st.session_state.get("cache_answers", False) and not st.session_state.pop("skip_cache_once", False), on_step=lambda step: st.write(safe.esc(step)),
                    scope=st.session_state.get("chat_scope", []),
                )
                status.update(label="Done", state="complete", expanded=False)
            answer = st.write_stream(safe.no_images(stream))
        except Exception as error:
            states.error(f"Could not answer: {str(error).splitlines()[0]}", "Try again", _retry, (question,))
            return
        record = trace.to_dict()  # complete now that the answer has been written
        _sources(ws, sources)
    exchange = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer, "sources": sources, "route": route, "trace": record},
    ]
    st.session_state.messages += exchange
    if _save(ws, question, exchange):
        st.rerun()  # the sidebar was drawn before this answer: redraw it so the chat list shows (and reorders) this chat


def _retry(question: str) -> None:
    st.session_state.suggested_question = question
    st.session_state.skip_cache_once = True


def _save(ws: Workspace, question: str, exchange: list[dict]) -> bool:
    """Keep the chat in Postgres. A failure here must not lose the answer the user is reading. True if it was saved."""
    try:
        if not st.session_state.get("conversation_id"):
            st.session_state.conversation_id = history.create_conversation(ws.user_id, question)
        return history.add_messages(ws.user_id, st.session_state.conversation_id, exchange)
    except Exception as error:
        st.warning(f"This answer could not be saved to your chat history: {str(error).splitlines()[0][:150]}", icon=":material/warning:")
        return False


def _route_note(route: str | None) -> None:
    if route in ROUTE_NOTES:
        st.caption(f":material/alt_route: {ROUTE_NOTES[route]}")


def _sources(ws: Workspace, sources: list[dict]) -> None:
    """One card per CV an answer was based on: who it is, the best excerpt and a link to the original file."""
    if not sources:
        return
    by_cv: dict[str, list[dict]] = {}
    for source in sources:
        by_cv.setdefault(source["file_name"], []).append(source)
    with st.expander(f"Sources · {len(by_cv)} CV{'' if len(by_cv) == 1 else 's'}", icon=":material/menu_book:"):
        for file_name, chunks in by_cv.items():
            first = chunks[0]
            with st.container(border=True):
                name_column, link_column = st.columns([4, 1], vertical_alignment="center")
                name_column.markdown(f"**{safe.esc(first.get('candidate_name') or file_name)}**")
                if link := _link(ws, file_name):
                    link_column.link_button("Open", link, icon=":material/open_in_new:", width="stretch")
                facts = [first.get("job_title"), file_name]
                st.caption(" · ".join(safe.esc(fact) for fact in facts if fact))
                for chunk in chunks[:2]:
                    snippet = " ".join((chunk.get("caption") or chunk["content"]).split())
                    page = f" (p.{chunk['page']})" if chunk.get("page") else ""
                    st.markdown(f"> {safe.esc(snippet[:220])}{'...' if len(snippet) > 220 else ''}{page}")


def _link(ws: Workspace, file_name: str) -> str | None:
    """A one-hour link to the user's own copy of the CV, or None if it cannot be made. Never built from index data."""
    try:
        return blob_storage.read_link(ws, file_name)
    except Exception:  # a missing link must never break the answer
        return None
