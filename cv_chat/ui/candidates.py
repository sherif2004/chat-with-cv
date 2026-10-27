"""Candidates view: a sortable table with the name, title and location read from each CV."""
import pandas as pd
import streamlit as st

from cv_chat.services import blob_storage, search_index
from cv_chat.ui import states
from cv_chat.workspace import Workspace

SORTS = {"Name": lambda c: (c["name"] or c["file_name"]).lower(), "Job title": lambda c: (c["title"] or "~").lower(), "Location": lambda c: (c["location"] or "~").lower()}
COLUMNS = 3
MAX_COMPARE = 3  # candidates that can be compared side by side


@st.cache_data(ttl=30, show_spinner=False)
def _profiles(ws: Workspace, cvs: tuple[str, ...]) -> dict[str, dict]:
    """The metadata of every CV by file name. `cvs` is part of the cache key, so adding or deleting a CV refreshes it."""
    return search_index.list_profiles(ws)


def candidate_list(ws: Workspace, cvs: list[str]) -> list[dict]:
    try:
        profiles = _profiles(ws, tuple(cvs))
    except Exception:  # the CVs are still listed, just without their details
        profiles = {}
    found = []
    for file_name in cvs:
        profile = profiles.get(file_name) or {}
        found.append({
            "file_name": file_name, "name": profile.get("candidate_name") or "", "title": profile.get("job_title") or "",
            "location": profile.get("location") or "", "email": profile.get("email") or "",
        })
    return found


def render(ws: Workspace, cvs: list[str]) -> None:
    if not cvs:
        states.empty("No CVs yet. Add some in the Library and they will show up here.", "Open the Library", _open_library, icon=":material/upload_file:")
        return
    candidates = candidate_list(ws, cvs)
    text = st.text_input("Filter", placeholder="Filter by name, title, location or file", label_visibility="collapsed", key="candidate_filter").strip().lower()
    shown = [
        c for c in candidates
        if not text or text in " ".join([c["name"], c["title"], c["location"], c["file_name"]]).lower()
    ]
    shown.sort(key=SORTS["Name"])
    if not shown:
        states.empty("No candidate matches this filter.", "Clear the filter", _clear_filters, icon=":material/search_off:")
        return

    bar = st.container()  # drawn above the table, filled once the table says what is selected
    frame = pd.DataFrame({
        "Name": [c["name"] or c["file_name"] for c in shown],
        "Job title": [c["title"] for c in shown],
        "Location": [c["location"] for c in shown],
        "Email": [c["email"] for c in shown],
        "CV": [_link(ws, c["file_name"]) for c in shown],
    })
    event = st.dataframe(  # one widget for every row (a card each made big lists slow); click a header to sort
        frame, hide_index=True, width="stretch", on_select="rerun", selection_mode="multi-row", key=f"candidates_table_{st.session_state.get('candidates_reset', 0)}",
        column_config={"CV": st.column_config.LinkColumn("CV", display_text="Open", help="Open the original file")},
    )
    chosen = [shown[i] for i in event.selection.rows if i < len(shown)]
    with bar:
        count_column, chat_column = st.columns([3, 2], vertical_alignment="center")
        count_column.caption(
            f"{len(chosen)} selected of {len(shown)}. Click a header to sort."
            if chosen else f"{len(shown)} of {len(candidates)} candidates. Select rows to chat with them or compare them."
        )
        chat_column.button(
            "Chat with selected", icon=":material/chat:", type="primary", width="stretch", disabled=not chosen,
            on_click=_chat_with, args=([c["file_name"] for c in chosen],), help="Answer only from the selected CVs",
        )
    if 2 <= len(chosen) <= MAX_COMPARE:
        _compare(chosen)
    elif len(chosen) > MAX_COMPARE:
        st.caption(f"Select up to {MAX_COMPARE} candidates to compare them side by side.")


def _compare(chosen: list[dict]) -> None:
    """The ticked candidates side by side, with buttons to clear the choice or ask the chat to compare them."""
    labels = [c["name"] or c["file_name"] for c in chosen]
    frame = pd.DataFrame(
        {
            label: [c["title"] or "-", c["location"] or "-", c["email"] or "-", c["file_name"]]
            for label, c in zip(labels, chosen)
        },
        index=["Job title", "Location", "Email", "File"],
    )
    with st.container(border=True):
        st.markdown(f"**Comparing {len(chosen)} candidates**")
        st.dataframe(frame, width="stretch")
        ask_column, clear_column = st.columns(2)
        ask_column.button(
            "Ask the chat to compare them", icon=":material/chat:", type="primary", width="stretch", on_click=_ask_to_compare,
            args=([c["file_name"] for c in chosen], labels),
        )
        clear_column.button("Clear the comparison", icon=":material/close:", width="stretch", on_click=_clear_comparison)


def _ask_to_compare(file_names: list[str], labels: list[str]) -> None:
    """Limit the chat to these CVs and start a comparison question."""
    st.session_state.chat_scope = list(file_names)
    st.session_state.scope_saved = list(file_names)
    st.session_state.suggested_question = "Compare these candidates: " + ", ".join(labels)
    st.session_state.view = "Chat"
    _clear_comparison()


def _clear_comparison() -> None:
    st.session_state.candidates_reset = st.session_state.get("candidates_reset", 0) + 1  # a new table key starts with nothing selected


def _clear_filters() -> None:
    st.session_state["candidate_filter"] = ""


def _open_library() -> None:
    st.session_state.view = "Library"


def _link(ws: Workspace, file_name: str) -> str | None:
    try:
        return blob_storage.read_link(ws, file_name)
    except Exception:  # a missing link must never break the page
        return None


def _chat_with(file_names: list[str]) -> None:
    """Limit the chat to these CVs and switch to it."""
    st.session_state.chat_scope = list(file_names)
    st.session_state.scope_saved = list(file_names)
    st.session_state.view = "Chat"
