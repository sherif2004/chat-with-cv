"""Candidates view: one card per CV, with the name, title, years and location read from the CV."""
import pandas as pd
import streamlit as st

from cv_chat.services import blob_storage, search_index
from cv_chat.ui import states
from cv_chat.ui.safe import esc
from cv_chat.workspace import Workspace

SORTS = {"Name": lambda c: (c["name"] or c["file_name"]).lower(), "Most experience": lambda c: -(c["years"] or -1), "Least experience": lambda c: c["years"] if c["years"] is not None else 1e9}
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
            "years": profile.get("years_experience"), "location": profile.get("location") or "", "email": profile.get("email") or "",
        })
    return found


def render(ws: Workspace, cvs: list[str]) -> None:
    if not cvs:
        states.empty("No CVs yet. Add some in the Library and they will show up here as cards.", "Open the Library", _open_library, icon=":material/upload_file:")
        return
    candidates = candidate_list(ws, cvs)
    search_column, years_column, sort_column = st.columns([3, 2, 2])
    text = search_column.text_input("Filter", placeholder="Name, title, location or file", label_visibility="collapsed", key="candidate_filter").strip().lower()
    min_years = years_column.number_input("Minimum years", min_value=0, max_value=60, value=0, step=1, help="Minimum years of experience", key="candidate_min_years")
    sort = sort_column.selectbox("Sort", list(SORTS), label_visibility="collapsed")

    shown = [
        c for c in candidates
        if (not text or text in " ".join([c["name"], c["title"], c["location"], c["file_name"]]).lower())
        and (not min_years or (c["years"] is not None and c["years"] >= min_years))  # years that could not be read never match
    ]
    shown.sort(key=SORTS[sort])
    st.caption(f"{len(shown)} of {len(candidates)} candidates")
    chosen = [c for c in candidates if st.session_state.get(f"compare_{c['file_name']}")]
    if len(chosen) >= 2:
        _compare(chosen)
    elif chosen:
        st.caption(f"Tick **Compare** on at least one more candidate (up to {MAX_COMPARE}) to see them side by side.")
    if not shown:
        states.empty("No candidate matches these filters.", "Clear the filters", _clear_filters, icon=":material/search_off:")
        return
    for start in range(0, len(shown), COLUMNS):
        for column, candidate in zip(st.columns(COLUMNS), shown[start : start + COLUMNS]):
            with column:
                _card(ws, candidate, full=len(chosen) >= MAX_COMPARE)


def _card(ws: Workspace, c: dict, full: bool = False) -> None:
    with st.container(border=True):
        st.markdown(f"**{esc(c['name'] or c['file_name'])}**")
        if c["title"]:
            st.caption(esc(c["title"]))
        if c["years"] is not None:
            st.markdown(f":material/work: {c['years']:g} years")
        if c["location"]:
            st.markdown(f":material/location_on: {esc(c['location'])}")
        if c["email"]:
            st.caption(f":material/mail: {esc(c['email'])}")
        st.caption(f":material/description: {esc(c['file_name'])}")
        link = _link(ws, c["file_name"])
        if link:
            st.link_button("Open CV", link, icon=":material/open_in_new:", width="stretch")
        key = f"compare_{c['file_name']}"
        st.checkbox("Compare", key=key, disabled=full and not st.session_state.get(key), help=f"Tick up to {MAX_COMPARE} candidates to compare them side by side")
        st.button(
            "Chat with this CV", key=f"chat_with_{c['file_name']}", icon=":material/chat:", width="stretch",
            help="Answer only from this CV", on_click=_chat_with, args=(c["file_name"],),
        )


def _compare(chosen: list[dict]) -> None:
    """The ticked candidates side by side, with buttons to clear the choice or ask the chat to compare them."""
    labels = [c["name"] or c["file_name"] for c in chosen]
    frame = pd.DataFrame(
        {
            label: [c["title"] or "-", "-" if c["years"] is None else f"{c['years']:g}", c["location"] or "-", c["email"] or "-", c["file_name"]]
            for label, c in zip(labels, chosen)
        },
        index=["Job title", "Years of experience", "Location", "Email", "File"],
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
    for key in [k for k in st.session_state if str(k).startswith("compare_")]:
        st.session_state[key] = False


def _clear_filters() -> None:
    st.session_state["candidate_filter"] = ""
    st.session_state["candidate_min_years"] = 0


def _open_library() -> None:
    st.session_state.view = "Library"


def _link(ws: Workspace, file_name: str) -> str | None:
    try:
        return blob_storage.read_link(ws, file_name)
    except Exception:  # a missing link must never break the page
        return None


def _chat_with(file_name: str) -> None:
    """Limit the chat to this CV and switch to it."""
    st.session_state.chat_scope = [file_name]
    st.session_state.scope_saved = [file_name]
    st.session_state.view = "Chat"
