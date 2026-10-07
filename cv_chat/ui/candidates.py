"""Candidates view: one card per CV, with the name, title, years and location read from the CV."""
import streamlit as st

from cv_chat.services import blob_storage, search_index
from cv_chat.ui.safe import esc
from cv_chat.workspace import Workspace

SORTS = {"Name": lambda c: (c["name"] or c["file_name"]).lower(), "Most experience": lambda c: -(c["years"] or -1), "Least experience": lambda c: c["years"] if c["years"] is not None else 1e9}
COLUMNS = 3


@st.cache_data(ttl=30, show_spinner=False)
def _profiles(ws: Workspace, cvs: tuple[str, ...]) -> dict[str, dict]:
    """The metadata of every CV by file name. `cvs` is part of the cache key, so adding or deleting a CV refreshes it."""
    return search_index.list_profiles(ws)


def _candidates(ws: Workspace, cvs: list[str]) -> list[dict]:
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
        st.info("No CVs yet. Upload and process some in the sidebar.", icon=":material/info:")
        return
    candidates = _candidates(ws, cvs)
    search_column, years_column, sort_column = st.columns([3, 2, 2])
    text = search_column.text_input("Filter", placeholder="Name, title, location or file", label_visibility="collapsed").strip().lower()
    min_years = years_column.number_input("Minimum years", min_value=0, max_value=60, value=0, step=1, help="Minimum years")
    sort = sort_column.selectbox("Sort", list(SORTS), label_visibility="collapsed")

    shown = [
        c for c in candidates
        if (not text or text in " ".join([c["name"], c["title"], c["location"], c["file_name"]]).lower())
        and (not min_years or (c["years"] is not None and c["years"] >= min_years))  # years that could not be read never match
    ]
    shown.sort(key=SORTS[sort])
    st.caption(f"{len(shown)} of {len(candidates)} candidates")
    if not shown:
        st.info("No candidate matches these filters.", icon=":material/search_off:")
        return
    for start in range(0, len(shown), COLUMNS):
        for column, candidate in zip(st.columns(COLUMNS), shown[start : start + COLUMNS]):
            with column:
                _card(ws, candidate)


def _card(ws: Workspace, c: dict) -> None:
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
        st.button(
            "Chat with this CV", key=f"chat_with_{c['file_name']}", icon=":material/chat:", width="stretch",
            help="Answer only from this CV", on_click=_chat_with, args=(c["file_name"],),
        )


def _link(ws: Workspace, file_name: str) -> str | None:
    try:
        return blob_storage.read_link(ws, file_name)
    except Exception:  # a missing link must never break the page
        return None


def _chat_with(file_name: str) -> None:
    """Limit the chat to this CV and switch to it."""
    st.session_state.chat_scope = [file_name]
    st.session_state.view = "Chat"
