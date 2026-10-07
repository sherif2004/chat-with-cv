"""The chat list in the sidebar: New chat, search, chats grouped by age, and open / rename / delete."""
from datetime import datetime

import streamlit as st

from cv_chat import history
from cv_chat.ui.safe import esc
from cv_chat.workspace import Workspace

PAGE = 30  # chats shown at first, and added by each "Show more"
GROUPS = ("Today", "Yesterday", "Previous 7 days", "Older")


def group_by_age(chats: list[history.Conversation], now: datetime | None = None) -> list[tuple[str, list[history.Conversation]]]:
    """The chats under Today, Yesterday, Previous 7 days and Older (by the day they were last used), keeping their order."""
    now = now or datetime.now().astimezone()
    groups: dict[str, list[history.Conversation]] = {name: [] for name in GROUPS}
    for chat in chats:
        age = (now.date() - chat.updated_at.astimezone(now.tzinfo).date()).days
        groups["Today" if age <= 0 else "Yesterday" if age == 1 else "Previous 7 days" if age <= 7 else "Older"].append(chat)
    return [(name, items) for name, items in groups.items() if items]


def render(ws: Workspace) -> None:
    st.button("New chat", icon=":material/add_comment:", type="primary", width="stretch", on_click=new_chat)
    query = st.text_input(
        "Search chats", key="chat_search", placeholder="Search chats", icon=":material/search:", label_visibility="collapsed",
    ).strip()
    limit = st.session_state.get("chat_limit", PAGE)
    try:
        chats = history.search_conversations(ws.user_id, query, limit) if query else history.list_conversations(ws.user_id, limit)
    except Exception as error:
        st.caption(f"Could not load your chats: {esc(str(error).splitlines()[0][:100])}")
        return
    if not chats:
        st.caption("No chat matches that search." if len(query) >= 2 else "Your chats will show up here.")
        return
    current = st.session_state.get("conversation_id")
    for name, items in group_by_age(chats):
        st.caption(name)
        for chat in items:
            _row(ws, chat, chat.id == current)
    if len(chats) >= limit:
        st.button("Show more", icon=":material/expand_more:", width="stretch", type="tertiary", on_click=_show_more)


def _row(ws: Workspace, chat: history.Conversation, active: bool) -> None:
    title_column, menu_column = st.columns([6, 1], vertical_alignment="center", gap="xxsmall")
    title_column.button(
        chat.title, key=f"open_{chat.id}", width="stretch", on_click=open_chat, args=(ws, chat.id),
        type="secondary" if active else "tertiary", icon=":material/chat_bubble:" if active else None,
    )
    with menu_column.popover("", icon=":material/more_vert:", type="tertiary", width="content", key=f"chatmenu_{chat.id}"):
        st.text_input("Name", value=chat.title, key=f"name_{chat.id}", max_chars=history.TITLE_LENGTH)
        st.button("Rename", key=f"rename_{chat.id}", icon=":material/edit:", width="stretch", on_click=rename_chat, args=(ws, chat.id, f"name_{chat.id}"))
        st.divider()
        confirmed = st.checkbox("Yes, delete this chat", key=f"confirm_{chat.id}")
        st.button(
            "Delete chat", key=f"delete_{chat.id}", icon=":material/delete:", width="stretch", disabled=not confirmed,
            on_click=delete_chat, args=(ws, chat.id),
        )


def _show_more() -> None:
    st.session_state.chat_limit = st.session_state.get("chat_limit", PAGE) + PAGE


def new_chat() -> None:
    st.session_state.messages = []
    st.session_state.conversation_id = None
    st.session_state.shown_messages = None


def open_chat(ws: Workspace, conversation_id: str) -> None:
    messages = history.load_messages(ws.user_id, conversation_id)
    if messages is None:  # deleted elsewhere, or not this user's
        st.session_state.notice = ("That chat no longer exists", ":material/info:")
        return
    st.session_state.messages = messages
    st.session_state.conversation_id = conversation_id
    st.session_state.shown_messages = None


def rename_chat(ws: Workspace, conversation_id: str, name_key: str) -> None:
    if not history.rename(ws.user_id, conversation_id, st.session_state.get(name_key, "")):
        st.session_state.notice = ("Could not rename the chat", ":material/error:")


def delete_chat(ws: Workspace, conversation_id: str) -> None:
    history.delete(ws.user_id, conversation_id)
    if st.session_state.get("conversation_id") == conversation_id:
        new_chat()
