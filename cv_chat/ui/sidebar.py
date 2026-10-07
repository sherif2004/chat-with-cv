"""Sidebar: navigation only. The chat list, and a small progress bar while CVs are being processed."""
import streamlit as st

from cv_chat.ui import chats, library
from cv_chat.ui.safe import esc
from cv_chat.workspace import Workspace


def render(ws: Workspace) -> None:
    if notice := st.session_state.pop("notice", None):  # the result of an action that ran before this rerun
        st.toast(esc(notice[0]), icon=notice[1])
    with st.sidebar:
        chats.render(ws)
        library.watch(ws)
