"""The same look for every "nothing here" and "something went wrong" message: one sentence and one button for the next step."""
from collections.abc import Callable

import streamlit as st

from cv_chat.ui.safe import esc


def empty(message: str, button: str | None = None, on_click: Callable | None = None, args: tuple = (), icon: str = ":material/info:") -> None:
    st.info(message, icon=icon)
    if button and on_click:
        st.button(button, icon=":material/arrow_forward:", type="primary", on_click=on_click, args=args)


def error(message: str, retry: str | None = "Try again", on_click: Callable | None = None, args: tuple = ()) -> None:
    """An error with a retry button. Without on_click the button just reruns the page, which tries the failed step again."""
    st.error(esc(message) if "[" in message else message, icon=":material/error:")
    if retry:
        st.button(retry, icon=":material/refresh:", on_click=on_click, args=args)
