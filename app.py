"""Chat with CVs: Streamlit entry point. Run with `uv run streamlit run app.py`."""
import streamlit as st

st.set_page_config(page_title="CV Chat", page_icon=":material/description:")

try:
    from cv_chat import config
    from cv_chat.ui import chat, sidebar
except KeyError as missing:  # config.py raises KeyError for a value missing from .env
    st.error(f"Missing {missing} in .env. Copy .env.example to .env and fill in your Azure values.", icon=":material/error:")
    st.stop()

st.session_state.setdefault("messages", [])
cvs = sidebar.render()
chat.render(has_cvs=len(cvs) >= config.MIN_CVS)
