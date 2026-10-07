"""Chat with CVs: Streamlit entry point. Run with `uv run streamlit run app.py`."""
import streamlit as st

st.set_page_config(page_title="CV Chat", page_icon=":material/description:")

from cv_chat.ui import accounts

user = accounts.current_user()  # shows the login screen and stops the page until someone is logged in
accounts.logout_button()

try:
    from cv_chat import config
    from cv_chat.workspace import Workspace
    from cv_chat.rag import ingest
    from cv_chat.ui import chat, sidebar
except KeyError as missing:  # config.py raises KeyError for a value missing from .env
    st.error(f"Missing {missing} in .env. Copy .env.example to .env and fill in your Azure values.", icon=":material/error:")
    st.stop()
except Exception as error:  # for example a malformed storage connection string
    st.error(f"Could not connect to Azure: {error}. Check the values in .env.", icon=":material/error:")
    st.stop()



@st.cache_resource(show_spinner="Checking your storage and search index...")
def _prepare_workspace(ws: Workspace) -> None:
    ingest.prepare(ws)  # creates this user's container and index; an index made by an older version gets the new fields


ws = Workspace(user.id)  # built only from the logged-in user's id, so every call below reaches only their own data
try:
    _prepare_workspace(ws)
except Exception as error:  # for example an index built for a different embedding model
    st.error(f"Could not prepare the search index: {error}", icon=":material/error:")
    st.stop()

st.session_state.setdefault("messages", [])
cvs = sidebar.render(ws)
chat.render(ws, has_cvs=len(cvs) >= config.MIN_CVS)
