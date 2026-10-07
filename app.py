"""Chat with CVs: Streamlit entry point. Run with `streamlit run app.py`."""
import streamlit as st

st.set_page_config(page_title="CV Chat", page_icon=":material/description:", layout="wide")

from cv_chat.ui import accounts, style

style.inject()  # the stylesheet and its animations, also for the login page
user = accounts.current_user()  # shows the login screen and stops the page until someone is logged in
accounts.account_menu(user)

try:
    from cv_chat import config
    from cv_chat.workspace import Workspace
    from cv_chat.rag import ingest
    from cv_chat.ui import candidates, chat, library, sidebar
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
sidebar.render(ws)
cvs = library.cv_names(ws)
view = st.segmented_control("View", ["Chat", "Candidates", "Library"], key="view", default="Chat", label_visibility="collapsed")
if view == "Candidates":  # only the chosen view is drawn, so the other pages cost nothing
    candidates.render(ws, cvs)
elif view == "Library":
    library.render(ws, cvs)
else:
    style.inject(reading_column=True)
    chat.render(ws, cvs, has_cvs=len(cvs) >= config.MIN_CVS)
