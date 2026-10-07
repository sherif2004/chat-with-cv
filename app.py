"""Chat with CVs: Streamlit entry point. Run with `streamlit run app.py`."""
import streamlit as st

st.set_page_config(page_title="CV Chat", page_icon=":material/description:", layout="wide")

from cv_chat.ui import accounts, profile, states, style  # none of these needs the Azure settings, so the error screens below always work

profile.start()  # timing of this page run, only when the app was started with CV_PROFILE=1
with profile.phase("login check"):
    style.inject()  # the stylesheet and its animations, also for the login page
    user = accounts.current_user()  # shows the login screen and stops the page until someone is logged in
with profile.phase("account menu"):
    accounts.account_menu(user)

try:
    with profile.phase("imports"):
        from cv_chat import config
        from cv_chat.workspace import Workspace
        from cv_chat.rag import ingest
        from cv_chat.ui import candidates, chat, library, sidebar
except KeyError as missing:  # config.py raises KeyError for a value missing from .env
    states.error(f"Missing {missing} in .env. Copy .env.example to .env, fill in your Azure values and restart the app.", "Check again")
    st.stop()
except Exception as error:  # for example a malformed storage connection string
    states.error(f"Could not connect to Azure: {error}. Check the values in .env.", "Try again")
    st.stop()



@st.cache_resource(show_spinner="Checking your storage and search index...")
def _prepare_workspace(ws: Workspace) -> None:
    ingest.prepare(ws)  # creates this user's container and index; an index made by an older version gets the new fields


ws = Workspace(user.id)  # built only from the logged-in user's id, so every call below reaches only their own data
try:
    with profile.phase("prepare workspace"):
        _prepare_workspace(ws)
except Exception as error:  # for example an index built for a different embedding model
    states.error(f"Could not prepare your storage and search index: {error}", "Try again")
    st.stop()

st.session_state.setdefault("messages", [])
with profile.phase("sidebar"):
    sidebar.render(ws)
with profile.phase("cv list"):
    cvs = library.cv_names(ws)
if st.session_state.get("view") is None:  # first visit, or the user clicked the selected view again
    st.session_state["view"] = "Chat"
view = st.segmented_control("View", ["Chat", "Candidates", "Library"], key="view", label_visibility="collapsed")  # no default: buttons set the view through session state
with profile.phase(f"view: {view}"):
    if view == "Candidates":  # only the chosen view is drawn, so the other pages cost nothing
        candidates.render(ws, cvs)
    elif view == "Library":
        library.render(ws, cvs)
    else:
        style.inject(reading_column=True)
        chat.render(ws, cvs, has_cvs=len(cvs) >= config.MIN_CVS)
profile.finish()
