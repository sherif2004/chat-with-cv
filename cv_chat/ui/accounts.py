"""Login and sign-up screen, and the cookie that keeps a user logged in across page reloads.

Streamlit can read cookies on the server (st.context.cookies) but cannot set them, so a tiny script in the page
writes the cookie. It is therefore not HttpOnly; it holds a random token whose hash is the only thing stored server-side.
"""
import json

import streamlit as st

from cv_chat import auth, db

COOKIE = "cv_session"


@st.cache_resource
def _prepare_database() -> None:
    db.init_schema()


def _provision(user: auth.User) -> None:
    """Create the new user's Azure container and search index. Anything half-made is removed if this fails."""
    from cv_chat.rag import ingest  # imported here: the login screen must work even when the Azure settings are wrong
    from cv_chat.workspace import Workspace

    ws = Workspace(user.id)
    try:
        ingest.prepare(ws)
    except Exception:
        try:
            ingest.destroy(ws)
        except Exception:  # the original error is the one worth showing
            pass
        raise


def _write_cookie(value: str, max_age: int) -> None:
    script = f"window.parent.document.cookie = {json.dumps(f'{COOKIE}={value}; path=/; max-age={max_age}; SameSite=Lax')};"
    st.iframe(f"<script>{script}</script>", height=1)  # a fixed string built here: no user text goes into it


def current_user() -> auth.User:
    """The logged-in user. Shows the login/sign-up screen and stops the page if there is none."""
    try:
        _prepare_database()
    except Exception as error:
        st.error(f"Could not reach Postgres: {error}. Start it with `docker compose up -d`.", icon=":material/error:")
        st.stop()
    if user := st.session_state.get("user"):
        return user
    cookie_token = st.context.cookies.get(COOKIE)
    if user := auth.user_for_token(cookie_token):
        st.session_state["user"], st.session_state["token"] = user, cookie_token
        return user
    screen = st.empty()
    with screen.container():
        token = _login_screen()
    if token is None:
        st.stop()
    screen.empty()
    _write_cookie(token, auth.SESSION_DAYS * 86400)  # no rerun: the script must stay on the page until the browser has run it
    st.session_state["user"], st.session_state["token"] = auth.user_for_token(token), token
    return st.session_state["user"]


def _login_screen() -> str | None:
    """The login and sign-up forms. Returns a session token once someone has logged in or signed up."""
    st.title("CV Chat")
    login_tab, signup_tab = st.tabs(["Log in", "Sign up"])
    with login_tab, st.form("login"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Log in", type="primary"):
            try:
                return auth.log_in(email, password)
            except auth.AuthError as error:
                st.error(str(error))
    with signup_tab, st.form("signup"):
        email = st.text_input("Email", key="signup_email")
        password = st.text_input(f"Password (at least {auth.MIN_PASSWORD} characters)", type="password", key="signup_password")
        if st.form_submit_button("Create account", type="primary"):
            try:
                auth.sign_up(email, password, on_created=_provision)
                return auth.log_in(email, password)
            except auth.AuthError as error:
                st.error(str(error))
    return None


def logout_button() -> None:
    st.sidebar.caption(st.session_state["user"].email)
    if st.sidebar.button("Log out", icon=":material/logout:"):
        auth.log_out(st.session_state.get("token"))
        _write_cookie("", 0)
        for key in list(st.session_state):
            del st.session_state[key]
        st.info("You are logged out. Reload the page to log in again.", icon=":material/logout:")
        st.stop()  # no rerun: the cookie script must stay on the page until the browser has run it
