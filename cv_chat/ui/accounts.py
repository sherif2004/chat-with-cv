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
    ended = False
    if user := st.session_state.get("user"):
        if auth.user_for_token(st.session_state.get("token")) == user:  # asked again on every page run, so a session ended elsewhere stops working here
            _recovery_gate()
            return user
        st.session_state.clear()
        ended = True
    cookie_token = st.context.cookies.get(COOKIE)
    if user := auth.user_for_token(cookie_token):
        st.session_state["user"], st.session_state["token"] = user, cookie_token
        return user
    screen = st.empty()
    with screen.container():
        if ended:
            st.warning("Your session has ended. Log in again.", icon=":material/lock:")
        token, new_code = _login_screen()
    if token is None:
        st.stop()
    screen.empty()
    _write_cookie(token, auth.SESSION_DAYS * 86400)  # no rerun: the script must stay on the page until the browser has run it
    st.session_state["user"], st.session_state["token"] = auth.user_for_token(token), token
    if new_code:
        st.session_state["recovery_code"] = new_code
    _recovery_gate()
    return st.session_state["user"]


def _recovery_gate() -> None:
    """Shows a new recovery code, once, until the user confirms they saved it. Only its hash is stored, so it cannot be shown again."""
    code = st.session_state.get("recovery_code")
    if not code:
        return
    _, middle, _ = st.columns([1, 2, 1])
    with middle:
        st.html('<div class="cv-brand"><div class="cv-brand-name">Save your recovery code</div></div>')
        with st.container(border=True):
            st.markdown("If you ever forget your password, this code lets you set a new one. **It is shown only once.**")
            st.code(code, language=None)
            st.caption("Keep it somewhere safe, such as a password manager. Anyone who has it can reset your password.")
            saved = st.checkbox("I have saved my recovery code")
            if st.button("Continue", type="primary", disabled=not saved, width="stretch"):
                del st.session_state["recovery_code"]
                st.rerun()
    st.stop()


def _login_screen() -> tuple[str | None, str | None]:
    """The login, sign-up and forgot-password forms. Returns (session token, new recovery code or None) once someone is in."""
    _, middle, _ = st.columns([1, 2, 1])
    with middle:
        st.html('<div class="cv-brand"><div class="cv-brand-name">CV Chat</div><div class="cv-brand-tag">Ask questions about your candidates</div></div>')
        with st.container(border=True):
            login_tab, signup_tab, forgot_tab = st.tabs(["Log in", "Sign up", "Forgot password"])
            with login_tab, st.form("login", border=False):
                email = st.text_input("Email", autocomplete="email")
                password = st.text_input("Password", type="password", autocomplete="current-password")
                if st.form_submit_button("Log in", type="primary", width="stretch"):
                    try:
                        return auth.log_in(email, password), None
                    except auth.AuthError as error:
                        st.error(str(error))
            with signup_tab, st.form("signup", border=False):
                email = st.text_input("Email", key="signup_email", autocomplete="email")
                password = st.text_input("Password", type="password", key="signup_password", autocomplete="new-password")
                st.caption(f"At least {auth.MIN_PASSWORD} characters. A few words in a row make a strong, easy-to-remember password.")
                if st.form_submit_button("Create account", type="primary", width="stretch"):
                    try:
                        user = auth.sign_up(email, password, on_created=_provision)
                        return auth.log_in(email, password), auth.issue_recovery_code(user.id)
                    except auth.AuthError as error:
                        st.error(str(error))
            with forgot_tab, st.form("forgot", border=False):
                st.caption("Use the recovery code you were given when you created your account.")
                email = st.text_input("Email", key="forgot_email", autocomplete="email")
                code = st.text_input("Recovery code", key="forgot_code", placeholder="XXXX-XXXX-XXXX-XXXX")
                password = st.text_input(f"New password (at least {auth.MIN_PASSWORD} characters)", type="password", key="forgot_password", autocomplete="new-password")
                if st.form_submit_button("Reset password", type="primary", width="stretch"):
                    try:
                        new_code = auth.reset_password(email, code, password)
                        return auth.log_in(email, password), new_code
                    except auth.AuthError as error:
                        st.error(str(error))
    return None, None


def account_menu(user: auth.User) -> None:
    """The account menu at the top of the sidebar: log out, change password, delete the account."""
    with st.sidebar, st.popover(user.email, icon=":material/account_circle:", width="stretch"):
        if st.button("Log out", icon=":material/logout:", width="stretch"):
            auth.log_out(st.session_state.get("token"))
            _end_session("You are logged out. Reload the page to log in again.")
        st.divider()
        st.markdown("**Chat settings**")
        st.toggle(
            "Query expansion", key="expand_queries",
            help="Also search two reworded versions of each question and merge the results. "
            "Finds more, but answers start 1 to 2 seconds later and each question uses three semantic searches.",
        )
        st.toggle(
            "Cache final answers", key="cache_answers",
            help="Reuse the answer to an identical question in an identical chat. Off by default: a cached answer can be out of date.",
        )
        st.button(
            "Clear cache", icon=":material/mop:", width="stretch", on_click=_clear_cache, args=(user,),
            help="Forget cached router results, searches and answers. This also happens whenever a CV is processed or deleted.",
        )
        st.divider()
        st.markdown("**Change password**")
        with st.form("change_password", clear_on_submit=True):
            current = st.text_input("Current password", type="password")
            new = st.text_input(f"New password (at least {auth.MIN_PASSWORD} characters)", type="password")
            if st.form_submit_button("Change password"):
                try:
                    auth.change_password(user.id, current, new, keep_token=st.session_state.get("token"))
                    st.toast("Password changed. You were logged out on your other devices.", icon=":material/task_alt:")
                except auth.AuthError as error:
                    st.error(str(error))
        st.divider()
        st.markdown("**Recovery code**")
        st.caption("A new code replaces the old one. Accounts made before recovery codes existed can get their first code here.")
        st.text_input("Your password", type="password", key="recovery_password", autocomplete="current-password")
        st.button("Get a new recovery code", icon=":material/key:", width="stretch", on_click=_new_recovery_code, args=(user,))
        st.divider()
        st.markdown("**Delete my account**")
        st.caption("This deletes your CVs, your search index, your chats and your account. It cannot be undone.")
        confirm = st.text_input("Type your email to confirm", key="delete_confirm")
        password = st.text_input("Your password", type="password", key="delete_password")
        if st.button("Delete my account", icon=":material/delete_forever:", type="primary", width="stretch",
                     disabled=confirm.strip().lower() != user.email or not password):
            if auth.check_password(user.id, password):
                _delete_account(user)
            else:
                st.error("Wrong password.")


def _new_recovery_code(user: auth.User) -> None:
    if not auth.check_password(user.id, st.session_state.get("recovery_password", "")):
        st.toast("Wrong password.", icon=":material/error:")
        return
    st.session_state["recovery_code"] = auth.issue_recovery_code(user.id)  # shown by _recovery_gate on the next run


def _clear_cache(user: auth.User) -> None:
    from cv_chat.rag.cache import cache_for  # imported here: the account menu must work even when the Azure settings are wrong
    from cv_chat.workspace import Workspace

    cache_for(Workspace(user.id)).clear()
    st.toast("Cache cleared", icon=":material/task_alt:")


def _delete_account(user: auth.User) -> None:
    """Azure data first: if that fails nothing else is removed, so the user can try again."""
    from cv_chat.rag import ingest, jobs  # imported here: the account menu must work even when the Azure settings are wrong
    from cv_chat.workspace import Workspace

    ws = Workspace(user.id)
    if jobs.queue_for(ws).active():
        st.error("CVs are still being processed. Wait until they finish, then delete your account.")
        return
    try:
        ingest.destroy(ws)
    except Exception as error:
        st.error(f"Could not delete your CVs and search index in Azure, so your account was not deleted: {str(error).splitlines()[0][:200]}")
        return
    auth.delete_user(user.id)  # also removes the sessions and the saved chats
    _end_session("Your account and all your data were deleted.")


def _end_session(message: str) -> None:
    """Forget the login in this browser. No rerun: the cookie script must stay on the page until the browser has run it."""
    _write_cookie("", 0)
    for key in list(st.session_state):
        del st.session_state[key]
    st.info(message, icon=":material/logout:")
    st.stop()
