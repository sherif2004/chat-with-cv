import pytest
from streamlit.testing.v1 import AppTest

from cv_chat.ui import accounts, style


# --- the session cookie ----------------------------------------------------------------------------------------------------

def test_the_cookie_is_marked_secure_when_the_app_is_served_over_https():
    assert accounts.cookie_text("abc", 100, https=True) == "cv_session=abc; path=/; max-age=100; SameSite=Lax; Secure"


def test_the_cookie_is_not_secure_over_plain_http_where_a_browser_would_refuse_to_keep_it():
    assert accounts.cookie_text("abc", 100, https=False) == "cv_session=abc; path=/; max-age=100; SameSite=Lax"


@pytest.mark.parametrize("url, expected", [("https://cv.example.com/", True), ("http://localhost:8501/", False), ("", False), (None, False)])
def test_https_is_read_from_the_address_the_browser_uses(monkeypatch, url, expected):
    monkeypatch.setattr(accounts.st, "context", type("Context", (), {"url": url})())
    assert accounts._served_over_https() is expected


def test_if_the_address_cannot_be_read_the_cookie_is_not_marked_secure(monkeypatch):
    class Broken:
        @property
        def url(self):
            raise RuntimeError("no script run")

    monkeypatch.setattr(accounts.st, "context", Broken())
    assert accounts._served_over_https() is False


# --- chat titles in the sidebar ----------------------------------------------------------------------------------------------

def sidebar_row():
    import uuid
    from datetime import datetime, timezone

    from cv_chat import history
    from cv_chat.ui import chats
    from cv_chat.workspace import Workspace

    title = "![x](https://attacker.example/?q=1) **bold** [link](https://a.example)"
    chats._row(Workspace(str(uuid.uuid4())), history.Conversation(str(uuid.uuid4()), title, datetime.now(timezone.utc)), False)


def test_a_chat_title_cannot_add_markdown_to_the_sidebar_button():
    at = AppTest.from_function(sidebar_row).run()
    assert not at.exception
    label = at.button[0].label
    assert "\\!\\[x\\]" in label and "\\*\\*bold\\*\\*" in label  # every Markdown character is escaped
    assert "![" not in label


# --- the stylesheet is sent once -----------------------------------------------------------------------------------------------

def test_the_stylesheet_and_the_reading_column_are_separate_so_the_chat_view_does_not_send_the_stylesheet_twice(monkeypatch):
    sent = []
    monkeypatch.setattr(style.st, "html", lambda body: sent.append(body))
    style.inject()
    style.reading_column()
    assert len(sent) == 2 and sent[0] == style._CSS and "max-width" in sent[1]
