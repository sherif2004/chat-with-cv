import pytest

from cv_chat import auth, history


@pytest.fixture
def user(clean):
    return auth.sign_up("jane@example.com", "correct horse battery")


def chat_with(user, title, *texts):
    conversation = history.create_conversation(user.id, title)
    history.add_messages(user.id, conversation, [{"role": "user" if n % 2 == 0 else "assistant", "content": text} for n, text in enumerate(texts)])
    return conversation


def test_a_search_finds_chats_by_title_and_by_message_text(user):
    by_title = chat_with(user, "Python developers", "hello")
    by_text = chat_with(user, "Other", "who knows PYTHON well?")
    chat_with(user, "Unrelated", "nothing here")
    assert {c.id for c in history.search_conversations(user.id, "python")} == {by_title, by_text}


def test_percent_and_underscore_in_a_search_are_taken_literally(user):
    hundred = chat_with(user, "a", "I am 100% sure")
    chat_with(user, "b", "I am 1000 sure")
    chat_with(user, "c", "snake_case names")
    chat_with(user, "d", "snakeXcase names")
    assert [c.id for c in history.search_conversations(user.id, "100%")] == [hundred]
    assert len(history.search_conversations(user.id, "snake_case")) == 1


def test_a_search_never_reaches_another_users_chats(user):
    other = auth.sign_up("omar@example.com", "correct horse battery")
    chat_with(other, "Secret python plans", "python")
    assert history.search_conversations(user.id, "python") == []


def test_the_search_index_on_message_text_exists(database):
    with database.pool().connection() as conn:
        has_extension = conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'").fetchone()
        has_index = conn.execute("SELECT 1 FROM pg_indexes WHERE indexname = 'messages_content_trgm'").fetchone()
    if not has_extension:
        pytest.skip("this Postgres does not allow the pg_trgm extension")
    assert has_index


def test_starting_the_app_again_changes_nothing(database):
    database.init_schema()
    database.init_schema()


def test_the_app_still_starts_when_the_search_index_cannot_be_made(database, monkeypatch):
    monkeypatch.setattr(database, "SEARCH_INDEX_SQL", "CREATE EXTENSION this_extension_does_not_exist")
    database.init_schema()  # logs a warning and goes on
