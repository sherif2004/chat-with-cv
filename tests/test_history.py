import pytest

from cv_chat import auth, db, history


@pytest.fixture
def ada():
    return auth.sign_up("ada@example.com", "correct horse").id


@pytest.fixture
def bob():
    return auth.sign_up("bob@example.com", "correct horse").id


def _exchange():
    return [
        {"role": "user", "content": "Who knows Python?"},
        {"role": "assistant", "content": "Ada does.", "sources": [{"file_name": "ada.pdf", "page": 1, "content": "Python", "score": 0.5}],
         "route": "simple", "trace": {"events": [{"kind": "search", "ms": 12.5}]}},
    ]


def test_titles():
    assert history.make_title("  Who   knows\nPython? ") == "Who knows Python?"
    assert history.make_title("") == "New chat"
    long = history.make_title("x" * 200)
    assert len(long) == history.TITLE_LENGTH and long.endswith("…")


def test_a_saved_chat_comes_back_exactly(ada):
    chat = history.create_conversation(ada, "Who knows Python?")
    assert history.add_messages(ada, chat, _exchange())
    assert history.load_messages(ada, chat) == _exchange()


def test_chats_are_listed_newest_first(ada):
    first = history.create_conversation(ada, "first")
    second = history.create_conversation(ada, "second")
    history.add_messages(ada, first, _exchange())  # activity moves a chat to the top
    assert [c.id for c in history.list_conversations(ada)] == [first, second]


def test_another_user_cannot_read_change_or_delete_a_chat(ada, bob):
    chat = history.create_conversation(ada, "private")
    history.add_messages(ada, chat, _exchange())
    assert history.load_messages(bob, chat) is None
    assert history.add_messages(bob, chat, _exchange()) is False
    assert history.rename(bob, chat, "hacked") is False
    assert history.delete(bob, chat) is False
    assert history.list_conversations(bob) == []
    assert history.load_messages(ada, chat) == _exchange()  # untouched
    assert history.list_conversations(ada)[0].title == "private"


@pytest.mark.parametrize("bad", ["", "not-a-uuid", "1; DROP TABLE messages", "00000000-0000-0000-0000-000000000000"])
def test_bad_or_unknown_ids_look_like_missing_chats(ada, bad):
    assert history.load_messages(ada, bad) is None
    assert history.add_messages(ada, bad, _exchange()) is False
    assert history.rename(ada, bad, "x") is False
    assert history.delete(ada, bad) is False


def test_rename_and_delete(ada):
    chat = history.create_conversation(ada, "old")
    assert history.rename(ada, chat, "  new   name ")
    assert history.list_conversations(ada)[0].title == "new name"
    history.add_messages(ada, chat, _exchange())
    assert history.delete(ada, chat)
    assert history.list_conversations(ada) == [] and history.load_messages(ada, chat) is None
    with db.pool().connection() as conn:
        assert conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 0  # the messages went with it


def test_deleting_an_account_deletes_its_chats(ada, bob):
    mine, theirs = history.create_conversation(ada, "a"), history.create_conversation(bob, "b")
    history.add_messages(ada, mine, _exchange())
    auth.delete_user(ada)
    with db.pool().connection() as conn:
        assert conn.execute("SELECT count(*) FROM conversations").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 0
    assert history.list_conversations(bob)[0].id == theirs


def test_the_list_is_limited(ada):
    for i in range(5):
        history.create_conversation(ada, f"chat {i}")
    assert len(history.list_conversations(ada, limit=3)) == 3
