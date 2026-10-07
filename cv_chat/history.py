"""Saved chats: conversations and their messages, in Postgres.

Every function takes the user's id and puts it in the SQL, so a conversation id alone never opens, changes or deletes
anyone else's chat: a conversation that is not the user's looks exactly like one that does not exist.
"""
import uuid
from dataclasses import dataclass
from datetime import datetime

from psycopg.types.json import Jsonb

from cv_chat import db

TITLE_LENGTH = 60


@dataclass(frozen=True)
class Conversation:
    id: str
    title: str
    updated_at: datetime


def make_title(question: str) -> str:
    title = " ".join(question.split())
    return (title[: TITLE_LENGTH - 1].rstrip() + "…") if len(title) > TITLE_LENGTH else (title or "New chat")


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(str(value))
        return True
    except ValueError:
        return False


def create_conversation(user_id: str, title: str) -> str:
    with db.pool().connection() as conn:
        return str(conn.execute(
            "INSERT INTO conversations (user_id, title) VALUES (%s, %s) RETURNING id", (user_id, make_title(title))
        ).fetchone()[0])


def list_conversations(user_id: str, limit: int = 30) -> list[Conversation]:
    with db.pool().connection() as conn:
        rows = conn.execute(
            "SELECT id, title, updated_at FROM conversations WHERE user_id = %s ORDER BY updated_at DESC LIMIT %s",
            (user_id, limit),
        ).fetchall()
    return [Conversation(str(row[0]), row[1], row[2]) for row in rows]


def search_conversations(user_id: str, query: str, limit: int = 30) -> list[Conversation]:
    """The user's chats whose title or any message contains the text, newest first. Never reaches another user's chats."""
    text = " ".join(query.split())
    if len(text) < 2:
        return []
    pattern = "%" + text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"  # the text is matched literally
    with db.pool().connection() as conn:
        rows = conn.execute(
            "SELECT c.id, c.title, c.updated_at FROM conversations c WHERE c.user_id = %s AND (c.title ILIKE %s ESCAPE '\\' "
            "OR EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.id AND m.content ILIKE %s ESCAPE '\\')) "
            "ORDER BY c.updated_at DESC LIMIT %s",
            (user_id, pattern, pattern, limit),
        ).fetchall()
    return [Conversation(str(row[0]), row[1], row[2]) for row in rows]


def title(user_id: str, conversation_id: str) -> str | None:
    """The title of one of the user's chats, or None if there is no such chat of theirs."""
    if not _is_uuid(conversation_id):
        return None
    with db.pool().connection() as conn:
        row = conn.execute("SELECT title FROM conversations WHERE id = %s AND user_id = %s", (conversation_id, user_id)).fetchone()
    return row[0] if row else None


def load_messages(user_id: str, conversation_id: str) -> list[dict] | None:
    """The messages of one of the user's chats, or None if there is no such chat of theirs."""
    if not _is_uuid(conversation_id):
        return None
    with db.pool().connection() as conn:
        if not conn.execute(
            "SELECT 1 FROM conversations WHERE id = %s AND user_id = %s", (conversation_id, user_id)
        ).fetchone():
            return None
        rows = conn.execute(
            "SELECT id, role, content, sources, route, trace, feedback FROM messages WHERE conversation_id = %s ORDER BY id",
            (conversation_id,),
        ).fetchall()
    messages = []
    for message_id, role, content, sources, route, trace, feedback in rows:
        message = {"id": message_id, "role": role, "content": content}
        if role == "assistant":
            message.update(sources=sources or [], route=route, trace=trace, feedback=feedback)
        messages.append(message)
    return messages


def add_messages(user_id: str, conversation_id: str, messages: list[dict]) -> bool:
    """Append messages to one of the user's chats. False (and nothing saved) if it is not theirs. Each message dict gets its
    database id in "id", so it can be given feedback later."""
    if not _is_uuid(conversation_id):
        return False
    with db.pool().connection() as conn:
        if not conn.execute(
            "UPDATE conversations SET updated_at = now() WHERE id = %s AND user_id = %s RETURNING 1",
            (conversation_id, user_id),
        ).fetchone():
            return False
        for message in messages:
            message["id"] = conn.execute(
                "INSERT INTO messages (conversation_id, role, content, sources, route, trace) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                (
                    conversation_id, message["role"], message["content"],
                    Jsonb(message["sources"]) if message.get("sources") is not None else None,
                    message.get("route"),
                    Jsonb(message["trace"]) if message.get("trace") is not None else None,
                ),
            ).fetchone()[0]
    return True


def set_feedback(user_id: str, message_id: int, value: int | None) -> bool:
    """Thumbs up (1), thumbs down (-1) or none (None) on an answer in one of the user's chats."""
    if value not in (1, -1, None) or not isinstance(message_id, int):
        return False
    with db.pool().connection() as conn:
        return conn.execute(
            "UPDATE messages SET feedback = %s WHERE id = %s AND role = 'assistant' AND conversation_id IN "
            "(SELECT id FROM conversations WHERE user_id = %s) RETURNING 1",
            (value, message_id, user_id),
        ).fetchone() is not None


def rename(user_id: str, conversation_id: str, title: str) -> bool:
    if not _is_uuid(conversation_id):
        return False
    with db.pool().connection() as conn:
        return conn.execute(
            "UPDATE conversations SET title = %s WHERE id = %s AND user_id = %s RETURNING 1",
            (make_title(title), conversation_id, user_id),
        ).fetchone() is not None


def delete(user_id: str, conversation_id: str) -> bool:
    if not _is_uuid(conversation_id):
        return False
    with db.pool().connection() as conn:
        return conn.execute(
            "DELETE FROM conversations WHERE id = %s AND user_id = %s RETURNING 1", (conversation_id, user_id)
        ).fetchone() is not None
