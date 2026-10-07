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
            "SELECT role, content, sources, route, trace FROM messages WHERE conversation_id = %s ORDER BY id",
            (conversation_id,),
        ).fetchall()
    messages = []
    for role, content, sources, route, trace in rows:
        message = {"role": role, "content": content}
        if role == "assistant":
            message.update(sources=sources or [], route=route, trace=trace)
        messages.append(message)
    return messages


def add_messages(user_id: str, conversation_id: str, messages: list[dict]) -> bool:
    """Append messages to one of the user's chats. False (and nothing saved) if it is not theirs."""
    if not _is_uuid(conversation_id):
        return False
    with db.pool().connection() as conn:
        if not conn.execute(
            "UPDATE conversations SET updated_at = now() WHERE id = %s AND user_id = %s RETURNING 1",
            (conversation_id, user_id),
        ).fetchone():
            return False
        for message in messages:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, sources, route, trace) VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    conversation_id, message["role"], message["content"],
                    Jsonb(message["sources"]) if message.get("sources") is not None else None,
                    message.get("route"),
                    Jsonb(message["trace"]) if message.get("trace") is not None else None,
                ),
            )
    return True


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
