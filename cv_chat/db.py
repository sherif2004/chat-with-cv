"""Postgres access: accounts, sessions, chat history and failed-login counts. Separate from config.py so it needs no Azure values."""
import logging

from psycopg_pool import ConnectionPool

from cv_chat.account_config import accounts

log = logging.getLogger(__name__)

SCHEMA = """
SELECT pg_advisory_xact_lock(2);  -- two processes starting together must not create the tables at the same time
CREATE TABLE IF NOT EXISTS users (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    email text NOT NULL UNIQUE,
    password_hash text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE users ADD COLUMN IF NOT EXISTS recovery_hash text;  -- SHA-256 of the one-time recovery code
CREATE TABLE IF NOT EXISTS sessions (
    token_hash text PRIMARY KEY,
    user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_user_idx ON sessions(user_id);
CREATE TABLE IF NOT EXISTS conversations (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS conversations_user_idx ON conversations(user_id, updated_at DESC);
CREATE TABLE IF NOT EXISTS messages (
    id bigserial PRIMARY KEY,
    conversation_id uuid NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role text NOT NULL CHECK (role IN ('user', 'assistant')),
    content text NOT NULL,
    sources jsonb,
    route text,
    trace jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_conversation_idx ON messages(conversation_id, id);
ALTER TABLE messages ADD COLUMN IF NOT EXISTS feedback smallint;  -- 1 thumbs up, -1 thumbs down
CREATE TABLE IF NOT EXISTS auth_failures (
    id bigserial PRIMARY KEY,
    key text NOT NULL,  -- login:<email>, reset:<email> or password:<user id>
    failed_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS auth_failures_key_idx ON auth_failures(key, failed_at DESC);
CREATE INDEX IF NOT EXISTS auth_failures_time_idx ON auth_failures(failed_at);
"""

# Searching chats is a substring search over every message; a trigram index makes it fast. It needs the pg_trgm extension, which
# some hosted databases do not allow: then the app works and the search just stays slower on big histories.
SEARCH_INDEX_SQL = """
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX IF NOT EXISTS messages_content_trgm ON messages USING gin (content gin_trgm_ops);
"""

_pool: ConnectionPool | None = None


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(accounts.database_url, min_size=1, max_size=accounts.pool_max, open=True)
    return _pool


def init_schema() -> None:
    with pool().connection() as conn:
        conn.execute(SCHEMA)
    try:
        with pool().connection() as conn:
            conn.execute(SEARCH_INDEX_SQL)
    except Exception as error:
        log.warning("no text-search index on chat messages (searching chats stays slower on big histories): %s", str(error).splitlines()[0])
