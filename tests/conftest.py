"""Azure values for the test run: cv_chat.config reads them when it is first imported. Also a throwaway Postgres database."""
import os
import uuid

import pytest

AZURE_ENV = {
    "AZURE_STORAGE_CONNECTION_STRING": "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=Zm9v;EndpointSuffix=core.windows.net",
    "AZURE_STORAGE_CONTAINER": "cvs",
    "AZURE_SEARCH_ENDPOINT": "https://search.example.net",
    "AZURE_SEARCH_KEY": "search-key",
    "AZURE_SEARCH_INDEX": "cvs-index",
    "AZURE_DOCINTEL_ENDPOINT": "https://docintel.example.net",
    "AZURE_DOCINTEL_KEY": "docintel-key",
    "AZURE_OPENAI_ENDPOINT": "https://openai.example.net",
    "AZURE_OPENAI_API_KEY": "openai-key",
    "AZURE_OPENAI_API_VERSION": "2024-10-21",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "embed",
    "AZURE_OPENAI_CHAT_DEPLOYMENT": "chat",
}

for name, value in AZURE_ENV.items():
    os.environ.setdefault(name, value)


@pytest.fixture(scope="module")
def database():
    """A new empty database with the app's schema, dropped afterwards. Never touches the database named in DATABASE_URL.
    The tests that use it are skipped when no Postgres answers."""
    import psycopg
    from psycopg.conninfo import make_conninfo

    from cv_chat import db

    try:
        admin = psycopg.connect(make_conninfo(db.accounts.database_url, dbname="postgres"), autocommit=True, connect_timeout=3)
    except psycopg.OperationalError:
        pytest.skip("no Postgres to test against")
    name = f"cvchat_test_{uuid.uuid4().hex[:8]}"
    admin.execute(f'CREATE DATABASE "{name}"')
    original_settings, original_pool = db.accounts, db._pool
    db.accounts = original_settings.model_copy(update={"database_url": make_conninfo(original_settings.database_url, dbname=name)})
    db._pool = None
    try:
        db.init_schema()
        yield db
    finally:
        if db._pool is not None:
            db._pool.close()
        db.accounts, db._pool = original_settings, original_pool
        admin.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s", (name,))
        admin.execute(f'DROP DATABASE IF EXISTS "{name}"')
        admin.close()


@pytest.fixture
def clean(database):
    """The test database with no users, chats or login failures."""
    with database.pool().connection() as conn:
        conn.execute("TRUNCATE users, auth_failures RESTART IDENTITY CASCADE")
    return database
