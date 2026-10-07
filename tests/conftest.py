"""Tests run against their own database (cvchat_test), never the one the app uses: they empty the tables."""
import os

import psycopg
import pytest

TEST_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://cvchat:cvchat@localhost:5432/cvchat_test")
os.environ["DATABASE_URL"] = TEST_URL  # must be set before cv_chat.db is imported

from cv_chat import db  # noqa: E402


def pytest_sessionstart(session):
    admin = TEST_URL.rsplit("/", 1)[0] + "/postgres"
    with psycopg.connect(admin, autocommit=True) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname = 'cvchat_test'").fetchone():
            conn.execute("CREATE DATABASE cvchat_test")


@pytest.fixture(autouse=True)
def clean_db():
    assert db.DATABASE_URL == TEST_URL and db.DATABASE_URL.endswith("_test")
    db.init_schema()
    with db.pool().connection() as conn:
        conn.execute("TRUNCATE sessions, users CASCADE")
    yield


@pytest.fixture(scope="session", autouse=True)
def close_pool():
    yield
    db.pool().close()
