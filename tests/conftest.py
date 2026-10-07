"""Tests run against their own database (cvchat_test), never the one the app uses: they empty the tables."""
import importlib.util
import os
import sys
from unittest.mock import MagicMock

import psycopg
import pytest

# Dummy Azure settings (set before cv_chat.config is imported, and load_dotenv never overrides them), so no test can
# reach a real Azure account.
for name, value in {
    "AZURE_STORAGE_CONNECTION_STRING": "UseDevelopmentStorage=true", "AZURE_STORAGE_CONTAINER": "cvs",
    "AZURE_SEARCH_ENDPOINT": "https://test.search.windows.net", "AZURE_SEARCH_KEY": "test", "AZURE_SEARCH_INDEX": "cvs-index",
    "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com", "AZURE_OPENAI_API_KEY": "test", "AZURE_OPENAI_API_VERSION": "2024-10-21",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "test", "AZURE_OPENAI_CHAT_DEPLOYMENT": "test",
}.items():
    os.environ[name] = value
if importlib.util.find_spec("docling") is None:  # the isolation tests do not read CVs, so Docling can be a stub
    for module in ("docling", "docling.datamodel", "docling.datamodel.base_models", "docling.datamodel.pipeline_options",
                   "docling.document_converter", "docling_core", "docling_core.types", "docling_core.types.doc"):
        sys.modules[module] = MagicMock()

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
        conn.execute("TRUNCATE messages, conversations, sessions, users CASCADE")
    yield


@pytest.fixture(scope="session", autouse=True)
def close_pool():
    yield
    db.pool().close()
