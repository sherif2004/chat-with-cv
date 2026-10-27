"""Settings of accounts and the Postgres database. Separate from config.py, which needs the Azure values: the login screen must
work (and say what is missing) before those are filled in.

Overridable from the environment or .env as CVCHAT_ACCOUNTS_<NAME>; the database address keeps its name, DATABASE_URL.
"""
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from cv_chat.config_files import ENV_FILES


class AccountSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILES, env_prefix="CVCHAT_ACCOUNTS_", extra="ignore", frozen=True)

    database_url: str = Field("postgresql://cvchat:cvchat@localhost:5432/cvchat", validation_alias=AliasChoices("DATABASE_URL"))  # tests point this at cvchat_test
    max_users: int = Field(15, ge=1)  # one Azure AI Search index per user, and the Basic tier allows 15
    min_password: int = Field(8, ge=8)  # it can be made stricter, never weaker than 8
    session_days: int = Field(14, ge=1)
    pool_max: int = Field(5, ge=1)  # Postgres connections kept


def load(env_files: tuple = ENV_FILES) -> AccountSettings:
    return AccountSettings(_env_file=env_files)


accounts = load()
