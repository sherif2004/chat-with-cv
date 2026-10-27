import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from cv_chat import account_config

ROOT = Path(__file__).resolve().parent.parent


def test_the_defaults():
    loaded = account_config.load(env_files=())
    assert (loaded.max_users, loaded.min_password, loaded.session_days) == (15, 8, 14)
    assert loaded.database_url.startswith("postgresql://")


def test_the_database_url_keeps_its_old_environment_name(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db:5432/other")
    assert account_config.load(env_files=()).database_url == "postgresql://u:p@db:5432/other"


def test_a_value_can_be_overridden_from_the_environment(monkeypatch):
    monkeypatch.setenv("CVCHAT_ACCOUNTS_MAX_USERS", "3")
    assert account_config.load(env_files=()).max_users == 3


@pytest.mark.parametrize("name, value", [("CVCHAT_ACCOUNTS_MAX_USERS", "0"), ("CVCHAT_ACCOUNTS_MIN_PASSWORD", "3"), ("CVCHAT_ACCOUNTS_SESSION_DAYS", "0")])
def test_values_out_of_range_are_refused(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        account_config.load(env_files=())


def test_it_loads_without_any_azure_value_so_the_login_screen_always_works():
    env = {"PATH": os.environ["PATH"], "PYTHONPATH": str(ROOT)}
    code = "import sys, cv_chat.auth, cv_chat.db; sys.exit(1 if 'cv_chat.config' in sys.modules else 0)"
    done = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
