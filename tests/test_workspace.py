import re

import pytest
from pydantic import ValidationError

from cv_chat import config
from cv_chat.workspace import Workspace
from tests.conftest import AZURE_ENV

CANONICAL = "abcdef01-2345-6789-abcd-ef0123456789"
AZURE_NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")  # what Azure accepts for a container or an index


@pytest.mark.parametrize("raw", [
    CANONICAL, CANONICAL.upper(), "{" + CANONICAL + "}", "urn:uuid:" + CANONICAL, CANONICAL.replace("-", ""),
])
def test_every_way_of_writing_a_uuid_gives_the_same_workspace(raw):
    workspace = Workspace(raw)
    assert workspace.user_id == CANONICAL and workspace == Workspace(CANONICAL)
    assert AZURE_NAME.match(workspace.container) and AZURE_NAME.match(workspace.index)


@pytest.mark.parametrize("raw", ["", "not-a-uuid", "../../etc", CANONICAL + "x", "abcdef01-2345-6789-abcd"])
def test_anything_that_is_not_a_uuid_is_refused(raw):
    with pytest.raises(ValueError):
        Workspace(raw)


def test_two_users_never_share_a_container_or_an_index():
    a, b = Workspace(CANONICAL), Workspace("00000000-0000-0000-0000-000000000001")
    assert a.container != b.container and a.index != b.index


@pytest.fixture
def azure_env(monkeypatch):
    for key, value in AZURE_ENV.items():
        monkeypatch.setenv(key, value)


@pytest.mark.parametrize("prefix", ["cvs", "my-cvs-2", "a", "a" * 30])
def test_a_good_container_prefix_is_accepted_and_lowercased(azure_env, monkeypatch, prefix):
    monkeypatch.setenv("AZURE_STORAGE_CONTAINER", prefix.upper())
    assert config.load(env_files=()).azure.storage_container == prefix


@pytest.mark.parametrize("prefix", ["my_cvs", "a" * 31, "-cvs", "cvs-", "cv  s", "cvs--x", "CVS.pdf", "cv/s"])
def test_a_container_prefix_azure_would_refuse_is_refused_at_start_up(azure_env, monkeypatch, prefix):
    monkeypatch.setenv("AZURE_STORAGE_CONTAINER", prefix)
    with pytest.raises(ValidationError):
        config.load(env_files=())


@pytest.mark.parametrize("prefix", ["cvs-index", "idx", "a" * 90])
def test_a_good_index_prefix_is_accepted(azure_env, monkeypatch, prefix):
    monkeypatch.setenv("AZURE_SEARCH_INDEX", prefix)
    assert config.load(env_files=()).azure.search_index == prefix


@pytest.mark.parametrize("prefix", ["-bad", "Has Space", "a" * 91, "under_score", "bad--dashes"])
def test_an_index_prefix_azure_would_refuse_is_refused_at_start_up(azure_env, monkeypatch, prefix):
    monkeypatch.setenv("AZURE_SEARCH_INDEX", prefix)
    with pytest.raises(ValidationError):
        config.load(env_files=())


def test_the_longest_allowed_prefixes_still_make_valid_azure_names(azure_env, monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_CONTAINER", "a" * 30)
    monkeypatch.setenv("AZURE_SEARCH_INDEX", "b" * 90)
    longest = config.load(env_files=()).azure
    suffix = 33  # "-" and the 32 hex digits of the user id
    assert len(longest.storage_container) + suffix <= 63 and len(longest.search_index) + suffix <= 128
