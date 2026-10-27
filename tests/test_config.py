import pytest
from pydantic import ValidationError

from cv_chat import config
from cv_chat.settings_error import describe
from tests.conftest import AZURE_ENV


@pytest.fixture
def azure_env(monkeypatch):
    for name, value in AZURE_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("AZURE_OPENAI_SMALL_DEPLOYMENT", raising=False)


def test_loads_defaults(azure_env):
    settings = config.load(env_files=())
    assert settings.retrieval.top_k == 20
    assert settings.chunking.max_chars == 1500
    assert settings.chunking.overlap == 300
    assert settings.azure.small_deployment == ""


def test_missing_azure_values_are_all_reported(azure_env, monkeypatch):
    monkeypatch.delenv("AZURE_SEARCH_KEY")
    monkeypatch.delenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    with pytest.raises(ValidationError) as caught:
        config.load(env_files=())
    message = describe(caught.value)
    assert "AZURE_SEARCH_KEY" in message and "AZURE_OPENAI_CHAT_DEPLOYMENT" in message


def test_empty_value_counts_as_missing(azure_env, monkeypatch):
    monkeypatch.setenv("AZURE_SEARCH_KEY", "   ")
    with pytest.raises(ValidationError):
        config.load(env_files=())


def test_openai_endpoint_keeps_only_the_resource_address(azure_env, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://r.openai.azure.com/openai/v1/")
    assert config.load(env_files=()).azure.openai_endpoint == "https://r.openai.azure.com"


def test_secrets_are_not_printed(azure_env):
    assert "search-key" not in repr(config.load(env_files=()))


def test_overlap_must_be_smaller_than_the_chunk(azure_env, monkeypatch):
    monkeypatch.setenv("CVCHAT_CHUNKING__MAX_CHARS", "500")
    monkeypatch.setenv("CVCHAT_CHUNKING__OVERLAP", "500")
    with pytest.raises(ValidationError):
        config.load(env_files=())


def test_search_k_cannot_pass_the_semantic_ranker_limit(azure_env, monkeypatch):
    monkeypatch.setenv("CVCHAT_RETRIEVAL__SEARCH_K", "51")
    with pytest.raises(ValidationError):
        config.load(env_files=())


def test_a_tunable_can_be_overridden_from_the_environment(azure_env, monkeypatch):
    monkeypatch.setenv("CVCHAT_RETRIEVAL__TOP_K", "7")
    assert config.load(env_files=()).retrieval.top_k == 7
