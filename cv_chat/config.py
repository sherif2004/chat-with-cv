"""All settings in one place, validated at start-up: Azure values from .env, tunables with sensible defaults.

Azure values keep their .env names (AZURE_SEARCH_KEY, ...). Every tunable can be overridden from the environment or .env as
CVCHAT_<GROUP>__<NAME>, for example CVCHAT_RETRIEVAL__TOP_K=10. Anything missing or out of range raises one ValidationError
that names all of it; settings_error.describe turns it into a message for the screen.
"""
import re
from pathlib import Path

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
ENV_FILES = (".env", ROOT / ".env")


def _alias(name: str) -> AliasChoices:
    return AliasChoices(name)


class AzureSettings(BaseSettings):
    """The Azure resources. Deployment names, not model names, for Azure OpenAI."""

    model_config = SettingsConfigDict(env_file=ENV_FILES, extra="ignore", populate_by_name=True, str_strip_whitespace=True, frozen=True)

    storage_connection_string: SecretStr = Field(min_length=1, validation_alias=_alias("AZURE_STORAGE_CONNECTION_STRING"))
    storage_container: str = Field(min_length=1, validation_alias=_alias("AZURE_STORAGE_CONTAINER"))  # prefix: <prefix>-<user id>

    search_endpoint: str = Field(min_length=1, validation_alias=_alias("AZURE_SEARCH_ENDPOINT"))
    search_key: SecretStr = Field(min_length=1, validation_alias=_alias("AZURE_SEARCH_KEY"))
    search_index: str = Field(min_length=1, validation_alias=_alias("AZURE_SEARCH_INDEX"))  # prefix: <prefix>-<user id>

    docintel_endpoint: str = Field(min_length=1, validation_alias=_alias("AZURE_DOCINTEL_ENDPOINT"))
    docintel_key: SecretStr = Field(min_length=1, validation_alias=_alias("AZURE_DOCINTEL_KEY"))

    openai_endpoint: str = Field(min_length=1, validation_alias=_alias("AZURE_OPENAI_ENDPOINT"))
    openai_api_key: SecretStr = Field(min_length=1, validation_alias=_alias("AZURE_OPENAI_API_KEY"))
    openai_api_version: str = Field(min_length=1, validation_alias=_alias("AZURE_OPENAI_API_VERSION"))
    embedding_deployment: str = Field(min_length=1, validation_alias=_alias("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"))
    chat_deployment: str = Field(min_length=1, validation_alias=_alias("AZURE_OPENAI_CHAT_DEPLOYMENT"))
    # Optional: a smaller, cheaper deployment for small jobs such as naming a chat. Without it a chat is named after its first question.
    small_deployment: str = Field("", validation_alias=_alias("AZURE_OPENAI_SMALL_DEPLOYMENT"))

    @field_validator("openai_endpoint")
    @classmethod
    def _resource_address_only(cls, value: str) -> str:
        """A pasted "/openai/v1" or a trailing slash would give 404s."""
        return re.split(r"/openai\b", value)[0].rstrip("/")


class _Group(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ChunkingSettings(_Group):
    max_chars: int = Field(1500, ge=200)  # a section shorter than this stays one chunk
    overlap: int = Field(300, ge=0)  # used only when a long section is split

    @model_validator(mode="after")
    def _overlap_is_smaller(self) -> "ChunkingSettings":
        if self.overlap >= self.max_chars:
            raise ValueError("overlap must be smaller than max_chars")
        return self


class RetrievalSettings(_Group):
    top_k: int = Field(20, ge=1)  # most candidates a question about everyone covers
    top_p: float = Field(0.8, gt=0, le=1)  # share of the relevance the kept candidates and excerpts must hold
    context_chars: int = Field(36000, ge=1000)  # most CV text sent to the chat model for one question
    history_chars: int = Field(6000, ge=0)  # the most recent chat sent along with a question
    search_k: int = Field(50, ge=1, le=50)  # chunks fetched and re-ranked per search (50 is the semantic ranker's limit)
    cover_rounds: int = Field(3, ge=0)  # extra searches aimed at the CVs the first search missed
    expanded_queries: int = Field(2, ge=0)  # alternative queries searched when query expansion is on
    rrf_k: int = Field(60, ge=1)  # reciprocal rank fusion constant: higher flattens the difference between ranks
    embed_cache_size: int = Field(256, ge=0)  # question embeddings kept in memory


class ExtractionSettings(_Group):
    cache_dir: str = ".cache/extracted"  # Document Intelligence output per file, so re-chunking skips the service call


class EntitiesSettings(_Group):
    model: str = "urchade/gliner_medium-v2.1"  # generic NER (GLiNER medium, Apache-2.0, 1.5 GB): name, title, email, phone, location


class AgentSettings(_Group):
    max_rounds: int = Field(5, ge=1)  # tool-using rounds the agent may take for one complex question
    max_seconds: int = Field(30, ge=1)  # time budget for those rounds; afterwards it answers from what it has found


class CacheSettings(_Group):
    size: int = Field(256, ge=1)  # entries kept per kind of cached result (router, search, answer)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILES, env_prefix="CVCHAT_", env_nested_delimiter="__", extra="ignore", frozen=True,
    )

    azure: AzureSettings
    chunking: ChunkingSettings = ChunkingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    extraction: ExtractionSettings = ExtractionSettings()
    entities: EntitiesSettings = EntitiesSettings()
    agent: AgentSettings = AgentSettings()
    cache: CacheSettings = CacheSettings()


def load(env_files: tuple = ENV_FILES) -> Settings:
    """Read the environment and the .env files. Raises one ValidationError naming everything wrong or missing."""
    return Settings(_env_file=env_files, azure=AzureSettings(_env_file=env_files))


settings = load()
