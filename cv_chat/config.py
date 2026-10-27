"""All settings in one place, validated at start-up: Azure values from .env, tunables with sensible defaults.

Azure values keep their .env names (AZURE_SEARCH_KEY, ...). Every tunable can be overridden from the environment or .env as
CVCHAT_<GROUP>__<NAME>, for example CVCHAT_RETRIEVAL__TOP_K=10. Anything missing or out of range raises one ValidationError
that names all of it; settings_error.describe turns it into a message for the screen.
"""
import re
from pathlib import Path

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
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
    max_chars: int = Field(4000, ge=200)  # a section shorter than this stays one chunk
    overlap: int = Field(400, ge=0)  # used only when a long section is split
    caps_headings: bool = True  # also treat a short ALL-CAPS line as a heading, for CVs whose headings the layout model missed
    caps_max_words: int = Field(5, ge=1)
    heading_max_chars: int = Field(40, ge=1)  # a longer line is never a heading

    @model_validator(mode="after")
    def _overlap_is_smaller(self) -> "ChunkingSettings":
        if self.overlap >= self.max_chars:
            raise ValueError("overlap must be smaller than max_chars")
        return self


class SectionsSettings(_Group):
    # Standard section types and the heading keywords that make a heading one of them (see processing/sections.py).
    # The order matters: the first type with a matching keyword wins, so "Programming Languages" is skills, not languages.
    types: dict[str, list[str]] = {
        "experience": ["experience", "employment", "work history", "career", "professional background", "internship"],
        "education": ["education", "academic", "qualification", "degree", "universit"],
        "skills": ["skill", "technolog", "competenc", "tools", "expertise", "tech stack", "programming", "technical"],
        "projects": ["project"],
        "summary": ["summary", "profile", "objective", "about"],
        "certifications": ["certif", "licen", "award", "training"],
        "languages": ["language"],
        "contact": ["contact"],
    }
    boosted: list[str] = ["experience", "skills"]  # chunks from these sections rank a little higher for keyword matches

    @model_validator(mode="after")
    def _consistent(self) -> "SectionsSettings":
        if "other" in self.types:
            raise ValueError('"other" is the type of every unmatched heading; it cannot have keywords')
        if any(not words or any(not word.strip() or word != word.lower() for word in words) for words in self.types.values()):
            raise ValueError("every section type needs at least one lower-case keyword")
        if not set(self.boosted) <= set(self.types):
            raise ValueError("boosted sections must be section types")
        return self


class RetrievalSettings(_Group):
    top_k: int = Field(20, ge=1)  # CVs read whole for a question about everyone (a comparison, a ranking, a list)
    top_k_focused: int = Field(3, ge=1)  # CVs read whole for any other question
    history_chars: int = Field(6000, ge=0)  # the most recent chat sent along with a question
    search_k: int = Field(50, ge=1, le=50)  # chunks fetched and re-ranked per search (50 is the semantic ranker's limit)
    cover_rounds: int = Field(3, ge=0)  # extra searches aimed at the CVs the first search missed
    expanded_queries: int = Field(2, ge=0)  # alternative queries searched when query expansion is on
    rrf_k: int = Field(60, ge=1)  # reciprocal rank fusion constant: higher flattens the difference between ranks
    embed_cache_size: int = Field(256, ge=0)  # question embeddings kept in memory


class ExtractionSettings(_Group):
    cache_dir: str = ".cache/extracted"  # Document Intelligence output per file, so re-chunking skips the service call
    version: int = Field(2, ge=1)  # bump when what is kept from a CV changes: cached output of the old rules is then ignored


class EntitiesSettings(_Group):
    model: str = "urchade/gliner_medium-v2.1"  # generic NER (GLiNER medium, Apache-2.0, 1.5 GB): name, title, email, phone, location


class IngestSettings(_Group):
    max_name_bytes: int = Field(255, ge=1, le=255)  # a file name, in UTF-8; the search key made from it must stay under Azure's 1024 limit


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
    sections: SectionsSettings = SectionsSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    extraction: ExtractionSettings = ExtractionSettings()
    entities: EntitiesSettings = EntitiesSettings()
    ingest: IngestSettings = IngestSettings()
    agent: AgentSettings = AgentSettings()
    cache: CacheSettings = CacheSettings()


def load(env_files: tuple = ENV_FILES) -> Settings:
    """Read the environment and the .env files. Raises one ValidationError naming everything wrong or missing."""
    return Settings(_env_file=env_files, azure=AzureSettings(_env_file=env_files))


settings = load()
