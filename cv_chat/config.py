"""All settings in one place, validated at start-up: Azure values from .env, tunables with sensible defaults.

Azure values keep their .env names (AZURE_SEARCH_KEY, ...). Every tunable can be overridden from the environment or .env as
CVCHAT_<GROUP>__<NAME>, for example CVCHAT_RETRIEVAL__TOP_K=10. Anything missing or out of range raises one ValidationError
that names all of it; settings_error.describe turns it into a message for the screen.
"""
import os
import re
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from cv_chat.config_files import ENV_FILES



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
    source_chars: int = Field(600, ge=100)  # a source (an excerpt in the Sources list, stored with the chat) is cut to this many characters
    search_k: int = Field(50, ge=1, le=50)  # chunks fetched and re-ranked per search (50 is the semantic ranker's limit)
    cover_rounds: int = Field(3, ge=0)  # extra searches aimed at the CVs the first search missed
    rrf_k: int = Field(60, ge=1)  # reciprocal rank fusion constant: higher flattens the difference between ranks
    embed_cache_size: int = Field(256, ge=0)  # question embeddings kept in memory


class ExtractionSettings(_Group):
    cache_dir: str = ".cache/extracted"  # Document Intelligence output per file, so re-chunking skips the service call
    version: int = Field(2, ge=1)  # bump when what is kept from a CV changes: cached output of the old rules is then ignored


class EntitiesSettings(_Group):
    model: str = "urchade/gliner_medium-v2.1"  # generic NER (GLiNER medium, Apache-2.0, 1.5 GB): name, title, email, phone, location
    labels: dict[str, str] = {  # each detail and the plain-language label the model is asked for (the wording matters: "city" finds places, "location" also finds universities)
        "candidate_name": "person name", "job_title": "job title", "email": "email address", "phone": "phone number", "location": "city",
    }
    threshold: float = Field(0.3, gt=0, le=1)  # the lowest confidence the model's answer may have
    window_words: int = Field(150, ge=20)  # the model reads about 384 tokens at a time
    max_value_chars: int = Field(100, ge=10)

    @model_validator(mode="after")
    def _known_labels(self) -> "EntitiesSettings":
        needed = {"candidate_name", "job_title", "email", "phone", "location"}
        if set(self.labels) != needed:
            raise ValueError(f"labels must be exactly: {', '.join(sorted(needed))}")
        return self


class ConcurrencySettings(_Group):
    embed_start: int = Field(4, ge=1)  # embedding requests in flight at first; halves on "too many requests", grows back to embed_max
    embed_max: int = Field(8, ge=1)
    interactive_reserved: int = Field(2, ge=0)  # embedding slots that ingesting CVs may not use, so a question never waits behind an upload
    chat_start: int = Field(6, ge=1)  # chat requests in flight at first (all users together); adapts like the embedding limit
    chat_max: int = Field(24, ge=1)
    per_user_chat: int = Field(3, ge=1)  # chat calls one user may have starting at once
    user_wait_seconds: float = Field(60, gt=0)  # after this long behind their own calls, a user's call goes on anyway
    rate_limit_retries: int = Field(8, ge=0)
    connection_retries: int = Field(4, ge=0)
    ingest_workers: int = Field(default_factory=lambda: max(2, min(8, os.cpu_count() or 2)), ge=1)  # CVs processed at once, all users together

    @model_validator(mode="after")
    def _start_within_max(self) -> "ConcurrencySettings":
        if self.embed_start > self.embed_max or self.chat_start > self.chat_max:
            raise ValueError("a start value cannot be above its max")
        return self


class IndexSettings(_Group):
    version: int = Field(3, ge=1)  # bump when the stored fields or keys change: every user's index is then rebuilt from the stored CVs
    analyzer: str = "en.microsoft"  # English stemming and stop words for keyword search ("built" also finds "building")
    semantic_config: str = "default"
    scoring_profile: str = "cv"
    text_weights: dict[str, float] = {"candidate_name": 3.0, "job_title": 2.0, "file_name": 2.0, "section": 1.5, "content": 1.0}
    section_boost: float = Field(1.5, gt=0)  # extra weight of the boosted sections (settings.sections.boosted)
    vector_metric: Literal["cosine", "dotProduct", "euclidean"] = "cosine"
    delete_batch: int = Field(500, ge=1, le=1000)  # chunks removed per request


class IngestSettings(_Group):
    pipeline_version: int = Field(2, ge=1)  # bump when processing logic changes: CVs processed by an older version are re-indexed
    max_name_bytes: int = Field(255, ge=1, le=255)  # a file name, in UTF-8; the search key made from it must stay under Azure's 1024 limit


class AgentSettings(_Group):
    max_rounds: int = Field(5, ge=1)  # tool-using rounds the agent may take for one complex question
    max_seconds: int = Field(30, ge=1)  # time budget for those rounds; afterwards it answers from what it has found
    per_cv_all: int = Field(1, ge=1)  # excerpts per CV from a search_cvs call over all CVs, so one search reaches many CVs
    per_cv_some: int = Field(2, ge=1)  # ... when the call names CVs
    per_cv_max: int = Field(5, ge=1)  # the most the model may ask for


class CacheSettings(_Group):
    size: int = Field(256, ge=1)  # entries kept per kind of cached result (router, search, answer)
    link_seconds: int = Field(600, ge=1, le=3000)  # a signed CV link is reused this long (links are signed for an hour)
    links: int = Field(2048, ge=1)  # signed links kept in memory
    cv_list_seconds: int = Field(60, ge=1)  # how long the list of a user's CVs is kept (adding or deleting a CV clears it at once)


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
    concurrency: ConcurrencySettings = ConcurrencySettings()
    index: IndexSettings = IndexSettings()
    ingest: IngestSettings = IngestSettings()
    agent: AgentSettings = AgentSettings()
    cache: CacheSettings = CacheSettings()


def load(env_files: tuple = ENV_FILES) -> Settings:
    """Read the environment and the .env files. Raises one ValidationError naming everything wrong or missing."""
    return Settings(_env_file=env_files, azure=AzureSettings(_env_file=env_files))


settings = load()
