"""All settings in one place: Azure values from .env, tunables as constants.

Missing .env values fail at import with one KeyError naming all of them.
"""
import os
import re

from dotenv import load_dotenv

load_dotenv()

_missing: list[str] = []


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        _missing.append(name)
    return value


# Azure Blob Storage
STORAGE_CONNECTION_STRING = _env("AZURE_STORAGE_CONNECTION_STRING")
STORAGE_CONTAINER = _env("AZURE_STORAGE_CONTAINER")

# Azure AI Search
SEARCH_ENDPOINT = _env("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = _env("AZURE_SEARCH_KEY")
SEARCH_INDEX = _env("AZURE_SEARCH_INDEX")

# Azure OpenAI
# Only the resource address is wanted: a pasted "/openai/v1" or trailing slash would give 404s.
OPENAI_ENDPOINT = re.split(r"/openai\b", _env("AZURE_OPENAI_ENDPOINT"))[0].rstrip("/")
OPENAI_API_KEY = _env("AZURE_OPENAI_API_KEY")
OPENAI_API_VERSION = _env("AZURE_OPENAI_API_VERSION")
EMBEDDING_DEPLOYMENT = _env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
CHAT_DEPLOYMENT = _env("AZURE_OPENAI_CHAT_DEPLOYMENT")

if _missing:
    raise KeyError(", ".join(_missing))

# Tunables
CHUNK_SIZE = 1500  # max characters per chunk; a section shorter than this stays one chunk
CHUNK_OVERLAP = CHUNK_SIZE // 5  # 20% overlap, used only when a long section is split
CONTEXT_CHARS = 36000  # the most CV text sent to the chat model for one question (about 9,000 tokens); excerpts are added by relevance until it is used
HISTORY_CHARS = 6000  # the most recent chat sent along with a question
RETRIEVE_K = 50  # chunks fetched and re-ranked per search, then grouped by candidate (50 is the semantic ranker's limit)
TOP_P = 0.8  # keep the most relevant candidates, and the most relevant excerpts of each, until they hold this share of the relevance
EXPANDED_QUERIES = 2  # alternative queries searched next to the main one when query expansion is on
RRF_K = 60  # reciprocal rank fusion constant: higher flattens the difference between ranks
EXTRACT_CACHE_DIR = ".cache/extracted"  # Docling output per file, so re-chunking skips the slow step
CACHE_SIZE = 256  # entries kept per kind of cached result (router, search, answer)
EMBED_CACHE_SIZE = 256  # question embeddings kept in memory
AGENT_MAX_ROUNDS = 5  # tool-using rounds the agent may take for one complex question
AGENT_MAX_SECONDS = 30  # time budget for those rounds; afterwards it answers from what it has found
NER_MODEL = "urchade/gliner_medium-v2.1"  # the generic NER model (GLiNER medium, Apache-2.0, 1.5 GB) that reads name, title, email, phone and location
