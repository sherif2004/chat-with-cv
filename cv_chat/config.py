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
DOCLING_DEVICE = "cpu"  # "cuda" if a GPU is available
MIN_CVS = 8  # CVs needed in the knowledge base before processing and chatting are enabled
MAX_WORKERS = 4  # CVs processed in parallel
TOP_K = 10  # chunks sent to the chat model per question
RETRIEVE_K = 30  # candidates fetched and reranked before the per-CV cap is applied
MAX_CHUNKS_PER_CV = 2  # so one CV cannot fill all TOP_K slots of a broad question
EXPANDED_QUERIES = 2  # alternative queries searched next to the main one when query expansion is on
RRF_K = 60  # reciprocal rank fusion constant: higher flattens the difference between ranks
MIN_FILTERED_RESULTS = 3  # fewer section-filtered hits than this -> search again without the filter
HISTORY_MESSAGES = 6  # recent chat messages sent along with each question
EMBED_BATCH = 16  # chunks per embedding request
EMBED_CONCURRENCY = 2  # embedding requests in flight at once, across all parallel CVs
EXTRACT_CACHE_DIR = ".cache/extracted"  # Docling output per file, so re-chunking skips the slow step
CACHE_SIZE = 256  # entries kept per kind of cached result (router, search, answer)
EMBED_CACHE_SIZE = 256  # question embeddings kept in memory
AGENT_MAX_ROUNDS = 5  # tool-using rounds the agent may take for one complex question
AGENT_MAX_SECONDS = 30  # time budget for those rounds; afterwards it answers from what it has found
AGENT_CV_CHARS = 12000  # most characters of one CV that get_cv hands to the model
AGENT_SEARCH_K = 20  # most excerpts one search_cvs call hands to the model
METADATA_CHARS = 12000  # most characters of a CV read for its metadata (name, title, years, contact)
MAX_YEARS = 60  # a years-of-experience value above this is treated as wrong
