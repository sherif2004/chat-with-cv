"""All settings in one place: Azure values from .env, tunables as constants.

A missing .env value fails at import with a KeyError naming the variable.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# Azure Blob Storage
STORAGE_CONNECTION_STRING = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
STORAGE_CONTAINER = os.environ["AZURE_STORAGE_CONTAINER"]

# Azure AI Search
SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]
SEARCH_INDEX = os.environ["AZURE_SEARCH_INDEX"]

# Azure OpenAI
OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
OPENAI_API_VERSION = os.environ["AZURE_OPENAI_API_VERSION"]
EMBEDDING_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
CHAT_DEPLOYMENT = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]

# Tunables
EMBEDDING_DIMENSIONS = 3072  # text-embedding-3-small (3072 for text-embedding-3-large)
CHUNK_SIZE = 1500  # max characters per chunk; a section shorter than this stays one chunk
CHUNK_OVERLAP = CHUNK_SIZE // 5  # 20% overlap, used only when a long section is split
DOCLING_DEVICE = "cpu"  # "cuda" if a GPU is available
MAX_WORKERS = 4  # CVs processed in parallel
TOP_K = 10  # chunks sent to the chat model per question
RETRIEVE_K = 30  # candidates fetched and reranked before the per-CV cap is applied
MAX_CHUNKS_PER_CV = 2  # so one CV cannot fill all TOP_K slots of a broad question
MIN_FILTERED_RESULTS = 3  # fewer section-filtered hits than this -> search again without the filter
HISTORY_MESSAGES = 6  # recent chat messages sent along with each question
EMBED_BATCH = 16  # chunks per embedding request
EMBED_CONCURRENCY = 2  # embedding requests in flight at once, across all parallel CVs
PIPELINE_VERSION = 2  # bump when extraction/chunking changes: CVs are then re-indexed even if the file is unchanged
EXTRACT_CACHE_DIR = ".cache/extracted"  # Docling output per file, so re-chunking skips the slow step
EMBED_CACHE_SIZE = 256  # question embeddings kept in memory
