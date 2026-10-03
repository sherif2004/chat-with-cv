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
EMBEDDING_DIMENSIONS = 1536  # text-embedding-3-small / ada-002 (3072 for text-embedding-3-large)
CHUNK_SIZE = 1500  # characters per chunk
CHUNK_OVERLAP = CHUNK_SIZE // 5  # 20% overlap between neighbouring chunks
MAX_WORKERS = 4  # CVs processed in parallel
