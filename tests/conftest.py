"""Azure values for the test run: cv_chat.config reads them when it is first imported."""
import os

AZURE_ENV = {
    "AZURE_STORAGE_CONNECTION_STRING": "UseDevelopmentStorage=true",
    "AZURE_STORAGE_CONTAINER": "cvs",
    "AZURE_SEARCH_ENDPOINT": "https://search.example.net",
    "AZURE_SEARCH_KEY": "search-key",
    "AZURE_SEARCH_INDEX": "cvs-index",
    "AZURE_DOCINTEL_ENDPOINT": "https://docintel.example.net",
    "AZURE_DOCINTEL_KEY": "docintel-key",
    "AZURE_OPENAI_ENDPOINT": "https://openai.example.net",
    "AZURE_OPENAI_API_KEY": "openai-key",
    "AZURE_OPENAI_API_VERSION": "2024-10-21",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "embed",
    "AZURE_OPENAI_CHAT_DEPLOYMENT": "chat",
}

for name, value in AZURE_ENV.items():
    os.environ.setdefault(name, value)
