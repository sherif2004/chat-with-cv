"""Azure OpenAI: embeddings for CV chunks and questions, and chat answers."""
from openai import AzureOpenAI

from cv_chat import config

_client = AzureOpenAI(
    azure_endpoint=config.OPENAI_ENDPOINT,
    api_key=config.OPENAI_API_KEY,
    api_version=config.OPENAI_API_VERSION,
    max_retries=5,  # the SDK backs off on 429s, which parallel ingestion can trigger
)


def embed(texts: list[str]) -> list[list[float]]:
    response = _client.embeddings.create(model=config.EMBEDDING_DEPLOYMENT, input=texts)
    return [item.embedding for item in response.data]


def chat(messages: list[dict]) -> str:
    response = _client.chat.completions.create(model=config.CHAT_DEPLOYMENT, messages=messages)
    return response.choices[0].message.content or ""
