"""Azure OpenAI: embeddings for CV chunks and questions, and chat answers."""
import threading
from collections.abc import Iterator

from openai import AzureOpenAI

from cv_chat import config

_client = AzureOpenAI(
    azure_endpoint=config.OPENAI_ENDPOINT,
    api_key=config.OPENAI_API_KEY,
    api_version=config.OPENAI_API_VERSION,
    max_retries=5,  # the SDK backs off on 429s, which parallel ingestion can trigger
)
_embed_gate = threading.Semaphore(config.EMBED_CONCURRENCY)  # shared by every parallel CV


def embed(texts: list[str]) -> list[list[float]]:
    """Embed in batches; the shared gate keeps parallel CVs from flooding the endpoint with requests."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), config.EMBED_BATCH):
        with _embed_gate:
            response = _client.embeddings.create(
                model=config.EMBEDDING_DEPLOYMENT, input=texts[start : start + config.EMBED_BATCH]
            )
        vectors.extend(item.embedding for item in response.data)
    return vectors


def chat(messages: list[dict], json_mode: bool = False) -> str:
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    response = _client.chat.completions.create(model=config.CHAT_DEPLOYMENT, messages=messages, **kwargs)
    return response.choices[0].message.content or ""


def chat_stream(messages: list[dict]) -> Iterator[str]:
    """Yield the answer piece by piece as the model writes it."""
    stream = _client.chat.completions.create(model=config.CHAT_DEPLOYMENT, messages=messages, stream=True)
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:  # some chunks carry only filter metadata
            yield chunk.choices[0].delta.content
