"""Azure OpenAI: embeddings for CV chunks and questions, and chat answers."""
import threading
from collections.abc import Callable, Iterator
from functools import lru_cache

from openai import AzureOpenAI, NotFoundError

from cv_chat import config

_client = AzureOpenAI(
    azure_endpoint=config.OPENAI_ENDPOINT,
    api_key=config.OPENAI_API_KEY,
    api_version=config.OPENAI_API_VERSION,
    max_retries=5,  # the SDK backs off on 429s, which parallel ingestion can trigger
)
_embed_gate = threading.Semaphore(config.EMBED_CONCURRENCY)  # shared by every parallel CV


def _explain_404(deployment: str, call: Callable):
    """Azure answers 404 for a wrong deployment name and for a wrong endpoint; say which things to check."""
    try:
        return call()
    except NotFoundError as error:
        raise RuntimeError(
            f"Azure OpenAI has no deployment named '{deployment}' at {config.OPENAI_ENDPOINT}. "
            "Use the deployment name (not the model name) exactly as in the portal, and check AZURE_OPENAI_ENDPOINT "
            "and AZURE_OPENAI_API_VERSION in .env."
        ) from error


@lru_cache(maxsize=1)
def embedding_dimensions() -> int:
    """Length of the embeddings the deployed model returns. The search index is created with this length."""
    return len(embed(["dimension probe"])[0])


def embed(texts: list[str]) -> list[list[float]]:
    """Embed in batches; the shared gate keeps parallel CVs from flooding the endpoint with requests."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), config.EMBED_BATCH):
        with _embed_gate:
            batch = texts[start : start + config.EMBED_BATCH]
            response = _explain_404(
                config.EMBEDDING_DEPLOYMENT,
                lambda: _client.embeddings.create(model=config.EMBEDDING_DEPLOYMENT, input=batch),
            )
        vectors.extend(item.embedding for item in response.data)
    return vectors


def chat(messages: list[dict], json_mode: bool = False) -> str:
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    response = _explain_404(
        config.CHAT_DEPLOYMENT,
        lambda: _client.chat.completions.create(model=config.CHAT_DEPLOYMENT, messages=messages, **kwargs),
    )
    return response.choices[0].message.content or ""


def chat_with_tools(messages: list[dict], tools: list[dict], timeout: float | None = None):
    """One turn of a tool-using conversation. Returns the model's message: its answer, or the tool calls it wants run."""
    kwargs = {"tools": tools} if tools else {}
    response = _explain_404(
        config.CHAT_DEPLOYMENT,
        lambda: _client.chat.completions.create(
            model=config.CHAT_DEPLOYMENT, messages=messages, timeout=timeout, **kwargs
        ),
    )
    return response.choices[0].message


def chat_stream(messages: list[dict]) -> Iterator[str]:
    """Yield the answer piece by piece as the model writes it."""
    stream = _explain_404(
        config.CHAT_DEPLOYMENT,
        lambda: _client.chat.completions.create(model=config.CHAT_DEPLOYMENT, messages=messages, stream=True),
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:  # some chunks carry only filter metadata
            yield chunk.choices[0].delta.content
