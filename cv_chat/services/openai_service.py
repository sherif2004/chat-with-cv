"""Azure OpenAI: embeddings for CV chunks and questions, and chat answers."""
import threading
from collections.abc import Callable, Iterator
from functools import lru_cache

from openai import AzureOpenAI, BadRequestError, NotFoundError

from cv_chat import config

_client = AzureOpenAI(
    azure_endpoint=config.OPENAI_ENDPOINT,
    api_key=config.OPENAI_API_KEY,
    api_version=config.OPENAI_API_VERSION,
    max_retries=5,  # the SDK backs off on 429s, which parallel ingestion can trigger
)
_embed_gate = threading.Semaphore(config.EMBED_CONCURRENCY)  # shared by every parallel CV


class ContentFilterError(RuntimeError):
    """Azure's content safety filter (for example its jailbreak detection) refused the prompt."""


def _explain_404(deployment: str, call: Callable):
    """Azure answers 404 for a wrong deployment name and for a wrong endpoint; say which things to check."""
    try:
        return call()
    except BadRequestError as error:
        if getattr(error, "code", None) == "content_filter":
            raise ContentFilterError("Azure's content safety filter refused the prompt.") from error
        raise
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


def chat_with_tools(messages: list[dict], tools: list[dict], timeout: float | None = None) -> tuple[str, object]:
    """One streamed turn of a tool-using conversation.

    Returns ("answer", pieces) as soon as the model starts writing text, so the caller can stream it on, or
    ("calls", [{"id", "name", "arguments"}, ...]) when the model asked for tools instead.
    """
    stream = _explain_404(
        config.CHAT_DEPLOYMENT,
        lambda: _client.chat.completions.create(
            model=config.CHAT_DEPLOYMENT, messages=messages, tools=tools, stream=True, timeout=timeout
        ),
    )
    calls: dict[int, dict] = {}
    for chunk in stream:
        if not chunk.choices:  # some chunks carry only filter metadata
            continue
        delta = chunk.choices[0].delta
        for call in delta.tool_calls or []:
            entry = calls.setdefault(call.index, {"id": "", "name": "", "arguments": ""})
            entry["id"] += call.id or ""
            if call.function:
                entry["name"] += call.function.name or ""
                entry["arguments"] += call.function.arguments or ""
        if delta.content and not calls:  # text before any tool call is the answer
            return "answer", _continue(delta.content, stream)
    return "calls", [calls[index] for index in sorted(calls)]


def _continue(first: str, stream) -> Iterator[str]:
    yield first
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def chat_stream(messages: list[dict]) -> Iterator[str]:
    """Yield the answer piece by piece as the model writes it."""
    stream = _explain_404(
        config.CHAT_DEPLOYMENT,
        lambda: _client.chat.completions.create(model=config.CHAT_DEPLOYMENT, messages=messages, stream=True),
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:  # some chunks carry only filter metadata
            yield chunk.choices[0].delta.content
