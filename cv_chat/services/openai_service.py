"""Azure OpenAI: embeddings for CV chunks and questions, and chat answers."""
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator
from functools import lru_cache

from openai import APIConnectionError, APITimeoutError, AzureOpenAI, BadRequestError, InternalServerError, NotFoundError, RateLimitError

from cv_chat import config

_client = AzureOpenAI(
    azure_endpoint=config.OPENAI_ENDPOINT,
    api_key=config.OPENAI_API_KEY,
    api_version=config.OPENAI_API_VERSION,
    max_retries=5,  # the SDK backs off on 429s, which parallel ingestion can trigger
)
log = logging.getLogger(__name__)


class AdaptiveLimit:
    """How many requests may be in flight at once. It starts at `start`, halves when Azure answers "too many requests", and
    grows by one after a run of requests that went through, up to `maximum`: it settles on what the deployment allows."""

    def __init__(self, start: int, maximum: int, grow_after: int = 10):
        self._condition = threading.Condition()
        self.limit, self.maximum, self._grow_after = start, maximum, grow_after
        self._active = self._ok = 0

    def __enter__(self) -> "AdaptiveLimit":
        with self._condition:
            while self._active >= self.limit:
                self._condition.wait()
            self._active += 1
        return self

    def __exit__(self, *exc) -> None:
        self.done()

    def done(self, rate_limited: bool = False) -> None:
        with self._condition:
            self._active -= 1
            if rate_limited:
                self.limit, self._ok = max(1, self.limit // 2), 0
            else:
                self._ok += 1
                if self._ok >= self._grow_after and self.limit < self.maximum:
                    self.limit, self._ok = self.limit + 1, 0
            self._condition.notify_all()


_CPUS = os.cpu_count() or 2
_embed_gate = AdaptiveLimit(start=min(4, _CPUS), maximum=max(2, min(8, _CPUS)))  # shared by every parallel CV
_embed_batch = 64  # texts per request: learned downwards if Azure refuses (some deployments take only 16)
_embed_lock = threading.Lock()


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


def _wait_seconds(error: RateLimitError, attempt: int) -> float:
    """How long Azure asks us to wait, or an exponential back-off when it does not say."""
    headers = getattr(getattr(error, "response", None), "headers", None) or {}
    try:
        if "retry-after-ms" in headers:
            return min(float(headers["retry-after-ms"]) / 1000, 60)
        if "retry-after" in headers:
            return min(float(headers["retry-after"]), 60)
    except ValueError:
        pass
    return min(2.0 ** attempt, 30)


def _looks_like_a_size_error(error: BadRequestError) -> bool:
    text = str(error).lower()
    return getattr(error, "code", None) != "content_filter" and any(w in text for w in ("too many", "maximum", "exceed", "array", "limit"))


def _embed_request(batch: list[str]):
    """One embeddings call. When Azure says "too many requests" it waits as long as Azure asks, lowers how many calls run at
    once for everyone, and tries again; connection and server errors are retried a few times."""
    attempt = 0
    while True:
        _embed_gate.__enter__()
        rate_limited = False
        try:
            return _explain_404(
                config.EMBEDDING_DEPLOYMENT,
                lambda: _client.with_options(max_retries=0).embeddings.create(model=config.EMBEDDING_DEPLOYMENT, input=batch),
            )
        except RateLimitError as error:
            rate_limited, wait = True, _wait_seconds(error, attempt)
            if attempt >= 8:
                raise
        except (APIConnectionError, APITimeoutError, InternalServerError):
            wait = min(2.0 ** attempt, 10)
            if attempt >= 4:
                raise
        finally:
            _embed_gate.done(rate_limited)
        attempt += 1
        log.info("embeddings call retried in %.1fs (attempt %d, %d calls allowed at once)", wait, attempt, _embed_gate.limit)
        time.sleep(wait)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed in batches. The batch size is learned (it drops if Azure refuses it) and how many calls run at once adapts to
    Azure's rate limit, so no hand-set numbers are needed."""
    global _embed_batch
    vectors: list[list[float]] = []
    start = 0
    while start < len(texts):
        size = _embed_batch
        try:
            response = _embed_request(texts[start : start + size])
        except BadRequestError as error:
            if size > 1 and _looks_like_a_size_error(error):
                with _embed_lock:
                    _embed_batch = max(1, min(_embed_batch, size) // 2)
                log.info("embeddings batch size lowered to %d", _embed_batch)
                continue
            raise
        vectors.extend(item.embedding for item in response.data)
        start += size
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
