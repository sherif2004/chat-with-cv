"""Azure OpenAI: embeddings for CV chunks and questions, and chat answers.

Many users share one deployment, so three things keep them out of each other's way:
- embedding calls have two kinds, background (ingesting CVs) and interactive (a question), sharing one adaptive limit; the
  background ones can never take the slots reserved for interactive ones, so a question is not queued behind an upload;
- chat calls have a limit of their own, which halves when Azure answers "too many requests" and grows back after a run of successes;
- one user can have only a few chat calls starting at once (see UserLimits), so one person with many tabs cannot crowd out the rest.
A limit is held until the answer has started to arrive, not while a long answer streams.
"""
import contextvars
import logging
import threading
import time
from collections.abc import Callable, Iterator
from functools import lru_cache

from openai import APIConnectionError, APITimeoutError, AzureOpenAI, BadRequestError, InternalServerError, NotFoundError, RateLimitError

from cv_chat.config import settings

_client = AzureOpenAI(
    azure_endpoint=settings.azure.openai_endpoint,
    api_key=settings.azure.openai_api_key.get_secret_value(),
    api_version=settings.azure.openai_api_version,
    max_retries=0,  # retries are made here, with the limits above, not blindly by the SDK
)
log = logging.getLogger(__name__)

current_user: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_user", default=None)
"""The user a question is answered for. qa.ask sets it in the thread that answers, so chat calls count against that user."""


class AdaptiveLimit:
    """How many requests may be in flight at once. It starts at `start`, halves when Azure answers "too many requests", and
    grows by one after a run of requests that went through, up to `maximum`: it settles on what the deployment allows."""

    def __init__(self, start: int, maximum: int, grow_after: int = 10):
        self._condition = threading.Condition()
        self.limit, self.maximum, self._grow_after = start, maximum, grow_after
        self._active = self._ok = 0

    def acquire(self, reserve: int = 0, timeout: float | None = None) -> bool:
        """Wait for a slot. `reserve` slots are kept free for callers that pass 0 (at least one call may always run when
        nothing else does). False if no slot became free within `timeout` seconds."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._active >= max(self.limit - reserve, 1):
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
            self._active += 1
            return True

    def __enter__(self) -> "AdaptiveLimit":
        self.acquire()
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


class UserSlot:
    """One of a user's slots; release it once, more calls do nothing."""

    def __init__(self, semaphore: threading.BoundedSemaphore | None):
        self._semaphore, self._lock = semaphore, threading.Lock()

    def release(self) -> None:
        with self._lock:
            semaphore, self._semaphore = self._semaphore, None
        if semaphore is not None:
            semaphore.release()


class UserLimits:
    """At most `cap` chat calls starting at once for any one user. A call without a user is not capped."""

    def __init__(self, cap: int):
        self._cap = cap
        self._lock = threading.Lock()
        self._semaphores: dict[str, threading.BoundedSemaphore] = {}

    def hold(self, user: str | None, timeout: float | None = None) -> UserSlot | None:
        """A slot for this user, or None if none became free within `timeout` seconds."""
        if user is None:
            return UserSlot(None)
        with self._lock:
            semaphore = self._semaphores.setdefault(user, threading.BoundedSemaphore(self._cap))
        return UserSlot(semaphore) if semaphore.acquire(timeout=timeout) else None


_concurrency = settings.concurrency
_embed_gate = AdaptiveLimit(start=_concurrency.embed_start, maximum=_concurrency.embed_max)  # shared by every CV and every question
_chat_gate = AdaptiveLimit(start=_concurrency.chat_start, maximum=_concurrency.chat_max)
_user_limits = UserLimits(_concurrency.per_user_chat)
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
            f"Azure OpenAI has no deployment named '{deployment}' at {settings.azure.openai_endpoint}. "
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


def call_with_retry(gate: AdaptiveLimit, call: Callable, reserve: int = 0):
    """Run one API call inside the gate. When Azure says "too many requests" it waits as long as Azure asks, lowers how many
    calls run at once for everyone, and tries again; connection and server errors are retried a few times."""
    attempt = 0
    while True:
        gate.acquire(reserve)
        rate_limited = False
        try:
            return call()
        except RateLimitError as error:
            rate_limited, wait = True, _wait_seconds(error, attempt)
            if attempt >= _concurrency.rate_limit_retries:
                raise
        except (APIConnectionError, APITimeoutError, InternalServerError):
            wait = min(2.0 ** attempt, 10)
            if attempt >= _concurrency.connection_retries:
                raise
        finally:
            gate.done(rate_limited)
        attempt += 1
        log.info("Azure OpenAI call retried in %.1fs (attempt %d, %d calls allowed at once)", wait, attempt, gate.limit)
        time.sleep(wait)


def _embed_request(batch: list[str], background: bool):
    return call_with_retry(
        _embed_gate,
        lambda: _explain_404(
            settings.azure.embedding_deployment,
            lambda: _client.embeddings.create(model=settings.azure.embedding_deployment, input=batch),
        ),
        reserve=_concurrency.interactive_reserved if background else 0,
    )


def embed(texts: list[str], background: bool = False) -> list[list[float]]:
    """Embed in batches. The batch size is learned (it drops if Azure refuses it) and how many calls run at once adapts to
    Azure's rate limit, so no hand-set numbers are needed. background=True is for ingesting CVs: those calls leave
    the slots reserved for questions free."""
    global _embed_batch
    vectors: list[list[float]] = []
    start = 0
    while start < len(texts):
        size = _embed_batch
        try:
            response = _embed_request(texts[start : start + size], background)
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


def _open_chat(deployment: str, create: Callable):
    """Start a chat call: wait for the user's turn and for the chat limit, retry rate limits. Returns what `create` returns
    (a finished reply, or a stream that has started)."""
    slot = _user_limits.hold(current_user.get(), timeout=_concurrency.user_wait_seconds)
    if slot is None:  # waited a long time behind this user's own calls: go on rather than hang
        log.warning("a user waited more than %ss for a free chat slot, going on", _concurrency.user_wait_seconds)
    try:
        return call_with_retry(_chat_gate, lambda: _explain_404(deployment, create))
    finally:
        if slot is not None:
            slot.release()


def chat(messages: list[dict], json_mode: bool = False, deployment: str | None = None) -> str:
    deployment = deployment or settings.azure.chat_deployment
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    response = _open_chat(deployment, lambda: _client.chat.completions.create(model=deployment, messages=messages, **kwargs))
    return response.choices[0].message.content or ""


def suggest_title(question: str, answer: str) -> str | None:
    """A short name for a new chat, written by the small model. None if no small model is set or it fails: the caller keeps the question."""
    if not settings.azure.small_deployment:
        return None
    try:
        title = chat(
            [
                {"role": "system", "content": "Name this chat in 2 to 6 words, in the language of the question. Reply with the name only: no quotes, no full stop."},
                {"role": "user", "content": f"Question: {question[:500]}\n\nAnswer: {answer[:600]}"},
            ],
            deployment=settings.azure.small_deployment,
        )
    except Exception as error:  # a chat name is a nicety: never lose the answer over it
        log.warning("naming the chat with the small model failed: %s", str(error).splitlines()[0][:150])
        return None
    title = " ".join(title.replace('"', " ").split()).strip(" .")
    return title or None


def chat_with_tools(messages: list[dict], tools: list[dict], timeout: float | None = None) -> tuple[str, object]:
    """One streamed turn of a tool-using conversation.

    Returns ("answer", pieces) as soon as the model starts writing text, so the caller can stream it on, or
    ("calls", [{"id", "name", "arguments"}, ...]) when the model asked for tools instead.
    """
    stream = _open_chat(
        settings.azure.chat_deployment,
        lambda: _client.chat.completions.create(
            model=settings.azure.chat_deployment, messages=messages, tools=tools, stream=True, timeout=timeout
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
    stream = _open_chat(
        settings.azure.chat_deployment,
        lambda: _client.chat.completions.create(model=settings.azure.chat_deployment, messages=messages, stream=True),
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:  # some chunks carry only filter metadata
            yield chunk.choices[0].delta.content
