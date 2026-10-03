"""Question answering over the indexed CVs: route the question, retrieve the best chunks, stream the answer."""
import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from cv_chat import config
from cv_chat.processing.sections import SECTION_TYPES
from cv_chat.rag import agent, retrieval
from cv_chat.rag.cache import ANSWER, ROUTE, cache
from cv_chat.services import openai_service

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You answer questions about a set of candidate CVs.
Use only the CV excerpts in the user's message. Each excerpt starts with a header like [CV: file name · section · p.N].
Name the candidate (or the CV file) behind every fact you state, and cite the evidence right after each claim as
[file name, p.N], using the file name and page from the excerpt header. Leave out ", p.N" when the header has no page.
When you compare candidates, give each candidate their own heading.
If the excerpts do not contain the answer, say that the uploaded CVs do not contain this information.
Write concise Markdown."""

CHAT_PROMPT = """You are the assistant of a "chat with CVs" app. The user's message needs no CV search
(a greeting, thanks, or something unrelated to the CVs). Reply briefly and politely.
Do not state any fact about a candidate. If they ask what you can do, say you answer questions about the uploaded CVs."""

ROUTER_PROMPT = f"""You prepare a message for a chat over a set of candidate CVs.
Given the chat so far and the latest message, reply with JSON: {{"route": "...", "query": "...", "sections": [...]}}.
- "route": one of
  "chat"    - a greeting, thanks, or anything that needs no CV search;
  "simple"  - a question that one search over the CVs can answer;
  "complex" - comparing, ranking, counting or listing across many CVs, or a question with several parts.
- "query": the latest message rewritten to stand alone, with pronouns and references resolved from the chat.
  Keep every name, skill and number. If it already stands alone, repeat it unchanged.
- "sections": the CV sections that hold the answer, chosen only from {SECTION_TYPES}.
  Use [] when the question is about the whole CV or you are unsure."""

EXPAND_PROMPT = f"""You help search a set of candidate CVs. Given a search query, reply with JSON: {{"queries": ["...", "..."]}}
with exactly {config.EXPANDED_QUERIES} alternative queries that look for the same thing:
1. keywords only: the key skills, job titles, technologies and names, no filler words;
2. worded the way a CV would say it (for "built APIs" write "designed and developed REST services").
Keep every name and number from the query. Do not add requirements the query does not have."""

ROUTES = ("chat", "simple", "complex")


@dataclass(frozen=True)
class Route:
    kind: str  # one of ROUTES
    query: str  # the message rewritten to stand alone
    sections: list[str]


def _route(question: str, history: list[dict]) -> Route:
    """Classify the message and make it stand alone in one model call. Falls back to a plain search of the raw question."""
    recent = _recent(history)
    key = (question, tuple((m["role"], m["content"]) for m in recent))
    if (cached := cache.get(ROUTE, key)) is not None:
        return cached
    token = cache.token()
    try:
        reply = openai_service.chat(
            [{"role": "system", "content": ROUTER_PROMPT}, *recent, {"role": "user", "content": question}],
            json_mode=True,
        )
        data = json.loads(reply)
        kind = data.get("route") if data.get("route") in ROUTES else "simple"
        query = str(data.get("query") or "").strip() or question
        sections = [name for name in data.get("sections", []) if name in SECTION_TYPES]
        route = Route(kind, query, sections)
    except Exception:  # routing only improves the search, so a failure must never block the answer
        return Route("simple", question, [])  # not cached: the next try may work
    cache.put(ROUTE, key, route, token)
    return route


def _recent(history: list[dict]) -> list[dict]:
    """Recent turns let follow-up questions refer back to earlier answers."""
    return [{"role": m["role"], "content": m["content"]} for m in history[-config.HISTORY_MESSAGES :]]


def _expand(query: str) -> list[str]:
    """Two alternative search queries (keyword style, and worded the way a CV would say it). [] if the call fails."""
    try:
        data = json.loads(openai_service.chat(
            [{"role": "system", "content": EXPAND_PROMPT}, {"role": "user", "content": query}], json_mode=True
        ))
        alternatives = [str(q).strip() for q in data.get("queries", []) if str(q).strip()]
    except Exception:  # expansion only widens the search, so a failure must never block the answer
        return []
    return [q for q in dict.fromkeys(alternatives) if q != query][: config.EXPANDED_QUERIES]


def _remember(stream: Iterator[str], sources: list[dict], key: tuple, token: int) -> Iterator[str]:
    """Pass the answer through and store it once it has been written completely."""
    pieces = []
    for piece in stream:
        pieces.append(piece)
        yield piece
    cache.put(ANSWER, key, ("".join(pieces), sources), token)


def ask(
    question: str,
    history: list[dict],
    expand: bool = False,
    cache_answers: bool = False,
    on_step: Callable[[str], None] = lambda step: None,
) -> tuple[Iterator[str], list[dict]]:
    """Find the relevant CV chunks and start the answer. Returns (answer text as a stream, sources).

    expand=True also searches reworded queries. cache_answers=True reuses the answer to an identical question with an
    identical chat. on_step is told what the agent is doing on complex questions.
    """
    question = question.strip()
    if not cache_answers:
        return _answer(question, history, expand, on_step)
    key = (question, tuple((m["role"], m["content"]) for m in _recent(history)), expand)
    if (cached := cache.get(ANSWER, key)) is not None:
        return iter([cached[0]]), cached[1]
    token = cache.token()
    stream, sources = _answer(question, history, expand, on_step)
    return _remember(stream, sources, key, token), sources


def _answer(question: str, history: list[dict], expand: bool, on_step: Callable[[str], None]) -> tuple[Iterator[str], list[dict]]:
    route = _route(question, history)
    if route.kind == "chat":  # no search, no sources
        messages = [{"role": "system", "content": CHAT_PROMPT}, *_recent(history), {"role": "user", "content": question}]
        return openai_service.chat_stream(messages), []
    if route.kind == "complex":  # needs more than the best chunks: let the agent plan its own searches
        try:
            return agent.run(route.query, _recent(history), on_step)
        except Exception as error:  # fall back to the plain flow rather than lose the answer
            log.warning("agent failed, using a single search: %s", error)
    queries = [route.query, *(_expand(route.query) if expand else [])]
    sources = retrieval.retrieve(queries, route.sections)
    excerpts = "\n\n".join(retrieval.format_excerpt(s) for s in sources)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *_recent(history),
        {"role": "user", "content": f"CV excerpts:\n\n{excerpts}\n\nQuestion: {question}"},
    ]
    return openai_service.chat_stream(messages), sources
