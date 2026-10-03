"""Question answering over the indexed CVs: route the question, retrieve the best chunks, stream the answer."""
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

from cv_chat import config
from cv_chat.processing.sections import SECTION_TYPES
from cv_chat.services import openai_service, search_index

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

ROUTES = ("chat", "simple", "complex")


@dataclass(frozen=True)
class Route:
    kind: str  # one of ROUTES
    query: str  # the message rewritten to stand alone
    sections: list[str]


@lru_cache(maxsize=config.EMBED_CACHE_SIZE)
def embed_question(question: str) -> tuple[float, ...]:
    """Cached in memory: asking the same question again skips the embedding call."""
    return tuple(openai_service.embed([question])[0])


def _route(question: str, history: list[dict]) -> Route:
    """Classify the message and make it stand alone in one model call. Falls back to a plain search of the raw question."""
    recent = _recent(history)
    try:
        reply = openai_service.chat(
            [{"role": "system", "content": ROUTER_PROMPT}, *recent, {"role": "user", "content": question}],
            json_mode=True,
        )
        data = json.loads(reply)
        kind = data.get("route") if data.get("route") in ROUTES else "simple"
        query = str(data.get("query") or "").strip() or question
        sections = [name for name in data.get("sections", []) if name in SECTION_TYPES]
        return Route(kind, query, sections)
    except Exception:  # routing only improves the search, so a failure must never block the answer
        return Route("simple", question, [])


def _recent(history: list[dict]) -> list[dict]:
    """Recent turns let follow-up questions refer back to earlier answers."""
    return [{"role": m["role"], "content": m["content"]} for m in history[-config.HISTORY_MESSAGES :]]


def _retrieve(query: str, sections: list[str]) -> list[dict]:
    vector = list(embed_question(query))
    results = search_index.hybrid_search(query, vector, config.RETRIEVE_K, sections)
    if sections and len(results) < config.MIN_FILTERED_RESULTS:  # the CVs may use unusual headings: search everything
        results = search_index.hybrid_search(query, vector, config.RETRIEVE_K)
    return _spread_over_cvs(results)


def _spread_over_cvs(results: list[dict]) -> list[dict]:
    """Keep the ranking but allow each CV only a few chunks, so broad questions reach many CVs."""
    per_cv: Counter[str] = Counter()
    picked = []
    for result in results:
        if per_cv[result["file_name"]] < config.MAX_CHUNKS_PER_CV:
            per_cv[result["file_name"]] += 1
            picked.append(result)
        if len(picked) == config.TOP_K:
            break
    return picked


def _excerpt(source: dict) -> str:
    where = f"{source['section']} · p.{source['page']}" if source.get("page") else source["section"]
    return f"[CV: {source['file_name']} · {where}]\n{source['content']}"


def ask(question: str, history: list[dict]) -> tuple[Iterator[str], list[dict]]:
    """Find the relevant CV chunks and start the answer. Returns (answer text as a stream, sources)."""
    question = question.strip()
    route = _route(question, history)
    if route.kind == "chat":  # no search, no sources
        messages = [{"role": "system", "content": CHAT_PROMPT}, *_recent(history), {"role": "user", "content": question}]
        return openai_service.chat_stream(messages), []
    sources = _retrieve(route.query, route.sections)
    excerpts = "\n\n".join(_excerpt(s) for s in sources)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *_recent(history),
        {"role": "user", "content": f"CV excerpts:\n\n{excerpts}\n\nQuestion: {question}"},
    ]
    return openai_service.chat_stream(messages), sources
