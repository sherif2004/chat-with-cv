"""Question answering over the indexed CVs: rewrite the question, retrieve the best chunks, stream the answer."""
import json
from collections import Counter
from collections.abc import Iterator
from functools import lru_cache

from cv_chat import config
from cv_chat.processing.sections import SECTION_TYPES
from cv_chat.services import openai_service, search_index

SYSTEM_PROMPT = """You answer questions about a set of candidate CVs.
Use only the CV excerpts in the user's message. Each excerpt starts with the CV file and section it comes from.
Name the candidate (or the CV file) behind every fact you state.
If the excerpts do not contain the answer, say that the uploaded CVs do not contain this information.
Write concise Markdown."""

REWRITE_PROMPT = f"""You prepare a search query over a set of candidate CVs.
Given the chat so far and the latest question, reply with JSON: {{"query": "...", "sections": [...]}}.
- "query": the latest question rewritten to stand alone, with pronouns and references resolved from the chat.
  Keep every name, skill and number. If it already stands alone, repeat it unchanged.
- "sections": the CV sections that hold the answer, chosen only from {SECTION_TYPES}.
  Use [] when the question is about the whole CV or you are unsure."""


@lru_cache(maxsize=config.EMBED_CACHE_SIZE)
def embed_question(question: str) -> tuple[float, ...]:
    """Cached in memory: asking the same question again skips the embedding call."""
    return tuple(openai_service.embed([question])[0])


def _rewrite(question: str, history: list[dict]) -> tuple[str, list[str]]:
    """Make a follow-up question stand alone and pick the CV sections to search. Falls back to the raw question."""
    recent = [{"role": m["role"], "content": m["content"]} for m in history[-config.HISTORY_MESSAGES :]]
    try:
        reply = openai_service.chat(
            [{"role": "system", "content": REWRITE_PROMPT}, *recent, {"role": "user", "content": question}],
            json_mode=True,
        )
        data = json.loads(reply)
        query = str(data.get("query") or "").strip() or question
        sections = [name for name in data.get("sections", []) if name in SECTION_TYPES]
        return query, sections
    except Exception:  # rewriting only improves the search, so a failure must never block the answer
        return question, []


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


def ask(question: str, history: list[dict]) -> tuple[Iterator[str], list[dict]]:
    """Find the relevant CV chunks and start the answer. Returns (answer text as a stream, sources)."""
    question = question.strip()
    query, sections = _rewrite(question, history)
    sources = _retrieve(query, sections)
    excerpts = "\n\n".join(f"[CV: {s['file_name']} · {s['section']}]\n{s['content']}" for s in sources)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        # Recent turns let follow-up questions refer back to earlier answers.
        *({"role": m["role"], "content": m["content"]} for m in history[-config.HISTORY_MESSAGES :]),
        {"role": "user", "content": f"CV excerpts:\n\n{excerpts}\n\nQuestion: {question}"},
    ]
    return openai_service.chat_stream(messages), sources
