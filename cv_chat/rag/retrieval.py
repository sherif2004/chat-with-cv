"""Finding CV text for a question: embed, hybrid search, pick the top-k CVs and give the model the sections it needs."""
import re
import time
from collections import Counter
from functools import lru_cache
from typing import NamedTuple

from cv_chat.config import settings
from cv_chat.rag import metadata
from cv_chat.rag.cache import SEARCH, cache_for
from cv_chat.rag.trace import Trace
from cv_chat.services import openai_service, search_index
from cv_chat.workspace import Workspace


@lru_cache(maxsize=settings.retrieval.embed_cache_size)
def embed_question(question: str) -> tuple[float, ...]:
    """Cached in memory: asking the same question again skips the embedding call."""
    return tuple(openai_service.embed([question])[0])


def search(
    ws: Workspace, query: str, sections: list[str], file_ids: list[str] | None = None, where: str | None = None, trace: Trace | None = None
) -> list[dict]:
    """One hybrid search with semantic re-ranking, boosting the sections the router picked (one Azure request). Cached."""
    start = time.perf_counter()
    info = {"sections": sections, "cvs": len(file_ids or []), "filter": where or "", "cached": False, "fallback": False, "embed_ms": 0}

    def record(results: list[dict]) -> list[dict]:
        if trace:
            trace.add("search", query, (time.perf_counter() - start) * 1000, results=len(results), **info)
        return results

    cache = cache_for(ws)
    key = (query, tuple(sections), tuple(file_ids or ()), where)
    if (cached := cache.get(SEARCH, key)) is not None:
        info["cached"] = True
        return record(cached)
    token = cache.token()
    embed_start = time.perf_counter()
    vector = list(embed_question(query))
    info["embed_ms"] = round((time.perf_counter() - embed_start) * 1000, 1)
    results = search_index.hybrid_search(ws, query, vector, settings.retrieval.search_k, boost_sections=sections, file_ids=file_ids, where=where)
    cache.put(SEARCH, key, results, token)
    return record(results)


class Retrieved(NamedTuple):
    context: list[dict]  # what the model reads: for each of the best CVs, the sections the search found in it (best CV first, each in reading order)
    sources: list[dict]  # the same chunks, shortened: what the Sources list shows and what is stored with the chat message


def best_cvs(results: list[dict], k: int) -> list[str]:
    """File names of the best `k` CVs, best first. A CV is as relevant as its best chunk."""
    best: dict[str, float] = {}
    for weight, chunk in zip(_weights(results), results):
        best[chunk["file_name"]] = max(best.get(chunk["file_name"], 0.0), weight)
    return sorted(best, key=best.__getitem__, reverse=True)[:k]


def shorten(chunk: dict) -> dict:
    """A copy of a chunk with its text cut to settings.retrieval.source_chars, for the Sources list and for storing."""
    limit = settings.retrieval.source_chars
    if len(chunk["content"]) <= limit:
        return dict(chunk)
    return {**chunk, "content": chunk["content"][:limit].rstrip() + "…"}


def retrieve(
    ws: Workspace, query: str, sections: list[str], names: list[str], trace: Trace | None = None,
    k: int = settings.retrieval.top_k, scoped: bool = False,
) -> Retrieved:
    """Find the `k` best CVs for the question and the sections of them that the model needs.

    `names` lists the file name of every CV that may answer; scoped=True means the user chose those CVs, so the search is
    limited to them. The first search returns only the best chunks, which can come from a few CVs, so the CVs it missed are
    searched again (at most cover_rounds times) until `k` CVs, or all of them, are covered: a question about everyone needs
    something from each candidate. The model then reads only the chunks the searches found in the best `k` CVs.
    """
    file_ids = names if scoped else None
    allowed = set(names)
    results = [result for result in search(ws, query, sections, file_ids, trace=trace) if result["file_name"] in allowed]
    aim = min(k, len(names))
    for _ in range(settings.retrieval.cover_rounds):
        found = {result["file_name"] for result in results}
        missing = [name for name in names if name not in found]
        if len(found) >= aim or not missing:
            break
        results = results + [result for result in search(ws, query, sections, missing, trace=trace) if result["file_name"] in allowed]
    rank = {name: position for position, name in enumerate(best_cvs(results, k))}
    chosen = sorted(
        (result for result in results if result["file_name"] in rank),
        key=lambda result: (rank[result["file_name"]], result.get("chunk_no") is None, result.get("chunk_no") or 0),
    )
    return Retrieved(chosen, [shorten(chunk) for chunk in chosen])


def _weights(results: list[dict]) -> list[float]:
    """How relevant each result is, from 0 to 1 (its score divided by the best score). Without scores (the search service
    returned none) the rank stands in for them, which makes every result look about equally relevant."""
    if not results:
        return []
    scores = [result.get("score") for result in results]
    if any(score is None or score <= 0 for score in scores):
        scores = [1 / (settings.retrieval.rrf_k + rank) for rank in range(1, len(results) + 1)]
    best = max(scores)
    return [score / best for score in scores]


def spread_over_cvs(results: list[dict], per_cv: int) -> list[dict]:
    """Keep the ranking but allow each CV only `per_cv` chunks, so a search reaches many CVs (used by the agent)."""
    taken: Counter[str] = Counter()
    picked = []
    for result in results:
        if taken[result["file_name"]] >= per_cv:
            continue
        taken[result["file_name"]] += 1
        picked.append(result)
    return picked


_TAG = re.compile(r"</?\s*cv_excerpt[^>]*>", re.IGNORECASE)


def as_data(text: str) -> str:
    """Wrap text that came from a CV (or a CV's name) in <cv_excerpt> tags, after removing any tags inside it, so the
    model treats it as data. Removing a tag can join the pieces around it into a new one, so repeat until stable."""
    while (stripped := _TAG.sub("", text)) != text:
        text = stripped
    return f"<cv_excerpt>\n{text}\n</cv_excerpt>"


def format_excerpt(chunk: dict, with_profile: bool = False) -> str:
    """A chunk as the model sees it: a header naming its CV, section and page, then the text, inside <cv_excerpt> tags.

    with_profile adds one line about the candidate (name, title, contact). The tags mark the text as data (see
    the system prompts), and tags inside the CV text are removed, so a CV cannot close the block early and pass its
    own text off as instructions.
    """
    where = f"{chunk['section']} · p.{chunk['page']}" if chunk.get("page") else chunk["section"]
    profile = metadata.profile_line(chunk) if with_profile else ""
    return as_data(f"[CV: {chunk['file_name']} · {where}]\n" + (f"Candidate: {profile}\n" if profile else "") + chunk["content"])


def format_excerpts(chunks: list[dict]) -> str:
    """Several chunks for the model. The first excerpt of each CV also carries the candidate line."""
    seen: set[str] = set()
    parts = []
    for chunk in chunks:
        parts.append(format_excerpt(chunk, with_profile=chunk["file_name"] not in seen))
        seen.add(chunk["file_name"])
    return "\n\n".join(parts)
