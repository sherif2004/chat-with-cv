"""Finding CV text for a question: embed, hybrid search, merge several searches, pick the top-k CVs and read them whole."""
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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
    """One hybrid search with semantic re-ranking, limited to the sections when there are enough hits. Cached."""
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
    results = _hybrid(ws, query, vector, sections, file_ids, where)
    cache.put(SEARCH, key, results, token)
    return record(results)


def fuse(result_lists: list[list[dict]]) -> list[dict]:
    """Reciprocal rank fusion: a chunk scores 1/(RRF_K + rank) in every list it appears in, so chunks that several
    queries found rise to the top. The order of a single list is kept."""
    scores: Counter[tuple] = Counter()
    chunks: dict[tuple, dict] = {}
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            key = (result["file_name"], result["section"], result["page"], result["content"])
            scores[key] += 1 / (settings.retrieval.rrf_k + rank)
            if key not in chunks or (result.get("score") or 0) > (chunks[key].get("score") or 0):
                chunks[key] = result  # keep the copy with the highest relevance score
    return [chunks[key] for key, _ in scores.most_common()]


class Retrieved(NamedTuple):
    context: list[dict]  # every chunk of the chosen CVs, best CV first, each CV in reading order: what the model reads
    matched: list[dict]  # the chunks the searches found in those CVs, best CV first: what the Sources list shows


def best_cvs(results: list[dict], k: int) -> list[str]:
    """File names of the best `k` CVs, best first. A CV is as relevant as its best chunk."""
    best: dict[str, float] = {}
    for weight, chunk in zip(_weights(results), results):
        best[chunk["file_name"]] = max(best.get(chunk["file_name"], 0.0), weight)
    return sorted(best, key=best.__getitem__, reverse=True)[:k]


def load_cvs(ws: Workspace, names: list[str], cvs: dict[str, str], trace: Trace | None = None) -> list[dict]:
    """Every chunk of each CV in `names` (in that order), each CV in reading order. `cvs` maps a file name to its file id."""
    if not names:
        return []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        loaded = list(pool.map(lambda name: search_index.get_cv_chunks(ws, cvs[name]), names))
    if trace:
        trace.add("load", "read whole CVs", (time.perf_counter() - start) * 1000, cvs=len(names), chunks=sum(len(chunks) for chunks in loaded))
    return [chunk for chunks in loaded for chunk in chunks]


def retrieve(
    ws: Workspace, queries: list[str], sections: list[str], cvs: dict[str, str], trace: Trace | None = None,
    k: int = settings.retrieval.top_k, scoped: bool = False,
) -> Retrieved:
    """Find the `k` best CVs for the question and return all of their text.

    `cvs` maps the file name of every CV that may answer to its file id; scoped=True means the user chose those CVs, so the
    searches are limited to them. Every query is searched (the first is the main one) and the lists are merged. The first search
    returns only the best chunks, which can come from a few CVs, so the CVs it missed are searched again (at most cover_rounds
    times) until `k` CVs, or all of them, are covered.
    """
    file_ids = list(cvs.values()) if scoped else None
    if len(queries) == 1:
        results = search(ws, queries[0], sections, file_ids, trace=trace)
    else:
        with ThreadPoolExecutor(max_workers=len(queries)) as pool:
            results = fuse(list(pool.map(lambda query: search(ws, query, sections, file_ids, trace=trace), queries)))
    results = [result for result in results if result["file_name"] in cvs]
    aim = min(k, len(cvs))
    for _ in range(settings.retrieval.cover_rounds):
        found = {result["file_name"] for result in results}
        missing = [file_id for name, file_id in cvs.items() if name not in found]
        if len(found) >= aim or not missing:
            break
        results = results + [result for result in search(ws, queries[0], sections, missing, trace=trace) if result["file_name"] in cvs]
    names = best_cvs(results, k)
    rank = {name: position for position, name in enumerate(names)}
    matched = sorted((result for result in results if result["file_name"] in rank), key=lambda result: rank[result["file_name"]])
    return Retrieved(load_cvs(ws, names, cvs, trace), matched)


def _hybrid(ws: Workspace, query: str, vector: list[float], sections: list[str], file_ids: list[str] | None, where: str | None) -> list[dict]:
    """When the router picked sections, search those sections and everything at the same time and merge the two lists, so
    a CV with unusual headings is still found and no "too few hits" threshold is needed."""
    if not sections:
        return search_index.hybrid_search(ws, query, vector, settings.retrieval.search_k, file_ids=file_ids, where=where)
    with ThreadPoolExecutor(max_workers=2) as pool:
        focused = pool.submit(search_index.hybrid_search, ws, query, vector, settings.retrieval.search_k, sections, file_ids, where)
        everything = pool.submit(search_index.hybrid_search, ws, query, vector, settings.retrieval.search_k, None, file_ids, where)
        return fuse([focused.result(), everything.result()])


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
