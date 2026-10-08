"""Finding CV chunks for a question: embed, hybrid search, merge several searches, spread over CVs."""
import math
import re
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from cv_chat import config
from cv_chat.rag import metadata
from cv_chat.rag.cache import SEARCH, cache_for
from cv_chat.rag.trace import Trace
from cv_chat.services import openai_service, search_index
from cv_chat.workspace import Workspace


@lru_cache(maxsize=config.EMBED_CACHE_SIZE)
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
            scores[key] += 1 / (config.RRF_K + rank)
            if key not in chunks or (result.get("score") or 0) > (chunks[key].get("score") or 0):
                chunks[key] = result  # keep the copy with the highest relevance score
    return [chunks[key] for key, _ in scores.most_common()]


def retrieve(
    ws: Workspace, queries: list[str], sections: list[str], trace: Trace | None = None, file_ids: list[str] | None = None,
) -> list[dict]:
    """Search every query (the first is the main one), merge the lists, then keep the relevant candidates and their relevant
    excerpts (see rank_cvs). file_ids limits it to those CVs."""
    if len(queries) == 1:
        results = search(ws, queries[0], sections, file_ids, trace=trace)
    else:
        with ThreadPoolExecutor(max_workers=len(queries)) as pool:
            results = fuse(list(pool.map(lambda query: search(ws, query, sections, file_ids, trace=trace), queries)))
    return rank_cvs(results)


def _hybrid(ws: Workspace, query: str, vector: list[float], sections: list[str], file_ids: list[str] | None, where: str | None) -> list[dict]:
    """When the router picked sections, search those sections and everything at the same time and merge the two lists, so
    a CV with unusual headings is still found and no "too few hits" threshold is needed."""
    if not sections:
        return search_index.hybrid_search(ws, query, vector, config.RETRIEVE_K, file_ids=file_ids, where=where)
    with ThreadPoolExecutor(max_workers=2) as pool:
        focused = pool.submit(search_index.hybrid_search, ws, query, vector, config.RETRIEVE_K, sections, file_ids, where)
        everything = pool.submit(search_index.hybrid_search, ws, query, vector, config.RETRIEVE_K, None, file_ids, where)
        return fuse([focused.result(), everything.result()])


def _weights(results: list[dict]) -> list[float]:
    """How relevant each result is, from 0 to 1 (its score divided by the best score). Without scores (the search service
    returned none) the rank stands in for them, which makes every result look about equally relevant."""
    scores = [result.get("score") for result in results]
    if not results or any(score is None or score <= 0 for score in scores):
        scores = [1 / (config.RRF_K + rank) for rank in range(1, len(results) + 1)]
    best = max(scores)
    return [score / best for score in scores]


def _temperature(weights: list[float]) -> float:
    """How sharply a higher relevance is favoured, taken from how spread out this search's relevances are: a search with one
    clear winner among weak results has a wide spread, so the winner takes most of the share; a search whose results are all
    alike has none, so they share it. A small floor keeps a perfectly flat list from dividing by zero."""
    return max(statistics.pstdev(weights), 0.05)


def _nucleus(weights: list[float], top_p: float, temperature: float) -> int:
    """How many of the weights (best first) to keep: the fewest that hold `top_p` of the total share, and at least one.
    A share is the weight's softmax, so a clear winner takes most of it and a close field shares it."""
    top = max(weights)
    exps = [math.exp((weight - top) / temperature) for weight in weights]  # relative to the best, so nothing overflows
    total = sum(exps)
    covered = 0.0
    for count, value in enumerate(exps, start=1):
        covered += value / total
        if covered >= top_p:
            return count
    return len(weights)


def rank_cvs(results: list[dict], top_p: float = config.TOP_P, budget: int = config.CONTEXT_CHARS) -> list[dict]:
    """Keep what is relevant instead of a fixed number. Top-p is applied twice:
    - candidates: a CV is as relevant as its best chunk; the best CVs are kept until they hold `top_p` of the relevance,
      so a question about one person keeps one or two and a broad question keeps many;
    - excerpts: for each kept CV, its best chunks are kept the same way.
    Then excerpts are added, most relevant CV first, until `budget` characters are used (the first excerpt always goes in).
    The result lists the CVs best first, each with its chunks in reading order, so the model sees a candidate's excerpts together."""
    if not results:
        return []
    weights = _weights(results)
    temperature = _temperature(weights)
    groups: dict[str, list[tuple[float, dict]]] = {}
    for weight, chunk in zip(weights, results):
        groups.setdefault(chunk["file_name"], []).append((weight, chunk))
    cvs = sorted(((max(w for w, _ in group), sorted(group, key=lambda item: -item[0])) for group in groups.values()), key=lambda cv: -cv[0])
    picked, used = [], 0
    for _, group in cvs[: _nucleus([best for best, _ in cvs], top_p, temperature)]:
        kept = []
        for _, chunk in group[: _nucleus([weight for weight, _ in group], top_p, temperature)]:
            if (picked or kept) and used + len(chunk["content"]) > budget:
                break
            kept.append(chunk)
            used += len(chunk["content"])
        picked += sorted(kept, key=lambda chunk: (chunk.get("page") is None, chunk.get("page") or 0))
        if used >= budget:
            break
    return picked


def spread_over_cvs(results: list[dict], per_cv: int, budget: int) -> list[dict]:
    """Keep the ranking but allow each CV only a few chunks, so a search reaches many CVs (used by the agent), until `budget`
    characters of text are used (the first result always goes in)."""
    taken: Counter[str] = Counter()
    picked, used = [], 0
    for result in results:
        if taken[result["file_name"]] >= per_cv:
            continue
        if picked and used + len(result["content"]) > budget:
            break
        taken[result["file_name"]] += 1
        picked.append(result)
        used += len(result["content"])
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
