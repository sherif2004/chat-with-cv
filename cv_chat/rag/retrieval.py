"""Finding CV chunks for a question: embed, hybrid search, merge several searches, spread over CVs."""
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from cv_chat import config
from cv_chat.rag import metadata
from cv_chat.rag.cache import SEARCH, cache
from cv_chat.services import openai_service, search_index


@lru_cache(maxsize=config.EMBED_CACHE_SIZE)
def embed_question(question: str) -> tuple[float, ...]:
    """Cached in memory: asking the same question again skips the embedding call."""
    return tuple(openai_service.embed([question])[0])


def search(query: str, sections: list[str], file_ids: list[str] | None = None, where: str | None = None) -> list[dict]:
    """One hybrid search with semantic re-ranking, limited to the sections when there are enough hits. Cached."""
    key = (query, tuple(sections), tuple(file_ids or ()), where)
    if (cached := cache.get(SEARCH, key)) is not None:
        return cached
    token = cache.token()
    vector = list(embed_question(query))
    results = search_index.hybrid_search(query, vector, config.RETRIEVE_K, sections, file_ids, where)
    if sections and len(results) < config.MIN_FILTERED_RESULTS:  # the CVs may use unusual headings: search everything
        results = search_index.hybrid_search(query, vector, config.RETRIEVE_K, file_ids=file_ids, where=where)
    cache.put(SEARCH, key, results, token)
    return results


def fuse(result_lists: list[list[dict]]) -> list[dict]:
    """Reciprocal rank fusion: a chunk scores 1/(RRF_K + rank) in every list it appears in, so chunks that several
    queries found rise to the top. The order of a single list is kept."""
    scores: Counter[tuple] = Counter()
    chunks: dict[tuple, dict] = {}
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            key = (result["file_name"], result["section"], result["page"], result["content"])
            scores[key] += 1 / (config.RRF_K + rank)
            chunks.setdefault(key, result)
    return [chunks[key] for key, _ in scores.most_common()]


def retrieve(queries: list[str], sections: list[str]) -> list[dict]:
    """Search every query (the first is the main one), merge the lists and keep the best chunks."""
    if len(queries) == 1:
        results = search(queries[0], sections)
    else:
        with ThreadPoolExecutor(max_workers=len(queries)) as pool:
            results = fuse(list(pool.map(lambda query: search(query, sections), queries)))
    return spread_over_cvs(results)


def spread_over_cvs(
    results: list[dict], per_cv: int = config.MAX_CHUNKS_PER_CV, limit: int = config.TOP_K
) -> list[dict]:
    """Keep the ranking but allow each CV only a few chunks, so broad questions reach many CVs."""
    taken: Counter[str] = Counter()
    picked = []
    for result in results:
        if taken[result["file_name"]] < per_cv:
            taken[result["file_name"]] += 1
            picked.append(result)
        if len(picked) == limit:
            break
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

    with_profile adds one line about the candidate (name, title, years, contact). The tags mark the text as data (see
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
