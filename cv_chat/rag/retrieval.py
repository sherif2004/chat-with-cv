"""Finding CV chunks for a question: embed, hybrid search, merge several searches, spread over CVs."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from cv_chat import config
from cv_chat.services import openai_service, search_index


@lru_cache(maxsize=config.EMBED_CACHE_SIZE)
def embed_question(question: str) -> tuple[float, ...]:
    """Cached in memory: asking the same question again skips the embedding call."""
    return tuple(openai_service.embed([question])[0])


def search(query: str, sections: list[str]) -> list[dict]:
    """One hybrid search with semantic re-ranking, limited to the sections when there are enough hits."""
    vector = list(embed_question(query))
    results = search_index.hybrid_search(query, vector, config.RETRIEVE_K, sections)
    if sections and len(results) < config.MIN_FILTERED_RESULTS:  # the CVs may use unusual headings: search everything
        results = search_index.hybrid_search(query, vector, config.RETRIEVE_K)
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


def spread_over_cvs(results: list[dict]) -> list[dict]:
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
