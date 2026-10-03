"""Question answering over the indexed CVs: retrieve the most relevant chunks, then answer with Azure OpenAI."""
from functools import lru_cache

from cv_chat import config
from cv_chat.services import openai_service, search_index

SYSTEM_PROMPT = """You answer questions about a set of candidate CVs.
Use only the CV excerpts in the user's message. Each excerpt starts with the CV file it comes from.
Name the candidate (or the CV file) behind every fact you state.
If the excerpts do not contain the answer, say that the uploaded CVs do not contain this information.
Write concise Markdown."""


@lru_cache(maxsize=config.EMBED_CACHE_SIZE)
def embed_question(question: str) -> tuple[float, ...]:
    """Cached in memory: asking the same question again skips the embedding call."""
    return tuple(openai_service.embed([question])[0])


def ask(question: str, history: list[dict]) -> tuple[str, list[dict]]:
    """Answer a question using the most relevant indexed CV excerpts."""
    sources = search_index.hybrid_search(question, list(embed_question(question)), config.TOP_K)
    if not sources:
        return "I couldn't find any relevant excerpts in the uploaded CVs.", []

    excerpts = "\n\n".join(
        f"CV: {source['file_name']}\nExcerpt:\n{source['content']}" for source in sources
    )
    recent_history = [
        {"role": message["role"], "content": message["content"]}
        for message in history[-config.HISTORY_MESSAGES :]
    ]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *recent_history,
        {"role": "user", "content": f"CV excerpts:\n\n{excerpts}\n\nQuestion: {question}"},
    ]
    return openai_service.chat(messages), sources
