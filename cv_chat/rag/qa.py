"""Question answering over the indexed CVs: rank the CVs with a search, then answer from their full text."""
from functools import lru_cache

from cv_chat import config
from cv_chat.processing.extract import extract_text
from cv_chat.services import blob_storage, openai_service, search_index

SYSTEM_PROMPT = """You answer questions about a set of candidate CVs.
Use only the CVs in the user's message. Each CV starts with its file name.
Name the candidate (or the CV file) behind every fact you state.
If the CVs do not contain the answer, say that the uploaded CVs do not contain this information.
Write concise Markdown."""


@lru_cache(maxsize=config.EMBED_CACHE_SIZE)
def embed_question(question: str) -> tuple[float, ...]:
    """Cached in memory: asking the same question again skips the embedding call."""
    return tuple(openai_service.embed([question])[0])


def ask(question: str, history: list[dict]) -> tuple[str, list[dict]]:
    """Answer a question from the full text of the most relevant CVs."""
    chunks = search_index.hybrid_search(question, list(embed_question(question)), config.TOP_K)
    if not chunks:
        return "I couldn't find any relevant excerpts in the uploaded CVs.", []

    # The search only ranks the CVs. The model then reads each of them in full,
    # so no candidate is judged on the few fragments the search happened to pick.
    full_texts = {}
    for name in dict.fromkeys(chunk["file_name"] for chunk in chunks):
        if (text := _full_text(name)) is not None:
            full_texts[name] = text
        if len(full_texts) == config.MAX_CVS:
            break
    if not full_texts:
        return "I couldn't find any relevant excerpts in the uploaded CVs.", []

    cvs = "\n\n".join(f"CV: {name}\n{text}" for name, text in full_texts.items())
    recent_history = [
        {"role": message["role"], "content": message["content"]}
        for message in history[-config.HISTORY_MESSAGES :]
    ]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *recent_history,
        {"role": "user", "content": f"CVs:\n\n{cvs}\n\nQuestion: {question}"},
    ]
    sources = [chunk for chunk in chunks if chunk["file_name"] in full_texts]
    return openai_service.chat(messages), sources


def _full_text(file_name: str) -> str | None:
    """The CV's text read from its original file, or None if the file was deleted from storage."""
    data = blob_storage.download_file(file_name)
    return None if data is None else extract_text(file_name, data)
