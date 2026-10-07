"""Starter questions for the welcome screen, built from the CVs that are indexed (their job titles and places).
No model call. Anything that does not look like plain words is left out, because it ends up as text on a button."""
import re
from collections import Counter

FIXED = [
    "Who has strong Python experience?",
    "Compare cloud and DevOps skills",
    "Who has the most work experience?",
    "Summarize everyone's education",
]
COUNT = 4
_GENERIC = {
    "senior", "junior", "lead", "principal", "staff", "chief", "head", "associate", "assistant", "intern", "trainee",
    "and", "the", "for", "of", "full", "stack", "software", "technical", "systems", "research", "network",
}
_WORD = re.compile(r"^[A-Za-z]{3,20}$")
_PLACE = re.compile(r"^[A-Za-z][A-Za-z .,'-]{1,38}$")


def _plural(word: str) -> str:
    return word if word.lower().endswith("s") else word + "s"


def build(candidates: list[dict]) -> list[str]:
    """Up to COUNT questions: the ones the data suggests first, then the fixed ones to fill the rest."""
    questions: list[str] = []

    words = Counter(
        word.capitalize()
        for c in candidates
        for word in {w for w in re.split(r"[\s,/|()-]+", c.get("title") or "") if _WORD.match(w) and w.lower() not in _GENERIC}
    )
    if words and (word := words.most_common(1)[0])[1] >= 2:
        questions.append(f"Compare the {_plural(word[0])}")

    places = Counter(c["location"].strip() for c in candidates if _PLACE.match((c.get("location") or "").strip()))
    if places and (place := places.most_common(1)[0])[1] >= 2:
        questions.append(f"Who is based in {place[0]}?")

    for fixed in FIXED:
        if len(questions) >= COUNT:
            break
        questions.append(fixed)
    return questions[:COUNT]
