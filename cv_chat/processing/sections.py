"""Map the heading a CV uses ("Work History", "Employment", ...) to one of a few standard section types.

The keywords are in the settings (settings.sections.types). A keyword matches from the start of a word, so "skill" finds
"Skills" and "Skillset" but not "Reskilling"; the first type whose keyword matches wins, in the order of the settings.
"""
import re

from cv_chat.config import settings

_PATTERNS = {
    name: re.compile(r"\b(?:" + "|".join(re.escape(keyword) for keyword in keywords) + ")", re.IGNORECASE)
    for name, keywords in settings.sections.types.items()
}
SECTION_TYPES = [*_PATTERNS, "other"]


def section_type(heading: str) -> str:
    return next((name for name, pattern in _PATTERNS.items() if pattern.search(heading)), "other")
