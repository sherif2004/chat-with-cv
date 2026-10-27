"""Narrowing the CVs by what was read from them (job title, location).

A filter is matched here, on the profile row of each CV, and not by Azure Search on words a model guessed: the model is shown the
job titles and locations that really exist, a plural matches a singular, and when nothing matches the answer lists what does
exist, so the filter can be corrected instead of silently returning nothing. The normal question flow and the agent both use it.
"""
import re
from typing import NamedTuple

from cv_chat.services import search_index
from cv_chat.workspace import Workspace

_WORD = re.compile(r"[a-z0-9+#.]+")


class Found(NamedTuple):
    names: list[str]  # the CVs that match every filter that was given
    titles: list[str]  # the job titles that exist in the CVs considered
    locations: list[str]  # the locations that exist in the CVs considered


def _stems(text: str | None) -> set[str]:
    """The words of a text, lower case, with a plural folded to its singular ("engineers" and "engineer" are one word)."""
    words = _WORD.findall((text or "").lower())
    return {word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word for word in words}


def find(ws: Workspace, job_title: str = "", location: str = "", scope: list[str] | None = None) -> Found:
    """The CVs where every word of the job title is in the CV's job title and every word of the location is in its location.
    A filter left empty (or with no usable word) matches everything; a CV whose detail could not be read never matches a
    filter on it. scope limits the CVs considered; empty means all."""
    profiles = {name: profile for name, profile in search_index.list_profiles(ws).items() if not scope or name in scope}
    wanted = {"job_title": _stems(job_title), "location": _stems(location)}
    names = sorted(name for name, profile in profiles.items() if all(words <= _stems(profile.get(field)) for field, words in wanted.items()))
    titles = sorted({profile["job_title"] for profile in profiles.values() if profile.get("job_title")})
    locations = sorted({profile["location"] for profile in profiles.values() if profile.get("location")})
    return Found(names, titles, locations)


def _plain(text: str) -> str:
    return text.replace("*", "").replace("`", "").strip()


def missing_message(job_title: str, location: str, found: Found, shown: int = 12) -> str:
    """The reply when no CV matches. Titles and locations come from the CVs, so they are shown as code to keep any markup inert."""
    asked = " and ".join(f"{label} **{_plain(value)}**" for label, value in (("the job title", job_title), ("the location", location)) if _plain(value))
    text = f"No CV matches {asked}."
    for label, values in (("job titles", found.titles), ("locations", found.locations)):
        if values:
            listed = ", ".join(f"`{_plain(value)}`" for value in values[:shown])
            text += f" The {label} in your CVs are: {listed}" + (" and more." if len(values) > shown else ".")
    return text + " Try other words, or ask about a skill instead."
