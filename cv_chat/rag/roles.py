"""Checking that a job role asked for really exists in the CVs, so a missing role is reported instead of answered with
loosely related candidates."""
import re

from cv_chat.services import search_index
from cv_chat.workspace import Workspace

_WORD = re.compile(r"[a-z0-9+#.]+")


def find(ws: Workspace, role: str, scope: list[str] | None = None) -> tuple[list[str], list[str]]:
    """(the CVs that have the role, the job titles found in the CVs in scope). A CV has the role when every word of it is in
    the CV's job title, or in the CV's text. scope limits the CVs considered; empty means all."""
    words = _WORD.findall(role.lower())
    profiles = {name: profile for name, profile in search_index.list_profiles(ws).items() if not scope or name in scope}
    if not words:
        return sorted(profiles), []
    matching = {name for name, profile in profiles.items() if all(word in (profile.get("job_title") or "").lower() for word in words)}
    matching |= search_index.cvs_mentioning(ws, " ".join(words)) & set(profiles)
    titles = sorted({profile["job_title"] for profile in profiles.values() if profile.get("job_title")})
    return sorted(matching), titles


def missing_message(role: str, titles: list[str], shown: int = 12) -> str:
    """The reply when no CV has the role. Titles come from the CVs, so they are shown as code to keep any markup in them inert."""
    text = f"No CV has the role **{role.replace('*', '')}**."
    if titles:
        listed = ", ".join(f"`{title.replace(chr(96), '')}`" for title in titles[:shown])
        text += f" The job titles in your CVs are: {listed}" + (" and more." if len(titles) > shown else ".")
    return text + " Try another role, or ask about a skill instead."
