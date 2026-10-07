"""How a CV's details (name, title, contact, location) are shown to the model and the agent.

The details themselves are read at upload time by processing/entities.py and stored on every chunk.
"""


def profile_line(meta: dict) -> str:
    """One line for the model and the agent: who this CV is. Empty when nothing is known."""
    parts = [meta.get("candidate_name"), meta.get("job_title"), meta.get("email"), meta.get("phone"), meta.get("location")]
    return " · ".join(part for part in parts if part)
