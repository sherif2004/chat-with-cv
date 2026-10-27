"""Turn the error from cv_chat.config into one line for the screen. Separate from config.py: that module is the one that fails."""
from pydantic import ValidationError


def describe(error: ValidationError) -> str:
    """The problems of a ValidationError in one line: the name of each missing value, and what is wrong with the others."""
    parts = []
    for problem in error.errors():
        name = ".".join(str(part) for part in problem["loc"] if part != "azure")
        parts.append(name if problem["type"] == "missing" else f"{name} ({problem['msg']})")
    return ", ".join(parts)
