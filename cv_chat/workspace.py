"""One user's Azure resources: a Blob container for the original CVs and a search index for their chunks.

A Workspace is built from the logged-in user's id and passed to every call that touches Azure, so a user's code can
only ever reach their own container and index. The names come from config (AZURE_STORAGE_CONTAINER and
AZURE_SEARCH_INDEX are prefixes) plus the user's id.
"""
import uuid
from dataclasses import dataclass

from cv_chat import config


@dataclass(frozen=True)
class Workspace:
    user_id: str

    def __post_init__(self):
        uuid.UUID(self.user_id)  # raises for anything that is not a UUID, so a name can never be built from free text

    @property
    def _suffix(self) -> str:
        return self.user_id.replace("-", "").lower()

    @property
    def container(self) -> str:
        return f"{config.STORAGE_CONTAINER.lower()}-{self._suffix}"

    @property
    def index(self) -> str:
        return f"{config.SEARCH_INDEX.lower()}-{self._suffix}"
