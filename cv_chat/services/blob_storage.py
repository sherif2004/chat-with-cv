"""Azure Blob Storage: keeps the original uploaded CV files, one container per user."""
import threading
import time
from datetime import datetime, timedelta, timezone

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobSasPermissions, BlobServiceClient, ContainerClient, generate_blob_sas

from cv_chat.config import settings
from cv_chat.workspace import Workspace

_links: dict[tuple[str, str], tuple[float, str]] = {}  # (container, file name) -> (when it was signed, the link)
_links_lock = threading.Lock()
_service = BlobServiceClient.from_connection_string(settings.azure.storage_connection_string.get_secret_value())


def _container(ws: Workspace) -> ContainerClient:
    return _service.get_container_client(ws.container)


def ensure_container(ws: Workspace) -> None:
    """Create the user's container if it is not there. Azure answers 409 both when it already exists (fine) and while a deleted
    container is still being removed, which takes up to a minute or more: that second case must not look like success, or every
    later upload fails with ContainerNotFound."""
    try:
        _container(ws).create_container()
    except ResourceExistsError as error:
        if getattr(error, "error_code", None) == "ContainerBeingDeleted":
            raise RuntimeError(
                f"The storage container '{ws.container}' was deleted a moment ago and Azure is still removing it. "
                "Wait about a minute and try again."
            ) from None


def delete_container(ws: Workspace) -> None:
    try:
        _container(ws).delete_container()
    except ResourceNotFoundError:
        pass


def upload_file(ws: Workspace, name: str, data: bytes) -> None:
    _container(ws).upload_blob(name, data, overwrite=True)


def download_file(ws: Workspace, name: str) -> bytes:
    return _container(ws).download_blob(name).readall()


def delete_file(ws: Workspace, name: str) -> None:
    try:
        _container(ws).delete_blob(name)
    except ResourceNotFoundError:  # already gone: the goal is reached
        pass


def list_files(ws: Workspace) -> list[str]:
    try:
        return sorted(blob.name for blob in _container(ws).list_blobs())
    except ResourceNotFoundError:  # the container is created on the first ingestion
        return []


def blob_url(ws: Workspace, name: str) -> str:
    """The plain address of a stored CV. The container is private, so on its own it does not open; see read_link."""
    return _container(ws).get_blob_client(name).url


def read_link(ws: Workspace, name: str, minutes: int = 60) -> str | None:
    """A link that opens one stored CV for a short time, or None if the storage account has no key to sign it with.

    Signed for this user's container only, read access only. Signing is cheap but a page asks for the same links on every click
    (and each new signature changes the page), so a link is kept in memory for settings.cache.link_seconds, far less than it is valid.
    """
    key = (ws.container, name)
    now = time.monotonic()
    with _links_lock:
        cached = _links.get(key)
    if cached is not None and now - cached[0] < settings.cache.link_seconds:
        return cached[1]
    account_key = getattr(_service.credential, "account_key", None)
    if not account_key:
        return None
    token = generate_blob_sas(
        account_name=_service.account_name, container_name=ws.container, blob_name=name, account_key=account_key,
        permission=BlobSasPermissions(read=True), expiry=datetime.now(timezone.utc) + timedelta(minutes=minutes),
    )
    link = f"{blob_url(ws, name)}?{token}"
    with _links_lock:
        _links.pop(key, None)
        _links[key] = (now, link)
        while len(_links) > settings.cache.links:
            _links.pop(next(iter(_links)))  # the oldest first
    return link
