"""Azure Blob Storage: keeps the original uploaded CV files, one container per user."""
from datetime import datetime, timedelta, timezone

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobSasPermissions, BlobServiceClient, ContainerClient, generate_blob_sas

from cv_chat import config
from cv_chat.workspace import Workspace

_service = BlobServiceClient.from_connection_string(config.STORAGE_CONNECTION_STRING)


def _container(ws: Workspace) -> ContainerClient:
    return _service.get_container_client(ws.container)


def ensure_container(ws: Workspace) -> None:
    try:
        _container(ws).create_container()
    except ResourceExistsError:
        pass


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

    Made on demand and never stored: the link is signed for this user's container only, read access only.
    """
    key = getattr(_service.credential, "account_key", None)
    if not key:
        return None
    token = generate_blob_sas(
        account_name=_service.account_name, container_name=ws.container, blob_name=name, account_key=key,
        permission=BlobSasPermissions(read=True), expiry=datetime.now(timezone.utc) + timedelta(minutes=minutes),
    )
    return f"{blob_url(ws, name)}?{token}"
