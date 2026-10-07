"""Azure Blob Storage: keeps the original uploaded CV files, one container per user."""
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient, ContainerClient

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
