"""Azure Blob Storage: keeps the original uploaded CV files."""
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient

from cv_chat import config

_container = BlobServiceClient.from_connection_string(config.STORAGE_CONNECTION_STRING).get_container_client(
    config.STORAGE_CONTAINER
)


def ensure_container() -> None:
    try:
        _container.create_container()
    except ResourceExistsError:
        pass


def upload_file(name: str, data: bytes) -> None:
    _container.upload_blob(name, data, overwrite=True)


def download_file(name: str) -> bytes:
    return _container.download_blob(name).readall()


def delete_file(name: str) -> None:
    try:
        _container.delete_blob(name)
    except ResourceNotFoundError:  # already gone: the goal is reached
        pass


def list_files() -> list[str]:
    try:
        return sorted(blob.name for blob in _container.list_blobs())
    except ResourceNotFoundError:  # the container is created on the first ingestion
        return []
