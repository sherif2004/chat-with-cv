import hashlib
from datetime import datetime, timezone

from azure.storage.blob import ContentSettings

from src.blob_client import get_container_client
from src.config import Settings


class AzureBlobStore:
    def __init__(self, settings: Settings):
        self._container = get_container_client(settings.container, settings=settings)

    def upload(self, filename: str, data: bytes) -> None:
        self._container.get_blob_client(filename).upload_blob(
            data,
            overwrite=True,
            content_settings=ContentSettings(
                content_type="application/pdf",
                content_md5=bytearray(hashlib.md5(data).digest()),
            ),
            metadata={
                "original_filename": filename,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def download(self, filename: str) -> bytes:
        return self._container.download_blob(filename).readall()

    def delete(self, filename: str) -> None:
        self._container.get_blob_client(filename).delete_blob(delete_snapshots="include")
