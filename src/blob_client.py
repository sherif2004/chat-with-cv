from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient

from src.config import Settings, load_settings


def get_service_client(settings: Settings | None = None) -> BlobServiceClient:
    settings = settings or load_settings()
    if settings.connection_string:
        return BlobServiceClient.from_connection_string(settings.connection_string)
    if settings.account_url:
        return BlobServiceClient(settings.account_url, credential=DefaultAzureCredential())
    raise ValueError(
        "Set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_URL in .env"
    )


def get_container_client(
    name: str | None = None, settings: Settings | None = None, create: bool = True
) -> ContainerClient:
    settings = settings or load_settings()
    service = get_service_client(settings)
    container = service.get_container_client(name or settings.container)
    if create and not container.exists():
        container.create_container()
    return container

