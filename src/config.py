import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    container: str
    connection_string: str
    account_url: str
    openai_endpoint: str = ""
    openai_api_key: str = ""
    openai_api_version: str = "2024-10-21"
    embed_deployment: str = ""
    embed_dimensions: int = 1536
    search_endpoint: str = ""
    search_key: str = ""
    search_index: str = "cv-chunks"
    ingest_workers: int = 4
    docling_device: str = "cpu"


def load_settings() -> Settings:
    return Settings(
        container=os.getenv("BLOB_CONTAINER", "pdfs"),
        connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING", ""),
        account_url=os.getenv("AZURE_STORAGE_ACCOUNT_URL", ""),
        openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        embed_deployment=os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", ""),
        embed_dimensions=int(os.getenv("EMBED_DIMENSIONS", "1536")),
        search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT", ""),
        search_key=os.getenv("AZURE_SEARCH_KEY", ""),
        search_index=os.getenv("AZURE_SEARCH_INDEX", "cv-chunks"),
        ingest_workers=int(os.getenv("INGEST_WORKERS", "4")),
        docling_device=os.getenv("DOCLING_DEVICE", "cpu").lower(),
    )


def require_ingest(settings: Settings) -> None:
    """Fail fast, naming every missing env var ingestion needs."""
    required = {
        "AZURE_OPENAI_ENDPOINT": settings.openai_endpoint,
        "AZURE_OPENAI_EMBED_DEPLOYMENT": settings.embed_deployment,
        "AZURE_SEARCH_ENDPOINT": settings.search_endpoint,
    }
    missing = [name for name, value in required.items() if not value]
    if not (settings.connection_string or settings.account_url):
        missing.append("AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_URL")
    if missing:
        raise ValueError("Missing in .env: " + ", ".join(missing))


def setup_logging() -> None:
    """Log everything under `src.*` to the terminal; level from LOG_LEVEL (default INFO)."""
    logger = logging.getLogger("src")
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    logger.propagate = False
