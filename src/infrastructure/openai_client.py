import threading

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

from src.config import Settings

_EMBED_BATCH = 16


def _client(s: Settings) -> AzureOpenAI:
    kwargs = dict(
        azure_endpoint=s.openai_endpoint,
        api_version=s.openai_api_version,
        max_retries=6,  # the SDK backs off on 429 and honours Retry-After
    )
    if s.openai_api_key:
        return AzureOpenAI(api_key=s.openai_api_key, **kwargs)
    provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    return AzureOpenAI(azure_ad_token_provider=provider, **kwargs)


class AzureEmbedder:
    """One shared instance: its semaphore caps concurrent calls across all ingest workers."""

    def __init__(self, s: Settings, max_concurrent: int = 2):
        self._client = _client(s)
        self._deployment = s.embed_deployment
        self._dimensions = s.embed_dimensions
        self._gate = threading.Semaphore(max_concurrent)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), _EMBED_BATCH):
            with self._gate:
                response = self._client.embeddings.create(
                    model=self._deployment,
                    input=texts[i : i + _EMBED_BATCH],
                    dimensions=self._dimensions,
                )
            vectors.extend(item.embedding for item in response.data)
        return vectors
