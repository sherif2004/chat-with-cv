"""Azure AI Search: the searchable index of CV chunks and their embeddings."""
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

from cv_chat import config

_credential = AzureKeyCredential(config.SEARCH_KEY)
_index_client = SearchIndexClient(config.SEARCH_ENDPOINT, _credential)
_search_client = SearchClient(config.SEARCH_ENDPOINT, config.SEARCH_INDEX, _credential)


def ensure_index() -> None:
    """Create the index, or update it in place if it already exists."""
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="file_name"),  # keyword search also matches names in file names
        SearchableField(name="content"),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=config.EMBEDDING_DIMENSIONS,
            vector_search_profile_name="default",
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
        profiles=[VectorSearchProfile(name="default", algorithm_configuration_name="hnsw")],
    )
    _index_client.create_or_update_index(SearchIndex(name=config.SEARCH_INDEX, fields=fields, vector_search=vector_search))


def upload_chunks(docs: list[dict]) -> None:
    results = _search_client.upload_documents(docs)
    failed = [result.key for result in results if not result.succeeded]
    if failed:
        raise RuntimeError(f"Search index rejected {len(failed)} chunk(s), e.g. {failed[0]}")


def hybrid_search(text: str, vector: list[float], k: int) -> list[dict]:
    """Keyword and vector search in one query; Azure AI Search merges both rankings."""
    results = _search_client.search(
        search_text=text,
        vector_queries=[VectorizedQuery(vector=vector, k_nearest_neighbors=k, fields="content_vector")],
        select=["file_name", "content"],
        top=k,
    )
    return [{"file_name": result["file_name"], "content": result["content"]} for result in results]
