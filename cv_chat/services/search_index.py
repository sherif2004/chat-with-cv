"""Azure AI Search: the searchable index of CV chunks and their embeddings."""
import logging

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

from cv_chat import config

log = logging.getLogger(__name__)
SEMANTIC_CONFIG = "default"

_credential = AzureKeyCredential(config.SEARCH_KEY)
_index_client = SearchIndexClient(config.SEARCH_ENDPOINT, _credential)
_search_client = SearchClient(config.SEARCH_ENDPOINT, config.SEARCH_INDEX, _credential)


def _existing_dimensions() -> int | None:
    try:
        index = _index_client.get_index(config.SEARCH_INDEX)
    except ResourceNotFoundError:
        return None
    return next((f.vector_search_dimensions for f in index.fields if f.name == "content_vector"), None)


def ensure_index(dimensions: int) -> None:
    """Create the index (vectors of the given length), or update it in place if it already exists."""
    existing = _existing_dimensions()
    if existing is not None and existing != dimensions:
        raise RuntimeError(
            f"The index '{config.SEARCH_INDEX}' stores vectors of length {existing}, but the embedding model returns "
            f"{dimensions}. Delete the index in the Azure portal or set a new AZURE_SEARCH_INDEX in .env, then process the CVs again."
        )
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="file_id", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="file_name"),  # keyword search also matches names in file names
        SimpleField(name="section", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="section_type", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True),
        SimpleField(name="content_hash", type=SearchFieldDataType.String),
        SearchableField(name="content"),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=dimensions,
            vector_search_profile_name="default",
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
        profiles=[VectorSearchProfile(name="default", algorithm_configuration_name="hnsw")],
    )
    semantic_search = SemanticSearch(configurations=[SemanticConfiguration(
        name=SEMANTIC_CONFIG,
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="file_name"),
            content_fields=[SemanticField(field_name="content")],
        ),
    )])
    _index_client.create_or_update_index(
        SearchIndex(name=config.SEARCH_INDEX, fields=fields, vector_search=vector_search, semantic_search=semantic_search)
    )


def upload_chunks(docs: list[dict]) -> None:
    results = _search_client.upload_documents(docs)
    failed = [result.key for result in results if not result.succeeded]
    if failed:
        raise RuntimeError(f"Search index rejected {len(failed)} chunk(s), e.g. {failed[0]}")


def get_hash(file_id: str) -> str | None:
    """The content hash stored for a file, or None if it is not indexed or its chunks disagree (interrupted run)."""
    results = _search_client.search("*", filter=f"file_id eq '{file_id}'", select=["content_hash"])
    hashes = {result["content_hash"] for result in results}
    return hashes.pop() if len(hashes) == 1 else None


def delete_stale_chunks(file_id: str, keep_ids: set[str]) -> None:
    """After re-indexing a changed file: drop chunks the new version no longer has."""
    results = _search_client.search("*", filter=f"file_id eq '{file_id}'", select=["id"])
    stale = [{"id": result["id"]} for result in results if result["id"] not in keep_ids]
    for start in range(0, len(stale), 500):
        _search_client.delete_documents(stale[start : start + 500])


def delete_file_chunks(file_id: str) -> None:
    delete_stale_chunks(file_id, set())


def hybrid_search(text: str, vector: list[float], k: int, section_types: list[str] | None = None) -> list[dict]:
    """Keyword and vector search in one query, merged, then re-ranked by the semantic ranker.

    section_types limits the search to those standard sections (see processing/sections.py).
    """
    section_filter = None
    if section_types:
        section_filter = f"search.in(section_type, '{','.join(section_types)}', ',')"
    kwargs = dict(
        search_text=text,
        vector_queries=[VectorizedQuery(vector=vector, k_nearest_neighbors=k, fields="content_vector")],
        filter=section_filter,
        select=["file_name", "section", "page", "content"],
        top=k,
    )
    try:
        results = list(_search_client.search(
            query_type="semantic",
            semantic_configuration_name=SEMANTIC_CONFIG,
            query_caption="extractive|highlight-false",  # the passage the ranker found most relevant, for Sources
            **kwargs,
        ))
    except HttpResponseError as error:
        if "semantic" not in str(error).lower():
            raise
        # The service has no semantic ranker (Free tier, or switched off): keep answering with plain hybrid search.
        log.warning("semantic ranker unavailable, using hybrid search only: %s", str(error).splitlines()[0])
        results = list(_search_client.search(**kwargs))
    found = []
    for result in results:
        captions = result.get("@search.captions") or []
        found.append({
            "file_name": result["file_name"], "section": result["section"], "page": result["page"],
            "content": result["content"], "caption": captions[0].text if captions and captions[0].text else "",
        })
    return found
