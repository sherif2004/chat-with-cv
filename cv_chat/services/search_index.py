"""Azure AI Search: the searchable index of CV chunks and their embeddings."""
import base64
import hashlib
import json
import logging
from functools import lru_cache

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    ScoringProfile,
    TagScoringFunction,
    TagScoringParameters,
    TextWeights,
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

from cv_chat.config import settings
from cv_chat.processing.sections import SECTION_TYPES
from cv_chat.workspace import Workspace

log = logging.getLogger(__name__)
META_FIELDS = ["candidate_name", "job_title", "email", "phone", "location"]

_semantic_available = True  # False once the service has said it has no semantic ranker
DELIMITERS = ("|", ";", "~", "^")  # for search.in: the first one that is in none of the file names


def chunk_key(file_id: str, number: int) -> str:
    """The document key of one chunk. A key may hold only letters, digits, _ - and =, and a file name may hold anything else,
    so the name is base64-encoded (URL-safe); the chunk number after the last "-" keeps the chunks of a CV in reading order."""
    return f"{base64.urlsafe_b64encode(file_id.encode()).decode()}-{number}"


def odata_string(value: str) -> str:
    """A text as an OData string literal: in quotes, with each quote inside doubled."""
    return "'" + value.replace("'", "''") + "'"


def file_id_filter(file_ids: list[str]) -> str | None:
    """An OData filter that matches the chunks of these CVs (the file id is the file name, which may hold quotes and commas)."""
    if not file_ids:
        return None
    if len(file_ids) == 1:
        return f"file_id eq {odata_string(file_ids[0])}"
    delimiter = next((d for d in DELIMITERS if not any(d in file_id for file_id in file_ids)), None)
    if delimiter is None:
        return "(" + " or ".join(f"file_id eq {odata_string(file_id)}" for file_id in file_ids) + ")"
    return f"search.in(file_id, {odata_string(delimiter.join(file_ids))}, {odata_string(delimiter)})"


_credential = AzureKeyCredential(settings.azure.search_key.get_secret_value())
_index_client = SearchIndexClient(settings.azure.search_endpoint, _credential)


@lru_cache(maxsize=64)
def _search_client_for(index: str) -> SearchClient:
    return SearchClient(settings.azure.search_endpoint, index, _credential)


def _client(ws: Workspace) -> SearchClient:
    return _search_client_for(ws.index)


def _existing_index(ws: Workspace) -> SearchIndex | None:
    try:
        return _index_client.get_index(ws.index)
    except ResourceNotFoundError:
        return None


def delete_index(ws: Workspace) -> None:
    try:
        _index_client.delete_index(ws.index)
    except ResourceNotFoundError:
        pass


MARKER_PREFIX = "cvschema_"  # the name of the marker field starts with this, followed by a hash of the whole index definition


def _fields(dimensions: int) -> list[SearchField]:
    return [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="file_id", type=SearchFieldDataType.String, filterable=True),  # the file name
        SimpleField(name="file_url", type=SearchFieldDataType.String),  # plain Blob address of the original file
        SearchableField(name="file_name", filterable=True),  # keyword search also matches names in file names
        SearchableField(name="section", filterable=True, facetable=True),  # headings are searchable and weighted too
        SimpleField(name="section_type", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True),
        SimpleField(name="chunk_no", type=SearchFieldDataType.Int32, filterable=True, sortable=True),  # the chunk's position in its CV
        SimpleField(name="content_hash", type=SearchFieldDataType.String),
        SearchableField(name="candidate_name", filterable=True),
        SearchableField(name="job_title", filterable=True),
        SimpleField(name="email", type=SearchFieldDataType.String),
        SimpleField(name="phone", type=SearchFieldDataType.String),
        SimpleField(name="location", type=SearchFieldDataType.String),
        SearchableField(name="content", analyzer_name=settings.index.analyzer),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=dimensions,
            vector_search_profile_name="default",
        ),
    ]


def desired_index(ws: Workspace, dimensions: int) -> SearchIndex:
    """The index this version of the app wants for the user. It carries a marker field named after a hash of the whole
    definition (fields, analyzer, vector size and metric, scoring, semantic ranking, and settings.index.version), so an index
    made by any other version is recognised by its missing marker."""
    parts = dict(
        fields=_fields(dimensions),
        vector_search=VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw", parameters=HnswParameters(metric=settings.index.vector_metric))],
            profiles=[VectorSearchProfile(name="default", algorithm_configuration_name="hnsw")],
        ),
        semantic_search=SemanticSearch(configurations=[SemanticConfiguration(
            name=settings.index.semantic_config,
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="file_name"),
                content_fields=[SemanticField(field_name="content")],
                keywords_fields=[SemanticField(field_name="section")],  # the heading tells the ranker what the text is about
            ),
        )]),
        scoring_profiles=[ScoringProfile(
            name=settings.index.scoring_profile,
            # Keyword matches in a name or title count more than the same words in the body text.
            text_weights=TextWeights(weights=settings.index.text_weights),
            functions=[TagScoringFunction(
                field_name="section_type", boost=settings.index.section_boost, parameters=TagScoringParameters(tags_parameter="boostSections"),
            )],
            function_aggregation="sum",
        )],
    )
    definition = json.dumps(SearchIndex(name="definition", **parts).as_dict(), sort_keys=True, default=str)
    signature = hashlib.md5(f"{definition}/{settings.index.version}".encode()).hexdigest()[:12]
    marker = SimpleField(name=f"{MARKER_PREFIX}{signature}", type=SearchFieldDataType.String)
    return SearchIndex(name=ws.index, **{**parts, "fields": [*parts["fields"], marker]})


def ensure_index(ws: Workspace, dimensions: int) -> bool:
    """Make the user's index the one this version wants (vectors of the given length). Azure cannot change an analyzer, a key
    or a vector size in place, so an index made by any other version is deleted and created again; the CVs stay in Blob Storage.
    Returns True when the index was just created, and so is empty: the stored CVs have to be indexed again."""
    wanted = desired_index(ws, dimensions)
    marker = next(field.name for field in wanted.fields if field.name.startswith(MARKER_PREFIX))
    existing = _existing_index(ws)
    if existing is not None and any(field.name == marker for field in existing.fields):
        return False
    if existing is not None:
        log.info("the search index %s is out of date, creating it again", ws.index)
        delete_index(ws)
    _index_client.create_or_update_index(wanted)
    return True


def upload_chunks(ws: Workspace, docs: list[dict]) -> None:
    results = _client(ws).upload_documents(docs)
    failed = [result.key for result in results if not result.succeeded]
    if failed:
        raise RuntimeError(f"Search index rejected {len(failed)} chunk(s), e.g. {failed[0]}")


def get_hash(ws: Workspace, file_id: str) -> str | None:
    """The content hash stored for a file, or None if it is not indexed or its chunks disagree (interrupted run)."""
    results = _client(ws).search("*", filter=file_id_filter([file_id]), select=["content_hash"])
    hashes = {result["content_hash"] for result in results}
    return hashes.pop() if len(hashes) == 1 else None


def delete_stale_chunks(ws: Workspace, file_id: str, keep_ids: set[str]) -> None:
    """After re-indexing a changed file: drop chunks the new version no longer has. Found by file name, which every chunk
    has had since the first version, so chunks written under an older id scheme are cleaned up too."""
    results = _client(ws).search("*", filter=f"file_name eq {odata_string(file_id)}", select=["id"])
    stale = [{"id": result["id"]} for result in results if result["id"] not in keep_ids]
    batch = settings.index.delete_batch
    for start in range(0, len(stale), batch):
        _client(ws).delete_documents(stale[start : start + batch])


def delete_file_chunks(ws: Workspace, file_id: str) -> None:
    delete_stale_chunks(ws, file_id, set())


def list_profiles(ws: Workspace) -> dict[str, dict]:
    """The metadata of every indexed CV, by file name: one row per CV (its first chunk)."""
    results = _client(ws).search("*", filter="chunk_no eq 0", select=["file_name", *META_FIELDS], top=1000)
    return {result["file_name"]: {name: result.get(name) for name in META_FIELDS} for result in results}


def get_cv_chunks(ws: Workspace, file_id: str) -> list[dict]:
    """Every chunk of one CV in reading order."""
    results = _client(ws).search(
        "*", filter=file_id_filter([file_id]), select=["chunk_no", "file_name", "file_url", "section", "section_type", "page", "content", *META_FIELDS], top=1000
    )
    chunks = sorted(results, key=lambda r: r["chunk_no"])
    return [
        {"file_name": c["file_name"], "file_url": c.get("file_url") or "", "section": c.get("section") or "", "section_type": c.get("section_type") or "other",
         "chunk_no": c["chunk_no"], "page": c.get("page"), "content": c["content"], "caption": "",
         **{name: c.get(name) for name in META_FIELDS}}
        for c in chunks
    ]


def hybrid_search(
    ws: Workspace,
    text: str,
    vector: list[float],
    k: int,
    boost_sections: list[str] | None = None,
    file_ids: list[str] | None = None,
    section_types: list[str] | None = None,
) -> list[dict]:
    """One Azure request that does keyword search, vector search and semantic re-ranking together.

    boost_sections are the standard sections (see processing/sections.py) whose chunks rank higher through the scoring profile;
    nothing is filtered out, so a CV with unusual headings is still found. Without them the default boosted sections are used.
    file_ids limits the search to those CVs; section_types, if given, restricts it to chunks of those standard section types
    (a hard filter, unlike boost_sections; unknown types are ignored).
    """
    global _semantic_available
    types = [name for name in section_types or [] if name in SECTION_TYPES]
    filters = [file_id_filter(file_ids or []), f"search.in(section_type, '{','.join(types)}', ',')" if types else None]
    kwargs = dict(
        search_text=text,
        vector_queries=[VectorizedQuery(vector=vector, k_nearest_neighbors=k, fields="content_vector")],
        filter=" and ".join(f for f in filters if f) or None,
        select=["file_name", "file_url", "section", "page", "chunk_no", "content", *META_FIELDS],
        scoring_profile=settings.index.scoring_profile,
        scoring_parameters=[f"boostSections-{','.join(boost_sections or settings.sections.boosted)}"],
        top=k,
    )
    results = None
    if _semantic_available:
        try:
            results = list(_client(ws).search(
                query_type="semantic",
                semantic_configuration_name=settings.index.semantic_config,
                query_caption="extractive|highlight-false",  # the passage the ranker found most relevant, for Sources
                **kwargs,
            ))
        except HttpResponseError as error:
            if "semantic" not in str(error).lower():
                raise
            # The service has no semantic ranker (Free tier, or switched off): plain hybrid search from now on, without asking again.
            log.warning("semantic ranker unavailable, using hybrid search only: %s", str(error).splitlines()[0])
            _semantic_available = False
    if results is None:
        results = list(_client(ws).search(**kwargs))
    found = []
    for result in results:
        captions = result.get("@search.captions") or []
        reranker = result.get("@search.reranker_score")
        found.append({
            "score": reranker if reranker is not None else result.get("@search.score"),  # how relevant Azure found the chunk
            "file_name": result["file_name"], "file_url": result.get("file_url") or "", "section": result.get("section") or "", "page": result.get("page"),
            "chunk_no": result.get("chunk_no"), "content": result["content"], "caption": captions[0].text if captions and captions[0].text else "",
            **{name: result.get(name) for name in META_FIELDS},
        })
    return found
