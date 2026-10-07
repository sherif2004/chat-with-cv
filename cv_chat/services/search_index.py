"""Azure AI Search: the searchable index of CV chunks and their embeddings."""
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

from cv_chat import config
from cv_chat.workspace import Workspace

log = logging.getLogger(__name__)
SEMANTIC_CONFIG = "default"
SCORING_PROFILE = "cv"
BOOSTED_SECTIONS = ["experience", "skills"]  # chunks from these sections rank a little higher for keyword matches
META_FIELDS = ["candidate_name", "job_title", "years_experience", "email", "phone", "location"]
ANALYZER = "en.microsoft"  # English stemming and stop words for keyword search ("built" also finds "building")

_credential = AzureKeyCredential(config.SEARCH_KEY)
_index_client = SearchIndexClient(config.SEARCH_ENDPOINT, _credential)


@lru_cache(maxsize=64)
def _search_client_for(index: str) -> SearchClient:
    return SearchClient(config.SEARCH_ENDPOINT, index, _credential)


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


def _outdated_settings(index: SearchIndex) -> list[str]:
    """Settings the index was created with that Azure cannot change in place. Empty when the index is current."""
    fields = {field.name: field for field in index.fields}
    problems = []
    if getattr(fields.get("content"), "analyzer_name", None) != ANALYZER:
        problems.append(f"the text analyzer of 'content' is not {ANALYZER}")
    if not getattr(fields.get("file_name"), "filterable", False):
        problems.append("'file_name' is not filterable")
    if "candidate_name" not in fields:
        problems.append("the CV metadata fields (candidate name, job title, years, contact) are missing")
    if not any(profile.name == SCORING_PROFILE for profile in index.scoring_profiles or []):
        problems.append("the scoring profile that boosts names, titles and sections is missing")
    algorithms = index.vector_search.algorithms if index.vector_search else []
    if not any(getattr(a.parameters, "metric", None) == "cosine" for a in algorithms):
        problems.append("the vector similarity metric is not cosine")
    return problems


def ensure_index(ws: Workspace, dimensions: int) -> None:
    """Create the index (vectors of the given length), or update it in place if it already exists."""
    existing = _existing_index(ws)
    if existing is not None:
        existing_dimensions = next((f.vector_search_dimensions for f in existing.fields if f.name == "content_vector"), None)
        if existing_dimensions is not None and existing_dimensions != dimensions:
            raise RuntimeError(
                f"The index '{ws.index}' stores vectors of length {existing_dimensions}, but the embedding model returns "
                f"{dimensions}. Delete the index in the Azure portal, then process the CVs again."
            )
        if problems := _outdated_settings(existing):
            raise RuntimeError(
                f"The index '{ws.index}' was made by an older version: {'; '.join(problems)}. Azure cannot change "
                "this in place. Delete the index in the Azure portal, restart the app, "
                "then click Update outdated CVs to index the stored CVs again."
            )
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="file_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="file_url", type=SearchFieldDataType.String),  # plain Blob address of the original file
        SearchableField(name="file_name", filterable=True),  # keyword search also matches names in file names
        SearchableField(name="section", filterable=True, facetable=True),  # headings are searchable and weighted too
        SimpleField(name="section_type", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True),
        SimpleField(name="content_hash", type=SearchFieldDataType.String),
        SearchableField(name="candidate_name", filterable=True),
        SearchableField(name="job_title", filterable=True),
        SimpleField(name="years_experience", type=SearchFieldDataType.Double, filterable=True, sortable=True),
        SimpleField(name="email", type=SearchFieldDataType.String),
        SimpleField(name="phone", type=SearchFieldDataType.String),
        SimpleField(name="location", type=SearchFieldDataType.String),
        SearchableField(name="content", analyzer_name=ANALYZER),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=dimensions,
            vector_search_profile_name="default",
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw", parameters=HnswParameters(metric="cosine"))],
        profiles=[VectorSearchProfile(name="default", algorithm_configuration_name="hnsw")],
    )
    semantic_search = SemanticSearch(configurations=[SemanticConfiguration(
        name=SEMANTIC_CONFIG,
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="file_name"),
            content_fields=[SemanticField(field_name="content")],
            keywords_fields=[SemanticField(field_name="section")],  # the heading tells the ranker what the text is about
        ),
    )])
    scoring_profiles = [ScoringProfile(
        name=SCORING_PROFILE,
        # Keyword matches in a name or title count more than the same words in the body text.
        text_weights=TextWeights(weights={"candidate_name": 3.0, "job_title": 2.0, "file_name": 2.0, "section": 1.5, "content": 1.0}),
        functions=[TagScoringFunction(
            field_name="section_type", boost=1.5, parameters=TagScoringParameters(tags_parameter="boostSections"),
        )],
        function_aggregation="sum",
    )]
    _index_client.create_or_update_index(SearchIndex(
        name=ws.index, fields=fields, vector_search=vector_search, semantic_search=semantic_search,
        scoring_profiles=scoring_profiles,
    ))


def upload_chunks(ws: Workspace, docs: list[dict]) -> None:
    results = _client(ws).upload_documents(docs)
    failed = [result.key for result in results if not result.succeeded]
    if failed:
        raise RuntimeError(f"Search index rejected {len(failed)} chunk(s), e.g. {failed[0]}")


def get_hash(ws: Workspace, file_id: str) -> str | None:
    """The content hash stored for a file, or None if it is not indexed or its chunks disagree (interrupted run)."""
    results = _client(ws).search("*", filter=f"file_id eq '{file_id}'", select=["content_hash"])
    hashes = {result["content_hash"] for result in results}
    return hashes.pop() if len(hashes) == 1 else None


def delete_stale_chunks(ws: Workspace, file_id: str, keep_ids: set[str]) -> None:
    """After re-indexing a changed file: drop chunks the new version no longer has."""
    results = _client(ws).search("*", filter=f"file_id eq '{file_id}'", select=["id"])
    stale = [{"id": result["id"]} for result in results if result["id"] not in keep_ids]
    for start in range(0, len(stale), 500):
        _client(ws).delete_documents(stale[start : start + 500])


def delete_file_chunks(ws: Workspace, file_id: str) -> None:
    delete_stale_chunks(ws, file_id, set())


def list_profiles(ws: Workspace) -> dict[str, dict]:
    """The metadata of every indexed CV, by file name (read from one chunk of each CV)."""
    results = _client(ws).search("*", select=["file_name", *META_FIELDS], top=1000)
    profiles: dict[str, dict] = {}
    for result in results:
        profiles.setdefault(result["file_name"], {name: result.get(name) for name in META_FIELDS})
    return profiles


def metadata_filter(min_years: float | None = None, max_years: float | None = None, job_title: str | None = None) -> str | None:
    """An OData filter on the CV metadata. CVs whose years could not be read never match a years filter."""
    parts = []
    if min_years is not None:
        parts.append(f"years_experience ge {float(min_years)}")
    if max_years is not None:
        parts.append(f"years_experience le {float(max_years)}")
    if job_title and (words := " ".join(job_title.replace("'", " ").split())):
        parts.append(f"search.ismatch('{words}', 'job_title', 'simple', 'all')")  # every word must appear in the title
    return " and ".join(parts) or None


def list_chunks(ws: Workspace) -> list[tuple[str, str]]:
    """(chunk id, file name) for every chunk in the index."""
    return [(result["id"], result["file_name"]) for result in _client(ws).search("*", select=["id", "file_name"])]


def delete_chunks(ws: Workspace, ids: list[str]) -> None:
    if ids:
        _client(ws).delete_documents([{"id": chunk_id} for chunk_id in ids])


def get_cv_chunks(ws: Workspace, file_id: str) -> list[dict]:
    """Every chunk of one CV in reading order."""
    results = _client(ws).search(
        "*", filter=f"file_id eq '{file_id}'", select=["id", "file_name", "file_url", "section", "page", "content", *META_FIELDS], top=1000
    )
    chunks = sorted(results, key=lambda r: int(r["id"].rsplit("-", 1)[1]))  # the id ends with the chunk's position
    return [
        {"file_name": c["file_name"], "file_url": c.get("file_url") or "", "section": c.get("section") or "", "page": c.get("page"), "content": c["content"], "caption": "",
         **{name: c.get(name) for name in META_FIELDS}}
        for c in chunks
    ]


def hybrid_search(
    ws: Workspace,
    text: str,
    vector: list[float],
    k: int,
    section_types: list[str] | None = None,
    file_ids: list[str] | None = None,
    where: str | None = None,
) -> list[dict]:
    """Keyword and vector search in one query, merged, then re-ranked by the semantic ranker.

    section_types limits the search to those standard sections (see processing/sections.py);
    file_ids limits it to those CVs; where is an extra OData filter (see metadata_filter).
    """
    filters = []
    if section_types:
        filters.append(f"search.in(section_type, '{','.join(section_types)}', ',')")
    if file_ids:
        filters.append(f"search.in(file_id, '{','.join(file_ids)}', ',')")
    if where:
        filters.append(f"({where})")
    section_filter = " and ".join(filters) or None
    kwargs = dict(
        search_text=text,
        vector_queries=[VectorizedQuery(vector=vector, k_nearest_neighbors=k, fields="content_vector")],
        filter=section_filter,
        select=["file_name", "file_url", "section", "page", "content", *META_FIELDS],
        scoring_profile=SCORING_PROFILE,
        scoring_parameters=[f"boostSections-{','.join(BOOSTED_SECTIONS)}"],
        top=k,
    )
    try:
        results = list(_client(ws).search(
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
        results = list(_client(ws).search(**kwargs))
    found = []
    for result in results:
        captions = result.get("@search.captions") or []
        reranker = result.get("@search.reranker_score")
        found.append({
            "score": reranker if reranker is not None else result.get("@search.score"),  # how relevant Azure found the chunk
            "file_name": result["file_name"], "file_url": result.get("file_url") or "", "section": result.get("section") or "", "page": result.get("page"),
            "content": result["content"], "caption": captions[0].text if captions and captions[0].text else "",
            **{name: result.get(name) for name in META_FIELDS},
        })
    return found
