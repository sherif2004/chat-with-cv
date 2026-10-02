from typing import Collection

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
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

from src.config import Settings
from src.domain.models import Chunk, CVInfo

_SEMANTIC = "default"
_SECTION_FIELD = SimpleField(name="section", type=SearchFieldDataType.String, filterable=True, facetable=True)


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class AzureSearchIndex:
    def __init__(self, s: Settings):
        credential = AzureKeyCredential(s.search_key) if s.search_key else DefaultAzureCredential()
        self._settings = s
        self._index_client = SearchIndexClient(s.search_endpoint, credential)
        self._client = SearchClient(s.search_endpoint, s.search_index, credential)

    def ensure(self) -> None:
        s = self._settings
        if s.search_index in list(self._index_client.list_index_names()):
            self._add_section_field()
            return
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="cv_id", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="filename", filterable=True),
            _SECTION_FIELD,
            SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True),
            SearchableField(name="content"),
            SimpleField(name="content_hash", type=SearchFieldDataType.String),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=s.embed_dimensions,
                vector_search_profile_name="vec-profile",
            ),
        ]
        self._index_client.create_index(SearchIndex(
            name=s.search_index,
            fields=fields,
            vector_search=VectorSearch(
                algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
                profiles=[VectorSearchProfile(name="vec-profile", algorithm_configuration_name="hnsw")],
            ),
            semantic_search=SemanticSearch(configurations=[SemanticConfiguration(
                name=_SEMANTIC,
                prioritized_fields=SemanticPrioritizedFields(content_fields=[SemanticField(field_name="content")]),
            )]),
        ))

    def _add_section_field(self) -> None:
        """Indexes created before the section column existed get it added in place (old chunks stay empty until re-indexed)."""
        index = self._index_client.get_index(self._settings.search_index)
        if any(f.name == "section" for f in index.fields):
            return
        index.fields.append(_SECTION_FIELD)
        self._index_client.create_or_update_index(index)

    def get_hash(self, cv_id: str) -> str | None:
        """The CV's content hash, or None if it is missing or its chunks disagree (interrupted re-index)."""
        results = self._client.search("*", filter=f"cv_id eq {_quote(cv_id)}", select=["content_hash"])
        hashes = {r["content_hash"] for r in results}
        return hashes.pop() if len(hashes) == 1 else None

    def delete_cv(self, cv_id: str, keep_ids: Collection[str] = ()) -> None:
        results = self._client.search("*", filter=f"cv_id eq {_quote(cv_id)}", select=["id"])
        ids = [{"id": r["id"]} for r in results if r["id"] not in keep_ids]
        for i in range(0, len(ids), 500):
            self._client.delete_documents(ids[i : i + 500])

    def upload(self, chunks: list[Chunk]) -> None:
        docs = [
            {
                "id": c.id, "cv_id": c.cv_id, "filename": c.filename,
                "section": c.section, "page": c.page, "content": c.content,
                "content_hash": c.content_hash, "content_vector": c.vector,
            }
            for c in chunks
        ]
        for i in range(0, len(docs), 100):
            results = self._client.upload_documents(docs[i : i + 100])
            failed = [r.key for r in results if not r.succeeded]
            if failed:
                raise RuntimeError(f"index rejected {len(failed)} chunk(s), e.g. {failed[0]}")

    def list_cvs(self) -> list[CVInfo]:
        found: dict[str, list] = {}
        for r in self._client.search("*", select=["cv_id", "filename"]):
            entry = found.setdefault(r["cv_id"], [r["filename"], 0])
            entry[1] += 1
        return sorted((CVInfo(cv, f, c) for cv, (f, c) in found.items()), key=lambda c: c.filename)
