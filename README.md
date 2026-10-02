# chat-with-cv — phase 1: ingestion

Upload CV PDFs → Azure Blob Storage → Docling extraction → section-aware chunking → Azure OpenAI embeddings → Azure AI Search.
Several CVs are ingested in parallel (`INGEST_WORKERS`); one failing PDF never blocks the rest. Unchanged files (same MD5) are skipped.

```bash
pip install -r requirements.txt
cp .env.example .env     # fill in Azure Storage, OpenAI embedding and AI Search values
streamlit run app.py     # upload / re-index / delete CVs
```

Output contract for the next phase: chunks in the Azure AI Search index (`src/infrastructure/search_index.py` defines the schema).
Scanned PDFs are reported as failed.
