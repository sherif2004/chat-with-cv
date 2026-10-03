# Chat with CVs

Upload CVs, make them searchable, and ask questions about them.
Built on **Azure Blob Storage**, **Azure AI Search** and **Azure OpenAI**.

## How it works

**Ingestion**: every uploaded CV goes through these steps, several CVs in parallel:

1. Extract the text (PDF with PyMuPDF, DOCX with python-docx).
2. Split it with a recursive character splitter (1500 characters, 20% overlap).
3. Embed each chunk with Azure OpenAI.
4. Upload the chunks and their vectors to Azure AI Search.
5. Store the original file in Azure Blob Storage.

One failing file (for example, a scanned PDF with no text) never stops the others.

## Project structure

```
cv_chat/
├── config.py       # the only place that reads .env
├── services/       # one thin module per Azure service
├── processing/     # text extraction and chunking (pure Python, no Azure)
└── rag/            # pipelines that combine services and processing
```

Imports flow one way: `rag` → `services` / `processing` → `config`.

## Setup

You need [uv](https://docs.astral.sh/uv/) and these Azure resources:

- a Storage account
- an AI Search service
- an Azure OpenAI resource with an embedding deployment (for example `text-embedding-3-small`)

```bash
uv sync
cp .env.example .env    # then fill in your Azure values
```

The Blob container and the search index are created automatically on the first ingestion.

## Try ingestion

Put some CVs in a local `cvs/` folder (it is git-ignored), then run:

```bash
uv run python -c "from pathlib import Path; from cv_chat.rag.ingest import process_cvs; [print(r) for r in process_cvs([(p.name, p.read_bytes()) for p in Path('cvs').iterdir()])]"
```

Each CV prints its chunk count, or the error that stopped it.
