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

**Question answering**: for every question:

1. Embed the question. Embeddings are cached in memory, so a repeated question skips this call.
2. Run a hybrid search (keyword + vector) in Azure AI Search for the 10 most relevant chunks.
3. Send the system prompt, the recent chat history, the chunks and the question to the Azure OpenAI chat model.
4. Return the answer together with the chunks it was based on.

## Project structure

```
app.py              # Streamlit entry point
.streamlit/         # dark theme
cv_chat/
├── config.py       # the only place that reads .env
├── services/       # one thin module per Azure service
├── processing/     # text extraction and chunking (pure Python, no Azure)
├── rag/            # pipelines that combine services and processing
└── ui/             # Streamlit sidebar and chat
```

Imports flow one way: `app.py` → `ui` → `rag` → `services` / `processing` → `config`.

## Setup

You need [uv](https://docs.astral.sh/uv/) and these Azure resources:

- a Storage account
- an AI Search service
- an Azure OpenAI resource with an embedding deployment (for example `text-embedding-3-small`) and a chat deployment (for example `gpt-4o-mini`)

```bash
uv sync
cp .env.example .env    # then fill in your Azure values
```

The Blob container and the search index are created automatically on the first ingestion.

## Run the app

```bash
uv run streamlit run app.py
```

1. In the sidebar, upload at least 8 CVs (PDF or DOCX) and click **Process CVs**. Each file reports its result as soon as it finishes.
2. Ask a question in the chat, or pick one of the suggested questions.
3. Open **Sources** under an answer to see which CVs it was based on.

Indexed CVs stay in Azure, so they are still there after a restart. **New chat** clears the conversation only.

## Command-line checks

These run the pipelines without the UI.

### Ingestion

Put some CVs in a local `cvs/` folder (it is git-ignored), then run:

```bash
uv run python -c "from pathlib import Path; from cv_chat.rag.ingest import process_cvs; [print(r) for r in process_cvs([(p.name, p.read_bytes()) for p in Path('cvs').iterdir()])]"
```

Each CV prints its chunk count, or the error that stopped it.

### Question answering

```bash
uv run python -c "from cv_chat.rag.qa import ask; print(ask('Which candidates know Python?', [])[0])"
```
