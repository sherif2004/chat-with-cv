<div align="center">

# Chat with CVs

**Upload a pile of CVs, then just ask questions about them.**

Every answer comes from the uploaded CVs, is written in the language you asked in, and shows which CVs and excerpts it was based on.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Azure](https://img.shields.io/badge/Cloud-Azure-0078D4?logo=microsoftazure&logoColor=white)
![uv](https://img.shields.io/badge/Env-uv-7C6CF6)

</div>

---

## Contents

- [What it does](#what-it-does)
- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Using the app](#using-the-app)
- [Accounts and Postgres](#accounts-and-postgres)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Good to know](#good-to-know)
- [Chat features in detail](#chat-features-in-detail)
- [Troubleshooting](#troubleshooting)
- [Run the pipeline without the UI](#run-the-pipeline-without-the-ui)

---

## What it does

| Step | What you do | What the app does |
|---|---|---|
| 0. Account | Sign up or log in | Gives you your own CV storage, search index and saved chats |
| 1. Upload | Drop PDF or DOCX CVs in the sidebar (at least 8 in total) | Reads and stores them |
| 2. Process | Click **Process CVs** | Reads each CV's layout, splits it by section and makes it searchable, several CVs at the same time. CVs that are already indexed and unchanged are skipped |
| 3. Ask | Type a question in the chat, in any language | Works out what kind of question it is, finds the relevant parts of the CVs (or plans several searches for a complex question) and streams an answer in your language, with `[file, p.N]` citations |
| 4. Check | Open **Sources** or **Details** under an answer | Sources: which CVs, and which excerpts, the answer used. Details: how the question was routed, what was searched, and where the time went |

The three required Azure services each have one job:

| Service | Job in this app |
|---|---|
| **Azure Blob Storage** | Keeps the original CV files |
| **Azure AI Search** | Holds the searchable CV content, finds the parts relevant to a question and re-ranks them (semantic ranker) |
| **Azure OpenAI** | Turns text into numbers for searching (embeddings) and writes the answers |

---

## How it works

This is a standard "chat with your documents" design, also called **RAG** (retrieval-augmented generation). The model never reads all the CVs. It only sees the few pieces that were found for the question.

### 1. Processing CVs (once per CV)

```mermaid
flowchart LR
    A["CV file<br/>PDF or DOCX"] --> H{"Already indexed<br/>and unchanged?"}
    H -- yes --> Z["Skip"]
    H -- no --> B["Extract with Docling<br/>layout, headings, tables"]
    B --> M["Read name, title,<br/>years, contact<br/>one model call"]
    M --> C["Split by section"]
    C --> D["Embed each chunk<br/>Azure OpenAI"]
    D --> E[("Azure AI Search<br/>chunks and vectors")]
    A --> F[("Azure Blob Storage<br/>original file")]
```

1. **Skip unchanged CVs.** Every chunk is stored with a hash of the file content (plus a fingerprint of the extraction, chunking, section and metadata code and settings). If the stored hash matches, the CV is skipped and shown as "already indexed, unchanged".
2. **Extract with Docling.** [Docling](https://github.com/docling-project/docling) reads the page layout, so multi-column CVs come out in the right reading order, tables become markdown, and section headings are detected. It handles PDF and DOCX. The result is cached on disk in `.cache/extracted/` (by file content), so changing the chunking never runs Docling again.
3. **Read the CV's metadata.** One model call reads the candidate's name, current job title, total years of experience (counted from the dated jobs), email, phone and location. They are stored on every chunk of that CV. If the call fails or is refused, the CV is still indexed with empty metadata. The values come from a model, so treat years of experience as an estimate. Contact details are personal data: they are stored in the search index like the rest of the CV text.
4. **Split by section.** The CV is cut at its headings (the ones Docling found, plus short ALL-CAPS lines). Each section becomes one chunk and starts with its heading. Only a section longer than `CHUNK_SIZE` (1500 characters) is split further, with 20% overlap, preferring line breaks. Text before the first heading goes under "Other".
5. **Label each chunk.** The heading is kept as `section` (as written in the CV) and also mapped to a standard `section_type`: experience, education, skills, projects, summary, certifications, languages, contact or other. The page number is stored too.
6. **Embed each chunk.** An embedding is a list of numbers that captures the meaning of a text. Each chunk is sent to Azure OpenAI together with its file name and the candidate's name and title, so even a chunk from page 3 still points to the right CV. Requests go out in batches of 16, and a shared limit of 2 requests in flight keeps parallel CVs from hitting rate limits.
7. **Save to Azure AI Search.** The new chunks are uploaded first, then chunks the new version no longer has are deleted, so a CV is never missing from the index.
8. **Save the original** to Azure Blob Storage, then clear the answer cache (see [Caching](#caching)).

Several CVs are processed in parallel (4 at a time) by a background queue, and the sidebar shows each file's state live. Docling runs one conversion at a time (its models are not thread-safe), but other CVs can embed and upload while one is being read. One CV failing, for example a scanned PDF with no text, never stops the others.

### 2. Answering a question (every time you ask)

```mermaid
flowchart TD
    Q["Your question<br/>+ recent chat"] --> R{"Router<br/>one model call"}
    R -- "chat (hi, thanks)" --> C["Reply without searching"]
    R -- simple --> X["Optional: reworded queries"]
    X --> S[("Azure AI Search<br/>hybrid search + semantic ranker<br/>name, title and section boosts")]
    S --> T["Top-p: keep the relevant CVs<br/>and their relevant excerpts"]
    T --> M["Azure OpenAI chat model<br/>streams the answer in your language"]
    R -- complex --> G["Agent: list_cvs, search_cvs, get_cv<br/>max 5 rounds, 30 s"]
    G --> M
    C --> A["Answer + Sources + Details"]
    M --> A
```

1. **Route and rewrite the question.** One quick model call decides whether the message needs a search at all (*chat*: greetings and thanks do not), needs one search (*simple*), or is *complex* (comparing, ranking, counting or listing across many CVs, or several questions in one). It also turns a follow-up like "what about his education?" into a standalone **English** search query using the recent chat (the CVs are English, so this works for a question asked in any language), and picks which CV sections hold the answer (for example `education`). If this step fails, the original question is searched as it is.
2. **Query expansion (optional, sidebar toggle).** The question is also reworded two ways and each version is searched; the result lists are merged with reciprocal rank fusion, so a chunk found by several queries rises to the top.
3. **Embed the query** the same way as the chunks. Embeddings of repeated queries are cached in memory.
4. **Hybrid search, filtered by section.** Azure AI Search runs two searches in one query and merges the rankings:
   - *keyword search* finds exact words, such as a skill, tool or name (English stemming: "built" also finds "building"),
   - *vector search* finds chunks with a similar meaning, even if the words differ ("cloud" finds "Azure").

   A scoring profile makes a keyword match in the candidate name or job title count more than the same words in the body text, and favours the *experience* and *skills* sections. When sections were picked, only those sections are searched. If fewer than 3 chunks match (some CVs use unusual headings), it searches everything instead.
5. **Semantic re-ranking.** Azure's semantic ranker reads the question and each of the top 30 candidates and re-orders them by how well they answer it. It also returns the most relevant passage of each chunk, which the Sources list shows.
6. **Keep what is relevant (top-p).** The search returns up to 50 chunks with the ranker's relevance score. They are grouped by CV, and a CV counts as relevant as its best chunk. Instead of a fixed number, the best CVs are kept until they hold 80% of the relevance (`TOP_P`), then the same is done for the excerpts inside each kept CV. Each score's share is its softmax, so a clear winner takes most of it and a close field shares it. A question about one person therefore sends one or two CVs, and a broad question such as "who knows Python?" sends many. `MAX_CVS` and `MAX_CHUNKS_PER_CV` only cap the extreme case where every score is the same. A CV's excerpts stay together, in reading order, with the best CV first.
7. **Ask the chat model.** It receives the excerpts (each labelled with CV, section and page, and the first one of every CV with a line about the candidate: name, title, years, contact), the question and the last few chat messages. It is told to answer only from the excerpts, to cite `[file name, p.N]`, to answer in the language of the question, and to say so when the answer is not there. The answer streams into the chat as it is written.
8. **Complex questions go to the agent** instead of steps 2 to 7. It can list all CVs with their name, title and years of experience, search (optionally limited to some CVs or filtered by years and job title) and read a whole CV. Its steps appear in the status box while it works.
9. **Show the sources and the details:** the CVs the answer came from, and a **Details** dropdown with the route, timeline, searches and excerpts used.

Any text that comes from a CV is treated as data, not as instructions (see [Safety](#safety)).

### What happens when you open the app

`uv run streamlit run app.py` starts [app.py](app.py). Streamlit runs that file from top to bottom at startup and again on every click or message:

1. Show the login and sign-up screen until someone is logged in (a session cookie keeps you logged in).
2. Load the settings and connect to Azure (a missing `.env` value shows a clear error), and make sure this user's container and index exist.
3. Draw the sidebar: account menu, chats, upload box, **Process CVs** button and the list of CVs.
4. Draw the main area: the **Chat** or **Candidates** view.

---

## Accounts and Postgres

Users sign up and log in with an email and password. Accounts, sessions and chats are stored in Postgres, which runs in Docker:

```bash
docker compose up -d          # starts Postgres on 127.0.0.1:5432
cp .env.example .env          # DATABASE_URL is already filled in for this setup
uv run streamlit run app.py
```

Passwords are stored as argon2 hashes. A login is a random token in a browser cookie; only its hash is stored in the database, and it expires after 14 days or when you log out. The session is checked against the database on every page run, so ending it on one device (log out, password change, account deletion) also ends an already-open tab on another device at its next click. Sign-ups are capped at 15 users, because each user gets their own search index and the Azure Basic tier allows 15.

### Account menu

Your email at the top of the sidebar opens the account menu:

- **Log out** ends this session.
- **Change password** asks for the current password and logs you out on your other devices.
- **Delete my account** (you type your email and enter your password to confirm) deletes your CVs, your search index, your saved chats and your account. Azure data is deleted first: if that fails, nothing else is removed and you can try again. It waits until CVs that are being processed have finished.

### Chats

Every answer is saved. **New chat** starts a fresh one, and past chats are listed below it (newest first). Click one to reopen it with its sources and details, or use its menu to rename or delete it. The model still only sees the last few messages of a chat.

### Candidates view

The **Candidates** switch above the chat shows one card per CV with the name, job title, years of experience, location and email read from the CV. Filter by text, set a minimum number of years, sort by name or experience, open the original CV, or press **Chat with this CV** to answer only from it. It reads the metadata of the first 1000 indexed chunks, so with very many CVs some may show without details.

### Each user has their own data

Signing up creates the user's own Blob container (`<AZURE_STORAGE_CONTAINER>-<user id>`) and their own Azure AI Search index (`<AZURE_SEARCH_INDEX>-<user id>`). If either cannot be created, the account is not created either. The two `.env` values are therefore **prefixes** now.

Every call that touches Azure takes the logged-in user's workspace as an argument, and the workspace is built only from that user's id, so a user's code cannot reach another user's CVs. The answer cache and the ingest queue are per user as well. Because each user needs a search index and Azure AI Search allows only a limited number per service (15 on the Basic tier), sign-ups are capped at 15.

CVs uploaded before this change stay in the old shared container and index (the plain prefix names) and are not moved to any user. Re-upload them from a user account, then delete the old container and index in the Azure portal when you no longer need them.

### Links to the original CVs

Every chunk is stored with the plain Blob address of its CV (`file_url`) next to the file name. The containers are private, so that address does not open on its own. In the **Sources** list, each CV name is a link to a signed, read-only address that works for one hour and is made on demand for the logged-in user's own container (it needs a storage connection string that includes the account key). The address is never given to the model.

CVs indexed before this feature have no address yet. Press **Update outdated CVs** to index them again.

File names from uploads must be plain names: no `/` or `\`, no control characters, not `.` or `..`, at most 200 characters. Other files are shown as failed in the status panel.

---

## Quick start

### 1. What you need

- [uv](https://docs.astral.sh/uv/) (it installs Python and all packages for you)
- An Azure account with these resources already created:
  - a **Storage account**
  - an **Azure AI Search** service on the **Basic tier or higher** (the semantic ranker is not available on Free)
  - an **Azure OpenAI** resource with two deployments: an **embedding** model (for example `text-embedding-3-large`) and a **chat** model (for example `gpt-4.1-mini`)

### 2. Install

```bash
git clone https://github.com/sherif2004/chat-with-cv.git
cd chat-with-cv
uv sync
```

### 3. Add your Azure settings

Copy the example file and fill it in (PowerShell: `Copy-Item .env.example .env`):

```bash
cp .env.example .env
```

| Variable | Where to find it in the Azure portal |
|---|---|
| `AZURE_STORAGE_CONNECTION_STRING` | Storage account, **Security + networking**, **Access keys**, connection string |
| `AZURE_STORAGE_CONTAINER` | A name prefix you choose, for example `cvs`. Each user gets `<prefix>-<user id>`, created at sign-up |
| `AZURE_SEARCH_ENDPOINT` | AI Search service, **Overview**, **Url** |
| `AZURE_SEARCH_KEY` | AI Search service, **Settings**, **Keys**, **Primary admin key** (use the admin key, a query key is read-only) |
| `AZURE_SEARCH_INDEX` | A name prefix you choose, for example `cvs-index`. Each user gets `<prefix>-<user id>`, created at sign-up |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource, **Keys and Endpoint**. Use only `https://<name>.openai.azure.com` |
| `AZURE_OPENAI_API_KEY` | Same page, **KEY 1** |
| `AZURE_OPENAI_API_VERSION` | Keep `2024-10-21` |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | The **deployment name** of your embedding model (not the model name) |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | The **deployment name** of your chat model (not the model name) |

`DATABASE_URL` (Postgres) is already filled in for the Docker setup below. `.env` holds secrets and is git-ignored. Never commit it.

### 4. Run

```bash
docker compose up -d      # Postgres for accounts and chats
uv run streamlit run app.py
```

The browser opens automatically. Each user's search index and Blob container are created when they sign up (and checked each time they log in). The first CV also loads the Docling layout models, which are downloaded on first use, so it takes noticeably longer than the next ones.

---

## Using the app

1. **Upload.** In the sidebar, drop CVs (PDF or DOCX) into the upload box. **Process CVs** stays disabled until the knowledge base would hold at least 8 CVs (the ones already indexed plus the new files), and the chat stays disabled while there are fewer than 8.
2. **Process.** Click **Process CVs**. The files are processed in the background, 4 at a time, and a live status list in the sidebar refreshes every 2 seconds. Each file shows *waiting*, then the stage it is in (*reading layout*, *reading name, title and experience*, *embedding*, *saving*), then a green check with its chunk count, a grey check if it was already indexed and unchanged, or a red mark and the reason if it failed. A progress bar and a line such as "4 running in parallel · 3 waiting" show the whole batch. You can keep using the app while it runs; when the last file finishes the CV list updates by itself. **Clear status** removes the finished entries.
3. **Ask.** Type a question in any language, or click one of the suggested questions on the welcome screen. The answer streams in as it is written, in the language you asked in. While it works, a status box shows what is happening (the router's decision, each search, each agent step).
4. **Check the sources.** Open **Sources** under an answer to see which CVs it used and the most relevant passage of each.
5. **Check the details.** Open **Details** (the line shows the route and the total time) for tabs with: a *Timeline* of every step with its time and share of the total (router, query expansion, each search, agent rounds and tools, the model's wait for its first word, writing the answer); the *Router* result (class, standalone question, sections searched); every *Search* (query, chunks found, filters, cached or not); the *Agent* rounds and tool calls; the exact *Excerpts used*; and the *Settings* (model deployments, query expansion, answer cache). Details are kept with each message for the whole chat.
6. **Manage a CV.** Open **Manage a CV** in the sidebar, pick a CV, then:
   - **Re-index** queues the stored file for processing again (for example after the pipeline changed) and shows it in the same status list,
   - **Delete** removes it from Blob Storage and from search, after a confirmation. If this leaves fewer than 8 CVs, the chat is disabled until you add more,
   - **Update outdated CVs** checks every stored CV against the current pipeline and re-indexes only those processed by an older version (the others show as skipped).
7. **Sidebar switches.**
   - **Query expansion** searches reworded versions of each question as well (finds more, answers start 1 to 2 seconds later, three searches per question),
   - **Cache final answers** reuses the answer to an identical question in an identical chat (off by default, because a cached answer can be out of date),
   - **Clear cache** forgets everything cached by hand.
8. **Start over.** **New chat** starts a new conversation and keeps the old one in your chat list. It does not delete any CVs.
9. **Browse candidates.** Switch to **Candidates** above the chat to see every CV as a card (see [Candidates view](#candidates-view)).

Your CVs stay in Azure, so they are still there after you close the app. The sidebar lists them again the next time you open it.

---

## Project structure

```
chat-with-cv/
├── app.py                    # Streamlit entry point
├── .env.example              # template for your Azure and Postgres settings
├── docker-compose.yml        # Postgres for accounts and chats
├── .cache/                   # saved Docling output (created on first run, git-ignored)
├── pyproject.toml            # dependencies (managed by uv)
├── uv.lock                   # exact package versions
└── cv_chat/
    ├── config.py             # reads .env and holds all settings
    ├── db.py                 # Postgres connection and tables (users, sessions, chats)
    ├── auth.py               # sign up, log in, sessions, change password, delete user
    ├── history.py            # saved chats, always scoped to one user
    ├── workspace.py          # one user's container and index names, passed to every Azure call
    ├── services/             # one small file per Azure service
    │   ├── blob_storage.py       # store and list CV files
    │   ├── search_index.py       # create the index, save chunks, hybrid + semantic search
    │   └── openai_service.py     # batched embeddings, chat answers, streaming
    ├── processing/           # no Azure
    │   ├── extract.py            # Docling: PDF / DOCX to pages and headings (cached)
    │   ├── chunking.py           # pages to section-based chunks
    │   └── sections.py           # heading to standard section type
    ├── rag/                  # the two pipelines
    │   ├── cache.py              # in-memory cache of router results, searches and answers
    │   ├── metadata.py           # one model call per CV: name, title, years, contact
    │   ├── trace.py              # what happened while one question was answered, with timings
    │   ├── agent.py              # tool-using agent for complex questions (capped rounds and time)
    │   ├── ingest.py             # upload flow: skip check, extract, chunk, embed, save; delete
    │   ├── jobs.py               # background queue: runs CVs in parallel and tracks each file's state
    │   ├── qa.py                 # question flow: route, expand, search or agent, stream answer, cache
    │   └── retrieval.py          # embed, hybrid search, rank fusion, top-p selection, excerpt formatting
    └── ui/                   # Streamlit screens
        ├── accounts.py           # login and sign-up screen, session cookie, account menu
        ├── candidates.py         # the Candidates view
        ├── sidebar.py            # chats, upload, process, list of CVs
        ├── chat.py               # conversation and sources
        ├── details.py            # the Details dropdown: route, timeline, searches, agent, excerpts
        ├── safe.py               # escapes CV text and strips images before anything is drawn
        └── styles.css            # styling of the welcome screen
```

The code is split so each part has one job and imports only what is below it:

```mermaid
flowchart TD
    app["app.py"] --> ui["ui/"]
    ui --> rag["rag/"]
    rag --> services["services/<br/>Azure"]
    rag --> processing["processing/<br/>pure Python"]
    services --> config["config.py"]
```

- `services/` talks to Azure and nothing else. Each file wraps one service.
- `processing/` never touches Azure or Streamlit, so it can be tested on its own. (`extract.py` needs Docling installed, but nothing else.)
- `rag/` combines the two into the upload and question pipelines. It never imports Streamlit.
- `ui/` only draws screens and calls `rag/`.

---

## Configuration

Azure values come from `.env` (see [Quick start](#3-add-your-azure-settings)). Everything else is a constant in [cv_chat/config.py](cv_chat/config.py):

| Setting | Default | Meaning |
|---|---|---|
| `CHUNK_SIZE` | `1500` | Longest chunk in characters. A section shorter than this stays one chunk |
| `CHUNK_OVERLAP` | `300` | Characters shared between pieces of a long section (20%) |
| `DOCLING_DEVICE` | `"cpu"` | Where Docling runs. Use `"cuda"` with a GPU for much faster extraction |
| `MIN_CVS` | `8` | CVs needed in the knowledge base before processing and chatting are enabled |
| `MAX_WORKERS` | `4` | CVs processed at the same time. Lower it if Azure OpenAI reports rate limits |
| `EMBED_BATCH` | `16` | Chunks per embedding request |
| `EMBED_CONCURRENCY` | `2` | Embedding requests in flight at once, across all CVs |
| `EXTRACT_CACHE_DIR` | `.cache/extracted` | Where Docling output is saved |
| `RETRIEVE_K` | `50` | Chunks fetched and re-ranked per search, then grouped by candidate (50 is the semantic ranker's limit) |
| `TOP_P` | `0.8` | Share of the relevance that the kept CVs, and the kept excerpts of each, must cover. Higher sends more, lower sends less |
| `TOP_P_TEMPERATURE` | `0.15` | How sharply a higher score is favoured when shares are worked out. Lower lets a clear winner take more |
| `MAX_CVS` | `10` | Most candidates sent to the chat model, however flat the scores are |
| `MAX_CHUNKS_PER_CV` | `4` | Most excerpts of one candidate sent to the chat model |
| `EXPANDED_QUERIES` | `2` | Alternative queries searched when **Query expansion** is on |
| `RRF_K` | `60` | Reciprocal rank fusion constant used to merge the searches |
| `CACHE_SIZE` | `256` | Entries kept per kind of cached result (router, search, answer) |
| `METADATA_CHARS` | `12000` | Most characters of a CV read for its metadata |
| `MAX_YEARS` | `60` | A years-of-experience value above this is treated as wrong |
| `AGENT_MAX_ROUNDS` | `5` | Tool rounds the agent may take for one complex question |
| `AGENT_MAX_SECONDS` | `30` | Time budget for those rounds, then it answers with what it found |
| `AGENT_SEARCH_K` | `20` | Most excerpts one agent search hands to the model (one per CV by default, so a search reaches up to 20 CVs) |
| `AGENT_CV_CHARS` | `12000` | Most characters of one CV the `get_cv` tool hands to the model |
| `MIN_FILTERED_RESULTS` | `3` | Fewer section-filtered hits than this and the search runs again on all sections |
| `HISTORY_MESSAGES` | `6` | Recent chat messages used for the router and sent with each question |
| `EMBED_CACHE_SIZE` | `256` | Query embeddings kept in memory |

The search index settings (text analyzer `en.microsoft`, field weights, boosted sections) are constants at the top of [cv_chat/services/search_index.py](cv_chat/services/search_index.py). Changing them needs a new index (see Troubleshooting).

---

## Good to know

- **Uploading the same file again is cheap.** If the content has not changed it is skipped. If it has, its chunks are replaced and any leftover chunks from the old version are deleted.
- **Same CV under a different file name counts as a different CV.** `cv.pdf` and `cv (1).pdf` are stored separately.
- **After changing extraction or chunking code or settings**, click **Update outdated CVs** (under **Manage a CV**), or **Process CVs** with the files again. The change is detected automatically and the CVs are re-indexed, even though the files are unchanged; CVs already up to date are skipped. (Any edit to `extract.py`, `chunking.py`, `sections.py` or `metadata.py`, even a comment, counts as a change.)
- **Using an index from an older version of the app:** delete it in the Azure portal (or set a new `AZURE_SEARCH_INDEX` name) and process the CVs again. Azure cannot change an analyzer, a field's filterable flag or the vector metric on an existing index, so the app stops at start-up with a message that names what is outdated. After recreating it, click **Update outdated CVs** to index the stored CVs again.
- **The first CV is slow.** Docling loads its layout models on first use. After that, extraction takes a few seconds per CV on CPU. A GPU (`DOCLING_DEVICE = "cuda"`) is much faster.
- **Delete is permanent.** It removes the original file from Blob Storage and the CV from the index.
- **Scanned PDFs go through OCR**, which is slower than reading a text PDF. A PDF with no readable text at all is reported as "No text found".
- **DOCX files have no page numbers**, so their chunks are stored as page 1.
- **Every question makes a router call** before the search, which adds a little time before the answer starts streaming. Query expansion adds one more call and two more searches. A complex question can take several model calls, often 10 to 30 seconds.
- **Simple questions** reach at most 10 chunks, 2 per CV, so a question about a large pile of CVs may not cover every one. That is what the agent is for.
- **The search is built for English CVs.** The keyword analyzer is English, and the router writes the search query in English. Questions can be in any language, but CVs in another language would need another analyzer and a new index.
- **Years of experience are an estimate** read by a model from the dated jobs ("Present" counts as today). A CV whose years could not be read never matches a years filter.
- **Contact details are personal data.** Email, phone and location are stored as fields in the search index, next to the CV text. Anyone with the search key can read them.
- **There is no login.** Anyone who can open the app sees every CV. Put it behind your own authentication before using real candidate data.
- **The caches live in the memory of the app process.** They are empty after a restart and are not shared between several copies of the app, but every browser session of one running app shares them. Entries that depend on a conversation (router results, final answers) include the recent chat in their key, so only an identical conversation can reuse them.
- **The embedding size is read from your embedding model** when the index is first created, and cannot be changed on an existing index. To switch to a model with a different size, delete the index in the Azure portal (or set a new `AZURE_SEARCH_INDEX` name) and process the CVs again.
- **Restart Streamlit after editing `.env`.** The file is read once at startup.

---

## Chat features in detail

The reasoning, limits and costs behind each part of the chat side.

### Router and citations

- **Router.** One model call classifies each message as *chat* (greetings, thanks, off-topic: no search), *simple* (one search) or *complex* (handled by the agent), and rewrites it into a standalone English search query. It is the same call that picks the CV sections, so it adds no second call.
- **Inline citations.** Answers cite their evidence as `[file name, p.N]` next to each claim, using the page number stored with every chunk. When comparing candidates, each gets their own heading.
- *Why:* no wasted searches on "hi" or "thanks", and every claim can be traced to a CV and page.

### Query expansion

- The question is expanded into two alternative search queries (one keyword-style, one worded the way a CV would say it). Each runs the usual hybrid search with semantic re-ranking, and the lists are merged with reciprocal rank fusion. A sidebar toggle turns it on and off.
- *Why:* CVs word the same thing differently ("built APIs" versus "REST services"), so one query can miss relevant chunks.
- *Cost:* one more model call and three searches per question, so answers start about 1 to 2 seconds later. Each search uses the semantic ranker, which has a monthly query quota on lower tiers.

### Agent for complex questions

- Questions the router marks *complex* go to a capped tool-using agent with three tools:
  - `list_cvs` returns every CV with its candidate name, job title and years of experience,
  - `search_cvs` searches, optionally limited to some CVs, filtered by `min_years`, `max_years` or `job_title`, with `per_cv` to see more of each CV (one excerpt per CV by default, up to 20 per search, so one search reaches many CVs),
  - `get_cv` reads a whole CV.
- It plans its searches, retries with different wording and answers only from what the tools returned. Its steps are shown while it works, and its answer streams.
- Limits: at most 5 rounds and about 30 seconds. When the budget is used up it answers from what it has found. If the agent fails it falls back to a single search.
- There is no location filter: the location field is stored but not filterable in the index.
- *Why:* ranking or counting needs more than the 10 best chunks, for example "who has the most experience?".
- *Cost:* several model calls, often 10 to 30 seconds per complex question. Simple questions are not affected.

### Caching

- **Router results** are cached per question and recent chat, so asking again skips that model call. A message that opens a chat and that the router classifies as *chat* ("hi", "thanks") is also cached by its text alone, so it is recognised again later in any chat. Decisions made with a chat history are never shared this way.
- **Chat replies** to such messages are cached by their text alone (always on), so saying "hi" again makes no model call at all. These replies are written without the chat history, so nothing from one conversation can reach another.
- **Search results** are cached per query, section filter and CV filter, so repeats skip the searches and do not use the semantic ranker quota again.
- **Final answers** can optionally be cached too, only for an identical question with identical chat history (sidebar switch, off by default).
- **Invalidation.** The whole cache is cleared whenever a CV is processed, re-indexed or deleted, so answers never cite a CV that was removed or changed. A result computed while a CV was changing is not stored. **Clear cache** in the sidebar clears it by hand.
- The cache is in memory, per app process. The in-memory cache of query embeddings (`EMBED_CACHE_SIZE`) is separate.

### Search boosts and CV metadata

Keyword matching uses a scoring profile (`cv`): a match in the candidate name counts 3 times as much as the same words in the body text, job title and file name 2 times, the section heading 1.5 times, and chunks from the *experience* and *skills* sections get a further boost. The semantic ranker then re-orders the best results as before. The keyword analyzer is English (`en.microsoft`), and the section headings are searchable. The metadata (name, title, years, contact) is read once per CV by one model call at upload time and stored on every chunk.

### Answer language

The answer, the greeting reply and the agent's answer use the language of your latest message. The search itself runs in English. File names, candidate names, job titles, technical terms and citations stay as written in the CVs. A few fixed messages (the content-filter note) and all buttons and labels stay in English.

### Safety

CV text is written by third parties and is treated as data, not as instructions:

- **Prompts.** Excerpts and the agent's CV list are wrapped in `<cv_excerpt>` tags, tags inside CV text are removed repeatedly (so a split tag cannot rebuild a closing tag), and the system prompts say to never follow instructions found inside them.
- **Extracted metadata** has angle brackets removed, and implausible years are dropped.
- **Azure's jailbreak filter.** If it refuses a request because of an excerpt, the app finds the flagged excerpt by halving the list, answers from the rest and adds a note naming the CV. It does not fail every question that retrieves that CV.
- **Drawing.** Text from CVs, file names and models is escaped before it is shown (so `![x](https://...)` cannot load an image), and image syntax in an answer is turned into a plain link. Links in answers are still clickable.

### Latency and the Details dropdown

Every answer records a trace: the router result, query expansion, each search (cached or not, with its filters), each agent round and tool call, the wait before the model's first word, and the time to write the answer. The **Details** dropdown shows it. Searches that run in parallel overlap, so the shares in the timeline can add up to more than 100%.

What stays as it is: the section filter, the limit of 2 chunks per CV for simple questions, streamed answers, and delete and re-index.

---

## Troubleshooting

Most problems now show a message that says what to check. These are the ones you may still meet:

| What you see | What it means and what to do |
|---|---|
| `Could not reach Postgres` on the login screen | Postgres is not running. Start it with `docker compose up -d` and reload the page |
| `Could not set up your storage, so the account was not created` | Azure refused to create the new user's container or search index (check the Azure values in `.env` and the index limit of your search tier). Nothing was created, so you can try again |
| `Missing 'AZURE_...' in .env` | Lists every empty or missing value at once. Copy `.env.example` to `.env`, fill them in and restart the app |
| `The index '...' was made by an older version` | The index settings changed (English text analyzer, filterable file names, explicit cosine metric, CV metadata fields, a scoring profile, searchable section headings) and Azure cannot change these in place. Delete the index in the Azure portal, or set a new `AZURE_SEARCH_INDEX` in `.env`, restart the app, then open **Manage a CV** and click **Update outdated CVs** to index the stored CVs again |
| `Could not connect to Azure` at start-up | A value in `.env` is malformed, most often the storage connection string. Copy it again from the portal |
| `Azure OpenAI has no deployment named '...'` | Use the **deployment name** (not the model name) exactly as in the portal. The endpoint is cleaned up for you (a trailing `/openai/v1` is removed), but the API version must look like `2024-10-21` |
| `The index '...' stores vectors of length N` | The index was created with another embedding model. Delete the index in the Azure portal or set a new `AZURE_SEARCH_INDEX`, then process the CVs again |
| `Could not prepare the search index: ... older version` | See the row above: delete the index (or use a new name), restart, and click **Update outdated CVs** |
| A note says excerpts from a CV *were left out because Azure's content safety filter flagged them* | Azure's jailbreak detection saw something that looks like an instruction in that CV's text. The answer used the other excerpts. Check the CV; the app keeps working |
| `Could not answer: ...` under a question | The model call failed (a wrong deployment name, a rate limit, a network error). The message says why. Try again, or lower `MAX_WORKERS` if it happens during an upload |
| `No text found (scanned PDF?)` | OCR ran and still found nothing: the file is blank or the image is unreadable. Try a clearer copy |
| `Could not list the CVs` in the sidebar | The storage connection string or container name is wrong, or the storage account is not reachable |
| **Process CVs** is greyed out | By design: no files are selected, or the indexed plus new CVs are still fewer than 8. The sidebar says how many more are needed |
| The chat input is disabled | By design: fewer than 8 CVs are in Azure |

If the search service has no semantic ranker (Free tier, or it is switched off), the app does not fail: it logs a warning and answers with plain hybrid search. Answers are less precisely ranked and Sources show the start of each chunk instead of the best passage.


---

## Run the pipeline without the UI

Handy for debugging. Everything works on one user's data, so you need a user id:

```bash
docker compose exec postgres psql -U cvchat -c "select id, email from users"
```

Put some CVs in a local `cvs/` folder (it is git-ignored) and run (replace `<user id>`):

```bash
uv run python -c "from pathlib import Path; from cv_chat.workspace import Workspace; from cv_chat.rag import ingest; ws = Workspace('<user id>'); ingest.prepare(ws); [print(p.name, ingest.process_cv(ws, p.name, p.read_bytes())) for p in Path('cvs').iterdir()]"
```

Each CV prints its chunk count and whether it was skipped as unchanged, or the error that stopped it.

To ask a question the same way, without the UI:

```bash
uv run python -c "from cv_chat.workspace import Workspace; from cv_chat.rag import qa; a = qa.ask(Workspace('<user id>'), 'Who knows Kubernetes?', []); print(''.join(a.stream)); print(a.route, a.trace.to_dict()['total_ms'], 'ms')"
```
