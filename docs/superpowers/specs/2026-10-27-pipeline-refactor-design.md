# Pipeline refactor: one validated settings object, top-k retrieval, sequential entity reading, self-migrating index

Status: draft for approval. Nothing is built yet.

## Why

The pipeline works, but its tunables are spread over `config.py` and many module-level constants (`INDEX_VERSION`,
`MAX_NAME_LENGTH`, NER thresholds, section regexes, cache sizes, batch sizes, the session length, and so on), and a few
behaviours are more complicated than the owner wants:

- two retrieval modes (top-p ranking with a character budget, and top-k) where only top-k is wanted;
- the file id is an md5 of the file name, which hides what it is;
- extraction silently drops page numbers and footers;
- candidate details are read by hunting for the email and cutting a "header block" around it;
- an index created by an older version needs a manual delete in the Azure portal;
- two users chatting at the same time can slow each other down through shared Azure rate-limit gates.

## Goals

1. One pydantic settings object holds every static value; all code reads from it; bad or missing values fail at start-up
   with one clear message.
2. Retrieval uses top-k only. `TOP_P` and `CONTEXT_CHARS` are gone.
3. The file id is the raw file name.
4. Extraction keeps everything Document Intelligence returns.
5. Chunking is section-aware: one chunk per section.
6. Candidate details are read in document order, header first, then each following section, until every field is filled.
7. The search index migrates itself: additive changes in place, breaking changes by rebuilding it from the stored originals.
8. Two (or more) users can chat at the same time without waiting on each other or on someone else's ingestion.

## Non-goals

- Years of experience stays out (as today).
- No change to auth, chat history schema, or the UI layout. UI code changes only where it reads a setting.
- No new Azure services. Postgres stays as is.
- Not fixing, in this project, the other problems noticed in review (`list_profiles` and `cvs_mentioning` read at most 1000
  chunks; the in-memory caches are per process). They are listed under "Noticed, not doing".

## Anti-goals (this would be a failure even if it runs)

- A CV disappearing from search or from Blob Storage during an automatic index rebuild.
- One user's data reachable by another (the per-user container and index must stay).
- A setting that is read from a different place in two modules.

## Design

### 1. Settings (`cv_chat/config.py`, pydantic-settings)

The file keeps its name. It defines nested pydantic models and one `settings` instance, loaded from `.env` and the
environment. Nested groups (names are a suggestion for the builder):

| Group | Holds |
|---|---|
| `azure` | storage connection string, container prefix; search endpoint, key, index prefix; Document Intelligence endpoint, key; OpenAI endpoint, key, API version, embedding / chat / small deployments |
| `retrieval` | `top_k`, `top_k_focused` (see open question 1), `search_k` (50), `cover_rounds`, `rrf_k`, `expanded_queries`, `history_chars` |
| `chunking` | `max_chars`, `overlap`, caps-heading fallback on/off, its max words and length |
| `extraction` | `cache_dir`, `extract_version` (part of the cache key), roles kept (all) |
| `entities` | model name, threshold, window words, labels, fallback labels |
| `sections` | section type to keywords table (see 5), boosted section types |
| `index` | analyzer, semantic config name, scoring profile and weights, vector metric, upload batch size |
| `ingest` | `pipeline_version`, file name limits, worker pool size |
| `agent` | max rounds, max seconds, per-CV result limits |
| `concurrency` | see 8 |
| `auth` | `max_users`, `min_password`, `session_days` |
| `cache` | sizes |

Rules:

- Every field has a type and a bound (`PositiveInt`, `Field(ge=..., le=...)`); cross-field checks use validators (for
  example `overlap < max_chars`, `top_k >= 1`, the semantic ranker's 50-result limit for `search_k`).
- Secrets are `SecretStr`.
- Tunables can be overridden by environment variables (`CVCHAT_RETRIEVAL__TOP_K=10`); Azure values keep their present
  `.env` names so `.env.example` barely changes.
- Postgres settings stay separate and load without the Azure values, as `db.py` does today, so the login screen works
  when Azure is not configured. `app.py` catches pydantic's `ValidationError` where it now catches `KeyError`, and shows
  the same screen listing every bad value.
- A pytest checks that no module defines a tunable of its own (a grep-style guard on the old constant names).

### 2. Retrieval: top-k only, whole CVs

- Delete `TOP_P`, `CONTEXT_CHARS`, `rank_cvs`, `_nucleus`, `_temperature`, and the character budget in `rank_top_k`.
- Every question: search (with the existing query expansion, fusion and cover rounds), rank CVs by their best chunk, keep
  the first `k`, then load **every chunk of each kept CV** (`search_index.get_cv_chunks`) in reading order and give them to
  the model, best CV first.
- The Sources list and the stored `sources` stay small: they hold the chunks the search matched, not the whole CVs.
- Agent: `search_cvs` keeps its per-CV limit but loses the character budget; `get_cv` returns the whole CV.
- Cost note: with the cap gone, a question carries up to `k` complete CVs (roughly 5 to 10k characters each).

### 3. File id is the raw file name

- `file_id` (a filterable field) holds the file name as uploaded. `file_id_for` becomes the identity function and is
  removed from call sites where the name is already at hand.
- Azure keys cannot contain dots or spaces, so the chunk key `id` is `urlsafe_base64(utf-8 name) + "-" + chunk number`.
  File names are limited to 255 bytes of UTF-8 so the key stays under Azure's 1024 limit. The key is never shown.
- Filters: single quotes are doubled in OData strings; for several files use `search.in` with a delimiter that does not
  occur in any of the names, or fall back to ORed `eq` clauses.
- Existing indexes are not compatible (ids and `file_id` change). This is handled by the automatic rebuild in 7.
- `validate_file_name` keeps its present rules and gets its limits from settings.

### 4. Extraction keeps everything (`processing/extract.py`)

- Stop skipping `pageNumber` and `pageFooter` paragraphs; keep every role Document Intelligence returns.
- Keep tables as markdown (their cell text is not repeated as paragraphs, so nothing is lost or doubled).
- Also keep figure captions and selection marks (checkbox states) if the result contains them.
- The disk cache key becomes `md5(file bytes) + extract_version`, so changing what is kept actually takes effect for CVs
  that are already cached. Today the key is the bytes only.
- Known gap, not in this design unless asked (open question 2): the text of a hyperlink is read but its URL target is not.

### 5. Chunking and sections

- `chunk_cv`: one chunk per section (heading plus its body). A section is split only when it is longer than
  `chunking.max_chars`, at line breaks, with the heading repeated and `chunking.overlap` characters of overlap.
- Headings come from the layout model; the short ALL-CAPS line fallback stays but is a setting. Text before the first
  heading is the "header" section.
- `sections.py`: the regexes stay (section filtering stays), but they become data in `settings.sections` (type to keyword
  list) and are compiled once. Proposed clean-up, to be confirmed while reviewing the table:
  - match whole words or word starts, not substrings;
  - check order fixed so "Programming Languages" is `skills`, not `languages`;
  - drop vague keywords (`about`, `school`, `course`, `personal`) that match unrelated headings.

### 6. Candidate details read in order (`processing/entities.py`)

- Remove the email-anchored header block and the page-1-only limit.
- Input is the document's sections in order (the output of the chunker), the header section first.
- For each section, in order, ask GLiNER (one label at a time, as today) only for the fields still empty; keep the first
  answer above threshold; stop as soon as all five fields are filled or the CV ends.
- The small repairs stay where they are still meaningful (widen a cut word, name and title sharing a line, ignore a
  "phone" answer that is the email); rules that depended on the header block are dropped.
- Risk: a CV whose header lacks a location may take one from a later section (for example a university). Mitigation: for
  `location`, accept only answers from the header section and sections of type `summary`, `contact` or `experience`.
- Chunking runs before entity reading, and the embeddings still get the candidate line, so the order in `process_cv` is:
  extract, chunk, read entities, embed.

### 7. Index that migrates itself (`services/search_index.py`)

- The desired index is built from code plus settings (fields, analyzer, vector dimensions and metric, scoring profile,
  semantic config). A stored schema signature (a hash of that definition and `pipeline_version`) is compared with the
  existing index at start-up and before ingestion.
- Additive change (a new field, a new semantic configuration): update in place, as today.
- Breaking change (analyzer, filterable flag, vector dimensions or metric, scoring profile, key scheme): delete the index,
  create the new one, and queue every CV found in Blob Storage for re-indexing in the background. The originals are never
  touched, and the extraction cache makes Document Intelligence calls free for files already read.
- While a rebuild runs the sidebar shows progress (the existing queue) and chat tells the user that N CVs are being
  re-indexed. A per-user lock prevents two sessions rebuilding at once.
- Replaces both the "delete the index in the portal" errors and the source-code hash in `_pipeline_fingerprint`. The
  fingerprint becomes a hash of the pipeline-relevant settings plus the hand-bumped `pipeline_version`, so editing a
  comment no longer re-indexes everything. Trade-off: after changing processing logic, `pipeline_version` must be bumped.

### 8. Concurrency (`services/openai_service.py`)

Today the embedding gate is shared by everyone and chat calls have no limit of their own.

- Separate gates: an **interactive** gate (question embeddings, router, expansion, answer, agent turns) and a
  **background** gate (ingestion embeddings), both adaptive (halve on 429, grow after a run of successes).
- Ingestion can never use the slots reserved for interactive calls (`concurrency.interactive_reserved`).
- A per-user cap on in-flight chat calls (`concurrency.per_user_chat`) so one user's agent run cannot starve another.
- Chat calls get the same 429 handling as embeddings (honour `Retry-After`, bounded retries) instead of relying on the
  SDK's blind retries, and the gate is held until the first byte, not for the whole streamed answer.
- All numbers are settings.
- Acceptance test: two scripted sessions ask questions at the same time while a third ingests 10 CVs; neither question
  waits on the ingestion, no 429 reaches the user, and time to first token for each is within the same order as when asked alone.

### 9. `workspace.py` (explanation, no behaviour change)

A `Workspace` is built from the logged-in user's UUID and carries the names of that user's Blob container and search
index. Every Azure call takes it, so code can only reach the user's own data. It stays exactly that; only the name
prefixes come from `settings`. The ingest queue and caches are already keyed by user.

## Constraints

- Python and Streamlit stay; add `pydantic` and `pydantic-settings` to `requirements.txt`.
- Azure AI Search key characters, semantic ranker's 50-result limit, Basic tier 15 indexes (hence `max_users`).
- The repo has no tests today; the plan adds some (settings validation, section matching, chunking, key encoding, filter
  escaping, ranking of CVs) and a concurrency test with the Azure calls faked.

## Left to the builder

Names of the settings classes and fields, module layout, how the concurrency gates are shared, the exact delimiter logic
for `search.in`, and the migration order of the work.

## Calls I made (please check)

- Keep the file name `config.py`; expose one `settings` object.
- Source-hash fingerprint replaced by settings hash + `pipeline_version`.
- Cache key for extraction includes `extract_version`.
- Chunk key is the base64 of the name plus a number; names limited to 255 bytes.
- Entity reading uses the chunker's sections as its windows; location restricted to header, summary, contact, experience.
- Section keyword clean-up list in 5.
- Sources stored in chat history stay the matched chunks only.

## Decisions made while building (the owner did not answer the open questions, so the recommendations were used)

- Open question 1: two values, `retrieval.top_k` (20, questions about everyone) and `retrieval.top_k_focused` (3, any other).
- Open question 2: hyperlink URLs are not recovered (no new dependency).
- The index is always replaced when its definition changes; the "update additive changes in place" path was dropped because
  removing the old marker field is itself a breaking change in Azure and one rule is simpler. Adding a field therefore
  rebuilds the index from the stored CVs, which costs embeddings and NER time but not Document Intelligence calls.
- Index definition is recognised by a marker field `cvschema_<hash>` instead of a stored signature (an Azure index cannot carry one).
- `index.version` (stored fields and keys) and `ingest.pipeline_version` (how a CV is processed) are two numbers.
- Chunking defaults changed to `max_chars` 4000 and `overlap` 400, since a section is meant to stay one chunk.
- Account and database settings are in `cv_chat/account_config.py`, so the login screen needs no Azure values; the minimum
  password length can only be raised above 8.
- Chunks written under the old md5 id scheme are removed by file name when a CV is re-indexed or deleted, and a rebuild removes them all.
- A limit is held until the response has started, including the per-user cap; a user waiting more than
  `concurrency.user_wait_seconds` goes on anyway rather than hang.
- Not done: `list_profiles` and `cvs_mentioning` still read at most 1000 chunks (see "Noticed, not doing").
- Known cost: with no character budget, a question carries up to `k` complete CVs. Lower `top_k` / `top_k_focused` or choose CVs in the chat to limit tokens.

## Open questions for the owner (answered above)

1. A question about one person would carry `k` whole CVs (default 20). Keep one `top_k` for every question, or use
   `top_k` for "everyone" questions and a small `top_k_focused` (suggested 3) for the rest? Both are top-k only.
2. Should extraction also recover hyperlink URLs (LinkedIn, GitHub) from the PDF? It needs `pypdf` and a few lines in
   `extract.py`; without it, only the link text is kept.

## Noticed, not doing

- `list_profiles` and `cvs_mentioning` use `top=1000` over chunks, so CVs can be missing from the role check and `list_cvs`
  once an index holds more than 1000 chunks.
- Caches and the NER lock are per process; running several workers needs shared storage.
- No throttling of login or reset attempts.
