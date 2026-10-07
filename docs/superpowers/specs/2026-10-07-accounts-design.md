# Accounts, per-user data, history and NER — design

Goal: several people use the app, each with their own CVs, search index and chat history. Scale: demo, under 15 users.

## Decisions
- **Auth:** custom, on Postgres (Docker, local). argon2 password hashes; random session token in a cookie, only its hash stored.
- **Isolation:** one Blob container (`cv-<user_id>`) and one Azure AI Search index (`cvs-<user_id>`) per user. Sign-ups capped at 15 (Basic tier index limit).
- **History:** conversations, messages and an ingest log in Postgres.
- **Candidate info:** spaCy NER + regex + date rules replace the LLM metadata call. Same output fields, so the index schema is unchanged.
- **UI:** Streamlit stays. Login/sign-up screen, conversation list in the sidebar.
- Login is email + password only. No email verification or password reset yet.

## Sub-projects (built in this order)
1. Postgres in Docker, `users`/`sessions`, login and sign-up. **(this branch, first)**
2. Per-user Azure resources: `blob_storage` and `search_index` take a user instead of using module globals; per-user caches; provisioning on sign-up with rollback.
3. Chat history and ingest log in Postgres.
4. NER extraction (`cv_chat/processing/entities.py`) replacing `rag/metadata.py`.
5. UI polish.

## Schema (sub-project 1)
- `users(id uuid pk, email citext unique, password_hash text, created_at)`
- `sessions(token_hash text pk, user_id fk, created_at, expires_at)`

## Security notes
- Passwords: argon2id via `argon2-cffi`. Generic "wrong email or password" message. Minimum length 8.
- Session token: 32 random bytes, stored as SHA-256 hash; 14-day expiry; deleted on logout.
- The shared in-memory answer/route cache must become per-user in sub-project 2, or one user's answer could be served to another.

## Testing
pytest against the Docker Postgres for the auth module (sign-up, duplicate email, login, wrong password, session expiry, logout, user cap).

## UI and history (approved 2026-10-07)
Built in this order; each step is its own commit(s).

- **A. Chat history + conversation list.** Tables `conversations(id, user_id, title, created_at, updated_at)` and `messages(id, conversation_id, role, content, sources jsonb, route, trace jsonb, created_at)`, both removed with the user. `cv_chat/history.py` takes the user id in every function and puts it in the SQL, so a conversation id alone never opens anyone's chat. Sidebar: New chat, a list of past chats (newest first, 30), rename and delete per chat; title = first question cut to 60 characters. The model still gets only the last `HISTORY_MESSAGES` messages.
- **B. Account area.** Log out, change password (current password required; other sessions are logged out), delete account (type the email; Azure data is deleted first and a failure leaves the database untouched).
- **C. Candidates view.** A Chat | Candidates switch; one card per CV from the indexed metadata, with filter, minimum years, sort, Open CV (signed link) and Chat with this CV. Only the selected view is drawn. Limit: `list_profiles` reads the first 1000 chunks.
- **D. Visual polish.** Centered login card, theme in `.streamlit/config.toml` and `styles.css` for light and dark, better empty states.

Checking the screens: a throwaway launcher outside the repo fakes Azure with in-memory data. It is not committed and proves nothing about real Azure.
