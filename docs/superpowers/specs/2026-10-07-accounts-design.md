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
