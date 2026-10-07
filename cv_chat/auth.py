"""Accounts: sign up, log in, and the sessions that keep a user logged in.

Passwords are stored as argon2 hashes. A session token is random; only its SHA-256 hash is stored,
so a leaked database does not give anyone a working login.
"""
import hashlib
import hmac
import re
import secrets
from collections.abc import Callable
from dataclasses import dataclass

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError

from cv_chat import db

MAX_USERS = 15  # one Azure AI Search index per user, and the Basic tier allows 15
MIN_PASSWORD = 8
SESSION_DAYS = 14

_hasher = PasswordHasher()
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_DUMMY_HASH = _hasher.hash("not-a-real-password")  # checked for unknown emails, so timing does not reveal which exist


class AuthError(ValueError):
    """A problem to show the user as it is."""


@dataclass(frozen=True)
class User:
    id: str
    email: str


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def sign_up(email: str, password: str, on_created: Callable[[User], None] | None = None) -> User:
    """Create an account. on_created sets up the user's other resources (their Azure container and index); if it fails,
    the account is removed again, so there are no half-made accounts."""
    email = email.strip().lower()
    if not _EMAIL.match(email) or len(email) > 254:
        raise AuthError("Enter a valid email address.")
    if len(password) < MIN_PASSWORD:
        raise AuthError(f"The password needs at least {MIN_PASSWORD} characters.")
    password_hash = _hasher.hash(password)
    with db.pool().connection() as conn:
        conn.execute("SELECT pg_advisory_xact_lock(1)")  # one sign-up at a time, so the user cap cannot be passed
        if conn.execute("SELECT count(*) FROM users").fetchone()[0] >= MAX_USERS:
            raise AuthError(f"Sign-ups are closed: this demo allows {MAX_USERS} users.")
        if conn.execute("SELECT 1 FROM users WHERE email = %s", (email,)).fetchone():
            raise AuthError("An account with this email already exists.")
        user_id = conn.execute(
            "INSERT INTO users (email, password_hash) VALUES (%s, %s) RETURNING id", (email, password_hash)
        ).fetchone()[0]
    user = User(str(user_id), email)
    if on_created is not None:
        try:
            on_created(user)
        except Exception as error:
            delete_user(user.id)
            raise AuthError(f"Could not set up your storage, so the account was not created: {str(error).splitlines()[0][:200]}") from error
    return user


_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O or 1/I, so a code copied by hand is not misread
_CODE_LENGTH = 16  # 32 symbols each: 80 bits, far too many to guess


def _normalise_code(code: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", code.upper())


def issue_recovery_code(user_id: str) -> str:
    """Make a new one-time recovery code (the old one stops working) and return it. Only its hash is stored, so it can
    be shown to the user this once and never again."""
    raw = "".join(secrets.choice(_CODE_ALPHABET) for _ in range(_CODE_LENGTH))
    with db.pool().connection() as conn:
        conn.execute("UPDATE users SET recovery_hash = %s WHERE id = %s", (_digest(raw), user_id))
    return "-".join(raw[i : i + 4] for i in range(0, _CODE_LENGTH, 4))


def reset_password(email: str, code: str, new_password: str) -> str:
    """Set a new password for someone who forgot theirs, using the recovery code from sign-up. Every session of the user
    is ended and a new recovery code is returned (the one just used no longer works)."""
    email = email.strip().lower()
    if len(new_password) < MIN_PASSWORD:
        raise AuthError(f"The new password needs at least {MIN_PASSWORD} characters.")
    with db.pool().connection() as conn:
        row = conn.execute("SELECT id, recovery_hash FROM users WHERE email = %s", (email,)).fetchone()
        stored = row[1] if row and row[1] else _digest("no-recovery-code")  # compared anyway, so timing does not show who exists
        matches = hmac.compare_digest(stored, _digest(_normalise_code(code)))
        if row is None or not row[1] or not matches:
            raise AuthError("Wrong email or recovery code.")
        conn.execute("UPDATE users SET password_hash = %s WHERE id = %s", (_hasher.hash(new_password), row[0]))
        conn.execute("DELETE FROM sessions WHERE user_id = %s", (row[0],))
    return issue_recovery_code(str(row[0]))


def check_password(user_id: str, password: str) -> bool:
    """True if this is the user's password. For actions that must not rest on a session alone, such as deleting the account."""
    with db.pool().connection() as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE id = %s", (user_id,)).fetchone()
    try:
        _hasher.verify(row[0] if row else _DUMMY_HASH, password)
    except VerificationError:
        return False
    return row is not None


def change_password(user_id: str, current: str, new: str, keep_token: str | None = None) -> None:
    """Set a new password after checking the current one. Every other session of the user is ended; keep_token stays."""
    if len(new) < MIN_PASSWORD:
        raise AuthError(f"The new password needs at least {MIN_PASSWORD} characters.")
    with db.pool().connection() as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE id = %s", (user_id,)).fetchone()
        try:
            _hasher.verify(row[0] if row else _DUMMY_HASH, current)
        except VerificationError:
            row = None
        if row is None:
            raise AuthError("The current password is wrong.")
        conn.execute("UPDATE users SET password_hash = %s WHERE id = %s", (_hasher.hash(new), user_id))
        conn.execute(
            "DELETE FROM sessions WHERE user_id = %s AND token_hash <> %s", (user_id, _digest(keep_token) if keep_token else "")
        )


def delete_user(user_id: str) -> None:
    """Remove an account and its sessions. The caller removes the user's Azure data."""
    with db.pool().connection() as conn:
        conn.execute("DELETE FROM users WHERE id = %s", (user_id,))


def log_in(email: str, password: str) -> str:
    """Check the password and return a new session token to store in the browser."""
    email = email.strip().lower()
    with db.pool().connection() as conn:
        row = conn.execute("SELECT id, password_hash FROM users WHERE email = %s", (email,)).fetchone()
        try:
            _hasher.verify(row[1] if row else _DUMMY_HASH, password)
        except VerificationError:
            row = None
        if row is None:
            raise AuthError("Wrong email or password.")
        token = secrets.token_urlsafe(32)
        conn.execute(
            "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (%s, %s, now() + make_interval(days => %s))",
            (_digest(token), row[0], SESSION_DAYS),
        )
        conn.execute("DELETE FROM sessions WHERE expires_at < now()")
    return token


def user_for_token(token: str | None) -> User | None:
    if not token:
        return None
    with db.pool().connection() as conn:
        row = conn.execute(
            "SELECT u.id, u.email FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash = %s AND s.expires_at > now()",
            (_digest(token),),
        ).fetchone()
    return User(str(row[0]), row[1]) if row else None


def log_out(token: str | None) -> None:
    if token:
        with db.pool().connection() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = %s", (_digest(token),))
