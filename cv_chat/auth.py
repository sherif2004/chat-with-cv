"""Accounts: sign up, log in, and the sessions that keep a user logged in.

Passwords are stored as argon2 hashes. A session token is random; only its SHA-256 hash is stored,
so a leaked database does not give anyone a working login.

Hashing takes about 50 ms, so it never happens while a database connection is held: the pool is small and every page run
asks it for the session. Failed logins, password checks and recovery-code guesses are counted per email (or per user) and
refused for a while once there are too many, so a password cannot be guessed without limit. An attempt is counted before
the password is checked, so a burst of guesses cannot slip through together.
"""
import hashlib
import hmac
import math
import re
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError

from cv_chat import db
from cv_chat.account_config import accounts as cfg


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


# ---- too many failures: a key is "login:<email>", "reset:<email>" or "password:<user id>"

def _reserve_attempt(key: str) -> None:
    """Count this attempt now, before the password is checked, and refuse it if the key has already used up its attempts.

    Counting first (under a lock for the key) is what makes the limit hold when many guesses arrive at once: each one sees the
    others. The attempt stays counted if it fails and is wiped by _clear_failures if it succeeds. A key is locked while it has
    max_failed_attempts attempts newer than lockout_minutes, and is free again when the oldest of them is that old.
    """
    with db.pool().connection() as conn:
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (key,))
        conn.execute("DELETE FROM auth_failures WHERE failed_at < now() - make_interval(mins => %s)", (cfg.lockout_minutes,))
        recent = conn.execute(
            "SELECT failed_at FROM auth_failures WHERE key = %s ORDER BY failed_at DESC LIMIT %s", (key, cfg.max_failed_attempts)
        ).fetchall()
        if len(recent) >= cfg.max_failed_attempts:
            free_at = recent[-1][0] + timedelta(minutes=cfg.lockout_minutes)
            minutes = max(1, math.ceil((free_at - datetime.now(free_at.tzinfo)).total_seconds() / 60))
            raise AuthError(f"Too many failed attempts. Try again in about {minutes} minute{'' if minutes == 1 else 's'}.")
        conn.execute("INSERT INTO auth_failures (key) VALUES (%s)", (key,))


def _clear_failures(key: str) -> None:
    with db.pool().connection() as conn:
        conn.execute("DELETE FROM auth_failures WHERE key = %s", (key,))


def _password_matches(stored_hash: str | None, password: str) -> bool:
    """argon2 check without a database connection. An unknown account is checked against a dummy hash, so timing does not reveal it."""
    try:
        _hasher.verify(stored_hash or _DUMMY_HASH, password)
    except VerificationError:
        return False
    return stored_hash is not None


def sign_up(email: str, password: str, on_created: Callable[[User], None] | None = None) -> User:
    """Create an account. on_created sets up the user's other resources (their Azure container and index); if it fails,
    the account is removed again, so there are no half-made accounts."""
    email = email.strip().lower()
    if not _EMAIL.match(email) or len(email) > 254:
        raise AuthError("Enter a valid email address.")
    if len(password) < cfg.min_password:
        raise AuthError(f"The password needs at least {cfg.min_password} characters.")
    password_hash = _hasher.hash(password)
    with db.pool().connection() as conn:
        conn.execute("SELECT pg_advisory_xact_lock(1)")  # one sign-up at a time, so the user cap cannot be passed
        if conn.execute("SELECT count(*) FROM users").fetchone()[0] >= cfg.max_users:
            raise AuthError(f"Sign-ups are closed: this demo allows {cfg.max_users} users.")
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
    if len(new_password) < cfg.min_password:
        raise AuthError(f"The new password needs at least {cfg.min_password} characters.")
    key = f"reset:{email}"
    _reserve_attempt(key)
    with db.pool().connection() as conn:
        row = conn.execute("SELECT id, recovery_hash FROM users WHERE email = %s", (email,)).fetchone()
    stored = row[1] if row and row[1] else _digest("no-recovery-code")  # compared anyway, so timing does not show who exists
    matches = hmac.compare_digest(stored, _digest(_normalise_code(code)))
    if row is None or not row[1] or not matches:
        raise AuthError("Wrong email or recovery code.")
    new_hash = _hasher.hash(new_password)
    with db.pool().connection() as conn:
        # only if the code is still the one that was checked: a code works once, also when two requests race
        changed = conn.execute(
            "UPDATE users SET password_hash = %s WHERE id = %s AND recovery_hash = %s RETURNING id", (new_hash, row[0], row[1])
        ).fetchone()
        if changed is None:
            raise AuthError("Wrong email or recovery code.")
        conn.execute("DELETE FROM sessions WHERE user_id = %s", (row[0],))
    _clear_failures(key)
    _clear_failures(f"login:{email}")  # the recovery code proves who this is: a lock on the login must not keep them out
    return issue_recovery_code(str(row[0]))


def check_password(user_id: str, password: str) -> bool:
    """True if this is the user's password. For actions that must not rest on a session alone, such as deleting the account.
    Raises AuthError when there have been too many wrong attempts."""
    key = f"password:{user_id}"
    _reserve_attempt(key)
    with db.pool().connection() as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE id = %s", (user_id,)).fetchone()
    if not _password_matches(row[0] if row else None, password):
        return False
    _clear_failures(key)
    return True


def change_password(user_id: str, current: str, new: str, keep_token: str | None = None) -> None:
    """Set a new password after checking the current one. Every other session of the user is ended; keep_token stays."""
    if len(new) < cfg.min_password:
        raise AuthError(f"The new password needs at least {cfg.min_password} characters.")
    key = f"password:{user_id}"
    _reserve_attempt(key)
    with db.pool().connection() as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE id = %s", (user_id,)).fetchone()
    if not _password_matches(row[0] if row else None, current):
        raise AuthError("The current password is wrong.")
    new_hash = _hasher.hash(new)
    with db.pool().connection() as conn:
        changed = conn.execute(
            "UPDATE users SET password_hash = %s WHERE id = %s AND password_hash = %s RETURNING id", (new_hash, user_id, row[0])
        ).fetchone()
        if changed is None:  # changed somewhere else while this request was hashing
            raise AuthError("Your password was changed somewhere else. Try again.")
        conn.execute(
            "DELETE FROM sessions WHERE user_id = %s AND token_hash <> %s", (user_id, _digest(keep_token) if keep_token else "")
        )
    _clear_failures(key)


def delete_user(user_id: str) -> None:
    """Remove an account and its sessions. The caller removes the user's Azure data."""
    with db.pool().connection() as conn:
        conn.execute("DELETE FROM users WHERE id = %s", (user_id,))


def log_in(email: str, password: str) -> str:
    """Check the password and return a new session token to store in the browser."""
    email = email.strip().lower()
    key = f"login:{email}"
    _reserve_attempt(key)
    with db.pool().connection() as conn:
        row = conn.execute("SELECT id, password_hash FROM users WHERE email = %s", (email,)).fetchone()
    if not _password_matches(row[1] if row else None, password):
        raise AuthError("Wrong email or password.")
    _clear_failures(key)
    token = secrets.token_urlsafe(32)
    with db.pool().connection() as conn:
        conn.execute(
            "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (%s, %s, now() + make_interval(days => %s))",
            (_digest(token), row[0], cfg.session_days),
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
