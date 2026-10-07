import pytest

from cv_chat import auth, db


def test_sign_up_then_log_in():
    user = auth.sign_up("  Ada@Example.com ", "correct horse")
    assert user.email == "ada@example.com"
    token = auth.log_in("ADA@example.com", "correct horse")
    assert auth.user_for_token(token) == user


def test_password_is_stored_hashed():
    auth.sign_up("a@b.co", "correct horse")
    with db.pool().connection() as conn:
        stored = conn.execute("SELECT password_hash FROM users").fetchone()[0]
    assert "correct horse" not in stored and stored.startswith("$argon2")


def test_duplicate_email_rejected():
    auth.sign_up("a@b.co", "correct horse")
    with pytest.raises(auth.AuthError, match="already exists"):
        auth.sign_up("A@B.co", "another pass")


@pytest.mark.parametrize("email,password", [("nope", "correct horse"), ("a@b.co", "short")])
def test_invalid_sign_up_rejected(email, password):
    with pytest.raises(auth.AuthError):
        auth.sign_up(email, password)


def test_wrong_password_and_unknown_email_look_the_same():
    auth.sign_up("a@b.co", "correct horse")
    for email, password in [("a@b.co", "wrong pass!"), ("x@b.co", "correct horse")]:
        with pytest.raises(auth.AuthError, match="Wrong email or password"):
            auth.log_in(email, password)


def test_token_is_stored_only_as_hash():
    auth.sign_up("a@b.co", "correct horse")
    token = auth.log_in("a@b.co", "correct horse")
    with db.pool().connection() as conn:
        stored = conn.execute("SELECT token_hash FROM sessions").fetchone()[0]
    assert token not in stored


def test_expired_session_is_not_accepted():
    auth.sign_up("a@b.co", "correct horse")
    token = auth.log_in("a@b.co", "correct horse")
    with db.pool().connection() as conn:
        conn.execute("UPDATE sessions SET expires_at = now() - interval '1 second'")
    assert auth.user_for_token(token) is None


def test_log_out_ends_the_session():
    auth.sign_up("a@b.co", "correct horse")
    token = auth.log_in("a@b.co", "correct horse")
    auth.log_out(token)
    assert auth.user_for_token(token) is None


def test_unknown_or_missing_token():
    assert auth.user_for_token(None) is None
    assert auth.user_for_token("garbage") is None


def test_user_cap(monkeypatch):
    monkeypatch.setattr(auth, "MAX_USERS", 2)
    auth.sign_up("a@b.co", "correct horse")
    auth.sign_up("b@b.co", "correct horse")
    with pytest.raises(auth.AuthError, match="closed"):
        auth.sign_up("c@b.co", "correct horse")


def test_failed_setup_removes_the_new_account():
    def broken(user):
        raise RuntimeError("Azure said no")

    with pytest.raises(auth.AuthError, match="account was not created"):
        auth.sign_up("a@b.co", "correct horse", on_created=broken)
    with db.pool().connection() as conn:
        assert conn.execute("SELECT count(*) FROM users").fetchone()[0] == 0
    auth.sign_up("a@b.co", "correct horse")  # the email and the user slot are free again


def test_setup_runs_with_the_new_user():
    seen = []
    user = auth.sign_up("a@b.co", "correct horse", on_created=seen.append)
    assert seen == [user]


def test_delete_user_removes_sessions():
    user = auth.sign_up("a@b.co", "correct horse")
    token = auth.log_in("a@b.co", "correct horse")
    auth.delete_user(user.id)
    assert auth.user_for_token(token) is None
