import pytest

from cv_chat import auth

PASSWORD = "correct horse battery"


@pytest.fixture
def user(clean):
    return auth.sign_up("jane@example.com", PASSWORD)


def fail_login(times, email="jane@example.com"):
    for _ in range(times):
        with pytest.raises(auth.AuthError, match="Wrong email or password"):
            auth.log_in(email, "wrong password!")


def age_failures(clean, minutes):
    with clean.pool().connection() as conn:
        conn.execute("UPDATE auth_failures SET failed_at = failed_at - make_interval(mins => %s)", (minutes,))


# --- logging in --------------------------------------------------------------------------------------------------------

def test_a_user_can_log_in_and_the_token_identifies_them(user):
    token = auth.log_in("JANE@example.com ", PASSWORD)
    assert auth.user_for_token(token) == user


def test_a_wrong_password_or_an_unknown_email_gets_the_same_answer(user):
    for email, password in [("jane@example.com", "nope nope nope"), ("nobody@example.com", PASSWORD)]:
        with pytest.raises(auth.AuthError, match="Wrong email or password"):
            auth.log_in(email, password)


# --- throttling --------------------------------------------------------------------------------------------------------

def test_after_too_many_failures_even_the_right_password_is_refused(user):
    fail_login(auth.cfg.max_failed_attempts)
    with pytest.raises(auth.AuthError, match="Too many failed attempts"):
        auth.log_in("jane@example.com", PASSWORD)


def test_the_message_says_roughly_how_long_to_wait(user):
    fail_login(auth.cfg.max_failed_attempts)
    with pytest.raises(auth.AuthError, match=rf"about {auth.cfg.lockout_minutes} minutes"):
        auth.log_in("jane@example.com", PASSWORD)


def test_one_email_being_locked_does_not_lock_another(user):
    auth.sign_up("omar@example.com", PASSWORD)
    fail_login(auth.cfg.max_failed_attempts)
    assert auth.log_in("omar@example.com", PASSWORD)


def test_an_unknown_email_is_throttled_too_so_the_lock_does_not_reveal_who_exists(clean):
    fail_login(auth.cfg.max_failed_attempts, email="ghost@example.com")
    with pytest.raises(auth.AuthError, match="Too many failed attempts"):
        auth.log_in("ghost@example.com", PASSWORD)


def test_a_good_login_clears_the_failures(user):
    fail_login(auth.cfg.max_failed_attempts - 1)
    assert auth.log_in("jane@example.com", PASSWORD)
    fail_login(auth.cfg.max_failed_attempts - 1)  # would have locked without the reset
    assert auth.log_in("jane@example.com", PASSWORD)


def test_the_lock_ends_when_the_failures_are_old_enough(clean, user):
    fail_login(auth.cfg.max_failed_attempts)
    age_failures(clean, auth.cfg.lockout_minutes + 1)
    assert auth.log_in("jane@example.com", PASSWORD)


def test_old_failures_are_forgotten_when_a_new_one_is_recorded(clean, user):
    fail_login(2)
    age_failures(clean, auth.cfg.lockout_minutes + 1)
    fail_login(1)
    with clean.pool().connection() as conn:
        assert conn.execute("SELECT count(*) FROM auth_failures").fetchone()[0] == 1


def test_password_checks_for_other_actions_share_one_counter_per_user(user):
    for _ in range(auth.cfg.max_failed_attempts):
        assert auth.check_password(user.id, "wrong password!") is False
    with pytest.raises(auth.AuthError, match="Too many failed attempts"):
        auth.change_password(user.id, PASSWORD, "another long password")
    with pytest.raises(auth.AuthError, match="Too many failed attempts"):
        auth.check_password(user.id, PASSWORD)


def test_a_right_password_check_is_true_and_clears_the_counter(user):
    assert auth.check_password(user.id, "wrong password!") is False
    assert auth.check_password(user.id, PASSWORD) is True


def test_recovery_code_guesses_are_throttled(user):
    auth.issue_recovery_code(user.id)
    for _ in range(auth.cfg.max_failed_attempts):
        with pytest.raises(auth.AuthError, match="Wrong email or recovery code"):
            auth.reset_password("jane@example.com", "AAAA-AAAA-AAAA-AAAA", "a new long password")
    with pytest.raises(auth.AuthError, match="Too many failed attempts"):
        auth.reset_password("jane@example.com", "AAAA-AAAA-AAAA-AAAA", "a new long password")


def test_a_burst_of_guesses_at_once_cannot_get_past_the_limit(user, monkeypatch):
    import threading

    checked = []
    real = auth._hasher

    class Counting:
        def verify(self, *args):
            checked.append(1)
            return real.verify(*args)

        def hash(self, *args):
            return real.hash(*args)

    monkeypatch.setattr(auth, "_hasher", Counting())
    results = []

    def guess():
        try:
            auth.log_in("jane@example.com", "wrong password!")
        except auth.AuthError as error:
            results.append(str(error))

    threads = [threading.Thread(target=guess) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(checked) == auth.cfg.max_failed_attempts  # the other 15 were refused before any password was checked
    assert sum("Too many failed attempts" in message for message in results) == 20 - auth.cfg.max_failed_attempts


def test_a_successful_reset_also_clears_a_lock_on_the_login(user):
    code = auth.issue_recovery_code(user.id)
    fail_login(auth.cfg.max_failed_attempts)  # someone else locked this email's login
    with pytest.raises(auth.AuthError, match="Too many failed attempts"):
        auth.log_in("jane@example.com", PASSWORD)
    auth.reset_password("jane@example.com", code, "a new long password")
    assert auth.log_in("jane@example.com", "a new long password")


# --- the other password flows still work -----------------------------------------------------------------------------------------

def test_reset_with_the_recovery_code_sets_the_password_ends_sessions_and_burns_the_code(user):
    code = auth.issue_recovery_code(user.id)
    old_token = auth.log_in("jane@example.com", PASSWORD)
    new_code = auth.reset_password("jane@example.com", code, "a new long password")
    assert auth.user_for_token(old_token) is None
    assert auth.log_in("jane@example.com", "a new long password")
    assert new_code != code
    with pytest.raises(auth.AuthError):
        auth.reset_password("jane@example.com", code, "yet another password")  # the code worked once


def test_changing_the_password_keeps_this_session_and_ends_the_others(user):
    keep, other = auth.log_in("jane@example.com", PASSWORD), auth.log_in("jane@example.com", PASSWORD)
    auth.change_password(user.id, PASSWORD, "a new long password", keep_token=keep)
    assert auth.user_for_token(keep) == user and auth.user_for_token(other) is None
    assert auth.log_in("jane@example.com", "a new long password")


def test_a_wrong_current_password_changes_nothing(user):
    with pytest.raises(auth.AuthError, match="current password is wrong"):
        auth.change_password(user.id, "wrong password!", "a new long password")
    assert auth.log_in("jane@example.com", PASSWORD)


def test_new_passwords_must_be_long_enough_without_counting_as_a_failure(user):
    with pytest.raises(auth.AuthError, match="at least"):
        auth.change_password(user.id, PASSWORD, "short")
    with pytest.raises(auth.AuthError, match="at least"):
        auth.reset_password("jane@example.com", "whatever", "short")
    with auth.db.pool().connection() as conn:
        assert conn.execute("SELECT count(*) FROM auth_failures").fetchone()[0] == 0


# --- hashing does not hold a database connection ---------------------------------------------------------------------------------

def idle_connections():
    stats = auth.db.pool().get_stats()
    return stats["pool_available"] == stats["pool_size"]


def test_no_connection_is_held_while_a_password_is_checked_or_hashed(user, monkeypatch):
    seen = []

    class Spy:
        """The real hasher, noting whether any database connection is checked out when it is used."""

        def __init__(self, real):
            self.real = real

        def verify(self, *args):
            seen.append(("verify", idle_connections()))
            return self.real.verify(*args)

        def hash(self, *args):
            seen.append(("hash", idle_connections()))
            return self.real.hash(*args)

    monkeypatch.setattr(auth, "_hasher", Spy(auth._hasher))
    code = auth.issue_recovery_code(user.id)

    auth.log_in("jane@example.com", PASSWORD)
    auth.change_password(user.id, PASSWORD, "a new long password")
    auth.reset_password("jane@example.com", code, "the third long password")

    assert len(seen) >= 4 and all(idle for _, idle in seen), seen
