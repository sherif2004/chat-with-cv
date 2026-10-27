import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from openai import RateLimitError

from cv_chat.services import openai_service as svc


def rate_limit_error(retry_after: str = "0") -> RateLimitError:
    response = httpx.Response(429, request=httpx.Request("POST", "http://azure"), headers={"retry-after": retry_after})
    return RateLimitError("slow down", response=response, body=None)


# --- the gate -------------------------------------------------------------------------------------------------------

def test_background_work_cannot_take_the_slots_reserved_for_questions():
    gate = svc.AdaptiveLimit(start=4, maximum=4)
    assert gate.acquire(reserve=2) and gate.acquire(reserve=2)  # two background calls fit
    assert not gate.acquire(reserve=2, timeout=0.05)  # a third would eat a reserved slot
    assert gate.acquire(timeout=0.05) and gate.acquire(timeout=0.05)  # questions still get both reserved slots
    assert not gate.acquire(timeout=0.05)  # and then the gate is full for everyone


def test_one_background_call_always_runs_when_nothing_else_does_even_if_the_limit_fell_below_the_reserve():
    gate = svc.AdaptiveLimit(start=1, maximum=4)
    assert gate.acquire(reserve=2, timeout=0.05)
    assert not gate.acquire(reserve=2, timeout=0.05)


def test_the_limit_halves_after_a_rate_limit_and_grows_back_after_a_run_of_successes():
    gate = svc.AdaptiveLimit(start=8, maximum=8, grow_after=3)
    gate.acquire()
    gate.done(rate_limited=True)
    assert gate.limit == 4
    for _ in range(3):
        gate.acquire()
        gate.done()
    assert gate.limit == 5


# --- retries --------------------------------------------------------------------------------------------------------

def test_a_rate_limited_call_is_retried_after_the_wait_and_lowers_the_limit(monkeypatch):
    monkeypatch.setattr(svc.time, "sleep", lambda seconds: None)
    gate = svc.AdaptiveLimit(start=8, maximum=8)
    attempts = []

    def call():
        attempts.append(1)
        if len(attempts) < 3:
            raise rate_limit_error()
        return "ok"

    assert svc.call_with_retry(gate, call) == "ok"
    assert len(attempts) == 3 and gate.limit == 2  # halved twice


def test_a_call_that_stays_rate_limited_eventually_raises(monkeypatch):
    monkeypatch.setattr(svc.time, "sleep", lambda seconds: None)
    gate = svc.AdaptiveLimit(start=4, maximum=4)
    with pytest.raises(RateLimitError):
        svc.call_with_retry(gate, lambda: (_ for _ in ()).throw(rate_limit_error()))


def test_the_wait_follows_the_retry_after_header():
    assert svc._wait_seconds(rate_limit_error("7"), 0) == 7.0


# --- per user -------------------------------------------------------------------------------------------------------

def test_a_user_over_the_cap_waits_but_another_user_does_not():
    limits = svc.UserLimits(cap=1)
    first = limits.hold("alice", timeout=0.05)
    assert first is not None
    assert limits.hold("alice", timeout=0.05) is None
    assert limits.hold("bob", timeout=0.05) is not None
    first.release()
    assert limits.hold("alice", timeout=0.05) is not None


def test_calls_without_a_user_are_not_capped():
    limits = svc.UserLimits(cap=1)
    assert limits.hold(None, timeout=0.05) is not None and limits.hold(None, timeout=0.05) is not None


# --- questions are not slowed down by ingestion ---------------------------------------------------------------------

class FakeEmbeddings:
    def create(self, model, input):
        time.sleep(0.2)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.0]) for _ in input])


def test_a_question_embedding_finishes_while_a_big_ingestion_is_still_running(monkeypatch):
    fake_client = SimpleNamespace(with_options=lambda **kwargs: fake_client, embeddings=FakeEmbeddings())
    monkeypatch.setattr(svc, "_client", fake_client)
    monkeypatch.setattr(svc, "_embed_gate", svc.AdaptiveLimit(start=4, maximum=4))
    monkeypatch.setattr(svc, "_embed_batch", 1)  # one text per request: 10 texts are 10 requests
    done = {}

    def ingest():
        svc.embed([f"chunk {n}" for n in range(10)], background=True)
        done["ingest"] = time.perf_counter()

    worker = threading.Thread(target=ingest)
    started = time.perf_counter()
    worker.start()
    time.sleep(0.05)  # the ingestion is now using all the slots it may have
    svc.embed(["a question"])
    done["question"] = time.perf_counter()
    worker.join()

    assert done["question"] - started < 0.6  # about one call; behind the ingestion it would take about a second
    assert done["question"] < done["ingest"]
