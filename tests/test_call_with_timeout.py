"""Tests for call_with_timeout (goldenswing.live.journal_lock).

The wrapper bounds blocking calls made while the journal lock is held
(health checks, broker fetches). These tests pin the cross-thread outcome
protocol:
- values (including None) propagate to the caller,
- exceptions propagate with the worker's traceback intact,
- timeouts raise TimeoutError promptly and abandon (never kill) the worker,
- concurrent calls never cross their outcomes.
"""
import threading
import time

import pytest

from goldenswing.live.journal_lock import call_with_timeout


def test_returns_value():
    assert call_with_timeout(lambda: 42, 5, label="t") == 42


def test_returns_none_distinguished_from_unset():
    # None is a legitimate result, not "no result yet".
    assert call_with_timeout(lambda: None, 5, label="t") is None


def test_propagates_exception():
    def boom():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        call_with_timeout(boom, 5, label="t")


def test_worker_traceback_preserved():
    def boom():
        raise RuntimeError("deep")

    try:
        call_with_timeout(boom, 5, label="t")
    except RuntimeError as e:
        frames = []
        tb = e.__traceback__
        while tb is not None:
            frames.append(tb.tb_frame.f_code.co_name)
            tb = tb.tb_next
        # The worker's frame survived the re-raise in the calling thread.
        assert "boom" in frames
    else:
        pytest.fail("expected RuntimeError")


def test_base_exception_transparency():
    def intr():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        call_with_timeout(intr, 5, label="t")


def test_timeout_raises_promptly_and_abandons():
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="timed out"):
        call_with_timeout(lambda: threading.Event().wait(60), 0.2,
                          label="hang")
    assert time.monotonic() - started < 5.0


def test_abandoned_worker_is_daemon():
    with pytest.raises(TimeoutError):
        call_with_timeout(lambda: threading.Event().wait(60), 0.2,
                          label="daemon-check")
    time.sleep(0.1)  # let the worker settle into its wait
    workers = [t for t in threading.enumerate()
               if t.name == "watchdog-daemon-check"]
    assert workers, "abandoned worker thread should still exist"
    assert all(t.daemon for t in workers), \
        "abandoned worker must not block interpreter shutdown"


def test_rejects_non_positive_timeout():
    with pytest.raises(ValueError):
        call_with_timeout(lambda: 1, 0, label="t")
    with pytest.raises(ValueError):
        call_with_timeout(lambda: 1, -1, label="t")


def test_concurrent_calls_do_not_cross_outcomes():
    results = {}
    errors = []

    def run(i):
        try:
            time.sleep(0.01 * (i % 5))
            results[i] = call_with_timeout(lambda: i, 10, label=f"c{i}")
        except Exception as e:  # noqa: BLE001 - test must not hide failures
            errors.append(e)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(15)
    assert not errors
    assert results == {i: i for i in range(30)}
