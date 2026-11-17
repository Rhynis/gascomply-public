"""Tests for the in-process fixed-window rate limiter (issue #90)."""

import time
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from app.core.ratelimit import RateLimiter


def _request(ip: str = "1.2.3.4") -> Mock:
    request = Mock()
    request.client.host = ip
    return request


def test_requests_under_the_limit_all_pass():
    limiter = RateLimiter(limit=3, window_seconds=60.0)
    request = _request()

    for _ in range(3):
        limiter(request)  # must not raise


def test_the_request_that_reaches_the_limit_is_rejected():
    limiter = RateLimiter(limit=3, window_seconds=60.0)
    request = _request()

    for _ in range(3):
        limiter(request)

    with pytest.raises(HTTPException) as excinfo:
        limiter(request)
    assert excinfo.value.status_code == 429


def test_the_429_names_how_long_to_wait():
    limiter = RateLimiter(limit=1, window_seconds=60.0)
    request = _request()
    limiter(request)

    with pytest.raises(HTTPException) as excinfo:
        limiter(request)

    retry_after = int(excinfo.value.headers["Retry-After"])
    assert 0 < retry_after <= 60


def test_two_different_clients_have_independent_limits():
    limiter = RateLimiter(limit=1, window_seconds=60.0)
    limiter(_request("1.1.1.1"))

    limiter(_request("2.2.2.2"))  # must not raise: a different key


def test_a_client_with_no_address_still_counts_under_one_shared_key():
    limiter = RateLimiter(limit=1, window_seconds=60.0)
    request = Mock()
    request.client = None
    limiter(request)

    with pytest.raises(HTTPException):
        limiter(request)


def test_a_new_window_resets_the_count():
    limiter = RateLimiter(limit=1, window_seconds=0.05)
    request = _request()
    limiter(request)

    with pytest.raises(HTTPException):
        limiter(request)

    time.sleep(0.06)

    limiter(request)  # must not raise: a new window has started
