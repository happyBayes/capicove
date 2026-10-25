"""Credential admission fails before dispatch for ambiguous transports."""

import importlib

import pytest

from capicove.runtime.remote_auth import bearer_token


def extract(headers: list[tuple[bytes, bytes]], query: bytes = b"") -> str:
    assert importlib.util.find_spec("capicove.runtime.remote_auth") is not None
    return bearer_token(headers, query=query)


def test_one_explicit_bearer_header() -> None:
    assert extract([(b"authorization", b"Bearer opaque-token")]) == "opaque-token"


@pytest.mark.parametrize(
    "headers,query",
    [
        ([], b""),
        ([(b"authorization", b"Bearer a"), (b"Authorization", b"Bearer b")], b""),
        ([(b"authorization", b"Basic a")], b""),
        ([(b"authorization", b"Bearer a b")], b""),
        ([(b"authorization", b"Bearer a\n")], b""),
        ([(b"authorization", b"Bearer a"), (b"cookie", b"token=b")], b""),
        ([(b"authorization", b"Bearer a")], b"access_token=b"),
        ([(b"authorization", b"Bearer a")], b"%61ccess_token=b"),
        ([(b"authorization", b"Bearer " + b"a" * 16385)], b""),
    ],
)
def test_ambiguous_or_alternative_credentials_rejected(
    headers: list[tuple[bytes, bytes]], query: bytes
) -> None:
    with pytest.raises(ValueError):
        extract(headers, query)
