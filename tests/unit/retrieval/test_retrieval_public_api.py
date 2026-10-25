"""Public retrieval package exports."""

from capicove import retrieval


def test_reverse_image_retrieval_public_api_is_exported() -> None:
    expected = {
        "InvocationAuthorizationContext",
        "RetrievalProfile",
        "RetrievalProfileRegistry",
        "RetrievalService",
        "default_retrieval_profiles",
    }

    assert expected.issubset(set(retrieval.__all__))
    assert all(getattr(retrieval, name) is not None for name in expected)
