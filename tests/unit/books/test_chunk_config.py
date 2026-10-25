from pathlib import Path

import pytest


def test_overrides_win_and_preset_protects_code() -> None:
    from capicove.books.chunk_config import resolve_chunk_config

    config = resolve_chunk_config(
        {"preset": "technical-code", "target_core_tokens": 500},
        {"target_core_tokens": 700},
    )
    assert config.target_core_tokens == 700
    assert "code" in config.protected_types
    assert config.context_before_tokens == 100


@pytest.mark.parametrize(
    "values",
    [
        {"target_tokens": 20},
        {"preset": "missing"},
        {"target_core_tokens": True},
        {"target_core_tokens": 1300},
        {"context_before_tokens": -1},
        {"oversize_policy": "truncate"},
        {"protected_types": ["arbitrary"]},
        {"merge_small_chunks": "false"},
        {"max_context_read_tokens": 0},
    ],
)
def test_invalid_configuration_is_not_silently_accepted(values: dict[str, object]) -> None:
    from capicove.books.chunk_config import resolve_chunk_config

    with pytest.raises((ValueError, TypeError)):
        resolve_chunk_config(overrides=values)


def test_read_budget_does_not_change_chunk_configuration() -> None:
    from capicove.books.chunk_config import resolve_chunk_config

    first = resolve_chunk_config()
    second = resolve_chunk_config(overrides={"max_context_read_tokens": 4000})
    assert first.chunk_values() == second.chunk_values()
    assert first.max_context_read_tokens != second.max_context_read_tokens


def test_toml_and_counter_are_explicit_and_offline(tmp_path: Path) -> None:
    from capicove.books.chunk_config import load_chunk_config, resolve_chunk_config
    from capicove.books.chunk_tokenizer import load_token_counter

    path = tmp_path / "chunking.toml"
    path.write_text('[chunking]\npreset = "math-heavy"\ncontext_after_tokens = 60\n')
    config = resolve_chunk_config(load_chunk_config(path))
    counter = load_token_counter(config)
    assert config.context_after_tokens == 60
    assert counter.count("A中😀") == 8
    assert counter.mode == "estimated"
    assert counter.fingerprint == "utf8-bytes-v1"


def test_toml_unknown_section_and_remote_tokenizer_fail(tmp_path: Path) -> None:
    from capicove.books.chunk_config import load_chunk_config, resolve_chunk_config

    path = tmp_path / "chunking.toml"
    path.write_text("[unknown]\nvalue = 1\n")
    with pytest.raises(ValueError, match="chunking"):
        load_chunk_config(path)
    with pytest.raises(ValueError, match="local"):
        resolve_chunk_config(overrides={"tokenizer": "https://example.com/tokenizer.json"})


def test_direct_config_applies_the_same_preset_as_resolver() -> None:
    from capicove.books.chunk_config import ChunkingConfig, resolve_chunk_config

    assert ChunkingConfig(preset="technical-code") == resolve_chunk_config(
        overrides={"preset": "technical-code"}
    )
