import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest import MonkeyPatch

from capicove.books.chunk_config import ChunkingConfig
from capicove.books.chunk_tokenizer import load_token_counter


def test_local_counter_disables_asset_truncation_and_padding(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    class Tokenizer:
        truncation = True
        padding = True

        def no_truncation(self) -> None:
            self.truncation = False

        def no_padding(self) -> None:
            self.padding = False

        def encode(self, text: str, *, add_special_tokens: bool) -> SimpleNamespace:
            assert not self.truncation and not self.padding and not add_special_tokens
            return SimpleNamespace(ids=list(text))

    tokenizer = Tokenizer()
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps({"truncation": {"max_length": 10}}))
    runtime = SimpleNamespace(
        __version__="test", Tokenizer=SimpleNamespace(from_str=lambda _: tokenizer)
    )
    monkeypatch.setattr("capicove.books.chunk_tokenizer.importlib.import_module", lambda _: runtime)
    counter = load_token_counter(ChunkingConfig(tokenizer="local:" + str(path)))
    assert counter.count("x" * 100) == 100
    assert counter.mode == "exact"
    assert counter.fingerprint.startswith("tokenizers:test:")


def test_local_counter_requires_explicit_regular_asset_and_optional_runtime(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    path = tmp_path / "tokenizer.json"
    config = ChunkingConfig(tokenizer="local:" + str(path))
    with pytest.raises(ValueError, match="regular"):
        load_token_counter(config)
    path.write_text("{}")

    def unavailable(name: str) -> None:
        raise ImportError(name)

    monkeypatch.setattr("capicove.books.chunk_tokenizer.importlib.import_module", unavailable)
    with pytest.raises(ValueError, match="optional tokenizers"):
        load_token_counter(config)
