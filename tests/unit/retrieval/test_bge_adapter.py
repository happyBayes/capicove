"""Local BGE boundary tests using authored assets and an injected runtime."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from capicove.model.local_assets import LocalModelAssets
from capicove.retrieval._adapters.bge import (
    BgeTextBatchEncoder,
    BgeTextEmbeddingProfile,
    bge_preprocessing_version,
)
from tests.unit.retrieval.test_text_embedding import text_space


def local_profile(tmp_path: Path, updates: dict[str, Any] | None = None) -> BgeTextEmbeddingProfile:
    """Create a tiny authored bundle; weights are never loaded by a real runtime."""
    configs: dict[str, Any] = {
        "modules.json": [
            {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
            {
                "idx": 1,
                "name": "1",
                "path": "1_Pooling",
                "type": "sentence_transformers.models.Pooling",
            },
        ],
        "config.json": {
            "model_type": "bert",
            "architectures": ["BertModel"],
            "hidden_size": 2,
            "max_position_embeddings": 32,
            "vocab_size": 128,
        },
        "sentence_bert_config.json": {"max_seq_length": 16, "do_lower_case": False},
        "config_sentence_transformers.json": {
            "prompts": {"query": "Q: ", "document": "D: "},
            "default_prompt_name": "query",
        },
        "tokenizer_config.json": {
            "tokenizer_class": "BertTokenizer",
            "model_max_length": 32,
            "do_lower_case": False,
        },
        "tokenizer.json": {"model": {"type": "WordPiece", "vocab": {"test": 0}}},
        "1_Pooling/config.json": {
            "word_embedding_dimension": 2,
            "pooling_mode_cls_token": True,
            "pooling_mode_mean_tokens": False,
            "pooling_mode_max_tokens": False,
            "pooling_mode_mean_sqrt_len_tokens": False,
        },
    }
    configs.update(updates or {})
    for name, value in configs.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"authored fake weights")
    files = tuple(
        sorted(
            (p.relative_to(tmp_path).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest())
            for p in tmp_path.rglob("*")
            if p.is_file()
        )
    )
    digest = hashlib.sha256(
        json.dumps(files, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assets = LocalModelAssets(tmp_path.resolve(), files, digest)
    preprocessing = bge_preprocessing_version(
        assets, query_prefix="Q: ", document_prefix="D: ", max_tokens=16
    )
    space = replace(
        text_space(), model_revision="sha256:" + digest, preprocessing_version=preprocessing
    )
    return BgeTextEmbeddingProfile(
        assets, space, query_prefix="Q: ", document_prefix="D: ", max_tokens=16
    )


def test_m3_profile_accepts_verified_xlm_roberta_dense_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy = local_profile(
        tmp_path,
        {
            "config.json": {
                "model_type": "xlm-roberta",
                "architectures": ["XLMRobertaModel"],
                "hidden_size": 2,
                "max_position_embeddings": 32,
                "vocab_size": 128,
            },
            "tokenizer_config.json": {
                "tokenizer_class": "XLMRobertaTokenizer",
                "model_max_length": 32,
            },
        },
    )
    preprocessing = bge_preprocessing_version(
        legacy.assets,
        query_prefix="Q: ",
        document_prefix="D: ",
        max_tokens=16,
        model_family="bge-m3",
    )
    profile = replace(
        legacy,
        model_family="bge-m3",
        space=replace(legacy.space, preprocessing_version=preprocessing),
    )
    runtime = FakeRuntime()
    original = runtime.__class__.__getitem__

    def transformer(self: FakeRuntime, index: int) -> Any:
        result = original(self, index)
        result.auto_model.config.model_type = "xlm-roberta"
        return result

    monkeypatch.setattr(FakeRuntime, "__getitem__", transformer)
    inject_runtime(monkeypatch, runtime)
    result = BgeTextBatchEncoder(profile, device="cpu", batch_size=1).encode(
        ("two",), space=profile.space, role="document"
    )
    assert result[0].vector == (1.0, 0.0)
    assert preprocessing != legacy.space.preprocessing_version


def test_m3_rejects_bert_bundle(tmp_path: Path) -> None:
    legacy = local_profile(tmp_path)
    preprocessing = bge_preprocessing_version(
        legacy.assets,
        query_prefix="Q: ",
        document_prefix="D: ",
        max_tokens=16,
        model_family="bge-m3",
    )
    profile = replace(
        legacy,
        model_family="bge-m3",
        space=replace(legacy.space, preprocessing_version=preprocessing),
    )
    with pytest.raises(ValueError, match="unsupported_bge_model"):
        BgeTextBatchEncoder(profile, device="cpu").encode(
            ("two",), space=profile.space, role="document"
        )


class FakeTokenizer:
    model_max_length = 32

    def __init__(self) -> None:
        self.inputs: list[str] = []

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        assert kwargs == {"add_special_tokens": True, "truncation": False, "padding": False}
        self.inputs.append(text)
        return {"input_ids": [1] + [3] * len(text) + [2]}


class FakeRuntime:
    max_seq_length = 16
    truncate_dim = None

    def __init__(self, output: Any = None) -> None:
        self.tokenizer = FakeTokenizer()
        self.output = output
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 2

    def __getitem__(self, index: int) -> Any:
        assert index == 0
        return SimpleNamespace(
            do_lower_case=False,
            auto_model=SimpleNamespace(
                config=SimpleNamespace(
                    model_type="bert", max_position_embeddings=32, hidden_size=2, vocab_size=128
                )
            ),
        )

    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        self.calls.append((texts, kwargs))
        if self.output is not None:
            return self.output
        return np.asarray(
            [[1.0, 0.0] if text.endswith("two") else [0.0, 1.0] for text in texts], dtype=np.float32
        )


def inject_runtime(
    monkeypatch: pytest.MonkeyPatch, runtime: FakeRuntime
) -> list[tuple[str, dict[str, Any]]]:
    loads: list[tuple[str, dict[str, Any]]] = []

    def load(root: str, **kwargs: Any) -> FakeRuntime:
        loads.append((root, kwargs))
        transformer = json.loads((Path(root) / "sentence_bert_config.json").read_text())
        tokenizer = json.loads((Path(root) / "tokenizer_config.json").read_text())
        runtime.max_seq_length = transformer["max_seq_length"]
        # Sentence Transformers 5.6 passes max_seq_length to AutoTokenizer
        # unless tokenizer_kwargs explicitly sets model_max_length.
        runtime.tokenizer.model_max_length = kwargs.get("tokenizer_kwargs", {}).get(
            "model_max_length", runtime.max_seq_length or tokenizer["model_max_length"]
        )
        return runtime

    def import_runtime(name: str) -> Any:
        assert name == "sentence_transformers"
        return SimpleNamespace(SentenceTransformer=load, __version__="5.6.0")

    monkeypatch.setattr("capicove.retrieval._adapters.bge.importlib.import_module", import_runtime)
    return loads


@pytest.mark.parametrize("role,prefix", [("query", "Q: "), ("document", "D: ")])
def test_local_bge_is_lazy_and_preserves_batch_order_and_exact_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str, prefix: str
) -> None:
    profile = local_profile(tmp_path)
    runtime = FakeRuntime()
    loads = inject_runtime(monkeypatch, runtime)
    encoder = BgeTextBatchEncoder(profile, device="cpu", batch_size=2)
    assert loads == []
    output = encoder.encode(("two", "one", "two"), space=profile.space, role=role)
    assert [item.vector for item in output] == [(1.0, 0.0), (0.0, 1.0), (1.0, 0.0)]
    assert all(item.space == profile.space for item in output)
    assert loads == [
        (
            str(tmp_path.resolve()),
            {
                "local_files_only": True,
                "trust_remote_code": False,
                "device": "cpu",
                "backend": "torch",
                "model_kwargs": {"use_safetensors": True},
            },
        )
    ]
    assert runtime.tokenizer.inputs == [prefix + "two", prefix + "one", prefix + "two"]
    assert runtime.calls == [
        (
            [prefix + "two", prefix + "one", prefix + "two"],
            {
                "prompt": "",
                "batch_size": 2,
                "show_progress_bar": False,
                "output_value": "sentence_embedding",
                "precision": "float32",
                "convert_to_numpy": True,
                "convert_to_tensor": False,
                "normalize_embeddings": True,
                "device": "cpu",
            },
        )
    ]
    encoder.encode(("one",), space=profile.space, role=role)
    assert len(loads) == 1


def test_overlong_input_is_counted_with_prefix_special_tokens_before_encode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = local_profile(tmp_path)
    runtime = FakeRuntime()
    inject_runtime(monkeypatch, runtime)
    encoder = BgeTextBatchEncoder(profile, device="cpu")
    assert encoder.encode(("x" * 11,), space=profile.space, role="query")
    runtime.calls.clear()
    with pytest.raises(ValueError, match="too_long"):
        encoder.encode(("ok", "x" * 12), space=profile.space, role="query")
    assert runtime.calls == []


def test_embedding_usage_counts_prefixes_special_tokens_and_unpadded_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = local_profile(tmp_path)
    runtime = FakeRuntime()
    inject_runtime(monkeypatch, runtime)
    encoder = BgeTextBatchEncoder(profile, device="cpu")

    vectors, counts = encoder.encode_with_token_counts(
        ("one", "longer"), space=profile.space, role="query", max_total_tokens=19
    )

    assert counts == (8, 11)
    assert len(vectors) == 2
    assert len(runtime.calls) == 1
    runtime.calls.clear()
    with pytest.raises(ValueError, match="embedding_batch_too_long"):
        encoder.encode_with_token_counts(
            ("one", "longer"), space=profile.space, role="query", max_total_tokens=18
        )
    assert runtime.calls == []


@pytest.mark.parametrize("budget", [True, 0, -1])
def test_embedding_usage_rejects_invalid_total_budget(tmp_path: Path, budget: int) -> None:
    profile = local_profile(tmp_path)
    encoder = BgeTextBatchEncoder(profile, device="cpu")
    with pytest.raises(ValueError, match="invalid_embedding_token_budget"):
        encoder.encode_with_token_counts(
            ("one",), space=profile.space, role="document", max_total_tokens=budget
        )


@pytest.mark.parametrize("transformer_limit,tokenizer_limit", [(16, 32), (24, 16)])
def test_differing_verified_limits_match_loaded_tokenizer_without_widening_input_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transformer_limit: int,
    tokenizer_limit: int,
) -> None:
    profile = local_profile(
        tmp_path,
        {
            "sentence_bert_config.json": {
                "max_seq_length": transformer_limit,
                "do_lower_case": False,
            },
            "tokenizer_config.json": {
                "tokenizer_class": "BertTokenizer",
                "model_max_length": tokenizer_limit,
                "do_lower_case": False,
            },
        },
    )
    runtime = FakeRuntime()
    inject_runtime(monkeypatch, runtime)
    encoder = BgeTextBatchEncoder(profile, device="cpu")
    # Eleven characters + the three-character prefix + two special tokens = 16.
    assert encoder.encode(("x" * 11,), space=profile.space, role="query")[0].vector == (0.0, 1.0)
    assert runtime.tokenizer.model_max_length == transformer_limit
    runtime.calls.clear()
    with pytest.raises(ValueError, match="embedding_input_too_long"):
        encoder.encode(("ok", "x" * 12), space=profile.space, role="query")
    assert runtime.calls == []


def test_preflight_uses_exact_sentence_transformer_strip_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = local_profile(tmp_path)
    runtime = FakeRuntime()
    inject_runtime(monkeypatch, runtime)
    BgeTextBatchEncoder(profile, device="cpu").encode(
        (" two  ",), space=profile.space, role="query"
    )
    assert runtime.tokenizer.inputs == ["Q:  two"]
    assert runtime.calls[0][0] == ["Q:  two"]


@pytest.mark.parametrize(
    "output",
    [
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        np.asarray([[1.0]]),
        np.asarray([[float("nan"), 1.0]]),
        np.asarray([[float("inf"), 1.0]]),
        np.asarray([[2.0, 0.0]]),
        np.asarray([[0.0, 0.0]]),
        np.asarray([[[1.0, 0.0]]]),
        np.asarray([[1, 0]], dtype=np.int8),
        {"dense_vecs": [[1.0, 0.0]], "lexical_weights": [{}]},
        [[1.0, 0.0]],
    ],
)
def test_bge_rejects_invalid_or_nondense_runtime_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output: Any
) -> None:
    profile = local_profile(tmp_path)
    inject_runtime(monkeypatch, FakeRuntime(output))
    with pytest.raises(ValueError):
        BgeTextBatchEncoder(profile, device="cpu").encode(
            ("one",), space=profile.space, role="query"
        )


@pytest.mark.parametrize("failure", ["missing", "tampered", "extra", "symlink", "unsafe_code"])
def test_bge_rejects_unsafe_assets_before_runtime_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    profile = local_profile(tmp_path)
    path = tmp_path / "model.safetensors"
    if failure == "missing":
        path.unlink()
    elif failure == "tampered":
        path.write_bytes(b"other weights")
    elif failure == "extra":
        (tmp_path / "extra.json").write_text("{}")
    elif failure == "symlink":
        path.unlink()
        path.symlink_to(tmp_path / "config.json")
    else:
        profile = local_profile(tmp_path, {"config.json": {"auto_map": {"AutoModel": "code.Bad"}}})
    loads = inject_runtime(monkeypatch, FakeRuntime())
    with pytest.raises(ValueError):
        BgeTextBatchEncoder(profile, device="cpu").encode(
            ("one",), space=profile.space, role="query"
        )
    assert loads == []


@pytest.mark.parametrize(
    "updates",
    [
        {"modules.json": [{"idx": 0, "name": "0", "path": "", "type": "evil.Module"}]},
        {
            "modules.json": [
                {
                    "idx": 0,
                    "name": "0",
                    "path": "../other",
                    "type": "sentence_transformers.models.Transformer",
                }
            ]
        },
        {
            "sentence_bert_config.json": {
                "max_seq_length": 16,
                "tokenizer_name_or_path": "remote/model",
            }
        },
        {"sentence_bert_config.json": {"max_seq_length": 8, "do_lower_case": False}},
        {
            "1_Pooling/config.json": {
                "word_embedding_dimension": 2,
                "pooling_mode_mean_tokens": True,
            }
        },
        {
            "config_sentence_transformers.json": {
                "prompts": {"query": "different", "document": "D: "}
            }
        },
        {"config_sentence_transformers.json": {"model_type": "SparseEncoder"}},
        {"config_sentence_transformers.json": {"truncate_dim": 2}},
        {"tokenizer_config.json": {"tokenizer_class": "CustomTokenizer", "model_max_length": 32}},
        {"tokenizer_config.json": {"tokenizer_class": "BertTokenizer", "model_max_length": 8}},
        {"config.json": {"model_type": "bert", "hidden_size": 3, "max_position_embeddings": 32}},
    ],
)
def test_bge_qualifies_verified_bundle_before_runtime_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, updates: dict[str, Any]
) -> None:
    profile = local_profile(tmp_path, updates)
    loads = inject_runtime(monkeypatch, FakeRuntime())
    with pytest.raises(ValueError):
        BgeTextBatchEncoder(profile, device="cpu").encode(
            ("one",), space=profile.space, role="query"
        )
    assert loads == []


@pytest.mark.parametrize(
    "field,value",
    [("query_prefix", "different"), ("document_prefix", "different"), ("max_tokens", 12)],
)
def test_preprocessing_identity_binds_role_mapping_and_window(
    tmp_path: Path, field: str, value: Any
) -> None:
    with pytest.raises(ValueError, match="preprocessing"):
        replace(local_profile(tmp_path), **{field: value})


def test_model_identity_binds_entire_verified_inventory(tmp_path: Path) -> None:
    profile = local_profile(tmp_path)
    with pytest.raises(ValueError, match="revision"):
        replace(profile, space=replace(profile.space, model_revision="other"))


@pytest.mark.parametrize("limit_owner", ["encoder", "tokenizer"])
def test_bge_rejects_runtime_context_smaller_than_qualified_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit_owner: str
) -> None:
    profile = local_profile(tmp_path)
    runtime = FakeRuntime()
    inject_runtime(monkeypatch, runtime)
    encoder = BgeTextBatchEncoder(profile, device="cpu")
    encoder.encode(("one",), space=profile.space, role="query")
    if limit_owner == "encoder":
        runtime.max_seq_length = 8
    else:
        runtime.tokenizer.model_max_length = 8
    runtime.calls.clear()
    with pytest.raises(ValueError):
        encoder.encode(("one",), space=profile.space, role="query")
    assert runtime.calls == []


@pytest.mark.parametrize("role", ["target", "passage"])
def test_bge_checks_role_before_runtime_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    profile = local_profile(tmp_path)
    loads = inject_runtime(monkeypatch, FakeRuntime())
    with pytest.raises(ValueError):
        BgeTextBatchEncoder(profile, device="cpu").encode(("one",), space=profile.space, role=role)
    assert loads == []


def test_bge_checks_complete_requested_space_before_runtime_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = local_profile(tmp_path)
    loads = inject_runtime(monkeypatch, FakeRuntime())
    with pytest.raises(ValueError):
        BgeTextBatchEncoder(profile, device="cpu").encode(
            ("one",), space=replace(profile.space, preprocessing_version="other"), role="query"
        )
    assert loads == []


def test_bge_imports_never_load_optional_runtimes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import capicove.retrieval; import capicove.retrieval._adapters.bge; import sys; assert not {'sentence_transformers', 'torch', 'transformers'} & sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_bge_rejects_boolean_module_indices_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = local_profile(
        tmp_path,
        {
            "modules.json": [
                {
                    "idx": False,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.models.Transformer",
                },
                {
                    "idx": True,
                    "name": "1",
                    "path": "1_Pooling",
                    "type": "sentence_transformers.models.Pooling",
                },
            ]
        },
    )
    loads = inject_runtime(monkeypatch, FakeRuntime())
    with pytest.raises(ValueError):
        BgeTextBatchEncoder(profile, device="cpu").encode(
            ("one",), space=profile.space, role="query"
        )
    assert loads == []


@pytest.mark.parametrize("device", ["auto", "", "gpu", None, ["cpu"], "cpu;download"])
def test_device_must_be_explicit_local_runtime_configuration(tmp_path: Path, device: Any) -> None:
    with pytest.raises(ValueError):
        BgeTextBatchEncoder(local_profile(tmp_path), device=device)


@pytest.mark.parametrize("batch_size", [0, -1, True, 2.5])
def test_runtime_batch_size_must_be_a_positive_integer(tmp_path: Path, batch_size: Any) -> None:
    with pytest.raises(ValueError):
        BgeTextBatchEncoder(local_profile(tmp_path), device="cpu", batch_size=batch_size)


def test_cached_runtime_rechecks_assets_before_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = local_profile(tmp_path)
    runtime = FakeRuntime()
    inject_runtime(monkeypatch, runtime)
    encoder = BgeTextBatchEncoder(profile, device="cpu")
    encoder.encode(("one",), space=profile.space, role="query")
    runtime.calls.clear()
    (tmp_path / "tokenizer.json").write_text("{}")
    with pytest.raises(ValueError):
        encoder.encode(("two",), space=profile.space, role="query")
    assert runtime.calls == []


@pytest.mark.parametrize(
    "failure", [ImportError("private runtime path"), RuntimeError("secret source")]
)
def test_optional_runtime_failures_do_not_expose_paths_or_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    profile = local_profile(tmp_path)

    def fail(name: str) -> Any:
        raise failure

    monkeypatch.setattr("capicove.retrieval._adapters.bge.importlib.import_module", fail)
    with pytest.raises(ValueError) as error:
        BgeTextBatchEncoder(profile, device="cpu").encode(
            ("source",), space=profile.space, role="query"
        )
    assert str(error.value) in {"text_embedding_runtime_unavailable", "local_embedding_failure"}


@pytest.mark.parametrize("version", ["5.2.3", "5.3.0", "5.7.0"])
def test_unreviewed_runtime_version_is_rejected_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    profile = local_profile(tmp_path)
    monkeypatch.setattr(
        "capicove.retrieval._adapters.bge.importlib.import_module",
        lambda name: SimpleNamespace(__version__=version),
    )
    with pytest.raises(ValueError, match="unsupported_sentence_transformers_version"):
        BgeTextBatchEncoder(profile, device="cpu").encode(
            ("source",), space=profile.space, role="query"
        )


@pytest.mark.parametrize("token_ids", [[], [True], [-1], [128], [[1]], "tokens"])
def test_invalid_tokenization_cannot_reach_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, token_ids: Any
) -> None:
    profile = local_profile(tmp_path)
    runtime = FakeRuntime()
    inject_runtime(monkeypatch, runtime)
    monkeypatch.setattr(runtime, "tokenizer", lambda *args, **kwargs: {"input_ids": token_ids})
    runtime.tokenizer.model_max_length = 32
    with pytest.raises(ValueError, match="tokenization"):
        BgeTextBatchEncoder(profile, device="cpu").encode(
            ("source",), space=profile.space, role="query"
        )
    assert runtime.calls == []
