"""Lazy dense BGE encoding from a verified, explicitly qualified local bundle.

This adapter supports BERT or explicitly selected BGE-M3 XLM-R + CLS pooling
layout, optionally followed by Normalize. The BGE family name alone does not
qualify a model. Hosts must provision a closed, immutable local inventory and
qualify their exact bundle's retrieval quality separately.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from dataclasses import dataclass
from threading import Lock
from typing import Any

from capicove.contracts import (
    EmbeddingSpaceRef,
    QueryModality,
    TargetType,
    VectorNormalization,
)
from capicove.model.local_assets import LocalModelAssets
from capicove.retrieval.models import TextEmbedding
from capicove.retrieval.text_embedding import _validate_embeddings, _validate_request


class _BgeError(ValueError):
    """Carry a bounded issue code without runtime paths or input text."""


def bge_preprocessing_version(
    assets: LocalModelAssets,
    *,
    query_prefix: str,
    document_prefix: str,
    max_tokens: int,
    model_family: str = "bge-bert",
) -> str:
    """Bind model/tokenizer/pooling bytes, role instructions and input policy.

    The asset manifest contains every tokenizer, pooling, normalization, model,
    and Sentence Transformers configuration file. Explicit role prefixes are
    concatenated before outer whitespace stripping, exactly as Transformer 5.6
    tokenizes text. Its verified max_seq_length sets the runtime tokenizer limit;
    no tokenizer override, additional lowercasing, or truncation is supported.
    """
    if not isinstance(assets, LocalModelAssets):
        raise ValueError("invalid_local_assets")
    if not isinstance(query_prefix, str) or not isinstance(document_prefix, str):
        raise ValueError("invalid_embedding_prefix")
    if type(max_tokens) is not int or not 0 < max_tokens < 10**12:
        raise ValueError("invalid_embedding_token_limit")
    if model_family not in {"bge-bert", "bge-m3"}:
        raise ValueError("unsupported_bge_model_family")
    payload = json.dumps(
        {
            "adapter": (
                "bge-bert-cls-st5.6-v2"
                if model_family == "bge-bert"
                else "bge-m3-xlm-roberta-cls-st5.6-v2"
            ),
            "assets": assets.manifest_sha256,
            "roles": {"query": query_prefix, "document": document_prefix},
            "max_tokens": max_tokens,
            "normalization": "l2",
            "input_policy": "prefix-then-strip-no-truncation",
            "tokenizer_limit_policy": "st-max-seq-length-no-override",
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class BgeTextEmbeddingProfile:
    """Startup-only dense BGE identity, never selected by text or a request.

    model_revision must identify the complete local manifest as sha256:<digest>.
    preprocessing_version must equal bge_preprocessing_version for these exact
    inputs. This pins tokenizer/pooling/normalization bytes and descriptor roles
    alongside the model. Empty prefixes are allowed only when explicitly given.
    """

    assets: LocalModelAssets
    space: EmbeddingSpaceRef
    query_prefix: str
    document_prefix: str
    max_tokens: int
    model_family: str = "bge-bert"

    def __post_init__(self) -> None:
        preprocessing = bge_preprocessing_version(
            self.assets,
            query_prefix=self.query_prefix,
            document_prefix=self.document_prefix,
            max_tokens=self.max_tokens,
            model_family=self.model_family,
        )
        if not isinstance(self.space, EmbeddingSpaceRef):
            raise ValueError("invalid_bge_space")
        if self.space.model_revision != "sha256:" + self.assets.manifest_sha256:
            raise ValueError("bge_model_revision_mismatch")
        if self.space.preprocessing_version != preprocessing:
            raise ValueError("bge_preprocessing_mismatch")
        if (
            self.space.provider_id != "local-bge"
            or self.space.schema_version != 1
            or self.space.normalization is not VectorNormalization.L2
            or self.space.query_modalities != frozenset({QueryModality.TEXT})
            or self.space.target_modalities != frozenset({TargetType.TEXT_CHUNK})
        ):
            raise ValueError("unsupported_bge_space")


def _read_json(assets: LocalModelAssets, name: str) -> Any:
    if name not in dict(assets.files):
        raise _BgeError("bge_configuration_missing")
    return json.loads((assets.root / name).read_bytes())


def _object(assets: LocalModelAssets, name: str) -> dict[str, Any]:
    value = _read_json(assets, name)
    if not isinstance(value, dict):
        raise _BgeError("invalid_bge_configuration")
    return value


def _positive_int(value: Any) -> bool:
    return type(value) is int and 0 < value < 10**12


def _verify_bundle(profile: BgeTextEmbeddingProfile) -> dict[str, Any]:
    assets = profile.assets
    try:
        assets.verify()
    except ValueError as error:
        raise _BgeError(str(error)) from None
    modules = _read_json(assets, "modules.json")
    expected = [
        {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
        {
            "idx": 1,
            "name": "1",
            "path": "1_Pooling",
            "type": "sentence_transformers.models.Pooling",
        },
    ]
    normalized = [
        *expected,
        {
            "idx": 2,
            "name": "2",
            "path": "2_Normalize",
            "type": "sentence_transformers.models.Normalize",
        },
    ]
    if (modules != expected and modules != normalized) or any(
        type(module["idx"]) is not int for module in modules
    ):
        raise _BgeError("unsupported_bge_modules")
    config = _object(assets, "config.json")
    model_type, architecture, tokenizer_classes = (
        ("bert", "BertModel", {"BertTokenizer", "BertTokenizerFast"})
        if profile.model_family == "bge-bert"
        else ("xlm-roberta", "XLMRobertaModel", {"XLMRobertaTokenizer", "XLMRobertaTokenizerFast"})
    )
    if (
        config.get("model_type") != model_type
        or config.get("architectures") != [architecture]
        or config.get("hidden_size") != profile.space.dimension
        or not _positive_int(config.get("vocab_size"))
        or not _positive_int(config.get("max_position_embeddings"))
        or config["max_position_embeddings"] < profile.max_tokens
        or not {"tokenizer.json", "model.safetensors"} <= dict(assets.files).keys()
    ):
        raise _BgeError("unsupported_bge_model")
    transformer = _object(assets, "sentence_bert_config.json")
    if (
        transformer.keys() - {"max_seq_length", "do_lower_case"}
        or transformer.get("do_lower_case", False) is not False
        or not _positive_int(transformer.get("max_seq_length"))
        or transformer["max_seq_length"] < profile.max_tokens
    ):
        raise _BgeError("unsupported_bge_preprocessing")
    tokenizer = _object(assets, "tokenizer_config.json")
    if (
        tokenizer.get("tokenizer_class") not in tokenizer_classes
        or not _positive_int(tokenizer.get("model_max_length"))
        or tokenizer["model_max_length"] < profile.max_tokens
    ):
        raise _BgeError("unsupported_bge_tokenizer")
    pooling = _object(assets, "1_Pooling/config.json")
    other_modes = {
        "pooling_mode_mean_tokens",
        "pooling_mode_max_tokens",
        "pooling_mode_mean_sqrt_len_tokens",
        "pooling_mode_weightedmean_tokens",
        "pooling_mode_lasttoken",
    }
    if (
        pooling.keys()
        - {*other_modes, "word_embedding_dimension", "pooling_mode_cls_token", "include_prompt"}
        or pooling.get("word_embedding_dimension") != profile.space.dimension
        or pooling.get("pooling_mode_cls_token") is not True
        # Pooling defaults mean pooling to True when its flag is absent.
        or pooling.get("pooling_mode_mean_tokens") is not False
        or any(pooling.get(name, False) is not False for name in other_modes)
        or pooling.get("include_prompt", True) is not True
    ):
        raise _BgeError("unsupported_bge_pooling")
    if "config_sentence_transformers.json" in dict(assets.files):
        st_config = _object(assets, "config_sentence_transformers.json")
        prompts = st_config.get("prompts", {})
        expected_prompts = {"query": profile.query_prefix, "document": profile.document_prefix}
        if (
            st_config.keys()
            - {"__version__", "prompts", "default_prompt_name", "similarity_fn_name", "model_type"}
            or st_config.get("model_type", "SentenceTransformer") != "SentenceTransformer"
            or not isinstance(prompts, dict)
            or any(
                key not in expected_prompts or value != expected_prompts[key]
                for key, value in prompts.items()
            )
            or st_config.get("default_prompt_name") not in (None, "query", "document")
        ):
            raise _BgeError("unsupported_bge_instructions")
    return {"model": config, "transformer": transformer, "tokenizer": tokenizer}


def _verify_runtime(encoder: Any, profile: BgeTextEmbeddingProfile, config: dict[str, Any]) -> None:
    model = encoder[0].auto_model.config
    if (
        encoder.get_sentence_embedding_dimension() != profile.space.dimension
        or encoder.truncate_dim is not None
        or encoder.max_seq_length != config["transformer"]["max_seq_length"]
        # Transformer 5.6 sets this AutoTokenizer constructor argument from
        # max_seq_length; tokenizer_config.json's original limit is overwritten.
        or encoder.tokenizer.model_max_length != config["transformer"]["max_seq_length"]
        or encoder[0].do_lower_case is not False
        or any(
            getattr(model, name) != config["model"][name]
            for name in ("model_type", "hidden_size", "max_position_embeddings", "vocab_size")
        )
    ):
        raise _BgeError("bge_runtime_configuration_mismatch")


class BgeTextBatchEncoder:
    """Implement ordered dense batches using optional Sentence Transformers 5.6.

    Loading and encoding are serialized. Asset bytes are reverified on each call;
    host filesystem permissions must prevent concurrent writes during loading.
    No remote fallback, dynamic modules, quantization, routing, sparse vectors,
    token embeddings, or implicit device selection is supported.
    """

    def __init__(
        self, profile: BgeTextEmbeddingProfile, *, device: str, batch_size: int = 32
    ) -> None:
        if not isinstance(profile, BgeTextEmbeddingProfile):
            raise ValueError("invalid_bge_profile")
        if not isinstance(device, str) or re.fullmatch(r"cpu|mps|cuda(?::[0-9]+)?", device) is None:
            raise ValueError("explicit_bge_device_required")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("invalid_embedding_batch_size")
        self._profile = profile
        self._device = device
        self._batch_size = batch_size
        self._encoder: Any = None
        self._lock = Lock()

    @property
    def space(self) -> EmbeddingSpaceRef:
        """Expose the complete immutable embedding identity for index binding."""
        return self._profile.space

    def close(self) -> None:
        """Release the optional runtime under the encoder's inference lock."""
        with self._lock:
            self._encoder = None

    def encode(
        self, texts: tuple[str, ...], *, space: EmbeddingSpaceRef, role: str
    ) -> tuple[TextEmbedding, ...]:
        """Encode a validated batch without changing its existing public contract."""
        return self.encode_with_token_counts(texts, space=space, role=role)[0]

    def encode_with_token_counts(
        self,
        texts: tuple[str, ...],
        *,
        space: EmbeddingSpaceRef,
        role: str,
        max_total_tokens: int | None = None,
    ) -> tuple[tuple[TextEmbedding, ...], tuple[int, ...]]:
        """Count every complete input before encoding any text in the batch."""
        if max_total_tokens is not None and (
            type(max_total_tokens) is not int or max_total_tokens < 1
        ):
            raise ValueError("invalid_embedding_token_budget")
        profile = self._profile
        _validate_request(texts, space=space, expected=profile.space, role=role)
        with self._lock:
            try:
                config = _verify_bundle(profile)
                if self._encoder is None:
                    runtime = importlib.import_module("sentence_transformers")
                    if re.fullmatch(r"5\.6\.\d+", runtime.__version__) is None:
                        raise _BgeError("unsupported_sentence_transformers_version")
                    encoder = runtime.SentenceTransformer(
                        str(profile.assets.root),
                        local_files_only=True,
                        trust_remote_code=False,
                        device=self._device,
                        backend="torch",
                        model_kwargs={"use_safetensors": True},
                    )
                    _verify_runtime(encoder, profile, config)
                    _verify_bundle(profile)
                    self._encoder = encoder
                _verify_runtime(self._encoder, profile, config)
                prefix = profile.query_prefix if role == "query" else profile.document_prefix
                prepared = [(prefix + text).strip() for text in texts]
                token_limit = min(
                    profile.max_tokens,
                    config["model"]["max_position_embeddings"],
                    config["transformer"]["max_seq_length"],
                    config["tokenizer"]["model_max_length"],
                )
                counts = []
                for text in prepared:
                    tokens = self._encoder.tokenizer(
                        text, add_special_tokens=True, truncation=False, padding=False
                    )["input_ids"]
                    if (
                        not isinstance(tokens, list)
                        or not tokens
                        or any(
                            type(token) is not int or not 0 <= token < config["model"]["vocab_size"]
                            for token in tokens
                        )
                    ):
                        raise _BgeError("invalid_bge_tokenization")
                    if len(tokens) > token_limit:
                        raise _BgeError("embedding_input_too_long")
                    counts.append(len(tokens))
                if max_total_tokens is not None and sum(counts) > max_total_tokens:
                    raise _BgeError("embedding_batch_too_long")
                raw = self._encoder.encode(
                    prepared,
                    # Prefixes have already been applied and counted. Disable the
                    # bundle's implicit default prompt, including document calls.
                    prompt="",
                    batch_size=self._batch_size,
                    show_progress_bar=False,
                    output_value="sentence_embedding",
                    precision="float32",
                    convert_to_numpy=True,
                    convert_to_tensor=False,
                    normalize_embeddings=True,
                    device=self._device,
                )
                import numpy as np

                if (
                    not isinstance(raw, np.ndarray)
                    or raw.ndim != 2
                    or raw.shape != (len(texts), space.dimension)
                    or raw.dtype.kind != "f"
                ):
                    raise _BgeError("invalid_dense_bge_output")
                output = tuple(TextEmbedding(space, tuple(row)) for row in raw.tolist())
                _validate_embeddings(output, count=len(texts), space=space)
                return output, tuple(counts)
            except _BgeError:
                raise
            except ImportError:
                raise _BgeError("text_embedding_runtime_unavailable") from None
            except Exception:
                raise _BgeError("local_embedding_failure") from None
