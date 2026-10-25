"""Resident offline BGE cross-encoder with bounded, untruncated text pairs."""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from threading import RLock
from typing import Any

from capicove.model.reranker_assets import MANIFEST_SHA256, MODEL_ID, reranker_assets
from capicove.retrieval.reranking import RerankResult, rank_scores, validate_rerank_request


def _load_runtime(root: Path, device: str) -> tuple[Any, Any, Any]:
    """Import optional SDKs only after explicit verified acquisition."""
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(root), local_files_only=True, trust_remote_code=False, use_fast=True
    )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        str(root), local_files_only=True, trust_remote_code=False, use_safetensors=True
    )
    if model.config.model_type != "xlm-roberta" or model.config.num_labels != 1:
        raise ValueError("incompatible_reranker_model")
    if device == "cpu":
        # Safetensors may retain a file-backed mapping on CPU. Allocate owned
        # storage so subsequent disk edits cannot alter verified resident weights.
        # Reassign cloned state without disturbing initialized nonpersistent
        # position/token buffers that are absent from the model state dict.
        state = {name: value.clone() for name, value in model.state_dict().items()}
        model.load_state_dict(state, strict=True, assign=True)
        del state
    else:
        model.to(device)
    model.eval()
    return torch, tokenizer, model


class BgeReranker:
    """Keep verified weights resident; serialize inference and explicit close.

    Files are hashed before and after loading. Inference uses in-memory weights
    and tokenizer only; edits on disk are observed at the next acquisition.
    Operators must prevent concurrent writes while loading local assets.
    """

    model_id = MODEL_ID
    model_revision = "sha256:" + MANIFEST_SHA256

    def __init__(
        self, root: Path, *, device: str = "cpu", batch_size: int = 8, max_tokens: int = 512
    ) -> None:
        if not re.fullmatch(r"cpu|mps|cuda(?::[0-9]+)?", device):
            raise ValueError("invalid reranker device")
        if (
            type(batch_size) is not int
            or not 1 <= batch_size <= 128
            or type(max_tokens) is not int
            or not 1 <= max_tokens <= 8192
        ):
            raise ValueError("invalid reranker limits")
        assets = reranker_assets(root)
        assets.verify()
        self._lock = RLock()
        self._runtime: tuple[Any, Any, Any] | None = _load_runtime(assets.root, device)
        try:
            assets.verify()
        except BaseException:
            self._runtime = None
            raise
        self.device = device
        self.batch_size = batch_size
        self.max_tokens = max_tokens

    def rerank(
        self, query: str, documents: tuple[str, ...], *, top_n: int | None = None
    ) -> RerankResult:
        """Reject overlong pairs before any forward pass; never silently truncate."""
        count = validate_rerank_request(query, documents, top_n)
        with self._lock:
            if self._runtime is None:
                raise RuntimeError("reranker is closed")
            torch, tokenizer, model = self._runtime
            encoded = tokenizer(
                [(query, doc) for doc in documents],
                padding=False,
                truncation=True,
                max_length=self.max_tokens + 1,
            )
            lengths = [len(ids) for ids in encoded["input_ids"]]
            if any(length > self.max_tokens for length in lengths) or sum(lengths) > 32768:
                raise ValueError("rerank_input_too_long")
            scores: list[float] = []
            with torch.inference_mode():
                for start in range(0, len(documents), self.batch_size):
                    batch = tokenizer.pad(
                        {
                            key: value[start : start + self.batch_size]
                            for key, value in encoded.items()
                        },
                        padding=True,
                        return_tensors="pt",
                    ).to(self.device)
                    values = model(**batch).logits.reshape(-1).float().cpu().tolist()
                    if len(values) != len(batch["input_ids"]):
                        raise ValueError("invalid_reranker_output")
                    scores.extend(values)
            if len(scores) != len(documents):
                raise ValueError("invalid_reranker_output")
            return RerankResult(
                self.model_id,
                self.model_revision,
                rank_scores(tuple(scores), top_n=count),
                sum(lengths),
            )

    def close(self) -> None:
        """Release resident model references, waiting for active inference."""
        with self._lock:
            self._runtime = None
