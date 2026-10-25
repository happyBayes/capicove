"""Local generation uses injected runtimes, never real models in ordinary tests."""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from capicove.graph_build._adapters.local_generation import (
    LocalStructuredGenerator,
    LocalTextTokenizer,
)
from capicove.graph_build.extraction import extraction_request
from capicove.graph_build.inference import DispatchControl, InferenceRunner
from capicove.graph_build.profiles import LocalGenerationProfile, StructuredRequest
from capicove.model.local_assets import LocalModelAssets
from tests.unit.graph_build.test_extraction import build_profile, unit
from tests.unit.graph_build.test_inference import SCOPE, SOURCES, Harness, validate
from tests.unit.model.test_local_assets import assets_at


TEMPLATE = "<{{ messages }}:assistant>"
GOOD = '{"entities":[],"relationships":[]}'


def test_request_tokenizer_accepts_transformers_batch_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = Runtime(monkeypatch)
    monkeypatch.setattr(
        runtime.tokenizer,
        "apply_chat_template",
        lambda *args, **kwargs: {
            "input_ids": [1, 10, 20],
            "attention_mask": [1, 1, 1],
        },
    )
    profile = profile_at(tmp_path)
    assert LocalTextTokenizer(profile).count_request(request_for(profile)) == 3


class Tensor:
    """Small CPU tensor stand-in at the optional runtime boundary."""

    def __init__(self, rows: list[list[int]], device: str = "cpu") -> None:
        self.rows = rows
        self.device = device
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, key: int) -> Tensor:
        return Tensor([self.rows[key]], self.device)

    def tolist(self) -> list[int]:
        return self.rows[0]


class FakeTokenizer:
    model_max_length = 32768
    chat_template = TEMPLATE
    eos_token_id = 0
    pad_token_id = 0
    bos_token_id = 1

    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []
        self.decoded: list[int] = []
        self.init_kwargs: dict[str, str] = {}

    def __len__(self) -> int:
        return 256

    def encode(self, text: str, *, add_special_tokens: bool, truncation: bool) -> list[int]:
        assert not truncation
        return ([1] if add_special_tokens else []) + [ord(char) for char in text]

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> list[int]:
        assert kwargs == {"tokenize": True, "add_generation_prompt": True, "truncation": False}
        self.messages = messages
        wire = (
            "<system>" + messages[0]["content"] + "<user>" + messages[1]["content"] + "<assistant>"
        )
        return [1] + [ord(char) for char in wire]

    def decode(self, ids: list[int], **kwargs: Any) -> str:
        assert kwargs == {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
        self.decoded = ids
        return "".join(chr(item) for item in ids if item != 0)


class Runtime:
    """Record loading/generation boundaries while returning authored token IDs."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tokenizer = FakeTokenizer()
        self.config = SimpleNamespace(
            max_position_embeddings=32768,
            vocab_size=256,
            is_encoder_decoder=False,
            eos_token_id=0,
            bos_token_id=1,
            pad_token_id=0,
        )
        self.loads: list[tuple[str, str, dict[str, Any]]] = []
        self.kwargs: dict[str, Any] = {}
        self.now = 10.0
        self.cancelled = False
        self.action = "success"
        self.new_tokens = [ord(char) for char in GOOD] + [0]
        self.devices: list[str] = []
        self.error: Exception | None = None
        self.events: list[str] = []
        self.stop_at: str | None = None
        self.stop_mode = "cancel"
        self.model = SimpleNamespace(
            config=self.config,
            to=self.to,
            eval=lambda: self.stage("model_eval"),
            get_input_embeddings=lambda: SimpleNamespace(num_embeddings=256),
            generate=self.generate,
        )
        transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=self.load_tokenizer),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=self.load_model),
            GenerationConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            StoppingCriteria=object,
            StoppingCriteriaList=list,
        )
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
            inference_mode=contextlib.nullcontext,
            tensor=lambda rows, **kwargs: Tensor(rows, kwargs["device"]),
            ones_like=lambda tensor: Tensor([[1] * tensor.shape[1]], tensor.device),
            long="long",
        )
        monkeypatch.setitem(sys.modules, "transformers", transformers)
        monkeypatch.setitem(sys.modules, "torch", torch)

    def load_tokenizer(self, root: str, **kwargs: Any) -> FakeTokenizer:
        self.loads.append(("tokenizer", root, kwargs))
        if self.error:
            raise self.error
        self.stage("tokenizer_load")
        return self.tokenizer

    def load_model(self, root: str, **kwargs: Any) -> Any:
        self.loads.append(("model", root, kwargs))
        self.stage("model_load")
        return self.model

    def to(self, device: str) -> Any:
        self.devices.append(device)
        self.stage("device_transfer")
        return self.model

    def stage(self, name: str) -> None:
        """Stop at the completion of an authored synchronous runtime operation."""
        self.events.append(name)
        if self.stop_at == name:
            if self.stop_mode == "cancel":
                self.cancelled = True
            else:
                self.now = 15.0

    def control(self) -> DispatchControl:
        return DispatchControl(15.0, lambda: self.now, lambda: self.cancelled)

    def generate(self, **kwargs: Any) -> Tensor:
        self.kwargs = kwargs
        if self.action == "cancel":
            self.cancelled = True
        elif self.action == "deadline":
            self.now = 15.0
        if self.action in {"cancel", "deadline"}:
            assert kwargs["stopping_criteria"][0](kwargs["input_ids"], None)
        if self.action == "error":
            raise RuntimeError("secret prompt and /private/path")
        prompt = kwargs["input_ids"].rows[0]
        if self.action == "bad_prefix":
            prompt = [99] * len(prompt)
        return Tensor([prompt + self.new_tokens])


def profile_at(tmp_path: Path, *, model_revision: str | None = None) -> LocalGenerationProfile:
    model = assets_at(
        tmp_path / "model",
        {
            "config.json": b'{"max_position_embeddings":32768,"vocab_size":256}',
            "model.safetensors": b"synthetic weights",
        },
    )
    tokenizer = assets_at(
        tmp_path / "tokenizer",
        {
            "tokenizer_config.json": json.dumps(
                {"model_max_length": 32768, "chat_template": TEMPLATE}
            ).encode(),
            "tokenizer.json": b"{}",
        },
    )
    return LocalGenerationProfile(
        model,
        tokenizer,
        model_revision or "sha256:" + model.manifest_sha256,
        "sha256:" + tokenizer.manifest_sha256,
        "cpu",
        32768,
        128,
    )


def request_for(profile: LocalGenerationProfile, **changes: Any) -> StructuredRequest:
    policy = replace(
        build_profile().inference,
        provider_id="local",
        model_id="sha256:" + profile.assets.manifest_sha256,
        model_revision=profile.model_revision,
        tokenizer_id="sha256:" + profile.tokenizer_assets.manifest_sha256,
        tokenizer_revision=profile.tokenizer_revision,
        context_window_tokens=profile.context_limit,
        max_output_tokens=128,
        **changes,
    )
    return extraction_request(unit(), replace(build_profile(), inference=policy))


def test_adapter_import_and_construction_do_not_start_optional_runtimes(tmp_path: Path) -> None:
    profile = profile_at(tmp_path)
    assert LocalStructuredGenerator(profile).model_revision == profile.model_revision
    assert LocalTextTokenizer(profile).fingerprint == LocalTextTokenizer(profile).fingerprint
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "import capicove.graph_build._adapters.local_generation; "
            "assert not {'torch', 'transformers'} & sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_exact_request_count_local_arguments_and_prompt_stripping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = Runtime(monkeypatch)
    profile = profile_at(tmp_path)
    tokenizer = LocalTextTokenizer(profile)
    generator = LocalStructuredGenerator(profile)
    request = request_for(profile)
    count = tokenizer.count_request(request)
    assert tokenizer.count("abc") == 3
    result = generator.generate(request, control=runtime.control())
    assert result.payload == GOOD.encode()
    assert (result.finish_reason, result.input_tokens, result.output_tokens) == (
        "stop",
        count,
        len(GOOD) + 1,
    )
    assert count > len(request.canonical_payload)
    assert [item["role"] for item in runtime.tokenizer.messages] == ["system", "user"]
    assert runtime.tokenizer.messages[1]["content"] == request.canonical_payload.decode()
    assert runtime.tokenizer.decoded == runtime.new_tokens
    assert runtime.devices == ["cpu"]
    for kind, root, kwargs in runtime.loads:
        assert root == str(
            profile.tokenizer_assets.root if kind == "tokenizer" else profile.assets.root
        )
        assert kwargs == (
            {"local_files_only": True, "trust_remote_code": False}
            | ({"use_safetensors": True} if kind == "model" else {})
        )
    decoding = runtime.kwargs["generation_config"]
    assert decoding.max_new_tokens == 128
    assert decoding.do_sample is False and decoding.num_beams == 1
    assert runtime.kwargs["max_time"] == 5.0


@pytest.mark.parametrize(
    "change",
    ["model_window", "tokenizer_window", "template", "vocabulary", "revision", "unsafe_weights"],
)
def test_incompatible_assets_fail_before_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    runtime = Runtime(monkeypatch)
    profile = profile_at(tmp_path)
    if change == "model_window":
        runtime.config.max_position_embeddings = 64
    elif change == "tokenizer_window":
        runtime.tokenizer.model_max_length = 64
    elif change == "template":
        runtime.tokenizer.chat_template = "different template"
    elif change == "vocabulary":
        runtime.config.vocab_size = 100
    elif change == "revision":
        profile = replace(profile, model_revision="a" * 40)
    else:
        profile = replace(profile, assets=assets_at(tmp_path / "other", {"config.json": b"{}"}))
        profile = replace(profile, model_revision="sha256:" + profile.assets.manifest_sha256)
    with pytest.raises(ValueError):
        LocalStructuredGenerator(profile).generate(request_for(profile), control=runtime.control())
    assert not runtime.kwargs


@pytest.mark.parametrize("device", ["mps", "cuda:0"])
def test_unavailable_explicit_device_cannot_fall_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    runtime = Runtime(monkeypatch)
    profile = replace(profile_at(tmp_path), device=device)
    with pytest.raises(ValueError, match="device_unavailable"):
        LocalStructuredGenerator(profile).generate(request_for(profile), control=runtime.control())
    assert not runtime.devices and not runtime.kwargs


@pytest.mark.parametrize("change", ["role", "temperature", "output", "identity", "context"])
def test_request_policy_must_match_local_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    runtime = Runtime(monkeypatch)
    profile = profile_at(tmp_path)
    changes_by_case: dict[str, dict[str, Any]] = {
        "role": {"input_role": "system"},
        "temperature": {"temperature": 1.0},
        "output": {},
        "identity": {},
        "context": {},
    }
    changes = changes_by_case[change]
    request = request_for(profile, **changes)
    if change == "output":
        profile = replace(profile, max_new_tokens=64)
    elif change == "identity":
        profile = profile_at(tmp_path / "different", model_revision="b" * 40)
    elif change == "context":
        profile = replace(profile, context_limit=256)
    with pytest.raises(ValueError):
        LocalStructuredGenerator(profile).generate(request, control=runtime.control())
    assert not runtime.kwargs


def test_exact_wire_reservation_is_rejected_without_truncation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = Runtime(monkeypatch)
    profile = replace(profile_at(tmp_path), context_limit=256)
    with pytest.raises(ValueError, match="context_window_exceeded"):
        LocalStructuredGenerator(profile).generate(request_for(profile), control=runtime.control())
    assert not runtime.kwargs


@pytest.mark.parametrize("mode", ["tokens", "bytes", "bad_prefix", "length", "unknown"])
def test_output_limits_and_finish_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    runtime = Runtime(monkeypatch)
    profile = profile_at(tmp_path)
    request = request_for(profile)
    if mode == "tokens":
        runtime.new_tokens = [65] * 129
    elif mode == "bytes":
        build = replace(
            build_profile(),
            inference=request.inference_policy,
            extraction_limits=replace(request.output_limits, max_response_bytes=4),
        )
        request = extraction_request(unit(), build)
    elif mode == "bad_prefix":
        runtime.action = mode
    else:
        runtime.new_tokens = [65] * (128 if mode == "length" else 2)
    if mode in {"tokens", "bytes", "bad_prefix"}:
        with pytest.raises(ValueError):
            LocalStructuredGenerator(profile).generate(request, control=runtime.control())
    else:
        result = LocalStructuredGenerator(profile).generate(request, control=runtime.control())
        assert result.finish_reason == mode


@pytest.mark.parametrize("mode", ["cancel", "deadline"])
def test_live_dispatch_limits_stop_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    runtime = Runtime(monkeypatch)
    runtime.action = mode
    profile = profile_at(tmp_path)
    if mode == "deadline":
        with pytest.raises(TimeoutError, match="generation_deadline_exceeded"):
            LocalStructuredGenerator(profile).generate(
                request_for(profile), control=runtime.control()
            )
    else:
        result = LocalStructuredGenerator(profile).generate(
            request_for(profile), control=runtime.control()
        )
        assert result.finish_reason == "cancelled"
        assert result.output_tokens == len(GOOD) + 1


@pytest.mark.parametrize("stage", ["load", "generate"])
def test_provider_exceptions_do_not_expose_paths_or_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    runtime = Runtime(monkeypatch)
    profile = profile_at(tmp_path)
    if stage == "load":
        runtime.error = RuntimeError("secret prompt and /private/path")
    else:
        runtime.action = "error"
    with pytest.raises(ValueError, match=r"^local_runtime_failure$") as error:
        LocalStructuredGenerator(profile).generate(request_for(profile), control=runtime.control())
    assert error.value.__suppress_context__


def test_tokenizer_fingerprint_pins_assets_and_revision(tmp_path: Path) -> None:
    profile = profile_at(tmp_path)
    tokenizer = LocalTextTokenizer(profile)
    assert len(tokenizer.fingerprint) == 64
    changed = replace(profile, tokenizer_revision="a" * 40)
    assert LocalTextTokenizer(changed).fingerprint != tokenizer.fingerprint


def test_source_revisions_are_bound_to_verified_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = Runtime(monkeypatch)
    profile = profile_at(tmp_path)
    revision = "a" * 40
    model = assets_at(
        tmp_path / "pinned_model",
        {
            "config.json": json.dumps(
                {"_commit_hash": revision, "max_position_embeddings": 32768, "vocab_size": 256}
            ).encode(),
            "model.safetensors": b"synthetic weights",
        },
    )
    tokenizer = assets_at(
        tmp_path / "pinned_tokenizer",
        {
            "tokenizer_config.json": json.dumps(
                {"_commit_hash": revision, "model_max_length": 32768, "chat_template": TEMPLATE}
            ).encode(),
            "tokenizer.json": b"{}",
        },
    )
    profile = replace(
        profile,
        assets=model,
        tokenizer_assets=tokenizer,
        model_revision=revision,
        tokenizer_revision=revision,
    )
    runtime.config._commit_hash = revision
    runtime.tokenizer.init_kwargs = {"_commit_hash": revision}
    assert (
        LocalStructuredGenerator(profile)
        .generate(request_for(profile), control=runtime.control())
        .payload
        == GOOD.encode()
    )
    runtime.tokenizer.init_kwargs = {"_commit_hash": "b" * 40}
    with pytest.raises(ValueError, match="tokenizer_revision_mismatch"):
        LocalTextTokenizer(profile).count_request(request_for(profile))


@pytest.mark.parametrize("mode", ["cancel", "deadline"])
def test_expired_dispatch_does_not_load_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    runtime = Runtime(monkeypatch)
    profile = profile_at(tmp_path)
    if mode == "cancel":
        runtime.cancelled = True
    else:
        runtime.now = 15.0
    if mode == "deadline":
        with pytest.raises(TimeoutError):
            LocalStructuredGenerator(profile).generate(
                request_for(profile), control=runtime.control()
            )
    else:
        result = LocalStructuredGenerator(profile).generate(
            request_for(profile), control=runtime.control()
        )
        assert (result.finish_reason, result.input_tokens, result.output_tokens) == (
            "cancelled",
            0,
            0,
        )
    assert not runtime.loads


def test_missing_local_model_fails_before_optional_runtime_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = Runtime(monkeypatch)
    profile = profile_at(tmp_path)
    profile = replace(profile, assets=replace(profile.assets, root=tmp_path / "absent"))
    with pytest.raises(ValueError, match="model_unavailable"):
        LocalStructuredGenerator(profile).generate(request_for(profile), control=runtime.control())
    assert not runtime.loads


def test_missing_optional_runtime_is_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = profile_at(tmp_path)
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ValueError, match="model_unavailable"):
        LocalTextTokenizer(profile).count("abc")


@pytest.mark.parametrize(
    "field,value",
    [
        ("device", "auto"),
        ("device", "cuda"),
        ("model_revision", "main"),
        ("tokenizer_revision", "latest"),
        ("max_new_tokens", True),
        ("context_limit", 128),
    ],
)
def test_local_profile_rejects_implicit_or_unbounded_choices(
    tmp_path: Path, field: str, value: Any
) -> None:
    with pytest.raises((ValueError, TypeError)):
        replace(profile_at(tmp_path), **{field: value})


def test_cooperative_cancellation_persists_a_terminal_runner_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = Runtime(monkeypatch)
    runtime.action = "cancel"
    profile = profile_at(tmp_path / "assets")
    harness = Harness(tmp_path / "runner", [])
    runner = InferenceRunner(
        generator=LocalStructuredGenerator(profile),
        tokenizer=LocalTextTokenizer(profile),
        gate=harness.gate,
        artifact_factory=lambda refs: harness.artifacts,
        receipts=harness.receipts,
        clock=lambda: runtime.now,
        wall_clock=harness.clock,
        sleeper=harness.clock.sleep,
        cancelled=lambda: runtime.cancelled,
    )
    receipt = runner.run(SCOPE, request_for(profile), SOURCES, validate)
    assert receipt.status == "cancelled"
    assert receipt.issue_codes == ("cancelled",)
    assert harness.receipts.get(SCOPE, receipt.key) == receipt


@pytest.mark.parametrize("stop_mode", ["deadline", "cancel"])
@pytest.mark.parametrize(
    "stage",
    [
        "model_verify_1",
        "model_config_read_1",
        "tokenizer_verify_1",
        "tokenizer_config_read_1",
        "tokenizer_load",
        "tokenizer_verify_2",
        "model_load",
        "device_transfer",
        "model_eval",
        "model_verify_2",
    ],
)
def test_loading_stops_before_starting_another_expensive_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop_mode: str, stage: str
) -> None:
    runtime = Runtime(monkeypatch)
    profile = profile_at(tmp_path)
    runtime.stop_at = stage
    runtime.stop_mode = stop_mode
    counts: dict[str, int] = {}
    original_verify = LocalModelAssets.verify
    original_read = Path.read_bytes

    def record(name: str) -> None:
        counts[name] = counts.get(name, 0) + 1
        runtime.stage(f"{name}_{counts[name]}")

    def recorded_verify(assets: LocalModelAssets) -> None:
        original_verify(assets)
        name = "model" if assets.root == profile.assets.root else "tokenizer"
        record(f"{name}_verify")

    def recorded_read(path: Path) -> bytes:
        result = original_read(path)
        if path.parent in (profile.assets.root, profile.tokenizer_assets.root):
            name = "model" if path.parent == profile.assets.root else "tokenizer"
            record(f"{name}_config_read")
        return result

    monkeypatch.setattr(LocalModelAssets, "verify", recorded_verify)
    monkeypatch.setattr(Path, "read_bytes", recorded_read)
    request = request_for(profile)
    if stop_mode == "deadline":
        with pytest.raises(TimeoutError, match="generation_deadline_exceeded"):
            LocalStructuredGenerator(profile).generate(request, control=runtime.control())
    else:
        result = LocalStructuredGenerator(profile).generate(request, control=runtime.control())
        assert (result.finish_reason, result.input_tokens, result.output_tokens) == (
            "cancelled",
            0,
            0,
        )
    assert stage in runtime.events
    assert runtime.events[runtime.events.index(stage) + 1 :] == []
    assert not runtime.kwargs
    assert not runtime.tokenizer.messages or stage not in {"tokenizer_load", "tokenizer_verify_2"}
