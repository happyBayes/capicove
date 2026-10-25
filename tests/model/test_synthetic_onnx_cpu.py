"""Lightweight ONNX Runtime checks that do not download model weights."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


onnx = pytest.importorskip("onnx", reason="install the model-test dependency group")
ort = pytest.importorskip("onnxruntime", reason="install capicove[models-cpu]")

pytestmark = [pytest.mark.model_onnx, pytest.mark.cpu]


def test_synthetic_onnx_runs_on_cpu_without_torch(tmp_path: Path) -> None:
    """Execute a tiny Add graph without installing the Torch runtime."""
    assert importlib.util.find_spec("torch") is None
    assert "torch" not in sys.modules

    left = onnx.helper.make_tensor_value_info("left", onnx.TensorProto.FLOAT, [1, 2])
    right = onnx.helper.make_tensor_value_info("right", onnx.TensorProto.FLOAT, [1, 2])
    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 2])
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Add", ["left", "right"], ["output"])],
        "synthetic_cpu_add",
        [left, right],
        [output],
    )
    model = onnx.helper.make_model(
        graph,
        ir_version=9,
        opset_imports=[onnx.helper.make_opsetid("", 17)],
    )
    model_path = tmp_path / "synthetic.onnx"
    onnx.save(model, model_path)

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    actual = session.run(
        ["output"],
        {
            "left": np.asarray([[1.0, 2.0]], dtype=np.float32),
            "right": np.asarray([[3.0, 4.0]], dtype=np.float32),
        },
    )[0]

    np.testing.assert_array_equal(actual, np.asarray([[4.0, 6.0]], dtype=np.float32))
    assert session.get_providers() == ["CPUExecutionProvider"]
