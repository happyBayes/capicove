"""Exercise real Torch conversion with authored tensors and no model downloads."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from capicove.processing.video.filtering._adapters.huggingface_frame_scorers import (
    ResolvedHuggingFaceModel,
    _clip_inference,
)
from capicove.processing.video.filtering.video_filter_ports import (
    ModelIncompatibleFilterError,
    SampledFrame,
)


pytestmark = pytest.mark.model_logic


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("invalid", [None, "nan", "zero", "shape"])
def test_clip_tensor_boundary(structured: bool, invalid: str | None) -> None:
    torch = pytest.importorskip("torch")

    def processor(**kwargs: Any) -> dict[str, Any]:
        images = kwargs["images"]
        assert images[0][0, 0].tolist() == [30, 20, 10]
        return {"pixel_values": torch.tensor(np.stack(images)), "input_ids": torch.tensor([[1]])}

    def features(**kwargs: Any) -> Any:
        values = torch.tensor([[1.0, 0.0]])
        if invalid == "nan":
            values[0, 0] = float("nan")
        elif invalid == "zero":
            values.zero_()
        elif invalid == "shape":
            values = values.flatten()
        return SimpleNamespace(pooler_output=values) if structured else values

    resolved = ResolvedHuggingFaceModel(
        "local",
        None,
        "a" * 64,
        SimpleNamespace(get_image_features=features, get_text_features=features),
        processor,
        {},
    )
    frame = SampledFrame(0, 0.0, 2, 2, np.full((2, 2, 3), [10, 20, 30], dtype=np.uint8))
    if invalid:
        with pytest.raises(ModelIncompatibleFilterError):
            _clip_inference(resolved, (frame,), "red")
    else:
        assert _clip_inference(resolved, (frame,), "red") == ([1.0], [1.0])
