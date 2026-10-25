# Image filters

Public filters cover size, shape, aspect ratio, subplot structure, watermark score,
aesthetics score, NSFW score, pair similarity, text matching, and text similarity.
Each filter returns a `FilterDecision` containing the decision, metrics, and
evidence instead of mutating or deleting source files.

Rule-based filters run with base dependencies. Learned filters consume injected
local scorer ports; callers choose and provision the model implementation.

```python
from capicove.processing.image import ImageSizeFilter

image_filter = ImageSizeFilter(min_size="100KB", max_size="10MB")
```

Consult the [image API](../../api/image.md) for constructor and result contracts.
