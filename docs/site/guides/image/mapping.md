# Image mapping and comparison

Image mappers produce derived evidence without overwriting the source. Current
operations include background removal, difference-area generation, and
difference-caption generation.

Difference results separate deterministic image measurements from optional model
captions. Derived files carry source and output hashes so callers can retain
lineage. Background removal requires an injected provider and explicit output
directory; the optional rembg adapter is installed with `capicove[image-rembg]`.

Comparison metrics such as bounding-box IoU, binary-mask IoU, coefficient of
variation, and information gain are available from the public image package.
