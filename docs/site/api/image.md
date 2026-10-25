# Image processing API

::: capicove.processing.image
    options:
      show_if_no_docstring: true

## pHash and LSH example

Integration example: provide a readable local `asset.png` before running this
block. It computes pHashes and enables invocation-local LSH acceleration:

```python
from capicove.processing.image import ImagePhash, ImageSource
from capicove.processing.image.deduplication import ImageRecord

result = ImagePhash(enable_lsh=True).process(
    (ImageRecord("asset", (ImageSource(path="asset.png"),)),),
    max_hamming_distance=2,
)
```

Search previously computed compatible hashes without reading images or creating a
persistent index:

```python
from capicove.processing.image import ImageLshSearch
from capicove.processing.image.deduplication import HashRecord

matches = (
    ImageLshSearch()
    .search(
        (0x0123456789ABCDEF,),
        (HashRecord("asset", (0x0123456789ABCDEE,)),),
        hash_version="phash-64-v1",
        max_hamming_distance=2,
    )
    .matches
)
```
