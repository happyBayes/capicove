# Perceptual image deduplication

Perceptual deduplication groups visually similar images using perceptual hashes. It
is designed for near-duplicate curation and does **not** prove byte identity.

`ImagePerceptualDeduplicator` is the local implementation. Install
`capicove[image-dedup]` for the imagededup adapter or `capicove[image-ray]` for the
Ray-backed execution path. Inputs and thresholds are explicit, and results preserve
which records were retained or grouped.

Use a cryptographic content hash when exact byte identity is required.

## Produce public pHashes

`ImagePhash` returns fixed-width lowercase 64-bit hashes, source SHA-256 values,
provider/hash lineage, and batch-local duplicate groups:

```python
from capicove.processing.image import ImagePhash, ImageSource
from capicove.processing.image.deduplication import ImageRecord

records = (
    ImageRecord("first", (ImageSource(path="images/first.png"),)),
    ImageRecord("second", (ImageSource(path="images/second.jpg"),)),
)
result = ImagePhash(max_hamming_distance=2, enable_lsh=True).process(records)

print(result.records[0].hashes)
print(result.groups)
```

`enable_lsh=True` accelerates candidate comparison within the batch. It does not
accelerate decoding or pHash calculation. LSH candidates are always verified with
exact Hamming distance, so connected components and stable winners match the
exhaustive path.

## Search existing hashes

`ImageLshSearch` is useful when hashes have already been computed:

```python
from capicove.processing.image import ImageLshSearch
from capicove.processing.image.deduplication import HashRecord

result = ImageLshSearch().search(
    (0x0123456789ABCDEF,),
    (
        HashRecord("first", (0x0123456789ABCDEE,)),
        HashRecord("other", (0xFFFFFFFFFFFFFFFF,)),
    ),
    hash_version="phash-64-v1",
    max_hamming_distance=2,
    top_k=100,
)
```

The search is bounded and stateless. Candidates must use the same hash version,
image count, and optional text hash. Results sort by exact distance and then stable
record ID. No persistent index or network service is created.
