# Video processing

Choose a video workflow:

- [splitting](splitting.md) by duration, keyframes, or scenes;
- [filtering](filtering.md) across metadata, visual, text, tag, and model criteria;
- [exact-file deduplication](../../cli/video-deduplicate.md) using the `video-dedup` extra;
- [retrieval](retrieval.md), a locally qualified experimental workflow requiring explicit index,
  encoder, and authorization configuration.

These workflows accept explicit local inputs and return structured results rather
than silently treating partial work as success. Source identity, backend lineage,
warnings, and failures remain available to callers.

Use [video splitting](splitting.md) when you need reproducible segments. Use
[video filtering](filtering.md) when you need to decide which assets satisfy a set
of criteria.
