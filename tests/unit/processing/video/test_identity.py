from capicove.processing.video.identity import configuration_hash, output_identity


def test_configuration_hash_is_independent_of_mapping_order() -> None:
    left = configuration_hash({"threshold": 27.0, "min_scene_len": 15})
    right = configuration_hash({"min_scene_len": 15, "threshold": 27.0})

    assert left == right


def test_configuration_hash_normalizes_nested_values() -> None:
    assert configuration_hash({"detector": "content", "options": {"x": 1}}) == (
        configuration_hash({"options": {"x": 1}, "detector": "content"})
    )


def test_output_identity_changes_with_segment_index() -> None:
    first = output_identity("a" * 64, "duration", "b" * 64, "ffmpeg:7", 0)
    second = output_identity("a" * 64, "duration", "b" * 64, "ffmpeg:7", 1)

    assert first != second
