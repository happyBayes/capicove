"""Run authored video through preparation, physical publication and transports."""

import json
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from capicove_retriever_elasticsearch.adapter import ElasticsearchTextRetriever
from capicove_retriever_elasticsearch.config import ElasticsearchConfiguration
from fastapi.testclient import TestClient

from capicove.contracts import ProcessingFingerprint
from capicove.processing.video.models import VideoSource
from capicove.processing.video.retrieval.models import VideoRetrievalPreparationRequest
from tests.integration.processing.video.test_retrieval_preparation import authored_video
from tests.unit.mcp.test_search import Server
from tests.unit.retrieval.test_video_models import visual_space
from tests.unit.tools.test_search import context


__all__ = ["authored_video"]


@pytest.mark.skipif(
    os.environ.get("CAPICOVE_TEST_REAL_ELASTICSEARCH") != "1",
    reason="explicit local TLS integration opt-in required",
)
@pytest.mark.parametrize("real_model", [False, True])
def test_local_build_reopen_and_query(
    tmp_path: Path, authored_video: bytes, capsys: pytest.CaptureFixture[str], real_model: bool
) -> None:
    from capicove.processing.image import ImageOcr, OcrWord
    from capicove.processing.image.ocr import OcrProviderOutput
    from capicove.runtime.video_search import LocalVideoSearchRuntime

    space = visual_space()
    local_model = None
    if real_model:
        if not os.environ.get("CAPICOVE_TEST_VIDEO_MODEL_ENDPOINT"):
            pytest.skip("explicit provisioned local model endpoint required")
        from capicove.model import load_packaged_manifest
        from capicove.retrieval.video_http_encoding import LocalSiglip2QueryEncoder

        spec = next(
            m for m in load_packaged_manifest().models if m.key == "siglip2-base-patch16-224"
        )
        space = replace(
            space,
            provider_id=spec.repository,
            model_revision=spec.revision,
            preprocessing_version=spec.preprocess_revision,
            dimension=768,
        )
        local_model = LocalSiglip2QueryEncoder(
            endpoint=os.environ["CAPICOVE_TEST_VIDEO_MODEL_ENDPOINT"],
            model_id=spec.repository,
            embedding_space=space,
        )

    class Ocr:
        provider_id = "authored-ocr"
        provider_version = "1"

        def recognize(self, image, languages):
            return OcrProviderOutput((OcrWord("red", 0, 0, 8, 8, 0.9, 1, 1, 1, 1, 1),))

    class Encoder:
        calls = 0

        def encode_preview_batch(self, previews):
            if local_model is not None:
                return local_model.encode_preview_batch(previews)
            return tuple((1.0, 0.0) for _ in previews)

        def encode(self, query, *, embedding_space, deadline):
            self.calls += 1
            if local_model is not None:
                return local_model.encode(query, embedding_space=embedding_space, deadline=deadline)
            return (1.0, 0.0)

    encoder = Encoder()
    backend = ElasticsearchTextRetriever(
        ElasticsearchConfiguration(
            hosts=(os.environ["CAPICOVE_VIDEO_ES_ENDPOINT"],),
            ca_certs=os.environ["CAPICOVE_VIDEO_ES_CA"],
            api_key=os.environ.get("CAPICOVE_TEST_ES_API_KEY"),
            index_prefix="capicove-video-build",
            active_alias="capicove-video-build-active",
        )
    )

    def runtime():
        return LocalVideoSearchRuntime(
            workspace_root=tmp_path,
            workspace_ref="build-to-query",
            corpus_refs=("authored",),
            embedding_space=space,
            encoder=encoder,
            text_backend=backend,
            text_index_format="elasticsearch-text-v1",
            image_ocr=ImageOcr(Ocr()),
            ocr_provider_fingerprint=ProcessingFingerprint("d" * 64),
        )

    fp = ProcessingFingerprint("b" * 64)
    request = VideoRetrievalPreparationRequest(
        VideoSource(data=authored_video),
        "artifact:authored",
        "evidence:authored",
        "representation:asr",
        "run:authored",
        "video-local-exact",
        fp,
        fp,
        fp,
        datetime.now(UTC) + timedelta(seconds=60),
    )
    with runtime() as built:
        publication = built.build(request)
        assert publication.warning_codes == ("asr_not_configured",)
        assert publication.generation_set_id
    with runtime() as reopened:
        arguments = {"query": {"type": "text", "text": "red"}, "targets": ["video_segment"]}
        direct = reopened.search_handler(context(), arguments)
        assert direct.result and direct.result["hits"]
        assert encoder.calls == 1  # One shared query vector for both visual lanes.
        assert direct.result["degraded"] is False
        with reopened.components.generation_sets.lease_active(
            workspace_ref="build-to-query", profile_ref="video-local-exact"
        ) as pin:
            assert pin.ocr_text is not None
        server = Server()
        callback = reopened.register_mcp(server)
        mcp = callback(arguments)
        with TestClient(reopened.http_app()) as client:
            http = client.post("/v1/search", json=arguments).json()
        assert reopened.cli(["search", "red", "--target", "video_segment"]) == 0
        cli = json.loads(capsys.readouterr().out)
        for result in (mcp, http, cli):
            assert result["result"]["hits"] == direct.result["hits"]
        with pytest.raises(ValueError, match="authorizer"):
            source = tmp_path / "authored.mp4"
            source.write_bytes(authored_video)
            reopened.build(replace(request, source=VideoSource(path=source)))
