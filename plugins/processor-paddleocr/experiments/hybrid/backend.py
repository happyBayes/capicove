"""Experimental local hybrid parser using existing public document contracts."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import platform
import subprocess
import tempfile
import unicodedata
from dataclasses import asdict
from pathlib import Path

from capicove_processor_paddleocr.backend import (
    _dominant_unsupported_script,
    _snapshot_authorized_source,
    _validate_ocr_language_hints,
)
from capicove_processor_paddleocr.models import load_book_balanced_manifest
from capicove_processor_paddleocr.runtime import (
    enforce_offline_runtime_policy,
    resolve_runtime_identity,
)
from evidence import assess, contains, merge_supplements, normalize
from fraction_layout import recognize

from capicove.contracts import (
    DocumentBlock,
    DocumentBlockType,
    DocumentPage,
    DocumentPageStatus,
    DocumentParseResult,
    DocumentParserReceipt,
    LocalPathSource,
    NormalizedBox,
)


ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def native_normalize(text):
    text = "".join(chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c for c in text)
    return normalize(text).replace("\u2013", "-")


def has_han(text):
    return any("CJK" in unicodedata.name(c, "") for c in text)


def model_for_region(region_text, document_text, language_hints):
    if (
        has_han(region_text)
        or has_han(document_text)
        or any(h.split("-")[0] in {"zh", "ja"} for h in language_hints)
    ):
        return "server"
    return "en"


def can_extract_raster(page):
    """Only bypass rendering for a proven untransformed opaque full-page raster."""
    if page.get("/Annots") or page.get("/Rotate", 0) or list(page.cropbox) != list(page.mediabox):
        return False
    if float(page.mediabox.left) != 0 or float(page.mediabox.bottom) != 0:
        return False
    contents = page.get_contents()
    operations = [] if contents is None else contents.operations
    if [op for _, op in operations] != [b"q", b"cm", b"Do", b"Q"]:
        return False
    matrix = list(map(float, operations[1][0]))
    if matrix != [
        float(page.mediabox.width),
        0.0,
        0.0,
        float(page.mediabox.height),
        0.0,
        0.0,
    ]:
        return False
    images = list(page.images)
    if len(images) != 1:
        return False
    obj = images[0].indirect_reference.get_object()
    return (
        obj.get("/ColorSpace") == "/DeviceGray"
        and obj.get("/BitsPerComponent") == 1
        and not obj.get("/Mask")
        and not obj.get("/SMask")
    )


class ExperimentalHybridBackend:
    parser_id = "local-hybrid"
    profile_id = "book-hybrid-local"
    profile_version = 1

    def __init__(self, configuration, output_root):
        self.configuration = copy.deepcopy(configuration)
        self.output_root = Path(output_root)
        self.models = {}
        self.assets = json.loads(Path(configuration["assets"]).read_text())
        self.asset_digest = digest(configuration["assets"])
        self.runtime = None

    def close(self):
        for model in self.models.values():
            model.close()
        self.models.clear()

    def _verify(self):
        if platform.system() != "Darwin":
            raise RuntimeError("This experimental profile requires macOS")
        if digest(self.configuration["assets"]) != self.asset_digest:
            raise ValueError("Asset manifest changed")
        for name, expected in self.configuration["binaries"].items():
            if digest(expected["path"]) != expected["sha256"]:
                raise ValueError(f"{name} executable changed")
        for model in self.assets["models"].values():
            for name, info in model["files"].items():
                if digest(Path(model["directory"]) / name) != info["sha256"]:
                    raise ValueError("Model asset changed")
        enforce_offline_runtime_policy()
        self.runtime = dict(
            resolve_runtime_identity(load_book_balanced_manifest().inference_engine)
        )

    def _read(self, kind, path):
        if kind not in self.models:
            from paddleocr import TextRecognition

            manifest = self.assets["models"][kind]
            name = "PP-OCRv5_server_rec" if kind == "server" else kind + "_PP-OCRv5_mobile_rec"
            self.models[kind] = TextRecognition(
                model_name=name,
                model_dir=manifest["directory"],
                device="cpu",
                cpu_threads=2,
                enable_mkldnn=False,
            )
        value = self.models[kind].predict(input=str(path), batch_size=1)[0]
        return {
            "text": str(value["rec_text"]),
            "score": float(value["rec_score"]),
            "model": kind,
        }

    def _command(self, name, arguments, timeout=60):
        command = [self.configuration["binaries"][name]["path"], *map(str, arguments)]
        result = subprocess.run(
            command, text=True, capture_output=True, timeout=timeout, check=False
        )
        if result.returncode:
            raise RuntimeError(f"{name} failed: {result.stderr[:500]}")
        if len(result.stdout) > 10_000_000:
            raise ValueError("Provider output exceeds diagnostic bound")
        return result.stdout

    def parse(self, request):
        if not isinstance(request.source, LocalPathSource):
            raise TypeError("An authorized local PDF is required")
        if (request.profile_id, request.profile_version) != (
            self.profile_id,
            self.profile_version,
        ):
            raise ValueError("Unexpected experimental profile")
        if (
            request.page_range is None
            or request.page_range.end_page_index - request.page_range.start_page_index != 1
        ):
            raise ValueError("Select exactly one page for this bounded profile")
        _validate_ocr_language_hints(request.language_hints)
        path = Path(request.source.path)
        if path.stat().st_size > 200_000_000:
            raise ValueError("Source exceeds bounded profile size")
        self.output_root.mkdir(exist_ok=False, parents=True)
        with tempfile.TemporaryDirectory(prefix="capicove-hybrid-source-") as temporary:
            snapshot = Path(temporary) / "source.pdf"
            _snapshot_authorized_source(path, snapshot, request.source_sha256)
            self._verify()
            try:
                result = self._parse_page(snapshot, request)
            finally:
                self._verify()
                if digest(path) != request.source_sha256:
                    raise ValueError("Authorized source changed during parsing")
        return result

    def _native_confirmation(self, path, index, output, width_pts, height_pts):
        """One fixed second rasterization, only to corroborate existing native text."""
        import pypdfium2 as pdfium
        from PIL import Image

        image_path = output / "native-confirmation-200.png"
        doc = pdfium.PdfDocument(path)
        page = doc[index]
        try:
            bitmap = page.render(scale=200 / 72)
            try:
                bitmap.to_pil().save(image_path, dpi=(200, 200))
            finally:
                bitmap.close()
        finally:
            page.close()
            doc.close()
        raw = self._command("vision", [image_path])
        (output / "native-confirmation-200.json").write_text(raw)
        observations = json.loads(raw)["observations"]
        if len(observations) > 1000:
            raise ValueError("Too many native confirmation regions")
        candidates = []
        with Image.open(image_path) as image:
            for i, observation in enumerate(observations):
                x, y, w, h = observation["box"]
                box = [
                    max(0, math.floor(x * image.width) - 2),
                    max(0, math.floor((1 - y - h) * image.height) - 2),
                    min(image.width, math.ceil((x + w) * image.width) + 2),
                    min(image.height, math.ceil((1 - y) * image.height) + 2),
                ]
                crop = output / f"native-confirmation-{i:04}.png"
                image.crop(box).save(crop)
                candidates.append(
                    {
                        "path": str(crop),
                        "normalized_box": [
                            box[0] / image.width,
                            box[1] / image.height,
                            box[2] / image.width,
                            box[3] / image.height,
                        ],
                    }
                )
        return candidates

    def _parse_page(self, path, request):
        import pypdfium2 as pdfium
        from PIL import Image
        from pypdf import PdfReader

        index = request.page_range.start_page_index
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported")
        if not 0 <= index < len(reader.pages):
            raise ValueError("Page index exceeds PDF bounds")
        source = self.output_root / "source"
        source.mkdir()
        work = self.output_root / "evidence"
        work.mkdir()
        crop_dir = work / "crops"
        crop_dir.mkdir()
        image_path = source / "native-scan.png"
        doc = pdfium.PdfDocument(path)
        page = doc[index]
        textpage = page.get_textpage()
        try:
            native = textpage.get_text_range()
            if _dominant_unsupported_script(native):
                raise ValueError("book-hybrid-local does not support the dominant PDF text script")
            width_pts, height_pts = page.get_size()
            chars = []
            for i in range(textpage.count_chars()):
                value = textpage.get_text_range(i, 1)
                if value.strip():
                    chars.append({"text": value, "box": list(textpage.get_charbox(i))})
            pypage = reader.pages[index]
            if can_extract_raster(pypage):
                image = pypage.images[0].image
                if image.width * image.height > 16_000_000:
                    raise ValueError("Raster exceeds pixel bound")
                image.save(
                    image_path,
                    dpi=(image.width * 72 / width_pts, image.height * 72 / height_pts),
                )
                route = "source-raster"
            else:
                if math.ceil(width_pts * 300 / 72) * math.ceil(height_pts * 300 / 72) > 16_000_000:
                    raise ValueError("Render exceeds pixel bound")
                bitmap = page.render(scale=300 / 72)
                try:
                    bitmap.to_pil().save(image_path, dpi=(300, 300))
                finally:
                    bitmap.close()
                route = "render-300dpi"
        finally:
            textpage.close()
            page.close()
            doc.close()
        vision_raw = self._command("vision", [image_path])
        (source / "vision.json").write_text(vision_raw)
        observations = json.loads(vision_raw)["observations"]
        if not observations or len(observations) > 1000:
            raise ValueError("Invalid primary region count")
        numeric_raw = self._command("numeric", [image_path])
        (source / "numeric.json").write_text(numeric_raw)
        numeric = json.loads(numeric_raw)
        if len(numeric) > 2000:
            raise ValueError("Too many numeric fields")
        tsv = self._command("tesseract", [image_path, "stdout", "-l", "eng", "tsv"])
        (source / "tesseract-native-geometry.tsv").write_text(tsv)
        items = []
        with Image.open(image_path) as image:
            width, height = image.size
            for kind, values in [("line", observations), ("numeric", numeric)]:
                for i, value in enumerate(values):
                    x, y, w, h = value["box"]
                    if not all(math.isfinite(v) for v in (x, y, w, h)) or min(w, h) <= 0:
                        raise ValueError("Invalid provider box")
                    box = [
                        max(0, math.floor(x * width) - 2),
                        max(0, math.floor((1 - y - h) * height) - 2),
                        min(width, math.ceil((x + w) * width) + 2),
                        min(height, math.ceil((1 - y) * height) + 2),
                    ]
                    if box[0] >= box[2] or box[1] >= box[3]:
                        raise ValueError("Empty crop")
                    crop = crop_dir / f"{len(items):04}.png"
                    image.crop(box).save(crop)
                    items.append(
                        {
                            "path": str(crop),
                            "sha256": digest(crop),
                            "kind": kind,
                            "observation": i,
                            "box": box,
                            "text": value.get("text", value.get("token")),
                            "line": value.get("line"),
                        }
                    )
        predictions = []
        line_items = [item for item in items if item["kind"] == "line"]
        native_groups = [[] for _ in line_items]
        for char in chars:
            left, bottom, right, top = char["box"]
            x = (left + right) / 2 * width / width_pts
            y = (1 - (bottom + top) / 2 / height_pts) * height
            matches = [i for i, item in enumerate(line_items) if contains(item["box"], x, y)]
            if len(matches) != 1:
                raise ValueError("Native character has unresolved region coverage")
            native_groups[matches[0]].append(char)
        for i, item in enumerate(items):
            row = {"index": i, "input_sha256": item["sha256"], "kind": item["kind"]}
            if item["kind"] == "line":
                model = model_for_region(item["text"], native, request.language_hints)
                row["paddle"] = self._read(model, item["path"])
                group = native_groups[item["observation"]]
                if group or has_han(item["text"]):
                    row["el"] = {
                        "text": "",
                        "not_applicable": "native or independently corroborated Han context",
                    }
                else:
                    row["el"] = self._read("el", item["path"])
            else:
                row["fraction"] = recognize(
                    item["path"], lambda crop: self._read("en", crop)["text"]
                )
            predictions.append(row)
        (work / "regions.json").write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n")
        (work / "predictions.json").write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2) + "\n"
        )
        (work / "inference-receipt.json").write_text(
            json.dumps(
                {
                    "output_sha256": digest(work / "predictions.json"),
                    "regions": digest(work / "regions.json"),
                }
            )
            + "\n"
        )
        result = assess(work, source)
        secondary_native = None
        for i, group in enumerate(native_groups):
            if not group:
                continue
            row = result["results"][i]
            text = "".join(c["text"] for c in sorted(group, key=lambda c: c["box"][0]))
            matches = [
                (provider, value)
                for provider, value in row["readings"].items()
                if provider in {"vision", "paddle"}
                and native_normalize(value) == native_normalize(text)
            ]
            if not matches:
                if secondary_native is None:
                    secondary_native = self._native_confirmation(
                        path, index, source, width_pts, height_pts
                    )
                positions = [
                    (
                        (c["box"][0] + c["box"][2]) / 2 / width_pts,
                        1 - (c["box"][1] + c["box"][3]) / 2 / height_pts,
                    )
                    for c in group
                ]
                candidates = [
                    candidate
                    for candidate in secondary_native
                    if all(contains(candidate["normalized_box"], x, y) for x, y in positions)
                ]
                for candidate in candidates:
                    reading = self._read(
                        model_for_region(text, native, request.language_hints),
                        candidate["path"],
                    )
                    if native_normalize(reading["text"]) == native_normalize(text):
                        matches.append(("paddle-native-confirmation", reading["text"]))
                        row["native_confirmation"] = {
                            "crop_sha256": digest(candidate["path"]),
                            "box": candidate["normalized_box"],
                            "dpi": 200,
                            "model": reading["model"],
                        }
                        break
                if not matches:
                    raise ValueError("Native and visible region text disagree")
            row["selected"] = list(matches[0])
            row["errors"] = []
            row["native_corroboration"] = text
        # Supplement every unmatched secondary word; no source-name exceptions.
        missing = {(tuple(w["box"]), w["text"]) for w in result["unassigned_words"]}
        groups = {}
        for row in csv.DictReader(tsv.splitlines(), delimiter="\t"):
            if row["level"] != "5" or not row["text"].strip():
                continue
            x, y, w, h = [int(row[k]) for k in ["left", "top", "width", "height"]]
            box = [x, y, x + w, y + h]
            if (tuple(box), row["text"]) not in missing:
                continue
            key = tuple(row[k] for k in ["block_num", "par_num", "line_num"])
            groups.setdefault(key, []).append({"text": row["text"], "box": box})
        supplements = []
        with Image.open(image_path) as image:
            for group in groups.values():
                group.sort(key=lambda w: w["box"][0])
                b = [
                    max(0, min(w["box"][0] for w in group) - 2),
                    max(0, min(w["box"][1] for w in group) - 2),
                    min(width, max(w["box"][2] for w in group) + 2),
                    min(height, max(w["box"][3] for w in group) + 2),
                ]
                crop = crop_dir / f"supplement-{len(supplements):04}.png"
                image.crop(b).save(crop)
                supplements.append(
                    {
                        "box": b,
                        "path": str(crop),
                        "sha256": digest(crop),
                        "source_words": group,
                        "text": " ".join(w["text"] for w in group),
                        "paddle": self._read("en", crop),
                        "el": self._read("el", crop),
                    }
                )
        result = merge_supplements(result, supplements)
        result.update(
            source_sha256=request.source_sha256,
            render_sha256=digest(image_path),
            render_route=route,
            source_page_index=index,
        )
        (work / "decision.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        if result["decision"] != "accept_candidate":
            raise ValueError("Unresolved scan region, script, or coverage evidence")
        blocks = []
        for order, row in enumerate(result["results"]):
            _provider, text = row["selected"]
            left, top, right, bottom = row["box"]
            blocks.append(
                DocumentBlock(
                    f"p{index:04d}:b{order:06d}",
                    index,
                    DocumentBlockType.LIST
                    if row.get("marker_pixels_verified")
                    else DocumentBlockType.TEXT,
                    text,
                    " ".join(text.split()),
                    NormalizedBox(left / width, top / height, right / width, bottom / height),
                    order,
                    0.0,
                )
            )
        page = DocumentPage(index, None, width, height, DocumentPageStatus.SUCCEEDED, tuple(blocks))
        configuration = {
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "runtime": self.runtime,
            "assets_sha256": self.asset_digest,
            "binaries": self.configuration["binaries"],
            "os": platform.mac_ver()[0],
            "render_policy": "source-raster-or-300dpi",
            "code": {p.name: digest(p) for p in ROOT.glob("*.py")},
        }
        configuration_sha = identity(configuration)
        receipt = DocumentParserReceipt(
            "capicove.processor.paddleocr",
            "0.1.0.dev0",
            self.profile_id,
            self.profile_version,
            self.asset_digest,
            configuration_sha,
            identity(
                {
                    "source": request.source_sha256,
                    "page": index,
                    "hints": request.language_hints,
                    "configuration": configuration_sha,
                }
            ),
        )
        return DocumentParseResult(
            request.source_sha256,
            identity({"page": asdict(page), "receipt": asdict(receipt)}),
            (page,),
            receipt,
            warnings=("Experimental local hybrid profile; not a general accuracy guarantee.",),
        )
