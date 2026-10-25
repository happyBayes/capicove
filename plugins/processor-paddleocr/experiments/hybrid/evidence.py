"""Experimental spatial corroboration; preserve raw complete provider readings."""

import csv
import hashlib
import json
import re
import unicodedata
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


ROOT = Path(__file__).resolve().parent


def normalize(text):
    table = str.maketrans({"\u2018": "'", "\u2019": "'", "“": '"', "”": '"', "″": '"'})
    return "".join(unicodedata.normalize("NFC", text.translate(table)).split())


def fraction_value(text):
    text = normalize(text)
    match = re.fullmatch(r'([0-9]+)([¼-¾⅐-⅞])"', text)
    if not match:
        return None
    decomposition = unicodedata.decomposition(match[2]).split()
    if not decomposition or decomposition[0] != "<fraction>":
        return None
    expanded = "".join(chr(int(code, 16)) for code in decomposition[1:])
    if "\u2044" not in expanded:
        return None
    numerator, denominator = expanded.split("\u2044")
    return Fraction(int(match[1])) + Fraction(int(numerator), int(denominator)), "inch"


def filled_bullet(image):
    ink = np.asarray(image.convert("L")) < 128
    y, x = np.nonzero(ink)
    if len(x) < 8:
        return False
    ink = ink[y.min() : y.max() + 1, x.min() : x.max() + 1]
    height, width = ink.shape
    if min(height, width) < 3 or not 0.75 <= width / height <= 1.33:
        return False
    _, count = ndimage.label(ink, structure=np.ones((3, 3), dtype=int))
    center = ink[
        height // 3 : max(height // 3 + 1, 2 * height // 3),
        width // 3 : max(width // 3 + 1, 2 * width // 3),
    ]
    return count == 1 and 0.60 <= float(ink.mean()) <= 0.92 and float(center.mean()) >= 0.90


def select_reading(readings, comparable=None):
    compared = readings if comparable is None else comparable
    for name, text in readings.items():
        if not text.strip():
            continue
        if any(
            other != name and normalize(compared[name]) == normalize(compared[other])
            for other in readings
        ):
            return name, text
    return None


def contains(box, x, y):
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assess(root, source):
    receipt = json.loads((root / "inference-receipt.json").read_text())
    if (
        digest(root / "predictions.json") != receipt["output_sha256"]
        or digest(root / "regions.json") != receipt["regions"]
    ):
        raise ValueError("Prediction identity changed")
    items = json.loads((root / "regions.json").read_text())
    predictions = json.loads((root / "predictions.json").read_text())
    if len(items) != len(predictions):
        raise ValueError("Missing predictions")
    for i, (item, prediction) in enumerate(zip(items, predictions, strict=True)):
        if (
            prediction["index"] != i
            or prediction["input_sha256"] != item["sha256"]
            or digest(Path(item["path"])) != item["sha256"]
        ):
            raise ValueError("Crop or row changed")
    lines = [
        (item, pred)
        for item, pred in zip(items, predictions, strict=True)
        if item["kind"] == "line"
    ]
    fields = [
        (item, pred)
        for item, pred in zip(items, predictions, strict=True)
        if item["kind"] == "numeric"
    ]
    with (source / "tesseract-native-geometry.tsv").open() as stream:
        words = []
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["level"] != "5" or not row["text"].strip():
                continue
            x, y, w, h = [int(row[k]) for k in ["left", "top", "width", "height"]]
            if min(w, h) <= 0:
                raise ValueError("Invalid word geometry")
            words.append({"text": row["text"], "box": [x, y, x + w, y + h]})
    assigned = [[] for _ in lines]
    unassigned = []
    for word in words:
        b = word["box"]
        matches = [
            i
            for i, (item, _) in enumerate(lines)
            if contains(item["box"], (b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        ]
        if len(matches) != 1:
            unassigned.append(word)
        else:
            assigned[matches[0]].append(word)
    results = []
    with Image.open(source / "native-scan.png") as image:
        for i, ((item, pred), group) in enumerate(zip(lines, assigned, strict=True)):
            group = sorted(group, key=lambda w: w["box"][0])
            original = {
                "vision": item["text"],
                "paddle": pred["paddle"]["text"],
                "tesseract": " ".join(w["text"] for w in group),
            }
            compared = original.copy()
            fraction_proofs = []
            errors = []
            greek = sum("GREEK" in unicodedata.name(c, "") for c in pred["el"]["text"])
            if greek:
                errors.append("unsupported_script_evidence")
            marker = bool(
                group
                and filled_bullet(image.crop(group[0]["box"]))
                and item["text"].startswith("•")
            )
            if marker:
                for name, text in compared.items():
                    if text and text[0] in "•·°©¢*":
                        compared[name] = text[1:].lstrip()
            for field, field_pred in fields:
                fraction = field_pred.get("fraction", {})
                if fraction.get("status") != "candidate" or field["line"] != item["text"]:
                    continue
                parsed = fraction_value(field["text"])
                expected = Fraction(fraction["whole"]) + Fraction(
                    fraction["numerator"], fraction["denominator"]
                )
                if parsed != (expected, fraction["unit"]):
                    errors.append("fraction_value_conflict")
                    continue
                b = field["box"]
                aligned = [
                    (j, w)
                    for j, w in enumerate(group)
                    if contains(
                        b,
                        (w["box"][0] + w["box"][2]) / 2,
                        (w["box"][1] + w["box"][3]) / 2,
                    )
                ]
                if len(aligned) != 1 or compared["vision"].count(field["text"]) != 1:
                    errors.append("fraction_alignment_ambiguous")
                    continue
                _, word = aligned[0]
                if compared["tesseract"].count(word["text"]) != 1:
                    errors.append("fraction_word_ambiguous")
                    continue
                # Only compare the surrounding text after verifying this exact field.
                compared["vision"] = compared["vision"].replace(
                    field["text"], "<VERIFIED_FIELD>", 1
                )
                compared["tesseract"] = compared["tesseract"].replace(
                    word["text"], "<VERIFIED_FIELD>", 1
                )
                fraction_proofs.append(
                    {
                        "crop_sha256": field["sha256"],
                        "box": b,
                        "value": str(expected),
                        "unit": fraction["unit"],
                        "vision_token": field["text"],
                        "secondary_token": word["text"],
                    }
                )
            selected = select_reading(original, compared)
            if selected is None:
                errors.append("uncorroborated_region")
            results.append(
                {
                    "region": i,
                    "box": item["box"],
                    "readings": original,
                    "compared": compared,
                    "marker_pixels_verified": marker,
                    "fraction_proofs": fraction_proofs,
                    "selected": selected,
                    "errors": errors,
                }
            )
    summary = {
        "canonical_success": False,
        "decision": "accept_candidate"
        if not unassigned and all(not r["errors"] for r in results)
        else "reject",
        "regions": len(results),
        "resolved": sum(not r["errors"] for r in results),
        "unassigned_words": unassigned,
        "results": results,
        "worker_sha256": digest(Path(__file__)),
    }
    (root / "scan-result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(
        "regions",
        summary["regions"],
        "resolved",
        summary["resolved"],
        "unassigned_words",
        len(unassigned),
        "decision",
        summary["decision"],
    )
    return summary


def merge_supplements(result, supplements):
    """Cover omitted words only with a second reader and unambiguous placement."""
    import copy

    merged = copy.deepcopy(result)
    rows = merged["results"]
    missing = {(tuple(w["box"]), w["text"]) for w in merged["unassigned_words"]}
    errors = []
    if len(missing) != len(merged["unassigned_words"]):
        errors.append("duplicate_unassigned_word")
    for supplement in supplements:
        keys = [(tuple(w["box"]), w["text"]) for w in supplement["source_words"]]
        if not keys or len(keys) != len(set(keys)) or any(k not in missing for k in keys):
            errors.append("supplement_source_mismatch")
            continue
        expected = " ".join(
            w["text"] for w in sorted(supplement["source_words"], key=lambda w: w["box"][0])
        )
        if supplement["text"] != expected or normalize(expected) != normalize(
            supplement["paddle"]["text"]
        ):
            errors.append("uncorroborated_supplement")
            continue
        if any("GREEK" in unicodedata.name(c, "") for c in supplement["el"]["text"]):
            errors.append("unsupported_supplement")
            continue
        b = supplement["box"]
        following = []
        for i, row in enumerate(rows):
            other = row["box"]
            overlap = min(b[2], other[2]) - max(b[0], other[0])
            if overlap >= 0.5 * min(b[2] - b[0], other[2] - other[0]) and other[1] >= b[3]:
                following.append((other[1] - b[3], i))
        if not following:
            errors.append("supplement_order_unresolved")
            continue
        following.sort()
        if len(following) > 1 and following[0][0] == following[1][0]:
            errors.append("supplement_order_ambiguous")
            continue
        rows.insert(
            following[0][1],
            {
                "region": "supplemental",
                "box": b,
                "selected": ["tesseract", expected],
                "readings": {
                    "tesseract": expected,
                    "paddle": supplement["paddle"]["text"],
                },
                "errors": [],
                "source_words": supplement["source_words"],
            },
        )
        missing.difference_update(keys)
    merged["unassigned_words"] = [{"box": list(box), "text": text} for box, text in sorted(missing)]
    merged["supplement_errors"] = errors
    merged["regions"] = len(rows)
    merged["resolved"] = sum(not row["errors"] for row in rows)
    merged["decision"] = (
        "accept_candidate"
        if not missing and not errors and all(not row["errors"] for row in rows)
        else "reject"
    )
    merged["canonical_success"] = False
    return merged
