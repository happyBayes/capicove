"""Read only spatially proven mixed fractions with an explicit digit provider."""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage


def recognize(path, digit_reader=None):
    with Image.open(path) as image:
        gray = image.convert("L")
    if gray.width * gray.height > 1_000_000:
        return {"status": "abstain", "reason": "image_too_large"}
    pixels = np.asarray(gray) < 128
    labels, _ = ndimage.label(pixels, structure=np.ones((3, 3), dtype=int))
    parts = []
    for index, slices in enumerate(ndimage.find_objects(labels), 1):
        y, x = slices
        mask = labels[slices] == index
        if int(mask.sum()) >= 3:
            parts.append({"box": (x.start, y.start, x.stop, y.stop), "mask": mask})
    if not parts:
        return {"status": "abstain", "reason": "no_ink"}
    tallest = max(p["box"][3] - p["box"][1] for p in parts)
    slashes = []
    for p in parts:
        b = p["box"]
        sy, sx = np.nonzero(p["mask"])
        if b[3] - b[1] < 0.65 * tallest or len(sx) < 5 or np.std(sx) == 0 or np.std(sy) == 0:
            continue
        if float(np.corrcoef(sx, sy)[0, 1]) < -0.9:
            slashes.append(p)
    if len(slashes) != 1:
        return {"status": "abstain", "reason": "slash_not_unique"}
    slash = slashes[0]
    sb = slash["box"]
    mid_y = (sb[1] + sb[3]) / 2
    mid_x = (sb[0] + sb[2]) / 2
    anchors = [
        p
        for p in parts
        if p is not slash and p["box"][2] <= sb[0] and p["box"][3] - p["box"][1] > 0.65 * tallest
    ]
    if len(anchors) != 1:
        return {"status": "abstain", "reason": "whole_number_not_unique"}
    whole = anchors[0]
    wb = whole["box"]
    groups = {"whole": [], "numerator": [], "denominator": [], "unit": []}
    for p in parts:
        if p is slash:
            continue
        b = p["box"]
        cx = (b[0] + b[2]) / 2
        cy = (b[1] + b[3]) / 2
        if min(b[2], wb[2]) > max(b[0], wb[0]):
            groups["whole"].append(p)
        elif cx < mid_x and cy < mid_y:
            groups["numerator"].append(p)
        elif cx > mid_x and cy > mid_y:
            groups["denominator"].append(p)
        elif cx > mid_x and cy < mid_y:
            groups["unit"].append(p)
        else:
            return {"status": "abstain", "reason": "unassigned_ink"}
    if any(not groups[name] for name in ["whole", "numerator", "denominator"]):
        return {"status": "abstain", "reason": "missing_digit_slot"}

    def box_of(group):
        return (
            min(p["box"][0] for p in group),
            min(p["box"][1] for p in group),
            max(p["box"][2] for p in group),
            max(p["box"][3] for p in group),
        )

    boxes = {name: box_of(group) for name, group in groups.items() if group}
    for name in ["numerator", "denominator"]:
        b = boxes[name]
        if b[3] - b[1] >= 0.8 * (wb[3] - wb[1]):
            return {"status": "abstain", "reason": "digit_scale"}
    unit = groups["unit"]
    if unit:
        if len(unit) != 2:
            return {"status": "abstain", "reason": "unit_mark_count"}
        for p in unit:
            b = p["box"]
            w = b[2] - b[0]
            h = b[3] - b[1]
            if b[0] <= boxes["denominator"][2] or h > 0.55 * tallest or h < 1.3 * w:
                return {"status": "abstain", "reason": "unit_mark_shape"}
    if digit_reader is None:
        raise ValueError("An explicitly selected digit reader is required")
    values = []
    with tempfile.TemporaryDirectory(prefix="capicove-fraction-") as tmp:
        for name in ["whole", "numerator", "denominator"]:
            b = boxes[name]
            ink = np.zeros((b[3] - b[1], b[2] - b[0]), dtype=bool)
            for p in groups[name]:
                x0, y0, x1, y1 = p["box"]
                ink[y0 - b[1] : y1 - b[1], x0 - b[0] : x1 - b[0]] |= p["mask"]
            digit = Image.fromarray(np.where(ink, 0, 255).astype("uint8"))
            digit = digit.resize((digit.width * 4, digit.height * 4), Image.Resampling.NEAREST)
            digit = ImageOps.expand(digit, border=20, fill=255)
            target = Path(tmp) / (name + ".png")
            digit.save(target)
            text = digit_reader(target)
            if len(text) != 1 or text not in "0123456789":
                return {
                    "status": "abstain",
                    "reason": "digit_unrecognized",
                    "slot": name,
                }
            values.append(int(text))
    w, n, d = values
    if not 0 < n < d:
        return {"status": "abstain", "reason": "not_proper_fraction"}
    return {
        "status": "candidate",
        "whole": w,
        "numerator": n,
        "denominator": d,
        "unit": "inch" if unit else None,
        "canonical_success": False,
    }
