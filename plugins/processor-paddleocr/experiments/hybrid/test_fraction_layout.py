"""Negative checks for conservative shape rejection."""

import tempfile
import unittest
from pathlib import Path

from fraction_layout import recognize
from PIL import Image, ImageDraw


class Tests(unittest.TestCase):
    def test_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "input.png"
            Image.new("L", (100, 50), 255).save(p)
            self.assertEqual(recognize(p)["status"], "abstain")

    def test_horizontal_shapes_are_not_fraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "input.png"
            im = Image.new("L", (100, 50), 255)
            draw = ImageDraw.Draw(im)
            for x in (10, 30, 50, 70):
                draw.rectangle((x, 10, x + 5, 30), fill=0)
            im.save(p)
            self.assertEqual(recognize(p)["status"], "abstain")
