"""Regression checks for preserving numeric disagreements and non-bullet symbols."""

import unittest
from fractions import Fraction

from evidence import filled_bullet, fraction_value, normalize, select_reading
from PIL import Image, ImageDraw


class Tests(unittest.TestCase):
    def test_fraction_is_exact(self):
        self.assertEqual(fraction_value('3½"'), (Fraction(7, 2), "inch"))
        self.assertEqual(fraction_value('7¾"'), (Fraction(31, 4), "inch"))
        for text in ['312"', '31½/"', '3%"', "3½ USD"]:
            self.assertIsNone(fraction_value(text))

    def test_no_digit_normalization(self):
        self.assertNotEqual(normalize('3½"'), normalize('312"'))
        self.assertNotEqual(normalize("16.30"), normalize("16.80"))

    def test_ring_is_not_filled_bullet(self):
        im = Image.new("L", (30, 30), 255)
        d = ImageDraw.Draw(im)
        d.ellipse((3, 3, 26, 26), outline=0, width=3)
        self.assertFalse(filled_bullet(im))
        d.ellipse((3, 3, 26, 26), fill=0)
        self.assertTrue(filled_bullet(im))

    def test_choose_whole_corroborated_reading(self):
        self.assertEqual(
            select_reading({"vision": "let you", "paddle": "let you", "tesseract": "Jet you"}),
            ("vision", "let you"),
        )

    def test_unknown_disagreement_rejects(self):
        self.assertIsNone(select_reading({"vision": "100", "paddle": "900", "tesseract": "800"}))

    def test_equivalent_quotes_retain_selected_raw(self):
        self.assertEqual(
            select_reading({"vision": "It's fine", "tesseract": "It\u2019s fine"}),
            ("vision", "It's fine"),
        )


class SupplementalTests(unittest.TestCase):
    def test_missing_heading_is_inserted_before_same_column_body(self):
        from evidence import merge_supplements

        result = {
            "results": [
                {
                    "region": 0,
                    "box": [10, 50, 90, 65],
                    "selected": ["vision", "Body"],
                    "errors": [],
                }
            ],
            "unassigned_words": [{"text": "Heading", "box": [10, 30, 60, 45]}],
        }
        supplements = [
            {
                "box": [8, 28, 62, 47],
                "source_words": result["unassigned_words"],
                "text": "Heading",
                "paddle": {"text": "Heading"},
                "el": {"text": "Heading"},
            }
        ]
        merged = merge_supplements(result, supplements)
        self.assertEqual(merged["decision"], "accept_candidate")
        self.assertEqual([r["selected"][1] for r in merged["results"]], ["Heading", "Body"])

    def test_uncorroborated_heading_stays_blocked(self):
        from evidence import merge_supplements

        result = {
            "results": [
                {
                    "region": 0,
                    "box": [10, 50, 90, 65],
                    "selected": ["vision", "Body"],
                    "errors": [],
                }
            ],
            "unassigned_words": [{"text": "Heading", "box": [10, 30, 60, 45]}],
        }
        supplements = [
            {
                "box": [8, 28, 62, 47],
                "source_words": result["unassigned_words"],
                "text": "Heading",
                "paddle": {"text": "Something"},
                "el": {"text": "Heading"},
            }
        ]
        self.assertEqual(merge_supplements(result, supplements)["decision"], "reject")
