from __future__ import annotations

import unittest

import numpy as np

from core.components.ocr.paddle_ocr import PaddleOcr
from core.types import Detection, Frame


class _FakeOcrEngine:
    def __init__(self, text: str = "高等数学", responses=None):
        self.text = text
        self.responses = list(responses) if responses is not None else None
        self.crop_shapes = []
        self.crops = []

    def __call__(self, crop):
        self.crop_shapes.append(crop.shape)
        self.crops.append(crop.copy())
        if self.responses is None:
            response = self.text
        else:
            response = self.responses[len(self.crops) - 1]
        if response is None:
            return [], []
        if isinstance(response, str):
            return [], [(response, 0.99)]
        return [], response


def _build_ocr(*, fallback: bool = True, full_frame: bool = False) -> PaddleOcr:
    ocr = object.__new__(PaddleOcr)
    ocr._engine = _FakeOcrEngine()
    ocr.candidate_classes = {"Book"}
    ocr.output_classes = ["W001", "W002", "W003", "W004"]
    ocr.template_keywords = [
        ["怪兽"],
        ["语文", "阅读", "写作", "国语", "作文"],
        ["数学", "高等数学", "线性代数", "函数", "几何"],
        ["自然", "物理", "化学", "科学", "地理", "生物"],
    ]
    ocr.enlarge = 1.0
    ocr.min_match_score = 80.0
    ocr.min_match_margin = 15.0
    ocr.min_text_length = 2
    ocr.retry_rotations = (90, 270)
    ocr.fallback_when_no_candidate = fallback
    ocr.fallback_candidate_class = "Table"
    ocr.full_frame_if_no_table = full_frame
    ocr.fallback_mask_known_classes = True
    ocr.fallback_mask_classes = {
        "Brush",
        "Earphone",
        "Cup",
        "ClothesHanger",
        "Chocolate",
        "MelonSeeds",
        "Sausage",
        "Chips",
        "Can",
        "Bottle",
        "Milk",
        "Water",
        "Peach",
        "Apple",
        "Banana",
        "Pear",
    }
    return ocr


def _frame() -> Frame:
    return Frame(
        frame_id="test",
        rgb=np.zeros((100, 200, 3), dtype=np.uint8),
        depth=None,
        timestamp=0.0,
    )


class PaddleOcrFallbackTest(unittest.TestCase):
    def test_exact_two_character_keyword_is_accepted(self):
        ocr = _build_ocr()

        class_name, score = ocr._classify("语文")

        self.assertEqual("W002", class_name)
        self.assertEqual(100.0, score)

    def test_single_character_is_rejected(self):
        ocr = _build_ocr()

        self.assertEqual((None, 0.0), ocr._classify("语"))

    def test_ambiguous_fuzzy_match_is_rejected(self):
        ocr = _build_ocr()
        ocr.output_classes = ["W002", "W005"]
        ocr.template_keywords = [["语文"], ["英语"]]
        ocr.min_match_score = 40.0

        self.assertEqual((None, 0.0), ocr._classify("语学"))

    def test_keyword_inside_noisy_text_is_accepted(self):
        ocr = _build_ocr()

        self.assertEqual(("W003", 100.0), ocr._classify("2数学5"))

    def test_failed_original_orientation_retries_90_and_270_degrees(self):
        ocr = _build_ocr()
        ocr._engine = _FakeOcrEngine(responses=["学数", None, "数学"])
        detections = [
            Detection(class_name="Book", bbox=(20, 10, 80, 50), score=0.8),
        ]

        results = ocr.process(_frame(), detections, table=1)

        self.assertEqual(1, len(results))
        self.assertEqual("W003", results[0].class_name)
        self.assertEqual("数学", results[0].evidence["text"])
        self.assertEqual(270, results[0].evidence["rotation"])
        self.assertEqual([(40, 60, 3), (60, 40, 3), (60, 40, 3)], ocr._engine.crop_shapes)

    def test_single_character_is_not_guessed_after_rotation_retries(self):
        ocr = _build_ocr()
        ocr._engine = _FakeOcrEngine(responses=["语", "语", "语"])
        detections = [
            Detection(class_name="Book", bbox=(20, 10, 80, 50), score=0.8),
        ]

        self.assertEqual([], ocr.process(_frame(), detections, table=1))
        self.assertEqual(3, len(ocr._engine.crops))

    def test_book_crop_takes_priority_over_table_fallback(self):
        ocr = _build_ocr()
        detections = [
            Detection(class_name="Table", bbox=(0, 0, 200, 100), score=1.0),
            Detection(class_name="Book", bbox=(20, 10, 80, 50), score=0.8),
        ]

        results = ocr.process(_frame(), detections, table=1)

        self.assertEqual(1, len(results))
        self.assertEqual((20, 10, 80, 50), results[0].bbox)
        self.assertEqual("book_crop", results[0].evidence["ocr_mode"])
        self.assertEqual("Book", results[0].evidence["source_detection_class"])
        self.assertEqual([(40, 60, 3)], ocr._engine.crop_shapes)

    def test_table_is_used_when_book_is_missing(self):
        ocr = _build_ocr()
        detections = [
            Detection(class_name="Table", bbox=(10, 20, 190, 90), score=1.0),
            Detection(class_name="Bottle", bbox=(30, 30, 50, 70), score=0.9),
            Detection(class_name="Brush", bbox=(191, 20, 199, 40), score=0.9),
        ]

        results = ocr.process(_frame(), detections, table=2)

        self.assertEqual(1, len(results))
        self.assertEqual((10, 20, 190, 90), results[0].bbox)
        self.assertEqual("table_fallback", results[0].evidence["ocr_mode"])
        self.assertEqual("Table", results[0].evidence["source_detection_class"])
        self.assertEqual([(70, 180, 3)], ocr._engine.crop_shapes)
        self.assertEqual(1, results[0].evidence["masked_detection_count"])
        self.assertEqual(
            ["Bottle"],
            results[0].evidence["masked_detection_classes"],
        )
        crop = ocr._engine.crops[0]
        self.assertTrue(np.all(crop[10:50, 20:40] == 127))
        self.assertTrue(np.all(crop[0:10, 0:20] == 0))

    def test_known_classes_are_not_masked_for_book_crop(self):
        ocr = _build_ocr()
        frame = Frame(
            frame_id="test",
            rgb=np.full((100, 200, 3), 42, dtype=np.uint8),
            depth=None,
            timestamp=0.0,
        )
        detections = [
            Detection(class_name="Book", bbox=(20, 10, 80, 50), score=0.8),
            Detection(class_name="Bottle", bbox=(30, 20, 50, 40), score=0.9),
        ]

        results = ocr.process(frame, detections, table=1)

        self.assertEqual(1, len(results))
        self.assertEqual(0, results[0].evidence["masked_detection_count"])
        self.assertTrue(np.all(ocr._engine.crops[0] == 42))

    def test_no_target_returns_no_result_when_full_frame_is_disabled(self):
        ocr = _build_ocr()

        results = ocr.process(_frame(), [], table=1)

        self.assertEqual([], results)
        self.assertEqual([], ocr._engine.crop_shapes)

    def test_full_frame_fallback_is_available_but_opt_in(self):
        ocr = _build_ocr(full_frame=True)

        results = ocr.process(_frame(), [], table=1)

        self.assertEqual(1, len(results))
        self.assertEqual((0, 0, 200, 100), results[0].bbox)
        self.assertEqual("full_frame_fallback", results[0].evidence["ocr_mode"])
        self.assertEqual([(100, 200, 3)], ocr._engine.crop_shapes)


if __name__ == "__main__":
    unittest.main()
