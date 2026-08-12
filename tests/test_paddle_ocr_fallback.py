from __future__ import annotations

import unittest

import numpy as np

from core.components.ocr.paddle_ocr import PaddleOcr
from core.types import Detection, Frame


class _FakeOcrEngine:
    def __init__(self, text: str = "高等数学"):
        self.text = text
        self.crop_shapes = []
        self.crops = []

    def __call__(self, crop):
        self.crop_shapes.append(crop.shape)
        self.crops.append(crop.copy())
        return [], [(self.text, 0.99)]


def _build_ocr(*, fallback: bool = True, full_frame: bool = False) -> PaddleOcr:
    ocr = object.__new__(PaddleOcr)
    ocr._engine = _FakeOcrEngine()
    ocr.candidate_classes = {"Book"}
    ocr.output_classes = ["W001", "W002", "W003", "W004"]
    ocr.templates = [
        "怪兽",
        "语文阅读写作诵读经典国语作文三字经论语诗词弟子规古诗文言文散文",
        "高等数学线性代数函数立体几何概率方程算术口算",
        "自然物理化学科学地理生物生态环境节气",
    ]
    ocr.enlarge = 1.0
    ocr.min_match_score = 60.0
    ocr.min_text_length = 2
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
    def test_two_character_keyword_is_accepted_at_threshold(self):
        ocr = _build_ocr()

        class_name, score = ocr._classify("语文")

        self.assertEqual("W002", class_name)
        self.assertEqual(60.0, score)

    def test_single_character_is_rejected(self):
        ocr = _build_ocr()

        self.assertEqual((None, 0.0), ocr._classify("语"))

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
