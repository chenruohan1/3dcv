"""PaddleOCR 文字识别：选择识别区域 → OCR → 模板模糊匹配为书本类别。

对应旧项目（3dcv_2025）的 OCRBookCropDetector，适配到当前框架的 BaseOcr 接口：
优先对上游检出的候选框（默认为 ``Book``）逐个裁剪；没有候选框时可回退到
``Table`` 框。裁剪区域送入 PaddleOCR 引擎后拼接识别文本，再用 rapidfuzz 与
配置的模板串做模糊匹配，命中则产出对应的书本物品名称检测项。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
from rapidfuzz import process

from core.components.ocr.base import BaseOcr
from core.components.ocr.paddleocr import ONNXPaddleOcr
from core.components.ocr.paddleocr.model_resolver import resolve_engine_config
from core.types import Detection, Frame


class PaddleOcr(BaseOcr):
    """基于 PaddleOCR ONNX 引擎的书本文字识别与分类。"""

    def __init__(self, config: Dict, class_registry: Optional[Dict] = None):
        if class_registry is None:
            raise ValueError("PaddleOcr requires shared class_registry config")

        _backend, engine_config = resolve_engine_config(config.get("engine", {}))
        # 引擎强制关闭 GPU、开启方向分类，与旧项目保持一致。
        self._engine = ONNXPaddleOcr(
            use_angle_cls=bool(config.get("use_angle_cls", True)),
            use_gpu=bool(config.get("use_gpu", False)),
            **engine_config,
        )

        # 只对这些类别的检测框做 OCR（默认书本）。
        self.candidate_classes = set(class_registry.get("ocr_candidate_classes", ["Book"]))
        self.output_classes = list(class_registry.get("ocr_output_classes", []))
        if not self.output_classes:
            raise ValueError("class_registry.ocr_output_classes must not be empty for OCR")

        # 每个输出类别对应一段模板文本，用于模糊匹配。
        ocr_templates = dict(class_registry.get("ocr_templates", {}))
        missing_templates = [
            class_name
            for class_name in self.output_classes
            if class_name not in ocr_templates
        ]
        if missing_templates:
            raise ValueError(
                "class_registry.ocr_templates must define every OCR output class: "
                + ", ".join(missing_templates)
            )
        self.templates = [str(ocr_templates[class_name]) for class_name in self.output_classes]

        self.enlarge = float(config.get("enlarge", 1.0))
        self.min_match_score = float(config.get("min_match_score", 0.0))
        self.min_text_length = int(config.get("min_text_length", 2))
        if self.min_text_length < 1:
            raise ValueError("ocr.min_text_length must be at least 1")
        self.fallback_when_no_candidate = bool(
            config.get("fallback_when_no_candidate", False)
        )
        self.fallback_candidate_class = str(
            config.get("fallback_candidate_class", "Table")
        ).strip()
        self.full_frame_if_no_table = bool(
            config.get("full_frame_if_no_table", False)
        )
        self.fallback_mask_known_classes = bool(
            config.get("fallback_mask_known_classes", True)
        )
        if self.fallback_when_no_candidate and not self.fallback_candidate_class:
            raise ValueError("ocr.fallback_candidate_class must not be empty")

        detector_classes = {
            str(class_name)
            for class_name in class_registry.get("detector_id_to_class", {}).values()
        }
        self.fallback_mask_classes = detector_classes.difference(
            self.candidate_classes,
            {self.fallback_candidate_class},
        )

    def process(self, frame: Frame, detections: List[Detection], table: int) -> List[Detection]:
        """优先识别候选框，无候选时按配置回退到桌面区域。"""
        if frame.rgb is None:
            return []

        results: List[Detection] = []
        for bbox, ocr_mode, source_class in self._select_ocr_targets(
            frame,
            detections,
        ):
            masked_detections = self._fallback_mask_detections(
                detections,
                ocr_mode,
                bbox,
            )
            text = self._read_text(
                frame.rgb,
                bbox,
                mask_bboxes=[detection.bbox for detection in masked_detections],
            )
            if not text:
                continue

            class_name, score = self._classify(text)
            if class_name is None:
                continue

            results.append(
                Detection(
                    class_name=class_name,
                    bbox=bbox,
                    score=1.0,
                    evidence={
                        "source": "ocr",
                        "ocr_mode": ocr_mode,
                        "source_detection_class": source_class,
                        "masked_detection_count": len(masked_detections),
                        "masked_detection_classes": sorted(
                            {detection.class_name for detection in masked_detections}
                        ),
                        "table": table,
                        "text": text,
                        "match_score": score,
                    },
                )
            )
        return results

    def _select_ocr_targets(
        self,
        frame: Frame,
        detections: List[Detection],
    ) -> List[Tuple[tuple[int, int, int, int], str, str]]:
        """选择 OCR 区域：Book 等候选框优先，否则回退到最佳 Table 框。"""
        candidates = [
            detection
            for detection in detections
            if detection.class_name in self.candidate_classes
        ]
        if candidates:
            return [
                (detection.bbox, "book_crop", detection.class_name)
                for detection in candidates
            ]

        if not self.fallback_when_no_candidate:
            return []

        fallback_candidates = [
            detection
            for detection in detections
            if detection.class_name == self.fallback_candidate_class
        ]
        if fallback_candidates:
            detection = max(
                fallback_candidates,
                key=lambda item: (
                    float(item.score),
                    max(0, item.bbox[2] - item.bbox[0])
                    * max(0, item.bbox[3] - item.bbox[1]),
                ),
            )
            return [
                (
                    detection.bbox,
                    "table_fallback",
                    detection.class_name,
                )
            ]

        if self.full_frame_if_no_table:
            height, width = frame.rgb.shape[:2]
            return [
                (
                    (0, 0, int(width), int(height)),
                    "full_frame_fallback",
                    "",
                )
            ]
        return []

    def _fallback_mask_detections(
        self,
        detections: List[Detection],
        ocr_mode: str,
        target_bbox: tuple[int, int, int, int],
    ) -> List[Detection]:
        """兜底 OCR 时遮蔽已识别的普通物品，避免包装文字误命中书本类别。"""
        if not self.fallback_mask_known_classes:
            return []
        if ocr_mode not in {"table_fallback", "full_frame_fallback"}:
            return []
        target_x1, target_y1, target_x2, target_y2 = target_bbox
        masked = []
        for detection in detections:
            if detection.class_name not in self.fallback_mask_classes:
                continue
            x1, y1, x2, y2 = detection.bbox
            if x2 <= target_x1 or x1 >= target_x2:
                continue
            if y2 <= target_y1 or y1 >= target_y2:
                continue
            masked.append(detection)
        return masked

    def _read_text(self, rgb, bbox, mask_bboxes=()) -> str:
        """裁剪 bbox 区域跑 OCR，把该区域内识别出的所有文本拼接成一个串。"""
        x1, y1, x2, y2 = (int(v) for v in bbox)
        height, width = rgb.shape[:2]
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))
        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return ""
        if mask_bboxes:
            crop = crop.copy()
            for mask_bbox in mask_bboxes:
                mask_x1, mask_y1, mask_x2, mask_y2 = (
                    int(value) for value in mask_bbox
                )
                mask_x1 = max(x1, min(mask_x1, x2))
                mask_x2 = max(x1, min(mask_x2, x2))
                mask_y1 = max(y1, min(mask_y1, y2))
                mask_y2 = max(y1, min(mask_y2, y2))
                if mask_x2 <= mask_x1 or mask_y2 <= mask_y1:
                    continue
                crop[
                    mask_y1 - y1 : mask_y2 - y1,
                    mask_x1 - x1 : mask_x2 - x1,
                ] = 127
        if self.enlarge > 1.0:
            crop = cv2.resize(
                crop,
                dsize=None,
                fx=self.enlarge,
                fy=self.enlarge,
                interpolation=cv2.INTER_LINEAR,
            )

        _boxes, rec_res = self._engine(crop)
        if not rec_res:
            return ""
        return "".join(text for text, _score in rec_res)

    def _classify(self, text: str):
        """用 rapidfuzz 把识别文本模糊匹配到模板，返回 (类别名, 相似度) 或 (None, 0)。"""
        text = str(text).strip()
        if len(text) < self.min_text_length:
            return None, 0.0
        _matched, score, index = process.extractOne(text, self.templates)
        if score >= self.min_match_score:
            return self.output_classes[index], float(score)
        return None, 0.0

    def close(self) -> None:
        """释放 PaddleOCR det/rec/cls 后端资源。"""
        close = getattr(self._engine, "close", None)
        if close is not None:
            close()
