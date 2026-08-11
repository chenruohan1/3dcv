"""PaddleOCR 文字识别：选择识别区域 → OCR → 模板模糊匹配为书本类别。

对应旧项目（3dcv_2025）的 OCRBookCropDetector，适配到当前框架的 BaseOcr 接口：
优先对上游检出的候选框（默认为 ``Book``）逐个裁剪；没有候选框时可回退到
``Table`` 框。裁剪区域送入 PaddleOCR 引擎后拼接识别文本，再与配置的类别关键词
逐项匹配；原方向分类失败时尝试 90/270 度旋转，命中则产出对应的书本物品名称检测项。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
from rapidfuzz import fuzz

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
        self.template_keywords = [
            self._parse_template_keywords(ocr_templates[class_name], class_name)
            for class_name in self.output_classes
        ]

        self.enlarge = float(config.get("enlarge", 1.0))
        self.min_match_score = float(config.get("min_match_score", 0.0))
        self.min_match_margin = float(config.get("min_match_margin", 0.0))
        self.min_text_length = int(config.get("min_text_length", 2))
        if self.min_text_length < 1:
            raise ValueError("ocr.min_text_length must be at least 1")
        retry_rotations = config.get("retry_rotations", [90, 270])
        self.retry_rotations = tuple(dict.fromkeys(int(value) for value in retry_rotations))
        invalid_rotations = [
            value for value in self.retry_rotations if value not in {90, 180, 270}
        ]
        if invalid_rotations:
            raise ValueError(
                "ocr.retry_rotations only supports 90, 180 and 270 degrees"
            )
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
            text, class_name, score, rotation, ocr_confidence = (
                self._read_and_classify(
                    frame.rgb,
                    bbox,
                    mask_bboxes=[
                        detection.bbox for detection in masked_detections
                    ],
                )
            )
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
                        "rotation": rotation,
                        "ocr_confidence": ocr_confidence,
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

    @staticmethod
    def _parse_template_keywords(value, class_name: str) -> List[str]:
        """把类别模板统一为非空关键词列表，同时兼容旧版字符串配置。"""
        if isinstance(value, str):
            keywords = [value.strip()]
        elif isinstance(value, (list, tuple)):
            keywords = [str(item).strip() for item in value]
        else:
            raise ValueError(
                f"class_registry.ocr_templates.{class_name} must be a string or list"
            )
        keywords = [keyword for keyword in keywords if keyword]
        if not keywords:
            raise ValueError(
                f"class_registry.ocr_templates.{class_name} must not be empty"
            )
        return keywords

    def _prepare_crop(self, rgb, bbox, mask_bboxes=()):
        """裁剪并遮蔽 OCR 区域，供原方向和旋转重试复用。"""
        x1, y1, x2, y2 = (int(v) for v in bbox)
        height, width = rgb.shape[:2]
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))
        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return None
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
        return crop

    @staticmethod
    def _rotate_crop(crop, rotation: int):
        if rotation == 0:
            return crop
        if rotation == 90:
            return cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        if rotation == 180:
            return cv2.rotate(crop, cv2.ROTATE_180)
        if rotation == 270:
            return cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        raise ValueError(f"unsupported OCR rotation: {rotation}")

    def _recognize_crop(self, crop, rotation: int = 0) -> Tuple[str, float]:
        """识别一个已准备好的裁剪图，返回拼接文本和按字符加权置信度。"""
        rotated = self._rotate_crop(crop, rotation)
        _boxes, rec_res = self._engine(rotated)
        if not rec_res:
            return "", 0.0

        parts = [
            (str(text).strip(), float(score))
            for text, score in rec_res
            if str(text).strip()
        ]
        if not parts:
            return "", 0.0
        total_length = sum(len(text) for text, _score in parts)
        confidence = sum(len(text) * score for text, score in parts) / total_length
        return "".join(text for text, _score in parts), float(confidence)

    def _read_text(self, rgb, bbox, mask_bboxes=()) -> str:
        """按原方向识别 bbox，保留供审计工具使用的兼容接口。"""
        crop = self._prepare_crop(rgb, bbox, mask_bboxes)
        if crop is None:
            return ""
        text, _confidence = self._recognize_crop(crop)
        return text

    def _read_and_classify(self, rgb, bbox, mask_bboxes=()):
        """原方向分类失败时尝试配置的旋转角度，并选择最可靠结果。"""
        crop = self._prepare_crop(rgb, bbox, mask_bboxes)
        if crop is None:
            return "", None, 0.0, 0, 0.0

        text, confidence = self._recognize_crop(crop, 0)
        class_name, match_score = self._classify(text)
        if class_name is not None:
            return text, class_name, match_score, 0, confidence

        best_raw = (text, confidence, 0)
        accepted = []
        for rotation in self.retry_rotations:
            rotated_text, rotated_confidence = self._recognize_crop(crop, rotation)
            if (rotated_confidence, len(rotated_text)) > (
                best_raw[1],
                len(best_raw[0]),
            ):
                best_raw = (rotated_text, rotated_confidence, rotation)

            rotated_class, rotated_score = self._classify(rotated_text)
            if rotated_class is not None:
                accepted.append(
                    (
                        rotated_score,
                        rotated_confidence,
                        len(rotated_text),
                        rotated_text,
                        rotated_class,
                        rotation,
                    )
                )

        if not accepted:
            raw_text, raw_confidence, raw_rotation = best_raw
            return raw_text, None, 0.0, raw_rotation, raw_confidence

        (
            best_score,
            best_confidence,
            _text_length,
            best_text,
            best_class,
            best_rotation,
        ) = max(accepted)
        return (
            best_text,
            best_class,
            float(best_score),
            best_rotation,
            float(best_confidence),
        )

    def _classify(self, text: str):
        """按类别关键词匹配文本；分数不足或类别不明确时拒绝。"""
        text = "".join(str(text).split())
        if len(text) < self.min_text_length:
            return None, 0.0

        class_scores = []
        for class_name, keywords in zip(self.output_classes, self.template_keywords):
            score = max(
                100.0 if keyword in text else float(fuzz.ratio(text, keyword))
                for keyword in keywords
            )
            class_scores.append((score, class_name))

        ranked = sorted(class_scores, reverse=True)
        best_score, best_class = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0
        if best_score < self.min_match_score:
            return None, 0.0
        if best_score - second_score < self.min_match_margin:
            return None, 0.0
        return best_class, float(best_score)

    def close(self) -> None:
        """释放 PaddleOCR det/rec/cls 后端资源。"""
        close = getattr(self._engine, "close", None)
        if close is not None:
            close()
