#!/usr/bin/env python3
"""Run one image through YOLO and the OCR table-fallback path."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from core.components.detector.yolo.factory import build_yolo_detector
from core.components.ocr.paddle_ocr import PaddleOcr
from core.config_loader import load_config
from core.types import Detection, Frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Book OCR or force the Table fallback on one image."
    )
    parser.add_argument("--image", required=True, help="RGB image path")
    parser.add_argument(
        "--model",
        default="models/yolov11s_2026.onnx",
        help="YOLO ONNX model path",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Project config path",
    )
    parser.add_argument(
        "--force-table-fallback",
        action="store_true",
        help="Remove OCR candidate detections before running OCR",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    model_path = Path(args.model)
    if not image_path.is_file():
        raise FileNotFoundError(f"image does not exist: {image_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO model does not exist: {model_path}")
    if model_path.suffix.lower() != ".onnx":
        raise ValueError("--model must point to an .onnx file")

    config = load_config(args.config)
    detector_config = dict(config["detector"])
    detector_config.update(
        {
            "backend": "onnx",
            "weights": str(model_path.with_suffix("")),
            "providers": ["CPUExecutionProvider"],
        }
    )

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"failed to read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    frame = Frame(
        frame_id=image_path.stem,
        rgb=rgb,
        depth=None,
        timestamp=0.0,
    )

    detector = build_yolo_detector(detector_config, config["class_registry"])
    ocr = PaddleOcr(config["ocr"], config["class_registry"])
    try:
        detections = detector.infer(frame, table=1)
        print("Detector results:")
        for detection in detections:
            print(
                f"  {detection.class_name:<16} "
                f"score={detection.score:.3f} bbox={detection.bbox}"
            )

        ocr_detections = detections
        if args.force_table_fallback:
            ocr_detections = [
                detection
                for detection in detections
                if detection.class_name not in ocr.candidate_classes
            ]
            print("\nOCR candidate detections removed to force Table fallback.")

        if not any(
            detection.class_name == ocr.fallback_candidate_class
            for detection in ocr_detections
        ):
            default_bbox = tuple(config["table_locator"]["default_bbox"])
            ocr_detections = [
                *ocr_detections,
                Detection(
                    class_name=ocr.fallback_candidate_class,
                    bbox=default_bbox,
                    score=1.0,
                    evidence={"source": "test_default_bbox"},
                ),
            ]
            print(f"Using configured default Table bbox: {default_bbox}")

        targets = ocr._select_ocr_targets(frame, ocr_detections)
        print("\nOCR targets:")
        if not targets:
            print("  none")

        for bbox, mode, source_class in targets:
            masked = ocr._fallback_mask_detections(
                ocr_detections,
                mode,
                bbox,
            )
            text = ocr._read_text(
                rgb,
                bbox,
                mask_bboxes=[detection.bbox for detection in masked],
            )
            classification = ocr._classify(text) if text else None
            print(f"  mode={mode} source={source_class} bbox={bbox}")
            print(
                "  masked="
                + str([detection.class_name for detection in masked])
            )
            print(f"  text={text!r}")
            print(f"  classification={classification}")

        results = ocr.process(frame, ocr_detections, table=1)
        print("\nFinal OCR detections:")
        if not results:
            print("  none (empty text or rejected by min_match_score)")
        for result in results:
            print(
                f"  class={result.class_name} bbox={result.bbox} "
                f"evidence={result.evidence}"
            )
    finally:
        ocr.close()
        detector.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
