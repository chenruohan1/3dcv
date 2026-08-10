#!/usr/bin/env python3
"""Audit OCR candidate and fallback behavior across evaluation sequences."""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import cv2

from core.components.detector.yolo.factory import build_yolo_detector
from core.components.ocr.paddle_ocr import PaddleOcr
from core.config_loader import load_config
from core.types import Detection, Frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the OCR strategy over RGB evaluation sequences."
    )
    parser.add_argument("--data-root", required=True, help="Directory containing eval_* folders")
    parser.add_argument(
        "--model",
        default="models/yolov11s_2026.onnx",
        help="YOLO ONNX model path",
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default="runs/ocr_strategy_audit")
    parser.add_argument("--read-interval", type=int, default=10)
    parser.add_argument(
        "--strategy",
        choices=("normal", "forced-fallback"),
        default="normal",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional global frame limit for a smoke test",
    )
    parser.add_argument("--no-save-review", action="store_true")
    return parser.parse_args()


def _bbox_text(bbox) -> str:
    return ",".join(str(int(value)) for value in bbox)


def _annotate_review(bgr, target_bbox, masked_detections, label: str):
    output = bgr.copy()
    for detection in masked_detections:
        x1, y1, x2, y2 = (int(value) for value in detection.bbox)
        cv2.rectangle(output, (x1, y1), (x2, y2), (128, 128, 128), 2)
    x1, y1, x2, y2 = (int(value) for value in target_bbox)
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(
        output,
        label,
        (max(0, x1), max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def main() -> int:
    args = parse_args()
    if args.read_interval <= 0:
        raise ValueError("--read-interval must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")

    data_root = Path(args.data_root)
    model_path = Path(args.model)
    output_dir = Path(args.output)
    review_dir = output_dir / "review_nonempty"
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {data_root}")
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO model does not exist: {model_path}")
    if model_path.suffix.lower() != ".onnx":
        raise ValueError("--model must point to an .onnx file")

    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_save_review:
        review_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    detector_config = dict(config["detector"])
    detector_config.update(
        {
            "backend": "onnx",
            "weights": str(model_path.with_suffix("")),
            "providers": ["CPUExecutionProvider"],
        }
    )

    detector = build_yolo_detector(detector_config, config["class_registry"])
    ocr = PaddleOcr(config["ocr"], config["class_registry"])

    counters = Counter()
    mode_counts = Counter()
    class_counts = Counter()
    sequence_counts = Counter()
    detector_times = []
    ocr_times = []
    started = time.perf_counter()
    csv_path = output_dir / "predictions.csv"
    fields = [
        "sequence",
        "frame_index",
        "frame_name",
        "image_path",
        "strategy",
        "book_detection_count",
        "detector_count",
        "table_source",
        "ocr_mode",
        "target_bbox",
        "masked_detection_count",
        "masked_detection_classes",
        "text",
        "text_length",
        "predicted_class",
        "match_score",
        "accepted",
        "detector_ms",
        "ocr_ms",
    ]

    try:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()

            stop = False
            for sequence_dir in sorted(data_root.glob("eval_*")):
                rgb_dir = sequence_dir / "rgb"
                if not rgb_dir.is_dir():
                    continue
                images = sorted(
                    path
                    for path in rgb_dir.iterdir()
                    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
                )
                selected = images[:: args.read_interval]
                for selected_index, image_path in enumerate(selected):
                    original_index = selected_index * args.read_interval + 1
                    if args.max_frames is not None and counters["frames"] >= args.max_frames:
                        stop = True
                        break

                    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if bgr is None:
                        counters["read_errors"] += 1
                        continue
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    frame = Frame(
                        frame_id=f"{sequence_dir.name}/{image_path.stem}",
                        rgb=rgb,
                        depth=None,
                        timestamp=0.0,
                    )

                    detector_started = time.perf_counter()
                    detections = detector.infer(frame, table=1)
                    detector_ms = (time.perf_counter() - detector_started) * 1000.0
                    detector_times.append(detector_ms)

                    book_detections = [
                        detection
                        for detection in detections
                        if detection.class_name in ocr.candidate_classes
                    ]
                    counters["frames"] += 1
                    sequence_counts[sequence_dir.name] += 1
                    if book_detections:
                        counters["frames_with_book_detection"] += 1

                    ocr_detections = detections
                    if args.strategy == "forced-fallback":
                        ocr_detections = [
                            detection
                            for detection in detections
                            if detection.class_name not in ocr.candidate_classes
                        ]

                    table_source = "detector"
                    if not any(
                        detection.class_name == ocr.fallback_candidate_class
                        for detection in ocr_detections
                    ):
                        table_source = "configured_default"
                        ocr_detections = [
                            *ocr_detections,
                            Detection(
                                class_name=ocr.fallback_candidate_class,
                                bbox=tuple(config["table_locator"]["default_bbox"]),
                                score=1.0,
                                evidence={"source": "audit_default_bbox"},
                            ),
                        ]

                    targets = ocr._select_ocr_targets(frame, ocr_detections)
                    if not targets:
                        counters["frames_without_target"] += 1
                        writer.writerow(
                            {
                                "sequence": sequence_dir.name,
                                "frame_index": original_index,
                                "frame_name": image_path.name,
                                "image_path": str(image_path),
                                "strategy": args.strategy,
                                "book_detection_count": len(book_detections),
                                "detector_count": len(detections),
                                "table_source": table_source,
                                "accepted": False,
                                "detector_ms": f"{detector_ms:.3f}",
                                "ocr_ms": "0.000",
                            }
                        )
                        continue

                    for target_index, (bbox, mode, _source_class) in enumerate(targets):
                        masked = ocr._fallback_mask_detections(
                            ocr_detections,
                            mode,
                            bbox,
                        )
                        ocr_started = time.perf_counter()
                        text = ocr._read_text(
                            rgb,
                            bbox,
                            mask_bboxes=[detection.bbox for detection in masked],
                        )
                        ocr_ms = (time.perf_counter() - ocr_started) * 1000.0
                        ocr_times.append(ocr_ms)
                        predicted_class, match_score = (
                            ocr._classify(text) if text else (None, 0.0)
                        )
                        accepted = predicted_class is not None

                        counters["targets"] += 1
                        mode_counts[mode] += 1
                        if text:
                            counters["nonempty_text_targets"] += 1
                        if accepted:
                            counters["accepted_targets"] += 1
                            class_counts[predicted_class] += 1
                            if mode == "table_fallback":
                                counters["accepted_table_fallback"] += 1
                            if not book_detections:
                                counters["accepted_without_book_detection"] += 1

                        writer.writerow(
                            {
                                "sequence": sequence_dir.name,
                                "frame_index": original_index,
                                "frame_name": image_path.name,
                                "image_path": str(image_path),
                                "strategy": args.strategy,
                                "book_detection_count": len(book_detections),
                                "detector_count": len(detections),
                                "table_source": table_source,
                                "ocr_mode": mode,
                                "target_bbox": _bbox_text(bbox),
                                "masked_detection_count": len(masked),
                                "masked_detection_classes": ";".join(
                                    sorted({detection.class_name for detection in masked})
                                ),
                                "text": text,
                                "text_length": len(text),
                                "predicted_class": predicted_class or "",
                                "match_score": f"{match_score:.6f}",
                                "accepted": accepted,
                                "detector_ms": f"{detector_ms:.3f}",
                                "ocr_ms": f"{ocr_ms:.3f}",
                            }
                        )

                        if text and not args.no_save_review:
                            label = f"{mode} {predicted_class or 'REJECT'} {match_score:.1f}"
                            review = _annotate_review(bgr, bbox, masked, label)
                            review_name = (
                                f"{sequence_dir.name}_{image_path.stem}_"
                                f"{target_index}_{predicted_class or 'REJECT'}.jpg"
                            )
                            cv2.imwrite(str(review_dir / review_name), review)

                    if counters["frames"] % 50 == 0:
                        csv_file.flush()
                        elapsed = time.perf_counter() - started
                        rate = counters["frames"] / elapsed if elapsed > 0 else 0.0
                        print(
                            f"progress frames={counters['frames']} "
                            f"accepted={counters['accepted_targets']} "
                            f"rate={rate:.2f} frame/s",
                            flush=True,
                        )
                if stop:
                    break
    finally:
        ocr.close()
        detector.close()

    elapsed = time.perf_counter() - started
    summary = {
        "strategy": args.strategy,
        "read_interval": args.read_interval,
        "elapsed_sec": round(elapsed, 3),
        "frame_rate": round(counters["frames"] / elapsed, 3) if elapsed > 0 else 0.0,
        "counters": dict(counters),
        "mode_counts": dict(mode_counts),
        "class_counts": dict(class_counts),
        "sequence_frame_counts": dict(sequence_counts),
        "mean_detector_ms": (
            round(sum(detector_times) / len(detector_times), 3) if detector_times else 0.0
        ),
        "mean_ocr_ms": round(sum(ocr_times) / len(ocr_times), 3) if ocr_times else 0.0,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")
    if not args.no_save_review:
        print(f"Review images: {review_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
