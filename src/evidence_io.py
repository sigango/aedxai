"""Read evidence-run shards into flat per-detection tables (CPU, no torch needed)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCALAR_METHOD_KEYS = (
    "pg",
    "ebpg",
    "oa",
    "insertion_auc",
    "deletion_auc",
    "sparsity",
    "composite",
    "pg_gt",
    "ebpg_gt",
    "oa_box",
    "insertion_auc_box",
    "deletion_auc_box",
    "composite_box",
    "explain_time",
    "eval_time",
    "is_fallback",
)

COCO_SMALL_AREA = 32**2
COCO_LARGE_AREA = 96**2


def _curve_features(insertion: list[float], deletion: list[float], prefix: str) -> dict[str, float]:
    # Local import keeps this module importable without cv2/torch.
    from .evidence import curve_features

    return curve_features(insertion, deletion, prefix=prefix)


def iter_image_records(run_dirs: Iterable[str | Path]) -> Iterator[dict[str, Any]]:
    """Yield the last successful record per (run, detector, image) across all shards."""
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        latest: dict[tuple[str, str], dict[str, Any]] = {}
        errors = 0
        for shard in sorted((run_dir / "records").glob("shard_*.jsonl")):
            with shard.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "error" in record:
                        errors += 1
                        continue
                    record["run"] = run_dir.name
                    latest[(str(record.get("detector")), str(record["image"]))] = record
        if errors:
            logger.info("%s: %d error records ignored (retried images keep their successful record).", run_dir, errors)
        yield from latest.values()


def load_vlm_records(run_dir: str | Path) -> dict[tuple[str, int], dict[str, float]]:
    """VLM verification results keyed by (image, det_id)."""
    results: dict[tuple[str, int], dict[str, float]] = {}
    for shard in sorted((Path(run_dir) / "vlm").glob("shard_*.jsonl")):
        with shard.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                results[(str(record["image"]), int(record["det_id"]))] = dict(record.get("features", {}))
    return results


def detections_frame(run_dirs: Iterable[str | Path], include_unexplained: bool = True) -> pd.DataFrame:
    """One row per detection with GT label and every evidence feature as a column."""
    run_dirs = [Path(path) for path in run_dirs]
    vlm_by_run = {run_dir.name: load_vlm_records(run_dir) for run_dir in run_dirs}
    rows: list[dict[str, Any]] = []
    for image_record in iter_image_records(run_dirs):
        vlm = vlm_by_run.get(image_record["run"], {})
        for det in image_record.get("detections", []):
            if not include_unexplained and not det.get("explained", False):
                continue
            x1, y1, x2, y2 = det["bbox"]
            area = max(0, x2 - x1) * max(0, y2 - y1)
            match = det.get("match", {})
            row: dict[str, Any] = {
                "run": image_record["run"],
                "detector": image_record.get("detector"),
                "image": image_record["image"],
                "image_id": image_record.get("image_id"),
                "det_id": int(det["det_id"]),
                "class_id": int(det["class_id"]),
                "class_name": det.get("class_name"),
                "confidence": float(det["confidence"]),
                "relative_size": det.get("relative_size"),
                "coco_size": "small" if area < COCO_SMALL_AREA else ("large" if area > COCO_LARGE_AREA else "medium"),
                "box_area": float(area),
                "bbox": json.dumps(det["bbox"]),
                "explained": bool(det.get("explained", False)),
                "label": match.get("label"),
                "is_tp": match.get("label") == "tp",
                "fp_type": match.get("fp_type"),
                "iou_gt": float(match.get("iou_gt", 0.0)),
            }
            for key, value in dict(image_record.get("attributes", {})).items():
                row[f"attr__{key}"] = value
            for key, value in dict(det.get("output", {})).items():
                row[f"out__{key}"] = value
            for key, value in dict(det.get("gradient") or {}).items():
                row[f"grad__{key}"] = value
            for method, record in dict(det.get("methods", {})).items():
                for key in SCALAR_METHOD_KEYS:
                    if key in record:
                        row[f"m__{method}__{key}"] = float(record[key]) if record[key] is not None else np.nan
                for key, value in dict(record.get("map_features", {})).items():
                    row[f"mf__{method}__{key}"] = value
                if "insertion_curve" in record:
                    for key, value in _curve_features(record["insertion_curve"], record["deletion_curve"], "").items():
                        row[f"cf__{method}__{key}"] = value
                if "insertion_curve_box" in record:
                    for key, value in _curve_features(
                        record["insertion_curve_box"], record["deletion_curve_box"], ""
                    ).items():
                        row[f"cfb__{method}__{key}"] = value
            for pair, features in dict(det.get("agreement", {})).items():
                first, second = pair.split("|")
                for key, value in features.items():
                    row[f"ag__{first}__{second}__{key}"] = value
            for key, value in vlm.get((image_record["image"], int(det["det_id"])), {}).items():
                row[f"vlm__{key}"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def images_for_ap(run_dirs: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Per-image GT and detections in the format expected by gt_matching.average_precision."""
    images: list[dict[str, Any]] = []
    for image_record in iter_image_records(run_dirs):
        images.append(
            {
                "run": image_record["run"],
                "detector": image_record.get("detector"),
                "image": image_record["image"],
                "gt": image_record.get("gt", []),
                "detections": [
                    {
                        "det_id": int(det["det_id"]),
                        "bbox": det["bbox"],
                        "class_id": int(det["class_id"]),
                        "score": float(det["confidence"]),
                        "explained": bool(det.get("explained", False)),
                    }
                    for det in image_record.get("detections", [])
                ],
            }
        )
    return images


def feature_columns(frame: pd.DataFrame, prefixes: Iterable[str]) -> list[str]:
    """Columns starting with any of the given prefixes (numeric only)."""
    prefixes = tuple(prefixes)
    return [
        column
        for column in frame.columns
        if column.startswith(prefixes) and pd.api.types.is_numeric_dtype(frame[column])
    ]


__all__ = [
    "detections_frame",
    "feature_columns",
    "images_for_ap",
    "iter_image_records",
    "load_vlm_records",
]
