"""Serialization helpers for PipelineResult — one directory per image."""

from __future__ import annotations

import json
import logging
from dataclasses import is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from .detector import Detection
    from .evaluator import EvalResult
    from .pipeline import PipelineResult
    from .xai_methods.base import SaliencyMap

logger = logging.getLogger(__name__)


def _to_jsonable(value: Any) -> Any:
    """Convert numpy / dataclass / path values into JSON-friendly primitives."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if is_dataclass(value):
        return {
            str(key): _to_jsonable(getattr(value, key))
            for key in value.__dataclass_fields__  # type: ignore[attr-defined]
        }
    return str(value)


def _detection_dict(detection: "Detection") -> dict[str, Any]:
    """Flatten a Detection object into a JSON-friendly dict."""
    x1, y1, x2, y2 = detection.bbox
    return {
        "detection_id": int(detection.detection_id),
        "class_id": int(detection.class_id),
        "class_name": str(getattr(detection, "class_name", "")),
        "confidence": float(detection.confidence),
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "relative_size": str(getattr(detection, "relative_size", "")),
    }


def _saliency_to_png(saliency_map: np.ndarray, output_path: Path) -> None:
    """Render a [0, 1] saliency map as an 8-bit colormapped PNG."""
    clipped = np.clip(np.nan_to_num(saliency_map, nan=0.0), 0.0, 1.0)
    heatmap = (clipped * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    cv2.imwrite(str(output_path), colored)


def _overlay_detections(image_path: str, detections: list["Detection"], output_path: Path) -> None:
    """Save the source image with bbox overlays for quick inspection."""
    image = cv2.imread(image_path)
    if image is None:
        return
    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection.bbox]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{getattr(detection, 'class_name', detection.class_id)} {detection.confidence:.2f}"
        cv2.putText(
            image,
            label,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output_path), image)


def save_pipeline_result(
    result: "PipelineResult",
    output_dir: str | Path,
    save_saliency_npy: bool = False,
) -> Path:
    """Write one result into a directory: JSON summary + saliency PNGs.

    Args:
        result: The PipelineResult returned by AEDXAIPipeline.run_on_image.
        output_dir: Parent directory; a subfolder named after the image stem is
            created inside it.
        save_saliency_npy: If True, also dump the raw float32 saliency arrays
            alongside the colormapped PNGs (useful for downstream analysis).

    Returns:
        The path to the created per-image directory.
    """
    parent = Path(output_dir)
    image_stem = Path(result.image_path).stem
    image_dir = parent / image_stem
    image_dir.mkdir(parents=True, exist_ok=True)

    detections = list(result.detections)
    saliency_maps: list["SaliencyMap"] = list(result.saliency_maps)
    eval_results: list["EvalResult"] = list(result.evaluation_results)
    reasoning = dict(result.selector_reasoning)

    # Align per-detection outputs by detection_id where possible; fall back to
    # positional alignment for older callers that don't thread ids through.
    detection_by_id = {int(det.detection_id): det for det in detections}
    saliency_by_id = {int(sal.detection_id): sal for sal in saliency_maps}

    per_detection: list[dict[str, Any]] = []
    for index, detection in enumerate(detections):
        detection_id = int(detection.detection_id)
        saliency = saliency_by_id.get(detection_id)
        if saliency is None and index < len(saliency_maps):
            saliency = saliency_maps[index]
        evaluation = eval_results[index] if index < len(eval_results) else None
        reasoning_entry = reasoning.get(detection_id, {})

        saliency_png_rel: str | None = None
        saliency_npy_rel: str | None = None
        if saliency is not None:
            saliency_png = image_dir / f"det_{detection_id:03d}_saliency.png"
            _saliency_to_png(np.asarray(saliency.map, dtype=np.float32), saliency_png)
            saliency_png_rel = saliency_png.name
            if save_saliency_npy:
                saliency_npy = image_dir / f"det_{detection_id:03d}_saliency.npy"
                np.save(saliency_npy, np.asarray(saliency.map, dtype=np.float32))
                saliency_npy_rel = saliency_npy.name

        per_detection.append(
            {
                "detection": _detection_dict(detection),
                "chosen_method": (
                    reasoning_entry.get("method")
                    or (getattr(saliency, "method_name", None) if saliency is not None else None)
                ),
                "reasoning": _to_jsonable(reasoning_entry) if reasoning_entry else None,
                "saliency": {
                    "method_name": getattr(saliency, "method_name", None) if saliency is not None else None,
                    "computation_time": float(getattr(saliency, "computation_time", 0.0)) if saliency is not None else None,
                    "png": saliency_png_rel,
                    "npy": saliency_npy_rel,
                },
                "metrics": evaluation.as_dict() if evaluation is not None else None,
            }
        )

    _overlay_detections(result.image_path, detections, image_dir / "detections.png")

    summary = {
        "image_path": str(result.image_path),
        "image_stem": image_stem,
        "num_detections": len(detections),
        "composite_score": (
            float(result.composite_score) if result.composite_score is not None else None
        ),
        "metadata": _to_jsonable(result.metadata),
        "assessments": _to_jsonable(result.assessments),
        "detections": per_detection,
    }

    json_path = image_dir / "result.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("Saved pipeline result -> %s", image_dir)
    return image_dir


__all__ = ["save_pipeline_result"]
