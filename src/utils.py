"""Shared utility helpers for logging, image I/O, seeding, and box geometry."""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

COCO_CLASSES: list[str] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with the AED-XAI default message format."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s %(name)s %(levelname)s] %(message)s",
        force=True,
    )


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as an RGB uint8 NumPy array."""
    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
        return np.array(rgb_image, dtype=np.uint8, copy=True)


def save_image(image: np.ndarray, path: str) -> None:
    """Save an RGB NumPy image array to disk, creating parent folders if needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(clipped).save(destination)


def bbox_area(bbox: list[int]) -> int:
    """Compute the area of an [x1, y1, x2, y2] bounding box."""
    x1, y1, x2, y2 = [int(value) for value in bbox]
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_iou(bbox1: list[int], bbox2: list[int]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] bounding boxes."""
    ax1, ay1, ax2, ay2 = [int(value) for value in bbox1]
    bx1, by1, bx2, by2 = [int(value) for value in bbox2]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = bbox_area([inter_x1, inter_y1, inter_x2, inter_y2])
    union_area = bbox_area(bbox1) + bbox_area(bbox2) - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def pairwise_iou(bboxes: list[list[int]]) -> np.ndarray:
    """Compute the dense NxN IoU matrix for a list of bounding boxes."""
    num_boxes = len(bboxes)
    matrix = np.zeros((num_boxes, num_boxes), dtype=np.float32)

    for row_index in range(num_boxes):
        for col_index in range(row_index, num_boxes):
            iou_value = bbox_iou(bboxes[row_index], bboxes[col_index])
            matrix[row_index, col_index] = iou_value
            matrix[col_index, row_index] = iou_value

    return matrix


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the preferred compute device for the current runtime."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_detections_json(detections: list[Any]) -> str:
    """Serialize detections into a stable JSON format for VLM prompting."""
    serialized: list[dict[str, Any]] = []

    for detection in detections:
        if is_dataclass(detection):
            payload: dict[str, Any] = asdict(detection)
        elif isinstance(detection, Mapping):
            payload = dict(detection)
        else:
            payload = {
                key: getattr(detection, key)
                for key in (
                    "detection_id",
                    "bbox",
                    "class_id",
                    "class_name",
                    "confidence",
                    "model_name",
                    "metadata",
                )
                if hasattr(detection, key)
            }

        serialized.append(
            {
                "detection_id": int(payload.get("detection_id", len(serialized))),
                "bbox": [int(value) for value in payload.get("bbox", [])],
                "class_id": int(payload.get("class_id", -1)),
                "class_name": str(payload.get("class_name", "unknown")),
                "confidence": float(payload.get("confidence", 0.0)),
                "model_name": payload.get("model_name"),
                "metadata": payload.get("metadata", {}),
            }
        )

    return json.dumps(serialized, indent=2)
