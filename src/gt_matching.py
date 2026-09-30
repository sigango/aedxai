"""Ground-truth loading, detection-to-GT matching, and AP for the evidence experiments.

The explanation metrics in ``src/evaluator.py`` stay annotation-free. Annotations
are used here only to label each detection as a true or false positive, which is
the independent endpoint the evidence experiments are scored on.

All annotation files are read in COCO format. Other datasets (VOC, BDD100K,
VisDrone) are converted by ``scripts/convert_annotations.py`` into COCO format
with COCO-80 category names, so class ids here always match ``Detection.class_id``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .utils import COCO_CLASSES

logger = logging.getLogger(__name__)

COCO_NAME_TO_IDX: dict[str, int] = {name: index for index, name in enumerate(COCO_CLASSES)}
# Category name used by converted datasets for class-agnostic "ignore" regions
# (e.g. VisDrone ignored areas). Loaded as crowd boxes with class_id = -1.
IGNORE_CATEGORY_NAME = "ignore"
IGNORE_CLASS_ID = -1


@dataclass(slots=True)
class GroundTruthBox:
    """One annotated object in [x1, y1, x2, y2] pixel coordinates."""

    bbox: list[float]
    class_id: int
    iscrowd: bool = False
    annotation_id: int = -1

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass(slots=True)
class ImageAnnotations:
    """All annotations for one image plus optional image-level attributes."""

    file_name: str
    image_id: int
    width: int
    height: int
    boxes: list[GroundTruthBox] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MatchResult:
    """Outcome of matching one detection against the image's ground truth."""

    label: str  # "tp", "fp", or "ignore" (matched a crowd region; excluded from AP)
    gt_index: int | None
    iou_gt: float
    gt_bbox: list[float] | None
    fp_type: str | None  # "duplicate", "localization", "classification", "background"
    best_iou_same_class: float
    best_iou_any_class: float

    @property
    def is_tp(self) -> bool:
        return self.label == "tp"


class AnnotationIndex:
    """File-name keyed lookup over a COCO-format annotation file."""

    def __init__(self, images: dict[str, ImageAnnotations], source: str = "") -> None:
        self._images = images
        self.source = source

    @classmethod
    def from_coco_json(cls, path: str | Path, assume_coco_ids: bool = True) -> "AnnotationIndex":
        """Load a COCO-format JSON; categories are mapped to COCO-80 indices by name.

        Categories whose name is not a COCO-80 class are dropped. With
        ``assume_coco_ids`` a category that fails name matching falls back to the
        official COCO 91-id mapping (useful for slightly renamed COCO files).
        """
        from .detector import COCO_91_TO_80

        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)

        category_map: dict[int, int] = {}
        for category in raw.get("categories", []):
            name = str(category.get("name", "")).strip().lower()
            if name == IGNORE_CATEGORY_NAME:
                category_map[int(category["id"])] = IGNORE_CLASS_ID
            elif name in COCO_NAME_TO_IDX:
                category_map[int(category["id"])] = COCO_NAME_TO_IDX[name]
            elif assume_coco_ids and int(category["id"]) in COCO_91_TO_80:
                category_map[int(category["id"])] = COCO_91_TO_80[int(category["id"])]

        images: dict[str, ImageAnnotations] = {}
        by_id: dict[int, ImageAnnotations] = {}
        for image in raw.get("images", []):
            entry = ImageAnnotations(
                file_name=Path(str(image["file_name"])).name,
                image_id=int(image["id"]),
                width=int(image.get("width", 0)),
                height=int(image.get("height", 0)),
                attributes=dict(image.get("attributes", {})),
            )
            images[entry.file_name] = entry
            by_id[entry.image_id] = entry

        dropped = 0
        for annotation in raw.get("annotations", []):
            entry = by_id.get(int(annotation["image_id"]))
            class_id = category_map.get(int(annotation["category_id"]))
            if entry is None or class_id is None:
                dropped += 1
                continue
            x, y, w, h = [float(value) for value in annotation["bbox"]]
            entry.boxes.append(
                GroundTruthBox(
                    bbox=[x, y, x + w, y + h],
                    class_id=class_id,
                    iscrowd=bool(annotation.get("iscrowd", 0)) or class_id == IGNORE_CLASS_ID,
                    annotation_id=int(annotation.get("id", -1)),
                )
            )
        if dropped:
            logger.info("Dropped %d annotations with unknown image or non-COCO-80 category from %s", dropped, path)
        return cls(images, source=str(path))

    def get(self, file_name: str) -> ImageAnnotations | None:
        """Return annotations for an image file name (basename match)."""
        return self._images.get(Path(file_name).name)

    def __contains__(self, file_name: str) -> bool:
        return Path(file_name).name in self._images

    def __len__(self) -> int:
        return len(self._images)

    def file_names(self) -> list[str]:
        return sorted(self._images)


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """IoU of two [x1, y1, x2, y2] boxes."""
    ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def _crowd_iou(det_box: Sequence[float], crowd_box: Sequence[float]) -> float:
    """COCO crowd overlap: intersection over the detection's own area."""
    ix1, iy1 = max(det_box[0], crowd_box[0]), max(det_box[1], crowd_box[1])
    ix2, iy2 = min(det_box[2], crowd_box[2]), min(det_box[3], crowd_box[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(0.0, det_box[2] - det_box[0]) * max(0.0, det_box[3] - det_box[1])
    return float(intersection / area) if area > 0 else 0.0


def match_detections(
    boxes: Sequence[Sequence[float]],
    class_ids: Sequence[int],
    scores: Sequence[float],
    gt_boxes: Sequence[GroundTruthBox],
    iou_threshold: float = 0.5,
) -> list[MatchResult]:
    """COCO-style greedy matching: highest score first, each GT matched at most once.

    Results are returned in the input order of ``boxes``.
    """
    num_dets = len(boxes)
    results: list[MatchResult | None] = [None] * num_dets
    matched_gt: set[int] = set()
    order = sorted(range(num_dets), key=lambda index: float(scores[index]), reverse=True)

    for det_index in order:
        det_box = [float(value) for value in boxes[det_index]]
        det_class = int(class_ids[det_index])

        best_any = 0.0
        best_same = 0.0
        best_same_matched = 0.0
        best_other_class = 0.0
        best_free_iou = 0.0
        best_free_index: int | None = None
        crowd_hit = False

        for gt_index, gt in enumerate(gt_boxes):
            if gt.iscrowd:
                same_or_any = gt.class_id == det_class or gt.class_id == IGNORE_CLASS_ID
                if same_or_any and _crowd_iou(det_box, gt.bbox) >= iou_threshold:
                    crowd_hit = True
                continue
            iou = box_iou(det_box, gt.bbox)
            best_any = max(best_any, iou)
            if gt.class_id != det_class:
                best_other_class = max(best_other_class, iou)
                continue
            best_same = max(best_same, iou)
            if gt_index in matched_gt:
                best_same_matched = max(best_same_matched, iou)
            elif iou > best_free_iou:
                best_free_iou = iou
                best_free_index = gt_index

        if best_free_index is not None and best_free_iou >= iou_threshold:
            matched_gt.add(best_free_index)
            results[det_index] = MatchResult(
                label="tp",
                gt_index=best_free_index,
                iou_gt=best_free_iou,
                gt_bbox=list(gt_boxes[best_free_index].bbox),
                fp_type=None,
                best_iou_same_class=best_same,
                best_iou_any_class=best_any,
            )
            continue

        if crowd_hit:
            label, fp_type = "ignore", None
        elif best_same_matched >= iou_threshold:
            label, fp_type = "fp", "duplicate"
        elif best_same >= 0.1:
            label, fp_type = "fp", "localization"
        elif best_other_class >= iou_threshold:
            label, fp_type = "fp", "classification"
        else:
            label, fp_type = "fp", "background"
        results[det_index] = MatchResult(
            label=label,
            gt_index=None,
            iou_gt=best_same,
            gt_bbox=None,
            fp_type=fp_type,
            best_iou_same_class=best_same,
            best_iou_any_class=best_any,
        )

    return [result for result in results if result is not None]


def average_precision(
    images: Iterable[Mapping[str, Any]],
    iou_threshold: float = 0.5,
    score_key: str = "score",
) -> dict[str, Any]:
    """COCO-style 101-point interpolated AP per class at one IoU threshold.

    Each image mapping holds ``detections`` (dicts with ``bbox``, ``class_id`` and
    ``score_key``) and ``gt`` (dicts with ``bbox``, ``class_id``, ``iscrowd``).
    Detections are re-matched with the given scores, so re-scoring changes TP/FP.
    """
    per_class_scores: dict[int, list[float]] = {}
    per_class_tp: dict[int, list[int]] = {}
    per_class_num_gt: dict[int, int] = {}

    for image in images:
        gt_list = [
            GroundTruthBox(
                bbox=[float(v) for v in gt["bbox"]],
                class_id=int(gt["class_id"]),
                iscrowd=bool(gt.get("iscrowd", False)),
            )
            for gt in image.get("gt", [])
        ]
        for gt in gt_list:
            if not gt.iscrowd:
                per_class_num_gt[gt.class_id] = per_class_num_gt.get(gt.class_id, 0) + 1

        detections = list(image.get("detections", []))
        if not detections:
            continue
        matches = match_detections(
            boxes=[det["bbox"] for det in detections],
            class_ids=[int(det["class_id"]) for det in detections],
            scores=[float(det[score_key]) for det in detections],
            gt_boxes=gt_list,
            iou_threshold=iou_threshold,
        )
        for det, match in zip(detections, matches):
            if match.label == "ignore":
                continue
            class_id = int(det["class_id"])
            per_class_scores.setdefault(class_id, []).append(float(det[score_key]))
            per_class_tp.setdefault(class_id, []).append(1 if match.is_tp else 0)

    recall_grid = np.linspace(0.0, 1.0, 101)
    per_class_ap: dict[int, float] = {}
    for class_id, num_gt in per_class_num_gt.items():
        if num_gt == 0:
            continue
        scores = np.asarray(per_class_scores.get(class_id, []), dtype=np.float64)
        tps = np.asarray(per_class_tp.get(class_id, []), dtype=np.float64)
        if scores.size == 0:
            per_class_ap[class_id] = 0.0
            continue
        order = np.argsort(-scores, kind="mergesort")
        tp_cum = np.cumsum(tps[order])
        fp_cum = np.cumsum(1.0 - tps[order])
        recall = tp_cum / num_gt
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        indices = np.searchsorted(recall, recall_grid, side="left")
        sampled = np.array([precision[i] if i < precision.size else 0.0 for i in indices])
        per_class_ap[class_id] = float(sampled.mean())

    mean_ap = float(np.mean(list(per_class_ap.values()))) if per_class_ap else 0.0
    return {"ap": mean_ap, "per_class_ap": per_class_ap, "num_classes": len(per_class_ap)}


__all__ = [
    "AnnotationIndex",
    "GroundTruthBox",
    "ImageAnnotations",
    "MatchResult",
    "average_precision",
    "box_iou",
    "match_detections",
]
