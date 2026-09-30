"""Tests for ground-truth loading, COCO-style matching, and AP."""

from __future__ import annotations

import json

import pytest

from src.gt_matching import AnnotationIndex, GroundTruthBox, average_precision, match_detections


def test_match_labels_tp_duplicate_localization_background() -> None:
    gts = [GroundTruthBox(bbox=[0, 0, 10, 10], class_id=0)]
    boxes = [[0, 0, 10, 10], [0, 0, 10, 9], [0, 0, 20, 20], [50, 50, 60, 60]]
    matches = match_detections(boxes, [0, 0, 0, 0], [0.9, 0.8, 0.7, 0.6], gts)
    assert [m.label for m in matches] == ["tp", "fp", "fp", "fp"]
    assert [m.fp_type for m in matches] == [None, "duplicate", "localization", "background"]
    assert matches[0].gt_bbox == [0, 0, 10, 10]


def test_match_uses_score_order_not_input_order() -> None:
    gts = [GroundTruthBox(bbox=[0, 0, 10, 10], class_id=0)]
    matches = match_detections([[0, 0, 10, 9], [0, 0, 10, 10]], [0, 0], [0.2, 0.9], gts)
    assert [m.label for m in matches] == ["fp", "tp"]


def test_classification_fp_and_crowd_ignore() -> None:
    gts = [
        GroundTruthBox(bbox=[0, 0, 10, 10], class_id=1),
        GroundTruthBox(bbox=[20, 20, 60, 60], class_id=0, iscrowd=True),
        GroundTruthBox(bbox=[100, 100, 140, 140], class_id=-1, iscrowd=True),
    ]
    matches = match_detections([[0, 0, 10, 10], [25, 25, 35, 35], [105, 105, 120, 120]], [0, 0, 5], [0.9, 0.8, 0.7], gts)
    assert matches[0].label == "fp" and matches[0].fp_type == "classification"
    assert matches[1].label == "ignore"
    assert matches[2].label == "ignore"  # class-agnostic ignore region


def test_annotation_index_maps_names_to_coco80(tmp_path) -> None:
    path = tmp_path / "ann.json"
    path.write_text(
        json.dumps(
            {
                "images": [{"id": 7, "file_name": "a/000007.jpg", "width": 64, "height": 48, "attributes": {"timeofday": "night"}}],
                "annotations": [
                    {"id": 1, "image_id": 7, "category_id": 3, "bbox": [1, 2, 10, 20], "iscrowd": 0},
                    {"id": 2, "image_id": 7, "category_id": 99, "bbox": [1, 2, 3, 4], "iscrowd": 0},
                ],
                "categories": [{"id": 3, "name": "car"}, {"id": 99, "name": "traffic sign"}],
            }
        )
    )
    index = AnnotationIndex.from_coco_json(path)
    entry = index.get("000007.jpg")
    assert entry is not None and entry.attributes == {"timeofday": "night"}
    assert len(entry.boxes) == 1
    assert entry.boxes[0].class_id == 2 and entry.boxes[0].bbox == [1.0, 2.0, 11.0, 22.0]


def test_average_precision_perfect_and_rescored() -> None:
    images = [
        {
            "gt": [{"bbox": [0, 0, 10, 10], "class_id": 0}],
            "detections": [
                {"bbox": [0, 0, 10, 10], "class_id": 0, "score": 0.4, "better": 0.9},
                {"bbox": [30, 30, 40, 40], "class_id": 0, "score": 0.8, "better": 0.1},
            ],
        }
    ]
    assert average_precision(images, score_key="better")["ap"] == pytest.approx(1.0)
    assert average_precision(images, score_key="score")["ap"] < 1.0
