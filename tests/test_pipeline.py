"""Smoke tests for AED-XAI pipeline-facing containers and fixtures."""

from __future__ import annotations

from pathlib import Path

from src.pipeline import PipelineResult


def test_sample_image_fixture_downloads(sample_image: Path) -> None:
    """The shared sample image fixture should download the expected COCO file."""
    assert sample_image.exists()
    assert sample_image.name == "000000000139.jpg"


def test_pipeline_result_container(sample_detections) -> None:
    """PipelineResult should preserve collected detections and scores."""
    result = PipelineResult(
        image_path="data/coco/val2017/000000000139.jpg",
        detections=sample_detections,
        composite_score=0.5,
    )

    assert len(result.detections) == 3
    assert result.composite_score == 0.5

