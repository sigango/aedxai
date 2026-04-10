"""Unit tests for VLM prompt building, response parsing, and drawing helpers."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from PIL import Image

from src.detector import Detection
from src.vlm_judge import Assessment, VLMJudge


def test_prompt_building() -> None:
    """_build_prompt produces a valid string with detection JSON inserted."""
    judge = VLMJudge()
    detections = [
        Detection(
            bbox=[10, 20, 100, 200],
            class_id=0,
            class_name="person",
            confidence=0.9,
            area=8100,
            relative_size="medium",
            detection_id=0,
        )
    ]
    prompt = judge._build_prompt(detections)
    assert "person" in prompt
    assert "0.9" in prompt or "0.90" in prompt
    assert "detections_json" not in prompt


def test_parse_valid_json() -> None:
    """_parse_response correctly parses well-formed JSON."""
    judge = VLMJudge()
    response = (
        '{"assessments": [{"detection_id": 0, "quality_score": 0.85, '
        '"scene_complexity": "medium", "object_relative_size": "large", '
        '"is_false_positive": false, "reasoning": "Good fit"}]}'
    )
    results = judge._parse_response(response, num_detections=1)
    assert len(results) == 1
    assert results[0].quality_score == 0.85
    assert results[0].scene_complexity == "medium"


def test_parse_malformed_json() -> None:
    """_parse_response repairs a common malformed JSON response."""
    judge = VLMJudge()
    response = (
        '{"assessments": [{"detection_id": 0, "quality_score": 0.85, '
        '"scene_complexity": "medium", "object_relative_size": "large", '
        '"is_false_positive": false, "reasoning": "Good fit"}]'
    )
    results = judge._parse_response(response, num_detections=1)
    assert len(results) == 1


def test_parse_markdown_wrapped() -> None:
    """_parse_response handles JSON wrapped in markdown code fences."""
    judge = VLMJudge()
    response = (
        "```json\n"
        '{"assessments": [{"detection_id": 0, "quality_score": 0.5, '
        '"scene_complexity": "low", "object_relative_size": "small", '
        '"is_false_positive": false, "reasoning": "ok"}]}\n'
        "```"
    )
    results = judge._parse_response(response, num_detections=1)
    assert len(results) == 1


def test_score_clamping() -> None:
    """Scores outside [0, 1] are clamped."""
    judge = VLMJudge()
    response = (
        '{"assessments": [{"detection_id": 0, "quality_score": 1.5, '
        '"scene_complexity": "low", "object_relative_size": "small", '
        '"is_false_positive": false, "reasoning": "ok"}]}'
    )
    results = judge._parse_response(response, num_detections=1)
    assert results[0].quality_score == 1.0


def test_draw_boxes() -> None:
    """_draw_boxes_on_image returns a PIL image with the original size."""
    judge = VLMJudge()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = [
        Detection(
            bbox=[10, 10, 100, 100],
            class_id=0,
            class_name="person",
            confidence=0.9,
            area=8100,
            relative_size="medium",
            detection_id=0,
        )
    ]
    result = judge._draw_boxes_on_image(image, detections)
    assert isinstance(result, Image.Image)
    assert result.size == (640, 480)


@pytest.mark.slow
def test_model_loads_on_gpu() -> None:
    """Qwen2.5-VL loads successfully on GPU with quantization."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    if importlib.util.find_spec("qwen_vl_utils") is None:
        pytest.skip("qwen-vl-utils is not installed.")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")

    judge = VLMJudge()
    judge.load_model()
    assert judge.model is not None
    vram_gb = torch.cuda.memory_allocated() / 1e9
    assert vram_gb < 12
    judge.unload_model()


def test_assessment_dataclass_fields() -> None:
    """Assessment preserves the parsed VLM output fields."""
    assessment = Assessment(
        detection_id=0,
        quality_score=0.91,
        scene_complexity="medium",
        object_relative_size="large",
        is_false_positive=False,
        reasoning="The box tightly covers the visible object.",
    )

    assert assessment.quality_score == 0.91
    assert assessment.is_false_positive is False
