"""VLM-based object detection quality assessment using Qwen2.5-VL."""

from __future__ import annotations

import gc
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import yaml
from PIL import Image, ImageColor, ImageDraw, ImageFont
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from .detector import Detection

logger = logging.getLogger(__name__)

_VALID_SCENE_COMPLEXITY = {"low", "medium", "high"}
_VALID_OBJECT_SIZES = {"small", "medium", "large"}


class ParseError(ValueError):
    """Raised when a VLM response cannot be converted into assessment objects."""


@dataclass(slots=True)
class Assessment:
    """Structured VLM judgment for a single detection."""

    detection_id: int
    quality_score: float
    scene_complexity: str
    object_relative_size: str
    is_false_positive: bool
    reasoning: str


@dataclass(slots=True)
class VLMJudgmentBatch:
    """Compatibility wrapper around a batch of VLM assessments."""

    assessments: list[Assessment]
    raw_response: str | None = None
    prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


VLMAssessment = Assessment


def _resolve_path(path: str) -> Path:
    """Resolve relative repo paths from either cwd or the package root."""
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path

    candidates = [
        Path.cwd() / raw_path,
        Path(__file__).resolve().parents[1] / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_vlm_config(config_path: str) -> dict[str, Any]:
    """Load the `vlm` section from the configured YAML file."""
    resolved_path = _resolve_path(config_path)
    with resolved_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    if "vlm" not in raw_config:
        raise KeyError(f"Missing 'vlm' section in config: {resolved_path}")
    return dict(raw_config["vlm"])


def _require_torch() -> Any:
    """Import torch lazily so parse-only tests do not require it."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for Qwen2.5-VL inference.") from exc
    return torch


def _coerce_bool(value: Any, default: bool = False) -> bool:
    """Convert common JSON-like values into booleans."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return default


def _clamp_score(value: Any, default: float = 0.0) -> float:
    """Clamp a quality score into the closed interval [0.0, 1.0]."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    return float(max(0.0, min(1.0, score)))


def _normalize_choice(value: Any, valid_values: set[str], default: str) -> str:
    """Normalize a categorical string into one of the allowed values."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in valid_values:
            return normalized
    return default


def _extract_assessment_payloads(payload: Any) -> list[dict[str, Any]]:
    """Extract assessment dictionaries from several plausible VLM payload shapes."""
    if isinstance(payload, dict):
        if isinstance(payload.get("assessments"), list):
            return [item for item in payload["assessments"] if isinstance(item, dict)]
        if any(
            key in payload
            for key in (
                "detection_id",
                "quality_score",
                "scene_complexity",
                "object_relative_size",
                "is_false_positive",
                "reasoning",
            )
        ):
            return [payload]
        return []

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    return []


def _build_assessment(payload: Mapping[str, Any], fallback_detection_id: int) -> Assessment:
    """Construct a validated assessment object from raw parsed JSON."""
    try:
        detection_id = int(payload.get("detection_id", fallback_detection_id))
    except (TypeError, ValueError):
        detection_id = fallback_detection_id

    return Assessment(
        detection_id=detection_id,
        quality_score=_clamp_score(payload.get("quality_score", 0.0), default=0.0),
        scene_complexity=_normalize_choice(
            payload.get("scene_complexity"),
            _VALID_SCENE_COMPLEXITY,
            default="medium",
        ),
        object_relative_size=_normalize_choice(
            payload.get("object_relative_size"),
            _VALID_OBJECT_SIZES,
            default="medium",
        ),
        is_false_positive=_coerce_bool(payload.get("is_false_positive"), default=False),
        reasoning=str(payload.get("reasoning", "")).strip(),
    )


def _basic_json_repair(candidate: str) -> str:
    """Apply a very small structural repair for common truncated JSON outputs."""
    repaired = candidate.strip()

    if "```" in repaired:
        repaired = re.sub(r"^```(?:json)?\s*|\s*```$", "", repaired, flags=re.DOTALL | re.IGNORECASE).strip()

    open_brackets = repaired.count("[")
    close_brackets = repaired.count("]")
    if open_brackets > close_brackets:
        repaired += "]" * (open_brackets - close_brackets)

    open_braces = repaired.count("{")
    close_braces = repaired.count("}")
    if open_braces > close_braces:
        repaired += "}" * (open_braces - close_braces)

    return repaired


class VLMJudge:
    """VLM-based detection quality judge using Qwen2.5-VL."""

    def __init__(self, config_path: str = "config/vlm_config.yaml") -> None:
        """Load configuration, but defer model loading for VRAM management."""
        self.config_path = str(config_path)
        self.config = _load_vlm_config(config_path)

        self.model_name = str(self.config.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct"))
        self.quantization = str(self.config.get("quantization", "int4")).lower()
        self.max_new_tokens = int(self.config.get("max_new_tokens", 1024))
        self.temperature = float(self.config.get("temperature", 0.0))
        self.top_p = float(self.config.get("top_p", 1.0))
        self.device_preference = str(self.config.get("device", "cuda")).lower()
        self.max_retries = int(self.config.get("max_retries", 3))

        self.prompt_template = str(self.config.get("prompt_template", "{detections_json}"))
        self.retry_prompt_template = str(self.config.get("retry_prompt_template", "{bbox}"))
        self.annotation_config: dict[str, Any] = dict(self.config.get("annotation", {}))

        self.model: Any | None = None
        self.processor: Any | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "VLMJudge":
        """Construct a judge from an in-memory configuration mapping."""
        instance = cls.__new__(cls)
        vlm_config = dict(config["vlm"] if "vlm" in config else config)

        instance.config_path = "<in-memory>"
        instance.config = vlm_config
        instance.model_name = str(vlm_config.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct"))
        instance.quantization = str(vlm_config.get("quantization", "int4")).lower()
        instance.max_new_tokens = int(vlm_config.get("max_new_tokens", 1024))
        instance.temperature = float(vlm_config.get("temperature", 0.0))
        instance.top_p = float(vlm_config.get("top_p", 1.0))
        instance.device_preference = str(vlm_config.get("device", "cuda")).lower()
        instance.max_retries = int(vlm_config.get("max_retries", 3))
        instance.prompt_template = str(vlm_config.get("prompt_template", "{detections_json}"))
        instance.retry_prompt_template = str(vlm_config.get("retry_prompt_template", "{bbox}"))
        instance.annotation_config = dict(vlm_config.get("annotation", {}))
        instance.model = None
        instance.processor = None
        return instance

    def load_model(self) -> None:
        """Load Qwen2.5-VL with the configured quantization mode."""
        if self.model is not None and self.processor is not None:
            return

        torch = _require_torch()
        if self.device_preference.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to load the configured Qwen2.5-VL model.")

        try:
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
                Qwen2_5_VLForConditionalGeneration,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers with Qwen2-VL support is required for VLM inference."
            ) from exc

        quantization_config: Any | None = None
        torch_dtype = torch.float16

        if self.quantization == "int4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.quantization == "int8":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif self.quantization != "fp16":
            logger.warning("Unknown quantization mode '%s'; falling back to fp16.", self.quantization)

        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "device_map": "auto",
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_name, **model_kwargs)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / 1e9
            logger.info("Loaded %s. CUDA memory allocated: %.2f GB", self.model_name, allocated_gb)
        else:
            logger.info("Loaded %s on non-CUDA device map.", self.model_name)

    def unload_model(self) -> None:
        """Unload the VLM and free cached GPU memory."""
        self.model = None
        self.processor = None

        gc.collect()
        try:
            torch = _require_torch()
        except ImportError:
            return

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def assess_detections(self, image: np.ndarray, detections: list["Detection"]) -> list[Assessment]:
        """Assess all detections in one image using the configured VLM."""
        if not detections:
            return []

        annotated_image = self._draw_boxes_on_image(image, detections)
        prompt = self._build_prompt(detections)

        try:
            response = self._run_qwen_inference(
                pil_image=annotated_image,
                prompt_text=prompt,
                max_new_tokens=self.max_new_tokens,
            )
            parsed = self._parse_response(response, num_detections=len(detections))
        except ParseError as exc:
            logger.warning("Batch VLM parsing failed; retrying per detection. Error: %s", exc)
            parsed = []

        parsed_by_id = {assessment.detection_id: assessment for assessment in parsed}
        final_assessments: list[Assessment] = []

        for detection in detections:
            assessment = parsed_by_id.get(detection.detection_id)
            if assessment is None:
                try:
                    assessment = self._retry_single_detection(image=image, detection=detection)
                except Exception as exc:
                    logger.warning(
                        "Single-detection retry failed for detection %s: %s",
                        detection.detection_id,
                        exc,
                    )
                    assessment = Assessment(
                        detection_id=detection.detection_id,
                        quality_score=0.0,
                        scene_complexity="medium",
                        object_relative_size=detection.relative_size,
                        is_false_positive=False,
                        reasoning="Fallback assessment after VLM failure.",
                    )

            final_assessments.append(self._validate_assessment(assessment))

        return final_assessments

    def _draw_boxes_on_image(self, image: np.ndarray, detections: list["Detection"]) -> Image.Image:
        """Draw numbered boxes and labels onto an RGB image for VLM consumption."""
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a NumPy array")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image must have shape (H, W, 3), got {image.shape}")

        rgb = np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))
        pil_image = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(pil_image)

        font_size = int(self.annotation_config.get("font_size", 18))
        line_width = int(self.annotation_config.get("line_width", 3))
        show_confidence = bool(self.annotation_config.get("show_confidence", True))
        show_class = bool(self.annotation_config.get("show_class", True))

        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

        colors = self.annotation_config.get(
            "colors",
            ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow"],
        )

        for index, detection in enumerate(detections):
            color = ImageColor.getrgb(str(colors[index % len(colors)]))
            x1, y1, x2, y2 = detection.bbox

            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

            label_parts = [f"#{detection.detection_id}"]
            if show_class:
                label_parts.append(detection.class_name)
            if show_confidence:
                label_parts.append(f"{detection.confidence:.2f}")
            label_text = " ".join(label_parts)

            bbox = draw.textbbox((x1, y1), label_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            bg_top = max(0, y1 - text_height - 8)
            bg_bottom = bg_top + text_height + 6
            bg_right = min(pil_image.width, x1 + text_width + 10)

            draw.rectangle([x1, bg_top, bg_right, bg_bottom], fill=color)
            draw.text((x1 + 4, bg_top + 2), label_text, fill="white", font=font)

        return pil_image

    def _build_prompt(self, detections: list["Detection"]) -> str:
        """Build the configured batch prompt with detection JSON inserted."""
        detections_json = json.dumps(
            [
                {
                    "id": detection.detection_id,
                    "bbox": [int(value) for value in detection.bbox],
                    "class": detection.class_name,
                    "confidence": round(float(detection.confidence), 4),
                }
                for detection in detections
            ],
            indent=2,
        )
        return self.build_prompt(detections_json)

    def _parse_response(self, response: str, num_detections: int) -> list[Assessment]:
        """Parse raw VLM output into validated assessment objects."""
        strategies: list[str] = [response]

        markdown_match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, flags=re.DOTALL | re.IGNORECASE)
        if markdown_match:
            strategies.append(markdown_match.group(1))

        try:
            from json_repair import repair_json
        except ImportError:
            repair_json = None

        parse_errors: list[str] = []

        for candidate in strategies:
            stripped = candidate.strip()
            if not stripped:
                continue

            for label, serialized in self._json_attempts(stripped, repair_json):
                try:
                    payload = json.loads(serialized)
                except json.JSONDecodeError as exc:
                    parse_errors.append(f"{label}: {exc}")
                    continue

                assessments_payload = _extract_assessment_payloads(payload)
                if not assessments_payload:
                    parse_errors.append(f"{label}: no assessments found")
                    continue

                assessments = [
                    self._validate_assessment(_build_assessment(item, fallback_detection_id=index))
                    for index, item in enumerate(assessments_payload)
                ]

                if len(assessments) != num_detections:
                    logger.warning(
                        "VLM returned %s assessments for %s detections; using parsed subset.",
                        len(assessments),
                        num_detections,
                    )
                return assessments

        raise ParseError(f"Unable to parse VLM response. Attempts: {' | '.join(parse_errors)}")

    def _retry_single_detection(self, image: np.ndarray, detection: "Detection") -> Assessment:
        """Retry assessment for a single detection using a simplified prompt."""
        annotated_image = self._draw_boxes_on_image(image, [detection])
        prompt = self.retry_prompt_template.replace("{detection_id}", str(detection.detection_id))
        prompt = prompt.replace("{bbox}", str([int(value) for value in detection.bbox]))
        prompt = prompt.replace("{class_name}", detection.class_name)
        prompt = prompt.replace("{confidence:.2f}", f"{detection.confidence:.2f}")
        prompt = prompt.replace("{confidence}", f"{detection.confidence}")

        last_error: Exception | None = None
        attempts = max(1, self.max_retries)
        for _ in range(attempts):
            try:
                response = self._run_qwen_inference(
                    pil_image=annotated_image,
                    prompt_text=prompt,
                    max_new_tokens=256,
                )
                assessments = self._parse_response(response, num_detections=1)
                if assessments:
                    assessment = assessments[0]
                    if assessment.detection_id != detection.detection_id:
                        assessment = Assessment(
                            detection_id=detection.detection_id,
                            quality_score=assessment.quality_score,
                            scene_complexity=assessment.scene_complexity,
                            object_relative_size=assessment.object_relative_size,
                            is_false_positive=assessment.is_false_positive,
                            reasoning=assessment.reasoning,
                        )
                    return self._validate_assessment(assessment)
            except Exception as exc:
                last_error = exc

        raise ParseError(f"Single-detection retry failed: {last_error}") from last_error

    def assess_detections_batch(
        self,
        images: list[np.ndarray],
        detections_per_image: list[list["Detection"]],
    ) -> list[list[Assessment]]:
        """Assess detections for multiple images sequentially with progress logging."""
        if len(images) != len(detections_per_image):
            raise ValueError("images and detections_per_image must have the same length")

        results: list[list[Assessment]] = []
        iterator = zip(images, detections_per_image, strict=True)
        for index, (image, detections) in enumerate(
            tqdm(iterator, total=len(images), desc="Assessing detections with VLM")
        ):
            start_time = time.perf_counter()
            assessments = self.assess_detections(image=image, detections=detections)
            elapsed = time.perf_counter() - start_time
            logger.info("Assessed image %s/%s in %.2fs", index + 1, len(images), elapsed)
            results.append(assessments)
        return results

    def annotate_image(self, image: np.ndarray, detections: Sequence["Detection"]) -> np.ndarray:
        """Compatibility wrapper returning the annotated image as a NumPy array."""
        return np.asarray(self._draw_boxes_on_image(image, list(detections)), dtype=np.uint8)

    def build_prompt(self, detections_json: str) -> str:
        """Compatibility wrapper that injects detection JSON into the prompt template."""
        return self.prompt_template.replace("{detections_json}", detections_json)

    def judge_detections(self, image: np.ndarray, detections: Sequence["Detection"]) -> VLMJudgmentBatch:
        """Compatibility wrapper returning the assessments in a batch container."""
        detections_list = list(detections)
        prompt = self._build_prompt(detections_list)
        assessments = self.assess_detections(image=image, detections=detections_list)
        return VLMJudgmentBatch(assessments=assessments, prompt=prompt)

    def judge_single_detection(self, image: np.ndarray, detection: "Detection") -> Assessment:
        """Compatibility wrapper for single-detection retry assessment."""
        return self._retry_single_detection(image=image, detection=detection)

    def parse_response(self, response_text: str, num_detections: int = 1) -> VLMJudgmentBatch:
        """Compatibility wrapper around the private parsing implementation."""
        return VLMJudgmentBatch(
            assessments=self._parse_response(response_text, num_detections=num_detections),
            raw_response=response_text,
        )

    def _run_qwen_inference(self, pil_image: Image.Image, prompt_text: str, max_new_tokens: int) -> str:
        """Run Qwen2.5-VL on one image and prompt using the required chat format."""
        self.load_model()
        assert self.model is not None
        assert self.processor is not None

        torch = _require_torch()
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise ImportError("qwen-vl-utils is required for Qwen2.5-VL inference.") from exc

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        model_device = getattr(self.model, "device", None)
        if model_device is None:
            try:
                model_device = next(self.model.parameters()).device
            except StopIteration:
                model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        inputs = inputs.to(model_device)

        try:
            with torch.no_grad():
                generate_kwargs: dict[str, Any] = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                }
                # Only pass temperature when actually sampling; greedy decoding ignores it
                # and some transformers versions raise ValueError when temperature=0.0 + do_sample=False
                if self.temperature > 0.0:
                    generate_kwargs["do_sample"] = True
                    generate_kwargs["temperature"] = self.temperature
                    generate_kwargs["top_p"] = self.top_p
                output_ids = self.model.generate(**inputs, **generate_kwargs)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            logger.warning(
                "VLM inference hit CUDA OOM; clearing cache and retrying. "
                "If this persists, unload detector/XAI modules before calling the judge."
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            with torch.no_grad():
                generate_kwargs: dict[str, Any] = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                }
                # Only pass temperature when actually sampling; greedy decoding ignores it
                # and some transformers versions raise ValueError when temperature=0.0 + do_sample=False
                if self.temperature > 0.0:
                    generate_kwargs["do_sample"] = True
                    generate_kwargs["temperature"] = self.temperature
                    generate_kwargs["top_p"] = self.top_p
                output_ids = self.model.generate(**inputs, **generate_kwargs)

        generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return str(response).strip()

    def _json_attempts(self, candidate: str, repair_json: Any | None) -> list[tuple[str, str]]:
        """Return candidate JSON strings in priority order."""
        attempts = [("raw", candidate)]
        if repair_json is not None:
            try:
                repaired = repair_json(candidate)
            except Exception as exc:
                logger.debug("json_repair failed on candidate: %s", exc)
            else:
                if isinstance(repaired, str):
                    attempts.append(("repaired", repaired))
                elif repaired is not None:
                    attempts.append(("repaired-serialized", json.dumps(repaired)))
        attempts.append(("basic-repair", _basic_json_repair(candidate)))
        return attempts

    def _validate_assessment(self, assessment: Assessment) -> Assessment:
        """Normalize and clamp parsed assessments into project-valid values."""
        return Assessment(
            detection_id=int(assessment.detection_id),
            quality_score=_clamp_score(assessment.quality_score),
            scene_complexity=_normalize_choice(
                assessment.scene_complexity,
                _VALID_SCENE_COMPLEXITY,
                default="medium",
            ),
            object_relative_size=_normalize_choice(
                assessment.object_relative_size,
                _VALID_OBJECT_SIZES,
                default="medium",
            ),
            is_false_positive=bool(assessment.is_false_positive),
            reasoning=str(assessment.reasoning).strip(),
        )


__all__ = [
    "Assessment",
    "ParseError",
    "VLMAssessment",
    "VLMJudge",
    "VLMJudgmentBatch",
]
