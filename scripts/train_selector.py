"""Generate selector training data and train the AED-XAI selector MLP."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

from src.detector import Detection, DetectorWrapper, compute_scene_complexity
from src.utils import load_image, set_seed, setup_logging
from src.xai_methods import get_explainer
from src.xai_selector import (
    FEATURE_COLUMNS,
    METHOD_NAMES,
    METHOD_TO_IDX,
    XAISelector,
    compute_image_entropy,
    encode_relative_size,
    encode_scene_complexity,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SelectorTrainingConfig:
    """Configuration bundle for selector training data generation and fitting."""

    detector_config: str = "config/detector_config.yaml"
    xai_config: str = "config/xai_config.yaml"
    eval_config: str = "config/eval_config.yaml"
    annotations_path: str = "data/coco/annotations/instances_val2017.json"
    images_dir: str = "data/coco/val2017"
    output_csv: str = "results/xai_selector_training_data.csv"
    progress_csv: str = "results/xai_selector_training_progress.csv"
    output_model_path: str = "data/checkpoints/xai_selector.pth"
    target_detections: int = 2000
    max_images: int = 400
    max_detections_per_image: int = 5
    checkpoint_every: int = 100
    resume: bool = False
    oracle_mode: bool = True
    sample_seed: int = 42
    methods: list[str] = field(default_factory=lambda: list(METHOD_NAMES))


class _FallbackAutoEvaluator:
    """Lightweight ODExAI-inspired evaluator fallback for selector data generation."""

    def evaluate_all(
        self,
        saliency: Any,
        bbox: list[int],
        model: Any,
        image: np.ndarray,
        detection: Detection,
    ) -> dict[str, float]:
        del model, image, detection
        saliency_map = getattr(saliency, "map", saliency)
        saliency_map = np.asarray(saliency_map, dtype=np.float32)
        x1, y1, x2, y2 = [int(value) for value in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(saliency_map.shape[1], x2)
        y2 = min(saliency_map.shape[0], y2)

        inside = saliency_map[y1:y2, x1:x2]
        total_mass = float(saliency_map.sum()) + 1e-8
        inside_mass = float(inside.sum()) if inside.size else 0.0

        max_index = np.unravel_index(int(np.argmax(saliency_map)), saliency_map.shape)
        pg = float(y1 <= max_index[0] < y2 and x1 <= max_index[1] < x2)
        ebpg = inside_mass / total_mass
        bbox_mean = float(inside.mean()) if inside.size else 0.0
        full_mean = float(saliency_map.mean()) + 1e-8
        oa = float(np.clip(0.6 * ebpg + 0.4 * (bbox_mean / full_mean), 0.0, 1.0))
        sparse_fraction = float(np.mean(saliency_map > saliency_map.mean()))
        sparsity = float(np.clip(1.0 - sparse_fraction, 0.0, 1.0))
        computation_time = float(getattr(saliency, "computation_time", 0.0))

        return {
            "pg": pg,
            "ebpg": ebpg,
            "oa": oa,
            "sparsity": sparsity,
            "computation_time": computation_time,
        }


class SelectorTrainingRunner:
    """Generate oracle-labeled training data and fit the AED-XAI selector MLP.

    Oracle Labeling Protocol (Section 3.3 of the paper):
        For each detection in the training corpus, ALL four XAI methods
        (GradCAM, G-CAME, D-CLOSE, LIME) are exhaustively evaluated.
        The method achieving the highest composite score
        ((PG + OA + Sparsity) / 3, normalised per metric
        across the dataset) is assigned as the oracle label.

        The selector MLP is then trained to predict this label from
        pre-explanation context features only:
            - class_id (COCO category index)
            - confidence (detector score)
            - relative_size_encoded {0=small, 1=medium, 2=large}
            - scene_complexity_encoded {0=simple, 1=moderate, 2=complex}
            - num_detections (scene crowdedness)
            - bbox_aspect_ratio (width / height)
            - image_entropy (local patch complexity)

        At inference, the selector predicts from these same features
        in <1 ms, enabling single-shot method selection with no
        additional forward passes.
    """

    def __init__(self, config: SelectorTrainingConfig) -> None:
        self.config = config
        self.detector_config = self._load_yaml_section(config.detector_config, "detector")
        self.xai_config = self._load_yaml_section(config.xai_config, "xai")
        self.eval_config = self._load_yaml_section(config.eval_config, "evaluation", required=False)

    def run(self) -> dict[str, Any]:
        """Generate selector training data, then train the selector if complete."""
        set_seed(self.config.sample_seed)
        if self.config.oracle_mode:
            logger.info(
                "Oracle labeling mode enabled: all configured XAI methods will be "
                "run on every selected detection to generate offline oracle labels."
            )
        progress_df = self._generate_training_data()
        final_df = self._finalize_training_dataframe(progress_df)

        final_df.to_csv(self.config.output_csv, index=False)
        logger.info("Saved selector training data to %s", self.config.output_csv)

        metric_columns = self._required_metric_columns()
        if not all(column in final_df.columns for column in metric_columns):
            logger.warning(
                "Training data is incomplete for some methods. Saved progress only; "
                "resume later with the missing methods to train the selector."
            )
            return {"status": "incomplete", "output_csv": self.config.output_csv}

        selector = XAISelector()
        training_metrics = selector.train(self.config.output_csv)
        selector.save_model(self.config.output_model_path)
        logger.info("Saved selector model to %s", self.config.output_model_path)
        return training_metrics

    def _generate_training_data(self) -> pd.DataFrame:
        """Generate raw per-detection metric rows with checkpointing support."""
        progress_path = Path(self.config.progress_csv)
        progress_path.parent.mkdir(parents=True, exist_ok=True)

        progress_df = pd.read_csv(progress_path) if self.config.resume and progress_path.exists() else pd.DataFrame()
        processed_keys = set(progress_df.get("sample_key", pd.Series(dtype=str)).astype(str).tolist())
        rows = progress_df.to_dict("records")

        detector = DetectorWrapper("yolox-s", config_path=self.config.detector_config)
        detector.load_model()
        model = detector.get_model()
        target_layer = detector.get_target_layer()
        explainers = self._build_explainers()
        evaluator = self._build_evaluator()
        coco = self._load_coco_annotations(self.config.annotations_path)
        sampled_image_ids = self._sample_diverse_image_ids(coco)

        new_count = 0
        total_collected = len(processed_keys)

        for image_id in tqdm(sampled_image_ids, desc="Generating selector data"):
            if total_collected >= self.config.target_detections:
                break

            image_info = coco["images_by_id"][image_id]
            image_path = Path(self.config.images_dir) / image_info["file_name"]
            if not image_path.exists():
                logger.warning("Skipping missing image: %s", image_path)
                continue

            image = load_image(str(image_path))
            detections = detector.detect(image)
            if not detections:
                continue

            scene_complexity = compute_scene_complexity(detections, detector.config)
            selected_detections = self._select_diverse_detections(detections)

            for detection in selected_detections:
                if total_collected >= self.config.target_detections:
                    break

                sample_key = self._sample_key(image_id, detection)
                if sample_key in processed_keys:
                    continue

                feature_row = self._build_feature_row(
                    image=image,
                    detection=detection,
                    scene_complexity=scene_complexity,
                    num_detections=len(detections),
                )
                feature_row["sample_key"] = sample_key

                for method_name, explainer in explainers.items():
                    saliency = explainer.explain(
                        model=model,
                        image=image,
                        detection=detection,
                        target_layer=target_layer if method_name in {"gradcam", "gcame"} else None,
                    )
                    metrics = evaluator.evaluate_all(
                        saliency=saliency,
                        bbox=detection.bbox,
                        model=model,
                        image=image,
                        detection=detection,
                    )

                    feature_row[f"pg_{method_name}"] = float(metrics["pg"])
                    feature_row[f"oa_{method_name}"] = float(metrics["oa"])
                    feature_row[f"sparsity_{method_name}"] = float(metrics["sparsity"])
                    feature_row[f"time_{method_name}"] = float(metrics["computation_time"])

                rows.append(feature_row)
                processed_keys.add(sample_key)
                new_count += 1
                total_collected += 1

                if new_count % self.config.checkpoint_every == 0:
                    checkpoint_df = pd.DataFrame(rows)
                    checkpoint_df.to_csv(progress_path, index=False)
                    logger.info("Checkpointed selector progress to %s (%d rows)", progress_path, len(checkpoint_df))

        progress_df = pd.DataFrame(rows)
        progress_df.to_csv(progress_path, index=False)
        logger.info("Saved selector progress to %s", progress_path)
        return progress_df

    def _build_explainers(self) -> dict[str, Any]:
        """Instantiate the configured XAI explainers for this run."""
        methods_config = dict(self.xai_config.get("methods", {}))
        explainers = {}
        for method_name in self.config.methods:
            if method_name not in methods_config:
                raise ValueError(f"Method '{method_name}' missing from xai_config.yaml")
            explainers[method_name] = get_explainer(method_name, methods_config[method_name])
        return explainers

    def _build_evaluator(self) -> Any:
        """Build the evaluator, preferring a real evaluator when available."""
        try:
            from src.evaluator import AutoEvaluator
        except Exception:
            logger.warning("src.evaluator.AutoEvaluator unavailable; using fallback evaluator.")
            return _FallbackAutoEvaluator()

        try:
            return AutoEvaluator(self.eval_config if self.eval_config else {})
        except Exception as exc:
            logger.warning("AutoEvaluator init failed (%s); using fallback evaluator.", exc)
            return _FallbackAutoEvaluator()

    def _build_feature_row(
        self,
        image: np.ndarray,
        detection: Detection,
        scene_complexity: str,
        num_detections: int,
    ) -> dict[str, Any]:
        """Compute the selector features for one detection."""
        x1, y1, x2, y2 = detection.bbox
        return {
            "class_id": int(detection.class_id),
            "confidence": float(detection.confidence),
            "relative_size_encoded": int(encode_relative_size(detection.relative_size)),
            "scene_complexity_encoded": int(encode_scene_complexity(scene_complexity)),
            "num_detections": int(num_detections),
            "bbox_aspect_ratio": float((x2 - x1) / max(y2 - y1, 1)),
            "image_entropy": float(compute_image_entropy(image, detection.bbox)),
        }

    def _finalize_training_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Normalize metrics across the dataset and assign best-method labels."""
        if dataframe.empty:
            raise ValueError("No selector training rows were generated.")

        dataframe = dataframe.copy()
        required_columns = self._required_metric_columns()
        present_columns = [column for column in required_columns if column in dataframe.columns]
        if not present_columns:
            return dataframe

        for metric_name in ("pg", "oa", "sparsity"):
            metric_columns = [f"{metric_name}_{method_name}" for method_name in METHOD_NAMES if f"{metric_name}_{method_name}" in dataframe.columns]
            values = dataframe[metric_columns].to_numpy(dtype=np.float32)
            metric_min = float(np.nanmin(values))
            metric_max = float(np.nanmax(values))
            scale = metric_max - metric_min + 1e-8
            for method_name in METHOD_NAMES:
                column = f"{metric_name}_{method_name}"
                if column in dataframe.columns:
                    dataframe[f"{metric_name}_norm_{method_name}"] = (dataframe[column] - metric_min) / scale

        available_methods = [
            method_name
            for method_name in METHOD_NAMES
            if all(
                column in dataframe.columns
                for column in (
                    f"pg_norm_{method_name}",
                    f"oa_norm_{method_name}",
                    f"sparsity_norm_{method_name}",
                )
            )
        ]

        if len(available_methods) == len(METHOD_NAMES):
            composites = []
            for method_name in METHOD_NAMES:
                composite = (
                    dataframe[f"pg_norm_{method_name}"]
                    + dataframe[f"oa_norm_{method_name}"]
                    + dataframe[f"sparsity_norm_{method_name}"]
                ) / 3.0
                dataframe[f"composite_{method_name}"] = composite
                composites.append(composite.to_numpy())

            # Log per-method mean composite score for diagnostics
            logger.info("=== PER-METHOD MEAN COMPOSITE SCORES ===")
            for method_name in METHOD_NAMES:
                col = f"composite_{method_name}"
                if col in dataframe.columns:
                    logger.info(
                        "  %-10s  mean_composite=%.4f  std=%.4f",
                        method_name,
                        float(dataframe[col].mean()),
                        float(dataframe[col].std()),
                    )

            composite_matrix = np.stack(composites, axis=1)
            best_indices = np.argmax(composite_matrix, axis=1)
            dataframe["best_method_label"] = best_indices.astype(np.int64)

            label_counts = pd.Series(best_indices).value_counts(normalize=True).sort_index()
            # --- Oracle label distribution (paper Table 3 data source) ---
            label_counts_abs = pd.Series(best_indices).value_counts().sort_index()
            logger.info(
                "=== ORACLE LABEL DISTRIBUTION (n=%d detections) ===",
                len(dataframe),
            )
            for method_name in METHOD_NAMES:
                idx = METHOD_TO_IDX[method_name]
                count = int(label_counts_abs.get(idx, 0))
                pct = float(label_counts.get(idx, 0.0)) * 100.0
                logger.info("  %-10s  idx=%d  count=%5d  pct=%5.1f%%", method_name, idx, count, pct)

            # Warn if any method is never selected (degenerate distribution)
            never_selected = [
                method_name for method_name in METHOD_NAMES
                if float(label_counts.get(METHOD_TO_IDX[method_name], 0.0)) == 0.0
            ]
            if never_selected:
                logger.warning(
                    "Methods NEVER selected as oracle best: %s. "
                    "Consider checking your composite weights or that these "
                    "methods are functioning correctly.",
                    never_selected,
                )
        else:
            logger.warning(
                "Only methods %s are present in progress data. "
                "Best-method labels will be deferred until all methods are available.",
                available_methods,
            )

        if "best_method_label" in dataframe.columns:
            final_columns = FEATURE_COLUMNS + [f"pg_{m}" for m in METHOD_NAMES if f"pg_{m}" in dataframe.columns]
            final_columns += [f"oa_{m}" for m in METHOD_NAMES if f"oa_{m}" in dataframe.columns]
            final_columns += [f"sparsity_{m}" for m in METHOD_NAMES if f"sparsity_{m}" in dataframe.columns]
            final_columns.append("best_method_label")
            return dataframe[final_columns]

        return dataframe

    def _load_coco_annotations(self, annotations_path: str) -> dict[str, Any]:
        """Load COCO annotations into lookup-friendly structures."""
        path = Path(annotations_path)
        if not path.exists():
            raise FileNotFoundError(f"COCO annotations not found: {path}")

        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)

        images_by_id = {int(image["id"]): image for image in raw.get("images", [])}
        categories_by_id = {int(category["id"]): category for category in raw.get("categories", [])}

        supercategory_to_image_ids: dict[str, set[int]] = defaultdict(set)
        for annotation in raw.get("annotations", []):
            image_id = int(annotation["image_id"])
            category = categories_by_id.get(int(annotation["category_id"]))
            if category is None:
                continue
            supercategory = str(category.get("supercategory", "unknown"))
            supercategory_to_image_ids[supercategory].add(image_id)

        return {
            "images_by_id": images_by_id,
            "supercategory_to_image_ids": supercategory_to_image_ids,
        }

    def _sample_diverse_image_ids(self, coco: Mapping[str, Any]) -> list[int]:
        """Sample a diverse image subset with minimum super-category coverage."""
        rng = np.random.default_rng(self.config.sample_seed)
        chosen: set[int] = set()

        for image_ids in coco["supercategory_to_image_ids"].values():
            candidates = sorted(image_ids)
            if not candidates:
                continue
            take = min(5, len(candidates))
            sampled = rng.choice(candidates, size=take, replace=False)
            chosen.update(int(value) for value in np.atleast_1d(sampled))

        all_ids = sorted(coco["images_by_id"].keys())
        remaining = [image_id for image_id in all_ids if image_id not in chosen]
        if len(chosen) < self.config.max_images and remaining:
            extra = min(self.config.max_images - len(chosen), len(remaining))
            sampled = rng.choice(remaining, size=extra, replace=False)
            chosen.update(int(value) for value in np.atleast_1d(sampled))

        return sorted(chosen)[: self.config.max_images]

    def _select_diverse_detections(self, detections: list[Detection]) -> list[Detection]:
        """Select up to K detections with rough diversity in size and confidence."""
        if len(detections) <= self.config.max_detections_per_image:
            return detections

        selected: list[Detection] = []
        seen_ids: set[int] = set()

        for size_name in ("small", "medium", "large"):
            candidates = [det for det in detections if det.relative_size == size_name]
            if candidates:
                candidate = max(candidates, key=lambda det: det.confidence)
                selected.append(candidate)
                seen_ids.add(candidate.detection_id)

        sorted_by_conf = sorted(detections, key=lambda det: det.confidence)
        for position in {0, len(sorted_by_conf) // 2, len(sorted_by_conf) - 1}:
            candidate = sorted_by_conf[position]
            if candidate.detection_id not in seen_ids:
                selected.append(candidate)
                seen_ids.add(candidate.detection_id)

        for candidate in sorted(detections, key=lambda det: det.confidence, reverse=True):
            if len(selected) >= self.config.max_detections_per_image:
                break
            if candidate.detection_id in seen_ids:
                continue
            selected.append(candidate)
            seen_ids.add(candidate.detection_id)

        return selected[: self.config.max_detections_per_image]

    @staticmethod
    def _load_yaml_section(path: str, section: str, required: bool = True) -> dict[str, Any]:
        """Load one named section from a YAML file."""
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if section not in raw:
            if required:
                raise KeyError(f"Missing '{section}' section in {path}")
            return {}
        return dict(raw[section])

    @staticmethod
    def _sample_key(image_id: int, detection: Detection) -> str:
        """Build a stable row key for resume/checkpoint support."""
        bbox = "-".join(str(int(value)) for value in detection.bbox)
        return f"{image_id}:{detection.class_id}:{bbox}"

    @staticmethod
    def _required_metric_columns() -> list[str]:
        """Return the required per-method metric columns for final labeling."""
        columns = []
        for metric_name in ("pg", "oa", "time"):
            for method_name in METHOD_NAMES:
                columns.append(f"{metric_name}_{method_name}")
        return columns


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for selector data generation and training."""
    parser = argparse.ArgumentParser(description="Generate data and train the AED-XAI selector.")
    parser.add_argument("--detector-config", default="config/detector_config.yaml")
    parser.add_argument("--xai-config", default="config/xai_config.yaml")
    parser.add_argument("--eval-config", default="config/eval_config.yaml")
    parser.add_argument("--annotations-path", default="data/coco/annotations/instances_val2017.json")
    parser.add_argument("--images-dir", default="data/coco/val2017")
    parser.add_argument("--output-csv", default="results/xai_selector_training_data.csv")
    parser.add_argument("--progress-csv", default="results/xai_selector_training_progress.csv")
    parser.add_argument("--output-model-path", default="data/checkpoints/xai_selector.pth")
    parser.add_argument("--target-detections", type=int, default=2000)
    parser.add_argument("--max-images", type=int, default=400)
    parser.add_argument("--max-detections-per-image", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--methods", default="gradcam,gcame,dclose,lime")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--oracle-mode",
        action="store_true",
        help=(
            "Enable oracle labeling mode: runs ALL configured XAI methods "
            "on every detection to generate offline oracle labels. "
            "This is the default and correct training procedure. "
            "Setting this flag makes the intent explicit and prints "
            "additional oracle-labeling diagnostics to the log."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    """Parse arguments and launch selector data generation and training."""
    parser = build_arg_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)

    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    config = SelectorTrainingConfig(
        detector_config=args.detector_config,
        xai_config=args.xai_config,
        eval_config=args.eval_config,
        annotations_path=args.annotations_path,
        images_dir=args.images_dir,
        output_csv=args.output_csv,
        progress_csv=args.progress_csv,
        output_model_path=args.output_model_path,
        target_detections=args.target_detections,
        max_images=args.max_images,
        max_detections_per_image=args.max_detections_per_image,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        oracle_mode=args.oracle_mode,
        sample_seed=args.sample_seed,
        methods=methods,
    )

    runner = SelectorTrainingRunner(config)
    metrics = runner.run()
    logger.info("Selector training run finished: %s", json.dumps(metrics, default=str))


if __name__ == "__main__":
    main()
