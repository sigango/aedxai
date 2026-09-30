"""Run fixed-method AED-XAI baselines without feedback-loop adaptation."""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import DetectorWrapper
from src.evaluator import AutoEvaluator, EvalResult
from src.provenance import resolve_image_paths, write_manifest
from src.utils import load_image, set_seed, setup_logging
from src.xai_methods import CAM_METHODS, base_method_name, get_explainer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BaselineRunConfig:
    """Configuration bundle for one fixed-XAI baseline run."""

    images_dir: Path
    num_images: int
    output: Path
    xai_method: str
    seed: int
    log_level: str
    detector_model: str
    detector_config: Path
    xai_config: Path
    eval_config: Path
    image_list: Path | None = None
    split: str | None = None


def _as_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root unless already absolute."""
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _mean_metric(evaluation_results: Iterable[EvalResult], attribute: str) -> float:
    """Return the mean of an EvalResult attribute, or 0.0 for empty inputs."""
    values = [float(getattr(result, attribute)) for result in evaluation_results]
    return float(np.mean(values)) if values else 0.0


def _row_from_eval_results(
    image_path: Path,
    num_detections: int,
    evaluation_results: list[EvalResult],
    has_error: bool,
    explain_time: float = 0.0,
    num_fallback: int = 0,
) -> dict[str, Any]:
    """Create one baseline CSV row from per-detection EvalResults."""
    composite_score = _mean_metric(evaluation_results, "composite")
    return {
        "image_path": image_path.name,
        "num_detections": int(num_detections),
        "composite_score": composite_score,
        "iterations": 1,
        "converged": False,
        "mean_pg": _mean_metric(evaluation_results, "pg"),
        "mean_oa": _mean_metric(evaluation_results, "oa"),
        "mean_ebpg": _mean_metric(evaluation_results, "ebpg"),
        "mean_sparsity": _mean_metric(evaluation_results, "sparsity"),
        "mean_insertion_auc": _mean_metric(evaluation_results, "insertion_auc"),
        "mean_deletion_auc": _mean_metric(evaluation_results, "deletion_auc"),
        "explain_time": float(explain_time),
        "num_fallback": int(num_fallback),
        "total_time": float(sum(float(result.computation_time) for result in evaluation_results)),
        "has_error": bool(has_error),
    }


def summarize_results(dataframe: pd.DataFrame, title: str = "Baseline Summary") -> dict[str, Any]:
    """Compute and print aggregate metrics for a baseline dataframe."""
    total = int(len(dataframe))
    summary: dict[str, Any] = {"count": total}
    metric_columns = [
        "composite_score",
        "mean_pg",
        "mean_oa",
        "mean_sparsity",
        "iterations",
        "explain_time",
        "total_time",
    ]

    print(f"\n=== {title} ===")
    print(f"Images processed: {total}")
    for column in metric_columns:
        mean = float(dataframe[column].mean()) if total else 0.0
        std = float(dataframe[column].std(ddof=0)) if total else 0.0
        summary[f"{column}_mean"] = mean
        summary[f"{column}_std"] = std
        print(f"{column:<20}: {mean:.4f} ± {std:.4f}")

    num_converged = int(dataframe["converged"].sum()) if total else 0
    num_errors = int(dataframe["has_error"].sum()) if total else 0
    summary["num_converged"] = num_converged
    summary["num_errors"] = num_errors
    summary["converged_fraction"] = float(num_converged / total) if total else 0.0
    summary["error_fraction"] = float(num_errors / total) if total else 0.0
    print(f"Converged        : {num_converged}/{total}")
    print(f"Errors           : {num_errors}/{total}")
    return summary


def _load_xai_method_config(path: Path, method_name: str) -> dict[str, Any]:
    """Load the method-specific XAI YAML section."""
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    methods = raw.get("xai", {}).get("methods", {})
    section_name = base_method_name(method_name)
    if section_name not in methods:
        if section_name in {"bbox_prior", "center_prior", "random"}:
            return {}
        raise KeyError(f"XAI method '{method_name}' not found in {path}")
    return dict(methods[section_name])


def run_baseline(config: BaselineRunConfig) -> pd.DataFrame:
    """Run one no-feedback fixed-XAI baseline and write its CSV output."""
    set_seed(config.seed)
    setup_logging(config.log_level)
    config.output.mkdir(parents=True, exist_ok=True)

    image_paths = resolve_image_paths(
        config.images_dir,
        image_list=config.image_list,
        split=config.split,
        num_images=config.num_images,
        extensions=(".jpg",),
    )
    if not image_paths:
        raise FileNotFoundError(f"No .jpg images found in {config.images_dir}")

    detector = DetectorWrapper(config.detector_model, config_path=str(config.detector_config))
    detector.load_model()
    method_config = _load_xai_method_config(config.xai_config, config.xai_method)
    explainer = get_explainer(config.xai_method, method_config)
    evaluator = AutoEvaluator(str(config.eval_config))
    target_layer = detector.get_target_layer() if config.xai_method in CAM_METHODS else None
    write_manifest(
        config.output,
        entry_point="scripts/run_baseline.py",
        args={key: getattr(config, key) for key in config.__slots__},
        config_paths={"detector": config.detector_config, "xai": config.xai_config, "eval": config.eval_config},
        image_names=[path.name for path in image_paths],
        file_name=f"manifest_{config.detector_model}_{config.xai_method}.json",
    )

    rows: list[dict[str, Any]] = []
    try:
        for image_path in tqdm(image_paths, desc=f"Baseline {config.xai_method}", unit="image"):
            try:
                image = load_image(str(image_path))
                detections = detector.detect(image)
                model = detector.get_model()
                eval_results: list[EvalResult] = []
                explain_time = 0.0
                num_fallback = 0

                for detection in detections:
                    saliency = explainer.explain(model, image, detection, target_layer)
                    explain_time += float(getattr(saliency, "computation_time", 0.0))
                    num_fallback += int(bool(getattr(saliency, "is_fallback", False)))
                    eval_result = evaluator.evaluate_all(
                        saliency_map=saliency,
                        bbox=detection.bbox,
                        model=model,
                        image=image,
                        detection=detection,
                    )
                    eval_results.append(eval_result)

                rows.append(
                    _row_from_eval_results(
                        image_path=image_path,
                        num_detections=len(detections),
                        evaluation_results=eval_results,
                        has_error=False,
                        explain_time=explain_time,
                        num_fallback=num_fallback,
                    )
                )
            except Exception as exc:
                logger.warning("Baseline failed for image %s: %s", image_path, exc, exc_info=True)
                rows.append(
                    _row_from_eval_results(
                        image_path=image_path,
                        num_detections=0,
                        evaluation_results=[],
                        has_error=True,
                    )
                )
    finally:
        detector.unload_model()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    dataframe = pd.DataFrame(rows)
    dataframe.insert(0, "detector", config.detector_model)
    if config.detector_model == "yolox-s":
        output_path = config.output / f"baseline_{config.xai_method}_results.csv"
    else:
        output_path = config.output / f"baseline_{config.detector_model}_{config.xai_method}_results.csv"
    dataframe.to_csv(output_path, index=False)
    logger.info("Saved baseline results to %s", output_path)
    summarize_results(dataframe, title=f"Baseline {config.xai_method} Summary")
    return dataframe


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for fixed-method baselines."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=str, default="data/coco/val2017")
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--output", type=str, default="results/baseline")
    parser.add_argument(
        "--xai-method",
        type=str,
        default="gradcam",
        choices=["gradcam", "gcame", "dclose", "lime", "gcame_raw", "gradcam_raw", "bbox_prior", "center_prior", "random"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--detector-model",
        type=str,
        default="yolox-s",
        choices=["yolox-s", "fasterrcnn_resnet50_fpn_v2"],
        help="Detector family to evaluate for this fixed-method baseline.",
    )
    parser.add_argument("--detector-config", type=str, default="config/detector_config.yaml")
    parser.add_argument("--xai-config", type=str, default="config/xai_config.yaml")
    parser.add_argument("--eval-config", type=str, default="config/eval_config.yaml")
    parser.add_argument("--image-list", type=str, default=None, help="Manifest from scripts/make_manifests.py.")
    parser.add_argument("--split", type=str, default=None, help="Split inside --image-list (e.g. test).")
    return parser


def main() -> None:
    """Parse arguments and run the selected baseline."""
    args = build_arg_parser().parse_args()
    config = BaselineRunConfig(
        images_dir=_as_project_path(args.images_dir),
        num_images=int(args.num_images),
        output=_as_project_path(args.output),
        xai_method=str(args.xai_method),
        seed=int(args.seed),
        log_level=str(args.log_level),
        detector_model=str(args.detector_model),
        detector_config=_as_project_path(args.detector_config),
        xai_config=_as_project_path(args.xai_config),
        eval_config=_as_project_path(args.eval_config),
        image_list=_as_project_path(args.image_list) if args.image_list else None,
        split=args.split,
    )
    run_baseline(config)


if __name__ == "__main__":
    main()
