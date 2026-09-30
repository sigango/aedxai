"""Run the full AED-XAI pipeline on a COCO image subset and export results."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import AEDXAIPipeline, PipelineResult
from src.provenance import resolve_image_paths, write_manifest
from src.utils import set_seed, setup_logging
from src.xai_selector import XAISelector

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ExperimentRunConfig:
    """Configuration for one full AED-XAI pipeline run."""

    images_dir: Path
    num_images: int
    output: Path
    seed: int
    selector_model: Path | None
    log_level: str
    detector_model: str | None = None
    detector_config: Path = Path("config/detector_config.yaml")
    vlm_config: Path = Path("config/vlm_config.yaml")
    xai_config: Path = Path("config/xai_config.yaml")
    eval_config: Path = Path("config/eval_config.yaml")
    image_list: Path | None = None
    split: str | None = None
    feedback_overrides: dict[str, Any] | None = None
    use_vlm: bool = True


def _as_project_path(path: str | Path) -> Path:
    """Resolve a CLI path relative to the project root unless already absolute."""
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _mean_metric(evaluation_results: Iterable[Any], attribute: str) -> float:
    """Return the mean of an EvalResult attribute, or 0.0 for empty inputs."""
    values = [float(getattr(result, attribute)) for result in evaluation_results]
    return float(np.mean(values)) if values else 0.0


def pipeline_result_to_row(result: PipelineResult) -> dict[str, Any]:
    """Flatten a PipelineResult into one CSV row."""
    eval_results = list(result.evaluation_results)
    has_error = "error" in result.metadata

    method_counts: dict[str, int] = {}
    for reasoning in result.selector_reasoning.values():
        method_name = str(reasoning.get("method", "unknown"))
        method_counts[method_name] = method_counts.get(method_name, 0) + 1

    return {
        "image_path": Path(result.image_path).name,
        "num_detections": int(len(result.detections)),
        "composite_score": float(result.composite_score) if result.composite_score is not None else 0.0,
        "iterations": int(result.metadata.get("iterations", 0 if has_error else 0)),
        "converged": bool(result.metadata.get("converged", False)),
        "escalations": int(result.metadata.get("escalations", 0)),
        "selected_methods": json.dumps(method_counts, sort_keys=True),
        "mean_pg": _mean_metric(eval_results, "pg"),
        "mean_oa": _mean_metric(eval_results, "oa"),
        "mean_ebpg": _mean_metric(eval_results, "ebpg"),
        "mean_sparsity": _mean_metric(eval_results, "sparsity"),
        "mean_insertion_auc": _mean_metric(eval_results, "insertion_auc"),
        "mean_deletion_auc": _mean_metric(eval_results, "deletion_auc"),
        "explain_time": float(
            sum(float(saliency.computation_time) for saliency in result.saliency_maps)
        ),
        "total_time": float(sum(float(result.computation_time) for result in eval_results)),
        "has_error": has_error,
    }


def pipeline_result_to_detection_rows(result: PipelineResult) -> list[dict[str, Any]]:
    """One row per final detection: chosen method, escalation outcome, metrics, fallback flag."""
    rows = []
    for detection, saliency, evaluation in zip(result.detections, result.saliency_maps, result.evaluation_results):
        reasoning = dict(result.selector_reasoning.get(detection.detection_id, {}))
        rows.append(
            {
                "image_path": Path(result.image_path).name,
                "det_id": int(detection.detection_id),
                "class_id": int(detection.class_id),
                "confidence": float(detection.confidence),
                "relative_size": detection.relative_size,
                "bbox": json.dumps([int(value) for value in detection.bbox]),
                "selected_method": reasoning.get("method"),
                "selector_source": reasoning.get("source"),
                "final_method": getattr(saliency, "method_name", None),
                "escalated": bool(reasoning.get("escalated", False)),
                "escalation_kept": reasoning.get("escalation_kept"),
                "is_fallback": bool(getattr(saliency, "is_fallback", False)),
                "pg": float(evaluation.pg),
                "ebpg": float(evaluation.ebpg),
                "oa": float(evaluation.oa),
                "sparsity": float(evaluation.sparsity),
                "composite": float(evaluation.composite),
                "explain_time": float(getattr(saliency, "computation_time", 0.0)),
                "eval_time": float(evaluation.computation_time),
            }
        )
    return rows


def summarize_results(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Compute and print aggregate metrics for an experiment dataframe."""
    total = int(len(dataframe))
    summary: dict[str, Any] = {"count": total}
    metric_columns = [
        "composite_score",
        "mean_pg",
        "mean_oa",
        "mean_sparsity",
        "iterations",
        "escalations",
        "explain_time",
        "total_time",
    ]

    print("\n=== AED-XAI Summary ===")
    print(f"Images processed: {total}")
    for column in metric_columns:
        if column not in dataframe:
            continue
        mean = float(dataframe[column].mean()) if total else 0.0
        std = float(dataframe[column].std(ddof=0)) if total else 0.0
        summary[f"{column}_mean"] = mean
        summary[f"{column}_std"] = std
        print(f"{column:<20}: {mean:.4f} ± {std:.4f}")

    num_converged = int(dataframe["converged"].sum()) if total and "converged" in dataframe else 0
    num_errors = int(dataframe["has_error"].sum()) if total and "has_error" in dataframe else 0
    summary["num_converged"] = num_converged
    summary["num_errors"] = num_errors
    summary["converged_fraction"] = float(num_converged / total) if total else 0.0
    summary["error_fraction"] = float(num_errors / total) if total else 0.0
    print(f"Converged        : {num_converged}/{total}")
    print(f"Errors           : {num_errors}/{total}")
    return summary


def run_experiment(config: ExperimentRunConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run AED-XAI on the configured image subset and persist CSV/JSON outputs."""
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

    logger.info("Running AED-XAI on %d images from %s", len(image_paths), config.images_dir)
    pipeline = AEDXAIPipeline(
        detector_config_path=str(config.detector_config),
        vlm_config_path=str(config.vlm_config),
        xai_config_path=str(config.xai_config),
        eval_config_path=str(config.eval_config),
        detector_model_name=config.detector_model,
        feedback_overrides=config.feedback_overrides,
        use_vlm=config.use_vlm,
    )

    try:
        pipeline.setup()
        if config.selector_model is not None and config.selector_model.exists():
            pipeline.xai_selector = XAISelector(model_path=str(config.selector_model))
            logger.info("Using selector model from %s", config.selector_model)
        elif config.selector_model is not None:
            logger.warning("Selector model not found at %s; using pipeline fallback selector.", config.selector_model)

        selector = pipeline.xai_selector
        detector_name = pipeline.detector.model_name if pipeline.detector is not None else str(config.detector_model)
        default_checkpoint = PROJECT_ROOT / "data" / "checkpoints" / f"xai_selector_{detector_name}.pth"
        write_manifest(
            config.output,
            entry_point="scripts/run_experiments.py",
            args={key: getattr(config, key) for key in config.__slots__},
            config_paths={
                "detector": config.detector_config,
                "vlm": config.vlm_config,
                "xai": config.xai_config,
                "eval": config.eval_config,
            },
            image_names=[path.name for path in image_paths],
            checkpoints={
                "selector": config.selector_model
                if config.selector_model is not None and config.selector_model.exists()
                else (default_checkpoint if default_checkpoint.exists() else None)
            },
            extra={
                "selector_is_trained": bool(getattr(selector, "is_trained", False)),
                "effective_feedback_config": pipeline.feedback_config(),
                "use_vlm": config.use_vlm,
            },
        )

        results: list[PipelineResult] = []
        for image_path in tqdm(image_paths, desc="AED-XAI images", unit="image"):
            try:
                results.append(pipeline.run_on_image(str(image_path)))
            except Exception as exc:
                logger.warning("Image failed and will be recorded as an error: %s", image_path, exc_info=True)
                results.append(
                    PipelineResult(
                        image_path=str(image_path),
                        detections=[],
                        evaluation_results=[],
                        composite_score=None,
                        metadata={"error": str(exc), "iterations": 0, "converged": False},
                    )
                )

        dataframe = pd.DataFrame([pipeline_result_to_row(result) for result in results])
        detector_name = (
            config.detector_model
            or (pipeline.detector.model_name if pipeline.detector is not None else "unknown")
        )
        dataframe.insert(0, "detector", detector_name)
        results_path = config.output / "results.csv"
        dataframe.to_csv(results_path, index=False)
        logger.info("Saved per-image results to %s", results_path)
        detections_path = config.output / "detections.csv"
        pd.DataFrame(
            [row for result in results for row in pipeline_result_to_detection_rows(result)]
        ).to_csv(detections_path, index=False)
        logger.info("Saved per-detection results to %s", detections_path)

        summary = summarize_results(dataframe)
        summary_path = config.output / "summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        logger.info("Saved summary to %s", summary_path)
        return dataframe, summary
    finally:
        pipeline.shutdown()


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for full AED-XAI experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=str, default="data/coco/val2017")
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--output", type=str, default="results/aedxai")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--detector-model",
        type=str,
        default=None,
        choices=["yolox-s", "fasterrcnn_resnet50_fpn_v2"],
        help="Optional detector override. If omitted, uses detector_config.yaml primary detector.",
    )
    parser.add_argument(
        "--selector-model",
        type=str,
        default="",
        help=(
            "Optional selector checkpoint override. If omitted, AEDXAIPipeline "
            "auto-loads data/checkpoints/xai_selector_<detector>.pth when present."
        ),
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--image-list", type=str, default=None, help="Manifest from scripts/make_manifests.py.")
    parser.add_argument("--split", type=str, default=None, help="Split inside --image-list (e.g. test).")
    parser.add_argument("--max-iterations", type=int, default=None, help="Override feedback.max_iterations (1 = off).")
    parser.add_argument(
        "--fixed-method",
        type=str,
        default=None,
        help="Bypass the selector and use this method for every detection (e.g. gcame).",
    )
    parser.add_argument(
        "--escalation",
        choices=["config", "on", "off"],
        default="config",
        help="Override feedback.escalation.enabled.",
    )
    parser.add_argument("--no-vlm", action="store_true", help="Do not load the VLM (scene complexity from detector).")
    return parser


def main() -> None:
    """Parse CLI arguments and run the full AED-XAI pipeline."""
    args = build_arg_parser().parse_args()
    config = ExperimentRunConfig(
        images_dir=_as_project_path(args.images_dir),
        num_images=int(args.num_images),
        output=_as_project_path(args.output),
        seed=int(args.seed),
        selector_model=_as_project_path(args.selector_model) if args.selector_model else None,
        log_level=str(args.log_level),
        detector_model=args.detector_model,
        detector_config=_as_project_path("config/detector_config.yaml"),
        vlm_config=_as_project_path("config/vlm_config.yaml"),
        xai_config=_as_project_path("config/xai_config.yaml"),
        eval_config=_as_project_path("config/eval_config.yaml"),
        image_list=_as_project_path(args.image_list) if args.image_list else None,
        split=args.split,
        feedback_overrides=_feedback_overrides(args),
        use_vlm=not args.no_vlm,
    )
    run_experiment(config)


def _feedback_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Translate CLI flags into feedback-config overrides for AEDXAIPipeline."""
    overrides: dict[str, Any] = {}
    if args.max_iterations is not None:
        overrides["max_iterations"] = int(args.max_iterations)
    if args.fixed_method:
        overrides["fixed_method"] = str(args.fixed_method)
    if args.escalation != "config":
        overrides["escalation"] = {"enabled": args.escalation == "on"}
    return overrides


if __name__ == "__main__":
    main()
