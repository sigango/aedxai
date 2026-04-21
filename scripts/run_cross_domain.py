"""Run AED-XAI cross-domain evaluation across multiple image datasets.

This script evaluates the full AED-XAI loop and/or fixed XAI baselines on
several image folders and exports a unified ``cross_domain.csv`` consumed by
``notebooks/05_results_visualization.ipynb``.

The evaluation intentionally does not require dataset ground-truth boxes.
AED-XAI's evaluator uses the detector's own boxes as pseudo-ground-truth for
annotation-free explanation metrics, so any folder of images can be used for
cross-domain stress testing.
"""

from __future__ import annotations

import argparse
import gc
import json
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
from src.pipeline import AEDXAIPipeline, PipelineResult
from src.utils import load_image, set_seed, setup_logging
from src.xai_methods import get_explainer
from src.xai_selector import XAISelector

logger = logging.getLogger(__name__)

XAI_METHODS = {"gradcam", "gcame", "dclose", "lime"}
SUPPORTED_METHODS = {"aedxai", *XAI_METHODS}
METHOD_DISPLAY = {
    "aedxai": "AED-XAI (Ours)",
    "gradcam": "GradCAM",
    "gcame": "G-CAME",
    "dclose": "D-CLOSE",
    "lime": "LIME",
}
DEFAULT_DATASETS = (
    "COCO=data/coco/val2017,"
    "VOC=data/voc/VOCdevkit/VOC2012/JPEGImages,"
    "BDD100K=data/bdd100k/images/100k/val,"
    "VisDrone=data/visdrone/VisDrone2019-DET-val/images,"
    "DOTA=data/dota/val/images,"
    "OpenImages=data/openimages/validation"
)


@dataclass(slots=True)
class DomainSpec:
    """One cross-domain dataset image source."""

    name: str
    images_dir: Path


@dataclass(slots=True)
class CrossDomainConfig:
    """Configuration for cross-domain AED-XAI evaluation."""

    domains: list[DomainSpec]
    methods: list[str]
    num_images: int
    output_dir: Path
    seed: int
    log_level: str
    image_exts: tuple[str, ...]
    recursive: bool
    skip_missing: bool
    detector_model: str
    selector_model: Path | None
    detector_config: Path
    vlm_config: Path
    xai_config: Path
    eval_config: Path


def _as_project_path(path: str | Path) -> Path:
    """Resolve a path relative to project root unless it is already absolute."""
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _parse_domains(value: str) -> list[DomainSpec]:
    """Parse NAME=PATH comma-separated dataset specifications."""
    domains: list[DomainSpec] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid dataset spec '{item}'. Expected NAME=PATH.")
        name, path = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid dataset spec '{item}': empty dataset name.")
        domains.append(DomainSpec(name=name, images_dir=_as_project_path(path.strip())))
    if not domains:
        raise ValueError("At least one dataset must be provided.")
    return domains


def _parse_methods(value: str) -> list[str]:
    """Parse comma-separated method names and validate them."""
    methods = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not methods:
        raise ValueError("At least one method must be provided.")
    unknown = [name for name in methods if name not in SUPPORTED_METHODS]
    if unknown:
        raise ValueError(f"Unknown methods {unknown}. Choose from {sorted(SUPPORTED_METHODS)}.")
    return methods


def _parse_exts(value: str) -> tuple[str, ...]:
    """Parse image extensions into a normalized tuple."""
    exts = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        exts.append(item if item.startswith(".") else f".{item}")
    return tuple(exts or [".jpg", ".jpeg", ".png"])


def _collect_images(domain: DomainSpec, exts: tuple[str, ...], recursive: bool, limit: int) -> list[Path]:
    """Collect sorted image paths for one domain."""
    iterator = domain.images_dir.rglob("*") if recursive else domain.images_dir.glob("*")
    paths = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in exts
    ]
    return sorted(paths)[:limit]


def _mean_metric(evaluation_results: Iterable[EvalResult], attribute: str) -> float:
    """Return the mean EvalResult attribute value, or NaN for empty results."""
    values = [float(getattr(result, attribute)) for result in evaluation_results]
    return float(np.mean(values)) if values else float("nan")


def _scene_complexity_from_pipeline(result: PipelineResult) -> str:
    """Return the majority VLM scene complexity label for one pipeline result."""
    values = [
        str(getattr(assessment, "scene_complexity", "unknown"))
        for assessment in getattr(result, "assessments", [])
        if getattr(assessment, "scene_complexity", None)
    ]
    return max(set(values), key=values.count) if values else "unknown"


def _empty_row(domain: str, method: str, image_path: Path, error: str) -> dict[str, Any]:
    """Create a failed per-image row with the common cross-domain schema."""
    return {
        "domain": domain,
        "method": METHOD_DISPLAY[method],
        "method_key": method,
        "image_id": image_path.name,
        "image_path": str(image_path),
        "num_detections": 0,
        "pg": float("nan"),
        "ebpg": float("nan"),
        "oa": float("nan"),
        "sparsity": float("nan"),
        "composite": float("nan"),
        "computation_time": 0.0,
        "iterations": 0,
        "converged": False,
        "scene_complexity": "unknown",
        "has_error": True,
        "error": error,
    }


def _row_from_pipeline_result(domain: str, result: PipelineResult) -> dict[str, Any]:
    """Flatten one AED-XAI PipelineResult into the cross-domain CSV schema."""
    eval_results = list(result.evaluation_results)
    return {
        "domain": domain,
        "method": METHOD_DISPLAY["aedxai"],
        "method_key": "aedxai",
        "image_id": Path(result.image_path).name,
        "image_path": result.image_path,
        "num_detections": int(len(result.detections)),
        "pg": _mean_metric(eval_results, "pg"),
        "ebpg": _mean_metric(eval_results, "ebpg"),
        "oa": _mean_metric(eval_results, "oa"),
        "sparsity": _mean_metric(eval_results, "sparsity"),
        "composite": float(result.composite_score) if result.composite_score is not None else float("nan"),
        "computation_time": float(sum(float(item.computation_time) for item in eval_results)),
        "iterations": int(result.metadata.get("iterations", 0)),
        "converged": bool(result.metadata.get("converged", False)),
        "scene_complexity": _scene_complexity_from_pipeline(result),
        "has_error": "error" in result.metadata,
        "error": str(result.metadata.get("error", "")),
    }


def _row_from_eval_results(
    domain: str,
    method: str,
    image_path: Path,
    num_detections: int,
    evaluation_results: list[EvalResult],
) -> dict[str, Any]:
    """Flatten fixed-method baseline EvalResults into the cross-domain schema."""
    return {
        "domain": domain,
        "method": METHOD_DISPLAY[method],
        "method_key": method,
        "image_id": image_path.name,
        "image_path": str(image_path),
        "num_detections": int(num_detections),
        "pg": _mean_metric(evaluation_results, "pg"),
        "ebpg": _mean_metric(evaluation_results, "ebpg"),
        "oa": _mean_metric(evaluation_results, "oa"),
        "sparsity": _mean_metric(evaluation_results, "sparsity"),
        "composite": _mean_metric(evaluation_results, "composite"),
        "computation_time": float(sum(float(item.computation_time) for item in evaluation_results)),
        "iterations": 1,
        "converged": False,
        "scene_complexity": "unknown",
        "has_error": False,
        "error": "",
    }


def _load_xai_config(path: Path) -> dict[str, Any]:
    """Load xai_config.yaml."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _run_aedxai(
    config: CrossDomainConfig,
    domain_images: dict[str, list[Path]],
) -> list[dict[str, Any]]:
    """Run full AED-XAI across all domains with one resident pipeline."""
    rows: list[dict[str, Any]] = []
    pipeline = AEDXAIPipeline(
        detector_config_path=str(config.detector_config),
        vlm_config_path=str(config.vlm_config),
        xai_config_path=str(config.xai_config),
        eval_config_path=str(config.eval_config),
        detector_model_name=config.detector_model,
    )
    try:
        pipeline.setup()
        if config.selector_model is not None:
            if config.selector_model.exists():
                pipeline.xai_selector = XAISelector(model_path=str(config.selector_model))
                logger.info("Using selector model from %s", config.selector_model)
            else:
                logger.warning("Selector model not found at %s; using pipeline fallback.", config.selector_model)

        for domain_name, image_paths in domain_images.items():
            for image_path in tqdm(image_paths, desc=f"AED-XAI {domain_name}", unit="image"):
                try:
                    result = pipeline.run_on_image(str(image_path))
                    rows.append(_row_from_pipeline_result(domain_name, result))
                except Exception as exc:
                    logger.warning("AED-XAI failed for %s/%s: %s", domain_name, image_path.name, exc, exc_info=True)
                    rows.append(_empty_row(domain_name, "aedxai", image_path, str(exc)))
    finally:
        pipeline.shutdown()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def _run_fixed_method(
    method: str,
    config: CrossDomainConfig,
    domain_images: dict[str, list[Path]],
) -> list[dict[str, Any]]:
    """Run one fixed XAI method across all domains."""
    rows: list[dict[str, Any]] = []
    detector = DetectorWrapper(config.detector_model, config_path=str(config.detector_config))
    try:
        detector.load_model()
        xai_config = _load_xai_config(config.xai_config)
        method_config = dict(xai_config.get("xai", {}).get("methods", {}).get(method, {}))
        if not method_config:
            raise KeyError(f"XAI method '{method}' not found in {config.xai_config}")
        explainer = get_explainer(method, method_config)
        evaluator = AutoEvaluator(str(config.eval_config))
        target_layer = detector.get_target_layer() if method in {"gradcam", "gcame"} else None
        model = detector.get_model()

        for domain_name, image_paths in domain_images.items():
            for image_path in tqdm(image_paths, desc=f"{METHOD_DISPLAY[method]} {domain_name}", unit="image"):
                try:
                    image = load_image(str(image_path))
                    detections = detector.detect(image)
                    eval_results: list[EvalResult] = []
                    for detection in detections:
                        saliency = explainer.explain(model, image, detection, target_layer)
                        eval_results.append(
                            evaluator.evaluate_all(
                                saliency_map=saliency,
                                bbox=detection.bbox,
                                model=model,
                                image=image,
                                detection=detection,
                            )
                        )
                    rows.append(
                        _row_from_eval_results(
                            domain=domain_name,
                            method=method,
                            image_path=image_path,
                            num_detections=len(detections),
                            evaluation_results=eval_results,
                        )
                    )
                except Exception as exc:
                    logger.warning(
                        "%s baseline failed for %s/%s: %s",
                        method,
                        domain_name,
                        image_path.name,
                        exc,
                        exc_info=True,
                    )
                    rows.append(_empty_row(domain_name, method, image_path, str(exc)))
    finally:
        detector.unload_model()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def summarize(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cross-domain rows by domain and method."""
    metric_cols = [
        "num_detections",
        "pg",
        "ebpg",
        "oa",
        "sparsity",
        "composite",
        "computation_time",
        "iterations",
    ]
    valid = dataframe[~dataframe["has_error"].fillna(False).astype(bool)].copy()
    summary = valid.groupby(["domain", "method"], as_index=False)[metric_cols].agg(["mean", "std"])
    summary.columns = [
        col[0] if col[1] == "" else f"{col[0]}_{col[1]}"
        for col in summary.columns.to_flat_index()
    ]
    errors = dataframe.groupby(["domain", "method"], as_index=False)["has_error"].sum()
    errors = errors.rename(columns={"has_error": "num_errors"})
    summary = summary.merge(errors, on=["domain", "method"], how="left")
    return summary


def run_cross_domain(config: CrossDomainConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run cross-domain evaluation and save outputs."""
    set_seed(config.seed)
    setup_logging(config.log_level)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    domain_images: dict[str, list[Path]] = {}
    for domain in config.domains:
        if not domain.images_dir.exists():
            message = f"Dataset '{domain.name}' images dir not found: {domain.images_dir}"
            if config.skip_missing:
                logger.warning("%s; skipping.", message)
                continue
            raise FileNotFoundError(message)
        image_paths = _collect_images(domain, config.image_exts, config.recursive, config.num_images)
        if not image_paths:
            message = f"Dataset '{domain.name}' has no matching images under {domain.images_dir}"
            if config.skip_missing:
                logger.warning("%s; skipping.", message)
                continue
            raise FileNotFoundError(message)
        domain_images[domain.name] = image_paths
        logger.info("Domain %-12s -> %d images from %s", domain.name, len(image_paths), domain.images_dir)

    if not domain_images:
        raise FileNotFoundError("No usable domains found. Check --datasets or disable --skip-missing.")

    all_rows: list[dict[str, Any]] = []
    if "aedxai" in config.methods:
        all_rows.extend(_run_aedxai(config, domain_images))

    for method in [name for name in config.methods if name in XAI_METHODS]:
        all_rows.extend(_run_fixed_method(method, config, domain_images))

    dataframe = pd.DataFrame(all_rows)
    if "detector" not in dataframe.columns:
        dataframe.insert(1, "detector", config.detector_model)
    summary = summarize(dataframe)

    csv_path = config.output_dir / "cross_domain.csv"
    per_image_path = config.output_dir / "cross_domain_per_image.csv"
    summary_path = config.output_dir / "cross_domain_summary.csv"
    metadata_path = config.output_dir / "cross_domain_summary.json"

    dataframe.to_csv(csv_path, index=False)
    dataframe.to_csv(per_image_path, index=False)
    summary.to_csv(summary_path, index=False)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "domains": {name: len(paths) for name, paths in domain_images.items()},
                "methods": config.methods,
                "num_rows": int(len(dataframe)),
                "num_errors": int(dataframe["has_error"].fillna(False).sum()),
                "output_csv": str(csv_path),
            },
            handle,
            indent=2,
        )

    logger.info("Wrote cross-domain rows    -> %s", csv_path)
    logger.info("Wrote per-image copy       -> %s", per_image_path)
    logger.info("Wrote cross-domain summary -> %s", summary_path)
    print("\n=== Cross-Domain Summary ===")
    print(summary.to_string(index=False))
    return dataframe, summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=DEFAULT_DATASETS, help="Comma-separated NAME=IMAGE_DIR entries.")
    parser.add_argument(
        "--methods",
        default="aedxai,gradcam,gcame,dclose,lime",
        help=f"Comma-separated methods from {sorted(SUPPORTED_METHODS)}.",
    )
    parser.add_argument("--num-images", type=int, default=200, help="Images per domain.")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--image-exts", default=".jpg,.jpeg,.png")
    parser.add_argument("--recursive", action="store_true", help="Search image folders recursively.")
    parser.add_argument("--strict", action="store_true", help="Fail when a configured dataset folder is missing.")
    parser.add_argument("--detector-model", default="yolox-s")
    parser.add_argument("--selector-model", default="", help="Optional selector checkpoint override.")
    parser.add_argument("--detector-config", default="config/detector_config.yaml")
    parser.add_argument("--vlm-config", default="config/vlm_config.yaml")
    parser.add_argument("--xai-config", default="config/xai_config.yaml")
    parser.add_argument("--eval-config", default="config/eval_config.yaml")
    return parser


def main() -> None:
    """Parse CLI args and run cross-domain evaluation."""
    args = build_arg_parser().parse_args()
    selector_model = _as_project_path(args.selector_model) if args.selector_model else None
    config = CrossDomainConfig(
        domains=_parse_domains(args.datasets),
        methods=_parse_methods(args.methods),
        num_images=int(args.num_images),
        output_dir=_as_project_path(args.output_dir),
        seed=int(args.seed),
        log_level=str(args.log_level),
        image_exts=_parse_exts(args.image_exts),
        recursive=bool(args.recursive),
        skip_missing=not bool(args.strict),
        detector_model=str(args.detector_model),
        selector_model=selector_model,
        detector_config=_as_project_path(args.detector_config),
        vlm_config=_as_project_path(args.vlm_config),
        xai_config=_as_project_path(args.xai_config),
        eval_config=_as_project_path(args.eval_config),
    )
    run_cross_domain(config)


if __name__ == "__main__":
    main()
