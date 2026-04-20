"""Run the AED-XAI pipeline on YOLOX-S and Faster R-CNN and compare results.

Usage (from repo root):
    python scripts/compare_detectors.py --max-images 50

Outputs:
    results/compare_detectors_per_image.csv   per-image, per-detector rows
    results/compare_detectors_summary.csv     mean/std per metric per detector
    console: summary table
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import AEDXAIPipeline, PipelineResult
from src.utils import set_seed, setup_logging

logger = logging.getLogger("compare_detectors")

SUPPORTED_DETECTORS = ["yolox-s", "fasterrcnn_resnet50_fpn_v2"]


def _pipeline_row(result: PipelineResult, detector_name: str) -> dict[str, Any]:
    """Flatten one PipelineResult into a row suitable for the comparison CSV."""
    eval_results = list(result.evaluation_results)

    def _mean(attr: str) -> float:
        values = [float(getattr(item, attr)) for item in eval_results]
        return float(np.mean(values)) if values else float("nan")

    return {
        "detector": detector_name,
        "image_path": Path(result.image_path).name,
        "num_detections": len(result.detections),
        "composite_score": float(result.composite_score) if result.composite_score is not None else float("nan"),
        "iterations": int(result.metadata.get("iterations", 0)),
        "converged": bool(result.metadata.get("converged", False)),
        "mean_pg": _mean("pg"),
        "mean_ebpg": _mean("ebpg"),
        "mean_oa": _mean("oa"),
        "mean_sparsity": _mean("sparsity"),
        "mean_insertion_auc": _mean("insertion_auc"),
        "mean_deletion_auc": _mean("deletion_auc"),
        "total_time": float(sum(float(item.computation_time) for item in eval_results)),
        "has_error": "error" in result.metadata,
    }


def run_pipeline_for_detector(
    detector_name: str,
    image_paths: list[Path],
    config_paths: dict[str, Path],
) -> list[dict[str, Any]]:
    """Run the full AED-XAI pipeline for one detector across all images."""
    logger.info("=== Running pipeline with detector=%s ===", detector_name)
    pipeline = AEDXAIPipeline(
        detector_config_path=str(config_paths["detector"]),
        vlm_config_path=str(config_paths["vlm"]),
        xai_config_path=str(config_paths["xai"]),
        eval_config_path=str(config_paths["eval"]),
        detector_model_name=detector_name,
    )

    rows: list[dict[str, Any]] = []
    try:
        pipeline.setup()
        for path in tqdm(image_paths, desc=f"{detector_name}"):
            try:
                result = pipeline.run_on_image(str(path))
                rows.append(_pipeline_row(result, detector_name))
            except Exception as exc:
                logger.warning("Pipeline failed for %s on %s: %s", detector_name, path.name, exc)
                rows.append(
                    {
                        "detector": detector_name,
                        "image_path": path.name,
                        "num_detections": 0,
                        "composite_score": float("nan"),
                        "iterations": 0,
                        "converged": False,
                        "mean_pg": float("nan"),
                        "mean_ebpg": float("nan"),
                        "mean_oa": float("nan"),
                        "mean_sparsity": float("nan"),
                        "mean_insertion_auc": float("nan"),
                        "mean_deletion_auc": float("nan"),
                        "total_time": 0.0,
                        "has_error": True,
                    }
                )
    finally:
        pipeline.shutdown()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    return rows


def summarize(per_image_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-image rows into a mean/std table per detector."""
    numeric_cols = [
        "num_detections",
        "composite_score",
        "iterations",
        "mean_pg",
        "mean_ebpg",
        "mean_oa",
        "mean_sparsity",
        "mean_insertion_auc",
        "mean_deletion_auc",
        "total_time",
    ]
    grouped = per_image_df.groupby("detector")[numeric_cols]
    summary = grouped.agg(["mean", "std"]).round(4)
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Compare AED-XAI pipeline across detectors.")
    parser.add_argument("--detector-config", default="config/detector_config.yaml")
    parser.add_argument("--vlm-config", default="config/vlm_config.yaml")
    parser.add_argument("--xai-config", default="config/xai_config.yaml")
    parser.add_argument("--eval-config", default="config/eval_config.yaml")
    parser.add_argument("--images-dir", default="data/coco/val2017")
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument(
        "--detectors",
        default=",".join(SUPPORTED_DETECTORS),
        help="Comma-separated detector model names to compare.",
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging(args.log_level)
    set_seed(args.seed)

    config_paths = {
        "detector": PROJECT_ROOT / args.detector_config,
        "vlm": PROJECT_ROOT / args.vlm_config,
        "xai": PROJECT_ROOT / args.xai_config,
        "eval": PROJECT_ROOT / args.eval_config,
    }

    images_dir = PROJECT_ROOT / args.images_dir
    image_paths = sorted(images_dir.glob("*.jpg"))[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"No images found under {images_dir}")
    logger.info("Running comparison on %d images from %s", len(image_paths), images_dir)

    detectors = [name.strip() for name in args.detectors.split(",") if name.strip()]
    for name in detectors:
        if name not in SUPPORTED_DETECTORS:
            raise ValueError(f"Unsupported detector '{name}'. Choose from: {SUPPORTED_DETECTORS}")

    # Run detectors sequentially to avoid VRAM contention — each loads and unloads cleanly.
    all_rows: list[dict[str, Any]] = []
    for detector_name in detectors:
        all_rows.extend(run_pipeline_for_detector(detector_name, image_paths, config_paths))

    per_image_df = pd.DataFrame(all_rows)
    summary_df = summarize(per_image_df)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = output_dir / "compare_detectors_per_image.csv"
    summary_path = output_dir / "compare_detectors_summary.csv"
    per_image_df.to_csv(per_image_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    logger.info("Wrote per-image rows -> %s", per_image_path)
    logger.info("Wrote summary        -> %s", summary_path)
    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
