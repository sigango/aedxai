"""Ablation: XAI selector accuracy vs number of training samples.

Trains a fresh XAI selector at each requested sample size on the same
held-out test set and reports label accuracy + the composite score
actually achieved when the selector is used at inference.

Usage (from repo root):
    python scripts/ablation_selector_size.py \
        --training-csv results/xai_selector_training_data.csv \
        --sizes 50,100,200,500,1000 \
        --seeds 42,123,456

Outputs:
    results/ablation_selector_size.csv          per (size, seed) row
    results/ablation_selector_size_summary.csv  mean/std per size
    results/figures/ablation_selector_size.png  accuracy + composite vs size
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import set_seed, setup_logging
from src.xai_selector import (
    IDX_TO_METHOD,
    METHOD_NAMES,
    METHOD_TO_IDX,
    XAISelector,
)

logger = logging.getLogger("ablation_selector_size")


COMPOSITE_PREFIX = "composite_"


def _require_composite_columns(dataframe: pd.DataFrame) -> list[str]:
    """Ensure the CSV has composite_{method} columns; return them."""
    missing = [method for method in METHOD_NAMES if f"{COMPOSITE_PREFIX}{method}" not in dataframe.columns]
    if missing:
        raise ValueError(
            "Training CSV is missing composite_{method} columns for: "
            f"{missing}. Regenerate with scripts/train_selector.py so that "
            "the finalize step writes per-method composite scores."
        )
    return [f"{COMPOSITE_PREFIX}{method}" for method in METHOD_NAMES]


def _evaluate_on_test(
    selector: XAISelector,
    test_df: pd.DataFrame,
    composite_columns: list[str],
) -> dict[str, float]:
    """Evaluate a trained selector on the held-out test DataFrame."""
    x_test, y_test = selector._prepare_dataframe_tensors(test_df)  # noqa: SLF001
    predicted_indices = selector._predict_indices(x_test)  # noqa: SLF001
    test_accuracy = float(np.mean(predicted_indices == y_test))

    composite_matrix = test_df[composite_columns].to_numpy(dtype=np.float32)
    row_indices = np.arange(composite_matrix.shape[0])
    achieved_composite = float(composite_matrix[row_indices, predicted_indices].mean())
    oracle_composite = float(composite_matrix.max(axis=1).mean())
    mean_composite_by_method = composite_matrix.mean(axis=0)
    random_baseline = float(mean_composite_by_method.mean())

    return {
        "test_accuracy": test_accuracy,
        "achieved_composite": achieved_composite,
        "oracle_composite": oracle_composite,
        "gap_to_oracle": oracle_composite - achieved_composite,
        "random_baseline_composite": random_baseline,
    }


def _stratify_if_possible(labels: np.ndarray, minimum_count: int = 2) -> np.ndarray | None:
    """Return labels for stratified splitting only if every class is populated."""
    counts = np.bincount(labels, minlength=len(METHOD_NAMES))
    nonzero = counts[counts > 0]
    if len(nonzero) > 0 and bool(np.all(nonzero >= minimum_count)):
        return labels
    return None


def run_ablation(
    training_csv: Path,
    sizes: list[int],
    seeds: list[int],
    test_fraction: float,
    epochs: int,
    detector: str | None = None,
) -> pd.DataFrame:
    """Run the ablation sweep and return per-run rows as a DataFrame."""
    dataframe = pd.read_csv(training_csv)
    if "best_method_label" not in dataframe.columns:
        raise ValueError("Training CSV must include 'best_method_label'.")
    composite_columns = _require_composite_columns(dataframe)

    dataframe = dataframe.reset_index(drop=True)
    labels = dataframe["best_method_label"].to_numpy(dtype=np.int64)

    stratify_full = _stratify_if_possible(labels)
    train_pool_df, test_df = train_test_split(
        dataframe,
        test_size=test_fraction,
        random_state=0,
        stratify=stratify_full,
    )
    train_pool_df = train_pool_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    logger.info(
        "Fixed test split: train_pool=%d  test=%d  (classes=%s)",
        len(train_pool_df),
        len(test_df),
        {name: int((test_df["best_method_label"] == index).sum()) for name, index in METHOD_TO_IDX.items()},
    )

    max_available = len(train_pool_df)
    rows: list[dict[str, Any]] = []

    for size in sizes:
        if size > max_available:
            logger.warning(
                "Requested size %d exceeds available pool %d; clamping to %d.",
                size,
                max_available,
                max_available,
            )
        effective_size = min(size, max_available)
        for seed in seeds:
            set_seed(seed)
            pool_labels = train_pool_df["best_method_label"].to_numpy(dtype=np.int64)
            stratify_pool = _stratify_if_possible(pool_labels)
            if effective_size >= len(train_pool_df):
                subsample_df = train_pool_df.copy()
            else:
                subsample_df, _ = train_test_split(
                    train_pool_df,
                    train_size=effective_size,
                    random_state=seed,
                    stratify=stratify_pool,
                )
            subsample_df = subsample_df.reset_index(drop=True)

            selector = XAISelector()
            training_result = selector._train_from_dataframe(  # noqa: SLF001
                dataframe=subsample_df,
                val_split=0.2,
                test_split=0.05,
                epochs=epochs,
                lr=0.001,
                batch_size=min(64, max(8, effective_size // 8)),
            )
            eval_metrics = _evaluate_on_test(selector, test_df, composite_columns)

            row = {
                "detector": detector if detector is not None else "",
                "train_size": int(effective_size),
                "seed": int(seed),
                "train_acc_internal": float(training_result["train_acc"]),
                "val_acc_internal": float(training_result["val_acc"]),
                **eval_metrics,
            }
            logger.info(
                "size=%4d seed=%3d | test_acc=%.4f achieved=%.4f oracle=%.4f gap=%.4f",
                row["train_size"],
                row["seed"],
                row["test_accuracy"],
                row["achieved_composite"],
                row["oracle_composite"],
                row["gap_to_oracle"],
            )
            rows.append(row)

    return pd.DataFrame(rows)


def summarize(per_run_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-run rows into mean/std per (detector, train_size)."""
    numeric_cols = [
        "test_accuracy",
        "achieved_composite",
        "oracle_composite",
        "gap_to_oracle",
        "random_baseline_composite",
    ]
    group_cols = ["detector", "train_size"] if "detector" in per_run_df.columns else ["train_size"]
    grouped = per_run_df.groupby(group_cols)[numeric_cols]
    summary = grouped.agg(["mean", "std"]).round(4)
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def plot_ablation(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Render a two-panel accuracy + composite plot from summary rows."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    if "detector" in summary_df.columns and summary_df["detector"].astype(str).str.len().any():
        grouped = summary_df.groupby("detector", sort=False)
    else:
        grouped = [("selector", summary_df)]

    for detector_name, detector_df in grouped:
        detector_df = detector_df.sort_values("train_size")
        sizes = detector_df["train_size"].to_numpy()
        label = str(detector_name) if str(detector_name) else "selector"

        axes[0].errorbar(
            sizes,
            detector_df["test_accuracy_mean"],
            yerr=detector_df["test_accuracy_std"],
            marker="o",
            capsize=4,
            label=label,
        )

        axes[1].errorbar(
            sizes,
            detector_df["achieved_composite_mean"],
            yerr=detector_df["achieved_composite_std"],
            marker="o",
            label=f"{label} selector",
            capsize=4,
        )
        axes[1].plot(sizes, detector_df["oracle_composite_mean"], linestyle="--", alpha=0.7, label=f"{label} oracle")
        axes[1].plot(
            sizes,
            detector_df["random_baseline_composite_mean"],
            linestyle=":",
            alpha=0.7,
            label=f"{label} random",
        )

    axes[0].set_xscale("log")
    axes[0].set_xlabel("Training samples")
    axes[0].set_ylabel("Held-out test accuracy")
    axes[0].set_title("Selector accuracy vs training size")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xscale("log")
    axes[1].set_xlabel("Training samples")
    axes[1].set_ylabel("Mean composite on test set")
    axes[1].set_title("Achieved composite vs training size")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote ablation plot -> %s", output_path)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="XAI selector training-size ablation.")
    parser.add_argument(
        "--training-csv",
        default="results/xai_selector_training_data.csv",
        help="Selector training CSV produced by scripts/train_selector.py.",
    )
    parser.add_argument(
        "--sizes",
        default="50,100,200,500,1000",
        help="Comma-separated training sample sizes to sweep.",
    )
    parser.add_argument(
        "--seeds",
        default="42,123,456",
        help="Comma-separated seeds to repeat each size.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--detector",
        default=None,
        help=(
            "Optional detector tag (e.g. yolox-s, fasterrcnn_resnet50_fpn_v2) "
            "written as a column in the output CSVs so notebook 05 can plot "
            "per-detector ablation lines. Appends to existing CSVs if present."
        ),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output CSVs instead of overwriting.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    sizes = [int(value.strip()) for value in args.sizes.split(",") if value.strip()]
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]

    training_csv = PROJECT_ROOT / args.training_csv
    if not training_csv.exists():
        raise FileNotFoundError(
            f"Training CSV not found: {training_csv}. "
            "Run scripts/train_selector.py first."
        )

    per_run_df = run_ablation(
        training_csv=training_csv,
        sizes=sizes,
        seeds=seeds,
        test_fraction=args.test_fraction,
        epochs=args.epochs,
        detector=args.detector,
    )
    summary_df = summarize(per_run_df)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    per_run_path = output_dir / "ablation_selector_size.csv"
    summary_path = output_dir / "ablation_selector_size_summary.csv"

    if args.append and per_run_path.exists():
        existing_runs = pd.read_csv(per_run_path)
        if "detector" not in existing_runs.columns:
            existing_runs["detector"] = ""
        combined_runs = pd.concat([existing_runs, per_run_df], ignore_index=True)
        combined_runs = combined_runs.drop_duplicates(
            subset=["detector", "train_size", "seed"], keep="last"
        ).reset_index(drop=True)
        combined_runs.to_csv(per_run_path, index=False)
        plot_summary_df = summarize(combined_runs)
        plot_summary_df.to_csv(summary_path, index=False)
    else:
        per_run_df.to_csv(per_run_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        plot_summary_df = summary_df
    plot_ablation(plot_summary_df, output_dir / "figures" / "ablation_selector_size.png")

    logger.info("Wrote per-run rows -> %s", per_run_path)
    logger.info("Wrote summary      -> %s", summary_path)
    print("\n=== Selector Training-Size Ablation ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
